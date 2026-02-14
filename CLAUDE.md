# FPL Manager — Claude Code Notes

## Project Goal

Build a fully autonomous FPL manager. This is NOT just a prediction tool — it should think and plan like a real FPL manager across the entire season:

- **Transfer planning**: Rolling 5-GW horizon with FT banking, price awareness, and fixture swings
- **Squad building**: Shape the squad toward upcoming fixture runs, not just next GW
- **Captain planning**: Joint captain optimization in the MILP solver + pre-planned captaincy calendar
- **Chip strategy**: Evaluate all 4 chips across every remaining GW using DGW/BGW awareness, squad-specific predictions, and chip synergies (WC→BB, FH+WC)
- **Price awareness**: Ownership-based price change predictions with probability scores
- **Reactive adjustments**: Auto-detect injuries, fixture changes, and prediction shifts that invalidate the plan — with SSE-driven alerts and one-click replan
- **Outcome tracking**: Record what was recommended vs what happened, track model accuracy over time

Every decision (transfer, captain, bench order, chip) is made in context of the bigger picture. The app produces a rolling multi-week plan that constantly recalculates as new information comes in.

## Repos

- **This project**: `https://github.com/invisi101/fplmanager` (active development)
- **Original predictor**: `https://github.com/invisi101/fplxti` (previous project this was forked from)

## Environment

- **Python**: Use `.venv/bin/python`, NOT system `python3` (system Python lacks project dependencies)
- **Run server**: `.venv/bin/python -m src.app` (serves on `http://127.0.0.1:9875`)
- **Port 9875**: Often has leftover processes from previous sessions. Kill with `lsof -ti:9875 | xargs kill -9` before starting
- **No build step**: Frontend is a single file at `src/templates/index.html` (inline CSS + JS). Just edit and refresh.

## My Manager ID

12904702

---

## Architecture Overview

Three-layer system: **Data → Features/Models → Strategy/Solver**, backed by SQLite and served via Flask.

### Project Structure

```
src/
├── app.py                  # Flask app, 40+ API endpoints, background task runner, SSE
├── templates/
│   └── index.html          # Entire frontend (single file, inline CSS + JS, dark theme)
├── data_fetcher.py         # GitHub CSV + FPL API data fetching with caching
├── feature_engineering.py  # ~1400 lines, 100+ features per player per GW
├── model.py                # XGBoost training: mean, quantile (Q80), decomposed sub-models
├── predict.py              # Prediction pipeline: 1-GW, 3-GW, 8-GW horizon, captain scores
├── backtest.py             # Walk-forward backtesting framework
├── solver.py               # MILP solvers: squad selection + transfer optimization + captain
├── strategy.py             # Strategic brain: ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer
├── season_manager.py       # Season orchestration: recommendations, outcomes, prices, plan health
├── season_db.py            # SQLite with 8 tables (season, snapshots, recommendations, outcomes, prices, fixtures, plans, changelog)
└── feature_selection.py    # Feature importance analysis (correlation, RF, Lasso, RFE)

models/     # Saved .joblib model files (gitignored)
output/     # predictions.csv, charts, locked_teams.json (gitignored)
cache/      # Cached data: 6h for GitHub CSVs, 30m for FPL API (gitignored)
```

---

## Data Pipeline

### Sources
1. **GitHub (FPL-Core-Insights)**: Historical match stats, player stats, player match stats for 2024-2025 and 2025-2026 seasons. Cached 6 hours.
2. **FPL API** (public, no auth): Current player data (prices, form, injuries, ownership), fixtures, manager picks/history. Cached 30 minutes.

### Feature Engineering (`feature_engineering.py`)
100+ features per player per GW including:
- Player rolling stats (3/5/10 GW windows): xG, xA, shots, touches, dribbles, crosses, tackles, goals, assists
- EWM features (span=5): Exponentially weighted xG, xA, xGOT
- Upside/volatility: xG volatility, form acceleration, big chance frequency
- Home/away form splits
- Opponent history: Player's historical performance vs specific opponents
- Team rolling stats: Goals scored, xG, clean sheets, big chances
- Opponent defensive stats: xG conceded, shots conceded
- Rest/congestion: Days rest, fixture congestion rate
- Fixture context: FDR, is_home, opponent_elo, multi-GW lookahead
- ICT/BPS: Influence, creativity, threat, bonus points
- Market data: Ownership, transfer momentum
- Availability: Chance of playing, availability rate
- Interaction features: xG × opponent goals conceded, CS opportunity

All features shifted by 1 GW to prevent leakage. DGW handling: multiple rows per fixture, targets divided by fixture count, predictions summed.

---

## Model Architecture

### Tier 1: Mean Regression (Primary)
- 4 models per position × 2 targets (next_gw_points, next_3gw_points)
- XGBoost `reg:squarederror`, walk-forward CV (20 splits)
- Sample weighting: current season 1.0, previous 0.5

### Tier 2: Quantile Models (Captain Picks)
- MID + FWD only, 80th percentile of next_gw_points
- `captain_score = 0.4 × mean + 0.6 × Q80` — captures explosive upside

### Tier 3: Decomposed Sub-Models
- ~20 models predicting individual components: goals, assists, clean sheets, bonus, saves
- Position-specific objectives (Poisson for counts, logistic for binary CS)
- Combined via FPL scoring rules with playing probability weighting

### Multi-GW Predictions
- 3-GW: Sum of three 1-GW predictions with correct opponent data per offset
- 8-GW horizon: Model predictions for near-term, fixture heuristics for distant GWs
- Confidence decays with distance (0.95 → 0.77 at GW+5)

---

## MILP Solver (`solver.py`)

### `solve_milp_team()` — Optimal squad from scratch
Two-tier MILP with optional captain optimization:
- **Variables**: `x_i` (in squad), `s_i` (starter), `c_i` (captain, when `captain_col` provided)
- **Objective**: max(0.9 × starting XI pts + 0.1 × bench pts + captain bonus)
- **Constraints**: Budget, positions (2/5/5/3), max 3 per team, 11 starters, formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), exactly 1 captain who is a starter
- **Returns**: starters, bench, total_cost, starting_points, captain_id

### `solve_transfer_milp()` — Optimal transfers
Same as above plus: `sum(x_i × is_current_i) >= 15 - max_transfers` — keeps at least (15 - N) current players.
- Budget = bank + sum(now_cost of current squad)
- Also supports joint captain optimization

---

## Strategic Planning Brain (`strategy.py`)

### ChipEvaluator
Evaluates all 4 chips (BB, TC, FH, WC) across every remaining GW:
- Near-term: Uses model predictions (solve MILP for FH/WC, bench sums for BB, best player for TC)
- Far-term: Fixture heuristics (DGW count, BGW count, FDR)
- Synergies: WC→BB (build BB-optimized squad), FH+WC (complementary strategy)

### MultiWeekPlanner
Rolling **5-GW** transfer planner:
- Forward simulation: tries each FT allocation for GW+1, picks path maximizing total points
- Considers: FT banking (save → use 2 next week), fixture swings, price change probability
- Reduces pool to top 200 players for efficiency
- Passes `captain_col` to MILP solver for captain-aware squad building

### CaptainPlanner
Pre-plans captaincy across the prediction horizon:
- Uses transfer plan squads to pick captain from the planned squad (not just current)
- Flags weak captain GWs (predicted < 4 pts)

### PlanSynthesizer
Combines all plans into a coherent timeline:
- Chip schedule (synergy-aware: uses WC→BB combos when valuable)
- Natural-language rationale explaining the overall strategy
- Comparison with previous plan → changelog

### Reactive Re-planning
- `detect_plan_invalidation()`: Checks injuries (critical), fixture changes (BB without DGW), prediction shifts (>50% captain drop), doubtful players
- `apply_availability_adjustments()`: Zeros predictions for injured/suspended players
- `check_plan_health()`: Lightweight check using bootstrap data (no prediction regeneration)
- Auto-triggers on data refresh via SSE `plan_invalidated` events

---

## Season Manager (`season_manager.py`)

Orchestrates everything for a full season:

### Weekly Workflow
1. **Refresh Data** → updates cache, detects availability issues, checks plan health
2. **Generate Recommendation** → multi-GW predictions, chip heatmap, transfer plan, captain plan, strategic plan synthesis, stores everything in DB
3. **Review Action Plan** → clear steps (transfer X out / Y in, set captain to Z, activate chip)
4. **Make Moves** → user executes on FPL website
5. **Record Results** → imports actual picks, compares to recommendation, tracks accuracy

### Price Tracking
- `track_prices()`: Snapshots prices for squad + top 30 transferred-in players
- `get_price_alerts()`: Raw net-transfer threshold alerts
- `predict_price_changes()`: Ownership-based algorithm approximation
  - `transfer_ratio = net_transfers / (ownership_pct × 100,000)`
  - Rise if ratio > 0.005, fall if < -0.005
  - Probability = min(1.0, abs(ratio) / 0.01)
- `get_price_history()`: Historical snapshots with date/price/net_transfers

### Pre-Season
- `generate_preseason_plan()`: MILP for initial squad (100.0m budget) + full-season chip schedule
- Falls back to price-based heuristic if no model predictions available

---

## Database Schema (`season_db.py`)

8 SQLite tables:
| Table | Purpose |
|-------|---------|
| `season` | Manager seasons (id, manager_id, name, start_gw, current_gw) |
| `gw_snapshot` | Per-GW state (squad_json, bank, team_value, points, rank, captain) |
| `recommendation` | Pre-GW advice (transfers_json, captain, chip, predicted_points) |
| `recommendation_outcome` | Post-GW tracking (followed_transfers, actual_points, point_delta) |
| `price_tracker` | Player price snapshots (price, transfers_in/out, snapshot_date) |
| `fixture_calendar` | GW × team fixture grid (fixture_count, fdr_avg, is_dgw, is_bgw) |
| `strategic_plan` | Full plan JSON + chip heatmap JSON (per season per GW) |
| `plan_changelog` | Plan change history (chip reschedule, captain change, reason) |

---

## API Endpoints

### Core
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/predictions` | Player predictions with filters/sorting |
| POST | `/api/refresh-data` | Re-fetch data, rebuild predictions, check plan health |
| POST | `/api/train` | Train all model tiers + generate predictions |
| POST | `/api/best-team` | MILP optimal squad |
| GET | `/api/my-team?manager_id=ID` | Import manager's FPL squad |
| POST | `/api/transfer-recommendations` | MILP transfer solver (with captain optimization) |
| GET | `/api/status` | SSE stream for live progress |

### Season Management
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/season/init` | Backfill season history |
| GET | `/api/season/dashboard` | Full dashboard (rank, budget, accuracy_history) |
| POST | `/api/season/generate-recommendation` | Generate strategic plan + recommendation |
| POST | `/api/season/record-results` | Import actual results, compare to advice |
| GET | `/api/season/action-plan` | Clear action items for next GW |
| GET | `/api/season/outcomes` | All recorded outcomes |

### Strategic Planning
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET/POST | `/api/season/strategic-plan` | Fetch/generate full strategic plan |
| GET | `/api/season/chip-heatmap` | Chip values across remaining GWs |
| GET | `/api/season/plan-health` | Check plan validity (injuries/fixtures) |
| GET | `/api/season/plan-changelog` | Plan change history |

### Prices
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/season/prices` | Latest prices + raw alerts |
| GET | `/api/season/price-predictions` | Ownership-based price predictions |
| GET | `/api/season/price-history` | Price movement history (date/price/transfers) |

### Other
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/backtest` | Walk-forward backtesting |
| POST | `/api/gw-compare` | Compare locked team vs hindsight-best |
| POST | `/api/preseason/generate` | Pre-season initial squad + chip plan |
| GET | `/api/season/fixtures` | Fixture calendar |

---

## UI Structure (`src/templates/index.html`)

Single-file frontend with dark theme, CSS variables, SSE progress, localStorage persistence.

### Main Tabs
1. **Predictions** — Sortable player table, position filters, search
2. **Best Team** — MILP squad selector with pitch visualization, lock-in for comparison
3. **My Team** — Import FPL squad, dual-pitch (actual pts / predicted), transfer recommendations
4. **Season** — Full season management dashboard

### Season Sub-tabs
- **Overview**: Rank chart, points bar chart, budget chart, prediction accuracy dual-line chart
- **Workflow**: Step indicators (Refresh → Recommend → Review → Execute → Record), action plan, outcomes
- **Fixtures**: GW × team fixture grid
- **Transfers**: Transfer history table
- **Chips**: Status (used/available) + values
- **Prices**: Alerts, ownership-based predictions (risers/fallers with probability bars), price history multi-line chart, squad prices table
- **Strategy**: Plan health banner (auto-detect + replan button), rationale, 5-GW transfer timeline cards, captain plan badges, chip schedule + synergy annotations, chip heatmap table (color-coded), plan changelog

### Charts
- `drawLineChart()` — Single-line (rank, budget)
- `drawBarChart()` — Bar chart (points per GW)
- `drawDualLineChart()` — Two-line with legend (predicted vs actual accuracy)
- Price history chart — Multi-line canvas for top 5 movers

---

## Testing the App

1. Kill any existing process: `lsof -ti:9875 | xargs kill -9`
2. Start: `.venv/bin/python -m src.app`
3. Only one background task runs at a time (train, backtest, etc.)
4. Test API: `curl -s http://127.0.0.1:9875/api/my-team?manager_id=12904702`

### Key test commands
```bash
# Strategic plan (5-GW timeline)
curl -s http://127.0.0.1:9875/api/season/strategic-plan?manager_id=12904702 | python3 -c "import sys,json; t=json.load(sys.stdin)['plan']['timeline']; print(f'{len(t)} GW entries')"

# Plan health
curl -s http://127.0.0.1:9875/api/season/plan-health?manager_id=12904702

# Price predictions
curl -s http://127.0.0.1:9875/api/season/price-predictions?manager_id=12904702

# Dashboard with accuracy history
curl -s http://127.0.0.1:9875/api/season/dashboard?manager_id=12904702 | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'accuracy_history: {len(d.get(\"accuracy_history\",[]))} entries')"
```

---

## What's Built (Complete)

All phases from the original roadmap are done:

- **Pre-season**: GW1 cold-start predictions, initial squad selection, full-season chip plan
- **Season dashboard**: GW-by-GW tracking of rank, points, budget, model accuracy
- **DGW/BGW detection**: Fixture calendar with DGW/BGW flags, factored into chip timing
- **Multi-week transfer planner**: 5-GW rolling horizon with FT banking simulation
- **Price change integration**: Ownership-based predictions with probability, price history charts
- **Captain optimization**: Joint MILP solver with captain decision variables (3n vars)
- **Chip strategy engine**: Full heatmap across all remaining GWs + synergy detection
- **Fixture rotation**: Fixture swing bonuses in transfer planning
- **Reactive re-planning**: Injury/fixture change detection with SSE alerts + one-click replan
- **Outcome tracking**: Recommendation vs actual comparison, accuracy history charts

## Remaining TODO: Rethink Backtesting & Feature Visualization

The following 4 UI buttons have been **removed from the frontend** (but all backend code is preserved):

- **Run Feature Selection** (`/api/feature-selection`) — `src/feature_selection.py` still exists
- **Model Importance** (`/api/xgb-importance`) — endpoint still in `src/app.py`
- **Feature Insights** panel — showed feature charts, reports, and XGBoost importance
- **Backtest** panel — walk-forward backtesting UI with per-GW breakdown

**Why removed**: After the dynamic season handling rewrite (multi-season support, graduated weights, generic data fetching), these features need rethinking. The backtest framework may need updates for multi-season walk-forward, and the feature visualization approach should be reconsidered given the new 100+ feature set across N seasons.

**To restore**: The backend endpoints (`/api/feature-selection`, `/api/xgb-importance`, `/api/feature-report`, `/api/xgb-importance-report`, `/api/backtest`, `/api/backtest-results`) and Python files (`src/backtest.py`, `src/feature_selection.py`) are all intact. To bring back the UI, re-add the buttons to the action bar in `src/templates/index.html`, re-add the HTML panels, CSS styles, and JS functions. Check git history for the removed code (commit after the dynamic season handling commit).

**What to consider when rebuilding**:
- Backtest should work seamlessly across all dynamically detected seasons
- Feature importance visualization could show per-season breakdowns
- Consider integrating model accuracy metrics into the Season dashboard instead of a separate panel
- The current walk-forward CV in `src/backtest.py` may need updating for 3+ season training

---

## Remaining TODO: Full Autonomy

The one remaining feature is **authenticated FPL API access for autonomous execution**:

### What it needs
1. **FPL Authentication**: Login with FPL credentials (email/password) to get session cookies
   - FPL uses `https://users.premierleague.com/accounts/login/` for auth
   - Returns session cookies needed for write endpoints
   - Credentials should be stored securely (env vars or encrypted config, never in code)

2. **Write API Endpoints**: Use authenticated session to:
   - `POST /api/transfers` — Execute transfers (player_in, player_out)
   - `POST /api/my-team/captain` — Set captain and vice-captain
   - `POST /api/my-team/` — Set starting XI and bench order
   - `POST /api/chips/activate` — Activate chips (wildcard, freehit, bboost, 3xc)

3. **Exact Selling Prices**: Authenticated API provides real selling prices (which account for price rise profit sharing — you only get 50% of price rises). Currently we use `now_cost` which is slightly generous.

4. **Execution Flow**: After generating a recommendation:
   - Show the action plan in the UI (already done)
   - Add "Execute All" button that calls authenticated endpoints
   - Confirm before executing (show what will happen)
   - Log what was executed vs what was planned
   - Handle errors gracefully (insufficient funds, player unavailable, deadline passed)

5. **Safety Guardrails**:
   - Never execute without explicit user confirmation (unless configured for full autopilot)
   - Deadline awareness: warn if close to GW deadline, refuse if past
   - Rollback info: show how to reverse transfers manually if something goes wrong
   - Rate limiting: respect FPL API rate limits
   - Dry-run mode: show what would happen without actually doing it

6. **Scheduling** (optional, for true autopilot):
   - Cron-like scheduler: refresh data daily, generate recommendation 24h before deadline
   - Auto-execute transfers N hours before deadline if confidence is high
   - Alert (email/push) if plan health issues detected

### Implementation approach
- Add `src/fpl_auth.py` for authentication + authenticated API calls
- Add execution methods to `SeasonManager` (execute_transfers, set_captain, activate_chip)
- Add `/api/season/execute` endpoint with confirmation flow
- Add "Execute" button to the Action Plan UI section
- Store credentials via environment variables (`FPL_EMAIL`, `FPL_PASSWORD`)
