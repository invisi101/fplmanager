# FPL Gaffer - the brAIn — Claude Code Notes

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

- **This project**: `https://github.com/invisi101/FPLGaffer` (active development)
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
output/     # predictions.csv, charts (gitignored)
cache/      # Cached data: 6h for GitHub CSVs, 30m for FPL API (gitignored)
```

---

## Data Pipeline

### Sources
1. **GitHub (FPL-Core-Insights)**: Historical match stats, player stats, player match stats for 2024-2025 and 2025-2026 seasons. Cached 6 hours.
2. **FPL API** (public, no auth): Current player data (prices, form, injuries, ownership), fixtures, manager picks/history/transfers. Cached 30 minutes.

### Feature Engineering (`feature_engineering.py`)
100+ features per player per GW including:
- Player rolling stats (3/5/10 GW windows): xG, xA, shots, touches, dribbles, crosses, tackles, goals, assists
- EWM features (span=5): Exponentially weighted xG, xA, xGOT
- Upside/volatility: xG volatility, form acceleration, big chance frequency
- Home/away form splits
- Opponent history: Player's historical performance vs specific opponents
- Team rolling stats: Goals scored, xG, clean sheets, big chances
- Opponent stats: Defensive (xG conceded, shots conceded) AND attacking (goals scored, xG)
- CBIT composite (clearances + blocks + interceptions + tackles) for DefCon prediction
- Rest/congestion: Days rest, fixture congestion rate
- Fixture context: FDR, is_home, opponent_elo, multi-GW lookahead
- ICT/BPS: Influence, creativity, threat, bonus points
- Market data: Ownership, transfer momentum
- Availability: Chance of playing, availability rate
- Interaction features: xG × opponent goals conceded, CS opportunity (based on opponent attacking xG)

All features shifted by 1 GW to prevent leakage. DGW handling: multiple rows per fixture, targets divided by fixture count, predictions summed.

---

## Model Architecture

### Tier 1: Mean Regression (Primary)
- 4 models (one per position) for `next_gw_points`
- XGBoost `reg:squarederror`, walk-forward CV (last 20 splits)
- Sample weighting: current season 1.0, previous 0.5
- Fixed hyperparameters: 150 trees, depth 5, lr 0.1, subsample 0.8 (`tune=False` in production — grid search was tested and degraded performance)
- 3-GW predictions are derived at inference by summing three 1-GW predictions with per-GW opponent data

### Tier 2: Quantile Models (Captain Picks)
- MID + FWD only, 80th percentile of next_gw_points
- `captain_score = 0.4 × mean + 0.6 × Q80` — captures explosive upside

### Tier 3: Decomposed Sub-Models
- Position-specific component models predicting individual scoring elements:
  - **GKP**: cs, goals_conceded, saves, bonus
  - **DEF**: goals, assists, cs, goals_conceded, bonus, defcon
  - **MID**: goals, assists, cs, bonus, defcon
  - **FWD**: goals, assists, bonus
- Poisson objectives for count data (goals, assists, bonus, saves, goals_conceded, defcon), squarederror for CS (fractional DGW targets)
- DefCon: Poisson CDF predicts P(CBIT >= threshold) where threshold = 10 (DEF) or 12 (MID), scores +2 pts
- Combined via FPL scoring rules with playing probability weighting
- Soft calibration caps per position (GKP=7, DEF=8, MID=10, FWD=10)

### Ensemble
- Production predictions use an **85/15 blend** of mean regression and decomposed sub-models (`ENSEMBLE_WEIGHT_DECOMPOSED = 0.15` in predict.py)
- Mean model drives prediction accuracy; decomposed weight preserves ranking signal in the bench/rotation zone

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

### `solve_transfer_milp_with_hits()` — Hit-aware wrapper
Evaluates 0..max_transfers, subtracts forced replacements from hit count, applies -4 per hit. Compares all options including 0-transfer baseline.

---

## Strategic Planning Brain (`strategy.py`)

### ChipEvaluator
Evaluates all 4 chips (BB, TC, FH, WC) across every remaining GW:
- Near-term: Uses model predictions (solve MILP for FH/WC, bench sums for BB, best starter for TC)
- Far-term: Fixture heuristics (DGW count, BGW count, FDR)
- Synergies: WC→BB (build BB-optimized squad), FH+WC (complementary strategy)

### MultiWeekPlanner
Rolling **5-GW** transfer planner:
- Tree search: generates all valid FT allocation sequences across the full 5-GW horizon, simulates each, picks the path maximizing total points
- Considers: FT banking (save vs spend at every GW), fixture swings, price change probability
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
- `get_price_history()`: Historical snapshots with date/price/net_transfers

---

## Database Schema (`season_db.py`)

8 SQLite tables:
| Table | Purpose |
|-------|---------|
| `season` | Manager seasons (id, manager_id, name, start_gw, current_gw) |
| `gw_snapshot` | Per-GW state (squad_json, bank, team_value, points, rank, captain, transfers_in/out) |
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
| GET | `/api/monsters` | Top 3 players across 8 monster categories |

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
| POST | `/api/gw-compare` | Compare manager's actual team vs hindsight-best for any past GW |
| POST | `/api/preseason/generate` | Pre-season initial squad + chip plan |
| GET | `/api/season/fixtures` | Fixture calendar |

---

## UI Structure (`src/templates/index.html`)

Single-file frontend with dark theme, CSS variables, SSE progress, localStorage persistence.

### Main Tabs
1. **Predictions** — Sortable player table, position filters, search
2. **Best Team** — MILP squad selector with pitch visualization
3. **GW Compare** — Compare your actual FPL team vs hindsight-best for any past GW (dual pitch with overlap highlighting)
4. **My Team** — Import FPL squad, dual-pitch (actual pts / predicted), transfer recommendations
5. **Season** — Full season management dashboard
6. **Monsters** — Top 3 players across 8 categories (Goal Monsters, DefCon Monsters, Closet Strikers, Assist Kings, Set Piece Merchants, Captain Monsters, Value Monsters, Clean Sheet Machines) with player photos and podium layout

### Season Sub-tabs
- **Overview**: Rank chart, points bar chart, budget chart, prediction accuracy dual-line chart
- **Workflow**: Step indicators (Refresh → Recommend → Review → Execute → Record), action plan, outcomes
- **Fixtures**: GW × team fixture grid
- **Transfers**: Transfer history table
- **Chips**: Status (used/available) + values
- **Prices**: Alerts, ownership-based predictions (risers/fallers with probability bars), price history multi-line chart, squad prices table
- **Strategy**: Plan health banner (auto-detect + replan button), rationale, 5-GW transfer timeline cards, captain plan badges, chip schedule + synergy annotations, chip heatmap table (color-coded), plan changelog

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

# GW Compare (manager's actual team vs hindsight-best)
curl -s http://127.0.0.1:9875/api/gw-compare -X POST -H 'Content-Type: application/json' -d '{"manager_id":12904702,"gameweek":20}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'My: {d[\"my_team\"][\"starting_actual\"]} pts, Best: {d[\"best_team\"][\"starting_actual\"]} pts, Capture: {d[\"capture_pct\"]}%')"
```

---

## FPL Rules Reference

Key FPL rules that affect the codebase logic:

- **Free Transfers (FTs)**: You get 1 FT per GW. Unused FTs roll over, max 5 banked. Using more than your FTs costs -4 pts each.
- **Wildcard**: Unlimited transfers for one GW. Squad changes are permanent. **FTs are preserved** — you keep the same FT count you had before playing the chip.
- **Free Hit**: Unlimited transfers for one GW. Squad reverts to pre-FH squad the following GW. **FTs are preserved** — same count as before the chip.
- **Bench Boost**: All 15 players score (not just starting 11). One GW only.
- **Triple Captain**: Captain scores 3× instead of 2×. One GW only.
- **Half-season reset**: All 4 chips (WC, FH, BB, TC) are available once per half (GW1-19 and GW20-38), giving 8 total chips per season. First-half chips expire at GW19 deadline and cannot be carried over.
- **Captain**: Doubles the selected player's points. Vice-captain scores double if captain doesn't play.
- **DGW/BGW**: Double Gameweek = team plays twice (points from both matches summed). Blank Gameweek = team doesn't play.
- **DefCon (2025-2026)**: +2 pts if CBIT (clearances + blocks + interceptions + tackles) >= threshold. Threshold: 10 for GKP/DEF, 12 for MID/FWD.
- **GKP goals**: Worth 10 pts (changed from 6 in 2025-2026 season).

---

## Known Limitations

Design trade-offs and API limitations that are documented but not fixed:

- **Static team_code for transferred players**: Decomposed targets, opponent history, and home/away form use the player's CURRENT team from the `players` table. Pre-transfer matches get the wrong team assignment. NaN propagation handles gracefully. Fix requires per-match team assignment data not available in the public API.
- **fixture_congestion includes predicted match's own rest**: The rolling window includes rest before the match being predicted. Borderline design choice — fixture schedule is known in advance, so it's not data leakage.
- **FT planner explores at most 1 hit per GW**: Taking 2+ hits per GW is not explored (almost never profitable in FPL).
- **3-GW prediction is a simple sum**: No adjustment for form regression, rotation risk increase, or injury probability over time.
- **Selling prices use `now_cost`**: The public API doesn't provide actual selling prices (which account for 50% profit sharing on price rises). The authenticated API does, but we don't use it yet.

---

## Build Pipeline & Releases

GitHub Actions workflow builds Windows and macOS executables via PyInstaller. Triggers on release creation or manual `workflow_dispatch`.

### Key files
- **`.github/workflows/build-exe.yml`** — Two parallel jobs (windows-latest, macos-latest). Python 3.12.
- **`fpl-predictor.spec`** / **`gaffer-mac.spec`** — PyInstaller specs for each platform.
- **`launcher.py`** — Entrypoint for PyInstaller builds. Starts Flask server and opens browser.
- **`setup-mac.sh`** — Run once after cloning. Creates venv, installs deps, builds `Gaffer.app` in `/Applications`.

### Creating a release
```bash
gh release create v1.2.0 --title "v1.2.0" --notes "Release notes here"
```

### Rebuilding a release (IMPORTANT)
**`gh release delete` does NOT delete the git tag.** You MUST delete both the release AND the tag, or just bump the version number:
```bash
gh release delete v1.2.0 --yes
git push origin --delete v1.2.0
git tag -d v1.2.0
gh release create v1.2.0 --title "v1.2.0" --notes "Release notes here"
```

### PyInstaller frozen-mode path detection
Every source file that references `output/`, `models/`, or `cache/` directories must use this pattern:

```python
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent   # directory containing the EXE
else:
    _BASE = Path(__file__).resolve().parent.parent  # project root from src/

OUTPUT_DIR = _BASE / "output"
```

All source files that need this pattern already have it. If a new file is added that reads/writes to `output/`, `models/`, `cache/`, or `*.db`, it must include this block.

### Spec file gotchas
- **xgboost.testing**: `collect_all("xgboost")` imports `xgboost.testing` which calls `pytest.importorskip("hypothesis")`. Fixed with `filter_submodules=lambda name: "testing" not in name`.
- **Hidden imports**: `src.strategy` must be listed explicitly.

---

## Full Audit Prompt

When the user says **"run full audit"**, execute the following comprehensive audit across the entire codebase. Use parallel agents to cover all files simultaneously. This audit should be run periodically, especially after significant changes.

### Mindset

You are an FPL manager who has entrusted this app with your entire season. Every recommendation — transfers, captains, chips, bench order — must be correct. A single logic bug can cost you hundreds of points over a season. Audit as if your mini-league title depends on it.

### What to check

**1. FPL Rule Compliance**
Does the code correctly implement every FPL rule? Trace through each rule and find the code that handles it. Look for:
- Free transfer banking: +1 per GW, max 5, cost -4 per extra transfer
- Chips: WC and FH preserve FTs at pre-chip count (no reset, no accrual). BB and TC are one-GW only. All 4 chips available once per half (GW1-19 and GW20-38), 8 total per season. First-half chips expire at GW19.
- Captain: doubles points. Vice-captain activates only if captain gets 0 minutes.
- Squad rules: 15 players, 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 from any team
- Formation: 11 starters, 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
- DGW: player plays twice in one GW, points from both matches summed
- BGW: team doesn't play, player scores 0
- DefCon: +2 pts if CBIT >= threshold (10 for GKP/DEF, 12 for MID/FWD)
- Selling prices: you only get 50% of price rises (we use now_cost from public API — note any places this matters)
- Transfers: hits (-4 per extra transfer) should be subtracted from point calculations where applicable

**2. Data Leakage**
For every feature that enters the model, trace it back to its source. At GW N (predicting GW N+1), the feature must only use data from GW N-1 and earlier. Look for:
- Missing `shift(1)` on any rolling/expanding/cumulative calculation
- Merge keys that allow future data to leak (e.g., merging on gameweek without ensuring the data is from the past)
- Target variables appearing in feature columns
- Any feature that could contain information about the match being predicted

**3. Off-by-One Errors**
These are the most common bugs in this codebase. For every calculation that involves gameweek offsets, time periods, or array indices:
- Is the shift in the right direction?
- Does "GW N" mean the same thing everywhere? (Some places: GW N = the GW being predicted. Others: GW N = the current GW whose data we have.)
- Are loop bounds correct? (< vs <=, range(1, n) vs range(0, n))
- Cross-season boundaries: does GW 1 of season 2 correctly follow GW 38 of season 1?

**4. State Tracking Across Multi-Step Simulations**
The multi-week planner, chip evaluator, and season manager all simulate multiple GWs forward. For each simulation:
- Is budget preserved correctly? (It should not shrink when the solver picks a cheaper squad)
- Are FTs tracked correctly per FPL rules? (Banking, spending, chip effects)
- Does the squad update correctly? (WC: permanent. FH: reverts. Normal transfers: permanent.)
- When the solver fails, does state still advance correctly?

**5. Hardcoded Values That Should Be Dynamic**
Search for magic numbers (1000, 100, 38, etc.) and ask: should this be a parameter? Common offenders:
- Budget values (should come from manager's actual bank + squad value)
- GW limits (38 is correct for PL but should be handled at season boundaries)
- Thresholds (probability cutoffs, point thresholds, pool sizes)

**6. DGW / BGW Handling**
DGWs and BGWs are edge cases that break a lot of logic. For every place that processes per-GW data:
- DGW: Are stats summed (not averaged) for the GW? Are predictions summed across both fixtures?
- BGW: Are teams with no fixture handled gracefully? (No division by zero, no missing data)
- Fixture maps: Do they show all fixtures for DGW teams, not just the first?

**7. Injury / Availability Propagation**
When a player is flagged as injured or unavailable:
- Is EVERY prediction column zeroed? (predicted_points, captain_score, Q80, decomposed components)
- Does it propagate to ALL future GWs, not just the immediate one?
- Is the player excluded from transfer recommendations, captain picks, AND chip evaluations?

**8. Cache and Staleness**
The app caches FPL API data (30min) and GitHub data (6h). For every place that reads cached data:
- Is the data fresh enough for the operation? (Recording results needs live data, not 30-min-old cache)
- Could stale data cause incorrect recommendations?
- Are there race conditions between cache refresh and data usage?

**9. Database Integrity**
- Do any INSERT/UPDATE operations risk CASCADE deletes or orphaned rows?
- Are there any SQL operations that silently discard data on conflict?
- Do all queries filter by the correct season_id / manager_id?

**10. Edge Cases and Boundaries**
- Season start (GW1): Are there division-by-zero or empty-data issues when no history exists?
- Season end (GW38): Does `_get_next_gw` return 39? Do planners try to plan beyond GW38?
- New player (no history): Do rolling features handle NaN gracefully?
- Transferred-out player: Does data stop cleanly or does stale data leak?
- Manager with 0 bank: Does the solver handle budget = squad_value correctly?
- Empty prediction DataFrame: Does every function handle this without crashing?

**11. Objective Function Correctness (MILP Solver)**
The solver is the heart of every recommendation. Verify:
- Captain bonus: Does it correctly value doubling the captain's points?
- Bench weight (0.1): Is this applied consistently and does it make sense?
- Are all constraints correct? (Position counts, team cap, formation, budget)
- When captain_col is provided, are the 3n variables (squad, starter, captain) correctly linked?

**12. Common Sense Check**
Step back from the code and think like an FPL manager:
- Would you trust these recommendations with your actual team?
- Does the captain pick make sense? (Highest-scoring available player, not injured, good fixture)
- Does the transfer plan make sense? (Not selling players with great upcoming fixtures)
- Does the chip timing make sense? (BB in DGW, FH in BGW, WC before fixture swing)
- Are price change predictions actually influencing transfer timing?

### Output format

For each issue found, report:
1. **Severity**: Critical / High / Medium
2. **File and line number(s)**
3. **What's wrong** (concrete, specific)
4. **What should happen** (expected behavior)
5. **Suggested fix** (code-level)
6. **Unintended consequences** of the fix (what else might break)

Group findings by file. Deduplicate — if the same pattern appears in multiple places, report it once with all locations listed.

### Important notes

- Do NOT report style issues, missing docstrings, or code quality suggestions. Only report things that produce wrong results or could cause failures.
- DO check the actual code — do not assume anything is correct just because it was correct before.
- If you're unsure whether something is a bug, include it with a note explaining your uncertainty.

---

## Remaining TODO: Rethink Backtesting & Feature Visualization

The following 4 UI buttons have been **removed from the frontend** (but all backend code is preserved):

- **Run Feature Selection** (`/api/feature-selection`) — `src/feature_selection.py` still exists
- **Model Importance** (`/api/xgb-importance`) — endpoint still in `src/app.py`
- **Feature Insights** panel — showed feature charts, reports, and XGBoost importance
- **Backtest** panel — walk-forward backtesting UI with per-GW breakdown

**Why removed**: After the dynamic season handling rewrite (multi-season support, graduated weights, generic data fetching), these features need rethinking.

**To restore**: The backend endpoints and Python files are all intact. Check git history for the removed UI code.

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

### Implementation approach
- Add `src/fpl_auth.py` for authentication + authenticated API calls
- Add execution methods to `SeasonManager` (execute_transfers, set_captain, activate_chip)
- Add `/api/season/execute` endpoint with confirmation flow
- Add "Execute" button to the Action Plan UI section
- Store credentials via environment variables (`FPL_EMAIL`, `FPL_PASSWORD`)
