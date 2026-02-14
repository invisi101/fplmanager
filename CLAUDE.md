# FPL Manager — Claude Code Notes

## Project Goal

Build a fully autonomous FPL manager. This is NOT just a prediction tool — it should think and plan like a real FPL manager across the entire season:

- **Transfer planning**: Consider holding transfers for 2 free transfers next week, price rises/falls, when to move early vs wait
- **Squad building**: Shape the squad toward upcoming fixture runs, not just next GW
- **Captain planning**: Know who you're captaining in advance based on fixture calendar
- **Chip strategy**: Plan all 4 chips (Bench Boost, Triple Captain, Free Hit, Wildcard) across remaining GWs — evaluate each chip's value in every future GW using DGW/BGW awareness, squad-specific predictions, and chip synergies (e.g. Wildcard to set up Bench Boost)
- **Reactive adjustments**: When players get injured or fixtures change, re-evaluate the entire plan — not just next week but downstream impact on transfers, chips, and captaincy

Every decision (transfer, captain, bench order, chip) should be made in context of the bigger picture. The app should produce a rolling multi-week plan that constantly recalculates as new information comes in.

## Repos

- **This project**: `https://github.com/invisi101/fplmanager` (active development)
- **Original predictor**: `https://github.com/invisi101/fplxti` (previous project this was forked from)

## Environment

- **Python**: Use `.venv/bin/python`, NOT system `python3` (system Python lacks project dependencies)
- **Run server**: `.venv/bin/python -m src.app` (serves on `http://127.0.0.1:9875`)
- **Port 9875**: Often has leftover processes from previous sessions. Kill with `lsof -ti:9875 | xargs kill -9` before starting
- **No build step**: Frontend is a single file at `src/templates/index.html` (inline CSS + JS). Just edit and refresh.

## Project Structure

- `src/app.py` — Flask app, API endpoints, MILP solvers
- `src/templates/index.html` — Entire frontend (single file)
- `src/data_fetcher.py` — Data fetching + caching
- `src/feature_engineering.py` — 100+ features per player per GW
- `src/model.py` — XGBoost training (mean, quantile, sub-models)
- `src/predict.py` — Prediction pipeline
- `src/backtest.py` — Walk-forward backtesting
- `src/strategy.py` — Strategic brain: ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer, reactive re-planning
- `src/season_manager.py` — Season tracking, recommendations, strategic plan orchestration
- `src/season_db.py` — SQLite database for season data (includes strategic_plan + plan_changelog tables)
- `src/solver.py` — MILP solver for optimal team selection
- `models/` — Saved .joblib model files (gitignored)
- `output/` — predictions.csv, charts, locked_teams.json (gitignored)
- `cache/` — Cached data (6h for GitHub CSVs, 30m for FPL API) (gitignored)

## Testing the App

1. Kill any existing process on port 9876
2. Start with `.venv/bin/python -m src.app`
3. Only one background task runs at a time (train, backtest, etc.)
4. Test API with curl, e.g.: `curl -s http://127.0.0.1:9875/api/my-team?manager_id=12904702`

## My Manager ID

12904702

## Strategic Planning Brain

The app now includes a full strategic planning system in `src/strategy.py`:

- **ChipEvaluator**: Evaluates all 4 chips across every remaining GW, producing a chip heatmap. Uses model predictions for near-term GWs and fixture-calendar heuristics for distant GWs. Detects WC→BB synergies.
- **MultiWeekPlanner**: Rolling 3-GW transfer planner with forward simulation. Considers FT banking, fixture swings, and price awareness. Reduces player pool to top ~150 for efficiency.
- **CaptainPlanner**: Pre-plans captaincy across the horizon, flags weak captain GWs.
- **PlanSynthesizer**: Combines all plans into a coherent timeline with chip schedule and natural-language rationale.
- **Reactive Re-planning**: `detect_plan_invalidation()` checks for injuries, fixture changes, and prediction shifts that invalidate the current plan. `apply_availability_adjustments()` zeros predictions for injured/doubtful players.

### Key API Endpoints
- `POST /api/season/strategic-plan` — Generate full strategic plan (background task)
- `GET /api/season/strategic-plan?manager_id=ID` — Fetch latest plan
- `GET /api/season/chip-heatmap?manager_id=ID` — Chip values heatmap
- `GET /api/season/plan-changelog?manager_id=ID` — Plan change history

### UI
The "Strategy" sub-tab under Season shows:
- Transfer timeline (3-5 GW cards with FT strategy, transfers, captain)
- Captain plan badges
- Chip schedule with synergy annotations
- Full chip heatmap table (color-coded by value)
- Plan changelog

## Full Project Context

See `CLAUDE_PROMPT.md` for comprehensive architecture, model details, and roadmap.
