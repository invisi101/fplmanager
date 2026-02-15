# FPL Gaffer Brain

Autonomous Fantasy Premier League manager powered by XGBoost predictions and MILP optimization. Plans transfers, captaincy, and chip strategy across a rolling 5-gameweek horizon.

Built with Python, Flask, scipy MILP, and a single-file vanilla JS frontend.

---

## What It Does

- **Predicts** individual player points using position-specific XGBoost models (mean + quantile + decomposed sub-models), with 100+ engineered features per player per gameweek
- **Optimizes** squad selection, transfers, and captaincy jointly via mixed-integer linear programming
- **Plans ahead** with a 5-GW rolling transfer planner that considers FT banking, fixture swings, and price movements
- **Evaluates chips** (Wildcard, Bench Boost, Triple Captain, Free Hit) across every remaining GW with DGW/BGW awareness and synergy detection
- **Tracks your season** — records recommendations vs actual results, rank trajectory, budget evolution, and model accuracy over time
- **Reacts** to injuries, fixture changes, and prediction shifts with auto-replan detection and alerts

---

## Installation

### Windows (standalone EXE)

1. Download `FPL-Predictor-Windows.zip` from the [latest release](https://github.com/invisi101/fplmanager/releases/latest)
2. Extract the zip
3. Run `FPL Predictor.exe` — the app opens in your browser automatically

No Python installation required.

### Mac / Linux (from source)

```bash
git clone https://github.com/invisi101/fplmanager.git
cd fplmanager
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

Open http://127.0.0.1:9875 in your browser.


---

## First Run

1. **Get Latest Data** — downloads player stats, fixtures, and team data from the FPL API and GitHub historical CSVs
2. **Train Models** — trains XGBoost models for all positions (takes a few minutes)
3. Predictions appear automatically — you're ready to go

Retrain periodically as the season progresses.

---

## Features

### Predictions

Sortable, searchable table of every player's predicted points. Filter by position. Columns include cost, form, predicted next GW and 3 GW points, fixture difficulty, upcoming opponents, and captain scores.

### Best Team

MILP solver builds the optimal 15-player squad within budget, with joint captain optimization. Displayed on an interactive pitch visualization.

### GW Compare

Pick any past gameweek and see your actual FPL team side-by-side with the highest-scoring possible team for that GW. Shows dual pitch visualization with overlap highlighting, capture percentage, and bench for both sides.

### My Team

Import your FPL squad by manager ID. Dual-pitch view showing actual GW points (with captain multiplier) alongside predicted next GW points. Shows squad value, bank, and free transfers.

**Transfer Recommender** — finds optimal transfers using the MILP solver. Set max transfers, choose 1GW or 3GW optimization, optionally enable Wildcard mode. Shows each transfer with predicted points gained, hit cost, and net gain.

### Season Manager

Track your entire FPL season from any gameweek:

- **Overview** — rank trajectory, points-per-GW, budget evolution, and model accuracy charts
- **Strategy** — 5-GW transfer timeline with captain plan, chip schedule with synergy annotations, chip heatmap, and plan changelog. Auto-replan alerts when injuries or fixture changes invalidate the current plan
- **Weekly Workflow** — generate transfer/captain/chip recommendations, then record actual results after the gameweek
- **Fixtures** — FDR-colored grid for all 20 teams with DGW/BGW detection
- **Prices** — ownership-based price change predictions with probability scores and price history charts
- **Transfer History** — complete log with cost, hits, and recommendation adherence
- **Chips** — tracks usage and estimates remaining chip value


---

## CLI

```bash
python -m src.predict                      # predictions only
python -m src.predict --train --tune       # train models then predict
python -m src.predict --feature-selection  # run feature selection
python -m src.predict --force-fetch        # force re-fetch all data
```

---

## Architecture

Three-layer system: **Data → Features/Models → Strategy/Solver**, backed by SQLite and served via Flask.

| Layer | Components |
|-------|-----------|
| Data | FPL API + GitHub CSVs, cached (30m / 6h), 100+ features per player per GW |
| Models | 4 position-specific mean models, 2 quantile (Q80) for captaincy, ~20 decomposed sub-models |
| Strategy | ChipEvaluator, MultiWeekPlanner (5-GW), CaptainPlanner, PlanSynthesizer, reactive re-planning |
| Solver | scipy MILP with joint captain optimization (3n decision variables) |
| Storage | SQLite with 8 tables for season tracking, plans, outcomes, price history |
| Frontend | Single HTML file, vanilla JS, dark theme, canvas charts, SSE for live progress |
