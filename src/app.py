"""Flask web app for FPL Points Predictor dashboard."""

import io
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 for stdout/stderr on Windows (player names contain non-ASCII)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_file

from src.data_fetcher import (
    CACHE_DIR,
    CACHE_MAX_AGE_SECONDS,
    load_all_data,
    fetch_event_live,
    fetch_fpl_api,
    fetch_manager_entry,
    fetch_manager_picks,
    fetch_manager_history,
)
from src.feature_engineering import build_features, get_feature_columns
from src.feature_selection import generate_xgb_importance_charts, run_feature_selection
from src.model import (
    MODEL_DIR,
    POSITION_GROUPS,
    TARGETS,
    load_model,
    train_all_models,
    train_all_quantile_models,
    train_all_sub_models,
)
from src.backtest import predict_single_gw, run_backtest
from src.predict import OUTPUT_DIR, format_predictions, run_predictions
from src.solver import scrub_nan as _scrub_nan, solve_milp_team as _solve_milp_team, solve_transfer_milp as _solve_transfer_milp, solve_transfer_milp_with_hits as _solve_transfer_milp_with_hits
from src.season_db import SeasonDB
from src.season_manager import SeasonManager, scrub_nan_recursive

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    """Prevent browsers from caching API responses."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Global state for background tasks and SSE
# ---------------------------------------------------------------------------
_task_lock = threading.Lock()
_current_task: dict | None = None  # {"name": str, "thread": Thread}
_sse_queues: list[queue.Queue] = []
_sse_queues_lock = threading.Lock()

# Cached data/features so we don't rebuild every request
_pipeline_cache: dict = {}
_pipeline_lock = threading.Lock()
_backtest_results: dict | None = None


def _broadcast(msg: str, event: str = "progress"):
    """Send an SSE message to all connected clients."""
    data = json.dumps({"message": msg, "event": event})
    with _sse_queues_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


class _PrintCapture:
    """Context manager that captures print() output and broadcasts via SSE."""

    def __init__(self):
        self._original_stdout = None
        self._buffer = io.StringIO()

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout

    def write(self, text):
        try:
            self._original_stdout.write(text)
        except UnicodeEncodeError:
            self._original_stdout.write(text.encode("ascii", errors="replace").decode("ascii"))
        self._buffer.write(text)
        # Broadcast non-empty lines
        stripped = text.strip()
        if stripped:
            _broadcast(stripped)

    def flush(self):
        self._original_stdout.flush()

    @property
    def output(self):
        return self._buffer.getvalue()


def _get_team_map(season: str = "") -> dict[int, str]:
    """Map team_code -> short_name from cached teams CSV."""
    if not season:
        from src.data_fetcher import detect_current_season
        season = detect_current_season()
    teams_csv = CACHE_DIR / f"{season}_teams.csv"
    if teams_csv.exists():
        teams = pd.read_csv(teams_csv, encoding="utf-8")
        return dict(zip(teams["code"], teams["short_name"]))
    return {}


def _get_next_gw() -> int | None:
    """Get the next gameweek number from cached FPL API bootstrap data."""
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not bootstrap_path.exists():
        return None
    data = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    for event in data.get("events", []):
        if event.get("is_next"):
            return event["id"]
    # If no 'is_next', fall back to the one after 'is_current' (capped at 38)
    for event in data.get("events", []):
        if event.get("is_current"):
            return min(event["id"] + 1, 38)
    return None


def _get_next_fixtures(n: int = 3) -> dict[int, list[str]]:
    """Map team_code -> list of next N opponent strings like 'BRE (A)'.

    Uses FPL API fixtures + bootstrap for team ID/code mapping.
    """
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    if not bootstrap_path.exists() or not fixtures_path.exists():
        return {}

    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))

    next_gw = _get_next_gw()
    if not next_gw:
        return {}

    # FPL API uses team 'id' (1-20) in fixtures; predictions.csv has team 'code'
    id_to_name = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    id_to_code = {t["id"]: t["code"] for t in bootstrap["teams"]}

    # Collect upcoming fixtures per team_id, sorted by GW
    upcoming = [f for f in fixtures if not f.get("finished") and f.get("event") and f["event"] >= next_gw]
    upcoming.sort(key=lambda f: f["event"])

    team_fixtures: dict[int, list[str]] = {}  # team_id -> [str]
    for f in upcoming:
        for side, opp_side, tag in [("team_h", "team_a", "H"), ("team_a", "team_h", "A")]:
            tid = f[side]
            if tid not in team_fixtures:
                team_fixtures[tid] = []
            if len(team_fixtures[tid]) < n:
                opp_name = id_to_name.get(f[opp_side], "?")
                team_fixtures[tid].append(f"{opp_name} ({tag})")

    # Convert team_id keys to team_code keys
    result: dict[int, list[str]] = {}
    for tid, flist in team_fixtures.items():
        code = id_to_code.get(tid)
        if code is not None:
            result[code] = flist
    return result


def _load_predictions_from_csv() -> pd.DataFrame | None:
    """Load predictions.csv if it exists."""
    csv_path = OUTPUT_DIR / "predictions.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8")
    return None


def _ensure_pipeline_data(force: bool = False):
    """Load data + build features if not cached (or if forced)."""
    with _pipeline_lock:
        if force or "df" not in _pipeline_cache:
            with _PrintCapture():
                data = load_all_data(force=force)
                df = build_features(data)
                _pipeline_cache["data"] = data
                _pipeline_cache["df"] = df
                _pipeline_cache["feature_cols"] = get_feature_columns(df)


def _run_in_background(name: str, fn):
    """Run fn in a background thread with print capture, guarded by _task_lock."""
    global _current_task

    if not _task_lock.acquire(blocking=False):
        return False

    def wrapper():
        global _current_task
        try:
            _broadcast(f"Starting: {name}", event="task_start")
            with _PrintCapture():
                fn()
            _broadcast(f"Finished: {name}", event="task_done")
        except Exception as exc:
            _broadcast(f"Error: {exc}", event="task_error")
        finally:
            _current_task = None
            _task_lock.release()

    t = threading.Thread(target=wrapper, daemon=True)
    _current_task = {"name": name, "thread": t}
    t.start()
    return True


def _cache_age_seconds() -> float | None:
    """Return age in seconds of the newest file in cache/, or None."""
    if not CACHE_DIR.exists():
        return None
    csvs = list(CACHE_DIR.glob("*.csv")) + list(CACHE_DIR.glob("*.json"))
    if not csvs:
        return None
    newest = max(f.stat().st_mtime for f in csvs)
    return time.time() - newest


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    df = _load_predictions_from_csv()
    if df is None:
        return jsonify({"players": [], "error": "No predictions found. Train models first."})

    team_map = _get_team_map()

    # Add team short name
    if "team_code" in df.columns:
        df["team"] = df["team_code"].map(team_map).fillna("")

    # Add next 3 fixtures
    fixture_map = _get_next_fixtures(3)
    if fixture_map and "team_code" in df.columns:
        df["next_3_fixtures"] = df["team_code"].map(
            lambda tc: ", ".join(fixture_map.get(tc, []))
        )

    # Apply filters
    position = request.args.get("position", "").upper()
    if position and position != "ALL":
        df = df[df["position"] == position]

    search = request.args.get("search", "").strip().lower()
    if search and "web_name" in df.columns:
        df = df[df["web_name"].str.lower().str.contains(search, na=False)]

    sort_by = request.args.get("sort", "predicted_next_gw_points")
    sort_dir = request.args.get("dir", "desc")
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir == "asc"))

    # Round floats for display
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(2)

    players = _scrub_nan(df.to_dict(orient="records"))
    return jsonify({"players": players})


@app.route("/api/refresh-data", methods=["POST"])
def api_refresh_data():
    def do_refresh():
        data = load_all_data(force=True)
        df = build_features(data)
        with _pipeline_lock:
            _pipeline_cache["data"] = data
            _pipeline_cache["df"] = df
            _pipeline_cache["feature_cols"] = get_feature_columns(df)

        # Re-run predictions if 1-GW models exist
        models_exist = all(
            load_model(pos, "next_gw_points") is not None
            for pos in POSITION_GROUPS
        )
        if models_exist:
            print("\nRegenerating predictions with fresh data...")
            preds = run_predictions(df, data=data)
            if not preds.empty:
                result = format_predictions(preds, df)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                result.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8")
                print("Predictions updated.")

        # Auto-replan: check if existing strategic plan needs updating
        try:
            from src.strategy import detect_plan_invalidation
            from src.predict import predict_future_range, get_latest_gw
            from src.feature_engineering import get_fixture_context

            # Check all active seasons for plan invalidation
            bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
            if bootstrap_path.exists():
                bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
                elements = bootstrap.get("elements", [])

                # Build squad changes from bootstrap
                squad_changes = {}
                for el in elements:
                    if el.get("status") != "a" or (el.get("chance_of_playing_next_round") is not None and el["chance_of_playing_next_round"] < 75):
                        squad_changes[el["id"]] = {
                            "status": el.get("status", "a"),
                            "chance_of_playing": el.get("chance_of_playing_next_round"),
                            "web_name": el.get("web_name", "Unknown"),
                        }

                if squad_changes:
                    print(f"\n  {len(squad_changes)} players with availability concerns detected.")
                    _broadcast(f"Availability check: {len(squad_changes)} players flagged", event="progress")

                # Check plan health for all active seasons
                with SeasonDB()._conn_ctx() as conn:
                    seasons = conn.execute("SELECT DISTINCT manager_id FROM season").fetchall()
                for s in seasons:
                    mid = s["manager_id"]
                    health = _season_mgr.check_plan_health(mid)
                    if not health["healthy"]:
                        desc = "; ".join(t["description"] for t in health["triggers"][:3])
                        print(f"  Plan health issue for manager {mid}: {desc}")
                        _broadcast(
                            f"Plan health issue for manager {mid}: {health['summary']}",
                            event="plan_invalidated",
                        )
        except Exception as exc:
            print(f"  Auto-replan check failed (non-fatal): {exc}")

    started = _run_in_background("Refresh Data", do_refresh)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/train", methods=["POST"])
def api_train():
    def do_train():
        _ensure_pipeline_data()
        df = _pipeline_cache["df"]

        print("Step 4: Training models...")
        results = train_all_models(df, tune=True)
        print("\n  Training Summary (mean models):")
        for r in results:
            print(f"    {r['position']:3s} / {r['target']:18s}: MAE = {r['mae']:.3f}")

        print("\n  Training quantile models (for captain picks)...")
        q_results = train_all_quantile_models(df)
        if q_results:
            print("\n  Training Summary (quantile models):")
            for r in q_results:
                print(f"    {r['position']:3s} / q80: MAE = {r['mae']:.3f}")

        print("\n  Training decomposed sub-models...")
        sub_results = train_all_sub_models(df)
        if sub_results:
            print("\n  Training Summary (sub-models):")
            for r in sub_results:
                print(f"    {r['position']:3s} / {r['component']:18s}: MAE = {r['mae']:.4f}")

        print("\nStep 5: Generating predictions...")
        data = _pipeline_cache.get("data")
        preds = run_predictions(df, data=data)
        if not preds.empty:
            result = format_predictions(preds, df)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            result.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8")
            print(f"Predictions saved.")

    started = _run_in_background("Train Models", do_train)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/feature-selection", methods=["POST"])
def api_feature_selection():
    def do_feature_selection():
        _ensure_pipeline_data()
        df = _pipeline_cache["df"]
        feature_cols = _pipeline_cache["feature_cols"]

        print("Step 3: Running feature selection...")
        run_feature_selection(df, feature_cols)

    started = _run_in_background("Feature Selection", do_feature_selection)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/status")
def api_status():
    """SSE endpoint for live progress updates."""
    q: queue.Queue = queue.Queue(maxsize=200)
    with _sse_queues_lock:
        _sse_queues.append(q)

    def stream():
        try:
            # Send initial status (read is atomic for dict ref assignment)
            task = _current_task
            if task:
                task_name = task["name"]
                payload = json.dumps({"message": f"Running: {task_name}", "event": "status"})
                yield f"data: {payload}\n\n"
            else:
                payload = json.dumps({"message": "Idle", "event": "status"})
                yield f"data: {payload}\n\n"
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # keepalive
                    yield f": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_queues_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/model-info")
def api_model_info():
    """Return model metadata: MAE, training date, feature counts."""
    info = []
    for position in POSITION_GROUPS:
        for target in TARGETS:
            model_path = MODEL_DIR / f"xgb_{position}_{target}.joblib"
            entry = {"position": position, "target": target, "exists": False}
            if model_path.exists():
                entry["exists"] = True
                entry["trained_at"] = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(model_path.stat().st_mtime)
                )
                md = load_model(position, target)
                if md:
                    entry["n_features"] = len(md.get("features", []))
            info.append(entry)

    # Cache freshness
    age = _cache_age_seconds()

    # Season info
    from src.data_fetcher import detect_current_season, get_all_seasons
    current = detect_current_season()

    return jsonify({
        "models": info,
        "next_gw": _get_next_gw(),
        "cache_age_seconds": age,
        "cache_max_age_seconds": CACHE_MAX_AGE_SECONDS,
        "current_season": current,
        "available_seasons": get_all_seasons(current),
    })


@app.route("/api/feature-charts/<filename>")
def api_feature_charts(filename):
    """Serve PNG chart files from output/."""
    if not filename.endswith(".png"):
        return jsonify({"error": "Not found"}), 404
    chart_path = (OUTPUT_DIR / filename).resolve()
    # Prevent path traversal — resolved path must be inside OUTPUT_DIR
    if not str(chart_path).startswith(str(OUTPUT_DIR.resolve())):
        return jsonify({"error": "Not found"}), 404
    if not chart_path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(chart_path, mimetype="image/png")


@app.route("/api/feature-report")
def api_feature_report():
    """Return parsed feature importance report."""
    report_path = OUTPUT_DIR / "feature_importance_report.txt"
    if not report_path.exists():
        return jsonify({"error": "No feature report found. Run feature selection first.", "sections": []})

    text = report_path.read_text(encoding="utf-8")

    # List available charts (exclude xgb_ charts — those are served separately)
    charts = sorted(
        f.name for f in OUTPUT_DIR.glob("*.png")
        if not f.name.startswith("xgb_")
    ) if OUTPUT_DIR.exists() else []

    return jsonify({"report": text, "charts": charts})


@app.route("/api/xgb-importance", methods=["POST"])
def api_xgb_importance():
    def do_xgb_importance():
        print("Extracting XGBoost model feature importances...")
        generate_xgb_importance_charts()

    started = _run_in_background("XGBoost Importance", do_xgb_importance)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/xgb-importance-report")
def api_xgb_importance_report():
    """Return parsed XGBoost importance report and chart filenames."""
    report_path = OUTPUT_DIR / "xgb_importance_report.txt"
    if not report_path.exists():
        return jsonify({"error": "No XGBoost importance report found. Run Model Importance first.", "sections": []})

    text = report_path.read_text(encoding="utf-8")

    # Only include xgb_ prefixed charts
    charts = sorted(
        f.name for f in OUTPUT_DIR.glob("xgb_*.png")
    ) if OUTPUT_DIR.exists() else []

    return jsonify({"report": text, "charts": charts})


@app.route("/api/best-team", methods=["POST"])
def api_best_team():
    """Select optimal squad via two-tier MILP optimisation.

    Decision variables: x_i (in squad), s_i (starter).
    Objective: maximise starting 11 points + 0.1 * bench points.
    """
    df = _load_predictions_from_csv()
    if df is None or df.empty:
        return jsonify({"error": "No predictions available. Train models first."}), 400

    body = request.get_json(silent=True) or {}
    try:
        budget = float(body.get("budget", 100.0))
    except (TypeError, ValueError):
        return jsonify({"error": "budget must be a number."}), 400
    if budget <= 0:
        return jsonify({"error": "budget must be positive."}), 400
    target = body.get("target", "predicted_next_gw_points")
    if target not in ("predicted_next_gw_points", "predicted_next_3gw_points"):
        target = "predicted_next_gw_points"

    required = ["position", "cost", target]
    if not all(c in df.columns for c in required):
        return jsonify({"error": "Predictions missing required columns."}), 400

    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        return jsonify({"error": "No valid players."}), 400

    team_map = _get_team_map()
    if "team_code" in df.columns:
        df["team"] = df["team_code"].map(team_map).fillna("")

    fixture_map = _get_next_fixtures(1)
    if fixture_map and "team_code" in df.columns:
        df["opponent"] = df["team_code"].map(
            lambda tc: fixture_map.get(tc, [""])[0] if fixture_map.get(tc) else ""
        )

    captain_col_arg = "captain_score" if "captain_score" in df.columns else None
    result = _solve_milp_team(df, target, budget=budget, captain_col=captain_col_arg)
    if result is None:
        return jsonify({"error": f"Could not find a valid team within budget {budget:.1f}. Try increasing the budget."})

    starters_df = pd.DataFrame(result["starters"])
    gw_col = "predicted_next_gw_points"
    gw3_col = "predicted_next_3gw_points"
    starting_gw = round(starters_df[gw_col].sum(), 2) if gw_col in starters_df.columns else 0
    starting_gw3 = round(starters_df[gw3_col].sum(), 2) if gw3_col in starters_df.columns else 0

    # Add captain bonus (captain doubles points)
    captain_id = result.get("captain_id")
    if captain_id and "player_id" in starters_df.columns:
        cap_row = starters_df[starters_df["player_id"] == captain_id]
        if not cap_row.empty:
            if gw_col in cap_row.columns:
                starting_gw += round(float(cap_row[gw_col].iloc[0]), 2)
            if gw3_col in cap_row.columns:
                starting_gw3 += round(float(cap_row[gw3_col].iloc[0]), 2)

    return jsonify({
        "players": result["players"],
        "total_cost": result["total_cost"],
        "starting_gw_points": starting_gw,
        "starting_gw3_points": starting_gw3,
        "budget": budget,
        "remaining": round(budget - result["total_cost"], 1),
        "target": target,
    })




@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run a comprehensive backtest over a range of gameweeks."""
    global _backtest_results

    body = request.get_json(silent=True) or {}
    try:
        start_gw = int(body.get("start_gw", 5))
        end_gw = int(body.get("end_gw", 25))
    except (TypeError, ValueError):
        return jsonify({"error": "start_gw and end_gw must be integers."}), 400
    if not (1 <= start_gw <= 38) or not (1 <= end_gw <= 38):
        return jsonify({"error": "start_gw and end_gw must be between 1 and 38."}), 400
    if start_gw >= end_gw:
        return jsonify({"error": "start_gw must be less than end_gw."}), 400

    # Optional multi-season support
    seasons = body.get("seasons", None)
    if seasons is not None:
        from src.data_fetcher import detect_current_season, get_all_seasons
        valid_seasons = set(get_all_seasons(detect_current_season()))
        if not isinstance(seasons, list) or not all(isinstance(s, str) for s in seasons):
            return jsonify({"error": "seasons must be a list of strings."}), 400
        invalid = [s for s in seasons if s not in valid_seasons]
        if invalid:
            return jsonify({"error": f"Invalid seasons: {invalid}. Valid: {sorted(valid_seasons)}"}), 400
        if not seasons:
            seasons = None  # Empty list -> default single-season

    def do_backtest():
        global _backtest_results
        _ensure_pipeline_data()
        df = _pipeline_cache["df"]

        _backtest_results = run_backtest(
            df, start_gw=start_gw, end_gw=end_gw, seasons=seasons,
        )

    started = _run_in_background("Backtest", do_backtest)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/backtest-results")
def api_backtest_results():
    if _backtest_results is None:
        return jsonify({"error": "No backtest results. Run a backtest first."}), 404
    return jsonify(_backtest_results)


@app.route("/api/gw-compare", methods=["POST"])
def api_gw_compare():
    """Compare a manager's actual FPL team against the hindsight-best team for a GW."""
    body = request.get_json(silent=True) or {}
    try:
        manager_id = int(body.get("manager_id", 0))
        gameweek = int(body.get("gameweek", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id and gameweek must be integers."}), 400
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    if not (1 <= gameweek <= 38):
        return jsonify({"error": "gameweek must be between 1 and 38."}), 400

    # Fetch manager's actual picks for this GW
    try:
        picks_data = fetch_manager_picks(manager_id, gameweek)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks for GW{gameweek}: {exc}"}), 404

    # Bootstrap for player info
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not bootstrap_path.exists():
        return jsonify({"error": "No cached bootstrap data. Click 'Get Latest Data' first."}), 400
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

    from src.data_fetcher import detect_current_season
    season = body.get("season", detect_current_season())
    team_map = _get_team_map(season)

    # Build feature matrix
    _ensure_pipeline_data()
    df = _pipeline_cache["df"]
    season_gw = df[(df["season"] == season) & (df["gameweek"] == gameweek)]
    if season_gw.empty:
        return jsonify({"error": f"No data for {season} GW{gameweek} in feature matrix."}), 404

    actuals = season_gw[["player_id", "event_points"]].drop_duplicates("player_id", keep="first")
    actuals_map = dict(zip(actuals["player_id"], actuals["event_points"]))

    # Enrich manager's picks
    picks = picks_data.get("picks", [])
    entry_history = picks_data.get("entry_history", {})
    # entry_history "value" is total value INCLUDING bank
    budget = round(entry_history.get("value", 0) / 10, 1)

    my_squad = []
    for pick in picks:
        eid = pick.get("element")
        el = elements_map.get(eid, {})
        team_id = el.get("team")
        tc = team_id_to_code.get(team_id)
        my_squad.append({
            "player_id": eid,
            "web_name": el.get("web_name", "Unknown"),
            "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
            "team_code": tc,
            "team": team_map.get(tc, ""),
            "cost": el.get("now_cost", 0) / 10,
            "actual": actuals_map.get(eid, 0),
            "multiplier": pick.get("multiplier", 1),
            "starter": pick.get("position", 12) <= 11,
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
        })

    my_starters = [p for p in my_squad if p["starter"]]
    my_bench = [p for p in my_squad if not p["starter"]]
    # Captain/VC get multiplier applied to actual points
    my_starting_actual = round(
        sum((p["actual"] or 0) * p["multiplier"] for p in my_starters), 1,
    )

    # Build pool for hindsight-best
    pool_df = season_gw.copy()
    pool_df = pool_df.drop(columns=["position"], errors="ignore")
    pool_df = pool_df.rename(columns={"position_clean": "position", "event_points": "actual"})
    if "team_code" in pool_df.columns:
        pool_df["team"] = pool_df["team_code"].map(team_map).fillna("")
    keep_cols = ["player_id", "web_name", "position", "team_code", "team", "cost", "actual"]
    keep_cols = [c for c in keep_cols if c in pool_df.columns]
    pool_df = pool_df[keep_cols].drop_duplicates("player_id", keep="first")

    best_result = _solve_milp_team(pool_df, "actual", budget=budget)
    if best_result is None:
        return jsonify({"error": "Could not solve hindsight-best team."}), 500

    # Add captain bonus: best scorer among starters gets doubled
    best_starters_pts = [p.get("actual", 0) or 0 for p in best_result["starters"]]
    best_captain_pts = max(best_starters_pts) if best_starters_pts else 0
    best_starting_actual = round(sum(best_starters_pts) + best_captain_pts, 1)

    # Overlap between manager starters and best starters
    my_ids = {p["player_id"] for p in my_starters}
    best_ids = {p["player_id"] for p in best_result["starters"]}
    overlap_ids = sorted(my_ids & best_ids)
    capture_pct = round(
        (my_starting_actual / best_starting_actual) * 100, 1,
    ) if best_starting_actual > 0 else 0

    return jsonify({
        "gameweek": gameweek,
        "budget": budget,
        "my_team": {
            "starters": _scrub_nan(my_starters),
            "bench": _scrub_nan(my_bench),
            "starting_actual": my_starting_actual,
        },
        "best_team": {
            "starters": best_result["starters"],
            "bench": best_result["bench"],
            "starting_actual": best_starting_actual,
        },
        "overlap_player_ids": overlap_ids,
        "overlap_count": len(overlap_ids),
        "capture_pct": capture_pct,
    })


def _calculate_free_transfers(history: dict) -> int:
    """Calculate free transfers available for the next GW.

    Walks through history["current"] (one entry per GW played).
    Each GW entry has event_transfers and event_transfers_cost.
    WC/FH preserve FTs at pre-chip count (no accrual, no consumption).
    """
    current = history.get("current", [])
    chips = history.get("chips", [])
    chip_events = {c["event"] for c in chips if c.get("name") in ("wildcard", "freehit")}

    first_event = current[0].get("event", 1) if current else 1
    ft = 1  # Start of season
    for i, gw_entry in enumerate(current):
        event = gw_entry.get("event")

        if event in chip_events:
            # WC/FH: FTs preserved at pre-chip count, no accrual
            continue

        transfers_made = gw_entry.get("event_transfers", 0)
        transfers_cost = gw_entry.get("event_transfers_cost", 0)

        # How many were paid (4 pts each)
        paid = transfers_cost // 4 if transfers_cost > 0 else 0
        free_used = transfers_made - paid

        ft = ft - free_used
        ft = max(ft, 0)  # Floor at 0 before accrual (consistent with season_manager)

        # Mid-season joiner: first GW's FT was consumed by team creation,
        # so don't bank it (skip the +1 roll for the debut GW).
        if i == 0 and first_event > 1:
            pass  # ft already floored above, skip accrual for debut GW
        else:
            # After this GW, roll forward: gain 1, cap at 5
            ft = min(ft + 1, 5)

    return max(ft, 1)


ELEMENT_TYPE_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


@app.route("/api/my-team")
def api_my_team():
    """Fetch a manager's current squad from the FPL API, enriched with predictions."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    try:
        entry = fetch_manager_entry(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch manager {manager_id}: {exc}"}), 404

    current_event = entry.get("current_event")
    if not current_event:
        # Pre-season: no picks yet
        return jsonify({
            "manager": {
                "id": manager_id,
                "name": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
                "team_name": entry.get("name", ""),
                "overall_rank": None,
                "overall_points": 0,
            },
            "pre_season": True,
            "current_event": None,
            "next_gw": 1,
            "bank": 100.0,
            "squad_value": 0,
            "sell_value": 0,
            "free_transfers": 0,
            "active_chip": None,
            "chips_used": [],
            "squad": [],
            "xi_actual_gw": 0,
            "xi_pred_gw": 0,
            "xi_pred_3gw": 0,
        })

    try:
        picks_data = fetch_manager_picks(manager_id, current_event)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks for GW{current_event}: {exc}"}), 404

    try:
        history = fetch_manager_history(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch manager history: {exc}"}), 404

    free_transfers = _calculate_free_transfers(history)

    # Load bootstrap for element lookup and team mapping
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not bootstrap_path.exists():
        return jsonify({"error": "No cached bootstrap data. Click 'Get Latest Data' first."}), 400

    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    team_map = _get_team_map()

    # Load predictions
    pred_df = _load_predictions_from_csv()
    pred_map = {}
    if pred_df is not None and not pred_df.empty:
        for _, row in pred_df.iterrows():
            pred_map[row.get("player_id")] = row.to_dict()

    # Next fixtures
    fixture_map = _get_next_fixtures(1)
    next_gw = _get_next_gw()

    # Picks entry_history: "value" is total value INCLUDING bank
    entry_history = picks_data.get("entry_history", {})
    bank = entry_history.get("bank", 0) / 10  # tenths -> millions
    squad_value = (entry_history.get("value", 0) - entry_history.get("bank", 0)) / 10

    # Active chip for this GW
    active_chip = picks_data.get("active_chip")

    # Chips used
    chips_used = [
        {"name": c.get("name"), "event": c.get("event")}
        for c in history.get("chips", [])
    ]

    picks = picks_data.get("picks", [])
    squad = []
    for pick in picks:
        element_id = pick.get("element")
        el = elements_map.get(element_id, {})
        team_id = el.get("team")
        team_code = team_id_to_code.get(team_id)
        team_short = team_map.get(team_code, "")
        position = ELEMENT_TYPE_MAP.get(el.get("element_type"), "")

        player = {
            "player_id": element_id,
            "web_name": el.get("web_name", "Unknown"),
            "position": position,
            "team_code": team_code,
            "team": team_short,
            "cost": el.get("now_cost", 0) / 10,
            "event_points_raw": el.get("event_points", 0),
            "multiplier": pick.get("multiplier", 1),
            "event_points": el.get("event_points", 0) * pick.get("multiplier", 1),
            "status": el.get("status", "a"),
            "news": el.get("news", ""),
            "chance_of_playing": el.get("chance_of_playing_next_round"),
            "starter": pick.get("position", 12) <= 11,
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "squad_position": pick.get("position"),
        }

        # Enrich from predictions
        pred = pred_map.get(element_id, {})
        player["predicted_next_gw_points"] = pred.get("predicted_next_gw_points")
        player["predicted_next_3gw_points"] = pred.get("predicted_next_3gw_points")
        player["fdr"] = pred.get("fdr")
        player["captain_score"] = pred.get("captain_score")

        # Next opponent
        if fixture_map and team_code:
            opps = fixture_map.get(team_code, [])
            player["opponent"] = opps[0] if opps else ""
        else:
            player["opponent"] = ""

        squad.append(player)

    # Compute starting XI points
    starters = [p for p in squad if p["starter"]]
    xi_pred_gw = sum(p.get("predicted_next_gw_points") or 0 for p in starters)
    xi_pred_3gw = sum(p.get("predicted_next_3gw_points") or 0 for p in starters)
    xi_actual_gw = sum(p.get("event_points") or 0 for p in starters)

    # Squad value: now_cost sum (matches FPL app) vs selling value (from API)
    squad_value_now_cost = round(sum(p["cost"] for p in squad), 1)

    result = {
        "manager": {
            "id": manager_id,
            "name": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
            "team_name": entry.get("name", ""),
            "overall_rank": entry.get("summary_overall_rank"),
            "overall_points": entry.get("summary_overall_points"),
        },
        "current_event": current_event,
        "next_gw": next_gw,
        "bank": round(bank, 1),
        "squad_value": squad_value_now_cost,
        "sell_value": round(squad_value, 1),
        "free_transfers": free_transfers,
        "active_chip": active_chip,
        "chips_used": chips_used,
        "squad": _scrub_nan(squad),
        "xi_actual_gw": round(xi_actual_gw, 1),
        "xi_pred_gw": round(xi_pred_gw, 2),
        "xi_pred_3gw": round(xi_pred_3gw, 2),
    }

    return jsonify(result)


@app.route("/api/transfer-recommendations", methods=["POST"])
def api_transfer_recommendations():
    """Recommend optimal transfers for a manager's current squad."""
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    max_transfers = body.get("max_transfers", 1)
    try:
        max_transfers = int(max_transfers)
    except (TypeError, ValueError):
        return jsonify({"error": "max_transfers must be an integer."}), 400

    wildcard = body.get("wildcard", False)
    if wildcard:
        max_transfers = 15

    target = body.get("target", "predicted_next_gw_points")
    if target not in ("predicted_next_gw_points", "predicted_next_3gw_points"):
        target = "predicted_next_gw_points"

    # --- Fetch current squad (reuse my-team logic) ---
    try:
        entry = fetch_manager_entry(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch manager {manager_id}: {exc}"}), 404

    current_event = entry.get("current_event")
    if not current_event:
        return jsonify({"error": "Manager has no current event."}), 400

    try:
        picks_data = fetch_manager_picks(manager_id, current_event)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch picks: {exc}"}), 404

    try:
        history = fetch_manager_history(manager_id)
    except Exception as exc:
        return jsonify({"error": f"Could not fetch history: {exc}"}), 404

    free_transfers = _calculate_free_transfers(history)

    # Bootstrap for element lookup
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not bootstrap_path.exists():
        return jsonify({"error": "No cached data. Click 'Get Latest Data' first."}), 400
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    elements_map = {el["id"]: el for el in bootstrap.get("elements", [])}
    team_id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

    entry_history = picks_data.get("entry_history", {})
    bank = entry_history.get("bank", 0) / 10

    # Current squad player IDs and their info
    picks = picks_data.get("picks", [])
    current_squad_ids = set()
    current_squad_map = {}  # player_id -> info
    current_squad_cost = 0.0
    for pick in picks:
        eid = pick.get("element")
        current_squad_ids.add(eid)
        el = elements_map.get(eid, {})
        tid = el.get("team")
        tc = team_id_to_code.get(tid)
        player_cost = el.get("now_cost", 0) / 10
        current_squad_cost += player_cost
        current_squad_map[eid] = {
            "player_id": eid,
            "web_name": el.get("web_name", "Unknown"),
            "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
            "team_code": tc,
            "cost": player_cost,
            "starter": pick.get("position", 12) <= 11,
        }

    # Budget = bank + sum of current squad now_cost
    # This ensures swapping a player only works if the replacement's now_cost
    # fits within the sold player's now_cost + bank. Slightly generous (real
    # selling price can be lower than now_cost) but best we can do without auth.
    total_budget = round(bank + current_squad_cost, 1)

    # --- Load full prediction pool ---
    pred_df = _load_predictions_from_csv()
    if pred_df is None or pred_df.empty:
        return jsonify({"error": "No predictions available. Train models first."}), 400

    required_cols = ["player_id", "position", "cost", target]
    if not all(c in pred_df.columns for c in required_cols):
        return jsonify({"error": "Predictions missing required columns."}), 400

    pool = pred_df.dropna(subset=["position", "cost", target]).copy()

    # Enrich with team names & opponents
    team_map = _get_team_map()
    if "team_code" in pool.columns:
        pool["team"] = pool["team_code"].map(team_map).fillna("")

    fixture_map = _get_next_fixtures(3)
    if fixture_map and "team_code" in pool.columns:
        pool["opponent"] = pool["team_code"].map(
            lambda tc: fixture_map.get(tc, [""])[0] if fixture_map.get(tc) else ""
        )
        pool["next_3_fixtures"] = pool["team_code"].map(
            lambda tc: ", ".join(fixture_map.get(tc, []))
        )

    # --- Compute current XI points (before transfers) ---
    pred_map = {}
    for _, row in pred_df.iterrows():
        pred_map[row.get("player_id")] = row.to_dict()

    current_starters = [p for p in current_squad_map.values() if p["starter"]]
    current_xi_points = round(
        sum(pred_map.get(p["player_id"], {}).get(target, 0) or 0 for p in current_starters), 2
    )

    # Add captain bonus (captain doubles points)
    for pick in picks:
        if pick.get("is_captain"):
            cap_pred = pred_map.get(pick["element"], {}).get(target, 0) or 0
            current_xi_points = round(current_xi_points + cap_pred, 2)
            break

    # --- Solve ---
    captain_col_arg = "captain_score" if "captain_score" in pool.columns else None
    if wildcard:
        result = _solve_transfer_milp(
            pool, current_squad_ids, target,
            budget=total_budget, max_transfers=max_transfers,
            captain_col=captain_col_arg,
        )
    else:
        result = _solve_transfer_milp_with_hits(
            pool, current_squad_ids, target,
            budget=total_budget, free_transfers=free_transfers,
            max_transfers=max_transfers,
            captain_col=captain_col_arg,
        )
    if result is None:
        return jsonify({"error": "Could not find a valid transfer solution."}), 400

    # --- Build transfer lists ---
    transfers_in = []
    transfers_out = []

    out_list = sorted(result["transfers_out_ids"])
    in_list = sorted(result["transfers_in_ids"])

    # Build lookup for new squad
    new_squad_map = {p["player_id"]: p for p in result["players"]}

    # Pair transfers by position for display
    out_players = []
    for pid in out_list:
        info = current_squad_map.get(pid, {})
        info["team"] = team_map.get(info.get("team_code"), "")
        pred = pred_map.get(pid, {})
        info[target] = pred.get(target)
        out_players.append(info)

    in_players = []
    for pid in in_list:
        info = new_squad_map.get(pid, {})
        in_players.append(info)

    # Sort both by position for neat pairing
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    out_players.sort(key=lambda p: (pos_order.get(p.get("position"), 9), -(p.get("cost") or 0)))
    in_players.sort(key=lambda p: (pos_order.get(p.get("position"), 9), -(p.get("cost") or 0)))

    # Pair them up for the OUT -> IN display
    for i, out_p in enumerate(out_players):
        in_p = in_players[i] if i < len(in_players) else {}
        out_p["replaced_by"] = in_p.get("web_name", "?")
        transfers_out.append(out_p)

    for i, in_p in enumerate(in_players):
        out_p = out_players[i] if i < len(out_players) else {}
        in_p["replaces"] = out_p.get("web_name", "?")
        transfers_in.append(in_p)

    n_transfers = len(in_list)
    if "hit_cost" in result:
        points_hit = result["hit_cost"]
    else:
        points_hit = max(0, n_transfers - free_transfers) * 4
    points_gained = round(result["starting_points"] - current_xi_points, 2)
    net_gain = round(points_gained - points_hit, 2)

    next_gw = _get_next_gw()

    return jsonify({
        "transfers_in": _scrub_nan(transfers_in),
        "transfers_out": _scrub_nan(transfers_out),
        "new_squad": {
            "starters": result["starters"],
            "bench": result["bench"],
        },
        "current_xi_points": current_xi_points,
        "new_xi_points": result["starting_points"],
        "points_gained": points_gained,
        "free_transfers": free_transfers,
        "n_transfers": n_transfers,
        "points_hit": points_hit,
        "net_gain": net_gain,
        "budget_before": total_budget,
        "budget_after": result["total_cost"],
        "bank_after": round(total_budget - result["total_cost"], 1),
        "target": target,
        "next_gw": next_gw,
        "wildcard": wildcard,
        "transfers_in_ids": list(result["transfers_in_ids"]),
    })


# ---------------------------------------------------------------------------
# PL Table & GW Scores Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/pl-table")
def api_pl_table():
    """Compute Premier League standings from fixture results."""
    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not fixtures_path.exists() or not bootstrap_path.exists():
        return jsonify({"error": "No cached data. Click 'Get Latest Data' first."}), 400

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))

    # Team ID -> name/short_name mapping
    team_info = {t["id"]: {"name": t["name"], "short": t["short_name"]} for t in bootstrap.get("teams", [])}

    # Accumulate stats from finished fixtures
    stats = {}  # team_id -> {p, w, d, l, gf, ga, pts, form: []}
    for t_id in team_info:
        stats[t_id] = {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "form": []}

    finished = [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]
    # Sort by event then kickoff for correct form ordering
    finished.sort(key=lambda f: (f.get("event", 0), f.get("kickoff_time", "")))

    for f in finished:
        h = f["team_h"]
        a = f["team_a"]
        hs = f["team_h_score"]
        as_ = f["team_a_score"]

        if h not in stats or a not in stats:
            continue

        stats[h]["p"] += 1
        stats[h]["gf"] += hs
        stats[h]["ga"] += as_
        stats[a]["p"] += 1
        stats[a]["gf"] += as_
        stats[a]["ga"] += hs

        if hs > as_:
            stats[h]["w"] += 1
            stats[h]["pts"] += 3
            stats[h]["form"].append("W")
            stats[a]["l"] += 1
            stats[a]["form"].append("L")
        elif hs < as_:
            stats[a]["w"] += 1
            stats[a]["pts"] += 3
            stats[a]["form"].append("W")
            stats[h]["l"] += 1
            stats[h]["form"].append("L")
        else:
            stats[h]["d"] += 1
            stats[h]["pts"] += 1
            stats[h]["form"].append("D")
            stats[a]["d"] += 1
            stats[a]["pts"] += 1
            stats[a]["form"].append("D")

    # Build table sorted by Pts desc, GD desc, GF desc
    table = []
    for t_id, s in stats.items():
        info = team_info.get(t_id, {})
        gd = s["gf"] - s["ga"]
        table.append({
            "team_id": t_id,
            "team": info.get("name", ""),
            "short": info.get("short", ""),
            "p": s["p"], "w": s["w"], "d": s["d"], "l": s["l"],
            "gf": s["gf"], "ga": s["ga"], "gd": gd, "points": s["pts"],
            "form": s["form"][-5:],  # last 5 results
        })
    table.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))

    # Add position
    for i, row in enumerate(table):
        row["pos"] = i + 1

    return jsonify({"table": table})


@app.route("/api/gw-scores")
def api_gw_scores():
    """Match details for a specific gameweek."""
    gameweek = request.args.get("gameweek", type=int)
    if not gameweek:
        # Default to latest finished GW
        next_gw = _get_next_gw()
        gameweek = (next_gw - 1) if next_gw and next_gw > 1 else 1

    fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
    bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
    if not fixtures_path.exists() or not bootstrap_path.exists():
        return jsonify({"error": "No cached data. Click 'Get Latest Data' first."}), 400

    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))

    team_info = {t["id"]: {"name": t["name"], "short": t["short_name"]} for t in bootstrap.get("teams", [])}
    element_map = {el["id"]: el.get("web_name", "Unknown") for el in bootstrap.get("elements", [])}

    gw_fixtures = [f for f in fixtures if f.get("event") == gameweek]
    gw_fixtures.sort(key=lambda f: f.get("kickoff_time") or "")

    def _extract_stat(stats_list, stat_name, side):
        """Extract stat entries for home ('h') or away ('a') side."""
        for stat in stats_list:
            if stat.get("identifier") == stat_name:
                return stat.get(side, [])
        return []

    matches = []
    for f in gw_fixtures:
        h_id = f["team_h"]
        a_id = f["team_a"]
        h_info = team_info.get(h_id, {})
        a_info = team_info.get(a_id, {})
        stats_list = f.get("stats", [])

        match = {
            "home_team": h_info.get("name", ""),
            "home_short": h_info.get("short", ""),
            "away_team": a_info.get("name", ""),
            "away_short": a_info.get("short", ""),
            "home_score": f.get("team_h_score"),
            "away_score": f.get("team_a_score"),
            "kickoff": f.get("kickoff_time"),
            "finished": f.get("finished", False),
            "started": f.get("started", False),
        }

        # Extract match events from stats
        for stat_key, output_key in [
            ("goals_scored", "goals"),
            ("assists", "assists"),
            ("yellow_cards", "yellow_cards"),
            ("red_cards", "red_cards"),
            ("bonus", "bonus"),
        ]:
            home_entries = _extract_stat(stats_list, stat_key, "h")
            away_entries = _extract_stat(stats_list, stat_key, "a")

            if stat_key == "bonus":
                match[output_key] = {
                    "home": [{"player": element_map.get(e["element"], "?"), "points": e["value"]} for e in home_entries],
                    "away": [{"player": element_map.get(e["element"], "?"), "points": e["value"]} for e in away_entries],
                }
            elif stat_key in ("goals_scored",):
                match[output_key] = {
                    "home": [{"player": element_map.get(e["element"], "?"), "count": e["value"]} for e in home_entries],
                    "away": [{"player": element_map.get(e["element"], "?"), "count": e["value"]} for e in away_entries],
                }
            else:
                match[output_key] = {
                    "home": [{"player": element_map.get(e["element"], "?")} for e in home_entries],
                    "away": [{"player": element_map.get(e["element"], "?")} for e in away_entries],
                }

        # Own goals
        og_home = _extract_stat(stats_list, "own_goals", "h")
        og_away = _extract_stat(stats_list, "own_goals", "a")
        match["own_goals"] = {
            "home": [{"player": element_map.get(e["element"], "?")} for e in og_home],
            "away": [{"player": element_map.get(e["element"], "?")} for e in og_away],
        }

        matches.append(match)

    return jsonify({"matches": matches, "gameweek": gameweek})


# ---------------------------------------------------------------------------
# Season Management Endpoints
# ---------------------------------------------------------------------------

_season_db = SeasonDB()
_season_mgr = SeasonManager(_season_db)


@app.route("/api/season/init", methods=["POST"])
def api_season_init():
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    def do_init():
        _season_mgr.initialize_season(manager_id, progress_fn=print)

    started = _run_in_background("Season Init", do_init)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/season/status")
def api_season_status():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"active": False})
    return jsonify({"active": True, "season": season})


@app.route("/api/season/delete", methods=["DELETE"])
def api_season_delete():
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    _season_db.delete_season(manager_id)
    return jsonify({"status": "deleted"})


@app.route("/api/season/dashboard")
def api_season_dashboard():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    dashboard = _season_mgr.get_season_dashboard(manager_id)
    return jsonify(dashboard)


@app.route("/api/season/generate-recommendation", methods=["POST"])
def api_season_generate_recommendation():
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    def do_recommend():
        _season_mgr.generate_recommendation(manager_id, progress_fn=print)

    started = _run_in_background("Generate Recommendation", do_recommend)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/season/record-results", methods=["POST"])
def api_season_record_results():
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    def do_record():
        _season_mgr.record_actual_results(manager_id, progress_fn=print)

    started = _run_in_background("Record Results", do_record)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/season/recommendations")
def api_season_recommendations():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404
    recs = _season_db.get_recommendations(season["id"])
    return jsonify({"recommendations": recs})


@app.route("/api/season/snapshots")
def api_season_snapshots():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404
    snaps = _season_db.get_snapshots(season["id"])
    return jsonify({"snapshots": snaps})


@app.route("/api/season/fixtures")
def api_season_fixtures():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    from_gw = request.args.get("from_gw", type=int)
    to_gw = request.args.get("to_gw", type=int)
    fixtures = _season_db.get_fixture_calendar(season["id"], from_gw=from_gw, to_gw=to_gw)
    return jsonify({"fixtures": fixtures})


@app.route("/api/season/chips")
def api_season_chips():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    chips_used = _season_db.get_chips_status(season["id"])

    # Determine current half from next GW
    next_gw = _get_next_gw()
    if next_gw is None:
        next_gw = 1
    current_half_start = 1 if next_gw <= 19 else 20
    current_half_end = 19 if next_gw <= 19 else 38
    other_half_start = 20 if next_gw <= 19 else 1
    other_half_end = 38 if next_gw <= 19 else 19

    all_chips = [
        {"name": "wildcard", "label": "Wildcard"},
        {"name": "freehit", "label": "Free Hit"},
        {"name": "bboost", "label": "Bench Boost"},
        {"name": "3xc", "label": "Triple Captain"},
    ]
    result = []
    for chip in all_chips:
        chip_uses = [c for c in chips_used if c["chip_used"] == chip["name"]]
        current_half_use = next(
            (c for c in chip_uses if current_half_start <= c["gameweek"] <= current_half_end), None
        )
        other_half_use = next(
            (c for c in chip_uses if other_half_start <= c["gameweek"] <= other_half_end), None
        )
        result.append({
            **chip,
            "used": current_half_use is not None,
            "used_gw": current_half_use["gameweek"] if current_half_use else None,
            "used_other_half": other_half_use is not None,
            "used_other_gw": other_half_use["gameweek"] if other_half_use else None,
        })

    # Get latest recommendation for chip values
    recs = _season_db.get_recommendations(season["id"])
    chip_values = {}
    if recs:
        latest_rec = recs[-1]
        try:
            chip_values = json.loads(latest_rec.get("chip_values_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

    return jsonify({"chips": result, "chip_values": chip_values})


@app.route("/api/season/prices")
def api_season_prices():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    prices = _season_db.get_latest_prices(season["id"])
    alerts = _season_mgr.get_price_alerts(season["id"])
    return jsonify({"prices": prices, "alerts": alerts})


@app.route("/api/season/update-prices", methods=["POST"])
def api_season_update_prices():
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    _season_mgr.track_prices(season["id"], manager_id)
    return jsonify({"status": "ok"})


@app.route("/api/season/update-fixtures", methods=["POST"])
def api_season_update_fixtures():
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    _season_mgr.update_fixture_calendar(season["id"])
    return jsonify({"status": "ok"})


@app.route("/api/season/bank-analysis")
def api_season_bank_analysis():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    recs = _season_db.get_recommendations(season["id"])
    if not recs:
        return jsonify({"error": "No recommendations yet."}), 404

    latest = recs[-1]
    try:
        analysis = json.loads(latest.get("bank_analysis_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        analysis = {}
    return jsonify({"bank_analysis": analysis, "gameweek": latest.get("gameweek")})


@app.route("/api/season/gw-detail")
def api_season_gw_detail():
    manager_id = request.args.get("manager_id")
    gameweek = request.args.get("gameweek", type=int)
    if not manager_id or not gameweek:
        return jsonify({"error": "manager_id and gameweek are required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    snapshot = _season_db.get_snapshot(season["id"], gameweek)
    rec = _season_db.get_recommendation(season["id"], gameweek)
    return jsonify({
        "snapshot": snapshot,
        "recommendation": rec,
    })


@app.route("/api/season/transfer-history")
def api_season_transfer_history():
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    history = _season_db.get_transfer_history(season["id"])
    return jsonify({"transfer_history": history})


# ---------------------------------------------------------------------------
# Strategic Planning Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/season/strategic-plan", methods=["GET", "POST"])
def api_season_strategic_plan():
    """Generate (POST) or fetch (GET) strategic plan."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        manager_id = body.get("manager_id")
        if not manager_id:
            return jsonify({"error": "manager_id is required."}), 400
        try:
            manager_id = int(manager_id)
        except (TypeError, ValueError):
            return jsonify({"error": "manager_id must be an integer."}), 400

        def do_plan():
            _season_mgr.generate_recommendation(manager_id, progress_fn=print)

        started = _run_in_background("Strategic Plan", do_plan)
        if not started:
            return jsonify({"error": "Another task is already running."}), 409
        return jsonify({"status": "started"})

    # GET
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    plan_row = _season_db.get_strategic_plan(season["id"])
    if not plan_row:
        return jsonify({"error": "No strategic plan generated yet."}), 404

    plan = {}
    heatmap = {}
    try:
        plan = json.loads(plan_row.get("plan_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        heatmap = json.loads(plan_row.get("chip_heatmap_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    changelog = _season_db.get_plan_changelog(season["id"], limit=20)

    return jsonify({
        "plan": plan,
        "chip_heatmap": heatmap,
        "as_of_gw": plan_row.get("as_of_gw"),
        "created_at": plan_row.get("created_at"),
        "changelog": changelog,
    })


@app.route("/api/season/chip-heatmap")
def api_season_chip_heatmap():
    """Chip values across remaining GWs."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    plan_row = _season_db.get_strategic_plan(season["id"])
    if not plan_row:
        return jsonify({"error": "No strategic plan generated yet."}), 404

    heatmap = {}
    try:
        heatmap = json.loads(plan_row.get("chip_heatmap_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass

    # Find best GW for each chip
    best_gws = {}
    for chip, gw_vals in heatmap.items():
        if gw_vals:
            # gw_vals keys are strings from JSON
            best_gw = max(gw_vals.items(), key=lambda x: x[1])
            best_gws[chip] = {"gw": best_gw[0], "value": best_gw[1]}

    return jsonify({
        "chip_heatmap": heatmap,
        "best_gws": best_gws,
        "as_of_gw": plan_row.get("as_of_gw"),
    })


@app.route("/api/season/action-plan")
def api_season_action_plan():
    """Build a clear action plan from the latest recommendation."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    result = _season_mgr.get_action_plan(manager_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/season/outcomes")
def api_season_outcomes():
    """Return all recorded outcomes for the season."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    outcomes = _season_mgr.get_outcomes(manager_id)
    return jsonify({"outcomes": outcomes})


@app.route("/api/preseason/generate", methods=["POST"])
def api_preseason_generate():
    """Generate pre-season plan (initial squad + chip plan)."""
    body = request.get_json(silent=True) or {}
    manager_id = body.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    def do_preseason():
        _season_mgr.generate_preseason_plan(
            manager_id, progress_fn=print,
        )

    started = _run_in_background("Pre-Season Plan", do_preseason)
    if not started:
        return jsonify({"error": "Another task is already running."}), 409
    return jsonify({"status": "started"})


@app.route("/api/preseason/result")
def api_preseason_result():
    """Fetch the pre-season plan (initial squad + chip plan)."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No pre-season plan generated yet."}), 404

    # Get GW1 recommendation which contains the initial squad
    rec = _season_db.get_recommendation(season["id"], 1)
    if not rec:
        return jsonify({"error": "No pre-season plan generated yet."}), 404

    squad = []
    try:
        squad = json.loads(rec.get("new_squad_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        pass

    # Get chip plan
    plan_row = _season_db.get_strategic_plan(season["id"])
    chip_schedule = {}
    chip_heatmap = {}
    if plan_row:
        try:
            plan = json.loads(plan_row.get("plan_json") or "{}")
            chip_schedule = plan.get("chip_schedule", {})
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            chip_heatmap = json.loads(plan_row.get("chip_heatmap_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

    return jsonify({
        "initial_squad": squad,
        "predicted_points": rec.get("predicted_points"),
        "captain": {"id": rec.get("captain_id"), "name": rec.get("captain_name")},
        "chip_schedule": chip_schedule,
        "chip_heatmap": chip_heatmap,
    })


@app.route("/api/season/plan-health")
def api_season_plan_health():
    """Check if the current strategic plan is still valid."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    result = _season_mgr.check_plan_health(manager_id)
    return jsonify(result)


@app.route("/api/season/price-predictions")
def api_season_price_predictions():
    """Predict price changes using ownership-based algorithm."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    predictions = _season_mgr.predict_price_changes(season["id"])
    risers = [p for p in predictions if p["direction"] == "rise"]
    fallers = [p for p in predictions if p["direction"] == "fall"]
    return jsonify({"predictions": predictions, "risers": risers, "fallers": fallers})


@app.route("/api/season/price-history")
def api_season_price_history():
    """Get price movement history for players."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    player_ids_str = request.args.get("player_ids", "")
    player_ids = None
    if player_ids_str:
        try:
            player_ids = [int(x.strip()) for x in player_ids_str.split(",") if x.strip()]
        except ValueError:
            return jsonify({"error": "player_ids must be comma-separated integers."}), 400

    days = request.args.get("days", 14, type=int)

    history = _season_mgr.get_price_history(season["id"], player_ids=player_ids, days=days)
    return jsonify({"history": history})


@app.route("/api/season/plan-changelog")
def api_season_plan_changelog():
    """Return plan change history."""
    manager_id = request.args.get("manager_id")
    if not manager_id:
        return jsonify({"error": "manager_id is required."}), 400
    try:
        manager_id = int(manager_id)
    except (TypeError, ValueError):
        return jsonify({"error": "manager_id must be an integer."}), 400

    season = _season_db.get_season(manager_id)
    if not season:
        return jsonify({"error": "No active season."}), 404

    changelog = _season_db.get_plan_changelog(season["id"])
    return jsonify({"changelog": changelog})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9875, debug=True, threaded=True)
