"""Business logic for season-long FPL management."""

import json
import time
from pathlib import Path

from src.data_fetcher import (
    CACHE_DIR,
    FPL_API_BASE,
    fetch_manager_entry,
    fetch_manager_history,
    fetch_manager_picks,
)
from src.season_db import SeasonDB
from src.solver import scrub_nan, solve_milp_team, solve_transfer_milp
from src.strategy import (
    ChipEvaluator,
    MultiWeekPlanner,
    CaptainPlanner,
    PlanSynthesizer,
    apply_availability_adjustments,
    detect_plan_invalidation,
)

ELEMENT_TYPE_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
ALL_CHIPS = {"wildcard", "freehit", "bboost", "3xc"}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def scrub_nan_recursive(obj):
    """Recursively replace NaN/inf with None in nested structures."""
    import math
    if isinstance(obj, dict):
        return {k: scrub_nan_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [scrub_nan_recursive(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


class SeasonManager:
    """Orchestrates season-long FPL management."""

    def __init__(self, db: SeasonDB | None = None):
        self.db = db or SeasonDB()

    # -------------------------------------------------------------------
    # Bootstrap helpers
    # -------------------------------------------------------------------

    def _load_bootstrap(self) -> dict:
        bootstrap_path = CACHE_DIR / "fpl_api_bootstrap.json"
        if not bootstrap_path.exists():
            raise FileNotFoundError("No cached bootstrap data. Click 'Get Latest Data' first.")
        return json.loads(bootstrap_path.read_text(encoding="utf-8"))

    def _load_fixtures(self) -> list:
        fixtures_path = CACHE_DIR / "fpl_api_fixtures.json"
        if not fixtures_path.exists():
            return []
        return json.loads(fixtures_path.read_text(encoding="utf-8"))

    def _get_elements_map(self, bootstrap: dict) -> dict:
        return {el["id"]: el for el in bootstrap.get("elements", [])}

    def _get_team_maps(self, bootstrap: dict):
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}
        return id_to_code, id_to_short, code_to_short

    def _get_next_gw(self, bootstrap: dict) -> int | None:
        for event in bootstrap.get("events", []):
            if event.get("is_next"):
                return event["id"]
        for event in bootstrap.get("events", []):
            if event.get("is_current"):
                return event["id"] + 1
        return None

    def _calculate_free_transfers(self, history: dict) -> int:
        current = history.get("current", [])
        chips = history.get("chips", [])
        chip_events = {c["event"] for c in chips if c.get("name") in ("wildcard", "freehit")}

        first_event = current[0].get("event", 1) if current else 1
        ft = 1
        for i, gw_entry in enumerate(current):
            event = gw_entry.get("event")
            transfers_made = gw_entry.get("event_transfers", 0)
            transfers_cost = gw_entry.get("event_transfers_cost", 0)
            paid = transfers_cost // 4 if transfers_cost > 0 else 0
            free_used = transfers_made - paid
            ft = ft - free_used
            # Mid-season joiner: first GW's FT was consumed by team creation
            if i == 0 and first_event > 1:
                ft = max(ft, 0)
            else:
                ft = min(ft + 1, 5)
            if event in chip_events:
                ft = 1
        return max(ft, 1)

    # -------------------------------------------------------------------
    # Initialize Season
    # -------------------------------------------------------------------

    def initialize_season(self, manager_id: int, progress_fn=None) -> dict:
        """Backfill season history from FPL API.

        Args:
            manager_id: FPL manager ID
            progress_fn: Optional callable(message: str) for progress updates

        Returns:
            Dict with season info.
        """
        def log(msg):
            if progress_fn:
                progress_fn(msg)
            else:
                print(msg)

        log(f"Fetching manager {manager_id} info...")
        entry = fetch_manager_entry(manager_id)
        manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
        team_name = entry.get("name", "")
        current_event = entry.get("current_event")

        if not current_event:
            raise ValueError("Manager has no current event (season not started?).")

        log(f"Manager: {manager_name} ({team_name})")
        log(f"Current GW: {current_event}")

        log("Fetching season history...")
        history = fetch_manager_history(manager_id)
        gw_entries = history.get("current", [])
        chips_used = history.get("chips", [])

        # Create season record
        start_gw = gw_entries[0]["event"] if gw_entries else 1
        season_id = self.db.create_season(
            manager_id=manager_id,
            manager_name=manager_name,
            team_name=team_name,
            season_name="2025-2026",
            start_gw=start_gw,
        )
        self.db.update_season_gw(season_id, current_event)

        log(f"Season created (ID: {season_id}). Backfilling {len(gw_entries)} gameweeks...")

        # Build chip map: event -> chip_name
        chip_map = {c["event"]: c["name"] for c in chips_used}

        # Load bootstrap for element lookups
        bootstrap = self._load_bootstrap()
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, id_to_short, code_to_short = self._get_team_maps(bootstrap)

        # Backfill each played GW
        for i, gw_data in enumerate(gw_entries):
            gw = gw_data["event"]
            log(f"  Backfilling GW{gw} ({i+1}/{len(gw_entries)})...")

            # Fetch picks for this GW
            try:
                picks_data = fetch_manager_picks(manager_id, gw)
                time.sleep(0.3)  # Rate limiting
            except Exception as exc:
                log(f"    Warning: Could not fetch picks for GW{gw}: {exc}")
                picks_data = {}

            # Build squad JSON
            picks = picks_data.get("picks", [])
            squad = []
            captain_id = None
            captain_name = None
            for pick in picks:
                eid = pick.get("element")
                el = elements_map.get(eid, {})
                tid = el.get("team")
                tc = id_to_code.get(tid)
                ts = code_to_short.get(tc, "")
                pos = ELEMENT_TYPE_MAP.get(el.get("element_type"), "")

                player = {
                    "player_id": eid,
                    "web_name": el.get("web_name", "Unknown"),
                    "position": pos,
                    "team_code": tc,
                    "team": ts,
                    "cost": el.get("now_cost", 0) / 10,
                    "starter": pick.get("position", 12) <= 11,
                    "is_captain": pick.get("is_captain", False),
                    "multiplier": pick.get("multiplier", 1),
                }
                squad.append(player)

                if pick.get("is_captain"):
                    captain_id = eid
                    captain_name = el.get("web_name", "Unknown")

            # Extract transfers
            entry_hist = picks_data.get("entry_history", {})

            self.db.save_gw_snapshot(
                season_id=season_id,
                gameweek=gw,
                squad_json=json.dumps(squad) if squad else None,
                bank=entry_hist.get("bank", gw_data.get("bank", 0)) / 10 if entry_hist else gw_data.get("bank", 0) / 10,
                team_value=entry_hist.get("value", gw_data.get("value", 0)) / 10 if entry_hist else gw_data.get("value", 0) / 10,
                free_transfers=None,  # Calculated on demand
                chip_used=chip_map.get(gw),
                points=gw_data.get("points"),
                total_points=gw_data.get("total_points"),
                overall_rank=gw_data.get("overall_rank"),
                transfers_in_json=None,  # Not available in history
                transfers_out_json=None,
                captain_id=captain_id,
                captain_name=captain_name,
                transfers_cost=gw_data.get("event_transfers_cost", 0),
            )

        # Build fixture calendar
        log("Building fixture calendar...")
        self.update_fixture_calendar(season_id)

        # Track current prices
        log("Tracking squad prices...")
        self.track_prices(season_id, manager_id)

        log(f"Season initialization complete! {len(gw_entries)} GWs backfilled.")
        return {
            "season_id": season_id,
            "manager_id": manager_id,
            "manager_name": manager_name,
            "team_name": team_name,
            "start_gw": start_gw,
            "current_gw": current_event,
            "gws_backfilled": len(gw_entries),
        }

    # -------------------------------------------------------------------
    # Generate Recommendation
    # -------------------------------------------------------------------

    def generate_recommendation(self, manager_id: int, progress_fn=None) -> dict:
        """Generate pre-GW transfer, captain, and chip advice.

        Now includes strategic planning: multi-GW predictions, chip heatmap,
        multi-week transfer plan, captain plan, and chip synergies.
        """
        def log(msg):
            if progress_fn:
                progress_fn(msg)
            else:
                print(msg)

        season = self.db.get_season(manager_id)
        if not season:
            raise ValueError("No active season. Initialize first.")
        season_id = season["id"]

        bootstrap = self._load_bootstrap()
        next_gw = self._get_next_gw(bootstrap)
        if not next_gw:
            raise ValueError("Could not determine next gameweek.")

        log(f"Generating recommendation for GW{next_gw}...")

        # Load predictions
        import pandas as pd
        pred_path = OUTPUT_DIR / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError("No predictions. Train models first.")
        pred_df = pd.read_csv(pred_path, encoding="utf-8")

        # Get current squad
        entry = fetch_manager_entry(manager_id)
        current_event = entry.get("current_event")
        picks_data = fetch_manager_picks(manager_id, current_event)
        history = fetch_manager_history(manager_id)
        free_transfers = self._calculate_free_transfers(history)

        elements_map = self._get_elements_map(bootstrap)
        id_to_code, id_to_short, code_to_short = self._get_team_maps(bootstrap)

        # Build current squad info
        picks = picks_data.get("picks", [])
        current_squad_ids = set()
        current_squad = []
        current_squad_cost = 0.0
        for pick in picks:
            eid = pick.get("element")
            current_squad_ids.add(eid)
            el = elements_map.get(eid, {})
            tid = el.get("team")
            tc = id_to_code.get(tid)
            player_cost = el.get("now_cost", 0) / 10
            current_squad_cost += player_cost
            current_squad.append({
                "player_id": eid,
                "web_name": el.get("web_name", "Unknown"),
                "position": ELEMENT_TYPE_MAP.get(el.get("element_type"), ""),
                "team_code": tc,
                "team": code_to_short.get(tc, ""),
                "cost": player_cost,
                "starter": pick.get("position", 12) <= 11,
            })

        entry_hist = picks_data.get("entry_history", {})
        bank = entry_hist.get("bank", 0) / 10
        total_budget = round(bank + current_squad_cost, 1)

        target_col = "predicted_next_gw_points"

        # Build pred map for current squad
        pred_map = {}
        for _, row in pred_df.iterrows():
            pred_map[row.get("player_id")] = row.to_dict()

        # Current XI points
        current_starters = [p for p in current_squad if p["starter"]]
        current_xi_points = round(
            sum(pred_map.get(p["player_id"], {}).get(target_col, 0) or 0 for p in current_starters), 2
        )

        # =====================================================================
        # STRATEGIC PLANNING: Multi-GW predictions + chip heatmap + transfer plan
        # =====================================================================
        strategic_plan = None
        chip_heatmap = {}
        multi_week_plan = []

        try:
            log("  Generating multi-GW predictions...")
            from src.data_fetcher import load_all_data
            from src.feature_engineering import build_features, get_fixture_context
            from src.predict import predict_future_range, get_latest_gw

            data = load_all_data()
            df = build_features(data)
            latest_gw = get_latest_gw(df)
            fixture_context = get_fixture_context(data)

            # Build current snapshot for predictions
            from src.model import CURRENT_SEASON
            current_snapshot = df[
                (df["season"] == CURRENT_SEASON) & (df["gameweek"] == latest_gw)
            ].copy()
            if current_snapshot.empty:
                current_snapshot = df[df["gameweek"] == latest_gw].copy()

            # Step 1: Multi-GW predictions (GW+1 to GW+8)
            future_predictions = predict_future_range(
                current_snapshot, df, fixture_context, latest_gw, horizon=8,
            )

            if future_predictions:
                # Enrich predictions with position/cost/team_code for solver
                for gw, gw_df in future_predictions.items():
                    meta = pred_df[["player_id", "position", "cost", "team_code", "web_name"]].drop_duplicates("player_id")
                    enriched = gw_df.merge(meta, on="player_id", how="left")
                    enriched["team"] = enriched["team_code"].map(code_to_short).fillna("")
                    future_predictions[gw] = enriched

                # Apply availability adjustments (Phase 4c)
                log("  Applying availability adjustments...")
                elements = bootstrap.get("elements", [])
                future_predictions = apply_availability_adjustments(
                    future_predictions, elements,
                )

                # Step 2: Chip heatmap
                log("  Evaluating chips across all GWs...")
                chip_evaluator = ChipEvaluator()

                # Determine available chips
                chips_used_list = history.get("chips", [])
                chips_used_names = {c["name"] for c in chips_used_list}
                available_chips = set()
                wc_events = [c["event"] for c in chips_used_list if c["name"] == "wildcard"]
                if next_gw <= 19:
                    if not any(e <= 19 for e in wc_events):
                        available_chips.add("wildcard")
                else:
                    if not any(e >= 20 for e in wc_events):
                        available_chips.add("wildcard")
                for chip in ["freehit", "bboost", "3xc"]:
                    if chip not in chips_used_names:
                        available_chips.add(chip)

                fixture_calendar = self.db.get_fixture_calendar(
                    season_id, from_gw=next_gw,
                )

                chip_heatmap = chip_evaluator.evaluate_all_chips(
                    current_squad_ids, total_budget, available_chips,
                    future_predictions, fixture_calendar,
                )

                # Step 2b: Chip synergies
                log("  Evaluating chip synergies...")
                chip_synergies = chip_evaluator.evaluate_chip_synergies(
                    chip_heatmap, available_chips,
                )

                # Step 3: Multi-week transfer plan
                log("  Planning transfers over 3 GWs...")
                planner = MultiWeekPlanner()
                price_alerts = self.get_price_alerts(season_id)

                # Determine chip plan from heatmap for transfer planner
                chip_plan_for_planner = None
                if chip_heatmap:
                    synthesizer = PlanSynthesizer()
                    chip_schedule = synthesizer._plan_chip_schedule(
                        chip_heatmap, chip_synergies, available_chips,
                    )
                    chip_plan_for_planner = {"chip_gws": chip_schedule}

                multi_week_plan = planner.plan_transfers(
                    current_squad_ids, total_budget, free_transfers,
                    future_predictions, fixture_calendar, price_alerts,
                    chip_plan=chip_plan_for_planner,
                )

                # Step 4: Captain plan
                log("  Planning captaincy...")
                captain_planner = CaptainPlanner()
                captain_plan = captain_planner.plan_captaincy(
                    current_squad_ids, future_predictions, multi_week_plan,
                )

                # Step 5: Synthesize full plan
                log("  Synthesizing strategic plan...")
                synthesizer = PlanSynthesizer()
                strategic_plan = synthesizer.synthesize(
                    multi_week_plan, captain_plan, chip_heatmap,
                    chip_synergies, available_chips,
                )

                # Step 6: Compare to previous plan, log changes
                prev_plan_row = self.db.get_strategic_plan(season_id)
                if prev_plan_row and prev_plan_row.get("plan_json"):
                    try:
                        prev_plan = json.loads(prev_plan_row["plan_json"])
                        self._log_plan_changes(
                            season_id, next_gw, prev_plan, strategic_plan,
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Step 7: Store strategic plan
                self.db.save_strategic_plan(
                    season_id=season_id,
                    as_of_gw=next_gw,
                    plan_json=json.dumps(scrub_nan_recursive(strategic_plan)),
                    chip_heatmap_json=json.dumps(scrub_nan_recursive(chip_heatmap)),
                )

                log("  Strategic plan saved.")
            else:
                log("  Warning: No future predictions generated (missing fixtures?)")

        except Exception as exc:
            log(f"  Strategic planning failed (non-fatal): {exc}")
            import traceback
            traceback.print_exc()

        # =====================================================================
        # EXISTING LOGIC: single-GW recommendation (backwards compatible)
        # =====================================================================

        # --- Use multi-week plan GW1 if available, else fall back to single-GW solver ---
        transfers = []
        new_squad_json = None
        predicted_points = current_xi_points

        if multi_week_plan and multi_week_plan[0].get("transfers_in"):
            # Use GW+1 from the multi-week plan
            gw1 = multi_week_plan[0]
            predicted_points = gw1.get("predicted_points", current_xi_points)

            # Build transfer pairs from the multi-week plan
            out_list = []
            in_list = []
            for t in gw1.get("transfers_out", []):
                pid = t.get("player_id")
                out_info = dict(next((p for p in current_squad if p["player_id"] == pid), {}))
                el = elements_map.get(pid, {})
                pred = pred_map.get(pid, {})
                out_info["event_points"] = el.get("event_points", 0)
                out_info["predicted_next_gw_points"] = pred.get("predicted_next_gw_points")
                out_info["predicted_next_3gw_points"] = pred.get("predicted_next_3gw_points")
                out_info["total_points"] = el.get("total_points", 0)
                out_info["direction"] = "out"
                out_list.append(out_info)
            for t in gw1.get("transfers_in", []):
                pid = t.get("player_id")
                in_info = dict(t)
                el = elements_map.get(pid, {})
                in_info["event_points"] = el.get("event_points", 0)
                in_info["total_points"] = el.get("total_points", 0)
                in_info["direction"] = "in"
                in_list.append(in_info)

            pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
            out_list.sort(key=lambda p: pos_order.get(p.get("position"), 9))
            in_list.sort(key=lambda p: pos_order.get(p.get("position"), 9))
            for i, out_p in enumerate(out_list):
                in_p = in_list[i] if i < len(in_list) else {}
                transfers.append({"out": out_p, "in": in_p})
        else:
            # Fall back to single-GW MILP solver
            log("  Running single-GW transfer solver (fallback)...")
            pool = pred_df.dropna(subset=["position", "cost", target_col]).copy()
            if "team_code" in pool.columns:
                pool["team"] = pool["team_code"].map(code_to_short).fillna("")

            result = solve_transfer_milp(
                pool, current_squad_ids, target_col,
                budget=total_budget, max_transfers=free_transfers,
            )

            if result:
                predicted_points = result["starting_points"]
                new_squad_json = json.dumps(result["players"])
                out_list = []
                in_list = []
                for pid in result.get("transfers_out_ids", set()):
                    out_info = dict(next((p for p in current_squad if p["player_id"] == pid), {}))
                    el = elements_map.get(pid, {})
                    pred = pred_map.get(pid, {})
                    out_info["event_points"] = el.get("event_points", 0)
                    out_info["predicted_next_gw_points"] = pred.get("predicted_next_gw_points")
                    out_info["predicted_next_3gw_points"] = pred.get("predicted_next_3gw_points")
                    out_info["total_points"] = el.get("total_points", 0)
                    out_info["direction"] = "out"
                    out_list.append(out_info)
                for pid in result.get("transfers_in_ids", set()):
                    in_info = dict(next((p for p in result["players"] if p["player_id"] == pid), {}))
                    el = elements_map.get(pid, {})
                    in_info["event_points"] = el.get("event_points", 0)
                    in_info["total_points"] = el.get("total_points", 0)
                    in_info["direction"] = "in"
                    in_list.append(in_info)
                pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
                out_list.sort(key=lambda p: pos_order.get(p.get("position"), 9))
                in_list.sort(key=lambda p: pos_order.get(p.get("position"), 9))
                for i, out_p in enumerate(out_list):
                    in_p = in_list[i] if i < len(in_list) else {}
                    transfers.append({"out": out_p, "in": in_p})

        # --- Captain pick ---
        log("  Picking captain...")
        captain_id = None
        captain_name = None

        # Use strategic captain plan if available
        if strategic_plan and strategic_plan.get("timeline"):
            first_entry = strategic_plan["timeline"][0]
            captain_id = first_entry.get("captain_id")
            captain_name = first_entry.get("captain_name")

        if captain_id is None:
            captain_col = "captain_score"
            if captain_col not in pred_df.columns:
                captain_col = target_col
            squad_ids = current_squad_ids
            if multi_week_plan and multi_week_plan[0].get("squad_ids"):
                squad_ids = set(multi_week_plan[0]["squad_ids"])
            squad_preds = pred_df[pred_df["player_id"].isin(squad_ids)].copy()
            if not squad_preds.empty and captain_col in squad_preds.columns:
                captain_row = squad_preds.loc[squad_preds[captain_col].idxmax()]
                captain_id = int(captain_row["player_id"])
                captain_name = captain_row.get("web_name", "Unknown")

        # --- Evaluate chips (single-GW, backwards compatible) ---
        log("  Evaluating chips...")
        chip_values = self._evaluate_chips(
            season_id, manager_id, bootstrap, pred_df, current_squad_ids,
            total_budget, history,
        )

        # Use strategic chip schedule for suggestion if available
        chip_suggestion = None
        if strategic_plan and strategic_plan.get("chip_schedule"):
            for chip_name, chip_gw in strategic_plan["chip_schedule"].items():
                if chip_gw == next_gw:
                    chip_suggestion = chip_name
                    break

        if chip_suggestion is None and chip_values:
            best_chip = max(chip_values.items(), key=lambda x: x[1])
            if best_chip[1] > 5.0:
                chip_suggestion = best_chip[0]

        # --- Bank analysis ---
        log("  Running bank vs use analysis...")
        bank_analysis = self._analyze_bank_vs_use(
            pred_df, current_squad_ids, total_budget, free_transfers,
            code_to_short,
        )

        # Save to DB
        self.db.save_recommendation(
            season_id=season_id,
            gameweek=next_gw,
            transfers_json=json.dumps(scrub_nan(transfers)),
            captain_id=captain_id,
            captain_name=captain_name,
            chip_suggestion=chip_suggestion,
            chip_values_json=json.dumps(chip_values),
            bank_analysis_json=json.dumps(bank_analysis),
            new_squad_json=new_squad_json,
            predicted_points=predicted_points,
        )

        log(f"Recommendation saved for GW{next_gw}.")
        return {
            "gameweek": next_gw,
            "transfers": scrub_nan(transfers),
            "captain": {"id": captain_id, "name": captain_name},
            "chip_suggestion": chip_suggestion,
            "chip_values": chip_values,
            "chip_heatmap": scrub_nan_recursive(chip_heatmap),
            "bank_analysis": bank_analysis,
            "predicted_points": predicted_points,
            "current_xi_points": current_xi_points,
            "free_transfers": free_transfers,
            "strategic_plan": scrub_nan_recursive(strategic_plan) if strategic_plan else None,
            "multi_week_plan": scrub_nan_recursive(multi_week_plan) if multi_week_plan else None,
        }

    def _log_plan_changes(self, season_id: int, gameweek: int,
                          old_plan: dict, new_plan: dict):
        """Compare old and new strategic plans and log differences."""
        old_schedule = old_plan.get("chip_schedule", {})
        new_schedule = new_plan.get("chip_schedule", {})

        for chip in set(list(old_schedule.keys()) + list(new_schedule.keys())):
            old_gw = old_schedule.get(chip)
            new_gw = new_schedule.get(chip)
            if old_gw != new_gw:
                self.db.save_plan_change(
                    season_id=season_id,
                    gameweek=gameweek,
                    change_type="chip_schedule",
                    description=f"{chip} rescheduled",
                    old_value=f"GW{old_gw}" if old_gw else "unscheduled",
                    new_value=f"GW{new_gw}" if new_gw else "unscheduled",
                    reason="Updated predictions/fixtures",
                )

        # Compare captain plan
        old_timeline = old_plan.get("timeline", [])
        new_timeline = new_plan.get("timeline", [])
        old_caps = {e["gw"]: e.get("captain_name") for e in old_timeline if "captain_name" in e}
        new_caps = {e["gw"]: e.get("captain_name") for e in new_timeline if "captain_name" in e}
        for gw in set(list(old_caps.keys()) + list(new_caps.keys())):
            if old_caps.get(gw) != new_caps.get(gw) and old_caps.get(gw) and new_caps.get(gw):
                self.db.save_plan_change(
                    season_id=season_id,
                    gameweek=gameweek,
                    change_type="captain",
                    description=f"GW{gw} captain changed",
                    old_value=old_caps.get(gw, ""),
                    new_value=new_caps.get(gw, ""),
                    reason="Updated predictions",
                )

    # -------------------------------------------------------------------
    # Record Actual Results
    # -------------------------------------------------------------------

    def record_actual_results(self, manager_id: int, progress_fn=None) -> dict:
        """Post-GW: import actual picks/results and compare to recommendation."""
        def log(msg):
            if progress_fn:
                progress_fn(msg)
            else:
                print(msg)

        season = self.db.get_season(manager_id)
        if not season:
            raise ValueError("No active season.")
        season_id = season["id"]

        entry = fetch_manager_entry(manager_id)
        current_event = entry.get("current_event")
        if not current_event:
            raise ValueError("No current event.")

        log(f"Recording results for GW{current_event}...")

        picks_data = fetch_manager_picks(manager_id, current_event)
        history = fetch_manager_history(manager_id)

        bootstrap = self._load_bootstrap()
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, _, code_to_short = self._get_team_maps(bootstrap)

        # Build squad
        picks = picks_data.get("picks", [])
        squad = []
        captain_id = None
        captain_name = None
        for pick in picks:
            eid = pick.get("element")
            el = elements_map.get(eid, {})
            tid = el.get("team")
            tc = id_to_code.get(tid)
            pos = ELEMENT_TYPE_MAP.get(el.get("element_type"), "")
            player = {
                "player_id": eid,
                "web_name": el.get("web_name", "Unknown"),
                "position": pos,
                "team_code": tc,
                "team": code_to_short.get(tc, ""),
                "cost": el.get("now_cost", 0) / 10,
                "starter": pick.get("position", 12) <= 11,
                "is_captain": pick.get("is_captain", False),
                "multiplier": pick.get("multiplier", 1),
                "event_points": el.get("event_points", 0) * pick.get("multiplier", 1),
            }
            squad.append(player)
            if pick.get("is_captain"):
                captain_id = eid
                captain_name = el.get("web_name", "Unknown")

        # Find GW data from history
        gw_entries = history.get("current", [])
        gw_data = next((g for g in gw_entries if g["event"] == current_event), {})
        chip_map = {c["event"]: c["name"] for c in history.get("chips", [])}

        entry_hist = picks_data.get("entry_history", {})

        # Save snapshot
        self.db.save_gw_snapshot(
            season_id=season_id,
            gameweek=current_event,
            squad_json=json.dumps(squad),
            bank=entry_hist.get("bank", 0) / 10,
            team_value=entry_hist.get("value", 0) / 10,
            free_transfers=self._calculate_free_transfers(history),
            chip_used=chip_map.get(current_event),
            points=gw_data.get("points"),
            total_points=gw_data.get("total_points"),
            overall_rank=gw_data.get("overall_rank"),
            captain_id=captain_id,
            captain_name=captain_name,
            transfers_cost=gw_data.get("event_transfers_cost", 0),
        )

        self.db.update_season_gw(season_id, current_event)

        # Compare to recommendation
        rec = self.db.get_recommendation(season_id, current_event)
        outcome = {}
        if rec:
            log("  Comparing to recommendation...")
            actual_points = gw_data.get("points", 0)
            recommended_points = rec.get("predicted_points", 0)
            point_delta = round((actual_points or 0) - (recommended_points or 0), 1)

            # Check if captain was followed
            followed_captain = 1 if captain_id == rec.get("captain_id") else 0

            # Check if transfers were followed
            rec_transfers = json.loads(rec.get("transfers_json") or "[]")
            rec_in_ids = {t["player_id"] for t in rec_transfers if t.get("direction") == "in"}
            actual_squad_ids = {p["player_id"] for p in squad}
            followed_transfers = 1 if rec_in_ids.issubset(actual_squad_ids) else 0

            self.db.save_outcome(
                season_id=season_id,
                gameweek=current_event,
                followed_transfers=followed_transfers,
                followed_captain=followed_captain,
                followed_chip=0,
                recommended_points=recommended_points,
                actual_points=actual_points,
                point_delta=point_delta,
            )
            outcome = {
                "followed_transfers": followed_transfers,
                "followed_captain": followed_captain,
                "recommended_points": recommended_points,
                "actual_points": actual_points,
                "point_delta": point_delta,
            }

        log(f"GW{current_event} results recorded.")
        return {
            "gameweek": current_event,
            "points": gw_data.get("points"),
            "total_points": gw_data.get("total_points"),
            "overall_rank": gw_data.get("overall_rank"),
            "outcome": outcome,
        }

    # -------------------------------------------------------------------
    # Chip Evaluation
    # -------------------------------------------------------------------

    def _evaluate_chips(self, season_id: int, manager_id: int,
                        bootstrap: dict, pred_df, current_squad_ids: set,
                        total_budget: float, history: dict) -> dict:
        """Estimate point value of each available chip."""
        import pandas as pd

        chips_used_list = history.get("chips", [])
        chips_used_names = {c["name"] for c in chips_used_list}

        # Determine which chips are still available
        available = set()
        # Wildcards: one per half (GW1-19, GW20-38)
        wc_events = [c["event"] for c in chips_used_list if c["name"] == "wildcard"]
        current_gw = bootstrap.get("events", [{}])
        next_gw = self._get_next_gw(bootstrap) or 1
        if next_gw <= 19:
            if not any(e <= 19 for e in wc_events):
                available.add("wildcard")
        else:
            if not any(e >= 20 for e in wc_events):
                available.add("wildcard")

        for chip in ["freehit", "bboost", "3xc"]:
            if chip not in chips_used_names:
                available.add(chip)

        target_col = "predicted_next_gw_points"
        chip_values = {}

        if target_col not in pred_df.columns:
            return {chip: 0.0 for chip in available}

        # Current XI predicted points
        squad_preds = pred_df[pred_df["player_id"].isin(current_squad_ids)]
        if squad_preds.empty:
            return {chip: 0.0 for chip in available}

        current_xi_pts = squad_preds.nlargest(11, target_col)[target_col].sum()

        if "bboost" in available:
            # Bench Boost: sum of bench predictions (4 lowest of 15)
            all_15 = squad_preds.nlargest(15, target_col)[target_col]
            bench_pts = all_15.tail(4).sum() if len(all_15) >= 15 else 0
            chip_values["bboost"] = round(bench_pts, 1)

        if "3xc" in available:
            # Triple Captain: extra captain points (highest predicted * 1)
            best_pred = squad_preds[target_col].max()
            chip_values["3xc"] = round(best_pred, 1)

        if "freehit" in available:
            # Free Hit: unconstrained best team vs current
            pool = pred_df.dropna(subset=["position", "cost", target_col]).copy()
            fh_result = solve_milp_team(pool, target_col, budget=1000)
            if fh_result:
                chip_values["freehit"] = round(fh_result["starting_points"] - current_xi_pts, 1)
            else:
                chip_values["freehit"] = 0.0

        if "wildcard" in available:
            # Wildcard: unconstrained best team vs current (similar to FH but permanent)
            pool = pred_df.dropna(subset=["position", "cost", target_col]).copy()
            wc_result = solve_milp_team(pool, target_col, budget=total_budget)
            if wc_result:
                chip_values["wildcard"] = round(wc_result["starting_points"] - current_xi_pts, 1)
            else:
                chip_values["wildcard"] = 0.0

        return chip_values

    # -------------------------------------------------------------------
    # Bank vs Use Analysis
    # -------------------------------------------------------------------

    def _analyze_bank_vs_use(self, pred_df, current_squad_ids: set,
                             total_budget: float, free_transfers: int,
                             code_to_short: dict) -> dict:
        """Multi-week lookahead: compare FT allocation strategies over 2 weeks.

        For each option (use 0, 1, 2... FTs this week), simulates:
          Week 1: solve with N transfers → squad + points
          Week 2: solve with rolled-over FTs from new squad → points
        Picks the strategy that maximizes total points over both weeks.
        """
        import pandas as pd

        gw_col = "predicted_next_gw_points"
        gw3_col = "predicted_next_3gw_points"
        if gw_col not in pred_df.columns:
            return {"recommendation": "insufficient_data"}

        pool = pred_df.dropna(subset=["position", "cost", gw_col]).copy()
        if "team_code" in pool.columns:
            pool["team"] = pool["team_code"].map(code_to_short).fillna("")

        # Current XI points (week 1 baseline with 0 transfers)
        current_pred = pool[pool["player_id"].isin(current_squad_ids)]
        current_xi_pts = current_pred.nlargest(11, gw_col)[gw_col].sum() if len(current_pred) >= 11 else 0

        # Simulate each strategy: use 0..min(ft, 3) FTs this week
        max_use = min(free_transfers, 3)
        strategies = []

        for use_now in range(0, max_use + 1):
            # Week 1: solve with use_now transfers
            if use_now == 0:
                week1_pts = current_xi_pts
                week1_squad_ids = current_squad_ids
                week1_budget = total_budget
            else:
                result_w1 = solve_transfer_milp(
                    pool, current_squad_ids, gw_col,
                    budget=total_budget, max_transfers=use_now,
                )
                if result_w1:
                    week1_pts = result_w1["starting_points"]
                    week1_squad_ids = {p["player_id"] for p in result_w1["players"]}
                    week1_budget = result_w1["total_cost"]
                else:
                    week1_pts = current_xi_pts
                    week1_squad_ids = current_squad_ids
                    week1_budget = total_budget

            # FTs available next week: unused FTs roll over + 1 new, capped at 5
            remaining = free_transfers - use_now
            next_week_fts = min(remaining + 1, 5)

            # Week 2: solve from week 1's squad with next_week_fts transfers
            # Same single-GW predictions — keeps numbers comparable across weeks
            result_w2 = solve_transfer_milp(
                pool, week1_squad_ids, gw_col,
                budget=week1_budget, max_transfers=next_week_fts,
            )
            week2_pts = result_w2["starting_points"] if result_w2 else 0

            total_pts = round(week1_pts + week2_pts, 2)

            strategies.append({
                "use_this_week": use_now,
                "fts_next_week": next_week_fts,
                "week1_points": round(week1_pts, 2),
                "week2_points": round(week2_pts, 2),
                "total_2week": total_pts,
            })

        # Pick best strategy
        best = max(strategies, key=lambda s: s["total_2week"])
        best_use = best["use_this_week"]

        # Determine recommendation
        if best_use == 0:
            rec = "bank"
        else:
            rec = f"use_{best_use}"

        # Build comparison summary
        bank_strat = next((s for s in strategies if s["use_this_week"] == 0), None)
        use1_strat = next((s for s in strategies if s["use_this_week"] == 1), None)

        return {
            "recommendation": rec,
            "best_strategy": best,
            "strategies": strategies,
            "free_transfers": free_transfers,
            "horizon": "2 weeks",
            # Backward-compatible fields
            "gain_from_transfer": round(
                (use1_strat["total_2week"] - bank_strat["total_2week"]) if use1_strat and bank_strat else 0, 2
            ),
        }

    # -------------------------------------------------------------------
    # Price Tracking
    # -------------------------------------------------------------------

    def track_prices(self, season_id: int, manager_id: int):
        """Snapshot prices for squad players + high-transfer players."""
        bootstrap = self._load_bootstrap()
        elements = bootstrap.get("elements", [])
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

        # Get current squad
        entry = fetch_manager_entry(manager_id)
        current_event = entry.get("current_event")
        if current_event:
            try:
                picks_data = fetch_manager_picks(manager_id, current_event)
                squad_ids = {p["element"] for p in picks_data.get("picks", [])}
            except Exception:
                squad_ids = set()
        else:
            squad_ids = set()

        # Top transferred-in players (watchlist)
        sorted_elements = sorted(elements, key=lambda e: e.get("transfers_in_event", 0), reverse=True)
        watchlist_ids = {e["id"] for e in sorted_elements[:30]}

        track_ids = squad_ids | watchlist_ids
        players = []
        for el in elements:
            if el["id"] in track_ids:
                tid = el.get("team")
                players.append({
                    "player_id": el["id"],
                    "web_name": el.get("web_name"),
                    "team_code": id_to_code.get(tid),
                    "price": el.get("now_cost", 0) / 10,
                    "transfers_in_event": el.get("transfers_in_event", 0),
                    "transfers_out_event": el.get("transfers_out_event", 0),
                })

        self.db.save_price_snapshots_bulk(season_id, players)

    def get_price_alerts(self, season_id: int) -> list[dict]:
        """Flag players likely to rise/fall based on transfer volume."""
        bootstrap = self._load_bootstrap()
        elements = bootstrap.get("elements", [])
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}

        alerts = []
        for el in elements:
            net = el.get("transfers_in_event", 0) - el.get("transfers_out_event", 0)
            total_transfers = el.get("transfers_in_event", 0) + el.get("transfers_out_event", 0)

            if total_transfers < 5000:
                continue

            if net > 20000:
                tc = id_to_code.get(el.get("team"))
                alerts.append({
                    "player_id": el["id"],
                    "web_name": el.get("web_name"),
                    "team": code_to_short.get(tc, ""),
                    "price": el.get("now_cost", 0) / 10,
                    "net_transfers": net,
                    "direction": "rise",
                })
            elif net < -20000:
                tc = id_to_code.get(el.get("team"))
                alerts.append({
                    "player_id": el["id"],
                    "web_name": el.get("web_name"),
                    "team": code_to_short.get(tc, ""),
                    "price": el.get("now_cost", 0) / 10,
                    "net_transfers": net,
                    "direction": "fall",
                })

        alerts.sort(key=lambda a: abs(a["net_transfers"]), reverse=True)
        return alerts

    # -------------------------------------------------------------------
    # Fixture Calendar
    # -------------------------------------------------------------------

    def update_fixture_calendar(self, season_id: int):
        """Parse fixtures API to build GW-by-team fixture grid."""
        bootstrap = self._load_bootstrap()
        fixtures = self._load_fixtures()
        if not fixtures:
            return

        id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        teams = {t["id"]: t for t in bootstrap.get("teams", [])}

        # Build per-team per-GW fixture data
        team_gw: dict[int, dict[int, list]] = {}  # team_id -> gw -> [fixtures]
        for f in fixtures:
            gw = f.get("event")
            if not gw:
                continue
            for side, opp_side, tag in [("team_h", "team_a", "H"), ("team_a", "team_h", "A")]:
                tid = f[side]
                opp_id = f[opp_side]
                opp_name = id_to_short.get(opp_id, "?")
                fdr = f.get(f"team_{tag.lower()}_difficulty", 3)
                if tid not in team_gw:
                    team_gw[tid] = {}
                if gw not in team_gw[tid]:
                    team_gw[tid][gw] = []
                team_gw[tid][gw].append({
                    "opponent": f"{opp_name} ({tag})",
                    "fdr": fdr,
                })

        # Build flat records
        records = []
        all_gws = set()
        for tid in team_gw:
            for gw in team_gw[tid]:
                all_gws.add(gw)

        for tid, gw_data in team_gw.items():
            team_short = id_to_short.get(tid, "?")
            for gw in sorted(all_gws):
                fxs = gw_data.get(gw, [])
                if not fxs:
                    # BGW: no fixture this GW
                    records.append({
                        "team_id": tid,
                        "team_short": team_short,
                        "gameweek": gw,
                        "fixture_count": 0,
                        "opponents_json": json.dumps([]),
                        "fdr_avg": None,
                        "is_dgw": 0,
                        "is_bgw": 1,
                    })
                else:
                    avg_fdr = round(sum(fx["fdr"] for fx in fxs) / len(fxs), 1)
                    records.append({
                        "team_id": tid,
                        "team_short": team_short,
                        "gameweek": gw,
                        "fixture_count": len(fxs),
                        "opponents_json": json.dumps([fx["opponent"] for fx in fxs]),
                        "fdr_avg": avg_fdr,
                        "is_dgw": 1 if len(fxs) >= 2 else 0,
                        "is_bgw": 0,
                    })

        self.db.save_fixture_calendar(season_id, records)

    # -------------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------------

    def get_season_dashboard(self, manager_id: int) -> dict:
        """Full dashboard data for the Season tab."""
        season = self.db.get_season(manager_id)
        if not season:
            return {"error": "No active season."}
        season_id = season["id"]

        snapshots = self.db.get_snapshots(season_id)
        rank_history = self.db.get_rank_history(season_id)
        budget_history = self.db.get_budget_history(season_id)
        chips_used = self.db.get_chips_status(season_id)
        accuracy = self.db.get_recommendation_accuracy(season_id)
        recommendations = self.db.get_recommendations(season_id)
        outcomes = self.db.get_outcomes(season_id)

        # Compute available chips
        used_chip_names = {c["chip_used"] for c in chips_used}
        all_chips_list = [
            {"name": "wildcard", "label": "Wildcard"},
            {"name": "freehit", "label": "Free Hit"},
            {"name": "bboost", "label": "Bench Boost"},
            {"name": "3xc", "label": "Triple Captain"},
        ]
        chips_status = []
        for chip in all_chips_list:
            used_in = next((c["gameweek"] for c in chips_used if c["chip_used"] == chip["name"]), None)
            chips_status.append({
                "name": chip["name"],
                "label": chip["label"],
                "used": chip["name"] in used_chip_names,
                "used_gw": used_in,
            })

        # Latest snapshot for summary
        latest = snapshots[-1] if snapshots else {}

        # Points per GW
        points_per_gw = [
            {"gameweek": s["gameweek"], "points": s.get("points", 0)}
            for s in snapshots
        ]

        return {
            "season": season,
            "summary": {
                "overall_rank": latest.get("overall_rank"),
                "total_points": latest.get("total_points"),
                "bank": latest.get("bank"),
                "team_value": latest.get("team_value"),
                "gameweek": latest.get("gameweek"),
            },
            "rank_history": rank_history,
            "budget_history": budget_history,
            "points_per_gw": points_per_gw,
            "chips_status": chips_status,
            "accuracy": accuracy,
            "recommendations_count": len(recommendations),
            "outcomes_count": len(outcomes),
        }
