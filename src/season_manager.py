"""Business logic for season-long FPL management."""

import json
import time
from pathlib import Path

from src.data_fetcher import (
    CACHE_DIR,
    FPL_API_BASE,
    fetch_event_live,
    fetch_manager_entry,
    fetch_manager_history,
    fetch_manager_picks,
    fetch_manager_transfers,
)
from src.season_db import SeasonDB
from src.solver import scrub_nan, solve_milp_team, solve_transfer_milp, solve_transfer_milp_with_hits
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
    """Recursively replace NaN/inf with None and numpy types with Python natives."""
    import math
    import numpy as np
    if isinstance(obj, dict):
        return {k: scrub_nan_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [scrub_nan_recursive(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
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
        # If no 'is_next', fall back to the one after 'is_current'
        for event in bootstrap.get("events", []):
            if event.get("is_current"):
                next_id = event["id"] + 1
                if next_id > 38:
                    return None  # Season is over
                return next_id
        return None

    def _calculate_free_transfers(self, history: dict) -> int:
        current = history.get("current", [])
        chips = history.get("chips", [])
        chip_events = {c["event"] for c in chips if c.get("name") in ("wildcard", "freehit")}

        first_event = current[0].get("event", 1) if current else 1
        ft = 1
        for i, gw_entry in enumerate(current):
            event = gw_entry.get("event")
            if event in chip_events:
                # WC/FH: FTs preserved at pre-chip count, no accrual
                continue
            transfers_made = gw_entry.get("event_transfers", 0)
            transfers_cost = gw_entry.get("event_transfers_cost", 0)
            paid = transfers_cost // 4 if transfers_cost > 0 else 0
            free_used = transfers_made - paid
            ft = ft - free_used
            ft = max(ft, 0)  # Prevent negative propagation from API inconsistencies
            # Mid-season joiner: first GW's FT was consumed by team creation
            if i == 0 and first_event > 1:
                ft = max(ft, 0)
            else:
                ft = min(ft + 1, 5)
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
            log("Season not started yet — switching to pre-season mode.")
            return self.generate_preseason_plan(manager_id, progress_fn=progress_fn)

        log(f"Manager: {manager_name} ({team_name})")
        log(f"Current GW: {current_event}")

        log("Fetching season history...")
        history = fetch_manager_history(manager_id)
        gw_entries = history.get("current", [])
        chips_used = history.get("chips", [])

        # Create season record
        from src.data_fetcher import detect_current_season
        start_gw = gw_entries[0]["event"] if gw_entries else 1
        season_id = self.db.create_season(
            manager_id=manager_id,
            manager_name=manager_name,
            team_name=team_name,
            season_name=detect_current_season(),
            start_gw=start_gw,
        )
        self.db.update_season_gw(season_id, current_event)

        # Clear stale recommendations, strategic plans, and outcomes so
        # the UI doesn't show outdated action plans after re-import.
        self.db.clear_generated_data(season_id)

        log(f"Season created (ID: {season_id}). Backfilling {len(gw_entries)} gameweeks...")

        # Build chip map: event -> chip_name
        chip_map = {c["event"]: c["name"] for c in chips_used}

        # Load bootstrap for element lookups
        bootstrap = self._load_bootstrap()
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, id_to_short, code_to_short = self._get_team_maps(bootstrap)

        # Fetch all transfers and group by GW
        log("Fetching transfer history...")
        all_transfers = fetch_manager_transfers(manager_id)
        transfers_by_gw = {}
        for t in all_transfers:
            gw = t["event"]
            transfers_by_gw.setdefault(gw, []).append(t)

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
                    # NOTE: Historical backfill uses current now_cost, not
                    # the price at that GW. FPL public API has no historical
                    # price data. Prices will be approximate for past GWs.
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

            # Build transfer in/out lists for this GW
            gw_transfers = transfers_by_gw.get(gw, [])
            t_in_list = []
            t_out_list = []
            for t in gw_transfers:
                el_in = elements_map.get(t["element_in"], {})
                el_out = elements_map.get(t["element_out"], {})
                t_in_list.append({
                    "player_id": t["element_in"],
                    "web_name": el_in.get("web_name", "Unknown"),
                    "cost": t.get("element_in_cost", 0) / 10,
                })
                t_out_list.append({
                    "player_id": t["element_out"],
                    "web_name": el_out.get("web_name", "Unknown"),
                    "cost": t.get("element_out_cost", 0) / 10,
                })

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
                transfers_in_json=json.dumps(t_in_list) if t_in_list else None,
                transfers_out_json=json.dumps(t_out_list) if t_out_list else None,
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
        if not current_event:
            return {
                "error": f"No GWs played yet (next GW is {next_gw}). "
                "Use pre-season plan for initial squad selection.",
            }
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
                "is_captain": pick.get("is_captain", False),
            })

        entry_hist = picks_data.get("entry_history", {})
        bank = entry_hist.get("bank", 0) / 10
        # Use entry_history "value" (squad selling value) when available;
        # this accounts for the 50% price-rise rule on selling.
        # Fall back to sum(now_cost) if value is missing (slightly optimistic).
        squad_value = entry_hist.get("value")
        if squad_value:
            total_budget = round(bank + squad_value / 10, 1)
        else:
            total_budget = round(bank + current_squad_cost, 1)

        target_col = "predicted_next_gw_points"

        # Build pred map for current squad
        pred_map = {}
        for _, row in pred_df.iterrows():
            pred_map[row.get("player_id")] = row.to_dict()

        # Current XI points (include captain bonus to match FPL actual_points scale)
        current_starters = [p for p in current_squad if p["starter"]]
        current_xi_points = round(
            sum(pred_map.get(p["player_id"], {}).get(target_col, 0) or 0 for p in current_starters), 2
        )
        # Add captain bonus (captain doubles their points)
        current_captain = next((p for p in current_squad if p.get("is_captain")), None)
        if current_captain:
            cap_pred = pred_map.get(current_captain["player_id"], {}).get(target_col, 0) or 0
            current_xi_points = round(current_xi_points + cap_pred, 2)

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
                # Enrich predictions with position/cost/team_code/captain_score for solver
                meta_cols = ["player_id", "position", "cost", "team_code", "web_name"]
                if "captain_score" in pred_df.columns:
                    meta_cols.append("captain_score")
                for gw, gw_df in future_predictions.items():
                    meta = pred_df[meta_cols].drop_duplicates("player_id")
                    enriched = gw_df.merge(meta, on="player_id", how="left")
                    enriched["team"] = enriched["team_code"].map(code_to_short).fillna("")
                    future_predictions[gw] = enriched

                # Apply availability adjustments (Phase 4c)
                log("  Applying availability adjustments...")
                elements = bootstrap.get("elements", [])
                future_predictions = apply_availability_adjustments(
                    future_predictions, elements,
                )

                # Bug 40 fix: captain_score is GW+1 specific (uses Q80 quantile).
                # For GW+2 onwards, fall back to predicted_points since Q80
                # doesn't apply to different fixtures.
                first_pred_gw = min(future_predictions.keys())
                for gw, gw_df in future_predictions.items():
                    if gw != first_pred_gw and "captain_score" in gw_df.columns:
                        gw_df["captain_score"] = gw_df["predicted_points"]

                # Step 2: Chip heatmap
                log("  Evaluating chips across all GWs...")
                chip_evaluator = ChipEvaluator()

                # Determine available chips (all chips reset at GW20)
                chips_used_list = history.get("chips", [])
                available_chips = set()
                for chip in ["wildcard", "freehit", "bboost", "3xc"]:
                    chip_events = [c["event"] for c in chips_used_list if c["name"] == chip]
                    if next_gw <= 19:
                        if not any(e <= 19 for e in chip_events):
                            available_chips.add(chip)
                    else:
                        if not any(e >= 20 for e in chip_events):
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
                log("  Planning transfers over 5 GWs...")
                planner = MultiWeekPlanner()
                price_alerts = self.predict_price_changes(season_id)

                # Determine chip plan from heatmap for transfer planner
                chip_plan_for_planner = None
                if chip_heatmap:
                    synthesizer = PlanSynthesizer()
                    chip_schedule = synthesizer._plan_chip_schedule(
                        chip_heatmap, chip_synergies, available_chips,
                    )
                    # Invert chip_schedule from {chip: gw} to {gw: chip}
                    chip_gws_by_gw = {gw: chip for chip, gw in chip_schedule.items()}
                    chip_plan_for_planner = {"chip_gws": chip_gws_by_gw}

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

        if multi_week_plan and multi_week_plan[0].get("new_squad"):
            # WC/FH GW: full squad from scratch
            gw1 = multi_week_plan[0]
            predicted_points = gw1.get("predicted_points", current_xi_points)
            # Mark captain on squad players from solver result
            solver_cap_id = None
            if strategic_plan and strategic_plan.get("timeline"):
                solver_cap_id = strategic_plan["timeline"][0].get("captain_id")
            for p in gw1["new_squad"]:
                p["is_captain"] = (p.get("player_id") == solver_cap_id)
                p["is_vice_captain"] = False
            new_squad_json = json.dumps(scrub_nan(gw1["new_squad"]))
            # No transfer pairs for WC/FH — the full squad is the recommendation
        elif multi_week_plan:
            # Bug 41 fix: Use GW+1 from the multi-week plan even when
            # transfers_in is empty (banking FT recommendation).
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

            # Bug 37 fix: Use solve_transfer_milp_with_hits to account for
            # hit penalties, and pass captain_col for captain optimization.
            cap_col = "captain_score" if "captain_score" in pool.columns else None
            result = solve_transfer_milp_with_hits(
                pool, current_squad_ids, target_col,
                budget=total_budget, free_transfers=free_transfers,
                max_transfers=min(free_transfers + 2, 5),
                captain_col=cap_col,
            )

            if result:
                predicted_points = result["starting_points"]
                # Mark starters on player records
                starter_ids = {p["player_id"] for p in result["starters"]}
                for p in result["players"]:
                    p["starter"] = p.get("player_id") in starter_ids
                new_squad_json = json.dumps(scrub_nan(result["players"]))
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

        # Use strategic chip schedule for suggestion if available.
        # If a strategic plan exists, trust its chip timing — only suggest a
        # chip if the plan schedules one for THIS GW.  The single-GW fallback
        # should only fire when no strategic plan was generated at all.
        chip_suggestion = None
        has_strategic_plan = strategic_plan and strategic_plan.get("chip_schedule")
        if has_strategic_plan:
            for chip_name, chip_gw in strategic_plan["chip_schedule"].items():
                if chip_gw == next_gw:
                    chip_suggestion = chip_name
                    break

        if chip_suggestion is None and not has_strategic_plan and chip_values:
            best_chip = max(chip_values.items(), key=lambda x: x[1])
            if best_chip[1] > 5.0:
                chip_suggestion = best_chip[0]

        # If fallback suggests WC/FH but multi-week plan didn't build a squad
        # for this GW, solve MILP now to generate the full chip squad
        if chip_suggestion in ("wildcard", "freehit") and new_squad_json is None:
            log(f"  Building {chip_suggestion} squad (fallback chip suggestion)...")
            pool = pred_df.dropna(subset=["position", "cost", target_col]).copy()
            if "team_code" in pool.columns:
                pool["team"] = pool["team_code"].map(code_to_short).fillna("")
            cap_col = "captain_score" if "captain_score" in pool.columns else None
            fh_result = solve_milp_team(
                pool, target_col, budget=total_budget, captain_col=cap_col,
            )
            if fh_result:
                # Mark captain on player records (solver returns captain_id separately)
                fh_captain_id = fh_result.get("captain_id")
                for p in fh_result["players"]:
                    p["is_captain"] = (p.get("player_id") == fh_captain_id)
                    p["is_vice_captain"] = False

                new_squad_json = json.dumps(scrub_nan(fh_result["players"]))
                predicted_points = fh_result["starting_points"]
                transfers = []  # WC/FH replaces entire squad, no transfer pairs
                # Update captain from the FH/WC squad
                if fh_captain_id:
                    captain_id = fh_captain_id
                    cap_player = next(
                        (p for p in fh_result["players"] if p.get("player_id") == captain_id),
                        None,
                    )
                    if cap_player:
                        captain_name = cap_player.get("web_name", captain_name)

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

        # Fetch bootstrap for player info (names, teams, costs)
        from src.data_fetcher import fetch_fpl_api
        bootstrap = fetch_fpl_api("bootstrap", force=True)
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, _, code_to_short = self._get_team_maps(bootstrap)

        # Bug 54 fix: use live event data for accurate per-GW points
        # instead of bootstrap event_points (which is always latest GW)
        live_points_map = {}
        try:
            live_data = fetch_event_live(current_event, force=True)
            for el_live in live_data.get("elements", []):
                live_points_map[el_live["id"]] = el_live.get("stats", {}).get("total_points", 0)
            log(f"  Loaded live points for GW{current_event} ({len(live_points_map)} players)")
        except Exception as exc:
            log(f"  Warning: could not fetch live event data ({exc}), falling back to bootstrap")

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
            # Use live event points when available, fall back to bootstrap
            raw_pts = live_points_map.get(eid, el.get("event_points", 0))
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
                "event_points": raw_pts * pick.get("multiplier", 1),
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

        # Fetch transfers for this GW
        all_transfers = fetch_manager_transfers(manager_id)
        gw_transfers = [t for t in all_transfers if t["event"] == current_event]
        t_in_list = []
        t_out_list = []
        for t in gw_transfers:
            el_in = elements_map.get(t["element_in"], {})
            el_out = elements_map.get(t["element_out"], {})
            t_in_list.append({
                "player_id": t["element_in"],
                "web_name": el_in.get("web_name", "Unknown"),
                "cost": t.get("element_in_cost", 0) / 10,
            })
            t_out_list.append({
                "player_id": t["element_out"],
                "web_name": el_out.get("web_name", "Unknown"),
                "cost": t.get("element_out_cost", 0) / 10,
            })

        # Save snapshot
        self.db.save_gw_snapshot(
            season_id=season_id,
            gameweek=current_event,
            squad_json=json.dumps(squad),
            bank=entry_hist.get("bank", gw_data.get("bank", 0)) / 10 if entry_hist else gw_data.get("bank", 0) / 10,
            team_value=entry_hist.get("value", gw_data.get("value", 0)) / 10 if entry_hist else gw_data.get("value", 0) / 10,
            free_transfers=self._calculate_free_transfers(history),
            chip_used=chip_map.get(current_event),
            points=gw_data.get("points"),
            total_points=gw_data.get("total_points"),
            overall_rank=gw_data.get("overall_rank"),
            transfers_in_json=json.dumps(t_in_list) if t_in_list else None,
            transfers_out_json=json.dumps(t_out_list) if t_out_list else None,
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
            rec_in_ids = {t["in"]["player_id"] for t in rec_transfers if t.get("in", {}).get("player_id")}
            actual_squad_ids = {p["player_id"] for p in squad}
            followed_transfers = 1 if rec_in_ids.issubset(actual_squad_ids) else 0

            # WC/FH squad comparison: compare full recommended squad to actual
            rec_chip = rec.get("chip_suggestion") if rec else None
            rec_new_squad = rec.get("new_squad_json") if rec else None
            if rec_chip in ("wildcard", "freehit") and rec_new_squad:
                try:
                    import json as _json
                    rec_squad = _json.loads(rec_new_squad)
                    rec_squad_ids = {p["player_id"] for p in rec_squad if "player_id" in p}
                    if rec_squad_ids:
                        overlap = rec_squad_ids & actual_squad_ids
                        followed_transfers = 1 if len(overlap) >= 13 else 0  # 13/15 threshold
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass

            actual_chip = chip_map.get(current_event)
            recommended_chip = rec.get("chip_suggestion")
            followed_chip = 1 if actual_chip == recommended_chip else 0

            self.db.save_outcome(
                season_id=season_id,
                gameweek=current_event,
                followed_transfers=followed_transfers,
                followed_captain=followed_captain,
                followed_chip=followed_chip,
                recommended_points=recommended_points,
                actual_points=actual_points,
                point_delta=point_delta,
            )
            outcome = {
                "followed_transfers": followed_transfers,
                "followed_captain": followed_captain,
                "followed_chip": followed_chip,
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
    # Action Plan
    # -------------------------------------------------------------------

    def _get_next_fixture_map(self, bootstrap: dict, next_gw: int) -> dict:
        """Map team_code -> opponent string like 'BOU (H)' for the given GW."""
        fixtures = self._load_fixtures()
        if not fixtures:
            return {}
        id_to_short = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        result = {}
        for f in fixtures:
            if f.get("event") != next_gw:
                continue
            for side, opp_side, tag in [("team_h", "team_a", "H"), ("team_a", "team_h", "A")]:
                tid = f[side]
                code = id_to_code.get(tid)
                opp_name = id_to_short.get(f[opp_side], "?")
                if code is not None:
                    existing = result.get(code, "")
                    fixture_str = f"{opp_name} ({tag})"
                    # Append for DGW (multiple fixtures in same GW)
                    result[code] = f"{existing} + {fixture_str}" if existing else fixture_str
        return result

    def get_action_plan(self, manager_id: int) -> dict:
        """Build clear action items from the latest recommendation + strategic plan."""
        season = self.db.get_season(manager_id)
        if not season:
            return {"error": "No active season."}
        season_id = season["id"]

        bootstrap = self._load_bootstrap()
        next_gw = self._get_next_gw(bootstrap)
        if not next_gw:
            return {"error": "Could not determine next gameweek."}

        # Get GW deadline
        deadline = None
        for event in bootstrap.get("events", []):
            if event.get("id") == next_gw:
                deadline = event.get("deadline_time", "")
                break

        rec = self.db.get_recommendation(season_id, next_gw)
        if not rec:
            return {"error": f"No recommendation for GW{next_gw}. Generate one first."}

        # Build fixture map for opponent lookup on incoming players
        fixture_map = self._get_next_fixture_map(bootstrap, next_gw)
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, _, code_to_short = self._get_team_maps(bootstrap)

        steps = []
        priority = 1

        def _player_dict(raw: dict, fixture_map: dict) -> dict:
            """Extract structured player data for the frontend."""
            pid = raw.get("player_id")
            team_code = raw.get("team_code")
            team = raw.get("team", "")

            # Enrich from bootstrap if team info missing
            if (not team_code or not team) and pid:
                el = elements_map.get(pid, {})
                if not team_code:
                    team_code = id_to_code.get(el.get("team"))
                if not team:
                    team = code_to_short.get(team_code, "")

            d = {
                "player_id": pid,
                "web_name": raw.get("web_name", "?"),
                "position": raw.get("position", ""),
                "team": team,
                "team_code": team_code,
                "cost": raw.get("cost"),
            }
            # Normalize predicted points field
            pts = raw.get("predicted_next_gw_points") or raw.get("predicted_points")
            d["predicted_next_gw_points"] = pts
            # Opponent lookup from fixture map
            d["opponent"] = fixture_map.get(team_code, "") if team_code else ""
            return d

        # Transfers
        transfers = json.loads(rec.get("transfers_json") or "[]")
        has_pairs = transfers and isinstance(transfers[0], dict) and "out" in transfers[0]
        if has_pairs and transfers:
            for pair in transfers:
                out_raw = pair.get("out", {})
                in_raw = pair.get("in", {})
                out_name = out_raw.get("web_name", "?")
                in_name = in_raw.get("web_name", "?")
                step = {
                    "action": "transfer",
                    "description": f"Transfer out {out_name}, bring in {in_name}",
                    "priority": priority,
                    "player_out": _player_dict(out_raw, fixture_map),
                    "player_in": _player_dict(in_raw, fixture_map),
                }
                steps.append(step)
                priority += 1
        elif not has_pairs:
            # Check for old format or no transfers
            in_t = [t for t in transfers if t.get("direction") == "in"]
            out_t = [t for t in transfers if t.get("direction") == "out"]
            for i in range(max(len(in_t), len(out_t))):
                out_raw = out_t[i] if i < len(out_t) else {}
                in_raw = in_t[i] if i < len(in_t) else {}
                out_name = out_raw.get("web_name", "?") if out_raw else "?"
                in_name = in_raw.get("web_name", "?") if in_raw else "?"
                step = {
                    "action": "transfer",
                    "description": f"Transfer out {out_name}, bring in {in_name}",
                    "priority": priority,
                }
                if out_raw:
                    step["player_out"] = _player_dict(out_raw, fixture_map)
                if in_raw:
                    step["player_in"] = _player_dict(in_raw, fixture_map)
                steps.append(step)
                priority += 1

        if not steps:
            steps.append({
                "action": "transfer",
                "description": "No transfers - bank your free transfer",
                "priority": priority,
            })
            priority += 1

        # Captain
        captain_name = rec.get("captain_name")
        captain_id = rec.get("captain_id")
        if captain_name:
            steps.append({
                "action": "captain",
                "description": f"Set captain to {captain_name}",
                "priority": priority,
                "captain_id": captain_id,
                "captain_name": captain_name,
            })
            priority += 1

        # Chip
        chip_suggestion = rec.get("chip_suggestion")
        if chip_suggestion:
            chip_labels = {
                "wildcard": "Wildcard", "freehit": "Free Hit",
                "bboost": "Bench Boost", "3xc": "Triple Captain",
            }
            chip_step = {
                "action": "chip",
                "description": f"Activate {chip_labels.get(chip_suggestion, chip_suggestion)}",
                "priority": priority,
            }

            # For WC/FH, include the full new squad
            if chip_suggestion in ("wildcard", "freehit"):
                raw_squad = rec.get("new_squad_json")
                if raw_squad:
                    try:
                        squad_players = json.loads(raw_squad)
                        # Enrich each player with team/opponent
                        enriched = []
                        for p in squad_players:
                            pid = p.get("player_id")
                            el = elements_map.get(pid, {})
                            tc = p.get("team_code") or id_to_code.get(el.get("team"))
                            team = code_to_short.get(tc, p.get("team", ""))
                            opp = fixture_map.get(tc, "")
                            enriched.append({
                                "player_id": pid,
                                "web_name": p.get("web_name") or el.get("web_name", "?"),
                                "position": p.get("position") or el.get("position", ""),
                                "team": team,
                                "cost": p.get("cost"),
                                "predicted_next_gw_points": p.get("predicted_points") or p.get("predicted_next_gw_points"),
                                "starter": p.get("starter", False),
                                "is_captain": p.get("is_captain", False),
                                "is_vice_captain": p.get("is_vice_captain", False),
                                "opponent": opp,
                            })
                        chip_step["new_squad"] = enriched
                    except (json.JSONDecodeError, TypeError):
                        pass

            steps.append(chip_step)
            priority += 1

        # Bench order hint
        steps.append({
            "action": "bench_order",
            "description": "Check bench order (highest predicted points first off bench)",
            "priority": priority,
        })

        # Rationale from strategic plan
        rationale = ""
        plan_row = self.db.get_strategic_plan(season_id)
        if plan_row and plan_row.get("plan_json"):
            try:
                plan = json.loads(plan_row["plan_json"])
                rationale = plan.get("rationale", "")
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "gameweek": next_gw,
            "deadline": deadline,
            "steps": steps,
            "rationale": rationale,
            "predicted_points": rec.get("predicted_points"),
        }

    # -------------------------------------------------------------------
    # Outcomes
    # -------------------------------------------------------------------

    def get_outcomes(self, manager_id: int) -> list[dict]:
        """Return all recorded outcomes for the season."""
        season = self.db.get_season(manager_id)
        if not season:
            return []
        return self.db.get_outcomes(season["id"])

    # -------------------------------------------------------------------
    # Plan Health Check
    # -------------------------------------------------------------------

    def check_plan_health(self, manager_id: int) -> dict:
        """Lightweight check: is the current strategic plan still valid?

        Uses bootstrap availability data + stored plan. Does NOT regenerate
        predictions (expensive). Returns {healthy, triggers, summary}.
        """
        season = self.db.get_season(manager_id)
        if not season:
            return {"healthy": True, "triggers": [], "summary": {"critical": 0, "moderate": 0}}
        season_id = season["id"]

        plan_row = self.db.get_strategic_plan(season_id)
        if not plan_row or not plan_row.get("plan_json"):
            return {"healthy": True, "triggers": [], "summary": {"critical": 0, "moderate": 0}}

        try:
            current_plan = json.loads(plan_row["plan_json"])
        except (json.JSONDecodeError, TypeError):
            return {"healthy": True, "triggers": [], "summary": {"critical": 0, "moderate": 0}}

        # Load bootstrap for availability
        try:
            bootstrap = self._load_bootstrap()
        except FileNotFoundError:
            return {"healthy": True, "triggers": [], "summary": {"critical": 0, "moderate": 0}}

        elements = bootstrap.get("elements", [])

        # Build squad_changes from injured/doubtful players
        squad_changes = {}
        for el in elements:
            status = el.get("status", "a")
            chance = el.get("chance_of_playing_next_round")
            if status != "a" or (chance is not None and chance < 75):
                squad_changes[el["id"]] = {
                    "status": status,
                    "chance_of_playing": chance,
                    "web_name": el.get("web_name", "Unknown"),
                }

        # Get fixture calendar for fixture change detection
        fixture_calendar = self.db.get_fixture_calendar(season_id)

        # Call detect_plan_invalidation (without new predictions — fixture/injury checks only)
        triggers = detect_plan_invalidation(
            current_plan,
            new_predictions={},  # No fresh predictions — skip prediction shift checks
            fixture_calendar=fixture_calendar,
            squad_changes=squad_changes,
        )

        critical = sum(1 for t in triggers if t["severity"] == "critical")
        moderate = sum(1 for t in triggers if t["severity"] == "moderate")

        return {
            "healthy": critical == 0,
            "triggers": triggers,
            "summary": {"critical": critical, "moderate": moderate},
        }

    # -------------------------------------------------------------------
    # Pre-Season Plan
    # -------------------------------------------------------------------

    def generate_preseason_plan(self, manager_id: int, progress_fn=None) -> dict:
        """Pre-GW1: select initial squad + full season chip plan."""
        def log(msg):
            if progress_fn:
                progress_fn(msg)
            else:
                print(msg)

        bootstrap = self._load_bootstrap()
        elements = bootstrap.get("elements", [])
        elements_map = self._get_elements_map(bootstrap)
        id_to_code, id_to_short, code_to_short = self._get_team_maps(bootstrap)

        # Check if season already started
        for event in bootstrap.get("events", []):
            if event.get("is_current") or event.get("is_next"):
                # Season has started — this endpoint shouldn't be used
                next_gw = event["id"]
                break
        else:
            next_gw = 1  # True pre-season

        log("Pre-season plan: generating predictions...")

        # Try to generate predictions
        import pandas as pd
        from src.predict import build_preseason_predictions, format_predictions

        try:
            from src.data_fetcher import load_all_data
            from src.feature_engineering import build_features

            data = load_all_data()
            df = build_features(data)
            preds = build_preseason_predictions(df, data, bootstrap)
        except Exception as exc:
            log(f"  Prediction generation failed: {exc}")
            preds = pd.DataFrame()

        if preds.empty:
            # Fall back to pure price-based heuristic
            log("  Using price-based heuristic for initial squad...")
            element_type_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
            rows = []
            for el in elements:
                cost = el.get("now_cost", 0) / 10
                rows.append({
                    "player_id": el["id"],
                    "web_name": el.get("web_name", "Unknown"),
                    "position": element_type_map.get(el.get("element_type"), "MID"),
                    "cost": cost,
                    "team_code": id_to_code.get(el.get("team")),
                    "team": id_to_short.get(el.get("team"), ""),
                    "predicted_next_gw_points": round((cost / 10) * 0.5, 2),
                })
            preds = pd.DataFrame(rows)
        else:
            # Enrich with metadata
            if "position_clean" in preds.columns and "position" not in preds.columns:
                preds = preds.rename(columns={"position_clean": "position"})

            # Add cost/team from bootstrap
            meta_rows = []
            for el in elements:
                tid = el.get("team")
                meta_rows.append({
                    "player_id": el["id"],
                    "cost": el.get("now_cost", 0) / 10,
                    "team_code": id_to_code.get(tid),
                    "team": id_to_short.get(tid, ""),
                })
            meta_df = pd.DataFrame(meta_rows)
            # Merge metadata, avoid duplicates
            for col in ["cost", "team_code", "team"]:
                if col in preds.columns:
                    preds = preds.drop(columns=[col])
            preds = preds.merge(meta_df, on="player_id", how="left")

        target_col = "predicted_next_gw_points"
        if target_col not in preds.columns:
            return {"error": "Could not generate predictions."}

        # Solve MILP for best initial squad (budget=100.0)
        log("  Solving for optimal initial squad (budget: 100.0m)...")
        pool = preds.dropna(subset=["position", "cost", target_col]).copy()

        result = solve_milp_team(pool, target_col, budget=100.0)
        if not result:
            return {"error": "Could not find valid initial squad."}

        log(f"  Initial squad: {result['total_cost']:.1f}m, XI predicted: {result['starting_points']:.1f} pts")

        # Create or update season record
        entry = None
        try:
            from src.data_fetcher import fetch_manager_entry
            entry = fetch_manager_entry(manager_id)
        except Exception:
            pass
        manager_name = ""
        team_name = ""
        if entry:
            manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
            team_name = entry.get("name", "")

        from src.data_fetcher import detect_current_season
        season_id = self.db.create_season(
            manager_id=manager_id,
            manager_name=manager_name,
            team_name=team_name,
            season_name=detect_current_season(),
            start_gw=1,
        )

        # Build fixture calendar
        log("  Building fixture calendar...")
        self.update_fixture_calendar(season_id)

        # Run chip evaluator for full-season plan
        log("  Evaluating chips for full season...")
        squad_ids = {p["player_id"] for p in result["players"]}
        fixture_calendar = self.db.get_fixture_calendar(season_id)

        # Build future predictions dict for chip evaluator
        future_preds = {}
        if not pool.empty:
            # Use same predictions for all GWs (rough pre-season estimate)
            gw1_df = pool.rename(columns={target_col: "predicted_points"}).copy()
            gw1_df["confidence"] = 0.5
            for gw in range(1, 6):
                future_preds[gw] = gw1_df.copy()

        chip_evaluator = ChipEvaluator()
        available_chips = {"wildcard", "freehit", "bboost", "3xc"}
        chip_heatmap = chip_evaluator.evaluate_all_chips(
            squad_ids, 100.0, available_chips,
            future_preds, fixture_calendar,
        )
        chip_synergies = chip_evaluator.evaluate_chip_synergies(
            chip_heatmap, available_chips,
        )

        synthesizer = PlanSynthesizer()
        chip_schedule = synthesizer._plan_chip_schedule(
            chip_heatmap, chip_synergies, available_chips,
        )

        # Pick captain: use solver captain_id, or highest-predicted starter
        best_captain_id = result.get("captain_id")
        best_captain_name = None
        if best_captain_id:
            cap_p = next((p for p in result["players"] if p.get("player_id") == best_captain_id), None)
            best_captain_name = cap_p.get("web_name") if cap_p else None
        if not best_captain_id and result["starters"]:
            best_starter = max(result["starters"], key=lambda p: p.get(target_col, 0))
            best_captain_id = best_starter.get("player_id")
            best_captain_name = best_starter.get("web_name")

        # Save recommendation for GW1
        self.db.save_recommendation(
            season_id=season_id,
            gameweek=1,
            transfers_json=json.dumps([]),
            captain_id=best_captain_id,
            captain_name=best_captain_name,
            chip_suggestion=None,
            chip_values_json=json.dumps({}),
            bank_analysis_json=json.dumps({}),
            new_squad_json=json.dumps(scrub_nan(result["players"])),
            predicted_points=result["starting_points"],
        )

        # Save strategic plan
        strategic_plan = {
            "timeline": [],
            "chip_schedule": chip_schedule,
            "chip_synergies": chip_synergies[:3],
            "rationale": f"Pre-season plan. Chip plan: {', '.join(f'{k} GW{v}' for k, v in chip_schedule.items())}.",
            "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }
        self.db.save_strategic_plan(
            season_id=season_id,
            as_of_gw=1,
            plan_json=json.dumps(scrub_nan_recursive(strategic_plan)),
            chip_heatmap_json=json.dumps(scrub_nan_recursive(chip_heatmap)),
        )

        log("Pre-season plan complete.")
        return {
            "initial_squad": scrub_nan(result["players"]),
            "total_cost": result["total_cost"],
            "starting_points": result["starting_points"],
            "chip_schedule": chip_schedule,
            "chip_heatmap": scrub_nan_recursive(chip_heatmap),
        }

    # -------------------------------------------------------------------
    # Chip Evaluation
    # -------------------------------------------------------------------

    def _evaluate_chips(self, season_id: int, manager_id: int,
                        bootstrap: dict, pred_df, current_squad_ids: set,
                        total_budget: float, history: dict) -> dict:
        """Estimate point value of each available chip.

        Uses fixture calendar for DGW/BGW awareness: boosts BB/TC value
        in DGW weeks, discounts if a better DGW week exists later.
        """
        import pandas as pd

        chips_used_list = history.get("chips", [])

        # Determine which chips are still available (all chips reset at GW20)
        available = set()
        next_gw = self._get_next_gw(bootstrap) or 1
        for chip in ["wildcard", "freehit", "bboost", "3xc"]:
            chip_events = [c["event"] for c in chips_used_list if c["name"] == chip]
            if next_gw <= 19:
                if not any(e <= 19 for e in chip_events):
                    available.add(chip)
            else:
                if not any(e >= 20 for e in chip_events):
                    available.add(chip)

        target_col = "predicted_next_gw_points"
        chip_values = {}

        if target_col not in pred_df.columns:
            return {chip: 0.0 for chip in available}

        # Current XI predicted points (formation-aware)
        squad_preds = pred_df[pred_df["player_id"].isin(current_squad_ids)]
        if squad_preds.empty:
            return {chip: 0.0 for chip in available}

        # Use formation-aware XI selection if position data is available
        _temp = squad_preds.copy()
        _temp["predicted_points"] = _temp[target_col]
        if "position" in _temp.columns:
            try:
                xi = MultiWeekPlanner._select_formation_xi(_temp)
                current_xi_pts = xi["predicted_points"].sum()
                bench_preds = _temp[~_temp.index.isin(xi.index)]
            except Exception:
                current_xi_pts = squad_preds.nlargest(11, target_col)[target_col].sum()
                bench_preds = None
        else:
            current_xi_pts = squad_preds.nlargest(11, target_col)[target_col].sum()
            bench_preds = None

        # Bug 55 fix: Compute cap_col for captain optimization in FH/WC evaluation
        cap_col = "captain_score" if "captain_score" in pred_df.columns else None

        # Bug 55 fix: Add captain bonus to current_xi_pts for fair comparison
        if cap_col and cap_col in squad_preds.columns:
            if "position" in _temp.columns and bench_preds is not None:
                # xi was computed above via formation selection
                best_cap_idx = xi[cap_col].idxmax() if cap_col in xi.columns else xi["predicted_points"].idxmax()
                current_xi_pts += xi.loc[best_cap_idx, "predicted_points"]  # captain doubles
            else:
                # Fallback: use best captain_score from top 11
                top11 = squad_preds.nlargest(11, target_col)
                if cap_col in top11.columns:
                    best_cap_idx = top11[cap_col].idxmax()
                    current_xi_pts += top11.loc[best_cap_idx, target_col]

        # Fixture awareness: count DGW teams for current and future GWs
        fixture_calendar = self.db.get_fixture_calendar(season_id, from_gw=next_gw)
        dgw_by_gw = {}
        for f in fixture_calendar:
            gw = f["gameweek"]
            if f.get("is_dgw"):
                dgw_by_gw[gw] = dgw_by_gw.get(gw, 0) + 1
        current_gw_dgw_count = dgw_by_gw.get(next_gw, 0)
        future_max_dgw = max(
            (count for gw, count in dgw_by_gw.items() if gw > next_gw),
            default=0,
        )

        if "bboost" in available:
            # Bench Boost: sum of bench predictions (formation-aware)
            if bench_preds is not None and len(bench_preds) >= 4:
                bench_pts = bench_preds["predicted_points"].sum()
            else:
                # Bug 52 fix: Handle <15 squad predictions gracefully
                all_n = squad_preds.nlargest(min(15, len(squad_preds)), target_col)[target_col]
                xi_count = min(11, len(all_n))
                bench_pts = all_n.iloc[xi_count:].sum() if len(all_n) > xi_count else 0
            # DGW boost: bench players with double fixtures are worth more
            dgw_boost = 1.0 + current_gw_dgw_count * 0.15
            bench_pts *= dgw_boost
            # Discount if a future GW has significantly more DGW teams
            if future_max_dgw > current_gw_dgw_count + 2:
                bench_pts *= 0.7  # Likely better to save BB
            chip_values["bboost"] = round(bench_pts, 1)

        if "3xc" in available:
            # Triple Captain: extra captain points (highest predicted * 1)
            best_pred = squad_preds[target_col].max()
            # DGW boost for TC too
            dgw_boost = 1.0 + current_gw_dgw_count * 0.1
            chip_values["3xc"] = round(best_pred * dgw_boost, 1)

        if "freehit" in available:
            # Free Hit: unconstrained best team vs current
            pool = pred_df.dropna(subset=["position", "cost", target_col]).copy()
            # Bug 55 fix: Pass captain_col for captain optimization
            fh_result = solve_milp_team(pool, target_col, budget=total_budget, captain_col=cap_col)
            if fh_result:
                chip_values["freehit"] = round(fh_result["starting_points"] - current_xi_pts, 1)
            else:
                chip_values["freehit"] = 0.0

        if "wildcard" in available:
            # Wildcard: unconstrained best team vs current (similar to FH but permanent)
            pool = pred_df.dropna(subset=["position", "cost", target_col]).copy()
            # Bug 55 fix: Pass captain_col for captain optimization
            wc_result = solve_milp_team(pool, target_col, budget=total_budget, captain_col=cap_col)
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
        # Sum best available players (up to 11) rather than requiring exactly 11
        current_xi_pts = current_pred.nlargest(min(11, len(current_pred)), gw_col)[gw_col].sum() if not current_pred.empty else 0

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
                    week1_budget = total_budget
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

    def predict_price_changes(self, season_id: int) -> list[dict]:
        """Predict price changes using ownership-based algorithm approximation.

        Uses: transfer_ratio = net_transfers / (ownership_pct * 100_000)
        Rise if ratio > 0.005, fall if < -0.005.
        Probability = min(1.0, abs(ratio) / 0.01).
        """
        bootstrap = self._load_bootstrap()
        elements = bootstrap.get("elements", [])
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
        code_to_short = {t["code"]: t["short_name"] for t in bootstrap.get("teams", [])}

        predictions = []
        for el in elements:
            net = el.get("transfers_in_event", 0) - el.get("transfers_out_event", 0)
            ownership = el.get("selected_by_percent")
            if ownership is None:
                continue
            try:
                ownership_pct = float(ownership)
            except (TypeError, ValueError):
                continue

            if ownership_pct < 0.1:
                continue

            transfer_ratio = net / (ownership_pct * 100_000)

            if abs(transfer_ratio) < 0.005:
                continue

            direction = "rise" if transfer_ratio > 0 else "fall"
            probability = min(1.0, abs(transfer_ratio) / 0.01)
            estimated_change = 0.1 if direction == "rise" else -0.1

            tc = id_to_code.get(el.get("team"))
            predictions.append({
                "player_id": el["id"],
                "web_name": el.get("web_name", "Unknown"),
                "team": code_to_short.get(tc, ""),
                "price": el.get("now_cost", 0) / 10,
                "ownership": ownership_pct,
                "net_transfers": net,
                "transfer_ratio": round(transfer_ratio, 6),
                "direction": direction,
                "probability": round(probability, 3),
                "estimated_change": estimated_change,
            })

        predictions.sort(key=lambda p: p["probability"], reverse=True)
        return predictions

    def get_price_history(self, season_id: int, player_ids: list[int] | None = None, days: int = 14) -> dict:
        """Get price history from price_tracker table.

        Returns {player_id: {web_name, snapshots: [{date, price, net_transfers}]}}.
        """
        all_history = self.db.get_price_history(season_id)
        if not all_history:
            return {}

        # Filter by player_ids if provided
        if player_ids:
            pid_set = set(player_ids)
            all_history = [h for h in all_history if h["player_id"] in pid_set]

        # Filter by days
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        all_history = [h for h in all_history if h.get("snapshot_date", "") >= cutoff]

        # Group by player_id
        result = {}
        for h in all_history:
            pid = h["player_id"]
            if pid not in result:
                result[pid] = {
                    "web_name": h.get("web_name", "Unknown"),
                    "snapshots": [],
                }
            net = (h.get("transfers_in_event") or 0) - (h.get("transfers_out_event") or 0)
            result[pid]["snapshots"].append({
                "date": h.get("snapshot_date"),
                "price": h.get("price"),
                "net_transfers": net,
            })

        return result

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
        id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
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
                        "team_code": id_to_code.get(tid),
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
                        "team_code": id_to_code.get(tid),
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

        # Compute available chips with half-season reset awareness
        # WC/FH reset at GW20 (available once per half: GW1-19 and GW20-38)
        # All 4 chips (WC, FH, BB, TC) available once per half (GW1-19 and GW20-38), 8 total
        current_gw = season.get("current_gw", 1)
        # Bug 53 fix: Use next_gw for chip availability check, since chips
        # are played in future GWs, not the current (already-played) one.
        next_gw = min(current_gw + 1, 38)
        all_chips_list = [
            {"name": "wildcard", "label": "Wildcard"},
            {"name": "freehit", "label": "Free Hit"},
            {"name": "bboost", "label": "Bench Boost"},
            {"name": "3xc", "label": "Triple Captain"},
        ]
        chips_status = []
        for chip in all_chips_list:
            chip_events = [c["gameweek"] for c in chips_used if c["chip_used"] == chip["name"]]
            # Determine which usage matters for current half
            if next_gw <= 19:
                used = any(e <= 19 for e in chip_events)
                used_in = next((e for e in chip_events if e <= 19), None)
            else:
                used = any(e >= 20 for e in chip_events)
                used_in = next((e for e in chip_events if e >= 20), None)
            chips_status.append({
                "name": chip["name"],
                "label": chip["label"],
                "used": used,
                "used_gw": used_in,
            })

        # Latest snapshot for summary
        latest = snapshots[-1] if snapshots else {}

        # Points per GW
        points_per_gw = [
            {"gameweek": s["gameweek"], "points": s.get("points", 0)}
            for s in snapshots
        ]

        # Build accuracy history from outcomes
        accuracy_history = []
        for o in outcomes:
            if o.get("recommended_points") is not None and o.get("actual_points") is not None:
                accuracy_history.append({
                    "gameweek": o["gameweek"],
                    "predicted_points": o["recommended_points"],
                    "actual_points": o["actual_points"],
                    "delta": o.get("point_delta", 0),
                })

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
            "accuracy_history": accuracy_history,
            "recommendations_count": len(recommendations),
            "outcomes_count": len(outcomes),
        }
