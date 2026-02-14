"""Strategic planning brain for FPL season management.

Contains ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer,
and reactive re-planning logic. All decisions are made in context of the
bigger picture — not just next GW.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.solver import scrub_nan, solve_milp_team, solve_transfer_milp


# ---------------------------------------------------------------------------
# ChipEvaluator — Phase 1b
# ---------------------------------------------------------------------------

class ChipEvaluator:
    """Evaluate chip value across all remaining GWs in the season."""

    def evaluate_all_chips(
        self,
        current_squad_ids: set[int],
        total_budget: float,
        available_chips: set[str],
        future_predictions: dict[int, pd.DataFrame],
        fixture_calendar: list[dict],
    ) -> dict[str, dict[int, float]]:
        """Return {chip_name: {gw: estimated_value}} for all remaining GWs.

        For GWs within the prediction horizon, uses model predictions.
        For GWs beyond, uses fixture-calendar heuristics only.
        """
        if not available_chips:
            return {}

        # Build fixture lookup: gw -> {team_id -> {fixture_count, is_dgw, is_bgw, fdr_avg}}
        fx_by_gw = self._build_fixture_lookup(fixture_calendar)

        # Get the set of GWs we have predictions for
        pred_gws = sorted(future_predictions.keys())
        if not pred_gws:
            return {}

        # All GWs from fixture calendar
        all_gws = sorted(fx_by_gw.keys())
        if not all_gws:
            all_gws = pred_gws

        # Build squad team mapping for heuristic evaluation
        # (we need to know which teams are in the current squad for BB/TC heuristics)

        chip_values: dict[str, dict[int, float]] = {}

        if "bboost" in available_chips:
            chip_values["bboost"] = self._evaluate_bench_boost(
                current_squad_ids, future_predictions, fx_by_gw, all_gws, pred_gws,
            )

        if "3xc" in available_chips:
            chip_values["3xc"] = self._evaluate_triple_captain(
                current_squad_ids, future_predictions, fx_by_gw, all_gws, pred_gws,
            )

        if "freehit" in available_chips:
            chip_values["freehit"] = self._evaluate_free_hit(
                current_squad_ids, total_budget, future_predictions,
                fx_by_gw, all_gws, pred_gws,
            )

        if "wildcard" in available_chips:
            chip_values["wildcard"] = self._evaluate_wildcard(
                current_squad_ids, total_budget, future_predictions,
                fx_by_gw, all_gws, pred_gws,
            )

        return chip_values

    def _build_fixture_lookup(self, fixture_calendar: list[dict]) -> dict:
        """Build gw -> {team_id -> fixture_info} lookup."""
        fx_by_gw: dict[int, dict[int, dict]] = {}
        for f in fixture_calendar:
            gw = f["gameweek"]
            tid = f["team_id"]
            if gw not in fx_by_gw:
                fx_by_gw[gw] = {}
            fx_by_gw[gw][tid] = {
                "fixture_count": f.get("fixture_count", 1),
                "is_dgw": f.get("is_dgw", 0),
                "is_bgw": f.get("is_bgw", 0),
                "fdr_avg": f.get("fdr_avg"),
            }
        return fx_by_gw

    def _count_dgw_teams(self, fx_by_gw: dict, gw: int) -> int:
        """Count how many teams have a DGW in the given GW."""
        if gw not in fx_by_gw:
            return 0
        return sum(1 for t in fx_by_gw[gw].values() if t.get("is_dgw"))

    def _count_bgw_teams(self, fx_by_gw: dict, gw: int) -> int:
        """Count how many teams have a BGW in the given GW."""
        if gw not in fx_by_gw:
            return 0
        return sum(1 for t in fx_by_gw[gw].values() if t.get("is_bgw"))

    def _avg_fdr(self, fx_by_gw: dict, gw: int) -> float:
        """Average FDR across all teams with fixtures in a GW."""
        if gw not in fx_by_gw:
            return 3.0
        fdrs = [t["fdr_avg"] for t in fx_by_gw[gw].values()
                if t.get("fdr_avg") is not None and not t.get("is_bgw")]
        return sum(fdrs) / len(fdrs) if fdrs else 3.0

    def _evaluate_bench_boost(
        self, current_squad_ids, future_predictions, fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Bench Boost value per GW.

        Within prediction horizon: sum of 4 bench predictions.
        Beyond: DGW count heuristic (more DGWs = higher bench value).
        """
        values = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                squad_preds = gw_df[gw_df["player_id"].isin(current_squad_ids)]
                if len(squad_preds) >= 15:
                    all_15 = squad_preds.nlargest(15, "predicted_points")
                    bench_pts = all_15.tail(4)["predicted_points"].sum()
                else:
                    bench_pts = squad_preds["predicted_points"].sum() * 0.25 if not squad_preds.empty else 0
                # DGW multiplier: bench players with DGW fixtures are worth more
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                dgw_boost = 1.0 + (n_dgw * 0.15)  # 15% boost per DGW team
                values[gw] = round(bench_pts * dgw_boost, 1)
            else:
                # Heuristic: base bench value ~6 pts, boosted by DGW count
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                n_bgw = self._count_bgw_teams(fx_by_gw, gw)
                base = 6.0
                values[gw] = round(base * (1 + n_dgw * 0.3) * (1 - n_bgw * 0.05), 1)

        return values

    def _evaluate_triple_captain(
        self, current_squad_ids, future_predictions, fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Triple Captain value per GW.

        Within prediction horizon: best player's predicted points (extra 1x).
        Beyond: heuristic based on DGW + low FDR.
        """
        values = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                squad_preds = gw_df[gw_df["player_id"].isin(current_squad_ids)]
                if not squad_preds.empty:
                    best = squad_preds["predicted_points"].max()
                    n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                    dgw_boost = 1.0 + (n_dgw * 0.1)
                    values[gw] = round(best * dgw_boost, 1)
                else:
                    values[gw] = 0.0
            else:
                # Heuristic: base TC value ~5 pts from premium captain
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                avg_fdr = self._avg_fdr(fx_by_gw, gw)
                fdr_factor = max(0, (3.5 - avg_fdr) / 2)  # Lower FDR = higher value
                values[gw] = round(5.0 * (1 + n_dgw * 0.4) * (1 + fdr_factor), 1)

        return values

    def _evaluate_free_hit(
        self, current_squad_ids, total_budget, future_predictions,
        fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Free Hit value per GW.

        Within prediction horizon: unconstrained best XI minus current squad points.
        Beyond: heuristic based on BGW count (FH most valuable in BGWs).
        """
        values = {}

        for gw in all_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                # Current squad points this GW
                squad_preds = gw_df[gw_df["player_id"].isin(current_squad_ids)]
                if not squad_preds.empty and len(squad_preds) >= 11:
                    current_pts = squad_preds.nlargest(11, "predicted_points")["predicted_points"].sum()
                else:
                    current_pts = squad_preds["predicted_points"].sum() if not squad_preds.empty else 0

                # Solve unconstrained best XI (need full pool with position/cost)
                pool = gw_df.copy()
                if "position" in pool.columns and "cost" in pool.columns:
                    fh_result = solve_milp_team(pool, "predicted_points", budget=1000)
                    if fh_result:
                        fh_pts = fh_result["starting_points"]
                        values[gw] = round(max(0, fh_pts - current_pts), 1)
                    else:
                        values[gw] = 0.0
                else:
                    # Can't solve without position/cost, use estimate
                    all_top11 = gw_df.nlargest(11, "predicted_points")["predicted_points"].sum()
                    values[gw] = round(max(0, all_top11 - current_pts), 1)
            else:
                # Heuristic: FH is great in BGWs, moderate otherwise
                n_bgw = self._count_bgw_teams(fx_by_gw, gw)
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                base = 3.0
                values[gw] = round(base + n_bgw * 2.0 + n_dgw * 0.5, 1)

        return values

    def _evaluate_wildcard(
        self, current_squad_ids, total_budget, future_predictions,
        fx_by_gw, all_gws, pred_gws,
    ) -> dict[int, float]:
        """Wildcard value per GW.

        Within prediction horizon: solve best squad over next 3 GWs minus current.
        Beyond: heuristic based on upcoming fixture swings.
        """
        values = {}

        for gw in all_gws:
            if gw in future_predictions:
                # Sum predictions over next 3 GWs from this point
                look_ahead_gws = [g for g in pred_gws if gw <= g <= gw + 2]
                if not look_ahead_gws:
                    values[gw] = 0.0
                    continue

                # Build combined prediction (sum over 3 GWs)
                combined = None
                for lag in look_ahead_gws:
                    if lag in future_predictions:
                        lag_df = future_predictions[lag][["player_id", "predicted_points"]].copy()
                        lag_df = lag_df.rename(columns={"predicted_points": f"pts_{lag}"})
                        if combined is None:
                            combined = lag_df
                        else:
                            combined = combined.merge(lag_df, on="player_id", how="outer")

                if combined is None:
                    values[gw] = 0.0
                    continue

                pts_cols = [c for c in combined.columns if c.startswith("pts_")]
                combined["total_pts"] = combined[pts_cols].sum(axis=1)

                # Current squad value over these GWs
                squad_total = combined[combined["player_id"].isin(current_squad_ids)]["total_pts"]
                if len(squad_total) >= 11:
                    current_3gw = squad_total.nlargest(11).sum()
                else:
                    current_3gw = squad_total.sum()

                # Solve best squad for the combined period
                # Need position/cost data - merge from first GW predictions
                first_gw_df = future_predictions[look_ahead_gws[0]]
                meta_cols = [c for c in first_gw_df.columns
                             if c in ("player_id", "position", "cost", "team_code", "team")]
                if "position" in first_gw_df.columns and "cost" in first_gw_df.columns:
                    pool = combined.merge(
                        first_gw_df[meta_cols].drop_duplicates("player_id"),
                        on="player_id", how="left",
                    )
                    pool = pool.dropna(subset=["position", "cost"])
                    wc_result = solve_milp_team(pool, "total_pts", budget=total_budget)
                    if wc_result:
                        values[gw] = round(max(0, wc_result["starting_points"] - current_3gw), 1)
                    else:
                        values[gw] = 0.0
                else:
                    values[gw] = 0.0
            else:
                # Heuristic: WC value based on fixture difficulty swings
                n_dgw = self._count_dgw_teams(fx_by_gw, gw)
                avg_fdr = self._avg_fdr(fx_by_gw, gw)
                fdr_improvement = max(0, (3.0 - avg_fdr))
                values[gw] = round(3.0 + n_dgw * 1.5 + fdr_improvement * 2, 1)

        return values

    def evaluate_chip_synergies(
        self,
        chip_values: dict[str, dict[int, float]],
        available_chips: set[str],
    ) -> list[dict]:
        """Evaluate chip pairings and synergies.

        WC in GW X + BB in GW X+1..X+3: WC value increases
        FH for BGW + WC for nearby DGW: complementary strategy
        """
        synergies = []

        if "wildcard" in available_chips and "bboost" in available_chips:
            wc_vals = chip_values.get("wildcard", {})
            bb_vals = chip_values.get("bboost", {})
            for wc_gw, wc_val in wc_vals.items():
                # Look for BB opportunities 1-3 GWs after WC
                for bb_offset in range(1, 4):
                    bb_gw = wc_gw + bb_offset
                    if bb_gw in bb_vals:
                        bb_val = bb_vals[bb_gw]
                        # WC→BB synergy: WC can build a BB-optimized squad
                        synergy_bonus = bb_val * 0.3  # 30% boost from squad optimization
                        combined = wc_val + bb_val + synergy_bonus
                        synergies.append({
                            "chips": ["wildcard", "bboost"],
                            "gws": [wc_gw, bb_gw],
                            "individual_values": [round(wc_val, 1), round(bb_val, 1)],
                            "synergy_bonus": round(synergy_bonus, 1),
                            "combined_value": round(combined, 1),
                            "description": f"WC GW{wc_gw} → BB GW{bb_gw}: build BB-optimized squad",
                        })

        if "freehit" in available_chips and "wildcard" in available_chips:
            fh_vals = chip_values.get("freehit", {})
            wc_vals = chip_values.get("wildcard", {})
            for fh_gw, fh_val in fh_vals.items():
                # FH for awkward GW, WC nearby to restructure
                for wc_offset in range(-2, 3):
                    if wc_offset == 0:
                        continue
                    wc_gw = fh_gw + wc_offset
                    if wc_gw in wc_vals:
                        wc_val = wc_vals[wc_gw]
                        combined = fh_val + wc_val
                        if combined > 10:  # Only report meaningful synergies
                            synergies.append({
                                "chips": ["freehit", "wildcard"],
                                "gws": [fh_gw, wc_gw],
                                "individual_values": [round(fh_val, 1), round(wc_val, 1)],
                                "synergy_bonus": 0.0,
                                "combined_value": round(combined, 1),
                                "description": f"FH GW{fh_gw} + WC GW{wc_gw}: complementary strategy",
                            })

        # Sort by combined value descending
        synergies.sort(key=lambda s: s["combined_value"], reverse=True)
        return synergies[:10]  # Top 10 synergy combinations


# ---------------------------------------------------------------------------
# MultiWeekPlanner — Phase 2
# ---------------------------------------------------------------------------

class MultiWeekPlanner:
    """Rolling 3-GW transfer planner with FT banking, fixture swings, and price awareness."""

    def plan_transfers(
        self,
        current_squad_ids: set[int],
        total_budget: float,
        free_transfers: int,
        future_predictions: dict[int, pd.DataFrame],
        fixture_calendar: list[dict],
        price_alerts: list[dict],
        chip_plan: dict | None = None,
    ) -> list[dict]:
        """Plan transfers over next 3 GWs with forward simulation.

        Returns list of {gw, transfers_in, transfers_out, ft_strategy, rationale,
        squad_ids, predicted_points}.
        """
        pred_gws = sorted(future_predictions.keys())
        if len(pred_gws) < 1:
            return []

        # Build price bonus map: player_id -> bonus points
        price_bonus = self._build_price_bonus(price_alerts)

        # Build fixture swing bonus map for each GW
        fx_lookup = {}
        for f in fixture_calendar:
            gw = f["gameweek"]
            if gw not in fx_lookup:
                fx_lookup[gw] = {}
            fx_lookup[gw][f["team_id"]] = f

        # Take first 3 prediction GWs for planning
        plan_gws = pred_gws[:3]

        # Reduce player pool to top ~150 by predicted points for efficiency
        first_gw_df = future_predictions[plan_gws[0]]
        top_pool_ids = set(first_gw_df.nlargest(150, "predicted_points")["player_id"].tolist())
        # Always include current squad
        top_pool_ids |= current_squad_ids

        # Filter predictions to reduced pool
        filtered_preds = {}
        for gw in plan_gws:
            if gw in future_predictions:
                gw_df = future_predictions[gw]
                filtered_preds[gw] = gw_df[gw_df["player_id"].isin(top_pool_ids)].copy()

        # Forward simulation: try each FT allocation for GW+1
        max_use = min(free_transfers, 3)
        best_path = None
        best_total = -float("inf")

        for use_gw1 in range(0, max_use + 1):
            # Skip chip GWs — if chip is planned for GW1, don't use FTs
            if chip_plan and plan_gws[0] in chip_plan.get("chip_gws", {}):
                chip_name = chip_plan["chip_gws"][plan_gws[0]]
                if chip_name in ("freehit", "wildcard"):
                    if use_gw1 > 0:
                        continue

            path = self._simulate_path(
                plan_gws, filtered_preds, current_squad_ids,
                total_budget, free_transfers, use_gw1,
                price_bonus, fx_lookup,
            )
            if path is None:
                continue

            total_pts = sum(step["predicted_points"] for step in path)
            if total_pts > best_total:
                best_total = total_pts
                best_path = path

        if best_path is None:
            return []

        # Annotate with rationale
        for step in best_path:
            step["rationale"] = self._build_rationale(step, free_transfers)

        return best_path

    def _build_price_bonus(self, price_alerts: list[dict]) -> dict[int, float]:
        """Convert price alerts to bonus points for likely risers."""
        bonus = {}
        for alert in price_alerts:
            if alert.get("direction") == "rise":
                # +0.3 pts per 0.1m expected rise (rough estimate from net transfers)
                net = alert.get("net_transfers", 0)
                estimated_rise = min(0.3, net / 100000)  # Cap at 0.3m
                bonus[alert["player_id"]] = round(estimated_rise * 3.0, 2)
        return bonus

    def _simulate_path(
        self, plan_gws, filtered_preds, current_squad_ids,
        total_budget, free_transfers, use_gw1,
        price_bonus, fx_lookup,
    ) -> list[dict] | None:
        """Simulate a transfer path over 3 GWs given use_gw1 transfers in GW1."""
        path = []
        squad_ids = set(current_squad_ids)
        budget = total_budget
        ft = free_transfers

        for i, gw in enumerate(plan_gws):
            if gw not in filtered_preds:
                break

            gw_df = filtered_preds[gw].copy()

            # Apply price bonus to predicted_points
            if price_bonus and i == 0:  # Only for immediate GW
                gw_df["predicted_points"] = gw_df.apply(
                    lambda r: r["predicted_points"] + price_bonus.get(r["player_id"], 0),
                    axis=1,
                )

            # Apply fixture swing bonus
            if gw in fx_lookup:
                gw_fx = fx_lookup[gw]
                def _fx_bonus(row):
                    if "team_code" not in row or pd.isna(row.get("team_code")):
                        return 0
                    fx = gw_fx.get(int(row["team_code"]), {})
                    fdr = fx.get("fdr_avg", 3.0)
                    if fdr is None:
                        return 0
                    # Bonus for easy fixtures (FDR < 3)
                    return max(0, (3.0 - fdr) * 0.3)
                if "team_code" in gw_df.columns:
                    gw_df["predicted_points"] = gw_df["predicted_points"] + gw_df.apply(_fx_bonus, axis=1)

            if i == 0:
                use_now = use_gw1
            else:
                # For GW2+: use available FTs (greedy — use all)
                use_now = min(ft, 2)

            if use_now == 0:
                # No transfers: keep current squad
                squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)]
                if len(squad_preds) >= 11:
                    pts = squad_preds.nlargest(11, "predicted_points")["predicted_points"].sum()
                else:
                    pts = squad_preds["predicted_points"].sum() if not squad_preds.empty else 0

                path.append({
                    "gw": gw,
                    "transfers_in": [],
                    "transfers_out": [],
                    "ft_used": 0,
                    "ft_available": ft,
                    "predicted_points": round(pts, 2),
                    "squad_ids": list(squad_ids),
                })

                # Roll forward FTs
                ft = min(ft + 1, 5)
            else:
                # Solve transfer MILP
                pool = gw_df.dropna(subset=["predicted_points"])
                if "position" in pool.columns and "cost" in pool.columns:
                    result = solve_transfer_milp(
                        pool, squad_ids, "predicted_points",
                        budget=budget, max_transfers=use_now,
                    )
                    if result:
                        pts = result["starting_points"]
                        new_squad_ids = {p["player_id"] for p in result["players"]}
                        transfers_out = squad_ids - new_squad_ids
                        transfers_in = new_squad_ids - squad_ids

                        path.append({
                            "gw": gw,
                            "transfers_in": [
                                {"player_id": p["player_id"],
                                 "web_name": p.get("web_name", ""),
                                 "position": p.get("position", ""),
                                 "cost": p.get("cost", 0),
                                 "predicted_points": round(p.get("predicted_points", 0), 2)}
                                for p in result["players"] if p["player_id"] in transfers_in
                            ],
                            "transfers_out": [
                                {"player_id": pid}
                                for pid in transfers_out
                            ],
                            "ft_used": len(transfers_in),
                            "ft_available": ft,
                            "predicted_points": round(pts, 2),
                            "squad_ids": list(new_squad_ids),
                        })

                        squad_ids = new_squad_ids
                        budget = result["total_cost"]
                        ft = min(ft - use_now + 1, 5)
                        ft = max(ft, 1)
                    else:
                        # Solver failed, keep current squad
                        squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)]
                        pts = squad_preds.nlargest(11, "predicted_points")["predicted_points"].sum() if len(squad_preds) >= 11 else 0
                        path.append({
                            "gw": gw,
                            "transfers_in": [],
                            "transfers_out": [],
                            "ft_used": 0,
                            "ft_available": ft,
                            "predicted_points": round(pts, 2),
                            "squad_ids": list(squad_ids),
                        })
                        ft = min(ft + 1, 5)
                else:
                    return None

        return path if path else None

    def _build_rationale(self, step: dict, original_ft: int) -> str:
        """Build natural-language rationale for a planning step."""
        gw = step["gw"]
        ft_used = step["ft_used"]
        ft_avail = step["ft_available"]
        transfers_in = step.get("transfers_in", [])

        if ft_used == 0:
            if ft_avail < 5:
                return f"GW{gw}: Bank transfer (save for next week, {ft_avail}→{min(ft_avail+1, 5)} FTs)"
            else:
                return f"GW{gw}: No valuable transfers found (FTs maxed at 5)"
        else:
            names = [t.get("web_name", "?") for t in transfers_in[:3]]
            names_str = ", ".join(names)
            return f"GW{gw}: Use {ft_used} FT(s) — bring in {names_str}"


# ---------------------------------------------------------------------------
# CaptainPlanner — Phase 3a
# ---------------------------------------------------------------------------

class CaptainPlanner:
    """Pre-plan captaincy across the prediction horizon."""

    def plan_captaincy(
        self,
        current_squad_ids: set[int],
        future_predictions: dict[int, pd.DataFrame],
        transfer_plan: list[dict] | None = None,
    ) -> list[dict]:
        """Plan captaincy across future GWs.

        Returns list of {gw, captain_id, captain_name, captain_points,
        vc_id, vc_name, confidence, weak_gw}.
        """
        captain_plan = []
        pred_gws = sorted(future_predictions.keys())

        for gw in pred_gws:
            gw_df = future_predictions[gw]

            # Use transfer plan squad if available for this GW
            squad_ids = current_squad_ids
            if transfer_plan:
                for step in transfer_plan:
                    if step["gw"] == gw and step.get("squad_ids"):
                        squad_ids = set(step["squad_ids"])
                        break

            squad_preds = gw_df[gw_df["player_id"].isin(squad_ids)].copy()
            if squad_preds.empty:
                continue

            # Sort by predicted points
            squad_preds = squad_preds.sort_values("predicted_points", ascending=False)

            captain = squad_preds.iloc[0]
            vc = squad_preds.iloc[1] if len(squad_preds) > 1 else captain

            captain_pts = captain["predicted_points"]
            confidence = captain.get("confidence", 1.0)

            # Flag weak captain GWs (captain predicted < 4 pts)
            weak_gw = captain_pts < 4.0

            captain_plan.append({
                "gw": gw,
                "captain_id": int(captain["player_id"]),
                "captain_name": captain.get("web_name", "Unknown"),
                "captain_points": round(captain_pts, 2),
                "vc_id": int(vc["player_id"]),
                "vc_name": vc.get("web_name", "Unknown"),
                "confidence": round(confidence, 2),
                "weak_gw": weak_gw,
            })

        return captain_plan


# ---------------------------------------------------------------------------
# PlanSynthesizer — Phase 3c
# ---------------------------------------------------------------------------

class PlanSynthesizer:
    """Combine transfer plan, captain plan, and chip schedule into one coherent plan."""

    def synthesize(
        self,
        transfer_plan: list[dict],
        captain_plan: list[dict],
        chip_heatmap: dict[str, dict[int, float]],
        chip_synergies: list[dict],
        available_chips: set[str],
    ) -> dict:
        """Produce a unified strategic plan with natural-language rationale.

        Returns {
            timeline: [{gw, transfers, captain, chip, confidence, rationale}],
            chip_schedule: {chip: gw},
            rationale: str,
        }
        """
        # Determine optimal chip schedule from heatmap + synergies
        chip_schedule = self._plan_chip_schedule(
            chip_heatmap, chip_synergies, available_chips,
        )

        # Build unified timeline
        all_gws = set()
        for step in transfer_plan:
            all_gws.add(step["gw"])
        for cap in captain_plan:
            all_gws.add(cap["gw"])

        timeline = []
        for gw in sorted(all_gws):
            entry = {"gw": gw}

            # Transfer info
            transfer_step = next((s for s in transfer_plan if s["gw"] == gw), None)
            if transfer_step:
                entry["transfers_in"] = transfer_step.get("transfers_in", [])
                entry["transfers_out"] = transfer_step.get("transfers_out", [])
                entry["ft_used"] = transfer_step.get("ft_used", 0)
                entry["ft_available"] = transfer_step.get("ft_available", 0)
                entry["transfer_rationale"] = transfer_step.get("rationale", "")
                entry["predicted_points"] = transfer_step.get("predicted_points", 0)

            # Captain info
            cap_step = next((c for c in captain_plan if c["gw"] == gw), None)
            if cap_step:
                entry["captain_id"] = cap_step["captain_id"]
                entry["captain_name"] = cap_step["captain_name"]
                entry["captain_points"] = cap_step["captain_points"]
                entry["vc_id"] = cap_step["vc_id"]
                entry["vc_name"] = cap_step["vc_name"]
                entry["weak_captain"] = cap_step.get("weak_gw", False)

            # Chip info
            for chip_name, chip_gw in chip_schedule.items():
                if chip_gw == gw:
                    entry["chip"] = chip_name
                    chip_val = chip_heatmap.get(chip_name, {}).get(gw, 0)
                    entry["chip_value"] = chip_val

            # Confidence (based on distance from current GW)
            entry["confidence"] = cap_step.get("confidence", 0.9) if cap_step else 0.9

            timeline.append(entry)

        # Build overall rationale
        rationale = self._build_rationale(timeline, chip_schedule, chip_synergies)

        return {
            "timeline": timeline,
            "chip_schedule": chip_schedule,
            "chip_synergies": chip_synergies[:3],  # Top 3
            "rationale": rationale,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _plan_chip_schedule(
        self,
        chip_heatmap: dict[str, dict[int, float]],
        chip_synergies: list[dict],
        available_chips: set[str],
    ) -> dict[str, int]:
        """Determine which GW to play each chip.

        Uses synergy-aware scheduling: if WC→BB synergy is the top combo,
        schedule both together.
        """
        schedule = {}

        # Check if top synergy is worth using
        if chip_synergies:
            top_syn = chip_synergies[0]
            # Use synergy if combined value > sum of individual peak values * 0.9
            syn_chips = top_syn["chips"]
            syn_gws = top_syn["gws"]
            individual_peaks = []
            for chip in syn_chips:
                if chip in chip_heatmap and chip in available_chips:
                    vals = chip_heatmap[chip]
                    if vals:
                        individual_peaks.append(max(vals.values()))
                    else:
                        individual_peaks.append(0)

            if len(individual_peaks) == len(syn_chips):
                peak_sum = sum(individual_peaks)
                if top_syn["combined_value"] > peak_sum * 0.9:
                    # Use synergy schedule
                    for chip, gw in zip(syn_chips, syn_gws):
                        if chip in available_chips:
                            schedule[chip] = gw

        # Schedule remaining chips by their peak GW
        for chip in available_chips:
            if chip in schedule:
                continue
            if chip in chip_heatmap:
                vals = chip_heatmap[chip]
                if vals:
                    # Avoid scheduling on same GW as another chip
                    used_gws = set(schedule.values())
                    sorted_gws = sorted(vals.items(), key=lambda x: x[1], reverse=True)
                    for gw, val in sorted_gws:
                        if gw not in used_gws:
                            schedule[chip] = gw
                            break

        return schedule

    def _build_rationale(
        self, timeline, chip_schedule, chip_synergies,
    ) -> str:
        """Build natural-language summary of the strategic plan."""
        parts = []

        if chip_schedule:
            chip_labels = {
                "wildcard": "Wildcard", "freehit": "Free Hit",
                "bboost": "Bench Boost", "3xc": "Triple Captain",
            }
            chip_parts = [f"{chip_labels.get(c, c)} in GW{g}" for c, g in chip_schedule.items()]
            parts.append(f"Chip plan: {', '.join(chip_parts)}.")

        if chip_synergies:
            top = chip_synergies[0]
            parts.append(f"Key synergy: {top['description']} (+{top['synergy_bonus']:.1f} pts bonus).")

        # Summarize transfer strategy
        bank_gws = [t for t in timeline if t.get("ft_used", 0) == 0]
        use_gws = [t for t in timeline if t.get("ft_used", 0) > 0]
        if bank_gws and use_gws:
            parts.append(
                f"Transfer strategy: bank in GW{','.join(str(t['gw']) for t in bank_gws)}, "
                f"use in GW{','.join(str(t['gw']) for t in use_gws)}."
            )

        # Flag weak captain GWs
        weak = [t for t in timeline if t.get("weak_captain")]
        if weak:
            parts.append(
                f"Weak captain GW(s): {','.join(str(t['gw']) for t in weak)} — "
                "consider transfer to upgrade premium captain option."
            )

        return " ".join(parts) if parts else "No significant strategic adjustments needed."


# ---------------------------------------------------------------------------
# Reactive Re-planning — Phase 4
# ---------------------------------------------------------------------------

def detect_plan_invalidation(
    current_plan: dict,
    new_predictions: dict[int, pd.DataFrame],
    fixture_calendar: list[dict],
    squad_changes: dict | None = None,
) -> list[dict]:
    """Detect changes that invalidate the current strategic plan.

    Returns list of {severity, type, description, affected_gws}.
    Severity: 'critical' (auto-replan), 'moderate' (warning), 'minor' (logged).
    """
    triggers = []

    if not current_plan or "timeline" not in current_plan:
        return triggers

    timeline = current_plan["timeline"]

    # Check for fixture changes (DGW/BGW)
    fx_by_gw = {}
    for f in fixture_calendar:
        gw = f["gameweek"]
        if gw not in fx_by_gw:
            fx_by_gw[gw] = []
        fx_by_gw[gw].append(f)

    # Check chip GW changes
    chip_schedule = current_plan.get("chip_schedule", {})
    for chip_name, chip_gw in chip_schedule.items():
        if chip_gw in fx_by_gw:
            gw_fixtures = fx_by_gw[chip_gw]
            n_dgw = sum(1 for f in gw_fixtures if f.get("is_dgw"))
            n_bgw = sum(1 for f in gw_fixtures if f.get("is_bgw"))

            # If we planned BB for a GW that's no longer a DGW
            if chip_name == "bboost" and n_dgw == 0:
                triggers.append({
                    "severity": "critical",
                    "type": "fixture_change",
                    "description": f"BB planned for GW{chip_gw} but no DGW fixtures found",
                    "affected_gws": [chip_gw],
                })

            # If FH was planned for a GW that's no longer a BGW
            if chip_name == "freehit" and n_bgw == 0:
                triggers.append({
                    "severity": "moderate",
                    "type": "fixture_change",
                    "description": f"FH planned for GW{chip_gw} but no BGW fixtures found",
                    "affected_gws": [chip_gw],
                })

    # Check for significant prediction shifts
    for entry in timeline:
        gw = entry["gw"]
        if gw not in new_predictions:
            continue

        new_preds = new_predictions[gw]
        old_captain_id = entry.get("captain_id")
        if old_captain_id:
            new_cap_pred = new_preds[new_preds["player_id"] == old_captain_id]
            if not new_cap_pred.empty:
                new_pts = new_cap_pred.iloc[0]["predicted_points"]
                old_pts = entry.get("captain_points", 0)
                if old_pts > 0 and new_pts < old_pts * 0.5:
                    triggers.append({
                        "severity": "critical",
                        "type": "prediction_shift",
                        "description": f"GW{gw} captain prediction dropped >50% ({old_pts:.1f}→{new_pts:.1f})",
                        "affected_gws": [gw],
                    })

        # Check planned transfers: are players being transferred in still worthwhile?
        transfers_in = entry.get("transfers_in", [])
        for t in transfers_in:
            pid = t.get("player_id")
            if pid:
                new_pred = new_preds[new_preds["player_id"] == pid]
                if not new_pred.empty:
                    new_pts = new_pred.iloc[0]["predicted_points"]
                    old_pts = t.get("predicted_points", 0)
                    if old_pts > 0 and new_pts < old_pts * 0.5:
                        triggers.append({
                            "severity": "moderate",
                            "type": "prediction_shift",
                            "description": f"GW{gw} transfer target {t.get('web_name', '?')} prediction dropped >50%",
                            "affected_gws": [gw],
                        })

    # Check for squad changes (injuries, suspensions)
    if squad_changes:
        for player_id, change in squad_changes.items():
            status = change.get("status", "a")
            chance = change.get("chance_of_playing", 100)
            name = change.get("web_name", "Unknown")

            if status == "i" or (chance is not None and chance < 25):
                # Check if this player is in any planned squad
                for entry in timeline:
                    squad_ids = entry.get("squad_ids", [])
                    if int(player_id) in [int(x) for x in squad_ids]:
                        triggers.append({
                            "severity": "critical",
                            "type": "injury",
                            "description": f"{name} injured/unavailable — in planned GW{entry['gw']} squad",
                            "affected_gws": [entry["gw"]],
                        })
                        break
            elif chance is not None and chance < 50:
                for entry in timeline:
                    squad_ids = entry.get("squad_ids", [])
                    if int(player_id) in [int(x) for x in squad_ids]:
                        triggers.append({
                            "severity": "moderate",
                            "type": "doubt",
                            "description": f"{name} doubtful ({chance}% chance) — in GW{entry['gw']} squad",
                            "affected_gws": [entry["gw"]],
                        })
                        break

    # Sort by severity
    severity_order = {"critical": 0, "moderate": 1, "minor": 2}
    triggers.sort(key=lambda t: severity_order.get(t["severity"], 9))

    return triggers


def apply_availability_adjustments(
    future_predictions: dict[int, pd.DataFrame],
    bootstrap_elements: list[dict],
) -> dict[int, pd.DataFrame]:
    """Zero out predictions for injured/unavailable players.

    - chance_of_playing < 50%: zero for GW+1
    - status == 'i' (injured): zero for all GWs
    """
    # Build availability map
    injured_ids = set()
    doubtful_ids = set()
    for el in bootstrap_elements:
        status = el.get("status", "a")
        chance = el.get("chance_of_playing_next_round")
        pid = el["id"]

        if status == "i" or status == "s":  # injured or suspended
            injured_ids.add(pid)
        elif chance is not None and chance < 50:
            doubtful_ids.add(pid)

    adjusted = {}
    gws = sorted(future_predictions.keys())
    for i, gw in enumerate(gws):
        gw_df = future_predictions[gw].copy()

        # Zero injured players for all GWs
        gw_df.loc[gw_df["player_id"].isin(injured_ids), "predicted_points"] = 0.0

        # Zero doubtful players for GW+1 only
        if i == 0:
            gw_df.loc[gw_df["player_id"].isin(doubtful_ids), "predicted_points"] = 0.0

        adjusted[gw] = gw_df

    return adjusted
