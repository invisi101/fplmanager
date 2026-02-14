"""Tests for src/strategy.py — availability adjustments, plan invalidation, captain planner."""

import pandas as pd
import pytest

from src.strategy import (
    CaptainPlanner,
    apply_availability_adjustments,
    detect_plan_invalidation,
)


# ---------------------------------------------------------------------------
# apply_availability_adjustments tests
# ---------------------------------------------------------------------------

class TestApplyAvailabilityAdjustments:
    def _make_predictions(self, player_ids, gws=(25, 26, 27)):
        """Helper: create simple future predictions dict."""
        preds = {}
        for gw in gws:
            rows = []
            for pid in player_ids:
                rows.append({
                    "player_id": pid,
                    "predicted_points": 5.0,
                    "captain_score": 6.0,
                    "predicted_next_gw_points_q80": 7.0,
                })
            preds[gw] = pd.DataFrame(rows)
        return preds

    def test_injured_zeroed_all_gws(self, bootstrap_elements):
        preds = self._make_predictions([1, 2, 3, 4, 5, 6])
        result = apply_availability_adjustments(preds, bootstrap_elements)

        # Player 2 (injured) should be 0 in ALL GWs
        for gw, df in result.items():
            p2 = df[df["player_id"] == 2].iloc[0]
            assert p2["predicted_points"] == 0.0
            assert p2["captain_score"] == 0.0
            assert p2["predicted_next_gw_points_q80"] == 0.0

    def test_suspended_zeroed_all_gws(self, bootstrap_elements):
        preds = self._make_predictions([1, 2, 3, 4, 5, 6])
        result = apply_availability_adjustments(preds, bootstrap_elements)

        # Player 3 (suspended) should be 0 in ALL GWs
        for gw, df in result.items():
            p3 = df[df["player_id"] == 3].iloc[0]
            assert p3["predicted_points"] == 0.0

    def test_doubtful_zeroed_gw_plus_1_only(self, bootstrap_elements):
        preds = self._make_predictions([1, 2, 3, 4, 5, 6])
        gws = sorted(preds.keys())
        result = apply_availability_adjustments(preds, bootstrap_elements)

        # Player 4 (25% chance) should be 0 in first GW only
        p4_gw1 = result[gws[0]][result[gws[0]]["player_id"] == 4].iloc[0]
        assert p4_gw1["predicted_points"] == 0.0

        # But not in later GWs
        p4_gw2 = result[gws[1]][result[gws[1]]["player_id"] == 4].iloc[0]
        assert p4_gw2["predicted_points"] == 5.0

    def test_healthy_player_unchanged(self, bootstrap_elements):
        preds = self._make_predictions([1, 2, 3, 4, 5, 6])
        result = apply_availability_adjustments(preds, bootstrap_elements)

        # Player 1 (healthy) should be unchanged everywhere
        for gw, df in result.items():
            p1 = df[df["player_id"] == 1].iloc[0]
            assert p1["predicted_points"] == 5.0

    def test_chance_exactly_50_not_zeroed(self, bootstrap_elements):
        """Player 5 has chance=50. The threshold is < 50, so 50 should NOT be zeroed."""
        preds = self._make_predictions([1, 2, 3, 4, 5, 6])
        result = apply_availability_adjustments(preds, bootstrap_elements)

        # Player 5 (chance=50) should be unchanged
        gws = sorted(result.keys())
        p5 = result[gws[0]][result[gws[0]]["player_id"] == 5].iloc[0]
        assert p5["predicted_points"] == 5.0

    def test_captain_score_zeroed_for_injured(self, bootstrap_elements):
        preds = self._make_predictions([1, 2])
        result = apply_availability_adjustments(preds, bootstrap_elements)

        for gw, df in result.items():
            p2 = df[df["player_id"] == 2].iloc[0]
            assert p2["captain_score"] == 0.0


# ---------------------------------------------------------------------------
# detect_plan_invalidation tests
# ---------------------------------------------------------------------------

class TestDetectPlanInvalidation:
    def _make_plan(self, timeline_entries):
        return {"timeline": timeline_entries, "chip_schedule": {}}

    def test_no_triggers_for_healthy_plan(self):
        plan = self._make_plan([
            {"gw": 25, "captain_id": 1, "captain_points": 8.0,
             "transfers_in": [], "squad_ids": [1, 2, 3]},
        ])
        preds = {25: pd.DataFrame([
            {"player_id": 1, "predicted_points": 8.0},
            {"player_id": 2, "predicted_points": 5.0},
            {"player_id": 3, "predicted_points": 4.0},
        ])}
        triggers = detect_plan_invalidation(plan, preds, [])
        assert len(triggers) == 0

    def test_captain_prediction_drop_triggers_critical(self):
        plan = self._make_plan([
            {"gw": 25, "captain_id": 1, "captain_points": 10.0,
             "transfers_in": [], "squad_ids": [1, 2]},
        ])
        preds = {25: pd.DataFrame([
            {"player_id": 1, "predicted_points": 3.0},  # Dropped >50%
            {"player_id": 2, "predicted_points": 5.0},
        ])}
        triggers = detect_plan_invalidation(plan, preds, [])
        assert any(t["severity"] == "critical" and t["type"] == "prediction_shift"
                    for t in triggers)

    def test_injury_in_planned_squad_triggers_critical(self):
        plan = self._make_plan([
            {"gw": 25, "captain_id": 1, "captain_points": 8.0,
             "transfers_in": [], "squad_ids": [1, 10]},
        ])
        squad_changes = {
            10: {"status": "i", "chance_of_playing": 0, "web_name": "InjuredGuy"},
        }
        triggers = detect_plan_invalidation(plan, {}, [], squad_changes)
        assert any(t["severity"] == "critical" and t["type"] == "injury"
                    for t in triggers)

    def test_bb_without_dgw_triggers_moderate(self):
        plan = {
            "timeline": [{"gw": 25}],
            "chip_schedule": {"bboost": 25},
        }
        # Fixtures without any DGW in GW25
        fixtures = [
            {"gameweek": 25, "team_id": 1, "is_dgw": 0, "is_bgw": 0},
            {"gameweek": 25, "team_id": 2, "is_dgw": 0, "is_bgw": 0},
        ]
        triggers = detect_plan_invalidation(plan, {}, fixtures)
        assert any(t["severity"] == "moderate" and "BB" in t["description"]
                    for t in triggers)

    def test_empty_plan_returns_no_triggers(self):
        triggers = detect_plan_invalidation({}, {}, [])
        assert triggers == []
        triggers = detect_plan_invalidation(None, {}, [])
        assert triggers == []


# ---------------------------------------------------------------------------
# CaptainPlanner tests
# ---------------------------------------------------------------------------

class TestCaptainPlanner:
    def _make_squad_preds(self, players, gws=(25, 26)):
        """Helper to build future_predictions from a list of player dicts."""
        preds = {}
        for gw in gws:
            preds[gw] = pd.DataFrame(players)
        return preds

    def test_picks_highest_scorer(self):
        planner = CaptainPlanner()
        players = [
            {"player_id": 1, "web_name": "Low", "predicted_points": 3.0, "captain_score": 3.5},
            {"player_id": 2, "web_name": "High", "predicted_points": 8.0, "captain_score": 9.0},
            {"player_id": 3, "web_name": "Mid", "predicted_points": 5.0, "captain_score": 5.5},
        ]
        preds = self._make_squad_preds(players, gws=(25,))
        squad_ids = {1, 2, 3}
        result = planner.plan_captaincy(squad_ids, preds)
        assert len(result) == 1
        assert result[0]["captain_id"] == 2

    def test_flags_weak_gw(self):
        planner = CaptainPlanner()
        players = [
            {"player_id": 1, "web_name": "Weak", "predicted_points": 2.0, "captain_score": 2.5},
            {"player_id": 2, "web_name": "AlsoWeak", "predicted_points": 1.5, "captain_score": 1.8},
        ]
        preds = self._make_squad_preds(players, gws=(25,))
        squad_ids = {1, 2}
        result = planner.plan_captaincy(squad_ids, preds)
        assert result[0]["weak_gw"] == True  # noqa: E712 (numpy bool)

    def test_uses_captain_score_when_available(self):
        planner = CaptainPlanner()
        # Player 1 has lower predicted_points but higher captain_score
        players = [
            {"player_id": 1, "web_name": "Explosive", "predicted_points": 5.0, "captain_score": 12.0},
            {"player_id": 2, "web_name": "Consistent", "predicted_points": 7.0, "captain_score": 7.5},
        ]
        preds = self._make_squad_preds(players, gws=(25,))
        squad_ids = {1, 2}
        result = planner.plan_captaincy(squad_ids, preds)
        assert result[0]["captain_id"] == 1  # Higher captain_score wins
