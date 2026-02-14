"""Tests for src/solver.py — MILP solver for squad selection and transfers."""

import math

import numpy as np
import pandas as pd
import pytest

from src.solver import scrub_nan, solve_milp_team, solve_transfer_milp


# ---------------------------------------------------------------------------
# scrub_nan tests
# ---------------------------------------------------------------------------

class TestScrubNan:
    def test_replaces_nan_with_none(self):
        records = [{"a": float("nan"), "b": 1.0}]
        result = scrub_nan(records)
        assert result[0]["a"] is None
        assert result[0]["b"] == 1.0

    def test_replaces_inf_with_none(self):
        records = [{"a": float("inf"), "b": float("-inf")}]
        result = scrub_nan(records)
        assert result[0]["a"] is None
        assert result[0]["b"] is None

    def test_does_not_mutate_input(self):
        original = [{"a": float("nan"), "b": 2.0}]
        scrub_nan(original)
        assert math.isnan(original[0]["a"])
        assert original[0]["b"] == 2.0

    def test_preserves_non_float_values(self):
        records = [{"name": "Salah", "pos": "MID", "count": 3, "score": 7.5}]
        result = scrub_nan(records)
        assert result[0] == {"name": "Salah", "pos": "MID", "count": 3, "score": 7.5}

    def test_empty_list(self):
        assert scrub_nan([]) == []


# ---------------------------------------------------------------------------
# solve_milp_team tests
# ---------------------------------------------------------------------------

class TestSolveMilpTeam:
    def test_returns_15_players(self, player_pool_df):
        result = solve_milp_team(player_pool_df, "predicted_points", budget=5000)
        assert result is not None
        assert len(result["starters"]) == 11
        assert len(result["bench"]) == 4

    def test_squad_positions(self, player_pool_df):
        result = solve_milp_team(player_pool_df, "predicted_points", budget=5000)
        all_players = result["starters"] + result["bench"]
        pos_counts = {}
        for p in all_players:
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1
        assert pos_counts.get("GKP", 0) == 2
        assert pos_counts.get("DEF", 0) == 5
        assert pos_counts.get("MID", 0) == 5
        assert pos_counts.get("FWD", 0) == 3

    def test_starter_formation_valid(self, player_pool_df):
        result = solve_milp_team(player_pool_df, "predicted_points", budget=5000)
        pos_counts = {}
        for p in result["starters"]:
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1
        assert pos_counts.get("GKP", 0) == 1
        assert 3 <= pos_counts.get("DEF", 0) <= 5
        assert 2 <= pos_counts.get("MID", 0) <= 5
        assert 1 <= pos_counts.get("FWD", 0) <= 3

    def test_max_three_per_team(self, player_pool_df):
        result = solve_milp_team(player_pool_df, "predicted_points", budget=5000)
        all_players = result["starters"] + result["bench"]
        team_counts = {}
        for p in all_players:
            tc = p.get("team_code")
            if tc is not None:
                team_counts[tc] = team_counts.get(tc, 0) + 1
        for tc, count in team_counts.items():
            assert count <= 3, f"Team {tc} has {count} players (max 3)"

    def test_budget_constraint(self, player_pool_df):
        # Minimum valid squad costs ~986; use 1000 as a tight constraint
        result = solve_milp_team(player_pool_df, "predicted_points", budget=1000)
        assert result is not None
        assert result["total_cost"] <= 1000

    def test_captain_is_starter(self, player_pool_df):
        result = solve_milp_team(
            player_pool_df, "predicted_points",
            budget=5000, captain_col="captain_score",
        )
        assert result is not None
        assert result["captain_id"] is not None
        starter_ids = {p["player_id"] for p in result["starters"]}
        assert result["captain_id"] in starter_ids

    def test_captain_bonus_in_starting_points(self, player_pool_df):
        result = solve_milp_team(
            player_pool_df, "predicted_points",
            budget=5000, captain_col="captain_score",
        )
        assert result is not None
        # starting_points = sum of starters + captain's predicted_points (doubled)
        base_sum = sum(p["predicted_points"] for p in result["starters"])
        captain_pts = next(
            p["predicted_points"] for p in result["starters"]
            if p["player_id"] == result["captain_id"]
        )
        assert result["starting_points"] == pytest.approx(base_sum + captain_pts, abs=0.1)

    def test_nan_captain_score_falls_back(self, player_pool_df):
        df = player_pool_df.copy()
        df["captain_score"] = np.nan  # All NaN
        result = solve_milp_team(
            df, "predicted_points",
            budget=5000, captain_col="captain_score",
        )
        # Should still pick a captain (falling back to predicted_points)
        assert result is not None
        assert result["captain_id"] is not None

    def test_returns_none_for_impossible_budget(self, player_pool_df):
        result = solve_milp_team(player_pool_df, "predicted_points", budget=1)
        assert result is None

    def test_returns_none_for_missing_columns(self, player_pool_df):
        df = player_pool_df.drop(columns=["cost"])
        result = solve_milp_team(df, "predicted_points")
        assert result is None

    def test_bench_sorted_gk_first(self, player_pool_df):
        result = solve_milp_team(player_pool_df, "predicted_points", budget=5000)
        bench = result["bench"]
        assert len(bench) == 4
        assert bench[0]["position"] == "GKP"
        # Remaining 3 are outfield sorted by descending predicted_points
        outfield_pts = [p["predicted_points"] for p in bench[1:]]
        assert outfield_pts == sorted(outfield_pts, reverse=True)


# ---------------------------------------------------------------------------
# solve_transfer_milp tests
# ---------------------------------------------------------------------------

class TestSolveTransferMilp:
    def test_respects_max_transfers(self, player_pool_df, current_squad_ids):
        result = solve_transfer_milp(
            player_pool_df, current_squad_ids, "predicted_points",
            budget=5000, max_transfers=1,
        )
        assert result is not None
        assert len(result["transfers_in_ids"]) <= 1
        assert len(result["transfers_out_ids"]) <= 1

    def test_zero_transfers_keeps_squad(self, player_pool_df, current_squad_ids):
        result = solve_transfer_milp(
            player_pool_df, current_squad_ids, "predicted_points",
            budget=5000, max_transfers=0,
        )
        assert result is not None
        new_ids = {p["player_id"] for p in result["players"]}
        assert new_ids == current_squad_ids

    def test_balanced_transfers(self, player_pool_df, current_squad_ids):
        result = solve_transfer_milp(
            player_pool_df, current_squad_ids, "predicted_points",
            budget=5000, max_transfers=2,
        )
        assert result is not None
        assert len(result["transfers_in_ids"]) == len(result["transfers_out_ids"])

    def test_valid_positions_after_transfers(self, player_pool_df, current_squad_ids):
        result = solve_transfer_milp(
            player_pool_df, current_squad_ids, "predicted_points",
            budget=5000, max_transfers=2,
        )
        assert result is not None
        all_players = result["starters"] + result["bench"]
        pos_counts = {}
        for p in all_players:
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1
        assert pos_counts.get("GKP", 0) == 2
        assert pos_counts.get("DEF", 0) == 5
        assert pos_counts.get("MID", 0) == 5
        assert pos_counts.get("FWD", 0) == 3

    def test_captain_with_transfers(self, player_pool_df, current_squad_ids):
        result = solve_transfer_milp(
            player_pool_df, current_squad_ids, "predicted_points",
            budget=5000, max_transfers=2, captain_col="captain_score",
        )
        assert result is not None
        assert result["captain_id"] is not None
        starter_ids = {p["player_id"] for p in result["starters"]}
        assert result["captain_id"] in starter_ids
