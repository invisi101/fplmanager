"""Shared test fixtures for FPL Manager test suite."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.season_db import SeasonDB


# ---------------------------------------------------------------------------
# Player pool: 40 players across 8 teams, all 4 positions
# Costs calibrated so a valid 15 (2/5/5/3) fits within 1000 (100.0m)
# ---------------------------------------------------------------------------

def _make_players():
    """Build a 50-player pool across 10 teams, all 4 positions.

    Each team gets: 1 GKP, 2 DEF, 1 MID, 1 FWD = 5 per team, 50 total.
    Totals: 10 GKP, 20 DEF, 10 MID, 10 FWD — enough for any valid squad.
    """
    players = []
    pid = 1
    teams = list(range(1, 11))  # team_code 1-10

    positions_per_team = ["GKP", "DEF", "DEF", "MID", "FWD"]

    for tc in teams:
        for pos in positions_per_team:
            cost_map = {"GKP": 45, "DEF": 50, "MID": 65, "FWD": 75}
            base_cost = cost_map[pos]
            pts_map = {"GKP": 3.5, "DEF": 4.0, "MID": 5.5, "FWD": 6.0}
            base_pts = pts_map[pos]

            cost = base_cost + (pid % 5) * 3
            pts = round(base_pts + (pid % 7) * 0.5, 1)
            cap_score = round(pts * 1.2 + (pid % 3) * 0.3, 1)

            players.append({
                "player_id": pid,
                "web_name": f"Player_{pid}",
                "position": pos,
                "cost": cost,
                "team_code": tc,
                "predicted_next_gw_points": pts,
                "predicted_points": pts,
                "captain_score": cap_score,
            })
            pid += 1

    return players


@pytest.fixture
def player_pool_df():
    """40 players across 8 teams, all 4 positions."""
    return pd.DataFrame(_make_players())


@pytest.fixture
def current_squad_ids(player_pool_df):
    """Valid 15-player squad: 2 GKP, 5 DEF, 5 MID, 3 FWD, max 3/team.

    Carefully picks from different teams to respect the max-3-per-team rule.
    """
    df = player_pool_df
    squad = []
    team_counts = {}

    def pick(position, count):
        candidates = df[df["position"] == position]
        picked = 0
        for _, row in candidates.iterrows():
            if picked >= count:
                break
            tc = row["team_code"]
            if team_counts.get(tc, 0) < 3:
                squad.append(row["player_id"])
                team_counts[tc] = team_counts.get(tc, 0) + 1
                picked += 1

    pick("GKP", 2)
    pick("DEF", 5)
    pick("MID", 5)
    pick("FWD", 3)

    assert len(squad) == 15
    return set(squad)


@pytest.fixture
def future_predictions(player_pool_df):
    """Dict of {gw: DataFrame} for 3 GWs with predicted_points column."""
    preds = {}
    base_df = player_pool_df.copy()
    for gw in [25, 26, 27]:
        gw_df = base_df.copy()
        # Vary points slightly per GW
        gw_df["predicted_points"] = gw_df["predicted_points"] + (gw - 25) * 0.2
        preds[gw] = gw_df
    return preds


@pytest.fixture
def fixture_calendar():
    """Fixture calendar with DGW in GW26, BGW in GW27 for team 1."""
    fixtures = []
    for tc in range(1, 9):
        for gw in [25, 26, 27]:
            f = {
                "team_id": tc,
                "team_code": tc,
                "team_short": f"T{tc}",
                "gameweek": gw,
                "fixture_count": 1,
                "fdr_avg": 3.0,
                "is_dgw": 0,
                "is_bgw": 0,
            }
            if gw == 26 and tc == 1:
                f["fixture_count"] = 2
                f["is_dgw"] = 1
            if gw == 27 and tc == 1:
                f["fixture_count"] = 0
                f["is_bgw"] = 1
            fixtures.append(f)
    return fixtures


@pytest.fixture
def simple_history():
    """FPL history dict: 5 GWs, 1 transfer per GW, no chips."""
    return {
        "current": [
            {"event": i, "event_transfers": 1, "event_transfers_cost": 0}
            for i in range(1, 6)
        ],
        "chips": [],
    }


@pytest.fixture
def wildcard_history():
    """FPL history with WC used in GW3."""
    return {
        "current": [
            {"event": 1, "event_transfers": 0, "event_transfers_cost": 0},
            {"event": 2, "event_transfers": 0, "event_transfers_cost": 0},
            {"event": 3, "event_transfers": 5, "event_transfers_cost": 0},  # WC
            {"event": 4, "event_transfers": 1, "event_transfers_cost": 0},
            {"event": 5, "event_transfers": 1, "event_transfers_cost": 0},
        ],
        "chips": [{"event": 3, "name": "wildcard"}],
    }


@pytest.fixture
def bootstrap_elements():
    """Elements with varied statuses for availability testing."""
    return [
        {"id": 1, "web_name": "Healthy_1", "status": "a", "chance_of_playing_next_round": 100},
        {"id": 2, "web_name": "Injured_2", "status": "i", "chance_of_playing_next_round": 0},
        {"id": 3, "web_name": "Suspended_3", "status": "s", "chance_of_playing_next_round": 0},
        {"id": 4, "web_name": "Doubtful_4", "status": "d", "chance_of_playing_next_round": 25},
        {"id": 5, "web_name": "Borderline_5", "status": "d", "chance_of_playing_next_round": 50},
        {"id": 6, "web_name": "Likely_6", "status": "d", "chance_of_playing_next_round": 75},
    ]


@pytest.fixture
def season_db(tmp_path):
    """SeasonDB backed by tmp_path (isolated per test)."""
    db_path = tmp_path / "test_season.db"
    return SeasonDB(db_path=db_path)
