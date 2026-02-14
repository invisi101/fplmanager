"""Tests for src/feature_engineering.py — get_feature_columns leakage checks."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import get_feature_columns


def _make_sample_df():
    """Build a DataFrame with typical columns from feature engineering."""
    return pd.DataFrame({
        # Metadata (should be excluded)
        "player_id": [1, 2],
        "gameweek": [10, 10],
        "season": ["2024-2025", "2024-2025"],
        "position": ["MID", "FWD"],
        "web_name": ["Salah", "Haaland"],
        "team_code": [14, 13],
        "opponent_code": [1, 2],
        # Targets (should be excluded)
        "next_gw_points": [8.0, 10.0],
        "next_3gw_points": [20.0, 25.0],
        "event_points": [7.0, 9.0],
        # Decomposed targets (should be excluded)
        "next_gw_goals": [1.0, 2.0],
        "next_gw_assists": [0.0, 1.0],
        "next_gw_cs": [0.0, 0.0],
        "next_gw_bonus": [2.0, 3.0],
        "next_gw_goals_conceded": [1.0, 0.0],
        "next_gw_saves": [0.0, 0.0],
        "next_gw_minutes": [90.0, 90.0],
        # Valid features (should be included)
        "player_xg_last3": [0.5, 0.8],
        "player_xa_last5": [0.3, 0.2],
        "fdr": [3.0, 2.0],
        "is_home": [1, 0],
        "opponent_goals_conceded_last5": [6.0, 4.0],
        "player_ewm_xg": [0.45, 0.75],
        "ict_index": [50.0, 65.0],
        "ownership_pct": [25.5, 35.2],
        # Extra excluded columns
        "penalties_order": [1, 0],
        "corners_order": [2, 0],
        "total_points": [150, 180],
        "transfers_out_event": [5000, 3000],
        "cumulative_minutes": [900, 850],
        "ep_next": [5.5, 6.2],
    })


class TestGetFeatureColumns:
    def test_excludes_target_columns(self):
        df = _make_sample_df()
        features = get_feature_columns(df)
        targets = {"next_gw_points", "next_3gw_points", "event_points"}
        assert targets.isdisjoint(set(features))

    def test_excludes_decomposed_targets(self):
        df = _make_sample_df()
        features = get_feature_columns(df)
        decomposed = {
            "next_gw_goals", "next_gw_assists", "next_gw_cs",
            "next_gw_bonus", "next_gw_goals_conceded", "next_gw_saves",
            "next_gw_minutes",
        }
        assert decomposed.isdisjoint(set(features))

    def test_excludes_metadata(self):
        df = _make_sample_df()
        features = get_feature_columns(df)
        metadata = {"player_id", "gameweek", "season", "position", "web_name",
                     "team_code", "opponent_code"}
        assert metadata.isdisjoint(set(features))

    def test_includes_valid_numeric_features(self):
        df = _make_sample_df()
        features = get_feature_columns(df)
        expected_present = {"player_xg_last3", "player_xa_last5", "fdr",
                            "opponent_goals_conceded_last5", "player_ewm_xg",
                            "ict_index", "ownership_pct"}
        for col in expected_present:
            assert col in features, f"{col} should be in features"

    def test_output_is_sorted(self):
        df = _make_sample_df()
        features = get_feature_columns(df)
        assert features == sorted(features)
