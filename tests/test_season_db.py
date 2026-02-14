"""Tests for src/season_db.py — SQLite CRUD operations."""

import json

import pytest


class TestSeasonCRUD:
    def test_create_and_get_season(self, season_db):
        sid = season_db.create_season(
            manager_id=123,
            manager_name="Test Manager",
            team_name="Test FC",
            season_name="2024-2025",
            start_gw=1,
        )
        assert sid > 0

        season = season_db.get_season(123, "2024-2025")
        assert season is not None
        assert season["manager_id"] == 123
        assert season["manager_name"] == "Test Manager"
        assert season["team_name"] == "Test FC"
        assert season["season_name"] == "2024-2025"

    def test_create_season_upsert_preserves_id(self, season_db):
        """Creating the same season twice should update, not duplicate."""
        sid1 = season_db.create_season(123, "OldName", "OldTeam", "2024-2025")
        sid2 = season_db.create_season(123, "NewName", "NewTeam", "2024-2025")
        # Same season_id returned (upsert)
        assert sid1 == sid2
        season = season_db.get_season(123, "2024-2025")
        assert season["manager_name"] == "NewName"


class TestGWSnapshot:
    def test_save_and_get_snapshot(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        season_db.save_gw_snapshot(
            sid, gameweek=5,
            squad_json='[1,2,3]', bank=50, team_value=1000,
            points=65, total_points=350, overall_rank=100000,
            captain_id=1, captain_name="Salah",
        )
        snap = season_db.get_snapshot(sid, 5)
        assert snap is not None
        assert snap["gameweek"] == 5
        assert snap["bank"] == 50
        assert snap["captain_name"] == "Salah"

    def test_get_all_snapshots_ordered(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        for gw in [3, 1, 2]:
            season_db.save_gw_snapshot(sid, gameweek=gw, points=gw * 10)
        snaps = season_db.get_snapshots(sid)
        assert len(snaps) == 3
        assert [s["gameweek"] for s in snaps] == [1, 2, 3]


class TestRecommendation:
    def test_save_and_get_recommendation(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        season_db.save_recommendation(
            sid, gameweek=10,
            transfers_json='[{"in": 5, "out": 3}]',
            captain_id=5, captain_name="Haaland",
            chip_suggestion="bboost",
            predicted_points=75.5,
        )
        rec = season_db.get_recommendation(sid, 10)
        assert rec is not None
        assert rec["captain_name"] == "Haaland"
        assert rec["chip_suggestion"] == "bboost"
        assert rec["predicted_points"] == 75.5


class TestOutcome:
    def test_save_and_get_outcome(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        season_db.save_outcome(
            sid, gameweek=10,
            followed_transfers=1, followed_captain=1, followed_chip=0,
            recommended_points=75.0, actual_points=68.0,
            point_delta=-7.0,
        )
        outcomes = season_db.get_outcomes(sid)
        assert len(outcomes) == 1
        assert outcomes[0]["actual_points"] == 68.0
        assert outcomes[0]["point_delta"] == -7.0


class TestFixtureCalendar:
    def test_save_and_get_fixtures(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        fixtures = [
            {"team_id": 1, "team_short": "ARS", "gameweek": 25,
             "fixture_count": 2, "fdr_avg": 2.5, "is_dgw": 1, "is_bgw": 0},
            {"team_id": 2, "team_short": "CHE", "gameweek": 25,
             "fixture_count": 1, "fdr_avg": 3.0, "is_dgw": 0, "is_bgw": 0},
        ]
        season_db.save_fixture_calendar(sid, fixtures)
        result = season_db.get_fixture_calendar(sid, from_gw=25, to_gw=25)
        assert len(result) == 2
        ars = next(f for f in result if f["team_short"] == "ARS")
        assert ars["is_dgw"] == 1
        assert ars["fixture_count"] == 2


class TestStrategicPlan:
    def test_save_and_get_plan(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        plan = {"timeline": [{"gw": 25, "chip": "bboost"}]}
        heatmap = {"bboost": {"25": 12.5}}
        season_db.save_strategic_plan(sid, 25, json.dumps(plan), json.dumps(heatmap))

        result = season_db.get_strategic_plan(sid, 25)
        assert result is not None
        loaded_plan = json.loads(result["plan_json"])
        assert loaded_plan["timeline"][0]["chip"] == "bboost"

    def test_get_latest_plan(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        season_db.save_strategic_plan(sid, 20, '{"v": 1}', '{}')
        season_db.save_strategic_plan(sid, 25, '{"v": 2}', '{}')

        result = season_db.get_strategic_plan(sid)  # No specific GW = latest
        assert result is not None
        assert json.loads(result["plan_json"])["v"] == 2


class TestPlanChangelog:
    def test_save_and_get_changelog(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        season_db.save_plan_change(
            sid, gameweek=25,
            change_type="chip_reschedule",
            description="Moved BB from GW26 to GW28",
            old_value="GW26", new_value="GW28",
            reason="GW26 DGW cancelled",
        )
        logs = season_db.get_plan_changelog(sid)
        assert len(logs) == 1
        assert logs[0]["change_type"] == "chip_reschedule"
        assert logs[0]["reason"] == "GW26 DGW cancelled"

    def test_changelog_multiple_entries(self, season_db):
        sid = season_db.create_season(123, season_name="2024-2025")
        season_db.save_plan_change(sid, 25, "captain_change", "Changed captain")
        season_db.save_plan_change(sid, 26, "transfer_change", "New target")
        logs = season_db.get_plan_changelog(sid)
        assert len(logs) == 2
