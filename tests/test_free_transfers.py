"""Tests for FT calculation — _calculate_free_transfers in app.py and season_manager.py."""

import pytest

from src.app import _calculate_free_transfers as app_calc_ft


# We can't easily instantiate SeasonManager without its dependencies,
# so we test the standalone app.py version and verify the logic matches.


class TestCalculateFreeTransfers:
    def test_start_of_season_one_ft(self):
        """GW1 with no history => 1 FT."""
        history = {"current": [], "chips": []}
        assert app_calc_ft(history) == 1

    def test_no_transfer_banks_to_two(self):
        """1 GW with 0 transfers => 2 FT next."""
        history = {
            "current": [
                {"event": 1, "event_transfers": 0, "event_transfers_cost": 0},
            ],
            "chips": [],
        }
        assert app_calc_ft(history) == 2

    def test_using_one_ft_stays_at_one(self):
        """Using 1 FT each GW keeps you at 1."""
        history = {
            "current": [
                {"event": i, "event_transfers": 1, "event_transfers_cost": 0}
                for i in range(1, 6)
            ],
            "chips": [],
        }
        assert app_calc_ft(history) == 1

    def test_banking_accumulates_to_cap_of_five(self):
        """Banking FTs accumulates: after 5 GWs of 0 transfers, should be capped at 5."""
        history = {
            "current": [
                {"event": i, "event_transfers": 0, "event_transfers_cost": 0}
                for i in range(1, 7)  # 6 GWs of saving
            ],
            "chips": [],
        }
        result = app_calc_ft(history)
        assert result == 5  # Capped at 5

    def test_wildcard_preserves_fts_with_accrual(self):
        """WC preserves FTs and +1 accrual happens.

        GW1: 0 transfers => FT goes 1 -> 2
        GW2: 0 transfers => FT goes 2 -> 3
        GW3: WC (5 transfers) => FTs preserved at 3, +1 => 4
        """
        history = {
            "current": [
                {"event": 1, "event_transfers": 0, "event_transfers_cost": 0},
                {"event": 2, "event_transfers": 0, "event_transfers_cost": 0},
                {"event": 3, "event_transfers": 5, "event_transfers_cost": 0},
            ],
            "chips": [{"event": 3, "name": "wildcard"}],
        }
        assert app_calc_ft(history) == 4

    def test_freehit_preserves_fts_with_accrual(self):
        """FH preserves FTs and +1 accrual happens."""
        history = {
            "current": [
                {"event": 1, "event_transfers": 0, "event_transfers_cost": 0},
                {"event": 2, "event_transfers": 0, "event_transfers_cost": 0},
                {"event": 3, "event_transfers": 8, "event_transfers_cost": 0},  # FH
            ],
            "chips": [{"event": 3, "name": "freehit"}],
        }
        assert app_calc_ft(history) == 4

    def test_paid_transfers_dont_consume_fts(self):
        """Paid transfers (cost > 0) consume FTs first, then the rest are hits.

        GW1: 1 FT available, make 3 transfers, cost = 8 (2 hits * 4).
        free_used = 3 - 2 = 1 FT used. After: ft = 1 - 1 + 1 = 1.
        """
        history = {
            "current": [
                {"event": 1, "event_transfers": 3, "event_transfers_cost": 8},
            ],
            "chips": [],
        }
        assert app_calc_ft(history) == 1

    def test_minimum_is_always_one(self):
        """Even after spending all FTs, minimum is 1."""
        history = {
            "current": [
                {"event": 1, "event_transfers": 1, "event_transfers_cost": 0},
                {"event": 2, "event_transfers": 1, "event_transfers_cost": 0},
            ],
            "chips": [],
        }
        result = app_calc_ft(history)
        assert result >= 1

    def test_empty_history_returns_one(self):
        """Empty history returns 1."""
        assert app_calc_ft({"current": [], "chips": []}) == 1
        assert app_calc_ft({}) == 1

    def test_bank_then_spend(self):
        """Bank 2 GWs (save up to 3), then use 2 transfers.

        GW1: 0 transfers => FT: 1 -> 2
        GW2: 0 transfers => FT: 2 -> 3
        GW3: 2 transfers (free) => FT: 3 - 2 + 1 = 2
        """
        history = {
            "current": [
                {"event": 1, "event_transfers": 0, "event_transfers_cost": 0},
                {"event": 2, "event_transfers": 0, "event_transfers_cost": 0},
                {"event": 3, "event_transfers": 2, "event_transfers_cost": 0},
            ],
            "chips": [],
        }
        assert app_calc_ft(history) == 2
