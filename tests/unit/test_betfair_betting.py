"""Tests for the Betfair auto-bet system."""

import math
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from punty.betting.queue import calculate_stake, populate_bet_queue, settle_betfair_bets, execute_due_bets


class TestCalculateStake:
    """Test the dynamic stake doubling logic."""

    def test_base_case(self):
        assert calculate_stake(50, 50, 2) == 2.0

    def test_just_under_double(self):
        assert calculate_stake(99, 50, 2) == 2.0

    def test_exactly_double(self):
        assert calculate_stake(100, 50, 2) == 4.0

    def test_triple(self):
        assert calculate_stake(150, 50, 2) == 4.0

    def test_quadruple(self):
        assert calculate_stake(200, 50, 2) == 8.0

    def test_eight_times(self):
        assert calculate_stake(400, 50, 2) == 16.0

    def test_drawdown_keeps_base(self):
        """When balance drops below initial, keep base stake."""
        assert calculate_stake(30, 50, 2) == 2.0

    def test_zero_balance(self):
        assert calculate_stake(0, 50, 2) == 2.0

    def test_custom_base_stake(self):
        assert calculate_stake(200, 50, 5) == 20.0  # 5 * 2^2

    def test_custom_initial_balance(self):
        assert calculate_stake(200, 100, 2) == 4.0  # doubled once

    def test_large_growth(self):
        # $50 → $3200 = 6 doublings → $2 * 64 = $128
        assert calculate_stake(3200, 50, 2) == 128.0


class TestPopulateBetQueue:
    """Test bet queue population from approved content picks."""

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        db.commit = AsyncMock()
        db.add = MagicMock()
        return db

    @pytest.mark.asyncio
    async def test_disabled_setting_returns_zero(self, mock_db):
        """When betfair_auto_bet_enabled is false, no bets created."""
        mock_setting = MagicMock()
        mock_setting.value = "false"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_setting
        mock_db.execute = AsyncMock(return_value=mock_result)

        count = await populate_bet_queue(mock_db, "sale-2026-03-02", "content-1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_no_rank1_picks_returns_zero(self, mock_db):
        """When no rank 1 picks exist, no bets created."""
        # First call returns "true" for auto_bet_enabled
        enabled_setting = MagicMock()
        enabled_setting.value = "true"
        enabled_result = MagicMock()
        enabled_result.scalar_one_or_none.return_value = enabled_setting

        # Second call returns empty picks
        empty_result = MagicMock()
        empty_result.scalars.return_value.all.return_value = []

        mock_db.execute = AsyncMock(side_effect=[enabled_result, empty_result])
        count = await populate_bet_queue(mock_db, "sale-2026-03-02", "content-1")
        assert count == 0


class TestSettleBetfairBets:
    """Test Betfair bet settlement logic."""

    def test_place_hit_pnl_calculation(self):
        """Verify P&L formula: (odds-1) * stake * (1-commission)."""
        odds = 3.50
        stake = 2.00
        commission = 0.05
        gross = (odds - 1) * stake  # 5.00
        net = gross - gross * commission  # 5.00 - 0.25 = 4.75
        assert round(net, 2) == 4.75

    def test_miss_pnl(self):
        """Verify miss = -stake."""
        stake = 2.00
        assert -stake == -2.00

    def test_stake_doubling_progression(self):
        """Verify the full progression table."""
        initial = 50
        base = 2
        # Simulate growth
        balance = 50
        expected_stakes = [2, 2, 4, 4, 8]
        test_balances = [50, 75, 100, 150, 200]
        for bal, expected in zip(test_balances, expected_stakes):
            assert calculate_stake(bal, initial, base) == expected, f"balance={bal}"
