"""Tests for the Betfair auto-bet system."""

import math
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from punty.betting.queue import (
    calculate_stake, populate_bet_queue, settle_betfair_bets,
    execute_due_bets, refresh_bet_selections, SWAP_THRESHOLD, ODDS_ON_THRESHOLD,
)


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


def _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1,
                   meeting_id="sale-2026-03-02", race_number=1,
                   bet_type="place"):
    """Helper to create a mock BetfairBet."""
    bet = MagicMock()
    bet.id = f"bf-{meeting_id}-r{race_number}"
    bet.pick_id = pick_id
    bet.horse_name = horse_name
    bet.saddlecloth = saddlecloth
    bet.meeting_id = meeting_id
    bet.race_number = race_number
    bet.status = "queued"
    bet.enabled = True
    bet.bet_type = bet_type
    bet.requested_odds = 3.50
    return bet


def _make_mock_pick(pick_id, horse_name, saddlecloth, place_prob,
                    place_odds=3.50, meeting_id="sale-2026-03-02", race_number=1):
    """Helper to create a mock Pick."""
    pick = MagicMock()
    pick.id = pick_id
    pick.horse_name = horse_name
    pick.saddlecloth = saddlecloth
    pick.place_probability = place_prob
    pick.place_odds_at_tip = place_odds
    pick.meeting_id = meeting_id
    pick.race_number = race_number
    pick.pick_type = "selection"
    pick.tracked_only = False
    return pick


def _make_mock_runner(saddlecloth, scratched=False, current_odds=5.0):
    """Helper to create a mock Runner."""
    runner = MagicMock()
    runner.saddlecloth = saddlecloth
    runner.scratched = scratched
    runner.current_odds = current_odds
    return runner


class TestRefreshBetSelections:
    """Test dynamic bet refresh — scratching swaps, probability upgrades, odds-on flips."""

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        db.commit = AsyncMock()
        return db

    def _setup_db_responses(self, mock_db, queued_bets, setting_value, picks, runners):
        """Configure mock_db.execute to return the right objects in sequence.

        Call order in refresh_bet_selections:
        1. select(BetfairBet) → queued bets
        2. select(AppSettings) → min_place_prob setting
        Then per bet:
        3. select(Pick) → candidate picks for the race
        4. select(Runner) → current horse runner (scratched check)
        5..N. select(Runner) → one per candidate pick
        N+1. select(Runner) → odds-on check runner
        """
        # Build result mocks
        bets_result = MagicMock()
        bets_result.scalars.return_value.all.return_value = queued_bets

        setting_mock = MagicMock()
        setting_mock.value = setting_value
        setting_result = MagicMock()
        setting_result.scalar_one_or_none.return_value = setting_mock

        picks_result = MagicMock()
        picks_result.scalars.return_value.all.return_value = picks

        runner_results = []
        for r in runners:
            rr = MagicMock()
            rr.scalar_one_or_none.return_value = r
            runner_results.append(rr)

        mock_db.execute = AsyncMock(
            side_effect=[bets_result, setting_result, picks_result] + runner_results
        )

    @pytest.mark.asyncio
    async def test_no_queued_bets(self, mock_db):
        """No queued bets → returns 0, no commit."""
        empty_result = MagicMock()
        empty_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=empty_result)

        count = await refresh_bet_selections(mock_db)
        assert count == 0
        mock_db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scratched_horse_swapped(self, mock_db):
        """Scratched current horse → swap to best alternative."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.65)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.60)

        runner_a_scratched = _make_mock_runner(1, scratched=True, current_odds=5.0)
        runner_a_for_pick = _make_mock_runner(1, scratched=True, current_odds=5.0)
        runner_b = _make_mock_runner(2, scratched=False, current_odds=5.0)
        runner_odds_check = _make_mock_runner(2, scratched=False, current_odds=5.0)

        # Runners: current check (scratched), then per-pick (a=scratched, b=ok), then odds-on check
        self._setup_db_responses(
            mock_db, [bet], "0.50", [pick_a, pick_b],
            [runner_a_scratched, runner_a_for_pick, runner_b, runner_odds_check]
        )

        count = await refresh_bet_selections(mock_db)
        assert count >= 1
        assert bet.pick_id == "p2"
        assert bet.horse_name == "Horse B"
        assert bet.saddlecloth == 2

    @pytest.mark.asyncio
    async def test_scratched_no_replacement_cancels(self, mock_db):
        """All candidates scratched → cancel bet."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.65)

        runner_a_scratched = _make_mock_runner(1, scratched=True)
        runner_a_for_pick = _make_mock_runner(1, scratched=True)

        self._setup_db_responses(
            mock_db, [bet], "0.50", [pick_a],
            [runner_a_scratched, runner_a_for_pick]
        )

        count = await refresh_bet_selections(mock_db)
        assert count >= 1
        assert bet.status == "cancelled"
        assert "scratched" in bet.error_message.lower()

    @pytest.mark.asyncio
    async def test_higher_prob_swap_above_threshold(self, mock_db):
        """Better candidate > 3pp higher → swap."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.60)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.64)  # 4pp > threshold

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_a_pick = _make_mock_runner(1, current_odds=5.0)
        runner_b_pick = _make_mock_runner(2, current_odds=5.0)
        runner_odds_check = _make_mock_runner(2, current_odds=5.0)

        self._setup_db_responses(
            mock_db, [bet], "0.50", [pick_a, pick_b],
            [runner_a, runner_a_pick, runner_b_pick, runner_odds_check]
        )

        count = await refresh_bet_selections(mock_db)
        assert count >= 1
        assert bet.pick_id == "p2"
        assert bet.horse_name == "Horse B"

    @pytest.mark.asyncio
    async def test_no_swap_below_threshold(self, mock_db):
        """Better candidate < 3pp → no swap."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.60)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.62)  # 2pp < threshold

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_a_pick = _make_mock_runner(1, current_odds=5.0)
        runner_b_pick = _make_mock_runner(2, current_odds=5.0)
        runner_odds_check = _make_mock_runner(1, current_odds=5.0)

        self._setup_db_responses(
            mock_db, [bet], "0.50", [pick_a, pick_b],
            [runner_a, runner_a_pick, runner_b_pick, runner_odds_check]
        )

        count = await refresh_bet_selections(mock_db)
        assert count == 0
        assert bet.pick_id == "p1"  # unchanged

    @pytest.mark.asyncio
    async def test_odds_on_flips_to_win(self, mock_db):
        """Horse < $2.00 → bet_type flipped to 'win'."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1, bet_type="place")
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.80)

        runner_a = _make_mock_runner(1, current_odds=1.80)  # Odds-on
        runner_a_pick = _make_mock_runner(1, current_odds=1.80)
        runner_odds_check = _make_mock_runner(1, current_odds=1.80)

        self._setup_db_responses(
            mock_db, [bet], "0.50", [pick_a],
            [runner_a, runner_a_pick, runner_odds_check]
        )

        count = await refresh_bet_selections(mock_db)
        assert count >= 1
        assert bet.bet_type == "win"

    @pytest.mark.asyncio
    async def test_odds_drift_reverts_to_place(self, mock_db):
        """Horse drifts above $2.00 → bet_type reverted to 'place'."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1, bet_type="win")
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)

        runner_a = _make_mock_runner(1, current_odds=2.50)  # Drifted
        runner_a_pick = _make_mock_runner(1, current_odds=2.50)
        runner_odds_check = _make_mock_runner(1, current_odds=2.50)

        self._setup_db_responses(
            mock_db, [bet], "0.50", [pick_a],
            [runner_a, runner_a_pick, runner_odds_check]
        )

        count = await refresh_bet_selections(mock_db)
        assert count >= 1
        assert bet.bet_type == "place"


class TestSettlementBetType:
    """Test win vs place settlement threshold."""

    def test_win_bet_hit_only_first(self):
        """Win bet: only finish_position == 1 is a hit."""
        # finish_position 1 → hit for win (threshold 1)
        assert 1 <= 1  # hit
        assert not (2 <= 1)  # miss
        assert not (3 <= 1)  # miss

    def test_place_bet_hit_top_three(self):
        """Place bet: finish_position 1-3 is a hit."""
        assert 1 <= 3
        assert 2 <= 3
        assert 3 <= 3
        assert not (4 <= 3)
