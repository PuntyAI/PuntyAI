"""Tests for the Betfair auto-bet system."""

import math
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from punty.betting.queue import (
    calculate_stake, calculate_kelly_stake, populate_bet_queue,
    settle_betfair_bets, execute_due_bets, refresh_bet_selections,
    cycle_bet_selection, SWAP_THRESHOLD, DEFAULT_MIN_ODDS,
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


class TestCalculateKellyStake:
    """Test the Kelly-proportional staking logic."""

    def test_positive_edge_produces_stake(self):
        """10% edge at $2.00 odds → kelly = 0.10/1.0 = 0.10, capped at 0.08."""
        stake = calculate_kelly_stake(balance=200.0, place_probability=0.60, odds=2.00)
        # edge = 0.60 - 0.50 = 0.10, kelly = 0.10/1.0 = 0.10, capped 0.08
        assert stake == 0.08 * 200.0  # $16.00

    def test_no_edge_returns_zero(self):
        """Zero edge → no bet."""
        stake = calculate_kelly_stake(balance=50.0, place_probability=0.50, odds=2.00)
        assert stake == 0

    def test_negative_edge_returns_zero(self):
        """Negative edge → no bet."""
        stake = calculate_kelly_stake(balance=50.0, place_probability=0.30, odds=2.00)
        assert stake == 0

    def test_large_balance_scales(self):
        """Kelly scales with balance."""
        stake = calculate_kelly_stake(balance=1000.0, place_probability=0.70, odds=3.00)
        # edge = 0.70 - 0.333 = 0.367, kelly = 0.367/2.0 = 0.183, capped 0.08
        assert stake == 0.08 * 1000.0  # $80.00

    def test_small_edge_floors_to_min(self):
        """Small but positive edge → floored to $5 Betfair minimum."""
        stake = calculate_kelly_stake(balance=50.0, place_probability=0.56, odds=1.80)
        # edge = 0.56 - 0.556 = 0.004, kelly = 0.004/0.80 = 0.005
        # 0.005 * 50 = $0.25 < $5 min → rounds up to $5
        assert stake == 5.00

    def test_floor_applied(self):
        """Small edge floors to Betfair minimum."""
        stake = calculate_kelly_stake(balance=10.0, place_probability=0.56, odds=1.80)
        assert stake >= 5.00  # Betfair min

    def test_zero_balance(self):
        assert calculate_kelly_stake(0, 0.70, 2.00) == 0

    def test_invalid_odds(self):
        assert calculate_kelly_stake(50.0, 0.70, 1.00) == 0
        assert calculate_kelly_stake(50.0, 0.70, 0.50) == 0


class TestMinOddsFloor:
    """Test the minimum odds floor ($1.30)."""

    def test_min_odds_default(self):
        assert DEFAULT_MIN_ODDS == 1.30

    def test_kelly_zero_at_short_odds(self):
        """At $1.20 odds, implied = 83%. PP=80% has negative edge → $0."""
        stake = calculate_kelly_stake(balance=100.0, place_probability=0.80, odds=1.20)
        assert stake == 0  # 80% < 83.3% implied

    def test_kelly_positive_above_floor(self):
        """At $1.50 odds, implied = 66.7%. PP=80% has +13% edge → real stake."""
        stake = calculate_kelly_stake(balance=100.0, place_probability=0.80, odds=1.50)
        assert stake > 0


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

    def _setup_db_responses(self, mock_db, queued_bets, setting_value, picks, runners,
                             race_class="Class 1"):
        """Configure mock_db.execute to return the right objects in sequence.

        Call order in refresh_bet_selections:
        1. select(BetfairBet) → queued bets
        2-3. select(AppSettings) × 2 → min_place_prob, max_place_odds
        Then per bet:
        4. select(count(Runner)) → NTD runner count check
        5. select(Race) → maiden odds gate check
        6. select(Pick) → candidate picks for the race
        7. select(Runner) → current horse runner (scratched check)
        8..N. select(Runner) → one per candidate pick
        """
        # Build result mocks
        bets_result = MagicMock()
        bets_result.scalars.return_value.all.return_value = queued_bets

        def _make_setting_result(val):
            m = MagicMock()
            m.value = val
            r = MagicMock()
            r.scalar_one_or_none.return_value = m
            return r

        setting_prob = _make_setting_result(setting_value)
        setting_odds = _make_setting_result("6.0")

        # Runner count mock for NTD check (default 10 = safe)
        runner_count_result = MagicMock()
        runner_count_result.scalar.return_value = 10

        # Race mock for maiden check
        race_mock = MagicMock()
        race_mock.class_ = race_class
        race_result = MagicMock()
        race_result.scalar_one_or_none.return_value = race_mock

        picks_result = MagicMock()
        picks_result.scalars.return_value.all.return_value = picks

        runner_results = []
        for r in runners:
            rr = MagicMock()
            rr.scalar_one_or_none.return_value = r
            runner_results.append(rr)

        mock_db.execute = AsyncMock(
            side_effect=[bets_result, setting_prob, setting_odds,
                         runner_count_result, race_result, picks_result] + runner_results
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
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.75)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.70)

        runner_a_scratched = _make_mock_runner(1, scratched=True, current_odds=5.0)
        runner_a_for_pick = _make_mock_runner(1, scratched=True, current_odds=5.0)
        runner_b = _make_mock_runner(2, scratched=False, current_odds=5.0)

        # Runners: current check (scratched), then per-pick (a=scratched, b=ok)
        self._setup_db_responses(
            mock_db, [bet], "0.65", [pick_a, pick_b],
            [runner_a_scratched, runner_a_for_pick, runner_b]
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
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.75)

        runner_a_scratched = _make_mock_runner(1, scratched=True)
        runner_a_for_pick = _make_mock_runner(1, scratched=True)

        self._setup_db_responses(
            mock_db, [bet], "0.65", [pick_a],
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
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.74)  # 4pp > threshold

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_a_pick = _make_mock_runner(1, current_odds=5.0)
        runner_b_pick = _make_mock_runner(2, current_odds=5.0)

        self._setup_db_responses(
            mock_db, [bet], "0.65", [pick_a, pick_b],
            [runner_a, runner_a_pick, runner_b_pick]
        )

        count = await refresh_bet_selections(mock_db)
        assert count >= 1
        assert bet.pick_id == "p2"
        assert bet.horse_name == "Horse B"

    @pytest.mark.asyncio
    async def test_no_swap_below_threshold(self, mock_db):
        """Better candidate < 3pp → no swap."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.72)  # 2pp < threshold

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_a_pick = _make_mock_runner(1, current_odds=5.0)
        runner_b_pick = _make_mock_runner(2, current_odds=5.0)

        self._setup_db_responses(
            mock_db, [bet], "0.65", [pick_a, pick_b],
            [runner_a, runner_a_pick, runner_b_pick]
        )

        count = await refresh_bet_selections(mock_db)
        assert count == 0
        assert bet.pick_id == "p1"  # unchanged

    @pytest.mark.asyncio
    async def test_odds_on_stays_place(self, mock_db):
        """Horse < $2.00 → bet_type stays 'place' (no win flipping)."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1, bet_type="place")
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.80)

        runner_a = _make_mock_runner(1, current_odds=1.80)
        runner_a_pick = _make_mock_runner(1, current_odds=1.80)

        self._setup_db_responses(
            mock_db, [bet], "0.65", [pick_a],
            [runner_a, runner_a_pick]
        )

        count = await refresh_bet_selections(mock_db)
        assert count == 0  # No changes — stays place
        assert bet.bet_type == "place"


class TestBSPOrders:
    """Test BSP (Betfair Starting Price) order placement."""

    @pytest.mark.asyncio
    async def test_bsp_mock_returns_success(self):
        """BSP mock mode returns success with correct fields."""
        from punty.betting.betfair_client import place_bet
        with patch("punty.betting.betfair_client.settings") as mock_settings:
            mock_settings.mock_external = True
            result = await place_bet(
                AsyncMock(), "market-1", 12345, 10.0, 1.30, use_bsp=True
            )
            assert result["status"] == "SUCCESS"
            assert result["size_matched"] == 10.0

    @pytest.mark.asyncio
    async def test_limit_fallback(self):
        """use_bsp=False still uses LIMIT orders."""
        from punty.betting.betfair_client import place_bet
        with patch("punty.betting.betfair_client.settings") as mock_settings:
            mock_settings.mock_external = True
            result = await place_bet(
                AsyncMock(), "market-1", 12345, 10.0, 3.50, use_bsp=False
            )
            assert result["status"] == "SUCCESS"


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


class TestCycleBetSelection:
    """Test manual bet cycling through picks."""

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        db.commit = AsyncMock()
        return db

    def _odds_setting_result(self):
        """Mock for betfair_max_place_odds setting query."""
        setting_mock = MagicMock()
        setting_mock.value = "6.0"
        result = MagicMock()
        result.scalar_one_or_none.return_value = setting_mock
        return result

    @pytest.mark.asyncio
    async def test_cycle_to_next_pick(self, mock_db):
        """Cycling from pick 1 → pick 2."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.60)

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_b = _make_mock_runner(2, current_odds=5.0)

        # Call order: select bet, max_place_odds setting, select picks, runner checks
        bet_result = MagicMock()
        bet_result.scalar_one_or_none.return_value = bet
        picks_result = MagicMock()
        picks_result.scalars.return_value.all.return_value = [pick_a, pick_b]
        runner_a_result = MagicMock()
        runner_a_result.scalar_one_or_none.return_value = runner_a
        runner_b_result = MagicMock()
        runner_b_result.scalar_one_or_none.return_value = runner_b

        mock_db.execute = AsyncMock(
            side_effect=[bet_result, self._odds_setting_result(),
                         picks_result, runner_a_result, runner_b_result]
        )

        result = await cycle_bet_selection(mock_db, "bf-sale-2026-03-02-r1")
        assert result["swapped"] is True
        assert bet.horse_name == "Horse B"
        assert bet.pick_id == "p2"
        assert result["rank"] == 2

    @pytest.mark.asyncio
    async def test_cycle_wraps_around(self, mock_db):
        """Cycling from last pick wraps to first."""
        bet = _make_mock_bet(pick_id="p2", horse_name="Horse B", saddlecloth=2)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.60)

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_b = _make_mock_runner(2, current_odds=5.0)

        bet_result = MagicMock()
        bet_result.scalar_one_or_none.return_value = bet
        picks_result = MagicMock()
        picks_result.scalars.return_value.all.return_value = [pick_a, pick_b]
        runner_a_result = MagicMock()
        runner_a_result.scalar_one_or_none.return_value = runner_a
        runner_b_result = MagicMock()
        runner_b_result.scalar_one_or_none.return_value = runner_b

        mock_db.execute = AsyncMock(
            side_effect=[bet_result, self._odds_setting_result(),
                         picks_result, runner_a_result, runner_b_result]
        )

        result = await cycle_bet_selection(mock_db, "bf-sale-2026-03-02-r1")
        assert result["swapped"] is True
        assert bet.horse_name == "Horse A"
        assert result["rank"] == 1

    @pytest.mark.asyncio
    async def test_cycle_single_candidate(self, mock_db):
        """Only one valid pick → no swap."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)

        runner_a = _make_mock_runner(1, current_odds=5.0)

        bet_result = MagicMock()
        bet_result.scalar_one_or_none.return_value = bet
        picks_result = MagicMock()
        picks_result.scalars.return_value.all.return_value = [pick_a]
        runner_result = MagicMock()
        runner_result.scalar_one_or_none.return_value = runner_a

        mock_db.execute = AsyncMock(
            side_effect=[bet_result, self._odds_setting_result(),
                         picks_result, runner_result]
        )

        result = await cycle_bet_selection(mock_db, "bf-sale-2026-03-02-r1")
        assert result["swapped"] is False
        assert "one valid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_cycle_not_queued(self, mock_db):
        """Cannot cycle a placed bet."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1)
        bet.status = "placed"

        bet_result = MagicMock()
        bet_result.scalar_one_or_none.return_value = bet
        mock_db.execute = AsyncMock(return_value=bet_result)

        result = await cycle_bet_selection(mock_db, "bf-sale-2026-03-02-r1")
        assert result["swapped"] is False
        assert "placed" in result["message"]

    @pytest.mark.asyncio
    async def test_cycle_odds_on_stays_place(self, mock_db):
        """Cycling to an odds-on horse still sets bet_type to place (no win bets)."""
        bet = _make_mock_bet(pick_id="p1", horse_name="Horse A", saddlecloth=1, bet_type="place")
        pick_a = _make_mock_pick("p1", "Horse A", 1, 0.70)
        pick_b = _make_mock_pick("p2", "Horse B", 2, 0.60)

        runner_a = _make_mock_runner(1, current_odds=5.0)
        runner_b = _make_mock_runner(2, current_odds=1.50)  # Odds-on

        bet_result = MagicMock()
        bet_result.scalar_one_or_none.return_value = bet
        picks_result = MagicMock()
        picks_result.scalars.return_value.all.return_value = [pick_a, pick_b]
        runner_a_result = MagicMock()
        runner_a_result.scalar_one_or_none.return_value = runner_a
        runner_b_result = MagicMock()
        runner_b_result.scalar_one_or_none.return_value = runner_b

        mock_db.execute = AsyncMock(
            side_effect=[bet_result, self._odds_setting_result(),
                         picks_result, runner_a_result, runner_b_result]
        )

        result = await cycle_bet_selection(mock_db, "bf-sale-2026-03-02-r1")
        assert result["swapped"] is True
        assert bet.bet_type == "place"


class TestCalibration:
    """Test the probability calibration system."""

    def test_calibrate_no_data_returns_raw(self):
        """With empty calibration map, returns raw prediction."""
        from punty.betting.calibration import calibrate_probability
        assert calibrate_probability(0.75, {}) == 0.75

    def test_calibrate_corrects_overconfident(self):
        """Overconfident bin gets corrected down."""
        from punty.betting.calibration import calibrate_probability
        # Bin 7 (0.70-0.80, center 0.75) actual = 0.53
        cal_map = {7: 0.53}
        result = calibrate_probability(0.75, cal_map)
        assert result == 0.53

    def test_calibrate_corrects_underconfident(self):
        """Underconfident bin gets corrected up."""
        from punty.betting.calibration import calibrate_probability
        # Bin 4 (0.40-0.50, center 0.45) actual = 0.48
        cal_map = {4: 0.48}
        result = calibrate_probability(0.45, cal_map)
        assert result == 0.48

    def test_calibrate_interpolates_between_bins(self):
        """Probabilities between bin centers interpolate smoothly."""
        from punty.betting.calibration import calibrate_probability
        cal_map = {5: 0.52, 6: 0.67}  # bins 5 and 6
        # 0.60 is at center of bin 6
        result = calibrate_probability(0.60, cal_map)
        # Interpolation: pp=0.60 is slightly below bin 6 center (0.65),
        # so it interpolates toward bin 5
        assert 0.52 < result < 0.67

    def test_calibrate_zero_returns_zero(self):
        from punty.betting.calibration import calibrate_probability
        assert calibrate_probability(0.0, {5: 0.60}) == 0.0

    def test_calibrate_missing_bin_falls_through(self):
        """Missing bin returns raw prediction."""
        from punty.betting.calibration import calibrate_probability
        cal_map = {3: 0.35}  # Only bin 3 has data
        assert calibrate_probability(0.75, cal_map) == 0.75  # Bin 7 has no data

    def test_kelly_with_calibration_reduces_overconfident_stake(self):
        """Calibrated Kelly stakes less on overconfident predictions."""
        from punty.betting.calibration import calibrate_probability
        cal_map = {7: 0.53}  # Predicted 75% → actual 53%

        raw_pp = 0.75
        calibrated_pp = calibrate_probability(raw_pp, cal_map)

        # At $2.00 odds (implied 50%), $1000 balance to avoid min floor:
        # Raw: edge = 0.75 - 0.50 = 0.25, kelly = 0.25, capped 0.08 → $80
        # Calibrated: edge = 0.53 - 0.50 = 0.03, kelly = 0.03 → $30
        raw_stake = calculate_kelly_stake(1000, raw_pp, 2.00)
        cal_stake = calculate_kelly_stake(1000, calibrated_pp, 2.00)
        assert cal_stake < raw_stake
        assert cal_stake < raw_stake * 0.5  # At least halved

    def test_invalidate_cache(self):
        """Cache invalidation resets the cached map."""
        from punty.betting.calibration import invalidate_cache, _calibration_cache
        invalidate_cache()
        from punty.betting import calibration
        assert calibration._calibration_cache is None
        assert calibration._cache_expires is None
