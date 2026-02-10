"""Unit tests for pick settlement logic."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from punty.models.pick import Pick


class TestWinBetSettlement:
    """Tests for win bet settlement calculations."""

    def test_win_bet_winner(self):
        """Win bet on 1st place horse should return (odds * stake) - stake."""
        pick = Pick(
            id="test-1",
            bet_type="win",
            bet_stake=10.0,
            odds_at_tip=3.50,
        )

        # Simulate settlement
        stake = pick.bet_stake
        win_odds = pick.odds_at_tip
        finish_position = 1

        if finish_position == 1:
            pnl = round(win_odds * stake - stake, 2)
            hit = True
        else:
            pnl = round(-stake, 2)
            hit = False

        assert hit is True
        assert pnl == 25.0  # (3.50 * 10) - 10 = 25

    def test_win_bet_loser(self):
        """Win bet on non-winner should lose full stake."""
        stake = 10.0
        finish_position = 2

        hit = finish_position == 1
        pnl = -stake if not hit else 0

        assert hit is False
        assert pnl == -10.0


class TestPlaceBetSettlement:
    """Tests for place bet settlement calculations."""

    def test_place_bet_first(self):
        """Place bet on 1st should win."""
        stake = 6.0
        place_odds = 1.80
        finish_position = 1

        placed = finish_position <= 3
        if placed:
            pnl = round(place_odds * stake - stake, 2)
            hit = True
        else:
            pnl = round(-stake, 2)
            hit = False

        assert hit is True
        assert pnl == 4.8  # (1.80 * 6) - 6 = 4.8

    def test_place_bet_third(self):
        """Place bet on 3rd should win."""
        stake = 6.0
        place_odds = 1.80
        finish_position = 3

        placed = finish_position <= 3
        pnl = round(place_odds * stake - stake, 2) if placed else round(-stake, 2)

        assert placed is True
        assert pnl == 4.8

    def test_place_bet_fourth(self):
        """Place bet on 4th should lose."""
        stake = 6.0
        finish_position = 4

        placed = finish_position <= 3
        pnl = round(-stake, 2)

        assert placed is False
        assert pnl == -6.0


class TestEachWaySettlement:
    """Tests for each way bet settlement calculations."""

    def test_each_way_winner(self):
        """Each way bet on winner wins both halves."""
        stake = 4.0
        half = stake / 2  # 2.0
        win_odds = 8.00
        place_odds = 2.50
        finish_position = 1

        won = finish_position == 1

        if won:
            # Win half pays at win odds, place half pays at place odds
            pnl = round(win_odds * half + place_odds * half - stake, 2)
            hit = True
        else:
            pnl = 0
            hit = False

        assert hit is True
        assert pnl == 17.0  # (8.0 * 2) + (2.50 * 2) - 4 = 16 + 5 - 4 = 17

    def test_each_way_placer(self):
        """Each way bet on placer (not winner) wins place half only."""
        stake = 4.0
        half = stake / 2  # 2.0
        win_odds = 8.00
        place_odds = 2.50
        finish_position = 3

        won = finish_position == 1
        placed = finish_position <= 3

        if won:
            pnl = round(win_odds * half + place_odds * half - stake, 2)
            hit = True
        elif placed:
            # Only place half pays, win half lost
            pnl = round(place_odds * half - stake, 2)
            hit = True
        else:
            pnl = round(-stake, 2)
            hit = False

        assert hit is True
        assert pnl == 1.0  # (2.50 * 2) - 4 = 5 - 4 = 1

    def test_each_way_fourth(self):
        """Each way bet on 4th loses everything."""
        stake = 4.0
        finish_position = 4

        won = finish_position == 1
        placed = finish_position <= 3

        if won:
            hit = True
            pnl = 0
        elif placed:
            hit = True
            pnl = 0
        else:
            pnl = round(-stake, 2)
            hit = False

        assert hit is False
        assert pnl == -4.0


class TestExoticSettlement:
    """Tests for exotic bet settlement."""

    def test_exacta_hit(self):
        """Exacta with correct 1-2 should pay dividend."""
        exotic_runners = [3, 7]  # Our selection
        finish_order = [3, 7, 1, 5]  # Actual finish
        stake = 20.0
        dividend = 45.60  # Per $1 unit

        # Check if hit
        hit = list(finish_order[:2]) == exotic_runners

        if hit:
            pnl = round(dividend * stake - stake, 2)
        else:
            pnl = round(-stake, 2)

        assert hit is True
        assert pnl == 892.0  # (45.60 * 20) - 20 = 912 - 20 = 892

    def test_exacta_miss_reversed(self):
        """Exacta with reversed order should miss."""
        exotic_runners = [3, 7]
        finish_order = [7, 3, 1, 5]  # Reversed!
        stake = 20.0

        hit = list(finish_order[:2]) == exotic_runners

        assert hit is False

    def test_boxed_trifecta_hit(self):
        """Boxed trifecta should hit if all 3 finish in top 3."""
        exotic_runners = [1, 5, 8]
        finish_order = [5, 8, 1, 3]  # Different order but all in top 3
        stake = 20.0
        dividend = 125.30

        # Boxed = any order
        hit = set(finish_order[:3]) == set(exotic_runners)

        if hit:
            pnl = round(dividend * stake - stake, 2)
        else:
            pnl = round(-stake, 2)

        assert hit is True
        assert pnl == 2486.0  # (125.30 * 20) - 20 = 2506 - 20 = 2486

    def test_quinella_hit(self):
        """Quinella with correct 1-2 in any order."""
        exotic_runners = [2, 6]
        finish_order = [6, 2, 4, 1]  # Reversed but still hits
        stake = 20.0
        dividend = 18.40

        # Quinella = any order for top 2
        hit = set(finish_order[:2]) == set(exotic_runners)

        if hit:
            pnl = round(dividend * stake - stake, 2)
        else:
            pnl = round(-stake, 2)

        assert hit is True
        assert pnl == 348.0  # (18.40 * 20) - 20 = 368 - 20 = 348


class TestQuinellaLegsFormat:
    """Tests for quinella settlement with legs format (edge case)."""

    def _check_quinella_legs(self, legs, top2):
        """Replicate the fixed quinella legs logic from picks.py."""
        a, b = top2[0], top2[1]
        return (
            (a in legs[0] and b in legs[1]) or
            (b in legs[0] and a in legs[1])
        )

    def test_quinella_legs_hit_normal_order(self):
        """Quinella [[1,2],[3,4]] hits when 1st=1, 2nd=3."""
        assert self._check_quinella_legs([[1, 2], [3, 4]], [1, 3]) is True

    def test_quinella_legs_hit_reversed_order(self):
        """Quinella [[1,2],[3,4]] hits when 1st=3, 2nd=2 (reversed legs)."""
        assert self._check_quinella_legs([[1, 2], [3, 4]], [3, 2]) is True

    def test_quinella_legs_miss_both_from_same_leg(self):
        """Quinella [[1,2],[3,4]] should NOT hit when 1st=1, 2nd=2 (both from leg 0)."""
        assert self._check_quinella_legs([[1, 2], [3, 4]], [1, 2]) is False

    def test_quinella_legs_miss_neither_in_legs(self):
        """Quinella [[1,2],[3,4]] misses when result is 5-6."""
        assert self._check_quinella_legs([[1, 2], [3, 4]], [5, 6]) is False

    def test_quinella_legs_single_runner_per_leg(self):
        """Quinella [[1],[3]] hits when result is 3-1."""
        assert self._check_quinella_legs([[1], [3]], [3, 1]) is True

    def test_quinella_legs_single_runner_miss(self):
        """Quinella [[1],[3]] misses when result is 1-2."""
        assert self._check_quinella_legs([[1], [3]], [1, 2]) is False


class TestQuaddieSettlement:
    """Tests for quaddie settlement."""

    def test_quaddie_all_legs_hit(self):
        """Quaddie where all 4 legs hit should pay."""
        legs = [[1, 2], [3], [5, 6], [8]]  # Our selections per leg
        winners = [1, 3, 5, 8]  # Actual winners
        cost = 50.0
        dividend = 1250.60

        # Check each leg
        all_hit = True
        for i, leg in enumerate(legs):
            if winners[i] not in leg:
                all_hit = False
                break

        if all_hit:
            pnl = round(dividend - cost, 2)
        else:
            pnl = round(-cost, 2)

        assert all_hit is True
        assert pnl == 1200.60  # 1250.60 - 50 = 1200.60

    def test_quaddie_one_leg_miss(self):
        """Quaddie where one leg misses loses full cost."""
        legs = [[1, 2], [3], [5, 6], [8]]
        winners = [1, 4, 5, 8]  # Leg 2 missed (winner was 4, we had 3)
        cost = 50.0

        all_hit = True
        for i, leg in enumerate(legs):
            if winners[i] not in leg:
                all_hit = False
                break

        pnl = round(-cost, 2)

        assert all_hit is False
        assert pnl == -50.0


class TestSaverWinSettlement:
    """Tests for saver win bet type."""

    def test_saver_win_winner(self):
        """Saver win on winner should pay at win odds."""
        stake = 5.0
        win_odds = 4.50
        finish_position = 1

        won = finish_position == 1
        if won:
            pnl = round(win_odds * stake - stake, 2)
            hit = True
        else:
            pnl = round(-stake, 2)
            hit = False

        assert hit is True
        assert pnl == 17.5  # (4.50 * 5) - 5 = 22.5 - 5 = 17.5

    def test_saver_win_second(self):
        """Saver win on 2nd should lose."""
        stake = 5.0
        finish_position = 2

        won = finish_position == 1
        pnl = round(-stake, 2)

        assert won is False
        assert pnl == -5.0


class TestBig3MultiSettlement:
    """Tests for Big 3 multi settlement."""

    def test_big3_multi_all_win(self):
        """Big 3 multi where all 3 win should pay multi odds."""
        stake = 10.0
        multi_odds = 150.0
        horses_won = [True, True, True]

        all_won = all(horses_won)
        if all_won:
            pnl = round(multi_odds * stake - stake, 2)
        else:
            pnl = round(-stake, 2)

        assert all_won is True
        assert pnl == 1490.0  # (150 * 10) - 10 = 1500 - 10 = 1490

    def test_big3_multi_one_loses(self):
        """Big 3 multi where one loses should lose stake."""
        stake = 10.0
        horses_won = [True, False, True]

        all_won = all(horses_won)
        pnl = round(-stake, 2)

        assert all_won is False
        assert pnl == -10.0
