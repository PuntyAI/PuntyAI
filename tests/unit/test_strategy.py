"""Unit tests for RAG betting strategy aggregation and context builder."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from punty.memory.strategy import (
    _make_stat,
    _pnl_str,
    _normalise_exotic,
    _generate_directives,
    _insight_text,
    aggregate_bet_type_performance,
    aggregate_tip_rank_performance,
    build_strategy_context,
    populate_pattern_insights,
)


class TestMakeStat:
    """Tests for _make_stat helper."""

    def test_basic_stat(self):
        s = _make_stat("selection", "Win", 100, 25, 800.0, -96.0, 4.50, 32.0)
        assert s["category"] == "selection"
        assert s["sub_type"] == "Win"
        assert s["bets"] == 100
        assert s["winners"] == 25
        assert s["strike_rate"] == 25.0
        assert s["staked"] == 800.0
        assert s["pnl"] == -96.0
        assert s["returned"] == 704.0  # 800 + (-96)
        assert s["roi"] == -12.0  # -96/800*100
        assert s["avg_odds"] == 4.50
        assert s["best_win_pnl"] == 32.0

    def test_zero_bets(self):
        s = _make_stat("exotic", "Trifecta", 0, 0, 0.0, 0.0, 0.0, 0.0)
        assert s["strike_rate"] == 0
        assert s["roi"] == 0

    def test_profitable_roi(self):
        s = _make_stat("selection", "Place", 50, 24, 300.0, 33.0, 2.10, 12.0)
        assert s["roi"] == 11.0  # 33/300*100
        assert s["strike_rate"] == 48.0

    def test_zero_staked_no_division_error(self):
        s = _make_stat("selection", "Win", 5, 1, 0.0, 0.0, 3.0, 0.0)
        assert s["roi"] == 0


class TestPnlStr:
    """Tests for _pnl_str formatting."""

    def test_positive(self):
        assert _pnl_str(45.50) == "+$45.50"

    def test_negative(self):
        assert _pnl_str(-12.75) == "-$12.75"

    def test_zero(self):
        assert _pnl_str(0.0) == "+$0.00"


class TestNormaliseExotic:
    """Tests for exotic type normalisation."""

    def test_trifecta_box(self):
        assert _normalise_exotic("trifecta_box") == "Trifecta Box"
        assert _normalise_exotic("Trifecta Box") == "Trifecta Box"

    def test_exacta_standout(self):
        assert _normalise_exotic("exacta_standout") == "Exacta Standout"

    def test_quinella(self):
        assert _normalise_exotic("quinella") == "Quinella"

    def test_first_four(self):
        assert _normalise_exotic("first_four") == "First Four"
        assert _normalise_exotic("first_4") == "First Four"

    def test_plain_trifecta(self):
        assert _normalise_exotic("trifecta") == "Trifecta"

    def test_plain_exacta(self):
        assert _normalise_exotic("exacta") == "Exacta"

    def test_unknown_passes_through(self):
        assert _normalise_exotic("some_new_type") == "Some_New_Type"


class TestInsightText:
    """Tests for _insight_text."""

    def test_profitable(self):
        stat = {"sub_type": "Place", "roi": 11.0, "strike_rate": 48.0, "bets": 50, "pnl": 33.0}
        text = _insight_text(stat, "all-time")
        assert "PROFITABLE" in text
        assert "Place" in text
        assert "48.0% SR" in text

    def test_losing(self):
        stat = {"sub_type": "Win", "roi": -25.0, "strike_rate": 20.0, "bets": 100, "pnl": -200.0}
        text = _insight_text(stat, "last 30 days")
        assert "LOSING" in text

    def test_breakeven(self):
        stat = {"sub_type": "Each Way", "roi": -2.0, "strike_rate": 35.0, "bets": 30, "pnl": -6.0}
        text = _insight_text(stat, "all-time")
        assert "BREAKEVEN" in text


class TestGenerateDirectives:
    """Tests for _generate_directives."""

    def test_lean_into_profitable(self):
        overall = [
            _make_stat("selection", "Place", 50, 24, 300.0, 33.0, 2.10, 12.0),  # +11% ROI
            _make_stat("selection", "Win", 100, 25, 800.0, -96.0, 4.50, 32.0),  # -12% ROI
        ]
        directives = _generate_directives(overall, [], [])
        assert any("LEAN INTO PLACE" in d for d in directives)
        assert any("REDUCE WIN" in d for d in directives)

    def test_roughie_profitable(self):
        ranks = [
            {"rank": 1, "label": "Top Pick", "strike_rate": 22.0, "roi": -8.0, "bets": 80},
            {"rank": 4, "label": "Roughie", "strike_rate": 8.3, "roi": 12.5, "bets": 60},
        ]
        directives = _generate_directives([], [], ranks)
        assert any("ROUGHIES are profitable" in d for d in directives)

    def test_roughie_low_sr(self):
        ranks = [
            {"rank": 1, "label": "Top Pick", "strike_rate": 22.0, "roi": 5.0, "bets": 80},
            {"rank": 4, "label": "Roughie", "strike_rate": 3.0, "roi": -30.0, "bets": 40},
        ]
        directives = _generate_directives([], [], ranks)
        assert any("Exotics only" in d for d in directives)

    def test_top_pick_losing_roi(self):
        ranks = [
            {"rank": 1, "label": "Top Pick", "strike_rate": 18.0, "roi": -15.0, "bets": 100},
        ]
        directives = _generate_directives([], [], ranks)
        assert any("TOP PICKS" in d for d in directives)

    def test_place_vs_win_comparison(self):
        overall = [
            _make_stat("selection", "Win", 100, 25, 800.0, -96.0, 4.50, 32.0),  # -12% ROI
            _make_stat("selection", "Place", 50, 24, 300.0, 33.0, 2.10, 12.0),  # +11% ROI
        ]
        directives = _generate_directives(overall, [], [])
        assert any("PLACE outperforming WIN" in d for d in directives)

    def test_each_way_profitable(self):
        overall = [
            _make_stat("selection", "Each Way", 40, 17, 400.0, 20.0, 6.50, 40.0),  # +5%
        ]
        directives = _generate_directives(overall, [], [])
        assert any("EACH WAY working" in d for d in directives)

    def test_each_way_bleeding(self):
        overall = [
            _make_stat("selection", "Each Way", 40, 8, 400.0, -100.0, 6.50, 10.0),  # -25%
        ]
        directives = _generate_directives(overall, [], [])
        assert any("EACH WAY bleeding" in d for d in directives)

    def test_momentum_positive(self):
        overall = [_make_stat("selection", "Win", 100, 20, 800.0, -50.0, 4.0, 20.0)]
        recent = [_make_stat("selection", "Win", 20, 6, 160.0, 30.0, 4.0, 20.0)]
        directives = _generate_directives(overall, recent, [])
        assert any("MOMENTUM POSITIVE" in d for d in directives)

    def test_recent_slump(self):
        overall = [_make_stat("selection", "Win", 100, 30, 800.0, 50.0, 4.0, 20.0)]
        recent = [_make_stat("selection", "Win", 20, 3, 160.0, -40.0, 4.0, 20.0)]
        directives = _generate_directives(overall, recent, [])
        assert any("RECENT SLUMP" in d for d in directives)

    def test_empty_data(self):
        directives = _generate_directives([], [], [])
        assert len(directives) == 1
        assert "more data" in directives[0]

    def test_losing_threshold(self):
        """Bet types with ROI > -10% should NOT be flagged as losing."""
        overall = [
            _make_stat("selection", "Win", 100, 25, 800.0, -40.0, 4.50, 32.0),  # -5% ROI
        ]
        directives = _generate_directives(overall, [], [])
        assert not any("REDUCE WIN" in d for d in directives)

    def test_small_sample_not_flagged_losing(self):
        """Bet types with fewer than 10 bets should NOT be flagged as losing."""
        overall = [
            _make_stat("selection", "Saver Win", 5, 0, 50.0, -50.0, 8.0, 0.0),  # -100% but only 5 bets
        ]
        directives = _generate_directives(overall, [], [])
        assert not any("REDUCE" in d for d in directives)


class TestBuildStrategyContext:
    """Tests for build_strategy_context output formatting."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_data(self):
        db = AsyncMock()
        with patch("punty.memory.strategy.aggregate_bet_type_performance", return_value=[]):
            result = await build_strategy_context(db)
        assert result == ""

    @pytest.mark.asyncio
    async def test_contains_all_sections(self):
        mock_overall = [
            _make_stat("selection", "Win", 100, 25, 800.0, -96.0, 4.50, 32.0),
            _make_stat("selection", "Place", 50, 24, 300.0, 33.0, 2.10, 12.0),
        ]
        mock_recent = [
            _make_stat("selection", "Win", 20, 5, 160.0, -20.0, 4.50, 15.0),
        ]
        mock_ranks = [
            {"rank": 1, "label": "Top Pick", "bets": 80, "winners": 18, "strike_rate": 22.5, "staked": 640.0, "pnl": -51.2, "roi": -8.0, "avg_odds": 3.80},
            {"rank": 4, "label": "Roughie", "bets": 60, "winners": 5, "strike_rate": 8.3, "staked": 300.0, "pnl": 37.5, "roi": 12.5, "avg_odds": 15.0},
        ]

        async def mock_agg(db, window_days=None):
            return mock_recent if window_days == 30 else mock_overall

        mock_pp = {"bets": 30, "winners": 9, "strike_rate": 30.0, "staked": 240.0, "pnl": -18.0, "roi": -7.5, "avg_odds": 4.20}

        async def mock_pp_agg(db, window_days=None):
            return mock_pp

        mock_recent_results = [
            "- [WIN] Top [PP]: FAST HORSE @ $3.50 Win → Finished 1 | +$25.00",
            "- [LOSS] 2nd: SLOW HORSE @ $5.00 Place → Finished 6 | -$6.00",
            "- [WIN] Exacta: runners [1, 3] — $20 stake | +$85.00",
        ]

        db = AsyncMock()
        with patch("punty.memory.strategy.aggregate_bet_type_performance", side_effect=mock_agg), \
             patch("punty.memory.strategy.aggregate_tip_rank_performance", return_value=mock_ranks), \
             patch("punty.memory.strategy.aggregate_puntys_pick_performance", side_effect=mock_pp_agg), \
             patch("punty.memory.strategy.get_recent_results_with_context", return_value=mock_recent_results):
            result = await build_strategy_context(db)

        assert "## YOUR BETTING TRACK RECORD" in result
        assert "ALL-TIME" in result
        assert "Bet Type Scorecard" in result
        assert "Last 30 Days" in result
        assert "Pick Rank Performance" in result
        assert "PUNTY'S PICK Performance" in result
        assert "STRATEGY DIRECTIVES" in result
        assert "ROI TARGETS" in result
        assert "RECENT RESULTS" in result
        assert "FAST HORSE" in result
        assert "PROFITABLE" in result
        assert "LOSING" in result
        assert "Top Pick" in result
        assert "Roughie" in result
