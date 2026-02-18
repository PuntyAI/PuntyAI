"""Tests for Big 3 Multi pre-calculation."""

import pytest

from punty.context.pre_big3 import (
    Big3Candidate,
    Big3Recommendation,
    calculate_pre_big3,
    format_pre_big3,
    MIN_WIN_PROB,
    MIN_MULTI_EV,
    POOL_TAKEOUT,
    MULTI_STAKE,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _race_ctx(race_number, picks=None, ranked=None, runners=None):
    """Build a mock race context dict."""
    if ranked is None:
        ranked = [
            {"saddlecloth": 1, "horse": "Horse1", "win_prob": 0.30},
            {"saddlecloth": 2, "horse": "Horse2", "win_prob": 0.20},
            {"saddlecloth": 3, "horse": "Horse3", "win_prob": 0.10},
        ]
    if runners is None:
        runners = [
            {"saddlecloth": i, "horse_name": f"Horse{i}", "current_odds": 3.0 + i}
            for i in range(1, 6)
        ]

    ctx = {
        "race_number": race_number,
        "probabilities": {"probability_ranked": ranked},
        "runners": runners,
    }
    if picks is not None:
        ctx["pre_selections"] = {"picks": picks}
    return ctx


def _pick(sc, name, odds, rank=1, reason="Good form"):
    """Build a mock pre-selection pick."""
    return {
        "saddlecloth": sc,
        "horse_name": name,
        "win_odds": odds,
        "rank": rank,
        "reason": reason,
    }


# ──────────────────────────────────────────────
# Tests: calculate_pre_big3
# ──────────────────────────────────────────────

class TestCalculatePreBig3:
    """Tests for the calculate_pre_big3 function."""

    def test_returns_none_with_fewer_than_3_races(self):
        contexts = [_race_ctx(1), _race_ctx(2)]
        assert calculate_pre_big3(contexts) is None

    def test_returns_recommendation_with_3_races(self):
        contexts = [_race_ctx(i) for i in range(1, 4)]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        assert isinstance(rec, Big3Recommendation)
        assert len(rec.horses) == 3

    def test_horses_from_different_races(self):
        contexts = [_race_ctx(i) for i in range(1, 5)]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        race_numbers = {h.race_number for h in rec.horses}
        assert len(race_numbers) == 3  # all from different races

    def test_picks_highest_ev_combination(self):
        # Race 1: strong fav with high prob
        # Race 2: moderate fav
        # Race 3: weak field - should still pick best
        # Race 4: very weak - should be excluded from optimal combo
        contexts = [
            _race_ctx(1, ranked=[
                {"saddlecloth": 1, "horse": "Star", "win_prob": 0.45},
                {"saddlecloth": 2, "horse": "Also", "win_prob": 0.20},
            ], runners=[
                {"saddlecloth": 1, "horse_name": "Star", "current_odds": 2.5},
                {"saddlecloth": 2, "horse_name": "Also", "current_odds": 5.0},
            ]),
            _race_ctx(2, ranked=[
                {"saddlecloth": 3, "horse": "Mid", "win_prob": 0.30},
            ], runners=[
                {"saddlecloth": 3, "horse_name": "Mid", "current_odds": 3.5},
            ]),
            _race_ctx(3, ranked=[
                {"saddlecloth": 5, "horse": "Outsider", "win_prob": 0.20},
            ], runners=[
                {"saddlecloth": 5, "horse_name": "Outsider", "current_odds": 5.0},
            ]),
        ]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        assert rec.multi_prob == pytest.approx(0.45 * 0.30 * 0.20, abs=0.001)

    def test_uses_pre_selections_when_available(self):
        picks = [
            _pick(1, "Picked", 3.0, rank=1),
            _pick(2, "Second", 5.0, rank=2),
        ]
        contexts = [
            _race_ctx(1, picks=picks),
            _race_ctx(2),
            _race_ctx(3),
        ]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        # Should include "Picked" from pre-selections
        names = {h.horse_name for h in rec.horses}
        assert "Picked" in names or "Second" in names

    def test_filters_below_min_win_prob(self):
        contexts = [
            _race_ctx(1, ranked=[
                {"saddlecloth": 1, "horse": "Weak", "win_prob": 0.05},
            ], runners=[
                {"saddlecloth": 1, "horse_name": "Weak", "current_odds": 20.0},
            ]),
            _race_ctx(2),
            _race_ctx(3),
        ]
        rec = calculate_pre_big3(contexts)
        # Race 1 has no candidates above MIN_WIN_PROB, so only 2 races available
        if rec is not None:
            race_nums = {h.race_number for h in rec.horses}
            assert 1 not in race_nums

    def test_filters_invalid_odds(self):
        contexts = [
            _race_ctx(1, ranked=[
                {"saddlecloth": 1, "horse": "NoOdds", "win_prob": 0.30},
            ], runners=[
                {"saddlecloth": 1, "horse_name": "NoOdds", "current_odds": 0},
            ]),
            _race_ctx(2),
            _race_ctx(3),
        ]
        rec = calculate_pre_big3(contexts)
        if rec is not None:
            for h in rec.horses:
                assert h.win_odds > 1

    def test_ev_calculation(self):
        contexts = [_race_ctx(i) for i in range(1, 4)]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        expected_ev = rec.multi_prob * rec.multi_odds * POOL_TAKEOUT
        assert rec.expected_value == pytest.approx(expected_ev, abs=0.001)
        assert rec.estimated_return == pytest.approx(expected_ev * MULTI_STAKE, abs=0.01)

    def test_stake_is_multi_stake(self):
        contexts = [_race_ctx(i) for i in range(1, 4)]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        assert rec.stake == MULTI_STAKE

    def test_skips_races_without_race_number(self):
        contexts = [
            {"probabilities": {"probability_ranked": []}},  # no race_number
            _race_ctx(2),
            _race_ctx(3),
            _race_ctx(4),
        ]
        rec = calculate_pre_big3(contexts)
        assert rec is not None
        assert all(h.race_number in (2, 3, 4) for h in rec.horses)

    def test_handles_string_win_prob(self):
        """Win probs expressed as percentage strings should be converted."""
        contexts = [
            _race_ctx(1, ranked=[
                {"saddlecloth": 1, "horse": "StrProb", "win_prob": "30%"},
                {"saddlecloth": 2, "horse": "Another", "win_prob": "20%"},
            ]),
            _race_ctx(2),
            _race_ctx(3),
        ]
        rec = calculate_pre_big3(contexts)
        assert rec is not None


# ──────────────────────────────────────────────
# Tests: format_pre_big3
# ──────────────────────────────────────────────

class TestFormatPreBig3:
    """Tests for the format_pre_big3 function."""

    def test_returns_empty_for_none(self):
        assert format_pre_big3(None) == ""

    def test_contains_horse_names(self):
        rec = Big3Recommendation(
            horses=[
                Big3Candidate(1, 1, "Alpha", 0.35, 3.0, 1, "Fast horse"),
                Big3Candidate(2, 3, "Beta", 0.25, 4.0, 1, "Good form"),
                Big3Candidate(3, 5, "Gamma", 0.20, 5.0, 1, "Value pick"),
            ],
            multi_prob=0.35 * 0.25 * 0.20,
            multi_odds=3.0 * 4.0 * 5.0,
            expected_value=0.35 * 0.25 * 0.20 * 60.0 * POOL_TAKEOUT,
            stake=MULTI_STAKE,
            estimated_return=0.35 * 0.25 * 0.20 * 60.0 * POOL_TAKEOUT * MULTI_STAKE,
        )
        text = format_pre_big3(rec)
        assert "Alpha" in text
        assert "Beta" in text
        assert "Gamma" in text

    def test_contains_ev_info(self):
        rec = Big3Recommendation(
            horses=[
                Big3Candidate(1, 1, "A", 0.30, 3.0, 1, ""),
                Big3Candidate(2, 2, "B", 0.25, 4.0, 1, ""),
                Big3Candidate(3, 3, "C", 0.20, 5.0, 1, ""),
            ],
            multi_prob=0.015,
            multi_odds=60.0,
            expected_value=0.765,
            stake=MULTI_STAKE,
            estimated_return=7.65,
        )
        text = format_pre_big3(rec)
        assert "EV" in text
        assert "Multi probability" in text

    def test_strong_ev_flag(self):
        rec = Big3Recommendation(
            horses=[
                Big3Candidate(1, 1, "A", 0.35, 2.5, 1, ""),
                Big3Candidate(2, 2, "B", 0.30, 3.0, 1, ""),
                Big3Candidate(3, 3, "C", 0.25, 4.0, 1, ""),
            ],
            multi_prob=0.02625,
            multi_odds=30.0,
            expected_value=1.3,
            stake=MULTI_STAKE,
            estimated_return=13.0,
        )
        text = format_pre_big3(rec)
        assert "Strong positive EV" in text

    def test_below_value_warning(self):
        rec = Big3Recommendation(
            horses=[
                Big3Candidate(1, 1, "A", 0.20, 3.0, 1, ""),
                Big3Candidate(2, 2, "B", 0.15, 4.0, 1, ""),
                Big3Candidate(3, 3, "C", 0.15, 5.0, 1, ""),
            ],
            multi_prob=0.0045,
            multi_odds=60.0,
            expected_value=0.23,
            stake=MULTI_STAKE,
            estimated_return=2.3,
        )
        text = format_pre_big3(rec)
        assert "WARNING" in text

    def test_includes_reasons(self):
        rec = Big3Recommendation(
            horses=[
                Big3Candidate(1, 1, "A", 0.30, 3.0, 1, "Great form"),
                Big3Candidate(2, 2, "B", 0.25, 4.0, 1, ""),
                Big3Candidate(3, 3, "C", 0.20, 5.0, 1, "Inside draw"),
            ],
            multi_prob=0.015,
            multi_odds=60.0,
            expected_value=0.77,
            stake=MULTI_STAKE,
            estimated_return=7.7,
        )
        text = format_pre_big3(rec)
        assert "Great form" in text
        assert "Inside draw" in text

    def test_strong_flag_on_high_prob_horse(self):
        rec = Big3Recommendation(
            horses=[
                Big3Candidate(1, 1, "Strong", 0.35, 2.5, 1, ""),
                Big3Candidate(2, 2, "Solid", 0.22, 4.0, 1, ""),
                Big3Candidate(3, 3, "Other", 0.18, 5.0, 1, ""),
            ],
            multi_prob=0.01386,
            multi_odds=50.0,
            expected_value=0.59,
            stake=MULTI_STAKE,
            estimated_return=5.9,
        )
        text = format_pre_big3(rec)
        assert "[STRONG]" in text
        assert "[SOLID]" in text
