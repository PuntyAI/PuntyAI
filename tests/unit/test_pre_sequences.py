"""Tests for algorithmic sequence lane construction."""

import pytest

from punty.context.pre_sequences import (
    BALANCED_OUTLAY,
    MAX_COMBOS_SKINNY,
    SKINNY_OUTLAY,
    WIDE_OUTLAY,
    LaneLeg,
    SequenceBlock,
    SequenceLane,
    _build_lane,
    _recommend_variant,
    _trim_to_combo_limit,
    _width_for_variant,
    build_all_sequence_lanes,
    build_sequence_lanes,
    format_sequence_lanes,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _leg_analysis(race_number: int, confidence: str = "MED", width: int = 2) -> dict:
    """Build a mock leg analysis dict."""
    runners = [
        {"saddlecloth": i, "horse_name": f"Horse{race_number}_{i}", "win_prob": 0.25 - i * 0.04}
        for i in range(1, 6)
    ]
    return {
        "race_number": race_number,
        "confidence": confidence,
        "suggested_width": width,
        "top_runners": runners,
    }


def _race_ctx(race_number: int) -> dict:
    """Build a mock race context dict."""
    return {
        "race_number": race_number,
        "runners": [
            {"saddlecloth": i, "horse_name": f"Horse{race_number}_{i}",
             "_win_prob_raw": 0.25 - i * 0.04, "scratched": False}
            for i in range(1, 8)
        ],
        "probabilities": {
            "probability_ranked": [
                {"horse": f"Horse{race_number}_{i}", "win_prob": f"{(25 - i * 4):.1f}%",
                 "saddlecloth": i}
                for i in range(1, 8)
            ],
        },
    }


# ──────────────────────────────────────────────
# Tests: _width_for_variant
# ──────────────────────────────────────────────

class TestWidthForVariant:
    def test_skinny_high_confidence(self):
        assert _width_for_variant("Skinny", "HIGH", 1) == 1

    def test_skinny_med_confidence(self):
        assert _width_for_variant("Skinny", "MED", 2) == 2

    def test_skinny_low_confidence(self):
        assert _width_for_variant("Skinny", "LOW", 3) == 2  # capped at 2

    def test_balanced_high_confidence(self):
        assert _width_for_variant("Balanced", "HIGH", 1) == 2

    def test_balanced_med_confidence(self):
        assert _width_for_variant("Balanced", "MED", 2) == 3

    def test_balanced_low_confidence(self):
        assert _width_for_variant("Balanced", "LOW", 3) == 3  # min(3, 3+1)

    def test_wide_high_confidence(self):
        assert _width_for_variant("Wide", "HIGH", 1) == 3

    def test_wide_med_confidence(self):
        assert _width_for_variant("Wide", "MED", 2) == 4

    def test_wide_low_confidence(self):
        assert _width_for_variant("Wide", "LOW", 3) == 4


# ──────────────────────────────────────────────
# Tests: _build_lane
# ──────────────────────────────────────────────

class TestBuildLane:
    def test_skinny_lane(self):
        legs = [
            _leg_analysis(5, "HIGH", 1),
            _leg_analysis(6, "MED", 2),
            _leg_analysis(7, "MED", 2),
            _leg_analysis(8, "LOW", 3),
        ]
        lane = _build_lane("Skinny", legs, SKINNY_OUTLAY)
        assert lane.variant == "Skinny"
        assert lane.total_outlay == SKINNY_OUTLAY
        assert lane.total_combos >= 1
        assert abs(lane.unit_price - SKINNY_OUTLAY / lane.total_combos) < 0.01

    def test_combos_multiplication(self):
        """Combos = product of runners per leg."""
        legs = [
            _leg_analysis(1, "HIGH"),
            _leg_analysis(2, "HIGH"),
            _leg_analysis(3, "HIGH"),
            _leg_analysis(4, "HIGH"),
        ]
        lane = _build_lane("Skinny", legs, SKINNY_OUTLAY)
        # All HIGH → 1 runner per leg → 1*1*1*1 = 1 combo
        assert lane.total_combos == 1

    def test_wide_lane_has_more_combos(self):
        legs = [
            _leg_analysis(1, "MED", 2),
            _leg_analysis(2, "MED", 2),
            _leg_analysis(3, "MED", 2),
            _leg_analysis(4, "MED", 2),
        ]
        skinny = _build_lane("Skinny", legs, SKINNY_OUTLAY)
        wide = _build_lane("Wide", legs, WIDE_OUTLAY)
        assert wide.total_combos >= skinny.total_combos

    def test_flexi_pct_is_unit_times_100(self):
        legs = [_leg_analysis(1, "MED", 2)] * 4
        lane = _build_lane("Balanced", legs, BALANCED_OUTLAY)
        assert abs(lane.flexi_pct - lane.unit_price * 100) < 0.1


# ──────────────────────────────────────────────
# Tests: _trim_to_combo_limit
# ──────────────────────────────────────────────

class TestTrimToComboLimit:
    def test_trims_widest_leg(self):
        legs = [
            LaneLeg(1, [1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"], "LOW"),
            LaneLeg(2, [1, 2], ["A", "B"], "HIGH"),
        ]
        # 5 * 2 = 10 combos — if limit is 8, should trim leg 1
        trimmed = _trim_to_combo_limit(legs, 8)
        total = 1
        for leg in trimmed:
            total *= len(leg.runners)
        assert total <= 8

    def test_no_trim_when_under_limit(self):
        legs = [
            LaneLeg(1, [1, 2], ["A", "B"], "MED"),
            LaneLeg(2, [1, 2], ["A", "B"], "MED"),
        ]
        trimmed = _trim_to_combo_limit(legs, 10)
        assert len(trimmed[0].runners) == 2
        assert len(trimmed[1].runners) == 2


# ──────────────────────────────────────────────
# Tests: _recommend_variant
# ──────────────────────────────────────────────

class TestRecommendVariant:
    def test_all_high_recommends_skinny(self):
        legs = [{"confidence": "HIGH"}] * 4
        variant, reason = _recommend_variant(legs)
        assert variant == "Skinny"

    def test_mixed_recommends_balanced(self):
        legs = [
            {"confidence": "HIGH"},
            {"confidence": "HIGH"},
            {"confidence": "MED"},
            {"confidence": "LOW"},
        ]
        variant, reason = _recommend_variant(legs)
        assert variant == "Balanced"

    def test_mostly_low_recommends_wide(self):
        legs = [
            {"confidence": "LOW"},
            {"confidence": "LOW"},
            {"confidence": "LOW"},
            {"confidence": "MED"},
        ]
        variant, reason = _recommend_variant(legs)
        assert variant == "Wide"


# ──────────────────────────────────────────────
# Tests: build_sequence_lanes
# ──────────────────────────────────────────────

class TestBuildSequenceLanes:
    def test_quaddie_basic(self):
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_sequence_lanes("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.sequence_type == "Quaddie"
        assert result.race_start == 5
        assert result.race_end == 8
        assert len(result.skinny.legs) == 4
        assert len(result.balanced.legs) == 4
        assert len(result.wide.legs) == 4

    def test_big6_has_6_legs(self):
        leg_analysis = [_leg_analysis(r) for r in range(3, 9)]
        race_contexts = [_race_ctx(r) for r in range(3, 9)]

        result = build_sequence_lanes("Big 6", (3, 8), leg_analysis, race_contexts)
        assert result is not None
        assert len(result.skinny.legs) == 6

    def test_missing_leg_returns_none(self):
        """Missing leg analysis for a race should return None."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 8)]  # only 3 legs for 4-leg quaddie
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_sequence_lanes("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is None

    def test_skinny_outlay_correct(self):
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_sequence_lanes("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result.skinny.total_outlay == SKINNY_OUTLAY
        assert result.balanced.total_outlay == BALANCED_OUTLAY
        assert result.wide.total_outlay == WIDE_OUTLAY

    def test_unit_price_correctness(self):
        leg_analysis = [_leg_analysis(r, "HIGH") for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_sequence_lanes("Quaddie", (5, 8), leg_analysis, race_contexts)
        # All HIGH → Skinny has 1 runner per leg → 1 combo → unit = $10
        assert result.skinny.total_combos == 1
        assert result.skinny.unit_price == SKINNY_OUTLAY


# ──────────────────────────────────────────────
# Tests: build_all_sequence_lanes
# ──────────────────────────────────────────────

class TestBuildAllSequenceLanes:
    def test_8_race_meeting(self):
        """8-race meeting should produce Early Quaddie, Quaddie, and Big 6."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        types = {r.sequence_type for r in results}
        assert "Early Quaddie" in types
        assert "Quaddie" in types
        assert "Big 6" in types

    def test_7_race_meeting_no_big6(self):
        """7-race meeting should NOT have Big 6."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 8)]
        race_contexts = [_race_ctx(r) for r in range(1, 8)]

        results = build_all_sequence_lanes(7, leg_analysis, race_contexts)
        types = {r.sequence_type for r in results}
        assert "Big 6" not in types
        assert "Quaddie" in types

    def test_empty_analysis(self):
        results = build_all_sequence_lanes(8, [], [])
        assert results == []


# ──────────────────────────────────────────────
# Tests: format_sequence_lanes
# ──────────────────────────────────────────────

class TestFormatSequenceLanes:
    def test_basic_format(self):
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        block = build_sequence_lanes("Quaddie", (5, 8), leg_analysis, race_contexts)
        formatted = format_sequence_lanes([block])

        assert "PRE-BUILT SEQUENCE LANES" in formatted
        assert "Quaddie" in formatted
        assert "Skinny" in formatted
        assert "Balanced" in formatted
        assert "Wide" in formatted
        assert "combos" in formatted
        assert "Recommended:" in formatted

    def test_empty_returns_empty(self):
        assert format_sequence_lanes([]) == ""

    def test_contains_saddlecloth_numbers(self):
        leg_analysis = [_leg_analysis(r, "HIGH") for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        block = build_sequence_lanes("Quaddie", (5, 8), leg_analysis, race_contexts)
        formatted = format_sequence_lanes([block])

        # Should contain saddlecloth numbers (integers) separated by commas
        assert "1" in formatted  # at minimum saddlecloth 1 should appear
