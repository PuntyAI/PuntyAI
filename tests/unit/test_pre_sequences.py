"""Tests for smart sequence lane construction (single optimised ticket)."""

import pytest

from punty.context.pre_sequences import (
    MIN_OUTLAY,
    MAX_OUTLAY,
    MIN_FLEXI_PCT,
    CHAOS_SHAPES,
    BANKER_SHAPES,
    LaneLeg,
    SmartSequence,
    SequenceBlock,
    SequenceLane,
    build_smart_sequence,
    build_all_sequence_lanes,
    format_sequence_lanes,
    _build_reason,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _leg_analysis(
    race_number: int,
    confidence: str = "MED",
    width: int = 3,
    odds_shape: str = "CLEAR_FAV",
    shape_width: int = 5,
) -> dict:
    """Build a mock leg analysis dict."""
    runners = [
        {
            "saddlecloth": i,
            "horse_name": f"Horse{race_number}_{i}",
            "win_prob": 0.30 - i * 0.04,
        }
        for i in range(1, 9)
    ]
    return {
        "race_number": race_number,
        "confidence": confidence,
        "suggested_width": width,
        "top_runners": runners,
        "odds_shape": odds_shape,
        "shape_width": shape_width,
    }


def _race_ctx(race_number: int) -> dict:
    """Build a mock race context dict."""
    return {
        "race_number": race_number,
        "runners": [
            {
                "saddlecloth": i,
                "horse_name": f"Horse{race_number}_{i}",
                "_win_prob_raw": 0.25 - i * 0.04,
                "scratched": False,
            }
            for i in range(1, 8)
        ],
        "probabilities": {
            "probability_ranked": [
                {
                    "horse": f"Horse{race_number}_{i}",
                    "win_prob": f"{(25 - i * 4):.1f}%",
                    "saddlecloth": i,
                }
                for i in range(1, 8)
            ],
        },
    }


# ──────────────────────────────────────────────
# Tests: build_smart_sequence
# ──────────────────────────────────────────────

class TestBuildSmartSequence:
    def test_quaddie_basic(self):
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.sequence_type == "Quaddie"
        assert result.race_start == 5
        assert result.race_end == 8
        assert len(result.legs) == 4
        assert MIN_OUTLAY <= result.total_outlay <= MAX_OUTLAY

    def test_big6_has_6_legs(self):
        leg_analysis = [_leg_analysis(r) for r in range(3, 9)]
        race_contexts = [_race_ctx(r) for r in range(3, 9)]

        result = build_smart_sequence("Big 6", (3, 8), leg_analysis, race_contexts)
        assert result is not None
        assert len(result.legs) == 6

    def test_missing_leg_returns_none(self):
        """Missing leg analysis for a race should return None."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 8)]  # only 3 legs
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is None

    def test_combos_product_of_leg_widths(self):
        """Total combos = product of runners per leg."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        expected = 1
        for leg in result.legs:
            expected *= len(leg.runners)
        assert result.total_combos == expected

    def test_flexi_pct_calculated_correctly(self):
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        expected_flexi = round(result.total_outlay / result.total_combos * 100, 1)
        assert result.flexi_pct == expected_flexi

    def test_standout_shape_uses_3_runners(self):
        """STANDOUT shape should produce 3 runners per leg."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="STANDOUT", shape_width=3)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        # 3^4 = 81 combos, well under budget, so no constraining
        assert result.total_combos == 81
        for leg in result.legs:
            assert len(leg.runners) == 3

    def test_mixed_shapes(self):
        """Mixed shapes should produce different widths per leg."""
        leg_analysis = [
            _leg_analysis(5, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(6, odds_shape="CLEAR_FAV", shape_width=5),
            _leg_analysis(7, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(8, odds_shape="TRIO", shape_width=7),
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        widths = [len(leg.runners) for leg in result.legs]
        # Standout legs should be tighter than open legs
        assert widths[0] <= widths[1]  # STANDOUT <= CLEAR_FAV
        assert widths[2] <= widths[3]  # STANDOUT <= TRIO

    def test_flexi_floor_maintained(self):
        """Total combos should not exceed max for 30% flexi floor."""
        max_combos = int(MAX_OUTLAY / (MIN_FLEXI_PCT / 100.0))  # max possible
        leg_analysis = [
            _leg_analysis(r, odds_shape="TRIO", shape_width=7)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.total_combos <= max_combos

    def test_risk_note_for_open_legs(self):
        """Many open legs should produce a risk warning."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="OPEN_BUNCH", shape_width=6)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert "RISKY" in result.risk_note or "risk" in result.risk_note.lower()

    def test_legs_have_odds_shape(self):
        """Each leg should carry its odds shape classification."""
        leg_analysis = [
            _leg_analysis(5, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(6, odds_shape="CLEAR_FAV", shape_width=5),
            _leg_analysis(7, odds_shape="TRIO", shape_width=7),
            _leg_analysis(8, odds_shape="DOMINANT", shape_width=4),
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.legs[0].odds_shape == "STANDOUT"
        assert result.legs[1].odds_shape == "CLEAR_FAV"
        assert result.legs[2].odds_shape == "TRIO"
        assert result.legs[3].odds_shape == "DOMINANT"

    def test_runner_names_populated(self):
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        for leg in result.legs:
            assert len(leg.runner_names) == len(leg.runners)
            assert all(name for name in leg.runner_names)

    def test_hit_probability_populated(self):
        """Hit probability should be calculated."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.hit_probability > 0

    def test_chaos_ratio_all_banker(self):
        """All banker legs should have chaos_ratio = 0."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="STANDOUT", shape_width=3)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.chaos_ratio == 0.0

    def test_chaos_ratio_all_chaos(self):
        """All chaos legs should have chaos_ratio = 1."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="OPEN_BUNCH", shape_width=6)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.chaos_ratio == 1.0

    def test_outlay_scales_with_chaos(self):
        """More chaos legs → higher outlay."""
        banker_legs = [
            _leg_analysis(r, odds_shape="STANDOUT", shape_width=3)
            for r in range(5, 9)
        ]
        chaos_legs = [
            _leg_analysis(r, odds_shape="OPEN_BUNCH", shape_width=6)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        banker_result = build_smart_sequence("Quaddie", (5, 8), banker_legs, race_contexts)
        chaos_result = build_smart_sequence("Quaddie", (5, 8), chaos_legs, race_contexts)

        assert banker_result is not None
        assert chaos_result is not None
        assert chaos_result.total_outlay > banker_result.total_outlay

    def test_estimated_return_positive(self):
        """Estimated return should be positive for reasonable sequences."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.estimated_return_pct > 0


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
        assert "Early Quaddie" in types

    def test_empty_analysis(self):
        results = build_all_sequence_lanes(8, [], [])
        assert results == []

    def test_blocks_have_smart_field(self):
        """Each SequenceBlock should contain a SmartSequence."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        for block in results:
            assert block.smart is not None
            assert isinstance(block.smart, SmartSequence)

    def test_legacy_balanced_lane_populated(self):
        """Legacy SequenceBlock.balanced should be populated from smart ticket."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        for block in results:
            assert block.balanced is not None
            assert block.balanced.variant == "Smart"

    def test_recommended_is_smart(self):
        """Recommended field should be 'Smart'."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        for block in results:
            assert block.recommended == "Smart"


# ──────────────────────────────────────────────
# Tests: format_sequence_lanes
# ──────────────────────────────────────────────

class TestFormatSequenceLanes:
    def test_basic_format(self):
        leg_analysis = [_leg_analysis(r) for r in range(1, 5)]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)

        assert "PRE-BUILT SEQUENCE BETS" in formatted
        assert "Smart" in formatted
        assert "combos" in formatted
        assert "flexi" in formatted

    def test_empty_returns_empty(self):
        assert format_sequence_lanes([]) == ""

    def test_contains_saddlecloth_numbers(self):
        leg_analysis = [_leg_analysis(r) for r in range(1, 5)]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "1" in formatted

    def test_contains_odds_shape(self):
        """Format should show odds shape per leg."""
        leg_analysis = [
            _leg_analysis(1, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(2, odds_shape="CLEAR_FAV", shape_width=5),
            _leg_analysis(3, odds_shape="TRIO", shape_width=7),
            _leg_analysis(4, odds_shape="DOMINANT", shape_width=4),
        ]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "STANDOUT" in formatted
        assert "CLEAR_FAV" in formatted
        assert "TRIO" in formatted
        assert "DOMINANT" in formatted

    def test_risk_warning_shown(self):
        """Risk warning should appear in formatted output."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="OPEN_BUNCH", shape_width=6)
            for r in range(1, 5)
        ]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "WARNING" in formatted

    def test_single_ticket_line(self):
        """Format should include exactly one Smart: line per sequence."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 5)]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        # Should NOT have Ticket A/B/C
        assert "Ticket A" not in formatted
        assert "Ticket B" not in formatted
        assert "Ticket C" not in formatted
        # Should have Smart: line
        assert "Smart:" in formatted

    def test_metrics_line_shown(self):
        """Format should show hit probability and estimated return metrics."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 5)]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "Hit prob" in formatted
        assert "Est return" in formatted
        assert "Chaos ratio" in formatted

    def test_chaos_marker_shown(self):
        """Chaos legs should be marked [CHAOS] in format."""
        leg_analysis = [
            _leg_analysis(1, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(2, odds_shape="TRIO", shape_width=7),
            _leg_analysis(3, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(4, odds_shape="OPEN_BUNCH", shape_width=6),
        ]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "[CHAOS]" in formatted

    def test_banker_marker_shown(self):
        """Banker legs should be marked [BANKER] in format."""
        leg_analysis = [
            _leg_analysis(1, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(2, odds_shape="CLEAR_FAV", shape_width=5),
            _leg_analysis(3, odds_shape="DOMINANT", shape_width=4),
            _leg_analysis(4, odds_shape="TRIO", shape_width=7),
        ]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "[BANKER]" in formatted


# ──────────────────────────────────────────────
# Tests: _build_reason
# ──────────────────────────────────────────────

class TestBuildReason:
    def test_includes_width_and_return(self):
        smart = SmartSequence(
            sequence_type="Quaddie",
            race_start=5,
            race_end=8,
            legs=[
                LaneLeg(5, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                LaneLeg(6, [1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"], "CLEAR_FAV", 5),
                LaneLeg(7, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                LaneLeg(8, [1, 2, 3, 4], ["A", "B", "C", "D"], "DOMINANT", 4),
            ],
            total_combos=180,
            flexi_pct=16.7,
            total_outlay=30.0,
            constrained=False,
            risk_note="",
            estimated_return_pct=45.0,
            hit_probability=12.5,
            chaos_ratio=0.0,
        )
        reason = _build_reason(smart)
        assert "3x5x3x4" in reason
        assert "180 combos" in reason
        assert "Hit prob" in reason
        assert "Est return" in reason

    def test_chaos_heavy_mentioned(self):
        smart = SmartSequence(
            sequence_type="Quaddie",
            race_start=5,
            race_end=8,
            legs=[
                LaneLeg(5, [1, 2], ["A", "B"], "TRIO", 7),
                LaneLeg(6, [1, 2], ["A", "B"], "OPEN_BUNCH", 6),
                LaneLeg(7, [1, 2], ["A", "B"], "TRIO", 7),
                LaneLeg(8, [1, 2], ["A", "B"], "STANDOUT", 3),
            ],
            total_combos=16,
            flexi_pct=187.5,
            total_outlay=30.0,
            constrained=False,
            risk_note="",
            estimated_return_pct=35.0,
            hit_probability=8.0,
            chaos_ratio=0.75,
        )
        reason = _build_reason(smart)
        assert "chaos" in reason.lower()
