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
    _is_chaos_leg,
    _is_anchor_leg,
    _optimiser_select,
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
    top_prob: float = 0.35,
    edge_base: float = 0.06,
) -> dict:
    """Build a mock leg analysis dict with realistic odds and edge."""
    runners = []
    for i in range(1, 9):
        wp = top_prob - (i - 1) * 0.04
        if wp < 0.02:
            wp = 0.02
        edge = edge_base - (i - 1) * 0.02
        implied_odds = 1.0 / wp if wp > 0 else 50.0
        runners.append({
            "saddlecloth": i,
            "horse_name": f"Horse{race_number}_{i}",
            "win_prob": wp,
            "edge": edge,
            "current_odds": round(implied_odds * 0.9, 2),  # slight overlay
        })
    return {
        "race_number": race_number,
        "confidence": confidence,
        "suggested_width": width,
        "top_runners": runners,
        "odds_shape": odds_shape,
        "shape_width": shape_width,
    }


def _race_ctx(race_number: int, num_runners: int = 7) -> dict:
    """Build a mock race context dict."""
    return {
        "race_number": race_number,
        "runners": [
            {
                "saddlecloth": i,
                "horse_name": f"Horse{race_number}_{i}",
                "_win_prob_raw": 0.25 - i * 0.04,
                "scratched": False,
                "current_odds": round(1.0 / max(0.25 - i * 0.04, 0.02) * 0.9, 2),
                "punty_value_rating": 1.0 + (0.05 if i <= 3 else -0.05),
            }
            for i in range(1, num_runners + 1)
        ],
        "probabilities": {
            "probability_ranked": [
                {
                    "horse": f"Horse{race_number}_{i}",
                    "win_prob": f"{(25 - i * 4):.1f}%",
                    "saddlecloth": i,
                }
                for i in range(1, num_runners + 1)
            ],
        },
    }


# ──────────────────────────────────────────────
# Tests: _is_chaos_leg / _is_anchor_leg
# ──────────────────────────────────────────────

class TestLegClassification:
    def test_chaos_large_field(self):
        """Field size >= 14 → chaos."""
        from punty.probability import SequenceLegAnalysis
        la = _leg_analysis(1, top_prob=0.30)
        leg = SequenceLegAnalysis(
            race_number=1,
            top_runners=la["top_runners"],
            leg_confidence="MED",
            suggested_width=3,
            odds_shape="CLEAR_FAV",
            shape_width=5,
        )
        leg._field_size = 14
        assert _is_chaos_leg(leg)

    def test_chaos_high_fav_odds(self):
        """Fav odds >= 5.0 → chaos."""
        from punty.probability import SequenceLegAnalysis
        # All runners have low probs → high implied odds
        runners = [
            {"saddlecloth": i, "horse_name": f"H{i}", "win_prob": 0.10,
             "edge": 0.02, "current_odds": 8.0}
            for i in range(1, 9)
        ]
        leg = SequenceLegAnalysis(
            race_number=1,
            top_runners=runners,
            leg_confidence="LOW",
            suggested_width=5,
            odds_shape="OPEN_BUNCH",
            shape_width=6,
        )
        leg._field_size = 10
        assert _is_chaos_leg(leg)

    def test_chaos_low_top_prob(self):
        """p_top1 <= 0.22 → chaos."""
        from punty.probability import SequenceLegAnalysis
        runners = [
            {"saddlecloth": i, "horse_name": f"H{i}", "win_prob": 0.20 - i * 0.01,
             "edge": 0.02, "current_odds": 5.0}
            for i in range(1, 9)
        ]
        leg = SequenceLegAnalysis(
            race_number=1,
            top_runners=runners,
            leg_confidence="LOW",
            suggested_width=5,
            odds_shape="TRIO",
            shape_width=7,
        )
        leg._field_size = 10
        assert _is_chaos_leg(leg)

    def test_anchor_strong_fav(self):
        """Strong fav: p_top1>=0.30, fav_odds<=4.0, gap>=0.10 → anchor."""
        from punty.probability import SequenceLegAnalysis
        runners = [
            {"saddlecloth": 1, "horse_name": "H1", "win_prob": 0.40, "edge": 0.10, "current_odds": 2.50},
            {"saddlecloth": 2, "horse_name": "H2", "win_prob": 0.20, "edge": 0.02, "current_odds": 5.00},
            {"saddlecloth": 3, "horse_name": "H3", "win_prob": 0.12, "edge": 0.01, "current_odds": 8.00},
        ]
        for i in range(4, 9):
            runners.append({"saddlecloth": i, "horse_name": f"H{i}", "win_prob": 0.05, "edge": -0.02, "current_odds": 20.0})
        leg = SequenceLegAnalysis(
            race_number=1,
            top_runners=runners,
            leg_confidence="HIGH",
            suggested_width=2,
            odds_shape="STANDOUT",
            shape_width=3,
        )
        leg._field_size = 8
        assert _is_anchor_leg(leg)
        assert not _is_chaos_leg(leg)

    def test_not_anchor_when_close_probs(self):
        """Small gap between top two → not anchor."""
        from punty.probability import SequenceLegAnalysis
        runners = [
            {"saddlecloth": 1, "horse_name": "H1", "win_prob": 0.32, "edge": 0.05, "current_odds": 3.00},
            {"saddlecloth": 2, "horse_name": "H2", "win_prob": 0.28, "edge": 0.03, "current_odds": 3.50},
        ]
        for i in range(3, 9):
            runners.append({"saddlecloth": i, "horse_name": f"H{i}", "win_prob": 0.08, "edge": -0.01, "current_odds": 12.0})
        leg = SequenceLegAnalysis(
            race_number=1,
            top_runners=runners,
            leg_confidence="MED",
            suggested_width=3,
            odds_shape="CLEAR_FAV",
            shape_width=5,
        )
        leg._field_size = 8
        assert not _is_anchor_leg(leg)  # gap only 0.04


# ──────────────────────────────────────────────
# Tests: _optimiser_select
# ──────────────────────────────────────────────

class TestOptimiser:
    def _make_legs(self, specs):
        """Build legs_data from specs: list of (top_prob, edge_base, field_size, odds_shape)."""
        from punty.probability import SequenceLegAnalysis
        legs = []
        for i, (tp, eb, fs, shape) in enumerate(specs, start=5):
            la = _leg_analysis(i, top_prob=tp, edge_base=eb, odds_shape=shape)
            leg = SequenceLegAnalysis(
                race_number=i,
                top_runners=la["top_runners"],
                leg_confidence="MED",
                suggested_width=3,
                odds_shape=shape,
                shape_width=5,
            )
            leg._field_size = fs
            legs.append(leg)
        return legs

    def test_pass_when_no_anchor_no_strong(self):
        """No anchor leg and no strong runner → PASS (None)."""
        specs = [
            (0.18, 0.02, 12, "OPEN_BUNCH"),
            (0.15, 0.01, 14, "WIDE_OPEN"),
            (0.20, 0.03, 10, "TRIO"),
            (0.19, 0.02, 11, "MID_FAV"),
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is None

    def test_anchor_leg_singled(self):
        """Anchor leg should get 1 runner."""
        specs = [
            (0.40, 0.10, 8, "STANDOUT"),   # anchor
            (0.25, 0.05, 10, "CLEAR_FAV"),  # normal
            (0.22, 0.03, 10, "TRIO"),       # normal
            (0.28, 0.06, 9, "CLEAR_FAV"),   # normal
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is not None
        assert len(result[0]) == 1  # anchor singled

    def test_overlay_only_selection(self):
        """Prefer runners with positive edge."""
        specs = [
            (0.35, 0.08, 8, "STANDOUT"),   # anchor
            (0.25, 0.05, 10, "CLEAR_FAV"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is not None
        for i, sel in enumerate(result):
            for r in sel:
                # All selected should have non-negative edge
                assert float(r.get("edge", 0)) >= -0.01, \
                    f"Leg {i}: runner {r.get('saddlecloth')} has negative edge {r.get('edge')}"

    def test_field_size_cap_small(self):
        """Field 8-10 → max 3 selections."""
        specs = [
            (0.35, 0.08, 8, "STANDOUT"),
            (0.20, 0.06, 8, "TRIO"),     # chaos but field 8 → max 3
            (0.25, 0.05, 9, "CLEAR_FAV"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is not None
        for i, sel in enumerate(result):
            field = getattr(legs[i], "_field_size", 8)
            if field <= 10:
                assert len(sel) <= 3, f"Leg {i}: {len(sel)} selections for field {field}"

    def test_no_half_field(self):
        """Never select >= 50% of field."""
        specs = [
            (0.35, 0.08, 8, "STANDOUT"),
            (0.25, 0.05, 6, "CLEAR_FAV"),  # small field
            (0.25, 0.05, 6, "CLEAR_FAV"),
            (0.25, 0.05, 8, "CLEAR_FAV"),
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is not None
        for i, sel in enumerate(result):
            field = getattr(legs[i], "_field_size", 8)
            assert len(sel) < field / 2, \
                f"Leg {i}: {len(sel)} of {field} runners (>= 50%)"

    def test_budget_in_range(self):
        """Outlay between MIN and MAX."""
        specs = [
            (0.35, 0.08, 10, "STANDOUT"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
            (0.25, 0.05, 12, "CLEAR_FAV"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is not None
        combos = 1
        for sel in result:
            combos *= len(sel)
        cost = combos * (MIN_FLEXI_PCT / 100.0)
        assert cost <= MAX_OUTLAY, f"Cost {cost} exceeds MAX_OUTLAY {MAX_OUTLAY}"

    def test_at_least_one_single(self):
        """At least one leg has 1 selection."""
        specs = [
            (0.35, 0.08, 10, "STANDOUT"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
            (0.25, 0.05, 12, "CLEAR_FAV"),
            (0.25, 0.05, 10, "CLEAR_FAV"),
        ]
        legs = self._make_legs(specs)
        result = _optimiser_select(legs, budget=50.0)
        assert result is not None
        widths = [len(sel) for sel in result]
        assert 1 in widths, f"No single leg found: {widths}"


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

    def test_mixed_shapes(self):
        """Mixed shapes should produce different widths per leg."""
        leg_analysis = [
            _leg_analysis(5, odds_shape="STANDOUT", shape_width=3, top_prob=0.40, edge_base=0.10),
            _leg_analysis(6, odds_shape="CLEAR_FAV", shape_width=5, top_prob=0.25, edge_base=0.05),
            _leg_analysis(7, odds_shape="STANDOUT", shape_width=3, top_prob=0.35, edge_base=0.08),
            _leg_analysis(8, odds_shape="TRIO", shape_width=7, top_prob=0.20, edge_base=0.03),
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        widths = [len(leg.runners) for leg in result.legs]
        # Anchor legs (STANDOUT with high prob) should be tighter
        assert widths[0] <= widths[3] or widths[2] <= widths[3]

    def test_risk_note_for_open_legs(self):
        """Many open legs should produce a risk warning."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="OPEN_BUNCH", shape_width=6, top_prob=0.18, edge_base=0.02)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        # May return None (PASS) since all chaos with low probs
        # That's actually correct behaviour — no anchor, no strong runner
        if result is not None:
            assert "RISKY" in result.risk_note or "risk" in result.risk_note.lower()

    def test_legs_have_odds_shape(self):
        """Each leg should carry its odds shape classification."""
        leg_analysis = [
            _leg_analysis(5, odds_shape="STANDOUT", shape_width=3, top_prob=0.40, edge_base=0.10),
            _leg_analysis(6, odds_shape="CLEAR_FAV", shape_width=5, top_prob=0.25, edge_base=0.05),
            _leg_analysis(7, odds_shape="TRIO", shape_width=7, top_prob=0.25, edge_base=0.05),
            _leg_analysis(8, odds_shape="DOMINANT", shape_width=4, top_prob=0.35, edge_base=0.08),
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

    def test_chaos_ratio_calculated(self):
        """Chaos ratio should reflect leg classification."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_smart_sequence("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert 0.0 <= result.chaos_ratio <= 1.0

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
            _leg_analysis(1, odds_shape="STANDOUT", shape_width=3, top_prob=0.40, edge_base=0.10),
            _leg_analysis(2, odds_shape="CLEAR_FAV", shape_width=5, top_prob=0.25, edge_base=0.05),
            _leg_analysis(3, odds_shape="TRIO", shape_width=7, top_prob=0.25, edge_base=0.05),
            _leg_analysis(4, odds_shape="DOMINANT", shape_width=4, top_prob=0.35, edge_base=0.08),
        ]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "STANDOUT" in formatted
        assert "CLEAR_FAV" in formatted
        assert "TRIO" in formatted
        assert "DOMINANT" in formatted

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

    def test_anchor_marker_shown(self):
        """Anchor (singled) legs should be marked [ANCHOR] in format."""
        leg_analysis = [
            _leg_analysis(1, odds_shape="STANDOUT", shape_width=3, top_prob=0.40, edge_base=0.10),
            _leg_analysis(2, odds_shape="CLEAR_FAV", shape_width=5, top_prob=0.25, edge_base=0.05),
            _leg_analysis(3, odds_shape="DOMINANT", shape_width=4, top_prob=0.35, edge_base=0.08),
            _leg_analysis(4, odds_shape="CLEAR_FAV", shape_width=5, top_prob=0.25, edge_base=0.05),
        ]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        # At least one leg should have [ANCHOR] (the singled leg)
        assert "[ANCHOR]" in formatted


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
            total_outlay=40.0,
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
            total_outlay=40.0,
            constrained=False,
            risk_note="",
            estimated_return_pct=35.0,
            hit_probability=8.0,
            chaos_ratio=0.75,
        )
        reason = _build_reason(smart)
        assert "chaos" in reason.lower()
