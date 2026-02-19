"""Tests for smart sequence lane construction and ABC multi-ticket structure."""

import pytest

from punty.context.pre_sequences import (
    MIN_OUTLAY,
    MAX_OUTLAY,
    MIN_FLEXI_PCT,
    TICKET_A_PCT,
    TICKET_B_PCT,
    TICKET_C_PCT,
    TICKET_C_MIN_FLEXI_PCT,
    CHAOS_SHAPES,
    LaneLeg,
    SmartSequence,
    ABCTicket,
    ABCSequence,
    SequenceBlock,
    SequenceLane,
    build_smart_sequence,
    build_abc_tickets,
    build_all_sequence_lanes,
    format_sequence_lanes,
    _build_reason_abc,
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
# Tests: build_smart_sequence (legacy)
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
        # 3^4 = 81 combos, well under 333, so no constraining
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


# ──────────────────────────────────────────────
# Tests: build_abc_tickets
# ──────────────────────────────────────────────

class TestBuildABCTickets:
    def test_returns_3_tickets(self):
        """ABC should produce exactly 3 tickets."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert len(result.tickets) == 3
        labels = [t.ticket_label for t in result.tickets]
        assert labels == ["A", "B", "C"]

    def test_ticket_descriptions(self):
        """Tickets should have correct descriptions."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.tickets[0].description == "Banker Press"
        assert result.tickets[1].description == "Value Spread"
        assert result.tickets[2].description == "Chaos Saver"

    def test_ticket_a_has_2_runners_per_leg(self):
        """Ticket A (Banker) should have exactly 2 runners per leg."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        for leg in result.tickets[0].legs:
            assert len(leg.runners) == 2

    def test_ticket_a_combos_are_power_of_2(self):
        """Ticket A with 2 runners/leg should have 2^N combos."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert result.tickets[0].total_combos == 2**4  # 16

    def test_budget_split_correct(self):
        """Budget should split roughly 50/30/20."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        total = result.total_outlay
        # Allow rounding tolerance
        assert abs(result.tickets[0].total_outlay - total * TICKET_A_PCT) <= 1
        assert abs(result.tickets[1].total_outlay - total * TICKET_B_PCT) <= 1
        budget_sum = sum(t.total_outlay for t in result.tickets)
        assert abs(budget_sum - total) < 0.01

    def test_ticket_b_wider_than_a(self):
        """Ticket B should generally have wider legs than Ticket A."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        a_combos = result.tickets[0].total_combos
        b_combos = result.tickets[1].total_combos
        assert b_combos >= a_combos

    def test_ticket_c_expands_chaos_legs(self):
        """Ticket C should have wider chaos legs than Ticket B."""
        leg_analysis = [
            _leg_analysis(5, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(6, odds_shape="OPEN_BUNCH", shape_width=6),
            _leg_analysis(7, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(8, odds_shape="TRIO", shape_width=7),
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        # Chaos legs are R6 (OPEN_BUNCH) and R8 (TRIO)
        assert 6 in result.chaos_legs
        assert 8 in result.chaos_legs
        # Ticket C chaos legs should be >= Ticket B chaos legs
        b_legs = result.tickets[1].legs
        c_legs = result.tickets[2].legs
        for i in range(4):
            b_width = len(b_legs[i].runners)
            c_width = len(c_legs[i].runners)
            assert c_width >= b_width

    def test_chaos_legs_identified(self):
        """Chaos legs should be those with CHAOS_SHAPES."""
        leg_analysis = [
            _leg_analysis(5, odds_shape="STANDOUT", shape_width=3),
            _leg_analysis(6, odds_shape="TRIO", shape_width=7),
            _leg_analysis(7, odds_shape="DOMINANT", shape_width=4),
            _leg_analysis(8, odds_shape="WIDE_OPEN", shape_width=6),
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert set(result.chaos_legs) == {6, 8}

    def test_missing_data_returns_none(self):
        """Missing leg data should return None."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 8)]  # only 3
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is None

    def test_outlay_within_bounds(self):
        """Total outlay should be within MIN-MAX range."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert MIN_OUTLAY <= result.total_outlay <= MAX_OUTLAY

    def test_big6_produces_6_legs(self):
        """Big 6 ABC should have 6 legs per ticket."""
        leg_analysis = [_leg_analysis(r) for r in range(3, 9)]
        race_contexts = [_race_ctx(r) for r in range(3, 9)]

        result = build_abc_tickets("Big 6", (3, 8), leg_analysis, race_contexts)
        assert result is not None
        for ticket in result.tickets:
            assert len(ticket.legs) == 6

    def test_risk_note_for_all_chaos(self):
        """All chaos legs should produce risk warning."""
        leg_analysis = [
            _leg_analysis(r, odds_shape="OPEN_BUNCH", shape_width=6)
            for r in range(5, 9)
        ]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
        assert result is not None
        assert "RISKY" in result.risk_note or "risk" in result.risk_note.lower()

    def test_estimated_return_positive(self):
        """Estimated return should be positive for reasonable sequences."""
        leg_analysis = [_leg_analysis(r) for r in range(5, 9)]
        race_contexts = [_race_ctx(r) for r in range(5, 9)]

        result = build_abc_tickets("Quaddie", (5, 8), leg_analysis, race_contexts)
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

    def test_blocks_have_abc_field(self):
        """Each SequenceBlock should contain an ABCSequence."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        for block in results:
            assert block.abc is not None
            assert isinstance(block.abc, ABCSequence)

    def test_legacy_balanced_lane_populated(self):
        """Legacy SequenceBlock.balanced should be populated from Ticket B."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        for block in results:
            assert block.balanced is not None
            assert block.balanced.variant == "Smart"

    def test_recommended_is_abc(self):
        """Recommended field should be 'ABC'."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 9)]
        race_contexts = [_race_ctx(r) for r in range(1, 9)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        for block in results:
            assert block.recommended == "ABC"


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
        assert "ABC" in formatted or "Ticket" in formatted
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

    def test_three_ticket_lines(self):
        """Format should include Ticket A, B, C lines."""
        leg_analysis = [_leg_analysis(r) for r in range(1, 5)]
        race_contexts = [_race_ctx(r) for r in range(1, 5)]

        results = build_all_sequence_lanes(8, leg_analysis, race_contexts)
        formatted = format_sequence_lanes(results)
        assert "Ticket A" in formatted
        assert "Ticket B" in formatted
        assert "Ticket C" in formatted

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


# ──────────────────────────────────────────────
# Tests: _build_reason_abc
# ──────────────────────────────────────────────

class TestBuildReasonABC:
    def test_includes_width_and_total(self):
        abc = ABCSequence(
            sequence_type="Quaddie",
            race_start=5,
            race_end=8,
            tickets=[
                ABCTicket("A", [
                    LaneLeg(5, [1, 2], ["A", "B"], "STANDOUT", 3),
                    LaneLeg(6, [1, 2], ["A", "B"], "CLEAR_FAV", 5),
                    LaneLeg(7, [1, 2], ["A", "B"], "STANDOUT", 3),
                    LaneLeg(8, [1, 2], ["A", "B"], "DOMINANT", 4),
                ], 16, 93.8, 15.0, "Banker Press"),
                ABCTicket("B", [
                    LaneLeg(5, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                    LaneLeg(6, [1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"], "CLEAR_FAV", 5),
                    LaneLeg(7, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                    LaneLeg(8, [1, 2, 3, 4], ["A", "B", "C", "D"], "DOMINANT", 4),
                ], 180, 5.0, 9.0, "Value Spread"),
                ABCTicket("C", [
                    LaneLeg(5, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                    LaneLeg(6, [1, 2, 3, 4, 5, 6], ["A", "B", "C", "D", "E", "F"], "CLEAR_FAV", 5),
                    LaneLeg(7, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                    LaneLeg(8, [1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"], "DOMINANT", 4),
                ], 270, 2.2, 6.0, "Chaos Saver"),
            ],
            total_outlay=30.0,
            risk_note="",
            estimated_return_pct=45.0,
            chaos_legs=[],
        )
        reason = _build_reason_abc(abc)
        assert "3x5x3x4" in reason
        assert "180 combos" in reason
        assert "$30" in reason

    def test_chaos_legs_mentioned(self):
        abc = ABCSequence(
            sequence_type="Quaddie",
            race_start=5,
            race_end=8,
            tickets=[
                ABCTicket("A", [], 16, 93.8, 15.0, "Banker Press"),
                ABCTicket("B", [
                    LaneLeg(5, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                    LaneLeg(6, [1, 2, 3], ["A", "B", "C"], "TRIO", 7),
                    LaneLeg(7, [1, 2, 3], ["A", "B", "C"], "STANDOUT", 3),
                    LaneLeg(8, [1, 2, 3], ["A", "B", "C"], "OPEN_BUNCH", 6),
                ], 81, 11.1, 9.0, "Value Spread"),
                ABCTicket("C", [], 200, 3.0, 6.0, "Chaos Saver"),
            ],
            total_outlay=30.0,
            risk_note="",
            estimated_return_pct=45.0,
            chaos_legs=[6, 8],
        )
        reason = _build_reason_abc(abc)
        assert "chaos" in reason.lower()
