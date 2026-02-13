"""Tests for stewards report excuse parsing."""

import pytest
from punty.context.stewards import (
    parse_stewards_excuses,
    format_excuse_summary,
    extract_form_excuses,
)


class TestParseStewardsExcuses:
    def test_held_up(self):
        excuses = parse_stewards_excuses(
            "Held up approaching the home turn, had to shift out for clear running"
        )
        assert "held_up" in excuses

    def test_checked(self):
        assert "held_up" in parse_stewards_excuses("Checked near the 400m mark")

    def test_steadied(self):
        assert "held_up" in parse_stewards_excuses("Steadied when racing in restricted room")

    def test_wide(self):
        excuses = parse_stewards_excuses("Raced wide without cover throughout")
        assert "wide" in excuses

    def test_three_wide(self):
        assert "wide" in parse_stewards_excuses("Raced three-wide for most of the event")

    def test_slow_start(self):
        excuses = parse_stewards_excuses("Slow to begin, settled back")
        assert "slow_start" in excuses

    def test_began_awkwardly(self):
        assert "slow_start" in parse_stewards_excuses("Began awkwardly and lost ground")

    def test_dwelt(self):
        assert "slow_start" in parse_stewards_excuses("Dwelt at the start")

    def test_bumped(self):
        excuses = parse_stewards_excuses("Bumped on jumping, hampered at the 800m")
        assert "bumped" in excuses

    def test_crowded(self):
        assert "bumped" in parse_stewards_excuses("Crowded for room on the turn")

    def test_eased(self):
        excuses = parse_stewards_excuses("Not persevered with in the final 200m")
        assert "eased" in excuses

    def test_not_tested(self):
        assert "eased" in parse_stewards_excuses("Not fully tested over the final stages")

    def test_ran_on(self):
        excuses = parse_stewards_excuses("Ran on well in the final 200m")
        assert "ran_on" in excuses

    def test_best_work_late(self):
        assert "ran_on" in parse_stewards_excuses("Best work late, closing stages")

    def test_interference_hung(self):
        assert "interference" in parse_stewards_excuses("Hung out badly rounding the home turn")

    def test_interference_lugged(self):
        assert "interference" in parse_stewards_excuses("Lugged in under pressure")

    def test_over_raced(self):
        assert "interference" in parse_stewards_excuses("Over-raced in the early stages")

    def test_multiple_excuses(self):
        excuses = parse_stewards_excuses(
            "Held up for clear running, then raced wide without cover, ran on well late"
        )
        assert "held_up" in excuses
        assert "wide" in excuses
        assert "ran_on" in excuses

    def test_severity_ordering(self):
        """held_up should come before wide, which comes before ran_on."""
        excuses = parse_stewards_excuses(
            "Ran on well late, raced wide, held up on the turn"
        )
        assert excuses.index("held_up") < excuses.index("wide")
        assert excuses.index("wide") < excuses.index("ran_on")

    def test_no_excuses(self):
        excuses = parse_stewards_excuses("Raced handy, led on the turn, won well")
        assert excuses == []

    def test_none_input(self):
        assert parse_stewards_excuses(None) == []

    def test_empty_string(self):
        assert parse_stewards_excuses("") == []

    def test_case_insensitive(self):
        assert "held_up" in parse_stewards_excuses("HELD UP FOR CLEAR RUNNING")
        assert "wide" in parse_stewards_excuses("RACED WIDE WITHOUT COVER")

    def test_bled(self):
        assert "medical" in parse_stewards_excuses("Bled from both nostrils")

    def test_lame(self):
        assert "medical" in parse_stewards_excuses("Found to be lame in the off foreleg")

    def test_pulled_up(self):
        assert "medical" in parse_stewards_excuses("Pulled up lame approaching the 200m")

    def test_lost_shoe(self):
        assert "medical" in parse_stewards_excuses("Lost the near hind shoe during the race")

    def test_went_amiss(self):
        assert "medical" in parse_stewards_excuses("Something went amiss near the 600m mark")

    def test_fell(self):
        assert "medical" in parse_stewards_excuses("Fell near the 400m")

    def test_medical_highest_severity(self):
        """Medical excuses should appear first in severity ordering."""
        excuses = parse_stewards_excuses("Bled from the nostrils, raced wide, held up")
        assert excuses[0] == "medical"

    def test_medical_format(self):
        assert format_excuse_summary(["medical"]) == "medical/physical issue"


class TestFormatExcuseSummary:
    def test_single(self):
        assert format_excuse_summary(["held_up"]) == "held up"

    def test_multiple(self):
        result = format_excuse_summary(["held_up", "wide"])
        assert result == "held up, raced wide"

    def test_empty(self):
        assert format_excuse_summary([]) == ""


class TestExtractFormExcuses:
    def _make_start(self, position, comment="", venue="Test", distance=1400):
        return {
            "venue": venue,
            "distance": distance,
            "position": position,
            "comment": comment,
        }

    def test_excuse_for_poor_finish(self):
        fh = [self._make_start(8, "Held up for clear running, raced wide")]
        result = extract_form_excuses(fh)
        assert len(result) == 1
        assert result[0]["position"] == 8
        assert "held_up" in result[0]["excuses"]
        assert "wide" in result[0]["excuses"]

    def test_no_excuse_for_winner(self):
        """Winners don't need excuses even if comment mentions issues."""
        fh = [self._make_start(1, "Raced wide without cover")]
        result = extract_form_excuses(fh)
        assert len(result) == 0

    def test_no_excuse_for_top3(self):
        """Placegetters don't need excuses."""
        fh = [self._make_start(3, "Held up for clear running")]
        result = extract_form_excuses(fh)
        assert len(result) == 0

    def test_excuse_for_4th(self):
        """4th place finish with excuse should be flagged."""
        fh = [self._make_start(4, "Checked near the 400m")]
        result = extract_form_excuses(fh)
        assert len(result) == 1

    def test_max_starts_limit(self):
        fh = [self._make_start(8, f"Held up") for _ in range(10)]
        result = extract_form_excuses(fh, max_starts=3)
        assert len(result) == 3

    def test_no_comment(self):
        fh = [self._make_start(8, "")]
        result = extract_form_excuses(fh)
        assert len(result) == 0

    def test_empty_history(self):
        assert extract_form_excuses([]) == []

    def test_none_history(self):
        assert extract_form_excuses(None) == []

    def test_mixed_results(self):
        """Only poor finishes with excuses should be returned."""
        fh = [
            self._make_start(1, "Won easily"),
            self._make_start(8, "Held up, raced wide"),
            self._make_start(2, "Ran on well"),
            self._make_start(6, "Slow to begin"),
        ]
        result = extract_form_excuses(fh)
        assert len(result) == 2
        assert result[0]["position"] == 8
        assert result[1]["position"] == 6
