"""Tests for weight-specific form analysis."""

import pytest
from punty.context.weight_analysis import analyse_weight_form


def _make_start(weight, position):
    return {"weight": weight, "position": position}


class TestAnalyseWeightForm:
    def test_basic_band_assignment(self):
        """Starts should be grouped into correct weight bands."""
        fh = [
            _make_start(54, 1),  # lighter (current=58)
            _make_start(55, 2),  # lighter
            _make_start(57, 3),  # similar
            _make_start(58, 4),  # similar
            _make_start(61, 5),  # heavier
        ]
        result = analyse_weight_form(fh, 58.0)
        assert result["bands"]["lighter"]["starts"] == 2
        assert result["bands"]["similar"]["starts"] == 2
        assert result["bands"]["heavier"]["starts"] == 1

    def test_wins_counted(self):
        fh = [
            _make_start(54, 1),  # win at lighter
            _make_start(55, 1),  # win at lighter
            _make_start(55, 3),  # place at lighter
            _make_start(60, 5),  # loss at heavier
            _make_start(61, 8),  # loss at heavier
        ]
        result = analyse_weight_form(fh, 58.0)
        assert result["bands"]["lighter"]["wins"] == 2
        assert result["bands"]["lighter"]["places"] == 3
        assert result["bands"]["heavier"]["wins"] == 0

    def test_weight_change_up(self):
        fh = [_make_start(55.5, 3)]
        result = analyse_weight_form(fh, 58.0)
        assert result["weight_change"] == 2.5

    def test_weight_change_down(self):
        fh = [_make_start(60.0, 4)]
        result = analyse_weight_form(fh, 58.0)
        assert result["weight_change"] == -2.0

    def test_weight_change_same(self):
        fh = [_make_start(58.0, 2)]
        result = analyse_weight_form(fh, 58.0)
        assert result["weight_change"] == 0.0

    def test_warning_zero_wins_heavier(self):
        """Should warn when 0 wins at heavier band with 2+ starts."""
        fh = [
            _make_start(54, 1),
            _make_start(55, 1),
            _make_start(61, 5),
            _make_start(62, 8),
        ]
        result = analyse_weight_form(fh, 58.0)
        assert result["warning"] is not None
        assert "0/2" in result["warning"]

    def test_no_warning_when_winning_heavy(self):
        fh = [
            _make_start(54, 1),
            _make_start(61, 1),
            _make_start(62, 3),
        ]
        result = analyse_weight_form(fh, 58.0)
        bands = result["bands"]
        # heavier has 1 win from 2 starts — no warning
        assert bands["heavier"]["wins"] == 1

    def test_optimal_band_detection(self):
        fh = [
            _make_start(54, 1),
            _make_start(55, 1),
            _make_start(55, 2),
            _make_start(58, 4),
            _make_start(57, 5),
        ]
        result = analyse_weight_form(fh, 58.0)
        assert result["optimal_band"] == "lighter"

    def test_empty_history(self):
        assert analyse_weight_form([], 58.0) == {}

    def test_none_history(self):
        assert analyse_weight_form(None, 58.0) == {}

    def test_no_current_weight(self):
        fh = [_make_start(55, 1)]
        assert analyse_weight_form(fh, 0) == {}

    def test_weight_as_string(self):
        """Weight values may be strings in PF data."""
        fh = [_make_start("55.5", "1")]
        result = analyse_weight_form(fh, 58.0)
        assert result["bands"]["lighter"]["wins"] == 1

    def test_missing_weight_skipped(self):
        fh = [
            {"position": 1},  # no weight
            _make_start(55, 2),
        ]
        result = analyse_weight_form(fh, 58.0)
        # Only 1 valid start
        total = sum(b["starts"] for b in result["bands"].values())
        assert total == 1

    def test_single_start_no_optimal(self):
        """Need at least 2 starts for optimal band."""
        fh = [_make_start(55, 1)]
        result = analyse_weight_form(fh, 58.0)
        assert result["optimal_band"] is None
