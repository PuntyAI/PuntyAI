"""Tests for post-generation content validation."""

import pytest

from punty.validation.content_validator import (
    EXOTIC_MIN_RUNNERS,
    FIRST4_RUNNERS,
    PUNTYS_PICK_MIN_PROB,
    STAKE_TOLERANCE,
    STAKE_TOTAL_TARGET,
    TRIFECTA_RUNNERS,
    WIN_BET_MIN_PROB,
    ValidationIssue,
    ValidationResult,
    _parse_exotic_runners,
    _validate_combo_maths,
    _validate_exotics,
    _validate_puntys_picks,
    _validate_selections,
    _validate_sequences,
    validate_content,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _runner(sc: int, name: str = "", win_prob: float = 0.15, odds: float = 5.0) -> dict:
    """Build a mock runner dict."""
    return {
        "saddlecloth": sc,
        "horse_name": name or f"Horse{sc}",
        "_win_prob_raw": win_prob,
        "current_odds": odds,
    }


def _selection(
    race: int,
    sc: int,
    name: str = "",
    bet_type: str = "Win",
    stake: float = 5.0,
    rank: int = 1,
) -> dict:
    """Build a mock selection pick."""
    return {
        "pick_type": "selection",
        "race_number": race,
        "saddlecloth": sc,
        "horse_name": name or f"Horse{sc}",
        "bet_type": bet_type,
        "bet_stake": stake,
        "tip_rank": rank,
    }


def _exotic(race: int, exotic_type: str, runners, stake: float = 20.0) -> dict:
    """Build a mock exotic pick."""
    return {
        "pick_type": "exotic",
        "race_number": race,
        "exotic_type": exotic_type,
        "exotic_runners": runners,
        "exotic_stake": stake,
    }


def _sequence(
    race: int,
    seq_type: str = "Quaddie",
    variant: str = "Skinny",
    legs=None,
    start_race: int = 5,
    combos: int = None,
) -> dict:
    """Build a mock sequence pick."""
    return {
        "pick_type": "sequence",
        "race_number": race,
        "sequence_type": seq_type,
        "sequence_variant": variant,
        "sequence_legs": legs or [[1, 2], [3], [1, 4], [2]],
        "sequence_start_race": start_race,
        "sequence_combos": combos,
    }


def _race_data(race_num: int, runners: list[dict] = None) -> dict:
    """Build a race data dict."""
    if runners is None:
        runners = [_runner(i) for i in range(1, 9)]
    return {
        "runners": runners,
        "field_size": len(runners),
    }


# ──────────────────────────────────────────────
# Tests: ValidationResult
# ──────────────────────────────────────────────

class TestValidationResult:
    def test_empty_is_valid(self):
        vr = ValidationResult()
        assert vr.is_valid
        assert vr.errors == []
        assert vr.warnings == []

    def test_error_makes_invalid(self):
        vr = ValidationResult()
        vr.issues.append(ValidationIssue("error", 1, "bad", "consistency"))
        assert not vr.is_valid
        assert len(vr.errors) == 1
        assert len(vr.warnings) == 0

    def test_warning_stays_valid(self):
        vr = ValidationResult()
        vr.issues.append(ValidationIssue("warning", 1, "hmm", "probability"))
        assert vr.is_valid
        assert len(vr.errors) == 0
        assert len(vr.warnings) == 1

    def test_mixed_issues(self):
        vr = ValidationResult()
        vr.issues.append(ValidationIssue("error", 1, "bad", "consistency"))
        vr.issues.append(ValidationIssue("warning", 2, "hmm", "probability"))
        assert not vr.is_valid
        assert len(vr.errors) == 1
        assert len(vr.warnings) == 1

    def test_summary_valid(self):
        vr = ValidationResult(picks_checked=12, races_checked=4)
        assert "Valid" in vr.summary()
        assert "12 picks" in vr.summary()

    def test_summary_with_errors(self):
        vr = ValidationResult(picks_checked=8, races_checked=3)
        vr.issues.append(ValidationIssue("error", 1, "bad", "consistency"))
        vr.issues.append(ValidationIssue("error", 2, "worse", "exotic"))
        assert "2 errors" in vr.summary()

    def test_summary_with_warnings(self):
        vr = ValidationResult(picks_checked=6, races_checked=2)
        vr.issues.append(ValidationIssue("warning", 1, "hmm", "probability"))
        assert "1 warning" in vr.summary()


# ──────────────────────────────────────────────
# Tests: _validate_selections
# ──────────────────────────────────────────────

class TestValidateSelections:
    def test_valid_selections(self):
        """Valid selections produce no issues."""
        runner_map = {1: _runner(1, win_prob=0.25), 2: _runner(2, win_prob=0.20)}
        picks = [
            _selection(1, 1, bet_type="Win", stake=10.0),
            _selection(1, 2, bet_type="Place", stake=10.0),
        ]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        assert len(result.issues) == 0

    def test_runner_not_found(self):
        """Selection on non-existent runner produces error."""
        runner_map = {1: _runner(1)}
        picks = [_selection(1, 99, name="Ghost", bet_type="Win")]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert len(errors) == 1
        assert "No.99" in errors[0].message
        assert errors[0].category == "consistency"

    def test_low_probability_win_bet(self):
        """Win bet on low-probability runner produces warning."""
        runner_map = {1: _runner(1, win_prob=0.05)}
        picks = [_selection(1, 1, bet_type="Win", stake=10.0)]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        warnings = [i for i in result.issues if i.level == "warning"]
        assert any("probability" in w.category for w in warnings)

    def test_saver_win_also_counts_as_win(self):
        """Saver Win counts as a win bet for mandatory check."""
        runner_map = {1: _runner(1, win_prob=0.20)}
        picks = [_selection(1, 1, bet_type="Saver Win", stake=20.0)]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        # Should not have "No Win bet" warning
        assert not any("mandatory" in i.message.lower() for i in result.issues)

    def test_no_win_bet_warning(self):
        """All Place bets (no Win) produces warning."""
        runner_map = {1: _runner(1), 2: _runner(2)}
        picks = [
            _selection(1, 1, bet_type="Place", stake=10.0),
            _selection(1, 2, bet_type="Place", stake=10.0),
        ]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        assert any("no win" in i.message.lower() for i in result.issues)

    def test_each_way_counts_as_win(self):
        """Each Way satisfies the mandatory Win bet requirement."""
        runner_map = {1: _runner(1, win_prob=0.20)}
        picks = [_selection(1, 1, bet_type="Each Way", stake=5.0)]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        assert not any("mandatory" in i.message.lower() for i in result.issues)

    def test_each_way_single_counted_in_stakes(self):
        """Each Way stake counts as single (E/W killed, treated as Place)."""
        runner_map = {1: _runner(1, win_prob=0.20)}
        # $10 E/W = $10 total now (no longer doubled)
        picks = [_selection(1, 1, bet_type="Each Way", stake=10.0)]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        stake_issues = [i for i in result.issues if i.category == "stake"]
        # $10 vs $20 target = $10 difference, exceeds $3 tolerance
        assert len(stake_issues) == 1

    def test_stake_over_target(self):
        """Stakes way over $20 produce warning."""
        runner_map = {1: _runner(1, win_prob=0.20), 2: _runner(2, win_prob=0.18)}
        picks = [
            _selection(1, 1, bet_type="Win", stake=20.0),
            _selection(1, 2, bet_type="Win", stake=20.0),
        ]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        stake_issues = [i for i in result.issues if i.category == "stake"]
        assert len(stake_issues) == 1
        assert "$40.00" in stake_issues[0].message

    def test_stake_within_tolerance(self):
        """Stakes within tolerance ($20 ± $3) produce no warning."""
        runner_map = {1: _runner(1, win_prob=0.20), 2: _runner(2, win_prob=0.18)}
        picks = [
            _selection(1, 1, bet_type="Win", stake=12.0),
            _selection(1, 2, bet_type="Win", stake=10.0),
        ]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        stake_issues = [i for i in result.issues if i.category == "stake"]
        assert len(stake_issues) == 0

    def test_zero_probability_skipped(self):
        """Runner with zero probability doesn't trigger warning."""
        runner_map = {1: _runner(1, win_prob=0)}
        picks = [_selection(1, 1, bet_type="Win", stake=20.0)]
        result = ValidationResult()
        _validate_selections(picks, runner_map, 1, result)
        prob_warnings = [i for i in result.issues if i.category == "probability"]
        assert len(prob_warnings) == 0


# ──────────────────────────────────────────────
# Tests: _validate_exotics
# ──────────────────────────────────────────────

class TestValidateExotics:
    def test_valid_trifecta(self):
        runner_map = {i: _runner(i) for i in range(1, 9)}
        picks = [_exotic(1, "Trifecta", [1, 3, 5])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        assert len(result.issues) == 0

    def test_trifecta_too_few_runners(self):
        runner_map = {i: _runner(i) for i in range(1, 9)}
        picks = [_exotic(1, "Trifecta", [1, 3])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert any("Trifecta" in e.message for e in errors)

    def test_first4_too_few_runners(self):
        runner_map = {i: _runner(i) for i in range(1, 9)}
        picks = [_exotic(1, "First 4", [1, 2, 3])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert any("First4" in e.message for e in errors)

    def test_exotic_runner_not_in_field(self):
        runner_map = {i: _runner(i) for i in range(1, 5)}
        picks = [_exotic(1, "Exacta", [1, 99])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert any("No.99" in e.message for e in errors)

    def test_no_parseable_runners(self):
        runner_map = {i: _runner(i) for i in range(1, 5)}
        picks = [_exotic(1, "Quinella", [])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert any("no parseable" in e.message.lower() for e in errors)

    def test_valid_exacta(self):
        runner_map = {i: _runner(i) for i in range(1, 9)}
        picks = [_exotic(1, "Exacta", [2, 5])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        assert len(result.issues) == 0

    def test_quinella_needs_two(self):
        runner_map = {i: _runner(i) for i in range(1, 9)}
        picks = [_exotic(1, "Quinella", [3])]
        result = ValidationResult()
        _validate_exotics(picks, runner_map, 1, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert len(errors) >= 1


# ──────────────────────────────────────────────
# Tests: _validate_sequences
# ──────────────────────────────────────────────

class TestValidateSequences:
    def test_valid_sequence(self):
        race_data = {r: _race_data(r) for r in range(5, 9)}
        picks = [_sequence(5, legs=[[1, 2], [3, 4], [1], [2, 5]], start_race=5)]
        result = ValidationResult()
        _validate_sequences(picks, race_data, 5, result)
        assert len(result.issues) == 0

    def test_leg_runner_not_in_field(self):
        race_data = {r: _race_data(r, [_runner(i) for i in range(1, 5)]) for r in range(5, 9)}
        # Runner 99 doesn't exist in any race
        picks = [_sequence(5, legs=[[1, 99], [3], [1], [2]], start_race=5)]
        result = ValidationResult()
        _validate_sequences(picks, race_data, 5, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert any("No.99" in e.message for e in errors)

    def test_empty_legs_no_error(self):
        race_data = {r: _race_data(r) for r in range(5, 9)}
        picks = [_sequence(5, legs=[], start_race=5)]
        result = ValidationResult()
        _validate_sequences(picks, race_data, 5, result)
        assert len(result.issues) == 0

    def test_string_legs_parsed(self):
        """JSON string legs should be parsed."""
        race_data = {r: _race_data(r) for r in range(5, 9)}
        picks = [_sequence(5, legs='[[1, 2], [3], [1, 4], [2]]', start_race=5)]
        result = ValidationResult()
        _validate_sequences(picks, race_data, 5, result)
        # Should parse successfully with no errors (all runners 1-8 exist)
        assert len(result.issues) == 0


# ──────────────────────────────────────────────
# Tests: _validate_combo_maths
# ──────────────────────────────────────────────

class TestValidateComboMaths:
    def test_correct_combos(self):
        """Stated combos matching calculated → no error."""
        pick = _sequence(5, legs=[[1, 2], [3], [1, 4], [2]], start_race=5, combos=4)
        # 2 * 1 * 2 * 1 = 4
        result = ValidationResult()
        _validate_combo_maths(pick, result)
        assert len(result.issues) == 0

    def test_wrong_combos(self):
        """Stated combos NOT matching → error."""
        pick = _sequence(5, legs=[[1, 2], [3], [1, 4], [2]], start_race=5, combos=8)
        # 2 * 1 * 2 * 1 = 4, stated 8
        result = ValidationResult()
        _validate_combo_maths(pick, result)
        errors = [i for i in result.issues if i.level == "error"]
        assert len(errors) == 1
        assert "Combo maths" in errors[0].message

    def test_no_stated_combos_ok(self):
        """No stated combos → no check, no error."""
        pick = _sequence(5, legs=[[1, 2], [3], [1, 4], [2]], start_race=5, combos=None)
        result = ValidationResult()
        _validate_combo_maths(pick, result)
        assert len(result.issues) == 0

    def test_string_legs_parsed(self):
        """JSON string legs should be parseable."""
        pick = {
            "pick_type": "sequence",
            "sequence_legs": '[[1, 2], [3, 4], [5]]',
            "sequence_start_race": 5,
            "sequence_combos": 4,
            "sequence_variant": "Balanced",
        }
        # 2 * 2 * 1 = 4
        result = ValidationResult()
        _validate_combo_maths(pick, result)
        assert len(result.issues) == 0


# ──────────────────────────────────────────────
# Tests: _validate_puntys_picks
# ──────────────────────────────────────────────

class TestValidatePuntysPicks:
    def test_good_probability(self):
        """Punty's Pick with sufficient probability → no warning."""
        picks = [_selection(1, 1, bet_type="Win", stake=10.0, rank=0)]
        race_data = {1: {"runners": [_runner(1, win_prob=0.25)], "field_size": 8}}
        result = ValidationResult()
        _validate_puntys_picks(picks, race_data, result)
        assert len(result.issues) == 0

    def test_low_probability_warning(self):
        """Punty's Pick below 15% → warning."""
        picks = [_selection(1, 1, bet_type="Win", stake=10.0, rank=0)]
        race_data = {1: {"runners": [_runner(1, win_prob=0.08)], "field_size": 8}}
        result = ValidationResult()
        _validate_puntys_picks(picks, race_data, result)
        warnings = [i for i in result.issues if i.level == "warning"]
        assert len(warnings) == 1
        assert "probability" in warnings[0].category

    def test_non_puntys_pick_ignored(self):
        """Regular selections (rank != 0) not checked as Punty's Pick."""
        picks = [_selection(1, 1, bet_type="Win", stake=10.0, rank=1)]
        race_data = {1: {"runners": [_runner(1, win_prob=0.05)], "field_size": 8}}
        result = ValidationResult()
        _validate_puntys_picks(picks, race_data, result)
        assert len(result.issues) == 0

    def test_zero_probability_skipped(self):
        """Punty's Pick with zero probability (not calculated) → no warning."""
        picks = [_selection(1, 1, bet_type="Win", stake=10.0, rank=0)]
        race_data = {1: {"runners": [_runner(1, win_prob=0)], "field_size": 8}}
        result = ValidationResult()
        _validate_puntys_picks(picks, race_data, result)
        assert len(result.issues) == 0

    def test_multiple_races(self):
        """Punty's Picks across multiple races all checked."""
        picks = [
            _selection(1, 1, bet_type="Win", stake=10.0, rank=0),
            _selection(2, 3, bet_type="Win", stake=10.0, rank=0),
        ]
        race_data = {
            1: {"runners": [_runner(1, win_prob=0.08)], "field_size": 8},
            2: {"runners": [_runner(3, win_prob=0.05)], "field_size": 10},
        }
        result = ValidationResult()
        _validate_puntys_picks(picks, race_data, result)
        warnings = [i for i in result.issues if i.level == "warning"]
        assert len(warnings) == 2


# ──────────────────────────────────────────────
# Tests: _parse_exotic_runners
# ──────────────────────────────────────────────

class TestParseExoticRunners:
    def test_list_of_ints(self):
        assert _parse_exotic_runners([1, 3, 5]) == [1, 3, 5]

    def test_list_of_strings(self):
        assert _parse_exotic_runners(["1", "3", "5"]) == [1, 3, 5]

    def test_json_string(self):
        assert _parse_exotic_runners("[1, 3, 5]") == [1, 3, 5]

    def test_comma_separated(self):
        assert _parse_exotic_runners("1, 3, 5") == [1, 3, 5]

    def test_slash_separated(self):
        assert _parse_exotic_runners("1/3/5") == [1, 3, 5]

    def test_empty_list(self):
        assert _parse_exotic_runners([]) == []

    def test_empty_string(self):
        assert _parse_exotic_runners("") == []

    def test_none(self):
        assert _parse_exotic_runners(None) == []

    def test_mixed_types_in_list(self):
        assert _parse_exotic_runners([1, "3", 5]) == [1, 3, 5]

    def test_non_numeric_strings_filtered(self):
        assert _parse_exotic_runners(["1", "abc", "3"]) == [1, 3]


# ──────────────────────────────────────────────
# Tests: validate_content (integration)
# ──────────────────────────────────────────────

class TestValidateContent:
    def test_valid_early_mail(self):
        """Fully valid picks produce no errors."""
        runners = [_runner(i, win_prob=0.15 + i * 0.02) for i in range(1, 9)]
        race_data = {
            r: {"runners": runners, "field_size": len(runners)}
            for r in range(1, 9)
        }
        picks = []
        for r in range(1, 9):
            picks.append(_selection(r, 1, bet_type="Win", stake=8.0, rank=0))
            picks.append(_selection(r, 2, bet_type="Place", stake=6.0, rank=1))
            picks.append(_selection(r, 3, bet_type="Place", stake=6.0, rank=2))
            picks.append(_exotic(r, "Trifecta", [1, 2, 3]))

        result = validate_content(picks, race_data)
        assert result.races_checked == 8
        assert result.picks_checked == len(picks)
        # May have stake warnings but no errors
        assert all(i.level != "error" for i in result.issues)

    def test_empty_picks(self):
        """No picks → valid result with zero counts."""
        result = validate_content([], {})
        assert result.is_valid
        assert result.races_checked == 0
        assert result.picks_checked == 0

    def test_mixed_errors_and_warnings(self):
        """Picks with various issues produce both errors and warnings."""
        runners = [_runner(i, win_prob=0.20) for i in range(1, 5)]
        race_data = {1: {"runners": runners, "field_size": 4}}
        picks = [
            # Selection on non-existent runner → error
            _selection(1, 99, bet_type="Win", stake=10.0, rank=1),
            # Valid selection
            _selection(1, 1, bet_type="Win", stake=10.0, rank=0),
            # Exotic with runner not in field → error
            _exotic(1, "Exacta", [1, 88]),
        ]
        result = validate_content(picks, race_data)
        assert not result.is_valid
        assert len(result.errors) >= 2  # runner 99 + runner 88

    def test_sequences_validated(self):
        """Sequence picks are validated for leg runners."""
        runners = [_runner(i) for i in range(1, 5)]
        race_data = {r: {"runners": runners, "field_size": 4} for r in range(5, 9)}
        picks = [
            _sequence(5, legs=[[1, 2], [3, 99], [1], [2]], start_race=5),
        ]
        result = validate_content(picks, race_data)
        errors = [i for i in result.issues if i.level == "error"]
        assert any("No.99" in e.message for e in errors)

    def test_puntys_pick_flagged(self):
        """Punty's Pick with low probability → warning."""
        runners = [_runner(1, win_prob=0.05)]
        race_data = {1: {"runners": runners, "field_size": 8}}
        picks = [_selection(1, 1, bet_type="Win", stake=20.0, rank=0)]
        result = validate_content(picks, race_data)
        warnings = [i for i in result.issues if i.category == "probability"]
        assert len(warnings) >= 1

    def test_multiple_races_grouped(self):
        """Picks across multiple races are grouped and validated per-race."""
        runners = [_runner(i, win_prob=0.20) for i in range(1, 5)]
        race_data = {
            1: {"runners": runners, "field_size": 4},
            2: {"runners": runners, "field_size": 4},
        }
        picks = [
            _selection(1, 1, bet_type="Win", stake=20.0, rank=0),
            _selection(2, 2, bet_type="Win", stake=20.0, rank=0),
        ]
        result = validate_content(picks, race_data)
        assert result.races_checked == 2
        assert result.picks_checked == 2
