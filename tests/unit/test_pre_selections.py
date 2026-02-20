"""Tests for deterministic pre-selection engine."""

import pytest

from punty.context.pre_selections import (
    EACH_WAY_MAX_ODDS,
    EACH_WAY_MIN_ODDS,
    EACH_WAY_MIN_PROB,
    EXOTIC_PUNTYS_PICK_VALUE,
    PLACE_MIN_PROB,
    RACE_POOL,
    ROUGHIE_MIN_ODDS,
    ROUGHIE_MIN_VALUE,
    WIN_MIN_PROB,
    WIN_MIN_VALUE,
    RecommendedExotic,
    RecommendedPick,
    RacePreSelections,
    _allocate_stakes,
    _build_candidates,
    _calculate_puntys_pick,
    _determine_bet_type,
    _ensure_win_bet,
    _estimate_place_odds,
    _expected_return,
    _generate_notes,
    _select_exotic,
    calculate_pre_selections,
    format_pre_selections,
)


# ──────────────────────────────────────────────
# Helper: build a runner dict matching context shape
# ──────────────────────────────────────────────

def _runner(
    sc: int,
    name: str,
    odds: float,
    win_prob: float = 0.15,
    place_prob: float = 0.45,
    value: float = 1.1,
    place_value: float = 1.05,
    place_odds: float | None = None,
    rec_stake: float = 3.0,
    scratched: bool = False,
) -> dict:
    return {
        "saddlecloth": sc,
        "horse_name": name,
        "current_odds": odds,
        "place_odds": place_odds or round((odds - 1) / 3 + 1, 2),
        "_win_prob_raw": win_prob,
        "_place_prob_raw": place_prob,
        "punty_win_probability": f"{win_prob * 100:.1f}%",
        "punty_place_probability": f"{place_prob * 100:.1f}%",
        "punty_value_rating": value,
        "punty_place_value_rating": place_value,
        "punty_recommended_stake": rec_stake,
        "scratched": scratched,
    }


def _race_context(runners: list[dict], race_number: int = 1) -> dict:
    """Build a minimal race context dict."""
    probs = {
        "probability_ranked": [
            {"horse": r["horse_name"], "win_prob": f"{r['_win_prob_raw']*100:.1f}%",
             "saddlecloth": r["saddlecloth"]}
            for r in runners if not r.get("scratched")
        ],
        "value_plays": [
            {"horse": r["horse_name"], "value": r["punty_value_rating"], "edge": 5.0}
            for r in runners
            if not r.get("scratched") and r["punty_value_rating"] > 1.05
        ],
        "exotic_combinations": [],
    }
    return {
        "race_number": race_number,
        "runners": runners,
        "probabilities": probs,
    }


# ──────────────────────────────────────────────
# Tests: _build_candidates
# ──────────────────────────────────────────────

class TestBuildCandidates:
    def test_excludes_scratched(self):
        runners = [
            _runner(1, "Alpha", 3.0, scratched=True),
            _runner(2, "Beta", 4.0),
        ]
        result = _build_candidates(runners)
        assert len(result) == 1
        assert result[0]["saddlecloth"] == 2

    def test_excludes_no_odds(self):
        runners = [
            _runner(1, "Alpha", 0),
            _runner(2, "Beta", 4.0),
        ]
        result = _build_candidates(runners)
        assert len(result) == 1

    def test_excludes_no_saddlecloth(self):
        r = _runner(0, "NoSaddle", 4.0)
        r["saddlecloth"] = None
        result = _build_candidates([r])
        assert len(result) == 0

    def test_calculates_ev(self):
        runners = [_runner(1, "Alpha", 5.0, win_prob=0.25)]
        result = _build_candidates(runners)
        assert len(result) == 1
        # EV = 0.25 * 5.0 - 1 = 0.25
        assert abs(result[0]["ev"] - 0.25) < 0.001


# ──────────────────────────────────────────────
# Tests: _determine_bet_type
# ──────────────────────────────────────────────

class TestDetermineBetType:
    def test_top_pick_win_when_strong(self):
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 3.5,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_top_pick_win_in_sweet_spot(self):
        """$4-$6 with good prob/value should be Win (edge: +60.8% ROI)."""
        c = {"win_prob": 0.22, "place_prob": 0.55, "odds": 6.0,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_top_pick_each_way_outside_win_sweet_spot(self):
        """$8 odds with moderate prob should get Each Way (outside $4-$6 sweet spot)."""
        c = {"win_prob": 0.18, "place_prob": 0.45, "odds": 8.0,
             "value_rating": 0.95, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Each Way"

    def test_top_pick_each_way_odds_too_low(self):
        """Short-priced horses shouldn't get Each Way."""
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 2.5,
             "value_rating": 1.10, "place_value_rating": 1.05}
        result = _determine_bet_type(c, rank=1, is_roughie=False)
        assert result == "Win"  # odds below $4, so Win not E/W

    def test_second_pick_each_way_in_sweet_spot(self):
        """Rank 2 in $4-$6 sweet spot gets Each Way (upside + place protection)."""
        c = {"win_prob": 0.22, "place_prob": 0.55, "odds": 4.0,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Each Way"

    def test_second_pick_each_way_strong_signal_dead_zone(self):
        """Rank 2 in $3-$4 dead zone gets Each Way with strong signal."""
        c = {"win_prob": 0.28, "place_prob": 0.55, "odds": 3.0,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Each Way"

    def test_second_pick_place_when_no_value(self):
        c = {"win_prob": 0.12, "place_prob": 0.45, "odds": 8.0,
             "value_rating": 0.95, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Place"

    def test_third_pick_place(self):
        c = {"win_prob": 0.15, "place_prob": 0.45, "odds": 6.0,
             "value_rating": 1.05, "place_value_rating": 1.10}
        assert _determine_bet_type(c, rank=3, is_roughie=False) == "Place"

    def test_roughie_defaults_to_place(self):
        c = {"win_prob": 0.08, "place_prob": 0.30, "odds": 12.0,
             "value_rating": 1.15, "place_value_rating": 1.10}
        assert _determine_bet_type(c, rank=4, is_roughie=True) == "Place"

    def test_roughie_win_with_strong_value(self):
        c = {"win_prob": 0.20, "place_prob": 0.50, "odds": 10.0,
             "value_rating": 1.25, "place_value_rating": 1.10}
        assert _determine_bet_type(c, rank=4, is_roughie=True) == "Win"


# ──────────────────────────────────────────────
# Tests: _ensure_win_bet
# ──────────────────────────────────────────────

class TestEnsureWinBet:
    def test_no_change_when_win_exists(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 8, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _ensure_win_bet(picks)
        assert picks[0].bet_type == "Win"
        assert picks[1].bet_type == "Place"

    def test_upgrades_top_to_win_when_all_place(self):
        picks = [
            RecommendedPick(1, 1, "A", "Place", 8, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _ensure_win_bet(picks)
        assert picks[0].bet_type == "Win"

    def test_each_way_counts_as_win(self):
        picks = [
            RecommendedPick(1, 1, "A", "Each Way", 8, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _ensure_win_bet(picks)
        assert picks[0].bet_type == "Each Way"  # unchanged


# ──────────────────────────────────────────────
# Tests: _allocate_stakes
# ──────────────────────────────────────────────

class TestAllocateStakes:
    def test_total_within_pool(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 0, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
            RecommendedPick(3, 3, "C", "Place", 0, 7.0, 2.5, 0.15, 0.4, 1.05, 1.0, 0.2),
            RecommendedPick(4, 4, "D", "Place", 0, 12.0, 3.0, 0.08, 0.3, 1.1, 1.05, 0.1, True),
        ]
        _allocate_stakes(picks, 20.0)
        total = sum(p.stake for p in picks)
        assert total <= 20.5  # allow small rounding

    def test_each_way_costs_double(self):
        picks = [
            RecommendedPick(1, 1, "A", "Each Way", 0, 6.0, 2.5, 0.2, 0.5, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 0, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _allocate_stakes(picks, 20.0)
        # Each Way actual cost = stake * 2
        actual_total = picks[0].stake * 2 + picks[1].stake
        assert actual_total <= 20.5

    def test_minimum_stake(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
        ]
        _allocate_stakes(picks, 20.0)
        assert picks[0].stake >= 1.0

    def test_empty_picks(self):
        _allocate_stakes([], 20.0)  # no crash


# ──────────────────────────────────────────────
# Tests: _select_exotic
# ──────────────────────────────────────────────

class TestSelectExotic:
    def test_prefers_overlap_with_selections(self):
        combos = [
            {"type": "Trifecta Box", "runners": [1, 2, 3], "runner_names": ["A", "B", "C"],
             "probability": "8.5%", "value": 1.30, "combos": 6, "format": "boxed"},
            {"type": "Trifecta Box", "runners": [5, 6, 7], "runner_names": ["E", "F", "G"],
             "probability": "9.0%", "value": 1.35, "combos": 6, "format": "boxed"},
        ]
        result = _select_exotic(combos, {1, 2, 3, 4})
        assert result is not None
        assert set(result.runners) == {1, 2, 3}  # overlaps with our picks

    def test_returns_none_when_empty(self):
        assert _select_exotic([], {1, 2}) is None

    def test_metro_venue_allowed(self):
        """Metro venues should allow exotics (no venue filter)."""
        combos = [
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": "12.5%", "value": 1.50, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2}, venue_type="metro_vic")
        assert result is not None

    def test_provincial_venue_allowed(self):
        """Provincial venues should allow exotics."""
        combos = [
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": "12.5%", "value": 1.50, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2}, venue_type="provincial")
        assert result is not None

    def test_parses_string_probability(self):
        combos = [
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": "12.5%", "value": 1.50, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2})
        assert result is not None
        assert abs(result.probability - 0.125) < 0.001

    def test_ev_scoring_picks_best_type(self):
        """Higher EV (prob × value) wins regardless of exotic type."""
        combos = [
            # Exacta: 10% prob × 1.3 value = 0.13 EV
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": 0.10, "value": 1.3, "combos": 1, "format": "flat"},
            # Trifecta Box: 8% prob × 2.0 value = 0.16 EV (higher)
            {"type": "Trifecta Box", "runners": [1, 2, 3], "runner_names": ["A", "B", "C"],
             "probability": 0.08, "value": 2.0, "combos": 6, "format": "boxed"},
        ]
        result = _select_exotic(combos, {1, 2, 3})
        assert result is not None
        assert result.exotic_type == "Trifecta Box"

    def test_all_types_can_win(self):
        """Every exotic type can be selected when it has the best EV."""
        for etype, fmt in [
            ("Quinella", "flat"), ("Exacta Standout", "standout"),
            ("Trifecta Box", "boxed"), ("First4", "legs"), ("First4 Box", "boxed"),
        ]:
            combos = [
                {"type": etype, "runners": [1, 2, 3], "runner_names": ["A", "B", "C"],
                 "probability": 0.15, "value": 2.0, "combos": 3, "format": fmt},
            ]
            result = _select_exotic(combos, {1, 2, 3})
            assert result is not None, f"{etype} should be selectable"
            assert result.exotic_type == etype


# ──────────────────────────────────────────────
# Tests: _calculate_puntys_pick
# ──────────────────────────────────────────────

class TestCalculatePuntysPick:
    def test_picks_best_ev_selection(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.05),
            RecommendedPick(2, 2, "Beta", "Place", 5, 5.0, 2.0, 0.20, 0.50, 1.10, 1.05, -0.1),
        ]
        pp = _calculate_puntys_pick(picks, None)
        assert pp is not None
        assert pp.pick_type == "selection"
        assert pp.horse_name == "Alpha"

    def test_exotic_wins_when_high_value(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.05, 1.0, 0.05),
        ]
        exotic = RecommendedExotic(
            exotic_type="Trifecta Box", runners=[1, 2, 3],
            runner_names=["A", "B", "C"], probability=0.085,
            value_ratio=2.0, num_combos=6, format="boxed",
        )
        pp = _calculate_puntys_pick(picks, exotic)
        assert pp is not None
        assert pp.pick_type == "exotic"
        assert pp.exotic_type == "Trifecta Box"

    def test_exotic_loses_when_low_value(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.5),
        ]
        exotic = RecommendedExotic(
            exotic_type="Quinella", runners=[1, 2],
            runner_names=["A", "B"], probability=0.20,
            value_ratio=1.3, num_combos=1, format="flat",
        )
        pp = _calculate_puntys_pick(picks, exotic)
        assert pp is not None
        assert pp.pick_type == "selection"  # exotic value below 1.5x

    def test_secondary_bet_added(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.5),
            RecommendedPick(2, 2, "Beta", "Place", 5, 5.0, 2.0, 0.20, 0.50, 1.10, 1.05, 0.1),
        ]
        pp = _calculate_puntys_pick(picks, None)
        assert pp.secondary_horse == "Beta"
        assert pp.secondary_bet_type == "Place"

    def test_no_secondary_when_negative_ev(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.5),
            RecommendedPick(2, 2, "Beta", "Place", 5, 5.0, 2.0, 0.10, 0.30, 0.90, 0.85, -0.3),
        ]
        pp = _calculate_puntys_pick(picks, None)
        assert pp.secondary_horse is None

    def test_empty_picks_returns_none(self):
        assert _calculate_puntys_pick([], None) is None


# ──────────────────────────────────────────────
# Tests: _estimate_place_odds
# ──────────────────────────────────────────────

class TestEstimatePlaceOdds:
    def test_basic_estimation(self):
        assert abs(_estimate_place_odds(7.0) - 3.0) < 0.01

    def test_short_odds(self):
        result = _estimate_place_odds(2.0)
        assert result >= 1.0

    def test_invalid_odds(self):
        assert _estimate_place_odds(1.0) == 1.0


# ──────────────────────────────────────────────
# Tests: _expected_return
# ──────────────────────────────────────────────

class TestExpectedReturn:
    def test_win_positive_ev(self):
        c = {"odds": 4.0, "place_odds": 1.8, "win_prob": 0.30, "place_prob": 0.60}
        er = _expected_return(c, "Win")
        # 0.30 * 4.0 - 1 = 0.2
        assert abs(er - 0.2) < 0.01

    def test_place_positive_ev(self):
        c = {"odds": 4.0, "place_odds": 1.8, "win_prob": 0.30, "place_prob": 0.60}
        er = _expected_return(c, "Place")
        # 0.60 * 1.8 - 1 = 0.08
        assert abs(er - 0.08) < 0.01

    def test_each_way_combines(self):
        c = {"odds": 4.0, "place_odds": 1.8, "win_prob": 0.30, "place_prob": 0.60}
        er = _expected_return(c, "Each Way")
        win_er = 0.30 * 4.0 - 1  # 0.2
        place_er = 0.60 * 1.8 - 1  # 0.08
        expected = (win_er + place_er) / 2  # 0.14
        assert abs(er - expected) < 0.01


# ──────────────────────────────────────────────
# Tests: _generate_notes
# ──────────────────────────────────────────────

class TestGenerateNotes:
    def test_no_value_note(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 8, 3.0, 1.5, 0.3, 0.6, 0.95, 0.90, -0.1),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.00, 1.0, 0.0),
        ]
        notes = _generate_notes(picks, None, [{"win_prob": 0.3}, {"win_prob": 0.2}])
        assert any("No clear value" in n for n in notes)

    def test_strong_value_note(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 8, 3.0, 1.5, 0.3, 0.6, 1.25, 1.10, 0.5),
        ]
        notes = _generate_notes(picks, None, [{"win_prob": 0.3}])
        assert any("Strong value" in n for n in notes)

    def test_wide_open_note(self):
        candidates = [{"win_prob": 0.12}, {"win_prob": 0.10}]
        notes = _generate_notes([], None, candidates)
        assert any("Wide-open" in n for n in notes)


# ──────────────────────────────────────────────
# Tests: calculate_pre_selections (integration)
# ──────────────────────────────────────────────

class TestCalculatePreSelections:
    def test_basic_race(self):
        runners = [
            _runner(1, "Alpha", 3.5, win_prob=0.28, place_prob=0.60, value=1.15),
            _runner(2, "Beta", 5.0, win_prob=0.20, place_prob=0.50, value=1.10),
            _runner(3, "Gamma", 8.0, win_prob=0.14, place_prob=0.40, value=1.05),
            _runner(4, "Delta", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "Epsilon", 15.0, win_prob=0.06, place_prob=0.20, value=0.90),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        assert isinstance(result, RacePreSelections)
        assert result.race_number == 1
        assert len(result.picks) == 4
        assert result.total_stake <= 20.5
        assert result.puntys_pick is not None

    def test_pick_order_by_ev(self):
        """Highest win probability runner should be pick #1."""
        runners = [
            _runner(1, "LowEV", 2.0, win_prob=0.10, value=0.80),
            _runner(2, "HighEV", 4.0, win_prob=0.35, value=1.20),
            _runner(3, "MidEV", 6.0, win_prob=0.18, value=1.10),
            _runner(4, "Roughie", 15.0, win_prob=0.06, value=1.15),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        assert result.picks[0].horse_name == "HighEV"

    def test_roughie_identified(self):
        """Runner with odds >= $8 and value >= 1.10 should be roughie."""
        runners = [
            _runner(1, "Fav", 2.5, win_prob=0.35, value=1.10),
            _runner(2, "Contender", 5.0, win_prob=0.20, value=1.08),
            _runner(3, "Chance", 7.0, win_prob=0.14, value=1.05),
            _runner(4, "Outsider", 15.0, win_prob=0.07, place_prob=0.25, value=1.25),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        roughies = [p for p in result.picks if p.is_roughie]
        assert len(roughies) == 1
        assert roughies[0].horse_name == "Outsider"

    def test_at_least_one_win_bet(self):
        """Mandatory rule: at least one Win/Saver Win/Each Way per race."""
        runners = [
            _runner(1, "A", 3.0, win_prob=0.10, place_prob=0.45, value=0.90, place_value=1.10),
            _runner(2, "B", 4.0, win_prob=0.10, place_prob=0.42, value=0.85, place_value=1.05),
            _runner(3, "C", 5.0, win_prob=0.08, place_prob=0.38, value=0.80, place_value=1.00),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        win_types = {"Win", "Saver Win", "Each Way"}
        assert any(p.bet_type in win_types for p in result.picks)

    def test_empty_runners(self):
        ctx = _race_context([])
        result = calculate_pre_selections(ctx)
        assert result.picks == []
        assert result.puntys_pick is None
        assert result.total_stake == 0.0

    def test_all_scratched(self):
        runners = [
            _runner(1, "A", 3.0, scratched=True),
            _runner(2, "B", 4.0, scratched=True),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        assert result.picks == []

    def test_two_runner_race(self):
        """Small field — should still produce picks."""
        runners = [
            _runner(1, "A", 2.0, win_prob=0.55, place_prob=0.80, value=1.10),
            _runner(2, "B", 3.0, win_prob=0.35, place_prob=0.65, value=1.05),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        assert len(result.picks) == 2
        assert result.total_stake <= 20.5

    def test_exotic_recommendation(self):
        runners = [
            _runner(1, "A", 3.0, win_prob=0.30, value=1.10),
            _runner(2, "B", 4.0, win_prob=0.22, value=1.08),
            _runner(3, "C", 6.0, win_prob=0.15, value=1.05),
        ]
        ctx = _race_context(runners)
        ctx["probabilities"]["exotic_combinations"] = [
            {"type": "Trifecta Box", "runners": [1, 2, 3],
             "runner_names": ["A", "B", "C"],
             "probability": "8.5%", "value": 1.40, "combos": 6, "format": "boxed"},
        ]
        result = calculate_pre_selections(ctx)
        assert result.exotic is not None
        assert result.exotic.exotic_type == "Trifecta Box"

    def test_puntys_pick_exotic_when_high_value(self):
        """Exotic should become Punty's Pick when value >= 1.5x."""
        # Anchor odds must be <= $3.50 for exotic filter to pass
        runners = [
            _runner(1, "A", 3.00, win_prob=0.35, value=1.05),
            _runner(2, "B", 4.50, win_prob=0.22, value=1.02),
        ]
        ctx = _race_context(runners)
        ctx["probabilities"]["exotic_combinations"] = [
            {"type": "Exacta Standout", "runners": [1, 2],
             "runner_names": ["A", "B"],
             "probability": "15.0%", "value": 2.0, "combos": 1, "format": "standout"},
        ]
        result = calculate_pre_selections(ctx)
        # Exotic value is 2.0x which is > 1.5x threshold
        if result.puntys_pick:
            # The exotic edge (2.0 - 1.0 = 1.0) should beat selection edge
            assert result.puntys_pick.pick_type == "exotic"


# ──────────────────────────────────────────────
# Tests: format_pre_selections
# ──────────────────────────────────────────────

class TestFormatPreSelections:
    def test_basic_formatting(self):
        runners = [
            _runner(1, "Alpha", 3.5, win_prob=0.28, place_prob=0.60, value=1.15),
            _runner(2, "Beta", 5.0, win_prob=0.20, place_prob=0.50, value=1.10),
            _runner(3, "Gamma", 8.0, win_prob=0.14, place_prob=0.40, value=1.05),
            _runner(4, "Delta", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        formatted = format_pre_selections(result)

        assert "LOCKED SELECTIONS" in formatted
        assert "Race 1" in formatted
        assert "Punty's Pick" in formatted
        assert "Total stake" in formatted
        assert "$" in formatted

    def test_exotic_in_output(self):
        runners = [
            _runner(1, "A", 3.0, win_prob=0.30),
            _runner(2, "B", 4.0, win_prob=0.25),
        ]
        ctx = _race_context(runners)
        ctx["probabilities"]["exotic_combinations"] = [
            {"type": "Exacta Standout", "runners": [1, 2],
             "runner_names": ["A", "B"],
             "probability": "20.0%", "value": 1.50, "combos": 1, "format": "standout"},
        ]
        result = calculate_pre_selections(ctx)
        formatted = format_pre_selections(result)
        assert "Exotic:" in formatted
        assert "Exacta Standout" in formatted

    def test_notes_in_output(self):
        runners = [
            _runner(1, "A", 3.0, win_prob=0.10, value=0.90),
            _runner(2, "B", 4.0, win_prob=0.10, value=0.95),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        formatted = format_pre_selections(result)
        assert "NOTE:" in formatted
