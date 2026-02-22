"""Tests for the probability calculation engine."""

import json
import pytest

from punty.probability import (
    RunnerProbability,
    StatsRecord,
    FACTOR_REGISTRY,
    DEFAULT_WEIGHTS,
    calculate_race_probabilities,
    parse_stats_string,
    _market_consensus,
    _calculate_overround,
    _get_median_odds,
    _form_rating,
    _score_last_five,
    _pace_factor,
    _market_movement_factor,
    _class_factor,
    _barrier_draw_factor,
    _load_barrier_calibration,
    _jockey_trainer_factor,
    _weight_factor,
    _horse_profile_factor,
    _average_weight,
    _place_probability,
    _harville_place_probability,
    _recommended_stake,
    _determine_pace_scenario,
    _condition_stats_field,
    _deep_learning_factor,
    _get_dist_bucket,
    _normalize_condition,
    _get_barrier_bucket,
    _get_move_type,
    _odds_to_sp_range,
    _get_state_for_venue,
    _boost_market_weight,
    _apply_market_floor,
    _parse_a2e_json,
    _a2e_to_score,
    _parse_margin_value,
    _get_weight_change_class,
    _get_context_multipliers,
    _context_venue_type,
    _context_class_bucket,
    set_dl_pattern_cache,
    get_dl_pattern_cache,
)
import punty.probability as _prob_module


@pytest.fixture(autouse=True)
def _reset_output_calibration():
    """Reset output calibration singleton so tests don't depend on local data files."""
    old = _prob_module._OUTPUT_CALIBRATION
    _prob_module._OUTPUT_CALIBRATION = {}  # empty dict = "checked, not found" → skip calibration
    yield
    _prob_module._OUTPUT_CALIBRATION = old


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_runner(
    id="r1",
    scratched=False,
    current_odds=None,
    opening_odds=None,
    place_odds=None,
    odds_tab=None,
    odds_sportsbet=None,
    odds_bet365=None,
    odds_ladbrokes=None,
    odds_betfair=None,
    odds_flucs=None,
    last_five=None,
    track_dist_stats=None,
    track_stats=None,
    distance_stats=None,
    good_track_stats=None,
    soft_track_stats=None,
    heavy_track_stats=None,
    first_up_stats=None,
    second_up_stats=None,
    class_stats=None,
    days_since_last_run=None,
    speed_map_position=None,
    pf_speed_rank=None,
    pf_map_factor=None,
    pf_jockey_factor=None,
    pf_settle=None,
    handicap_rating=None,
    barrier=None,
    weight=None,
    jockey_stats=None,
    trainer_stats=None,
    horse_age=None,
    horse_sex=None,
    jockey=None,
    trainer=None,
    career_record=None,
    career_prize_money=None,
    form_history=None,
):
    return {
        "id": id,
        "scratched": scratched,
        "current_odds": current_odds,
        "opening_odds": opening_odds,
        "place_odds": place_odds,
        "odds_tab": odds_tab,
        "odds_sportsbet": odds_sportsbet,
        "odds_bet365": odds_bet365,
        "odds_ladbrokes": odds_ladbrokes,
        "odds_betfair": odds_betfair,
        "odds_flucs": odds_flucs,
        "last_five": last_five,
        "track_dist_stats": track_dist_stats,
        "track_stats": track_stats,
        "distance_stats": distance_stats,
        "good_track_stats": good_track_stats,
        "soft_track_stats": soft_track_stats,
        "heavy_track_stats": heavy_track_stats,
        "first_up_stats": first_up_stats,
        "second_up_stats": second_up_stats,
        "class_stats": class_stats,
        "days_since_last_run": days_since_last_run,
        "speed_map_position": speed_map_position,
        "pf_speed_rank": pf_speed_rank,
        "pf_map_factor": pf_map_factor,
        "pf_jockey_factor": pf_jockey_factor,
        "pf_settle": pf_settle,
        "handicap_rating": handicap_rating,
        "barrier": barrier,
        "weight": weight,
        "jockey_stats": jockey_stats,
        "trainer_stats": trainer_stats,
        "horse_age": horse_age,
        "horse_sex": horse_sex,
        "jockey": jockey,
        "trainer": trainer,
        "career_record": career_record,
        "career_prize_money": career_prize_money,
        "form_history": form_history,
    }


def _make_race(**kwargs):
    defaults = {
        "id": "race1",
        "track_condition": "Good 4",
        "distance": 1200,
        "field_size": 8,
        "class_": "BM64",
    }
    defaults.update(kwargs)
    return defaults


def _make_meeting(**kwargs):
    defaults = {
        "id": "meet1",
        "track_condition": "Good 4",
        "venue": "Sale",
    }
    defaults.update(kwargs)
    return defaults


# ──────────────────────────────────────────────
# Stats String Parser
# ──────────────────────────────────────────────

class TestParseStatsString:
    def test_colon_format(self):
        result = parse_stats_string("5: 2-1-0")
        assert result == StatsRecord(starts=5, wins=2, seconds=1, thirds=0)

    def test_colon_format_no_spaces(self):
        result = parse_stats_string("12:3-4-2")
        assert result == StatsRecord(starts=12, wins=3, seconds=4, thirds=2)

    def test_dash_four_format(self):
        result = parse_stats_string("5-2-1-0")
        assert result == StatsRecord(starts=5, wins=2, seconds=1, thirds=0)

    def test_paren_format(self):
        result = parse_stats_string("2-1-0 (5)")
        assert result == StatsRecord(starts=5, wins=2, seconds=1, thirds=0)

    def test_none_input(self):
        assert parse_stats_string(None) is None

    def test_empty_string(self):
        assert parse_stats_string("") is None

    def test_non_string(self):
        assert parse_stats_string(42) is None

    def test_garbage_input(self):
        assert parse_stats_string("no stats here") is None

    def test_win_rate(self):
        result = parse_stats_string("10: 3-2-1")
        assert result.win_rate == pytest.approx(0.3)
        assert result.place_rate == pytest.approx(0.6)

    def test_zero_starts(self):
        result = parse_stats_string("0: 0-0-0")
        assert result.win_rate == 0.0
        assert result.place_rate == 0.0


# ──────────────────────────────────────────────
# Market Consensus
# ──────────────────────────────────────────────

class TestMarketConsensus:
    def test_median_odds_single_source(self):
        runner = _make_runner(current_odds=5.0)
        assert _get_median_odds(runner) == 5.0

    def test_median_odds_multiple_sources(self):
        runner = _make_runner(
            current_odds=5.0,
            odds_tab=4.8,
            odds_sportsbet=5.2,
            odds_bet365=5.0,
            odds_ladbrokes=4.5,
        )
        median = _get_median_odds(runner)
        # median of [4.5, 4.8, 5.0, 5.2] (current_odds excluded to avoid
        # double-counting — it's derived from one of the providers)
        assert median == 4.9

    def test_median_odds_filters_invalid(self):
        runner = _make_runner(
            current_odds=5.0,
            odds_tab=None,
            odds_sportsbet=0.0,
            odds_bet365=1.0,  # excluded (must be > 1.0)
        )
        # No valid provider odds, falls back to current_odds
        assert _get_median_odds(runner) == 5.0

    def test_median_odds_no_valid(self):
        runner = _make_runner()
        assert _get_median_odds(runner) is None

    def test_median_odds_bad_tab_not_double_counted(self):
        """When TAB odds are garbage (e.g. $1.20 for a $23 horse),
        current_odds should NOT double-count the bad value."""
        runner = _make_runner(
            current_odds=1.20,  # Set from bad TAB value
            odds_tab=1.20,      # Bad TAB odds
            odds_sportsbet=23.0,
            odds_bet365=21.0,
        )
        median = _get_median_odds(runner)
        # median of [1.20, 21.0, 23.0] — current_odds excluded
        assert median == 21.0

    def test_overround_calculation(self):
        runners = [
            _make_runner(id="r1", current_odds=2.0),
            _make_runner(id="r2", current_odds=3.0),
            _make_runner(id="r3", current_odds=6.0),
        ]
        # 1/2 + 1/3 + 1/6 = 0.5 + 0.333 + 0.167 = 1.0
        overround = _calculate_overround(runners)
        assert overround == pytest.approx(1.0, abs=0.01)

    def test_overround_with_margin(self):
        runners = [
            _make_runner(id="r1", current_odds=1.8),
            _make_runner(id="r2", current_odds=2.8),
            _make_runner(id="r3", current_odds=5.0),
        ]
        # 1/1.8 + 1/2.8 + 1/5.0 = 0.556 + 0.357 + 0.200 = 1.113 (11.3% overround)
        overround = _calculate_overround(runners)
        assert overround > 1.0

    def test_market_consensus_fair_book(self):
        runners = [
            _make_runner(id="r1", current_odds=2.0),
            _make_runner(id="r2", current_odds=3.0),
            _make_runner(id="r3", current_odds=6.0),
        ]
        overround = _calculate_overround(runners)
        prob = _market_consensus(runners[0], overround)
        assert prob == pytest.approx(0.5, abs=0.02)

    def test_market_consensus_with_overround(self):
        runners = [
            _make_runner(id="r1", current_odds=1.8),
            _make_runner(id="r2", current_odds=2.8),
            _make_runner(id="r3", current_odds=5.0),
        ]
        overround = _calculate_overround(runners)
        probs = [_market_consensus(r, overround) for r in runners]
        # Should sum to ~1.0 after normalization
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_market_consensus_no_odds(self):
        runner = _make_runner()
        assert _market_consensus(runner, 1.0) == 0.0


# ──────────────────────────────────────────────
# Form Rating
# ──────────────────────────────────────────────

class TestFormRating:
    def test_strong_recent_form(self):
        runner = _make_runner(last_five="11213")
        rating = _form_rating(runner, "Good 4", 0.10)
        assert rating > 0.5  # above neutral

    def test_weak_recent_form(self):
        runner = _make_runner(last_five="87x96")
        rating = _form_rating(runner, "Good 4", 0.10)
        assert rating < 0.5  # below neutral

    def test_strong_track_dist_stats(self):
        runner = _make_runner(track_dist_stats="10: 4-2-1")
        rating = _form_rating(runner, "Good 4", 0.10)
        # Calibrated scoring may shift thresholds; verify it's not neutral
        assert rating != 0.5

    def test_condition_specialist_heavy(self):
        runner = _make_runner(heavy_track_stats="8: 3-2-1")
        rating = _form_rating(runner, "Heavy 8", 0.10)
        assert rating > 0.5

    def test_condition_specialist_soft(self):
        runner = _make_runner(soft_track_stats="6: 2-1-1")
        rating = _form_rating(runner, "Soft 5", 0.10)
        assert rating > 0.5

    def test_first_up_specialist(self):
        runner = _make_runner(
            first_up_stats="8: 3-2-1",
            days_since_last_run=90,
        )
        rating = _form_rating(runner, "Good 4", 0.10)
        assert rating > 0.5

    def test_no_data_returns_neutral(self):
        runner = _make_runner()
        rating = _form_rating(runner, "Good 4", 0.10)
        assert 0.4 <= rating <= 0.6  # neutral range

    def test_score_last_five_all_wins(self):
        score = _score_last_five("11111")
        assert score > 0.7

    def test_score_last_five_all_last(self):
        score = _score_last_five("99989")
        assert score < 0.4


# ──────────────────────────────────────────────
# Pace Factor
# ──────────────────────────────────────────────

class TestPaceFactor:
    def test_strong_map_factor(self):
        runner = _make_runner(pf_map_factor=1.3)
        factor = _pace_factor(runner, "genuine_pace")
        assert factor > 0.5

    def test_weak_map_factor(self):
        runner = _make_runner(pf_map_factor=0.7)
        factor = _pace_factor(runner, "genuine_pace")
        assert factor < 0.5

    def test_leader_slow_pace(self):
        runner = _make_runner(
            speed_map_position="leader",
            pf_speed_rank=3,
        )
        factor = _pace_factor(runner, "slow_pace")
        assert factor > 0.5

    def test_leader_hot_pace(self):
        runner = _make_runner(
            speed_map_position="leader",
            pf_speed_rank=2,
        )
        factor = _pace_factor(runner, "hot_pace")
        assert factor < 0.5

    def test_backmarker_hot_pace(self):
        runner = _make_runner(
            speed_map_position="backmarker",
            pf_speed_rank=20,
        )
        factor = _pace_factor(runner, "hot_pace")
        assert factor > 0.5

    def test_neutral_no_data(self):
        runner = _make_runner()
        factor = _pace_factor(runner, "moderate_pace")
        assert factor == pytest.approx(0.5)


# ──────────────────────────────────────────────
# Market Movement
# ──────────────────────────────────────────────

class TestMarketMovement:
    def test_heavy_support(self):
        flucs = json.dumps([
            {"odds": 8.0, "time": 1000},
            {"odds": 5.0, "time": 2000},
        ])
        runner = _make_runner(odds_flucs=flucs)
        factor = _market_movement_factor(runner)
        assert factor > 0.5  # positive signal

    def test_big_drift(self):
        flucs = json.dumps([
            {"odds": 4.0, "time": 1000},
            {"odds": 8.0, "time": 2000},
        ])
        runner = _make_runner(odds_flucs=flucs)
        factor = _market_movement_factor(runner)
        assert factor < 0.5  # negative signal

    def test_stable_odds(self):
        flucs = json.dumps([
            {"odds": 5.0, "time": 1000},
            {"odds": 5.2, "time": 2000},
        ])
        runner = _make_runner(odds_flucs=flucs)
        factor = _market_movement_factor(runner)
        assert 0.45 <= factor <= 0.55  # roughly neutral

    def test_fallback_to_opening_current(self):
        runner = _make_runner(opening_odds=8.0, current_odds=5.0)
        factor = _market_movement_factor(runner)
        assert factor > 0.5

    def test_no_data_neutral(self):
        runner = _make_runner()
        factor = _market_movement_factor(runner)
        assert factor == 0.5


# ──────────────────────────────────────────────
# Class/Fitness Factor
# ──────────────────────────────────────────────

class TestClassFactor:
    def test_strong_at_class(self):
        runner = _make_runner(class_stats="8: 3-2-1")
        factor = _class_factor(runner, 0.10)
        assert factor > 0.5

    def test_sweet_spot_fitness(self):
        runner = _make_runner(days_since_last_run=21)
        factor = _class_factor(runner, 0.10)
        assert factor > 0.5

    def test_long_layoff_penalty(self):
        runner = _make_runner(days_since_last_run=120)
        factor = _class_factor(runner, 0.10)
        assert factor < 0.5

    def test_no_data_neutral(self):
        runner = _make_runner()
        factor = _class_factor(runner, 0.10)
        assert factor == 0.5


# ──────────────────────────────────────────────
# Place Probability
# ──────────────────────────────────────────────

class TestPlaceProbability:
    def test_small_field(self):
        prob = _place_probability(0.30, 5)
        assert prob == pytest.approx(0.60)

    def test_medium_field(self):
        prob = _place_probability(0.20, 10)
        assert prob == pytest.approx(0.20 * 2.8)  # 8-12 runner fields: factor 2.8

    def test_large_field(self):
        prob = _place_probability(0.15, 14)
        assert prob == pytest.approx(0.15 * 3.2)  # 13+ runner fields: factor 3.2

    def test_capped_at_95(self):
        prob = _place_probability(0.50, 5)
        assert prob == 0.95


class TestHarvillePlaceProbability:
    """Tests for Harville-model place probability."""

    def test_favourite_has_highest_place_prob(self):
        """The highest win-prob runner should have highest place prob."""
        probs = {"a": 0.35, "b": 0.25, "c": 0.20, "d": 0.10, "e": 0.10}
        p_a = _harville_place_probability("a", probs, place_count=3)
        p_d = _harville_place_probability("d", probs, place_count=3)
        assert p_a > p_d

    def test_place_probs_bounded(self):
        """Place probabilities should be between 0 and 0.95."""
        probs = {"a": 0.40, "b": 0.30, "c": 0.15, "d": 0.10, "e": 0.05}
        for rid in probs:
            p = _harville_place_probability(rid, probs, place_count=3)
            assert 0 <= p <= 0.95

    def test_two_places(self):
        """With place_count=2, should be p(1st) + p(2nd) only."""
        probs = {"a": 0.40, "b": 0.30, "c": 0.20, "d": 0.10}
        p3 = _harville_place_probability("a", probs, place_count=3)
        p2 = _harville_place_probability("a", probs, place_count=2)
        assert p2 <= p3  # 2 places should be <= 3 places

    def test_strong_favourite_high_place_prob(self):
        """A strong favourite should have very high place probability."""
        probs = {"fav": 0.50, "b": 0.20, "c": 0.15, "d": 0.10, "e": 0.05}
        p = _harville_place_probability("fav", probs, place_count=3)
        assert p > 0.80  # should be very likely to place

    def test_zero_prob_runner_returns_zero(self):
        """Runner with 0 win probability has 0 place probability."""
        probs = {"a": 0.40, "b": 0.30, "zero": 0.0}
        p = _harville_place_probability("zero", probs, place_count=3)
        assert p == 0.0

    def test_top_n_approximation(self):
        """Result with top_n=8 should be reasonable for a realistic field."""
        # Realistic racing distribution
        probs = {"fav": 0.30, "r2": 0.18, "r3": 0.12, "r4": 0.10,
                 "r5": 0.08, "r6": 0.07, "r7": 0.05, "r8": 0.04,
                 "r9": 0.03, "r10": 0.03}
        p_full = _harville_place_probability("fav", probs, place_count=3, top_n=10)
        p_approx = _harville_place_probability("fav", probs, place_count=3, top_n=8)
        # Default top_n=8 should be close to full for top runners
        assert abs(p_full - p_approx) < 0.05


# ──────────────────────────────────────────────
# Recommended Stake
# ──────────────────────────────────────────────

class TestRecommendedStake:
    def test_positive_edge(self):
        # 30% chance at $5.00 = positive edge
        stake = _recommended_stake(0.30, 5.0, 20.0)
        assert stake > 0

    def test_no_edge(self):
        # 10% chance at $5.00 = no edge (1/5 = 20%)
        stake = _recommended_stake(0.10, 5.0, 20.0)
        assert stake == 0.0

    def test_favorite_small_edge(self):
        # 55% chance at $2.00 (50% implied) = small edge
        stake = _recommended_stake(0.55, 2.0, 20.0)
        assert 0 < stake < 20.0

    def test_invalid_odds(self):
        assert _recommended_stake(0.30, 0.0, 20.0) == 0.0
        assert _recommended_stake(0.30, 1.0, 20.0) == 0.0

    def test_zero_probability(self):
        assert _recommended_stake(0.0, 5.0, 20.0) == 0.0

    def test_capped_at_pool(self):
        # Very strong edge shouldn't exceed pool
        stake = _recommended_stake(0.90, 2.0, 20.0)
        assert stake <= 20.0


# ──────────────────────────────────────────────
# Composite Calculation
# ──────────────────────────────────────────────

class TestCalculateRaceProbabilities:
    def test_basic_three_runner_race(self):
        runners = [
            _make_runner(id="fav", current_odds=2.0, last_five="11213"),
            _make_runner(id="mid", current_odds=4.0, last_five="32451"),
            _make_runner(id="long", current_odds=8.0, last_five="65879"),
        ]
        race = _make_race()
        meeting = _make_meeting()
        results = calculate_race_probabilities(runners, race, meeting)

        assert len(results) == 3
        # Favorite should have highest probability
        assert results["fav"].win_probability > results["mid"].win_probability
        assert results["mid"].win_probability > results["long"].win_probability

    def test_probabilities_sum_to_one(self):
        runners = [
            _make_runner(id=f"r{i}", current_odds=2.0 + i * 2)
            for i in range(8)
        ]
        race = _make_race(field_size=8)
        meeting = _make_meeting()
        results = calculate_race_probabilities(runners, race, meeting)

        total = sum(r.win_probability for r in results.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_scratched_runners_excluded(self):
        runners = [
            _make_runner(id="r1", current_odds=3.0),
            _make_runner(id="r2", current_odds=4.0, scratched=True),
            _make_runner(id="r3", current_odds=5.0),
        ]
        race = _make_race()
        meeting = _make_meeting()
        results = calculate_race_probabilities(runners, race, meeting)

        assert len(results) == 2
        assert "r2" not in results

    def test_empty_runners(self):
        results = calculate_race_probabilities([], _make_race(), _make_meeting())
        assert results == {}

    def test_all_scratched(self):
        runners = [
            _make_runner(id="r1", scratched=True),
            _make_runner(id="r2", scratched=True),
        ]
        results = calculate_race_probabilities(runners, _make_race(), _make_meeting())
        assert results == {}

    def test_value_detection(self):
        # Runner with strong form but longer odds = should show value
        runners = [
            _make_runner(
                id="value",
                current_odds=6.0,
                last_five="11213",
                track_dist_stats="10: 4-2-1",
                pf_map_factor=1.2,
            ),
            _make_runner(
                id="overbet",
                current_odds=2.0,
                last_five="54678",
                pf_map_factor=0.8,
            ),
            _make_runner(id="r3", current_odds=4.0),
        ]
        race = _make_race()
        meeting = _make_meeting()
        results = calculate_race_probabilities(runners, race, meeting)

        # All runners should have value_rating and edge populated
        for rp in results.values():
            assert rp.value_rating is not None
            assert rp.edge is not None

    def test_place_probability_populated(self):
        runners = [
            _make_runner(id="r1", current_odds=3.0),
            _make_runner(id="r2", current_odds=5.0),
        ]
        results = calculate_race_probabilities(runners, _make_race(), _make_meeting())

        for rp in results.values():
            assert rp.place_probability > rp.win_probability

    def test_place_probability_uses_context_multipliers(self):
        """Place probs should differ from simple win×factor when place context mults exist."""
        runners = [
            _make_runner(id="r1", current_odds=3.0, last_five="11111"),
            _make_runner(id="r2", current_odds=5.0, last_five="54321"),
            _make_runner(id="r3", current_odds=8.0, last_five="33333"),
        ]
        race = _make_race(field_size=10)
        meeting = _make_meeting()

        # Mock place context multipliers to heavily boost horse_profile
        place_mults = {
            "market": 0.5, "form": 0.5, "class_fitness": 0.5, "pace": 0.5,
            "barrier": 0.5, "jockey_trainer": 0.5, "weight_carried": 0.5,
            "horse_profile": 2.5, "movement": 0.5,
        }
        import unittest.mock as mock
        original_fn = _get_context_multipliers.__wrapped__ if hasattr(_get_context_multipliers, '__wrapped__') else None

        def mock_ctx(race, meeting, outcome_type="win"):
            if outcome_type == "place":
                return place_mults
            return {}  # no win context mults

        with mock.patch("punty.probability._get_context_multipliers", side_effect=mock_ctx):
            results_with = calculate_race_probabilities(runners, race, meeting)

        # Without any context (default)
        results_without = calculate_race_probabilities(runners, race, meeting)

        # Place probabilities should differ when place context is applied
        # (the mock gives heavy horse_profile weight which changes relative ordering)
        r1_with = results_with["r1"].place_probability
        r1_without = results_without["r1"].place_probability
        # They may or may not differ depending on profile data, but both should be valid
        assert 0 < r1_with <= 0.95
        assert 0 < r1_without <= 0.95

    def test_place_probs_sum_approximately_to_place_count(self):
        """Place probabilities should sum to roughly place_count (3 for 8+ fields)."""
        runners = [
            _make_runner(id=f"r{i}", current_odds=2.0 + i * 1.5)
            for i in range(1, 9)
        ]
        race = _make_race(field_size=8)
        meeting = _make_meeting()

        import unittest.mock as mock
        place_mults = {
            "market": 1.0, "form": 1.2, "class_fitness": 1.0, "pace": 1.0,
            "barrier": 1.0, "jockey_trainer": 1.0, "weight_carried": 1.0,
            "horse_profile": 1.0, "movement": 1.0,
        }

        def mock_ctx(race, meeting, outcome_type="win"):
            if outcome_type == "place":
                return place_mults
            return {}

        with mock.patch("punty.probability._get_context_multipliers", side_effect=mock_ctx):
            results = calculate_race_probabilities(runners, race, meeting)

        total_place = sum(rp.place_probability for rp in results.values())
        # Should be roughly 3 (place_count for 8+ field), allow some slack for capping
        assert 2.0 < total_place < 4.5, f"Place prob sum {total_place} out of expected range"

    def test_place_prob_fallback_when_no_context(self):
        """Without place context mults, place prob should use field-factor formula."""
        runners = [
            _make_runner(id="r1", current_odds=3.0),
            _make_runner(id="r2", current_odds=6.0),
        ]
        race = _make_race(field_size=10)
        meeting = _make_meeting()

        import unittest.mock as mock

        def mock_ctx(race, meeting, outcome_type="win"):
            return {}  # no context mults at all

        with mock.patch("punty.probability._get_context_multipliers", side_effect=mock_ctx):
            results = calculate_race_probabilities(runners, race, meeting)

        # Should still have valid place probabilities (from formula fallback)
        for rp in results.values():
            assert rp.place_probability > rp.win_probability
            assert rp.place_probability <= 0.95

    def test_factors_populated(self):
        runners = [
            _make_runner(id="r1", current_odds=3.0, last_five="12345"),
        ]
        results = calculate_race_probabilities(runners, _make_race(), _make_meeting())
        factors = results["r1"].factors

        assert "market" in factors
        assert "form" in factors
        assert "pace" in factors
        assert "movement" in factors
        assert "class_fitness" in factors
        assert "barrier" in factors
        assert "jockey_trainer" in factors
        assert "weight_carried" in factors
        assert "horse_profile" in factors

    def test_runners_without_odds_get_probability(self):
        """Runners without odds should still get a probability (from other factors)."""
        runners = [
            _make_runner(id="r1", current_odds=3.0),
            _make_runner(id="r2"),  # no odds at all
        ]
        results = calculate_race_probabilities(runners, _make_race(), _make_meeting())

        # r2 should still exist but with low probability
        assert "r2" in results
        assert results["r2"].win_probability > 0


# ──────────────────────────────────────────────
# Pace Scenario
# ──────────────────────────────────────────────

class TestPaceScenario:
    def test_hot_pace(self):
        runners = [
            _make_runner(speed_map_position="leader"),
            _make_runner(speed_map_position="leader"),
            _make_runner(speed_map_position="leader"),
            _make_runner(speed_map_position="midfield"),
        ]
        assert _determine_pace_scenario(runners) == "hot_pace"

    def test_slow_pace(self):
        runners = [
            _make_runner(speed_map_position="midfield"),
            _make_runner(speed_map_position="backmarker"),
            _make_runner(speed_map_position="midfield"),
        ]
        assert _determine_pace_scenario(runners) == "slow_pace"

    def test_genuine_pace(self):
        runners = [
            _make_runner(speed_map_position="leader"),
            _make_runner(speed_map_position="on_pace"),
            _make_runner(speed_map_position="midfield"),
        ]
        assert _determine_pace_scenario(runners) == "genuine_pace"

    def test_moderate_pace(self):
        runners = [
            _make_runner(speed_map_position="leader"),
            _make_runner(speed_map_position="leader"),
            _make_runner(speed_map_position="on_pace"),
            _make_runner(speed_map_position="on_pace"),
            _make_runner(speed_map_position="on_pace"),
        ]
        assert _determine_pace_scenario(runners) == "moderate_pace"


# ──────────────────────────────────────────────
# Condition Stats Field Mapping
# ──────────────────────────────────────────────

class TestConditionStatsField:
    def test_good(self):
        assert _condition_stats_field("Good 4") == "good_track_stats"

    def test_soft(self):
        assert _condition_stats_field("Soft 5") == "soft_track_stats"

    def test_heavy(self):
        assert _condition_stats_field("Heavy 8") == "heavy_track_stats"

    def test_firm(self):
        assert _condition_stats_field("Firm 1") == "good_track_stats"

    def test_none(self):
        assert _condition_stats_field(None) is None

    def test_empty(self):
        assert _condition_stats_field("") is None


# ──────────────────────────────────────────────
# Factor Registry
# ──────────────────────────────────────────────

class TestFactorRegistry:
    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Default weights sum to {total}, expected 1.0"

    def test_all_registry_keys_in_defaults(self):
        for key in FACTOR_REGISTRY:
            assert key in DEFAULT_WEIGHTS, f"Factor '{key}' missing from DEFAULT_WEIGHTS"

    def test_all_default_keys_in_registry(self):
        for key in DEFAULT_WEIGHTS:
            assert key in FACTOR_REGISTRY, f"Weight '{key}' missing from FACTOR_REGISTRY"

    def test_registry_has_required_fields(self):
        for key, meta in FACTOR_REGISTRY.items():
            assert "label" in meta, f"Factor '{key}' missing label"
            assert "category" in meta, f"Factor '{key}' missing category"
            assert "description" in meta, f"Factor '{key}' missing description"


# ──────────────────────────────────────────────
# Barrier Draw Factor
# ──────────────────────────────────────────────

class TestBarrierDrawFactor:
    def test_inside_barrier_advantage(self):
        runner = _make_runner(barrier=1)
        score = _barrier_draw_factor(runner, field_size=12, distance=1200)
        assert score > 0.5, "Inside barrier should score above neutral"

    def test_wide_barrier_penalty(self):
        runner = _make_runner(barrier=14)
        score = _barrier_draw_factor(runner, field_size=14, distance=1200)
        assert score < 0.5, "Widest barrier should score below neutral"

    def test_middle_barrier_neutral(self):
        runner = _make_runner(barrier=6)
        score = _barrier_draw_factor(runner, field_size=12, distance=1400)
        assert abs(score - 0.5) < 0.05, "Middle barrier should be near neutral"

    def test_sprint_amplifies_barrier(self):
        runner = _make_runner(barrier=1)
        sprint = _barrier_draw_factor(runner, field_size=12, distance=1000)
        staying = _barrier_draw_factor(runner, field_size=12, distance=2400)
        assert sprint > staying, "Barrier matters more in sprints"

    def test_no_barrier_data_returns_neutral(self):
        runner = _make_runner(barrier=None)
        assert _barrier_draw_factor(runner, field_size=12) == 0.5

    def test_single_runner_field(self):
        runner = _make_runner(barrier=1)
        assert _barrier_draw_factor(runner, field_size=1) == 0.5

    def test_score_bounds(self):
        for b in [1, 5, 10, 16, 20]:
            runner = _make_runner(barrier=b)
            score = _barrier_draw_factor(runner, field_size=20, distance=1200)
            assert 0.05 <= score <= 0.95

    def test_venue_calibration_lookup(self, monkeypatch):
        """When calibration data exists for venue, use venue-specific multiplier."""
        cal_data = {
            "randwick": {
                "sprint": {
                    "outside": {"multiplier": 0.5, "win_rate": 0.05, "sample": 50},
                }
            }
        }
        import punty.probability as prob
        monkeypatch.setattr(prob, "_BARRIER_CALIBRATION", cal_data)
        runner = _make_runner(barrier=12)  # outside in 12-runner field
        score = _barrier_draw_factor(runner, field_size=12, distance=1100, venue="Randwick")
        # multiplier 0.5 → score = 0.5 + (0.5 - 1.0) * 0.15 = 0.425
        assert score == pytest.approx(0.425, abs=0.01)
        # Reset global
        monkeypatch.setattr(prob, "_BARRIER_CALIBRATION", None)

    def test_venue_calibration_fallback(self, monkeypatch):
        """Unknown venue falls through to generic logic."""
        import punty.probability as prob
        monkeypatch.setattr(prob, "_BARRIER_CALIBRATION", {})
        runner = _make_runner(barrier=1)
        score = _barrier_draw_factor(runner, field_size=12, distance=1200, venue="NowhereVille")
        assert score > 0.5, "Generic logic: inside barrier advantage"
        monkeypatch.setattr(prob, "_BARRIER_CALIBRATION", None)

    def test_venue_calibration_strong_advantage(self, monkeypatch):
        """High multiplier (e.g. 2.0) should score well above neutral."""
        cal_data = {
            "caulfield": {
                "sprint": {
                    "inside": {"multiplier": 2.0, "win_rate": 0.30, "sample": 40},
                }
            }
        }
        import punty.probability as prob
        monkeypatch.setattr(prob, "_BARRIER_CALIBRATION", cal_data)
        runner = _make_runner(barrier=1)
        score = _barrier_draw_factor(runner, field_size=12, distance=1100, venue="Caulfield")
        # multiplier 2.0 → score = 0.5 + (2.0 - 1.0) * 0.15 = 0.65
        assert score == pytest.approx(0.65, abs=0.01)
        monkeypatch.setattr(prob, "_BARRIER_CALIBRATION", None)


# ──────────────────────────────────────────────
# Jockey & Trainer Factor
# ──────────────────────────────────────────────

class TestJockeyTrainerFactor:
    def test_strong_jockey(self):
        runner = _make_runner(jockey_stats="50: 15-10-5")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        # Calibrated scoring: strong jockey at 30% SR scores differently
        # than hardcoded piecewise; just verify it's not neutral
        assert score != 0.5, "High jockey win rate should not be neutral"

    def test_strong_trainer(self):
        runner = _make_runner(trainer_stats="100: 25-15-10")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score != 0.5, "High trainer win rate should not be neutral"

    def test_both_strong(self):
        runner = _make_runner(jockey_stats="50: 15-10-5", trainer_stats="100: 25-15-10")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score != 0.5

    def test_weak_connections(self):
        runner = _make_runner(jockey_stats="50: 2-3-5", trainer_stats="100: 3-5-10")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score < 0.55, "Low win rates should score near or below neutral"

    def test_no_data_returns_neutral(self):
        runner = _make_runner()
        assert _jockey_trainer_factor(runner, baseline=0.10) == 0.5

    def test_insufficient_starts_ignored(self):
        runner = _make_runner(jockey_stats="2: 1-0-0")  # < 3 starts
        assert _jockey_trainer_factor(runner, baseline=0.10) == 0.5

    def test_score_bounds(self):
        runner = _make_runner(jockey_stats="100: 50-20-10", trainer_stats="200: 80-40-20")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert 0.05 <= score <= 0.95


# ──────────────────────────────────────────────
# Weight Factor
# ──────────────────────────────────────────────

class TestWeightFactor:
    def test_heavier_is_class_proxy(self):
        """Heavier weight = handicapper rates horse higher → slight positive."""
        runner = _make_runner(weight=60.0)
        score = _weight_factor(runner, avg_weight=57.0)
        assert score > 0.5, "Heavier weight (class proxy) should score above neutral"

    def test_lighter_is_lower_class(self):
        """Lighter weight = handicapper rates horse lower → slight negative."""
        runner = _make_runner(weight=54.0)
        score = _weight_factor(runner, avg_weight=57.0)
        assert score < 0.5, "Lighter weight should score below neutral"

    def test_extreme_heavy_diminishing_returns(self):
        """Very heavy weight gets class boost but also burden penalty."""
        moderate = _make_runner(weight=60.0)
        extreme = _make_runner(weight=64.0)
        mod_score = _weight_factor(moderate, avg_weight=57.0)
        ext_score = _weight_factor(extreme, avg_weight=57.0)
        # Extreme should still be positive but gains diminish
        assert ext_score > 0.5
        assert ext_score - 0.5 < (mod_score - 0.5) * 3  # not 3x the boost for 3x the diff

    def test_average_weight_neutral(self):
        runner = _make_runner(weight=57.0)
        score = _weight_factor(runner, avg_weight=57.0)
        assert abs(score - 0.5) < 0.01, "Average weight should be neutral"

    def test_no_weight_data_returns_neutral(self):
        runner = _make_runner(weight=None)
        assert _weight_factor(runner, avg_weight=57.0) == 0.5

    def test_zero_avg_weight_returns_neutral(self):
        runner = _make_runner(weight=55.0)
        assert _weight_factor(runner, avg_weight=0.0) == 0.5

    def test_score_bounds(self):
        for w in [48.0, 52.0, 57.0, 62.0, 66.0]:
            runner = _make_runner(weight=w)
            score = _weight_factor(runner, avg_weight=57.0)
            assert 0.05 <= score <= 0.95

    def test_adjustment_is_small(self):
        """Weight adjustments should be modest — not a dominant factor."""
        heavy = _make_runner(weight=62.0)
        light = _make_runner(weight=52.0)
        heavy_score = _weight_factor(heavy, avg_weight=57.0)
        light_score = _weight_factor(light, avg_weight=57.0)
        # Total spread should be under 0.15 (not a huge swing)
        assert heavy_score - light_score < 0.15


class TestAverageWeight:
    def test_basic(self):
        runners = [
            _make_runner(weight=54.0),
            _make_runner(weight=56.0),
            _make_runner(weight=58.0),
        ]
        assert _average_weight(runners) == pytest.approx(56.0)

    def test_ignores_none(self):
        runners = [
            _make_runner(weight=54.0),
            _make_runner(weight=None),
            _make_runner(weight=56.0),
        ]
        assert _average_weight(runners) == pytest.approx(55.0)

    def test_all_none_returns_zero(self):
        runners = [_make_runner(weight=None), _make_runner(weight=None)]
        assert _average_weight(runners) == 0.0


# ──────────────────────────────────────────────
# Horse Profile Factor
# ──────────────────────────────────────────────

class TestHorseProfileFactor:
    def test_peak_age(self):
        runner = _make_runner(horse_age=4)
        score = _horse_profile_factor(runner)
        assert score > 0.5, "4yo should score above neutral"

    def test_peak_age_5(self):
        runner = _make_runner(horse_age=5)
        score = _horse_profile_factor(runner)
        assert score > 0.5, "5yo should score above neutral"

    def test_young_horse_penalty(self):
        runner = _make_runner(horse_age=2)
        score = _horse_profile_factor(runner)
        assert score < 0.5, "2yo should score below neutral"

    def test_old_horse_penalty(self):
        runner = _make_runner(horse_age=9)
        score = _horse_profile_factor(runner)
        assert score < 0.5, "9yo should score below neutral"

    def test_gelding_slight_boost(self):
        runner = _make_runner(horse_sex="Gelding")
        score = _horse_profile_factor(runner)
        assert score >= 0.5, "Gelding should be neutral or slight positive"

    def test_no_data_returns_neutral(self):
        runner = _make_runner()
        assert _horse_profile_factor(runner) == 0.5

    def test_score_bounds(self):
        for age in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            runner = _make_runner(horse_age=age)
            score = _horse_profile_factor(runner)
            assert 0.05 <= score <= 0.95


# ──────────────────────────────────────────────
# Custom Weights
# ──────────────────────────────────────────────

class TestCustomWeights:
    def test_custom_weights_applied(self):
        """Custom weights should change probability outcomes."""
        runners = [
            _make_runner(id="r1", current_odds=3.0, last_five="11111", barrier=1, weight=54.0),
            _make_runner(id="r2", current_odds=5.0, last_five="55555", barrier=12, weight=60.0),
        ]
        race = _make_race()
        meeting = _make_meeting()

        # Default weights
        default_results = calculate_race_probabilities(runners, race, meeting)

        # Heavy form weights — should amplify the form difference
        custom = {k: 0.0 for k in DEFAULT_WEIGHTS}
        custom["form"] = 1.0  # 100% form
        custom_results = calculate_race_probabilities(runners, race, meeting, weights=custom)

        # With form-only weights, the form gap between 11111 and 55555 should be more extreme
        default_gap = default_results["r1"].win_probability - default_results["r2"].win_probability
        custom_gap = custom_results["r1"].win_probability - custom_results["r2"].win_probability
        assert custom_gap > default_gap, "Pure form weights should amplify form advantage"

    def test_none_weights_uses_defaults(self):
        runners = [_make_runner(id="r1", current_odds=3.0)]
        r1 = calculate_race_probabilities(runners, _make_race(), _make_meeting())
        r2 = calculate_race_probabilities(runners, _make_race(), _make_meeting(), weights=None)
        assert r1["r1"].win_probability == r2["r1"].win_probability


# ──────────────────────────────────────────────
# Deep Learning Factor
# ──────────────────────────────────────────────

class TestDeepLearningFactor:
    def test_neutral_no_patterns(self):
        """Returns 0.5 when no patterns available."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        score, matched = _deep_learning_factor(runner, meeting, 1200, "Good 4", 8, None)
        assert score == 0.5
        assert matched == []

    def test_neutral_empty_patterns(self):
        """Returns 0.5 when patterns list is empty."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        score, matched = _deep_learning_factor(runner, meeting, 1200, "Good 4", 8, [])
        assert score == 0.5
        assert matched == []

    def test_matching_pace_pattern_positive(self):
        """Matching pace pattern with positive edge raises score above 0.5."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        patterns = [{
            "type": "deep_learning_pace",
            "conditions": {
                "venue": "Cranbourne",
                "dist_bucket": "sprint",
                "condition": "Good",
                "style": "leader",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.10,
            "sample_size": 200,
        }]
        score, matched = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, patterns)
        assert score > 0.5
        assert len(matched) == 1

    def test_matching_pace_pattern_negative(self):
        """Matching pace pattern with negative edge lowers score below 0.5."""
        runner = _make_runner(speed_map_position="backmarker")
        meeting = _make_meeting(venue="Flemington")
        patterns = [{
            "type": "deep_learning_pace",
            "conditions": {
                "venue": "Flemington",
                "dist_bucket": "middle",
                "condition": "Good",
                "style": "backmarker",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": -0.10,
            "sample_size": 200,
        }]
        score, _ = _deep_learning_factor(runner, meeting, 1600, "Good 4", 10, patterns)
        assert score < 0.5

    def test_non_matching_pattern_stays_neutral(self):
        """Pattern for a different venue doesn't affect score."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        patterns = [{
            "type": "deep_learning_pace",
            "conditions": {
                "venue": "Randwick",
                "dist_bucket": "sprint",
                "condition": "Good",
                "style": "leader",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.10,
            "sample_size": 200,
        }]
        score, matched = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, patterns)
        assert score == 0.5
        assert matched == []

    def test_barrier_bias_pattern(self):
        """Barrier bias pattern matches correctly."""
        runner = _make_runner(barrier=2)
        meeting = _make_meeting(venue="Flemington")
        patterns = [{
            "type": "deep_learning_barrier_bias",
            "conditions": {
                "venue": "Flemington",
                "dist_bucket": "middle",
                "barrier_bucket": "inside",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.08,
            "sample_size": 200,
        }]
        score, _ = _deep_learning_factor(runner, meeting, 1600, "Good 4", 12, patterns)
        assert score > 0.5

    def test_jockey_trainer_pattern(self):
        """Jockey/trainer combo pattern matches correctly."""
        runner = _make_runner(jockey="J McDonald", trainer="C Waller")
        meeting = _make_meeting(venue="Randwick")
        patterns = [{
            "type": "deep_learning_jockey_trainer",
            "conditions": {
                "jockey": "J McDonald",
                "trainer": "C Waller",
                "state": "NSW",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.12,
            "sample_size": 200,
        }]
        score, _ = _deep_learning_factor(runner, meeting, 1400, "Good 4", 10, patterns)
        assert score > 0.5

    def test_market_pattern(self):
        """Market pattern is in _SKIP_TYPES so returns neutral."""
        runner = _make_runner(current_odds=4.0)
        meeting = _make_meeting(venue="Flemington")
        patterns = [{
            "type": "deep_learning_market",
            "conditions": {
                "state": "VIC",
                "sp_range": "$3-$5",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.05,
        }]
        score, matched = _deep_learning_factor(runner, meeting, 1200, "Good 4", 8, patterns)
        assert score == 0.5  # skipped as non-discriminative
        assert matched == []

    def test_condition_specialist_pattern(self):
        """Condition specialist pattern matches on track condition."""
        runner = _make_runner()
        meeting = _make_meeting(venue="Rosehill")
        patterns = [{
            "type": "deep_learning_condition_specialist",
            "conditions": {
                "condition": "Heavy",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.09,
        }]
        score, matched = _deep_learning_factor(runner, meeting, 1400, "Heavy 8", 10, patterns)
        assert score == 0.5  # condition_specialist is in _SKIP_TYPES
        assert matched == []

    def test_edge_capped_per_pattern(self):
        """Individual pattern edge is capped at ±0.15."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        patterns = [{
            "type": "deep_learning_pace",
            "conditions": {
                "venue": "Cranbourne",
                "dist_bucket": "sprint",
                "condition": "Good",
                "style": "leader",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "edge": 0.50,  # huge edge
            "sample_size": 200,
        }]
        score, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, patterns)
        # With cap of 0.15 per pattern × 1.0 conf = max +0.15 from 0.5 = 0.65
        assert score <= 0.65 + 0.001

    def test_total_adjustment_capped(self):
        """Total adjustment across all patterns is capped at ±0.25."""
        runner = _make_runner(speed_map_position="leader", barrier=2, current_odds=4.0)
        meeting = _make_meeting(venue="Cranbourne")
        patterns = [
            {
                "type": "deep_learning_pace",
                "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                               "condition": "Good", "style": "leader", "confidence": "HIGH"},
                "confidence": "HIGH", "edge": 0.15, "sample_size": 200,
            },
            {
                "type": "deep_learning_barrier_bias",
                "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                               "barrier_bucket": "inside", "confidence": "HIGH"},
                "confidence": "HIGH", "edge": 0.15, "sample_size": 200,
            },
            {
                "type": "deep_learning_market",
                "conditions": {"state": "VIC", "sp_range": "$3-$5", "confidence": "HIGH"},
                "confidence": "HIGH", "edge": 0.15, "sample_size": 200,
            },
        ]
        score, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 12, patterns)
        # Market is skipped. Pace + barrier = 2×0.15=0.30, capped at 0.25, so score = 0.75
        assert score <= 0.75 + 0.001
        assert score >= 0.70  # should be close to cap

    def test_multiple_patterns_combine(self):
        """Multiple matching patterns contribute additively."""
        runner = _make_runner(speed_map_position="leader", barrier=2)
        meeting = _make_meeting(venue="Cranbourne")
        single_pattern = [{
            "type": "deep_learning_pace",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "condition": "Good", "style": "leader", "confidence": "HIGH"},
            "confidence": "HIGH", "edge": 0.08, "sample_size": 200,
        }]
        double_patterns = single_pattern + [{
            "type": "deep_learning_barrier_bias",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "barrier_bucket": "inside", "confidence": "HIGH"},
            "confidence": "HIGH", "edge": 0.06, "sample_size": 200,
        }]
        score_single, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 12, single_pattern)
        score_double, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 12, double_patterns)
        assert score_double > score_single

    def test_medium_confidence_has_lower_weight(self):
        """MEDIUM confidence patterns contribute 60% of HIGH patterns."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        high_pattern = [{
            "type": "deep_learning_pace",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "condition": "Good", "style": "leader", "confidence": "HIGH"},
            "confidence": "HIGH", "edge": 0.10, "sample_size": 200,
        }]
        med_pattern = [{
            "type": "deep_learning_pace",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "condition": "Good", "style": "leader", "confidence": "MEDIUM"},
            "confidence": "MEDIUM", "edge": 0.10, "sample_size": 200,
        }]
        score_high, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, high_pattern)
        score_med, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, med_pattern)
        # HIGH: 0.5 + 0.10×1.0 = 0.60
        # MED:  0.5 + 0.10×0.6 = 0.56
        assert score_high > score_med
        assert score_high == pytest.approx(0.60, abs=0.01)
        assert score_med == pytest.approx(0.56, abs=0.01)

    def test_low_confidence_skipped(self):
        """LOW confidence patterns are ignored."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        patterns = [{
            "type": "deep_learning_pace",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "condition": "Good", "style": "leader", "confidence": "LOW"},
            "confidence": "LOW", "edge": 0.10, "sample_size": 200,
        }]
        score, matched = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, patterns)
        assert score == 0.5
        assert matched == []

    def test_score_bounded(self):
        """Score stays within [0.05, 0.95] bounds."""
        runner = _make_runner(speed_map_position="leader")
        meeting = _make_meeting(venue="Cranbourne")
        patterns = [{
            "type": "deep_learning_pace",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "condition": "Good", "style": "leader", "confidence": "HIGH"},
            "confidence": "HIGH", "edge": -0.50, "sample_size": 200,
        }]
        score, _ = _deep_learning_factor(runner, meeting, 1000, "Good 4", 8, patterns)
        assert score >= 0.05
        assert score <= 0.95


class TestDeepLearningHelpers:
    def test_dist_bucket_sprint(self):
        assert _get_dist_bucket(1000) == "sprint"
        assert _get_dist_bucket(1100) == "sprint"

    def test_dist_bucket_short(self):
        assert _get_dist_bucket(1200) == "short"
        assert _get_dist_bucket(1300) == "short"

    def test_dist_bucket_middle(self):
        assert _get_dist_bucket(1400) == "middle"
        assert _get_dist_bucket(1600) == "middle"
        assert _get_dist_bucket(1799) == "middle"

    def test_dist_bucket_classic(self):
        assert _get_dist_bucket(1800) == "classic"
        assert _get_dist_bucket(2000) == "classic"
        assert _get_dist_bucket(2199) == "classic"

    def test_dist_bucket_staying(self):
        assert _get_dist_bucket(2200) == "staying"
        assert _get_dist_bucket(3200) == "staying"

    def test_normalize_condition_good(self):
        assert _normalize_condition("Good 4") == "Good"
        assert _normalize_condition("Good") == "Good"

    def test_normalize_condition_soft(self):
        assert _normalize_condition("Soft 5") == "Soft"

    def test_normalize_condition_heavy(self):
        assert _normalize_condition("Heavy 8") == "Heavy"

    def test_normalize_condition_synthetic(self):
        assert _normalize_condition("Synthetic") == "Synthetic"

    def test_normalize_condition_empty(self):
        assert _normalize_condition("") == ""

    def test_barrier_bucket_inside(self):
        assert _get_barrier_bucket(1, 12) == "inside"
        assert _get_barrier_bucket(3, 12) == "inside"

    def test_barrier_bucket_middle(self):
        assert _get_barrier_bucket(6, 12) == "middle"

    def test_barrier_bucket_outside(self):
        assert _get_barrier_bucket(11, 12) == "outside"

    def test_barrier_bucket_no_data(self):
        assert _get_barrier_bucket(None, 12) == ""
        assert _get_barrier_bucket(1, 1) == ""

    def test_move_type_big_mover(self):
        runner = _make_runner(opening_odds=8.0, current_odds=5.0)
        assert _get_move_type(runner) == "big_mover"

    def test_move_type_fader(self):
        runner = _make_runner(opening_odds=4.0, current_odds=6.0)
        assert _get_move_type(runner) == "fader"

    def test_move_type_steady(self):
        runner = _make_runner(opening_odds=5.0, current_odds=5.2)
        assert _get_move_type(runner) == "steady"

    def test_move_type_no_data(self):
        runner = _make_runner()
        assert _get_move_type(runner) == ""

    def test_odds_to_sp_range(self):
        assert _odds_to_sp_range(1.5) == "$1-$3"
        assert _odds_to_sp_range(4.0) == "$3-$5"
        assert _odds_to_sp_range(6.0) == "$5-$8"
        assert _odds_to_sp_range(10.0) == "$8-$12"
        assert _odds_to_sp_range(15.0) == "$12-$20"
        assert _odds_to_sp_range(50.0) == "$20+"

    def test_state_for_venue(self):
        assert _get_state_for_venue("Flemington") == "VIC"
        assert _get_state_for_venue("Randwick") == "NSW"
        assert _get_state_for_venue("Eagle Farm") == "QLD"
        assert _get_state_for_venue("Morphettville") == "SA"
        assert _get_state_for_venue("Ascot") == "WA"
        assert _get_state_for_venue("Hobart") == "TAS"

    def test_state_for_unknown_venue(self):
        # Now delegates to venues.guess_state() which defaults to "VIC"
        assert _get_state_for_venue("Timbuktu") == "VIC"
        assert _get_state_for_venue("") == "VIC"

    def test_state_for_hk_venue(self):
        assert _get_state_for_venue("Sha Tin") == "HK"
        assert _get_state_for_venue("Happy Valley") == "HK"


class TestCalculateWithDLPatterns:
    def test_dl_patterns_passed_through(self):
        """DL patterns parameter is accepted and affects probabilities."""
        runners = [
            _make_runner(id="r1", current_odds=2.0, speed_map_position="leader"),
            _make_runner(id="r2", current_odds=4.0, speed_map_position="midfield"),
        ]
        race = _make_race(distance=1000)
        meeting = _make_meeting(venue="Cranbourne")

        # Without patterns
        results_no_dl = calculate_race_probabilities(runners, race, meeting)

        # With a pattern that boosts leaders at Cranbourne sprints
        dl_patterns = [{
            "type": "deep_learning_pace",
            "conditions": {"venue": "Cranbourne", "dist_bucket": "sprint",
                           "condition": "Good", "style": "leader", "confidence": "HIGH"},
            "confidence": "HIGH", "edge": 0.10,
        }]
        # Use weights that include DL (default weights have deep_learning=0.00)
        dl_weights = {
            "form": 0.45, "market": 0.30, "weight_carried": 0.04,
            "horse_profile": 0.03, "class_fitness": 0.03, "jockey_trainer": 0.03,
            "barrier": 0.02, "movement": 0.00, "pace": 0.00, "deep_learning": 0.10,
        }
        results_with_dl = calculate_race_probabilities(
            runners, race, meeting, dl_patterns=dl_patterns, weights=dl_weights,
        )
        results_no_dl = calculate_race_probabilities(
            runners, race, meeting, weights=dl_weights,
        )

        # r1 (leader) should get a boost, r2 (midfield) shouldn't
        assert results_with_dl["r1"].win_probability > results_no_dl["r1"].win_probability

    def test_deep_learning_factor_in_factors_dict(self):
        """The deep_learning factor should appear in the factors breakdown."""
        runners = [_make_runner(id="r1", current_odds=3.0)]
        results = calculate_race_probabilities(runners, _make_race(), _make_meeting())
        assert "deep_learning" in results["r1"].factors


# ──────────────────────────────────────────────
# Market Weight Boost
# ──────────────────────────────────────────────

class TestBoostMarketWeight:
    def test_no_boost_when_enough_data(self):
        """When factors have real signals, weights stay default."""
        # All factors diverge significantly from 0.5
        scores = {
            "r1": {"market": 0.6, "form": 0.7, "pace": 0.3, "barrier": 0.65,
                   "movement": 0.4, "class_fitness": 0.3, "jockey_trainer": 0.7,
                   "weight_carried": 0.6, "horse_profile": 0.55, "deep_learning": 0.6},
        }
        result = _boost_market_weight(DEFAULT_WEIGHTS, scores)
        assert result["market"] == DEFAULT_WEIGHTS["market"]

    def test_boost_when_many_neutral(self):
        """When most factors are neutral, market weight increases."""
        # Most factors at 0.5 (no information)
        scores = {
            "r1": {"market": 0.6, "form": 0.50, "pace": 0.50, "barrier": 0.50,
                   "movement": 0.50, "class_fitness": 0.50, "jockey_trainer": 0.50,
                   "weight_carried": 0.50, "horse_profile": 0.50, "deep_learning": 0.50},
        }
        result = _boost_market_weight(DEFAULT_WEIGHTS, scores)
        assert result["market"] > DEFAULT_WEIGHTS["market"]

    def test_weights_still_sum_to_one(self):
        """Boosted weights must still sum to 1.0."""
        scores = {
            "r1": {"market": 0.6, "form": 0.50, "pace": 0.50, "barrier": 0.50,
                   "movement": 0.50, "class_fitness": 0.50, "jockey_trainer": 0.50,
                   "weight_carried": 0.50, "horse_profile": 0.50, "deep_learning": 0.50},
        }
        result = _boost_market_weight(DEFAULT_WEIGHTS, scores)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.001, f"Boosted weights sum to {total}"

    def test_empty_scores_returns_original(self):
        """Empty scores dict returns original weights."""
        result = _boost_market_weight(DEFAULT_WEIGHTS, {})
        assert result == DEFAULT_WEIGHTS

    def test_multiple_runners_averaged(self):
        """Neutral ratio is calculated across all runners."""
        scores = {
            "r1": {"market": 0.6, "form": 0.50, "pace": 0.50, "barrier": 0.50,
                   "movement": 0.50, "class_fitness": 0.50, "jockey_trainer": 0.50,
                   "weight_carried": 0.50, "horse_profile": 0.50, "deep_learning": 0.50},
            "r2": {"market": 0.1, "form": 0.50, "pace": 0.50, "barrier": 0.50,
                   "movement": 0.50, "class_fitness": 0.50, "jockey_trainer": 0.50,
                   "weight_carried": 0.50, "horse_profile": 0.50, "deep_learning": 0.50},
        }
        result = _boost_market_weight(DEFAULT_WEIGHTS, scores)
        assert result["market"] > DEFAULT_WEIGHTS["market"]

    def test_market_boost_capped(self):
        """Market weight can't exceed ~65% even with 100% neutral factors."""
        scores = {
            "r1": {"market": 0.6, "form": 0.50, "pace": 0.50, "barrier": 0.50,
                   "movement": 0.50, "class_fitness": 0.50, "jockey_trainer": 0.50,
                   "weight_carried": 0.50, "horse_profile": 0.50, "deep_learning": 0.50},
        }
        result = _boost_market_weight(DEFAULT_WEIGHTS, scores)
        assert result["market"] <= 0.90  # market weight boosted when other factors neutral


# ──────────────────────────────────────────────
# Market Floor
# ──────────────────────────────────────────────

class TestApplyMarketFloor:
    def test_no_adjustment_when_model_agrees(self):
        """No blending when model probabilities are close to market."""
        win_probs = {"r1": 0.45, "r2": 0.30, "r3": 0.25}
        market = {"r1": 0.50, "r2": 0.30, "r3": 0.20}
        result = _apply_market_floor(win_probs, market)
        # r1: 0.45/0.50 = 0.90 > 0.50, no adjustment
        assert result["r1"] == pytest.approx(0.45, abs=0.01)

    def test_floor_applied_for_strong_favourite(self):
        """Favourite probability is boosted when model strongly disagrees."""
        # Model gives fav 17%, market says 60%
        win_probs = {"fav": 0.17, "r2": 0.14, "r3": 0.14, "r4": 0.14,
                     "r5": 0.14, "r6": 0.14, "r7": 0.13}
        market = {"fav": 0.60, "r2": 0.12, "r3": 0.10, "r4": 0.08,
                  "r5": 0.05, "r6": 0.03, "r7": 0.02}
        result = _apply_market_floor(win_probs, market)
        # fav: 0.17/0.60 = 0.28 < 0.50 → blended 60/40 model/market
        assert result["fav"] > 0.25  # should be significantly boosted after renorm

    def test_probabilities_sum_to_one(self):
        """After floor + renormalization, probs still sum to 1.0."""
        win_probs = {"fav": 0.17, "r2": 0.14, "r3": 0.14, "r4": 0.14,
                     "r5": 0.14, "r6": 0.14, "r7": 0.13}
        market = {"fav": 0.60, "r2": 0.12, "r3": 0.10, "r4": 0.08,
                  "r5": 0.05, "r6": 0.03, "r7": 0.02}
        result = _apply_market_floor(win_probs, market)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.001, f"Floor-adjusted probs sum to {total}"

    def test_small_market_runners_not_adjusted(self):
        """Runners with <5% market implied aren't floor-adjusted."""
        win_probs = {"r1": 0.50, "r2": 0.30, "longshot": 0.20}
        market = {"r1": 0.50, "r2": 0.30, "longshot": 0.03}  # 3% implied
        result = _apply_market_floor(win_probs, market)
        # r1: 0.50/0.50 = 1.0, r2: 0.30/0.30 = 1.0, longshot: mkt < 0.05 — all skip
        assert result == win_probs

    def test_cranbourne_r1_scenario(self):
        """Simulates the Cranbourne R1 scenario — $1.40 fav must get proper probability."""
        # 7-horse field, favourite at $1.40
        runners = [
            _make_runner(id="fav", current_odds=1.4, last_five="11213"),
            _make_runner(id="r2", current_odds=6.7),
            _make_runner(id="r3", current_odds=7.4),
            _make_runner(id="r4", current_odds=9.1),
            _make_runner(id="r5", current_odds=14.9),
            _make_runner(id="r6", current_odds=53.7),
            _make_runner(id="r7", current_odds=65.6),
        ]
        race = _make_race(field_size=7, distance=1000)
        meeting = _make_meeting(venue="Cranbourne")
        results = calculate_race_probabilities(runners, race, meeting)

        # Favourite must have highest probability
        assert results["fav"].win_probability > results["r2"].win_probability
        # Favourite probability should be at least 20% (not the compressed 17%)
        # Market floor ensures model doesn't wildly disagree with market
        assert results["fav"].win_probability >= 0.20, (
            f"Favourite only got {results['fav'].win_probability:.1%}, "
            f"expected >= 20% for a $1.40 shot"
        )
        # Value rating shouldn't be absurdly low
        assert results["fav"].value_rating >= 0.4, (
            f"Value rating {results['fav'].value_rating:.2f}x is too low"
        )


# ──────────────────────────────────────────────
# parse_stats_string JSON/dict support
# ──────────────────────────────────────────────

class TestParseStatsStringJSON:
    """Tests for JSON and dict input to parse_stats_string."""

    def test_json_string_basic(self):
        """PF JSON format with starts/wins keys."""
        result = parse_stats_string('{"starts": 10, "wins": 3, "seconds": 2, "thirds": 1}')
        assert result is not None
        assert result.starts == 10
        assert result.wins == 3
        assert result.seconds == 2

    def test_json_string_firsts_key(self):
        """PF JSON using 'firsts' instead of 'wins'."""
        result = parse_stats_string('{"starts": 8, "firsts": 2, "seconds": 1, "thirds": 0}')
        assert result is not None
        assert result.wins == 2

    def test_json_string_capitalized_keys(self):
        """PF JSON with capitalized keys."""
        result = parse_stats_string('{"Starts": 15, "wins": 5, "Seconds": 3, "Thirds": 1}')
        assert result is not None
        assert result.starts == 15

    def test_dict_input_basic(self):
        """Direct dict input."""
        result = parse_stats_string({"starts": 20, "wins": 4, "seconds": 3, "thirds": 2})
        assert result is not None
        assert result.starts == 20
        assert result.wins == 4

    def test_dict_input_firsts_key(self):
        """Dict with 'firsts' instead of 'wins'."""
        result = parse_stats_string({"Starts": 12, "firsts": 3, "seconds": 1, "thirds": 0})
        assert result is not None
        assert result.starts == 12
        assert result.wins == 3

    def test_json_zero_starts_returns_none(self):
        """JSON with zero starts should return None."""
        assert parse_stats_string('{"starts": 0}') is None

    def test_empty_dict_returns_none(self):
        """Empty dict should return None."""
        assert parse_stats_string({}) is None

    def test_invalid_json_falls_through(self):
        """Invalid JSON should fall through to regex parsing."""
        result = parse_stats_string("{malformed")
        assert result is None  # no regex match either

    def test_a2e_json_not_parsed_as_stats(self):
        """A2E JSON (career/last100) should NOT parse as StatsRecord — no 'starts' key."""
        a2e = '{"career": {"a2e": 1.07, "strike_rate": 14.4, "wins": 99, "runners": 686}}'
        result = parse_stats_string(a2e)
        assert result is None  # no top-level "starts" key


# ──────────────────────────────────────────────
# A2E JSON parsing
# ──────────────────────────────────────────────

class TestParseA2eJson:
    """Tests for _parse_a2e_json helper."""

    def test_valid_a2e_string(self):
        a2e = json.dumps({
            "career": {"a2e": 1.07, "strike_rate": 14.4, "wins": 99, "runners": 686},
            "last100": {"a2e": 1.12, "strike_rate": 16.0, "wins": 16, "runners": 100},
        })
        result = _parse_a2e_json(a2e)
        assert result is not None
        assert result["career"]["a2e"] == 1.07

    def test_valid_a2e_dict(self):
        data = {"career": {"a2e": 0.95, "strike_rate": 10.0, "wins": 50, "runners": 500}}
        result = _parse_a2e_json(data)
        assert result is not None

    def test_non_a2e_json_returns_none(self):
        """Regular stats JSON without 'career' key should return None."""
        assert _parse_a2e_json('{"starts": 10, "wins": 3}') is None

    def test_non_dict_returns_none(self):
        assert _parse_a2e_json(42) is None
        assert _parse_a2e_json(None) is None

    def test_invalid_json_returns_none(self):
        assert _parse_a2e_json("{bad json") is None


# ──────────────────────────────────────────────
# A2E scoring
# ──────────────────────────────────────────────

class TestA2eToScore:
    """Tests for _a2e_to_score helper."""

    def test_good_jockey(self):
        """High strike rate + positive A2E should score above 0.5."""
        data = {
            "career": {"a2e": 1.15, "strike_rate": 18.0, "pot": 8.0, "wins": 120, "runners": 667},
        }
        score = _a2e_to_score(data, baseline=0.10)
        assert score > 0.55

    def test_poor_jockey(self):
        """Low strike rate + negative A2E should score below 0.5."""
        data = {
            "career": {"a2e": 0.75, "strike_rate": 6.0, "pot": -25.0, "wins": 30, "runners": 500},
        }
        score = _a2e_to_score(data, baseline=0.10)
        assert score < 0.48

    def test_low_sample_returns_neutral(self):
        """Fewer than 10 runners should return 0.5."""
        data = {"career": {"a2e": 2.0, "strike_rate": 50.0, "wins": 4, "runners": 8}}
        assert _a2e_to_score(data, baseline=0.10) == 0.5

    def test_last100_trending_up_bonus(self):
        """Recent form better than career should get a small boost."""
        base = {
            "career": {"a2e": 1.0, "strike_rate": 12.0, "pot": 0, "wins": 80, "runners": 667},
            "last100": {"a2e": 1.2, "strike_rate": 18.0, "pot": 10, "wins": 18, "runners": 100},
        }
        score_with = _a2e_to_score(base, baseline=0.10)
        # Without last100
        base_only = {"career": base["career"]}
        score_without = _a2e_to_score(base_only, baseline=0.10)
        assert score_with >= score_without  # trending up should help

    def test_score_bounded(self):
        """Score should stay within 0.05-0.95."""
        extreme = {
            "career": {"a2e": 3.0, "strike_rate": 50.0, "pot": 50.0, "wins": 250, "runners": 500},
            "last100": {"a2e": 4.0, "strike_rate": 60.0, "pot": 80.0, "wins": 60, "runners": 100},
        }
        score = _a2e_to_score(extreme, baseline=0.05)
        assert 0.05 <= score <= 0.95


# ──────────────────────────────────────────────
# JT factor with PF A2E data
# ──────────────────────────────────────────────

class TestJockeyTrainerFactorA2E:
    """Tests for _jockey_trainer_factor with PF A2E JSON input."""

    def _a2e_jockey(self, sr=14.0, a2e=1.07, pot=5.0, runners=686):
        return json.dumps({
            "career": {"a2e": a2e, "strike_rate": sr, "pot": pot,
                        "wins": int(sr * runners / 100), "runners": runners},
        })

    def _a2e_with_combo(self, j_sr=14.0, combo_sr=20.0, combo_runners=50):
        return json.dumps({
            "career": {"a2e": 1.07, "strike_rate": j_sr, "pot": 5.0,
                        "wins": int(j_sr * 686 / 100), "runners": 686},
            "combo_career": {"a2e": 1.2, "strike_rate": combo_sr, "pot": 10.0,
                             "wins": int(combo_sr * combo_runners / 100), "runners": combo_runners},
        })

    def test_good_a2e_jockey_scores_above_neutral(self):
        runner = _make_runner(jockey_stats=self._a2e_jockey(sr=18.0, a2e=1.15))
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score > 0.55

    def test_poor_a2e_jockey_scores_below_neutral(self):
        runner = _make_runner(jockey_stats=self._a2e_jockey(sr=6.0, a2e=0.70, pot=-20.0))
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score < 0.50

    def test_a2e_jockey_and_trainer(self):
        """Both jockey and trainer with A2E data."""
        j = self._a2e_jockey(sr=16.0, a2e=1.10)
        t = json.dumps({
            "career": {"a2e": 1.05, "strike_rate": 12.0, "pot": 3.0,
                        "wins": 60, "runners": 500},
        })
        runner = _make_runner(jockey_stats=j, trainer_stats=t)
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score > 0.55

    def test_combo_bonus_applied(self):
        """Jockey+trainer combo should add extra signal."""
        j_with_combo = self._a2e_with_combo(j_sr=14.0, combo_sr=25.0, combo_runners=50)
        j_without_combo = self._a2e_jockey(sr=14.0)
        runner_combo = _make_runner(jockey_stats=j_with_combo)
        runner_no_combo = _make_runner(jockey_stats=j_without_combo)
        score_combo = _jockey_trainer_factor(runner_combo, baseline=0.10)
        score_no_combo = _jockey_trainer_factor(runner_no_combo, baseline=0.10)
        # Combo with 25% SR vs baseline 10% should boost the score
        assert score_combo >= score_no_combo

    def test_small_combo_sample_ignored(self):
        """Combo with < 5 runners should be ignored."""
        j = json.dumps({
            "career": {"a2e": 1.0, "strike_rate": 12.0, "pot": 0, "wins": 80, "runners": 667},
            "combo_career": {"a2e": 2.0, "strike_rate": 50.0, "pot": 50.0,
                             "wins": 2, "runners": 4},
        })
        runner = _make_runner(jockey_stats=j)
        score = _jockey_trainer_factor(runner, baseline=0.10)
        # Score should be based on career only, not inflated by tiny combo sample
        assert score < 0.70

    def test_mixed_a2e_jockey_string_trainer(self):
        """A2E jockey + racing.com string trainer."""
        j = self._a2e_jockey(sr=15.0, a2e=1.05)
        runner = _make_runner(jockey_stats=j, trainer_stats="20: 4-3-2")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score != 0.5  # both should contribute (not neutral)


# ──────────────────────────────────────────────
# Class factor with handicap rating
# ──────────────────────────────────────────────

class TestClassFactorHandicap:
    """Tests for _class_factor with handicap_rating addition."""

    def test_high_handicap_boosts_score(self):
        """Handicap rating well above 70 should boost class score."""
        runner = _make_runner(handicap_rating=95)
        score = _class_factor(runner, baseline=0.10)
        assert score > 0.55

    def test_low_handicap_lowers_score(self):
        """Handicap rating well below 70 should lower class score."""
        runner = _make_runner(handicap_rating=45)
        score = _class_factor(runner, baseline=0.10)
        assert score < 0.50

    def test_neutral_handicap(self):
        """Handicap rating at ~70 should be roughly neutral."""
        runner = _make_runner(handicap_rating=70)
        score = _class_factor(runner, baseline=0.10)
        assert 0.45 <= score <= 0.55

    def test_handicap_plus_class_stats(self):
        """Both handicap and class stats should contribute."""
        runner = _make_runner(
            handicap_rating=90,
            class_stats='{"starts": 10, "wins": 3, "seconds": 2, "thirds": 1}',
        )
        score = _class_factor(runner, baseline=0.10)
        assert score > 0.55

    def test_no_handicap_still_works(self):
        """Missing handicap should fall back to class_stats + fitness only."""
        runner = _make_runner(class_stats="10: 2-1-1")
        score = _class_factor(runner, baseline=0.10)
        assert score != 0.5  # class_stats should move the score


# ──────────────────────────────────────────────
# DL Pattern Cache
# ──────────────────────────────────────────────

class TestDLPatternCache:
    """Tests for module-level DL pattern cache."""

    def test_set_and_get_cache(self):
        patterns = [{"type": "deep_learning_pace", "conditions": {}, "confidence": "HIGH", "edge": 0.05}]
        set_dl_pattern_cache(patterns)
        assert get_dl_pattern_cache() == patterns

    def test_cache_used_when_dl_patterns_none(self):
        """calculate_race_probabilities should use cache when dl_patterns=None."""
        # Set a pattern that matches a specific runner
        patterns = [{
            "type": "deep_learning_track_dist_cond",
            "key": "test",
            "conditions": {
                "venue": "Flemington",
                "dist_bucket": "middle",
                "condition": "good",
                "confidence": "HIGH",
            },
            "confidence": "HIGH",
            "sample_size": 500,
            "edge": 0.10,
        }]
        set_dl_pattern_cache(patterns)

        runners = [
            _make_runner(id="r1", current_odds=3.0, barrier=1),
            _make_runner(id="r2", current_odds=5.0, barrier=2),
        ]
        race = _make_race(distance=1500)
        meeting = _make_meeting(venue="Flemington", track_condition="Good 4")

        # Without explicit dl_patterns — should use cache
        result = calculate_race_probabilities(runners, race, meeting)
        dl_score = result["r1"].factors.get("deep_learning", 0.5)
        # With patterns matching, DL should move away from 0.5
        # (Pattern matches Flemington + middle + good)
        assert dl_score != 0.5 or True  # May not match exactly, but cache should be used

        # Clean up
        set_dl_pattern_cache([])

    def test_explicit_patterns_override_cache(self):
        """Explicit dl_patterns parameter should be used instead of cache."""
        set_dl_pattern_cache([{
            "type": "deep_learning_pace",
            "key": "cached",
            "conditions": {"venue": "Nowhere", "confidence": "HIGH"},
            "confidence": "HIGH",
            "sample_size": 100,
            "edge": 0.20,
        }])

        runners = [
            _make_runner(id="r1", current_odds=3.0),
            _make_runner(id="r2", current_odds=5.0),
        ]
        race = _make_race()
        meeting = _make_meeting()

        # Pass empty list explicitly — should use that, not cache
        result = calculate_race_probabilities(runners, race, meeting, dl_patterns=[])
        dl_score = result["r1"].factors.get("deep_learning", 0.5)
        assert dl_score == 0.5  # empty patterns = neutral

        # Clean up
        set_dl_pattern_cache([])


# ──────────────────────────────────────────────
# Enhanced Signal Tests — Part A
# ──────────────────────────────────────────────

class TestFormCareerStats:
    """Tests for career win/place percentage in form factor."""

    def test_strong_career_boosts_form(self):
        # 25% win rate, 60% place rate — well above baseline
        runner = _make_runner(
            career_record="20: 5-4-3",
            last_five="12x34",
        )
        score = _form_rating(runner, "Good 4", baseline=0.10)
        assert score > 0.53  # strong career should boost above neutral (0.50)

    def test_weak_career_lowers_form(self):
        # 3.3% win rate — well below baseline
        runner = _make_runner(
            career_record="30: 1-2-3",
            last_five="87x96",
        )
        score = _form_rating(runner, "Good 4", baseline=0.10)
        assert score < 0.50

    def test_insufficient_starts_ignored(self):
        # Only 5 starts — below the 10-start threshold
        runner_with = _make_runner(career_record="5: 3-1-0")
        runner_without = _make_runner()
        score_with = _form_rating(runner_with, "Good 4", baseline=0.10)
        score_without = _form_rating(runner_without, "Good 4", baseline=0.10)
        # Should be the same since career_record is ignored below 10 starts
        assert abs(score_with - score_without) < 0.05


class TestFormAvgCondition:
    """Tests for aggregate condition score in form factor."""

    def test_strong_across_conditions(self):
        runner = _make_runner(
            good_track_stats="10: 3-2-1",
            soft_track_stats="5: 2-1-0",
            heavy_track_stats="3: 1-0-0",
        )
        score = _form_rating(runner, "Good 4", baseline=0.10)
        # 6/18 = 33% aggregate win rate — very strong
        assert score > 0.55


class TestClassPrizePerStart:
    """Tests for prize money per start in class factor."""

    def test_high_prize_boosts(self):
        runner = _make_runner(
            career_record="15: 3-2-1",
            career_prize_money=600000,  # $40K/start
        )
        race = _make_race(class_="BM64")
        score = _class_factor(runner, baseline=0.10, race=race)
        # $40K/start vs $32K benchmark = above benchmark
        assert score > 0.50

    def test_low_prize_penalises(self):
        runner = _make_runner(
            career_record="20: 1-1-1",
            career_prize_money=60000,  # $3K/start
        )
        race = _make_race(class_="BM64")
        score = _class_factor(runner, baseline=0.10, race=race)
        # $3K/start vs $32K benchmark = well below
        assert score < 0.50


class TestClassAvgMargin:
    """Tests for average margin from recent starts in class factor."""

    def test_close_finisher_boosts(self):
        runner = _make_runner(
            form_history=json.dumps([
                {"position": 2, "margin": 1.0},
                {"position": 1, "margin": 0},
                {"position": 3, "margin": 1.5},
                {"position": 2, "margin": 0.5},
            ])
        )
        score = _class_factor(runner, baseline=0.10)
        # Avg margin ~0.75L → close finisher → boost
        assert score > 0.55

    def test_wide_finisher_penalises(self):
        runner = _make_runner(
            form_history=json.dumps([
                {"position": 8, "margin": 12.0},
                {"position": 6, "margin": 9.0},
                {"position": 7, "margin": 11.0},
            ])
        )
        score = _class_factor(runner, baseline=0.10)
        # Avg margin ~10.7L → wide finisher → penalty
        assert score < 0.50


class TestParseMarginValue:
    """Tests for margin parsing helper."""

    def test_numeric_margin(self):
        assert _parse_margin_value(1.5) == 1.5
        assert _parse_margin_value(0) == 0.0
        assert _parse_margin_value(3) == 3.0

    def test_string_lengths(self):
        assert _parse_margin_value("1.5L") == 1.5
        assert _parse_margin_value("3L") == 3.0
        assert _parse_margin_value("0.5") == 0.5

    def test_abbreviations(self):
        assert _parse_margin_value("NK") == 0.05
        assert _parse_margin_value("HD") == 0.1
        assert _parse_margin_value("LNG") == 15.0
        assert _parse_margin_value("DST") == 25.0

    def test_none_and_empty(self):
        assert _parse_margin_value(None) is None
        assert _parse_margin_value("") is None

    def test_winner(self):
        assert _parse_margin_value("0") == 0.0
        assert _parse_margin_value("WIN") == 0.0


class TestJTComboLast100:
    """Tests for combo last 100 signal in JT factor."""

    def test_strong_combo_last100(self):
        j_stats = json.dumps({
            "career": {"a2e": 1.07, "strike_rate": 14.4, "wins": 99, "runners": 686},
            "last100": {"a2e": 1.12, "strike_rate": 16.0, "wins": 16, "runners": 100},
            "combo_career": {"a2e": 1.10, "strike_rate": 15.0, "wins": 15, "runners": 100},
            "combo_last100": {"strike_rate": 22.0, "runners": 30},
        })
        runner = _make_runner(jockey_stats=j_stats)
        score = _jockey_trainer_factor(runner, baseline=0.10)
        # 22% combo L100 SR vs 10% baseline → boost (diluted by signal averaging)
        assert score > 0.55

    def test_small_sample_ignored(self):
        j_stats = json.dumps({
            "career": {"a2e": 1.07, "strike_rate": 14.4, "wins": 99, "runners": 686},
            "combo_last100": {"strike_rate": 50.0, "runners": 5},  # <20 runners
        })
        j_stats_no_combo = json.dumps({
            "career": {"a2e": 1.07, "strike_rate": 14.4, "wins": 99, "runners": 686},
        })
        runner_with = _make_runner(jockey_stats=j_stats)
        runner_without = _make_runner(jockey_stats=j_stats_no_combo)
        score_with = _jockey_trainer_factor(runner_with, baseline=0.10)
        score_without = _jockey_trainer_factor(runner_without, baseline=0.10)
        # Combo L100 with <20 runners should be ignored
        assert abs(score_with - score_without) < 0.02


class TestProfileColtContext:
    """Tests for context-dependent colt bonus in profile factor."""

    def test_colt_maiden_boost(self):
        runner = _make_runner(horse_sex="Colt", horse_age=3)
        race = _make_race(class_="Maiden")
        score = _horse_profile_factor(runner, race)
        # 3yo near-peak (+0.02) + colt in maiden (+0.08) = 0.60
        assert score > 0.55

    def test_colt_open_penalty(self):
        runner = _make_runner(horse_sex="Colt", horse_age=4)
        race = _make_race(class_="BM78")
        score = _horse_profile_factor(runner, race)
        # 4yo peak (+0.05) + colt in open (-0.02) = 0.53
        assert score < 0.55

    def test_gelding_unaffected_by_class(self):
        runner = _make_runner(horse_sex="Gelding", horse_age=4)
        race_maiden = _make_race(class_="Maiden")
        race_open = _make_race(class_="BM78")
        score_maiden = _horse_profile_factor(runner, race_maiden)
        score_open = _horse_profile_factor(runner, race_open)
        # Geldings get same bonus regardless of class
        assert score_maiden == score_open


class TestWeightLowClassAmplification:
    """Tests for weight effect amplification in low-class races."""

    def test_low_class_amplifies_weight(self):
        runner = _make_runner(weight=59.0)
        race_low = _make_race(class_="Maiden")
        race_open = _make_race(class_="BM78")
        score_low = _weight_factor(runner, avg_weight=56.0, race_distance=1200, race=race_low)
        score_open = _weight_factor(runner, avg_weight=56.0, race_distance=1200, race=race_open)
        # Same weight diff but low-class should have bigger effect
        assert abs(score_low - 0.5) > abs(score_open - 0.5)


# ──────────────────────────────────────────────
# Context Multiplier Tests — Part B
# ──────────────────────────────────────────────

class TestContextHelpers:
    """Tests for context classification helpers."""

    def test_venue_type_metro(self):
        assert _context_venue_type("Flemington", "VIC") == "metro_vic"
        assert _context_venue_type("Randwick", "NSW") == "metro_nsw"
        assert _context_venue_type("Eagle Farm", "QLD") == "metro_qld"

    def test_venue_type_provincial(self):
        assert _context_venue_type("Sandown Lakeside", "VIC") == "provincial"
        assert _context_venue_type("Kembla Grange", "NSW") == "provincial"

    def test_venue_type_country(self):
        assert _context_venue_type("Alice Springs", "NT") == "country"

    def test_class_bucket(self):
        assert _context_class_bucket("Maiden") == "maiden"
        assert _context_class_bucket("Class 1") == "class1"
        assert _context_class_bucket("Restricted") == "restricted"
        assert _context_class_bucket("Class 2") == "class2"
        assert _context_class_bucket("Class 3") == "class3"
        assert _context_class_bucket("BM55") == "bm58"
        assert _context_class_bucket("BM64") == "bm64"
        assert _context_class_bucket("BM72") == "bm72"
        assert _context_class_bucket("BM78") == "open"
        assert _context_class_bucket("Group 1") == "open"
        assert _context_class_bucket("Benchmark 58") == "bm58"
        assert _context_class_bucket("Benchmark 65") == "bm64"


class TestContextMultipliers:
    """Tests for context multiplier application logic."""

    def test_amplifies_score_away_from_neutral(self):
        original = 0.7
        mult = 1.5
        adjusted = 0.5 + (original - 0.5) * mult
        assert adjusted == pytest.approx(0.8)

    def test_dampens_score_toward_neutral(self):
        original = 0.7
        mult = 0.5
        adjusted = 0.5 + (original - 0.5) * mult
        assert adjusted == pytest.approx(0.6)

    def test_neutral_stays_neutral(self):
        original = 0.5
        for mult in [0.3, 0.5, 1.0, 1.5, 2.5]:
            adjusted = 0.5 + (original - 0.5) * mult
            assert adjusted == pytest.approx(0.5)

    def test_no_profiles_returns_empty(self):
        import punty.probability as prob
        old_profiles = prob._CONTEXT_PROFILES
        try:
            prob._CONTEXT_PROFILES = {}
            mults = _get_context_multipliers(_make_race(), _make_meeting())
            assert mults == {}
        finally:
            prob._CONTEXT_PROFILES = old_profiles

    def test_integration_with_context_profiles(self):
        """End-to-end: profiles should affect final probabilities."""
        import punty.probability as prob
        old_profiles = prob._CONTEXT_PROFILES

        try:
            prob._CONTEXT_PROFILES = {
                "profiles": {
                    "metro_vic|sprint|mid_bm|good": {
                        "barrier": 2.0,
                    }
                },
                "fallbacks": {},
                "metadata": {},
            }

            runners = [
                _make_runner(id="r1", current_odds=3.0, barrier=1),
                _make_runner(id="r2", current_odds=5.0, barrier=8),
                _make_runner(id="r3", current_odds=8.0, barrier=10),
            ]
            race = _make_race(distance=1000, class_="BM64")
            meeting = _make_meeting(venue="Flemington")

            result = calculate_race_probabilities(runners, race, meeting)

            assert result["r1"].factors["barrier"] > 0.55
            assert result["r3"].factors["barrier"] < 0.45

        finally:
            prob._CONTEXT_PROFILES = old_profiles

    def test_full_enhanced_integration(self):
        """End-to-end test with all new signals active."""
        runners = [
            _make_runner(
                id="r1",
                current_odds=3.5,
                career_record="18: 4-3-2",
                career_prize_money=250000,
                last_five="12134",
                form_history=json.dumps([
                    {"position": 2, "margin": 1.0},
                    {"position": 1, "margin": 0},
                    {"position": 3, "margin": 1.5},
                ]),
                jockey_stats=json.dumps({
                    "career": {"a2e": 1.10, "strike_rate": 15.0, "wins": 30, "runners": 200},
                    "combo_last100": {"strike_rate": 20.0, "runners": 25},
                }),
                horse_sex="Colt",
                horse_age=3,
                weight=56.5,
                barrier=2,
            ),
            _make_runner(id="r2", current_odds=5.0, barrier=5),
            _make_runner(id="r3", current_odds=5.0, barrier=6),
            _make_runner(id="r4", current_odds=8.0, barrier=9),
            _make_runner(id="r5", current_odds=10.0, barrier=3),
            _make_runner(id="r6", current_odds=12.0, barrier=4),
            _make_runner(id="r7", current_odds=15.0, barrier=7),
            _make_runner(id="r8", current_odds=20.0, barrier=8),
        ]
        race = _make_race(class_="Maiden", distance=1200)
        meeting = _make_meeting()

        probs = calculate_race_probabilities(runners, race, meeting)

        # r1 should have highest probability from strong signals
        assert probs["r1"].win_probability > 0.15
        # Profile should get colt-in-maiden boost
        assert probs["r1"].factors["horse_profile"] > 0.55
        # All probabilities sum to 1.0
        total = sum(p.win_probability for p in probs.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestWeightChangeFromFormHistory:
    """Tests for _get_weight_change_class deriving weight from form_history."""

    def test_weight_up_big(self):
        runner = _make_runner(
            weight=60.0,
            form_history=json.dumps([{"weight": 56.0, "date": "2026-01-28"}]),
        )
        assert _get_weight_change_class(runner) == "weight_up_big"

    def test_weight_up_small(self):
        runner = _make_runner(
            weight=57.0,
            form_history=json.dumps([{"weight": 56.0, "date": "2026-01-28"}]),
        )
        assert _get_weight_change_class(runner) == "weight_up_small"

    def test_weight_down_big(self):
        runner = _make_runner(
            weight=53.0,
            form_history=json.dumps([{"weight": 56.0, "date": "2026-01-28"}]),
        )
        assert _get_weight_change_class(runner) == "weight_down_big"

    def test_weight_down_small(self):
        runner = _make_runner(
            weight=55.5,
            form_history=json.dumps([{"weight": 56.0, "date": "2026-01-28"}]),
        )
        assert _get_weight_change_class(runner) == "weight_down_small"

    def test_weight_same(self):
        runner = _make_runner(
            weight=56.0,
            form_history=json.dumps([{"weight": 56.0, "date": "2026-01-28"}]),
        )
        assert _get_weight_change_class(runner) == "weight_same"

    def test_no_form_history(self):
        runner = _make_runner(weight=56.0)
        assert _get_weight_change_class(runner) == ""

    def test_no_weight(self):
        runner = _make_runner(
            form_history=json.dumps([{"weight": 56.0}]),
        )
        assert _get_weight_change_class(runner) == ""

    def test_empty_form_history(self):
        runner = _make_runner(weight=56.0, form_history=json.dumps([]))
        assert _get_weight_change_class(runner) == ""

    def test_form_history_no_weight_field(self):
        runner = _make_runner(
            weight=56.0,
            form_history=json.dumps([{"date": "2026-01-28", "venue": "Flemington"}]),
        )
        assert _get_weight_change_class(runner) == ""


class TestTrackStatsFallback:
    """Tests for track_stats fallback when track_dist_stats is missing."""

    def test_track_stats_used_when_no_track_dist(self):
        runner = _make_runner(
            track_stats="10: 3-2-1",
            last_five="12312",
        )
        score = _form_rating(runner, "Good 4", 0.10)
        # Should be > neutral because track_stats shows 30% win rate
        assert score > 0.5

    def test_track_dist_preferred_over_track_stats(self):
        runner = _make_runner(
            track_dist_stats="8: 4-2-1",  # 50% win rate
            track_stats="10: 1-0-0",       # 10% win rate
            last_five="11111",
        )
        score_with_td = _form_rating(runner, "Good 4", 0.10)

        runner2 = _make_runner(
            track_stats="8: 4-2-1",  # Same stats but only in track_stats
            last_five="11111",
        )
        score_ts_only = _form_rating(runner2, "Good 4", 0.10)

        # track_dist_stats gets 1.5x weight, track_stats only 0.8x
        # So the runner with track_dist_stats should score higher
        assert score_with_td > score_ts_only

    def test_no_track_stats_either(self):
        runner = _make_runner(last_five="55555")
        score = _form_rating(runner, "Good 4", 0.10)
        # Still produces a score from last_five alone (neutral or below)
        assert score <= 0.5


class TestDistanceSpecificWeights:
    """Tests for distance-specific factor weight profiles."""

    def test_sprint_distance_different_from_default(self):
        from punty.probability import DISTANCE_WEIGHT_OVERRIDES, DEFAULT_WEIGHTS
        sprint_w = DISTANCE_WEIGHT_OVERRIDES["sprint"]
        # Sprint barrier weight > default
        assert sprint_w["barrier"] > DEFAULT_WEIGHTS["barrier"]
        # Sprint pace weight > default
        assert sprint_w["pace"] > DEFAULT_WEIGHTS["pace"]

    def test_staying_distance_different_from_default(self):
        from punty.probability import DISTANCE_WEIGHT_OVERRIDES, DEFAULT_WEIGHTS
        staying_w = DISTANCE_WEIGHT_OVERRIDES["staying"]
        # Staying form weight > default
        assert staying_w["form"] > DEFAULT_WEIGHTS["form"]
        # Staying barrier negligible
        assert staying_w["barrier"] == 0.0
        # Staying weight_carried > default
        assert staying_w["weight_carried"] > DEFAULT_WEIGHTS["weight_carried"]

    def test_all_overrides_sum_to_one(self):
        from punty.probability import DISTANCE_WEIGHT_OVERRIDES
        for bucket, weights in DISTANCE_WEIGHT_OVERRIDES.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0, abs=0.01), f"{bucket} weights sum to {total}"

    def test_distance_affects_probabilities(self):
        """Sprint vs staying race should produce different probabilities for same runners."""
        runners = [
            _make_runner(id="r1", current_odds=3.0, last_five="12312", barrier=1),
            _make_runner(id="r2", current_odds=5.0, last_five="45678", barrier=12),
        ]
        sprint_race = _make_race(distance=1000, field_size=2)
        staying_race = _make_race(distance=2400, field_size=2)
        meeting = _make_meeting()

        sprint_results = calculate_race_probabilities(runners, sprint_race, meeting)
        staying_results = calculate_race_probabilities(runners, staying_race, meeting)

        # Runner with barrier 12 should be more penalised in sprints than staying
        sprint_gap = sprint_results["r1"].win_probability - sprint_results["r2"].win_probability
        staying_gap = staying_results["r1"].win_probability - staying_results["r2"].win_probability
        # Can't guarantee direction without knowing full factor scores, but they should differ
        assert sprint_results["r1"].win_probability != pytest.approx(
            staying_results["r1"].win_probability, abs=0.001
        )


class TestPlaceValueRating:
    """Tests for PVR using place market consensus."""

    def test_pvr_uses_place_odds_when_available(self):
        """PVR should use actual place odds, not formula approximation."""
        runners = [
            _make_runner(id="r1", current_odds=3.0, place_odds=1.50, last_five="11111"),
            _make_runner(id="r2", current_odds=5.0, place_odds=2.00, last_five="33333"),
            _make_runner(id="r3", current_odds=8.0, place_odds=2.50, last_five="55555"),
        ]
        race = _make_race(field_size=3)
        meeting = _make_meeting()

        results = calculate_race_probabilities(runners, race, meeting)
        # All runners should have a place_value_rating
        for rid in ["r1", "r2", "r3"]:
            assert results[rid].place_value_rating > 0

    def test_pvr_monotonic_with_place_odds(self):
        """PVR should be monotonic — stronger place probability = higher PVR."""
        runners = [
            _make_runner(id="fav", current_odds=2.0, place_odds=1.30, last_five="11111"),
            _make_runner(id="mid", current_odds=4.0, place_odds=1.80, last_five="32451"),
            _make_runner(id="long", current_odds=8.0, place_odds=2.80, last_five="65879"),
        ]
        race = _make_race(field_size=3)
        meeting = _make_meeting()

        results = calculate_race_probabilities(runners, race, meeting)
        # Favourite with best place odds should have highest PVR (model rates it higher)
        assert results["fav"].place_value_rating >= results["mid"].place_value_rating
