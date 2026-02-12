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
    _jockey_trainer_factor,
    _weight_factor,
    _horse_profile_factor,
    _average_weight,
    _place_probability,
    _recommended_stake,
    _determine_pace_scenario,
    _condition_stats_field,
)


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
        assert median == 5.0  # median of [4.5, 4.8, 5.0, 5.0, 5.2]

    def test_median_odds_filters_invalid(self):
        runner = _make_runner(
            current_odds=5.0,
            odds_tab=None,
            odds_sportsbet=0.0,
            odds_bet365=1.0,  # excluded (must be > 1.0)
        )
        assert _get_median_odds(runner) == 5.0

    def test_median_odds_no_valid(self):
        runner = _make_runner()
        assert _get_median_odds(runner) is None

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
        assert rating > 0.5

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
        assert prob == pytest.approx(0.60)

    def test_large_field(self):
        prob = _place_probability(0.15, 14)
        assert prob == pytest.approx(0.15 * 3.3)

    def test_capped_at_95(self):
        prob = _place_probability(0.50, 5)
        assert prob == 0.95


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

        # The value runner should have a value_rating > 1.0
        assert results["value"].value_rating > 1.0

    def test_place_probability_populated(self):
        runners = [
            _make_runner(id="r1", current_odds=3.0),
            _make_runner(id="r2", current_odds=5.0),
        ]
        results = calculate_race_probabilities(runners, _make_race(), _make_meeting())

        for rp in results.values():
            assert rp.place_probability > rp.win_probability

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


# ──────────────────────────────────────────────
# Jockey & Trainer Factor
# ──────────────────────────────────────────────

class TestJockeyTrainerFactor:
    def test_strong_jockey(self):
        runner = _make_runner(jockey_stats="50: 15-10-5")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score > 0.5, "High jockey win rate should score above neutral"

    def test_strong_trainer(self):
        runner = _make_runner(trainer_stats="100: 25-15-10")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score > 0.5, "High trainer win rate should score above neutral"

    def test_both_strong(self):
        runner = _make_runner(jockey_stats="50: 15-10-5", trainer_stats="100: 25-15-10")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score > 0.5

    def test_weak_connections(self):
        runner = _make_runner(jockey_stats="50: 2-3-5", trainer_stats="100: 3-5-10")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert score < 0.55, "Low win rates should score near or below neutral"

    def test_no_data_returns_neutral(self):
        runner = _make_runner()
        assert _jockey_trainer_factor(runner, baseline=0.10) == 0.5

    def test_insufficient_starts_ignored(self):
        runner = _make_runner(jockey_stats="3: 2-1-0")  # < 5 starts
        assert _jockey_trainer_factor(runner, baseline=0.10) == 0.5

    def test_score_bounds(self):
        runner = _make_runner(jockey_stats="100: 50-20-10", trainer_stats="200: 80-40-20")
        score = _jockey_trainer_factor(runner, baseline=0.10)
        assert 0.05 <= score <= 0.95


# ──────────────────────────────────────────────
# Weight Factor
# ──────────────────────────────────────────────

class TestWeightFactor:
    def test_lighter_is_advantage(self):
        runner = _make_runner(weight=54.0)
        score = _weight_factor(runner, avg_weight=57.0)
        assert score > 0.5, "Lighter weight should score above neutral"

    def test_heavier_is_disadvantage(self):
        runner = _make_runner(weight=60.0)
        score = _weight_factor(runner, avg_weight=57.0)
        assert score < 0.5, "Heavier weight should score below neutral"

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
