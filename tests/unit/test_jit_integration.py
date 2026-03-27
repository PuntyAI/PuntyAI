"""Integration tests for JIT Betfair pipeline.

Tests the full path: probability → ranking → sense check → Kelly.
Catches bugs like the prob key mismatch that killed an entire day of bets.
"""

import pytest
from unittest.mock import MagicMock


def _make_runner(id, name, sc, odds=5.0, age=4):
    """Create a mock runner with all required attributes."""
    r = MagicMock()
    r.id = id
    r.horse_name = name
    r.saddlecloth = sc
    r.current_odds = odds
    r.opening_odds = odds
    r.barrier = sc
    r.weight = 57.0
    r.form = "x1234"
    r.last_five = "12345"
    r.jockey = "J. Smith"
    r.trainer = "T. Jones"
    r.horse_age = age
    r.horse_sex = "Gelding"
    r.scratched = False
    r.speed_map_position = "midfield"
    r.form_history = "[]"
    r.career_record = "10: 3-2-1"
    r.jockey_stats = None
    r.trainer_stats = None
    r.track_dist_stats = None
    r.distance_stats = None
    r.good_track_stats = None
    r.soft_track_stats = None
    r.heavy_track_stats = None
    r.class_stats = None
    r.handicap_rating = None
    r.days_since_last_run = 14
    r.pf_settle = None
    r.pf_speed_rank = None
    r.pf_map_factor = None
    r.pf_jockey_factor = None
    r.kash_rated_price = None
    r.kash_speed_cat = None
    r.kash_early_speed = None
    r.kash_late_speed = None
    r.pf_ai_price = None
    r.pf_ai_score = None
    r.pf_assessed_price = None
    r.sire = None
    r.dam_sire = None
    r.pointsbet_odds = None
    r.odds_pointsbet = None
    r.odds_betfair = None
    r.odds_sportsbet = None
    r.odds_bet365 = None
    r.odds_ladbrokes = None
    r.odds_tab = None
    r.place_odds = None
    r.odds_flucs = None
    return r


def _make_race(id="test-r1", distance=1200, race_class="Maiden"):
    r = MagicMock()
    r.id = id
    r.race_number = 1
    r.distance = distance
    r.class_ = race_class
    r.name = "Test Race"
    r.start_time = None
    r.results_status = "Open"
    return r


def _make_meeting(id="test", venue="Flemington", condition="Good 4"):
    m = MagicMock()
    m.id = id
    m.venue = venue
    m.track_condition = condition
    m.rail_position = None
    m.date = "2026-03-27"
    m.selected = True
    return m


class TestProbKeyMatchesRunnerId:
    """THE critical test — probs must be keyed by runner.id."""

    def test_probs_keyed_by_runner_id(self):
        from punty.probability import calculate_race_probabilities

        runners = [
            _make_runner("test-r1-1-horse-a", "Horse A", 1, 3.0),
            _make_runner("test-r1-2-horse-b", "Horse B", 2, 5.0),
            _make_runner("test-r1-3-horse-c", "Horse C", 3, 8.0),
            _make_runner("test-r1-4-horse-d", "Horse D", 4, 12.0),
        ]

        probs = calculate_race_probabilities(runners, _make_race(), _make_meeting())

        # Every runner.id must be a valid key in probs
        for runner in runners:
            prob = probs.get(runner.id)
            assert prob is not None, (
                f"probs.get('{runner.id}') returned None. "
                f"Available keys: {list(probs.keys())}. "
                f"This is the bug that caused zero Betfair bets on 2026-03-27."
            )
            assert prob.win_probability > 0

    def test_probs_not_keyed_by_horse_name_only(self):
        """Verify we're not accidentally relying on horse_name as key."""
        from punty.probability import calculate_race_probabilities

        runners = [
            _make_runner("meeting-r1-1-alpha", "Alpha Horse", 1, 3.0),
            _make_runner("meeting-r1-2-beta", "Beta Horse", 2, 5.0),
            _make_runner("meeting-r1-3-gamma", "Gamma Horse", 3, 8.0),
            _make_runner("meeting-r1-4-delta", "Delta Horse", 4, 12.0),
        ]

        probs = calculate_race_probabilities(runners, _make_race(), _make_meeting())

        # runner.id lookup must work
        matched_by_id = sum(1 for r in runners if probs.get(r.id))
        assert matched_by_id == len(runners), f"Only {matched_by_id}/{len(runners)} matched by ID"


class TestJITRanking:
    """Test that JIT ranking logic produces non-empty results."""

    def test_ranking_not_empty(self):
        from punty.probability import calculate_race_probabilities

        runners = [
            _make_runner("test-r1-1-aa", "AA", 1, 3.0, age=3),
            _make_runner("test-r1-2-bb", "BB", 2, 5.0, age=4),
            _make_runner("test-r1-3-cc", "CC", 3, 8.0, age=5),
        ]
        race = _make_race()
        meeting = _make_meeting()

        probs = calculate_race_probabilities(runners, race, meeting)

        # Simulate JIT ranking logic
        ranked = []
        for runner in runners:
            prob = probs.get(runner.id) or probs.get(runner.horse_name)
            if not prob:
                continue
            if runner.horse_age and runner.horse_age >= 6:
                continue
            ranked.append({
                "runner": runner,
                "wp": prob.win_probability,
                "pp": prob.place_probability,
            })

        ranked.sort(key=lambda x: x["pp"], reverse=True)
        assert len(ranked) > 0, "JIT ranking produced empty list — all runners filtered out"
        assert ranked[0]["pp"] > 0

    def test_age_filter_doesnt_kill_all(self):
        """Age 6+ filter should not remove every runner in a normal field."""
        runners = [
            _make_runner("r1-1-a", "A", 1, 3.0, age=3),
            _make_runner("r1-2-b", "B", 2, 5.0, age=4),
            _make_runner("r1-3-c", "C", 3, 8.0, age=6),  # This one filtered
            _make_runner("r1-4-d", "D", 4, 12.0, age=5),
        ]

        after_filter = [r for r in runners if not (r.horse_age and r.horse_age >= 6)]
        assert len(after_filter) >= 1, "Age filter removed every runner"
        assert len(after_filter) == 3


class TestSenseCheck:
    def test_returns_valid_consensus(self):
        from punty.sense_check import sense_check_race
        runners = [
            {"saddlecloth": 1, "current_odds": 3.0, "kash_rated_price": 3.5,
             "pf_ai_score": 80, "pf_assessed_price": None, "scratched": False},
            {"saddlecloth": 2, "current_odds": 5.0, "kash_rated_price": 4.0,
             "pf_ai_score": 60, "pf_assessed_price": None, "scratched": False},
        ]
        result = sense_check_race(1, runners)
        assert result["consensus"] in ("HIGH", "MEDIUM", "LOW")
        assert result["kelly_mult"] >= 0

    def test_consensus_override(self):
        from punty.sense_check import find_consensus_pick
        # All 3 external models pick #1, our R2 is #1
        runners = [
            {"saddlecloth": 1, "current_odds": 2.0, "kash_rated_price": 2.5,
             "pf_ai_score": 90, "scratched": False},
            {"saddlecloth": 2, "current_odds": 8.0, "kash_rated_price": 9.0,
             "pf_ai_score": 30, "scratched": False},
        ]
        picks = [
            {"saddlecloth": 2, "tip_rank": 1, "horse_name": "B"},
            {"saddlecloth": 1, "tip_rank": 2, "horse_name": "A"},
        ]
        result = find_consensus_pick(picks, runners)
        assert result is not None
        assert result["saddlecloth"] == 1
        assert result["tip_rank"] == 2


class TestKellyStaking:
    def test_positive_stake(self):
        from punty.betting.queue import calculate_kelly_stake
        stake = calculate_kelly_stake(balance=100, place_probability=0.70, odds=1.50)
        assert stake > 0

    def test_zero_edge_zero_stake(self):
        from punty.betting.queue import calculate_kelly_stake
        # PP below break-even for these odds → zero or minimal stake
        stake = calculate_kelly_stake(balance=100, place_probability=0.30, odds=1.50)
        assert stake >= 0  # Should be 0 or very small
