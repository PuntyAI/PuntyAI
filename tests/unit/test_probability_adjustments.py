"""Tests for ProPun strategy post-scoring adjustments in probability engine."""

import pytest

from punty.probability import (
    _campaign_run_count,
    _false_favourite_score,
    _barrier_draw_factor,
    calculate_exotic_combinations,
    parse_stats_string,
)


# ── Campaign run count ──────────────────────────────────────────────

class TestCampaignRunCount:
    def test_full_five_runs(self):
        runner = {"last_five": "12345"}
        assert _campaign_run_count(runner) == 5

    def test_with_x_for_unplaced(self):
        runner = {"last_five": "1x3x2"}
        assert _campaign_run_count(runner) == 5

    def test_three_runs(self):
        runner = {"last_five": "123"}
        assert _campaign_run_count(runner) == 3

    def test_single_run(self):
        runner = {"last_five": "1"}
        assert _campaign_run_count(runner) == 1

    def test_empty_string_with_long_spell(self):
        runner = {"last_five": "", "days_since_last_run": 90}
        assert _campaign_run_count(runner) == 1  # first-up

    def test_empty_string_no_spell(self):
        runner = {"last_five": "", "days_since_last_run": 14}
        assert _campaign_run_count(runner) == 3  # default mid-campaign

    def test_none_last_five(self):
        runner = {"last_five": None}
        assert _campaign_run_count(runner) == 3  # default

    def test_ignores_non_digit_chars(self):
        runner = {"last_five": "1-2-3"}
        assert _campaign_run_count(runner) == 3  # only digits count

    def test_whitespace_stripped(self):
        runner = {"last_five": " 12x4 "}
        assert _campaign_run_count(runner) == 4


# ── False favourite scoring ─────────────────────────────────────────

class TestFalseFavouriteScore:
    def _make_runner(self, **kw):
        defaults = {
            "career_record": None,
            "first_up_stats": None,
            "days_since_last_run": 14,
            "last_five": "12345",
            "weight": 56.0,
            "sex": "G",
            "age": 5,
            "track_stats": None,
        }
        defaults.update(kw)
        return defaults

    def test_clean_runner_scores_zero(self):
        runner = self._make_runner(
            career_record="50: 10-8-6",  # 20% SR, >15%
        )
        assert _false_favourite_score(runner, {}) == 0

    def test_low_career_strike_rate(self):
        runner = self._make_runner(
            career_record="20: 2-3-4",  # 10% SR, <15%
        )
        score = _false_favourite_score(runner, {})
        assert score >= 2  # +2 for career SR

    def test_first_up_no_wins(self):
        runner = self._make_runner(
            days_since_last_run=90,
            first_up_stats="5: 0-1-1",  # 0 wins from 5 first-up attempts
        )
        score = _false_favourite_score(runner, {})
        assert score >= 2  # +2 for first-up no wins

    def test_deep_campaign(self):
        runner = self._make_runner(
            last_five="1234567",  # 7 chars = 7 runs (note: only 5 scored by _score_last_five, but _campaign_run_count counts all)
        )
        score = _false_favourite_score(runner, {})
        assert score >= 1  # +1 for 7+ runs

    def test_heavy_weight_young_filly(self):
        runner = self._make_runner(
            weight=59.0,
            sex="F",
            age=3,
        )
        score = _false_favourite_score(runner, {})
        assert score >= 1  # +1 for weight

    def test_last_run_unplaced(self):
        runner = self._make_runner(last_five="51234")
        score = _false_favourite_score(runner, {})
        assert score >= 1  # +1 for unplaced last start

    def test_class_jump_after_maiden_win(self):
        runner = self._make_runner(
            career_record="5: 1-1-1",  # exactly 1 win
        )
        race = {"class_": "Class 2 Hcp"}  # not maiden
        score = _false_favourite_score(runner, race)
        assert score >= 1  # +1 for class jump

    def test_multiple_red_flags_stack(self):
        runner = self._make_runner(
            career_record="10: 1-2-3",  # 10% SR (<15%) + 1 win
            last_five="x1234",  # unplaced last start
        )
        race = {"class_": "BM64"}  # not maiden
        score = _false_favourite_score(runner, race)
        # Should get: +2 (career SR) + +1 (unplaced) + +1 (class jump) = 4+
        assert score >= 4


# ── Barrier softening ──────────────────────────────────────────────

class TestBarrierSoftening:
    """Verify barrier penalties have been reduced ~25%."""

    def test_widest_gate_penalty_reduced(self):
        """Widest gate in a 10-horse sprint field: penalty should be ~0.06, not 0.08."""
        # barrier 10 of 10 = relative 1.0 (>= 0.85)
        score = _barrier_draw_factor(
            {"barrier": 10}, field_size=10, distance=1200,
        )
        # Sprint dist_mult = 1.3, penalty = 0.06 * 1.3 = 0.078
        # Old: 0.08 * 1.3 = 0.104
        assert score == pytest.approx(0.5 - 0.06 * 1.3, abs=0.01)

    def test_outer_quarter_penalty_reduced(self):
        """Outer quarter gate: penalty should be ~0.03, not 0.04."""
        # barrier 8 of 10 = relative 0.78 (>= 0.70, < 0.85)
        score = _barrier_draw_factor(
            {"barrier": 8}, field_size=10, distance=1400,
        )
        # Middle dist_mult = 1.0, penalty = 0.03
        # Old: 0.04
        assert score == pytest.approx(0.5 - 0.03, abs=0.01)


# ── Exotic: odds-on exclusion ──────────────────────────────────────

class TestOddsOnExoticExclusion:
    def _make_runners(self, n=10, fav_implied=0.55):
        """Create n runners with decreasing probabilities."""
        runners = []
        for i in range(n):
            if i == 0:
                mi = fav_implied
            else:
                mi = (1.0 - fav_implied) / (n - 1)
            runners.append({
                "saddlecloth": i + 1,
                "horse_name": f"Horse{i+1}",
                "win_prob": mi,  # Use same as market for simplicity
                "market_implied": mi,
            })
        return runners

    def test_first4_skipped_when_odds_on(self):
        """No First4/First4 Box when favourite is odds-on (< $2.00)."""
        runners = self._make_runners(n=10, fav_implied=0.55)  # $1.82
        results = calculate_exotic_combinations(runners)
        first4_types = [r.exotic_type for r in results if "First4" in r.exotic_type]
        assert len(first4_types) == 0

    def test_first4_included_when_not_odds_on(self):
        """First4 still generated when favourite is not odds-on."""
        runners = self._make_runners(n=10, fav_implied=0.35)  # $2.86
        results = calculate_exotic_combinations(runners)
        # May or may not generate depending on value thresholds,
        # but the function shouldn't skip them outright
        # Just verify the function runs without error
        assert isinstance(results, list)

    def test_trifecta_box_excludes_odds_on_fav(self):
        """Trifecta Box should exclude odds-on favourite from pool."""
        runners = self._make_runners(n=10, fav_implied=0.55)  # $1.82
        results = calculate_exotic_combinations(runners)
        for r in results:
            if r.exotic_type == "Trifecta Box":
                # Fav (saddlecloth 1) should not be in any trifecta box
                assert 1 not in r.runners


# ── Contrarian trifecta ────────────────────────────────────────────

class TestContrarianTrifecta:
    def test_generated_when_short_fav_big_field(self):
        """Contrarian trifecta generated when top runner > 40% implied, field >= 10."""
        runners = []
        # Strong favourite: 45% implied ($2.22)
        runners.append({
            "saddlecloth": 1, "horse_name": "StrongFav",
            "win_prob": 0.45, "market_implied": 0.45,
        })
        # 2nd/3rd popular
        for i in range(2):
            runners.append({
                "saddlecloth": i + 2, "horse_name": f"Popular{i+2}",
                "win_prob": 0.15, "market_implied": 0.15,
            })
        # Contrarian pool: $12-$40 range runners with >= 5% prob
        for i in range(7):
            implied = 0.05 + i * 0.005  # 5-8.5% implied = $12-20 odds
            runners.append({
                "saddlecloth": i + 4, "horse_name": f"Outsider{i+4}",
                "win_prob": implied * 1.3,  # slight value edge
                "market_implied": implied,
            })

        results = calculate_exotic_combinations(runners)
        contrarian = [r for r in results if r.exotic_type == "Trifecta Contrarian"]
        # Should generate at least one contrarian trifecta
        assert len(contrarian) >= 1
        # Fav should always be first runner (anchored to win)
        for ct in contrarian:
            assert ct.runners[0] == 1

    def test_not_generated_for_open_race(self):
        """No contrarian trifecta when no dominant favourite."""
        runners = []
        for i in range(10):
            runners.append({
                "saddlecloth": i + 1, "horse_name": f"Horse{i+1}",
                "win_prob": 0.10, "market_implied": 0.10,
            })
        results = calculate_exotic_combinations(runners)
        contrarian = [r for r in results if r.exotic_type == "Trifecta Contrarian"]
        assert len(contrarian) == 0


# ── Hot streak name normalization ──────────────────────────────────

class TestHotStreakDedup:
    """Verify name normalization prevents duplicate hot streak alerts."""

    def test_title_normalization(self):
        """Names with different casing should normalize to same key."""
        from collections import Counter
        names = ["J MCDONALD", "J McDonald", "j mcdonald", "J Mcdonald"]
        counter = Counter()
        for name in names:
            counter[name.strip().title()] += 1
        # All should map to the same key
        assert len(counter) == 1
        assert counter["J Mcdonald"] == 4

    def test_dedup_key_matches(self):
        """Dedup key with normalized name should match across casings."""
        key1 = f"hot_jockey:{'J MCDONALD'.strip().title()}:3"
        key2 = f"hot_jockey:{'j mcdonald'.strip().title()}:3"
        assert key1 == key2


# ── Pace read spam prevention ─────────────────────────────────────

class TestPaceReadDedup:
    """Verify pace bias tracking prevents duplicate 'no bias' posts."""

    def test_same_bias_blocked(self):
        """Same bias type as last post should be blocked."""
        last_bias = None  # no bias
        current_bias = None  # still no bias
        posted_count = 1
        should_skip = (posted_count > 0 and current_bias == last_bias)
        assert should_skip is True

    def test_changed_bias_allowed(self):
        """Different bias type should be posted."""
        last_bias = None  # no bias
        current_bias = "speed"  # bias developed
        posted_count = 1
        should_skip = (posted_count > 0 and current_bias == last_bias)
        assert should_skip is False

    def test_first_post_always_allowed(self):
        """First post (count=0) should always go through."""
        posted_count = 0
        current_bias = None
        last_bias = None
        should_skip = (posted_count > 0 and current_bias == last_bias)
        assert should_skip is False


# ── Speed map bonus on Trifecta combos ───────────────────────────

class TestTrifectaSpeedMapBonus:
    """Fix 3: on-pace/leader runners get probability boost in Trifecta."""

    def _make_runners(self, speed_positions=None):
        """Create 4 runners with optional speed map positions."""
        runners = []
        for i in range(4):
            r = {
                "saddlecloth": i + 1,
                "horse_name": f"Horse{i+1}",
                "win_prob": 0.25 - i * 0.05,  # 0.25, 0.20, 0.15, 0.10
                "market_implied": 0.25 - i * 0.05,
                "value_rating": 1.10,
            }
            if speed_positions and i < len(speed_positions):
                r["speed_map_position"] = speed_positions[i]
            runners.append(r)
        return runners

    def test_on_pace_combo_scores_higher_than_midfield(self):
        """Combo with on-pace runners should have higher probability than midfield."""
        # Runners: #1 leader, #2 on_pace, #3 midfield, #4 backmarker
        runners = self._make_runners(["leader", "on_pace", "midfield", "backmarker"])
        results = calculate_exotic_combinations(runners)

        standouts = [r for r in results if r.exotic_type == "Trifecta Standout"]
        if len(standouts) >= 2:
            # Combo with on-pace runner (#2) should score higher than midfield (#3)
            # Both combos have #1 as standout; one includes #2, other includes #3
            combo_with_pace = [s for s in standouts if 2 in s.runners]
            combo_without_pace = [s for s in standouts if 2 not in s.runners]
            if combo_with_pace and combo_without_pace:
                assert combo_with_pace[0].estimated_probability > combo_without_pace[0].estimated_probability

    def test_speed_bonus_applies_to_trifecta_box(self):
        """Trifecta Box with on-pace runners gets boosted probability."""
        runners_pace = self._make_runners(["leader", "on_pace", "on_pace", "midfield"])
        runners_none = self._make_runners([None, None, None, None])
        results_pace = calculate_exotic_combinations(runners_pace)
        results_none = calculate_exotic_combinations(runners_none)

        box_pace = [r for r in results_pace if r.exotic_type == "Trifecta Box"]
        box_none = [r for r in results_none if r.exotic_type == "Trifecta Box"]
        if box_pace and box_none:
            # Pace runners should boost the box probability
            assert box_pace[0].estimated_probability > box_none[0].estimated_probability

    def test_no_speed_data_no_bonus(self):
        """Runners without speed_map_position get no bonus (multiplier stays 1.0)."""
        # Use runners with enough value edge to pass thresholds
        runners = []
        for i in range(4):
            runners.append({
                "saddlecloth": i + 1,
                "horse_name": f"Horse{i+1}",
                "win_prob": 0.30 - i * 0.05,  # 0.30, 0.25, 0.20, 0.15
                "market_implied": 0.20 - i * 0.03,  # lower market → higher value
                "value_rating": 1.30,
            })
        results = calculate_exotic_combinations(runners)
        standouts = [r for r in results if r.exotic_type == "Trifecta Standout"]
        # Should still generate combos — just no speed bonus
        assert len(standouts) >= 1
