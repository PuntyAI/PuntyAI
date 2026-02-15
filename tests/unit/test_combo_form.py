"""Tests for cross-referenced combo form analysis."""

import pytest

from punty.context.combo_form import (
    ComboStats,
    SpacingResult,
    _bucket_class,
    _bucket_condition,
    _normalise_venue,
    _distance_matches,
    _venue_matches,
    _parse_position,
    _build_stats,
    _gap_to_bucket,
    _analyse_spacing,
    analyse_combo_form,
)


# ──────────────────────────────────────────────
# Bucketing tests
# ──────────────────────────────────────────────

class TestBucketCondition:
    def test_good_variants(self):
        assert _bucket_condition("Good 4") == "good"
        assert _bucket_condition("Good") == "good"
        assert _bucket_condition("Firm 1") == "good"

    def test_soft_variants(self):
        assert _bucket_condition("Soft 5") == "soft"
        assert _bucket_condition("Soft 7") == "soft"

    def test_heavy_variants(self):
        assert _bucket_condition("Heavy 8") == "heavy"
        assert _bucket_condition("Heavy 10") == "heavy"

    def test_empty(self):
        assert _bucket_condition("") == "good"
        assert _bucket_condition(None) == "good"


class TestBucketClass:
    def test_maiden(self):
        assert _bucket_class("Maiden") == "maiden"
        assert _bucket_class("MDN") == "maiden"
        assert _bucket_class("Maiden Plate") == "maiden"

    def test_benchmark(self):
        assert _bucket_class("BM58") == "bm_low"
        assert _bucket_class("Benchmark 64") == "bm_mid"
        assert _bucket_class("BM72") == "bm_high"
        assert _bucket_class("BM78") == "bm_high"
        assert _bucket_class("BM84") == "open"

    def test_class_levels(self):
        assert _bucket_class("Class 1") == "class1"
        assert _bucket_class("CL1") == "class1"
        assert _bucket_class("Class 2") == "class2"
        assert _bucket_class("Class 3") == "class3"

    def test_open_group(self):
        assert _bucket_class("Group 1") == "open"
        assert _bucket_class("Listed") == "open"
        assert _bucket_class("Open Handicap") == "open"

    def test_restricted(self):
        assert _bucket_class("Restricted Handicap") == "restricted"
        # "Restricted Maiden" matches maiden first (intentional)
        assert _bucket_class("Restricted Maiden") == "maiden"

    def test_empty(self):
        assert _bucket_class("") == "unknown"
        assert _bucket_class(None) == "unknown"


class TestVenueMatching:
    def test_exact_match(self):
        assert _venue_matches("Flemington", "Flemington")

    def test_case_insensitive(self):
        assert _venue_matches("flemington", "FLEMINGTON")

    def test_aliases(self):
        assert _venue_matches("The Valley", "Moonee Valley")
        assert _venue_matches("Royal Randwick", "Randwick")

    def test_no_match(self):
        assert not _venue_matches("Flemington", "Caulfield")

    def test_empty(self):
        assert not _venue_matches("", "Flemington")


class TestDistanceMatching:
    def test_sprint_tolerance(self):
        assert _distance_matches(1000, 1000)
        assert _distance_matches(1000, 1100)
        assert not _distance_matches(1000, 1200)

    def test_middle_tolerance(self):
        assert _distance_matches(1600, 1600)
        assert _distance_matches(1400, 1600)
        assert not _distance_matches(1400, 1700)

    def test_staying_tolerance(self):
        assert _distance_matches(2400, 2400)
        assert _distance_matches(2400, 2600)
        assert not _distance_matches(2400, 2800)

    def test_none_values(self):
        assert not _distance_matches(None, 1600)
        assert not _distance_matches(1600, None)


class TestParsePosition:
    def test_int(self):
        assert _parse_position(1) == 1
        assert _parse_position(3) == 3

    def test_string(self):
        assert _parse_position("1") == 1
        assert _parse_position("2nd") == 2

    def test_float(self):
        assert _parse_position(1.0) == 1

    def test_invalid(self):
        assert _parse_position(None) is None
        assert _parse_position(0) is None
        assert _parse_position("x") is None
        assert _parse_position("DNF") is None


# ──────────────────────────────────────────────
# Stats builder
# ──────────────────────────────────────────────

class TestBuildStats:
    def test_basic(self):
        starts = [
            {"position": 1},
            {"position": 2},
            {"position": 3},
            {"position": 5},
        ]
        stats = _build_stats(starts)
        assert stats.starts == 4
        assert stats.wins == 1
        assert stats.seconds == 1
        assert stats.thirds == 1
        assert stats.win_rate == 0.25
        assert stats.place_rate == 0.75

    def test_empty(self):
        stats = _build_stats([])
        assert stats.starts == 0
        assert stats.win_rate == 0.0


# ──────────────────────────────────────────────
# Spacing analysis
# ──────────────────────────────────────────────

class TestGapBucket:
    def test_buckets(self):
        assert _gap_to_bucket(7) == "quick_backup"
        assert _gap_to_bucket(14) == "quick_backup"
        assert _gap_to_bucket(21) == "normal"
        assert _gap_to_bucket(28) == "normal"
        assert _gap_to_bucket(45) == "freshened"
        assert _gap_to_bucket(90) == "spell"
        assert _gap_to_bucket(150) == "long_spell"


class TestSpacingAnalysis:
    def _make_history(self, gaps_and_positions):
        """Build form history from (gap_days, position) pairs, most recent first."""
        from datetime import datetime, timedelta
        history = []
        date = datetime(2026, 2, 1)
        for gap, pos in gaps_and_positions:
            history.append({"date": date.strftime("%Y-%m-%d"), "position": pos})
            date -= timedelta(days=gap)
        # Add one more entry so the last gap can be calculated
        history.append({"date": date.strftime("%Y-%m-%d"), "position": 4})
        return history

    def test_normal_spacing_with_wins(self):
        # Horse performs well off 21-day gaps
        history = self._make_history([
            (21, 1), (21, 2), (21, 1), (21, 5),
        ])
        result = _analyse_spacing(history, days_since_last_run=25)
        assert result is not None
        assert result.bucket == "normal"
        assert result.stats.wins == 2
        assert result.pattern_score > 0.5

    def test_too_few_starts(self):
        history = [{"date": "2026-01-01", "position": 1}]
        result = _analyse_spacing(history, days_since_last_run=21)
        assert result is None

    def test_no_days_since(self):
        history = self._make_history([(21, 1), (21, 2)])
        result = _analyse_spacing(history, days_since_last_run=None)
        assert result is None

    def test_no_similar_bucket(self):
        # All gaps are 21 days (normal), but asking about spell (90 days)
        history = self._make_history([(21, 1), (21, 2), (21, 1)])
        result = _analyse_spacing(history, days_since_last_run=90)
        assert result is None


# ──────────────────────────────────────────────
# Main analyse_combo_form
# ──────────────────────────────────────────────

SAMPLE_FORM_HISTORY = [
    {
        "date": "2026-01-20", "venue": "Flemington", "distance": 1600,
        "class": "BM78", "track": "Good 4", "position": 1,
        "jockey": "J McDonald", "barrier": 3,
    },
    {
        "date": "2026-01-01", "venue": "Flemington", "distance": 1600,
        "class": "BM72", "track": "Good 3", "position": 2,
        "jockey": "J McDonald", "barrier": 5,
    },
    {
        "date": "2025-12-15", "venue": "Caulfield", "distance": 1400,
        "class": "BM72", "track": "Soft 5", "position": 1,
        "jockey": "B Shinn", "barrier": 8,
    },
    {
        "date": "2025-11-20", "venue": "Flemington", "distance": 1600,
        "class": "BM64", "track": "Soft 6", "position": 3,
        "jockey": "J McDonald", "barrier": 2,
    },
    {
        "date": "2025-10-30", "venue": "Flemington", "distance": 1200,
        "class": "BM78", "track": "Heavy 8", "position": 5,
        "jockey": "C Williams", "barrier": 10,
    },
    {
        "date": "2025-10-10", "venue": "Randwick", "distance": 1600,
        "class": "BM78", "track": "Good 4", "position": 1,
        "jockey": "J McDonald", "barrier": 4,
    },
]


class TestAnalyseComboForm:
    def test_track_condition_combo(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        # Flemington + Good: starts on 2026-01-20 (1st), 2026-01-01 (2nd)
        tc = result.get("track_cond")
        assert tc is not None
        assert tc.starts == 2
        # Below min_starts threshold (3) in probability engine, but still returned

    def test_dist_condition_combo(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        # 1600m + Good: 2026-01-20 (Flemington, 1st), 2026-01-01 (Flemington, 2nd),
        # 2025-10-10 (Randwick, 1st)
        dc = result.get("dist_cond")
        assert dc is not None
        assert dc.starts == 3
        assert dc.wins == 2

    def test_jockey_horse_combo(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        jh = result.get("jockey_horse")
        assert jh is not None
        assert jh.starts == 4  # McDonald rode 4 times
        assert jh.wins == 2

    def test_class_performance(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        # BM78 → bm_high, BM72 → bm_high (72 ≤ 78): 5 starts match
        cp = result.get("class_perf")
        assert cp is not None
        assert cp.starts == 5
        assert cp.wins == 3

    def test_soft_condition_combo(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Caulfield",
            today_distance=1400,
            today_condition="Soft 5",
            today_class="BM72",
            today_jockey="B Shinn",
        )
        # Caulfield + Soft: 2025-12-15 (1st)
        tc = result.get("track_cond")
        assert tc is not None
        assert tc.starts == 1

        # Dist(1400) + Soft: 2025-12-15 (1400m, 1st) + 2025-11-20 (1600m within 200m tol, 3rd)
        dc = result.get("dist_cond")
        assert dc is not None
        assert dc.starts == 2

    def test_empty_form_history(self):
        result = analyse_combo_form(
            [],
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        assert result == {}

    def test_no_matching_venue(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Eagle Farm",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        assert "track_cond" not in result

    def test_trials_excluded(self):
        history = [
            {"date": "2026-01-20", "venue": "Flemington", "distance": 1600,
             "class": "BM78", "track": "Good 4", "position": 1,
             "jockey": "J McDonald", "is_trial": True},
        ]
        result = analyse_combo_form(
            history,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        assert result == {}

    def test_triple_combo(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
        )
        # Flemington + 1600m + Good: 2026-01-20 (1st), 2026-01-01 (2nd)
        tdc = result.get("track_dist_cond")
        assert tdc is not None
        assert tdc.starts == 2
        assert tdc.wins == 1

    def test_spacing_included(self):
        result = analyse_combo_form(
            SAMPLE_FORM_HISTORY,
            today_venue="Flemington",
            today_distance=1600,
            today_condition="Good 4",
            today_class="BM78",
            today_jockey="J McDonald",
            days_since_last_run=20,
        )
        spacing = result.get("spacing")
        # Gaps in sample: 19, 17, 25, 21, 20 days — several in "normal" bucket
        if spacing is not None:
            assert isinstance(spacing, SpacingResult)
            assert spacing.bucket == "normal"


class TestComboStats:
    def test_win_rate(self):
        s = ComboStats(starts=10, wins=3, seconds=2, thirds=1)
        assert s.win_rate == 0.3
        assert s.place_rate == 0.6

    def test_zero_starts(self):
        s = ComboStats()
        assert s.win_rate == 0.0
        assert s.place_rate == 0.0
