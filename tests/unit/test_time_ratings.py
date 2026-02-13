"""Tests for standard time comparison."""

import pytest
from punty.context.time_ratings import (
    parse_race_time,
    rate_form_times,
    _distance_bucket,
    _condition_group,
    _venue_key,
    clear_cache,
)


class TestParseRaceTime:
    def test_mm_ss(self):
        assert parse_race_time("1:35.20") == 95.20

    def test_short_race(self):
        assert parse_race_time("0:58.40") == 58.40

    def test_long_race(self):
        assert parse_race_time("2:05.10") == 125.10

    def test_seconds_only(self):
        assert parse_race_time("58.40") == 58.40

    def test_none(self):
        assert parse_race_time(None) is None

    def test_empty_string(self):
        assert parse_race_time("") is None

    def test_invalid(self):
        assert parse_race_time("abc") is None

    def test_whitespace(self):
        assert parse_race_time(" 1:35.20 ") == 95.20

    def test_zero_minutes(self):
        """A short sprint like 0:55.00."""
        assert parse_race_time("0:55.00") == 55.00


class TestDistanceBucket:
    def test_sprint(self):
        assert _distance_bucket(1000) == "sprint"
        assert _distance_bucket(1200) == "sprint"

    def test_short(self):
        assert _distance_bucket(1300) == "short"
        assert _distance_bucket(1400) == "short"

    def test_middle(self):
        assert _distance_bucket(1600) == "middle"
        assert _distance_bucket(1800) == "middle"

    def test_staying(self):
        assert _distance_bucket(2000) == "staying"
        assert _distance_bucket(3200) == "staying"


class TestConditionGroup:
    def test_good(self):
        assert _condition_group("Good 4") == "Good"
        assert _condition_group("Good") == "Good"

    def test_soft(self):
        assert _condition_group("Soft 5") == "Soft"
        assert _condition_group("Soft 7") == "Soft"

    def test_heavy(self):
        assert _condition_group("Heavy 8") == "Heavy"
        assert _condition_group("Heavy 10") == "Heavy"

    def test_firm(self):
        assert _condition_group("Firm 1") == "Firm"

    def test_empty(self):
        assert _condition_group("") == "Good"

    def test_none(self):
        assert _condition_group(None) == "Good"


class TestVenueKey:
    def test_basic(self):
        assert _venue_key("Flemington") == "flemington"

    def test_with_spaces(self):
        assert _venue_key("Eagle Farm") == "eaglefarm"

    def test_with_special_chars(self):
        assert _venue_key("Moonee Valley") == "mooneevalley"


class TestRateFormTimes:
    STANDARD_TIMES = {
        "flemington_middle_Good": {"median": 97.50, "count": 100},
        "randwick_sprint_Good": {"median": 68.00, "count": 80},
        "flemington_middle_Soft": {"median": 99.00, "count": 50},
    }

    def _make_start(self, venue, distance, track, time_str):
        return {
            "venue": venue,
            "distance": distance,
            "track": track,
            "time": time_str,
        }

    def test_fast_rating(self):
        """More than 2% faster than standard."""
        fh = [self._make_start("Flemington", 1600, "Good 4", "1:33.00")]
        # 93.0 vs 97.5 standard = -4.6%
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 1
        assert results[0]["rating"] == "FAST"
        assert results[0]["diff_pct"] < -2

    def test_slow_rating(self):
        """More than 2% slower than standard."""
        fh = [self._make_start("Flemington", 1600, "Good 4", "1:40.00")]
        # 100.0 vs 97.5 = +2.6%
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 1
        assert results[0]["rating"] == "SLOW"

    def test_standard_rating(self):
        """Within 2% of standard."""
        fh = [self._make_start("Flemington", 1600, "Good 4", "1:37.00")]
        # 97.0 vs 97.5 = -0.5%
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 1
        assert results[0]["rating"] == "STANDARD"

    def test_condition_matching(self):
        """Should match the correct condition group."""
        fh = [self._make_start("Flemington", 1600, "Soft 5", "1:35.00")]
        # 95.0 vs 99.0 (Soft standard) = -4.0%
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 1
        assert results[0]["rating"] == "FAST"
        assert results[0]["standard_secs"] == 99.0

    def test_no_matching_standard(self):
        """Should skip starts with no matching standard time."""
        fh = [self._make_start("Unknown Track", 1600, "Good", "1:35.00")]
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 0

    def test_no_time_field(self):
        fh = [{"venue": "Flemington", "distance": 1600, "track": "Good"}]
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 0

    def test_empty_standard_times(self):
        fh = [self._make_start("Flemington", 1600, "Good", "1:35.00")]
        results = rate_form_times(fh, {})
        assert len(results) == 0

    def test_multiple_starts(self):
        fh = [
            self._make_start("Flemington", 1600, "Good 4", "1:33.00"),  # FAST
            self._make_start("Randwick", 1000, "Good 3", "1:04.00"),    # SLOW (64s vs 68s = -5.9% actually FAST)
            self._make_start("Flemington", 1600, "Good", "1:37.50"),    # STANDARD
        ]
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 3

    def test_max_starts(self):
        fh = [self._make_start("Flemington", 1600, "Good", "1:35.00")] * 10
        results = rate_form_times(fh, self.STANDARD_TIMES, max_starts=3)
        assert len(results) == 3

    def test_fallback_to_good(self):
        """If no standard for specific condition, fall back to Good."""
        fh = [self._make_start("Flemington", 1600, "Firm 1", "1:33.00")]
        # No "flemington_middle_Firm" but should fall back to "flemington_middle_Good"
        results = rate_form_times(fh, self.STANDARD_TIMES)
        assert len(results) == 1
        assert results[0]["standard_secs"] == 97.50
