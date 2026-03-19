"""Tests for Betfair meta-model bet selector."""

import math
from unittest.mock import patch, MagicMock

import pytest

from punty.betting.meta_model import (
    extract_meta_features,
    predict_place_probability,
    should_bet,
    clear_cache,
    META_FEATURE_NAMES,
    NUM_META_FEATURES,
    DEFAULT_WP_THRESHOLD,
    _safe_float,
)


# ── Helpers ──


def _make_features(**overrides) -> list[float]:
    """Build a meta-feature vector with sensible defaults."""
    defaults = {
        "wp": 0.25,
        "wp_margin": 0.08,
        "odds": 3.50,
        "field_size": 10,
        "distance_bucket": 3.0,
        "class_bucket": 3.0,
        "track_cond_bucket": 2.0,
        "venue_type": 1.0,
        "barrier_relative": 0.3,
        "age": 4.0,
        "days_since": 14.0,
        "form_score": 0.65,
        "form_trend": -0.5,
        "value_rating": 1.2,
        "speed_map_pos": 2.0,
        "weight_diff": -0.5,
        "career_win_pct": 0.18,
        "career_place_pct": 0.42,
    }
    defaults.update(overrides)
    return extract_meta_features(**defaults)


# ── Meta-feature extraction ──


class TestExtractMetaFeatures:
    def test_correct_length(self):
        features = _make_features()
        assert len(features) == NUM_META_FEATURES

    def test_correct_order(self):
        features = extract_meta_features(
            wp=0.30, wp_margin=0.10, odds=2.50, field_size=8,
            distance_bucket=2.0, class_bucket=4.0, track_cond_bucket=3.0,
            venue_type=1.0, barrier_relative=0.2, age=5.0, days_since=21.0,
            form_score=0.80, form_trend=-1.0, value_rating=1.5,
            speed_map_pos=1.0, weight_diff=-1.0,
            career_win_pct=0.22, career_place_pct=0.50,
        )
        assert features[0] == 0.30   # wp
        assert features[1] == 0.10   # wp_margin
        assert features[2] == 2.50   # odds
        assert features[3] == 8.0    # field_size
        assert features[4] == 2.0    # distance_bucket
        assert features[5] == 4.0    # class_bucket
        assert features[6] == 3.0    # track_cond_bucket
        assert features[7] == 1.0    # venue_type
        assert features[8] == 0.2    # barrier_relative
        assert features[9] == 5.0    # age
        assert features[10] == 21.0  # days_since
        assert features[11] == 0.80  # form_score
        assert features[12] == -1.0  # form_trend
        assert features[13] == 1.5   # value_rating
        assert features[14] == 1.0   # speed_map_pos
        assert features[15] == -1.0  # weight_diff
        assert features[16] == 0.22  # career_win_pct
        assert features[17] == 0.50  # career_place_pct

    def test_handles_nan_inputs(self):
        features = extract_meta_features(
            wp=0.25, wp_margin=0.05, odds=float("nan"),
            field_size=10, distance_bucket=3.0, class_bucket=3.0,
            track_cond_bucket=2.0, venue_type=1.0,
            barrier_relative=float("nan"), age=float("nan"),
            days_since=float("nan"), form_score=float("nan"),
            form_trend=float("nan"), value_rating=float("nan"),
            speed_map_pos=float("nan"), weight_diff=float("nan"),
            career_win_pct=float("nan"), career_place_pct=float("nan"),
        )
        assert len(features) == NUM_META_FEATURES
        # First two should be valid
        assert features[0] == 0.25
        assert features[1] == 0.05
        # Odds should be NaN
        assert math.isnan(features[2])

    def test_handles_none_inputs(self):
        features = extract_meta_features(
            wp=0.20, wp_margin=None, odds=None,
            field_size=12, distance_bucket=None, class_bucket=None,
            track_cond_bucket=None, venue_type=None,
            barrier_relative=None, age=None,
            days_since=None, form_score=None,
            form_trend=None, value_rating=None,
            speed_map_pos=None, weight_diff=None,
            career_win_pct=None, career_place_pct=None,
        )
        assert len(features) == NUM_META_FEATURES
        assert features[0] == 0.20
        # None should become NaN
        assert math.isnan(features[1])  # wp_margin
        assert math.isnan(features[2])  # odds

    def test_feature_names_count(self):
        assert len(META_FEATURE_NAMES) == NUM_META_FEATURES
        assert NUM_META_FEATURES == 18


# ── Safe float ──


class TestSafeFloat:
    def test_normal_values(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(0) == 0.0
        assert _safe_float(-1.5) == -1.5

    def test_none_returns_nan(self):
        assert math.isnan(_safe_float(None))

    def test_infinity_returns_nan(self):
        assert math.isnan(_safe_float(float("inf")))
        assert math.isnan(_safe_float(float("-inf")))

    def test_nan_returns_nan(self):
        assert math.isnan(_safe_float(float("nan")))

    def test_string_returns_nan(self):
        assert math.isnan(_safe_float("not_a_number"))

    def test_numeric_string(self):
        assert _safe_float("3.14") == 3.14


# ── Prediction (model unavailable fallback) ──


class TestPredictFallback:
    def setup_method(self):
        clear_cache()

    def test_predict_returns_negative_when_no_model(self):
        features = _make_features()
        prob = predict_place_probability(features)
        assert prob == -1.0

    def test_should_bet_falls_back_to_wp_above_threshold(self):
        features = _make_features(wp=0.25)
        result, prob, reason = should_bet(features, threshold=0.65, wp=0.25)
        assert result is True
        assert prob == 0.25
        assert "WP fallback" in reason

    def test_should_bet_falls_back_to_wp_below_threshold(self):
        features = _make_features(wp=0.18)
        result, prob, reason = should_bet(features, threshold=0.65, wp=0.18)
        assert result is False
        assert prob == 0.18
        assert "WP fallback" in reason

    def test_should_bet_no_wp_available(self):
        features = _make_features()
        result, prob, reason = should_bet(features, threshold=0.65, wp=None)
        assert result is False
        assert "No model and no WP" in reason

    def test_should_bet_wp_at_exact_threshold(self):
        """WP exactly at 22% should pass."""
        features = _make_features(wp=0.22)
        result, prob, reason = should_bet(features, threshold=0.65, wp=0.22)
        assert result is True


# ── Prediction (model available) ──


class TestPredictWithModel:
    def setup_method(self):
        clear_cache()

    def test_predict_with_mock_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.72]
        mock_model.num_trees.return_value = 50

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()
                prob = predict_place_probability(features)
                assert prob == 0.72

    def test_should_bet_with_model_above_threshold(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75]

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()
                result, prob, reason = should_bet(features, threshold=0.65, wp=0.25)
                assert result is True
                assert prob == 0.75
                assert "meta-model" in reason

    def test_should_bet_with_model_below_threshold(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.55]

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()
                result, prob, reason = should_bet(features, threshold=0.65, wp=0.25)
                assert result is False
                assert prob == 0.55
                assert "meta-model" in reason

    def test_should_bet_exact_threshold(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.65]

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()
                result, prob, reason = should_bet(features, threshold=0.65, wp=0.25)
                assert result is True
                assert prob == 0.65

    def test_predict_handles_model_error(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model error")

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()
                prob = predict_place_probability(features)
                assert prob == -1.0


# ── Meta-model availability ──


class TestMetaModelAvailability:
    def setup_method(self):
        clear_cache()

    def test_not_available_when_no_file(self):
        with patch("punty.betting.meta_model.META_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            from punty.betting.meta_model import meta_model_available
            clear_cache()
            assert meta_model_available() is False

    def test_clear_cache_resets_state(self):
        import punty.betting.meta_model as mm
        mm._meta_failed = True
        clear_cache()
        assert mm._meta_failed is False
        assert mm._meta_model is None


# ── Custom threshold ──


class TestCustomThreshold:
    def setup_method(self):
        clear_cache()

    def test_higher_threshold_rejects_more(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.70]

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()

                # At 0.65 threshold — should pass
                result1, _, _ = should_bet(features, threshold=0.65)
                assert result1 is True

                # At 0.75 threshold — should fail
                result2, _, _ = should_bet(features, threshold=0.75)
                assert result2 is False

    def test_zero_threshold_accepts_all(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.01]

        with patch("punty.betting.meta_model._meta_model", mock_model):
            with patch("punty.betting.meta_model._meta_failed", False):
                features = _make_features()
                result, _, _ = should_bet(features, threshold=0.0)
                assert result is True


# ── Edge cases ──


class TestEdgeCases:
    def test_all_nan_features(self):
        """Model should handle all-NaN features gracefully."""
        features = [float("nan")] * NUM_META_FEATURES
        # Should return -1.0 (no model loaded), not crash
        clear_cache()
        prob = predict_place_probability(features)
        assert prob == -1.0

    def test_empty_features_list(self):
        """Wrong-length features should not crash."""
        clear_cache()
        prob = predict_place_probability([])
        assert prob == -1.0

    def test_very_high_odds(self):
        features = _make_features(odds=101.0)
        assert len(features) == NUM_META_FEATURES
        assert features[2] == 101.0

    def test_very_small_field(self):
        features = _make_features(field_size=3)
        assert features[3] == 3.0

    def test_negative_form_trend(self):
        """Negative trend = improving, should be valid."""
        features = _make_features(form_trend=-2.5)
        assert features[12] == -2.5

    def test_default_wp_threshold_value(self):
        """Ensure the fallback threshold matches the documented 22%."""
        assert DEFAULT_WP_THRESHOLD == 0.22
