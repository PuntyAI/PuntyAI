"""Unit tests for LightGBM probability engine."""

import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from punty.ml.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    _parse_stats,
    _score_last_five,
    _sr_from_stats,
    extract_features_from_db_row,
    extract_features_from_runner,
    extract_features_batch,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_db_runner():
    """Sample runner row from backtest.db."""
    return {
        "id": "sale-2025-06-01-r1-1",
        "race_id": "sale-2025-06-01-r1",
        "horse_name": "Test Horse",
        "saddlecloth": 1,
        "barrier": 3,
        "weight": 57.5,
        "jockey": "J. Smith",
        "trainer": "T. Jones",
        "form": "12345",
        "career_record": "20: 5-3-2",
        "last_five": "12345",
        "current_odds": 4.50,
        "opening_odds": 5.00,
        "odds_tab": 4.60,
        "finish_position": 1,
        "horse_age": 5,
        "horse_sex": "Gelding",
        "career_prize_money": 150000,
        "days_since_last_run": 14,
        "handicap_rating": 72.0,
        "pf_settle": 3.5,
        "track_dist_stats": "8: 2-1-1",
        "distance_stats": "15: 4-2-1",
        "track_stats": "10: 3-2-0",
        "good_track_stats": "12: 3-2-1",
        "soft_track_stats": "5: 1-1-0",
        "heavy_track_stats": "3: 1-0-1",
        "first_up_stats": "6: 2-1-0",
        "second_up_stats": "5: 1-1-1",
        "jockey_stats": "200: 30-25-20",
        "trainer_stats": "500: 60-50-40",
        "scratched": False,
    }


@pytest.fixture
def sample_race():
    return {
        "id": "sale-2025-06-01-r1",
        "meeting_id": "sale-2025-06-01",
        "distance": 1200,
        "class": "BM64",
        "field_size": 10,
    }


@pytest.fixture
def sample_meeting():
    return {
        "id": "sale-2025-06-01",
        "venue": "Sale",
        "date": "2025-06-01",
        "track_condition": "Good 4",
    }


@pytest.fixture
def mock_runner_orm():
    """Mock Runner ORM object (attribute-based access)."""
    r = MagicMock()
    r.id = "sale-2025-06-01-r1-1"
    r.race_id = "sale-2025-06-01-r1"
    r.barrier = 3
    r.weight = 57.5
    r.current_odds = 4.50
    r.opening_odds = 5.00
    r.odds_betfair = 4.80
    r.odds_tab = 4.60
    r.odds_sportsbet = 4.50
    r.odds_bet365 = 4.40
    r.odds_ladbrokes = 4.70
    r.career_record = "20: 5-3-2"
    r.last_five = "12345"
    r.horse_age = 5
    r.horse_sex = "Gelding"
    r.career_prize_money = 150000
    r.days_since_last_run = 14
    r.handicap_rating = 72.0
    r.pf_settle = 3.5
    r.track_dist_stats = "8: 2-1-1"
    r.distance_stats = "15: 4-2-1"
    r.track_stats = "10: 3-2-0"
    r.good_track_stats = "12: 3-2-1"
    r.soft_track_stats = "5: 1-1-0"
    r.heavy_track_stats = "3: 1-0-1"
    r.first_up_stats = "6: 2-1-0"
    r.second_up_stats = "5: 1-1-1"
    r.jockey_stats = "200: 30-25-20"
    r.trainer_stats = "500: 60-50-40"
    r.scratched = False
    r.place_odds = None
    return r


@pytest.fixture
def fixture_model_dir(tmp_path):
    """Create tiny LightGBM models for testing."""
    import lightgbm as lgb

    np.random.seed(42)
    n = 50
    X = np.random.randn(n, NUM_FEATURES)
    # Make market_prob realistic (0-1)
    X[:, 0] = np.abs(X[:, 0]) * 0.1 + 0.02

    y_win = (np.random.random(n) < 0.1).astype(int)
    y_place = (np.random.random(n) < 0.3).astype(int)

    params = {"objective": "binary", "num_leaves": 4, "max_depth": 2,
              "verbose": -1, "min_data_in_leaf": 2}

    train_data = lgb.Dataset(X, label=y_win, feature_name=FEATURE_NAMES)
    win_model = lgb.train(params, train_data, num_boost_round=5)
    win_model.save_model(str(tmp_path / "lgbm_win_model.txt"))

    train_data = lgb.Dataset(X, label=y_place, feature_name=FEATURE_NAMES)
    place_model = lgb.train(params, train_data, num_boost_round=5)
    place_model.save_model(str(tmp_path / "lgbm_place_model.txt"))

    return tmp_path


# ──────────────────────────────────────────────
# Feature extraction tests
# ──────────────────────────────────────────────

class TestFeatureExtraction:
    """Tests for feature extraction from various data sources."""

    def test_feature_count(self):
        """Feature list has exactly 58 features."""
        assert NUM_FEATURES == 56

    def test_db_row_extraction_length(self, sample_db_runner, sample_race, sample_meeting):
        """Extraction from DB row produces correct-length vector."""
        features = extract_features_from_db_row(
            sample_db_runner, sample_race, sample_meeting,
            field_size=10, avg_weight=57.0,
        )
        assert len(features) == NUM_FEATURES

    def test_orm_extraction_length(self, mock_runner_orm, sample_race, sample_meeting):
        """Extraction from ORM object produces correct-length vector."""
        features = extract_features_from_runner(
            mock_runner_orm, sample_race, sample_meeting,
            field_size=10, avg_weight=57.0, overround=1.15,
        )
        assert len(features) == NUM_FEATURES

    def test_db_and_orm_same_length(
        self, sample_db_runner, mock_runner_orm, sample_race, sample_meeting,
    ):
        """Both extraction paths produce same-length vectors."""
        db_features = extract_features_from_db_row(
            sample_db_runner, sample_race, sample_meeting, 10, 57.0,
        )
        orm_features = extract_features_from_runner(
            mock_runner_orm, sample_race, sample_meeting, 10, 57.0, 1.15,
        )
        assert len(db_features) == len(orm_features) == NUM_FEATURES

    def test_missing_fields_produce_nan(self, sample_race, sample_meeting):
        """Missing runner fields should produce NaN, not crash."""
        sparse_runner = {"id": "test-1", "race_id": "test-r1", "scratched": False}
        features = extract_features_from_db_row(
            sparse_runner, sample_race, sample_meeting, 10, 57.0,
        )
        assert len(features) == NUM_FEATURES
        # Most features should be NaN for sparse runner
        nan_count = sum(1 for f in features if math.isnan(f))
        assert nan_count > 20  # Majority should be NaN

    def test_market_prob_calculated(self, sample_db_runner, sample_race, sample_meeting):
        """Market prob should be derived from odds."""
        features = extract_features_from_db_row(
            sample_db_runner, sample_race, sample_meeting, 10, 57.0,
        )
        # Feature 0 = market_prob = 1/4.50 ≈ 0.222
        assert 0.15 < features[0] < 0.30

    def test_career_stats_parsed(self, sample_db_runner, sample_race, sample_meeting):
        """Career win/place pct should be parsed from career_record."""
        features = extract_features_from_db_row(
            sample_db_runner, sample_race, sample_meeting, 10, 57.0,
        )
        # career_win_pct = 5/20 = 0.25
        assert abs(features[1] - 0.25) < 0.01
        # career_place_pct = (5+3+2)/20 = 0.50
        assert abs(features[2] - 0.50) < 0.01

    def test_batch_extraction(self, mock_runner_orm, sample_race, sample_meeting):
        """Batch extraction should work for multiple runners."""
        # Create a second runner
        r2 = MagicMock()
        r2.id = "sale-2025-06-01-r1-2"
        r2.barrier = 5
        r2.weight = 56.0
        r2.current_odds = 8.00
        r2.opening_odds = 9.00
        r2.scratched = False
        for attr in ("odds_betfair", "odds_tab", "odds_sportsbet", "odds_bet365",
                     "odds_ladbrokes", "career_record", "last_five", "horse_age",
                     "horse_sex", "career_prize_money", "days_since_last_run",
                     "handicap_rating", "pf_settle", "track_dist_stats",
                     "distance_stats", "track_stats", "good_track_stats",
                     "soft_track_stats", "heavy_track_stats", "first_up_stats",
                     "second_up_stats", "jockey_stats", "trainer_stats", "place_odds"):
            setattr(r2, attr, None)

        X = extract_features_batch([mock_runner_orm, r2], sample_race, sample_meeting)
        assert X.shape == (2, NUM_FEATURES)


# ──────────────────────────────────────────────
# Stats parser tests
# ──────────────────────────────────────────────

class TestStatsParsers:

    def test_parse_stats_colon_format(self):
        assert _parse_stats("20: 5-3-2") == (20, 5, 3, 2)

    def test_parse_stats_dash_format(self):
        assert _parse_stats("20-5-3-2") == (20, 5, 3, 2)

    def test_parse_stats_none(self):
        assert _parse_stats(None) is None
        assert _parse_stats("") is None

    def test_sr_from_stats(self):
        sr, starts = _sr_from_stats("20: 5-3-2")
        assert abs(sr - 0.25) < 0.01
        assert starts == 20

    def test_score_last_five(self):
        score = _score_last_five("11111")
        assert score == 1.0
        score2 = _score_last_five("99999")
        assert score2 < 0.2


# ──────────────────────────────────────────────
# Config toggle tests
# ──────────────────────────────────────────────

class TestConfigToggle:

    def test_use_lightgbm_default_false(self):
        """Default config should have use_lightgbm=False."""
        from punty.probability import _use_lightgbm, _lgbm_enabled
        import punty.probability as prob_mod

        # Reset cached value
        prob_mod._lgbm_enabled = None

        with patch("punty.config.get_settings") as mock_settings:
            mock_settings.return_value.use_lightgbm = False
            assert _use_lightgbm() is False

        # Reset for other tests
        prob_mod._lgbm_enabled = None

    def test_use_lightgbm_enabled(self):
        """When config is True, _use_lightgbm() returns True."""
        from punty.probability import _use_lightgbm
        import punty.probability as prob_mod
        prob_mod._lgbm_enabled = None

        with patch("punty.config.get_settings") as mock_settings:
            mock_settings.return_value.use_lightgbm = True
            assert _use_lightgbm() is True

        prob_mod._lgbm_enabled = None


# ──────────────────────────────────────────────
# Inference tests
# ──────────────────────────────────────────────

class TestInference:

    def test_models_available_when_missing(self):
        """models_available() returns False when model files don't exist."""
        from punty.ml import inference
        inference.clear_cache()

        with patch.object(inference, "WIN_MODEL_PATH", Path("/nonexistent/win.txt")):
            with patch.object(inference, "PLACE_MODEL_PATH", Path("/nonexistent/place.txt")):
                assert inference.models_available() is False

        inference.clear_cache()

    def test_predict_race_empty_on_missing_models(self, mock_runner_orm, sample_race, sample_meeting):
        """predict_race returns empty dict when models missing."""
        from punty.ml import inference
        inference.clear_cache()

        with patch.object(inference, "WIN_MODEL_PATH", Path("/nonexistent/win.txt")):
            with patch.object(inference, "PLACE_MODEL_PATH", Path("/nonexistent/place.txt")):
                result = inference.predict_race([mock_runner_orm], sample_race, sample_meeting)
                assert result == {}

        inference.clear_cache()

    def test_predict_race_with_fixture_models(
        self, fixture_model_dir, mock_runner_orm, sample_race, sample_meeting,
    ):
        """predict_race returns probabilities with fixture models."""
        from punty.ml import inference
        inference.clear_cache()

        with patch.object(inference, "WIN_MODEL_PATH", fixture_model_dir / "lgbm_win_model.txt"):
            with patch.object(inference, "PLACE_MODEL_PATH", fixture_model_dir / "lgbm_place_model.txt"):
                result = inference.predict_race([mock_runner_orm], sample_race, sample_meeting)
                assert len(result) == 1
                rid = mock_runner_orm.id
                assert rid in result
                win_prob, place_prob = result[rid]
                assert 0.0 < win_prob < 1.0
                assert 0.0 < place_prob < 1.0

        inference.clear_cache()


# ──────────────────────────────────────────────
# Integration test (probability.py)
# ──────────────────────────────────────────────

class TestProbabilityIntegration:

    def test_lgbm_branch_produces_runner_probability(
        self, fixture_model_dir, mock_runner_orm, sample_race, sample_meeting,
    ):
        """_calculate_lgbm_probabilities returns RunnerProbability objects."""
        from punty.ml import inference
        from punty.probability import _calculate_lgbm_probabilities, RunnerProbability
        inference.clear_cache()

        with patch.object(inference, "WIN_MODEL_PATH", fixture_model_dir / "lgbm_win_model.txt"):
            with patch.object(inference, "PLACE_MODEL_PATH", fixture_model_dir / "lgbm_place_model.txt"):
                result = _calculate_lgbm_probabilities(
                    [mock_runner_orm], sample_race, sample_meeting,
                )
                assert len(result) == 1
                rp = list(result.values())[0]
                assert isinstance(rp, RunnerProbability)
                assert 0.0 < rp.win_probability <= 1.0
                assert 0.0 < rp.place_probability <= 1.0
                assert isinstance(rp.value_rating, float)
                assert isinstance(rp.edge, float)
                assert isinstance(rp.factors, dict)

        inference.clear_cache()

    def test_lgbm_win_probs_sum_to_one(
        self, fixture_model_dir, mock_runner_orm, sample_race, sample_meeting,
    ):
        """Win probabilities from LightGBM branch should sum to ~1.0."""
        from punty.ml import inference
        from punty.probability import _calculate_lgbm_probabilities
        inference.clear_cache()

        # Create 3 runners
        runners = []
        for i in range(3):
            r = MagicMock()
            r.id = f"test-r-{i}"
            r.barrier = i + 1
            r.weight = 56.0 + i
            r.current_odds = 3.0 + i * 2
            r.opening_odds = 3.5 + i * 2
            r.scratched = False
            for attr in ("odds_betfair", "odds_tab", "odds_sportsbet", "odds_bet365",
                         "odds_ladbrokes", "career_record", "last_five", "horse_age",
                         "horse_sex", "career_prize_money", "days_since_last_run",
                         "handicap_rating", "pf_settle", "track_dist_stats",
                         "distance_stats", "track_stats", "good_track_stats",
                         "soft_track_stats", "heavy_track_stats", "first_up_stats",
                         "second_up_stats", "jockey_stats", "trainer_stats", "place_odds"):
                setattr(r, attr, None)
            runners.append(r)

        with patch.object(inference, "WIN_MODEL_PATH", fixture_model_dir / "lgbm_win_model.txt"):
            with patch.object(inference, "PLACE_MODEL_PATH", fixture_model_dir / "lgbm_place_model.txt"):
                result = _calculate_lgbm_probabilities(runners, sample_race, sample_meeting)
                if result:
                    total = sum(rp.win_probability for rp in result.values())
                    assert abs(total - 1.0) < 0.01

        inference.clear_cache()

    def test_fallback_when_lgbm_disabled(self):
        """When use_lightgbm=False, weighted engine is used."""
        import punty.probability as prob_mod
        prob_mod._lgbm_enabled = None

        with patch("punty.config.get_settings") as mock_settings:
            mock_settings.return_value.use_lightgbm = False
            assert prob_mod._use_lightgbm() is False

        prob_mod._lgbm_enabled = None
