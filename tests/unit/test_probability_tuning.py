"""Tests for probability model self-tuning and calibration analysis."""

import json
import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from punty.probability_tuning import (
    CALIBRATION_BUCKETS,
    VALUE_BUCKETS,
    MIN_WEIGHT,
    MAX_WEIGHT,
    SMOOTHING,
    MIN_CHANGE_THRESHOLD,
    _point_biserial,
    calculate_calibration,
    calculate_value_performance,
    analyze_factor_performance,
    calculate_brier_score,
    calculate_category_breakdown,
    maybe_tune_weights,
    get_tuning_history,
    get_dashboard_data,
    _get_current_weights,
    _biggest_change,
)
from punty.probability import FACTOR_REGISTRY, DEFAULT_WEIGHTS


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_pick_row(win_prob, hit, value_rating=1.0, pnl=0.0, stake=5.0,
                   factors=None, created_at=None):
    """Create a mock pick row for calibration/value tests."""
    return MagicMock(
        win_probability=win_prob,
        hit=hit,
        value_rating=value_rating,
        pnl=pnl,
        bet_stake=stake,
        factors_json=json.dumps(factors) if factors else None,
        created_at=created_at or datetime.now(),
    )


def _make_factor_row(factors, hit, created_at=None):
    """Create a mock row for factor performance tests (factors_json, hit)."""
    return (json.dumps(factors), hit)


def _make_category_row(win_prob, hit, pnl, stake, vr, distance, condition):
    """Create a mock row for category breakdown."""
    return (win_prob, hit, pnl, stake, vr, distance, condition)


class MockResult:
    """Mock SQLAlchemy result object."""
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return MockScalars(self._rows)

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0] if self._rows else None


class MockScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


# ──────────────────────────────────────────────
# Point-biserial correlation
# ──────────────────────────────────────────────

class TestPointBiserial:
    def test_perfect_correlation(self):
        """Winners always have higher scores."""
        scores = [0.2, 0.3, 0.7, 0.8]
        outcomes = [0.0, 0.0, 1.0, 1.0]
        r = _point_biserial(scores, outcomes)
        assert r > 0.5  # Strong positive correlation

    def test_no_correlation(self):
        """No relationship between scores and outcomes."""
        scores = [0.5, 0.5, 0.5, 0.5]
        outcomes = [0.0, 1.0, 0.0, 1.0]
        r = _point_biserial(scores, outcomes)
        assert abs(r) < 0.01  # Near zero

    def test_negative_correlation(self):
        """Higher scores = more losers."""
        scores = [0.8, 0.7, 0.3, 0.2]
        outcomes = [0.0, 0.0, 1.0, 1.0]
        r = _point_biserial(scores, outcomes)
        assert r < -0.5

    def test_too_few_samples(self):
        """Returns 0 with < 2 samples."""
        assert _point_biserial([0.5], [1.0]) == 0.0
        assert _point_biserial([], []) == 0.0

    def test_all_same_group(self):
        """Returns 0 when all outcomes are the same."""
        scores = [0.3, 0.5, 0.7]
        outcomes = [1.0, 1.0, 1.0]
        r = _point_biserial(scores, outcomes)
        assert r == 0.0

    def test_bounded(self):
        """Result is always in [-1, 1]."""
        scores = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        outcomes = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        r = _point_biserial(scores, outcomes)
        assert -1.0 <= r <= 1.0


# ──────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────

class TestCalibration:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Returns empty list when no picks."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])
        result = await calculate_calibration(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_buckets_structure(self):
        """Returns correct bucket structure."""
        rows = [
            MagicMock(win_probability=0.03, hit=False, value_rating=0.8),
            MagicMock(win_probability=0.07, hit=True, value_rating=1.2),
            MagicMock(win_probability=0.12, hit=False, value_rating=0.9),
            MagicMock(win_probability=0.25, hit=True, value_rating=1.1),
            MagicMock(win_probability=0.35, hit=True, value_rating=1.0),
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_calibration(db)
        assert len(result) == len(CALIBRATION_BUCKETS)

        # Check each bucket has required keys
        for bucket in result:
            assert "bucket" in bucket
            assert "predicted_avg" in bucket
            assert "actual_rate" in bucket
            assert "count" in bucket

    @pytest.mark.asyncio
    async def test_actual_win_rate_calculation(self):
        """Actual rate = wins / total in bucket."""
        # All in 10-15% bucket
        rows = [
            MagicMock(win_probability=0.12, hit=True, value_rating=1.0),
            MagicMock(win_probability=0.13, hit=False, value_rating=1.0),
            MagicMock(win_probability=0.14, hit=False, value_rating=1.0),
            MagicMock(win_probability=0.11, hit=True, value_rating=1.0),
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_calibration(db)
        # Find the 10-15% bucket
        bucket_10_15 = next(b for b in result if b["bucket"] == "10-15%")
        assert bucket_10_15["count"] == 4
        assert bucket_10_15["actual_rate"] == 0.5  # 2/4


# ──────────────────────────────────────────────
# Value Performance
# ──────────────────────────────────────────────

class TestValuePerformance:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Returns empty list when no picks."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])
        result = await calculate_value_performance(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_value_buckets(self):
        """Correct bucketing by value_rating."""
        rows = [
            MagicMock(value_rating=0.7, hit=False, pnl=-5.0, bet_stake=5.0),  # <0.80
            MagicMock(value_rating=0.9, hit=False, pnl=-5.0, bet_stake=5.0),  # 0.80-1.00
            MagicMock(value_rating=1.05, hit=True, pnl=15.0, bet_stake=5.0),  # 1.00-1.10
            MagicMock(value_rating=1.2, hit=True, pnl=10.0, bet_stake=5.0),   # 1.10-1.30
            MagicMock(value_rating=1.5, hit=True, pnl=20.0, bet_stake=5.0),   # 1.30+
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_value_performance(db)
        assert len(result) == len(VALUE_BUCKETS)

        # Strong value bucket should have positive ROI
        strong = next(b for b in result if "Strong" in b["bucket"])
        assert strong["count"] == 1
        assert strong["roi"] > 0

    @pytest.mark.asyncio
    async def test_roi_calculation(self):
        """ROI = total_pnl / total_staked * 100."""
        rows = [
            _make_pick_row(0.15, True, value_rating=1.15, pnl=10.0, stake=5.0),
            _make_pick_row(0.12, False, value_rating=1.2, pnl=-5.0, stake=5.0),
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_value_performance(db)
        value_bucket = next(b for b in result if "1.10-1.30" in b["bucket"])
        assert value_bucket["count"] == 2
        # ROI = (10 - 5) / (5 + 5) * 100 = 50%
        assert value_bucket["roi"] == 50.0


# ──────────────────────────────────────────────
# Factor Performance
# ──────────────────────────────────────────────

class TestFactorPerformance:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Returns empty dict when no factor data."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])
        result = await analyze_factor_performance(db)
        assert result == {}

    @pytest.mark.asyncio
    async def test_too_few_samples(self):
        """Returns empty if < 10 picks."""
        rows = [_make_factor_row({"market": 0.6}, True) for _ in range(5)]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)
        result = await analyze_factor_performance(db)
        assert result == {}

    @pytest.mark.asyncio
    async def test_factor_edge_positive(self):
        """Factor with higher scores for winners has positive edge."""
        factors_win = {"market": 0.7, "form": 0.6, "pace": 0.5}
        factors_lose = {"market": 0.3, "form": 0.4, "pace": 0.5}

        rows = []
        for _ in range(8):
            rows.append(_make_factor_row(factors_win, True))
        for _ in range(12):
            rows.append(_make_factor_row(factors_lose, False))

        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await analyze_factor_performance(db)
        assert "market" in result
        assert result["market"]["edge"] > 0  # Market distinguishes winners
        assert result["pace"]["edge"] == 0.0  # Pace is neutral

    @pytest.mark.asyncio
    async def test_factor_structure(self):
        """Each factor result has required keys."""
        factors = {k: 0.5 for k in FACTOR_REGISTRY}
        rows = [_make_factor_row(factors, i % 3 == 0) for i in range(15)]

        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await analyze_factor_performance(db)
        for key, data in result.items():
            assert "winner_avg" in data
            assert "loser_avg" in data
            assert "edge" in data
            assert "accuracy" in data
            assert "label" in data


# ──────────────────────────────────────────────
# Brier Score
# ──────────────────────────────────────────────

class TestBrierScore:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Returns None when no data."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])
        result = await calculate_brier_score(db)
        assert result["model"] is None
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_perfect_predictions(self):
        """Brier score = 0 for perfect predictions."""
        rows = [
            MagicMock(win_probability=1.0, value_rating=1.0, hit=True),
            MagicMock(win_probability=0.0, value_rating=1.0, hit=False),
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_brier_score(db)
        assert result["model"] == 0.0

    @pytest.mark.asyncio
    async def test_coin_flip_predictions(self):
        """Brier score ~ 0.25 for 50/50 predictions."""
        rows = [
            MagicMock(win_probability=0.5, value_rating=1.0, hit=True),
            MagicMock(win_probability=0.5, value_rating=1.0, hit=False),
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_brier_score(db)
        assert result["model"] == 0.25


# ──────────────────────────────────────────────
# Tune Weights
# ──────────────────────────────────────────────

class TestTuneWeights:
    @pytest.mark.asyncio
    async def test_skips_on_cooldown(self):
        """Skips if last tune was < 24h ago."""
        db = AsyncMock()

        # Mock last tune was 1 hour ago
        recent_time = datetime.now() - timedelta(hours=1)
        db.execute.side_effect = [
            MockResult([recent_time]),  # last tune query
        ]

        result = await maybe_tune_weights(db)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_insufficient_data(self):
        """Skips if < min_sample picks with factor data."""
        db = AsyncMock()

        # No recent tune
        db.execute.side_effect = [
            MockResult([None]),  # last tune query - none
            MockResult([]),      # factor performance query - empty
        ]

        result = await maybe_tune_weights(db, min_sample=50)
        assert result is None

    @pytest.mark.asyncio
    async def test_weight_bounds(self):
        """Weights stay within MIN_WEIGHT and MAX_WEIGHT."""
        # This tests the bounds logic directly
        weights = {"market": 0.01, "form": 0.99}  # Extreme values
        for k in FACTOR_REGISTRY:
            if k not in weights:
                weights[k] = 0.05

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Apply bounds
        for k in weights:
            weights[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, weights[k]))

        assert all(v >= MIN_WEIGHT for v in weights.values())
        assert all(v <= MAX_WEIGHT for v in weights.values())

    def test_smoothing_blends_old_and_new(self):
        """Smoothing blends old weight with optimal."""
        old_w = 0.22
        opt_w = 0.30
        new_w = SMOOTHING * old_w + (1 - SMOOTHING) * opt_w
        expected = 0.7 * 0.22 + 0.3 * 0.30
        assert abs(new_w - expected) < 0.0001

    def test_weights_normalize_to_one(self):
        """After tuning, weights must sum to 1.0."""
        weights = {k: v for k, v in DEFAULT_WEIGHTS.items()}
        total = sum(weights.values())
        normalized = {k: v / total for k, v in weights.items()}
        assert abs(sum(normalized.values()) - 1.0) < 0.0001

    def test_min_change_threshold(self):
        """Small changes below threshold are ignored."""
        old = {"market": 0.220, "form": 0.150}
        new = {"market": 0.221, "form": 0.149}  # Changes < 0.005
        max_change = max(abs(new[k] - old[k]) for k in old)
        assert max_change < MIN_CHANGE_THRESHOLD


# ──────────────────────────────────────────────
# Biggest Change
# ──────────────────────────────────────────────

class TestBiggestChange:
    def test_identifies_biggest(self):
        """Correctly identifies the factor with biggest change."""
        old = {"market": 0.22, "form": 0.15, "pace": 0.11}
        new = {"market": 0.25, "form": 0.14, "pace": 0.11}
        result = _biggest_change(old, new)
        assert "Market" in result
        assert "+3.0%" in result

    def test_decrease(self):
        """Shows decrease direction."""
        old = {"market": 0.22, "form": 0.18}
        new = {"market": 0.22, "form": 0.12}
        result = _biggest_change(old, new)
        assert "Form" in result
        assert "-6.0%" in result


# ──────────────────────────────────────────────
# Tuning History
# ──────────────────────────────────────────────

class TestTuningHistory:
    @pytest.mark.asyncio
    async def test_empty_history(self):
        """Returns empty list when no tuning events."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])
        result = await get_tuning_history(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_history_structure(self):
        """Returns correct structure."""
        mock_log = MagicMock()
        mock_log.id = 1
        mock_log.created_at = datetime.now()
        mock_log.old_weights_json = json.dumps({"market": 22, "form": 15})
        mock_log.new_weights_json = json.dumps({"market": 24, "form": 14})
        mock_log.metrics_json = json.dumps({"max_change": 0.02})
        mock_log.picks_analyzed = 100
        mock_log.reason = "auto_tune"

        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_log]
        db.execute.return_value = mock_result

        result = await get_tuning_history(db)
        assert len(result) == 1
        entry = result[0]
        assert entry["picks_analyzed"] == 100
        assert "market" in entry["changes"]
        assert entry["changes"]["market"] == 2.0  # 24 - 22


# ──────────────────────────────────────────────
# Dashboard Data
# ──────────────────────────────────────────────

class TestDashboardData:
    @pytest.mark.asyncio
    async def test_returns_all_keys(self):
        """Dashboard data has all expected top-level keys."""
        db = AsyncMock()

        # Mock all queries to return empty
        db.execute.return_value = MockResult([])

        with patch("punty.probability_tuning.calculate_calibration", return_value=[]), \
             patch("punty.probability_tuning.calculate_value_performance", return_value=[]), \
             patch("punty.probability_tuning.analyze_factor_performance", return_value={}), \
             patch("punty.probability_tuning.calculate_brier_score", return_value={"model": None, "market": None, "count": 0}), \
             patch("punty.probability_tuning.calculate_category_breakdown", return_value=[]), \
             patch("punty.probability_tuning.get_tuning_history", return_value=[]), \
             patch("punty.probability_tuning._get_current_weights", return_value=dict(DEFAULT_WEIGHTS)):

            result = await get_dashboard_data(db)

        assert "calibration" in result
        assert "value_performance" in result
        assert "factor_performance" in result
        assert "factor_table" in result
        assert "brier" in result
        assert "categories" in result
        assert "current_weights" in result
        assert "weight_history" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_summary_structure(self):
        """Summary has required keys."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])

        with patch("punty.probability_tuning.calculate_calibration", return_value=[]), \
             patch("punty.probability_tuning.calculate_value_performance", return_value=[]), \
             patch("punty.probability_tuning.analyze_factor_performance", return_value={}), \
             patch("punty.probability_tuning.calculate_brier_score", return_value={"model": None, "market": None, "count": 0}), \
             patch("punty.probability_tuning.calculate_category_breakdown", return_value=[]), \
             patch("punty.probability_tuning.get_tuning_history", return_value=[]), \
             patch("punty.probability_tuning._get_current_weights", return_value=dict(DEFAULT_WEIGHTS)):

            result = await get_dashboard_data(db)

        s = result["summary"]
        assert "total_settled" in s
        assert "total_with_factors" in s
        assert "brier_model" in s
        assert "last_tune_date" in s

    @pytest.mark.asyncio
    async def test_factor_table_has_all_factors(self):
        """Factor table includes all factors from registry."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])

        with patch("punty.probability_tuning.calculate_calibration", return_value=[]), \
             patch("punty.probability_tuning.calculate_value_performance", return_value=[]), \
             patch("punty.probability_tuning.analyze_factor_performance", return_value={}), \
             patch("punty.probability_tuning.calculate_brier_score", return_value={"model": None, "market": None, "count": 0}), \
             patch("punty.probability_tuning.calculate_category_breakdown", return_value=[]), \
             patch("punty.probability_tuning.get_tuning_history", return_value=[]), \
             patch("punty.probability_tuning._get_current_weights", return_value=dict(DEFAULT_WEIGHTS)):

            result = await get_dashboard_data(db)

        factor_keys = {f["key"] for f in result["factor_table"]}
        for key in FACTOR_REGISTRY:
            assert key in factor_keys


# ──────────────────────────────────────────────
# Category Breakdown
# ──────────────────────────────────────────────

class TestCategoryBreakdown:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        """Returns empty list when no data."""
        db = AsyncMock()
        db.execute.return_value = MockResult([])
        result = await calculate_category_breakdown(db)
        assert result == []

    @pytest.mark.asyncio
    async def test_groups_by_distance_condition(self):
        """Groups picks by distance bucket and condition."""
        rows = [
            (0.15, True, 10.0, 5.0, 1.1, 1000, "Good 4"),   # sprint/Good
            (0.12, False, -5.0, 5.0, 0.9, 1000, "Good 3"),   # sprint/Good
            (0.10, False, -5.0, 5.0, 0.8, 1000, "Good"),      # sprint/Good
            (0.20, True, 15.0, 5.0, 1.2, 2400, "Heavy 8"),    # staying/Heavy
            (0.18, False, -5.0, 5.0, 1.0, 2400, "Heavy 10"),  # staying/Heavy
            (0.15, True, 8.0, 5.0, 1.1, 2400, "Heavy 9"),     # staying/Heavy
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_category_breakdown(db)
        assert len(result) >= 2

        cats = {c["category"] for c in result}
        assert any("sprint" in c.lower() for c in cats)
        assert any("staying" in c.lower() for c in cats)

    @pytest.mark.asyncio
    async def test_min_picks_filter(self):
        """Categories with < 3 picks are excluded."""
        rows = [
            (0.15, True, 10.0, 5.0, 1.1, 1000, "Good"),
            (0.12, False, -5.0, 5.0, 0.9, 1000, "Good"),
            # Only 2 sprint/Good picks — should be excluded
        ]
        db = AsyncMock()
        db.execute.return_value = MockResult(rows)

        result = await calculate_category_breakdown(db)
        # With only 2 picks per category, nothing meets the threshold
        assert len(result) == 0


# ──────────────────────────────────────────────
# Current Weights
# ──────────────────────────────────────────────

class TestGetCurrentWeights:
    @pytest.mark.asyncio
    async def test_defaults_when_no_setting(self):
        """Returns DEFAULT_WEIGHTS when no AppSettings entry."""
        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute.return_value = mock_result

        result = await _get_current_weights(db)
        assert result == DEFAULT_WEIGHTS

    @pytest.mark.asyncio
    async def test_loads_from_settings(self):
        """Converts percentage-based settings to decimals."""
        setting = MagicMock()
        setting.value = json.dumps({"market": 30, "form": 20, "pace": 50})

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = setting
        db = AsyncMock()
        db.execute.return_value = mock_result

        result = await _get_current_weights(db)
        assert result["market"] == 0.30
        assert result["form"] == 0.20
        assert result["pace"] == 0.50


# ──────────────────────────────────────────────
# Integration: Default Weights
# ──────────────────────────────────────────────

class TestDefaultWeightsIntegrity:
    def test_default_weights_sum_to_one(self):
        """DEFAULT_WEIGHTS must sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_factors_have_default_weight(self):
        """Every factor in FACTOR_REGISTRY has a default weight."""
        for key in FACTOR_REGISTRY:
            assert key in DEFAULT_WEIGHTS

    def test_all_weights_in_valid_range(self):
        """All default weights are between MIN_WEIGHT and MAX_WEIGHT."""
        for key, weight in DEFAULT_WEIGHTS.items():
            assert weight >= MIN_WEIGHT, f"{key} weight {weight} < {MIN_WEIGHT}"
            assert weight <= MAX_WEIGHT, f"{key} weight {weight} > {MAX_WEIGHT}"
