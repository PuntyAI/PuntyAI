"""Tests for bet type threshold tuning engine."""

import json
import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from punty.bet_type_tuning import (
    _trim_outliers,
    _wilson_confidence,
    _time_weighted_picks,
    _clamp,
    _smooth,
    _find_optimal_threshold,
    _find_optimal_value_threshold,
    _compute_selection_thresholds,
    _compute_exotic_thresholds,
    _compute_sequence_thresholds,
    _bucket_performance,
    load_bet_thresholds,
    analyze_selection_performance,
    analyze_exotic_performance,
    analyze_sequence_performance,
    maybe_tune_bet_thresholds,
    get_bet_type_dashboard,
    DEFAULT_SELECTION_THRESHOLDS,
    DEFAULT_EXOTIC_THRESHOLDS,
    DEFAULT_SEQUENCE_THRESHOLDS,
    BOUNDS,
    SMOOTHING,
    PROB_BUCKETS,
    VALUE_BUCKETS,
    MIN_CONFIDENCE_SAMPLES,
)


# ── Helper factories ────────────────────────────────────────────────────────


def _make_pick(
    bet_type="win",
    win_prob=0.25,
    place_prob=0.50,
    value_rating=1.1,
    odds=4.0,
    place_odds=1.80,
    hit=True,
    pnl=3.0,
    tip_rank=1,
    exotic_type=None,
    exotic_stake=None,
    sequence_variant=None,
    confidence=None,
    created_at=None,
):
    return {
        "id": "test",
        "bet_type": bet_type,
        "win_prob": win_prob,
        "place_prob": place_prob,
        "value_rating": value_rating,
        "odds": odds,
        "place_odds": place_odds,
        "hit": hit,
        "pnl": pnl,
        "tip_rank": tip_rank,
        "exotic_type": exotic_type,
        "exotic_stake": exotic_stake,
        "sequence_type": None,
        "sequence_variant": sequence_variant,
        "confidence": confidence,
        "created_at": created_at or datetime(2026, 2, 10),
    }


# ── Statistical helpers ─────────────────────────────────────────────────────


class TestTrimOutliers:
    def test_removes_extreme_pnl(self):
        picks = [_make_pick(pnl=2.0) for _ in range(50)]
        picks.append(_make_pick(pnl=500.0))  # extreme outlier
        result = _trim_outliers(picks)
        assert len(result) == 50  # outlier removed

    def test_keeps_normal_variation(self):
        picks = [_make_pick(pnl=i) for i in range(-5, 6)]
        result = _trim_outliers(picks)
        assert len(result) == len(picks)

    def test_handles_empty_list(self):
        assert _trim_outliers([]) == []

    def test_handles_small_list(self):
        picks = [_make_pick(pnl=1.0), _make_pick(pnl=2.0)]
        assert _trim_outliers(picks) == picks

    def test_handles_all_same_pnl(self):
        picks = [_make_pick(pnl=5.0) for _ in range(10)]
        result = _trim_outliers(picks)
        assert len(result) == 10


class TestWilsonConfidence:
    def test_perfect_hit_rate(self):
        lo, hi = _wilson_confidence(10, 10)
        assert lo > 0.7
        assert hi <= 1.0

    def test_zero_hit_rate(self):
        lo, hi = _wilson_confidence(0, 10)
        assert lo == 0.0
        assert hi < 0.3

    def test_fifty_percent(self):
        lo, hi = _wilson_confidence(50, 100)
        assert 0.4 < lo < 0.5
        assert 0.5 < hi < 0.6

    def test_empty_sample(self):
        lo, hi = _wilson_confidence(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_small_sample(self):
        lo, hi = _wilson_confidence(1, 3)
        assert lo < 0.33
        assert hi > 0.33


class TestTimeWeightedPicks:
    def test_recent_picks_weighted_more(self):
        cutoff = datetime(2026, 1, 1)
        recent = _make_pick(created_at=datetime(2026, 2, 1))  # within 30d
        older = _make_pick(created_at=datetime(2026, 1, 10))  # >30d ago
        result = _time_weighted_picks([recent, older], cutoff)
        recent_count = sum(1 for p in result if p["created_at"] == recent["created_at"])
        older_count = sum(1 for p in result if p["created_at"] == older["created_at"])
        assert recent_count == 3
        assert older_count == 2


class TestClamp:
    def test_clamp_below_min(self):
        assert _clamp(0.05, "win_min_prob") == BOUNDS["win_min_prob"][0]

    def test_clamp_above_max(self):
        assert _clamp(0.50, "win_min_prob") == BOUNDS["win_min_prob"][1]

    def test_within_bounds(self):
        assert _clamp(0.20, "win_min_prob") == 0.20

    def test_unknown_key_passthrough(self):
        assert _clamp(99.9, "unknown_key") == 99.9


class TestSmooth:
    def test_smoothing_ratio(self):
        result = _smooth(0.18, 0.22)
        expected = SMOOTHING * 0.18 + (1 - SMOOTHING) * 0.22
        assert abs(result - expected) < 0.001

    def test_no_change(self):
        assert _smooth(0.20, 0.20) == 0.20


# ── Threshold finding ───────────────────────────────────────────────────────


class TestFindOptimalThreshold:
    def test_finds_profitable_threshold(self):
        """Picks with higher win_prob should have better ROI."""
        picks = []
        # Low prob picks lose money
        for _ in range(30):
            picks.append(_make_pick(win_prob=0.10, hit=False, pnl=-5.0))
        # High prob picks make money
        for _ in range(30):
            picks.append(_make_pick(win_prob=0.25, hit=True, pnl=10.0))
        for _ in range(10):
            picks.append(_make_pick(win_prob=0.25, hit=False, pnl=-5.0))

        result = _find_optimal_threshold(picks, "win_prob", "win")
        # Should find a threshold where ROI is positive
        assert result["threshold"] > 0
        assert result["roi"] > 0

    def test_returns_zero_if_insufficient_samples(self):
        picks = [_make_pick(win_prob=0.30, hit=True, pnl=5.0)]
        result = _find_optimal_threshold(picks, "win_prob", "win")
        assert result["threshold"] == 0.0

    def test_handles_all_losers(self):
        picks = [_make_pick(win_prob=0.15, hit=False, pnl=-5.0) for _ in range(20)]
        result = _find_optimal_threshold(picks, "win_prob", "win")
        # Should still return a result (best of bad options)
        assert isinstance(result["roi"], float)


class TestFindOptimalValueThreshold:
    def test_finds_profitable_value(self):
        picks = []
        # Low value exotics lose
        for _ in range(15):
            picks.append(_make_pick(value_rating=0.8, hit=False, pnl=-20.0, exotic_stake=20.0))
        # High value exotics win
        for _ in range(10):
            picks.append(_make_pick(value_rating=1.5, hit=True, pnl=100.0, exotic_stake=20.0))

        result = _find_optimal_value_threshold(picks)
        # Should find a threshold where ROI is positive
        assert result["threshold"] > 0
        assert result["roi"] > 0

    def test_returns_zero_if_no_data(self):
        result = _find_optimal_value_threshold([])
        assert result["threshold"] == 0.0


# ── Bucket performance ──────────────────────────────────────────────────────


class TestBucketPerformance:
    def test_groups_by_probability(self):
        picks = [
            _make_pick(win_prob=0.12, hit=True, pnl=5.0),
            _make_pick(win_prob=0.12, hit=False, pnl=-5.0),
            _make_pick(win_prob=0.22, hit=True, pnl=8.0),
        ]
        buckets = [(0.10, 0.15), (0.15, 0.25)]
        result = _bucket_performance(picks, "win_prob", buckets)
        assert len(result) == 2
        assert result[0]["count"] == 2
        assert result[0]["strike_rate"] == 0.5
        assert result[1]["count"] == 1
        assert result[1]["strike_rate"] == 1.0

    def test_empty_bucket(self):
        picks = [_make_pick(win_prob=0.50)]
        buckets = [(0.10, 0.20)]
        result = _bucket_performance(picks, "win_prob", buckets)
        assert result[0]["count"] == 0

    def test_includes_confidence_intervals(self):
        picks = [_make_pick(win_prob=0.15, hit=True, pnl=3.0) for _ in range(20)]
        buckets = [(0.10, 0.20)]
        result = _bucket_performance(picks, "win_prob", buckets)
        assert result[0]["lower_ci"] > 0
        assert result[0]["upper_ci"] <= 1.0


# ── Threshold computation ───────────────────────────────────────────────────


class TestComputeSelectionThresholds:
    def test_smooths_toward_optimal(self):
        analysis = {
            "win": {
                "count": 50,
                "optimal_threshold": {"threshold": 0.22, "sample_size": 50},
            },
            "place": {
                "count": 30,
                "optimal_threshold": {"threshold": 0.38, "sample_size": 30},
            },
        }
        current = {**DEFAULT_SELECTION_THRESHOLDS}
        result = _compute_selection_thresholds(analysis, current)

        # Should move toward 0.22 but not jump there
        assert result["win_min_prob"] > current["win_min_prob"]
        assert result["win_min_prob"] < 0.22

    def test_respects_bounds(self):
        analysis = {
            "win": {
                "count": 50,
                "optimal_threshold": {"threshold": 0.50, "sample_size": 50},
            },
        }
        current = {**DEFAULT_SELECTION_THRESHOLDS}
        result = _compute_selection_thresholds(analysis, current)
        assert result["win_min_prob"] <= BOUNDS["win_min_prob"][1]

    def test_skips_low_sample(self):
        analysis = {
            "win": {
                "count": 3,
                "optimal_threshold": {"threshold": 0.30, "sample_size": 3},
            },
        }
        current = {**DEFAULT_SELECTION_THRESHOLDS}
        result = _compute_selection_thresholds(analysis, current)
        assert result["win_min_prob"] == current["win_min_prob"]

    def test_no_change_on_empty_analysis(self):
        result = _compute_selection_thresholds({}, {**DEFAULT_SELECTION_THRESHOLDS})
        assert result == DEFAULT_SELECTION_THRESHOLDS


class TestComputeExoticThresholds:
    def test_smooths_value_threshold(self):
        analysis = {
            "Trifecta": {
                "count": 20,
                "optimal_value": {"threshold": 1.5, "sample_size": 20},
            },
        }
        current = {**DEFAULT_EXOTIC_THRESHOLDS}
        result = _compute_exotic_thresholds(analysis, current)
        assert result["min_value_trifecta"] > current["min_value_trifecta"]
        assert result["min_value_trifecta"] < 1.5

    def test_respects_bounds(self):
        analysis = {
            "Exacta": {
                "count": 20,
                "optimal_value": {"threshold": 5.0, "sample_size": 20},
            },
        }
        current = {**DEFAULT_EXOTIC_THRESHOLDS}
        result = _compute_exotic_thresholds(analysis, current)
        assert result["min_value_exacta"] <= BOUNDS["min_value_exacta"][1]


class TestComputeSequenceThresholds:
    def test_widens_on_low_hit_rate(self):
        analysis = {
            "skinny": {
                "count": 30,
                "hit_rate": 0.02,  # well below target 0.08
                "roi": -0.30,
            },
        }
        current = {**DEFAULT_SEQUENCE_THRESHOLDS}
        result = _compute_sequence_thresholds(analysis, current)
        # Should recommend wider legs (at least some increase after smoothing)
        # With 70/30 smoothing, a +1 width becomes partial increase
        for conf in ("high", "med", "low"):
            key = f"width_{conf}_skinny"
            assert result[key] >= current[key]

    def test_narrows_on_high_hit_low_roi(self):
        analysis = {
            "wide": {
                "count": 30,
                "hit_rate": 0.30,  # above 0.18 * 1.3 = 0.234
                "roi": -0.10,     # losing money
            },
        }
        current = {**DEFAULT_SEQUENCE_THRESHOLDS}
        result = _compute_sequence_thresholds(analysis, current)
        # Should suggest narrowing (hitting fine but too expensive)
        for conf in ("high", "med", "low"):
            key = f"width_{conf}_wide"
            assert result[key] <= current[key]

    def test_no_change_on_good_performance(self):
        analysis = {
            "balanced": {
                "count": 30,
                "hit_rate": 0.12,  # right on target
                "roi": 0.05,      # positive
            },
        }
        current = {**DEFAULT_SEQUENCE_THRESHOLDS}
        result = _compute_sequence_thresholds(analysis, current)
        for conf in ("high", "med", "low"):
            key = f"width_{conf}_balanced"
            assert result[key] == current[key]

    def test_respects_width_bounds(self):
        analysis = {
            "skinny": {
                "count": 30,
                "hit_rate": 0.001,  # terrible
                "roi": -0.90,
            },
        }
        current = {"width_high_skinny": 5, "width_med_skinny": 5, "width_low_skinny": 5}
        result = _compute_sequence_thresholds(analysis, current)
        assert result["width_high_skinny"] <= 5
        assert result["width_low_skinny"] <= 5


# ── Load / save ─────────────────────────────────────────────────────────────


class TestLoadBetThresholds:
    @pytest.mark.asyncio
    async def test_returns_defaults_when_no_setting(self):
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result_mock)

        thresholds = await load_bet_thresholds(db)
        assert thresholds["selection"] == DEFAULT_SELECTION_THRESHOLDS
        assert thresholds["exotic"] == DEFAULT_EXOTIC_THRESHOLDS
        assert thresholds["sequence"] == DEFAULT_SEQUENCE_THRESHOLDS

    @pytest.mark.asyncio
    async def test_loads_stored_values(self):
        stored = {
            "selection": {"win_min_prob": 0.22},
            "exotic": {"min_value_trifecta": 1.4},
            "sequence": {"width_high_skinny": 2},
        }
        setting = MagicMock()
        setting.value = json.dumps(stored)

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = setting
        db.execute = AsyncMock(return_value=result_mock)

        thresholds = await load_bet_thresholds(db)
        # Stored value overrides default
        assert thresholds["selection"]["win_min_prob"] == 0.22
        # Default preserved for non-stored keys
        assert thresholds["selection"]["place_min_prob"] == DEFAULT_SELECTION_THRESHOLDS["place_min_prob"]

    @pytest.mark.asyncio
    async def test_handles_corrupt_json(self):
        setting = MagicMock()
        setting.value = "not json"

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = setting
        db.execute = AsyncMock(return_value=result_mock)

        thresholds = await load_bet_thresholds(db)
        assert thresholds["selection"] == DEFAULT_SELECTION_THRESHOLDS


# ── Integration: maybe_tune_bet_thresholds ──────────────────────────────────


class TestMaybeTuneBetThresholds:
    @pytest.mark.asyncio
    async def test_skips_on_cooldown(self):
        db = AsyncMock()
        # Return recent tuning log timestamp
        recent = datetime(2026, 2, 14, 10, 0)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = recent
        db.execute = AsyncMock(return_value=result_mock)

        with patch("punty.bet_type_tuning.melb_now_naive", return_value=datetime(2026, 2, 14, 12, 0)):
            result = await maybe_tune_bet_thresholds(db)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_on_no_picks(self):
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Cooldown check — no recent tuning
                result.scalar_one_or_none.return_value = None
            else:
                # All pick queries + threshold load return empty
                result.scalar_one_or_none.return_value = None
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute
        db.add = MagicMock()
        db.commit = AsyncMock()

        with patch("punty.bet_type_tuning.melb_now_naive", return_value=datetime(2026, 2, 14, 20, 0)):
            result = await maybe_tune_bet_thresholds(db)
        assert result is None


# ── Dashboard data ──────────────────────────────────────────────────────────


class TestGetBetTypeDashboard:
    @pytest.mark.asyncio
    async def test_returns_all_sections(self):
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        result_mock.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result_mock)

        data = await get_bet_type_dashboard(db)
        assert "selection" in data
        assert "exotic" in data
        assert "sequence" in data
        assert "current_thresholds" in data


# ── Defaults match current hardcoded values ─────────────────────────────────


class TestDefaultsMatchHardcoded:
    """Verify defaults match the current hardcoded constants in pre_selections.py."""

    def test_win_min_prob(self):
        assert DEFAULT_SELECTION_THRESHOLDS["win_min_prob"] == 0.18

    def test_win_min_value(self):
        assert DEFAULT_SELECTION_THRESHOLDS["win_min_value"] == 0.90

    def test_saver_win_min_prob(self):
        assert DEFAULT_SELECTION_THRESHOLDS["saver_win_min_prob"] == 0.14

    def test_place_min_prob(self):
        assert DEFAULT_SELECTION_THRESHOLDS["place_min_prob"] == 0.35

    def test_place_min_value(self):
        assert DEFAULT_SELECTION_THRESHOLDS["place_min_value"] == 0.95

    def test_each_way_min_prob(self):
        assert DEFAULT_SELECTION_THRESHOLDS["each_way_min_prob"] == 0.15

    def test_each_way_max_prob(self):
        assert DEFAULT_SELECTION_THRESHOLDS["each_way_max_prob"] == 0.40

    def test_each_way_min_odds(self):
        assert DEFAULT_SELECTION_THRESHOLDS["each_way_min_odds"] == 4.0

    def test_each_way_max_odds(self):
        assert DEFAULT_SELECTION_THRESHOLDS["each_way_max_odds"] == 20.0

    def test_exotic_puntys_pick(self):
        assert DEFAULT_EXOTIC_THRESHOLDS["puntys_pick_value"] == 1.5

    def test_sequence_skinny_high(self):
        assert DEFAULT_SEQUENCE_THRESHOLDS["width_high_skinny"] == 1

    def test_sequence_balanced_med(self):
        assert DEFAULT_SEQUENCE_THRESHOLDS["width_med_balanced"] == 3

    def test_sequence_wide_low(self):
        assert DEFAULT_SEQUENCE_THRESHOLDS["width_low_wide"] == 4
