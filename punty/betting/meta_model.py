"""Betfair meta-model — learned bet selector for the LGBM rank 1 pick.

A lightweight LightGBM binary classifier trained on "does our LGBM's top pick
actually place?" with sample weights proportional to place dividends (optimise
ROI, not just strike rate).

Runs ON TOP of the main LGBM: after the rank model picks the best runner,
the meta-model decides whether the race context predicts a reliable place result.

Graceful fallback: if model file not found, falls back to WP >= 22% threshold.
"""

import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
META_MODEL_PATH = MODEL_DIR / "bf_meta_model.txt"
META_METADATA_PATH = MODEL_DIR / "bf_meta_metadata.json"

# Module-level cache — loaded once on first call
_meta_model: Optional[Any] = None
_meta_failed: bool = False

# Meta-feature names — ORDER MUST MATCH extraction
META_FEATURE_NAMES = [
    "wp",                 # main LGBM's win probability for this runner
    "wp_margin",          # gap between rank 1 and rank 2 WP (confidence margin)
    "odds",               # market odds (shorter = safer)
    "field_size",         # number of runners
    "distance_bucket",    # 1-5 (sprint to staying)
    "class_bucket",       # 1-6 (maiden to group)
    "track_cond_bucket",  # 1-5 (firm to synthetic)
    "venue_type",         # 1=metro, 3=country
    "barrier_relative",   # 0-1 (inside to outside)
    "age",                # horse age
    "days_since",         # days since last run
    "form_score",         # last 5 form score
    "form_trend",         # improving/declining (slope)
    "value_rating",       # WP / market implied (overlay)
    "speed_map_pos",      # 1-4 (leader to backmarker)
    "weight_diff",        # relative to field average
    "career_win_pct",     # career strike rate
    "career_place_pct",   # career place rate
]

NUM_META_FEATURES = len(META_FEATURE_NAMES)

# Default WP threshold — used as fallback when meta-model unavailable
DEFAULT_WP_THRESHOLD = 0.22


def _load_meta_model() -> bool:
    """Load meta-model from disk. Returns True if loaded successfully."""
    global _meta_model, _meta_failed

    if _meta_model is not None:
        return True
    if _meta_failed:
        return False

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — meta-model unavailable")
        _meta_failed = True
        return False

    if not META_MODEL_PATH.exists():
        logger.info("Meta-model not found at %s — using WP fallback", META_MODEL_PATH)
        _meta_failed = True
        return False

    try:
        _meta_model = lgb.Booster(model_file=str(META_MODEL_PATH))
        logger.info(
            "Betfair meta-model loaded: %d trees from %s",
            _meta_model.num_trees(), META_MODEL_PATH,
        )
        return True
    except Exception as e:
        logger.error("Failed to load meta-model: %s", e)
        _meta_failed = True
        return False


def meta_model_available() -> bool:
    """Check if the meta-model is loaded or loadable."""
    if _meta_model is not None:
        return True
    if _meta_failed:
        return False
    return _load_meta_model()


def _safe_float(val: Any) -> float:
    """Convert to float, NaN if None or invalid."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def extract_meta_features(
    wp: float,
    wp_margin: float,
    odds: float,
    field_size: int,
    distance_bucket: float,
    class_bucket: float,
    track_cond_bucket: float,
    venue_type: float,
    barrier_relative: float,
    age: float,
    days_since: float,
    form_score: float,
    form_trend: float,
    value_rating: float,
    speed_map_pos: float,
    weight_diff: float,
    career_win_pct: float,
    career_place_pct: float,
) -> list[float]:
    """Build the meta-feature vector. Order must match META_FEATURE_NAMES."""
    return [
        _safe_float(wp),
        _safe_float(wp_margin),
        _safe_float(odds),
        float(field_size),
        _safe_float(distance_bucket),
        _safe_float(class_bucket),
        _safe_float(track_cond_bucket),
        _safe_float(venue_type),
        _safe_float(barrier_relative),
        _safe_float(age),
        _safe_float(days_since),
        _safe_float(form_score),
        _safe_float(form_trend),
        _safe_float(value_rating),
        _safe_float(speed_map_pos),
        _safe_float(weight_diff),
        _safe_float(career_win_pct),
        _safe_float(career_place_pct),
    ]


def predict_place_probability(features: list[float]) -> float:
    """Predict probability that rank 1 pick will place.

    Args:
        features: Meta-feature vector (18 floats, order per META_FEATURE_NAMES).

    Returns:
        Float 0-1 probability. Returns -1.0 if model unavailable.
    """
    if not _load_meta_model():
        return -1.0

    try:
        X = np.array([features], dtype=np.float64)
        prob = _meta_model.predict(X)[0]
        return float(prob)
    except Exception as e:
        logger.error("Meta-model prediction failed: %s", e)
        return -1.0


def should_bet(
    features: list[float],
    threshold: float = 0.65,
    wp: Optional[float] = None,
) -> tuple[bool, float, str]:
    """Decide whether to bet on this pick.

    Args:
        features: Meta-feature vector.
        threshold: Meta-model probability threshold for betting.
        wp: Win probability (used for fallback if model unavailable).

    Returns:
        Tuple of (should_bet, probability, reason).
        reason describes why the decision was made.
    """
    prob = predict_place_probability(features)

    if prob < 0:
        # Model unavailable — fall back to WP threshold
        if wp is not None and wp >= DEFAULT_WP_THRESHOLD:
            return True, wp, f"WP fallback ({wp:.0%} >= {DEFAULT_WP_THRESHOLD:.0%})"
        elif wp is not None:
            return False, wp, f"WP fallback ({wp:.0%} < {DEFAULT_WP_THRESHOLD:.0%})"
        else:
            return False, 0.0, "No model and no WP available"

    if prob >= threshold:
        return True, prob, f"meta-model ({prob:.0%} >= {threshold:.0%})"
    else:
        return False, prob, f"meta-model ({prob:.0%} < {threshold:.0%})"


def clear_cache():
    """Clear cached model (for testing or reload)."""
    global _meta_model, _meta_failed
    _meta_model = None
    _meta_failed = False
