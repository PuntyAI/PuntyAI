"""Bet type meta-model — learned Win/Each Way/Place selector per pick.

A LightGBM multi-class classifier trained on Proform historical data that
predicts the most profitable bet type (Win, Each Way, Place) for each
selection given race context and pick quality.

Replaces hard-coded _determine_bet_type() rules with learned decisions
from ~79K simulated selections across a full year of racing.

Graceful fallback: if model file not found, returns None and caller uses
existing rules.
"""

import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
BETTYPE_MODEL_PATH = MODEL_DIR / "bettype_meta_model.txt"
BETTYPE_METADATA_PATH = MODEL_DIR / "bettype_meta_metadata.json"

# Module-level cache
_bt_model: Optional[Any] = None
_bt_failed: bool = False

# Bet type classes — model predicts index, we map to label
BET_TYPE_CLASSES = ["Win", "Each Way", "Place"]

# Feature names — ORDER MUST MATCH extraction
BETTYPE_FEATURE_NAMES = [
    # Pick identity (2 features)
    "tip_rank",           # 1-4
    "is_roughie",         # 0 or 1

    # Pick quality (4 features)
    "win_prob",           # LGBM win probability for this pick
    "place_prob",         # Harville place probability
    "value_rating",       # WP / market implied (overlay)
    "odds",               # market odds

    # Race context (6 features)
    "field_size",         # number of runners
    "distance_bucket",    # 1-5
    "class_bucket",       # 1-6
    "track_cond_bucket",  # 1-5
    "venue_type",         # 1=metro, 2=provincial, 3=country
    "prize_money_bucket", # 1-4

    # Race shape (3 features)
    "rank1_wp",           # top pick's WP (confidence indicator)
    "wp_spread",          # rank1 - rank3 WP
    "gap_to_next",        # WP gap from this pick to next rank

    # Market context (2 features)
    "fav_odds",           # favourite odds
    "place_odds",         # place odds for this pick (if available)
]

NUM_BETTYPE_FEATURES = len(BETTYPE_FEATURE_NAMES)


def _load_bt_model() -> bool:
    """Load bet type model from disk."""
    global _bt_model, _bt_failed

    if _bt_model is not None:
        return True
    if _bt_failed:
        return False

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — bet type model unavailable")
        _bt_failed = True
        return False

    if not BETTYPE_MODEL_PATH.exists():
        logger.info("Bet type model not found at %s — using rules", BETTYPE_MODEL_PATH)
        _bt_failed = True
        return False

    try:
        _bt_model = lgb.Booster(model_file=str(BETTYPE_MODEL_PATH))
        logger.info(
            "Bet type meta-model loaded: %d trees from %s",
            _bt_model.num_trees(), BETTYPE_MODEL_PATH,
        )
        return True
    except Exception as e:
        logger.error("Failed to load bet type model: %s", e)
        _bt_failed = True
        return False


def bettype_model_available() -> bool:
    """Check if the bet type model is loaded or loadable."""
    if _bt_model is not None:
        return True
    if _bt_failed:
        return False
    return _load_bt_model()


def _safe_float(val: Any) -> float:
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _prize_money_bucket(pm: float) -> float:
    if pm >= 100_000:
        return 4.0
    elif pm >= 50_000:
        return 3.0
    elif pm >= 25_000:
        return 2.0
    return 1.0


def extract_bettype_features(
    *,
    tip_rank: int,
    is_roughie: bool,
    win_prob: float,
    place_prob: float,
    value_rating: float,
    odds: float,
    field_size: int,
    distance_bucket: float,
    class_bucket: float,
    track_cond_bucket: float,
    venue_type: float,
    prize_money: float,
    rank1_wp: float,
    wp_spread: float,
    gap_to_next: float,
    fav_odds: float,
    place_odds: float = 0.0,
) -> list[float]:
    """Build the bet type feature vector."""
    return [
        float(tip_rank),
        1.0 if is_roughie else 0.0,
        _safe_float(win_prob),
        _safe_float(place_prob),
        _safe_float(value_rating),
        _safe_float(odds),
        float(field_size),
        _safe_float(distance_bucket),
        _safe_float(class_bucket),
        _safe_float(track_cond_bucket),
        _safe_float(venue_type),
        _prize_money_bucket(prize_money),
        _safe_float(rank1_wp),
        _safe_float(wp_spread),
        _safe_float(gap_to_next),
        _safe_float(fav_odds),
        _safe_float(place_odds),
    ]


def predict_best_bet_type(features: list[float]) -> tuple[str, list[float]] | None:
    """Predict the most profitable bet type for this pick.

    Returns:
        (best_type, probabilities) where probabilities is [P(Win), P(EW), P(Place)].
        Returns None if model unavailable.
    """
    if not _load_bt_model():
        return None

    try:
        X = np.array([features], dtype=np.float64)
        raw_preds = _bt_model.predict(X)[0]

        # Multi-class: raw_preds is array of probabilities per class
        if hasattr(raw_preds, '__len__') and len(raw_preds) == 3:
            probs = list(raw_preds)
        else:
            # Binary fallback — shouldn't happen with multiclass
            probs = [float(raw_preds), 0.0, 1.0 - float(raw_preds)]

        best_idx = int(np.argmax(probs))
        return BET_TYPE_CLASSES[best_idx], probs
    except Exception as e:
        logger.error("Bet type prediction failed: %s", e)
        return None


def recommend_bet_type(
    *,
    tip_rank: int,
    is_roughie: bool,
    win_prob: float,
    place_prob: float,
    value_rating: float,
    odds: float,
    field_size: int,
    distance_bucket: float,
    class_bucket: float,
    track_cond_bucket: float,
    venue_type: float,
    prize_money: float,
    rank1_wp: float,
    wp_spread: float,
    gap_to_next: float,
    fav_odds: float,
    place_odds: float = 0.0,
) -> tuple[str, str] | None:
    """Recommend bet type with reason.

    Returns:
        (bet_type, reason) or None if model unavailable.
    """
    # ≤4 runners: no place market, always Win
    if field_size <= 4:
        return "Win", "no place market (≤4 runners)"

    features = extract_bettype_features(
        tip_rank=tip_rank,
        is_roughie=is_roughie,
        win_prob=win_prob,
        place_prob=place_prob,
        value_rating=value_rating,
        odds=odds,
        field_size=field_size,
        distance_bucket=distance_bucket,
        class_bucket=class_bucket,
        track_cond_bucket=track_cond_bucket,
        venue_type=venue_type,
        prize_money=prize_money,
        rank1_wp=rank1_wp,
        wp_spread=wp_spread,
        gap_to_next=gap_to_next,
        fav_odds=fav_odds,
        place_odds=place_odds,
    )

    result = predict_best_bet_type(features)
    if result is None:
        return None

    bet_type, probs = result
    win_p, ew_p, place_p = probs

    reason = f"model: Win={win_p:.0%} EW={ew_p:.0%} Place={place_p:.0%}"
    return bet_type, reason


def clear_cache():
    """Clear cached model."""
    global _bt_model, _bt_failed
    _bt_model = None
    _bt_failed = False
