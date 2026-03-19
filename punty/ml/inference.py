"""LightGBM model inference for race probability prediction.

Supports two model architectures:
  - LambdaRank (v7+): single ranking model → softmax → win probs → Harville place probs
  - Binary (v5/v6): separate win + place models (legacy fallback)

Loads trained models from punty/data/, caches at module level.
Graceful degradation: returns empty dict if models not found or prediction fails.
"""

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
RANK_MODEL_PATH = MODEL_DIR / "lgbm_rank_model.txt"
# Legacy paths (v5/v6 binary classifiers)
WIN_MODEL_PATH = MODEL_DIR / "lgbm_win_model.txt"
PLACE_MODEL_PATH = MODEL_DIR / "lgbm_place_model.txt"

# Module-level model cache — loaded once on first call
_models: dict = {}


def _load_models() -> bool:
    """Load models from disk. Prefers rank model, falls back to win+place."""
    if "rank" in _models or ("win" in _models and "place" in _models):
        return True

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — LightGBM predictions unavailable")
        _models["_failed"] = True
        return False

    if _models.get("_failed"):
        return False

    # Try LambdaRank model first (v7+)
    if RANK_MODEL_PATH.exists():
        try:
            _models["rank"] = lgb.Booster(model_file=str(RANK_MODEL_PATH))
            _models["_type"] = "rank"
            logger.info(
                "LightGBM rank model loaded: %d trees",
                _models["rank"].num_trees(),
            )
            return True
        except Exception as e:
            logger.error("Failed to load rank model: %s", e)

    # Fall back to legacy win+place models (v5/v6)
    if WIN_MODEL_PATH.exists() and PLACE_MODEL_PATH.exists():
        try:
            _models["win"] = lgb.Booster(model_file=str(WIN_MODEL_PATH))
            _models["place"] = lgb.Booster(model_file=str(PLACE_MODEL_PATH))
            _models["_type"] = "binary"
            logger.info(
                "LightGBM binary models loaded: win=%d trees, place=%d trees",
                _models["win"].num_trees(),
                _models["place"].num_trees(),
            )
            return True
        except Exception as e:
            logger.error("Failed to load binary models: %s", e)

    logger.warning(
        "No LightGBM model files found at %s — run training first", MODEL_DIR,
    )
    _models["_failed"] = True
    return False


def models_available() -> bool:
    """Check if LightGBM models are loaded or loadable."""
    if "rank" in _models or ("win" in _models and "place" in _models):
        return True
    if _models.get("_failed"):
        return False
    return _load_models()


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Convert ranking scores to probabilities via softmax.

    Numerically stable: subtracts max before exponentiation to prevent overflow.
    Returns array summing to 1.0.
    """
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def predict_race(
    runners: list, race: Any, meeting: Any,
) -> dict[str, tuple[float, float]]:
    """Predict win and place probabilities for all active runners.

    For LambdaRank: ranking scores → softmax → win probs, place probs via Harville.
    For legacy binary: raw model predictions with place >= win enforcement.

    Args:
        runners: List of Runner ORM objects
        race: Race ORM object
        meeting: Meeting ORM object

    Returns:
        Dict mapping runner_id → (win_prob, place_prob).
        Empty dict if models unavailable or prediction fails.
    """
    if not _load_models():
        return {}

    try:
        from punty.ml.features import extract_features_batch, _get

        active = [r for r in runners if not _get(r, "scratched", False)]
        if not active:
            return {}

        X = extract_features_batch(active, race, meeting)
        if X.shape[0] == 0:
            return {}

        model_type = _models.get("_type", "binary")

        if model_type == "rank":
            return _predict_rank(X, active)
        else:
            return _predict_binary(X, active)

    except Exception as e:
        logger.error("LightGBM prediction failed: %s", e, exc_info=True)
        return {}


def _predict_rank(X: np.ndarray, active: list) -> dict[str, tuple[float, float]]:
    """LambdaRank prediction: single model → softmax → Harville."""
    from punty.ml.features import _get

    model = _models["rank"]
    model_n_features = model.num_feature()
    if X.shape[1] > model_n_features:
        X = X[:, :model_n_features]

    raw_scores = model.predict(X)
    win_probs = _softmax(raw_scores)

    # Build win prob dict for Harville
    field_size = len(active)
    place_count = 2 if field_size <= 7 else 3
    rids = [_get(r, "id", "") for r in active]
    win_dict = {rids[i]: float(win_probs[i]) for i in range(len(active))}

    # Place probs via Harville (mathematically guarantees place >= win)
    from punty.probability import _harville_place_probability
    place_dict = {rid: _harville_place_probability(rid, win_dict, place_count) for rid in rids}

    results = {}
    for i, runner in enumerate(active):
        rid = rids[i]
        results[rid] = (win_dict[rid], place_dict[rid])

    return results


def _predict_binary(X: np.ndarray, active: list) -> dict[str, tuple[float, float]]:
    """Legacy binary prediction: separate win + place models."""
    from punty.ml.features import _get

    model_n_features = _models["win"].num_feature()
    if X.shape[1] > model_n_features:
        X = X[:, :model_n_features]

    win_probs = _models["win"].predict(X)
    place_probs = _models["place"].predict(X)

    results = {}
    for i, runner in enumerate(active):
        rid = _get(runner, "id", "")
        wp = float(win_probs[i])
        pp = float(place_probs[i])
        # Enforce: place_prob >= win_prob (placing includes winning).
        if pp < wp:
            pp = wp
        results[rid] = (wp, pp)

    return results


def clear_cache():
    """Clear cached models (for testing)."""
    _models.clear()
