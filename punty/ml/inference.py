"""LightGBM model inference for race probability prediction.

Loads trained win and place models from punty/data/, caches them at
module level, and provides predict_race() for integration with probability.py.

Graceful degradation: returns empty dict if models not found or prediction fails.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
WIN_MODEL_PATH = MODEL_DIR / "lgbm_win_model.txt"
PLACE_MODEL_PATH = MODEL_DIR / "lgbm_place_model.txt"

# Module-level model cache — loaded once on first call
_models: dict = {}


def _load_models() -> bool:
    """Load win and place models from disk. Returns True if both loaded."""
    if "win" in _models and "place" in _models:
        return True

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — LightGBM predictions unavailable")
        _models["_failed"] = True
        return False

    if _models.get("_failed"):
        return False

    if not WIN_MODEL_PATH.exists() or not PLACE_MODEL_PATH.exists():
        logger.warning(
            "LightGBM model files not found at %s — run scripts/train_lightgbm.py first",
            MODEL_DIR,
        )
        _models["_failed"] = True
        return False

    try:
        _models["win"] = lgb.Booster(model_file=str(WIN_MODEL_PATH))
        _models["place"] = lgb.Booster(model_file=str(PLACE_MODEL_PATH))
        logger.info(
            "LightGBM models loaded: win=%d trees, place=%d trees",
            _models["win"].num_trees(),
            _models["place"].num_trees(),
        )
        return True
    except Exception as e:
        logger.error("Failed to load LightGBM models: %s", e)
        _models["_failed"] = True
        return False


def models_available() -> bool:
    """Check if LightGBM models are loaded or loadable."""
    if "win" in _models and "place" in _models:
        return True
    if _models.get("_failed"):
        return False
    return _load_models()


def predict_race(
    runners: list, race: Any, meeting: Any,
) -> dict[str, tuple[float, float]]:
    """Predict win and place probabilities for all active runners.

    Args:
        runners: List of Runner ORM objects
        race: Race ORM object
        meeting: Meeting ORM object

    Returns:
        Dict mapping runner_id → (win_prob, place_prob).
        Raw probabilities (not normalized to sum to 1).
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

        win_probs = _models["win"].predict(X)
        place_probs = _models["place"].predict(X)

        results = {}
        for i, runner in enumerate(active):
            rid = _get(runner, "id", "")
            results[rid] = (float(win_probs[i]), float(place_probs[i]))

        return results

    except Exception as e:
        logger.error("LightGBM prediction failed: %s", e, exc_info=True)
        return {}


def clear_cache():
    """Clear cached models (for testing)."""
    _models.clear()
