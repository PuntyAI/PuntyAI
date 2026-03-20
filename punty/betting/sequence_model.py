"""Sequence/multi meta-model — learned play/skip gate for multi-race bets.

A lightweight LightGBM binary classifier trained on "does this sequence type
hit for our LGBM-ranked picks in this meeting context?" with sample weights
proportional to estimated dividends.

Covers: Early Quaddie, Quaddie, Big 6, Big3 Multi.

At inference: predicts P(hit) for the proposed sequence. If below threshold,
advises skipping. Replaces hard-coded track condition gates, confidence
floors, and return % thresholds.

Graceful fallback: if model file not found, all sequences are played
(no gating).
"""

import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
SEQ_MODEL_PATH = MODEL_DIR / "sequence_meta_model.txt"
SEQ_METADATA_PATH = MODEL_DIR / "sequence_meta_metadata.json"

# Module-level cache
_seq_model: Optional[Any] = None
_seq_failed: bool = False

# Sequence type encoding
SEQ_TYPE_CODES = {
    "Early Quaddie": 1,
    "Quaddie": 2,
    "Big 6": 3,
    "Big3 Multi": 4,
}

# Feature names — ORDER MUST MATCH extraction
SEQ_FEATURE_NAMES = [
    # Sequence type (2 features)
    "seq_type_code",      # 1-4 encoding
    "num_legs",           # 4 or 6 (quaddie vs big6) or 3 (big3 multi)

    # Per-leg pick quality — padded to 6 legs (8 features)
    "leg1_top_wp",        # rank 1 WP in leg 1
    "leg2_top_wp",
    "leg3_top_wp",
    "leg4_top_wp",
    "leg5_top_wp",        # 0 for quaddies (only 4 legs)
    "leg6_top_wp",        # 0 for quaddies
    "avg_leg_wp",         # average top-pick WP across all legs
    "min_leg_wp",         # weakest leg (chaos indicator)

    # Field context (4 features)
    "avg_field_size",     # average runners across legs
    "max_field_size",     # largest field (hardest leg)
    "min_field_size",     # smallest field
    "field_variance",     # spread of field sizes

    # Track / meeting context (2 features)
    "track_cond_bucket",  # 1-5 (firm to synthetic)
    "venue_type",         # 1=metro, 2=provincial, 3=country

    # Combo context (3 features)
    "total_combos",       # how many combinations in the ticket
    "estimated_return_pct",  # estimated return %
    "hit_probability",    # model's estimated P(all legs hit)
]

NUM_SEQ_FEATURES = len(SEQ_FEATURE_NAMES)

# Default threshold — play if P(hit) >= this
DEFAULT_SEQ_THRESHOLD = 0.15


def _load_seq_model() -> bool:
    """Load sequence meta-model from disk."""
    global _seq_model, _seq_failed

    if _seq_model is not None:
        return True
    if _seq_failed:
        return False

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — sequence meta-model unavailable")
        _seq_failed = True
        return False

    if not SEQ_MODEL_PATH.exists():
        logger.info("Sequence meta-model not found at %s — playing all sequences",
                     SEQ_MODEL_PATH)
        _seq_failed = True
        return False

    try:
        _seq_model = lgb.Booster(model_file=str(SEQ_MODEL_PATH))
        logger.info(
            "Sequence meta-model loaded: %d trees from %s",
            _seq_model.num_trees(), SEQ_MODEL_PATH,
        )
        return True
    except Exception as e:
        logger.error("Failed to load sequence meta-model: %s", e)
        _seq_failed = True
        return False


def sequence_model_available() -> bool:
    """Check if the sequence meta-model is loaded or loadable."""
    if _seq_model is not None:
        return True
    if _seq_failed:
        return False
    return _load_seq_model()


def _safe_float(val: Any) -> float:
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _track_cond_bucket(condition: str) -> float:
    """Ordinal encode track condition."""
    if not condition:
        return 0.0
    c = condition.lower().strip()
    if "firm" in c or (c.startswith("f") and len(c) <= 2):
        return 1.0
    if "good" in c or c.startswith("g"):
        return 2.0
    if "soft" in c or "dead" in c or (c.startswith("s") and (len(c) <= 2 or c[1:2].isdigit())):
        return 3.0
    if "heavy" in c or (c.startswith("h") and len(c) <= 3):
        return 4.0
    if "synth" in c or "syn" in c:
        return 5.0
    return 0.0


def extract_sequence_features(
    *,
    sequence_type: str,
    leg_wps: list[float],
    leg_field_sizes: list[float],
    track_condition: str = "",
    venue_type: float = 0.0,
    total_combos: int = 1,
    estimated_return_pct: float = 0.0,
    hit_probability: float = 0.0,
) -> list[float]:
    """Build the sequence feature vector."""
    num_legs = len(leg_wps)
    type_code = float(SEQ_TYPE_CODES.get(sequence_type, 0))

    # Pad leg WPs to 6
    padded_wps = list(leg_wps) + [0.0] * (6 - len(leg_wps))
    avg_wp = sum(leg_wps) / len(leg_wps) if leg_wps else 0.0
    min_wp = min(leg_wps) if leg_wps else 0.0

    # Field sizes
    fields = list(leg_field_sizes) if leg_field_sizes else [10.0] * num_legs
    avg_field = sum(fields) / len(fields) if fields else 10.0
    max_field = max(fields) if fields else 10.0
    min_field = min(fields) if fields else 10.0
    field_var = max_field - min_field

    tc_bucket = _track_cond_bucket(track_condition)

    return [
        type_code,
        float(num_legs),
        _safe_float(padded_wps[0]),
        _safe_float(padded_wps[1]),
        _safe_float(padded_wps[2]),
        _safe_float(padded_wps[3]),
        _safe_float(padded_wps[4]),
        _safe_float(padded_wps[5]),
        _safe_float(avg_wp),
        _safe_float(min_wp),
        _safe_float(avg_field),
        _safe_float(max_field),
        _safe_float(min_field),
        _safe_float(field_var),
        tc_bucket,
        _safe_float(venue_type),
        float(total_combos),
        _safe_float(estimated_return_pct),
        _safe_float(hit_probability),
    ]


def should_play_sequence(
    *,
    sequence_type: str,
    leg_wps: list[float],
    leg_field_sizes: list[float],
    track_condition: str = "",
    venue_type: float = 0.0,
    total_combos: int = 1,
    estimated_return_pct: float = 0.0,
    hit_probability: float = 0.0,
    threshold: float = DEFAULT_SEQ_THRESHOLD,
) -> tuple[bool, float, str]:
    """Decide whether to play this sequence.

    Returns:
        (should_play, probability, reason)
    """
    if not _load_seq_model():
        # No model — play everything (no gating)
        return True, 0.0, "no model (playing all)"

    features = extract_sequence_features(
        sequence_type=sequence_type,
        leg_wps=leg_wps,
        leg_field_sizes=leg_field_sizes,
        track_condition=track_condition,
        venue_type=venue_type,
        total_combos=total_combos,
        estimated_return_pct=estimated_return_pct,
        hit_probability=hit_probability,
    )

    try:
        X = np.array([features], dtype=np.float64)
        prob = float(_seq_model.predict(X)[0])
    except Exception as e:
        logger.error("Sequence meta-model prediction failed: %s", e)
        return True, 0.0, f"prediction error ({e})"

    if prob >= threshold:
        return True, prob, f"meta-model ({prob:.0%} >= {threshold:.0%})"
    else:
        return False, prob, f"meta-model ({prob:.0%} < {threshold:.0%})"


def clear_cache():
    """Clear cached model (for testing or reload)."""
    global _seq_model, _seq_failed
    _seq_model = None
    _seq_failed = False
