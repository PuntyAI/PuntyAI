"""Exotic meta-model — learned exotic type selector for race context.

A lightweight LightGBM binary classifier trained on "does this exotic type hit
for our LGBM-ranked picks in this race context?" with sample weights proportional
to estimated dividends (optimise ROI, not just strike rate).

At inference: scores each candidate exotic by predicted hit probability, then
multiplies by value_ratio to pick the best expected-value exotic for the race.

Context-aware features capture race shape, pick quality, field dynamics, and
exotic structure — learning which combinations work in which conditions.

Graceful fallback: if model file not found, returns None and caller uses
hand-tuned routing.
"""

import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
EXOTIC_MODEL_PATH = MODEL_DIR / "exotic_meta_model.txt"
EXOTIC_METADATA_PATH = MODEL_DIR / "exotic_meta_metadata.json"

# Module-level cache — loaded once on first call
_exotic_model: Optional[Any] = None
_exotic_failed: bool = False

# Exotic type encoding for the model
EXOTIC_TYPE_CODES = {
    "Quinella": 1,
    "Quinella Box": 2,
    "Exacta": 3,
    "Exacta Standout": 4,
    "Trifecta Standout": 5,
    "Trifecta Box": 6,
    "First4": 7,
    "First4 Box": 8,
}

# Feature names — ORDER MUST MATCH extraction
EXOTIC_FEATURE_NAMES = [
    # Race context (6 features)
    "field_size",         # number of runners
    "distance_bucket",    # 1-5 (sprint to staying)
    "class_bucket",       # 1-6 (maiden to group)
    "track_cond_bucket",  # 1-5 (firm to synthetic)
    "venue_type",         # 1=metro, 2=provincial, 3=country
    "prize_money_bucket", # 1-4 (low to feature)

    # Race shape (3 features)
    "wp_spread",          # top1 WP - top3 WP (dominant vs open)
    "wp_hhi",             # Herfindahl index of top 4 WPs (concentration)
    "top1_wp",            # rank 1 win probability

    # Pick quality (7 features)
    "rank1_wp",           # WP of our rank 1 pick
    "rank2_wp",           # WP of our rank 2 pick
    "rank3_wp",           # WP of our rank 3 pick
    "rank4_wp",           # WP of our rank 4 pick (roughie)
    "gap_12",             # WP gap between rank 1 and rank 2
    "gap_23",             # WP gap between rank 2 and rank 3
    "gap_34",             # WP gap between rank 3 and rank 4

    # Market context (3 features)
    "fav_odds",           # favourite SP odds
    "odds_spread",        # SP range (longest - shortest in top 4)
    "avg_top4_value",     # average value rating of top 4 (WP / market implied)

    # Exotic type (3 features)
    "exotic_type_code",   # 1-8 encoding per EXOTIC_TYPE_CODES
    "num_combo_runners",  # number of runners in the combo
    "num_combos",         # number of permutations/combinations

    # Combo composition (5 features) — WHO is in this specific combo
    "anchor_rank",        # rank of the 1st runner in combo (1=top pick, 4=roughie)
    "anchor_wp",          # WP of the anchor runner
    "best_combo_rank",    # best (lowest) rank among combo runners
    "worst_combo_rank",   # worst (highest) rank — roughie=4
    "avg_combo_wp",       # average WP of runners in this combo
]

NUM_EXOTIC_FEATURES = len(EXOTIC_FEATURE_NAMES)


def _load_exotic_model() -> bool:
    """Load exotic meta-model from disk. Returns True if loaded."""
    global _exotic_model, _exotic_failed

    if _exotic_model is not None:
        return True
    if _exotic_failed:
        return False

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — exotic meta-model unavailable")
        _exotic_failed = True
        return False

    if not EXOTIC_MODEL_PATH.exists():
        logger.info("Exotic meta-model not found at %s — using hand-tuned routing",
                     EXOTIC_MODEL_PATH)
        _exotic_failed = True
        return False

    try:
        _exotic_model = lgb.Booster(model_file=str(EXOTIC_MODEL_PATH))
        logger.info(
            "Exotic meta-model loaded: %d trees from %s",
            _exotic_model.num_trees(), EXOTIC_MODEL_PATH,
        )
        return True
    except Exception as e:
        logger.error("Failed to load exotic meta-model: %s", e)
        _exotic_failed = True
        return False


def exotic_model_available() -> bool:
    """Check if the exotic meta-model is loaded or loadable."""
    if _exotic_model is not None:
        return True
    if _exotic_failed:
        return False
    return _load_exotic_model()


def _safe_float(val: Any) -> float:
    """Convert to float, NaN if None or invalid."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _prize_money_bucket(prize_money: float) -> float:
    """Bucket prize money into 4 tiers."""
    if prize_money >= 100_000:
        return 4.0  # feature/group race
    elif prize_money >= 50_000:
        return 3.0  # metro quality
    elif prize_money >= 25_000:
        return 2.0  # provincial quality
    return 1.0  # country/low


def extract_exotic_features(
    *,
    field_size: int,
    distance_bucket: float,
    class_bucket: float,
    track_cond_bucket: float,
    venue_type: float,
    prize_money: float,
    rank_wps: list[float],
    rank_odds: list[float],
    exotic_type: str,
    num_combo_runners: int,
    num_combos: int,
    combo_runner_ranks: list[int] | None = None,
    combo_runner_wps: list[float] | None = None,
) -> list[float]:
    """Build the exotic feature vector. Order must match EXOTIC_FEATURE_NAMES.

    Args:
        field_size: Number of runners in the race.
        distance_bucket: 1-5 distance category.
        class_bucket: 1-6 race class category.
        track_cond_bucket: 1-5 track condition category.
        venue_type: 1=metro, 2=provincial, 3=country.
        prize_money: Race prize money in dollars.
        rank_wps: Win probabilities for our top 4 picks [rank1, rank2, rank3, rank4].
        rank_odds: SP odds for our top 4 picks.
        exotic_type: Exotic type name (e.g. "Quinella Box").
        num_combo_runners: Number of runners in the exotic combination.
        num_combos: Number of permutations/combinations.
        combo_runner_ranks: Ranks (1-4) of the runners in this specific combo.
        combo_runner_wps: WPs of the runners in this specific combo.
    """
    nan = float("nan")

    # Pad to 4 if we have fewer picks
    wps = list(rank_wps) + [0.0] * (4 - len(rank_wps))
    odds = list(rank_odds) + [nan] * (4 - len(rank_odds))

    # Race shape
    wp_spread = wps[0] - wps[2] if len(wps) >= 3 else wps[0] - wps[-1]
    wp_sum = sum(w ** 2 for w in wps[:4])
    wp_total = sum(wps[:4])
    wp_hhi = wp_sum / (wp_total ** 2) if wp_total > 0 else 0.25

    # Pick gaps
    gap_12 = wps[0] - wps[1]
    gap_23 = wps[1] - wps[2]
    gap_34 = wps[2] - wps[3]

    # Market context
    valid_odds = [o for o in odds[:4] if o and not math.isnan(o) and o > 1.0]
    fav_odds = min(valid_odds) if valid_odds else nan
    odds_spread = max(valid_odds) - min(valid_odds) if len(valid_odds) >= 2 else nan

    # Average value rating
    values = []
    for w, o in zip(wps[:4], odds[:4]):
        if o and not math.isnan(o) and o > 1.0:
            mkt_implied = 1.0 / o
            values.append(w / mkt_implied if mkt_implied > 0 else nan)
    avg_value = sum(v for v in values if not math.isnan(v)) / len(values) if values else nan

    # Exotic type code
    type_code = float(EXOTIC_TYPE_CODES.get(exotic_type, 0))

    # Combo composition — who's actually in this specific combo
    c_ranks = combo_runner_ranks or [1, 2]  # default to top 2
    c_wps = combo_runner_wps or wps[:2]
    anchor_rank = float(c_ranks[0]) if c_ranks else nan
    anchor_wp = _safe_float(c_wps[0]) if c_wps else nan
    best_combo_rank = float(min(c_ranks)) if c_ranks else nan
    worst_combo_rank = float(max(c_ranks)) if c_ranks else nan
    avg_combo_wp = sum(c_wps) / len(c_wps) if c_wps else nan

    return [
        float(field_size),
        _safe_float(distance_bucket),
        _safe_float(class_bucket),
        _safe_float(track_cond_bucket),
        _safe_float(venue_type),
        _prize_money_bucket(prize_money),
        _safe_float(wp_spread),
        _safe_float(wp_hhi),
        _safe_float(wps[0]),
        _safe_float(wps[0]),
        _safe_float(wps[1]),
        _safe_float(wps[2]),
        _safe_float(wps[3]),
        _safe_float(gap_12),
        _safe_float(gap_23),
        _safe_float(gap_34),
        _safe_float(fav_odds),
        _safe_float(odds_spread),
        _safe_float(avg_value),
        type_code,
        float(num_combo_runners),
        float(num_combos),
        _safe_float(anchor_rank),
        _safe_float(anchor_wp),
        _safe_float(best_combo_rank),
        _safe_float(worst_combo_rank),
        _safe_float(avg_combo_wp),
    ]


def predict_exotic_hit_probability(features: list[float]) -> float:
    """Predict probability that an exotic type hits for this race context.

    Args:
        features: Exotic feature vector (26 floats, order per EXOTIC_FEATURE_NAMES).

    Returns:
        Float 0-1 probability. Returns -1.0 if model unavailable.
    """
    if not _load_exotic_model():
        return -1.0

    try:
        X = np.array([features], dtype=np.float64)
        prob = _exotic_model.predict(X)[0]
        return float(prob)
    except Exception as e:
        logger.error("Exotic meta-model prediction failed: %s", e)
        return -1.0


def score_exotic_candidates(
    candidates: list[dict],
    *,
    field_size: int,
    distance_bucket: float,
    class_bucket: float,
    track_cond_bucket: float,
    venue_type: float,
    prize_money: float,
    rank_wps: list[float],
    rank_odds: list[float],
    saddlecloth_to_rank: dict[int, int] | None = None,
    saddlecloth_to_wp: dict[int, float] | None = None,
) -> list[tuple[dict, float, float]]:
    """Score all candidate exotics using the meta-model.

    Args:
        candidates: List of exotic combo dicts from calculate_exotic_combinations.
        saddlecloth_to_rank: Map saddlecloth → rank (1-4) for combo composition.
        saddlecloth_to_wp: Map saddlecloth → WP for combo composition.
        (remaining kwargs): Race context features.

    Returns:
        List of (combo_dict, predicted_hit_prob, score) sorted by score descending.
        score = predicted_hit_prob × value_ratio.
        Returns empty list if model unavailable.
    """
    if not _load_exotic_model():
        return []

    sc_rank = saddlecloth_to_rank or {}
    sc_wp = saddlecloth_to_wp or {}

    results = []
    for ec in candidates:
        exotic_type = ec.get("type", "")
        if exotic_type not in EXOTIC_TYPE_CODES:
            continue

        # Map combo runners to their ranks and WPs
        runners = ec.get("runners", [])
        combo_ranks = [sc_rank.get(r, 4) for r in runners]  # default rank 4 if unknown
        combo_wps = [sc_wp.get(r, 0.05) for r in runners]

        features = extract_exotic_features(
            field_size=field_size,
            distance_bucket=distance_bucket,
            class_bucket=class_bucket,
            track_cond_bucket=track_cond_bucket,
            venue_type=venue_type,
            prize_money=prize_money,
            rank_wps=rank_wps,
            rank_odds=rank_odds,
            exotic_type=exotic_type,
            num_combo_runners=len(runners),
            num_combos=max(1, ec.get("combos", 1)),
            combo_runner_ranks=combo_ranks,
            combo_runner_wps=combo_wps,
        )

        prob = predict_exotic_hit_probability(features)
        if prob < 0:
            return []  # Model failed, bail out

        value = ec.get("value", 1.0)
        score = prob * value
        results.append((ec, prob, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def clear_cache():
    """Clear cached model (for testing or reload)."""
    global _exotic_model, _exotic_failed
    _exotic_model = None
    _exotic_failed = False
