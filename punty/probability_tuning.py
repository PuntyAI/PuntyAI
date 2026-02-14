"""Probability model self-tuning and calibration analysis.

Analyzes settled picks to evaluate factor performance, calibrate
probability predictions, and automatically adjust factor weights
to improve accuracy over time.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, case, and_
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.models.pick import Pick
from punty.models.meeting import Meeting, Race
from punty.models.settings import AppSettings
from punty.memory.models import TuningLog
from punty.probability import DEFAULT_WEIGHTS, FACTOR_REGISTRY

logger = logging.getLogger(__name__)

# Calibration buckets (probability ranges)
CALIBRATION_BUCKETS = [
    (0.00, 0.05, "0-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, 0.20, "15-20%"),
    (0.20, 0.30, "20-30%"),
    (0.30, 0.50, "30-50%"),
    (0.50, 1.01, "50%+"),
]

# Value rating buckets
VALUE_BUCKETS = [
    (0.00, 0.80, "<0.80 (Unders)"),
    (0.80, 1.00, "0.80-1.00"),
    (1.00, 1.10, "1.00-1.10"),
    (1.10, 1.30, "1.10-1.30 (Value)"),
    (1.30, 99.0, "1.30+ (Strong Value)"),
]

# Tuning constraints
MIN_WEIGHT = 0.00   # Allow zeroed dead factors
MAX_WEIGHT = 0.45   # Allow calibrated market weight (40%)
SMOOTHING = 0.70    # 70% old weight, 30% optimal
MIN_CHANGE_THRESHOLD = 0.005  # Only save if any factor changes by >0.5%
COOLDOWN_HOURS = 24  # Minimum hours between tuning runs


async def calculate_calibration(
    db: AsyncSession, lookback_days: int = 60,
) -> list[dict]:
    """Predicted probability vs actual win rate by bucket.

    Uses all settled selection picks with win_probability.
    Returns list of {bucket, predicted_avg, actual_rate, count, market_rate}.
    """
    cutoff = melb_now_naive() - timedelta(days=lookback_days)

    result = await db.execute(
        select(Pick.win_probability, Pick.hit, Pick.value_rating)
        .where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.win_probability.isnot(None),
            Pick.created_at >= cutoff,
        )
    )
    rows = result.all()

    if not rows:
        return []

    buckets = []
    for low, high, label in CALIBRATION_BUCKETS:
        in_bucket = [r for r in rows if low <= (r.win_probability or 0) < high]
        if not in_bucket:
            buckets.append({
                "bucket": label, "predicted_avg": 0, "actual_rate": 0,
                "count": 0, "market_rate": 0,
            })
            continue

        predicted_avg = sum(r.win_probability for r in in_bucket) / len(in_bucket)
        actual_wins = sum(1 for r in in_bucket if r.hit)
        actual_rate = actual_wins / len(in_bucket)

        # Market implied rate = predicted / value_rating (approximation)
        market_rates = []
        for r in in_bucket:
            vr = r.value_rating
            if vr and vr > 0:
                market_rates.append(r.win_probability / vr)
        market_rate = sum(market_rates) / len(market_rates) if market_rates else predicted_avg

        buckets.append({
            "bucket": label,
            "predicted_avg": round(predicted_avg, 4),
            "actual_rate": round(actual_rate, 4),
            "count": len(in_bucket),
            "market_rate": round(market_rate, 4),
        })

    return buckets


async def calculate_value_performance(
    db: AsyncSession, lookback_days: int = 60,
) -> list[dict]:
    """ROI by value_rating bucket."""
    cutoff = melb_now_naive() - timedelta(days=lookback_days)

    result = await db.execute(
        select(Pick.value_rating, Pick.hit, Pick.pnl, Pick.bet_stake)
        .where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.value_rating.isnot(None),
            Pick.created_at >= cutoff,
        )
    )
    rows = result.all()

    if not rows:
        return []

    buckets = []
    for low, high, label in VALUE_BUCKETS:
        in_bucket = [r for r in rows if low <= (r.value_rating or 0) < high]
        if not in_bucket:
            buckets.append({
                "bucket": label, "count": 0, "hit_rate": 0,
                "avg_pnl": 0, "total_pnl": 0, "roi": 0,
            })
            continue

        count = len(in_bucket)
        wins = sum(1 for r in in_bucket if r.hit)
        total_pnl = sum(r.pnl or 0 for r in in_bucket)
        total_staked = sum(r.bet_stake or 0 for r in in_bucket)

        buckets.append({
            "bucket": label,
            "count": count,
            "hit_rate": round(wins / count, 4) if count else 0,
            "avg_pnl": round(total_pnl / count, 2) if count else 0,
            "total_pnl": round(total_pnl, 2),
            "roi": round(total_pnl / total_staked * 100, 1) if total_staked else 0,
        })

    return buckets


async def analyze_factor_performance(
    db: AsyncSession, lookback_days: int = 60,
) -> dict:
    """Analyze which factors best predict winners vs losers.

    Returns dict of factor_key -> {winner_avg, loser_avg, edge, accuracy, count}.
    Requires picks with factors_json populated.
    """
    cutoff = melb_now_naive() - timedelta(days=lookback_days)

    result = await db.execute(
        select(Pick.factors_json, Pick.hit)
        .where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.factors_json.isnot(None),
            Pick.created_at >= cutoff,
        )
    )
    rows = result.all()

    if not rows:
        return {}

    # Parse factor scores
    picks_data = []
    for factors_json, hit in rows:
        try:
            factors = json.loads(factors_json)
            picks_data.append({"factors": factors, "hit": bool(hit)})
        except (json.JSONDecodeError, TypeError):
            continue

    if len(picks_data) < 10:
        return {}

    # Analyze each factor
    factor_keys = list(FACTOR_REGISTRY.keys())
    performance = {}

    for key in factor_keys:
        winner_scores = [p["factors"].get(key, 0.5) for p in picks_data if p["hit"]]
        loser_scores = [p["factors"].get(key, 0.5) for p in picks_data if not p["hit"]]

        if not winner_scores or not loser_scores:
            continue

        winner_avg = sum(winner_scores) / len(winner_scores)
        loser_avg = sum(loser_scores) / len(loser_scores)
        edge = winner_avg - loser_avg

        # Point-biserial correlation approximation
        all_scores = [p["factors"].get(key, 0.5) for p in picks_data]
        all_hits = [1.0 if p["hit"] else 0.0 for p in picks_data]
        accuracy = _point_biserial(all_scores, all_hits)

        performance[key] = {
            "label": FACTOR_REGISTRY[key]["label"],
            "category": FACTOR_REGISTRY[key]["category"],
            "winner_avg": round(winner_avg, 4),
            "loser_avg": round(loser_avg, 4),
            "edge": round(edge, 4),
            "accuracy": round(accuracy, 4),
            "count": len(picks_data),
        }

    return performance


def _point_biserial(scores: list[float], outcomes: list[float]) -> float:
    """Calculate point-biserial correlation between continuous scores and binary outcomes."""
    n = len(scores)
    if n < 2:
        return 0.0

    mean_all = sum(scores) / n
    group_1 = [s for s, o in zip(scores, outcomes) if o > 0.5]
    group_0 = [s for s, o in zip(scores, outcomes) if o <= 0.5]

    if not group_1 or not group_0:
        return 0.0

    mean_1 = sum(group_1) / len(group_1)
    mean_0 = sum(group_0) / len(group_0)
    n1 = len(group_1)
    n0 = len(group_0)

    # Standard deviation of all scores
    var = sum((s - mean_all) ** 2 for s in scores) / n
    if var <= 0:
        return 0.0
    sd = math.sqrt(var)

    # Point-biserial correlation
    rpb = (mean_1 - mean_0) / sd * math.sqrt(n1 * n0 / (n * n))
    return max(-1.0, min(1.0, rpb))


async def calculate_brier_score(
    db: AsyncSession, lookback_days: int = 60,
) -> dict:
    """Calculate Brier score for model and market baseline.

    Lower is better. Perfect = 0.0, coin flip = 0.25.
    """
    cutoff = melb_now_naive() - timedelta(days=lookback_days)

    result = await db.execute(
        select(Pick.win_probability, Pick.value_rating, Pick.hit)
        .where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.win_probability.isnot(None),
            Pick.created_at >= cutoff,
        )
    )
    rows = result.all()

    if not rows:
        return {"model": None, "market": None, "count": 0}

    model_sum = 0.0
    market_sum = 0.0
    count = 0

    for row in rows:
        wp = row.win_probability
        vr = row.value_rating
        hit = row.hit
        outcome = 1.0 if hit else 0.0
        model_sum += (wp - outcome) ** 2

        # Market implied = our_prob / value_rating
        if vr and vr > 0:
            market_prob = wp / vr
            market_sum += (market_prob - outcome) ** 2
        else:
            market_sum += (wp - outcome) ** 2
        count += 1

    return {
        "model": round(model_sum / count, 4) if count else None,
        "market": round(market_sum / count, 4) if count else None,
        "count": count,
    }


async def calculate_category_breakdown(
    db: AsyncSession, lookback_days: int = 60,
) -> list[dict]:
    """Performance breakdown by race category (distance bucket + condition).

    Groups settled picks by their race characteristics.
    """
    cutoff = melb_now_naive() - timedelta(days=lookback_days)

    result = await db.execute(
        select(
            Pick.win_probability, Pick.hit, Pick.pnl, Pick.bet_stake,
            Pick.value_rating, Race.distance, Meeting.track_condition,
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .outerjoin(
            Race,
            and_(Race.meeting_id == Pick.meeting_id, Race.race_number == Pick.race_number),
        )
        .where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.win_probability.isnot(None),
            Pick.created_at >= cutoff,
        )
    )
    rows = result.all()

    if not rows:
        return []

    # Group by distance bucket + condition
    from punty.probability import _get_dist_bucket, _normalize_condition

    categories = {}
    for wp, hit, pnl, stake, vr, distance, condition in rows:
        dist_b = _get_dist_bucket(distance or 1400)
        cond = _normalize_condition(condition or "")
        cat_key = f"{dist_b} / {cond}" if cond else dist_b

        if cat_key not in categories:
            categories[cat_key] = {
                "category": cat_key,
                "picks": 0, "wins": 0,
                "total_pnl": 0, "total_staked": 0,
                "brier_sum": 0,
            }

        c = categories[cat_key]
        c["picks"] += 1
        if hit:
            c["wins"] += 1
        c["total_pnl"] += pnl or 0
        c["total_staked"] += stake or 0
        outcome = 1.0 if hit else 0.0
        c["brier_sum"] += (wp - outcome) ** 2

    result_list = []
    for cat in sorted(categories.values(), key=lambda x: -x["picks"]):
        if cat["picks"] < 3:
            continue
        result_list.append({
            "category": cat["category"],
            "picks": cat["picks"],
            "wins": cat["wins"],
            "hit_rate": round(cat["wins"] / cat["picks"], 4),
            "total_pnl": round(cat["total_pnl"], 2),
            "roi": round(cat["total_pnl"] / cat["total_staked"] * 100, 1) if cat["total_staked"] else 0,
            "brier": round(cat["brier_sum"] / cat["picks"], 4),
        })

    return result_list


async def _get_current_weights(db: AsyncSession) -> dict[str, float]:
    """Load current probability weights from AppSettings (as decimals 0-1)."""
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == "probability_weights")
    )
    setting = result.scalar_one_or_none()

    if setting and setting.value:
        stored = json.loads(setting.value)
        # Settings stores percentages (0-100), convert to decimals
        return {k: v / 100.0 for k, v in stored.items()}

    return dict(DEFAULT_WEIGHTS)


async def _save_weights(
    db: AsyncSession, weights: dict[str, float],
    old_weights: dict[str, float], metrics: dict,
    picks_analyzed: int,
) -> None:
    """Save tuned weights to AppSettings and log the change."""
    # Convert to percentages for storage
    weights_pct = {k: round(v * 100, 1) for k, v in weights.items()}
    old_pct = {k: round(v * 100, 1) for k, v in old_weights.items()}
    weights_json = json.dumps(weights_pct)

    result = await db.execute(
        select(AppSettings).where(AppSettings.key == "probability_weights")
    )
    setting = result.scalar_one_or_none()

    if setting:
        setting.value = weights_json
    else:
        setting = AppSettings(
            key="probability_weights",
            value=weights_json,
            description="Probability model factor weights (percentages)",
        )
        db.add(setting)

    # Log the change
    log = TuningLog(
        old_weights_json=json.dumps(old_pct),
        new_weights_json=json.dumps(weights_pct),
        metrics_json=json.dumps(metrics),
        reason="auto_tune",
        picks_analyzed=picks_analyzed,
    )
    db.add(log)

    await db.flush()
    logger.info(
        f"Probability weights auto-tuned ({picks_analyzed} picks analyzed). "
        f"Biggest change: {_biggest_change(old_weights, weights)}"
    )


def _biggest_change(old: dict, new: dict) -> str:
    """Format the biggest weight change for logging."""
    max_key = ""
    max_delta = 0.0
    for k in old:
        delta = abs(new.get(k, 0) - old.get(k, 0))
        if delta > max_delta:
            max_delta = delta
            max_key = k
    direction = "+" if new.get(max_key, 0) > old.get(max_key, 0) else "-"
    label = FACTOR_REGISTRY.get(max_key, {}).get("label", max_key)
    return f"{label} {direction}{max_delta * 100:.1f}%"


async def maybe_tune_weights(
    db: AsyncSession,
    min_sample: int = 50,
    learning_rate: float = 0.03,
) -> Optional[dict]:
    """Auto-tune probability weights based on settled results.

    Returns dict describing changes, or None if skipped.
    """
    # Check cooldown — don't tune more than once per 24 hours
    cutoff_time = melb_now_naive() - timedelta(hours=COOLDOWN_HOURS)
    last_tune = await db.execute(
        select(TuningLog.created_at)
        .order_by(TuningLog.created_at.desc())
        .limit(1)
    )
    last_row = last_tune.scalar_one_or_none()
    if last_row and last_row >= cutoff_time:
        logger.debug("Probability tuning skipped: cooldown period active")
        return None

    # Load factor performance data
    factor_perf = await analyze_factor_performance(db, lookback_days=60)
    if not factor_perf:
        logger.debug("Probability tuning skipped: no factor data available")
        return None

    # Check minimum sample
    sample_size = next(iter(factor_perf.values()), {}).get("count", 0)
    if sample_size < min_sample:
        logger.debug(f"Probability tuning skipped: only {sample_size}/{min_sample} picks with factor data")
        return None

    current_weights = await _get_current_weights(db)

    # Calculate optimal weights based on factor accuracy
    accuracies = {}
    for key in FACTOR_REGISTRY:
        perf = factor_perf.get(key)
        if perf:
            # Use absolute accuracy (correlation) as quality signal
            # Add small base to prevent any factor from going to zero
            accuracies[key] = max(0.01, abs(perf["accuracy"]) + 0.05)
        else:
            accuracies[key] = 0.05  # neutral for unknown factors

    # Softmax-like normalization to get optimal weights
    total_accuracy = sum(accuracies.values())
    optimal_weights = {k: v / total_accuracy for k, v in accuracies.items()}

    # Smooth: blend current with optimal
    new_weights = {}
    for key in FACTOR_REGISTRY:
        old_w = current_weights.get(key, DEFAULT_WEIGHTS.get(key, 0.05))
        opt_w = optimal_weights.get(key, 0.05)
        new_w = SMOOTHING * old_w + (1 - SMOOTHING) * opt_w
        new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))
        new_weights[key] = new_w

    # Normalize to sum to 1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    # Check if change is significant enough to save
    max_change = max(
        abs(new_weights.get(k, 0) - current_weights.get(k, 0))
        for k in FACTOR_REGISTRY
    )

    if max_change < MIN_CHANGE_THRESHOLD:
        logger.debug(f"Probability tuning skipped: max change {max_change:.4f} below threshold")
        return None

    # Build metrics for the log
    metrics = {
        "factor_performance": {
            k: {"edge": v["edge"], "accuracy": v["accuracy"]}
            for k, v in factor_perf.items()
        },
        "max_change": round(max_change, 4),
        "sample_size": sample_size,
    }

    # Calculate Brier score for the log
    brier = await calculate_brier_score(db, lookback_days=60)
    metrics["brier_score"] = brier

    # Save
    await _save_weights(db, new_weights, current_weights, metrics, sample_size)

    # Build summary
    changes = {}
    for key in FACTOR_REGISTRY:
        old_pct = current_weights.get(key, 0) * 100
        new_pct = new_weights[key] * 100
        delta = new_pct - old_pct
        if abs(delta) >= 0.1:
            changes[key] = {
                "label": FACTOR_REGISTRY[key]["label"],
                "old": round(old_pct, 1),
                "new": round(new_pct, 1),
                "delta": round(delta, 1),
            }

    return {
        "picks_analyzed": sample_size,
        "changes": changes,
        "max_change_pct": round(max_change * 100, 1),
    }


async def get_tuning_history(
    db: AsyncSession, limit: int = 20,
) -> list[dict]:
    """Get recent tuning events for dashboard."""
    result = await db.execute(
        select(TuningLog)
        .where(TuningLog.reason == "auto_tune")
        .order_by(TuningLog.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()

    history = []
    for row in rows:
        try:
            old_w = json.loads(row.old_weights_json)
            new_w = json.loads(row.new_weights_json)
            metrics = json.loads(row.metrics_json) if row.metrics_json else {}
        except (json.JSONDecodeError, TypeError):
            old_w, new_w, metrics = {}, {}, {}

        # Find biggest change (skip non-numeric values from other tuning systems)
        changes = {}
        for key in set(list(old_w.keys()) + list(new_w.keys())):
            old_val = old_w.get(key, 0)
            new_val = new_w.get(key, 0)
            if not isinstance(old_val, (int, float)) or not isinstance(new_val, (int, float)):
                continue
            delta = (new_val or 0) - (old_val or 0)
            if abs(delta) >= 0.1:
                changes[key] = round(delta, 1)

        history.append({
            "id": row.id,
            "date": row.created_at.isoformat() if row.created_at else None,
            "old_weights": old_w,
            "new_weights": new_w,
            "changes": changes,
            "picks_analyzed": row.picks_analyzed,
            "metrics": metrics,
            "reason": row.reason,
        })

    return history


async def analyze_context_performance(
    db: AsyncSession, lookback_days: int = 60,
) -> dict:
    """Analyze model performance by racing context (venue type / distance / class).

    Groups settled picks by context and computes per-context Brier score,
    strike rate, and ROI. This reveals where context multipliers are helping
    or hurting and which contexts need better profiles.

    Returns {
        "by_venue_type": [...],
        "by_distance": [...],
        "by_class": [...],
        "context_impact": {"with": brier, "without_estimate": brier, "delta": ...},
    }
    """
    cutoff = melb_now_naive() - timedelta(days=lookback_days)

    result = await db.execute(
        select(
            Pick.win_probability, Pick.hit, Pick.pnl, Pick.bet_stake,
            Pick.value_rating, Pick.factors_json,
            Race.distance, Race.class_,
            Meeting.venue, Meeting.track_condition,
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .outerjoin(
            Race,
            and_(Race.meeting_id == Pick.meeting_id, Race.race_number == Pick.race_number),
        )
        .where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.win_probability.isnot(None),
            Pick.created_at >= cutoff,
        )
    )
    rows = result.all()

    if not rows:
        return {}

    from punty.probability import (
        _get_dist_bucket, _context_class_bucket, _context_venue_type,
        _get_state_for_venue,
    )

    # Group by various dimensions
    venue_groups = {}
    dist_groups = {}
    class_groups = {}

    for wp, hit, pnl, stake, vr, factors_json, distance, race_class, venue, condition in rows:
        outcome = 1.0 if hit else 0.0
        brier_val = (wp - outcome) ** 2

        # Market Brier for comparison
        market_prob = (wp / vr) if vr and vr > 0 else wp
        market_brier = (market_prob - outcome) ** 2

        entry = {
            "brier": brier_val,
            "market_brier": market_brier,
            "hit": bool(hit),
            "pnl": pnl or 0,
            "stake": stake or 0,
        }

        # Venue type grouping
        state = _get_state_for_venue(venue or "")
        vtype = _context_venue_type(venue or "", state)
        venue_groups.setdefault(vtype, []).append(entry)

        # Distance grouping
        dbucket = _get_dist_bucket(distance or 1400)
        dist_groups.setdefault(dbucket, []).append(entry)

        # Class grouping
        cbucket = _context_class_bucket(race_class or "")
        class_groups.setdefault(cbucket, []).append(entry)

    def _summarize(groups: dict) -> list:
        summary = []
        for key, entries in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(entries) < 5:
                continue
            n = len(entries)
            brier = sum(e["brier"] for e in entries) / n
            market_b = sum(e["market_brier"] for e in entries) / n
            wins = sum(1 for e in entries if e["hit"])
            total_pnl = sum(e["pnl"] for e in entries)
            total_staked = sum(e["stake"] for e in entries)
            summary.append({
                "context": key,
                "picks": n,
                "brier": round(brier, 4),
                "market_brier": round(market_b, 4),
                "beats_market": brier < market_b,
                "hit_rate": round(wins / n, 4),
                "total_pnl": round(total_pnl, 2),
                "roi": round(total_pnl / total_staked * 100, 1) if total_staked else 0,
            })
        return summary

    return {
        "by_venue_type": _summarize(venue_groups),
        "by_distance": _summarize(dist_groups),
        "by_class": _summarize(class_groups),
    }


async def get_dashboard_data(db: AsyncSession) -> dict:
    """Aggregate all dashboard data in one call."""
    # Try 60-day window first; fall back to all-time if empty
    calibration = await calculate_calibration(db, lookback_days=60)
    lookback = 60
    if not calibration or all(c["count"] == 0 for c in calibration):
        calibration = await calculate_calibration(db, lookback_days=3650)
        lookback = 3650

    value_perf = await calculate_value_performance(db, lookback_days=lookback)
    factor_perf = await analyze_factor_performance(db, lookback_days=lookback)
    brier = await calculate_brier_score(db, lookback_days=lookback)
    categories = await calculate_category_breakdown(db, lookback_days=lookback)
    context_perf = await analyze_context_performance(db, lookback_days=lookback)
    weight_history = await get_tuning_history(db, limit=20)
    current_weights = await _get_current_weights(db)

    # Convert current weights to percentages for display
    weights_pct = {k: round(v * 100, 1) for k, v in current_weights.items()}

    # Build factor table data (merge weights + performance)
    factor_table = []
    for key, meta in FACTOR_REGISTRY.items():
        perf = factor_perf.get(key, {})
        factor_table.append({
            "key": key,
            "label": meta["label"],
            "category": meta["category"],
            "weight": weights_pct.get(key, 0),
            "default_weight": round(DEFAULT_WEIGHTS.get(key, 0) * 100, 1),
            "edge": perf.get("edge", 0),
            "accuracy": perf.get("accuracy", 0),
            "winner_avg": perf.get("winner_avg", 0),
            "loser_avg": perf.get("loser_avg", 0),
        })

    # Count total settled picks and picks with factor data
    total_result = await db.execute(
        select(func.count(Pick.id)).where(
            Pick.pick_type == "selection",
            Pick.settled == True,
        )
    )
    total_settled = total_result.scalar() or 0

    factor_result = await db.execute(
        select(func.count(Pick.id)).where(
            Pick.pick_type == "selection",
            Pick.settled == True,
            Pick.factors_json.isnot(None),
        )
    )
    total_with_factors = factor_result.scalar() or 0

    # Last tuning info
    last_tune_result = await db.execute(
        select(TuningLog).order_by(TuningLog.created_at.desc()).limit(1)
    )
    last_tune = last_tune_result.scalar_one_or_none()

    return {
        "calibration": calibration,
        "value_performance": value_perf,
        "factor_performance": factor_perf,
        "factor_table": factor_table,
        "brier": brier,
        "categories": categories,
        "context_performance": context_perf,
        "current_weights": weights_pct,
        "weight_history": weight_history,
        "summary": {
            "total_settled": total_settled,
            "total_with_factors": total_with_factors,
            "brier_model": brier.get("model"),
            "brier_market": brier.get("market"),
            "last_tune_date": last_tune.created_at.isoformat() if last_tune else None,
            "last_tune_picks": last_tune.picks_analyzed if last_tune else 0,
        },
    }
