"""Bet type threshold tuning — learns optimal probability thresholds from settled picks.

Analyzes historical settled picks to find the probability sweet spots for each
bet type (win, place, each way, saver win), exotic type, and sequence variant.
Uses simulation-based threshold finding with multi-layer outlier protection.
"""

import json
import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.models.pick import Pick
from punty.models.settings import AppSettings
from punty.memory.models import TuningLog

logger = logging.getLogger(__name__)

# ── Tuning constraints ──────────────────────────────────────────────────────

COOLDOWN_HOURS = 24
SMOOTHING = 0.70  # 70% old, 30% new
MIN_CHANGE_THRESHOLD = 0.01  # 1% change required to save
LOOKBACK_DAYS = 60
RECENT_DAYS = 30  # Recent window gets 60% weight
RECENT_WEIGHT = 0.60
OLDER_WEIGHT = 0.40
OUTLIER_SIGMA = 3.0  # Trim picks with P&L > 3 sigma from mean
MIN_CONFIDENCE_SAMPLES = 10  # Need at least 10 samples per bucket

# Minimum samples required
MIN_SELECTION_PICKS = 100
MIN_EXOTIC_PICKS = 20
MIN_SEQUENCE_PICKS = 30

# ── Default thresholds (match current hardcoded values) ─────────────────────

DEFAULT_SELECTION_THRESHOLDS = {
    "win_min_prob": 0.18,
    "win_min_value": 0.90,
    "saver_win_min_prob": 0.14,
    "place_min_prob": 0.35,
    "place_min_value": 0.95,
    "each_way_min_prob": 0.15,
    "each_way_max_prob": 0.40,
    "each_way_min_odds": 4.0,
    "each_way_max_odds": 20.0,
}

DEFAULT_EXOTIC_THRESHOLDS = {
    "min_value_trifecta": 1.2,
    "min_value_exacta": 1.1,
    "min_value_quinella": 1.0,
    "min_value_first4": 1.5,
    "puntys_pick_value": 1.5,
}

DEFAULT_SEQUENCE_THRESHOLDS = {
    "width_high_skinny": 1,
    "width_med_skinny": 2,
    "width_low_skinny": 2,
    "width_high_balanced": 2,
    "width_med_balanced": 3,
    "width_low_balanced": 3,
    "width_high_wide": 3,
    "width_med_wide": 4,
    "width_low_wide": 4,
}

# ── Hard bounds (prevents degenerate configs) ───────────────────────────────

BOUNDS = {
    "win_min_prob": (0.10, 0.30),
    "win_min_value": (0.70, 1.10),
    "saver_win_min_prob": (0.08, 0.25),
    "place_min_prob": (0.25, 0.50),
    "place_min_value": (0.80, 1.10),
    "each_way_min_prob": (0.10, 0.25),
    "each_way_max_prob": (0.30, 0.50),
    "each_way_min_odds": (2.0, 6.0),
    "each_way_max_odds": (12.0, 30.0),
    "min_value_trifecta": (0.8, 2.0),
    "min_value_exacta": (0.8, 2.0),
    "min_value_quinella": (0.8, 2.0),
    "min_value_first4": (0.8, 2.0),
    "puntys_pick_value": (1.0, 3.0),
}

SEQUENCE_WIDTH_BOUNDS = (1, 5)


# ── Loading / saving ────────────────────────────────────────────────────────


async def load_bet_thresholds(db: AsyncSession) -> dict:
    """Load tuned bet type thresholds from AppSettings, with defaults."""
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == "bet_type_thresholds")
    )
    setting = result.scalar_one_or_none()

    defaults = {
        "selection": {**DEFAULT_SELECTION_THRESHOLDS},
        "exotic": {**DEFAULT_EXOTIC_THRESHOLDS},
        "sequence": {**DEFAULT_SEQUENCE_THRESHOLDS},
    }

    if not setting or not setting.value:
        return defaults

    try:
        stored = json.loads(setting.value)
        # Merge stored values over defaults (so new keys get defaults)
        for section in ("selection", "exotic", "sequence"):
            if section in stored:
                defaults[section].update(stored[section])
        return defaults
    except (json.JSONDecodeError, TypeError):
        return defaults


async def _save_thresholds(
    db: AsyncSession,
    thresholds: dict,
    old_thresholds: dict,
    metrics: dict,
    picks_analyzed: int,
) -> None:
    """Save tuned thresholds to AppSettings and log the change."""
    thresholds["last_tuned"] = melb_now_naive().isoformat()

    result = await db.execute(
        select(AppSettings).where(AppSettings.key == "bet_type_thresholds")
    )
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = json.dumps(thresholds)
    else:
        db.add(AppSettings(key="bet_type_thresholds", value=json.dumps(thresholds)))

    # Log to TuningLog (reuse existing model, differentiate by reason)
    log = TuningLog(
        old_weights_json=json.dumps(old_thresholds),
        new_weights_json=json.dumps(thresholds),
        metrics_json=json.dumps(metrics),
        reason="bet_type_tune",
        picks_analyzed=picks_analyzed,
    )
    db.add(log)
    await db.commit()
    logger.info(f"Bet type thresholds tuned ({picks_analyzed} picks analyzed)")


# ── Statistical helpers ─────────────────────────────────────────────────────


def _trim_outliers(picks: list[dict], sigma: float = OUTLIER_SIGMA) -> list[dict]:
    """Remove picks with P&L beyond sigma standard deviations from mean."""
    if len(picks) < 5:
        return picks
    pnls = [p["pnl"] for p in picks if p["pnl"] is not None]
    if not pnls:
        return picks
    mean_pnl = statistics.mean(pnls)
    try:
        std_pnl = statistics.stdev(pnls)
    except statistics.StatisticsError:
        return picks
    if std_pnl == 0:
        return picks
    lo = mean_pnl - sigma * std_pnl
    hi = mean_pnl + sigma * std_pnl
    return [p for p in picks if p["pnl"] is not None and lo <= p["pnl"] <= hi]


def _wilson_confidence(hits: int, total: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson score interval for hit rate (90% confidence with z=1.645).

    Returns (lower, upper) bounds on the true hit rate.
    """
    if total == 0:
        return (0.0, 1.0)
    p = hits / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _time_weighted_picks(
    picks: list[dict],
    cutoff: datetime,
) -> list[dict]:
    """Weight recent picks more heavily by duplicating them.

    Recent (last 30 days): 60% weight → included 3 times
    Older (30-60 days): 40% weight → included 2 times
    This approximates 60/40 weighting in aggregate statistics.
    """
    recent_cutoff = cutoff + timedelta(days=LOOKBACK_DAYS - RECENT_DAYS)
    weighted = []
    for p in picks:
        created = p.get("created_at")
        if created and created >= recent_cutoff:
            weighted.extend([p, p, p])  # 3x for recent
        else:
            weighted.extend([p, p])  # 2x for older
    return weighted


def _clamp(value: float, key: str) -> float:
    """Clamp a threshold value to its hard bounds."""
    if key in BOUNDS:
        lo, hi = BOUNDS[key]
        return max(lo, min(hi, value))
    return value


def _smooth(old_val: float, new_val: float) -> float:
    """Apply 70/30 smoothing: 70% old + 30% new."""
    return SMOOTHING * old_val + (1 - SMOOTHING) * new_val


# ── Selection analysis ──────────────────────────────────────────────────────


async def _fetch_settled_picks(
    db: AsyncSession,
    pick_type: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> list[dict]:
    """Fetch settled picks of a given type from the lookback window."""
    cutoff = melb_now_naive() - timedelta(days=lookback_days)
    result = await db.execute(
        select(Pick).where(
            Pick.pick_type == pick_type,
            Pick.settled == True,
            Pick.created_at >= cutoff,
        )
    )
    picks = []
    for p in result.scalars().all():
        picks.append({
            "id": p.id,
            "bet_type": p.bet_type,
            "win_prob": p.win_probability,
            "place_prob": p.place_probability,
            "value_rating": p.value_rating,
            "odds": p.odds_at_tip,
            "place_odds": p.place_odds_at_tip,
            "hit": p.hit,
            "pnl": p.pnl,
            "tip_rank": p.tip_rank,
            "exotic_type": p.exotic_type,
            "exotic_stake": p.exotic_stake,
            "sequence_type": p.sequence_type,
            "sequence_variant": p.sequence_variant,
            "confidence": p.confidence,
            "created_at": p.created_at,
        })
    return picks


def _bucket_performance(
    picks: list[dict],
    key: str,
    buckets: list[tuple[float, float]],
) -> list[dict]:
    """Calculate strike rate and ROI per probability/value bucket."""
    results = []
    for lo, hi in buckets:
        in_bucket = [p for p in picks if p.get(key) is not None and lo <= p[key] < hi]
        if not in_bucket:
            results.append({
                "range": f"{lo:.0%}-{hi:.0%}" if hi <= 1.0 else f"{lo:.1f}-{hi:.1f}",
                "count": 0, "strike_rate": 0, "roi": 0, "avg_pnl": 0,
                "lower_ci": 0, "upper_ci": 0,
            })
            continue
        hits = sum(1 for p in in_bucket if p["hit"])
        total_staked = sum(abs(p["pnl"]) if not p["hit"] else 0 for p in in_bucket)
        total_pnl = sum(p["pnl"] for p in in_bucket if p["pnl"] is not None)
        count = len(in_bucket)
        strike_rate = hits / count if count else 0
        # ROI: total P&L / total amount at risk
        total_risk = sum(
            (p.get("exotic_stake") or abs(p["pnl"]) if not p["hit"] else p["pnl"] / ((p["odds"] or 2) - 1) if p["pnl"] and p["pnl"] > 0 else abs(p["pnl"]) if p["pnl"] else 0)
            for p in in_bucket
        )
        roi = total_pnl / max(1, total_risk) if total_risk else 0
        lower_ci, upper_ci = _wilson_confidence(hits, count)
        results.append({
            "range": f"{lo:.0%}-{hi:.0%}" if hi <= 1.0 else f"{lo:.1f}-{hi:.1f}",
            "count": count,
            "strike_rate": round(strike_rate, 4),
            "roi": round(roi, 4),
            "avg_pnl": round(total_pnl / count, 2) if count else 0,
            "total_pnl": round(total_pnl, 2),
            "lower_ci": round(lower_ci, 4),
            "upper_ci": round(upper_ci, 4),
        })
    return results


PROB_BUCKETS = [(i / 100, (i + 5) / 100) for i in range(0, 50, 5)] + [(0.50, 1.01)]
VALUE_BUCKETS = [(0.0, 0.8), (0.8, 1.0), (1.0, 1.1), (1.1, 1.3), (1.3, 2.0), (2.0, 99.0)]


async def analyze_selection_performance(
    db: AsyncSession,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    """Analyze settled selection picks by bet type.

    Returns per-bet-type performance including:
    - count, strike_rate, roi, avg_pnl
    - probability bucket breakdown
    - optimal threshold (simulation-based)
    """
    raw_picks = await _fetch_settled_picks(db, "selection", lookback_days)
    if not raw_picks:
        return {}

    cutoff = melb_now_naive() - timedelta(days=lookback_days)
    trimmed = _trim_outliers(raw_picks)
    picks = _time_weighted_picks(trimmed, cutoff)

    result = {}
    for bt in ("win", "saver_win", "place", "each_way"):
        bt_picks = [p for p in picks if p["bet_type"] == bt]
        raw_bt = [p for p in trimmed if p["bet_type"] == bt]
        if not bt_picks:
            result[bt] = {"count": 0, "strike_rate": 0, "roi": 0}
            continue

        hits = sum(1 for p in bt_picks if p["hit"])
        total_pnl = sum(p["pnl"] for p in bt_picks if p["pnl"] is not None)
        count = len(bt_picks)
        strike_rate = hits / count if count else 0

        # Optimal threshold simulation
        prob_key = "place_prob" if bt == "place" else "win_prob"
        optimal_threshold = _find_optimal_threshold(raw_bt, prob_key, bt)

        # Bucket performance
        buckets = _bucket_performance(raw_bt, prob_key, PROB_BUCKETS)

        result[bt] = {
            "count": len(raw_bt),
            "strike_rate": round(strike_rate, 4),
            "roi": round(total_pnl / max(1, count), 2),
            "total_pnl": round(sum(p["pnl"] for p in raw_bt if p["pnl"] is not None), 2),
            "optimal_threshold": optimal_threshold,
            "buckets": buckets,
        }

    return result


def _find_optimal_threshold(
    picks: list[dict],
    prob_key: str,
    bet_type: str,
) -> dict:
    """Simulate different probability thresholds to find the one that maximizes ROI.

    For each threshold candidate (0.05 to 0.50 in 0.01 steps):
    - Filter picks to those at or above the threshold
    - Calculate ROI and strike rate
    - Pick the threshold with best ROI that meets minimum strike rate

    Returns {threshold, roi, strike_rate, sample_size, improvement_pct}.
    """
    min_strike = {"win": 0.20, "saver_win": 0.18, "place": 0.35, "each_way": 0.22}
    required_strike = min_strike.get(bet_type, 0.20)

    best = {"threshold": 0.0, "roi": -999, "strike_rate": 0, "sample_size": 0}

    for thresh_int in range(5, 51):
        threshold = thresh_int / 100
        above = [p for p in picks if p.get(prob_key) is not None and p[prob_key] >= threshold]
        if len(above) < MIN_CONFIDENCE_SAMPLES:
            continue

        hits = sum(1 for p in above if p["hit"])
        strike_rate = hits / len(above)
        total_pnl = sum(p["pnl"] for p in above if p["pnl"] is not None)
        roi = total_pnl / len(above)

        if strike_rate >= required_strike and roi > best["roi"]:
            best = {
                "threshold": threshold,
                "roi": round(roi, 4),
                "strike_rate": round(strike_rate, 4),
                "sample_size": len(above),
            }

    # If nothing beats min strike rate, find best ROI regardless
    if best["threshold"] == 0.0:
        for thresh_int in range(5, 51):
            threshold = thresh_int / 100
            above = [p for p in picks if p.get(prob_key) is not None and p[prob_key] >= threshold]
            if len(above) < MIN_CONFIDENCE_SAMPLES:
                continue
            total_pnl = sum(p["pnl"] for p in above if p["pnl"] is not None)
            hits = sum(1 for p in above if p["hit"])
            roi = total_pnl / len(above)
            if roi > best["roi"]:
                best = {
                    "threshold": threshold,
                    "roi": round(roi, 4),
                    "strike_rate": round(hits / len(above), 4),
                    "sample_size": len(above),
                }

    return best


# ── Exotic analysis ─────────────────────────────────────────────────────────


async def analyze_exotic_performance(
    db: AsyncSession,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    """Analyze settled exotic picks by exotic type.

    Returns per-type: count, hit_rate, roi, avg_value, optimal_value_threshold.
    """
    raw_picks = await _fetch_settled_picks(db, "exotic", lookback_days)
    if not raw_picks:
        return {}

    trimmed = _trim_outliers(raw_picks)
    result = {}

    for etype in ("Trifecta", "Exacta", "Quinella", "First4"):
        type_picks = [p for p in trimmed if p.get("exotic_type") and etype.lower() in p["exotic_type"].lower()]
        if not type_picks:
            result[etype] = {"count": 0, "hit_rate": 0, "roi": 0}
            continue

        hits = sum(1 for p in type_picks if p["hit"])
        total_pnl = sum(p["pnl"] for p in type_picks if p["pnl"] is not None)
        count = len(type_picks)
        hit_rate = hits / count if count else 0
        total_staked = sum(p.get("exotic_stake") or 20.0 for p in type_picks)
        roi = total_pnl / total_staked if total_staked else 0
        avg_value = statistics.mean([p["value_rating"] for p in type_picks if p.get("value_rating")]) if any(p.get("value_rating") for p in type_picks) else 0

        # Find optimal value threshold
        optimal_value = _find_optimal_value_threshold(type_picks)

        result[etype] = {
            "count": count,
            "hit_rate": round(hit_rate, 4),
            "roi": round(roi, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_value": round(avg_value, 3),
            "optimal_value": optimal_value,
        }

    return result


def _find_optimal_value_threshold(picks: list[dict]) -> dict:
    """Find the value_rating threshold that maximizes ROI for exotics."""
    best = {"threshold": 0.0, "roi": -999, "hit_rate": 0, "sample_size": 0}

    for thresh_int in range(60, 201, 5):  # 0.60 to 2.00 in 0.05 steps
        threshold = thresh_int / 100
        above = [p for p in picks if p.get("value_rating") is not None and p["value_rating"] >= threshold]
        if len(above) < 5:
            continue

        hits = sum(1 for p in above if p["hit"])
        total_pnl = sum(p["pnl"] for p in above if p["pnl"] is not None)
        total_staked = sum(p.get("exotic_stake") or 20.0 for p in above)
        roi = total_pnl / total_staked if total_staked else 0

        if roi > best["roi"]:
            best = {
                "threshold": threshold,
                "roi": round(roi, 4),
                "hit_rate": round(hits / len(above), 4),
                "sample_size": len(above),
            }

    return best


# ── Sequence analysis ───────────────────────────────────────────────────────


async def analyze_sequence_performance(
    db: AsyncSession,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    """Analyze settled sequence picks by variant.

    Returns per-variant: count, hit_rate, roi, total_pnl, leg_analysis.
    """
    raw_picks = await _fetch_settled_picks(db, "sequence", lookback_days)
    if not raw_picks:
        return {}

    trimmed = _trim_outliers(raw_picks)
    result = {}

    for variant in ("skinny", "balanced", "wide"):
        var_picks = [p for p in trimmed if p.get("sequence_variant") == variant]
        if not var_picks:
            result[variant] = {"count": 0, "hit_rate": 0, "roi": 0}
            continue

        hits = sum(1 for p in var_picks if p["hit"])
        total_pnl = sum(p["pnl"] for p in var_picks if p["pnl"] is not None)
        count = len(var_picks)
        hit_rate = hits / count if count else 0
        total_staked = sum(p.get("exotic_stake") or 10.0 for p in var_picks)
        roi = total_pnl / total_staked if total_staked else 0

        # Confidence level analysis (from pick confidence field)
        conf_perf = {}
        for conf in ("HIGH", "MED", "LOW"):
            conf_picks = [p for p in var_picks if p.get("confidence") == conf]
            if conf_picks:
                c_hits = sum(1 for p in conf_picks if p["hit"])
                conf_perf[conf] = {
                    "count": len(conf_picks),
                    "hit_rate": round(c_hits / len(conf_picks), 4),
                }

        result[variant] = {
            "count": count,
            "hit_rate": round(hit_rate, 4),
            "roi": round(roi, 4),
            "total_pnl": round(total_pnl, 2),
            "total_staked": round(total_staked, 2),
            "confidence_breakdown": conf_perf,
        }

    return result


# ── Combined tuner ──────────────────────────────────────────────────────────


def _compute_selection_thresholds(
    analysis: dict,
    current: dict,
) -> dict:
    """Compute new selection thresholds from analysis, smoothed with current."""
    new = {**current}

    # Win threshold: use optimal from simulation
    win_data = analysis.get("win", {})
    if win_data.get("optimal_threshold", {}).get("threshold"):
        opt = win_data["optimal_threshold"]["threshold"]
        if win_data["optimal_threshold"].get("sample_size", 0) >= MIN_CONFIDENCE_SAMPLES:
            new["win_min_prob"] = _clamp(_smooth(current["win_min_prob"], opt), "win_min_prob")

    # Saver win threshold
    sw_data = analysis.get("saver_win", {})
    if sw_data.get("optimal_threshold", {}).get("threshold"):
        opt = sw_data["optimal_threshold"]["threshold"]
        if sw_data["optimal_threshold"].get("sample_size", 0) >= MIN_CONFIDENCE_SAMPLES:
            new["saver_win_min_prob"] = _clamp(_smooth(current["saver_win_min_prob"], opt), "saver_win_min_prob")

    # Place threshold
    pl_data = analysis.get("place", {})
    if pl_data.get("optimal_threshold", {}).get("threshold"):
        opt = pl_data["optimal_threshold"]["threshold"]
        if pl_data["optimal_threshold"].get("sample_size", 0) >= MIN_CONFIDENCE_SAMPLES:
            new["place_min_prob"] = _clamp(_smooth(current["place_min_prob"], opt), "place_min_prob")

    # Each way thresholds
    ew_data = analysis.get("each_way", {})
    if ew_data.get("optimal_threshold", {}).get("threshold"):
        opt = ew_data["optimal_threshold"]["threshold"]
        if ew_data["optimal_threshold"].get("sample_size", 0) >= MIN_CONFIDENCE_SAMPLES:
            new["each_way_min_prob"] = _clamp(_smooth(current["each_way_min_prob"], opt), "each_way_min_prob")

    return new


def _compute_exotic_thresholds(
    analysis: dict,
    current: dict,
) -> dict:
    """Compute new exotic thresholds from analysis."""
    new = {**current}

    type_to_key = {
        "Trifecta": "min_value_trifecta",
        "Exacta": "min_value_exacta",
        "Quinella": "min_value_quinella",
        "First4": "min_value_first4",
    }

    for etype, key in type_to_key.items():
        edata = analysis.get(etype, {})
        opt_value = edata.get("optimal_value", {})
        if opt_value.get("threshold") and opt_value.get("sample_size", 0) >= 5:
            new[key] = _clamp(_smooth(current[key], opt_value["threshold"]), key)

    return new


def _compute_sequence_thresholds(
    analysis: dict,
    current: dict,
) -> dict:
    """Compute new sequence width thresholds from analysis.

    If a variant's ROI is positive, keep or narrow widths (it's working).
    If ROI is negative and hit_rate is low, widen to improve hit rate.
    If ROI is negative and hit_rate is OK, narrow to reduce cost.
    """
    new = {**current}

    for variant in ("skinny", "balanced", "wide"):
        vdata = analysis.get(variant, {})
        if not vdata or vdata.get("count", 0) < 10:
            continue

        hit_rate = vdata.get("hit_rate", 0)
        roi = vdata.get("roi", 0)

        # Hit rate targets by variant
        targets = {"skinny": 0.08, "balanced": 0.12, "wide": 0.18}
        target_hit = targets.get(variant, 0.12)

        for conf in ("high", "med", "low"):
            key = f"width_{conf}_{variant}"
            cur_width = current.get(key, 2)

            if hit_rate < target_hit * 0.7:
                # Hit rate well below target → widen by 1
                optimal = cur_width + 1
            elif hit_rate > target_hit * 1.3 and roi < 0:
                # Hitting enough but losing money → narrow by 1 (reduce cost)
                optimal = cur_width - 1
            else:
                optimal = cur_width  # no change

            optimal = max(SEQUENCE_WIDTH_BOUNDS[0], min(SEQUENCE_WIDTH_BOUNDS[1], optimal))
            # Round smooth to int
            smoothed = SMOOTHING * cur_width + (1 - SMOOTHING) * optimal
            new[key] = max(SEQUENCE_WIDTH_BOUNDS[0], min(SEQUENCE_WIDTH_BOUNDS[1], round(smoothed)))

    return new


async def maybe_tune_bet_thresholds(
    db: AsyncSession,
) -> Optional[dict]:
    """Auto-tune bet type thresholds based on settled results.

    Returns dict describing changes, or None if skipped.
    """
    # Check cooldown
    cutoff_time = melb_now_naive() - timedelta(hours=COOLDOWN_HOURS)
    last_tune = await db.execute(
        select(TuningLog.created_at)
        .where(TuningLog.reason == "bet_type_tune")
        .order_by(TuningLog.created_at.desc())
        .limit(1)
    )
    last_row = last_tune.scalar_one_or_none()
    if last_row and last_row >= cutoff_time:
        logger.debug("Bet type tuning skipped: cooldown period active")
        return None

    # Load current thresholds
    current = await load_bet_thresholds(db)
    old_thresholds = json.loads(json.dumps(current))  # deep copy

    # ── Analyze selections ──────────────────────────────────────────────
    sel_analysis = await analyze_selection_performance(db)
    total_selections = sum(v.get("count", 0) for v in sel_analysis.values())

    # ── Analyze exotics ─────────────────────────────────────────────────
    exotic_analysis = await analyze_exotic_performance(db)
    total_exotics = sum(v.get("count", 0) for v in exotic_analysis.values())

    # ── Analyze sequences ───────────────────────────────────────────────
    seq_analysis = await analyze_sequence_performance(db)
    total_sequences = sum(v.get("count", 0) for v in seq_analysis.values())

    total_picks = total_selections + total_exotics + total_sequences
    if total_picks == 0:
        logger.debug("Bet type tuning skipped: no settled picks")
        return None

    # ── Compute new thresholds ──────────────────────────────────────────
    new_thresholds = {**current}

    if total_selections >= MIN_SELECTION_PICKS:
        new_thresholds["selection"] = _compute_selection_thresholds(
            sel_analysis, current["selection"]
        )

    if total_exotics >= MIN_EXOTIC_PICKS:
        new_thresholds["exotic"] = _compute_exotic_thresholds(
            exotic_analysis, current["exotic"]
        )

    if total_sequences >= MIN_SEQUENCE_PICKS:
        new_thresholds["sequence"] = _compute_sequence_thresholds(
            seq_analysis, current["sequence"]
        )

    # ── Check significance ──────────────────────────────────────────────
    max_change = 0.0
    max_change_key = ""
    for section in ("selection", "exotic", "sequence"):
        for key in new_thresholds.get(section, {}):
            if key == "last_tuned":
                continue
            old_v = old_thresholds.get(section, {}).get(key, 0)
            new_v = new_thresholds.get(section, {}).get(key, 0)
            if old_v:
                change = abs(new_v - old_v) / abs(old_v)
            else:
                change = abs(new_v - old_v)
            if change > max_change:
                max_change = change
                max_change_key = f"{section}.{key}"

    if max_change < MIN_CHANGE_THRESHOLD:
        logger.debug(f"Bet type tuning skipped: max change {max_change:.4f} below threshold")
        return None

    # ── Build metrics ───────────────────────────────────────────────────
    metrics = {
        "selection_analysis": {
            bt: {k: v for k, v in data.items() if k != "buckets"}
            for bt, data in sel_analysis.items()
        },
        "exotic_analysis": exotic_analysis,
        "sequence_analysis": seq_analysis,
        "max_change": round(max_change, 4),
        "max_change_key": max_change_key,
        "total_selections": total_selections,
        "total_exotics": total_exotics,
        "total_sequences": total_sequences,
    }

    # ── Save ────────────────────────────────────────────────────────────
    await _save_thresholds(db, new_thresholds, old_thresholds, metrics, total_picks)

    # Build change summary
    changes = {}
    for section in ("selection", "exotic", "sequence"):
        for key in new_thresholds.get(section, {}):
            old_v = old_thresholds.get(section, {}).get(key)
            new_v = new_thresholds.get(section, {}).get(key)
            if old_v is not None and new_v is not None and old_v != new_v:
                changes[f"{section}.{key}"] = {
                    "old": old_v,
                    "new": new_v,
                    "delta": round(new_v - old_v, 4),
                }

    return {
        "picks_analyzed": total_picks,
        "changes": changes,
        "max_change_pct": round(max_change * 100, 1),
        "max_change_key": max_change_key,
    }


# ── Dashboard data ──────────────────────────────────────────────────────────


async def get_bet_type_dashboard(db: AsyncSession) -> dict:
    """Get all bet type performance data for the calibration dashboard."""
    sel = await analyze_selection_performance(db)
    exotic = await analyze_exotic_performance(db)
    seq = await analyze_sequence_performance(db)
    current = await load_bet_thresholds(db)

    return {
        "selection": sel,
        "exotic": exotic,
        "sequence": seq,
        "current_thresholds": current,
    }
