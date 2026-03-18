"""Probability calibration — learns from actual results to correct predictions.

Builds separate win and place calibration maps from settled bet history.
Win calibration: bins win_probability for Win/Saver Win bets, checks hit rate.
Place calibration: bins place_probability for Place bets, checks hit rate.

Applied in two places:
  1. Kelly staking (Betfair queue) — calibrated PP before edge calculation
  2. Probability engine (weighted fallback) — corrects win/place overconfidence

Updates automatically on every settlement cycle. More data = better calibration.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, case, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Calibration bins: [0.0, 0.1), [0.1, 0.2), ... [0.9, 1.0]
NUM_BINS = 10
BIN_WIDTH = 0.10
MIN_SAMPLES = 15  # Need at least 15 bets in a bin to trust it

# Cache: recalculate at most once per hour
_calibration_cache: Optional[dict] = None  # {"win": {bin: rate}, "place": {bin: rate}}
_cache_expires: Optional[datetime] = None
CACHE_TTL = timedelta(hours=1)


def _bin_index(pp: float) -> int:
    """Map a probability to its bin index."""
    return max(0, min(NUM_BINS - 1, int(pp / BIN_WIDTH)))


async def _build_calibration_map_for_type(
    db: AsyncSession, bet_type: str,
) -> dict[int, float]:
    """Build predicted→actual calibration for a specific bet type.

    Args:
        bet_type: "win" or "place"

    Returns: {bin_index: actual_strike_rate} for bins with enough data.
    """
    if bet_type == "win":
        prob_col = "win_probability"
        type_filter = "bet_type IN ('win', 'saver_win')"
        label = "Win"
    else:
        prob_col = "place_probability"
        type_filter = "bet_type = 'place'"
        label = "Place"

    result = await db.execute(
        text(f"""
            SELECT
                CAST({prob_col} / 0.10 AS INTEGER) as prob_bin,
                COUNT(*) as cnt,
                SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as hits
            FROM picks
            WHERE settled = 1
              AND pick_type = 'selection'
              AND {type_filter}
              AND {prob_col} IS NOT NULL
              AND {prob_col} > 0
            GROUP BY prob_bin
            ORDER BY prob_bin
        """)
    )
    rows = result.fetchall()

    cal_map = {}
    for row in rows:
        bin_idx = min(row[0], NUM_BINS - 1)
        count = row[1]
        hits = row[2]
        if count >= MIN_SAMPLES:
            cal_map[bin_idx] = hits / count
            predicted = (bin_idx + 0.5) * BIN_WIDTH
            actual = cal_map[bin_idx]
            direction = "OVER" if predicted > actual + 0.03 else "UNDER" if predicted < actual - 0.03 else "OK"
            logger.debug(
                f"{label} calibration bin {bin_idx} ({predicted:.0%}): "
                f"actual={actual:.1%} n={count} [{direction}]"
            )

    return cal_map


async def build_calibration_map(db: AsyncSession) -> dict[str, dict[int, float]]:
    """Build win + place calibration maps from settled bets.

    Returns: {"win": {bin: rate}, "place": {bin: rate}}
    """
    win_map = await _build_calibration_map_for_type(db, "win")
    place_map = await _build_calibration_map_for_type(db, "place")
    return {"win": win_map, "place": place_map}


async def get_calibration_map(
    db: AsyncSession, bet_type: str = "place",
) -> dict[int, float]:
    """Get cached calibration map for a bet type, rebuilding if stale.

    Args:
        bet_type: "win" or "place" (default "place" for backward compat)
    """
    global _calibration_cache, _cache_expires

    now = datetime.utcnow()
    if _calibration_cache is not None and _cache_expires and now < _cache_expires:
        return _calibration_cache.get(bet_type, {})

    _calibration_cache = await build_calibration_map(db)
    _cache_expires = now + CACHE_TTL
    win_bins = len(_calibration_cache.get("win", {}))
    place_bins = len(_calibration_cache.get("place", {}))
    logger.info(f"Calibration maps rebuilt: win={win_bins} bins, place={place_bins} bins")
    return _calibration_cache.get(bet_type, {})


def calibrate_probability(pp: float, cal_map: dict[int, float]) -> float:
    """Apply calibration correction to a predicted probability.

    Uses isotonic-style interpolation between bins for smooth corrections.
    Falls back to raw prediction if bin has insufficient data.
    """
    if not cal_map or pp <= 0:
        return pp

    bin_idx = _bin_index(pp)

    # Direct lookup if bin has data
    if bin_idx in cal_map:
        # Interpolate within the bin for smoother correction
        bin_center = (bin_idx + 0.5) * BIN_WIDTH
        actual_center = cal_map[bin_idx]

        # Check adjacent bins for interpolation
        if pp > bin_center and (bin_idx + 1) in cal_map:
            # Interpolate toward next bin
            next_center = ((bin_idx + 1) + 0.5) * BIN_WIDTH
            next_actual = cal_map[bin_idx + 1]
            t = (pp - bin_center) / BIN_WIDTH
            return actual_center + t * (next_actual - actual_center)
        elif pp < bin_center and (bin_idx - 1) in cal_map:
            # Interpolate toward previous bin
            prev_center = ((bin_idx - 1) + 0.5) * BIN_WIDTH
            prev_actual = cal_map[bin_idx - 1]
            t = (bin_center - pp) / BIN_WIDTH
            return actual_center + t * (prev_actual - actual_center)

        return actual_center

    # No data for this bin — use raw prediction
    return pp


async def calibrated_kelly_stake(
    db: AsyncSession,
    balance: float,
    predicted_pp: float,
    odds: float,
    max_fraction: float = 0.08,
    min_stake: float = 0.50,
    min_calibrated_pp: float = 0.55,
) -> float:
    """Kelly staking with calibrated probabilities.

    This is the core learning function: it corrects overconfident predictions
    before calculating edge, so Kelly doesn't over-bet on false confidence.

    PP floor (min_calibrated_pp): rejects bets where calibrated probability
    falls below threshold. Backtest: 0.55 floor yields 70% strike rate
    (vs 64% unfiltered) with same returns — pure risk reduction.
    """
    from punty.betting.queue import calculate_kelly_stake

    if predicted_pp <= 0 or odds <= 1 or balance <= 0:
        return 0

    cal_map = await get_calibration_map(db)
    actual_pp = calibrate_probability(predicted_pp, cal_map)

    # Log significant corrections
    if abs(actual_pp - predicted_pp) > 0.05:
        logger.info(
            f"Calibration correction: {predicted_pp:.0%} -- {actual_pp:.0%} "
            f"(odds ${odds:.2f}, edge shift {(actual_pp - predicted_pp):+.0%})"
        )

    # PP floor removed — trust PP ranking to select best bets per meeting.
    # Still calibrate for accurate Kelly sizing.
    if actual_pp < min_calibrated_pp:
        logger.info(
            f"PP below old floor: calibrated {actual_pp:.0%} < {min_calibrated_pp:.0%} "
            f"(predicted {predicted_pp:.0%}, odds ${odds:.2f}) — proceeding anyway"
        )

    return calculate_kelly_stake(balance, actual_pp, odds, max_fraction, min_stake)


async def get_all_calibration_maps(
    db: AsyncSession,
) -> dict[str, dict[int, float]]:
    """Get both win and place calibration maps (cached).

    Returns: {"win": {bin: rate}, "place": {bin: rate}}
    For use in probability engine to correct both win and place predictions.
    """
    global _calibration_cache, _cache_expires

    now = datetime.utcnow()
    if _calibration_cache is not None and _cache_expires and now < _cache_expires:
        return _calibration_cache

    _calibration_cache = await build_calibration_map(db)
    _cache_expires = now + CACHE_TTL
    return _calibration_cache


def invalidate_cache():
    """Force recalculation on next call (call after settlement)."""
    global _calibration_cache, _cache_expires
    _calibration_cache = None
    _cache_expires = None
