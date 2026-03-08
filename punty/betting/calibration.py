"""Probability calibration — learns from actual results to correct LGBM predictions.

The LGBM model's place probabilities are miscalibrated above 0.60:
  Predicted 0.70 → Actual 0.53 (overconfident by 17%)
  Predicted 0.80 → Actual 0.58 (overconfident by 22%)

This module builds a calibration map from settled bet history and applies it
before Kelly staking, so we bet based on REALITY not predictions.

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
_calibration_cache: Optional[dict] = None
_cache_expires: Optional[datetime] = None
CACHE_TTL = timedelta(hours=1)


def _bin_index(pp: float) -> int:
    """Map a probability to its bin index."""
    return max(0, min(NUM_BINS - 1, int(pp / BIN_WIDTH)))


async def build_calibration_map(db: AsyncSession) -> dict[int, float]:
    """Build predicted→actual calibration from all settled place bets.

    Returns: {bin_index: actual_strike_rate} for bins with enough data.
    Bins without enough data return None (use raw prediction).
    """
    from punty.models.pick import Pick

    # Query: group settled place bets by PP bucket, get actual strike rate
    result = await db.execute(
        text("""
            SELECT
                CAST(place_probability / 0.10 AS INTEGER) as pp_bin,
                COUNT(*) as cnt,
                SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as hits
            FROM picks
            WHERE settled = 1
              AND pick_type = 'selection'
              AND bet_type IN ('place', 'win', 'saver_win')
              AND place_probability IS NOT NULL
              AND place_probability > 0
            GROUP BY pp_bin
            ORDER BY pp_bin
        """)
    )
    rows = result.fetchall()

    cal_map = {}
    for row in rows:
        bin_idx = min(row[0], NUM_BINS - 1)  # Clamp 1.0 edge case
        count = row[1]
        hits = row[2]
        if count >= MIN_SAMPLES:
            cal_map[bin_idx] = hits / count
            predicted = (bin_idx + 0.5) * BIN_WIDTH
            actual = cal_map[bin_idx]
            direction = "OVER" if predicted > actual + 0.03 else "UNDER" if predicted < actual - 0.03 else "OK"
            logger.debug(
                f"Calibration bin {bin_idx} ({predicted:.0%}): "
                f"actual={actual:.1%} n={count} [{direction}]"
            )

    return cal_map


async def get_calibration_map(db: AsyncSession) -> dict[int, float]:
    """Get cached calibration map, rebuilding if stale."""
    global _calibration_cache, _cache_expires

    now = datetime.utcnow()
    if _calibration_cache is not None and _cache_expires and now < _cache_expires:
        return _calibration_cache

    _calibration_cache = await build_calibration_map(db)
    _cache_expires = now + CACHE_TTL
    logger.info(f"Calibration map rebuilt: {len(_calibration_cache)} bins calibrated")
    return _calibration_cache


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

    # PP floor gate: reject bets where calibrated probability is too low
    if actual_pp < min_calibrated_pp:
        logger.info(
            f"PP floor rejection: calibrated {actual_pp:.0%} < {min_calibrated_pp:.0%} "
            f"(predicted {predicted_pp:.0%}, odds ${odds:.2f})"
        )
        return 0

    return calculate_kelly_stake(balance, actual_pp, odds, max_fraction, min_stake)


def invalidate_cache():
    """Force recalculation on next call (call after settlement)."""
    global _calibration_cache, _cache_expires
    _calibration_cache = None
    _cache_expires = None
