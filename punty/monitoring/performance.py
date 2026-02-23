"""Performance monitoring: daily digest, rolling comparisons, regression detection."""

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today, MELB_TZ
from punty.models.meeting import Meeting
from punty.models.pick import Pick
from punty.results.picks import get_performance_summary, get_performance_history

logger = logging.getLogger(__name__)

# Regression thresholds
ROI_FLOOR = -15.0  # Alert if 7-day ROI below -15%
STRIKE_RATE_DROP = 5.0  # Alert if SR drops 5+ pp week-over-week
CONSECUTIVE_LOSS_DAYS = 3  # Alert after 3 consecutive losing days
DAILY_LOSS_LIMIT = -200.0  # Alert if single day loss exceeds $200

_loss_alert_sent_date: Optional[date] = None  # Only alert once per day


async def compute_daily_digest(db: AsyncSession, target_date: date) -> dict:
    """Compute daily P&L digest. Reuses get_performance_summary()."""
    summary = await get_performance_summary(db, target_date)
    return summary


async def compute_rolling_comparison(db: AsyncSession) -> dict:
    """Compare last 7 days vs previous 7 days."""
    today = melb_today()
    current = await get_performance_history(db, today - timedelta(days=6), today)
    previous = await get_performance_history(db, today - timedelta(days=13), today - timedelta(days=7))

    current_pnl = sum(d["pnl"] for d in current)
    previous_pnl = sum(d["pnl"] for d in previous)
    current_bets = sum(d["bets"] for d in current)
    previous_bets = sum(d["bets"] for d in previous)
    current_winners = sum(d["winners"] for d in current)
    previous_winners = sum(d["winners"] for d in previous)

    current_sr = (current_winners / current_bets * 100) if current_bets else 0.0
    previous_sr = (previous_winners / previous_bets * 100) if previous_bets else 0.0

    return {
        "current_7d_pnl": round(current_pnl, 2),
        "previous_7d_pnl": round(previous_pnl, 2),
        "pnl_delta": round(current_pnl - previous_pnl, 2),
        "current_7d_bets": current_bets,
        "previous_7d_bets": previous_bets,
        "current_strike_rate": round(current_sr, 1),
        "previous_strike_rate": round(previous_sr, 1),
        "sr_delta": round(current_sr - previous_sr, 1),
    }


async def check_regressions(db: AsyncSession) -> list[str]:
    """Check for performance regressions. Returns list of alert messages."""
    alerts = []
    today = melb_today()
    history = await get_performance_history(db, today - timedelta(days=6), today)

    if not history:
        return alerts

    # 7-day ROI check
    total_pnl = sum(d["pnl"] for d in history)
    total_bets = sum(d["bets"] for d in history)
    # Approximate staked as bets × $20 average
    total_staked = total_bets * 20
    if total_staked > 0:
        roi = total_pnl / total_staked * 100
        if roi < ROI_FLOOR:
            alerts.append(f"7-day ROI at {roi:.1f}% (below {ROI_FLOOR}% floor)")

    # Strike rate drop week-over-week
    comparison = await compute_rolling_comparison(db)
    if comparison["previous_7d_bets"] >= 10 and comparison["sr_delta"] < -STRIKE_RATE_DROP:
        alerts.append(
            f"Strike rate dropped {abs(comparison['sr_delta']):.1f}pp: "
            f"{comparison['previous_strike_rate']:.1f}% \u2192 {comparison['current_strike_rate']:.1f}%"
        )

    # Consecutive losing days
    consecutive_losses = 0
    for day in reversed(history):
        if day["pnl"] < 0:
            consecutive_losses += 1
        else:
            break
    if consecutive_losses >= CONSECUTIVE_LOSS_DAYS:
        alerts.append(f"{consecutive_losses} consecutive losing days")

    # Daily loss limit (most recent day)
    if history[-1]["pnl"] < DAILY_LOSS_LIMIT:
        alerts.append(
            f"Today's loss ${abs(history[-1]['pnl']):.0f} "
            f"exceeds ${abs(DAILY_LOSS_LIMIT):.0f} limit"
        )

    return alerts


async def check_calibration_drift(
    db: AsyncSession, window_days: int = 14
) -> Optional[str]:
    """Compare live prediction accuracy vs expected calibration.

    Checks if win_probability stored on Pick rows still matches
    actual outcomes within acceptable bounds.
    """
    cutoff = melb_today() - timedelta(days=window_days)

    result = await db.execute(
        select(Pick.win_probability, Pick.hit)
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.win_probability.isnot(None),
            Meeting.date >= cutoff,
        )
    )
    rows = result.all()

    if len(rows) < 50:
        return None

    # Bucket into 10% bins
    bins: dict[float, list] = {}
    for prob, hit in rows:
        if prob is None:
            continue
        bin_key = round(prob * 10) / 10  # 0.0, 0.1, 0.2, ...
        if bin_key not in bins:
            bins[bin_key] = [0.0, 0, 0]  # sum_prob, hits, count
        bins[bin_key][0] += prob
        bins[bin_key][1] += 1 if hit else 0
        bins[bin_key][2] += 1

    drift_signals = []
    for bin_key in sorted(bins.keys()):
        sum_prob, actual_hits, count = bins[bin_key]
        if count < 10:
            continue
        predicted_rate = sum_prob / count
        actual_rate = actual_hits / count
        gap = actual_rate - predicted_rate
        if abs(gap) > 0.10:
            direction = "underconfident" if gap > 0 else "overconfident"
            drift_signals.append(
                f"  {bin_key:.0%} band: {direction} by {abs(gap):.0%} (n={count})"
            )

    if drift_signals:
        return "Calibration drift:\n" + "\n".join(drift_signals)
    return None


async def check_intraday_loss(db: AsyncSession) -> Optional[str]:
    """Check if today's cumulative P&L has breached the daily loss limit.

    Called from monitor.py after each race settlement.
    Only fires once per day to avoid alert spam.
    """
    global _loss_alert_sent_date
    today = melb_today()

    if _loss_alert_sent_date == today:
        return None

    result = await db.execute(
        select(func.sum(Pick.pnl))
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Meeting.date == today,
            Pick.settled == True,
            Pick.pick_type != "big3",
        )
    )
    total_pnl = float(result.scalar() or 0)

    if total_pnl < DAILY_LOSS_LIMIT:
        _loss_alert_sent_date = today
        return (
            f"\u26a0\ufe0f Daily loss alert: ${abs(total_pnl):.0f} lost today "
            f"(limit: ${abs(DAILY_LOSS_LIMIT):.0f})"
        )
    return None
