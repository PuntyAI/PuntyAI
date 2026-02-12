"""Weekly P&L summary for blog 'The Ledger' section."""

import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import select, func, case, and_
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.pick import Pick
from punty.models.meeting import Meeting

logger = logging.getLogger(__name__)


async def build_weekly_ledger(
    db: AsyncSession, window_days: int = 7,
) -> dict[str, Any]:
    """Build weekly P&L summary with trend comparison to previous week.

    Returns structured dict for blog prompt injection.
    """
    today = melb_today()
    this_week_start = today - timedelta(days=window_days)
    last_week_start = this_week_start - timedelta(days=window_days)

    this_week = await _period_stats(db, this_week_start, today)
    last_week = await _period_stats(db, last_week_start, this_week_start)

    # Trend calculation
    trend = "flat"
    pnl_change = 0.0
    if last_week["total_staked"] > 0 and this_week["total_staked"] > 0:
        pnl_change = this_week["total_pnl"] - last_week["total_pnl"]
        if pnl_change > 5:
            trend = "up"
        elif pnl_change < -5:
            trend = "down"

    # Best and worst individual bets this week
    best_bet = await _best_worst_bet(db, this_week_start, today, best=True)
    worst_bet = await _best_worst_bet(db, this_week_start, today, best=False)

    # Per bet-type breakdown
    bet_type_breakdown = await _bet_type_breakdown(db, this_week_start, today)

    # Streak tracking
    streak = await _current_streak(db)

    return {
        "this_week": this_week,
        "last_week": last_week,
        "trend": trend,
        "pnl_change": round(pnl_change, 2),
        "best_bet": best_bet,
        "worst_bet": worst_bet,
        "bet_type_breakdown": bet_type_breakdown,
        "streak": streak,
    }


async def _period_stats(
    db: AsyncSession, start_date, end_date,
) -> dict[str, Any]:
    """Aggregate stats for a date period."""
    q = (
        select(
            func.count(Pick.id).label("total_bets"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("winners"),
            func.sum(Pick.bet_stake).label("staked"),
            func.sum(Pick.pnl).label("pnl"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Meeting.date >= start_date,
            Meeting.date < end_date,
        )
    )
    row = (await db.execute(q)).one()
    bets = row.total_bets or 0
    winners = int(row.winners or 0)
    staked = float(row.staked or 0)
    pnl = float(row.pnl or 0)
    sr = round(winners / bets * 100, 1) if bets > 0 else 0
    roi = round(pnl / staked * 100, 1) if staked > 0 else 0

    return {
        "total_bets": bets,
        "winners": winners,
        "strike_rate": sr,
        "total_staked": round(staked, 2),
        "total_pnl": round(pnl, 2),
        "roi": roi,
    }


async def _best_worst_bet(
    db: AsyncSession, start_date, end_date, best: bool = True,
) -> dict[str, Any] | None:
    """Find the best or worst single bet in a period."""
    order = Pick.pnl.desc() if best else Pick.pnl.asc()
    q = (
        select(
            Pick.horse_name,
            Pick.odds_at_tip,
            Pick.bet_type,
            Pick.pnl,
            Pick.bet_stake,
            Meeting.venue,
            Pick.race_number,
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Meeting.date >= start_date,
            Meeting.date < end_date,
        )
        .order_by(order)
        .limit(1)
    )
    row = (await db.execute(q)).first()
    if not row or row.pnl is None:
        return None
    return {
        "horse": row.horse_name,
        "odds": float(row.odds_at_tip or 0),
        "bet_type": row.bet_type,
        "pnl": round(float(row.pnl or 0), 2),
        "stake": float(row.bet_stake or 0),
        "venue": row.venue,
        "race_number": row.race_number,
    }


async def _bet_type_breakdown(
    db: AsyncSession, start_date, end_date,
) -> list[dict]:
    """P&L breakdown by bet type for a period."""
    q = (
        select(
            Pick.bet_type,
            func.count(Pick.id).label("bets"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("wins"),
            func.sum(Pick.bet_stake).label("staked"),
            func.sum(Pick.pnl).label("pnl"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type.isnot(None),
            Meeting.date >= start_date,
            Meeting.date < end_date,
        )
        .group_by(Pick.bet_type)
    )
    results = []
    for row in (await db.execute(q)).all():
        bt, bets, wins, staked, pnl = row
        staked = float(staked or 0)
        pnl = float(pnl or 0)
        wins = int(wins or 0)
        sr = round(wins / bets * 100, 1) if bets > 0 else 0
        roi = round(pnl / staked * 100, 1) if staked > 0 else 0
        results.append({
            "bet_type": (bt or "win").replace("_", " ").title(),
            "bets": bets, "wins": wins, "strike_rate": sr,
            "staked": round(staked, 2), "pnl": round(pnl, 2), "roi": roi,
        })
    return sorted(results, key=lambda r: r["pnl"], reverse=True)


async def _current_streak(db: AsyncSession) -> dict[str, Any]:
    """Calculate current winning/losing day streak."""
    # Get last 14 meeting dates with their aggregate daily P&L
    q = (
        select(
            Meeting.date,
            func.sum(Pick.pnl).label("daily_pnl"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(Pick.settled == True, Pick.pick_type == "selection")
        .group_by(Meeting.date)
        .order_by(Meeting.date.desc())
        .limit(14)
    )
    rows = (await db.execute(q)).all()
    if not rows:
        return {"type": "none", "count": 0}

    # Count consecutive days of same sign
    first_pnl = float(rows[0].daily_pnl or 0)
    streak_type = "winning" if first_pnl > 0 else "losing"
    count = 0
    for row in rows:
        pnl = float(row.daily_pnl or 0)
        if (streak_type == "winning" and pnl > 0) or (streak_type == "losing" and pnl <= 0):
            count += 1
        else:
            break

    return {"type": streak_type, "count": count}
