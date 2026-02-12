"""Weekly awards computation from settled picks data."""

import logging
from datetime import timedelta
from typing import Any, Optional

from sqlalchemy import select, func, case, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.pick import Pick
from punty.models.meeting import Meeting, Race, Runner

logger = logging.getLogger(__name__)


async def compute_weekly_awards(
    db: AsyncSession, window_days: int = 7,
) -> dict[str, Any]:
    """Compute Punty Awards from last N days of settled picks.

    Returns dict with keys:
      jockey_of_the_week, roughie_of_the_week, value_bomb,
      track_to_watch, wooden_spoon, power_rankings
    """
    cutoff = melb_today() - timedelta(days=window_days)
    base_filter = [Pick.settled == True, Pick.settled_at >= cutoff]

    awards: dict[str, Any] = {}

    # ── Jockey of the Week ─────────────────────────────────────────────────
    jockey_q = (
        select(
            Runner.jockey,
            func.count(Pick.id).label("bets"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("wins"),
            func.sum(Pick.pnl).label("pnl"),
            func.sum(Pick.bet_stake).label("staked"),
        )
        .join(Race, and_(
            Pick.meeting_id == Race.meeting_id,
            Pick.race_number == Race.race_number,
        ))
        .join(Runner, and_(
            Runner.race_id == Race.id,
            Runner.saddlecloth == Pick.saddlecloth,
        ))
        .where(*base_filter, Pick.pick_type == "selection", Runner.jockey.isnot(None))
        .group_by(Runner.jockey)
        .having(func.count(Pick.id) >= 3)
        .order_by(desc("pnl"))
        .limit(1)
    )
    row = (await db.execute(jockey_q)).first()
    if row:
        staked = float(row.staked or 0)
        roi = round(float(row.pnl or 0) / staked * 100, 1) if staked > 0 else 0
        awards["jockey_of_the_week"] = {
            "name": row.jockey,
            "bets": row.bets,
            "wins": int(row.wins or 0),
            "pnl": round(float(row.pnl or 0), 2),
            "roi": roi,
        }

    # ── Roughie of the Week ────────────────────────────────────────────────
    roughie_q = (
        select(
            Pick.horse_name,
            Pick.odds_at_tip,
            Pick.pnl,
            Meeting.venue,
            Pick.race_number,
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(*base_filter, Pick.pick_type == "selection",
               Pick.hit == True, Pick.odds_at_tip >= 10.0)
        .order_by(desc(Pick.odds_at_tip))
        .limit(1)
    )
    row = (await db.execute(roughie_q)).first()
    if row:
        awards["roughie_of_the_week"] = {
            "horse": row.horse_name,
            "odds": float(row.odds_at_tip or 0),
            "pnl": round(float(row.pnl or 0), 2),
            "venue": row.venue,
            "race_number": row.race_number,
        }

    # ── Value Bomb ─────────────────────────────────────────────────────────
    value_q = (
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
        .where(*base_filter, Pick.hit == True, Pick.pnl > 0)
        .order_by(desc(Pick.pnl))
        .limit(1)
    )
    row = (await db.execute(value_q)).first()
    if row:
        awards["value_bomb"] = {
            "horse": row.horse_name,
            "odds": float(row.odds_at_tip or 0),
            "bet_type": row.bet_type,
            "pnl": round(float(row.pnl or 0), 2),
            "stake": float(row.bet_stake or 0),
            "venue": row.venue,
            "race_number": row.race_number,
        }

    # ── Track to Watch ─────────────────────────────────────────────────────
    track_q = (
        select(
            Meeting.venue,
            func.count(Pick.id).label("bets"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("wins"),
            func.sum(Pick.pnl).label("pnl"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(*base_filter, Pick.pick_type == "selection")
        .group_by(Meeting.venue)
        .having(func.count(Pick.id) >= 3)
        .order_by(desc("pnl"))
        .limit(1)
    )
    row = (await db.execute(track_q)).first()
    if row:
        sr = round(int(row.wins or 0) / row.bets * 100, 1) if row.bets > 0 else 0
        awards["track_to_watch"] = {
            "venue": row.venue,
            "bets": row.bets,
            "wins": int(row.wins or 0),
            "strike_rate": sr,
            "pnl": round(float(row.pnl or 0), 2),
        }

    # ── Wooden Spoon (worst venue or biggest single loss) ──────────────────
    spoon_q = (
        select(
            Meeting.venue,
            func.count(Pick.id).label("bets"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("wins"),
            func.sum(Pick.pnl).label("pnl"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(*base_filter, Pick.pick_type == "selection")
        .group_by(Meeting.venue)
        .having(func.count(Pick.id) >= 3)
        .order_by("pnl")  # ascending = worst P&L first
        .limit(1)
    )
    row = (await db.execute(spoon_q)).first()
    if row and float(row.pnl or 0) < 0:
        awards["wooden_spoon"] = {
            "venue": row.venue,
            "bets": row.bets,
            "wins": int(row.wins or 0),
            "pnl": round(float(row.pnl or 0), 2),
        }

    # ── Power Rankings (top 5 jockeys + trainers last 30 days) ─────────────
    cutoff_30 = melb_today() - timedelta(days=30)
    power_filter = [Pick.settled == True, Pick.settled_at >= cutoff_30]

    power_rankings = {"jockeys": [], "trainers": []}
    for field, label in [(Runner.jockey, "jockeys"), (Runner.trainer, "trainers")]:
        pq = (
            select(
                field,
                func.count(Pick.id).label("bets"),
                func.sum(case((Pick.hit == True, 1), else_=0)).label("wins"),
                func.sum(Pick.pnl).label("pnl"),
            )
            .join(Race, and_(
                Pick.meeting_id == Race.meeting_id,
                Pick.race_number == Race.race_number,
            ))
            .join(Runner, and_(
                Runner.race_id == Race.id,
                Runner.saddlecloth == Pick.saddlecloth,
            ))
            .where(*power_filter, Pick.pick_type == "selection", field.isnot(None))
            .group_by(field)
            .having(func.count(Pick.id) >= 5)
            .order_by(desc("pnl"))
            .limit(5)
        )
        for row in (await db.execute(pq)).all():
            name, bets, wins, pnl = row
            wins = int(wins or 0)
            pnl = float(pnl or 0)
            sr = round(wins / bets * 100, 1) if bets > 0 else 0
            power_rankings[label].append({
                "name": name, "bets": bets, "wins": wins,
                "strike_rate": sr, "pnl": round(pnl, 2),
            })
    awards["power_rankings"] = power_rankings

    return awards
