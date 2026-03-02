"""Betfair auto-bet API endpoints."""

import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.database import get_db
from punty.models.betfair_bet import BetfairBet

logger = logging.getLogger(__name__)
router = APIRouter()


class BalanceUpdate(BaseModel):
    balance: float


class StakeUpdate(BaseModel):
    stake: float


class ToggleAll(BaseModel):
    enabled: bool
    meeting_id: str | None = None


@router.get("/queue")
async def get_queue(db: AsyncSession = Depends(get_db)):
    """Get today's bet queue grouped by meeting."""
    from punty.betting.queue import get_queue_summary
    return await get_queue_summary(db)


@router.put("/queue/{bet_id}/toggle")
async def toggle_bet(bet_id: str, db: AsyncSession = Depends(get_db)):
    """Toggle a single bet on/off."""
    result = await db.execute(select(BetfairBet).where(BetfairBet.id == bet_id))
    bet = result.scalar_one_or_none()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    if bet.status != "queued":
        raise HTTPException(status_code=400, detail=f"Cannot toggle bet in '{bet.status}' status")
    bet.enabled = not bet.enabled
    await db.commit()
    return bet.to_dict()


@router.put("/queue/{bet_id}/cycle")
async def cycle_bet(bet_id: str, db: AsyncSession = Depends(get_db)):
    """Cycle a queued bet to the next-best pick for the race."""
    from punty.betting.queue import cycle_bet_selection
    result = await cycle_bet_selection(db, bet_id)
    if not result.get("swapped") and result.get("message") == "Bet not found":
        raise HTTPException(status_code=404, detail="Bet not found")
    return result


@router.put("/queue/{bet_id}/stake")
async def update_bet_stake(bet_id: str, body: StakeUpdate, db: AsyncSession = Depends(get_db)):
    """Update stake for a single queued bet."""
    result = await db.execute(select(BetfairBet).where(BetfairBet.id == bet_id))
    bet = result.scalar_one_or_none()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    if bet.status != "queued":
        raise HTTPException(status_code=400, detail=f"Cannot edit bet in '{bet.status}' status")
    if body.stake < 0.50:
        raise HTTPException(status_code=400, detail="Minimum stake is $0.50")
    bet.stake = round(body.stake, 2)
    await db.commit()
    return bet.to_dict()


@router.put("/queue/stake-all")
async def update_all_stakes(body: StakeUpdate, db: AsyncSession = Depends(get_db)):
    """Set stake for all queued bets."""
    if body.stake < 0.50:
        raise HTTPException(status_code=400, detail="Minimum stake is $0.50")
    result = await db.execute(select(BetfairBet).where(BetfairBet.status == "queued"))
    bets = result.scalars().all()
    for bet in bets:
        bet.stake = round(body.stake, 2)
    await db.commit()
    return {"updated": len(bets), "stake": body.stake}


@router.put("/queue/toggle-all")
async def toggle_all_bets(body: ToggleAll, db: AsyncSession = Depends(get_db)):
    """Bulk enable/disable queued bets."""
    query = select(BetfairBet).where(BetfairBet.status == "queued")
    if body.meeting_id:
        query = query.where(BetfairBet.meeting_id == body.meeting_id)
    result = await db.execute(query)
    bets = result.scalars().all()
    for bet in bets:
        bet.enabled = body.enabled
    await db.commit()
    return {"updated": len(bets), "enabled": body.enabled}


@router.get("/summary")
async def get_summary(db: AsyncSession = Depends(get_db)):
    """Get P&L summary, balance, streak."""
    from punty.betting.queue import get_queue_summary
    return await get_queue_summary(db)


@router.get("/history")
async def get_history(
    days: int = 30,
    status: str = "all",
    venue: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Get settled bet history with filters and summary stats."""
    from datetime import timedelta
    cutoff_date = melb_today() - timedelta(days=days) if days > 0 else None
    result = await db.execute(
        select(BetfairBet).where(
            BetfairBet.settled == True,
        ).order_by(BetfairBet.settled_at.desc())
    )
    bets = result.scalars().all()
    # Filter by date from meeting_id (contains date string)
    filtered = []
    for b in bets:
        # Date filter
        if cutoff_date:
            try:
                parts = b.meeting_id.rsplit("-", 3)
                if len(parts) >= 4:
                    bet_date = date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
                    if bet_date < cutoff_date:
                        continue
            except (ValueError, IndexError):
                pass
        # Status filter
        if status == "won" and not b.hit:
            continue
        if status == "lost" and b.hit:
            continue
        # Venue filter (match against meeting_id prefix before date)
        if venue:
            venue_part = b.meeting_id.rsplit("-", 3)[0] if len(b.meeting_id.rsplit("-", 3)) >= 4 else b.meeting_id
            if venue.lower() not in venue_part.lower():
                continue
        filtered.append(b.to_dict())

    # Summary stats
    total = len(filtered)
    wins = sum(1 for b in filtered if b.get("hit"))
    losses = total - wins
    total_pnl = sum(b.get("pnl") or 0 for b in filtered)
    total_staked = sum(b.get("stake") or 0 for b in filtered)
    summary = {
        "total": total,
        "wins": wins,
        "losses": losses,
        "pnl": round(total_pnl, 2),
        "roi": round((total_pnl / total_staked * 100) if total_staked > 0 else 0, 1),
        "strike_rate": round((wins / total * 100) if total > 0 else 0, 1),
        "avg_stake": round((total_staked / total) if total > 0 else 0, 2),
    }
    return {"bets": filtered, "summary": summary}


@router.get("/status")
async def get_scheduler_status():
    """Get scheduler running/stopped status."""
    from punty.betting.scheduler import betfair_scheduler
    return betfair_scheduler.status()


@router.post("/scheduler/start")
async def start_scheduler():
    """Start the bet scheduler."""
    from punty.betting.scheduler import betfair_scheduler
    betfair_scheduler.start()
    return {"status": "started"}


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the bet scheduler."""
    from punty.betting.scheduler import betfair_scheduler
    betfair_scheduler.stop()
    return {"status": "stopped"}


@router.get("/balance")
async def get_balance(db: AsyncSession = Depends(get_db)):
    """Get current tracked balance."""
    from punty.betting.queue import get_balance as _get_balance, calculate_stake, _get_setting, DEFAULT_INITIAL_BALANCE
    balance = await _get_balance(db)
    initial = float(await _get_setting(db, "betfair_initial_balance", str(DEFAULT_INITIAL_BALANCE)))
    return {
        "balance": balance,
        "initial_balance": initial,
        "current_stake": calculate_stake(balance, initial),
    }


@router.put("/balance")
async def update_balance(body: BalanceUpdate, db: AsyncSession = Depends(get_db)):
    """Manually set balance."""
    from punty.betting.queue import set_balance
    await set_balance(db, body.balance)
    return {"balance": body.balance}
