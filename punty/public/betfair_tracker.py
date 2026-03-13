"""Betfair tracker routes — password-protected public view."""

from datetime import date

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

from punty.config import melb_today
from punty.models.database import async_session

from punty.public.deps import (
    templates,
    _BF_TRACKER_PASSWORD,
    _BF_TRACKER_COOKIE,
    _bf_make_token,
    _bf_check_auth,
)

router = APIRouter()


@router.get("/betfair-tracker", response_class=HTMLResponse)
async def betfair_tracker(request: Request):
    """Public betfair tracker — shows password gate or tracker page."""
    if not _bf_check_auth(request):
        return templates.TemplateResponse("betfair_tracker_login.html", {"request": request, "error": ""})
    return templates.TemplateResponse("betfair_tracker.html", {"request": request})


@router.post("/betfair-tracker", response_class=HTMLResponse)
async def betfair_tracker_login(request: Request, password: str = Form(...)):
    """Verify password and set auth cookie."""
    if password != _BF_TRACKER_PASSWORD:
        return templates.TemplateResponse("betfair_tracker_login.html", {"request": request, "error": "Wrong password"})
    response = RedirectResponse(url="/betfair-tracker", status_code=303)
    response.set_cookie(
        _BF_TRACKER_COOKIE,
        _bf_make_token(),
        max_age=90 * 86400,  # 90 days
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/api/betfair-tracker/queue")
async def bf_tracker_queue(request: Request):
    """Public read-only betfair queue — today's bets + stats."""
    if not _bf_check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from punty.betting.queue import get_queue_summary
    async with async_session() as db:
        return await get_queue_summary(db)


@router.get("/api/betfair-tracker/history")
async def bf_tracker_history(request: Request, days: int = 30, status: str = "all", venue: str = ""):
    """Public read-only betfair history."""
    if not _bf_check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from datetime import timedelta
    from punty.models.betfair_bet import BetfairBet
    from punty.models.pick import Pick
    from sqlalchemy import select
    cutoff_date = melb_today() - timedelta(days=days) if days > 0 else None
    async with async_session() as db:
        result = await db.execute(
            select(BetfairBet).where(BetfairBet.settled == True).order_by(BetfairBet.settled_at.desc())
        )
        bets = result.scalars().all()

        # Batch load picks for PP enrichment
        pick_ids = [b.pick_id for b in bets if b.pick_id]
        pp_map = {}
        if pick_ids:
            pick_result = await db.execute(select(Pick.id, Pick.place_probability).where(Pick.id.in_(pick_ids)))
            pp_map = {row[0]: row[1] for row in pick_result.all()}

    filtered = []
    for b in bets:
        if cutoff_date:
            try:
                parts = b.meeting_id.rsplit("-", 3)
                if len(parts) >= 4:
                    bet_date = date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
                    if bet_date < cutoff_date:
                        continue
            except (ValueError, IndexError):
                pass
        if status == "won" and not b.hit:
            continue
        if status == "lost" and b.hit:
            continue
        if venue:
            venue_part = b.meeting_id.rsplit("-", 3)[0] if len(b.meeting_id.rsplit("-", 3)) >= 4 else b.meeting_id
            if venue.lower() not in venue_part.lower():
                continue
        d = b.to_dict()
        d["place_probability"] = pp_map.get(b.pick_id)
        filtered.append(d)
    total = len(filtered)
    wins = sum(1 for b in filtered if b.get("hit"))
    total_pnl = sum(b.get("pnl") or 0 for b in filtered)
    total_staked = sum(b.get("stake") or 0 for b in filtered)
    summary = {
        "total": total,
        "wins": wins,
        "losses": total - wins,
        "pnl": round(total_pnl, 2),
        "roi": round((total_pnl / total_staked * 100) if total_staked > 0 else 0, 1),
        "strike_rate": round((wins / total * 100) if total > 0 else 0, 1),
        "avg_stake": round((total_staked / total) if total > 0 else 0, 2),
    }
    return {"bets": filtered, "summary": summary}


@router.get("/api/betfair-tracker/dashboard")
async def bf_tracker_dashboard(request: Request):
    """Public read-only dashboard stats — time-period breakdowns."""
    if not _bf_check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from datetime import timedelta, datetime
    from punty.models.betfair_bet import BetfairBet
    from punty.betting.queue import get_balance, calculate_stake, _get_setting, DEFAULT_INITIAL_BALANCE
    from sqlalchemy import select

    today = melb_today()
    now = datetime.now()

    async with async_session() as db:
        # All settled bets
        result = await db.execute(
            select(BetfairBet).where(BetfairBet.settled == True).order_by(BetfairBet.settled_at.desc())
        )
        all_bets = result.scalars().all()

        balance = await get_balance(db)
        initial = float(await _get_setting(db, "betfair_initial_balance", str(DEFAULT_INITIAL_BALANCE)))
        stake = calculate_stake(balance, initial)

    def _bet_date(b):
        try:
            parts = b.meeting_id.rsplit("-", 3)
            if len(parts) >= 4:
                return date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
        except (ValueError, IndexError):
            pass
        return None

    def _calc_stats(bets_list):
        total = len(bets_list)
        if total == 0:
            return {"bets": 0, "wins": 0, "losses": 0, "pnl": 0, "roi": 0, "strike_rate": 0, "staked": 0}
        wins = sum(1 for b in bets_list if b.hit)
        total_pnl = sum(b.pnl or 0 for b in bets_list)
        total_staked = sum(b.stake or 0 for b in bets_list)
        return {
            "bets": total,
            "wins": wins,
            "losses": total - wins,
            "pnl": round(total_pnl, 2),
            "roi": round((total_pnl / total_staked * 100) if total_staked > 0 else 0, 1),
            "strike_rate": round((wins / total * 100) if total > 0 else 0, 1),
            "staked": round(total_staked, 2),
        }

    def _calc_period(bets_list):
        """Calculate stats + best/worst for a period."""
        stats = _calc_stats(bets_list)
        # Best win: highest positive P&L; Worst loss: most negative P&L
        winners = [b for b in bets_list if (b.pnl or 0) > 0]
        losers = [b for b in bets_list if (b.pnl or 0) < 0]
        if winners:
            best = max(winners, key=lambda b: b.pnl)
            stats["best_bet"] = {"horse": best.horse_name, "pnl": best.pnl, "meeting": best.meeting_id}
        else:
            stats["best_bet"] = None
        if losers:
            worst = min(losers, key=lambda b: b.pnl)
            stats["worst_bet"] = {"horse": worst.horse_name, "pnl": worst.pnl, "meeting": worst.meeting_id}
        else:
            stats["worst_bet"] = None
        # Running P&L for this period
        running = []
        cumulative = 0
        for b in sorted(bets_list, key=lambda b: b.settled_at or b.created_at):
            cumulative += b.pnl or 0
            running.append({"id": b.id, "pnl": round(cumulative, 2), "date": _bet_date(b).isoformat() if _bet_date(b) else ""})
        stats["running_pnl"] = running
        return stats

    # Bucket bets by period
    today_bets = [b for b in all_bets if _bet_date(b) == today]
    yesterday_bets = [b for b in all_bets if _bet_date(b) == today - timedelta(days=1)]
    week_bets = [b for b in all_bets if _bet_date(b) and _bet_date(b) >= today - timedelta(days=7)]
    month_bets = [b for b in all_bets if _bet_date(b) and _bet_date(b) >= today - timedelta(days=30)]

    # Daily breakdown for chart (last 14 days)
    daily = []
    for i in range(13, -1, -1):
        d = today - timedelta(days=i)
        day_bets = [b for b in all_bets if _bet_date(b) == d]
        stats = _calc_stats(day_bets)
        stats["date"] = d.isoformat()
        stats["label"] = d.strftime("%d %b")
        daily.append(stats)

    # Each period includes by_type, best/worst, running_pnl
    periods = {
        "today": _calc_period(today_bets),
        "yesterday": _calc_period(yesterday_bets),
        "week": _calc_period(week_bets),
        "month": _calc_period(month_bets),
        "all_time": _calc_period(all_bets),
    }

    return {
        "balance": balance,
        "initial_balance": initial,
        "current_stake": stake,
        "periods": periods,
        "daily": daily,
    }


@router.get("/api/betfair-tracker/balance")
async def bf_tracker_balance(request: Request):
    """Public read-only balance."""
    if not _bf_check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from punty.betting.queue import get_balance, calculate_stake, _get_setting, DEFAULT_INITIAL_BALANCE
    from sqlalchemy import select
    async with async_session() as db:
        balance = await get_balance(db)
        initial = float(await _get_setting(db, "betfair_initial_balance", str(DEFAULT_INITIAL_BALANCE)))
    return {"balance": balance, "initial_balance": initial, "current_stake": calculate_stake(balance, initial)}
