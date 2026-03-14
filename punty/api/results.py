"""Results API — monitor control, manual checks, P&L summary."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import JSONResponse

from punty.models.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/monitor-status")
async def monitor_status(request: Request):
    """Get results monitor status."""
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        return {"running": False, "error": "Monitor not initialized"}
    return monitor.status()


@router.post("/monitor/start")
async def start_monitor(request: Request):
    """Start the results monitor."""
    if not request.session.get("user"):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    monitor.start()
    return {"status": "started"}


@router.post("/monitor/stop")
async def stop_monitor(request: Request):
    """Stop the results monitor."""
    if not request.session.get("user"):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    monitor.stop()
    return {"status": "stopped"}


@router.post("/{meeting_id}/check")
async def check_meeting(meeting_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Manual one-shot results check for a meeting."""
    from punty.models.meeting import Meeting
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting not found: {meeting_id}")
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    try:
        await monitor.check_single_meeting(meeting_id)
        return {"status": "checked", "meeting_id": meeting_id}
    except Exception as e:
        logger.error(f"Manual check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def performance_summary(
    date: str = None,
    db: AsyncSession = Depends(get_db),
):
    """Get picks P&L performance summary for a date."""
    from datetime import date as date_type
    from punty.results.picks import get_performance_summary

    if date:
        try:
            target = date_type.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")
    else:
        from punty.config import melb_today
        target = melb_today()

    try:
        return await get_performance_summary(db, target)
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/history")
async def performance_history(
    start: str = None,
    end: str = None,
    db: AsyncSession = Depends(get_db),
):
    """Get daily P&L history for a date range."""
    from datetime import date as date_type, timedelta
    from punty.results.picks import get_performance_history

    from punty.config import melb_today
    today = melb_today()
    if end:
        try:
            end_date = date_type.fromisoformat(end)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end date format, use YYYY-MM-DD")
    else:
        end_date = today

    if start:
        try:
            start_date = date_type.fromisoformat(start)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start date format, use YYYY-MM-DD")
    else:
        start_date = end_date - timedelta(days=7)

    try:
        days = await get_performance_history(db, start_date, end_date)
        return {"start": start_date.isoformat(), "end": end_date.isoformat(), "days": days}
    except Exception as e:
        logger.error(f"Performance history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{meeting_id}/summary")
async def meeting_summary(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get picks vs results P&L summary for a meeting."""
    from punty.models.meeting import Meeting
    from punty.results.tracker import build_meeting_summary
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting not found: {meeting_id}")
    try:
        summary = await build_meeting_summary(db, meeting_id)
        return summary
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{meeting_id}/race/{race_number}/sectionals")
async def scrape_race_sectionals(
    meeting_id: str,
    race_number: int,
    db: AsyncSession = Depends(get_db),
):
    """Manually scrape sectional times for a race.

    Sectional times show actual running positions at each checkpoint.
    These are typically available 10-15 minutes after a race finishes.
    """
    import json as _json
    from punty.models.meeting import Meeting, Race
    from punty.scrapers.racing_com import RacingComScraper

    # Get meeting
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting not found: {meeting_id}")

    race_id = f"{meeting_id}-r{race_number}"
    race = await db.get(Race, race_id)
    if not race:
        raise HTTPException(status_code=404, detail=f"Race not found: {race_id}")

    # Scrape sectionals
    scraper = RacingComScraper()
    try:
        sectional_data = await scraper.scrape_sectional_times(
            meeting.venue, meeting.date, race_number
        )
    finally:
        await scraper.close()

    if not sectional_data or not sectional_data.get("horses"):
        return {
            "status": "not_available",
            "message": "Sectional times not yet available for this race. Try again in a few minutes.",
        }

    # Store the data
    race.sectional_times = _json.dumps(sectional_data)
    race.has_sectionals = True
    await db.commit()

    return {
        "status": "success",
        "race_id": race_id,
        "horses_count": len(sectional_data.get("horses", [])),
        "data": sectional_data,
    }


@router.get("/{meeting_id}/race/{race_number}/sectionals")
async def get_race_sectionals(
    meeting_id: str,
    race_number: int,
    db: AsyncSession = Depends(get_db),
):
    """Get stored sectional times for a race."""
    import json as _json
    from punty.models.meeting import Race

    race_id = f"{meeting_id}-r{race_number}"
    race = await db.get(Race, race_id)
    if not race:
        raise HTTPException(status_code=404, detail=f"Race not found: {race_id}")

    if not race.sectional_times:
        return {
            "status": "not_available",
            "has_sectionals": race.has_sectionals,
            "message": "No sectional times stored for this race.",
        }

    try:
        data = _json.loads(race.sectional_times)
        return {
            "status": "available",
            "race_id": race_id,
            "data": data,
        }
    except (_json.JSONDecodeError, TypeError):
        return {"error": "Failed to parse stored sectional data"}


@router.post("/settle-past")
async def settle_past_races(db: AsyncSession = Depends(get_db)):
    """Settle all past races that have results but unsettled picks.

    This catches any picks that were missed by the results monitor,
    e.g. due to server restarts or network issues.
    """
    from sqlalchemy import select, and_, or_
    from punty.models.meeting import Meeting, Race
    from punty.models.pick import Pick
    from punty.results.picks import settle_picks_for_race
    from punty.config import melb_today

    today = melb_today()
    settled_total = 0
    errors = []

    # Find races with results (Paying/Closed) that have unsettled picks
    result = await db.execute(
        select(Race.meeting_id, Race.race_number, Race.id)
        .join(Pick, and_(
            Pick.meeting_id == Race.meeting_id,
            Pick.race_number == Race.race_number,
        ))
        .join(Meeting, Meeting.id == Race.meeting_id)
        .where(
            and_(
                Race.results_status.in_(["Paying", "Closed"]),
                Pick.settled == False,
                Meeting.date <= today,
            )
        )
        .distinct()
    )
    races_to_settle = result.all()

    for meeting_id, race_number, race_id in races_to_settle:
        try:
            count = await settle_picks_for_race(db, meeting_id, race_number)
            settled_total += count
            if count > 0:
                logger.info(f"Settled {count} picks for {meeting_id} R{race_number}")
                # Backfill form history for future lookups
                try:
                    from punty.results.monitor import _backfill_form_history
                    await _backfill_form_history(db, meeting_id, race_number)
                except Exception as e:
                    logger.warning(f"Failed to backfill form history for {meeting_id} R{race_number}: {e}")
        except Exception as e:
            errors.append(f"{meeting_id} R{race_number}: {e}")
            logger.error(f"Failed to settle {meeting_id} R{race_number}: {e}")

    return {
        "status": "done",
        "races_checked": len(races_to_settle),
        "picks_settled": settled_total,
        "errors": errors,
    }


@router.post("/{meeting_id}/race/{race_number}/dividend")
async def add_manual_dividend(
    meeting_id: str,
    race_number: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Manually add or update an exotic dividend for a race.

    Merges into existing exotic_results JSON, then re-settles affected picks.
    Accepts JSON body: {"exotic_type": "quinella", "dividend": 24.50}
    Optional: {"runners": "1-3"} for result combo display.
    """
    if not request.session.get("user"):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    import json
    from punty.models.meeting import Meeting, Race
    from punty.results.picks import settle_picks_for_race

    body = await request.json()
    exotic_type = body.get("exotic_type", "").strip().lower()
    dividend = body.get("dividend")
    runners_combo = body.get("runners", "")

    if not exotic_type:
        raise HTTPException(status_code=400, detail="exotic_type is required")
    if dividend is None:
        raise HTTPException(status_code=400, detail="dividend is required")
    try:
        dividend = float(dividend)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="dividend must be a number")
    if dividend <= 0:
        raise HTTPException(status_code=400, detail="dividend must be positive")

    # Load race
    race_id = f"{meeting_id}-r{race_number}"
    race = await db.get(Race, race_id)
    if not race:
        raise HTTPException(status_code=404, detail=f"Race not found: {race_id}")

    # Merge into existing exotic_results
    existing = {}
    if race.exotic_results:
        try:
            existing = json.loads(race.exotic_results)
        except (json.JSONDecodeError, TypeError):
            existing = {}

    old_value = existing.get(exotic_type)
    existing[exotic_type] = dividend
    race.exotic_results = json.dumps(existing)
    await db.commit()

    logger.info(
        f"Manual dividend: {meeting_id} R{race_number} {exotic_type}=${dividend:.2f} "
        f"(was: {old_value})"
    )

    # Re-settle picks for this race
    settled_count = 0
    try:
        # First unsettled any picks that might need re-settlement
        from punty.models.pick import Pick
        from sqlalchemy import select, and_

        # Find picks for this race that are exotic/sequence type
        picks_result = await db.execute(
            select(Pick).where(
                and_(
                    Pick.meeting_id == meeting_id,
                    Pick.race_number == race_number,
                )
            )
        )
        picks = picks_result.scalars().all()

        # For exotic picks on this race, or sequence picks with last leg on this race,
        # mark unsettled so they get re-processed
        for pick in picks:
            if pick.pick_type in ("exotic", "sequence") and pick.settled:
                if pick.pnl == 0.0 or pick.pnl is None:
                    # Only re-settle if currently at $0 (missing dividend)
                    pick.settled = False
                    pick.pnl = None
                    pick.hit = None
        await db.commit()

        # Also check sequences where the last leg is this race
        seq_result = await db.execute(
            select(Pick).where(
                and_(
                    Pick.meeting_id == meeting_id,
                    Pick.pick_type == "sequence",
                )
            )
        )
        seq_picks = seq_result.scalars().all()
        for sp in seq_picks:
            if sp.sequence_legs:
                try:
                    legs = json.loads(sp.sequence_legs)
                    start = sp.sequence_start_race or 1
                    last_leg_race = start + len(legs) - 1
                    if last_leg_race == race_number and sp.settled and (sp.pnl == 0.0 or sp.pnl is None):
                        sp.settled = False
                        sp.pnl = None
                        sp.hit = None
                except (json.JSONDecodeError, TypeError):
                    pass
        await db.commit()

        settled_count = await settle_picks_for_race(db, meeting_id, race_number)
    except Exception as e:
        logger.error(f"Re-settlement after manual dividend failed: {e}")

    return {
        "status": "ok",
        "race_id": race_id,
        "exotic_type": exotic_type,
        "dividend": dividend,
        "old_value": old_value,
        "picks_resettled": settled_count,
    }


@router.get("/unsettled")
async def get_unsettled_picks(db: AsyncSession = Depends(get_db)):
    """Get all unsettled picks grouped by meeting, for the admin settlement dashboard."""
    from sqlalchemy import select, and_
    from punty.models.pick import Pick
    from punty.models.meeting import Meeting, Race

    result = await db.execute(
        select(Pick, Meeting.venue, Meeting.date, Race.results_status)
        .join(Meeting, Meeting.id == Pick.meeting_id)
        .outerjoin(Race, and_(
            Race.meeting_id == Pick.meeting_id,
            Race.race_number == Pick.race_number,
        ))
        .where(Pick.settled == False)
        .order_by(Meeting.date.desc(), Pick.meeting_id, Pick.race_number)
    )
    rows = result.all()

    unsettled = []
    for pick, venue, meet_date, results_status in rows:
        unsettled.append({
            "pick_id": pick.id,
            "meeting_id": pick.meeting_id,
            "venue": venue,
            "date": meet_date.isoformat() if meet_date else None,
            "race_number": pick.race_number,
            "pick_type": pick.pick_type,
            "exotic_type": pick.exotic_type,
            "sequence_type": pick.sequence_type,
            "horse_name": pick.horse_name,
            "bet_stake": float(pick.bet_stake) if pick.bet_stake else None,
            "exotic_stake": float(pick.exotic_stake) if pick.exotic_stake else None,
            "results_status": results_status,
        })

    return {"unsettled": unsettled, "count": len(unsettled)}


@router.get("/wins/recent")
async def get_recent_wins_api(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Get recent winning picks with Punty celebration phrases."""
    from punty.results.picks import get_recent_wins
    try:
        wins = await get_recent_wins(db, limit=limit)
        return {"wins": wins}
    except Exception as e:
        logger.error(f"Recent wins failed: {e}")
        return {"error": str(e), "wins": []}


@router.get("/wins/stats")
async def get_all_time_stats_api(db: AsyncSession = Depends(get_db)):
    """Get all-time win statistics for dashboard hero stats."""
    from punty.results.picks import get_all_time_stats
    try:
        stats = await get_all_time_stats(db)
        return stats
    except Exception as e:
        logger.error(f"All-time stats failed: {e}")
        return {"today_winners": 0, "total_winners": 0, "collected": 0}


@router.get("/monitoring/digest")
async def monitoring_digest(db: AsyncSession = Depends(get_db)):
    """Get today's performance digest (same data sent via Telegram at 23:00)."""
    from punty.config import melb_today
    from punty.monitoring.performance import (
        compute_daily_digest,
        compute_rolling_comparison,
        check_regressions,
        check_calibration_drift,
    )
    try:
        today = melb_today()
        digest = await compute_daily_digest(db, today)
        comparison = await compute_rolling_comparison(db)
        regressions = await check_regressions(db)
        calibration = await check_calibration_drift(db)
        return {
            "digest": digest,
            "comparison": comparison,
            "regressions": regressions,
            "calibration": calibration,
        }
    except Exception as e:
        logger.error(f"Monitoring digest failed: {e}")
        return {"error": str(e)}
