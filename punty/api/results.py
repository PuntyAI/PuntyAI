"""Results API — monitor control, manual checks, P&L summary."""

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

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
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        return {"error": "Monitor not initialized"}
    monitor.start()
    return {"status": "started"}


@router.post("/monitor/stop")
async def stop_monitor(request: Request):
    """Stop the results monitor."""
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        return {"error": "Monitor not initialized"}
    monitor.stop()
    return {"status": "stopped"}


@router.post("/{meeting_id}/check")
async def check_meeting(meeting_id: str, request: Request):
    """Manual one-shot results check for a meeting."""
    monitor = getattr(request.app.state, "results_monitor", None)
    if not monitor:
        return {"error": "Monitor not initialized"}
    try:
        await monitor.check_single_meeting(meeting_id)
        return {"status": "checked", "meeting_id": meeting_id}
    except Exception as e:
        logger.error(f"Manual check failed: {e}")
        return {"status": "error", "error": str(e)}


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
            return {"error": "Invalid date format, use YYYY-MM-DD"}
    else:
        from punty.config import melb_today
        target = melb_today()

    try:
        return await get_performance_summary(db, target)
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        return {"error": str(e)}


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
            return {"error": "Invalid end date format, use YYYY-MM-DD"}
    else:
        end_date = today

    if start:
        try:
            start_date = date_type.fromisoformat(start)
        except ValueError:
            return {"error": "Invalid start date format, use YYYY-MM-DD"}
    else:
        start_date = end_date - timedelta(days=7)

    try:
        days = await get_performance_history(db, start_date, end_date)
        return {"start": start_date.isoformat(), "end": end_date.isoformat(), "days": days}
    except Exception as e:
        logger.error(f"Performance history failed: {e}")
        return {"error": str(e)}


@router.get("/{meeting_id}/summary")
async def meeting_summary(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get picks vs results P&L summary for a meeting."""
    from punty.results.tracker import build_meeting_summary
    try:
        summary = await build_meeting_summary(db, meeting_id)
        return summary
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        return {"error": str(e)}


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
        return {"error": f"Meeting not found: {meeting_id}"}

    race_id = f"{meeting_id}-r{race_number}"
    race = await db.get(Race, race_id)
    if not race:
        return {"error": f"Race not found: {race_id}"}

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
        return {"error": f"Race not found: {race_id}"}

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
