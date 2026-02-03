"""API endpoints for race meetings."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from punty.config import melb_today
from punty.models.database import get_db
from punty.models.meeting import Meeting, Race

router = APIRouter()


@router.get("/")
async def list_meetings(db: AsyncSession = Depends(get_db)):
    """List all race meetings."""
    result = await db.execute(select(Meeting).order_by(Meeting.date.desc()))
    meetings = result.scalars().all()
    return [m.to_dict() for m in meetings]


@router.get("/today")
async def list_today(db: AsyncSession = Depends(get_db)):
    """List today's meetings."""
    today = melb_today()
    result = await db.execute(
        select(Meeting).where(Meeting.date == today).order_by(Meeting.venue)
    )
    meetings = result.scalars().all()
    return [m.to_dict() for m in meetings]


@router.get("/selected")
async def list_selected(db: AsyncSession = Depends(get_db)):
    """List only selected meetings for today."""
    today = melb_today()
    result = await db.execute(
        select(Meeting).where(Meeting.date == today, Meeting.selected == True).order_by(Meeting.venue)
    )
    meetings = result.scalars().all()
    return [m.to_dict() for m in meetings]


@router.get("/{meeting_id}")
async def get_meeting(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific meeting with races and runners."""
    result = await db.execute(
        select(Meeting)
        .where(Meeting.id == meeting_id)
        .options(selectinload(Meeting.races).selectinload(Race.runners))
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting.to_dict(include_races=True)


@router.post("/scrape-calendar")
async def scrape_calendar_endpoint(db: AsyncSession = Depends(get_db)):
    """Trigger calendar scrape for today's meetings."""
    from punty.scrapers.orchestrator import scrape_calendar

    meetings = await scrape_calendar(db)
    return {"status": "ok", "count": len(meetings), "meetings": meetings}


@router.put("/{meeting_id}/select")
async def toggle_select(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Toggle meeting selection on/off."""
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting.selected = not meeting.selected
    await db.commit()
    return {"id": meeting_id, "selected": meeting.selected}


@router.post("/{meeting_id}/scrape-full")
async def scrape_full_endpoint(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Run all scrapers for a meeting."""
    from punty.scrapers.orchestrator import scrape_meeting_full

    try:
        result = await scrape_meeting_full(meeting_id, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{meeting_id}/scrape-stream")
async def scrape_stream_endpoint(meeting_id: str):
    """SSE stream of scrape progress for a meeting."""
    from punty.scrapers.orchestrator import scrape_meeting_full_stream
    from punty.models.database import async_session

    async def event_generator():
        # Create session inside generator to avoid FastAPI closing it before streaming starts
        async with async_session() as db:
            async for event in scrape_meeting_full_stream(meeting_id, db):
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{meeting_id}/refresh-odds")
async def refresh_odds_endpoint(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Quick odds/scratchings refresh for a meeting."""
    from punty.scrapers.orchestrator import refresh_odds

    try:
        result = await refresh_odds(meeting_id, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{meeting_id}/speed-maps-stream")
async def speed_maps_stream_endpoint(meeting_id: str):
    """SSE stream of speed map scrape progress for a meeting."""
    from punty.scrapers.orchestrator import scrape_speed_maps_stream
    from punty.models.database import async_session

    async def event_generator():
        # Create session inside generator to avoid FastAPI closing it before streaming starts
        async with async_session() as db:
            async for event in scrape_speed_maps_stream(meeting_id, db):
                yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/scrape")
async def scrape_meeting(venue: str, date: str, db: AsyncSession = Depends(get_db)):
    """Trigger scraping for a race meeting (legacy endpoint)."""
    return {"status": "queued", "venue": venue, "date": date}
