"""API endpoints for race meetings."""

import json

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, or_
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


# =============================================================================
# BULK ENDPOINTS - Must be defined BEFORE /{meeting_id} routes to avoid
# FastAPI matching "bulk" as a meeting_id
# =============================================================================

@router.get("/bulk/scrape-stream")
async def bulk_scrape_stream():
    """SSE stream for scraping all selected meetings sequentially."""
    from punty.scrapers.orchestrator import scrape_meeting_full_stream
    from punty.models.database import async_session

    logger = logging.getLogger(__name__)

    async def event_generator():
        try:
            async with async_session() as db:
                # Get selected meetings
                today = melb_today()
                logger.info(f"Bulk scrape: querying for date={today}")
                result = await db.execute(
                    select(Meeting).where(
                        Meeting.date == today,
                        Meeting.selected == True,
                        # Include NULL meeting_type (legacy) and non-trial meetings
                        or_(Meeting.meeting_type == None, Meeting.meeting_type != "trial")
                    ).order_by(Meeting.venue)
                )
                meetings = result.scalars().all()
                # Store meeting IDs and venues to avoid lazy loading issues
                meeting_list = [(m.id, m.venue) for m in meetings]
                logger.info(f"Bulk scrape: found {len(meeting_list)} meetings: {[v for _, v in meeting_list]}")

                if not meeting_list:
                    yield f"data: {json.dumps({'step': 0, 'total': 0, 'label': 'No meetings selected', 'status': 'complete'})}\n\n"
                    return

                total_meetings = len(meeting_list)
                yield f"data: {json.dumps({'step': 0, 'total': total_meetings, 'label': f'Starting scrape of {total_meetings} meetings...', 'status': 'running'})}\n\n"

                for idx, (meeting_id, venue) in enumerate(meeting_list):
                    meeting_num = idx + 1
                    logger.info(f"Bulk scrape: starting {venue} ({meeting_num}/{total_meetings})")
                    yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': venue, 'label': f'Scraping {venue} ({meeting_num}/{total_meetings})...', 'status': 'running'})}\n\n"

                    try:
                        errors = []
                        async for event in scrape_meeting_full_stream(meeting_id, db):
                            # Forward sub-events with meeting context
                            event['meeting'] = venue
                            event['meeting_num'] = meeting_num
                            event['total_meetings'] = total_meetings
                            yield f"data: {json.dumps(event)}\n\n"
                            if event.get('status') == 'error':
                                errors.append(event.get('label', 'Unknown error'))

                        if errors:
                            yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': venue, 'label': f'{venue}: completed with errors', 'status': 'done', 'errors': errors})}\n\n"
                        else:
                            yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': venue, 'label': f'{venue}: complete', 'status': 'done'})}\n\n"

                    except Exception as e:
                        logger.error(f"Bulk scrape error for {venue}: {e}")
                        yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': venue, 'label': f'{venue}: error - {str(e)}', 'status': 'error'})}\n\n"

                    # Brief delay between meetings to avoid rate limiting
                    import asyncio
                    await asyncio.sleep(2)

                yield f"data: {json.dumps({'step': total_meetings, 'total': total_meetings, 'label': f'All {total_meetings} meetings scraped', 'status': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Bulk scrape fatal error: {e}")
            yield f"data: {json.dumps({'step': 0, 'total': 0, 'label': f'Fatal error: {str(e)}', 'status': 'error'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/bulk/speed-maps-stream")
async def bulk_speed_maps_stream():
    """SSE stream for fetching speed maps for all selected meetings."""
    from punty.scrapers.orchestrator import scrape_speed_maps_stream
    from punty.models.database import async_session

    async def event_generator():
        async with async_session() as db:
            today = melb_today()
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                    or_(Meeting.meeting_type == None, Meeting.meeting_type != "trial")
                ).order_by(Meeting.venue)
            )
            meetings = result.scalars().all()

            if not meetings:
                yield f"data: {json.dumps({'step': 0, 'total': 0, 'label': 'No meetings selected', 'status': 'complete'})}\n\n"
                return

            total_meetings = len(meetings)
            yield f"data: {json.dumps({'step': 0, 'total': total_meetings, 'label': f'Fetching speed maps for {total_meetings} meetings...', 'status': 'running'})}\n\n"

            for idx, meeting in enumerate(meetings):
                meeting_num = idx + 1
                yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': meeting.venue, 'label': f'Fetching {meeting.venue} ({meeting_num}/{total_meetings})...', 'status': 'running'})}\n\n"

                try:
                    async for event in scrape_speed_maps_stream(meeting.id, db):
                        event['meeting'] = meeting.venue
                        event['meeting_num'] = meeting_num
                        event['total_meetings'] = total_meetings
                        yield f"data: {json.dumps(event)}\n\n"

                    yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': meeting.venue, 'label': f'{meeting.venue}: complete', 'status': 'done'})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': meeting.venue, 'label': f'{meeting.venue}: error - {str(e)}', 'status': 'error'})}\n\n"

                import asyncio
                await asyncio.sleep(2)

            yield f"data: {json.dumps({'step': total_meetings, 'total': total_meetings, 'label': f'Speed maps complete for {total_meetings} meetings', 'status': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/bulk/generate-early-mail-stream")
async def bulk_generate_early_mail_stream():
    """SSE stream for generating early mail for all selected meetings."""
    from punty.ai.generator import generate_content_stream
    from punty.models.database import async_session

    async def event_generator():
        async with async_session() as db:
            today = melb_today()
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                    or_(Meeting.meeting_type == None, Meeting.meeting_type != "trial")
                ).order_by(Meeting.venue)
            )
            meetings = result.scalars().all()

            if not meetings:
                yield f"data: {json.dumps({'step': 0, 'total': 0, 'label': 'No meetings selected', 'status': 'complete'})}\n\n"
                return

            total_meetings = len(meetings)
            yield f"data: {json.dumps({'step': 0, 'total': total_meetings, 'label': f'Generating early mail for {total_meetings} meetings...', 'status': 'running'})}\n\n"

            for idx, meeting in enumerate(meetings):
                meeting_num = idx + 1
                yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': meeting.venue, 'label': f'Generating {meeting.venue} ({meeting_num}/{total_meetings})...', 'status': 'running'})}\n\n"

                try:
                    async for event in generate_content_stream(meeting.id, "early_mail", db):
                        event['meeting'] = meeting.venue
                        event['meeting_num'] = meeting_num
                        event['total_meetings'] = total_meetings
                        yield f"data: {json.dumps(event)}\n\n"

                    yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': meeting.venue, 'label': f'{meeting.venue}: complete', 'status': 'done'})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'step': meeting_num, 'total': total_meetings, 'meeting': meeting.venue, 'label': f'{meeting.venue}: error - {str(e)}', 'status': 'error'})}\n\n"

                import asyncio
                await asyncio.sleep(1)

            yield f"data: {json.dumps({'step': total_meetings, 'total': total_meetings, 'label': f'Early mail generated for {total_meetings} meetings', 'status': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =============================================================================
# MEETING-SPECIFIC ENDPOINTS - /{meeting_id} routes come AFTER /bulk/ routes
# =============================================================================

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
