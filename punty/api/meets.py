"""API endpoints for race meetings."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.database import get_db

router = APIRouter()


@router.get("/")
async def list_meetings(db: AsyncSession = Depends(get_db)):
    """List all race meetings."""
    from punty.models.meeting import Meeting
    from sqlalchemy import select

    result = await db.execute(select(Meeting).order_by(Meeting.date.desc()))
    meetings = result.scalars().all()
    return [m.to_dict() for m in meetings]


@router.get("/{meeting_id}")
async def get_meeting(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific meeting with races and runners."""
    from punty.models.meeting import Meeting
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Meeting)
        .where(Meeting.id == meeting_id)
        .options(selectinload(Meeting.races))
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting.to_dict(include_races=True)


@router.post("/scrape")
async def scrape_meeting(venue: str, date: str, db: AsyncSession = Depends(get_db)):
    """Trigger scraping for a race meeting."""
    # Will be implemented with scrapers
    return {"status": "queued", "venue": venue, "date": date}
