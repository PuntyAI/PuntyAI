"""Web routes for the dashboard."""

from pathlib import Path
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from punty.models.database import get_db
from punty.models.meeting import Meeting, Race
from punty.models.content import Content, ContentStatus, ScheduledJob

router = APIRouter()

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    """Main dashboard page."""
    # Get today's meetings
    from datetime import date

    today = date.today()
    result = await db.execute(
        select(Meeting).where(Meeting.date == today).options(selectinload(Meeting.races)).order_by(Meeting.venue)
    )
    todays_meetings = result.scalars().all()

    # Get pending reviews count
    result = await db.execute(
        select(func.count(Content.id)).where(Content.status == ContentStatus.PENDING_REVIEW.value)
    )
    pending_reviews = result.scalar() or 0

    # Get recent content
    result = await db.execute(
        select(Content).order_by(Content.created_at.desc()).limit(10)
    )
    recent_content = result.scalars().all()

    # Get active jobs
    result = await db.execute(select(ScheduledJob).where(ScheduledJob.enabled == True))
    active_jobs = result.scalars().all()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "todays_meetings": todays_meetings,
            "pending_reviews": pending_reviews,
            "recent_content": recent_content,
            "active_jobs": active_jobs,
        },
    )


@router.get("/meets", response_class=HTMLResponse)
async def meets_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Race meetings management page — shows today's meetings first."""
    from datetime import date as date_type

    today = date_type.today()

    # Today's meetings first
    result = await db.execute(
        select(Meeting).where(Meeting.date == today).options(selectinload(Meeting.races)).order_by(Meeting.venue)
    )
    todays = result.scalars().all()

    # If no today meetings, fall back to all recent
    if todays:
        meetings = todays
    else:
        result = await db.execute(select(Meeting).options(selectinload(Meeting.races)).order_by(Meeting.date.desc()).limit(50))
        meetings = result.scalars().all()

    return templates.TemplateResponse(
        "meets.html",
        {
            "request": request,
            "meetings": meetings,
        },
    )


@router.get("/meets/{meeting_id}", response_class=HTMLResponse)
async def meeting_detail(
    meeting_id: str, request: Request, db: AsyncSession = Depends(get_db)
):
    """Meeting detail page with races."""

    result = await db.execute(
        select(Meeting)
        .where(Meeting.id == meeting_id)
        .options(selectinload(Meeting.races).selectinload(Race.runners))
    )
    meeting = result.scalar_one_or_none()

    if not meeting:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Meeting not found"},
            status_code=404,
        )

    # Get content for this meeting
    result = await db.execute(
        select(Content).where(Content.meeting_id == meeting_id).order_by(Content.created_at.desc())
    )
    content_items = result.scalars().all()

    return templates.TemplateResponse(
        "meeting_detail.html",
        {
            "request": request,
            "meeting": meeting,
            "content_items": content_items,
        },
    )


@router.get("/content", response_class=HTMLResponse)
async def content_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Content management page."""
    result = await db.execute(select(Content).order_by(Content.created_at.desc()).limit(100))
    content_items = result.scalars().all()

    # Group by status
    status_counts = {}
    for status in ContentStatus:
        result = await db.execute(
            select(func.count(Content.id)).where(Content.status == status.value)
        )
        status_counts[status.value] = result.scalar() or 0

    return templates.TemplateResponse(
        "content.html",
        {
            "request": request,
            "content_items": content_items,
            "status_counts": status_counts,
        },
    )


@router.get("/review", response_class=HTMLResponse)
async def review_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Content review queue page."""
    result = await db.execute(
        select(Content)
        .where(Content.status == ContentStatus.PENDING_REVIEW.value)
        .order_by(Content.created_at.asc())
    )
    review_items = result.scalars().all()

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "review_items": review_items,
        },
    )


@router.get("/review/{content_id}", response_class=HTMLResponse)
async def review_detail(
    content_id: str, request: Request, db: AsyncSession = Depends(get_db)
):
    """Single content review page."""

    result = await db.execute(
        select(Content)
        .where(Content.id == content_id)
        .options(selectinload(Content.meeting), selectinload(Content.context_snapshot))
    )
    content = result.scalar_one_or_none()

    if not content:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Content not found"},
            status_code=404,
        )

    return templates.TemplateResponse(
        "review_detail.html",
        {
            "request": request,
            "content": content,
        },
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Settings page for API keys, scheduler config, prompts."""
    from punty.models.settings import AnalysisWeights, AppSettings
    from pathlib import Path

    # Get all jobs
    result = await db.execute(select(ScheduledJob).order_by(ScheduledJob.job_type))
    jobs = result.scalars().all()

    # Get analysis weights
    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    weights_record = result.scalar_one_or_none()

    if weights_record:
        weights = weights_record.weights
    else:
        weights = AnalysisWeights.DEFAULT_WEIGHTS

    # Get all app settings
    result = await db.execute(select(AppSettings))
    settings_records = result.scalars().all()
    settings = {s.key: s.to_dict() for s in settings_records}

    # Add defaults for any missing
    for key, default in AppSettings.DEFAULTS.items():
        if key not in settings:
            settings[key] = {
                "key": key,
                "value": default["value"],
                "description": default["description"],
            }

    # Load personality prompt
    personality_path = Path(__file__).parent.parent.parent / "prompts" / "personality.md"
    personality_prompt = ""
    if personality_path.exists():
        personality_prompt = personality_path.read_text(encoding="utf-8")

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "jobs": jobs,
            "weights": weights,
            "weight_labels": AnalysisWeights.WEIGHT_LABELS,
            "weight_options": AnalysisWeights.WEIGHT_OPTIONS,
            "settings": settings,
            "personality_prompt": personality_prompt,
        },
    )
