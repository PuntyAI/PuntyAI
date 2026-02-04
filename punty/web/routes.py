"""Web routes for the dashboard."""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from punty.config import melb_now
from punty.models.database import get_db
from punty.models.meeting import Meeting, Race
from punty.models.pick import Pick
from punty.models.content import Content, ContentStatus, ScheduledJob


router = APIRouter()

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else {}

from punty.config import MELB_TZ
from datetime import timezone

def _melb(dt, fmt='%H:%M'):
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MELB_TZ).strftime(fmt)

def _melb_iso(dt):
    """Convert naive Melbourne datetime to ISO format with timezone for JavaScript."""
    if dt is None:
        return ''
    # The datetime is stored as naive Melbourne local time
    # Attach Melbourne timezone then convert to ISO format
    melb_dt = dt.replace(tzinfo=MELB_TZ)
    return melb_dt.isoformat()

templates.env.filters["melb"] = _melb
templates.env.filters["melb_iso"] = _melb_iso


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    """Main dashboard page."""
    today = melb_now().date()
    now = melb_now()
    result = await db.execute(
        select(Meeting).where(
            Meeting.date == today,
            Meeting.meeting_type.in_(["race", None]),
        ).options(selectinload(Meeting.races)).order_by(Meeting.venue)
    )
    todays_meetings = result.scalars().all()

    # Calculate race progress for each meeting
    meeting_progress = {}
    now_naive = now.replace(tzinfo=None)  # For comparing with naive DB times
    for meeting in todays_meetings:
        races = meeting.races or []
        total_races = len(races)
        if total_races == 0:
            meeting_progress[meeting.id] = {"status": "no_races", "label": "No races", "selected": meeting.selected or False}
            continue

        # Count completed races (Paying or Closed status)
        completed = sum(1 for r in races if r.results_status in ("Paying", "Closed", "Final"))

        # Get first race start time (start_time is naive Melbourne time in DB)
        races_with_time = [r for r in races if r.start_time]
        first_race = min(races_with_time, key=lambda r: r.start_time) if races_with_time else None
        first_start = first_race.start_time if first_race else None

        progress = {"selected": meeting.selected or False}
        if completed == total_races:
            progress.update({"status": "completed", "label": "Completed"})
        elif completed > 0:
            remaining = total_races - completed
            progress.update({"status": "in_progress", "label": f"{remaining} to go", "completed": completed, "total": total_races})
        elif first_start and first_start > now_naive:
            # Races haven't started - show countdown (attach tz for JS)
            first_start_aware = first_start.replace(tzinfo=MELB_TZ)
            progress.update({"status": "not_started", "label": "Starts soon", "first_race_iso": first_start_aware.isoformat()})
        else:
            progress.update({"status": "in_progress", "label": f"{total_races} to go", "completed": 0, "total": total_races})
        meeting_progress[meeting.id] = progress

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

    # Get results monitor status
    monitor = getattr(request.app.state, "results_monitor", None)
    monitor_status = monitor.status() if monitor else {"running": False}

    # Get performance summary for today + all-time cumulative P&L
    performance = None
    cumulative_pnl = []
    try:
        from punty.results.picks import get_performance_summary, get_cumulative_pnl
        performance = await get_performance_summary(db, today)
        cumulative_pnl = await get_cumulative_pnl(db)
    except Exception:
        pass

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "now": melb_now,
            "todays_meetings": todays_meetings,
            "meeting_progress": meeting_progress,
            "pending_reviews": pending_reviews,
            "recent_content": recent_content,
            "active_jobs": active_jobs,
            "monitor_status": monitor_status,
            "performance": performance,
            "cumulative_pnl": cumulative_pnl,
        },
    )


@router.get("/meets", response_class=HTMLResponse)
async def meets_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Race meetings management page — shows today's meetings first."""
    today = melb_now().date()
    now = melb_now()
    now_naive = now.replace(tzinfo=None)

    # Today's meetings first (exclude trials/jumpouts)
    result = await db.execute(
        select(Meeting).where(
            Meeting.date == today,
            Meeting.meeting_type.in_(["race", None]),
        ).options(selectinload(Meeting.races)).order_by(Meeting.venue)
    )
    todays = result.scalars().all()

    # If no today meetings, fall back to all recent
    if todays:
        meetings = todays
    else:
        result = await db.execute(select(Meeting).options(selectinload(Meeting.races)).order_by(Meeting.date.desc()).limit(50))
        meetings = result.scalars().all()

    # Calculate race progress for each meeting
    meeting_progress = {}
    for meeting in meetings:
        races = meeting.races or []
        total_races = len(races)
        if total_races == 0:
            meeting_progress[meeting.id] = {"status": "no_races", "label": "No races"}
            continue

        # Count completed races (Paying or Closed status)
        completed = sum(1 for r in races if r.results_status in ("Paying", "Closed", "Final"))

        # Get first race start time
        races_with_time = [r for r in races if r.start_time]
        first_race = min(races_with_time, key=lambda r: r.start_time) if races_with_time else None
        first_start = first_race.start_time if first_race else None

        if completed == total_races:
            meeting_progress[meeting.id] = {"status": "completed", "label": "Completed"}
        elif completed > 0:
            remaining = total_races - completed
            meeting_progress[meeting.id] = {"status": "in_progress", "label": f"{remaining} to go"}
        elif first_start and first_start > now_naive:
            first_start_aware = first_start.replace(tzinfo=MELB_TZ)
            meeting_progress[meeting.id] = {"status": "not_started", "label": "Starts soon", "first_race_iso": first_start_aware.isoformat()}
        else:
            meeting_progress[meeting.id] = {"status": "in_progress", "label": f"{total_races} to go"}

    return templates.TemplateResponse(
        "meets.html",
        {
            "request": request,
            "meetings": meetings,
            "meeting_progress": meeting_progress,
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

    # Get all meetings for the same date (for the meeting switcher dropdown)
    all_meetings_result = await db.execute(
        select(Meeting).where(
            Meeting.date == meeting.date,
            Meeting.meeting_type.in_(["race", None]),
        ).order_by(Meeting.venue)
    )
    all_meetings = all_meetings_result.scalars().all()

    # Get content for this meeting
    result = await db.execute(
        select(Content).where(Content.meeting_id == meeting_id).order_by(Content.created_at.desc())
    )
    content_items = result.scalars().all()

    # Load picks for this meeting, grouped by race number
    pick_result = await db.execute(
        select(Pick).where(Pick.meeting_id == meeting_id).order_by(Pick.tip_rank)
    )
    all_picks = pick_result.scalars().all()
    picks_by_race = {}
    meeting_picks = []  # big3_multi, sequences
    sequences_by_race = {}  # {race_num: [{pick, leg_index, saddlecloths}]}
    for p in all_picks:
        if p.pick_type in ("big3", "big3_multi"):
            meeting_picks.append(p)
        elif p.pick_type == "sequence":
            meeting_picks.append(p)
            # Map each leg to its race number
            if p.sequence_start_race and p.sequence_legs:
                legs = json.loads(p.sequence_legs)
                for i, leg in enumerate(legs):
                    race_num = p.sequence_start_race + i
                    sequences_by_race.setdefault(race_num, []).append({
                        "pick": p,
                        "leg_index": i,
                        "saddlecloths": leg,
                        "label": f"{p.sequence_type or 'Seq'} ({p.sequence_variant or ''})" if p.sequence_variant else (p.sequence_type or "Sequence"),
                    })
        elif p.race_number is not None:
            picks_by_race.setdefault(p.race_number, []).append(p)

    # Check if race previews are enabled
    from punty.models.settings import AppSettings
    setting_result = await db.execute(
        select(AppSettings).where(AppSettings.key == "enable_race_previews")
    )
    setting = setting_result.scalar_one_or_none()
    race_previews_enabled = setting and setting.value == "true"

    return templates.TemplateResponse(
        "meeting_detail.html",
        {
            "request": request,
            "meeting": meeting,
            "all_meetings": all_meetings,
            "content_items": content_items,
            "picks_by_race": picks_by_race,
            "meeting_picks": meeting_picks,
            "sequences_by_race": sequences_by_race,
            "race_previews_enabled": race_previews_enabled,
        },
    )


@router.get("/content", response_class=HTMLResponse)
async def content_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Content management page."""
    result = await db.execute(select(Content).order_by(Content.created_at.desc()).limit(100))
    content_items = result.scalars().all()

    # Group by status — single query instead of N queries
    result = await db.execute(
        select(Content.status, func.count(Content.id))
        .group_by(Content.status)
    )
    status_counts = {status.value: 0 for status in ContentStatus}
    for row in result.all():
        status_counts[row[0]] = row[1]

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

    # Get all content items from the same date for prev/next navigation
    today = melb_now().date()
    all_content_result = await db.execute(
        select(Content)
        .where(Content.meeting_id.like(f"%-{today.isoformat()}%"))
        .order_by(Content.meeting_id, Content.created_at.desc())
    )
    all_content = all_content_result.scalars().all()

    # Find prev/next content
    prev_content = None
    next_content = None
    for i, c in enumerate(all_content):
        if c.id == content_id:
            if i > 0:
                prev_content = all_content[i - 1]
            if i < len(all_content) - 1:
                next_content = all_content[i + 1]
            break

    return templates.TemplateResponse(
        "review_detail.html",
        {
            "request": request,
            "content": content,
            "prev_content": prev_content,
            "next_content": next_content,
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

    # Build API key status for template
    from punty.models.settings import get_api_key
    from punty.config import settings as app_settings
    openai_key = await get_api_key(db, "openai_api_key", app_settings.openai_api_key)
    twitter_key = await get_api_key(db, "twitter_api_key", app_settings.twitter_api_key)
    whatsapp_token = await get_api_key(db, "whatsapp_api_token", app_settings.whatsapp_api_token)
    # Only pass masked values to template — never expose full keys in HTML
    api_key_status = {
        "openai": ("..." + openai_key[-4:]) if openai_key else "",
        "twitter": ("..." + twitter_key[-4:]) if twitter_key else "",
        "whatsapp": ("..." + whatsapp_token[-4:]) if whatsapp_token else "",
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
            "api_key_status": api_key_status,
        },
    )
