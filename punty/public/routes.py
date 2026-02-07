"""Public website routes - no authentication required."""

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_, or_

from punty.config import melb_today, melb_now, MELB_TZ
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race

router = APIRouter()

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)


async def get_next_race() -> dict:
    """Get the next upcoming race for countdown display."""
    async with async_session() as db:
        today = melb_today()
        now = melb_now()
        now_naive = now.replace(tzinfo=None)

        # Get today's selected meetings
        meetings_result = await db.execute(
            select(Meeting).where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
        )
        meetings = {m.id: m for m in meetings_result.scalars().all()}

        if not meetings:
            return {"has_next": False}

        # Get all races for today's meetings that haven't finished yet
        # Note: notin_ doesn't handle NULL correctly, so we need explicit OR for NULL values
        races_result = await db.execute(
            select(Race).where(
                and_(
                    Race.meeting_id.in_(list(meetings.keys())),
                    Race.start_time.isnot(None),
                    or_(
                        Race.results_status.is_(None),
                        Race.results_status.notin_(["Paying", "Closed", "Final"]),
                    ),
                )
            ).order_by(Race.start_time)
        )
        races = races_result.scalars().all()

        # Find the next race (first one with start_time > now)
        next_race = None
        for race in races:
            if race.start_time and race.start_time > now_naive:
                next_race = race
                break

        if not next_race:
            return {"has_next": False, "all_done": True}

        meeting = meetings.get(next_race.meeting_id)
        # Add timezone info for JavaScript
        start_time_aware = next_race.start_time.replace(tzinfo=MELB_TZ)

        return {
            "has_next": True,
            "venue": meeting.venue if meeting else "Unknown",
            "race_number": next_race.race_number,
            "race_name": next_race.name,
            "start_time_iso": start_time_aware.isoformat(),
            "start_time_formatted": next_race.start_time.strftime("%H:%M"),
        }


async def get_recent_wins_public(limit: int = 15) -> dict:
    """Get recent wins for public ticker with Punty celebrations."""
    from punty.results.celebrations import get_celebration

    async with async_session() as db:
        # Get recent settled wins, ordered by settled_at descending
        result = await db.execute(
            select(Pick, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                and_(
                    Pick.settled == True,
                    Pick.hit == True,
                    Pick.pnl > 0,  # Only profitable wins
                )
            )
            .order_by(Pick.settled_at.desc())
            .limit(limit)
        )
        rows = result.all()

        wins = []
        for pick, meeting in rows:
            # Calculate stake and return
            stake = pick.bet_stake or pick.exotic_stake or 1.0
            returned = stake + (pick.pnl or 0)

            # Build display name
            if pick.pick_type == "selection":
                display_name = f"{pick.horse_name or 'Runner'} R{pick.race_number}"
            elif pick.pick_type == "exotic":
                display_name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"
            elif pick.pick_type == "sequence":
                display_name = f"{pick.sequence_type or 'Sequence'}"
            elif pick.pick_type == "big3_multi":
                display_name = "Big 3 Multi"
            else:
                display_name = f"Win R{pick.race_number}"

            wins.append({
                "id": pick.id,
                "venue": meeting.venue,
                "display_name": display_name,
                "pick_type": pick.pick_type,
                "stake": round(stake, 2),
                "returned": round(returned, 2),
                "pnl": round(pick.pnl, 2),
                "celebration": get_celebration(pick.pnl, pick.pick_type),
            })

        return {"wins": wins}


async def get_winner_stats() -> dict:
    """Get winner statistics for today and all-time."""
    from punty.models.meeting import Race

    async with async_session() as db:
        today = melb_today()

        # Today's winners (selections that hit)
        today_result = await db.execute(
            select(func.count(Pick.id)).where(
                and_(
                    Pick.pick_type == "selection",
                    Pick.hit == True,
                    Pick.settled == True,
                    Pick.created_at >= today.isoformat(),
                )
            )
        )
        today_winners = today_result.scalar() or 0

        # Check if all races today are complete
        today_meetings_result = await db.execute(
            select(Meeting).where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                )
            )
        )
        today_meetings = today_meetings_result.scalars().all()
        meeting_ids = [m.id for m in today_meetings]

        # Get races for today's meetings
        all_races_complete = False
        if meeting_ids:
            races_result = await db.execute(
                select(Race).where(Race.meeting_id.in_(meeting_ids))
            )
            races = races_result.scalars().all()
            if races:
                # Check if all races have final results (Paying or Closed)
                complete_statuses = {"Paying", "Closed"}
                all_races_complete = all(
                    r.results_status in complete_statuses for r in races
                )

        # All-time winners
        alltime_result = await db.execute(
            select(func.count(Pick.id)).where(
                and_(
                    Pick.pick_type == "selection",
                    Pick.hit == True,
                    Pick.settled == True,
                )
            )
        )
        alltime_winners = alltime_result.scalar() or 0

        # All-time total collected (sum of returns from winning selections = pnl + stake)
        alltime_winnings_result = await db.execute(
            select(func.sum(Pick.pnl + Pick.bet_stake)).where(
                and_(
                    Pick.pick_type == "selection",
                    Pick.hit == True,
                    Pick.settled == True,
                    Pick.pnl > 0,
                )
            )
        )
        alltime_winnings = alltime_winnings_result.scalar() or 0.0

        # Get early mail content for today (sent to Twitter)
        early_mail_result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "early_mail",
                    Content.sent_to_twitter == True,
                    Content.meeting_id.in_(meeting_ids) if meeting_ids else False,
                )
            )
        )
        early_mail_content = early_mail_result.scalars().all()

        # Build list of today's tips with Twitter links
        todays_tips = []
        for content in early_mail_content:
            meeting = next((m for m in today_meetings if m.id == content.meeting_id), None)
            if meeting:
                todays_tips.append({
                    "venue": meeting.venue,
                    "twitter_url": f"https://twitter.com/PuntyAI",  # Link to profile, or specific tweet if stored
                })

        return {
            "today_winners": today_winners,
            "alltime_winners": alltime_winners,
            "alltime_winnings": round(alltime_winnings, 2),
            "todays_tips": todays_tips,
            "meetings_today": len(today_meetings),
            "all_races_complete": all_races_complete,
        }


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Public homepage."""
    stats = await get_winner_stats()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
        }
    )


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About us page."""
    return templates.TemplateResponse("about.html", {"request": request})


@router.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    """How it works page."""
    return templates.TemplateResponse("how-it-works.html", {"request": request})


@router.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    """Contact page."""
    return templates.TemplateResponse("contact.html", {"request": request})


@router.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):
    """Terms of service page."""
    return templates.TemplateResponse("terms.html", {"request": request})


@router.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    """Privacy policy page."""
    return templates.TemplateResponse("privacy.html", {"request": request})


async def get_tips_calendar(page: int = 1, per_page: int = 30) -> dict:
    """Get meetings with sent content, grouped by date for calendar view."""
    from sqlalchemy import desc
    from collections import defaultdict

    async with async_session() as db:
        # Get meetings that have sent early mail content
        result = await db.execute(
            select(Meeting, Content)
            .join(Content, Content.meeting_id == Meeting.id)
            .where(
                and_(
                    Content.content_type == "early_mail",
                    Content.status == "sent",
                )
            )
            .order_by(desc(Meeting.date), Meeting.venue)
        )
        rows = result.all()

        # Group by date
        meetings_by_date = defaultdict(list)
        seen_meetings = set()
        for meeting, content in rows:
            if meeting.id in seen_meetings:
                continue
            seen_meetings.add(meeting.id)
            date_key = meeting.date.isoformat() if meeting.date else "unknown"
            meetings_by_date[date_key].append({
                "id": meeting.id,
                "venue": meeting.venue,
                "date": meeting.date,
                "date_formatted": meeting.date.strftime("%a %d %b %Y") if meeting.date else "",
                "track_condition": meeting.track_condition,
                "slug": meeting.id,  # Use meeting_id as URL slug
            })

        # Convert to list sorted by date desc
        dates = sorted(meetings_by_date.keys(), reverse=True)
        total = len(dates)
        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        # Paginate dates
        offset = (page - 1) * per_page
        paginated_dates = dates[offset:offset + per_page]

        calendar = []
        for date_key in paginated_dates:
            meetings = meetings_by_date[date_key]
            if meetings:
                calendar.append({
                    "date": date_key,
                    "date_formatted": meetings[0]["date_formatted"],
                    "meetings": meetings,
                })

        return {
            "calendar": calendar,
            "page": page,
            "per_page": per_page,
            "total_dates": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }


async def get_meeting_tips(meeting_id: str) -> dict | None:
    """Get early mail and wrap-up content for a specific meeting."""
    async with async_session() as db:
        # Get meeting
        meeting_result = await db.execute(
            select(Meeting).where(Meeting.id == meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        if not meeting:
            return None

        # Get early mail (sent)
        early_mail_result = await db.execute(
            select(Content).where(
                and_(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "early_mail",
                    Content.status == "sent",
                )
            ).order_by(Content.created_at.desc())
        )
        early_mail = early_mail_result.scalar_one_or_none()

        # Get wrap-up (sent)
        wrapup_result = await db.execute(
            select(Content).where(
                and_(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "meeting_wrapup",
                    Content.status == "sent",
                )
            ).order_by(Content.created_at.desc())
        )
        wrapup = wrapup_result.scalar_one_or_none()

        # If no sent content at all, return None
        if not early_mail and not wrapup:
            return None

        return {
            "meeting": {
                "id": meeting.id,
                "venue": meeting.venue,
                "date": meeting.date.isoformat() if meeting.date else None,
                "date_formatted": meeting.date.strftime("%A, %d %B %Y") if meeting.date else "",
                "track_condition": meeting.track_condition,
                "weather": meeting.weather,
                "rail_position": meeting.rail_position,
            },
            "early_mail": {
                "content": early_mail.raw_content,
                "created_at": early_mail.created_at.isoformat() if early_mail.created_at else None,
            } if early_mail else None,
            "wrapup": {
                "content": wrapup.raw_content,
                "created_at": wrapup.created_at.isoformat() if wrapup.created_at else None,
            } if wrapup else None,
        }


@router.get("/tips", response_class=HTMLResponse)
async def tips_page(request: Request, page: int = 1):
    """Public tips calendar page showing meetings by date."""
    calendar_data = await get_tips_calendar(page=page, per_page=30)
    return templates.TemplateResponse(
        "tips.html",
        {
            "request": request,
            "calendar": calendar_data["calendar"],
            "page": calendar_data["page"],
            "total_pages": calendar_data["total_pages"],
            "total_dates": calendar_data["total_dates"],
            "has_next": calendar_data["has_next"],
            "has_prev": calendar_data["has_prev"],
        }
    )


@router.get("/tips/{meeting_id}", response_class=HTMLResponse)
async def meeting_tips_page(request: Request, meeting_id: str):
    """Public meeting detail page with early mail and wrap-up."""
    from fastapi import HTTPException

    data = await get_meeting_tips(meeting_id)
    if not data:
        raise HTTPException(status_code=404, detail="Meeting not found or no tips available")

    return templates.TemplateResponse(
        "meeting_tips.html",
        {
            "request": request,
            "meeting": data["meeting"],
            "early_mail": data["early_mail"],
            "wrapup": data["wrapup"],
        }
    )
