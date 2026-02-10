"""Public website routes - no authentication required."""

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_, or_

from punty.config import melb_today, melb_now, MELB_TZ
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race
from punty.models.live_update import LiveUpdate

router = APIRouter()

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)

# Static files directory
static_dir = Path(__file__).parent / "static"


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

        # Today's winners (all pick types that hit)
        today_result = await db.execute(
            select(func.count(Pick.id))
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                    Meeting.date == today,
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

        # All-time winners (all pick types)
        alltime_result = await db.execute(
            select(func.count(Pick.id)).where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                )
            )
        )
        alltime_winners = alltime_result.scalar() or 0

        # All-time total collected (sum of returns from all winning bets)
        alltime_winnings_result = await db.execute(
            select(
                func.sum(Pick.bet_stake + Pick.pnl).filter(Pick.bet_stake.isnot(None)),
                func.sum(Pick.exotic_stake + Pick.pnl).filter(Pick.exotic_stake.isnot(None)),
            ).where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                )
            )
        )
        row = alltime_winnings_result.one()
        alltime_winnings = (row[0] or 0) + (row[1] or 0)

        # Get early mail content for today (approved or sent)
        early_mail_result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "early_mail",
                    Content.status.in_(["approved", "sent"]),
                    Content.meeting_id.in_(meeting_ids) if meeting_ids else False,
                )
            )
        )
        early_mail_content = early_mail_result.scalars().all()

        # Build list of today's tips with internal links
        todays_tips = []
        for content in early_mail_content:
            meeting = next((m for m in today_meetings if m.id == content.meeting_id), None)
            if meeting:
                todays_tips.append({
                    "venue": meeting.venue,
                    "meeting_id": meeting.id,
                })

        # Strike rates per tip rank (selections only)
        from sqlalchemy import case
        pick_ranks = []
        labels = {1: "Top Pick", 2: "2nd Pick", 3: "3rd Pick", 4: "Roughie"}
        for rank in [1, 2, 3, 4]:
            sr_result = await db.execute(
                select(
                    func.count(Pick.id),
                    func.sum(case((Pick.hit == True, 1), else_=0)),
                ).where(and_(
                    Pick.settled == True,
                    Pick.pick_type == "selection",
                    Pick.tip_rank == rank,
                ))
            )
            row = sr_result.one()
            total = row[0] or 0
            hits = int(row[1] or 0)
            rate = round(hits / total * 100, 1) if total > 0 else 0
            pick_ranks.append({
                "label": labels[rank],
                "rank": rank,
                "hits": hits,
                "total": total,
                "rate": rate,
            })

        return {
            "today_winners": today_winners,
            "alltime_winners": alltime_winners,
            "alltime_winnings": round(alltime_winnings, 2),
            "todays_tips": todays_tips,
            "meetings_today": len(today_meetings),
            "all_races_complete": all_races_complete,
            "pick_ranks": pick_ranks,
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


@router.get("/glossary", response_class=HTMLResponse)
async def glossary(request: Request):
    """Racing glossary page."""
    return templates.TemplateResponse("glossary.html", {"request": request})


@router.get("/calculator", response_class=HTMLResponse)
async def calculator(request: Request):
    """Betting calculator page."""
    return templates.TemplateResponse("calculator.html", {"request": request})


@router.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):
    """Terms of service page."""
    return templates.TemplateResponse("terms.html", {"request": request})


@router.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    """Privacy policy page."""
    return templates.TemplateResponse("privacy.html", {"request": request})


@router.get("/sitemap.xml")
async def sitemap():
    """Serve sitemap.xml for search engines."""
    return FileResponse(static_dir / "sitemap.xml", media_type="application/xml")


@router.get("/robots.txt")
async def robots():
    """Serve robots.txt for search engines."""
    return FileResponse(static_dir / "robots.txt", media_type="text/plain")


async def get_tips_calendar(
    page: int = 1,
    per_page: int = 30,
    venue: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> dict:
    """Get meetings with sent content, grouped by date for calendar view."""
    from sqlalchemy import desc, distinct
    from collections import defaultdict

    async with async_session() as db:
        # Build shared filter conditions
        filters = [
            Content.content_type == "early_mail",
            Content.status.in_(["approved", "sent"]),
            Meeting.date.isnot(None),
        ]
        if venue:
            filters.append(Meeting.venue.ilike(f"%{venue}%"))
        if date_from:
            filters.append(Meeting.date >= date_from)
        if date_to:
            filters.append(Meeting.date <= date_to)

        # Count total distinct dates with sent early mail content
        count_result = await db.execute(
            select(func.count(distinct(Meeting.date)))
            .join(Content, Content.meeting_id == Meeting.id)
            .where(and_(*filters))
        )
        total = count_result.scalar() or 0
        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        # Get the paginated distinct dates
        offset = (page - 1) * per_page
        dates_result = await db.execute(
            select(distinct(Meeting.date))
            .join(Content, Content.meeting_id == Meeting.id)
            .where(and_(*filters))
            .order_by(desc(Meeting.date))
            .offset(offset)
            .limit(per_page)
        )
        page_dates = [row[0] for row in dates_result.all()]

        if not page_dates:
            return {
                "calendar": [],
                "page": page,
                "per_page": per_page,
                "total_dates": total,
                "total_pages": total_pages,
                "has_next": False,
                "has_prev": page > 1,
            }

        # Fetch meetings for only the paginated dates
        meeting_filters = [
            Content.content_type == "early_mail",
            Content.status.in_(["approved", "sent"]),
            Meeting.date.in_(page_dates),
        ]
        if venue:
            meeting_filters.append(Meeting.venue.ilike(f"%{venue}%"))

        result = await db.execute(
            select(Meeting)
            .join(Content, Content.meeting_id == Meeting.id)
            .where(and_(*meeting_filters))
            .order_by(desc(Meeting.date), Meeting.venue)
        )
        meetings = result.scalars().all()

        # Group by date (deduplicate by meeting id)
        meetings_by_date = defaultdict(list)
        seen_meetings = set()
        for meeting in meetings:
            if meeting.id in seen_meetings:
                continue
            seen_meetings.add(meeting.id)
            date_key = meeting.date.isoformat()
            meetings_by_date[date_key].append({
                "id": meeting.id,
                "venue": meeting.venue,
                "date": meeting.date,
                "date_formatted": meeting.date.strftime("%a %d %b %Y") if meeting.date else "",
                "track_condition": meeting.track_condition,
                "slug": meeting.id,
            })

        # Build calendar in date order (already sorted by query)
        calendar = []
        for d in sorted(page_dates, reverse=True):
            date_key = d.isoformat()
            day_meetings = meetings_by_date.get(date_key, [])
            if day_meetings:
                calendar.append({
                    "date": date_key,
                    "date_formatted": day_meetings[0]["date_formatted"],
                    "meetings": day_meetings,
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

        # Get early mail (approved or sent)
        early_mail_result = await db.execute(
            select(Content).where(
                and_(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "early_mail",
                    Content.status.in_(["approved", "sent"]),
                )
            ).order_by(Content.created_at.desc())
        )
        early_mail = early_mail_result.scalars().first()

        # Get wrap-up (approved or sent) — use .first() since meetings can have multiple wrapups
        wrapup_result = await db.execute(
            select(Content).where(
                and_(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "meeting_wrapup",
                    Content.status.in_(["approved", "sent"]),
                )
            ).order_by(Content.created_at.desc())
        )
        wrapup = wrapup_result.scalars().first()

        # If no sent content at all, return None
        if not early_mail and not wrapup:
            return None

        # Get Punty's winning picks for win tick display
        from punty.models.pick import Pick
        import json as _json
        winners_result = await db.execute(
            select(Pick).where(
                and_(
                    Pick.meeting_id == meeting_id,
                    Pick.settled == True,
                    Pick.hit == True,
                )
            )
        )
        winners_map = {}       # {race_number: [saddlecloth, ...]} for selections/big3
        winning_exotics = {}   # {race_number: exotic_type} for hit exotics
        winning_sequences = [] # [{type, variant, start_race}] for hit sequences
        for pick in winners_result.scalars().all():
            if pick.pick_type in ("selection", "big3") and pick.race_number and pick.saddlecloth:
                winners_map.setdefault(pick.race_number, []).append(pick.saddlecloth)
            elif pick.pick_type == "exotic" and pick.race_number and pick.exotic_type:
                winning_exotics[pick.race_number] = pick.exotic_type
            elif pick.pick_type == "sequence" and pick.sequence_type:
                winning_sequences.append({
                    "type": pick.sequence_type,
                    "variant": pick.sequence_variant,
                })
            elif pick.pick_type == "big3_multi":
                winning_sequences.append({
                    "type": "big3_multi",
                    "variant": None,
                })

        # Get live updates (celebrations + pace analysis)
        updates_result = await db.execute(
            select(LiveUpdate).where(
                LiveUpdate.meeting_id == meeting_id,
            ).order_by(LiveUpdate.created_at.asc())
        )
        live_updates = [
            {
                "type": u.update_type,
                "content": u.content,
                "horse_name": u.horse_name,
                "odds": u.odds,
                "pnl": u.pnl,
                "race_number": u.race_number,
                "tweet_id": u.tweet_id,
                "created_at": u.created_at.strftime("%I:%M %p").lstrip("0") if u.created_at else None,
            }
            for u in updates_result.scalars().all()
        ]

        # Venue historical stats (Punty's track record here)
        from sqlalchemy import case
        venue_stats = None
        if meeting.venue:
            venue_meetings = select(Meeting.id).where(
                and_(Meeting.venue == meeting.venue, Meeting.id != meeting.id)
            )
            venue_sel = await db.execute(
                select(
                    func.count(Pick.id),
                    func.sum(case((Pick.hit == True, 1), else_=0)),
                    func.sum(Pick.pnl),
                    func.sum(Pick.bet_stake),
                ).where(and_(
                    Pick.settled == True,
                    Pick.pick_type == "selection",
                    Pick.meeting_id.in_(venue_meetings),
                ))
            )
            row = venue_sel.one()
            total, hits, pnl, staked = int(row[0] or 0), int(row[1] or 0), float(row[2] or 0), float(row[3] or 0)
            if total >= 4:
                venue_stats = {
                    "total": total,
                    "hits": hits,
                    "rate": round(hits / total * 100, 1) if total > 0 else 0,
                    "pnl": round(pnl, 2),
                    "staked": round(staked, 2),
                    "meetings": 0,
                }
                # Count distinct meetings
                mc = await db.execute(
                    select(func.count(func.distinct(Pick.meeting_id))).where(and_(
                        Pick.settled == True,
                        Pick.pick_type == "selection",
                        Pick.meeting_id.in_(venue_meetings),
                    ))
                )
                venue_stats["meetings"] = mc.scalar() or 0

        # Generate seed from meeting date for consistent rotation
        seed = hash(meeting.id) if meeting.id else 0

        from punty.formatters.html import format_html

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
                "content": format_html(early_mail.raw_content, "early_mail", seed),
                "created_at": early_mail.created_at.isoformat() if early_mail.created_at else None,
            } if early_mail else None,
            "wrapup": {
                "content": format_html(wrapup.raw_content, "meeting_wrapup", seed + 1),
                "created_at": wrapup.created_at.isoformat() if wrapup.created_at else None,
            } if wrapup else None,
            "winners": winners_map,
            "winning_exotics": winning_exotics,
            "winning_sequences": winning_sequences,
            "live_updates": live_updates,
            "venue_stats": venue_stats,
        }


@router.get("/tips", response_class=HTMLResponse)
async def tips_page(
    request: Request,
    page: int = 1,
    venue: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    """Public tips calendar page showing meetings by date."""
    from datetime import date as date_type

    # Parse date strings
    parsed_from = None
    parsed_to = None
    if date_from:
        try:
            parsed_from = date_type.fromisoformat(date_from)
        except ValueError:
            pass
    if date_to:
        try:
            parsed_to = date_type.fromisoformat(date_to)
        except ValueError:
            pass

    calendar_data = await get_tips_calendar(
        page=page, per_page=30, venue=venue,
        date_from=parsed_from, date_to=parsed_to,
    )

    # Build filter query string for pagination links
    filter_params = ""
    if venue:
        filter_params += f"&venue={venue}"
    if date_from:
        filter_params += f"&date_from={date_from}"
    if date_to:
        filter_params += f"&date_to={date_to}"

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
            "venue": venue or "",
            "date_from": date_from or "",
            "date_to": date_to or "",
            "filter_params": filter_params,
        }
    )


async def get_bet_type_stats(
    venue: str | None = None,
    state: str | None = None,
    distance_min: int | None = None,
    distance_max: int | None = None,
    track_condition: str | None = None,
    race_class: str | None = None,
    jockey: str | None = None,
    trainer: str | None = None,
    horse_sex: str | None = None,
    tip_rank: int | None = None,
    odds_min: float | None = None,
    odds_max: float | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    field_size_min: int | None = None,
    field_size_max: int | None = None,
    weather: str | None = None,
    barrier_min: int | None = None,
    barrier_max: int | None = None,
    today: bool = False,
) -> list[dict]:
    """Get strike rate, won/total, and P&L for every bet type with optional filters."""
    from sqlalchemy import case
    from punty.models.meeting import Runner

    # Desired display order
    _SEL_ORDER = ["Win", "Saver Win", "Place", "Each Way"]
    _EXOTIC_ORDER = ["Quinella", "Exacta", "Exacta Standout", "Trifecta", "Trifecta Box", "Trifecta Standout", "First Four"]
    _SEQ_ORDER = ["Early Quaddie", "Quaddie", "Big6"]

    def _normalise_exotic(raw: str) -> str:
        low = raw.lower().strip()
        if low in ("box trifecta", "trifecta box", "trifecta (box)", "trifecta (boxed)", "trifecta boxed"):
            return "Trifecta Box"
        if low in ("exacta standout", "exacta (standout)"):
            return "Exacta Standout"
        if low in ("trifecta standout", "trifecta (standout)"):
            return "Trifecta Standout"
        if low in ("first four", "first 4", "first four (boxed)", "first four box"):
            return "First Four"
        return raw

    def _esc(s: str) -> str:
        """Escape ILIKE wildcards in user input."""
        return s.replace("%", "\\%").replace("_", "\\_")

    # Build filter condition lists by table
    meeting_conds = []
    if venue:
        meeting_conds.append(Meeting.venue.ilike(f"%{_esc(venue)}%"))
    if state:
        from punty.memory.assessment import TRACK_STATE_MAP
        state_tracks = [t for t, s in TRACK_STATE_MAP.items() if s == state.upper()]
        if state_tracks:
            meeting_conds.append(or_(*[Meeting.venue.ilike(f"%{t}%") for t in state_tracks]))
    if date_from:
        meeting_conds.append(Meeting.date >= date_from)
    if date_to:
        meeting_conds.append(Meeting.date <= date_to)
    if track_condition:
        meeting_conds.append(Meeting.track_condition.ilike(f"%{_esc(track_condition)}%"))
    if weather:
        meeting_conds.append(Meeting.weather_condition.ilike(f"%{_esc(weather)}%"))
    if today:
        meeting_conds.append(Meeting.date == melb_today())

    race_conds = []
    if distance_min:
        race_conds.append(Race.distance >= distance_min)
    if distance_max:
        race_conds.append(Race.distance <= distance_max)
    if race_class:
        race_conds.append(Race.class_.ilike(f"%{_esc(race_class)}%"))
    if field_size_min:
        race_conds.append(Race.field_size >= field_size_min)
    if field_size_max:
        race_conds.append(Race.field_size <= field_size_max)

    runner_conds = []
    if jockey:
        runner_conds.append(Runner.jockey.ilike(f"%{_esc(jockey)}%"))
    if trainer:
        runner_conds.append(Runner.trainer.ilike(f"%{_esc(trainer)}%"))
    if horse_sex:
        runner_conds.append(Runner.horse_sex == horse_sex)
    if barrier_min:
        runner_conds.append(Runner.barrier >= barrier_min)
    if barrier_max:
        runner_conds.append(Runner.barrier <= barrier_max)

    pick_conds = []
    if tip_rank:
        pick_conds.append(Pick.tip_rank == tip_rank)
    if odds_min:
        pick_conds.append(Pick.odds_at_tip >= odds_min)
    if odds_max:
        pick_conds.append(Pick.odds_at_tip <= odds_max)

    needs_meeting = bool(meeting_conds)
    needs_race = bool(race_conds) or bool(runner_conds)
    needs_runner = bool(runner_conds)
    has_runner_or_pick_filters = needs_runner or bool(pick_conds)

    def _apply_joins(query, include_runner=False):
        """Add JOINs to query based on active filters."""
        if needs_meeting:
            query = query.join(Meeting, Pick.meeting_id == Meeting.id)
        if needs_race or (include_runner and needs_runner):
            query = query.join(Race, and_(
                Race.meeting_id == Pick.meeting_id,
                Race.race_number == Pick.race_number,
            ))
        if include_runner and needs_runner:
            query = query.join(Runner, and_(
                Runner.race_id == Race.id,
                Runner.saddlecloth == Pick.saddlecloth,
            ))
        return query

    def _extra_conds(include_runner=False, include_pick=False):
        """Collect filter conditions for the query."""
        conds = list(meeting_conds) + list(race_conds)
        if include_runner:
            conds.extend(runner_conds)
        if include_pick:
            conds.extend(pick_conds)
        return conds

    async with async_session() as db:
        # --- Selections by bet_type (exclude exotics_only) ---
        sel_query = select(
            Pick.bet_type,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.pnl),
            func.sum(Pick.bet_stake),
        )
        sel_query = _apply_joins(sel_query, include_runner=True)
        sel_conds = [
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type.isnot(None),
            Pick.bet_type != "exotics_only",
        ] + _extra_conds(include_runner=True, include_pick=True)
        sel_result = await db.execute(
            sel_query.where(and_(*sel_conds)).group_by(Pick.bet_type)
        )

        sel_stats = {}
        for bet_type, total, hits, pnl, staked in sel_result.all():
            hits = int(hits or 0)
            pnl = float(pnl or 0)
            staked = float(staked or 0)
            rate = round(hits / total * 100, 1) if total > 0 else 0
            label = (bet_type or "unknown").replace("_", " ").title()
            sel_stats[label] = {
                "category": "Selections",
                "type": label,
                "won": hits,
                "total": total,
                "rate": rate,
                "pnl": round(pnl, 2),
                "staked": round(staked, 2),
            }

        # --- Exotics (skip when runner/pick filters active) ---
        exotic_stats: dict[str, dict] = {}
        if not has_runner_or_pick_filters:
            exotic_query = select(
                Pick.exotic_type,
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.exotic_stake),
            )
            exotic_query = _apply_joins(exotic_query, include_runner=False)
            exotic_conds = [
                Pick.settled == True,
                Pick.pick_type == "exotic",
                Pick.exotic_type.isnot(None),
            ] + _extra_conds()
            exotic_result = await db.execute(
                exotic_query.where(and_(*exotic_conds)).group_by(Pick.exotic_type)
            )

            for exotic_type, total, hits, pnl, staked in exotic_result.all():
                label = _normalise_exotic(exotic_type)
                hits = int(hits or 0)
                pnl = float(pnl or 0)
                staked = float(staked or 0)
                if label in exotic_stats:
                    exotic_stats[label]["won"] += hits
                    exotic_stats[label]["total"] += total
                    exotic_stats[label]["pnl"] = round(exotic_stats[label]["pnl"] + pnl, 2)
                    exotic_stats[label]["staked"] = round(exotic_stats[label]["staked"] + staked, 2)
                else:
                    exotic_stats[label] = {
                        "category": "Exotics",
                        "type": label,
                        "won": hits,
                        "total": total,
                        "rate": 0,
                        "pnl": round(pnl, 2),
                        "staked": round(staked, 2),
                    }
            for s in exotic_stats.values():
                s["rate"] = round(s["won"] / s["total"] * 100, 1) if s["total"] > 0 else 0

        # --- Sequences (skip when runner/pick filters active) ---
        seq_stats = {}
        if not has_runner_or_pick_filters:
            seq_query = select(
                Pick.sequence_type,
                Pick.sequence_variant,
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.exotic_stake),
            )
            seq_query = _apply_joins(seq_query, include_runner=False)
            seq_conds = [
                Pick.settled == True,
                Pick.pick_type == "sequence",
            ] + _extra_conds()
            seq_result = await db.execute(
                seq_query.where(and_(*seq_conds)).group_by(Pick.sequence_type, Pick.sequence_variant)
            )

            for seq_type, seq_variant, total, hits, pnl, staked in seq_result.all():
                hits = int(hits or 0)
                pnl = float(pnl or 0)
                staked = float(staked or 0)
                rate = round(hits / total * 100, 1) if total > 0 else 0
                label = (seq_type or "Sequence").replace("_", " ").title()
                if seq_variant:
                    label += f" ({seq_variant.title()})"
                seq_stats[label] = {
                    "category": "Sequences",
                    "type": label,
                    "won": hits,
                    "total": total,
                    "rate": rate,
                    "pnl": round(pnl, 2),
                    "staked": round(staked, 2),
                }

        # --- Big 3 Multi (skip when runner/pick filters active) ---
        big3_stat = None
        if not has_runner_or_pick_filters:
            big3_query = select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.exotic_stake),
            )
            big3_query = _apply_joins(big3_query, include_runner=False)
            big3_conds = [
                Pick.settled == True,
                Pick.pick_type == "big3_multi",
            ] + _extra_conds()
            big3_result = await db.execute(
                big3_query.where(and_(*big3_conds))
            )
            row = big3_result.one()
            if row[0] and row[0] > 0:
                hits = int(row[1] or 0)
                pnl = float(row[2] or 0)
                staked = float(row[3] or 0)
                rate = round(hits / row[0] * 100, 1) if row[0] > 0 else 0
                big3_stat = {
                    "category": "Multi",
                    "type": "Big 3 Multi",
                    "won": hits,
                    "total": row[0],
                    "rate": rate,
                    "pnl": round(pnl, 2),
                    "staked": round(staked, 2),
                }

        # Build ordered output
        stats = []
        for key in _SEL_ORDER:
            if key in sel_stats:
                stats.append(sel_stats[key])
        for key in _EXOTIC_ORDER:
            if key in exotic_stats:
                stats.append(exotic_stats[key])
        for key, val in exotic_stats.items():
            if key not in _EXOTIC_ORDER:
                stats.append(val)
        for prefix in _SEQ_ORDER:
            for key in sorted(seq_stats.keys()):
                if key.startswith(prefix):
                    stats.append(seq_stats[key])
        if big3_stat:
            stats.append(big3_stat)

        return stats


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Public stats page with filterable performance data."""
    return templates.TemplateResponse("stats.html", {"request": request})


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
            "winners": data.get("winners", {}),
            "winning_exotics": data.get("winning_exotics", {}),
            "winning_sequences": data.get("winning_sequences", []),
            "live_updates": data.get("live_updates", []),
            "venue_stats": data.get("venue_stats"),
        }
    )
