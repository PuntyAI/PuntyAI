"""Public website routes - no authentication required."""

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_, or_

from punty.config import melb_today, melb_now, MELB_TZ
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race, Runner
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
                    Meeting.selected == True,  # Exclude deselected meetings
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


async def get_winner_stats(today: bool = False) -> dict:
    """Get winner statistics for today and all-time."""
    from punty.models.meeting import Race

    async with async_session() as db:
        today_date = melb_today()

        # Today's winners (all pick types that hit)
        today_result = await db.execute(
            select(func.count(Pick.id))
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                    Meeting.date == today_date,
                )
            )
        )
        today_winners = today_result.scalar() or 0

        # Check if all races today are complete
        today_meetings_result = await db.execute(
            select(Meeting).where(
                and_(
                    Meeting.date == today_date,
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
        from punty.memory.assessment import _get_state_from_track as get_state_for_track
        todays_tips = []
        for content in early_mail_content:
            meeting = next((m for m in today_meetings if m.id == content.meeting_id), None)
            if meeting:
                todays_tips.append({
                    "venue": meeting.venue,
                    "meeting_id": meeting.id,
                    "state": get_state_for_track(meeting.venue) or "VIC",
                })

        # Strike rates per tip rank (selections only)
        from sqlalchemy import case
        pick_ranks = []
        labels = {1: "Top Pick", 2: "2nd Pick", 3: "3rd Pick", 4: "Roughie"}
        rank_conds = [
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type != "exotics_only",
        ]
        if today:
            rank_conds.append(Meeting.date == today_date)
        for rank in [1, 2, 3, 4]:
            q = select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.bet_stake),
            ).where(and_(*rank_conds, Pick.tip_rank == rank))
            if today:
                q = q.join(Meeting, Pick.meeting_id == Meeting.id)
            sr_result = await db.execute(q)
            row = sr_result.one()
            total = row[0] or 0
            hits = int(row[1] or 0)
            pnl = float(row[2] or 0)
            staked = float(row[3] or 0)
            rate = round(hits / total * 100, 1) if total > 0 else 0
            roi = round(pnl / staked * 100, 1) if staked > 0 else 0
            pick_ranks.append({
                "label": labels[rank],
                "rank": rank,
                "hits": hits,
                "total": total,
                "rate": rate,
                "pnl": round(pnl, 2),
                "staked": round(staked, 2),
                "roi": roi,
            })

        # Best Bet — highest PNL winning pick today, fallback to recent
        best_bet = None

        def _build_best_bet(bp, bm, is_today=True):
            stake = bp.bet_stake or bp.exotic_stake or 0
            pnl = float(bp.pnl or 0)
            roi = round(pnl / stake * 100, 1) if stake > 0 else 0
            returned = stake + pnl

            if bp.pick_type == "selection":
                display = bp.horse_name or "Runner"
                bet_label = (bp.bet_type or "win").replace("_", " ").title()
            elif bp.pick_type == "exotic":
                display = bp.exotic_type or "Exotic"
                bet_label = "Exotic"
            elif bp.pick_type == "sequence":
                display = (bp.sequence_type or "Sequence").replace("_", " ").title()
                bet_label = "Sequence"
            elif bp.pick_type == "big3_multi":
                display = "Big 3 Multi"
                bet_label = "Multi"
            else:
                display = "Winner"
                bet_label = bp.pick_type or "Bet"

            # Show place odds for place bets, win odds otherwise
            if bp.bet_type == "place" and bp.place_odds_at_tip:
                display_odds = float(bp.place_odds_at_tip)
            else:
                display_odds = float(bp.odds_at_tip) if bp.odds_at_tip else None

            return {
                "display_name": display,
                "venue": bm.venue,
                "race_number": bp.race_number,
                "bet_type": bet_label,
                "odds": display_odds,
                "stake": round(stake, 2),
                "returned": round(returned, 2),
                "pnl": round(pnl, 2),
                "roi": roi,
                "pick_type": bp.pick_type,
                "meeting_id": bm.id,
                "is_today": is_today,
            }

        # Try today first
        if meeting_ids:
            best_bet_result = await db.execute(
                select(Pick, Meeting)
                .join(Meeting, Pick.meeting_id == Meeting.id)
                .where(
                    and_(
                        Pick.settled == True,
                        Pick.hit == True,
                        Pick.pnl > 0,
                        Meeting.date == today_date,
                        Meeting.selected == True,
                    )
                )
                .order_by(Pick.pnl.desc())
                .limit(1)
            )
            best_row = best_bet_result.first()
            if best_row:
                best_bet = _build_best_bet(*best_row, is_today=True)

        # Fallback: most recent winning bet from last 7 days
        if not best_bet:
            from datetime import timedelta
            recent_cutoff = today_date - timedelta(days=7)
            recent_result = await db.execute(
                select(Pick, Meeting)
                .join(Meeting, Pick.meeting_id == Meeting.id)
                .where(
                    and_(
                        Pick.settled == True,
                        Pick.hit == True,
                        Pick.pnl > 0,
                        Meeting.date >= recent_cutoff,
                        Meeting.date < today_date,
                        Meeting.selected == True,
                    )
                )
                .order_by(Pick.pnl.desc())
                .limit(1)
            )
            recent_row = recent_result.first()
            if recent_row:
                best_bet = _build_best_bet(*recent_row, is_today=False)

        # Punty's Picks — only selections flagged as is_puntys_pick
        # Use pp_hit/pp_pnl when set (Punty's Pick had different bet type than main selection)
        from sqlalchemy import literal_column
        pp_hit_expr = func.coalesce(Pick.pp_hit, Pick.hit)
        pp_pnl_expr = func.coalesce(Pick.pp_pnl, Pick.pnl)
        pp_conds = [
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.is_puntys_pick == True,
            Pick.bet_type != "exotics_only",
        ]
        if today:
            pp_conds.append(Meeting.date == today_date)
        pp_q = select(
            func.count(Pick.id),
            func.sum(case((pp_hit_expr == True, 1), else_=0)),
            func.sum(pp_pnl_expr),
            func.sum(Pick.bet_stake),
        ).where(and_(*pp_conds))
        if today:
            pp_q = pp_q.join(Meeting, Pick.meeting_id == Meeting.id)
        pp_result = await db.execute(pp_q)
        pp_row = pp_result.one()
        pp_total = pp_row[0] or 0
        pp_hits = int(pp_row[1] or 0)
        pp_pnl = float(pp_row[2] or 0)
        pp_staked = float(pp_row[3] or 0)
        pp_rate = round(pp_hits / pp_total * 100, 1) if pp_total > 0 else 0
        pp_roi = round(pp_pnl / pp_staked * 100, 1) if pp_staked > 0 else 0
        pick_ranks.append({
            "label": "Punty's Picks",
            "rank": 0,
            "hits": pp_hits,
            "total": pp_total,
            "rate": pp_rate,
            "pnl": round(pp_pnl, 2),
            "staked": round(pp_staked, 2),
            "roi": pp_roi,
        })

        return {
            "today_winners": today_winners,
            "alltime_winners": alltime_winners,
            "alltime_winnings": round(alltime_winnings, 2),
            "todays_tips": todays_tips,
            "meetings_today": len(today_meetings),
            "all_races_complete": all_races_complete,
            "pick_ranks": pick_ranks,
            "best_bet": best_bet,
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


@router.get("/blog", response_class=HTMLResponse)
async def blog_listing(request: Request, page: int = 1):
    """Public blog listing page."""
    import re
    per_page = 10

    async with async_session() as db:
        # Count total published blogs
        count_result = await db.execute(
            select(func.count(Content.id)).where(
                and_(
                    Content.content_type == "weekly_blog",
                    Content.status.in_(["approved", "sent"]),
                )
            )
        )
        total = count_result.scalar() or 0
        total_pages = max(1, (total + per_page - 1) // per_page)

        # Fetch page of blogs
        offset = (page - 1) * per_page
        result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "weekly_blog",
                    Content.status.in_(["approved", "sent"]),
                )
            ).order_by(Content.created_at.desc())
            .offset(offset).limit(per_page)
        )
        blogs_raw = result.scalars().all()

        blogs = []
        for b in blogs_raw:
            # Extract excerpt (first substantial paragraph)
            excerpt = ""
            if b.raw_content:
                for line in b.raw_content.split("\n"):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or stripped.startswith("*FROM") or stripped.startswith("---"):
                        continue
                    if len(stripped) > 50:
                        excerpt = re.sub(r'\*+', '', stripped)[:200]
                        if len(stripped) > 200:
                            excerpt = excerpt.rsplit(" ", 1)[0] + "..."
                        break

            blogs.append({
                "slug": b.blog_slug or b.id,
                "title": b.blog_title or "From the Horse's Mouth",
                "date_formatted": b.blog_week_start.strftime("%A, %d %B %Y") if b.blog_week_start else (
                    b.created_at.strftime("%A, %d %B %Y") if b.created_at else ""
                ),
                "excerpt": excerpt,
            })

    return templates.TemplateResponse(
        "blog.html",
        {
            "request": request,
            "blogs": blogs,
            "page": page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }
    )


@router.get("/blog/{slug}", response_class=HTMLResponse)
async def blog_post(request: Request, slug: str):
    """Public individual blog post page."""
    from fastapi import HTTPException
    from punty.formatters.blog import format_blog_html
    import re

    async with async_session() as db:
        result = await db.execute(
            select(Content).where(
                and_(
                    Content.blog_slug == slug,
                    Content.content_type == "weekly_blog",
                    Content.status.in_(["approved", "sent"]),
                )
            )
        )
        content = result.scalar_one_or_none()
        if not content:
            raise HTTPException(status_code=404, detail="Blog post not found")

        # Format to HTML
        html_content = format_blog_html(content.raw_content or "")

        # Extract excerpt for meta tags
        excerpt = ""
        if content.raw_content:
            for line in content.raw_content.split("\n"):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("*FROM") or stripped.startswith("---"):
                    continue
                if len(stripped) > 50:
                    excerpt = re.sub(r'\*+', '', stripped)[:160]
                    break

        blog_date = content.blog_week_start or (content.created_at.date() if content.created_at else None)
        blog_data = {
            "title": content.blog_title or "From the Horse's Mouth",
            "slug": content.blog_slug,
            "html_content": html_content,
            "excerpt": excerpt,
            "date_formatted": blog_date.strftime("%A, %d %B %Y") if blog_date else "",
            "date_iso": blog_date.isoformat() if blog_date else "",
        }

        # Prev/next navigation
        prev_result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "weekly_blog",
                    Content.status.in_(["approved", "sent"]),
                    Content.created_at < content.created_at,
                )
            ).order_by(Content.created_at.desc()).limit(1)
        )
        prev_blog_raw = prev_result.scalar_one_or_none()
        prev_blog = {"slug": prev_blog_raw.blog_slug, "title": prev_blog_raw.blog_title or "Previous"} if prev_blog_raw else None

        next_result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "weekly_blog",
                    Content.status.in_(["approved", "sent"]),
                    Content.created_at > content.created_at,
                )
            ).order_by(Content.created_at.asc()).limit(1)
        )
        next_blog_raw = next_result.scalar_one_or_none()
        next_blog = {"slug": next_blog_raw.blog_slug, "title": next_blog_raw.blog_title or "Next"} if next_blog_raw else None

    return templates.TemplateResponse(
        "blog_post.html",
        {
            "request": request,
            "blog": blog_data,
            "prev_blog": prev_blog,
            "next_blog": next_blog,
        }
    )


@router.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):
    """Terms of service page."""
    return templates.TemplateResponse("terms.html", {"request": request})


@router.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    """Privacy policy page."""
    return templates.TemplateResponse("privacy.html", {"request": request})


@router.get("/data-deletion", response_class=HTMLResponse)
async def data_deletion(request: Request):
    """Data deletion instructions page."""
    return templates.TemplateResponse("data-deletion.html", {"request": request})


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
        # Hard floor: never show tips before 2026-02-17 (pre-audit data removed)
        TIPS_START_DATE = date(2026, 2, 17)
        effective_from = max(date_from, TIPS_START_DATE) if date_from else TIPS_START_DATE

        filters = [
            Content.content_type == "early_mail",
            Content.status.in_(["approved", "sent"]),
            Meeting.date.isnot(None),
            Meeting.date >= effective_from,
        ]
        if venue:
            filters.append(Meeting.venue.ilike(f"%{venue}%"))
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


def _norm_exotic(raw: str) -> str:
    """Normalize exotic type names for display."""
    low = raw.lower().strip()
    return {
        "box trifecta": "Trifecta", "trifecta box": "Trifecta",
        "trifecta (box)": "Trifecta", "trifecta (boxed)": "Trifecta",
        "trifecta boxed": "Trifecta", "exacta standout": "Exacta",
        "exacta (standout)": "Exacta",
        "trifecta standout": "Trifecta",
        "trifecta (standout)": "Trifecta",
        "first four": "First 4", "first 4": "First 4",
        "first four (boxed)": "First 4", "first four box": "First 4",
        "first4": "First 4", "first4 box": "First 4",
        "first 4 standout": "First 4", "first four standout": "First 4",
    }.get(low, raw)


def _compute_pick_data(all_picks: list) -> dict:
    """Compute winners, sequences, selections lookup, and stats from a single picks list.

    Returns dict with keys: winners_map, winning_exotics, winning_sequences,
    sequence_results, picks_lookup, meeting_stats.
    """
    winners_map = {}       # {race_number: [saddlecloth, ...]}
    winning_exotics = {}   # {race_number: exotic_type}
    winning_sequences = [] # [{type, variant}]
    sequence_results = []
    picks_lookup = {}      # {race_number: {saddlecloth: {...}}}

    # Accumulators for stats
    sel_stats = {}   # {bet_type: {total, hits, pnl, staked}}
    ex_stats = {}    # {norm_type: {total, hits, pnl, staked}}
    seq_stats = {}   # {(seq_type, variant): {total, hits, pnl, staked}}
    b3_stats = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}

    for pick in all_picks:
        # --- Selection picks lookup (all, not just settled) ---
        if pick.pick_type == "selection" and pick.race_number and pick.saddlecloth:
            picks_lookup.setdefault(pick.race_number, {})[pick.saddlecloth] = {
                "tip_rank": pick.tip_rank,
                "bet_type": pick.bet_type,
                "hit": pick.hit,
                "pnl": float(pick.pnl) if pick.pnl is not None else None,
            }

        # --- Winners (settled + hit) ---
        if pick.settled and pick.hit:
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
                winning_sequences.append({"type": "big3_multi", "variant": None})

        # --- Settled sequence/multi results ---
        if pick.settled and pick.pick_type in ("sequence", "big3_multi"):
            seq_type = pick.sequence_type or pick.pick_type
            label = (pick.sequence_type or "multi").replace("_", " ").title()
            sequence_results.append({
                "type": seq_type,
                "variant": pick.sequence_variant,
                "label": label,
                "hit": bool(pick.hit),
                "pnl": float(pick.pnl) if pick.pnl is not None else 0.0,
                "stake": float(pick.exotic_stake or pick.bet_stake or 0),
            })

        # --- Stats accumulators (settled only) ---
        if not pick.settled:
            continue

        if pick.pick_type == "selection" and pick.bet_type and pick.bet_type != "exotics_only":
            bt = pick.bet_type
            if bt not in sel_stats:
                sel_stats[bt] = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}
            sel_stats[bt]["total"] += 1
            if pick.hit:
                sel_stats[bt]["hits"] += 1
            sel_stats[bt]["pnl"] += float(pick.pnl or 0)
            sel_stats[bt]["staked"] += float(pick.bet_stake or 0)

        elif pick.pick_type == "exotic" and pick.exotic_type:
            label = _norm_exotic(pick.exotic_type)
            if label not in ex_stats:
                ex_stats[label] = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}
            ex_stats[label]["total"] += 1
            if pick.hit:
                ex_stats[label]["hits"] += 1
            ex_stats[label]["pnl"] += float(pick.pnl or 0)
            ex_stats[label]["staked"] += float(pick.exotic_stake or 0)

        elif pick.pick_type == "sequence":
            key = (pick.sequence_type, pick.sequence_variant)
            if key not in seq_stats:
                seq_stats[key] = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}
            seq_stats[key]["total"] += 1
            if pick.hit:
                seq_stats[key]["hits"] += 1
            seq_stats[key]["pnl"] += float(pick.pnl or 0)
            seq_stats[key]["staked"] += float(pick.exotic_stake or 0)

        elif pick.pick_type == "big3_multi":
            b3_stats["total"] += 1
            if pick.hit:
                b3_stats["hits"] += 1
            b3_stats["pnl"] += float(pick.pnl or 0)
            b3_stats["staked"] += float(pick.exotic_stake or 0)

    # --- Build meeting_stats list ---
    meeting_stats = []

    # Selections by bet_type (ordered)
    for k in ["win", "saver_win", "place", "each_way"]:
        if k in sel_stats:
            s = sel_stats[k]
            label = k.replace("_", " ").title()
            meeting_stats.append({
                "category": "Selections", "type": label,
                "won": s["hits"], "total": s["total"],
                "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
                "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
            })

    # Exotics (ordered)
    ex_order = ["Quinella", "Exacta", "Trifecta", "First 4"]
    for k in ex_order:
        if k in ex_stats:
            s = ex_stats[k]
            meeting_stats.append({
                "category": "Exotics", "type": k,
                "won": s["hits"], "total": s["total"],
                "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
                "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
            })
    for k, s in ex_stats.items():
        if k not in ex_order:
            meeting_stats.append({
                "category": "Exotics", "type": k,
                "won": s["hits"], "total": s["total"],
                "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
                "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
            })

    # Sequences
    for (st, sv), s in seq_stats.items():
        label = (st or "Sequence").replace("_", " ").title()
        meeting_stats.append({
            "category": "Sequences", "type": label,
            "won": s["hits"], "total": s["total"],
            "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
            "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
        })

    # Big3 Multi
    if b3_stats["total"] > 0:
        meeting_stats.append({
            "category": "Multi", "type": "Big 3 Multi",
            "won": b3_stats["hits"], "total": b3_stats["total"],
            "rate": round(b3_stats["hits"] / b3_stats["total"] * 100, 1) if b3_stats["total"] else 0,
            "pnl": round(b3_stats["pnl"], 2), "staked": round(b3_stats["staked"], 2),
        })

    return {
        "winners_map": winners_map,
        "winning_exotics": winning_exotics,
        "winning_sequences": winning_sequences,
        "sequence_results": sequence_results,
        "picks_lookup": picks_lookup,
        "meeting_stats": meeting_stats,
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

        # Get early mail + wrapup in one query (2 content rows max)
        content_result = await db.execute(
            select(Content).where(
                and_(
                    Content.meeting_id == meeting_id,
                    Content.content_type.in_(["early_mail", "meeting_wrapup"]),
                    Content.status.in_(["approved", "sent"]),
                )
            ).order_by(Content.created_at.desc())
        )
        all_content = content_result.scalars().all()
        early_mail = next((c for c in all_content if c.content_type == "early_mail"), None)
        wrapup = next((c for c in all_content if c.content_type == "meeting_wrapup"), None)

        # If no sent content at all, return None
        if not early_mail and not wrapup:
            return None

        # Load ALL picks for this meeting in one query — compute everything in Python
        picks_result = await db.execute(
            select(Pick).where(Pick.meeting_id == meeting_id)
        )
        all_picks = picks_result.scalars().all()
        pick_data = _compute_pick_data(all_picks)

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

        # Fetch races and runners for form guide display
        races_result = await db.execute(
            select(Race).where(Race.meeting_id == meeting_id).order_by(Race.race_number)
        )
        races_list = races_result.scalars().all()

        race_ids = [r.id for r in races_list]
        runners_by_race = {}
        if race_ids:
            runners_result = await db.execute(
                select(Runner).where(Runner.race_id.in_(race_ids)).order_by(Runner.saddlecloth)
            )
            for runner in runners_result.scalars().all():
                runners_by_race.setdefault(runner.race_id, []).append(runner)

        picks_lookup = pick_data["picks_lookup"]

        # Build races data for template
        races_data = []
        for race in races_list:
            race_runners = runners_by_race.get(race.id, [])
            race_picks = picks_lookup.get(race.race_number, {})
            runners_data = []
            for r in race_runners:
                pick = race_picks.get(r.saddlecloth)
                runners_data.append({
                    "sc": r.saddlecloth, "name": r.horse_name,
                    "b": r.barrier, "j": r.jockey, "t": r.trainer,
                    "w": float(r.weight) if r.weight else None,
                    "odds": float(r.current_odds) if r.current_odds else None,
                    "form": r.last_five or r.form,
                    "smp": r.speed_map_position,
                    "fp": r.finish_position,
                    "wd": float(r.win_dividend) if r.win_dividend else None,
                    "pd": float(r.place_dividend) if r.place_dividend else None,
                    "x": bool(r.scratched),
                    "pick": pick,
                })
            races_data.append({
                "num": race.race_number, "name": race.name,
                "dist": race.distance, "cls": race.class_,
                "prize": race.prize_money,
                "time": race.start_time.strftime("%H:%M") if race.start_time else None,
                "status": race.results_status,
                "runners": runners_data,
            })

        # Venue historical stats (Punty's track record here) — single query
        from sqlalchemy import case
        venue_stats = None
        if meeting.venue:
            venue_meetings = select(Meeting.id).where(
                and_(Meeting.venue == meeting.venue, Meeting.id != meeting.id)
            )
            venue_result = await db.execute(
                select(
                    func.count(Pick.id),
                    func.sum(case((Pick.hit == True, 1), else_=0)),
                    func.sum(Pick.pnl),
                    func.sum(Pick.bet_stake),
                    func.count(func.distinct(Pick.meeting_id)),
                ).where(and_(
                    Pick.settled == True,
                    Pick.pick_type == "selection",
                    Pick.meeting_id.in_(venue_meetings),
                ))
            )
            row = venue_result.one()
            total, hits, pnl, staked, meetings_count = (
                int(row[0] or 0), int(row[1] or 0), float(row[2] or 0),
                float(row[3] or 0), int(row[4] or 0),
            )
            if total >= 4:
                venue_stats = {
                    "total": total,
                    "hits": hits,
                    "rate": round(hits / total * 100, 1) if total > 0 else 0,
                    "pnl": round(pnl, 2),
                    "staked": round(staked, 2),
                    "meetings": meetings_count,
                }

        # Other meetings on the same day (for venue switcher)
        same_day_meetings = []
        if meeting.date:
            same_day_result = await db.execute(
                select(Meeting).where(
                    and_(
                        Meeting.date == meeting.date,
                        Meeting.selected == True,
                        Meeting.id != meeting.id,
                    )
                ).order_by(Meeting.venue)
            )
            # Only include meetings that have approved/sent early mail
            for m in same_day_result.scalars().all():
                em_check = await db.execute(
                    select(Content.id).where(
                        and_(
                            Content.meeting_id == m.id,
                            Content.content_type == "early_mail",
                            Content.status.in_(["approved", "sent"]),
                        )
                    ).limit(1)
                )
                if em_check.scalar_one_or_none() is not None:
                    same_day_meetings.append({
                        "id": m.id,
                        "venue": m.venue,
                    })

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
            "winners": pick_data["winners_map"],
            "winning_exotics": pick_data["winning_exotics"],
            "winning_sequences": pick_data["winning_sequences"],
            "sequence_results": pick_data["sequence_results"],
            "live_updates": live_updates,
            "venue_stats": venue_stats,
            "races": races_data,
            "meeting_stats": pick_data["meeting_stats"],
            "same_day_meetings": same_day_meetings,
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
    _EXOTIC_ORDER = ["Quinella", "Exacta", "Trifecta", "First 4"]
    _SEQ_ORDER = ["Early Quaddie", "Quaddie", "Big6"]
    _VARIANT_ORDER = ["Skinny", "Balanced", "Wide"]

    def _normalise_exotic(raw: str) -> str:
        low = raw.lower().strip()
        if low in ("box trifecta", "trifecta box", "trifecta (box)", "trifecta (boxed)", "trifecta boxed",
                    "trifecta standout", "trifecta (standout)"):
            return "Trifecta"
        if low in ("exacta standout", "exacta (standout)"):
            return "Exacta"
        if low in ("first four", "first 4", "first four (boxed)", "first four box",
                    "first4", "first4 box", "first 4 standout", "first four standout"):
            return "First 4"
        return raw

    def _esc(s: str) -> str:
        """Escape ILIKE wildcards in user input."""
        return s.replace("%", "\\%").replace("_", "\\_")

    # Build filter condition lists by table
    # Hard floor: exclude pre-audit data
    meeting_conds = [Meeting.date >= date(2026, 2, 17)]
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
            for variant in _VARIANT_ORDER:
                key = f"{prefix} ({variant})"
                if key in seq_stats:
                    stats.append(seq_stats[key])
            # Catch any variants not in _VARIANT_ORDER
            for key in sorted(seq_stats.keys()):
                if key.startswith(prefix) and key not in [f"{prefix} ({v})" for v in _VARIANT_ORDER]:
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

    # Dynamic meta tags for SEO
    venue = data["meeting"].get("venue", "")
    date_str = data["meeting"].get("date_formatted", "")
    meta_title = f"{venue} Racing Tips {date_str} | PuntyAI"
    meta_desc = f"AI racing tips and form guide for {venue} on {date_str}. Early mail analysis, selections, exotics, and live race updates."

    response = templates.TemplateResponse(
        "meeting_tips.html",
        {
            "request": request,
            "meeting": data["meeting"],
            "early_mail": data["early_mail"],
            "wrapup": data["wrapup"],
            "winners": data.get("winners", {}),
            "winning_exotics": data.get("winning_exotics", {}),
            "winning_sequences": data.get("winning_sequences", []),
            "sequence_results": data.get("sequence_results", []),
            "live_updates": data.get("live_updates", []),
            "venue_stats": data.get("venue_stats"),
            "races": data.get("races", []),
            "meeting_stats": data.get("meeting_stats", []),
            "same_day_meetings": data.get("same_day_meetings", []),
            "meta_title": meta_title,
            "meta_description": meta_desc,
        }
    )
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


# ──────────────────────────────────────────────
# SEO: Dynamic sitemaps + llms.txt
# ──────────────────────────────────────────────

@router.get("/llms.txt")
async def llms_txt():
    """Serve llms.txt for AI crawler discovery."""
    return FileResponse(static_dir / "llms.txt", media_type="text/plain")


@router.get("/sitemap.xml")
async def sitemap_index():
    """Main sitemap index linking to sub-sitemaps."""
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    xml += '  <sitemap><loc>https://punty.ai/sitemap-static.xml</loc></sitemap>\n'
    xml += '  <sitemap><loc>https://punty.ai/sitemap-blog.xml</loc></sitemap>\n'
    xml += '  <sitemap><loc>https://punty.ai/sitemap-tips.xml</loc></sitemap>\n'
    xml += '</sitemapindex>'
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-static.xml")
async def sitemap_static():
    """Static pages sitemap."""
    pages = [
        ("https://punty.ai/", "1.0", "daily"),
        ("https://punty.ai/tips", "0.9", "daily"),
        ("https://punty.ai/stats", "0.8", "daily"),
        ("https://punty.ai/blog", "0.8", "daily"),
        ("https://punty.ai/how-it-works", "0.7", "monthly"),
        ("https://punty.ai/calculator", "0.7", "monthly"),
        ("https://punty.ai/glossary", "0.7", "monthly"),
        ("https://punty.ai/about", "0.6", "monthly"),
        ("https://punty.ai/contact", "0.5", "monthly"),
    ]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for loc, priority, freq in pages:
        xml += f'  <url><loc>{loc}</loc><changefreq>{freq}</changefreq><priority>{priority}</priority></url>\n'
    xml += '</urlset>'
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-blog.xml")
async def sitemap_blog():
    """Dynamic sitemap for all published blog posts."""
    async with async_session() as db:
        result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "weekly_blog",
                    Content.status.in_(["approved", "sent"]),
                    Content.blog_slug.isnot(None),
                )
            ).order_by(Content.created_at.desc())
        )
        blogs = result.scalars().all()

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for blog in blogs:
        lastmod = blog.created_at.strftime("%Y-%m-%d") if blog.created_at else ""
        xml += f'  <url>\n'
        xml += f'    <loc>https://punty.ai/blog/{blog.blog_slug}</loc>\n'
        if lastmod:
            xml += f'    <lastmod>{lastmod}</lastmod>\n'
        xml += f'    <changefreq>never</changefreq>\n'
        xml += f'    <priority>0.8</priority>\n'
        xml += f'  </url>\n'
    xml += '</urlset>'
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-tips.xml")
async def sitemap_tips():
    """Dynamic sitemap for meeting tips pages."""
    async with async_session() as db:
        result = await db.execute(
            select(Meeting.id, Meeting.date).join(
                Content, Content.meeting_id == Meeting.id
            ).where(
                and_(
                    Content.content_type == "early_mail",
                    Content.status.in_(["approved", "sent"]),
                    Meeting.date >= date(2026, 2, 17),
                )
            ).order_by(Meeting.date.desc()).limit(500)
        )
        meetings = result.all()

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for mid, mdate in meetings:
        lastmod = mdate.strftime("%Y-%m-%d") if mdate else ""
        xml += f'  <url>\n'
        xml += f'    <loc>https://punty.ai/tips/{mid}</loc>\n'
        if lastmod:
            xml += f'    <lastmod>{lastmod}</lastmod>\n'
        xml += f'    <changefreq>never</changefreq>\n'
        xml += f'    <priority>0.7</priority>\n'
        xml += f'  </url>\n'
    xml += '</urlset>'
    return Response(content=xml, media_type="application/xml")
