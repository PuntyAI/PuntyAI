"""Public website pages — static/content pages, SEO sitemaps."""

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from sqlalchemy import select, func, and_, or_, text, case

from punty.config import melb_today, melb_now
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race

from punty.public.deps import templates, static_dir

router = APIRouter()


async def get_recent_wins_public(limit: int = 15) -> dict:
    """Get recent wins for public ticker with Punty celebrations."""
    from punty.results.celebrations import get_celebration

    async with async_session() as db:
        # Subquery: content IDs that are NOT superseded/rejected
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

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
                    Pick.content_id.in_(active_content),
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

        # Exclude picks from superseded/rejected content
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        # Today's winners (all pick types that hit, excluding tracked-only)
        today_result = await db.execute(
            select(func.count(Pick.id))
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                    Meeting.date == today_date,
                    Meeting.selected == True,
                    Pick.content_id.in_(active_content),
                    or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
                )
            )
        )
        today_winners = today_result.scalar() or 0

        # Check if all races today are complete
        # Include meetings with approved content (e.g. abandoned meetings still have tips)
        meetings_with_content = select(Content.meeting_id).where(
            Content.content_type == "early_mail",
            Content.status.in_(["approved", "sent"]),
        ).scalar_subquery()
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

        # All-time winners (all pick types, excluding tracked-only)
        alltime_result = await db.execute(
            select(func.count(Pick.id))
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                    Meeting.selected == True,
                    Pick.content_id.in_(active_content),
                    or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
                )
            )
        )
        alltime_winners = alltime_result.scalar() or 0

        # All-time total collected (sum of returns from all winning bets)
        alltime_winnings_result = await db.execute(
            select(
                func.sum(Pick.bet_stake + Pick.pnl).filter(Pick.bet_stake.isnot(None)),
                func.sum(Pick.exotic_stake + Pick.pnl).filter(Pick.exotic_stake.isnot(None)),
            )
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
                    Meeting.selected == True,
                )
            )
        )
        row = alltime_winnings_result.one_or_none()
        alltime_winnings = ((row[0] or 0) + (row[1] or 0)) if row else 0

        # Get early mail content for today (approved or sent)
        early_mail_result = await db.execute(
            select(Content).where(
                and_(
                    Content.content_type == "early_mail",
                    Content.status.in_(["approved", "sent"]),
                    Content.meeting_id.in_(meeting_ids) if meeting_ids else text("1=0"),
                )
            )
        )
        early_mail_content = early_mail_result.scalars().all()

        # Build list of today's tips with internal links
        from punty.venues import guess_state as get_state_for_track
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
        pick_ranks = []
        labels = {1: "Top Pick", 2: "2nd Pick", 3: "3rd Pick", 4: "Roughie"}
        rank_conds = [
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type != "exotics_only",
            Meeting.selected == True,
            or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
        ]
        if today:
            rank_conds.append(Meeting.date == today_date)
        for rank in [1, 2, 3, 4]:
            q = select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.bet_stake),
            ).join(Meeting, Pick.meeting_id == Meeting.id).where(and_(*rank_conds, Pick.tip_rank == rank))
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
                        Pick.content_id.in_(active_content),
                        or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
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
                        Pick.content_id.in_(active_content),
                        or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
                    )
                )
                .order_by(Pick.pnl.desc())
                .limit(1)
            )
            recent_row = recent_result.first()
            if recent_row:
                best_bet = _build_best_bet(*recent_row, is_today=False)

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
            Meeting.selected == True,
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
            Meeting.selected == True,
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


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Public homepage."""
    import asyncio
    import random
    from pathlib import Path
    from punty.results.picks import get_performance_history
    from punty.venues import guess_state as get_state_for_track, is_metro

    today = melb_today()
    thirty_ago = today - timedelta(days=30)

    # Pick 3 random social images for CTA carousel
    social_dir = Path("data/social_images")
    social_imgs = sorted(social_dir.glob("punty_promo_*.png")) if social_dir.exists() else []
    cta_images = [f"/social-img/{p.name}" for p in random.sample(social_imgs, min(3, len(social_imgs)))] if social_imgs else []

    stats, next_race_data = await asyncio.gather(
        get_winner_stats(),
        _get_next_race_import(),
    )

    # 30-day P&L + strike rate
    async with async_session() as db:
        perf_history = await get_performance_history(db, thirty_ago, today)

        # 7-day P&L
        seven_ago = today - timedelta(days=7)
        perf_7d = await get_performance_history(db, seven_ago, today)

        # All-time strike rate fallback
        alltime_stats = await db.execute(
            select(
                func.count(Pick.id).label("total"),
                func.sum(case((Pick.hit == True, 1), else_=0)).label("hits"),
            )
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                Pick.settled == True,
                Meeting.selected == True,
                or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
                Pick.pick_type == "selection",
            )
        )
        at_row = alltime_stats.one_or_none()
        alltime_total = (at_row.total or 0) if at_row else 0
        alltime_hits = (at_row.hits or 0) if at_row else 0

    pnl_30d = sum(d["pnl"] for d in perf_history)
    pnl_7d = sum(d["pnl"] for d in perf_7d)
    staked_7d = sum(d.get("staked", 0) for d in perf_7d)
    roi_7d = round(pnl_7d / staked_7d * 100, 1) if staked_7d else 0
    bets_30d = sum(d["bets"] for d in perf_history)
    hits_30d = sum(d.get("hits", 0) for d in perf_history)
    strike_30d = round(hits_30d / bets_30d * 100, 1) if bets_30d else 0
    # Use all-time strike rate if 30d has no data
    if not strike_30d and alltime_total:
        strike_30d = round(alltime_hits / alltime_total * 100, 1)

    # Today's meetings with metro flag
    async with async_session() as db:
        from sqlalchemy import func as sa_func
        next_jump_sq = (
            select(
                Race.meeting_id,
                sa_func.min(Race.start_time).label("next_jump"),
            )
            .where(or_(
                Race.results_status.is_(None),
                Race.results_status.in_(["Open", "scheduled"]),
            ))
            .group_by(Race.meeting_id)
            .subquery()
        )
        meetings_result = await db.execute(
            select(Meeting, next_jump_sq.c.next_jump)
            .outerjoin(next_jump_sq, Meeting.id == next_jump_sq.c.meeting_id)
            .where(and_(
                Meeting.date == today,
                Meeting.selected == True,
                Meeting.meeting_type.in_(["race", None]),
            ))
            .order_by(next_jump_sq.c.next_jump.asc().nullslast())
        )
        today_meetings = meetings_result.all()

    meetings = []
    for m, next_jump in today_meetings:
        meetings.append({
            "venue": m.venue,
            "meeting_id": m.id,
            "state": get_state_for_track(m.venue) or "AUS",
            "track_condition": m.track_condition,
            "next_jump": next_jump.isoformat() if next_jump else None,
            "is_metro": is_metro(m.venue),
        })

    # Sort: metro first, then by next_jump
    meetings.sort(key=lambda x: (not x["is_metro"], x["next_jump"] or "9999"))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
            "meetings": meetings,
            "roi_7d": roi_7d,
            "strike_30d": strike_30d,
            "next_race": next_race_data,
            "cta_images": cta_images,
        }
    )


async def _get_next_race_import():
    """Import and call get_next_race from dashboard to avoid circular imports."""
    from punty.public.dashboard import get_next_race
    return await get_next_race()


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
                    Meeting.selected == True,
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
