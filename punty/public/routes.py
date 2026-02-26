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

from punty.config import settings as _app_settings
templates.env.globals["is_staging"] = _app_settings.is_staging


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
            "meeting_id": next_race.meeting_id,
            "race_number": next_race.race_number,
            "race_name": next_race.name,
            "distance": next_race.distance,
            "class": next_race.class_,
            "start_time_iso": start_time_aware.isoformat(),
            "start_time_formatted": next_race.start_time.strftime("%H:%M"),
        }


async def _get_picks_for_race(meeting_id: str, race_number: int) -> list[dict]:
    """Direct query for picks in a specific race — fallback when upcoming filter misses them."""
    import json as _json_fb

    async with async_session() as db:
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        result = await db.execute(
            select(Pick)
            .where(
                Pick.meeting_id == meeting_id,
                Pick.race_number == race_number,
                Pick.content_id.in_(active_content),
                Pick.settled == False,
            )
            .order_by(Pick.tip_rank.nullslast())
        )
        picks = result.scalars().all()

    out = []
    for pick in picks:
        if pick.pick_type not in ("selection", "exotic"):
            continue

        if pick.pick_type == "selection":
            name = pick.horse_name or "Runner"
        else:
            name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"

        exotic_runners = None
        if pick.pick_type == "exotic" and pick.exotic_runners:
            try:
                exotic_runners = _json_fb.loads(pick.exotic_runners) if isinstance(pick.exotic_runners, str) else pick.exotic_runners
            except (ValueError, TypeError):
                exotic_runners = None
            if exotic_runners and isinstance(exotic_runners, list) and any(isinstance(r, list) for r in exotic_runners):
                exotic_runners = [item for sub in exotic_runners for item in (sub if isinstance(sub, list) else [sub])]

        wp = round(pick.win_probability * 100, 1) if pick.win_probability else None
        pp = round(pick.place_probability * 100, 1) if pick.place_probability else None
        bt_lower = (pick.bet_type or "").lower()
        show_prob = pp if bt_lower == "place" and pp else wp

        out.append({
            "name": name,
            "saddlecloth": pick.saddlecloth,
            "venue": "",
            "meeting_id": meeting_id,
            "race_number": race_number,
            "pick_type": pick.pick_type,
            "bet_type": (pick.bet_type or "").replace("_", " ").title(),
            "exotic_type": pick.exotic_type,
            "exotic_runners": exotic_runners,
            "odds": pick.odds_at_tip,
            "stake": round(pick.bet_stake or pick.exotic_stake or 0, 2),
            "tip_rank": pick.tip_rank,
            "value_rating": round(pick.value_rating, 2) if pick.value_rating else None,
            "win_prob": wp,
            "place_prob": pp,
            "show_prob": show_prob,
            "confidence": pick.confidence,
            "is_puntys_pick": pick.is_puntys_pick or False,
            "start_time": None,
            "is_edge": (pick.value_rating or 0) >= 1.1 and pick.pick_type == "selection",
        })
    return out


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
                    Pick.content_id.in_(active_content),
                    or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
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

        # All-time winners (all pick types, excluding tracked-only)
        alltime_result = await db.execute(
            select(func.count(Pick.id)).where(
                and_(
                    Pick.hit == True,
                    Pick.settled == True,
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
        from sqlalchemy import case
        pick_ranks = []
        labels = {1: "Top Pick", 2: "2nd Pick", 3: "3rd Pick", 4: "Roughie"}
        rank_conds = [
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type != "exotics_only",
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


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Public homepage."""
    import asyncio
    from datetime import timedelta
    from punty.results.picks import get_performance_history
    from punty.venues import guess_state as get_state_for_track, is_metro

    today = melb_today()
    thirty_ago = today - timedelta(days=30)

    stats, next_race_data = await asyncio.gather(
        get_winner_stats(),
        get_next_race(),
    )

    # 30-day P&L + strike rate
    async with async_session() as db:
        perf_history = await get_performance_history(db, thirty_ago, today)

        # Biggest single win (highest positive pnl)
        biggest_win_result = await db.execute(
            select(func.max(Pick.pnl)).where(
                Pick.settled == True,
                Pick.pnl > 0,
                or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
            )
        )
        biggest_win = biggest_win_result.scalar() or 0

        # All-time strike rate fallback
        from sqlalchemy import case
        alltime_stats = await db.execute(
            select(
                func.count(Pick.id).label("total"),
                func.sum(case((Pick.hit == True, 1), else_=0)).label("hits"),
            ).where(
                Pick.settled == True,
                or_(Pick.tracked_only == False, Pick.tracked_only.is_(None)),
                Pick.pick_type == "selection",
            )
        )
        at_row = alltime_stats.one()
        alltime_total = at_row.total or 0
        alltime_hits = at_row.hits or 0

    pnl_30d = sum(d["pnl"] for d in perf_history)
    amount_won_30d = sum(d["pnl"] for d in perf_history if d["pnl"] > 0)
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
            "pnl_30d": round(pnl_30d, 2),
            "amount_won_30d": round(amount_won_30d, 2),
            "strike_30d": strike_30d,
            "biggest_win": round(biggest_win, 2),
            "next_race": next_race_data,
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
    losing_exotics = {}    # {race_number: exotic_type}  — settled but NOT hit
    winning_sequences = [] # [{type, variant}]
    sequence_results = []
    picks_lookup = {}      # {race_number: {saddlecloth: {...}}}
    pp_picks = {}          # {race_number: saddlecloth}  — Punty's Pick per race

    # Accumulators for stats
    sel_stats = {}   # {bet_type: {total, hits, pnl, staked}}
    ex_stats = {}    # {norm_type: {total, hits, pnl, staked}}
    seq_stats = {}   # {(seq_type, variant): {total, hits, pnl, staked}}
    b3_stats = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}

    for pick in all_picks:
        # --- Selection picks lookup (all, not just settled) ---
        if pick.pick_type == "selection" and pick.race_number and pick.saddlecloth:
            # Edge/confidence calculation
            bt_lower = (pick.bet_type or "").lower()
            model_prob = None
            if "place" in bt_lower and pick.place_probability:
                model_prob = round(pick.place_probability * 100, 1)
            elif pick.win_probability:
                model_prob = round(pick.win_probability * 100, 1)
            market_implied = round(100 / pick.odds_at_tip, 1) if pick.odds_at_tip else None
            edge_pct = round(model_prob - market_implied, 1) if model_prob and market_implied else None
            if edge_pct is not None:
                if edge_pct >= 10:
                    confidence = "HIGH EDGE"
                elif edge_pct >= 5:
                    confidence = "VALUE"
                elif edge_pct >= 3:
                    confidence = "EDGE"
                else:
                    confidence = "SPECULATIVE"
            else:
                confidence = None

            picks_lookup.setdefault(pick.race_number, {})[pick.saddlecloth] = {
                "tip_rank": pick.tip_rank,
                "bet_type": pick.bet_type,
                "hit": pick.hit,
                "pnl": float(pick.pnl) if pick.pnl is not None else None,
                "odds": float(pick.odds_at_tip) if pick.odds_at_tip else None,
                "stake": float(pick.bet_stake) if pick.bet_stake else None,
                "model_prob": model_prob,
                "market_implied": market_implied,
                "edge_pct": edge_pct,
                "confidence": confidence,
                "is_puntys_pick": bool(pick.is_puntys_pick),
                "is_roughie": bool(pick.is_roughie) if hasattr(pick, 'is_roughie') else (pick.tip_rank == 4),
            }
            # Track Punty's Pick per race
            if pick.is_puntys_pick:
                pp_picks[pick.race_number] = pick.saddlecloth

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

        # --- Losing exotics (settled but NOT hit) ---
        if pick.settled and not pick.hit:
            if pick.pick_type == "exotic" and pick.race_number and pick.exotic_type:
                losing_exotics[pick.race_number] = pick.exotic_type

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
        "losing_exotics": losing_exotics,
        "winning_sequences": winning_sequences,
        "sequence_results": sequence_results,
        "picks_lookup": picks_lookup,
        "pp_picks": pp_picks,
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

        # Load picks only from active (approved/sent) content — exclude superseded/rejected
        active_content_ids = [c.id for c in all_content]
        if active_content_ids:
            picks_result = await db.execute(
                select(Pick).where(
                    Pick.meeting_id == meeting_id,
                    Pick.content_id.in_(active_content_ids),
                )
            )
        else:
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

        # Build scratched picks + alternatives for template
        scratched_picks = {}   # {race_number: [saddlecloth, ...]}
        alternatives = {}      # {race_number: {name, sc, odds}}
        for rd in races_data:
            rn = rd["num"]
            scr = [r["sc"] for r in rd["runners"] if r["x"] and r.get("pick")]
            if not scr:
                continue
            scratched_picks[rn] = scr
            pick_scs = {r["sc"] for r in rd["runners"] if r.get("pick")}
            cands = sorted(
                [r for r in rd["runners"] if not r["x"] and r["sc"] not in pick_scs and r.get("odds")],
                key=lambda r: r["odds"]
            )
            if cands:
                alternatives[rn] = {"name": cands[0]["name"], "sc": cands[0]["sc"], "odds": cands[0]["odds"]}

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
            "losing_exotics": pick_data["losing_exotics"],
            "winning_sequences": pick_data["winning_sequences"],
            "pp_picks": pick_data["pp_picks"],
            "sequence_results": pick_data["sequence_results"],
            "live_updates": live_updates,
            "venue_stats": venue_stats,
            "races": races_data,
            "meeting_stats": pick_data["meeting_stats"],
            "same_day_meetings": same_day_meetings,
            "scratched_picks": scratched_picks,
            "alternatives": alternatives,
        }


@router.get("/tips", response_class=HTMLResponse)
async def tips_dashboard(request: Request):
    """Racing dashboard — one-stop page for today's racing."""
    import asyncio
    from datetime import timedelta
    from punty.results.picks import get_performance_history

    today = melb_today()
    week_ago = today - timedelta(days=7)

    # Run all data fetches in parallel
    dashboard, stats, next_race_data, recent_wins = await asyncio.gather(
        get_daily_dashboard(),
        get_winner_stats(today=True),
        get_next_race(),
        get_recent_wins_public(15),
    )

    # 7-day performance history
    async with async_session() as db:
        perf_history = await get_performance_history(db, week_ago, today)

    seven_day_pnl = sum(d["pnl"] for d in perf_history)
    seven_day_bets = sum(d["bets"] for d in perf_history)
    seven_day_roi = round(seven_day_pnl / (seven_day_bets * 10) * 100, 1) if seven_day_bets else 0

    # Always fetch today's selected meetings with next-race jump times
    from punty.venues import guess_state as get_state_for_track
    async with async_session() as db:
        # Get meetings with their next un-run race start_time for sorting
        next_jump_sq = (
            select(
                Race.meeting_id,
                func.min(Race.start_time).label("next_jump"),
                func.min(Race.race_number).label("next_race_num"),
            )
            .where(
                or_(
                    Race.results_status.is_(None),
                    Race.results_status.in_(["Open", "scheduled"]),
                )
            )
            .group_by(Race.meeting_id)
            .subquery()
        )
        meetings_result = await db.execute(
            select(Meeting, next_jump_sq.c.next_jump, next_jump_sq.c.next_race_num)
            .outerjoin(next_jump_sq, Meeting.id == next_jump_sq.c.meeting_id)
            .where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
            .order_by(next_jump_sq.c.next_jump.asc().nullslast())
        )
        today_meetings = meetings_result.all()

    from punty.venues import is_metro
    meetings_list = []
    for m, next_jump, next_race_num in today_meetings:
        meetings_list.append({
            "venue": m.venue,
            "meeting_id": m.id,
            "state": get_state_for_track(m.venue) or "AUS",
            "track_condition": m.track_condition,
            "next_jump": next_jump.isoformat() if next_jump else None,
            "next_race_num": next_race_num,
            "is_metro": is_metro(m.venue),
        })

    # Enrich todays_tips with track_condition
    for tip in stats.get("todays_tips", []):
        tip["track_condition"] = None
        match = next((m for m in meetings_list if m["meeting_id"] == tip.get("meeting_id")), None)
        if match:
            tip["track_condition"] = match["track_condition"]

    # Best of meets + sequence bets
    best_of, sequences = await asyncio.gather(
        get_best_of_meets(),
        get_all_sequences(),
    )

    # Match next race to upcoming picks
    next_race_picks = []
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        for u in dashboard.get("upcoming", []):
            if u.get("meeting_id") == nr_mid and u.get("race_number") == nr_rn:
                next_race_picks.append(u)

        # Fallback: direct query if upcoming filter returned empty
        if not next_race_picks:
            next_race_picks = await _get_picks_for_race(nr_mid, nr_rn)

    # Build scratched saddlecloths for next race
    next_race_scratched = set()
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        race_id = f"{nr_mid}-r{nr_rn}"
        async with async_session() as db:
            runners_res = await db.execute(
                select(Runner).where(Runner.race_id == race_id, Runner.scratched == True)
            )
            next_race_scratched = {r.saddlecloth for r in runners_res.scalars().all() if r.saddlecloth}

    # Build primary_play from next race picks (Punty's Pick or rank 1)
    primary_play = None
    if next_race_picks and next_race_data.get("has_next"):
        # Prefer Punty's Pick, else rank 1
        pp = next((p for p in next_race_picks if p.get("is_puntys_pick")), None)
        if not pp:
            pp = next((p for p in next_race_picks if p.get("tip_rank") == 1), None)
        if not pp and next_race_picks:
            pp = next_race_picks[0]

        if pp and pp.get("odds"):
            bt_lower = (pp.get("bet_type") or "").lower()
            if "place" in bt_lower and pp.get("place_prob"):
                model_prob = pp["place_prob"] / 100
            elif pp.get("win_prob"):
                model_prob = pp["win_prob"] / 100
            else:
                model_prob = (pp.get("place_prob") or pp.get("win_prob") or 0) / 100
            market_implied = 1 / pp["odds"] if pp["odds"] else 0
            edge_pct = round((model_prob - market_implied) * 100, 1)
            if edge_pct >= 10:
                confidence = "HIGH EDGE"
            elif edge_pct >= 5:
                confidence = "VALUE"
            elif edge_pct >= 3:
                confidence = "EDGE"
            else:
                confidence = "SMALL EDGE"
            primary_play = {
                **pp,
                "model_prob": round(model_prob * 100, 1),
                "market_implied": round(market_implied * 100, 1),
                "edge_pct": edge_pct,
                "confidence": confidence,
            }

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dashboard": dashboard,
            "stats": stats,
            "next_race": next_race_data,
            "next_race_picks": next_race_picks,
            "next_race_scratched": next_race_scratched,
            "recent_wins": recent_wins,
            "perf_history": perf_history,
            "seven_day_pnl": round(seven_day_pnl, 2),
            "seven_day_roi": seven_day_roi,
            "best_of": best_of,
            "sequences": sequences,
            "meetings_list": meetings_list,
            "primary_play": primary_play,
        }
    )


@router.get("/tips/_glance", response_class=HTMLResponse)
async def tips_glance(request: Request):
    """HTMX partial: at-a-glance strip refresh."""
    import asyncio
    from datetime import timedelta
    from punty.results.picks import get_performance_history

    today = melb_today()
    week_ago = today - timedelta(days=7)

    dashboard, stats, next_race_data = await asyncio.gather(
        get_daily_dashboard(),
        get_winner_stats(today=True),
        get_next_race(),
    )

    async with async_session() as db:
        perf_history = await get_performance_history(db, week_ago, today)

    seven_day_pnl = sum(d["pnl"] for d in perf_history)
    seven_day_bets = sum(d["bets"] for d in perf_history)
    seven_day_roi = round(seven_day_pnl / (seven_day_bets * 10) * 100, 1) if seven_day_bets else 0

    # Quick meetings count (independent of content approval)
    async with async_session() as db:
        meetings_count_result = await db.execute(
            select(func.count()).select_from(Meeting).where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
        )
        meetings_count = meetings_count_result.scalar() or 0

    return templates.TemplateResponse(
        "partials/glance_strip.html",
        {
            "request": request,
            "dashboard": dashboard,
            "stats": stats,
            "next_race": next_race_data,
            "seven_day_pnl": round(seven_day_pnl, 2),
            "seven_day_roi": seven_day_roi,
            "meetings_count": meetings_count,
        }
    )


@router.get("/tips/_next-race", response_class=HTMLResponse)
async def tips_next_race(request: Request):
    """HTMX partial: next race hero refresh."""
    import asyncio

    dashboard, next_race_data = await asyncio.gather(
        get_daily_dashboard(),
        get_next_race(),
    )

    next_race_picks = []
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        for u in dashboard.get("upcoming", []):
            if u.get("meeting_id") == nr_mid and u.get("race_number") == nr_rn:
                next_race_picks.append(u)

        # Fallback: direct query if upcoming filter returned empty
        if not next_race_picks:
            next_race_picks = await _get_picks_for_race(nr_mid, nr_rn)

    # Build scratched saddlecloths for next race
    next_race_scratched = set()
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        race_id = f"{nr_mid}-r{nr_rn}"
        async with async_session() as db:
            runners_res = await db.execute(
                select(Runner).where(Runner.race_id == race_id, Runner.scratched == True)
            )
            next_race_scratched = {r.saddlecloth for r in runners_res.scalars().all() if r.saddlecloth}

    return templates.TemplateResponse(
        "partials/next_race.html",
        {
            "request": request,
            "next_race": next_race_data,
            "next_race_picks": next_race_picks,
            "next_race_scratched": next_race_scratched,
        }
    )


@router.get("/tips/_live-edge", response_class=HTMLResponse)
async def tips_live_edge(request: Request):
    """HTMX partial: Live Edge Zone refresh."""
    import asyncio

    dashboard, next_race_data = await asyncio.gather(
        get_daily_dashboard(),
        get_next_race(),
    )

    next_race_picks = []
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        for u in dashboard.get("upcoming", []):
            if u.get("meeting_id") == nr_mid and u.get("race_number") == nr_rn:
                next_race_picks.append(u)
        if not next_race_picks:
            next_race_picks = await _get_picks_for_race(nr_mid, nr_rn)

    # Build primary_play
    primary_play = None
    if next_race_picks and next_race_data.get("has_next"):
        pp = next((p for p in next_race_picks if p.get("is_puntys_pick")), None)
        if not pp:
            pp = next((p for p in next_race_picks if p.get("tip_rank") == 1), None)
        if not pp and next_race_picks:
            pp = next_race_picks[0]

        if pp and pp.get("odds"):
            bt_lower = (pp.get("bet_type") or "").lower()
            if "place" in bt_lower and pp.get("place_prob"):
                model_prob = pp["place_prob"] / 100
            elif pp.get("win_prob"):
                model_prob = pp["win_prob"] / 100
            else:
                model_prob = (pp.get("place_prob") or pp.get("win_prob") or 0) / 100
            market_implied = 1 / pp["odds"] if pp["odds"] else 0
            edge_pct = round((model_prob - market_implied) * 100, 1)
            if edge_pct >= 10:
                confidence = "HIGH EDGE"
            elif edge_pct >= 5:
                confidence = "VALUE"
            elif edge_pct >= 3:
                confidence = "EDGE"
            else:
                confidence = "SMALL EDGE"
            primary_play = {
                **pp,
                "model_prob": round(model_prob * 100, 1),
                "market_implied": round(market_implied * 100, 1),
                "edge_pct": edge_pct,
                "confidence": confidence,
            }

    return templates.TemplateResponse(
        "partials/live_edge.html",
        {
            "request": request,
            "next_race": next_race_data,
            "next_race_picks": next_race_picks,
            "primary_play": primary_play,
        }
    )


@router.get("/tips/_results-feed", response_class=HTMLResponse)
async def tips_results_feed(request: Request):
    """HTMX partial: results feed refresh."""
    import asyncio

    dashboard, recent_wins = await asyncio.gather(
        get_daily_dashboard(),
        get_recent_wins_public(15),
    )

    return templates.TemplateResponse(
        "partials/results_feed.html",
        {
            "request": request,
            "dashboard": dashboard,
            "recent_wins": recent_wins,
        }
    )


async def get_best_of_meets() -> dict:
    """Get best winner, roughie, and exotic pick per today's meeting."""
    today = melb_today()

    async with async_session() as db:
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        result = await db.execute(
            select(Pick, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                Meeting.date == today,
                Meeting.selected == True,
                Pick.content_id.in_(active_content),
            )
        )
        rows = result.all()

    from punty.venues import guess_state as get_state_for_track

    meets = {}
    for pick, meeting in rows:
        mid = meeting.id
        if mid not in meets:
            meets[mid] = {
                "venue": meeting.venue,
                "meeting_id": mid,
                "state": get_state_for_track(meeting.venue) or "AUS",
                "best_winner": None,
                "roughie": None,
                "exotic": None,
            }

        # Best Winner: selection, tip_rank <= 3, Win/Saver Win only, highest win_prob
        bt_lower = (pick.bet_type or "").lower()
        if (pick.pick_type == "selection"
                and (pick.tip_rank or 99) <= 3
                and bt_lower in ("win", "saver_win", "saver win")
                and (pick.win_probability or 0) >= 0.22):
            current = meets[mid]["best_winner"]
            if not current or (pick.win_probability or 0) > (current.get("win_prob") or 0):
                meets[mid]["best_winner"] = {
                    "horse": pick.horse_name,
                    "race": pick.race_number,
                    "odds": pick.odds_at_tip,
                    "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                    "win_prob": pick.win_probability,
                    "tip_rank": pick.tip_rank,
                    "is_puntys_pick": pick.is_puntys_pick or False,
                }

        # Best Roughie: tip_rank == 4, highest value_rating, odds >= $8
        if (pick.pick_type == "selection"
                and pick.tip_rank == 4
                and (pick.odds_at_tip or 0) >= 8
                and (pick.win_probability or 0) >= 0.08):
            current = meets[mid]["roughie"]
            if not current or (pick.value_rating or 0) > (current.get("value_rating") or 0):
                meets[mid]["roughie"] = {
                    "horse": pick.horse_name,
                    "race": pick.race_number,
                    "odds": pick.odds_at_tip,
                    "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                    "value_rating": pick.value_rating,
                }

        # Best Exotic: exotic with highest stake per meeting (prefer higher race)
        if pick.pick_type == "exotic":
            import json as _json3
            runners = pick.exotic_runners
            if isinstance(runners, str):
                try:
                    runners = _json3.loads(runners)
                except (ValueError, TypeError):
                    runners = []
            # Flatten nested arrays (e.g. [[5], [8, 2, 1]] → [5, 8, 2, 1])
            if runners and isinstance(runners, list) and any(isinstance(r, list) for r in runners):
                runners = [item for sub in runners for item in (sub if isinstance(sub, list) else [sub])]
            new_exotic = {
                "type": pick.exotic_type,
                "race": pick.race_number,
                "runners": runners or [],
                "stake": pick.exotic_stake or 0,
            }
            current = meets[mid]["exotic"]
            # Replace if no current, higher stake, or same stake but later race
            if (not current
                    or (new_exotic["stake"] or 0) > (current.get("stake") or 0)
                    or ((new_exotic["stake"] or 0) == (current.get("stake") or 0)
                        and (new_exotic["race"] or 0) > (current.get("race") or 0))):
                meets[mid]["exotic"] = new_exotic

    return meets


async def get_all_sequences() -> list:
    """Get all sequence bets (quaddies, big6, etc.) for today's meetings.
    Excludes sequences whose first leg race has already started."""
    today = melb_today()
    import json as _json
    from datetime import datetime, timezone

    async with async_session() as db:
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        result = await db.execute(
            select(Pick, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                Meeting.date == today,
                Meeting.selected == True,
                Pick.content_id.in_(active_content),
                Pick.pick_type == "sequence",
            )
        )
        rows = result.all()

        # Fetch race start times for filtering started sequences
        race_times = {}
        if rows:
            meeting_ids = list({m.id for _, m in rows})
            races_result = await db.execute(
                select(Race.meeting_id, Race.race_number, Race.start_time)
                .where(Race.meeting_id.in_(meeting_ids))
            )
            for mid, rn, st in races_result.all():
                race_times[(mid, rn)] = st

    if not rows:
        return []

    from zoneinfo import ZoneInfo
    MELB_TZ = ZoneInfo("Australia/Melbourne")
    now = datetime.now(MELB_TZ)
    sequences = []
    for pick, meeting in rows:
        # Skip sequences whose last leg has already started (fully underway)
        start_race = pick.sequence_start_race or 1
        legs_data = pick.sequence_legs
        if isinstance(legs_data, str):
            try:
                legs_data = _json.loads(legs_data)
            except (ValueError, TypeError):
                legs_data = []
        num_legs = len(legs_data) if legs_data else 4
        last_leg_race = start_race + num_legs - 1
        last_leg_time = race_times.get((meeting.id, last_leg_race))
        if last_leg_time:
            if last_leg_time.tzinfo is None:
                last_leg_time = last_leg_time.replace(tzinfo=MELB_TZ)
            if last_leg_time <= now:
                continue
        first_leg_time = race_times.get((meeting.id, start_race))
        legs = legs_data

        combo_count = 1
        for leg in (legs or []):
            if isinstance(leg, list):
                combo_count *= len(leg)
            elif isinstance(leg, dict):
                combo_count *= len(leg.get("runners", [1]))

        # Build compact leg summaries (e.g. "R5: 3,9,6")
        leg_summaries = []
        start = pick.sequence_start_race or 1
        for i, leg in enumerate(legs or []):
            if isinstance(leg, list):
                runners = leg
            elif isinstance(leg, dict):
                runners = leg.get("runners", [])
            else:
                runners = [leg]
            leg_summaries.append({
                "race": start + i,
                "runners": [str(r) for r in runners],
            })

        # Store first leg time for sorting and display
        flt = first_leg_time
        if flt and flt.tzinfo is None:
            from zoneinfo import ZoneInfo
            flt = flt.replace(tzinfo=ZoneInfo("Australia/Melbourne"))

        sequences.append({
            "venue": meeting.venue,
            "meeting_id": meeting.id,
            "type": (pick.sequence_type or "quaddie").title(),
            "variant": (pick.sequence_variant or "").title(),
            "legs": leg_summaries,
            "combo_count": combo_count,
            "stake": pick.exotic_stake or pick.bet_stake,
            "first_leg_time": flt.isoformat() if flt else None,
            "first_leg_race": start_race,
        })

    # Sort by first leg time (soonest first = "up next")
    sequences.sort(key=lambda s: s.get("first_leg_time") or "9999")

    return sequences


@router.get("/tips/archive", response_class=HTMLResponse)
async def tips_archive(
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
        from punty.venues import get_venues_for_state
        state_tracks = get_venues_for_state(state)
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


async def get_daily_dashboard() -> dict:
    """Build today's daily scoreboard — results, upcoming, timeline, insights."""
    from sqlalchemy import case, desc, func as sa_func
    from collections import Counter, defaultdict
    from punty.results.celebrations import get_celebration
    from punty.results.picks import get_performance_summary

    today = melb_today()
    now = melb_now()
    now_naive = now.replace(tzinfo=None)

    async with async_session() as db:
        # ── Performance summary (reuse existing) ──
        performance = await get_performance_summary(db, today)

        # ── Subquery: active content only ──
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        # ── ALL picks today (settled + unsettled) with runner/race details ──
        # Use coalesce so sequences (race_number=NULL) join via sequence_start_race
        pick_race_num = sa_func.coalesce(Pick.race_number, Pick.sequence_start_race)
        result = await db.execute(
            select(Pick, Runner, Race, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .outerjoin(Race, and_(
                Race.meeting_id == Pick.meeting_id,
                Race.race_number == pick_race_num,
            ))
            .outerjoin(Runner, and_(
                Runner.race_id == Race.id,
                Runner.saddlecloth == Pick.saddlecloth,
            ))
            .where(
                Meeting.date == today,
                Meeting.selected == True,
                Pick.content_id.in_(active_content),
            )
            .order_by(Meeting.venue, Race.race_number, Pick.tip_rank.nullslast())
        )
        all_rows = result.all()

        # Split into settled and unsettled
        settled_rows = [(p, r, rc, m) for p, r, rc, m in all_rows if p.settled]
        unsettled_rows = [(p, r, rc, m) for p, r, rc, m in all_rows if not p.settled]

        # Sort settled by PNL desc for big wins
        settled_by_pnl = sorted(settled_rows, key=lambda x: x[0].pnl or 0, reverse=True)

        # ── Big Wins (top 10 by PNL) ──
        big_wins = []
        for pick, runner, race, meeting in settled_by_pnl:
            if not pick.hit or not pick.pnl or pick.pnl <= 0:
                continue
            big_wins.append(_build_win_card(pick, runner, race, meeting, get_celebration))
            if len(big_wins) >= 10:
                break

        # ── P&L Timeline (settled picks ordered by settled_at for chart) ──
        timeline = []
        settled_chrono = sorted(settled_rows, key=lambda x: x[0].settled_at or x[0].created_at)
        running_pnl = 0.0
        for pick, runner, race, meeting in settled_chrono:
            pnl = pick.pnl or 0
            running_pnl += pnl
            # Use race start time or settled_at for x-axis
            ts = (race.start_time if race else None) or pick.settled_at or pick.created_at
            rn = pick.race_number or pick.sequence_start_race or "?"
            timeline.append({
                "time": ts.strftime("%H:%M") if ts else "?",
                "pnl": round(pnl, 2),
                "cumulative": round(running_pnl, 2),
                "label": f"{meeting.venue} R{rn}",
                "hit": bool(pick.hit),
            })

        # ── Upcoming bets (unsettled, race not finished) ──
        upcoming = []
        for pick, runner, race, meeting in unsettled_rows:
            if pick.pick_type not in ("selection", "exotic", "sequence", "big3_multi"):
                continue
            # Race status
            rs = ((race.results_status if race else None) or "").lower()
            if rs in ("paying", "closed", "final"):
                continue  # Already finished, just not settled yet — skip
            # Display name
            if pick.pick_type == "selection":
                name = pick.horse_name or "Runner"
            elif pick.pick_type == "exotic":
                name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"
            elif pick.pick_type == "sequence":
                name = f"{(pick.sequence_variant or '').title()} {pick.sequence_type or 'Sequence'}"
            else:
                name = pick.horse_name or f"R{pick.race_number}"

            stake = pick.bet_stake or pick.exotic_stake or 0
            # Probability: use place_prob for Place bets, win_prob otherwise
            wp = round(pick.win_probability * 100, 1) if pick.win_probability else None
            pp = round(pick.place_probability * 100, 1) if pick.place_probability else None
            bt_lower = (pick.bet_type or "").lower()
            show_prob = pp if bt_lower == "place" and pp else wp

            # Exotic runners as flat list
            exotic_runners = None
            if pick.pick_type == "exotic" and pick.exotic_runners:
                import json as _json2
                try:
                    exotic_runners = _json2.loads(pick.exotic_runners) if isinstance(pick.exotic_runners, str) else pick.exotic_runners
                except (ValueError, TypeError):
                    exotic_runners = None
                # Flatten nested arrays (e.g. [[5], [8, 2, 1]] → [5, 8, 2, 1])
                if exotic_runners and isinstance(exotic_runners, list) and any(isinstance(r, list) for r in exotic_runners):
                    exotic_runners = [item for sub in exotic_runners for item in (sub if isinstance(sub, list) else [sub])]

            upcoming.append({
                "name": name,
                "saddlecloth": pick.saddlecloth,
                "venue": meeting.venue,
                "meeting_id": meeting.id,
                "race_number": pick.race_number,
                "pick_type": pick.pick_type,
                "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                "exotic_type": pick.exotic_type,
                "exotic_runners": exotic_runners,
                "odds": pick.odds_at_tip,
                "stake": round(stake, 2),
                "tip_rank": pick.tip_rank,
                "value_rating": round(pick.value_rating, 2) if pick.value_rating else None,
                "win_prob": wp,
                "place_prob": pp,
                "show_prob": show_prob,
                "confidence": pick.confidence,
                "is_puntys_pick": pick.is_puntys_pick or False,
                "start_time": race.start_time.strftime("%H:%M") if race and race.start_time else None,
                "is_edge": (pick.value_rating or 0) >= 1.1 and pick.pick_type == "selection",
            })

        # ── Edge picks (upcoming selections with value_rating >= 1.1) ──
        edge_picks = [u for u in upcoming if u.get("is_edge")]
        edge_picks.sort(key=lambda x: x.get("value_rating") or 0, reverse=True)

        # ── Bet type breakdown (all settled picks, with ROI/SR) ──
        bt_stats = defaultdict(lambda: {"bets": 0, "winners": 0, "staked": 0.0, "pnl": 0.0})
        for pick, runner, race, meeting in settled_rows:
            # Label by pick type: selections use bet_type, exotics use exotic_type, sequences use sequence_type
            if pick.pick_type == "selection":
                bt = (pick.bet_type or "unknown").replace("_", " ").title()
            elif pick.pick_type == "exotic":
                bt = (pick.exotic_type or "Exotic").replace("_", " ").title()
            elif pick.pick_type in ("sequence", "big3_multi"):
                bt = (pick.sequence_type or pick.pick_type or "Sequence").replace("_", " ").title()
            else:
                bt = (pick.pick_type or "unknown").replace("_", " ").title()
            bt_stats[bt]["bets"] += 1
            stake = pick.exotic_stake if pick.pick_type == "exotic" else pick.bet_stake
            bt_stats[bt]["staked"] += stake or 0
            bt_stats[bt]["pnl"] += pick.pnl or 0
            if pick.hit:
                bt_stats[bt]["winners"] += 1
        bet_types = []
        for bt, data in sorted(bt_stats.items()):
            data["strike_rate"] = round(data["winners"] / data["bets"] * 100, 1) if data["bets"] else 0
            data["roi"] = round(data["pnl"] / data["staked"] * 100, 1) if data["staked"] else 0
            data["name"] = bt
            data["staked"] = round(data["staked"], 2)
            data["pnl"] = round(data["pnl"], 2)
            bet_types.append(data)

        # ── Insights ──
        insights = []
        winning_selections = [
            (p, r, rc, m) for p, r, rc, m in settled_rows
            if p.hit and p.pnl and p.pnl > 0 and p.pick_type == "selection"
        ]
        all_selections = [
            (p, r, rc, m) for p, r, rc, m in settled_rows if p.pick_type == "selection"
        ]

        # Hot Jockey
        jockey_wins = Counter()
        jockey_rides = Counter()
        for pick, runner, race, meeting in all_selections:
            j = runner.jockey if runner else None
            if j:
                jockey_rides[j] += 1
                if pick.hit:
                    jockey_wins[j] += 1
        if jockey_wins:
            hot_j, hot_j_wins = jockey_wins.most_common(1)[0]
            hot_j_rides = jockey_rides[hot_j]
            parts = hot_j.split()
            short_j = f"{parts[0][0]} {' '.join(parts[1:])}" if len(parts) > 1 else hot_j
            insights.append({
                "icon": "fire", "label": "Hot Jockey",
                "text": f"{short_j} ({hot_j_wins}/{hot_j_rides} winners)",
            })

        # Hot Trainer
        trainer_wins = Counter()
        trainer_runs = Counter()
        for pick, runner, race, meeting in all_selections:
            t = runner.trainer if runner else None
            if t:
                trainer_runs[t] += 1
                if pick.hit:
                    trainer_wins[t] += 1
        if trainer_wins:
            hot_t, hot_t_wins = trainer_wins.most_common(1)[0]
            hot_t_runs = trainer_runs[hot_t]
            parts = hot_t.split()
            short_t = f"{parts[0][0]} {' '.join(parts[1:])}" if len(parts) > 1 else hot_t
            insights.append({
                "icon": "trophy", "label": "Hot Trainer",
                "text": f"{short_t} ({hot_t_wins}/{hot_t_runs} winners)",
            })

        # Best Odds Win
        if winning_selections:
            best_odds_win = max(winning_selections, key=lambda x: x[0].odds_at_tip or 0)
            bo_pick = best_odds_win[0]
            if bo_pick.odds_at_tip and bo_pick.odds_at_tip >= 4.0:
                insights.append({
                    "icon": "zap", "label": "Best Odds",
                    "text": f"{bo_pick.horse_name} at ${bo_pick.odds_at_tip:.2f} saluted",
                })

        # Frontrunners vs closers
        pace_wins = Counter()
        for pick, runner, race, meeting in winning_selections:
            smp = (runner.speed_map_position if runner else "") or ""
            if smp.lower() in ("leader", "on_pace"):
                pace_wins["front"] += 1
            elif smp.lower() in ("midfield", "backmarker"):
                pace_wins["back"] += 1
        total_pace = pace_wins["front"] + pace_wins["back"]
        if total_pace >= 3:
            insights.append({
                "icon": "horse", "label": "Pace",
                "text": f"{pace_wins['front']}/{total_pace} winners led or sat on pace",
            })

        # Best Venue
        venue_pnl = Counter()
        venue_bets = Counter()
        for pick, runner, race, meeting in settled_rows:
            venue_bets[meeting.venue] += 1
            venue_pnl[meeting.venue] += pick.pnl or 0
        if venue_pnl:
            best_venue = max(venue_pnl, key=venue_pnl.get)
            bv_pnl = venue_pnl[best_venue]
            bv_bets = venue_bets[best_venue]
            if bv_pnl > 0:
                insights.append({
                    "icon": "pin", "label": "Best Venue",
                    "text": f"{best_venue} +${bv_pnl:.0f} from {bv_bets} bets",
                })

        # Best bet type
        if bet_types:
            best_bt = max(bet_types, key=lambda x: x["pnl"])
            if best_bt["pnl"] > 0:
                insights.append({
                    "icon": "chart", "label": "Bet Type",
                    "text": f"{best_bt['name']} bets leading today ({best_bt['roi']:+.0f}% ROI)",
                })

        # ── Venue breakdown ──
        venues = []
        # Include unsettled counts too
        venue_upcoming = Counter()
        for pick, runner, race, meeting in unsettled_rows:
            venue_upcoming[meeting.venue] += 1
        all_venue_names = set(list(venue_bets.keys()) + list(venue_upcoming.keys()))
        for v in sorted(all_venue_names):
            v_wins = sum(1 for p, r, rc, m in settled_rows if m.venue == v and p.hit and p.pnl and p.pnl > 0)
            v_total = venue_bets.get(v, 0)
            v_pnl = venue_pnl.get(v, 0)
            v_up = venue_upcoming.get(v, 0)
            # Get meeting_id for cross-link
            v_mid = None
            for p, r, rc, m in all_rows:
                if m.venue == v:
                    v_mid = m.id
                    break
            venues.append({
                "name": v,
                "meeting_id": v_mid,
                "bets": v_total,
                "winners": v_wins,
                "upcoming": v_up,
                "pnl": round(v_pnl, 2),
            })

        # ── Summary counts for has_data logic ──
        total_picks = len(all_rows)
        total_settled = len(settled_rows)
        total_upcoming = len(unsettled_rows)

        return {
            "date": today.strftime("%d %B %Y").lstrip("0"),
            "date_iso": today.isoformat(),
            "performance": performance,
            "big_wins": big_wins,
            "timeline": timeline,
            "upcoming": upcoming[:30],  # Cap at 30 for page size
            "edge_picks": edge_picks[:8],
            "bet_types": bet_types,
            "insights": insights,
            "venues": venues,
            "total_picks": total_picks,
            "total_settled": total_settled,
            "total_upcoming": total_upcoming,
            "has_data": total_picks > 0,
            "has_settled": total_settled > 0,
        }


def _build_win_card(pick, runner, race, meeting, get_celebration) -> dict:
    """Build a win card dict for the big wins section."""
    stake = pick.bet_stake or pick.exotic_stake or 1.0
    returned = stake + (pick.pnl or 0)

    if pick.pick_type == "selection":
        display_name = pick.horse_name or "Runner"
    elif pick.pick_type == "exotic":
        display_name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"
    elif pick.pick_type == "sequence":
        display_name = f"{(pick.sequence_variant or '').title()} {pick.sequence_type or 'Sequence'}"
    elif pick.pick_type == "big3_multi":
        display_name = "Big 3 Multi"
    else:
        display_name = pick.horse_name or f"R{pick.race_number}"

    smp = (runner.speed_map_position if runner else None) or ""
    running_style = {
        "leader": "Led all the way", "on_pace": "Sat on pace",
        "midfield": "Settled midfield", "backmarker": "Came from behind", "": None,
    }.get(smp.lower(), smp.title() if smp else None)

    margin = runner.result_margin if runner else None
    if margin and pick.pick_type == "selection":
        margin_text = f"Won by {margin}"
    elif pick.pick_type == "exotic":
        margin_text = f"Paid ${returned:.2f}"
    else:
        margin_text = None

    return {
        "display_name": display_name,
        "venue": meeting.venue,
        "meeting_id": meeting.id,
        "race_number": pick.race_number,
        "race_name": race.name if race else None,
        "jockey": runner.jockey if runner else None,
        "trainer": runner.trainer if runner else None,
        "odds": pick.odds_at_tip,
        "bet_type": (pick.bet_type or "").replace("_", " ").title(),
        "stake": round(stake, 2),
        "returned": round(returned, 2),
        "pnl": round(pick.pnl, 2),
        "tip_rank": pick.tip_rank,
        "pick_type": pick.pick_type,
        "running_style": running_style,
        "margin_text": margin_text,
        "celebration": get_celebration(pick.pnl, pick.pick_type),
        "is_puntys_pick": pick.is_puntys_pick or False,
    }


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Today's daily dashboard — celebrating wins, insights, and performance."""
    dashboard = await get_daily_dashboard()
    return templates.TemplateResponse(
        "stats.html",
        {"request": request, **dashboard},
    )


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

    # Find next upcoming race time for sidebar countdown
    next_race_time = None
    from datetime import datetime
    now = datetime.now()
    for r in data.get("races", []):
        if r.get("time") and r.get("status") not in ("Paying", "Closed", "Final"):
            try:
                race_dt = datetime.strptime(
                    data["meeting"].get("date", "") + "T" + r["time"],
                    "%Y-%m-%dT%H:%M"
                )
                if race_dt > now:
                    next_race_time = race_dt.strftime("%Y-%m-%dT%H:%M:00+11:00")
                    break
            except (ValueError, TypeError):
                pass

    response = templates.TemplateResponse(
        "meeting_tips.html",
        {
            "request": request,
            "meeting": data["meeting"],
            "early_mail": data["early_mail"],
            "wrapup": data["wrapup"],
            "winners": data.get("winners", {}),
            "winning_exotics": data.get("winning_exotics", {}),
            "losing_exotics": data.get("losing_exotics", {}),
            "winning_sequences": data.get("winning_sequences", []),
            "pp_picks": data.get("pp_picks", {}),
            "sequence_results": data.get("sequence_results", []),
            "live_updates": data.get("live_updates", []),
            "venue_stats": data.get("venue_stats"),
            "races": data.get("races", []),
            "meeting_stats": data.get("meeting_stats", []),
            "same_day_meetings": data.get("same_day_meetings", []),
            "scratched_picks": data.get("scratched_picks", {}),
            "alternatives": data.get("alternatives", {}),
            "next_race_time": next_race_time,
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
