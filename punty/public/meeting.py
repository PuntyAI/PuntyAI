"""Meeting detail page route."""

from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func, and_, or_, case

from punty.config import melb_today
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race, Runner
from punty.models.live_update import LiveUpdate

from punty.public.deps import templates
from punty.public.dashboard import _compute_pick_data

router = APIRouter()


async def get_meeting_tips(meeting_id: str) -> dict | None:
    """Get early mail and wrap-up content for a specific meeting."""
    async with async_session() as db:
        # Get meeting
        meeting_result = await db.execute(
            select(Meeting).where(Meeting.id == meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        if not meeting or not meeting.selected:
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
            ).order_by(LiveUpdate.created_at.desc())
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
        venue_stats = None
        if meeting.venue:
            venue_meetings = select(Meeting.id).where(
                Meeting.venue == meeting.venue
            )
            venue_result = await db.execute(
                select(
                    func.count(Pick.id),
                    func.sum(case((Pick.hit == True, 1), else_=0)),
                    func.sum(Pick.pnl),
                    func.sum(
                        case(
                            (Pick.pick_type == "exotic", Pick.exotic_stake),
                            else_=Pick.bet_stake,
                        )
                    ),
                    func.count(func.distinct(Pick.meeting_id)),
                ).where(and_(
                    Pick.settled == True,
                    Pick.pick_type.in_(["selection", "exotic", "sequence"]),
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
    meeting_finished = False
    from punty.config import melb_now_naive
    now = melb_now_naive()
    races = data.get("races", [])
    has_unfinished = False
    for r in races:
        status = r.get("status") or ""
        if status in ("Paying", "Closed", "Final"):
            continue
        has_unfinished = True
        if r.get("time"):
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
    if races and not has_unfinished:
        meeting_finished = True

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
            "meeting_finished": meeting_finished,
            "meta_title": meta_title,
            "meta_description": meta_desc,
        }
    )
    response.headers["Cache-Control"] = "public, max-age=300"
    return response
