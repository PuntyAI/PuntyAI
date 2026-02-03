"""Web routes for Group One Glory."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sqlalchemy import select, func

from punty.config import melb_now, MELB_TZ
from punty.models.database import get_db
from punty.models.glory import G1User
from punty.glory.auth import (
    get_current_user,
    require_user,
    require_admin,
    generate_csrf_token,
)
from punty.glory.services.competition import CompetitionService
from punty.glory.services.race import RaceService
from punty.glory.services.pick import PickService
from punty.glory.services.leaderboard import LeaderboardService


router = APIRouter(prefix="/group1glory", redirect_slashes=False)

# Templates directory (parent folder so 'glory/base.html' resolves correctly)
templates_dir = Path(__file__).parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=templates_dir)
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else {}


def _melb(dt, fmt='%H:%M'):
    """Format datetime to Melbourne timezone."""
    if dt is None:
        return ''
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MELB_TZ).strftime(fmt)


def _melb_date(dt, fmt='%a %d %b'):
    """Format date to Melbourne timezone with day name."""
    if dt is None:
        return ''
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(MELB_TZ)
    return dt.strftime(fmt)


templates.env.filters["melb"] = _melb
templates.env.filters["melb_date"] = _melb_date


def _base_context(request: Request, db: AsyncSession = None) -> dict:
    """Get base template context with user and CSRF token."""
    user = get_current_user(request)
    return {
        "request": request,
        "user": user,
        "csrf_token": generate_csrf_token(request),
        "now": melb_now,
    }


# --- Setup Route (for creating first admin) ---


@router.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Setup page for creating the first admin user."""
    import os

    # Check if any users exist
    result = await db.execute(select(func.count(G1User.id)))
    user_count = result.scalar() or 0

    # Check if setup key is configured
    setup_key_configured = bool(os.environ.get("G1_SETUP_KEY", ""))

    return templates.TemplateResponse(
        "glory/setup.html",
        {
            "request": request,
            "already_setup": user_count > 0 and not setup_key_configured,
            "need_setup_key": user_count > 0 and setup_key_configured,
            "error": request.query_params.get("error"),
        },
    )


# --- Public Routes ---


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, db: AsyncSession = Depends(get_db)):
    """Home page."""
    context = _base_context(request)

    # Get active competition
    comp_service = CompetitionService(db)
    competition = await comp_service.get_active_competition(include_races=True)

    if competition:
        # Get leaderboard preview
        lb_service = LeaderboardService(db)
        top_3 = await lb_service.get_top_3(competition.id)
        context["top_3"] = top_3
        context["competition"] = competition

        # Get upcoming races
        race_service = RaceService(db)
        upcoming_races = [
            r for r in competition.races
            if r.status in ("open", "nominations", "final_field")
        ]
        upcoming_races.sort(key=lambda r: r.race_date)
        context["upcoming_races"] = upcoming_races[:5]

    return templates.TemplateResponse("glory/home.html", context)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/group1glory/picks", status_code=302)

    error = request.query_params.get("error")
    return templates.TemplateResponse(
        "glory/login.html",
        {"request": request, "error": error, "csrf_token": generate_csrf_token(request)},
    )


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/group1glory/picks", status_code=302)

    error = request.query_params.get("error")
    return templates.TemplateResponse(
        "glory/register.html",
        {"request": request, "error": error, "csrf_token": generate_csrf_token(request)},
    )


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About/rules page."""
    context = _base_context(request)
    return templates.TemplateResponse("glory/about.html", context)


@router.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    """Terms and conditions page."""
    context = _base_context(request)
    return templates.TemplateResponse("glory/terms.html", context)


@router.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    """Contact us page."""
    context = _base_context(request)
    return templates.TemplateResponse("glory/contact.html", context)


# --- Authenticated Routes ---


@router.get("/picks", response_class=HTMLResponse)
async def picks_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Pick Your Horses page."""
    user = require_user(request)
    context = _base_context(request)

    # Get active competition with races and horses
    comp_service = CompetitionService(db)
    competition = await comp_service.get_active_competition(include_races=True)

    if not competition:
        return templates.TemplateResponse("glory/picks.html", context)

    context["competition"] = competition

    # Group races by date
    race_service = RaceService(db)
    races_by_date = await race_service.get_races_grouped_by_date(competition.id)
    context["races_by_date"] = races_by_date

    # Get user's picks
    pick_service = PickService(db)
    user_picks = await pick_service.get_user_picks_for_competition(
        user["id"], competition.id
    )
    # Create lookup by race_id
    picks_by_race = {p.race_id: p for p in user_picks}
    context["picks_by_race"] = picks_by_race

    # Get pick summary
    pick_summary = await pick_service.get_pick_summary_for_user(
        user["id"], competition.id
    )
    context["pick_summary"] = pick_summary

    return templates.TemplateResponse("glory/picks.html", context)


@router.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Leaderboard page."""
    user = require_user(request)
    context = _base_context(request)

    # Get active competition
    comp_service = CompetitionService(db)
    competition = await comp_service.get_active_competition(include_races=True)

    if not competition:
        return templates.TemplateResponse("glory/leaderboard.html", context)

    context["competition"] = competition

    # Get competition stats
    stats = await comp_service.get_competition_stats(competition.id)
    context["stats"] = stats

    # Get full leaderboard
    lb_service = LeaderboardService(db)
    leaderboard = await lb_service.get_leaderboard(competition.id)
    context["leaderboard"] = leaderboard

    # Get current user's rank
    user_rank = await lb_service.get_user_rank(user["id"], competition.id)
    context["user_rank"] = user_rank

    return templates.TemplateResponse("glory/leaderboard.html", context)


# --- Admin Routes ---


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    """Admin dashboard."""
    user = require_admin(request)
    context = _base_context(request)

    # Get all competitions
    comp_service = CompetitionService(db)
    competitions = await comp_service.list_competitions()
    context["competitions"] = competitions

    # Get active competition stats
    active_comp = await comp_service.get_active_competition(include_races=True)
    if active_comp:
        stats = await comp_service.get_competition_stats(active_comp.id)
        context["active_competition"] = active_comp
        context["stats"] = stats

    return templates.TemplateResponse("glory/admin/dashboard.html", context)


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request, db: AsyncSession = Depends(get_db)):
    """Manage users."""
    user = require_admin(request)
    context = _base_context(request)

    # Get all users
    result = await db.execute(
        select(G1User).order_by(G1User.created_at.desc())
    )
    users = result.scalars().all()
    context["users"] = [u.to_dict() for u in users]

    return templates.TemplateResponse("glory/admin/users.html", context)


@router.get("/admin/competitions", response_class=HTMLResponse)
async def admin_competitions(request: Request, db: AsyncSession = Depends(get_db)):
    """Manage competitions."""
    user = require_admin(request)
    context = _base_context(request)

    comp_service = CompetitionService(db)
    competitions = await comp_service.list_competitions(include_races=True)
    context["competitions"] = competitions

    return templates.TemplateResponse("glory/admin/competitions.html", context)


@router.get("/admin/competitions/{competition_id}", response_class=HTMLResponse)
async def admin_competition_detail(
    competition_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Competition detail/edit page."""
    user = require_admin(request)
    context = _base_context(request)

    comp_service = CompetitionService(db)
    competition = await comp_service.get_competition(competition_id, include_races=True)

    if not competition:
        return templates.TemplateResponse(
            "glory/admin/error.html",
            {"request": request, "error": "Competition not found"},
            status_code=404,
        )

    context["competition"] = competition

    # Get stats
    stats = await comp_service.get_competition_stats(competition_id)
    context["stats"] = stats

    return templates.TemplateResponse("glory/admin/competition_detail.html", context)


@router.get("/admin/races", response_class=HTMLResponse)
async def admin_races(request: Request, db: AsyncSession = Depends(get_db)):
    """Manage races."""
    user = require_admin(request)
    context = _base_context(request)

    # Get active competition's races
    comp_service = CompetitionService(db)
    competition = await comp_service.get_active_competition(include_races=True)

    if competition:
        context["competition"] = competition

        # Group races by status
        races_by_status = {
            "nominations": [],
            "final_field": [],
            "open": [],
            "closed": [],
            "resulted": [],
        }
        for race in competition.races:
            if race.status in races_by_status:
                races_by_status[race.status].append(race)

        context["races_by_status"] = races_by_status

    return templates.TemplateResponse("glory/admin/races.html", context)


@router.get("/admin/races/{race_id}", response_class=HTMLResponse)
async def admin_race_detail(
    race_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Race detail page with horses and picks."""
    user = require_admin(request)
    context = _base_context(request)

    race_service = RaceService(db)
    race = await race_service.get_race(race_id, include_horses=True)

    if not race:
        return templates.TemplateResponse(
            "glory/admin/error.html",
            {"request": request, "error": "Race not found"},
            status_code=404,
        )

    context["race"] = race

    # Get all picks for this race
    pick_service = PickService(db)
    picks = await pick_service.get_all_picks_for_race(race_id)
    context["picks"] = picks

    return templates.TemplateResponse("glory/admin/race_detail.html", context)


@router.get("/admin/results", response_class=HTMLResponse)
async def admin_results(request: Request, db: AsyncSession = Depends(get_db)):
    """Enter race results."""
    user = require_admin(request)
    context = _base_context(request)

    # Get active competition's races that need results
    comp_service = CompetitionService(db)
    competition = await comp_service.get_active_competition(include_races=True)

    if competition:
        context["competition"] = competition

        # Races that are closed but not resulted
        pending_results = [
            r for r in competition.races
            if r.status == "closed"
        ]
        pending_results.sort(key=lambda r: r.race_date)
        context["pending_results"] = pending_results

        # Recently resulted races
        resulted = [
            r for r in competition.races
            if r.status == "resulted"
        ]
        resulted.sort(key=lambda r: r.race_date, reverse=True)
        context["resulted_races"] = resulted[:10]

    return templates.TemplateResponse("glory/admin/results.html", context)


@router.get("/admin/results/{race_id}", response_class=HTMLResponse)
async def admin_result_entry(
    race_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Result entry form for a specific race."""
    user = require_admin(request)
    context = _base_context(request)

    race_service = RaceService(db)
    race = await race_service.get_race(race_id, include_horses=True)

    if not race:
        return templates.TemplateResponse(
            "glory/admin/error.html",
            {"request": request, "error": "Race not found"},
            status_code=404,
        )

    context["race"] = race

    # Get non-scratched horses for result entry
    horses = [h for h in race.horses if not h.is_scratched]
    horses.sort(key=lambda h: (h.saddlecloth or 99, h.barrier or 99, h.name))
    context["horses"] = horses

    return templates.TemplateResponse("glory/admin/result_entry.html", context)


@router.get("/admin/picks", response_class=HTMLResponse)
async def admin_picks(request: Request, db: AsyncSession = Depends(get_db)):
    """View all user picks."""
    user = require_admin(request)
    context = _base_context(request)

    # Get active competition
    comp_service = CompetitionService(db)
    competition = await comp_service.get_active_competition(include_races=True)

    if competition:
        context["competition"] = competition

        # Get all resulted races
        resulted_races = [
            r for r in competition.races
            if r.status == "resulted"
        ]
        resulted_races.sort(key=lambda r: r.race_date, reverse=True)

        # For each race, get all picks
        pick_service = PickService(db)
        races_with_picks = []
        for race in resulted_races[:10]:
            picks = await pick_service.get_all_picks_for_race(race.id)
            races_with_picks.append({
                "race": race,
                "picks": picks,
            })

        context["races_with_picks"] = races_with_picks

    return templates.TemplateResponse("glory/admin/picks.html", context)
