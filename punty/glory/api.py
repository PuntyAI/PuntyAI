"""API endpoints for Group One Glory."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select, func, update

from punty.models.database import get_db
from punty.models.glory import G1User
from punty.glory.auth import (
    get_current_user,
    require_user,
    require_admin,
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_id,
    login_user,
    logout_user,
)
from punty.glory.services.competition import CompetitionService
from punty.glory.services.race import RaceService
from punty.glory.services.pick import PickService
from punty.glory.services.leaderboard import LeaderboardService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/group1glory/api", redirect_slashes=False)


# --- Pydantic Models ---


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class PickRequest(BaseModel):
    race_id: str
    horse_id: str


class CompetitionCreate(BaseModel):
    name: str
    start_date: str  # ISO format
    end_date: str
    prize_pool: Optional[int] = None


class RaceCreate(BaseModel):
    competition_id: str
    race_name: str
    venue: str
    race_date: str  # ISO format datetime
    distance: int
    prize_money: Optional[int] = None
    race_number: Optional[int] = None


class RaceStatusUpdate(BaseModel):
    status: str
    tipping_closes_at: Optional[str] = None


class ResultEntry(BaseModel):
    results: list[tuple[str, int]]  # [(horse_id, position), ...]


class SetupAdminRequest(BaseModel):
    email: str
    password: str
    display_name: str
    setup_key: str  # Must match G1_SETUP_KEY env var


class PromoteUserRequest(BaseModel):
    user_id: str
    is_admin: bool


# --- Setup Endpoint (First-Run Only) ---


@router.post("/setup")
async def setup_admin(
    request: Request,
    data: SetupAdminRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create the first admin user. Only works when no users exist OR with valid setup key.

    The setup_key must match the G1_SETUP_KEY environment variable.
    If no setup key is configured, this endpoint only works when there are no users.
    """
    import os

    # Check setup key
    setup_key = os.environ.get("G1_SETUP_KEY", "")

    # Count existing users
    result = await db.execute(select(func.count(G1User.id)))
    user_count = result.scalar() or 0

    # If users exist, require valid setup key
    if user_count > 0:
        if not setup_key or data.setup_key != setup_key:
            raise HTTPException(
                status_code=403,
                detail="Setup is only available before any users are created, or with a valid setup key",
            )

    # Check if email already exists
    existing = await get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists",
        )

    # Validate password
    if len(data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters",
        )

    # Create admin user
    user = await create_user(
        db,
        email=data.email,
        password=data.password,
        display_name=data.display_name,
        is_admin=True,  # Setup creates admin
    )

    # Log them in
    login_user(request, user)

    logger.info(f"Admin user created via setup: {data.email}")
    return {"status": "ok", "user": user.to_dict(), "message": "Admin account created successfully"}


# --- Auth Endpoints ---


@router.post("/auth/register")
async def register(
    request: Request,
    data: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user."""
    # Check if email already exists
    existing = await get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists",
        )

    # Validate password
    if len(data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters",
        )

    # Validate display name
    if len(data.display_name) < 2:
        raise HTTPException(
            status_code=400,
            detail="Display name must be at least 2 characters",
        )

    # Create user
    user = await create_user(
        db,
        email=data.email,
        password=data.password,
        display_name=data.display_name,
    )

    # Log them in
    login_user(request, user)

    return {"status": "ok", "user": user.to_dict()}


@router.post("/auth/login")
async def login(
    request: Request,
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Log in a user."""
    user = await authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    login_user(request, user)
    return {"status": "ok", "user": user.to_dict()}


@router.post("/auth/logout")
async def logout(request: Request):
    """Log out the current user."""
    logout_user(request)
    return {"status": "ok"}


@router.get("/auth/me")
async def get_me(request: Request):
    """Get the current user's info."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# --- Competition Endpoints ---


@router.get("/competitions")
async def list_competitions(db: AsyncSession = Depends(get_db)):
    """List all competitions."""
    service = CompetitionService(db)
    competitions = await service.list_competitions()
    return [c.to_dict() for c in competitions]


@router.get("/competitions/active")
async def get_active_competition(db: AsyncSession = Depends(get_db)):
    """Get the currently active competition."""
    service = CompetitionService(db)
    competition = await service.get_active_competition(include_races=True)
    if not competition:
        return None
    return competition.to_dict(include_races=True)


@router.get("/competitions/{competition_id}")
async def get_competition(
    competition_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific competition."""
    service = CompetitionService(db)
    competition = await service.get_competition(competition_id, include_races=True)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    return competition.to_dict(include_races=True)


@router.post("/admin/competitions")
async def create_competition(
    request: Request,
    data: CompetitionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new competition (admin only)."""
    require_admin(request)

    service = CompetitionService(db)
    competition = await service.create_competition(
        name=data.name,
        start_date=datetime.fromisoformat(data.start_date).date(),
        end_date=datetime.fromisoformat(data.end_date).date(),
        prize_pool=data.prize_pool,
    )
    return competition.to_dict()


@router.post("/admin/competitions/{competition_id}/activate")
async def activate_competition(
    competition_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Activate a competition (admin only)."""
    require_admin(request)

    service = CompetitionService(db)
    competition = await service.activate_competition(competition_id)
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    return competition.to_dict()


@router.delete("/admin/competitions/{competition_id}")
async def delete_competition(
    competition_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Delete a competition (admin only)."""
    require_admin(request)

    service = CompetitionService(db)
    deleted = await service.delete_competition(competition_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Competition not found")
    return {"status": "ok"}


# --- Race Endpoints ---


@router.get("/races")
async def list_races(
    competition_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List races, optionally filtered by competition."""
    if not competition_id:
        # Get active competition
        comp_service = CompetitionService(db)
        competition = await comp_service.get_active_competition()
        if not competition:
            return []
        competition_id = competition.id

    service = RaceService(db)
    races = await service.list_races_by_competition(competition_id, include_horses=True)
    return [r.to_dict(include_horses=True) for r in races]


@router.get("/races/{race_id}")
async def get_race(
    race_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific race with horses."""
    service = RaceService(db)
    race = await service.get_race(race_id, include_horses=True)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
    return race.to_dict(include_horses=True)


@router.get("/races/{race_id}/horses")
async def get_horses(
    race_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get horses for a race."""
    service = RaceService(db)
    horses = await service.list_horses(race_id)
    return [h.to_dict() for h in horses]


@router.post("/admin/races")
async def create_race(
    request: Request,
    data: RaceCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new race (admin only)."""
    require_admin(request)

    service = RaceService(db)
    race = await service.create_race(
        competition_id=data.competition_id,
        race_name=data.race_name,
        venue=data.venue,
        race_date=datetime.fromisoformat(data.race_date),
        distance=data.distance,
        prize_money=data.prize_money,
        race_number=data.race_number,
    )
    return race.to_dict()


@router.put("/admin/races/{race_id}/status")
async def update_race_status(
    race_id: str,
    request: Request,
    data: RaceStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update race status (admin only)."""
    require_admin(request)

    service = RaceService(db)
    tipping_closes = None
    if data.tipping_closes_at:
        tipping_closes = datetime.fromisoformat(data.tipping_closes_at)

    race = await service.update_race_status(
        race_id,
        status=data.status,
        tipping_closes_at=tipping_closes,
    )
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
    return race.to_dict()


@router.post("/admin/races/{race_id}/import-horses")
async def import_horses(
    race_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Import horses from Racing.com for a race (admin only)."""
    require_admin(request)

    race_service = RaceService(db)
    race = await race_service.get_race(race_id)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    # Import from Racing.com scraper
    from punty.scrapers.racing_com import RacingComScraper

    scraper = RacingComScraper()
    try:
        data = await scraper.scrape_meeting(race.venue, race.race_date.date())
    except Exception as e:
        logger.error(f"Failed to scrape Racing.com: {e}")
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")

    # Find the matching race by name or number
    matching_race = None
    for scraped_race in data.get("races", []):
        if (
            scraped_race.get("race_number") == race.race_number
            or race.race_name.lower() in scraped_race.get("name", "").lower()
        ):
            matching_race = scraped_race
            break

    if not matching_race:
        raise HTTPException(
            status_code=404,
            detail="Could not find matching race in scraped data",
        )

    # Clear existing horses
    await race_service.clear_horses(race_id)

    # Get runners for this race
    race_runners = [
        r for r in data.get("runners", [])
        if r.get("race_id") == matching_race.get("id")
    ]

    # Add horses
    horses_added = 0
    for runner in race_runners:
        await race_service.add_horse(
            race_id=race_id,
            name=runner.get("horse_name"),
            barrier=runner.get("barrier"),
            weight=runner.get("weight"),
            jockey=runner.get("jockey"),
            trainer=runner.get("trainer"),
            odds=runner.get("current_odds") or runner.get("odds_tab"),
            form=runner.get("form"),
            career_record=runner.get("career_record"),
            career_prize=runner.get("career_prize_money"),
            saddlecloth=runner.get("saddlecloth"),
            horse_age=runner.get("horse_age"),
            horse_sex=runner.get("horse_sex"),
            sire=runner.get("sire"),
            dam=runner.get("dam"),
            last_five=runner.get("last_five"),
            comment=runner.get("comment_short"),
        )
        horses_added += 1

    return {"status": "ok", "horses_added": horses_added}


@router.delete("/admin/races/{race_id}")
async def delete_race(
    race_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Delete a race (admin only)."""
    require_admin(request)

    service = RaceService(db)
    deleted = await service.delete_race(race_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Race not found")
    return {"status": "ok"}


# --- Results Endpoints ---


@router.post("/admin/results/{race_id}")
async def enter_results(
    race_id: str,
    request: Request,
    data: ResultEntry,
    db: AsyncSession = Depends(get_db),
):
    """Enter race results (admin only)."""
    require_admin(request)

    race_service = RaceService(db)

    try:
        race = await race_service.enter_results(race_id, data.results)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Calculate points for all picks
    pick_service = PickService(db)
    points = await pick_service.calculate_points_for_race(race_id)

    return {
        "status": "ok",
        "race": race.to_dict(),
        "points_awarded": points,
    }


# --- Pick Endpoints ---


@router.post("/picks")
async def submit_pick(
    request: Request,
    data: PickRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a pick for a race."""
    user = require_user(request)

    service = PickService(db)
    try:
        pick = await service.submit_pick(
            user_id=user["id"],
            race_id=data.race_id,
            horse_id=data.horse_id,
        )
        return pick.to_dict(include_horse=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/picks")
async def get_my_picks(
    request: Request,
    competition_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get the current user's picks."""
    user = require_user(request)

    if not competition_id:
        comp_service = CompetitionService(db)
        competition = await comp_service.get_active_competition()
        if not competition:
            return []
        competition_id = competition.id

    service = PickService(db)
    picks = await service.get_user_picks_for_competition(user["id"], competition_id)
    return [p.to_dict(include_horse=True) for p in picks]


@router.get("/picks/{race_id}")
async def get_my_pick_for_race(
    race_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get the current user's pick for a specific race."""
    user = require_user(request)

    service = PickService(db)
    pick = await service.get_user_pick_for_race(user["id"], race_id)
    if not pick:
        return None
    return pick.to_dict(include_horse=True)


@router.delete("/picks/{race_id}")
async def delete_pick(
    race_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Delete a pick for a race (if still open)."""
    user = require_user(request)

    service = PickService(db)
    deleted = await service.delete_pick(user["id"], race_id)
    if not deleted:
        raise HTTPException(
            status_code=400,
            detail="Could not delete pick - race may be closed",
        )
    return {"status": "ok"}


# --- Leaderboard Endpoints ---


@router.get("/leaderboard")
async def get_leaderboard(
    competition_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get the leaderboard for a competition."""
    if not competition_id:
        comp_service = CompetitionService(db)
        competition = await comp_service.get_active_competition()
        if not competition:
            return []
        competition_id = competition.id

    service = LeaderboardService(db)
    entries = await service.get_leaderboard(competition_id)
    return [
        {
            "rank": e.rank,
            "user_id": e.user_id,
            "display_name": e.display_name,
            "total_points": e.total_points,
            "races_picked": e.races_picked,
            "winners": e.winners,
            "places": e.places,
            "points_behind": e.points_behind,
        }
        for e in entries
    ]


@router.get("/honor-board")
async def get_honor_board(db: AsyncSession = Depends(get_db)):
    """Get the honor board of past champions."""
    service = LeaderboardService(db)
    return await service.get_honor_board()


# --- Admin User Management ---


@router.get("/admin/users")
async def list_users(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """List all users (admin only)."""
    require_admin(request)

    result = await db.execute(
        select(G1User).order_by(G1User.created_at.desc())
    )
    users = result.scalars().all()
    return [u.to_dict() for u in users]


@router.put("/admin/users/{user_id}/admin")
async def set_user_admin_status(
    user_id: str,
    request: Request,
    data: PromoteUserRequest,
    db: AsyncSession = Depends(get_db),
):
    """Promote or demote a user to/from admin (admin only)."""
    current_user = require_admin(request)

    # Prevent demoting yourself
    if user_id == current_user["id"] and not data.is_admin:
        raise HTTPException(
            status_code=400,
            detail="You cannot remove your own admin status",
        )

    # Find the user
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update admin status
    user.is_admin = data.is_admin
    await db.commit()
    await db.refresh(user)

    action = "promoted to admin" if data.is_admin else "demoted from admin"
    logger.info(f"User {user.email} {action} by {current_user['email']}")

    return {"status": "ok", "user": user.to_dict()}


# --- Calendar Scraper Endpoint ---


@router.post("/admin/scrape-calendar")
async def scrape_calendar(
    request: Request,
    year: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """Scrape Racing Australia Group 1 calendar (admin only)."""
    require_admin(request)

    from punty.glory.services.calendar_scraper import RacingAustraliaCalendarScraper

    scraper = RacingAustraliaCalendarScraper()
    if year is None:
        year = datetime.now().year

    try:
        races = await scraper.scrape_group1_calendar(year)
        return {"status": "ok", "races_found": len(races), "races": races}
    except Exception as e:
        logger.error(f"Calendar scrape failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")


@router.post("/admin/import-calendar")
async def import_calendar_races(
    request: Request,
    competition_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Import scraped calendar races into a competition (admin only)."""
    require_admin(request)

    from punty.glory.services.calendar_scraper import RacingAustraliaCalendarScraper

    scraper = RacingAustraliaCalendarScraper()
    year = datetime.now().year

    try:
        races = await scraper.scrape_group1_calendar(year)
    except Exception as e:
        logger.error(f"Calendar scrape failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")

    # Import races into competition
    race_service = RaceService(db)
    imported = 0

    for race_data in races:
        await race_service.create_race(
            competition_id=competition_id,
            race_name=race_data["race_name"],
            venue=race_data["venue"],
            race_date=race_data["race_date"],
            distance=race_data["distance"],
            prize_money=race_data.get("prize_money"),
            external_id=race_data.get("external_id"),
        )
        imported += 1

    return {"status": "ok", "races_imported": imported}
