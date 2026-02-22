"""FastAPI application entry point for PuntyAI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from punty.config import settings
from punty.auth import AuthMiddleware, CSRFMiddleware, router as auth_router, PUBLIC_SITE_HOSTS, PUBLIC_SITE_PATHS, PUBLIC_SITE_PREFIXES_EXTRA
from punty.rate_limit import RateLimitMiddleware
from punty.models.database import init_db
from punty.web.routes import router as web_router
from punty.public.routes import router as public_router
from punty.api import meets, content, scheduler, delivery, settings as settings_api, results as results_api, weather as weather_api, analytics as analytics_api
from punty.results.monitor import ResultsMonitor


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Set Cache-Control headers for static assets and public API responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path

        if path.startswith("/static/"):
            if any(path.endswith(ext) for ext in (".mp4", ".webm", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff2", ".woff")):
                response.headers["Cache-Control"] = "public, max-age=2592000, immutable"  # 30 days
            elif any(path.endswith(ext) for ext in (".css", ".js")):
                response.headers["Cache-Control"] = "public, max-age=86400"  # 1 day
        elif path.startswith("/api/public/"):
            response.headers["Cache-Control"] = "public, max-age=30"  # 30 seconds

        return response


class HostnameRoutingMiddleware(BaseHTTPMiddleware):
    """Route requests based on hostname: punty.ai -> public site, app.punty.ai -> admin."""

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").lower()
        path = request.url.path

        # Check if this is a public site host (punty.ai, not app.punty.ai)
        is_public_host = any(host.startswith(h) or host == h for h in PUBLIC_SITE_HOSTS)

        # Debug logging
        logger.debug(f"HostnameRouting: host={host}, path={path}, is_public={is_public_host}, in_paths={path in PUBLIC_SITE_PATHS}")

        # If on public host and requesting a public site path, rewrite to /public prefix
        # This allows both routers to coexist
        is_public_path = path in PUBLIC_SITE_PATHS or any(path.startswith(p) for p in PUBLIC_SITE_PREFIXES_EXTRA)
        if is_public_host and is_public_path:
            # Modify scope to add internal prefix for routing
            # Handle root path specially to avoid trailing slash redirect
            if path == "/":
                request.scope["path"] = "/public"
            else:
                request.scope["path"] = f"/public{path}"
            logger.debug(f"HostnameRouting: rewrote path to {request.scope['path']}")

        return await call_next(request)


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    logger.info("Starting PuntyAI...")

    # Validate secret key in production
    if not settings.debug and settings.secret_key == "change-me-in-production":
        raise RuntimeError(
            "PUNTY_SECRET_KEY must be set in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    # Ensure data directory exists
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info(f"Database initialized at {settings.db_path}")

    # Load personality prompt from DB into cache
    from punty.ai.generator import _personality_cache
    from punty.models.database import async_session
    from punty.models.settings import AppSettings
    from sqlalchemy import select as sa_select
    async with async_session() as db:
        result = await db.execute(sa_select(AppSettings).where(AppSettings.key == "personality_prompt"))
        setting = result.scalar_one_or_none()
        if setting and setting.value:
            _personality_cache.set(setting.value)
            logger.info("Personality prompt loaded from database")

    if not settings.disable_background:
        # Start scheduler
        from punty.scheduler.manager import scheduler_manager
        await scheduler_manager.start()
        await scheduler_manager.setup_daily_morning_job()

        # Set up per-meeting automation jobs for today's meetings
        automation_result = await scheduler_manager.setup_daily_automation()
        scheduled_meetings = automation_result.get("meetings_scheduled", [])
        logger.info(f"Scheduler started - calendar scrape at 00:05, morning scrape at 05:00, {len(scheduled_meetings)} meetings scheduled for today")

        # Schedule daily P&L digest at 23:00 AEDT
        from apscheduler.triggers.cron import CronTrigger
        from punty.config import MELB_TZ
        from punty.monitoring.alerts import send_daily_digest
        scheduler_manager.scheduler.add_job(
            send_daily_digest,
            CronTrigger(hour=23, minute=0, timezone=MELB_TZ),
            id="daily_pnl_digest",
            args=[app],
            replace_existing=True,
        )
        logger.info("Daily P&L digest scheduled for 23:00 AEDT")

        # Settle any past races with unsettled picks (catches restarts/missed settlements)
        try:
            from punty.results.picks import settle_picks_for_race
            from punty.models.pick import Pick
            from punty.models.meeting import Meeting, Race
            from sqlalchemy import and_
            from punty.config import melb_today
            async with async_session() as settle_db:
                today = melb_today()
                unsettled_result = await settle_db.execute(
                    sa_select(Race.meeting_id, Race.race_number)
                    .join(Pick, and_(
                        Pick.meeting_id == Race.meeting_id,
                        Pick.race_number == Race.race_number,
                    ))
                    .join(Meeting, Meeting.id == Race.meeting_id)
                    .where(
                        and_(
                            Race.results_status.in_(["Paying", "Closed"]),
                            Pick.settled == False,
                            Meeting.date <= today,
                        )
                    )
                    .distinct()
                )
                races_to_settle = unsettled_result.all()
                if races_to_settle:
                    settled_total = 0
                    for meeting_id, race_number in races_to_settle:
                        try:
                            count = await settle_picks_for_race(settle_db, meeting_id, race_number)
                            settled_total += count
                        except Exception as e:
                            logger.warning(f"Startup settlement failed for {meeting_id} R{race_number}: {e}")
                    logger.info(f"Startup settlement: settled {settled_total} picks across {len(races_to_settle)} races")
        except Exception as e:
            logger.warning(f"Startup settlement check failed: {e}")

        # Start Telegram bot
        from punty.telegram.bot import TelegramBot
        telegram_bot = TelegramBot(app)
        if await telegram_bot.start():
            app.state.telegram_bot = telegram_bot
        else:
            app.state.telegram_bot = None

        # Initialize results monitor
        monitor = ResultsMonitor(app)
        app.state.results_monitor = monitor

        # Always start monitor — the poll loop checks _should_be_monitoring()
        # each iteration and idles outside racing hours. Previously gated on
        # _should_be_monitoring() at boot, which meant the monitor never started
        # if the app launched before the first race (e.g. 5am morning restart).
        monitor.start()
        logger.info("Results monitor started (will idle until racing window)")
    else:
        app.state.telegram_bot = None
        app.state.results_monitor = None
        logger.info("Background services disabled (PUNTY_DISABLE_BACKGROUND=true)")

    yield

    # Shutdown
    logger.info("Shutting down PuntyAI...")
    if app.state.telegram_bot:
        await app.state.telegram_bot.stop()
    if app.state.results_monitor:
        app.state.results_monitor.stop()
    if not settings.disable_background:
        from punty.scheduler.manager import scheduler_manager
        await scheduler_manager.stop()

    # Close Playwright browser if it was started
    try:
        from punty.scrapers.playwright_base import close_browser
        await close_browser()
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="PuntyAI",
    description="AI-powered horse racing content generator",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware stack (order matters — added in reverse, outermost first):
# SessionMiddleware → HostnameRoutingMiddleware → RateLimit → AuthMiddleware → CSRFMiddleware → GZip → CacheControl
app.add_middleware(CacheControlMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CSRFMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(HostnameRoutingMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    same_site="lax",
    https_only=not settings.debug,
)

# Auth routes (login, callback, logout)
app.include_router(auth_router)

# Mount static files
static_path = Path(__file__).parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Include routers
# Public site routes (served via hostname routing middleware on punty.ai)
app.include_router(public_router, prefix="/public", tags=["public"])
# Admin dashboard routes (served on app.punty.ai)
app.include_router(web_router)
app.include_router(meets.router, prefix="/api/meets", tags=["meets"])
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(scheduler.router, prefix="/api/scheduler", tags=["scheduler"])
app.include_router(delivery.router, prefix="/api/delivery", tags=["delivery"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["settings"])
app.include_router(results_api.router, prefix="/api/results", tags=["results"])
app.include_router(weather_api.router, prefix="/api/weather", tags=["weather"])
app.include_router(analytics_api.router, prefix="/api/analytics", tags=["analytics"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/api/webhook/trigger-morning-scrape")
async def trigger_morning_scrape(request: Request):
    """Internal webhook to trigger morning scrape. Localhost only."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Localhost only")
    import asyncio
    from punty.scheduler.jobs import daily_morning_scrape
    asyncio.create_task(daily_morning_scrape())
    return {"status": "triggered", "message": "Morning scrape started in background"}


@app.post("/api/webhook/trigger-calendar-scrape")
async def trigger_calendar_scrape(request: Request):
    """Internal webhook to trigger calendar scrape. Localhost only."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Localhost only")
    import asyncio
    from punty.scheduler.jobs import daily_calendar_scrape
    asyncio.create_task(daily_calendar_scrape())
    return {"status": "triggered", "message": "Calendar scrape started in background"}


@app.get("/api/public/stats")
async def public_stats(today: bool = False):
    """Get public stats for homepage."""
    from punty.public.routes import get_winner_stats
    return await get_winner_stats(today=today)


@app.get("/api/public/wins")
async def public_recent_wins(limit: int = 15):
    """Get recent wins for public ticker."""
    from punty.public.routes import get_recent_wins_public
    return await get_recent_wins_public(limit=limit)


@app.get("/api/public/next-race")
async def public_next_race():
    """Get next upcoming race for countdown."""
    from punty.public.routes import get_next_race
    return await get_next_race()


@app.get("/api/public/bet-type-stats")
async def public_bet_type_stats(
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
    date_from: str | None = None,
    date_to: str | None = None,
    field_size_min: int | None = None,
    field_size_max: int | None = None,
    weather: str | None = None,
    barrier_min: int | None = None,
    barrier_max: int | None = None,
    today: bool = False,
):
    """Get performance stats for every bet type with optional filters."""
    from datetime import date as date_type
    from punty.public.routes import get_bet_type_stats

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

    return await get_bet_type_stats(
        venue=venue, state=state,
        distance_min=distance_min, distance_max=distance_max,
        track_condition=track_condition, race_class=race_class,
        jockey=jockey, trainer=trainer, horse_sex=horse_sex,
        tip_rank=tip_rank, odds_min=odds_min, odds_max=odds_max,
        date_from=parsed_from, date_to=parsed_to,
        field_size_min=field_size_min, field_size_max=field_size_max,
        weather=weather, barrier_min=barrier_min, barrier_max=barrier_max,
        today=today,
    )


@app.get("/api/public/filter-options")
async def public_filter_options():
    """Get distinct filter values for stats page autocomplete."""
    from punty.models.database import async_session
    from punty.models.meeting import Meeting, Race, Runner
    from punty.models.content import Content
    from punty.models.pick import Pick
    from sqlalchemy import select, distinct, and_

    async with async_session() as db:
        # Venues with settled picks
        venue_result = await db.execute(
            select(distinct(Meeting.venue))
            .join(Pick, Pick.meeting_id == Meeting.id)
            .where(Pick.settled == True)
            .order_by(Meeting.venue)
        )
        venues = [r[0] for r in venue_result.all() if r[0]]

        # Jockeys from runners that have settled picks
        jockey_result = await db.execute(
            select(distinct(Runner.jockey))
            .join(Race, Runner.race_id == Race.id)
            .join(Pick, and_(
                Pick.meeting_id == Race.meeting_id,
                Pick.race_number == Race.race_number,
                Pick.saddlecloth == Runner.saddlecloth,
                Pick.pick_type == "selection",
            ))
            .where(and_(Pick.settled == True, Runner.jockey.isnot(None)))
            .order_by(Runner.jockey)
        )
        jockeys = [r[0] for r in jockey_result.all() if r[0]]

        # Trainers
        trainer_result = await db.execute(
            select(distinct(Runner.trainer))
            .join(Race, Runner.race_id == Race.id)
            .join(Pick, and_(
                Pick.meeting_id == Race.meeting_id,
                Pick.race_number == Race.race_number,
                Pick.saddlecloth == Runner.saddlecloth,
                Pick.pick_type == "selection",
            ))
            .where(and_(Pick.settled == True, Runner.trainer.isnot(None)))
            .order_by(Runner.trainer)
        )
        trainers = [r[0] for r in trainer_result.all() if r[0]]

        # Race classes
        class_result = await db.execute(
            select(distinct(Race.class_))
            .join(Pick, and_(
                Pick.meeting_id == Race.meeting_id,
                Pick.race_number == Race.race_number,
            ))
            .where(and_(Pick.settled == True, Race.class_.isnot(None)))
            .order_by(Race.class_)
        )
        classes = [r[0] for r in class_result.all() if r[0]]

        # Track conditions
        tc_result = await db.execute(
            select(distinct(Meeting.track_condition))
            .join(Pick, Pick.meeting_id == Meeting.id)
            .where(and_(Pick.settled == True, Meeting.track_condition.isnot(None)))
            .order_by(Meeting.track_condition)
        )
        track_conditions = [r[0] for r in tc_result.all() if r[0]]

        # Weather conditions
        weather_result = await db.execute(
            select(distinct(Meeting.weather_condition))
            .join(Pick, Pick.meeting_id == Meeting.id)
            .where(and_(Pick.settled == True, Meeting.weather_condition.isnot(None)))
            .order_by(Meeting.weather_condition)
        )
        weather_conditions = [r[0] for r in weather_result.all() if r[0]]

    return {
        "venues": venues,
        "jockeys": jockeys,
        "trainers": trainers,
        "classes": classes,
        "track_conditions": track_conditions,
        "weather": weather_conditions,
    }


@app.get("/api/public/venues")
async def public_venues():
    """Get distinct venues that have sent tips, for search autocomplete."""
    from punty.models.database import async_session
    from punty.models.meeting import Meeting
    from punty.models.content import Content
    from sqlalchemy import select, distinct, and_

    async with async_session() as db:
        result = await db.execute(
            select(distinct(Meeting.venue))
            .join(Content, Content.meeting_id == Meeting.id)
            .where(and_(Content.content_type == "early_mail", Content.status == "sent"))
            .order_by(Meeting.venue)
        )
        venues = [row[0] for row in result.all() if row[0]]
    return {"venues": venues}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "punty.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
