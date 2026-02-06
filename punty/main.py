"""FastAPI application entry point for PuntyAI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from punty.config import settings
from punty.auth import AuthMiddleware, CSRFMiddleware, router as auth_router, PUBLIC_SITE_HOSTS, PUBLIC_SITE_PATHS
from punty.models.database import init_db
from punty.web.routes import router as web_router
from punty.public.routes import router as public_router
from punty.api import meets, content, scheduler, delivery, settings as settings_api, results as results_api
from punty.results.monitor import ResultsMonitor


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
        if is_public_host and path in PUBLIC_SITE_PATHS:
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

    # Start scheduler
    from punty.scheduler.manager import scheduler_manager
    await scheduler_manager.start()
    await scheduler_manager.setup_daily_morning_job()
    morning_time = scheduler_manager.get_morning_job_time()
    logger.info(f"Scheduler started - morning prep scheduled for {morning_time}")

    # Initialize results monitor
    monitor = ResultsMonitor(app)
    app.state.results_monitor = monitor

    # Auto-start monitor if within active monitoring period
    if await monitor._should_be_monitoring():
        monitor.start()
        logger.info("Results monitor auto-started (within active monitoring period)")
    else:
        logger.info("Results monitor initialized but not started (outside monitoring period)")

    yield

    # Shutdown
    logger.info("Shutting down PuntyAI...")
    monitor.stop()
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
# SessionMiddleware → HostnameRoutingMiddleware → AuthMiddleware → CSRFMiddleware
app.add_middleware(CSRFMiddleware)
app.add_middleware(AuthMiddleware)
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/api/public/stats")
async def public_stats():
    """Get public stats for homepage."""
    from punty.public.routes import get_winner_stats
    return await get_winner_stats()


@app.get("/api/public/wins")
async def public_recent_wins(limit: int = 15):
    """Get recent wins for public ticker."""
    from punty.public.routes import get_recent_wins_public
    return await get_recent_wins_public(limit=limit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "punty.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
