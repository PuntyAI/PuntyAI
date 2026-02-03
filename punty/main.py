"""FastAPI application entry point for PuntyAI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from punty.config import settings
from punty.auth import AuthMiddleware, CSRFMiddleware, router as auth_router
from punty.models.database import init_db
from punty.web.routes import router as web_router
from punty.api import meets, content, scheduler, delivery, settings as settings_api, results as results_api
from punty.results.monitor import ResultsMonitor
from punty.glory.routes import router as glory_router
from punty.glory.api import router as glory_api_router
from punty.glory.auth import GloryAuthMiddleware, GloryCSRFMiddleware


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

    # Start scheduler (will be implemented later)
    # await scheduler_manager.start()

    # Initialize results monitor
    monitor = ResultsMonitor(app)
    app.state.results_monitor = monitor

    yield

    # Shutdown
    logger.info("Shutting down PuntyAI...")
    monitor.stop()
    # await scheduler_manager.stop()

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
# SessionMiddleware → AuthMiddleware → CSRFMiddleware
app.add_middleware(CSRFMiddleware)
app.add_middleware(AuthMiddleware)
# Glory-specific middleware (runs after general auth, handles /group1glory/ paths)
app.add_middleware(GloryCSRFMiddleware)
app.add_middleware(GloryAuthMiddleware)
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
app.include_router(web_router)
app.include_router(meets.router, prefix="/api/meets", tags=["meets"])
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(scheduler.router, prefix="/api/scheduler", tags=["scheduler"])
app.include_router(delivery.router, prefix="/api/delivery", tags=["delivery"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["settings"])
app.include_router(results_api.router, prefix="/api/results", tags=["results"])

# Group One Glory routes (separate tipping competition module)
app.include_router(glory_router, tags=["glory"])
app.include_router(glory_api_router, tags=["glory-api"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "punty.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
