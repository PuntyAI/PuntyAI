"""FastAPI application entry point for PuntyAI."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from punty.config import settings
from punty.models.database import init_db
from punty.web.routes import router as web_router
from punty.api import meets, content, scheduler, delivery, settings as settings_api


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

    # Ensure data directory exists
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info(f"Database initialized at {settings.db_path}")

    # Start scheduler (will be implemented later)
    # await scheduler_manager.start()

    yield

    # Shutdown
    logger.info("Shutting down PuntyAI...")
    # await scheduler_manager.stop()


# Create FastAPI app
app = FastAPI(
    title="PuntyAI",
    description="AI-powered horse racing content generator",
    version="0.1.0",
    lifespan=lifespan,
)

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
