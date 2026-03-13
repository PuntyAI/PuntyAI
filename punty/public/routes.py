"""Public website routes — aggregates sub-module routers."""
from fastapi import APIRouter

from punty.public.pages import router as pages_router
from punty.public.dashboard import router as dashboard_router
from punty.public.meeting import router as meeting_router
from punty.public.stats import router as stats_router
from punty.public.betfair_tracker import router as betfair_router

# Re-export data functions consumed by main.py and tests
from punty.public.pages import get_winner_stats, get_recent_wins_public, get_tips_calendar  # noqa: F401
from punty.public.dashboard import get_next_race  # noqa: F401
from punty.public.stats import get_bet_type_stats  # noqa: F401
from punty.public.meeting import get_meeting_tips  # noqa: F401

router = APIRouter()
router.include_router(pages_router)
router.include_router(dashboard_router)
router.include_router(meeting_router)
router.include_router(stats_router)
router.include_router(betfair_router)
