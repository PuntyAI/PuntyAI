"""Group One Glory - Tipping competition for Melbourne/Sydney Group 1 races."""

from punty.glory.routes import router as glory_router
from punty.glory.api import router as glory_api_router

__all__ = ["glory_router", "glory_api_router"]
