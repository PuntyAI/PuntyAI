"""Web scrapers for Australian racing data."""

from punty.scrapers.base import BaseScraper, ScraperError
from punty.scrapers.racing_com import RacingComScraper

__all__ = [
    "BaseScraper",
    "ScraperError",
    "RacingComScraper",
]
