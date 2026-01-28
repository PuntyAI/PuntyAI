"""Web scrapers for Australian racing data."""

from punty.scrapers.base import BaseScraper, ScraperError
from punty.scrapers.racing_com import RacingComScraper
from punty.scrapers.tab import TabScraper
from punty.scrapers.punters import PuntersScraper

__all__ = [
    "BaseScraper",
    "ScraperError",
    "RacingComScraper",
    "TabScraper",
    "PuntersScraper",
]
