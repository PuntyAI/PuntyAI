"""Base scraper class with common functionality."""

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ScraperError(Exception):
    """Exception raised when scraping fails."""

    pass


class BaseScraper(ABC):
    """Base class for all scrapers."""

    # Default headers to mimic a browser
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    def __init__(self, timeout: float = 30.0):
        """Initialize scraper with HTTP client."""
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> str:
        """Fetch URL and return HTML content."""
        try:
            logger.info(f"Fetching: {url}")
            response = await self.client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise ScraperError(f"HTTP {e.response.status_code}: {url}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise ScraperError(f"Request failed: {url}")

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML into BeautifulSoup object."""
        return BeautifulSoup(html, "lxml")

    @abstractmethod
    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape meeting data for a venue and date.

        Returns dict with:
        - meeting: Meeting data
        - races: List of race data
        - runners: List of runner data
        """
        pass

    @abstractmethod
    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Scrape race results for a venue and date.

        Returns list of result data.
        """
        pass

    def generate_meeting_id(self, venue: str, race_date: date) -> str:
        """Generate unique meeting ID."""
        venue_slug = venue.lower().replace(" ", "-")
        return f"{venue_slug}-{race_date.isoformat()}"

    def generate_race_id(self, meeting_id: str, race_number: int) -> str:
        """Generate unique race ID."""
        return f"{meeting_id}-r{race_number}"

    def generate_runner_id(self, race_id: str, barrier: int, horse_name: str) -> str:
        """Generate unique runner ID."""
        horse_slug = horse_name.lower().replace(" ", "-").replace("'", "")[:20]
        return f"{race_id}-{barrier}-{horse_slug}"

    @staticmethod
    def clean_text(text: Optional[str]) -> Optional[str]:
        """Clean and normalize text."""
        if text is None:
            return None
        return " ".join(text.strip().split())

    @staticmethod
    def parse_odds(odds_str: Optional[str]) -> Optional[float]:
        """Parse odds string to float."""
        if not odds_str:
            return None
        try:
            # Remove $ sign and any other characters
            cleaned = odds_str.replace("$", "").replace(",", "").strip()
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def parse_weight(weight_str: Optional[str]) -> Optional[float]:
        """Parse weight string to float (in kg)."""
        if not weight_str:
            return None
        try:
            cleaned = weight_str.replace("kg", "").strip()
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def parse_distance(distance_str: Optional[str]) -> Optional[int]:
        """Parse distance string to meters."""
        if not distance_str:
            return None
        try:
            cleaned = distance_str.lower().replace("m", "").replace(",", "").strip()
            return int(cleaned)
        except ValueError:
            return None

    @staticmethod
    def parse_prize_money(prize_str: Optional[str]) -> Optional[int]:
        """Parse prize money string to integer."""
        if not prize_str:
            return None
        try:
            cleaned = prize_str.replace("$", "").replace(",", "").strip()
            return int(float(cleaned))
        except ValueError:
            return None
