"""PointsBet greyhound odds scraper via public JSON API.

Reuses the same PointsBet API as the thoroughbred scraper but filters for
greyhound racing markets. The API structure is identical — the difference
is in the category/sport filter.

API endpoints (same base as thoroughbred):
  - Meetings: GET api.au.pointsbet.com/api/racing/v4/meetings?startDate=...&endDate=...
    Filter: categoryName == "Greyhound Racing" or similar
  - Race detail: GET api.au.pointsbet.com/api/v2/racing/races/{eventId}

Key differences from thoroughbred PointsBet scraper:
  - Filter meetings by greyhound category (not thoroughbred)
  - dog_name instead of horse_name
  - box_number instead of saddlecloth
  - No jockey field in runner data
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx

from punty.greyhounds.models import GREYHOUND_VENUES, venue_state

logger = logging.getLogger(__name__)

API_BASE = "https://api.au.pointsbet.com"
MAX_VALID_ODDS = 501.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://pointsbet.com.au",
    "Referer": "https://pointsbet.com.au/",
}

# PointsBet category name for greyhounds (confirm via API inspection)
_GREYHOUND_CATEGORY = "Greyhound Racing"
_TARGET_STATES = {"VIC", "NSW", "QLD"}


def _normalize_name(name: str) -> str:
    """Normalize dog name for cross-provider matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())


class PointsBetGreyhoundScraper:
    """Scrape greyhound odds from PointsBet public JSON API.

    Usage:
        scraper = PointsBetGreyhoundScraper()
        odds = await scraper.scrape_odds_for_meeting("Sandown Park", date(2026, 3, 27))
    """

    async def scrape_odds_for_meeting(
        self,
        venue: str,
        race_date: date,
        race_count: int = 12,
    ) -> list[dict[str, Any]]:
        """Fetch PointsBet odds for all greyhound races at a venue.

        Returns list of dicts:
            {race_number, dog_name, box_number,
             current_odds, opening_odds, place_odds, scratched}

        TODO: Implement API calls. Pattern matches thoroughbred PointsBet scraper.
        """
        # Step 1: Find race IDs for this venue from meetings endpoint
        race_ids = await self._find_race_ids(venue, race_date)
        if not race_ids:
            logger.warning(f"PointsBet Greyhound: no races found for {venue} on {race_date}")
            return []

        logger.info(f"PointsBet Greyhound: found {len(race_ids)} races for {venue}")

        # Step 2: Fetch odds for each race
        all_odds: list[dict] = []
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            for race_num, event_id in race_ids:
                try:
                    odds = await self._fetch_race_odds(client, event_id, race_num)
                    all_odds.extend(odds)
                except Exception as e:
                    logger.error(f"PointsBet Greyhound R{race_num}: {e}")

        return all_odds

    async def _find_race_ids(
        self, venue: str, race_date: date
    ) -> list[tuple[int, str]]:
        """Find PointsBet event IDs for greyhound races at a venue.

        Returns list of (race_number, event_id) tuples.

        TODO: Implement. Query meetings endpoint, filter by greyhound category + venue.
        """
        # URL: {API_BASE}/api/racing/v4/meetings?startDate={date}&endDate={date}
        # Filter: meeting.categoryName == _GREYHOUND_CATEGORY
        # Match venue name (fuzzy — PointsBet may use different naming)
        # Extract race event IDs
        raise NotImplementedError("PointsBet greyhound race ID lookup not yet implemented")

    async def _fetch_race_odds(
        self, client: httpx.AsyncClient, event_id: str, race_number: int
    ) -> list[dict[str, Any]]:
        """Fetch odds for a single greyhound race.

        Returns list of dicts per runner:
            {race_number, dog_name, box_number,
             current_odds, opening_odds, place_odds, scratched}

        TODO: Implement. Query race detail endpoint, extract runner odds.
        """
        # URL: {API_BASE}/api/v2/racing/races/{event_id}
        # Parse response:
        #   - runners[].name -> dog_name
        #   - runners[].number -> box_number (NOT saddlecloth)
        #   - runners[].fixedWin.odds -> current_odds
        #   - runners[].fixedPlace.odds -> place_odds
        #   - runners[].scratched -> scratched flag
        raise NotImplementedError("PointsBet greyhound race odds fetch not yet implemented")
