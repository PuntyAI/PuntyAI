"""Betfair Exchange greyhound scraper — exchange odds for greyhound markets.

Uses the same Betfair API as the thoroughbred scraper but with:
  - event_type_id = "4339" (greyhound racing, not "7" for horse racing)
  - market_type = "WIN" or "PLACE"

Reuses cert-based authentication from the main Betfair scraper.
Requires same app_settings: betfair_username, betfair_password,
betfair_app_key, betfair_cert_path, betfair_key_path.

Key differences from thoroughbred Betfair scraper:
  - event_type_id "4339" instead of "7"
  - Runner names are dog names
  - Runner numbers are box numbers (1-8)
  - Typically 8-runner fields (simpler matching)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.settings import AppSettings
from punty.greyhounds.models import GREYHOUND_VENUES, venue_state

logger = logging.getLogger(__name__)

# Betfair API constants
BETFAIR_EVENT_TYPE_GREYHOUND = "4339"
BETFAIR_EXCHANGE_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
BETFAIR_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"

# Module-level session cache (shared with thoroughbred if same credentials)
_session_cache: dict[str, Any] = {"token": None, "expires": 0}


def _normalize_name(name: str) -> str:
    """Normalize dog name for cross-provider matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())


class BetfairGreyhoundScraper:
    """Fetch greyhound exchange odds from Betfair API.

    Usage:
        scraper = await BetfairGreyhoundScraper.from_settings(db_session)
        if scraper:
            odds = await scraper.scrape_odds_for_meeting("Sandown Park", date(2026, 3, 27))
    """

    def __init__(
        self,
        username: str,
        password: str,
        app_key: str,
        cert_path: str,
        key_path: str,
    ):
        self.username = username
        self.password = password
        self.app_key = app_key
        self.cert_path = cert_path
        self.key_path = key_path

    @classmethod
    async def from_settings(cls, db: AsyncSession) -> Optional["BetfairGreyhoundScraper"]:
        """Create scraper from app_settings DB table. Returns None if not configured.

        Reuses the same credentials as the thoroughbred Betfair scraper.
        """
        result = await db.execute(
            select(AppSettings).where(
                AppSettings.key.in_([
                    "betfair_username", "betfair_password",
                    "betfair_app_key", "betfair_cert_path", "betfair_key_path",
                ])
            )
        )
        settings_map = {row.key: row.value for row in result.scalars()}
        required = ["betfair_username", "betfair_password", "betfair_app_key",
                     "betfair_cert_path", "betfair_key_path"]
        if not all(settings_map.get(k) for k in required):
            return None
        return cls(
            username=settings_map["betfair_username"],
            password=settings_map["betfair_password"],
            app_key=settings_map["betfair_app_key"],
            cert_path=settings_map["betfair_cert_path"],
            key_path=settings_map["betfair_key_path"],
        )

    async def scrape_odds_for_meeting(
        self,
        venue: str,
        race_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch Betfair exchange odds for greyhound races at a venue.

        Returns list of dicts:
            {race_number, dog_name, box_number, odds_betfair, scratched}

        TODO: Implement API calls using BETFAIR_EVENT_TYPE_GREYHOUND.
        """
        # Step 1: Login / reuse session token
        # Step 2: listMarketCatalogue with filter:
        #   eventTypeIds: [BETFAIR_EVENT_TYPE_GREYHOUND]
        #   marketCountries: ["AU"]
        #   marketStartTime: {from: race_date, to: race_date + 1day}
        #   venues: [venue]
        #   marketTypeCodes: ["WIN"]  (or "PLACE")
        # Step 3: listMarketBook for each market to get runner prices
        # Step 4: Match runners by name/number
        logger.info(f"Betfair Greyhound: scraping odds for {venue} on {race_date}")
        raise NotImplementedError("Betfair greyhound odds scraping not yet implemented")

    async def resolve_market(
        self,
        venue: str,
        race_date: date,
        race_number: int,
        market_type: str = "PLACE",
    ) -> Optional[dict[str, Any]]:
        """Resolve a Betfair market for a specific greyhound race.

        Returns: {market_id, runners: [{selection_id, dog_name, box_number}]} or None.

        Used by the betting queue to place bets on greyhound races.

        TODO: Implement market resolution.
        """
        # Same pattern as thoroughbred betfair_client.resolve_place_market
        # but with event_type_id = BETFAIR_EVENT_TYPE_GREYHOUND
        raise NotImplementedError("Betfair greyhound market resolution not yet implemented")

    async def _login(self) -> str:
        """Authenticate with Betfair cert-based login. Returns session token.

        TODO: Implement — identical to thoroughbred Betfair scraper login.
        Can potentially share the session with the thoroughbred scraper.
        """
        # Check cache first
        if _session_cache["token"] and time.time() < _session_cache["expires"]:
            return _session_cache["token"]

        # POST to BETFAIR_LOGIN_URL with cert
        raise NotImplementedError("Betfair login not yet implemented for greyhounds")

    async def _api_call(self, method: str, params: dict) -> Any:
        """Make a Betfair API call.

        TODO: Implement — identical to thoroughbred Betfair scraper pattern.
        """
        # POST to {BETFAIR_EXCHANGE_URL}/{method}
        # Headers: X-Application: app_key, X-Authentication: session_token
        # Body: JSON params
        raise NotImplementedError("Betfair API call not yet implemented for greyhounds")
