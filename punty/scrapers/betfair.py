"""Betfair Exchange API scraper for live odds.

Uses cert-based authentication to fetch current back prices from
the Betfair Exchange for Australian horse racing markets. Provides
odds_betfair and acts as a fallback when racing.com odds are missing.

Requires app_settings:
  - betfair_username
  - betfair_password
  - betfair_app_key
  - betfair_cert_path
  - betfair_key_path
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

logger = logging.getLogger(__name__)

# Session token cache (module-level singleton)
_session_cache: dict[str, Any] = {"token": None, "expires": 0}


def _normalize_name(name: str) -> str:
    """Normalize horse name for cross-provider matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())


def _strip_venue_prefix(venue: str) -> str:
    """Strip sponsor prefixes like 'Beaumont' from 'Beaumont Newcastle'."""
    # Common sponsor prefixes on Australian venues
    prefixes = [
        "Beaumont", "Ladbrokes", "TAB", "Sportsbet", "Bet365",
        "Neds", "Pointsbet", "Unibet", "BetEasy",
    ]
    for prefix in prefixes:
        if venue.startswith(prefix + " "):
            return venue[len(prefix) + 1:]
    return venue


class BetfairScraper:
    """Betfair Exchange API scraper for live odds."""

    LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
    BETTING_URL = "https://api-au.betfair.com/exchange/betting/json-rpc/v1"

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
    async def from_settings(cls, db: AsyncSession) -> Optional["BetfairScraper"]:
        """Load credentials from app_settings DB table."""
        keys = [
            "betfair_username", "betfair_password", "betfair_app_key",
            "betfair_cert_path", "betfair_key_path",
        ]
        result = await db.execute(
            select(AppSettings).where(AppSettings.key.in_(keys))
        )
        settings = {s.key: s.value for s in result.scalars().all()}

        username = settings.get("betfair_username")
        password = settings.get("betfair_password")
        app_key = settings.get("betfair_app_key")
        cert_path = settings.get("betfair_cert_path")
        key_path = settings.get("betfair_key_path")

        if not all([username, password, app_key, cert_path, key_path]):
            logger.debug("Betfair credentials not configured — skipping")
            return None

        return cls(username, password, app_key, cert_path, key_path)

    async def _get_session_token(self) -> str:
        """Get a valid session token, using cache or refreshing via cert login."""
        global _session_cache

        # Return cached token if still valid (with 60s buffer)
        if _session_cache["token"] and time.time() < _session_cache["expires"] - 60:
            return _session_cache["token"]

        logger.info("Betfair: logging in with cert auth")
        async with httpx.AsyncClient(
            cert=(self.cert_path, self.key_path),
            timeout=15.0,
        ) as client:
            resp = await client.post(
                self.LOGIN_URL,
                data={"username": self.username, "password": self.password},
                headers={
                    "X-Application": self.app_key,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("loginStatus")
        token = data.get("sessionToken")
        if status != "SUCCESS" or not token:
            raise RuntimeError(f"Betfair login failed: {status} — {data}")

        # Cache for 20 minutes
        _session_cache["token"] = token
        _session_cache["expires"] = time.time() + 1200
        logger.info("Betfair: login successful")
        return token

    async def _api_call(self, method: str, params: dict) -> Any:
        """Make a Betfair JSON-RPC API call."""
        token = await self._get_session_token()
        payload = {
            "jsonrpc": "2.0",
            "method": f"SportsAPING/v1.0/{method}",
            "params": params,
            "id": 1,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                self.BETTING_URL,
                json=[payload],
                headers={
                    "X-Application": self.app_key,
                    "X-Authentication": token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # JSON-RPC returns array of results
        if isinstance(data, list) and data:
            result = data[0]
        else:
            result = data

        if "error" in result:
            error = result["error"]
            # Clear session cache on auth errors
            if "INVALID_SESSION" in str(error):
                _session_cache["token"] = None
            raise RuntimeError(f"Betfair API error ({method}): {error}")

        return result.get("result", [])

    async def get_odds_for_meeting(
        self, venue: str, race_date: date, meeting_id: str
    ) -> list[dict]:
        """Get Betfair exchange odds for all races at a venue.

        Returns list of dicts: {race_id, horse_name, odds_betfair}
        """
        # Build venue search term — strip sponsor prefix
        search_venue = _strip_venue_prefix(venue)

        # Date range for market filter (full day in UTC)
        date_from = datetime(race_date.year, race_date.month, race_date.day,
                             tzinfo=timezone.utc).isoformat()
        date_to = datetime(race_date.year, race_date.month, race_date.day,
                           23, 59, 59, tzinfo=timezone.utc).isoformat()

        # 1. List WIN markets for this venue/date
        catalogue = await self._api_call("listMarketCatalogue", {
            "filter": {
                "eventTypeIds": ["7"],
                "marketCountries": ["AU"],
                "marketTypeCodes": ["WIN"],
                "venues": [search_venue],
                "marketStartTime": {"from": date_from, "to": date_to},
            },
            "marketProjection": [
                "RUNNER_DESCRIPTION",
                "RUNNER_METADATA",
                "EVENT",
                "MARKET_START_TIME",
                "MARKET_DESCRIPTION",
            ],
            "maxResults": "25",
            "sort": "FIRST_TO_START",
        })

        if not catalogue:
            logger.info(f"Betfair: no markets found for '{search_venue}' on {race_date}")
            return []

        logger.info(f"Betfair: found {len(catalogue)} WIN markets for {search_venue}")

        # 2. Get market IDs and fetch prices
        market_ids = [m["marketId"] for m in catalogue]
        books = await self._api_call("listMarketBook", {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS"],
            },
        })

        # Build lookup: marketId -> book
        book_by_id = {b["marketId"]: b for b in books} if books else {}

        # 3. Match runners and extract odds
        results = []
        for market in catalogue:
            market_id = market["marketId"]
            book = book_by_id.get(market_id, {})
            book_runners = {r["selectionId"]: r for r in book.get("runners", [])}

            # Determine race number from market name (e.g., "R1 1200m Mdn")
            market_name = market.get("marketName", "")
            race_num = self._extract_race_number(market_name, market)
            if not race_num:
                continue

            race_id = f"{meeting_id}-r{race_num}"

            for runner in market.get("runners", []):
                selection_id = runner["selectionId"]
                horse_name = runner.get("runnerName", "")
                if not horse_name:
                    continue

                # Strip saddlecloth prefix (e.g. "1. Arties Magic" → "Arties Magic")
                horse_name = re.sub(r"^\d+\.\s*", "", horse_name)

                # Check runner status — Betfair marks scratchings as REMOVED
                book_runner = book_runners.get(selection_id, {})
                runner_status = book_runner.get("status", "ACTIVE")
                if runner_status == "REMOVED":
                    results.append({
                        "race_id": race_id,
                        "horse_name": horse_name,
                        "odds_betfair": 0,
                        "scratched": True,
                    })
                    continue

                # Get best back price from book
                price = self._get_best_back(book_runner)
                if not price:
                    # Fallback to lastPriceTraded
                    price = book_runner.get("lastPriceTraded")
                if not price or price <= 1.0:
                    continue

                results.append({
                    "race_id": race_id,
                    "horse_name": horse_name,
                    "odds_betfair": round(price, 2),
                    "scratched": False,
                })

        logger.info(
            f"Betfair: {len(results)} runner odds from {len(catalogue)} markets "
            f"for {venue}"
        )
        return results

    @staticmethod
    def _extract_race_number(market_name: str, market: dict) -> Optional[int]:
        """Extract race number from Betfair market name or description."""
        # Try market name: "R1 1200m Mdn" or "1. Some Race Name"
        m = re.match(r"R(\d+)\b", market_name)
        if m:
            return int(m.group(1))
        m = re.match(r"(\d+)\.", market_name)
        if m:
            return int(m.group(1))

        # Try description
        desc = market.get("description", {})
        if isinstance(desc, dict):
            market_time = desc.get("marketTime")
            # Could use time-based matching as fallback
        return None

    @staticmethod
    def _get_best_back(book_runner: dict) -> Optional[float]:
        """Get the best available back price for a runner."""
        ex = book_runner.get("ex", {})
        backs = ex.get("availableToBack", [])
        if backs:
            # First entry is the best (highest) back price
            return backs[0].get("price")
        return None

    async def fetch_place_odds(
        self, venue: str, race_date: date, meeting_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch PLACE market odds from Betfair Exchange.

        Returns list of dicts: {race_id, horse_name, place_odds_betfair}
        """
        search_venue = _strip_venue_prefix(venue)
        date_from = datetime(race_date.year, race_date.month, race_date.day,
                             tzinfo=timezone.utc).isoformat()
        date_to = datetime(race_date.year, race_date.month, race_date.day,
                           23, 59, 59, tzinfo=timezone.utc).isoformat()

        catalogue = await self._api_call("listMarketCatalogue", {
            "filter": {
                "eventTypeIds": ["7"],
                "marketCountries": ["AU"],
                "marketTypeCodes": ["PLACE"],
                "venues": [search_venue],
                "marketStartTime": {"from": date_from, "to": date_to},
            },
            "marketProjection": [
                "RUNNER_DESCRIPTION",
                "RUNNER_METADATA",
                "EVENT",
                "MARKET_START_TIME",
                "MARKET_DESCRIPTION",
            ],
            "maxResults": "25",
            "sort": "FIRST_TO_START",
        })

        if not catalogue:
            logger.debug(f"Betfair: no PLACE markets for '{search_venue}' on {race_date}")
            return []

        logger.info(f"Betfair: found {len(catalogue)} PLACE markets for {search_venue}")

        market_ids = [m["marketId"] for m in catalogue]
        books = await self._api_call("listMarketBook", {
            "marketIds": market_ids,
            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        })
        book_by_id = {b["marketId"]: b for b in books} if books else {}

        results = []
        for market in catalogue:
            market_id = market["marketId"]
            book = book_by_id.get(market_id, {})
            book_runners = {r["selectionId"]: r for r in book.get("runners", [])}

            market_name = market.get("marketName", "")
            race_num = self._extract_race_number(market_name, market)
            if not race_num:
                continue

            race_id = f"{meeting_id}-r{race_num}"
            for runner in market.get("runners", []):
                selection_id = runner["selectionId"]
                horse_name = runner.get("runnerName", "")
                if not horse_name:
                    continue
                horse_name = re.sub(r"^\d+\.\s*", "", horse_name)

                book_runner = book_runners.get(selection_id, {})
                price = self._get_best_back(book_runner)
                if not price:
                    price = book_runner.get("lastPriceTraded")
                if not price or price <= 1.0:
                    continue

                results.append({
                    "race_id": race_id,
                    "horse_name": horse_name,
                    "place_odds_betfair": round(price, 2),
                })

        logger.info(
            f"Betfair: {len(results)} place odds from {len(catalogue)} PLACE markets "
            f"for {venue}"
        )
        return results
