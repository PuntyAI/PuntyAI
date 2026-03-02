"""Betfair bet placement client — wraps BetfairScraper for place bet operations."""

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import settings

logger = logging.getLogger(__name__)


async def _get_scraper(db: AsyncSession):
    """Get a configured BetfairScraper instance, or None if not configured."""
    from punty.scrapers.betfair import BetfairScraper
    return await BetfairScraper.from_settings(db)


async def resolve_place_market(
    db: AsyncSession,
    venue: str,
    race_date: date,
    meeting_id: str,
    race_number: int,
) -> Optional[dict[str, Any]]:
    """Resolve a Betfair PLACE market for a specific race.

    Returns: {market_id, runners: [{selection_id, horse_name}]} or None
    """
    if settings.mock_external:
        logger.info(f"[MOCK] resolve_place_market: {venue} R{race_number}")
        return {
            "market_id": f"mock-{meeting_id}-r{race_number}",
            "runners": [{"selection_id": 12345, "horse_name": "Mock Horse"}],
        }

    scraper = await _get_scraper(db)
    if not scraper:
        logger.warning("Betfair not configured — cannot resolve place market")
        return None

    from punty.scrapers.betfair import _strip_venue_prefix

    search_venue = _strip_venue_prefix(venue)
    date_from = datetime(race_date.year, race_date.month, race_date.day,
                         tzinfo=timezone.utc).isoformat()
    date_to = datetime(race_date.year, race_date.month, race_date.day,
                       23, 59, 59, tzinfo=timezone.utc).isoformat()

    try:
        catalogue = await scraper._api_call("listMarketCatalogue", {
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
            ],
            "maxResults": "25",
            "sort": "FIRST_TO_START",
        })
    except Exception as e:
        logger.error(f"Betfair listMarketCatalogue failed: {e}")
        return None

    if not catalogue:
        logger.debug(f"Betfair: no PLACE catalogue for {venue}")
        return None

    # PLACE markets are all named "To Be Placed" — no race number in the name.
    # Match by cross-referencing our Race start_time with market start times.
    from punty.models.meeting import Race
    from sqlalchemy import select as sa_select
    race_result = await db.execute(
        sa_select(Race).where(
            Race.meeting_id == meeting_id,
            Race.race_number == race_number,
        )
    )
    race = race_result.scalar_one_or_none()

    target_market = None
    if race and race.start_time:
        # Match by closest start time (within 5 min tolerance)
        race_utc = race.start_time
        # start_time is stored as naive Melbourne time; Betfair returns UTC
        # Convert Melbourne naive → UTC by subtracting ~11h (AEDT) or ~10h (AEST)
        import zoneinfo
        melb_tz = zoneinfo.ZoneInfo("Australia/Melbourne")
        race_aware = race_utc.replace(tzinfo=melb_tz)
        race_utc_dt = race_aware.astimezone(timezone.utc)

        best_diff = None
        for market in catalogue:
            ms = market.get("marketStartTime", "")
            if not ms:
                continue
            # Parse Betfair ISO time "2026-03-02T06:34:00.000Z"
            market_dt = datetime.fromisoformat(ms.replace("Z", "+00:00"))
            diff = abs((market_dt - race_utc_dt).total_seconds())
            if diff < 300 and (best_diff is None or diff < best_diff):  # within 5 min
                best_diff = diff
                target_market = market

    if not target_market:
        # Fallback: sorted FIRST_TO_START, so Nth market = race N
        # Only works when catalogue has all races for the meeting
        idx = race_number - 1
        if idx < len(catalogue):
            target_market = catalogue[idx]
            logger.info(f"Betfair: matched {venue} R{race_number} by position (index {idx})")

    if not target_market:
        logger.debug(f"Betfair: no PLACE market found for {venue} R{race_number}")
        return None

    runners = []
    for runner in target_market.get("runners", []):
        name = runner.get("runnerName", "")
        name = re.sub(r"^\d+\.\s*", "", name)  # strip saddlecloth prefix
        runners.append({
            "selection_id": runner["selectionId"],
            "horse_name": name,
        })
    return {
        "market_id": target_market["marketId"],
        "runners": runners,
    }


async def get_place_odds(
    db: AsyncSession,
    market_id: str,
    selection_id: int,
) -> Optional[float]:
    """Get current best back price for a selection in a PLACE market."""
    if settings.mock_external:
        return 3.50

    scraper = await _get_scraper(db)
    if not scraper:
        return None

    try:
        books = await scraper._api_call("listMarketBook", {
            "marketIds": [market_id],
            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        })
    except Exception as e:
        logger.error(f"Betfair listMarketBook failed: {e}")
        return None

    if not books:
        return None

    for book in books:
        for runner in book.get("runners", []):
            if runner.get("selectionId") == selection_id:
                return scraper._get_best_back(runner)
    return None


async def place_bet(
    db: AsyncSession,
    market_id: str,
    selection_id: int,
    stake: float,
    price: float,
) -> dict[str, Any]:
    """Place a BACK bet on Betfair Exchange.

    Returns: {bet_id, status, size_matched, average_price_matched} or {status: 'failed', error: ...}
    """
    if settings.mock_external:
        logger.info(f"[MOCK] place_bet: market={market_id} sel={selection_id} ${stake} @ {price}")
        return {
            "bet_id": f"mock-{market_id}-{selection_id}",
            "status": "SUCCESS",
            "size_matched": stake,
            "average_price_matched": price,
        }

    scraper = await _get_scraper(db)
    if not scraper:
        return {"status": "failed", "error": "Betfair not configured"}

    try:
        result = await scraper._api_call("placeOrders", {
            "marketId": market_id,
            "instructions": [{
                "selectionId": selection_id,
                "side": "BACK",
                "orderType": "LIMIT",
                "limitOrder": {
                    "size": round(stake, 2),
                    "price": round(price, 2),
                    "persistenceType": "LAPSE",
                },
            }],
        })
    except Exception as e:
        logger.error(f"Betfair placeOrders failed: {e}")
        return {"status": "failed", "error": str(e)}

    if not result or not isinstance(result, dict):
        return {"status": "failed", "error": f"Unexpected response: {result}"}

    status = result.get("status", "FAILURE")
    reports = result.get("instructionReports", [])

    if status == "SUCCESS" and reports:
        report = reports[0]
        return {
            "bet_id": report.get("betId", ""),
            "status": report.get("status", "SUCCESS"),
            "size_matched": report.get("sizeMatched", 0),
            "average_price_matched": report.get("averagePriceMatched", 0),
        }

    error = result.get("errorCode", "UNKNOWN")
    if reports:
        error = reports[0].get("errorCode", error)
    return {"status": "failed", "error": error}


async def get_account_balance(db: AsyncSession) -> Optional[float]:
    """Get available Betfair account balance."""
    if settings.mock_external:
        return 50.00

    scraper = await _get_scraper(db)
    if not scraper:
        return None

    try:
        token = await scraper._get_session_token()
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api-au.betfair.com/exchange/account/json-rpc/v1",
                json=[{
                    "jsonrpc": "2.0",
                    "method": "AccountAPING/v1.0/getAccountFunds",
                    "params": {},
                    "id": 1,
                }],
                headers={
                    "X-Application": scraper.app_key,
                    "X-Authentication": token,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list) and data:
            result = data[0].get("result", {})
            return result.get("availableToBetBalance")
    except Exception as e:
        logger.error(f"Betfair getAccountFunds failed: {e}")
    return None
