"""Betfairlightweight client wrapper — replaces raw httpx BetfairScraper API calls.

Uses betfairlightweight for:
- Cert-based authentication with automatic session management
- Type-safe API responses
- Connection pooling (single requests.Session)
- Retry logic via tenacity
- Foundation for Flumine streaming/simulation

Keeps our async interface by running bflw (sync) in a thread executor.
"""

import asyncio
import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import betfairlightweight
from betfairlightweight.filters import (
    market_filter,
    market_projection,
    price_projection,
)

logger = logging.getLogger(__name__)

# Singleton client
_client: Optional[betfairlightweight.APIClient] = None
_client_lock = asyncio.Lock()


async def get_client(db=None) -> Optional[betfairlightweight.APIClient]:
    """Get or create a betfairlightweight APIClient.

    Loads credentials from app_settings DB table on first call.
    Returns cached client on subsequent calls.
    """
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client

        if db is None:
            from punty.models.database import async_session
            async with async_session() as db:
                return await _create_client(db)
        else:
            return await _create_client(db)


async def _create_client(db) -> Optional[betfairlightweight.APIClient]:
    """Create and login a betfairlightweight client."""
    global _client
    from sqlalchemy import select
    from punty.models.settings import AppSettings

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
        logger.debug("Betfair credentials not configured")
        return None

    try:
        client = betfairlightweight.APIClient(
            username=username,
            password=password,
            app_key=app_key,
            certs=cert_path,
            locale="en",
            lightweight=False,  # Full response objects for type safety
        )

        # Login in thread executor (bflw is sync)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client.login)
        logger.info("betfairlightweight: login successful")

        _client = client
        return client
    except Exception as e:
        logger.error(f"betfairlightweight login failed: {e}")
        return None


async def _run_sync(func, *args, **kwargs):
    """Run a synchronous bflw call in the thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ── Market Discovery ──

async def list_market_catalogue(
    venue: str,
    race_date: date,
    market_type: str = "PLACE",
    country: str = "AU",
    max_results: int = 25,
) -> list:
    """List market catalogue for a venue/date/type.

    Returns list of MarketCatalogue objects with runner info.
    """
    client = await get_client()
    if not client:
        return []

    date_from = datetime(race_date.year, race_date.month, race_date.day,
                         tzinfo=timezone.utc).isoformat()
    date_to = datetime(race_date.year, race_date.month, race_date.day,
                       23, 59, 59, tzinfo=timezone.utc).isoformat()

    from punty.scrapers.betfair import _strip_venue_prefix
    search_venue = _strip_venue_prefix(venue)

    try:
        mf = market_filter(
            event_type_ids=["7"],
            market_countries=[country],
            market_type_codes=[market_type],
            venues=[search_venue],
            market_start_time={"from": date_from, "to": date_to},
        )
        mp = ["RUNNER_DESCRIPTION", "RUNNER_METADATA", "EVENT", "MARKET_START_TIME"]

        catalogues = await _run_sync(
            client.betting.list_market_catalogue,
            filter=mf,
            market_projection=mp,
            max_results=max_results,
            sort="FIRST_TO_START",
        )
        return catalogues or []
    except Exception as e:
        logger.error(f"list_market_catalogue failed: {e}")
        return []


async def resolve_place_market(
    venue: str,
    race_date: date,
    race_number: int,
    country: str = "AU",
) -> Optional[dict[str, Any]]:
    """Find a PLACE market for a specific race.

    Returns: {market_id, runners: [{selection_id, horse_name}]} or None
    """
    catalogues = await list_market_catalogue(
        venue, race_date, market_type="PLACE", country=country
    )

    if not catalogues:
        logger.info(f"No PLACE markets for {venue} on {race_date}")
        return None

    # Match by race number (position in sorted catalogue)
    if race_number > len(catalogues):
        logger.warning(f"Race {race_number} > {len(catalogues)} markets for {venue}")
        return None

    market = catalogues[race_number - 1]
    runners = []
    if hasattr(market, "runners") and market.runners:
        for r in market.runners:
            runners.append({
                "selection_id": r.selection_id,
                "horse_name": r.runner_name,
            })

    return {
        "market_id": market.market_id,
        "runners": runners,
    }


# ── Order Placement ──

async def place_bsp_place_bet(
    market_id: str,
    selection_id: int,
    stake: float,
    min_odds: float = 1.20,
) -> Optional[dict]:
    """Place a BSP LIMIT_ON_CLOSE back bet (place market).

    Returns: {bet_id, status, size_matched} or None on failure.
    """
    client = await get_client()
    if not client:
        return None

    try:
        from betfairlightweight.filters import limit_on_close_order

        order = limit_on_close_order(
            liability=stake,
            price=min_odds,
        )

        instructions = [{
            "orderType": "LIMIT_ON_CLOSE",
            "selectionId": selection_id,
            "side": "BACK",
            "limitOnCloseOrder": {
                "liability": stake,
                "price": min_odds,
            },
        }]

        result = await _run_sync(
            client.betting.place_orders,
            market_id=market_id,
            instructions=instructions,
        )

        if result and hasattr(result, "instruction_reports"):
            report = result.instruction_reports[0]
            return {
                "bet_id": getattr(report, "bet_id", None),
                "status": getattr(report, "status", "UNKNOWN"),
                "size_matched": getattr(report, "size_matched", 0),
                "error_code": getattr(report, "error_code", None),
            }

        return {"bet_id": None, "status": "UNKNOWN", "error_code": "NO_REPORT"}
    except Exception as e:
        logger.error(f"place_bsp_place_bet failed: {e}")
        return None


# ── Settlement ──

async def get_bet_result(bet_id: str) -> Optional[dict]:
    """Check settlement status of a bet.

    Returns: {status, profit, size_matched, price_matched} or None.
    """
    client = await get_client()
    if not client:
        return None

    try:
        # Check cleared orders first (settled)
        cleared = await _run_sync(
            client.betting.list_cleared_orders,
            bet_status="SETTLED",
            bet_ids=[bet_id],
        )

        if cleared and hasattr(cleared, "orders") and cleared.orders:
            order = cleared.orders[0]
            return {
                "status": "SETTLED",
                "profit": getattr(order, "profit", 0),
                "size_matched": getattr(order, "size_matched", 0),
                "price_matched": getattr(order, "price_matched", 0),
            }

        # Check for voided/lapsed/cancelled
        for status in ["VOIDED", "LAPSED", "CANCELLED"]:
            cleared = await _run_sync(
                client.betting.list_cleared_orders,
                bet_status=status,
                bet_ids=[bet_id],
            )
            if cleared and hasattr(cleared, "orders") and cleared.orders:
                return {
                    "status": status,
                    "profit": 0,
                    "size_matched": 0,
                    "price_matched": 0,
                }

        # Check current orders (still live)
        current = await _run_sync(
            client.betting.list_current_orders,
            bet_ids=[bet_id],
        )
        if current and hasattr(current, "orders") and current.orders:
            order = current.orders[0]
            return {
                "status": getattr(order, "status", "PENDING"),
                "profit": 0,
                "size_matched": getattr(order, "size_matched", 0),
                "price_matched": getattr(order, "price_matched", 0),
            }

        return None
    except Exception as e:
        logger.error(f"get_bet_result failed for {bet_id}: {e}")
        return None


# ── Account ──

async def get_account_funds() -> Optional[dict]:
    """Get current account balance.

    Returns: {available, exposure, total} or None.
    """
    client = await get_client()
    if not client:
        return None

    try:
        funds = await _run_sync(client.account.get_account_funds)
        return {
            "available": getattr(funds, "available_to_bet_balance", 0),
            "exposure": getattr(funds, "exposure", 0),
            "total": (
                getattr(funds, "available_to_bet_balance", 0)
                + abs(getattr(funds, "exposure", 0))
            ),
        }
    except Exception as e:
        logger.error(f"get_account_funds failed: {e}")
        return None


# ── Market Prices (for live odds) ──

async def get_market_book(market_id: str) -> Optional[dict]:
    """Get current prices for a market.

    Returns: {runners: [{selection_id, back_price, lay_price, last_traded}]}
    """
    client = await get_client()
    if not client:
        return None

    try:
        pp = price_projection(price_data=["EX_BEST_OFFERS"])
        books = await _run_sync(
            client.betting.list_market_book,
            market_ids=[market_id],
            price_projection=pp,
        )

        if not books:
            return None

        book = books[0]
        runners = []
        if hasattr(book, "runners"):
            for r in book.runners:
                back_price = 0
                lay_price = 0
                if hasattr(r, "ex"):
                    if r.ex.available_to_back:
                        back_price = r.ex.available_to_back[0].price
                    if r.ex.available_to_lay:
                        lay_price = r.ex.available_to_lay[0].price

                runners.append({
                    "selection_id": r.selection_id,
                    "back_price": back_price,
                    "lay_price": lay_price,
                    "last_traded": getattr(r, "last_price_traded", 0),
                    "status": getattr(r, "status", "ACTIVE"),
                })

        return {
            "market_id": book.market_id,
            "status": getattr(book, "status", "UNKNOWN"),
            "runners": runners,
        }
    except Exception as e:
        logger.error(f"get_market_book failed: {e}")
        return None
