"""Flumine streaming framework integration for Betfair.

Runs Flumine in a background daemon thread alongside FastAPI.
Streams AU/NZ horse racing markets (WIN + PLACE), caches live
prices, and provides order placement via LimitOnCloseOrder (BSP).

The existing JIT evaluation (jit.py) and scheduler (scheduler.py)
remain unchanged — they just read from this cache instead of
making per-call httpx requests.

Falls back gracefully if flumine is not installed or credentials
are missing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Graceful import ──────────────────────────────────────────────
try:
    import betfairlightweight
    from betfairlightweight.filters import streaming_market_filter
    from betfairlightweight.resources import MarketBook

    from flumine import BaseStrategy, Flumine, clients
    from flumine.events.events import TerminationEvent
    from flumine.order.ordertype import LimitOnCloseOrder
    from flumine.order.trade import Trade

    _FLUMINE_AVAILABLE = True
except ImportError:
    _FLUMINE_AVAILABLE = False
    logger.debug("flumine not installed — streaming unavailable")


# ── Cache data classes ───────────────────────────────────────────

@dataclass
class RunnerSnapshot:
    """Point-in-time snapshot of a Betfair runner's prices."""
    selection_id: int
    horse_name: str
    back_price: float = 0.0
    lay_price: float = 0.0
    last_traded: float = 0.0
    status: str = "ACTIVE"


@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of a Betfair market."""
    market_id: str
    venue: str
    market_type: str   # "WIN" or "PLACE"
    start_time: datetime
    status: str
    runners: list[RunnerSnapshot] = field(default_factory=list)
    updated_at: float = 0.0  # time.monotonic()


# ── Strategy ─────────────────────────────────────────────────────

if _FLUMINE_AVAILABLE:

    class PuntyStrategy(BaseStrategy):
        """Streams market data into a shared cache. No bet logic here."""

        def __init__(self, manager: FlumineManager, **kwargs):
            super().__init__(**kwargs)
            self._manager = manager

        def check_market_book(self, market, market_book: MarketBook) -> bool:
            return market_book.status != "CLOSED"

        def process_market_book(self, market, market_book: MarketBook) -> None:
            """Update the shared market cache on every streaming tick."""
            try:
                md = market_book.market_definition
                venue = getattr(md, "venue", "") or ""
                market_type = getattr(md, "market_type", "") or ""
                start_time_str = getattr(md, "market_time", None)

                if isinstance(start_time_str, str):
                    start_time = datetime.fromisoformat(
                        start_time_str.replace("Z", "+00:00")
                    )
                elif isinstance(start_time_str, datetime):
                    start_time = start_time_str
                else:
                    start_time = datetime.now(tz=timezone.utc)

                # Build name lookup — streaming runners only have selection_id
                # + prices. Names come from catalogue (arrives ~60s after first
                # stream tick). Start from carry-forward, then override with
                # fresh data if available.
                cat_names = {}

                # Base: carry forward names from previous cache entry
                old = self._manager._market_cache.get(market_book.market_id)
                if old:
                    for r in old.runners:
                        if r.horse_name:
                            cat_names[r.selection_id] = r.horse_name

                # Override with market_definition.runners (stream metadata)
                if md and hasattr(md, "runners") and md.runners:
                    for dr in md.runners:
                        if isinstance(dr, dict):
                            sid = dr.get("id") or dr.get("selection_id")
                            rname = dr.get("name", "")
                        else:
                            sid = getattr(dr, "selection_id", None) or getattr(dr, "id", None)
                            rname = getattr(dr, "name", "") or ""
                        rname = re.sub(r"^\d+\.\s*", "", rname)
                        if sid and rname:
                            cat_names[sid] = rname

                # Override with market_catalogue (best source, populated ~60s in)
                cat = getattr(market, "market_catalogue", None)
                if cat and hasattr(cat, "runners") and cat.runners:
                    for cr in cat.runners:
                        sid = getattr(cr, "selection_id", None)
                        rname = getattr(cr, "runner_name", "") or ""
                        rname = re.sub(r"^\d+\.\s*", "", rname)
                        if sid and rname:
                            cat_names[sid] = rname

                runners = []
                for r in market_book.runners:
                    back_price = 0.0
                    lay_price = 0.0
                    if hasattr(r, "ex") and r.ex:
                        if r.ex.available_to_back:
                            b = r.ex.available_to_back[0]
                            back_price = b.get("price", 0) if isinstance(b, dict) else getattr(b, "price", 0)
                        if r.ex.available_to_lay:
                            l = r.ex.available_to_lay[0]
                            lay_price = l.get("price", 0) if isinstance(l, dict) else getattr(l, "price", 0)

                    name = cat_names.get(r.selection_id, "")

                    runners.append(RunnerSnapshot(
                        selection_id=r.selection_id,
                        horse_name=name,
                        back_price=back_price,
                        lay_price=lay_price,
                        last_traded=getattr(r, "last_price_traded", 0) or 0,
                        status=getattr(r, "status", "ACTIVE"),
                    ))

                snapshot = MarketSnapshot(
                    market_id=market_book.market_id,
                    venue=venue,
                    market_type=market_type,
                    start_time=start_time,
                    status=market_book.status,
                    runners=runners,
                    updated_at=time.monotonic(),
                )
                # Atomic dict update (thread-safe via GIL)
                is_new = market_book.market_id not in self._manager._market_cache
                named = sum(1 for r in runners if r.horse_name)

                self._manager._market_cache[market_book.market_id] = snapshot

                if is_new and venue:
                    logger.info(
                        f"Flumine cache: +{market_book.market_id} "
                        f"venue={venue} type={market_type} "
                        f"start={start_time} runners={len(runners)} "
                        f"named={named}"
                    )

            except Exception as e:
                logger.warning(f"PuntyStrategy cache update error: {e}", exc_info=True)

        def process_orders(self, market, orders: list) -> None:
            for order in orders:
                logger.info(
                    f"Flumine order update: sel={order.selection_id} "
                    f"status={order.status} matched={order.size_matched} "
                    f"@ {order.average_price_matched}"
                )

        def process_closed_market(self, market, market_book) -> None:
            market_id = market_book.market_id
            self._manager._market_cache.pop(market_id, None)
            logger.debug(f"Flumine market closed: {market_id}")


# ── Manager ──────────────────────────────────────────────────────

class FlumineManager:
    """Manages Flumine lifecycle and provides thread-safe market cache."""

    def __init__(self):
        self._framework = None
        self._thread: Optional[threading.Thread] = None
        self._market_cache: dict[str, MarketSnapshot] = {}
        self._running = False
        self._trading_client = None

    def is_available(self) -> bool:
        """True if flumine is installed AND streaming is running."""
        return _FLUMINE_AVAILABLE and self._running

    async def start(self, db=None) -> bool:
        """Start Flumine in a background daemon thread.

        Loads credentials from app_settings, creates bflw APIClient,
        wraps in Flumine client, starts streaming AU/NZ horse racing.

        Returns True on success, False if deps missing or creds not configured.
        """
        if not _FLUMINE_AVAILABLE:
            logger.info("Flumine not installed — skipping")
            return False

        if self._running:
            logger.debug("Flumine already running")
            return True

        # Get a logged-in bflw client
        from punty.betting.bflw_client import get_client
        trading = await get_client(db)
        if not trading:
            logger.warning("Flumine: no Betfair credentials — skipping")
            return False

        self._trading_client = trading

        try:
            client = clients.BetfairClient(trading)
            framework = Flumine(client=client)

            # Strategy: stream all AU/NZ horse racing WIN + PLACE markets
            strategy = PuntyStrategy(
                manager=self,
                market_filter=streaming_market_filter(
                    event_type_ids=["7"],
                    country_codes=["AU", "NZ"],
                    market_types=["WIN", "PLACE"],
                ),
                max_order_exposure=200,
                max_selection_exposure=200,
            )
            framework.add_strategy(strategy)

            self._framework = framework
            self._running = True

            self._thread = threading.Thread(
                target=self._run_framework,
                name="flumine-main",
                daemon=True,
            )
            self._thread.start()

            logger.info(
                "Flumine started — streaming AU/NZ horse racing "
                "(WIN + PLACE markets)"
            )
            return True

        except Exception as e:
            logger.error(f"Flumine startup failed: {e}", exc_info=True)
            self._running = False
            return False

    def _run_framework(self):
        """Target for the daemon thread. Flumine.run() blocks until stopped."""
        try:
            self._framework.run()
        except Exception as e:
            logger.error(f"Flumine thread crashed: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("Flumine thread exited")

    def stop(self):
        """Gracefully stop Flumine and join the thread."""
        if not self._running:
            return

        self._running = False

        if self._framework:
            try:
                self._framework.handler_queue.put(TerminationEvent({}))
            except Exception as e:
                logger.debug(f"Flumine termination event: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("Flumine thread did not stop within 10s")

        self._market_cache.clear()
        logger.info("Flumine stopped")

    # ── Market Cache Reads ───────────────────────────────────────

    def get_markets_for_race(
        self,
        venue: str,
        race_date: date,
        race_number: int,
        market_type: str = "PLACE",
    ) -> Optional[dict[str, Any]]:
        """Find a cached market for a specific race.

        Matches by venue name + start time proximity (within 5 min of
        the race's expected start). Falls back to positional matching.

        Returns: {market_id, runners: [{selection_id, horse_name}]} or None
        """
        if not self._running:
            return None

        from punty.scrapers.betfair import _strip_venue_prefix
        search_venue = _strip_venue_prefix(venue).lower()

        STALE_SECONDS = 300  # reject snapshots older than 5 min
        now = time.monotonic()

        # Log cache state on first lookup per race for diagnostics
        cache_snapshot = list(self._market_cache.values())
        if cache_snapshot:
            venues_in_cache = set()
            for s in cache_snapshot:
                if now - s.updated_at <= STALE_SECONDS and s.venue:
                    venues_in_cache.add(f"{s.venue}({s.market_type})")
            logger.info(
                f"Flumine cache lookup: venue={search_venue} type={market_type} "
                f"date={race_date} R{race_number} | "
                f"cache={len(cache_snapshot)} markets, "
                f"venues={sorted(venues_in_cache)[:15]}"
            )

        candidates = []
        for snap in cache_snapshot:
            if now - snap.updated_at > STALE_SECONDS:
                continue
            if snap.market_type != market_type:
                continue
            snap_venue = snap.venue.lower()
            if snap_venue != search_venue and search_venue not in snap_venue:
                continue
            # Date check: convert UTC start_time to Melbourne date
            if snap.start_time:
                import zoneinfo
                melb = zoneinfo.ZoneInfo("Australia/Melbourne")
                snap_melb = snap.start_time.astimezone(melb) if snap.start_time.tzinfo else snap.start_time
                snap_date = snap_melb.date()
            else:
                snap_date = None
            if snap_date != race_date:
                continue
            candidates.append(snap)

        if not candidates:
            logger.info(f"Flumine cache miss: no {market_type} markets for {search_venue} on {race_date}")
            return None

        # Sort by start time to match race_number positionally
        candidates.sort(key=lambda s: s.start_time)

        # Try positional match (Nth market = race N)
        idx = race_number - 1
        if idx < len(candidates):
            snap = candidates[idx]
            result = self._snap_to_dict(snap)

            # If runners have no names, backfill from bflw catalogue (one-time)
            named = sum(1 for r in result["runners"] if r["horse_name"])
            if not named and self._trading_client:
                try:
                    self._backfill_names(snap.market_id, result)
                except Exception as e:
                    logger.debug(f"Flumine name backfill failed: {e}")

            return result

        return None

    def _backfill_names(self, market_id: str, result: dict) -> None:
        """One-time bflw listMarketCatalogue call to get runner names."""
        catalogues = self._trading_client.betting.list_market_catalogue(
            filter={"marketIds": [market_id]},
            market_projection=["RUNNER_DESCRIPTION"],
            max_results=1,
        )
        if not catalogues:
            return
        cat = catalogues[0]
        name_map = {}
        for cr in cat.runners:
            sid = cr.selection_id
            rname = re.sub(r"^\d+\.\s*", "", cr.runner_name or "")
            if sid and rname:
                name_map[sid] = rname
        # Update result runners
        for r in result["runners"]:
            if not r["horse_name"] and r["selection_id"] in name_map:
                r["horse_name"] = name_map[r["selection_id"]]
        # Also update the cache so future lookups have names
        snap = self._market_cache.get(market_id)
        if snap:
            for sr in snap.runners:
                if not sr.horse_name and sr.selection_id in name_map:
                    sr.horse_name = name_map[sr.selection_id]
        named = sum(1 for r in result["runners"] if r["horse_name"])
        logger.info(f"Flumine name backfill: {market_id} → {named} runners named")

    def get_markets_for_race_by_time(
        self,
        venue: str,
        race_start_time: datetime,
        market_type: str = "PLACE",
    ) -> Optional[dict[str, Any]]:
        """Find a cached market by matching start time (within 5 min)."""
        if not self._running:
            return None

        from punty.scrapers.betfair import _strip_venue_prefix
        search_venue = _strip_venue_prefix(venue).lower()

        now = time.monotonic()
        best_snap = None
        best_diff = None

        for snap in self._market_cache.values():
            if now - snap.updated_at > 300:
                continue
            if snap.market_type != market_type:
                continue
            snap_venue = snap.venue.lower()
            if snap_venue != search_venue and search_venue not in snap_venue:
                continue

            diff = abs((snap.start_time - race_start_time).total_seconds())
            if diff < 300 and (best_diff is None or diff < best_diff):
                best_diff = diff
                best_snap = snap

        if best_snap:
            return self._snap_to_dict(best_snap)
        return None

    @staticmethod
    def _snap_to_dict(snap: MarketSnapshot) -> dict[str, Any]:
        """Convert a MarketSnapshot to the dict format jit.py expects."""
        return {
            "market_id": snap.market_id,
            "runners": [
                {
                    "selection_id": r.selection_id,
                    "horse_name": r.horse_name,
                }
                for r in snap.runners
                if r.status == "ACTIVE"
            ],
        }

    def get_runner_price(
        self, market_id: str, selection_id: int
    ) -> Optional[float]:
        """Get the current best back price for a runner from cache."""
        snap = self._market_cache.get(market_id)
        if not snap:
            return None
        if time.monotonic() - snap.updated_at > 300:
            return None
        for r in snap.runners:
            if r.selection_id == selection_id and r.status == "ACTIVE":
                return r.back_price if r.back_price > 0 else None
        return None

    # ── Order Placement ──────────────────────────────────────────

    def place_bsp_order(
        self,
        market_id: str,
        selection_id: int,
        stake: float,
        min_price: float = 1.20,
    ) -> dict[str, Any]:
        """Place a BSP LIMIT_ON_CLOSE back bet via Flumine.

        Must be called from within Flumine's thread context or via
        run_in_executor from the async loop.

        Returns: {bet_id, status, size_matched, order_type} or {status: 'failed', error: ...}
        """
        if not self._framework or not self._running:
            return {"status": "failed", "error": "Flumine not running"}

        # Find the Market object in Flumine's market registry
        target_market = None
        for market in self._framework.markets:
            if market.market_id == market_id:
                target_market = market
                break

        if not target_market:
            return {
                "status": "failed",
                "error": f"Market {market_id} not in Flumine registry",
            }

        try:
            trade = Trade(
                market_id=market_id,
                selection_id=selection_id,
                handicap=0,
                strategy=self._framework.strategies[0],
            )
            order = trade.create_order(
                side="BACK",
                order_type=LimitOnCloseOrder(
                    liability=round(stake, 2),
                    price=round(min_price, 2),
                ),
            )
            target_market.place_order(order)

            return {
                "bet_id": getattr(order, "bet_id", None),
                "status": "SUCCESS",
                "size_matched": getattr(order, "size_matched", 0),
                "order_type": "BSP",
            }
        except Exception as e:
            logger.error(f"Flumine place_bsp_order failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    # ── Account ──────────────────────────────────────────────────

    def get_account_balance(self) -> Optional[float]:
        """Read cached account balance from Flumine's client."""
        if not self._framework or not self._running:
            return None

        try:
            client = self._framework.clients.get_default()
            if client and client.account_funds:
                return getattr(
                    client.account_funds, "available_to_bet_balance", None
                )
        except Exception as e:
            logger.debug(f"Flumine get_account_balance: {e}")
        return None

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return current status for the API endpoint."""
        return {
            "available": _FLUMINE_AVAILABLE,
            "running": self._running,
            "markets_cached": len(self._market_cache),
            "thread_alive": (
                self._thread.is_alive() if self._thread else False
            ),
        }


# ── Module-level singleton ───────────────────────────────────────
flumine_manager = FlumineManager()
