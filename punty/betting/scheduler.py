"""Background scheduler for automated Betfair bet placement."""

import asyncio
import logging
from datetime import datetime

from punty.config import melb_now_naive
from punty.models.database import async_session

logger = logging.getLogger(__name__)


class BetfairBetScheduler:
    """Polls every 30s to execute due Betfair bets, similar to ResultsMonitor."""

    def __init__(self):
        self.running = False
        self.last_check: datetime | None = None
        self.bets_placed_today = 0
        self._task: asyncio.Task | None = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("BetfairBetScheduler started")

    def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("BetfairBetScheduler stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "bets_placed_today": self.bets_placed_today,
        }

    async def _poll_loop(self):
        while self.running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"BetfairBetScheduler error: {e}", exc_info=True)
            await asyncio.sleep(30)

    async def _tick(self):
        from punty.betting.queue import execute_due_bets, refresh_bet_selections

        self.last_check = melb_now_naive()
        async with async_session() as db:
            swaps = await refresh_bet_selections(db)
            if swaps:
                logger.info(f"BetfairBetScheduler: {swaps} bet swap(s) this tick")
            placed = await execute_due_bets(db)
            if placed:
                self.bets_placed_today += placed
                logger.info(f"BetfairBetScheduler: placed {placed} bets this tick")


# Module-level singleton
betfair_scheduler = BetfairBetScheduler()
