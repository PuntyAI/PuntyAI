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
        self._tick_count = 0  # 30s ticks — sync balance every 120 ticks (1 hour)

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._poll_loop())
        # Recover zombie bets left in "placing" from a previous crash/restart
        asyncio.create_task(self._recover_zombie_bets())
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
        from punty.betting.queue import execute_due_bets, refresh_bet_selections, sync_betfair_balance

        self.last_check = melb_now_naive()
        self._tick_count += 1
        async with async_session() as db:
            swaps = await refresh_bet_selections(db)
            if swaps:
                logger.info(f"BetfairBetScheduler: {swaps} bet swap(s) this tick")
            placed = await execute_due_bets(db)
            if placed:
                self.bets_placed_today += placed
                logger.info(f"BetfairBetScheduler: placed {placed} bets this tick")

            # Sync balance from Betfair API every hour (120 × 30s ticks)
            if self._tick_count % 120 == 0:
                synced = await sync_betfair_balance(db)
                if not synced:
                    logger.warning("BetfairBetScheduler: hourly balance sync failed")


    async def _recover_zombie_bets(self):
        """Reset bets stuck in 'placing' from a crash/restart.

        If a bet was mid-placement when the server died, it will be stuck
        in 'placing' forever. Reset to 'queued' if the race hasn't started,
        or 'cancelled' if the window has passed.
        """
        try:
            from punty.models.betfair_bet import BetfairBet
            from sqlalchemy import select

            async with async_session() as db:
                result = await db.execute(
                    select(BetfairBet).where(BetfairBet.status == "placing")
                )
                zombies = result.scalars().all()
                if not zombies:
                    return

                now = melb_now_naive()
                recovered = 0
                cancelled = 0
                for bet in zombies:
                    if bet.scheduled_at and bet.scheduled_at > now:
                        # Race hasn't started — re-queue
                        bet.status = "queued"
                        bet.error_message = None
                        recovered += 1
                    else:
                        # Race already started — cancel
                        bet.status = "cancelled"
                        bet.error_message = "Recovered from crash — missed window"
                        cancelled += 1
                await db.commit()
                logger.info(
                    f"Recovered {recovered} zombie bets (re-queued), "
                    f"cancelled {cancelled} (missed window)"
                )
        except Exception as e:
            logger.error(f"Zombie bet recovery failed: {e}")


# Module-level singleton
betfair_scheduler = BetfairBetScheduler()
