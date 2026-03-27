"""Background scheduler for Betfair betting — JIT evaluation at T-5min."""

import asyncio
import logging
from datetime import datetime, timedelta

from punty.config import melb_now_naive, melb_today
from punty.models.database import async_session

logger = logging.getLogger(__name__)


class BetfairBetScheduler:
    """Polls every 30s. Scans for races starting in ~5 minutes and runs
    JIT evaluation (fresh probability + gates + Kelly + place bet)."""

    def __init__(self):
        self.running = False
        self.last_check: datetime | None = None
        self.bets_placed_today = 0
        self._task: asyncio.Task | None = None
        self._tick_count = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._poll_loop())
        asyncio.create_task(self._recover_zombie_bets())
        logger.info("BetfairBetScheduler started (JIT mode)")

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
        from sqlalchemy import select
        from punty.models.meeting import Meeting, Race
        from punty.models.betfair_bet import BetfairBet
        from punty.betting.jit import evaluate_and_bet_race
        from punty.betting.queue import sync_betfair_balance

        self.last_check = melb_now_naive()
        self._tick_count += 1
        now = self.last_check

        async with async_session() as db:
            # ── JIT: Find races starting in ~5 minutes ──
            # Window: [T-5m30s, T-4m30s] — 1-minute window, 30s poll catches it
            window_start = now + timedelta(minutes=4, seconds=30)
            window_end = now + timedelta(minutes=5, seconds=30)

            from sqlalchemy import or_
            race_result = await db.execute(
                select(Race).join(Meeting, Meeting.id == Race.meeting_id).where(
                    Meeting.date == melb_today(),
                    Meeting.selected == True,
                    Race.start_time >= window_start,
                    Race.start_time <= window_end,
                    or_(Race.results_status == "Open", Race.results_status == None),
                )
            )
            upcoming = race_result.scalars().all()

            for race in upcoming:
                # Check not already evaluated
                existing = await db.execute(
                    select(BetfairBet).where(
                        BetfairBet.meeting_id == race.meeting_id,
                        BetfairBet.race_number == race.race_number,
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                # JIT evaluate and bet
                try:
                    result = await evaluate_and_bet_race(db, race.meeting_id, race.race_number)
                    action = result.get("action", "?")
                    reason = result.get("reason", "")
                    horse = result.get("horse_name", "")

                    if action == "bet_placed":
                        self.bets_placed_today += 1
                        logger.info(
                            f"JIT BET: R{race.race_number} {horse} — {reason}"
                        )
                    elif action == "skipped":
                        logger.info(
                            f"JIT SKIP: {race.meeting_id} R{race.race_number} — {reason}"
                        )
                    else:
                        logger.warning(
                            f"JIT {action.upper()}: {race.meeting_id} R{race.race_number} — {reason}"
                        )
                except Exception as e:
                    logger.error(
                        f"JIT error for {race.meeting_id} R{race.race_number}: {e}",
                        exc_info=True,
                    )

            # ── Legacy: still execute any manually-queued bets ──
            from punty.betting.queue import execute_due_bets
            legacy_placed = await execute_due_bets(db)
            if legacy_placed:
                self.bets_placed_today += legacy_placed

            # ── Hourly balance sync ──
            if self._tick_count % 120 == 0:
                synced = await sync_betfair_balance(db)
                if not synced:
                    logger.warning("BetfairBetScheduler: hourly balance sync failed")

    async def _recover_zombie_bets(self):
        """Reset bets stuck in 'placing' from a crash/restart."""
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
                        bet.status = "cancelled"
                        bet.error_message = "Recovered from crash — JIT will re-evaluate"
                        cancelled += 1
                    else:
                        bet.status = "cancelled"
                        bet.error_message = "Recovered from crash — missed window"
                        cancelled += 1
                await db.commit()
                if cancelled:
                    logger.info(f"Recovered {cancelled} zombie bets (cancelled)")
        except Exception as e:
            logger.error(f"Zombie bet recovery failed: {e}")


# Module-level singleton
betfair_scheduler = BetfairBetScheduler()
