"""Background results monitor — polls racing.com for completed races."""

import asyncio
import logging
import random
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

AEST = ZoneInfo("Australia/Melbourne")

# Racing typically runs between 11:00 and 18:30 AEST
RACING_START = time(10, 30)
RACING_END = time(19, 0)

# Poll interval range (seconds) — randomised each cycle
POLL_MIN = 90
POLL_MAX = 180
# Backoff interval when erroring
ERROR_POLL_MIN = 240
ERROR_POLL_MAX = 360
# Idle interval when outside racing hours
IDLE_POLL = 600


class ResultsMonitor:
    """Polls racing.com for completed races and triggers results generation."""

    def __init__(self, app):
        self.app = app
        self.running = False
        self.processed_races: dict[str, set[int]] = {}  # {meeting_id: {race_nums}}
        self.wrapups_generated: set[str] = set()
        self.task: Optional[asyncio.Task] = None
        self.last_check: Optional[datetime] = None
        self.poll_interval = POLL_MIN  # display value
        self.consecutive_errors = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._poll_loop())
        logger.info("Results monitor started")

    def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            self.task = None
        logger.info("Results monitor stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "processed_races": {k: list(v) for k, v in self.processed_races.items()},
            "wrapups_generated": list(self.wrapups_generated),
            "poll_interval": self.poll_interval,
            "consecutive_errors": self.consecutive_errors,
        }

    def _is_racing_hours(self) -> bool:
        """Check if current AEST time is within typical racing hours."""
        now_aest = datetime.now(AEST).time()
        return RACING_START <= now_aest <= RACING_END

    def _next_interval(self) -> float:
        """Calculate next poll interval with jitter."""
        if self.consecutive_errors >= 5:
            interval = random.uniform(ERROR_POLL_MIN, ERROR_POLL_MAX)
        elif not self._is_racing_hours():
            interval = IDLE_POLL + random.uniform(0, 60)
        else:
            interval = random.uniform(POLL_MIN, POLL_MAX)
        self.poll_interval = round(interval)
        return interval

    async def _poll_loop(self):
        self.consecutive_errors = 0
        while self.running:
            try:
                if self._is_racing_hours() or self._has_unfinished_meetings():
                    await self._check_all_meetings()
                    self.last_check = datetime.now(AEST)
                    self.consecutive_errors = 0
                else:
                    logger.debug("Outside racing hours — skipping check")
                    self.last_check = datetime.now(AEST)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Results monitor error ({self.consecutive_errors} consecutive): {e}", exc_info=True)
                if self.consecutive_errors >= 5:
                    logger.critical("Results monitor hit 5 consecutive errors — backing off")

            interval = self._next_interval()
            logger.debug(f"Next poll in {interval:.0f}s")
            await asyncio.sleep(interval)

    def _has_unfinished_meetings(self) -> bool:
        """Check if any tracked meetings still have unprocessed races."""
        for meeting_id, processed in self.processed_races.items():
            # If we've started tracking but haven't generated a wrapup, there's work to do
            if meeting_id not in self.wrapups_generated:
                return True
        return False

    async def _check_all_meetings(self):
        from punty.models.database import async_session
        from punty.models.meeting import Meeting, Race

        async with async_session() as db:
            from punty.config import melb_today
            today = melb_today()
            result = await db.execute(
                select(Meeting).where(Meeting.date == today)
            )
            meetings = result.scalars().all()

            for meeting in meetings:
                # Skip meetings where wrapup is already done
                if meeting.id in self.wrapups_generated:
                    continue

                # Only check meetings that have races (have been scraped)
                race_result = await db.execute(
                    select(Race).where(Race.meeting_id == meeting.id)
                )
                races = race_result.scalars().all()
                if not races:
                    continue

                # Skip if all races already processed
                processed = self.processed_races.get(meeting.id, set())
                if len(processed) >= len(races) and meeting.id in self.wrapups_generated:
                    continue

                await self._check_meeting(db, meeting, races)

                # Small random delay between meetings to avoid burst traffic
                if len(meetings) > 1:
                    await asyncio.sleep(random.uniform(3, 8))

            await db.commit()

    async def _check_meeting(self, db: AsyncSession, meeting, races):
        from punty.scrapers.racing_com import RacingComScraper
        from punty.scrapers.orchestrator import upsert_race_results
        from punty.ai.generator import ContentGenerator

        meeting_id = meeting.id
        if meeting_id not in self.processed_races:
            self.processed_races[meeting_id] = set()

        scraper = RacingComScraper()
        try:
            statuses = await scraper.check_race_statuses(meeting.venue, meeting.date)
        except Exception as e:
            logger.error(f"Failed to check statuses for {meeting.venue}: {e}")
            return
        finally:
            await scraper.close()

        for race_num, status in statuses.items():
            if status in ("Paying", "Closed") and race_num not in self.processed_races[meeting_id]:
                logger.info(f"New result: {meeting.venue} R{race_num} ({status})")
                try:
                    # Random delay before scraping detailed results
                    await asyncio.sleep(random.uniform(2, 6))

                    scraper2 = RacingComScraper()
                    try:
                        results_data = await scraper2.scrape_race_result(
                            meeting.venue, meeting.date, race_num
                        )
                    finally:
                        await scraper2.close()

                    await upsert_race_results(db, meeting_id, race_num, results_data)

                    # Verify runners were actually updated before proceeding
                    from sqlalchemy import select as sa_select
                    from punty.models.meeting import Runner as RunnerModel
                    race_id = f"{meeting_id}-r{race_num}"
                    check = await db.execute(
                        sa_select(RunnerModel).where(
                            RunnerModel.race_id == race_id,
                            RunnerModel.finish_position.isnot(None),
                        ).limit(1)
                    )
                    if not check.scalar_one_or_none():
                        logger.warning(f"No runner positions set for {meeting.venue} R{race_num} — skipping settlement & generation")
                        continue

                    # Settle picks for this race
                    try:
                        from punty.results.picks import settle_picks_for_race
                        await settle_picks_for_race(db, meeting_id, race_num)
                    except Exception as e:
                        logger.error(f"Failed to settle picks for {meeting.venue} R{race_num}: {e}")

                    # Check if results generation is enabled
                    from punty.models.settings import AppSettings
                    results_setting = await db.execute(
                        select(AppSettings).where(AppSettings.key == "enable_results")
                    )
                    results_enabled = results_setting.scalar_one_or_none()
                    if not results_enabled or results_enabled.value == "true":
                        generator = ContentGenerator(db)
                        await generator.generate_results(meeting_id, race_num, save=True)
                    else:
                        logger.info(f"Results generation disabled — skipping {meeting.venue} R{race_num}")

                    self.processed_races[meeting_id].add(race_num)
                    logger.info(f"Processed result: {meeting.venue} R{race_num}")
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to process result {meeting.venue} R{race_num}: {e}\n{traceback.format_exc()}")

        # Backfill exotic dividends from TabTouch for races missing them
        await self._backfill_tabtouch_exotics(db, meeting, statuses)

        # Check if all races done for wrap-up
        total_races = len(races)
        paying_count = sum(1 for s in statuses.values() if s in ("Paying", "Closed"))
        if paying_count >= total_races and meeting_id not in self.wrapups_generated:
            # Check if meeting wrapup is enabled
            from punty.models.settings import AppSettings
            wrapup_setting = await db.execute(
                select(AppSettings).where(AppSettings.key == "enable_meeting_wrapup")
            )
            wrapup_enabled = wrapup_setting.scalar_one_or_none()
            if wrapup_enabled and wrapup_enabled.value != "true":
                logger.info(f"All races done for {meeting.venue} — skipping wrap-up (disabled in settings)")
                self.wrapups_generated.add(meeting_id)  # Mark as done to avoid retrying
            else:
                logger.info(f"All races done for {meeting.venue} — generating wrap-up")
                try:
                    generator = ContentGenerator(db)
                    await generator.generate_meeting_wrapup(meeting_id, save=True)
                    self.wrapups_generated.add(meeting_id)
                    logger.info(f"Wrap-up generated for {meeting.venue}")
                except Exception as e:
                    logger.error(f"Failed to generate wrap-up for {meeting.venue}: {e}")

    async def _backfill_tabtouch_exotics(self, db, meeting, statuses):
        """Fetch exotic dividends from TabTouch for races that are missing them."""
        import json as _json
        from punty.models.meeting import Race

        # Only bother if we have paying races
        paying_races = [rn for rn, s in statuses.items() if s in ("Paying", "Closed")]
        if not paying_races:
            return

        # Check which paying races are missing exotic_results
        missing = []
        for rn in paying_races:
            race_id = f"{meeting.id}-r{rn}"
            race = await db.get(Race, race_id)
            if race and not race.exotic_results:
                missing.append(rn)

        if not missing:
            return

        try:
            from punty.scrapers.tabtouch import find_venue_code, scrape_meeting_exotics

            venue_code = await find_venue_code(meeting.venue, meeting.date)
            if not venue_code:
                logger.debug(f"No TabTouch venue code found for {meeting.venue}")
                return

            exotics_by_race = await scrape_meeting_exotics(venue_code, meeting.date)
            if not exotics_by_race:
                return

            updated = 0
            for rn, exotics in exotics_by_race.items():
                race_id = f"{meeting.id}-r{rn}"
                race = await db.get(Race, race_id)
                if race and not race.exotic_results and exotics:
                    race.exotic_results = _json.dumps(exotics)
                    updated += 1

            if updated:
                await db.flush()
                logger.info(f"TabTouch backfilled exotics for {updated} races at {meeting.venue}")

                # Re-settle sequence picks that may now have dividend data
                # First, unsettled any sequence picks that were hit but got $0 pnl (no dividend at time)
                from punty.models.pick import Pick
                from sqlalchemy import update
                await db.execute(
                    update(Pick).where(
                        Pick.meeting_id == meeting.id,
                        Pick.pick_type == "sequence",
                        Pick.settled == True,
                        Pick.hit == True,
                        Pick.pnl == 0.0,
                    ).values(settled=False, settled_at=None)
                )
                await db.flush()

                from punty.results.picks import settle_picks_for_race
                for rn in exotics_by_race:
                    try:
                        await settle_picks_for_race(db, meeting.id, rn)
                    except Exception as e:
                        logger.error(f"Failed to re-settle picks after TabTouch backfill R{rn}: {e}")

        except Exception as e:
            logger.debug(f"TabTouch exotic backfill failed for {meeting.venue}: {e}")

    async def check_single_meeting(self, meeting_id: str):
        """Manual one-shot check for a specific meeting."""
        from punty.models.database import async_session
        from punty.models.meeting import Meeting, Race

        async with async_session() as db:
            meeting = await db.get(Meeting, meeting_id)
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            race_result = await db.execute(
                select(Race).where(Race.meeting_id == meeting_id)
            )
            races = race_result.scalars().all()
            if not races:
                raise ValueError(f"No races found for {meeting_id}")

            await self._check_meeting(db, meeting, races)
            await db.commit()
