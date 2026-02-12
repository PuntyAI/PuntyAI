"""Background results monitor — polls racing.com for completed races."""

import asyncio
import logging
import random
from datetime import datetime, time, timedelta
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import MELB_TZ, melb_now

logger = logging.getLogger(__name__)

AEST = MELB_TZ  # Alias for backwards compat

# Monitor stops 20 minutes after last race completes (or wrap-up generated)
POST_RACING_BUFFER = timedelta(minutes=20)

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
        self.pace_updates_posted: dict[str, int] = {}  # {meeting_id: count}
        self.alerted_changes: dict[str, set[str]] = {}  # {meeting_id: set of dedup keys}
        self.last_change_check: dict[str, datetime] = {}  # rate limit per meeting
        self.last_jockey_check: dict[str, datetime] = {}  # Playwright rate limit
        self.last_weather_check: dict[str, datetime] = {}  # WillyWeather rate limit
        self.task: Optional[asyncio.Task] = None
        self.last_check: Optional[datetime] = None
        self.poll_interval = POLL_MIN  # display value
        self.consecutive_errors = 0
        self._last_reset_date = melb_now().date()

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
            "alerted_changes": {k: len(v) for k, v in self.alerted_changes.items()},
        }

    async def _should_be_monitoring(self) -> bool:
        """Check if monitor should be active based on actual race times."""
        from punty.models.database import async_session
        from punty.models.meeting import Meeting, Race
        from punty.config import melb_today

        now_aest = melb_now()

        async with async_session() as db:
            today = melb_today()

            # Get all selected meetings for today
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                )
            )
            meetings = result.scalars().all()

            if not meetings:
                return False

            meeting_ids = [m.id for m in meetings]

            # Get first and last race start times across all selected meetings
            first_result = await db.execute(
                select(func.min(Race.start_time)).where(
                    Race.meeting_id.in_(meeting_ids)
                )
            )
            first_race_time = first_result.scalar_one_or_none()

            last_result = await db.execute(
                select(func.max(Race.start_time)).where(
                    Race.meeting_id.in_(meeting_ids)
                )
            )
            last_race_time = last_result.scalar_one_or_none()

            if not first_race_time:
                # No races loaded yet - don't monitor until data exists
                return False

            # Don't start until first race time (DST-safe: use tz-aware comparisons)
            first_time = first_race_time.time() if isinstance(first_race_time, datetime) else first_race_time
            first_dt = datetime.combine(today, first_time, tzinfo=AEST)

            if now_aest < first_dt:
                return False

            # Check if any races are still incomplete
            incomplete_result = await db.execute(
                select(Race).where(
                    Race.meeting_id.in_(meeting_ids),
                    Race.results_status.notin_(["Paying", "Closed", "Final"])
                ).limit(1)
            )
            has_incomplete = incomplete_result.scalar_one_or_none() is not None

            if has_incomplete:
                return True

            # All races complete — check if wrap-ups are done for all meetings
            from punty.models.content import Content
            wrapup_result = await db.execute(
                select(func.count(Content.id)).where(
                    Content.meeting_id.in_(meeting_ids),
                    Content.content_type == "meeting_wrapup",
                    Content.status.notin_(["rejected", "superseded"]),
                )
            )
            wrapup_count = wrapup_result.scalar() or 0
            all_wrapups_done = wrapup_count >= len(meeting_ids)

            if not all_wrapups_done:
                # Still waiting for wrap-ups to generate
                return True

            # All races complete + all wrap-ups done — apply buffer (DST-safe)
            last_time = last_race_time.time() if isinstance(last_race_time, datetime) else last_race_time
            last_dt = datetime.combine(today, last_time, tzinfo=AEST)
            cutoff = last_dt + POST_RACING_BUFFER

            if now_aest <= cutoff:
                return True

            logger.info(f"All races complete, wrap-ups done, buffer passed (last race: {last_race_time}) — monitor going idle")
            return False

    async def _next_interval(self) -> float:
        """Calculate next poll interval with jitter."""
        if self.consecutive_errors >= 5:
            interval = random.uniform(ERROR_POLL_MIN, ERROR_POLL_MAX)
        elif not await self._should_be_monitoring():
            interval = IDLE_POLL + random.uniform(0, 60)
        else:
            interval = random.uniform(POLL_MIN, POLL_MAX)
        self.poll_interval = round(interval)
        return interval

    async def _poll_loop(self):
        self.consecutive_errors = 0

        # One-time retrospective check for missed celebrations/alerts after restart
        try:
            await self._check_retrospective_updates()
        except Exception as e:
            logger.warning(f"Retrospective check failed: {e}", exc_info=True)

        while self.running:
            # Reset tracking state at midnight to prevent unbounded growth
            today = melb_now().date()
            if today > self._last_reset_date:
                old_races = sum(len(s) for s in self.processed_races.values())
                old_wrapups = len(self.wrapups_generated)
                self.processed_races.clear()
                self.wrapups_generated.clear()
                self.pace_updates_posted.clear()
                self.alerted_changes.clear()
                self.last_change_check.clear()
                self.last_jockey_check.clear()
                self.last_weather_check.clear()
                self._last_reset_date = today
                logger.info(f"Daily reset: cleared {old_races} processed races, {old_wrapups} wrapups")

            try:
                if await self._should_be_monitoring():
                    await self._check_all_meetings()
                    self.last_check = melb_now()
                    self.consecutive_errors = 0
                else:
                    logger.debug("Outside active monitoring window — skipping check")
                    self.last_check = melb_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Results monitor error ({self.consecutive_errors} consecutive): {e}", exc_info=True)
                if self.consecutive_errors >= 5:
                    logger.critical("Results monitor hit 5 consecutive errors — backing off")

            interval = await self._next_interval()
            logger.debug(f"Next poll in {interval:.0f}s")
            await asyncio.sleep(interval)

    async def _check_retrospective_updates(self):
        """Check for missed celebrations and change alerts after restart.

        Runs once on monitor startup. Finds settled races from today that
        are missing LiveUpdate records for celebrations/clean sweeps, and
        picks affected by scratchings with no alert posted.
        """
        from punty.models.database import async_session
        from punty.models.meeting import Meeting, Race
        from punty.models.pick import Pick
        from punty.models.live_update import LiveUpdate
        from punty.config import melb_today
        from sqlalchemy import and_

        async with async_session() as db:
            today = melb_today()

            # Find today's selected meetings
            meetings_result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                )
            )
            meetings = meetings_result.scalars().all()
            if not meetings:
                return

            meeting_map = {m.id: m for m in meetings}
            meeting_ids = list(meeting_map.keys())

            # Find races with settled picks today
            settled_result = await db.execute(
                select(Pick.meeting_id, Pick.race_number)
                .where(
                    Pick.meeting_id.in_(meeting_ids),
                    Pick.pick_type == "selection",
                    Pick.settled == True,
                )
                .distinct()
            )
            settled_races = settled_result.all()

            if not settled_races:
                return

            # Find existing LiveUpdates for today (to avoid duplicates)
            existing_result = await db.execute(
                select(LiveUpdate.meeting_id, LiveUpdate.race_number, LiveUpdate.update_type)
                .where(LiveUpdate.meeting_id.in_(meeting_ids))
            )
            existing_updates = set()
            for row in existing_result.all():
                existing_updates.add((row[0], row[1], row[2]))

            posted = 0
            for meeting_id, race_number in settled_races:
                meeting = meeting_map.get(meeting_id)
                if not meeting:
                    continue

                # Check for missed clean sweep
                if (meeting_id, race_number, "clean_sweep") not in existing_updates:
                    try:
                        await self._check_clean_sweep(db, meeting, race_number)
                        # Check if one was actually posted
                        check = await db.execute(
                            select(LiveUpdate).where(
                                LiveUpdate.meeting_id == meeting_id,
                                LiveUpdate.race_number == race_number,
                                LiveUpdate.update_type == "clean_sweep",
                            ).limit(1)
                        )
                        if check.scalar_one_or_none():
                            posted += 1
                    except Exception as e:
                        logger.debug(f"Retrospective clean sweep check failed for {meeting_id} R{race_number}: {e}")

                # Check for missed big win celebrations
                if (meeting_id, race_number, "celebration") not in existing_updates:
                    try:
                        await self._check_big_wins(db, meeting_id, race_number)
                    except Exception as e:
                        logger.debug(f"Retrospective big win check failed for {meeting_id} R{race_number}: {e}")

            # Check for missed scratching alerts on unsettled picks
            for meeting_id in meeting_ids:
                meeting = meeting_map[meeting_id]
                # Find our unsettled picks
                unsettled_result = await db.execute(
                    select(Pick).where(
                        Pick.meeting_id == meeting_id,
                        Pick.pick_type == "selection",
                        Pick.settled == False,
                    )
                )
                unsettled_picks = unsettled_result.scalars().all()
                if not unsettled_picks:
                    continue

                # Check if any picked runners are scratched
                from punty.models.meeting import Runner
                for pick in unsettled_picks:
                    if not pick.race_number or not pick.horse_name:
                        continue
                    race_id = f"{meeting_id}-r{pick.race_number}"
                    runner_result = await db.execute(
                        select(Runner).where(
                            Runner.race_id == race_id,
                            Runner.horse_name == pick.horse_name,
                            Runner.scratched == True,
                        ).limit(1)
                    )
                    scratched_runner = runner_result.scalar_one_or_none()
                    if not scratched_runner:
                        continue

                    # Check if alert already exists
                    dedup_key = f"scratching:R{pick.race_number}:{pick.horse_name}"
                    if meeting_id not in self.alerted_changes:
                        self.alerted_changes[meeting_id] = set()
                    if dedup_key in self.alerted_changes[meeting_id]:
                        continue

                    existing_alert = await db.execute(
                        select(LiveUpdate).where(
                            LiveUpdate.meeting_id == meeting_id,
                            LiveUpdate.race_number == pick.race_number,
                            LiveUpdate.update_type == "scratching_alert",
                            LiveUpdate.horse_name == pick.horse_name,
                        ).limit(1)
                    )
                    if existing_alert.scalar_one_or_none():
                        self.alerted_changes[meeting_id].add(dedup_key)
                        continue

                    # Compose and post scratching alert
                    from punty.results.change_detection import (
                        find_impacted_picks, find_alternative, compose_scratching_alert
                    )
                    impacts = await find_impacted_picks(
                        db, meeting_id, pick.race_number,
                        pick.horse_name, scratched_runner.saddlecloth,
                    )
                    alternative = await find_alternative(
                        db, meeting_id, pick.race_number,
                        scratched_runner.saddlecloth,
                    )
                    from punty.results.change_detection import ChangeAlert
                    alert = ChangeAlert(
                        change_type="scratching",
                        meeting_id=meeting_id,
                        race_number=pick.race_number,
                        horse_name=pick.horse_name,
                        saddlecloth=scratched_runner.saddlecloth,
                        message=compose_scratching_alert(
                            pick.horse_name, pick.race_number,
                            impacts, alternative,
                        ),
                        impacts=impacts,
                        alternative=alternative,
                    )
                    self.alerted_changes[meeting_id].add(dedup_key)
                    try:
                        await self._post_change_alert(db, meeting, alert)
                        posted += 1
                        logger.info(
                            f"Retrospective scratching alert: {pick.horse_name} "
                            f"(R{pick.race_number}, {meeting.venue})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to post retrospective scratching alert: {e}")

            await db.commit()

            if posted:
                logger.info(f"Retrospective check: posted {posted} missed updates")
            else:
                logger.info("Retrospective check: no missed updates")

    async def _check_all_meetings(self):
        from punty.models.database import async_session
        from punty.models.meeting import Meeting, Race

        async with async_session() as db:
            from punty.config import melb_today
            today = melb_today()
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,  # Only process active/selected meetings
                )
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

        # Pre-race change detection (scratchings, track condition, jockey/gear)
        try:
            await self._check_pre_race_changes(db, meeting, races, statuses)
        except Exception as e:
            logger.warning(f"Pre-race change check failed for {meeting.venue}: {e}", exc_info=True)

        # Weather refresh (every 30 min per meeting)
        try:
            await self._check_weather_changes(db, meeting)
        except Exception as e:
            logger.warning(f"Weather check failed for {meeting.venue}: {e}", exc_info=True)

        for race_num, status in statuses.items():
            if status in ("Paying", "Closed") and race_num not in self.processed_races[meeting_id]:
                # Check if race already has results in DB (from before restart)
                from punty.models.meeting import Runner as RunnerModel
                race_id = f"{meeting_id}-r{race_num}"
                existing_check = await db.execute(
                    select(RunnerModel).where(
                        RunnerModel.race_id == race_id,
                        RunnerModel.finish_position.isnot(None),
                    ).limit(1)
                )
                if existing_check.scalar_one_or_none():
                    # Results exist - but check if picks need settlement
                    from punty.models.pick import Pick
                    unsettled = await db.execute(
                        select(Pick).where(
                            Pick.meeting_id == meeting_id,
                            Pick.race_number == race_num,
                            Pick.settled == False,
                            Pick.pick_type.in_(["selection", "big3", "exotic"]),
                        ).limit(1)
                    )
                    if unsettled.scalar_one_or_none():
                        # Run settlement for unsettled picks
                        logger.info(f"Settling unsettled picks for {meeting.venue} R{race_num}")
                        try:
                            from punty.results.picks import settle_picks_for_race
                            await settle_picks_for_race(db, meeting_id, race_num)
                        except Exception as e:
                            logger.error(f"Failed to settle picks for {meeting.venue} R{race_num}: {e}")

                    self.processed_races[meeting_id].add(race_num)
                    logger.debug(f"Skipping {meeting.venue} R{race_num} — already has results")
                    continue

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

                    # Try to scrape sectional times (may not be available yet - typically 10-15min delay)
                    try:
                        await self._scrape_sectional_times(db, meeting, race_num)
                    except Exception as e:
                        logger.debug(f"Sectional times not yet available for {meeting.venue} R{race_num}: {e}")

                    # Check if results generation is enabled
                    from punty.models.settings import AppSettings
                    results_setting = await db.execute(
                        select(AppSettings).where(AppSettings.key == "enable_results")
                    )
                    results_enabled = results_setting.scalar_one_or_none()
                    if not results_enabled or results_enabled.value == "true":
                        # Check if result content already exists (prevents duplicates on restart)
                        from punty.models.content import Content
                        existing_result = await db.execute(
                            select(Content).where(
                                Content.meeting_id == meeting_id,
                                Content.content_type == "race_result",
                                Content.race_id == race_id,
                                Content.status.notin_(["rejected", "superseded"]),
                            )
                        )
                        if existing_result.scalar_one_or_none():
                            logger.info(f"Result content already exists for {meeting.venue} R{race_num} — skipping")
                        else:
                            generator = ContentGenerator(db)
                            await generator.generate_results(meeting_id, race_num, save=True)
                    else:
                        logger.info(f"Results generation disabled — skipping {meeting.venue} R{race_num}")

                    self.processed_races[meeting_id].add(race_num)
                    logger.info(f"Processed result: {meeting.venue} R{race_num}")

                    # Check for big wins and post celebration tweet replies
                    try:
                        await self._check_big_wins(db, meeting_id, race_num)
                    except Exception as e:
                        logger.debug(f"Big win check failed for {meeting.venue} R{race_num}: {e}")

                    # Check for clean sweep (all selections won)
                    try:
                        await self._check_clean_sweep(db, meeting, race_num)
                    except Exception as e:
                        logger.debug(f"Clean sweep check failed for {meeting.venue} R{race_num}: {e}")

                    # Check for pace bias and post analysis tweet (after race 4+)
                    try:
                        await self._check_pace_bias(db, meeting, statuses, race_num)
                    except Exception as e:
                        logger.debug(f"Pace analysis skipped for {meeting.venue} R{race_num}: {e}")
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to process result {meeting.venue} R{race_num}: {e}\n{traceback.format_exc()}")

        # Backfill exotic dividends from TabTouch for races missing them
        await self._backfill_tabtouch_exotics(db, meeting, statuses)

        # Backfill sectional times for races that are missing them (10-15min delay after race)
        await self._backfill_sectionals(db, meeting, races)

        # Check if all races done for wrap-up
        # CRITICAL: Verify actual results exist in DB, not just racing.com status
        total_races = len(races)
        paying_count = sum(1 for s in statuses.values() if s in ("Paying", "Closed"))
        results_count = 0
        if paying_count >= total_races:
            from punty.models.meeting import Runner as RunnerModel
            for race in races:
                has_result = await db.execute(
                    select(RunnerModel).where(
                        RunnerModel.race_id == race.id,
                        RunnerModel.finish_position.isnot(None),
                    ).limit(1)
                )
                if has_result.scalar_one_or_none():
                    results_count += 1
            if results_count < total_races:
                logger.warning(
                    f"{meeting.venue}: {paying_count}/{total_races} races paying but only "
                    f"{results_count}/{total_races} have results scraped — holding wrap-up"
                )
        if paying_count >= total_races and results_count >= total_races and meeting_id not in self.wrapups_generated:
            # Check if a wrapup already exists in DB (prevents duplicates on restart)
            from punty.models.content import Content
            existing_wrapup = await db.execute(
                select(Content).where(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "meeting_wrapup",
                    Content.status.notin_(["rejected", "superseded"]),
                )
            )
            if existing_wrapup.scalar_one_or_none():
                logger.info(f"Wrap-up already exists for {meeting.venue} — skipping")
                self.wrapups_generated.add(meeting_id)
                return

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
                    content = await generator.generate_meeting_wrapup(meeting_id, save=True)
                    self.wrapups_generated.add(meeting_id)
                    logger.info(f"Wrap-up generated for {meeting.venue}")

                    # Auto-approve and post to Twitter + Facebook
                    if content:
                        from punty.scheduler.automation import auto_approve_content, auto_post_to_twitter, auto_post_to_facebook
                        content_id = content.get('content_id') if isinstance(content, dict) else None
                        if content_id:
                            approval = await auto_approve_content(content_id, db)
                            if approval.get("status") == "approved":
                                logger.info(f"Wrap-up auto-approved for {meeting.venue}")
                                twitter_result = await auto_post_to_twitter(content_id, db)
                                facebook_result = await auto_post_to_facebook(content_id, db)
                                # Alert on delivery failures
                                failures = []
                                if twitter_result.get("status") == "error":
                                    failures.append(f"Twitter: {twitter_result.get('message', 'unknown')}")
                                if facebook_result.get("status") == "error":
                                    failures.append(f"Facebook: {facebook_result.get('message', 'unknown')}")
                                if failures:
                                    from punty.scheduler.automation import _send_delivery_failure_alert
                                    await _send_delivery_failure_alert(db, content_id, failures)
                            else:
                                logger.warning(f"Wrap-up auto-approval failed for {meeting.venue}: {approval.get('issues')}")
                except Exception as e:
                    logger.error(f"Failed to generate wrap-up for {meeting.venue}: {e}")

    async def _get_thread_parent(self, db: AsyncSession, meeting_id: str, early_mail_tweet_id: str) -> str:
        """Get the last tweet ID in the thread chain for this meeting.

        Twitter threads require each reply to chain off the previous one
        (not all reply to the root). This finds the most recent LiveUpdate
        tweet for the meeting and returns its ID, falling back to the
        early mail tweet if no updates have been posted yet.
        """
        from punty.models.live_update import LiveUpdate
        result = await db.execute(
            select(LiveUpdate.tweet_id).where(
                LiveUpdate.meeting_id == meeting_id,
                LiveUpdate.tweet_id.isnot(None),
            ).order_by(LiveUpdate.created_at.desc()).limit(1)
        )
        last_tweet_id = result.scalar_one_or_none()
        return last_tweet_id or early_mail_tweet_id

    async def _check_big_wins(self, db: AsyncSession, meeting_id: str, race_number: int):
        """Check for big wins (>= 5x outlay) and post celebration replies."""
        from punty.models.pick import Pick
        from punty.models.content import Content
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery
        from punty.delivery.facebook import FacebookDelivery
        from punty.results.celebrations import compose_celebration_tweet

        result = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting_id,
                Pick.race_number == race_number,
                Pick.pick_type == "selection",
                Pick.settled == True,
                Pick.hit == True,
                Pick.pnl > 0,
            )
        )
        for pick in result.scalars().all():
            stake = pick.bet_stake or 1.0
            if pick.pnl < 4 * stake:
                continue

            # Find the meeting's early mail
            em_result = await db.execute(
                select(Content).where(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "early_mail",
                ).order_by(Content.created_at.desc())
            )
            early_mail = em_result.scalars().first()
            if not early_mail:
                return

            collect = stake + pick.pnl
            tweet_text = compose_celebration_tweet(
                horse_name=pick.horse_name or "Winner",
                odds=pick.odds_at_tip or 0,
                stake=stake,
                collect=collect,
                bet_type=pick.bet_type or "Win",
            )

            # Post to Twitter (chain off last tweet in thread)
            reply_tweet_id = None
            thread_parent = None
            if early_mail.twitter_id:
                thread_parent = await self._get_thread_parent(db, meeting_id, early_mail.twitter_id)
                twitter = TwitterDelivery(db)
                if await twitter.is_configured():
                    try:
                        reply_result = await twitter.post_reply(thread_parent, tweet_text)
                        reply_tweet_id = reply_result.get("tweet_id")
                        logger.info(f"Posted celebration reply for {pick.horse_name} ({pick.pnl:+.2f})")
                    except Exception as e:
                        logger.warning(f"Failed to post celebration reply: {e}")

            # Facebook live updates disabled until commenting permission is resolved
            fb_post_id = None

            # Save to DB regardless of post success
            update = LiveUpdate(
                meeting_id=meeting_id,
                race_number=race_number,
                update_type="celebration",
                content=tweet_text,
                tweet_id=reply_tweet_id,
                parent_tweet_id=thread_parent or early_mail.twitter_id,
                facebook_comment_id=fb_post_id,
                parent_facebook_id=early_mail.facebook_id,
                horse_name=pick.horse_name,
                odds=pick.odds_at_tip,
                pnl=pick.pnl,
            )
            db.add(update)
            await db.flush()

    async def _check_clean_sweep(self, db: AsyncSession, meeting, race_number: int):
        """Check if ALL selections in a race won — post a big celebration."""
        from punty.models.pick import Pick
        from punty.models.content import Content
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery
        from punty.delivery.facebook import FacebookDelivery
        from punty.results.celebrations import compose_clean_sweep_tweet

        # Get all selections for this race
        result = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting.id,
                Pick.race_number == race_number,
                Pick.pick_type == "selection",
                Pick.settled == True,
            )
        )
        selections = result.scalars().all()

        if len(selections) < 2:
            return

        # Check ALL selections hit (placed or won)
        if not all(s.hit for s in selections):
            return

        # Build winners info + totals
        winners = []
        total_pnl = 0.0
        total_collect = 0.0
        for s in selections:
            winners.append({
                "horse_name": s.horse_name or "Winner",
                "odds": s.odds_at_tip or 0,
            })
            total_pnl += s.pnl or 0
            stake = s.bet_stake or 1.0
            total_collect += stake + (s.pnl or 0)

        tweet_text = compose_clean_sweep_tweet(
            venue=meeting.venue,
            race_number=race_number,
            winners=winners,
            total_pnl=total_pnl,
            total_collect=total_collect,
        )

        logger.info(f"CLEAN SWEEP! {meeting.venue} R{race_number} — all {len(selections)} selections hit!")

        # Find early mail for threading
        em_result = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting.id,
                Content.content_type == "early_mail",
            ).order_by(Content.created_at.desc())
        )
        early_mail = em_result.scalars().first()
        if not early_mail:
            return

        # Post to Twitter (chain off last tweet in thread)
        reply_tweet_id = None
        thread_parent = None
        if early_mail.twitter_id:
            thread_parent = await self._get_thread_parent(db, meeting.id, early_mail.twitter_id)
            twitter = TwitterDelivery(db)
            if await twitter.is_configured():
                try:
                    reply_result = await twitter.post_reply(thread_parent, tweet_text)
                    reply_tweet_id = reply_result.get("tweet_id")
                    logger.info(f"Posted clean sweep celebration to Twitter")
                except Exception as e:
                    logger.warning(f"Failed to post clean sweep to Twitter: {e}")

        # Post to Facebook
        # Facebook live updates disabled until commenting permission is resolved
        fb_post_id = None

        update = LiveUpdate(
            meeting_id=meeting.id,
            race_number=race_number,
            update_type="clean_sweep",
            content=tweet_text,
            tweet_id=reply_tweet_id,
            parent_tweet_id=thread_parent or (early_mail.twitter_id if early_mail else None),
            facebook_comment_id=fb_post_id,
            parent_facebook_id=early_mail.facebook_id if early_mail else None,
            pnl=total_pnl,
        )
        db.add(update)
        await db.flush()

    async def _check_pace_bias(self, db: AsyncSession, meeting, statuses: dict, just_completed: int):
        """Analyze pace bias and post reply if significant pattern detected."""
        from punty.results.pace_analysis import analyze_pace_bias, compose_pace_tweet, find_bias_fits
        from punty.models.content import Content
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery
        from punty.delivery.facebook import FacebookDelivery

        meeting_id = meeting.id

        # Limit to 3 pace updates per meeting
        posted_count = self.pace_updates_posted.get(meeting_id, 0)
        if posted_count >= 3:
            return

        completed_races = sorted(self.processed_races.get(meeting_id, set()))
        total_races = len(statuses)

        # Start analysis after race 3 for 7-race meets, race 4 for 8+
        min_races = 3 if total_races <= 7 else 4
        if len(completed_races) < min_races:
            return
        if just_completed >= total_races:
            return

        races_remaining = total_races - just_completed

        bias_result = await analyze_pace_bias(db, meeting_id, completed_races)
        if not bias_result:
            return
        # First check always posts (even "maps are tracking" when no bias)
        # Subsequent checks only post if a bias has developed
        if not bias_result.bias_detected and posted_count > 0:
            return

        # Find the meeting's early mail
        em_result = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting_id,
                Content.content_type == "early_mail",
            ).order_by(Content.created_at.desc())
        )
        early_mail = em_result.scalars().first()
        if not early_mail:
            return
        if not early_mail.twitter_id and not early_mail.facebook_id:
            return

        # Find horses in remaining races that fit the bias pattern
        horse_suggestions = []
        if bias_result.bias_detected:
            try:
                horse_suggestions = await find_bias_fits(
                    db, meeting_id, completed_races, len(statuses), bias_result.bias_type
                )
            except Exception as e:
                logger.debug(f"Failed to find bias fits: {e}")

        tweet_text = compose_pace_tweet(bias_result, meeting.venue, races_remaining, horse_suggestions)
        if not tweet_text:
            return

        # Post to Twitter (chain off last tweet in thread)
        reply_tweet_id = None
        thread_parent = None
        if early_mail.twitter_id:
            thread_parent = await self._get_thread_parent(db, meeting_id, early_mail.twitter_id)
            twitter = TwitterDelivery(db)
            if await twitter.is_configured():
                try:
                    reply_result = await twitter.post_reply(thread_parent, tweet_text)
                    reply_tweet_id = reply_result.get("tweet_id")
                    self.pace_updates_posted[meeting_id] = posted_count + 1
                    logger.info(f"Posted pace analysis for {meeting.venue} (update {posted_count + 1}, bias: {bias_result.bias_type})")
                except Exception as e:
                    logger.warning(f"Failed to post pace analysis reply: {e}")

        # Facebook live updates disabled until commenting permission is resolved
        fb_post_id = None

        # Save to DB regardless of post success
        update = LiveUpdate(
            meeting_id=meeting_id,
            race_number=just_completed,
            update_type="pace_analysis",
            content=tweet_text,
            tweet_id=reply_tweet_id,
            parent_tweet_id=thread_parent or early_mail.twitter_id,
            facebook_comment_id=fb_post_id,
            parent_facebook_id=early_mail.facebook_id,
        )
        db.add(update)
        await db.flush()
        self.pace_updates_posted[meeting_id] = posted_count + 1

    async def _scrape_sectional_times(self, db: AsyncSession, meeting, race_num: int):
        """Scrape and store post-race sectional times for a race.

        Sectional times are typically available 10-15 minutes after the race finishes.
        This captures position and time at each checkpoint to compare with speed map predictions.
        """
        import json as _json
        from punty.models.meeting import Race
        from punty.scrapers.racing_com import RacingComScraper

        race_id = f"{meeting.id}-r{race_num}"
        race = await db.get(Race, race_id)
        if not race:
            return

        # Skip if we already have sectional data
        if race.sectional_times:
            return

        scraper = RacingComScraper()
        try:
            # Pass meet_code for CSV fallback if available
            meet_code = getattr(meeting, 'meet_code', None)
            sectional_data = await scraper.scrape_sectional_times(
                meeting.venue, meeting.date, race_num, meet_code=meet_code
            )
            # Store meet_code if we captured it
            if sectional_data and sectional_data.get("meet_code") and not meeting.meet_code:
                meeting.meet_code = sectional_data["meet_code"]
        finally:
            await scraper.close()

        if not sectional_data or not sectional_data.get("horses"):
            # Sectionals not available yet - will be picked up on future poll
            race.has_sectionals = False
            await db.flush()
            return

        # Store the sectional data
        race.sectional_times = _json.dumps(sectional_data)
        race.has_sectionals = True
        await db.flush()

        logger.info(f"Sectional times captured for {meeting.venue} R{race_num} ({len(sectional_data['horses'])} horses)")

    async def _backfill_sectionals(self, db: AsyncSession, meeting, races):
        """Try to backfill sectional times for races that are missing them."""
        import json as _json
        from punty.models.meeting import Race

        for race in races:
            # Skip if already has sectionals or explicitly marked as not having them
            if race.sectional_times:
                continue
            if race.has_sectionals is False:
                continue
            # Only try for races with results
            if race.results_status not in ("Paying", "Closed", "Final"):
                continue

            try:
                await self._scrape_sectional_times(db, meeting, race.race_number)
            except Exception as e:
                logger.debug(f"Sectional backfill failed for {meeting.venue} R{race.race_number}: {e}")

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

    # ── Pre-race change detection ──────────────────────────────────────────

    async def _check_pre_race_changes(self, db: AsyncSession, meeting, races, statuses: dict):
        """Check for scratchings, track condition changes, and jockey/gear swaps on upcoming races."""
        from punty.results.change_detection import (
            take_snapshot, detect_scratching_changes,
            detect_track_condition_change, detect_jockey_gear_changes,
        )

        meeting_id = meeting.id

        # Check if live alerts are enabled
        from punty.models.settings import AppSettings
        setting_result = await db.execute(
            select(AppSettings).where(AppSettings.key == "enable_live_alerts")
        )
        setting = setting_result.scalar_one_or_none()
        if setting and setting.value != "true":
            return

        # Rate limit: max 1 check per 3 min per meeting
        now = melb_now()
        last = self.last_change_check.get(meeting_id)
        if last and (now - last).total_seconds() < 180:
            return

        # Find upcoming race numbers (status=Open, start_time > now + 5min)
        upcoming = []
        for race in races:
            status = statuses.get(race.race_number, "Open")
            if status != "Open":
                continue
            if race.start_time:
                from datetime import time as _time
                st = race.start_time if isinstance(race.start_time, _time) else race.start_time.time()
                race_start = datetime.combine(meeting.date, st, tzinfo=MELB_TZ)
                if race_start < now + timedelta(minutes=5):
                    continue
            upcoming.append(race.race_number)

        if not upcoming:
            logger.debug(f"No upcoming races for {meeting.venue} change detection")
            return

        logger.info(f"Checking pre-race changes for {meeting.venue}: races {upcoming}")

        # Snapshot current DB state before refresh
        snapshot = await take_snapshot(db, meeting_id, upcoming)

        # Refresh odds + scratchings via TAB (httpx, lightweight)
        try:
            from punty.scrapers.orchestrator import refresh_odds
            await refresh_odds(meeting_id, db)
        except Exception as e:
            logger.warning(f"Odds refresh failed for {meeting.venue}: {e}")

        self.last_change_check[meeting_id] = now

        # Detect scratching changes (TAB refresh already applied scratchings)
        alerts = await detect_scratching_changes(db, meeting_id, upcoming, snapshot)

        # Jockey/gear check (less frequent — every 10 min, Playwright)
        # Also updates track condition from racing.com
        jockey_last = self.last_jockey_check.get(meeting_id)
        if not jockey_last or (now - jockey_last).total_seconds() >= 600:
            try:
                from punty.scrapers.playwright_base import is_scrape_in_progress
                in_progress, _ = is_scrape_in_progress()
                if not in_progress:
                    from punty.scrapers.racing_com import RacingComScraper
                    field_scraper = RacingComScraper()
                    try:
                        field_data = await field_scraper.check_race_fields(
                            meeting.venue, meeting.date, upcoming
                        )
                        # Apply jockey/gear/track updates to DB
                        await self._apply_field_changes(db, meeting_id, field_data)
                        # Detect jockey/gear changes
                        jg_alerts = await detect_jockey_gear_changes(
                            db, meeting_id, upcoming, snapshot
                        )
                        alerts.extend(jg_alerts)
                    finally:
                        await field_scraper.close()
                    self.last_jockey_check[meeting_id] = now
            except Exception as e:
                logger.warning(f"Jockey/gear check failed for {meeting.venue}: {e}", exc_info=True)

        # Detect track condition change (AFTER both TAB + racing.com updates)
        track_alert = await detect_track_condition_change(db, meeting_id, snapshot)
        if track_alert:
            alerts.append(track_alert)

        if alerts:
            logger.info(f"Change detection for {meeting.venue}: {len(alerts)} alerts — {[a.change_type for a in alerts]}")
        else:
            logger.info(f"Change detection for {meeting.venue}: no changes detected")

        # Dedup and post
        if meeting_id not in self.alerted_changes:
            self.alerted_changes[meeting_id] = set()

        for alert in alerts:
            key = alert.dedup_key
            if key in self.alerted_changes[meeting_id]:
                continue
            self.alerted_changes[meeting_id].add(key)

            try:
                await self._post_change_alert(db, meeting, alert)
            except Exception as e:
                logger.warning(f"Failed to post {alert.change_type} alert: {e}")

    async def _check_weather_changes(self, db: AsyncSession, meeting):
        """Check for significant weather changes via WillyWeather (every 30 min)."""
        meeting_id = meeting.id
        now = melb_now()

        # Rate limit: 30 min per meeting
        last = self.last_weather_check.get(meeting_id)
        if last and (now - last).total_seconds() < 1800:
            return

        from punty.scrapers.willyweather import WillyWeatherScraper, analyse_wind_impact

        ww = await WillyWeatherScraper.from_settings(db)
        if not ww:
            return

        try:
            weather = await ww.get_weather(meeting.venue)
        finally:
            await ww.close()

        self.last_weather_check[meeting_id] = now

        if not weather:
            return

        changes = []

        # Check wind shift
        new_wind = weather.get("wind_speed")
        old_wind = meeting.weather_wind_speed
        new_dir = weather.get("wind_direction")
        old_dir = meeting.weather_wind_dir
        if new_wind is not None and old_wind is not None:
            wind_diff = abs(new_wind - old_wind)
            if wind_diff >= 10:
                changes.append(f"Wind shifted to {new_wind}km/h {new_dir or ''} (was {old_wind}km/h {old_dir or ''})")
            elif new_dir and old_dir and new_dir != old_dir and new_wind >= 15:
                changes.append(f"Wind direction changed: {old_dir} → {new_dir} at {new_wind}km/h")

        # Check temp change
        new_temp = weather.get("temp")
        old_temp = meeting.weather_temp
        if new_temp is not None and old_temp is not None:
            temp_diff = abs(new_temp - old_temp)
            if temp_diff >= 3:
                changes.append(f"Temperature {'up' if new_temp > old_temp else 'down'} to {new_temp}°C (was {old_temp}°C)")

        # Check humidity change (significant for track conditions)
        new_humidity = weather.get("humidity")
        old_humidity = getattr(meeting, "weather_humidity", None)
        if new_humidity is not None and old_humidity is not None:
            humidity_diff = abs(new_humidity - old_humidity)
            if humidity_diff >= 15:
                changes.append(f"Humidity {'up' if new_humidity > old_humidity else 'down'} to {new_humidity}% (was {old_humidity}%)")

        # Check rain starting
        obs = weather.get("observation", {})
        if obs:
            rain_since_9am = obs.get("rain_since_9am")
            if rain_since_9am and rain_since_9am > 0:
                old_rainfall = meeting.rainfall
                was_dry = not old_rainfall or "0%" in str(old_rainfall) or old_rainfall == 0
                if was_dry:
                    changes.append(f"Rain recorded: {rain_since_9am}mm since 9am")

        if not changes:
            # Still update weather fields silently
            self._apply_weather_to_meeting(meeting, weather)
            await db.commit()
            return

        # Build alert message
        wind_analysis = analyse_wind_impact(
            meeting.venue,
            new_wind or 0,
            new_dir or "",
        )
        parts = [f"Weather update at {meeting.venue}:"]
        parts.extend(changes)
        if wind_analysis and wind_analysis["strength"] != "negligible":
            parts.append(wind_analysis["description"])
        message = "\n".join(parts)

        # Update meeting fields
        self._apply_weather_to_meeting(meeting, weather)
        await db.commit()

        # Post via change alert system
        from punty.results.change_detection import ChangeAlert
        alert = ChangeAlert(
            change_type="weather",
            meeting_id=meeting_id,
            message=message,
        )
        dedup_key = alert.dedup_key
        if meeting_id not in self.alerted_changes:
            self.alerted_changes[meeting_id] = set()
        if dedup_key not in self.alerted_changes[meeting_id]:
            self.alerted_changes[meeting_id].add(dedup_key)
            await self._post_change_alert(db, meeting, alert)
            logger.info(f"Weather alert posted for {meeting.venue}: {'; '.join(changes)}")

    @staticmethod
    def _apply_weather_to_meeting(meeting, weather: dict):
        """Update meeting weather fields from WillyWeather data."""
        if weather.get("temp") is not None:
            meeting.weather_temp = weather["temp"]
        if weather.get("wind_speed") is not None:
            meeting.weather_wind_speed = weather["wind_speed"]
        if weather.get("wind_direction"):
            meeting.weather_wind_dir = weather["wind_direction"]
        if weather.get("condition"):
            meeting.weather_condition = weather["condition"]
        if weather.get("humidity") is not None:
            meeting.weather_humidity = weather["humidity"]
        if weather.get("rainfall_chance") is not None:
            meeting.rainfall = f"{weather['rainfall_chance']}% chance, {weather['rainfall_amount']}mm"

    async def _post_change_alert(self, db: AsyncSession, meeting, alert):
        """Post a change detection alert to Twitter and Facebook."""
        from punty.models.content import Content
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery
        from punty.delivery.facebook import FacebookDelivery

        # Find the meeting's early mail for reply threading
        em_result = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting.id,
                Content.content_type == "early_mail",
            ).order_by(Content.created_at.desc())
        )
        early_mail = em_result.scalars().first()
        if not early_mail:
            logger.info(f"No early mail for {meeting.venue} — skipping change alert")
            return
        if not early_mail.twitter_id and not early_mail.facebook_id:
            logger.info(f"Early mail not posted for {meeting.venue} — skipping change alert")
            return

        tweet_text = alert.message
        logger.info(f"Posting {alert.change_type} alert for {meeting.venue}: {tweet_text[:80]}...")

        # Post to Twitter as reply (chain off last tweet in thread)
        reply_tweet_id = None
        thread_parent = None
        if early_mail.twitter_id:
            thread_parent = await self._get_thread_parent(db, meeting.id, early_mail.twitter_id)
            twitter = TwitterDelivery(db)
            if await twitter.is_configured():
                try:
                    reply_result = await twitter.post_reply(thread_parent, tweet_text)
                    reply_tweet_id = reply_result.get("tweet_id")
                except Exception as e:
                    logger.warning(f"Failed to post {alert.change_type} Twitter reply: {e}")

        # Post to Facebook as standalone update
        fb_post_id = None
        # Facebook live updates disabled until commenting permission is resolved

        # Save to LiveUpdate
        update_type_map = {
            "scratching": "scratching_alert",
            "track_condition": "track_alert",
            "jockey_change": "jockey_alert",
            "gear_change": "gear_alert",
            "weather": "weather_alert",
        }
        update = LiveUpdate(
            meeting_id=meeting.id,
            race_number=alert.race_number,
            update_type=update_type_map.get(alert.change_type, "change_alert"),
            content=tweet_text,
            tweet_id=reply_tweet_id,
            parent_tweet_id=thread_parent or early_mail.twitter_id,
            facebook_comment_id=fb_post_id,
            parent_facebook_id=early_mail.facebook_id,
            horse_name=alert.horse_name,
        )
        db.add(update)
        await db.flush()

    async def _apply_field_changes(self, db: AsyncSession, meeting_id: str, field_data: dict):
        """Persist jockey/gear updates from racing.com field check."""
        from punty.models.meeting import Runner

        for race_num, runners in field_data.get("races", {}).items():
            race_id = f"{meeting_id}-r{race_num}"
            for r in runners:
                horse_name = r.get("horse_name")
                if not horse_name:
                    continue
                result = await db.execute(
                    select(Runner).where(
                        Runner.race_id == race_id,
                        Runner.horse_name == horse_name,
                    ).limit(1)
                )
                runner = result.scalar_one_or_none()
                if not runner:
                    continue

                if r.get("scratched") and not runner.scratched:
                    runner.scratched = True
                if r.get("jockey") and r["jockey"] != runner.jockey:
                    runner.jockey = r["jockey"]
                if r.get("gear") and r["gear"] != runner.gear:
                    runner.gear = r["gear"]
                if r.get("gear_changes") and r["gear_changes"] != runner.gear_changes:
                    runner.gear_changes = r["gear_changes"]

        # Track condition from racing.com — only update if more specific
        # (prevents "Good 4" being overwritten by bare "Good", which causes
        # flip-flopping between sources and noisy change alerts)
        tc = field_data.get("meeting", {}).get("track_condition")
        if tc:
            from punty.models.meeting import Meeting
            from punty.scrapers.orchestrator import _is_more_specific
            meeting = await db.get(Meeting, meeting_id)
            if meeting and _is_more_specific(tc, meeting.track_condition or ""):
                meeting.track_condition = tc

        await db.flush()

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

            # Clear processed state so manual check re-processes all races
            # This ensures settlement runs for any unsettled picks
            if meeting_id in self.processed_races:
                self.processed_races[meeting_id].clear()

            await self._check_meeting(db, meeting, races)
            await db.commit()
