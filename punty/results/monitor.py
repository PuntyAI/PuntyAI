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


async def _backfill_form_history(db: AsyncSession, meeting_id: str, race_number: int) -> None:
    """Store race result data as form history entries for future lookups.

    After a race settles, save each runner's result (venue, distance, class,
    position, margin, weight, jockey, barrier, SP) as a form_history JSON entry.
    Next time the horse runs and form_history is scraped, these entries can be
    merged to fill gaps in the scraper's coverage.
    """
    import json
    from punty.models.meeting import Meeting, Race, Runner

    race_id = f"{meeting_id}-r{race_number}"

    # Get meeting + race + runners
    meeting_result = await db.execute(
        select(Meeting).where(Meeting.id == meeting_id)
    )
    meeting = meeting_result.scalar_one_or_none()
    if not meeting:
        return

    race_result = await db.execute(
        select(Race).where(Race.id == race_id)
    )
    race = race_result.scalar_one_or_none()
    if not race:
        return

    runners_result = await db.execute(
        select(Runner).where(
            Runner.race_id == race_id,
            Runner.scratched == False,
            Runner.finish_position.isnot(None),
            Runner.finish_position > 0,
        )
    )
    runners = runners_result.scalars().all()
    if not runners:
        return

    backfilled = 0
    for runner in runners:
        # Build form entry from result data
        entry = {
            "venue": meeting.venue,
            "date": str(meeting.date),
            "distance": race.distance,
            "class": race.class_,
            "track_condition": meeting.track_condition,
            "position": runner.finish_position,
            "margin": runner.result_margin,
            "weight": runner.weight,
            "jockey": runner.jockey,
            "barrier": runner.barrier,
            "sp": runner.current_odds,
            "prize": race.prize_money,
            "field_size": len(runners),
            "backfilled": True,
        }

        # Merge into existing form_history if present
        try:
            existing = json.loads(runner.form_history) if runner.form_history else []
        except (json.JSONDecodeError, TypeError):
            existing = []

        # Check if this race is already in form history (avoid duplicates)
        already_has = any(
            e.get("venue") == meeting.venue
            and str(e.get("date", "")) == str(meeting.date)
            and e.get("distance") == race.distance
            for e in existing
        )
        if already_has:
            continue

        # Prepend (most recent first)
        updated = [entry] + existing
        runner.form_history = json.dumps(updated)
        backfilled += 1

    if backfilled > 0:
        await db.commit()
        logger.info(f"Backfilled form history for {backfilled} runners in {meeting.venue} R{race_number}")


class ResultsMonitor:
    """Polls racing.com for completed races and triggers results generation."""

    def __init__(self, app):
        self.app = app
        self.running = False
        self.processed_races: dict[str, set[int]] = {}  # {meeting_id: {race_nums}}
        self.wrapups_generated: set[str] = set()
        self.pace_updates_posted: dict[str, int] = {}  # {meeting_id: count}
        self.weather_alerts_posted: dict[str, int] = {}  # {meeting_id: count}
        self.alerted_changes: dict[str, set[str]] = {}  # {meeting_id: set of dedup keys}
        self.last_change_check: dict[str, datetime] = {}  # rate limit per meeting
        self.last_jockey_check: dict[str, datetime] = {}  # Playwright rate limit
        self.last_weather_check: dict[str, datetime] = {}  # WillyWeather rate limit
        self.last_track_change: dict[str, datetime] = {}  # cooldown after track condition change
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
                    Race.results_status.notin_(["Paying", "Closed", "Final", "Abandoned"])
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
                self.weather_alerts_posted.clear()
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

                # Check for missed big win celebrations (selections + exotics)
                # Always run — _check_big_wins has per-pick deduplication
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
            status_data = await scraper.check_race_statuses(meeting.venue, meeting.date)
            statuses = status_data["statuses"]
            scraped_tc = status_data.get("track_condition")
        except Exception as e:
            logger.error(f"Failed to check statuses for {meeting.venue}: {e}")
            return
        finally:
            await scraper.close()

        # ── Abandonment detection ───────────────────────────────────────
        # If ALL races are "Abandoned", post alert, void picks, deselect meeting.
        if statuses and all(s == "Abandoned" for s in statuses.values()):
            dedup = self.alerted_changes.get(meeting_id, set())
            if "abandonment" not in dedup:
                logger.warning(f"Meeting ABANDONED: {meeting.venue} ({meeting_id})")
                from punty.results.change_detection import compose_abandonment_alert, ChangeAlert
                alert = ChangeAlert(
                    change_type="abandonment",
                    meeting_id=meeting_id,
                    message=compose_abandonment_alert(meeting.venue),
                )
                try:
                    await self._post_change_alert(db, meeting, alert)
                except Exception as e:
                    logger.warning(f"Failed to post abandonment alert for {meeting.venue}: {e}")
                if meeting_id not in self.alerted_changes:
                    self.alerted_changes[meeting_id] = set()
                self.alerted_changes[meeting_id].add("abandonment")

                # Void all unsettled picks (abandoned = no win, no loss)
                from punty.results.picks import void_picks_for_meeting
                try:
                    await void_picks_for_meeting(db, meeting_id)
                except Exception as e:
                    logger.warning(f"Failed to void picks for {meeting.venue}: {e}")

                # Delete orphan track alert tweets (Good→Heavy looks stupid if abandoned)
                try:
                    from punty.models.live_update import LiveUpdate
                    from punty.delivery.twitter import TwitterDelivery
                    track_updates = await db.execute(
                        select(LiveUpdate).where(
                            LiveUpdate.meeting_id == meeting_id,
                            LiveUpdate.update_type == "track_alert",
                        )
                    )
                    twitter = TwitterDelivery(db)
                    if await twitter.is_configured():
                        for update in track_updates.scalars():
                            try:
                                await twitter.delete_tweet(update.tweet_id)
                                logger.info(f"Deleted orphan track tweet {update.tweet_id} for abandoned {meeting.venue}")
                            except Exception as e:
                                logger.debug(f"Could not delete track tweet {update.tweet_id}: {e}")
                except Exception as e:
                    logger.debug(f"Track tweet cleanup failed for {meeting.venue}: {e}")

                # Deselect meeting to stop further monitoring
                meeting.selected = False
                await db.flush()
                await db.commit()

            return  # Skip all further processing for abandoned meetings

        # Suppress track condition alerts when >50% races abandoned (meeting winding down)
        abandoned_count = sum(1 for s in statuses.values() if s == "Abandoned")
        if statuses and abandoned_count > len(statuses) / 2:
            logger.info(
                f"Suppressing track alerts for {meeting.venue}: "
                f"{abandoned_count}/{len(statuses)} races abandoned"
            )
            scraped_tc = None

        # Update track condition if changed (piggybacks on status poll — no extra cost)
        # Only update if the base category actually changed (e.g. Soft→Good)
        # or if the new value is more specific (e.g. Good→Good 4).
        # Prevents flip-flop when racing.com returns "Good" but PF has "Good 4".
        if scraped_tc and scraped_tc != meeting.track_condition:
            from punty.results.change_detection import _base_condition
            from punty.scrapers.orchestrator import _should_update_condition
            old_tc = meeting.track_condition
            old_base = _base_condition(old_tc) if old_tc else ""
            new_base = _base_condition(scraped_tc)

            # Cooldown: if we changed the track condition recently (within 30 min),
            # reject base category reversals to prevent Good→Soft→Good bouncing.
            # Only specificity upgrades (same base) bypass cooldown.
            now = melb_now()
            last_change = self.last_track_change.get(meeting_id)
            cooldown_active = last_change and (now - last_change).total_seconds() < 1800

            if old_base != new_base and cooldown_active:
                logger.info(
                    f"Track condition cooldown active for {meeting.venue}: "
                    f"ignoring {old_tc} → {scraped_tc} (changed {int((now - last_change).total_seconds())}s ago)"
                )
                scraped_tc = None
            elif _should_update_condition(scraped_tc, old_tc or ""):
                # Accept: base category changed (real upgrade/downgrade) OR new is more specific
                meeting.track_condition = scraped_tc
                if old_base != new_base:
                    self.last_track_change[meeting_id] = now
                await db.flush()
                logger.warning(f"Track condition changed for {meeting.venue}: {old_tc} → {scraped_tc}")
            else:
                logger.debug(f"Ignoring less-specific condition for {meeting.venue}: {old_tc} → {scraped_tc}")
                scraped_tc = None  # Prevent alert firing below

            # Fire track upgrade/downgrade alert (only for genuine condition changes,
            # not just rating fluctuations like Good 4 → Good)
            if old_tc and scraped_tc:
                from punty.results.change_detection import compose_track_alert, ChangeAlert
                if old_base != new_base:
                    # Dedup: don't re-alert same transition
                    if meeting_id not in self.alerted_changes:
                        self.alerted_changes[meeting_id] = set()
                    pair = sorted([old_tc or "", scraped_tc or ""])
                    dedup_key = f"track:{pair[0]}->{pair[1]}"
                    if dedup_key not in self.alerted_changes[meeting_id]:
                        self.alerted_changes[meeting_id].add(dedup_key)
                        alert = ChangeAlert(
                            change_type="track_condition",
                            meeting_id=meeting_id,
                            old_value=old_tc,
                            new_value=scraped_tc,
                            message=compose_track_alert(meeting.venue, old_tc, scraped_tc),
                        )
                        try:
                            await self._post_change_alert(db, meeting, alert)
                            logger.info(f"Track alert posted for {meeting.venue}: {old_tc} → {scraped_tc}")
                        except Exception as e:
                            logger.warning(f"Failed to post track alert for {meeting.venue}: {e}")

        # Void picks for individually abandoned races (partial abandonment)
        for race_num, status in statuses.items():
            if status == "Abandoned" and race_num not in self.processed_races[meeting_id]:
                logger.info(f"Race abandoned: {meeting.venue} R{race_num}")
                from punty.results.picks import void_picks_for_race
                try:
                    await void_picks_for_race(db, meeting_id, race_num)
                except Exception as e:
                    logger.warning(f"Failed to void picks for {meeting.venue} R{race_num}: {e}")
                self.processed_races[meeting_id].add(race_num)

        # Process results FIRST — this is time-critical for live settlement
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

                    # Backfill form history — store result data for future lookups
                    try:
                        await _backfill_form_history(db, meeting_id, race_num)
                    except Exception as e:
                        logger.debug(f"Form history backfill failed for {meeting.venue} R{race_num}: {e}")

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

        # Pre-race change detection AFTER results (scratchings, track condition, jockey/gear)
        try:
            await self._check_pre_race_changes(db, meeting, races, statuses)
        except Exception as e:
            logger.warning(f"Pre-race change check failed for {meeting.venue}: {e}", exc_info=True)

        # Weather refresh (every 30 min per meeting)
        try:
            await self._check_weather_changes(db, meeting)
        except Exception as e:
            logger.warning(f"Weather check failed for {meeting.venue}: {e}", exc_info=True)

        # Backfill exotic dividends from TabTouch for races missing them
        await self._backfill_tabtouch_exotics(db, meeting, statuses)

        # Backfill place dividends for races where win_dividend exists but place_dividend is missing
        await self._backfill_place_dividends(db, meeting, statuses)

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
            if existing_wrapup.scalars().first():
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

    async def _get_posted_early_mail(self, db: AsyncSession, meeting_id: str):
        """Find the early mail with a social media post for threading live updates.

        Prefers the newest early mail that has been posted (has twitter_id or facebook_id).
        When a new early mail supersedes the old one but hasn't been sent yet,
        falls back to the superseded one so live updates still thread correctly
        instead of being silently dropped.
        """
        from punty.models.content import Content
        result = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting_id,
                Content.content_type == "early_mail",
            ).order_by(Content.created_at.desc())
        )
        all_early_mails = result.scalars().all()
        for em in all_early_mails:
            if em.twitter_id or em.facebook_id:
                return em
        # No posted early mail at all
        return all_early_mails[0] if all_early_mails else None

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
        """Check for big wins and post celebration replies.

        Triggers on:
        - Selections: P&L >= 4x stake (i.e. 5x return)
        - Exotics: Any hit (trifecta/exacta/quinella/first4)
        - Any bet type: P&L >= $100 absolute
        """
        from punty.models.pick import Pick
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery
        from punty.results.celebrations import compose_celebration_tweet, compose_exotic_celebration_tweet

        # Check selections AND exotics with positive P&L
        result = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting_id,
                Pick.race_number == race_number,
                Pick.pick_type.in_(["selection", "exotic"]),
                Pick.settled == True,
                Pick.hit == True,
                Pick.pnl > 0,
            )
        )
        for pick in result.scalars().all():
            stake = pick.bet_stake or pick.exotic_stake or 1.0
            pnl = pick.pnl or 0

            # Celebration triggers: 4x return for selections, OR any exotic hit, OR $100+ profit
            is_big_selection = pick.pick_type == "selection" and pnl >= 4 * stake
            is_exotic_hit = pick.pick_type == "exotic"
            is_hundred_plus = pnl >= 100

            if not (is_big_selection or is_exotic_hit or is_hundred_plus):
                continue

            # Check if we already posted a celebration for this pick
            existing = await db.execute(
                select(LiveUpdate).where(
                    LiveUpdate.meeting_id == meeting_id,
                    LiveUpdate.race_number == race_number,
                    LiveUpdate.update_type == "celebration",
                    LiveUpdate.horse_name == (pick.horse_name or pick.exotic_type or "Winner"),
                )
            )
            if existing.scalar_one_or_none():
                continue

            # Find the meeting's early mail (falls back to superseded if new one not yet posted)
            early_mail = await self._get_posted_early_mail(db, meeting_id)
            if not early_mail:
                return

            collect = stake + pnl

            # Compose appropriate celebration tweet
            if pick.pick_type == "exotic":
                # Get venue for exotic celebrations
                from punty.models.meeting import Meeting
                meeting_result = await db.execute(
                    select(Meeting.venue).where(Meeting.id == meeting_id)
                )
                venue = meeting_result.scalar_one_or_none() or ""
                tweet_text = compose_exotic_celebration_tweet(
                    exotic_type=pick.exotic_type or "Exotic",
                    pnl=pnl,
                    stake=stake,
                    collect=collect,
                    venue=venue,
                    race_number=race_number,
                )
                log_name = pick.exotic_type or "Exotic"
            else:
                tweet_text = compose_celebration_tweet(
                    horse_name=pick.horse_name or "Winner",
                    odds=pick.odds_at_tip or 0,
                    stake=stake,
                    collect=collect,
                    bet_type=pick.bet_type or "Win",
                )
                log_name = pick.horse_name or "Winner"

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
                        logger.info(f"Posted celebration reply for {log_name} ({pnl:+.2f})")
                    except Exception as e:
                        logger.warning(f"Failed to post celebration reply: {e}")

            # Facebook live updates disabled (standalone posts not useful without commenting)
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
                parent_facebook_id=early_mail.facebook_id if early_mail else None,
                horse_name=pick.horse_name or pick.exotic_type,
                odds=pick.odds_at_tip,
                pnl=pnl,
            )
            db.add(update)
            await db.flush()

    async def _check_clean_sweep(self, db: AsyncSession, meeting, race_number: int):
        """Check if ALL selections in a race won — post a big celebration."""
        from punty.models.pick import Pick
        from punty.models.content import Content
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery
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

        # Find early mail for threading (falls back to superseded if new one not yet posted)
        early_mail = await self._get_posted_early_mail(db, meeting.id)
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

        # Facebook live updates disabled (standalone posts not useful without commenting)
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

        # Find the meeting's early mail (falls back to superseded if new one not yet posted)
        early_mail = await self._get_posted_early_mail(db, meeting_id)
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

        # Post to Facebook as standalone post
        # (commenting requires pages_manage_engagement — App Review pending)
        # Facebook live updates disabled (standalone posts not useful without commenting)
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

    async def _backfill_place_dividends(self, db, meeting, statuses):
        """Re-scrape races where placed runners are missing place dividends.

        Racing.com sometimes doesn't include place dividends in the initial
        result scrape (they may finalize slightly after win dividends). This
        backfill re-scrapes those races and updates the missing dividends.
        """
        from punty.models.meeting import Runner as RunnerModel, Race

        # Only check paying/closed races
        paying_races = [rn for rn, s in statuses.items() if s in ("Paying", "Closed")]
        if not paying_races:
            return

        # Find races with placed runners missing place_dividend
        missing_races = []
        for rn in paying_races:
            race_id = f"{meeting.id}-r{rn}"
            # Count runners that have a finish position in places but no place_dividend
            result = await db.execute(
                select(RunnerModel).where(
                    RunnerModel.race_id == race_id,
                    RunnerModel.finish_position.isnot(None),
                    RunnerModel.finish_position <= 3,
                    RunnerModel.finish_position > 0,
                )
            )
            placed_runners = result.scalars().all()
            # Backfill if any placed runner is missing place_dividend or has
            # a suspiciously high value (>$50 is very rare for place dividends)
            if any(
                not r.place_dividend or r.place_dividend > 50
                for r in placed_runners
            ):
                missing_races.append(rn)

        if not missing_races:
            return

        # Rate limit: only attempt backfill once per meeting per monitor cycle
        backfill_key = f"place_div_{meeting.id}"
        if hasattr(self, '_place_div_attempts'):
            attempts = self._place_div_attempts.get(backfill_key, 0)
            if attempts >= 5:  # Stop after 5 attempts
                return
        else:
            self._place_div_attempts = {}

        self._place_div_attempts[backfill_key] = self._place_div_attempts.get(backfill_key, 0) + 1

        logger.info(
            f"Backfilling place dividends for {meeting.venue}: "
            f"{len(missing_races)} race(s) missing place_dividend "
            f"(attempt {self._place_div_attempts[backfill_key]})"
        )

        try:
            scraper = RacingComScraper()
            try:
                updated = 0
                for rn in missing_races:
                    try:
                        results_data = await scraper.scrape_race_result(
                            meeting.venue, meeting.date, rn
                        )
                    except Exception as e:
                        logger.debug(f"Place dividend backfill scrape failed for {meeting.venue} R{rn}: {e}")
                        continue

                    race_id = f"{meeting.id}-r{rn}"
                    for result in results_data.get("results", []):
                        place_div = result.get("place_dividend")
                        if place_div is None or place_div <= 0:
                            continue

                        horse_name = result.get("horse_name", "")
                        saddlecloth = result.get("saddlecloth")

                        # Find runner by name first, then saddlecloth
                        runner = None
                        if horse_name:
                            r = await db.execute(
                                select(RunnerModel).where(
                                    RunnerModel.race_id == race_id,
                                    RunnerModel.horse_name == horse_name,
                                ).limit(1)
                            )
                            runner = r.scalar_one_or_none()
                        if not runner and saddlecloth is not None:
                            r = await db.execute(
                                select(RunnerModel).where(
                                    RunnerModel.race_id == race_id,
                                    RunnerModel.saddlecloth == int(saddlecloth),
                                ).limit(1)
                            )
                            runner = r.scalar_one_or_none()

                        if runner:
                            # Always overwrite — existing value may be a garbage
                            # estimate from formula approximation
                            if runner.place_dividend != place_div:
                                runner.place_dividend = place_div
                                updated += 1

                if updated:
                    await db.flush()
                    logger.info(f"Backfilled {updated} place dividend(s) for {meeting.venue}")
            finally:
                await scraper.close()
        except Exception as e:
            logger.debug(f"Place dividend backfill failed for {meeting.venue}: {e}")

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
        # Skip if cooldown is active (prevents Good→Soft→Good bouncing)
        last_tc_change = self.last_track_change.get(meeting_id)
        tc_cooldown = last_tc_change and (now - last_tc_change).total_seconds() < 1800
        if not tc_cooldown:
            track_alert = await detect_track_condition_change(db, meeting_id, snapshot)
            if track_alert:
                alerts.append(track_alert)
                self.last_track_change[meeting_id] = now

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

        # Alert for track-impacting weather: rain, significant wind, storms
        obs = weather.get("observation", {})
        if obs:
            # Rain alerts
            rain_since_9am = obs.get("rain_since_9am")
            if rain_since_9am and rain_since_9am > 0:
                old_rainfall = meeting.rainfall
                was_dry = not old_rainfall or "0%" in str(old_rainfall) or old_rainfall == 0
                if was_dry:
                    changes.append(f"Rain recorded: {rain_since_9am}mm since 9am")
                elif rain_since_9am >= 5:
                    changes.append(f"Heavy rain: {rain_since_9am}mm since 9am")

            # Significant wind change (gusts ≥40 km/h or sustained ≥30 km/h)
            gust = obs.get("wind_gust") or weather.get("wind_gust")
            wind_speed = weather.get("wind_speed") or obs.get("wind_speed")
            if gust and gust >= 40:
                changes.append(f"Strong wind gusts: {gust} km/h")
            elif wind_speed and wind_speed >= 30:
                changes.append(f"Strong winds: {wind_speed} km/h sustained")

            # Storm detection (heavy rain + strong wind = storm conditions)
            if rain_since_9am and rain_since_9am >= 3 and gust and gust >= 35:
                changes.append("Storm conditions detected")

        # Cap at 2 weather alerts per meeting
        if meeting_id not in self.weather_alerts_posted:
            self.weather_alerts_posted[meeting_id] = 0
        if self.weather_alerts_posted[meeting_id] >= 2:
            changes = []

        if not changes:
            # Still update weather fields silently
            self._apply_weather_to_meeting(meeting, weather)
            await db.commit()
            return

        # Build alert message
        parts = [f"Weather update at {meeting.venue}:"]
        parts.extend(changes)
        message = "\n".join(parts)

        # Update meeting fields
        self._apply_weather_to_meeting(meeting, weather)
        await db.commit()

        # Post via change alert system (with DB-level dedup to survive restarts)
        from punty.results.change_detection import ChangeAlert
        from punty.models.live_update import LiveUpdate

        alert = ChangeAlert(
            change_type="weather",
            meeting_id=meeting_id,
            message=message,
        )
        dedup_key = alert.dedup_key
        if meeting_id not in self.alerted_changes:
            self.alerted_changes[meeting_id] = set()
        if dedup_key not in self.alerted_changes[meeting_id]:
            # DB-level check: has this exact message already been posted today?
            existing = await db.execute(
                select(LiveUpdate.id).where(
                    LiveUpdate.meeting_id == meeting_id,
                    LiveUpdate.update_type == "weather_alert",
                    LiveUpdate.content == message,
                ).limit(1)
            )
            if existing.scalar():
                self.alerted_changes[meeting_id].add(dedup_key)
                logger.info(f"Weather alert already posted for {meeting.venue} (DB dedup)")
            else:
                self.alerted_changes[meeting_id].add(dedup_key)
                await self._post_change_alert(db, meeting, alert)
                self.weather_alerts_posted[meeting_id] = self.weather_alerts_posted.get(meeting_id, 0) + 1
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
        """Post a change detection alert to Twitter."""
        from punty.models.live_update import LiveUpdate
        from punty.delivery.twitter import TwitterDelivery

        # Find the meeting's early mail for reply threading
        early_mail = await self._get_posted_early_mail(db, meeting.id)
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

        # Post to Facebook as standalone post
        # (commenting requires pages_manage_engagement — App Review pending)
        fb_post_id = None
        if early_mail.facebook_id:
            # Facebook live updates disabled (standalone posts not useful without commenting)
            pass

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
            from punty.scrapers.orchestrator import _should_update_condition
            meeting = await db.get(Meeting, meeting_id)
            if meeting and _should_update_condition(tc, meeting.track_condition or ""):
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
