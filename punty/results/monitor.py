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
        while self.running:
            # Reset tracking state at midnight to prevent unbounded growth
            today = melb_now().date()
            if today > self._last_reset_date:
                old_races = sum(len(s) for s in self.processed_races.values())
                old_wrapups = len(self.wrapups_generated)
                self.processed_races.clear()
                self.wrapups_generated.clear()
                self.pace_updates_posted.clear()
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
                                await auto_post_to_twitter(content_id, db)
                                await auto_post_to_facebook(content_id, db)
                            else:
                                logger.warning(f"Wrap-up auto-approval failed for {meeting.venue}: {approval.get('issues')}")
                except Exception as e:
                    logger.error(f"Failed to generate wrap-up for {meeting.venue}: {e}")

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

            # Post to Twitter
            reply_tweet_id = None
            if early_mail.twitter_id:
                twitter = TwitterDelivery(db)
                if await twitter.is_configured():
                    try:
                        reply_result = await twitter.post_reply(early_mail.twitter_id, tweet_text)
                        reply_tweet_id = reply_result.get("tweet_id")
                        logger.info(f"Posted celebration reply for {pick.horse_name} ({pick.pnl:+.2f})")
                    except Exception as e:
                        logger.warning(f"Failed to post celebration reply: {e}")

            # Post to Facebook as standalone update
            fb_post_id = None
            fb = FacebookDelivery(db)
            if await fb.is_configured():
                try:
                    fb_result = await fb.post_update(tweet_text)
                    fb_post_id = fb_result.get("post_id")
                    logger.info(f"Posted Facebook celebration for {pick.horse_name}")
                except Exception as e:
                    logger.warning(f"Failed to post Facebook celebration: {e}")

            # Save to DB regardless of post success
            update = LiveUpdate(
                meeting_id=meeting_id,
                race_number=race_number,
                update_type="celebration",
                content=tweet_text,
                tweet_id=reply_tweet_id,
                parent_tweet_id=early_mail.twitter_id,
                facebook_comment_id=fb_post_id,
                parent_facebook_id=early_mail.facebook_id,
                horse_name=pick.horse_name,
                odds=pick.odds_at_tip,
                pnl=pick.pnl,
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

        # Post to Twitter
        reply_tweet_id = None
        if early_mail.twitter_id:
            twitter = TwitterDelivery(db)
            if await twitter.is_configured():
                try:
                    reply_result = await twitter.post_reply(early_mail.twitter_id, tweet_text)
                    reply_tweet_id = reply_result.get("tweet_id")
                    self.pace_updates_posted[meeting_id] = posted_count + 1
                    logger.info(f"Posted pace analysis for {meeting.venue} (update {posted_count + 1}, bias: {bias_result.bias_type})")
                except Exception as e:
                    logger.warning(f"Failed to post pace analysis reply: {e}")

        # Post to Facebook as standalone update
        fb_post_id = None
        fb = FacebookDelivery(db)
        if await fb.is_configured():
            try:
                fb_result = await fb.post_update(tweet_text)
                fb_post_id = fb_result.get("post_id")
                if not reply_tweet_id:
                    self.pace_updates_posted[meeting_id] = posted_count + 1
                logger.info(f"Posted Facebook pace analysis for {meeting.venue}")
            except Exception as e:
                logger.warning(f"Failed to post Facebook pace analysis: {e}")

        # Save to DB regardless of post success
        update = LiveUpdate(
            meeting_id=meeting_id,
            race_number=just_completed,
            update_type="pace_analysis",
            content=tweet_text,
            tweet_id=reply_tweet_id,
            parent_tweet_id=early_mail.twitter_id,
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
