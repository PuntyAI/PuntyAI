"""Scheduler manager for background jobs."""

import logging
import random
from datetime import datetime, timedelta, time
from typing import Optional, Callable, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from punty.scheduler.jobs import JobType

logger = logging.getLogger(__name__)


def get_random_morning_time() -> tuple[int, int]:
    """Get a random time between 7:30 AM and 8:30 AM.

    Returns (hour, minute) tuple.
    """
    # Random minute between 7:30 (450 mins) and 8:30 (510 mins)
    total_minutes = random.randint(7 * 60 + 30, 8 * 60 + 30)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return hour, minute


class SchedulerManager:
    """Manages background job scheduling."""

    def __init__(self):
        from punty.config import MELB_TZ
        self.scheduler = AsyncIOScheduler(timezone=MELB_TZ)
        self._started = False

    async def start(self) -> None:
        """Start the scheduler."""
        if not self._started:
            self.scheduler.start()
            self._started = True
            logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._started:
            self.scheduler.shutdown(wait=True)
            self._started = False
            logger.info("Scheduler stopped")

    def add_job(
        self,
        job_id: str,
        func: Callable,
        trigger_type: str = "interval",
        **trigger_kwargs
    ) -> None:
        """Add a job to the scheduler.

        Args:
            job_id: Unique identifier for the job
            func: The async function to run
            trigger_type: One of 'interval', 'cron', 'date'
            **trigger_kwargs: Arguments for the trigger
        """
        from punty.config import MELB_TZ

        if trigger_type == "interval":
            trigger = IntervalTrigger(**trigger_kwargs)
        elif trigger_type == "cron":
            if "timezone" not in trigger_kwargs:
                trigger_kwargs["timezone"] = MELB_TZ
            trigger = CronTrigger(**trigger_kwargs)
        elif trigger_type == "date":
            if "timezone" not in trigger_kwargs:
                trigger_kwargs["timezone"] = MELB_TZ
            trigger = DateTrigger(**trigger_kwargs)
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")

        self.scheduler.add_job(
            func,
            trigger,
            id=job_id,
            replace_existing=True,
            misfire_grace_time=60,
        )
        logger.info(f"Added job: {job_id} with {trigger_type} trigger")

    def remove_job(self, job_id: str) -> None:
        """Remove a job from the scheduler."""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
        except Exception as e:
            logger.warning(f"Could not remove job {job_id}: {e}")

    def get_job(self, job_id: str) -> Optional[Any]:
        """Get a job by ID."""
        return self.scheduler.get_job(job_id)

    def get_jobs(self) -> list:
        """Get all scheduled jobs."""
        return self.scheduler.get_jobs()

    def pause_job(self, job_id: str) -> None:
        """Pause a job."""
        self.scheduler.pause_job(job_id)
        logger.info(f"Paused job: {job_id}")

    def resume_job(self, job_id: str) -> None:
        """Resume a paused job."""
        self.scheduler.resume_job(job_id)
        logger.info(f"Resumed job: {job_id}")

    async def setup_meeting_jobs(
        self,
        meeting_id: str,
        venue: str,
        race_date: datetime,
        first_race_time: Optional[datetime] = None,
    ) -> None:
        """Set up all scheduled jobs for a race meeting.

        This creates the standard job schedule:
        - 6:00 AM: Scrape race cards
        - Every 30min from 9:00 AM: Scrape speed maps
        - Every 15min from 10:00 AM: Scrape odds
        - 30min before first race: Scrape late mail
        - After each race: Scrape results (triggered manually or by time)
        """
        from punty.scheduler.jobs import (
            scrape_race_cards,
            scrape_speed_maps,
            scrape_odds,
            check_context_changes,
        )
        from punty.models.database import async_session

        # Helper to create job wrapper with db session
        def create_job_wrapper(job_func, **kwargs):
            async def wrapper():
                async with async_session() as db:
                    try:
                        result = await job_func(db, meeting_id, venue, race_date.date(), **kwargs)
                        logger.info(f"Job completed: {job_func.__name__} - {result}")
                    except Exception as e:
                        logger.error(f"Job failed: {job_func.__name__} - {e}")
            return wrapper

        race_day = race_date.date()

        # Only schedule if race day is today or in the future
        from punty.config import melb_today
        if race_day < melb_today():
            logger.info(f"Skipping job setup for past date: {race_day}")
            return

        # Scrape race cards at 6:00 AM on race day
        self.add_job(
            f"{meeting_id}-race-cards",
            create_job_wrapper(scrape_race_cards),
            trigger_type="cron",
            hour=6,
            minute=0,
            day=race_day.day,
            month=race_day.month,
        )

        # Scrape speed maps every 30 minutes from 9:00 AM to first race
        self.add_job(
            f"{meeting_id}-speed-maps",
            create_job_wrapper(scrape_speed_maps),
            trigger_type="cron",
            hour="9-14",
            minute="0,30",
            day=race_day.day,
            month=race_day.month,
        )

        # Scrape odds every 15 minutes from 10:00 AM
        self.add_job(
            f"{meeting_id}-odds",
            create_job_wrapper(scrape_odds),
            trigger_type="cron",
            hour="10-18",
            minute="*/15",
            day=race_day.day,
            month=race_day.month,
        )

        # Check context changes after speed maps/odds updates
        async def check_changes_wrapper():
            async with async_session() as db:
                try:
                    result = await check_context_changes(db, meeting_id)
                    if result["status"] == "changes_detected":
                        logger.warning(
                            f"Significant changes detected for {meeting_id}: {result['changes']}"
                        )
                        # TODO: Trigger re-evaluation workflow
                except Exception as e:
                    logger.error(f"Context check failed: {e}")

        self.add_job(
            f"{meeting_id}-context-check",
            check_changes_wrapper,
            trigger_type="cron",
            hour="9-14",
            minute="5,35",  # 5 minutes after speed map scrape
            day=race_day.day,
            month=race_day.month,
        )

        logger.info(f"Set up jobs for meeting: {meeting_id}")

    def remove_meeting_jobs(self, meeting_id: str) -> None:
        """Remove all jobs for a meeting."""
        job_suffixes = ["race-cards", "speed-maps", "odds", "context-check", "results"]
        for suffix in job_suffixes:
            job_id = f"{meeting_id}-{suffix}"
            self.remove_job(job_id)

    async def setup_daily_morning_job(self) -> None:
        """Set up the daily scheduled jobs.

        - 00:05 AM: Calendar scrape (find meetings, auto-select, initial data load)
        - 05:00 AM: Morning scrape (full re-scrape + speed maps + generate early mail)
        - Thursday 22:00: Weekly pattern refresh (patterns, awards, ledger, news)
        - Friday 08:00: Weekly blog generation

        Early mail is generated during morning scrape (~05:30) and approved
        but not posted. meeting_pre_race_job (T-90m) compares snapshots and
        posts existing content or regenerates if material changes are detected.
        """
        from punty.scheduler.jobs import daily_calendar_scrape, daily_morning_scrape
        from punty.scheduler.jobs import weekly_pattern_refresh, weekly_blog_job
        from punty.config import MELB_TZ

        logger.info("Scheduling daily calendar scrape job for 00:05 Melbourne time")

        # Calendar scrape runs at 12:05 AM
        self.add_job(
            "daily-calendar-scrape",
            daily_calendar_scrape,
            trigger_type="cron",
            hour=0,
            minute=5,
            timezone=MELB_TZ,
        )

        # Morning scrape runs at 5:00 AM — heavy initial data load
        self.add_job(
            "daily-morning-scrape",
            daily_morning_scrape,
            trigger_type="cron",
            hour=5,
            minute=0,
            timezone=MELB_TZ,
        )

        # Weekly pattern refresh — Thursday 10pm
        self.add_job(
            "weekly-pattern-refresh",
            weekly_pattern_refresh,
            trigger_type="cron",
            day_of_week="thu",
            hour=22,
            minute=0,
            timezone=MELB_TZ,
        )

        # Weekly blog generation — Friday 8am
        self.add_job(
            "weekly-blog",
            weekly_blog_job,
            trigger_type="cron",
            day_of_week="fri",
            hour=8,
            minute=0,
            timezone=MELB_TZ,
        )

        logger.info("Daily calendar scrape job configured")
        logger.info("Daily morning scrape job configured (05:00)")
        logger.info("Weekly pattern refresh job configured (Thu 22:00)")
        logger.info("Weekly blog job configured (Fri 08:00)")

    async def setup_legacy_morning_job(self) -> None:
        """Legacy method - kept for backwards compatibility.

        This now just calls setup_daily_morning_job().
        """
        await self.setup_daily_morning_job()

    def get_morning_job_time(self) -> Optional[str]:
        """Get the scheduled time for the morning scrape job."""
        job = self.get_job("daily-morning-scrape")
        if job and job.next_run_time:
            return job.next_run_time.strftime("%H:%M")
        return None

    async def setup_meeting_automation(self, meeting_id: str) -> dict:
        """Schedule automation jobs for a meeting based on race times.

        - Pre-race job: 90 min before first race (refresh + compare + post/regen)
          Morning content is already generated at ~05:30 by morning_generate_all().
        - Post-race job: 30 min after last race

        Returns: dict with scheduled job times
        """
        from punty.models.database import async_session
        from punty.models.meeting import Meeting, Race
        from punty.scheduler.jobs import meeting_pre_race_job, meeting_post_race_job
        from punty.config import MELB_TZ, melb_now
        from sqlalchemy import select, func

        async with async_session() as db:
            # Get meeting
            result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
            meeting = result.scalar_one_or_none()

            if not meeting:
                logger.warning(f"Meeting not found for automation: {meeting_id}")
                return {"error": f"Meeting not found: {meeting_id}"}

            # Get first and last race times
            race_result = await db.execute(
                select(
                    func.min(Race.start_time).label("first_race"),
                    func.max(Race.start_time).label("last_race"),
                ).where(Race.meeting_id == meeting_id)
            )
            row = race_result.one_or_none()

            if not row or not row.first_race:
                logger.warning(f"No race times found for {meeting_id}")
                return {"error": f"No race times for: {meeting_id}"}

            first_race_time = row.first_race
            last_race_time = row.last_race

            # Calculate job times
            from datetime import timedelta

            # Pre-race: 90 minutes before first race (morning gen already done)
            pre_race_time = first_race_time - timedelta(minutes=90)

            # Post-race: 60 minutes after last race (wrap-up generation + social delivery)
            post_race_time = last_race_time + timedelta(minutes=60)

            now = melb_now().replace(tzinfo=None)

            scheduled = {
                "meeting_id": meeting_id,
                "venue": meeting.venue,
                "first_race": first_race_time.strftime("%H:%M"),
                "last_race": last_race_time.strftime("%H:%M"),
                "jobs": [],
            }

            # Schedule pre-race job if still in the future
            if pre_race_time > now:
                # Create wrapper function for the job
                async def pre_race_wrapper():
                    return await meeting_pre_race_job(meeting_id)

                self.add_job(
                    f"{meeting_id}-pre-race",
                    pre_race_wrapper,
                    trigger_type="date",
                    run_date=pre_race_time,
                )
                scheduled["jobs"].append({
                    "type": "pre_race",
                    "time": pre_race_time.strftime("%H:%M"),
                })
                logger.info(f"Scheduled pre-race job for {meeting.venue} at {pre_race_time.strftime('%H:%M')}")
            else:
                # Pre-race time passed — check if content was already generated.
                # If not (e.g. restart killed the original job), run it now as catch-up.
                from punty.models.content import Content
                content_result = await db.execute(
                    select(Content).where(
                        Content.meeting_id == meeting_id,
                        Content.content_type == "early_mail",
                        Content.status.notin_(["rejected", "superseded"]),
                    )
                )
                existing_content = content_result.scalars().first()

                if existing_content:
                    # Content exists — but was it posted to socials?
                    if existing_content.status in ("approved",) and not existing_content.twitter_id:
                        logger.warning(
                            f"Pre-race time passed for {meeting.venue} — content exists but NOT posted to socials, "
                            f"running catch-up posting NOW"
                        )
                        import asyncio
                        from punty.scheduler.automation import post_existing_content

                        async def _catchup_post(cid=existing_content.id):
                            from punty.models.database import async_session
                            async with async_session() as post_db:
                                await post_existing_content(cid, post_db)
                                await post_db.commit()

                        asyncio.create_task(_catchup_post())
                        scheduled["jobs"].append({
                            "type": "pre_race",
                            "time": "catch-up-post",
                        })
                    else:
                        logger.info(f"Pre-race time passed for {meeting.venue} — content already exists and posted, skipping")
                else:
                    # Only catch up if the last race hasn't finished yet
                    if last_race_time > now:
                        logger.warning(
                            f"Pre-race time passed for {meeting.venue} but NO content exists — "
                            f"running catch-up pre-race job NOW"
                        )
                        import asyncio
                        asyncio.create_task(meeting_pre_race_job(meeting_id))
                        scheduled["jobs"].append({
                            "type": "pre_race",
                            "time": "catch-up",
                        })
                    else:
                        logger.info(f"Pre-race time passed for {meeting.venue} — meeting already finished, skipping")

            # Schedule post-race job if still in the future
            if post_race_time > now:
                async def post_race_wrapper():
                    return await meeting_post_race_job(meeting_id)

                self.add_job(
                    f"{meeting_id}-post-race",
                    post_race_wrapper,
                    trigger_type="date",
                    run_date=post_race_time,
                )
                scheduled["jobs"].append({
                    "type": "post_race",
                    "time": post_race_time.strftime("%H:%M"),
                })
                logger.info(f"Scheduled post-race job for {meeting.venue} at {post_race_time.strftime('%H:%M')}")
            else:
                # Post-race time passed — check if wrap-up was generated
                from punty.models.content import Content
                wrapup_result = await db.execute(
                    select(Content).where(
                        Content.meeting_id == meeting_id,
                        Content.content_type == "meeting_wrapup",
                        Content.status.notin_(["rejected", "superseded"]),
                    )
                )
                existing_wrapup = wrapup_result.scalars().first()

                if existing_wrapup:
                    logger.info(f"Post-race time passed for {meeting.venue} — wrapup already exists, skipping")
                else:
                    logger.warning(
                        f"Post-race time passed for {meeting.venue} but NO wrapup exists — "
                        f"running catch-up post-race job NOW"
                    )
                    import asyncio
                    asyncio.create_task(meeting_post_race_job(meeting_id))
                    scheduled["jobs"].append({
                        "type": "post_race",
                        "time": "catch-up",
                    })

            return scheduled

    async def setup_daily_automation(self) -> dict:
        """Set up automation for all selected meetings today.

        Called after midnight calendar scrape or on app startup.
        """
        from punty.models.database import async_session
        from punty.models.meeting import Meeting
        from punty.config import melb_today
        from sqlalchemy import select, or_

        results = {
            "meetings_scheduled": [],
            "errors": [],
        }

        async with async_session() as db:
            # Get all selected meetings for today
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == melb_today(),
                    Meeting.selected == True,
                    or_(Meeting.meeting_type == None, Meeting.meeting_type == "race")
                )
            )
            meetings = result.scalars().all()

            for meeting in meetings:
                try:
                    scheduled = await self.setup_meeting_automation(meeting.id)
                    if "error" not in scheduled:
                        results["meetings_scheduled"].append(scheduled)
                    else:
                        results["errors"].append(scheduled["error"])
                except Exception as e:
                    logger.error(f"Failed to setup automation for {meeting.venue}: {e}")
                    results["errors"].append(f"{meeting.venue}: {str(e)}")

        logger.info(f"Daily automation setup complete: {len(results['meetings_scheduled'])} meetings")
        return results


# Global scheduler instance
scheduler_manager = SchedulerManager()
