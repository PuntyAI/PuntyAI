"""Scheduler manager for background jobs."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from punty.scheduler.jobs import JobType

logger = logging.getLogger(__name__)


class SchedulerManager:
    """Manages background job scheduling."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
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
        if trigger_type == "interval":
            trigger = IntervalTrigger(**trigger_kwargs)
        elif trigger_type == "cron":
            trigger = CronTrigger(**trigger_kwargs)
        elif trigger_type == "date":
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
        if race_day < datetime.now().date():
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


# Global scheduler instance
scheduler_manager = SchedulerManager()
