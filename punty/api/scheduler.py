"""API endpoints for scheduler management."""

from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.database import get_db

router = APIRouter()


class JobConfig(BaseModel):
    """Configuration for a scheduled job."""

    meeting_id: str
    job_type: str  # scrape_race_cards, scrape_speed_maps, scrape_odds, scrape_results
    cron_expression: Optional[str] = None
    enabled: bool = True


@router.get("/jobs")
async def list_jobs(meeting_id: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """List all scheduled jobs."""
    from punty.models.content import ScheduledJob
    from sqlalchemy import select

    query = select(ScheduledJob)
    if meeting_id:
        query = query.where(ScheduledJob.meeting_id == meeting_id)

    result = await db.execute(query)
    jobs = result.scalars().all()
    return [j.to_dict() for j in jobs]


@router.post("/jobs")
async def create_job(config: JobConfig, db: AsyncSession = Depends(get_db)):
    """Create a new scheduled job."""
    from punty.models.content import ScheduledJob
    import uuid

    job = ScheduledJob(
        id=str(uuid.uuid4()),
        meeting_id=config.meeting_id,
        job_type=config.job_type,
        cron_expression=config.cron_expression,
        enabled=config.enabled,
    )
    db.add(job)
    await db.commit()
    return job.to_dict()


@router.put("/jobs/{job_id}/toggle")
async def toggle_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Enable or disable a job."""
    from punty.models.content import ScheduledJob
    from sqlalchemy import select
    from fastapi import HTTPException

    result = await db.execute(select(ScheduledJob).where(ScheduledJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.enabled = not job.enabled
    await db.commit()
    return job.to_dict()


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a scheduled job."""
    from punty.models.content import ScheduledJob
    from sqlalchemy import select
    from fastapi import HTTPException

    result = await db.execute(select(ScheduledJob).where(ScheduledJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await db.delete(job)
    await db.commit()
    return {"status": "deleted"}


@router.post("/run/{job_type}")
async def run_job_now(job_type: str, meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Manually trigger a job to run immediately."""
    # Will be implemented with scheduler
    return {"status": "triggered", "job_type": job_type, "meeting_id": meeting_id}


@router.get("/status")
async def scheduler_status():
    """Get scheduler status and scheduled jobs."""
    from punty.scheduler.manager import scheduler_manager

    jobs = scheduler_manager.get_jobs()
    job_info = []
    for job in jobs:
        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        job_info.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run,
            "next_run_formatted": job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z") if job.next_run_time else None,
        })

    morning_time = scheduler_manager.get_morning_job_time()

    return {
        "running": scheduler_manager._started,
        "jobs": job_info,
        "morning_prep_time": morning_time,
    }


@router.post("/morning-prep/run")
async def run_morning_prep_now():
    """Manually trigger the morning prep job to run immediately."""
    from punty.scheduler.jobs import daily_morning_prep
    import asyncio

    # Run in background
    asyncio.create_task(daily_morning_prep())

    return {"status": "triggered", "message": "Morning prep job started in background"}


@router.post("/morning-prep/reschedule")
async def reschedule_morning_prep():
    """Reschedule the morning prep job with a new random time."""
    from punty.scheduler.manager import scheduler_manager, get_random_morning_time
    from punty.scheduler.jobs import daily_morning_prep
    from punty.config import MELB_TZ

    scheduler_manager.remove_job("daily-morning-prep")

    hour, minute = get_random_morning_time()
    scheduler_manager.add_job(
        "daily-morning-prep",
        daily_morning_prep,
        trigger_type="cron",
        hour=hour,
        minute=minute,
        timezone=MELB_TZ,
    )

    return {
        "status": "rescheduled",
        "new_time": f"{hour:02d}:{minute:02d}",
    }


@router.post("/morning-prep/toggle")
async def toggle_morning_prep():
    """Pause or resume the morning prep automation."""
    from punty.scheduler.manager import scheduler_manager

    job = scheduler_manager.get_job("daily-morning-prep")
    if not job:
        return {"status": "error", "message": "Job not found"}

    # Check if job is paused by looking at next_run_time
    if job.next_run_time is None:
        # Job is paused, resume it
        scheduler_manager.resume_job("daily-morning-prep")
        return {"status": "resumed", "paused": False}
    else:
        # Job is running, pause it
        scheduler_manager.pause_job("daily-morning-prep")
        return {"status": "paused", "paused": True}


@router.get("/morning-prep/paused")
async def get_morning_prep_paused():
    """Check if morning prep is paused."""
    from punty.scheduler.manager import scheduler_manager

    job = scheduler_manager.get_job("daily-morning-prep")
    if not job:
        return {"paused": True, "exists": False}

    return {"paused": job.next_run_time is None, "exists": True}


@router.get("/activity-log")
async def get_activity_log(limit: int = 50):
    """Get recent activity log entries."""
    from punty.scheduler.activity_log import activity_log

    entries = activity_log.get_entries(limit=limit)
    return {"entries": entries}


@router.get("/full-status")
async def get_full_status():
    """Get scheduler status, pending jobs, and activity log."""
    from punty.scheduler.manager import scheduler_manager
    from punty.scheduler.activity_log import activity_log
    from punty.scrapers.playwright_base import is_scrape_in_progress

    # Get APScheduler jobs
    jobs = scheduler_manager.get_jobs()
    pending_jobs = []
    for job in jobs:
        if job.next_run_time:
            pending_jobs.append({
                "id": job.id,
                "name": job.name or job.id,
                "next_run": job.next_run_time.strftime("%H:%M:%S") if job.next_run_time else None,
                "next_run_date": job.next_run_time.strftime("%Y-%m-%d") if job.next_run_time else None,
            })

    # Sort by next run time
    pending_jobs.sort(key=lambda x: x["next_run"] or "99:99:99")

    # Check if scrape is in progress
    scrape_active, current_scrape = is_scrape_in_progress()

    return {
        "scheduler_running": scheduler_manager._started,
        "morning_prep_time": scheduler_manager.get_morning_job_time(),
        "pending_jobs": pending_jobs[:10],  # Top 10 pending
        "scrape_in_progress": scrape_active,
        "current_scrape": current_scrape,
        "activity_log": activity_log.get_entries(limit=30),
    }
