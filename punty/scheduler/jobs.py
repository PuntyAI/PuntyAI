"""Job definitions for scheduled tasks."""

import asyncio
import logging
import random
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Rate limit handling for batch jobs
RATE_LIMIT_QUEUE_PAUSE = 60  # seconds to pause queue when rate limited

# Calendar retry: max retries until noon Melbourne time
CALENDAR_RETRY_MAX_HOUR = 12  # stop retrying after noon


def _schedule_calendar_retry(scheduler_manager, race_date: date) -> None:
    """Schedule an hourly retry of the calendar scrape if no meetings were found.

    Racing.com often doesn't publish next-day meetings until the morning,
    so the midnight scrape may find 0 meetings. This schedules retries
    every hour until meetings are found or noon is reached.
    """
    from punty.config import melb_now, MELB_TZ

    now = melb_now()
    next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)

    # Don't retry past noon — if still no meetings, it's genuinely empty
    if next_hour.hour >= CALENDAR_RETRY_MAX_HOUR:
        logger.info(f"Calendar retry skipped — past {CALENDAR_RETRY_MAX_HOUR}:00, no more retries")
        return

    job_id = f"calendar-retry-{next_hour.strftime('%H%M')}"

    # Check if a retry is already scheduled
    if scheduler_manager.get_job(job_id):
        logger.info(f"Calendar retry already scheduled: {job_id}")
        return

    scheduler_manager.add_job(
        job_id,
        daily_calendar_scrape,
        trigger_type="date",
        run_date=next_hour.replace(tzinfo=None),
    )
    logger.info(f"Scheduled calendar retry at {next_hour.strftime('%H:%M')} Melbourne time")


class JobType(str, Enum):
    """Types of scheduled jobs."""

    SCRAPE_RACE_CARDS = "scrape_race_cards"
    SCRAPE_SPEED_MAPS = "scrape_speed_maps"
    SCRAPE_ODDS = "scrape_odds"
    SCRAPE_LATE_MAIL = "scrape_late_mail"
    SCRAPE_RESULTS = "scrape_results"
    GENERATE_EARLY_MAIL = "generate_early_mail"
    GENERATE_RACE_PREVIEW = "generate_race_preview"
    CHECK_CONTEXT_CHANGES = "check_context_changes"
    DAILY_MORNING_PREP = "daily_morning_prep"


async def daily_calendar_scrape() -> dict:
    """
    Daily calendar scrape job that runs shortly after midnight (12:05 AM).

    This job:
    1. Scrapes the racing calendar to get today's meetings
    2. Auto-selects all meetings (skipping abandoned)
    3. Scrapes full race data for each meeting (runners, form, odds, track conditions)
    4. Deselects meetings with no races (abandoned/cancelled)
    5. Schedules meeting-specific automation jobs (pre-race, post-race)

    If 0 meetings are found, schedules hourly retries until meetings appear
    (racing.com often doesn't publish next-day meetings until morning).

    Note: Early mail generation is handled by meeting_pre_race_job
    which runs 2 hours before each meeting's first race.
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting, Race, Runner
    from punty.config import melb_today, melb_now
    from punty.scrapers.calendar import scrape_calendar
    from punty.scheduler.activity_log import log_scheduler_job, log_system
    from punty.scheduler.manager import scheduler_manager
    from sqlalchemy import select, or_

    logger.info("Starting daily calendar scrape job")
    log_scheduler_job("Calendar Scrape Started", status="info")
    results = {
        "started_at": melb_now().isoformat(),
        "calendar_scraped": False,
        "meetings_found": 0,
        "meetings_selected": 0,
        "automation_scheduled": [],
        "errors": [],
    }

    today = melb_today()

    async with async_session() as db:
        # Step 1: Scrape calendar
        try:
            logger.info("Step 1: Scraping racing calendar...")
            meetings_data = await scrape_calendar(today)
            results["calendar_scraped"] = True
            results["meetings_found"] = len(meetings_data) if meetings_data else 0
            logger.info(f"Found {results['meetings_found']} meetings")

            # If 0 meetings found, schedule hourly retry (calendar may not be populated yet)
            if not meetings_data:
                _schedule_calendar_retry(scheduler_manager, today)
                results["retry_scheduled"] = True

            # Create Meeting records for any new meetings found
            for m in meetings_data:
                meeting_id = f"{m['venue'].lower().replace(' ', '-')}-{today.isoformat()}"
                existing = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
                if not existing.scalar_one_or_none():
                    new_meeting = Meeting(
                        id=meeting_id,
                        venue=m['venue'],
                        date=today,
                        meeting_type=m.get('meeting_type', 'race'),
                    )
                    db.add(new_meeting)
                    logger.info(f"Created meeting: {m['venue']}")
            await db.commit()
        except Exception as e:
            logger.error(f"Calendar scrape failed: {e}")
            results["errors"].append(f"Calendar: {str(e)}")
            # Also retry on error
            _schedule_calendar_retry(scheduler_manager, today)
            results["retry_scheduled"] = True

        # Step 2: Auto-select ALL meetings
        try:
            logger.info("Step 2: Auto-selecting all meetings...")

            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
            meetings = result.scalars().all()

            selected_count = 0
            for meeting in meetings:
                if not meeting.selected:
                    meeting.selected = True
                    selected_count += 1
                    logger.info(f"Auto-selected: {meeting.venue}")

            await db.commit()
            results["meetings_selected"] = selected_count
            logger.info(f"Auto-selected {selected_count} meetings")
        except Exception as e:
            logger.error(f"Auto-select failed: {e}")
            results["errors"].append(f"Auto-select: {str(e)}")

        # Step 3: Lightweight fields-only scrape (no form history, no racing.com)
        try:
            from punty.scrapers.orchestrator import scrape_meeting_fields_only
            from punty.scrapers.punting_form import PuntingFormScraper, clear_meetings_cache

            # Clear stale meeting cache from previous day
            clear_meetings_cache()

            logger.info("Step 3: Scraping fields data for all meetings...")
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                )
            )
            selected_meetings = result.scalars().all()
            scrape_count = 0

            # Shared PF scraper for the entire batch
            pf_scraper = await PuntingFormScraper.from_settings(db)
            try:
                for meeting in selected_meetings:
                    try:
                        logger.info(f"Scraping fields for {meeting.venue}...")
                        await scrape_meeting_fields_only(meeting.id, db, pf_scraper=pf_scraper)
                        scrape_count += 1
                    except Exception as e:
                        logger.error(f"Scrape failed for {meeting.venue}: {e}")
                        results["errors"].append(f"Scrape {meeting.venue}: {str(e)}")
            finally:
                await pf_scraper.close()

            results["meetings_scraped"] = scrape_count
            logger.info(f"Scraped fields for {scrape_count}/{len(selected_meetings)} meetings")
        except Exception as e:
            logger.error(f"Fields scrape failed: {e}")
            results["errors"].append(f"Fields scrape: {str(e)}")

        # Step 3b: Deselect abandoned meetings (no races after scrape)
        try:
            logger.info("Step 3b: Checking for abandoned/empty meetings...")
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                )
            )
            selected_meetings = result.scalars().all()
            abandoned = []

            for meeting in selected_meetings:
                race_result = await db.execute(
                    select(Race).where(Race.meeting_id == meeting.id).limit(1)
                )
                if not race_result.scalar_one_or_none():
                    meeting.selected = False
                    abandoned.append(meeting.venue)
                    logger.info(f"Deselected {meeting.venue} — no races found (abandoned/cancelled)")

            if abandoned:
                await db.commit()
                results["abandoned"] = abandoned
                logger.info(f"Deselected {len(abandoned)} abandoned meetings: {abandoned}")
        except Exception as e:
            logger.error(f"Abandoned check failed: {e}")
            results["errors"].append(f"Abandoned check: {str(e)}")

    # Step 4: Schedule meeting automation
    try:
        logger.info("Step 4: Scheduling meeting automation...")
        automation_result = await scheduler_manager.setup_daily_automation()
        results["automation_scheduled"] = automation_result.get("meetings_scheduled", [])
        if automation_result.get("errors"):
            results["errors"].extend(automation_result["errors"])
    except Exception as e:
        logger.error(f"Automation setup failed: {e}")
        results["errors"].append(f"Automation: {str(e)}")

    results["completed_at"] = melb_now().isoformat()
    logger.info(f"Daily calendar scrape complete: {results}")

    # Log completion
    scheduled = len(results.get("automation_scheduled", []))
    errors = len(results.get("errors", []))
    if errors:
        log_system(f"Calendar scrape complete: {scheduled} meetings scheduled, {errors} errors", status="warning")
    else:
        log_system(f"Calendar scrape complete: {scheduled} meetings scheduled", status="success")

    # Send email notification
    try:
        from punty.delivery.email import send_email
        from punty.models.settings import AppSettings

        async with async_session() as db:
            result = await db.execute(select(AppSettings).where(AppSettings.key == "notification_email"))
            setting = result.scalar_one_or_none()
            notification_email = setting.value if setting and setting.value else None

        if notification_email:
            meetings_info = "\n".join([
                f"- {m['venue']}: pre-race {m['jobs'][0]['time'] if m['jobs'] else 'N/A'}"
                for m in results.get("automation_scheduled", [])
            ])

            await send_email(
                to_email=notification_email,
                subject=f"PuntyAI Calendar: {results['meetings_found']} meetings found",
                body_html=f"""
                <h2>Daily Calendar Scrape Complete</h2>
                <p><strong>Meetings found:</strong> {results['meetings_found']}</p>
                <p><strong>Scheduled for automation:</strong> {scheduled}</p>
                <pre>{meetings_info}</pre>
                {"<p style='color:red'>Errors: " + ", ".join(results['errors']) + "</p>" if errors else ""}
                """,
                body_text=f"Calendar: {results['meetings_found']} meetings, {scheduled} scheduled",
            )
            results["email_sent"] = True
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        results["email_sent"] = False

    return results


async def daily_morning_scrape() -> dict:
    """
    Daily morning scrape job — runs at 5:00 AM Melbourne time.

    This is the heavy initial data load for the day:
    1. Full re-scrape for all selected meetings (runners, form, odds, track conditions)
    2. Speed maps for all selected meetings

    By 5:00 AM, racing.com should have complete field data even if the
    midnight calendar scrape found incomplete data.

    The later pre-race job (2h before first race) only applies lightweight
    updates (odds refresh, scratchings) rather than repeating this heavy scrape.
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting, Race
    from punty.config import melb_today, melb_now
    from punty.scrapers.orchestrator import scrape_meeting_full, scrape_speed_maps_stream
    from punty.scheduler.activity_log import log_scheduler_job, log_system
    from sqlalchemy import select, or_

    logger.info("Starting daily morning scrape job (5:00 AM)")
    log_scheduler_job("Morning Scrape Started", status="info")
    results = {
        "started_at": melb_now().isoformat(),
        "meetings_scraped": [],
        "speed_maps_done": [],
        "errors": [],
    }

    today = melb_today()

    async with async_session() as db:
        # Get all selected meetings for today
        result = await db.execute(
            select(Meeting).where(
                Meeting.date == today,
                Meeting.selected == True,
                or_(Meeting.meeting_type == None, Meeting.meeting_type == "race"),
            ).order_by(Meeting.venue)
        )
        meetings = result.scalars().all()

        if not meetings:
            logger.info("No selected meetings for today — morning scrape skipped")
            results["skipped"] = True
            log_system("Morning scrape: no meetings to scrape", status="info")
            return results

        logger.info(f"Morning scrape: {len(meetings)} meetings to process")

        # Step 1: Full data scrape for each meeting (shared PF scraper)
        from punty.scrapers.punting_form import PuntingFormScraper
        pf_scraper = await PuntingFormScraper.from_settings(db)
        try:
            for meeting in meetings:
                try:
                    logger.info(f"Scraping full data for {meeting.venue}...")
                    await scrape_meeting_full(meeting.id, db, pf_scraper=pf_scraper)
                    results["meetings_scraped"].append(meeting.venue)
                    logger.info(f"Full scrape complete for {meeting.venue}")
                except Exception as e:
                    logger.error(f"Full scrape failed for {meeting.venue}: {e}")
                    results["errors"].append(f"scrape {meeting.venue}: {str(e)}")
        finally:
            await pf_scraper.close()

        # Step 1b: Deselect meetings with no races (abandoned/cancelled after scrape)
        try:
            for meeting in meetings:
                if meeting.venue not in results["meetings_scraped"]:
                    continue
                race_result = await db.execute(
                    select(Race).where(Race.meeting_id == meeting.id).limit(1)
                )
                if not race_result.scalar_one_or_none():
                    meeting.selected = False
                    logger.info(f"Deselected {meeting.venue} — no races found after morning scrape")
            await db.commit()
        except Exception as e:
            logger.error(f"Abandoned check failed: {e}")

        # Step 2: Speed maps for each meeting (still selected)
        result = await db.execute(
            select(Meeting).where(
                Meeting.date == today,
                Meeting.selected == True,
                or_(Meeting.meeting_type == None, Meeting.meeting_type == "race"),
            ).order_by(Meeting.venue)
        )
        still_selected = result.scalars().all()

        for meeting in still_selected:
            try:
                logger.info(f"Scraping speed maps for {meeting.venue}...")
                async for _ in scrape_speed_maps_stream(meeting.id, db):
                    pass
                results["speed_maps_done"].append(meeting.venue)
                logger.info(f"Speed maps complete for {meeting.venue}")
            except Exception as e:
                logger.error(f"Speed maps failed for {meeting.venue}: {e}")
                results["errors"].append(f"speed_maps {meeting.venue}: {str(e)}")

    results["completed_at"] = melb_now().isoformat()

    # Log completion
    scraped = len(results["meetings_scraped"])
    speed_maps = len(results["speed_maps_done"])
    errors = len(results["errors"])
    if errors:
        log_system(f"Morning scrape complete: {scraped} scraped, {speed_maps} speed maps, {errors} errors", status="warning")
    else:
        log_system(f"Morning scrape complete: {scraped} scraped, {speed_maps} speed maps", status="success")

    logger.info(f"Daily morning scrape complete: {results}")
    return results


async def daily_morning_prep() -> dict:
    """
    Legacy daily morning preparation job - now redirects to calendar scrape.

    Note: This job is deprecated. Use daily_calendar_scrape instead.
    Keeping for backwards compatibility with existing scheduled jobs.
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting, Race, Runner
    from punty.config import melb_today, melb_now
    from punty.scrapers.calendar import scrape_calendar
    from punty.scrapers.orchestrator import scrape_meeting_full, scrape_speed_maps_stream
    from punty.ai.generator import ContentGenerator
    from punty.scheduler.activity_log import log_scheduler_job, log_system
    from sqlalchemy import select, or_

    logger.info("Starting daily morning prep job")
    log_scheduler_job("Morning Prep Started", status="info")
    results = {
        "started_at": melb_now().isoformat(),
        "calendar_scraped": False,
        "meetings_found": 0,
        "meetings_scraped": [],
        "early_mail_generated": [],
        "errors": [],
    }

    today = melb_today()

    async with async_session() as db:
        # Step 1: Scrape calendar
        try:
            logger.info("Step 1: Scraping racing calendar...")
            meetings_data = await scrape_calendar(today)
            results["calendar_scraped"] = True
            results["meetings_found"] = len(meetings_data) if meetings_data else 0
            logger.info(f"Found {results['meetings_found']} meetings")

            # Create Meeting records for any new meetings found
            from punty.models.meeting import Meeting
            for m in meetings_data:
                meeting_id = f"{m['venue'].lower().replace(' ', '-')}-{today.isoformat()}"
                existing = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
                if not existing.scalar_one_or_none():
                    new_meeting = Meeting(
                        id=meeting_id,
                        venue=m['venue'],
                        date=today,
                        meeting_type=m.get('meeting_type', 'race'),
                    )
                    db.add(new_meeting)
                    logger.info(f"Created meeting: {m['venue']}")
            await db.commit()
        except Exception as e:
            logger.error(f"Calendar scrape failed: {e}")
            results["errors"].append(f"Calendar: {str(e)}")

        # Step 2: Auto-select ALL meetings
        try:
            logger.info("Step 2: Auto-selecting all meetings...")

            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
            meetings = result.scalars().all()

            selected_count = 0
            for meeting in meetings:
                if not meeting.selected:
                    meeting.selected = True
                    selected_count += 1
                    logger.info(f"Auto-selected: {meeting.venue}")

            await db.commit()
            logger.info(f"Auto-selected {selected_count} meetings")
        except Exception as e:
            logger.error(f"Auto-select failed: {e}")
            results["errors"].append(f"Auto-select: {str(e)}")

        # Step 3: Scrape all data for selected meetings
        try:
            logger.info("Step 3: Scraping data for selected meetings...")
            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                    or_(Meeting.meeting_type == None, Meeting.meeting_type == "race")
                ).order_by(Meeting.venue)
            )
            selected_meetings = result.scalars().all()

            for meeting in selected_meetings:
                try:
                    logger.info(f"Scraping data for {meeting.venue}...")
                    # Scrape meeting data (form, odds, etc)
                    await scrape_meeting_full(meeting.id, db)

                    # Scrape speed maps
                    async for _ in scrape_speed_maps_stream(meeting.id, db):
                        pass  # Just run through the generator

                    results["meetings_scraped"].append(meeting.venue)
                    logger.info(f"Completed scraping {meeting.venue}")
                except Exception as e:
                    logger.error(f"Scrape failed for {meeting.venue}: {e}")
                    results["errors"].append(f"Scrape {meeting.venue}: {str(e)}")

            await db.commit()
        except Exception as e:
            logger.error(f"Data scrape step failed: {e}")
            results["errors"].append(f"Data scrape: {str(e)}")

        # Step 3b: Deactivate meetings without speed maps
        try:
            logger.info("Step 3b: Checking for meetings without speed maps...")
            from punty.models.meeting import Runner

            result = await db.execute(
                select(Meeting).where(
                    Meeting.date == today,
                    Meeting.selected == True,
                    or_(Meeting.meeting_type == None, Meeting.meeting_type == "race")
                )
            )
            selected_meetings = result.scalars().all()

            deactivated = []
            for meeting in selected_meetings:
                # Check if meeting has speed map data (False = incomplete, None = not attempted)
                if meeting.speed_map_complete is not True:
                    meeting.selected = False
                    deactivated.append(meeting.venue)
                    logger.info(f"Deactivated {meeting.venue} - no speed map data")

            if deactivated:
                await db.commit()
                results["deactivated_no_speedmaps"] = deactivated
                logger.info(f"Deactivated {len(deactivated)} meetings without speed maps: {deactivated}")
            else:
                logger.info("All meetings have speed map data")
        except Exception as e:
            logger.error(f"Speed map check failed: {e}")
            results["errors"].append(f"Speed map check: {str(e)}")

        # Step 4: Generate early mail for selected meetings (if enabled)
        from punty.models.settings import AppSettings
        em_setting = await db.execute(select(AppSettings).where(AppSettings.key == "enable_early_mail"))
        em_enabled = em_setting.scalar_one_or_none()
        early_mail_enabled = not em_enabled or em_enabled.value == "true"  # Default on

        if not early_mail_enabled:
            logger.info("Step 4: Skipping early mail generation (disabled in settings)")
            results["early_mail_skipped"] = True
        else:
            try:
                logger.info("Step 4: Generating early mail...")
                result = await db.execute(
                    select(Meeting).where(
                        Meeting.date == today,
                        Meeting.selected == True,
                        or_(Meeting.meeting_type == None, Meeting.meeting_type == "race")
                    ).order_by(Meeting.venue)
                )
                selected_meetings = result.scalars().all()

                generator = ContentGenerator(db)

                for meeting in selected_meetings:
                    try:
                        logger.info(f"Generating early mail for {meeting.venue}...")
                        async for event in generator.generate_early_mail_stream(meeting.id):
                            if event.get("status") == "error":
                                raise Exception(event.get("label", "Unknown error"))

                        results["early_mail_generated"].append(meeting.venue)
                        logger.info(f"Early mail generated for {meeting.venue}")
                    except Exception as e:
                        error_str = str(e).lower()
                        logger.error(f"Early mail failed for {meeting.venue}: {e}")
                        results["errors"].append(f"Early mail {meeting.venue}: {str(e)}")

                        # If rate limited, pause before next meeting
                        if "rate limit" in error_str or "429" in error_str:
                            logger.warning(f"Rate limit detected — pausing queue for {RATE_LIMIT_QUEUE_PAUSE}s")
                            await asyncio.sleep(RATE_LIMIT_QUEUE_PAUSE)
            except Exception as e:
                logger.error(f"Early mail step failed: {e}")
                results["errors"].append(f"Early mail: {str(e)}")

    results["completed_at"] = melb_now().isoformat()
    logger.info(f"Daily morning prep complete: {results}")

    # Log completion
    scraped = len(results.get("meetings_scraped", []))
    generated = len(results.get("early_mail_generated", []))
    errors = len(results.get("errors", []))
    if errors:
        log_system(f"Morning Prep Complete: {scraped} scraped, {generated} generated, {errors} errors", status="warning")
    else:
        log_system(f"Morning Prep Complete: {scraped} scraped, {generated} generated", status="success")

    # Send email notification
    try:
        from punty.delivery.email import send_email, format_morning_prep_email
        from punty.models.settings import AppSettings
        from sqlalchemy import select

        # Get notification email from settings
        async with async_session() as db:
            result = await db.execute(select(AppSettings).where(AppSettings.key == "notification_email"))
            setting = result.scalar_one_or_none()
            notification_email = setting.value if setting and setting.value else "punty@punty.ai"

        subject, body_html, body_text = format_morning_prep_email(results)
        email_result = await send_email(
            to_email=notification_email,
            subject=subject,
            body_html=body_html,
            body_text=body_text,
        )
        logger.info(f"Morning prep email notification: {email_result}")
        results["email_sent"] = email_result.get("status") == "sent"
    except Exception as e:
        logger.error(f"Failed to send morning prep email: {e}")
        results["email_sent"] = False
        results["email_error"] = str(e)

    return results


async def scrape_race_cards(
    db: AsyncSession, meeting_id: str, venue: str, race_date: date
) -> dict:
    """Scrape race cards and create/update meeting data."""
    from punty.scrapers.racing_com import RacingComScraper
    from punty.models.meeting import Meeting, Race, Runner
    from sqlalchemy import select
    import uuid

    logger.info(f"Running scrape_race_cards for {venue} on {race_date}")

    scraper = RacingComScraper()
    try:
        data = await scraper.scrape_meeting(venue, race_date)

        # Create or update meeting
        result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
        meeting = result.scalar_one_or_none()

        if not meeting:
            meeting = Meeting(
                id=meeting_id,
                venue=data["meeting"]["venue"],
                date=data["meeting"]["date"],
                track_condition=data["meeting"].get("track_condition"),
                weather=data["meeting"].get("weather"),
                rail_position=data["meeting"].get("rail_position"),
            )
            db.add(meeting)
        else:
            meeting.track_condition = data["meeting"].get("track_condition") or meeting.track_condition
            meeting.weather = data["meeting"].get("weather") or meeting.weather
            meeting.rail_position = data["meeting"].get("rail_position") or meeting.rail_position

        # Create races
        for race_data in data["races"]:
            result = await db.execute(select(Race).where(Race.id == race_data["id"]))
            race = result.scalar_one_or_none()

            if not race:
                race = Race(**race_data)
                db.add(race)
            else:
                for key, value in race_data.items():
                    if key != "id" and value is not None:
                        setattr(race, key, value)

        # Create runners
        for runner_data in data["runners"]:
            result = await db.execute(select(Runner).where(Runner.id == runner_data["id"]))
            runner = result.scalar_one_or_none()

            if not runner:
                runner = Runner(**runner_data)
                db.add(runner)
            else:
                for key, value in runner_data.items():
                    if key != "id" and value is not None:
                        setattr(runner, key, value)

        await db.commit()

        return {
            "status": "success",
            "races_count": len(data["races"]),
            "runners_count": len(data["runners"]),
        }

    finally:
        await scraper.close()


async def scrape_speed_maps(
    db: AsyncSession, meeting_id: str, venue: str, race_date: date
) -> dict:
    """Scrape speed maps and update runner positions."""
    from punty.scrapers.punters import PuntersScraper
    from punty.models.meeting import Runner, Race
    from sqlalchemy import select

    logger.info(f"Running scrape_speed_maps for {venue} on {race_date}")

    scraper = PuntersScraper()
    try:
        data = await scraper.scrape_meeting(venue, race_date)
        speed_maps = data.get("speed_maps", {})
        comments = data.get("comments", {})

        updated_count = 0

        # Get all races for this meeting
        result = await db.execute(
            select(Race).where(Race.meeting_id == meeting_id)
        )
        races = result.scalars().all()

        for race in races:
            race_speed_maps = speed_maps.get(race.race_number, [])
            race_comments = comments.get(race.race_number, {})

            # Get runners for this race
            result = await db.execute(
                select(Runner).where(Runner.race_id == race.id)
            )
            runners = result.scalars().all()

            for runner in runners:
                # Find speed map position
                for sm in race_speed_maps:
                    if sm["horse_name"].lower() == runner.horse_name.lower():
                        if sm.get("speed_map_position"):
                            runner.speed_map_position = sm["speed_map_position"]
                            updated_count += 1
                        break

                # Add comment if available
                comment = race_comments.get(runner.horse_name)
                if comment:
                    runner.comments = comment

        await db.commit()

        return {
            "status": "success",
            "updated_count": updated_count,
        }

    finally:
        await scraper.close()


async def scrape_odds(
    db: AsyncSession, meeting_id: str, venue: str, race_date: date
) -> dict:
    """Scrape live odds and update runner data."""
    from punty.scrapers.tab import TabScraper
    from punty.models.meeting import Runner, Race
    from sqlalchemy import select

    logger.info(f"Running scrape_odds for {venue} on {race_date}")

    scraper = TabScraper()
    try:
        data = await scraper.scrape_meeting(venue, race_date)
        runners_odds = data.get("runners_odds", [])

        updated_count = 0
        scratchings_count = 0

        # Get all races for this meeting
        result = await db.execute(
            select(Race).where(Race.meeting_id == meeting_id)
        )
        races = result.scalars().all()
        races_by_num = {r.race_number: r for r in races}

        for odds_data in runners_odds:
            race = races_by_num.get(odds_data.get("race_number"))
            if not race:
                continue

            # Find runner
            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race.id,
                    Runner.horse_name.ilike(odds_data["horse_name"])
                )
            )
            runner = result.scalar_one_or_none()

            if not runner:
                # Try matching by barrier
                result = await db.execute(
                    select(Runner).where(
                        Runner.race_id == race.id,
                        Runner.barrier == odds_data.get("barrier")
                    )
                )
                runner = result.scalar_one_or_none()

            if runner:
                if odds_data.get("current_odds"):
                    runner.current_odds = odds_data["current_odds"]
                    updated_count += 1
                if odds_data.get("opening_odds") and not runner.opening_odds:
                    runner.opening_odds = odds_data["opening_odds"]
                if odds_data.get("scratched") and not runner.scratched:
                    runner.scratched = True
                    runner.scratching_reason = odds_data.get("scratching_reason")
                    scratchings_count += 1

        await db.commit()

        return {
            "status": "success",
            "updated_count": updated_count,
            "scratchings_count": scratchings_count,
        }

    finally:
        await scraper.close()


async def scrape_results(
    db: AsyncSession, meeting_id: str, venue: str, race_date: date, race_number: Optional[int] = None
) -> dict:
    """Scrape race results."""
    from punty.scrapers.tab import TabScraper
    from punty.models.meeting import Race, Runner, Result
    from sqlalchemy import select
    import uuid

    logger.info(f"Running scrape_results for {venue} on {race_date}")

    scraper = TabScraper()
    try:
        results_data = await scraper.scrape_results(venue, race_date)

        created_count = 0

        # Get races
        query = select(Race).where(Race.meeting_id == meeting_id)
        if race_number:
            query = query.where(Race.race_number == race_number)

        result = await db.execute(query)
        races = result.scalars().all()
        races_by_num = {r.race_number: r for r in races}

        for result_data in results_data:
            race = races_by_num.get(result_data.get("race_number"))
            if not race:
                continue

            # Find runner
            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race.id,
                    Runner.horse_name.ilike(result_data["horse_name"])
                )
            )
            runner = result.scalar_one_or_none()

            if runner:
                # Check if result already exists
                existing = await db.execute(
                    select(Result).where(
                        Result.race_id == race.id,
                        Result.runner_id == runner.id
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                race_result = Result(
                    id=str(uuid.uuid4()),
                    race_id=race.id,
                    runner_id=runner.id,
                    position=result_data["position"],
                    dividend_win=result_data.get("dividend_win"),
                    dividend_place=result_data.get("dividend_place"),
                )
                db.add(race_result)
                created_count += 1

                # Mark race as finished if we have winner
                if result_data["position"] == 1:
                    race.status = "finished"

        await db.commit()

        return {
            "status": "success",
            "created_count": created_count,
        }

    finally:
        await scraper.close()


async def check_context_changes(
    db: AsyncSession, meeting_id: str
) -> dict:
    """Check for significant changes in context that might affect tips."""
    from punty.context.diff import detect_significant_changes
    from punty.context.versioning import create_context_snapshot

    logger.info(f"Checking context changes for {meeting_id}")

    # Create new snapshot
    snapshot = await create_context_snapshot(db, meeting_id)

    if not snapshot:
        return {"status": "no_changes", "changes": []}

    # Detect significant changes
    changes = snapshot.get("significant_changes", [])

    return {
        "status": "changes_detected" if changes else "no_changes",
        "changes": changes,
        "snapshot_version": snapshot.get("version"),
    }


async def meeting_pre_race_job(meeting_id: str) -> dict:
    """Run 2 hours before first race for a meeting.

    This is a lightweight update pass — the heavy scrape was done at 5:00 AM
    by daily_morning_scrape. This job just applies final updates before
    generating the early mail.

    Steps:
    1. Refresh odds + scratchings (TAB)
    2. Refresh track conditions + weather (racingaustralia.horse)
    3. Check jockey/gear changes (racing.com field check)
    4. Refresh speed maps
    5. Generate early mail
    6. Auto-approve if valid
    7. Post to Twitter + Facebook

    Returns: job result dict
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting, Race
    from punty.models.content import Content, ContentStatus
    from punty.config import melb_now
    from punty.scrapers.orchestrator import refresh_odds, scrape_speed_maps_stream
    from punty.ai.generator import ContentGenerator
    from punty.scheduler.activity_log import log_scheduler_job, log_system
    from punty.scheduler.automation import auto_approve_and_post
    from sqlalchemy import select

    logger.info(f"Starting pre-race job for {meeting_id}")
    log_scheduler_job(f"Pre-race job started: {meeting_id}", status="info")

    results = {
        "meeting_id": meeting_id,
        "started_at": melb_now().isoformat(),
        "steps": [],
        "errors": [],
    }

    async with async_session() as db:
        # Get meeting
        result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
        meeting = result.scalar_one_or_none()

        if not meeting:
            results["errors"].append(f"Meeting not found: {meeting_id}")
            return results

        venue = meeting.venue

        # Step 1: Refresh odds + scratchings via TAB (lightweight)
        try:
            logger.info(f"Step 1: Refreshing odds/scratchings for {venue}...")
            await refresh_odds(meeting_id, db)
            results["steps"].append("refresh_odds: success")
        except Exception as e:
            logger.error(f"Odds refresh failed for {venue}: {e}")
            results["errors"].append(f"refresh_odds: {str(e)}")

        # Step 2: Refresh track conditions + weather + scratchings via API
        try:
            logger.info(f"Step 2: Refreshing conditions/weather for {venue}...")
            from punty.scrapers.punting_form import PuntingFormScraper
            from punty.scrapers.orchestrator import _apply_pf_conditions
            from punty.models.meeting import Runner

            pf = await PuntingFormScraper.from_settings(db)
            try:
                # 2a: Conditions/weather
                if not meeting.track_condition_locked:
                    cond = await pf.get_conditions_for_venue(venue)
                    if cond:
                        _apply_pf_conditions(meeting, cond)
                        await db.commit()
                        logger.info(f"Conditions for {venue}: {meeting.track_condition} | rain={cond.get('rainfall')}mm")

                # 2b: Scratchings (catches scratchings TAB may have missed)
                pf_meeting_id = await pf.resolve_meeting_id(venue, meeting.date)
                if pf_meeting_id:
                    scratchings = await pf.get_scratchings_for_meeting(pf_meeting_id)
                    scratch_count = 0
                    for s in scratchings:
                        race_num = s.get("race_number")
                        tab_no = s.get("tab_no")
                        if race_num and tab_no:
                            race_id = f"{meeting_id}-r{race_num}"
                            runner_result = await db.execute(
                                select(Runner).where(
                                    Runner.race_id == race_id,
                                    Runner.saddlecloth == tab_no,
                                    Runner.scratched == False,
                                ).limit(1)
                            )
                            runner = runner_result.scalar_one_or_none()
                            if runner:
                                runner.scratched = True
                                scratch_count += 1
                                logger.info(f"Pre-race scratching: {runner.horse_name} (R{race_num} No.{tab_no})")
                    if scratch_count:
                        await db.commit()
                        logger.info(f"Applied {scratch_count} scratchings for {venue}")
            finally:
                await pf.close()
            results["steps"].append("conditions_refresh: success")
        except Exception as e:
            logger.error(f"Conditions refresh failed for {venue}: {e}")
            results["errors"].append(f"conditions_refresh: {str(e)}")

        # Step 2c: WillyWeather enhanced weather (supplements PF data)
        try:
            from punty.scrapers.willyweather import WillyWeatherScraper
            ww = await WillyWeatherScraper.from_settings(db)
            if ww:
                try:
                    weather_data = await ww.get_weather(venue)
                    if weather_data:
                        if weather_data.get("temp") is not None:
                            meeting.weather_temp = weather_data["temp"]
                        if weather_data.get("wind_speed") is not None:
                            meeting.weather_wind_speed = weather_data["wind_speed"]
                        if weather_data.get("wind_direction"):
                            meeting.weather_wind_dir = weather_data["wind_direction"]
                        if weather_data.get("condition"):
                            meeting.weather_condition = weather_data["condition"]
                        if weather_data.get("humidity") is not None:
                            meeting.weather_humidity = weather_data["humidity"]
                        if weather_data.get("rainfall_chance") is not None:
                            meeting.rainfall = f"{weather_data['rainfall_chance']}% chance, {weather_data['rainfall_amount']}mm"
                        await db.commit()
                        logger.info(
                            f"WillyWeather updated: {venue} — "
                            f"{weather_data.get('condition')}, {weather_data.get('temp')}°C, "
                            f"wind {weather_data.get('wind_speed')}km/h {weather_data.get('wind_direction')}, "
                            f"humidity {weather_data.get('humidity')}%"
                        )
                        results["steps"].append("willyweather: success")
                finally:
                    await ww.close()
        except Exception as e:
            logger.warning(f"WillyWeather refresh failed for {venue}: {e}")
            results["errors"].append(f"willyweather: {str(e)}")

        # Step 3: Check jockey/gear changes via racing.com (Playwright)
        try:
            logger.info(f"Step 3: Checking jockey/gear changes for {venue}...")
            from punty.scrapers.racing_com import RacingComScraper
            from punty.models.meeting import Runner

            # Get all race numbers
            race_result = await db.execute(
                select(Race.race_number).where(Race.meeting_id == meeting_id)
            )
            race_numbers = [r[0] for r in race_result.all()]

            if race_numbers:
                field_scraper = RacingComScraper()
                try:
                    field_data = await field_scraper.check_race_fields(
                        venue, meeting.date, race_numbers
                    )
                    # Apply jockey/gear/scratching updates
                    for race_num, runners in field_data.get("races", {}).items():
                        race_id = f"{meeting_id}-r{race_num}"
                        for r in runners:
                            horse_name = r.get("horse_name")
                            if not horse_name:
                                continue
                            runner_result = await db.execute(
                                select(Runner).where(
                                    Runner.race_id == race_id,
                                    Runner.horse_name == horse_name,
                                ).limit(1)
                            )
                            runner = runner_result.scalar_one_or_none()
                            if not runner:
                                continue
                            if r.get("scratched") and not runner.scratched:
                                runner.scratched = True
                                logger.info(f"Late scratching: {horse_name} (R{race_num})")
                            if r.get("jockey") and r["jockey"] != runner.jockey:
                                logger.info(f"Jockey change: {horse_name} (R{race_num}) {runner.jockey} → {r['jockey']}")
                                runner.jockey = r["jockey"]
                            if r.get("gear") and r["gear"] != runner.gear:
                                runner.gear = r["gear"]
                            if r.get("gear_changes") and r["gear_changes"] != runner.gear_changes:
                                logger.info(f"Gear change: {horse_name} (R{race_num}) {r['gear_changes']}")
                                runner.gear_changes = r["gear_changes"]

                    # Update track condition if changed
                    tc = field_data.get("meeting", {}).get("track_condition")
                    if tc and not meeting.track_condition_locked:
                        from punty.scrapers.orchestrator import _is_more_specific, _normalise_track
                        if _normalise_track(tc) != _normalise_track(meeting.track_condition or ""):
                            if _is_more_specific(tc, meeting.track_condition):
                                logger.info(f"Track condition update: {meeting.track_condition!r} → {tc!r}")
                                meeting.track_condition = tc

                    await db.commit()
                finally:
                    await field_scraper.close()
            results["steps"].append("jockey_gear_check: success")
        except Exception as e:
            logger.error(f"Jockey/gear check failed for {venue}: {e}")
            results["errors"].append(f"jockey_gear_check: {str(e)}")

        # Step 4: Refresh speed maps
        try:
            logger.info(f"Step 4: Refreshing speed maps for {venue}...")
            async for _ in scrape_speed_maps_stream(meeting_id, db):
                pass
            results["steps"].append("refresh_speed_maps: success")
        except Exception as e:
            logger.error(f"Speed maps failed for {venue}: {e}")
            results["errors"].append(f"refresh_speed_maps: {str(e)}")

        # Step 5: Generate early mail
        content_id = None
        try:
            logger.info(f"Step 5: Generating early mail for {venue}...")
            generator = ContentGenerator(db)

            async for event in generator.generate_early_mail_stream(meeting_id):
                if event.get("status") == "error":
                    raise Exception(event.get("label", "Unknown error"))
                # content_id is nested inside event["result"]
                if event.get("content_id"):
                    content_id = event.get("content_id")
                elif event.get("result", {}).get("content_id"):
                    content_id = event["result"]["content_id"]

            results["steps"].append("generate_early_mail: success")
            results["content_id"] = content_id
        except Exception as e:
            logger.error(f"Early mail generation failed for {venue}: {e}")
            results["errors"].append(f"generate_early_mail: {str(e)}")

            # If rate limited, note it
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit detected for {venue}")
                results["rate_limited"] = True

        # Step 6 & 7: Auto-approve and post to Twitter + Facebook
        if content_id:
            try:
                logger.info(f"Step 6-7: Auto-approving and posting for {venue}...")
                post_result = await auto_approve_and_post(content_id, db)
                results["steps"].append(f"auto_approve_post: {post_result.get('status')}")
                results["post_result"] = post_result
            except Exception as e:
                logger.error(f"Auto-approve/post failed for {venue}: {e}")
                results["errors"].append(f"auto_approve_post: {str(e)}")

    results["completed_at"] = melb_now().isoformat()

    # Log completion
    if results["errors"]:
        log_system(f"Pre-race job completed with errors: {venue}", status="warning")
    else:
        log_system(f"Pre-race job completed: {venue}", status="success")

    logger.info(f"Pre-race job complete for {meeting_id}: {results}")
    return results


async def meeting_post_race_job(meeting_id: str, retry_count: int = 0, job_started_at: str | None = None) -> dict:
    """Run 30 minutes after last race for a meeting.

    Steps:
    1. Check all picks are settled
    2. If not settled, reschedule retry (max 3 retries, max 2h total)
    3. Generate wrap-up
    4. Auto-approve if valid
    5. Post to Twitter
    6. Log activity

    Returns: job result dict
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting
    from punty.config import melb_now
    from punty.ai.generator import ContentGenerator
    from punty.scheduler.activity_log import log_scheduler_job, log_system
    from punty.scheduler.automation import auto_approve_and_post, check_all_settled
    from sqlalchemy import select

    MAX_RETRIES = 3
    RETRY_DELAY_MINUTES = 10
    MAX_ELAPSED_HOURS = 2

    # Track job start time across retries
    from datetime import datetime as _dt
    if job_started_at is None:
        job_started_at = melb_now().isoformat()
    first_start = _dt.fromisoformat(job_started_at)
    elapsed_hours = (melb_now() - first_start).total_seconds() / 3600

    logger.info(f"Starting post-race job for {meeting_id} (retry: {retry_count}, elapsed: {elapsed_hours:.1f}h)")
    log_scheduler_job(f"Post-race job started: {meeting_id}", status="info")

    results = {
        "meeting_id": meeting_id,
        "started_at": melb_now().isoformat(),
        "retry_count": retry_count,
        "steps": [],
        "errors": [],
    }

    async with async_session() as db:
        # Get meeting
        result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
        meeting = result.scalar_one_or_none()

        if not meeting:
            results["errors"].append(f"Meeting not found: {meeting_id}")
            return results

        venue = meeting.venue

        # Step 1: Check all picks are settled
        all_settled, settled_count, total_count = await check_all_settled(meeting_id, db)
        results["settlement"] = {
            "all_settled": all_settled,
            "settled": settled_count,
            "total": total_count,
        }

        if not all_settled and retry_count < MAX_RETRIES and elapsed_hours < MAX_ELAPSED_HOURS:
            logger.info(f"Not all picks settled ({settled_count}/{total_count}), scheduling retry...")
            results["steps"].append(f"settlement_check: incomplete ({settled_count}/{total_count})")
            results["retry_scheduled"] = True

            # Schedule retry, passing job_started_at to track total elapsed time
            from punty.scheduler.manager import scheduler_manager
            from datetime import timedelta

            _started = job_started_at
            retry_time = melb_now() + timedelta(minutes=RETRY_DELAY_MINUTES)
            scheduler_manager.add_job(
                f"{meeting_id}-post-race-retry-{retry_count + 1}",
                lambda: meeting_post_race_job(meeting_id, retry_count + 1, job_started_at=_started),
                trigger_type="date",
                run_date=retry_time,
            )

            log_system(f"Settlement incomplete, retry scheduled: {venue}", status="info")
            return results

        if not all_settled:
            logger.warning(f"Settlement timeout ({elapsed_hours:.1f}h elapsed or max retries). Proceeding with wrap-up anyway.")
            results["steps"].append(f"settlement_timeout: proceeding ({settled_count}/{total_count} settled)")

        if not all_settled:
            logger.warning(f"Max retries reached, proceeding with unsettled picks: {meeting_id}")
            results["steps"].append("settlement_check: max_retries_reached")

        # Step 2: Generate wrap-up
        content_id = None
        try:
            logger.info(f"Step 2: Generating wrap-up for {venue}...")
            generator = ContentGenerator(db)

            async for event in generator.generate_meeting_wrapup_stream(meeting_id):
                if event.get("status") == "error":
                    raise Exception(event.get("label", "Unknown error"))
                if event.get("content_id"):
                    content_id = event.get("content_id")

            results["steps"].append("generate_wrapup: success")
            results["content_id"] = content_id
        except Exception as e:
            logger.error(f"Wrap-up generation failed for {venue}: {e}")
            results["errors"].append(f"generate_wrapup: {str(e)}")

        # Step 3 & 4: Auto-approve and post to Twitter
        if content_id:
            try:
                logger.info(f"Step 3-4: Auto-approving and posting wrap-up for {venue}...")
                post_result = await auto_approve_and_post(content_id, db)
                results["steps"].append(f"auto_approve_post: {post_result.get('status')}")
                results["post_result"] = post_result
            except Exception as e:
                logger.error(f"Auto-approve/post failed for {venue}: {e}")
                results["errors"].append(f"auto_approve_post: {str(e)}")

    results["completed_at"] = melb_now().isoformat()

    # Log completion
    if results["errors"]:
        log_system(f"Post-race job completed with errors: {venue}", status="warning")
    else:
        log_system(f"Post-race job completed: {venue}", status="success")

    logger.info(f"Post-race job complete for {meeting_id}: {results}")
    return results
