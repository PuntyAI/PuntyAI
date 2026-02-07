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


async def daily_morning_prep() -> dict:
    """
    Daily morning preparation job that runs between 7:30-8:30 AM Melbourne time.

    This job:
    1. Scrapes the racing calendar to get today's meetings
    2. Auto-selects metro meetings
    3. Scrapes all data (form + speed maps) for selected meetings
    4. Generates early mail for all selected meetings
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
