"""Job definitions for scheduled tasks."""

import asyncio
import logging
import random
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

from sqlalchemy import select
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
        run_date=next_hour,
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
            from punty.venues import venue_slug
            for m in meetings_data:
                meeting_id = f"{venue_slug(m['venue'])}-{today.isoformat()}"
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
            # Also clear HKJC ranking cache for fresh daily stats
            try:
                from punty.scrapers.tab_playwright import _hkjc_ranking_cache
                _hkjc_ranking_cache.clear()
            except Exception:
                pass

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

        # Step 1b: Quality gate — deselect meetings with poor data quality
        from punty.models.meeting import Runner
        try:
            for meeting in meetings:
                if meeting.venue not in results["meetings_scraped"]:
                    continue

                # Check 1: No races at all (abandoned/cancelled)
                race_result = await db.execute(
                    select(Race).where(Race.meeting_id == meeting.id)
                )
                races = race_result.scalars().all()
                if not races:
                    meeting.selected = False
                    logger.info(f"Deselected {meeting.venue} — no races found after morning scrape")
                    continue

                # Check 2: Too few races (< 4 = picnic/novelty meeting)
                if len(races) < 4:
                    meeting.selected = False
                    logger.info(
                        f"Deselected {meeting.venue} — only {len(races)} races "
                        f"(minimum 4 for quality gate)"
                    )
                    continue

                # Check 3: Low odds coverage (< 50% runners with odds)
                total_runners = 0
                runners_with_odds = 0
                for race in races:
                    runner_result = await db.execute(
                        select(Runner).where(
                            Runner.race_id == race.id,
                            Runner.scratched != True,
                        )
                    )
                    race_runners = runner_result.scalars().all()
                    total_runners += len(race_runners)
                    runners_with_odds += sum(
                        1 for r in race_runners
                        if (r.current_odds and r.current_odds > 1.0)
                        or (r.odds_betfair and r.odds_betfair > 1.0)
                    )

                if total_runners > 0:
                    coverage = runners_with_odds / total_runners
                    if coverage < 0.5:
                        meeting.selected = False
                        logger.warning(
                            f"Deselected {meeting.venue} — odds coverage "
                            f"{coverage:.0%} ({runners_with_odds}/{total_runners} runners). "
                            f"No bookmaker coverage for this venue."
                        )
                        continue

            await db.commit()
        except Exception as e:
            logger.error(f"Quality gate check failed: {e}")

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

    # Step 3: Generate early mail for all meetings (approve only, no social post)
    try:
        gen_results = await morning_generate_all()
        results["generation"] = gen_results
    except Exception as e:
        logger.error(f"Morning generation failed: {e}")
        results["errors"].append(f"morning_generate: {str(e)}")

    results["completed_at"] = melb_now().isoformat()

    # Log completion
    scraped = len(results["meetings_scraped"])
    speed_maps = len(results["speed_maps_done"])
    generated = len(results.get("generation", {}).get("generated", []))
    errors = len(results["errors"])
    if errors:
        log_system(
            f"Morning scrape complete: {scraped} scraped, {speed_maps} speed maps, "
            f"{generated} generated, {errors} errors",
            status="warning",
        )
    else:
        log_system(
            f"Morning scrape complete: {scraped} scraped, {speed_maps} speed maps, "
            f"{generated} generated",
            status="success",
        )

    logger.info(f"Daily morning scrape complete: {results}")
    return results


async def morning_generate_all() -> dict:
    """Generate and approve early mail for all selected meetings today.

    Called at the end of the morning scrape (~05:30). Content is approved
    but NOT posted to socials — the pre-race job handles posting later,
    regenerating only if material changes are detected.

    Returns: summary dict with generated/skipped/failed lists
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting
    from punty.config import melb_today, melb_now
    from punty.ai.generator import ContentGenerator
    from punty.context.versioning import create_context_snapshot
    from punty.scheduler.automation import validate_meeting_readiness, auto_approve_content
    from punty.scheduler.activity_log import log_system
    from sqlalchemy import select, or_

    results = {
        "started_at": melb_now().isoformat(),
        "generated": [],
        "skipped": [],
        "failed": [],
    }

    async with async_session() as db:
        # Get all selected meetings for today
        result = await db.execute(
            select(Meeting).where(
                Meeting.date == melb_today(),
                Meeting.selected == True,
                or_(Meeting.meeting_type == None, Meeting.meeting_type == "race"),
            ).order_by(Meeting.venue)
        )
        meetings = result.scalars().all()

        if not meetings:
            logger.info("No selected meetings for morning generation")
            return results

        logger.info(f"Morning generation starting for {len(meetings)} meetings")

        for meeting in meetings:
            venue = meeting.venue
            meeting_id = meeting.id

            try:
                # Check readiness
                is_ready, issues = await validate_meeting_readiness(meeting_id, db)
                if not is_ready:
                    logger.warning(f"Skipping {venue} — not ready: {issues}")
                    results["skipped"].append({"venue": venue, "reason": "; ".join(issues)})
                    continue

                # Create baseline context snapshot for later comparison
                await create_context_snapshot(db, meeting_id, force=True)

                # Generate early mail
                content_id = None
                generator = ContentGenerator(db)
                async for event in generator.generate_early_mail_stream(meeting_id):
                    if event.get("status") == "error":
                        raise Exception(event.get("label", "Unknown error"))
                    if event.get("content_id"):
                        content_id = event.get("content_id")
                    elif event.get("result", {}).get("content_id"):
                        content_id = event["result"]["content_id"]

                if not content_id:
                    raise Exception("No content_id returned from generation")

                # Auto-approve (no social post)
                approve_result = await auto_approve_content(content_id, db)
                if approve_result["status"] != "approved":
                    logger.warning(f"Morning approval failed for {venue}: {approve_result}")
                    results["failed"].append({
                        "venue": venue,
                        "reason": f"approval: {approve_result.get('issues', [])}",
                    })
                    continue

                results["generated"].append(venue)
                logger.info(f"Morning generation complete for {venue}: {content_id}")

            except Exception as e:
                logger.error(f"Morning generation failed for {venue}: {e}")
                results["failed"].append({"venue": venue, "reason": str(e)})

                # Rate limit handling: pause before next meeting
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    logger.warning(f"Rate limited — pausing {RATE_LIMIT_QUEUE_PAUSE}s before next meeting")
                    await asyncio.sleep(RATE_LIMIT_QUEUE_PAUSE)

    results["completed_at"] = melb_now().isoformat()
    gen_count = len(results["generated"])
    skip_count = len(results["skipped"])
    fail_count = len(results["failed"])
    logger.info(f"Morning generation complete: {gen_count} generated, {skip_count} skipped, {fail_count} failed")

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
    """Scrape live odds and update runner data via racing.com (multi-bookmaker)."""
    from punty.scrapers.orchestrator import refresh_odds

    logger.info(f"Running scrape_odds for {venue} on {race_date}")
    result = await refresh_odds(meeting_id, db)
    return {"status": result.get("status", "error")}


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
    """Run 90 minutes before first race for a meeting.

    Two-phase flow: morning_generate_all() already generated + approved
    content at ~05:30. This job refreshes data, compares snapshots, and
    either posts the existing content or regenerates if material changes
    (scratchings, track condition, jockey/gear, speed map flips) are detected.

    Steps:
    1. Refresh odds + scratchings (TAB)
    2. Refresh track conditions + weather
    3. Check jockey/gear changes (racing.com field check)
    4. Refresh speed maps
    5. Check for existing morning content
    6. Compare context snapshots — detect material changes
    7. If material changes: regenerate → approve → post
       If no changes: post existing content
       If no morning content: full generate → approve → post (fallback)

    Returns: job result dict
    """
    from punty.models.database import async_session
    from punty.models.meeting import Meeting, Race
    from punty.models.content import Content, ContentStatus
    from punty.config import melb_now
    from punty.scrapers.orchestrator import refresh_odds, scrape_speed_maps_stream
    from punty.ai.generator import ContentGenerator
    from punty.scheduler.activity_log import log_scheduler_job, log_system
    from punty.scheduler.automation import auto_approve_and_post, post_existing_content
    from punty.context.versioning import create_context_snapshot
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
            from punty.scrapers.orchestrator import refresh_track_conditions
            from punty.models.meeting import Runner

            pf = await PuntingFormScraper.from_settings(db)
            try:
                # 2a: Conditions/weather — single gatekeeper
                pf_cond = await pf.get_conditions_for_venue(venue)
                await refresh_track_conditions(meeting, pf_cond=pf_cond, source="pre_race")
                await db.commit()

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
                    # Apply jockey/gear/scratching/odds updates
                    odds_updated = 0
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
                            # Always apply fresh odds (overnight odds can be wildly stale)
                            odds_data = r.get("odds")
                            if odds_data:
                                best_odds = (
                                    odds_data.get("odds_betfair")
                                    or odds_data.get("odds_tab")
                                    or odds_data.get("odds_sportsbet")
                                    or odds_data.get("odds_bet365")
                                    or odds_data.get("odds_ladbrokes")
                                )
                                if best_odds:
                                    runner.current_odds = best_odds
                                    if not runner.opening_odds:
                                        runner.opening_odds = best_odds
                                    odds_updated += 1
                                for field in ("odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes", "odds_betfair"):
                                    val = odds_data.get(field)
                                    if val:
                                        setattr(runner, field, val)
                                if odds_data.get("place_odds"):
                                    runner.place_odds = odds_data["place_odds"]
                                if odds_data.get("odds_flucs") and not runner.odds_flucs:
                                    runner.odds_flucs = odds_data["odds_flucs"]
                    if odds_updated:
                        logger.info(f"Updated odds for {odds_updated} runners in {venue}")

                    # Track condition NOT updated here — handled by refresh_track_conditions() gatekeeper
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

        # Step 4b: Validate meeting readiness before generation
        from punty.scheduler.automation import validate_meeting_readiness
        is_ready, readiness_issues = await validate_meeting_readiness(meeting_id, db)
        if not is_ready:
            logger.warning(f"Meeting {venue} failed readiness check: {readiness_issues}")
            log_system(f"Skipping {venue} — failed readiness: {'; '.join(readiness_issues)}", status="warning")
            results["errors"].append(f"readiness_check: {'; '.join(readiness_issues)}")
            results["skipped"] = True
            results["completed_at"] = melb_now().isoformat()
            return results

        results["steps"].append("readiness_check: passed")

        # Step 5: Check for existing morning-generated content
        existing_result = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting_id,
                Content.content_type == "early_mail",
                Content.status.in_(["approved", "sent"]),
            ).order_by(Content.created_at.desc()).limit(1)
        )
        morning_content = existing_result.scalar_one_or_none()

        if morning_content:
            # Step 6: Create fresh snapshot and compare to morning baseline
            needs_regen = False
            try:
                snapshot = await create_context_snapshot(db, meeting_id, force=True)
                if snapshot and snapshot.get("significant_changes"):
                    MATERIAL_TYPES = {
                        "scratching", "track_condition", "speed_map_change",
                        "jockey_change", "gear_change",
                    }
                    material = [
                        c for c in snapshot["significant_changes"]
                        if c["type"] in MATERIAL_TYPES
                    ]
                    if material:
                        needs_regen = True
                        descriptions = [c["description"] for c in material]
                        logger.info(f"Material changes for {venue}: {descriptions}")
                        results["steps"].append(f"snapshot_compare: material changes ({len(material)})")
                    else:
                        results["steps"].append("snapshot_compare: no material changes")
                else:
                    results["steps"].append("snapshot_compare: no changes")
            except Exception as e:
                logger.error(f"Snapshot comparison failed for {venue}: {e}")
                results["errors"].append(f"snapshot_compare: {str(e)}")
                # On snapshot failure, fall through to regen for safety
                needs_regen = True

            if needs_regen:
                # Step 7a: Regenerate — material changes detected
                logger.info(f"Regenerating early mail for {venue} due to material changes...")
                content_id = None
                try:
                    generator = ContentGenerator(db)
                    async for event in generator.generate_early_mail_stream(meeting_id):
                        if event.get("status") == "error":
                            raise Exception(event.get("label", "Unknown error"))
                        if event.get("content_id"):
                            content_id = event.get("content_id")
                        elif event.get("result", {}).get("content_id"):
                            content_id = event["result"]["content_id"]

                    if content_id:
                        post_result = await auto_approve_and_post(content_id, db)
                        results["steps"].append(f"regen_approve_post: {post_result.get('status')}")
                        results["post_result"] = post_result
                        results["content_id"] = content_id
                    else:
                        results["errors"].append("regen: no content_id returned")
                except Exception as e:
                    logger.error(f"Regeneration failed for {venue}: {e}")
                    results["errors"].append(f"regen: {str(e)}")
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "429" in error_str:
                        results["rate_limited"] = True
            else:
                # Step 7b: No material changes — post existing content to socials
                logger.info(f"No material changes for {venue} — posting existing content")
                try:
                    post_result = await post_existing_content(morning_content.id, db)
                    results["steps"].append(f"post_existing: {post_result.get('status')}")
                    results["post_result"] = post_result
                    results["content_id"] = morning_content.id
                except Exception as e:
                    logger.error(f"Post existing failed for {venue}: {e}")
                    results["errors"].append(f"post_existing: {str(e)}")
        else:
            # No morning content: full generate → approve → post (fallback)
            logger.info(f"No morning content for {venue} — full generation")
            content_id = None
            try:
                generator = ContentGenerator(db)
                async for event in generator.generate_early_mail_stream(meeting_id):
                    if event.get("status") == "error":
                        raise Exception(event.get("label", "Unknown error"))
                    if event.get("content_id"):
                        content_id = event.get("content_id")
                    elif event.get("result", {}).get("content_id"):
                        content_id = event["result"]["content_id"]

                results["steps"].append("generate_early_mail: success")
                results["content_id"] = content_id
            except Exception as e:
                logger.error(f"Early mail generation failed for {venue}: {e}")
                results["errors"].append(f"generate_early_mail: {str(e)}")
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    results["rate_limited"] = True

            if content_id:
                try:
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
    from punty.models.content import Content
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
            from functools import partial
            from punty.scheduler.manager import scheduler_manager
            from datetime import timedelta

            retry_time = melb_now() + timedelta(minutes=RETRY_DELAY_MINUTES)
            scheduler_manager.add_job(
                f"{meeting_id}-post-race-retry-{retry_count + 1}",
                partial(meeting_post_race_job, meeting_id, retry_count + 1, job_started_at=job_started_at),
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

        # Step 2: Generate wrap-up (skip if one already exists)
        content_id = None
        existing_wrapup = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting_id,
                Content.content_type == "meeting_wrapup",
                Content.status.notin_(["rejected", "superseded"]),
            )
        )
        existing_wrap = existing_wrapup.scalars().first()
        if existing_wrap:
            logger.info(f"Wrap-up already exists for {venue} — skipping generation")
            results["steps"].append("generate_wrapup: skipped (already exists)")
            # If wrap exists but hasn't been approved yet, try auto-approving it
            if existing_wrap.status in ("pending_review", "draft"):
                content_id = existing_wrap.id
                logger.info(f"Existing wrap-up for {venue} is {existing_wrap.status} — will attempt auto-approve")
        else:
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

        # Step 5: Auto-tune probability weights
        try:
            from punty.probability_tuning import maybe_tune_weights
            tune_result = await maybe_tune_weights(db)
            if tune_result:
                results["steps"].append(f"probability_tune: adjusted {tune_result['max_change_pct']:.1f}%")
                await db.commit()
            else:
                results["steps"].append("probability_tune: skipped (cooldown/insufficient data)")
        except Exception as e:
            logger.warning(f"Probability tuning failed: {e}")
            results["errors"].append(f"probability_tune: {str(e)}")

        # Step 6: Auto-tune bet type thresholds
        try:
            from punty.bet_type_tuning import maybe_tune_bet_thresholds
            bt_result = await maybe_tune_bet_thresholds(db)
            if bt_result:
                results["steps"].append(
                    f"bet_type_tune: adjusted {bt_result['max_change_key']} "
                    f"({bt_result['max_change_pct']:.1f}%, {bt_result['picks_analyzed']} picks)"
                )
            else:
                results["steps"].append("bet_type_tune: skipped (cooldown/insufficient data)")
        except Exception as e:
            logger.warning(f"Bet type tuning failed: {e}")
            results["errors"].append(f"bet_type_tune: {str(e)}")

    results["completed_at"] = melb_now().isoformat()

    # Log completion
    if results["errors"]:
        log_system(f"Post-race job completed with errors: {venue}", status="warning")
    else:
        log_system(f"Post-race job completed: {venue}", status="success")

    logger.info(f"Post-race job complete for {meeting_id}: {results}")
    return results


async def weekly_pattern_refresh() -> dict:
    """Thursday night pattern refresh — builds all data needed for Friday blog.

    Runs: patterns, awards, ledger, future nominations, news headlines.
    Stores results in AppSettings for the blog context builder.
    """
    from punty.models.database import async_session
    from punty.models.settings import AppSettings
    from punty.scheduler.activity_log import log_system
    import json

    results = {"steps": [], "errors": []}
    log_system("Starting weekly pattern refresh", status="info")

    async with async_session() as db:
        # Step 1: Deep pattern analysis
        try:
            from punty.patterns.engine import run_deep_pattern_analysis
            patterns = await run_deep_pattern_analysis(db)
            # Store as JSON in AppSettings
            await _upsert_setting(db, "weekly_patterns", json.dumps(patterns))
            results["steps"].append(f"patterns: {len(patterns)} dimensions")
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            results["errors"].append(f"patterns: {e}")

        # Step 2: Weekly awards
        try:
            from punty.patterns.awards import compute_weekly_awards
            awards = await compute_weekly_awards(db)
            await _upsert_setting(db, "weekly_awards", json.dumps(awards))
            results["steps"].append(f"awards: {len(awards)} categories")
        except Exception as e:
            logger.error(f"Awards computation failed: {e}")
            results["errors"].append(f"awards: {e}")

        # Step 3: Weekly ledger
        try:
            from punty.patterns.weekly_summary import build_weekly_ledger
            ledger = await build_weekly_ledger(db)
            await _upsert_setting(db, "weekly_ledger", json.dumps(ledger))
            results["steps"].append(f"ledger: P&L ${ledger.get('this_week', {}).get('total_pnl', 0):+.2f}")
        except Exception as e:
            logger.error(f"Ledger computation failed: {e}")
            results["errors"].append(f"ledger: {e}")

        # Step 4: Future nominations
        try:
            from punty.scrapers.future_races import scrape_future_group_races
            future = await scrape_future_group_races(db)
            results["steps"].append(f"future_races: {future.get('group_races_found', 0)} races")
        except Exception as e:
            logger.error(f"Future races scrape failed: {e}")
            results["errors"].append(f"future_races: {e}")

        # Step 5: News headlines
        try:
            from punty.scrapers.news import NewsScraper
            scraper = NewsScraper()
            headlines = await scraper.scrape_headlines()
            await _upsert_setting(db, "recent_news_headlines", json.dumps(headlines))
            results["steps"].append(f"news: {len(headlines)} headlines")
        except Exception as e:
            logger.error(f"News scrape failed: {e}")
            results["errors"].append(f"news: {e}")

        await db.commit()

    if results["errors"]:
        log_system(f"Pattern refresh completed with errors: {results['errors']}", status="warning")
    else:
        log_system("Pattern refresh complete", status="success")

    logger.info(f"Weekly pattern refresh complete: {results}")
    return results


async def weekly_blog_job() -> dict:
    """Friday morning blog generation job.

    1. Check pattern data freshness (run inline if stale)
    2. Generate blog via AI
    3. Validate and auto-approve
    4. Post teaser to Twitter + Facebook
    """
    from punty.models.database import async_session
    from punty.models.settings import AppSettings
    from punty.scheduler.activity_log import log_system
    from punty.ai.generator import ContentGenerator
    from punty.scheduler.automation import auto_approve_and_post

    results = {"steps": [], "errors": []}
    log_system("Starting weekly blog generation", status="info")

    async with async_session() as db:
        # Step 1: Check pattern data freshness
        result = await db.execute(
            select(AppSettings).where(AppSettings.key == "weekly_patterns")
        )
        setting = result.scalar_one_or_none()
        if not setting or not setting.value:
            logger.info("Pattern data stale — running inline refresh")
            results["steps"].append("pattern_refresh: inline")
            await weekly_pattern_refresh()
        else:
            results["steps"].append("pattern_data: fresh")

    # Step 2: Generate blog (needs fresh db session)
    content_id = None
    async with async_session() as db:
        try:
            generator = ContentGenerator(db)
            async for event in generator.generate_weekly_blog_stream():
                if event.get("status") == "error":
                    raise Exception(event.get("label", "Unknown error"))
                if event.get("result", {}).get("content_id"):
                    content_id = event["result"]["content_id"]

            results["steps"].append("generate_blog: success")
            results["content_id"] = content_id
        except Exception as e:
            logger.error(f"Blog generation failed: {e}")
            results["errors"].append(f"generate_blog: {e}")

    # Step 3-4: Auto-approve and post
    if content_id:
        async with async_session() as db:
            try:
                post_result = await auto_approve_and_post(content_id, db)
                results["steps"].append(f"auto_approve_post: {post_result.get('status')}")
                results["post_result"] = post_result
            except Exception as e:
                logger.error(f"Blog auto-approve/post failed: {e}")
                results["errors"].append(f"auto_approve_post: {e}")

    if results["errors"]:
        log_system(f"Blog job completed with errors: {results['errors']}", status="warning")
    else:
        log_system("Weekly blog published", status="success")

    logger.info(f"Weekly blog job complete: {results}")
    return results


async def _upsert_setting(db, key: str, value: str):
    """Upsert an AppSettings row."""
    from punty.models.settings import AppSettings

    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = value
    else:
        db.add(AppSettings(key=key, value=value))
    await db.flush()
