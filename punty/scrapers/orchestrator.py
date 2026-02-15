"""Scraping orchestrator — coordinates scrapers and stores results."""

import json as _json
import logging
from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.meeting import Meeting, Race, Runner

logger = logging.getLogger(__name__)


def _normalise_track(cond: str | None) -> str:
    """Normalise track condition for comparison (prevents false change alerts)."""
    if not cond:
        return ""
    import re
    return re.sub(r"\s+", " ", cond.strip().lower().replace("(", "").replace(")", ""))


def _is_more_specific(new_cond: str | None, old_cond: str | None) -> bool:
    """Return True if new condition is more specific (has rating number) than old.

    Never overwrites a rated condition (e.g. "Good 4") with a different
    base category (e.g. "Soft 5") — protects against stale PF data
    overriding accurate TAB/Racing.com conditions.
    """
    import re
    if not new_cond:
        return False
    if not old_cond:
        return True
    new_has_rating = bool(re.search(r"\d", new_cond))
    old_has_rating = bool(re.search(r"\d", old_cond))
    # New has rating, old doesn't → more specific
    if new_has_rating and not old_has_rating:
        return True
    # Old has rating, new doesn't → keep old
    if not new_has_rating and old_has_rating:
        return False
    # Both have ratings — only update if same base category
    # (don't let "Soft 5" overwrite "Good 4")
    if new_has_rating and old_has_rating:
        new_base = new_cond.strip().split()[0].lower()
        old_base = old_cond.strip().split()[0].lower()
        if new_base != old_base:
            return False
    # Same base or neither has rating — newer is preferred
    return True


# All Runner fields that come from the scraper
RUNNER_FIELDS = [
    "saddlecloth", "barrier", "weight", "jockey", "trainer", "form", "current_odds", "opening_odds", "place_odds",
    "scratched", "comments", "horse_age", "horse_sex", "horse_colour",
    "sire", "dam", "dam_sire", "career_prize_money", "last_five",
    "days_since_last_run", "handicap_rating", "speed_value",
    "track_dist_stats", "track_stats", "distance_stats",
    "first_up_stats", "second_up_stats", "good_track_stats",
    "soft_track_stats", "heavy_track_stats", "jockey_stats", "trainer_stats", "class_stats",
    "gear", "gear_changes", "stewards_comment", "comment_long", "comment_short",
    "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes",
    "odds_betfair", "odds_flucs", "trainer_location", "form_history",
    # Pace analysis insights
    "pf_speed_rank", "pf_settle", "pf_map_factor", "pf_jockey_factor",
]

# Race fields from scraper
RACE_FIELDS = [
    "name", "distance", "class_", "prize_money", "start_time",
    "track_condition", "race_type", "age_restriction", "weight_type", "field_size",
]

# Meeting fields from scraper
MEETING_FIELDS = [
    "track_condition", "weather", "rail_position",
    "penetrometer", "weather_condition", "weather_temp",
    "weather_wind_speed", "weather_wind_dir", "weather_humidity", "rail_bias_comment",
    "rainfall", "irrigation", "going_stick",
]

# Fields where Racing Australia is authoritative — always overwrite PF data
_RA_AUTH_RUNNER_FIELDS = ["scratched", "barrier", "weight", "jockey"]
_RA_AUTH_RACE_FIELDS = ["distance", "class_", "field_size", "start_time"]


async def scrape_calendar(db: AsyncSession) -> list[dict]:
    """Scrape today's calendar and populate meetings in DB."""
    from punty.scrapers.calendar import scrape_calendar as _scrape

    today = melb_today()
    raw_meetings = await _scrape(today)
    results = []

    for m in raw_meetings:
        venue = m["venue"]
        venue_slug = venue.lower().replace(" ", "-")
        meeting_id = f"{venue_slug}-{today.isoformat()}"

        existing = await db.get(Meeting, meeting_id)
        if existing:
            if not existing.source:
                existing.source = "racing.com/calendar"
            # Update meeting_type if calendar detected it as trial/jumpout
            if m.get("meeting_type") in ("trial", "jumpout"):
                existing.meeting_type = m["meeting_type"]
            results.append(existing.to_dict())
            continue

        # Use meeting_type from calendar data if available, otherwise classify by venue name
        meeting_type = m.get("meeting_type") or _classify_meeting_type(venue)
        meeting = Meeting(
            id=meeting_id,
            venue=venue,
            date=today,
            selected=False,
            source="racing.com/calendar",
            meeting_type=meeting_type,
        )
        db.add(meeting)
        results.append({
            "id": meeting_id,
            "venue": venue,
            "date": today.isoformat(),
            "state": m.get("state", ""),
            "num_races": m.get("num_races", 0),
            "selected": False,
            "source": "racing.com/calendar",
            "meeting_type": meeting_type,
        })

    await db.commit()
    logger.info(f"Calendar scrape complete: {len(results)} meetings")
    return results


async def scrape_meeting_fields_only(meeting_id: str, db: AsyncSession, pf_scraper=None) -> dict:
    """Lightweight scrape: fields + conditions only. No form history, scratchings, or racing.com.

    Used by midnight calendar scrape where we only need race times and basic
    runner data for scheduling automation. The 5am morning scrape does the full dump.

    Args:
        pf_scraper: Optional shared PuntingFormScraper instance (avoids re-creating per meeting).
    """
    from punty.scheduler.activity_log import log_scrape_start, log_scrape_complete, log_scrape_error

    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    venue = meeting.venue
    race_date = meeting.date
    errors = []
    owns_pf = pf_scraper is None

    log_scrape_start(venue)

    try:
        if pf_scraper is None:
            from punty.scrapers.punting_form import PuntingFormScraper
            pf_scraper = await PuntingFormScraper.from_settings(db)

        data = await pf_scraper.scrape_meeting_fields_only(venue, race_date)
        await _upsert_meeting_data(db, meeting, data)
    except Exception as e:
        logger.error(f"Fields-only scrape failed for {venue}: {e}")
        errors.append(f"fields_only: {e}")
        # Fallback 1: Racing Australia free fields (httpx, fast)
        try:
            from punty.scrapers.ra_fields import scrape_ra_fields
            data = await scrape_ra_fields(venue, race_date, meeting_id)
            if data:
                await _upsert_meeting_data(db, meeting, data)
                logger.info(f"RA fields fallback succeeded for {venue}")
            else:
                raise ValueError("RA fields returned no data")
        except Exception as e_ra:
            logger.error(f"RA fields fallback failed for {venue}: {e_ra}")
            errors.append(f"ra_fields_fallback: {e_ra}")
            # Fallback 2: racing.com (Playwright, slower)
            try:
                from punty.scrapers.racing_com import RacingComScraper
                scraper = RacingComScraper()
                try:
                    data = await scraper.scrape_meeting(venue, race_date)
                    await _upsert_meeting_data(db, meeting, data)
                finally:
                    await scraper.close()
            except Exception as e2:
                logger.error(f"racing.com fallback also failed: {e2}")
                errors.append(f"racing.com_fallback: {e2}")

    # Conditions (uses session-level cache — one API call for all meetings)
    if not meeting.track_condition_locked:
        try:
            cond = await pf_scraper.get_conditions_for_venue(venue)
            if cond:
                _apply_pf_conditions(meeting, cond)
        except Exception as e:
            logger.error(f"Conditions failed for {venue}: {e}")
            errors.append(f"conditions: {e}")

        # Racing Australia — authoritative override
        try:
            from punty.scrapers.track_conditions import get_conditions_for_meeting
            ra_cond = await get_conditions_for_meeting(venue)
            if ra_cond:
                _apply_ra_conditions(meeting, ra_cond)
        except Exception as e:
            logger.error(f"RA conditions failed for {venue}: {e}")

    # RA Free Fields cross-check (only when PF succeeded)
    pf_failed = any("fields_only" in e for e in errors)
    if not pf_failed:
        try:
            ra_xcheck = await _cross_check_ra_fields(db, meeting, meeting_id)
            if ra_xcheck.get("mismatches"):
                logger.info(f"RA cross-check: {ra_xcheck['mismatches']} corrections for {venue}")
        except Exception as e:
            logger.warning(f"RA cross-check failed for {venue}: {e}")

    if owns_pf and pf_scraper:
        await pf_scraper.close()

    # Classify empty meetings as trials
    if not pf_failed and (not meeting.meeting_type or meeting.meeting_type == "race"):
        race_count = await db.execute(
            select(Race).where(Race.meeting_id == meeting_id).limit(1)
        )
        if not race_count.scalar_one_or_none():
            meeting.meeting_type = _classify_meeting_type(venue)
            if meeting.meeting_type == "race":
                meeting.meeting_type = "trial"
                logger.info(f"No races found for {venue} — classified as trial")

    # Fill derived fields (days_since_last_run, class_stats) from form_history
    await _fill_derived_fields(db, meeting_id)

    await db.commit()

    if errors:
        log_scrape_error(venue, "; ".join(errors))
    else:
        log_scrape_complete(venue)

    return {"meeting_id": meeting_id, "errors": errors}


async def scrape_meeting_full(meeting_id: str, db: AsyncSession, pf_scraper=None) -> dict:
    """Run all scrapers for a selected meeting and merge data into DB.

    Pipeline: fields → conditions → racing.com (odds+comments supplement) → TAB odds

    Args:
        pf_scraper: Optional shared PuntingFormScraper instance (avoids re-creating per meeting).
    """
    from punty.scrapers.playwright_base import scrape_lock
    from punty.scheduler.activity_log import log_scrape_start, log_scrape_complete, log_scrape_error

    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    venue = meeting.venue
    race_date = meeting.date
    errors = []
    owns_pf = pf_scraper is None

    log_scrape_start(venue)

    # Acquire scrape lock to prevent concurrent Playwright operations
    async with scrape_lock(venue):
        # Step 1: Primary API — runner/race data
        try:
            if pf_scraper is None:
                from punty.scrapers.punting_form import PuntingFormScraper
                pf_scraper = await PuntingFormScraper.from_settings(db)
            data = await pf_scraper.scrape_meeting_data(venue, race_date)
            await _upsert_meeting_data(db, meeting, data)
        except Exception as e:
            logger.error(f"Primary scrape failed: {e}")
            errors.append(f"primary: {e}")
            # Fallback 1: Racing Australia free fields (httpx, fast)
            try:
                from punty.scrapers.ra_fields import scrape_ra_fields
                data = await scrape_ra_fields(venue, race_date, meeting_id)
                if data:
                    await _upsert_meeting_data(db, meeting, data)
                    logger.info(f"RA fields fallback succeeded for {venue}")
                else:
                    raise ValueError("RA fields returned no data")
            except Exception as e_ra:
                logger.error(f"RA fields fallback failed for {venue}: {e_ra}")
                errors.append(f"ra_fields_fallback: {e_ra}")
                # Fallback 2: racing.com (Playwright, slower)
                try:
                    from punty.scrapers.racing_com import RacingComScraper
                    scraper = RacingComScraper()
                    try:
                        data = await scraper.scrape_meeting(venue, race_date)
                        await _upsert_meeting_data(db, meeting, data)
                    finally:
                        await scraper.close()
                except Exception as e2:
                    logger.error(f"racing.com fallback also failed: {e2}")
                    errors.append(f"racing.com_fallback: {e2}")

        # Step 2: Conditions — track/weather data
        if not meeting.track_condition_locked:
            try:
                if pf_scraper is None:
                    from punty.scrapers.punting_form import PuntingFormScraper
                    pf_scraper = await PuntingFormScraper.from_settings(db)
                cond = await pf_scraper.get_conditions_for_venue(venue)
                if cond:
                    _apply_pf_conditions(meeting, cond)
            except Exception as e:
                logger.error(f"Conditions failed: {e}")
                errors.append(f"conditions: {e}")

            # Step 2b: Racing Australia — authoritative override
            try:
                from punty.scrapers.track_conditions import get_conditions_for_meeting
                ra_cond = await get_conditions_for_meeting(venue)
                if ra_cond:
                    _apply_ra_conditions(meeting, ra_cond)
            except Exception as e:
                logger.error(f"RA conditions failed for {venue}: {e}")

        # Step 2c: RA Free Fields cross-check (only when PF succeeded)
        pf_ok = not any("primary" in e or "fields" in e for e in errors)
        if pf_ok:
            try:
                ra_xcheck = await _cross_check_ra_fields(db, meeting, meeting_id)
                if ra_xcheck.get("mismatches"):
                    logger.info(f"RA cross-check: {ra_xcheck['mismatches']} corrections for {venue}")
            except Exception as e:
                logger.warning(f"RA cross-check failed for {venue}: {e}")

        # Step 3: racing.com — supplementary odds + comments only
        try:
            from punty.scrapers.racing_com import RacingComScraper
            scraper = RacingComScraper()
            try:
                data = await scraper.scrape_meeting(venue, race_date)
                await _merge_racing_com_supplement(db, meeting_id, data)
            finally:
                await scraper.close()
        except Exception as e:
            logger.error(f"racing.com supplement failed: {e}")
            errors.append(f"racing.com_supplement: {e}")

        # Close PF scraper only if we own it
        if owns_pf and pf_scraper:
            await pf_scraper.close()

        # If no races were found, classify as trial/jumpout
        pf_failed = any("primary" in e for e in errors)
        if not pf_failed and (not meeting.meeting_type or meeting.meeting_type == "race"):
            race_count = await db.execute(
                select(Race).where(Race.meeting_id == meeting_id).limit(1)
            )
            if not race_count.scalar_one_or_none():
                meeting.meeting_type = _classify_meeting_type(venue)
                if meeting.meeting_type == "race":
                    meeting.meeting_type = "trial"
                    logger.info(f"No races found for {venue} — classified as trial")

        # Fill derived fields (days_since_last_run, class_stats) from form_history
        await _fill_derived_fields(db, meeting_id)

        await db.commit()

    if errors:
        log_scrape_error(venue, "; ".join(errors))
    else:
        log_scrape_complete(venue)

    return {"meeting_id": meeting_id, "errors": errors}


async def scrape_meeting_full_stream(meeting_id: str, db: AsyncSession) -> AsyncGenerator[dict, None]:
    """Run all scrapers for a meeting, yielding progress events."""
    logger.info(f"scrape_meeting_full_stream called for meeting_id={meeting_id}")

    # Check if another scrape is in progress
    from punty.scrapers.playwright_base import is_scrape_in_progress, scrape_lock
    in_progress, current = is_scrape_in_progress()
    if in_progress:
        yield {"step": 0, "total": 1, "label": f"Another scrape in progress: {current}. Please wait.", "status": "error"}
        return

    # Use explicit select instead of db.get() to avoid session state issues
    from sqlalchemy import select
    try:
        result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
        meeting = result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"DB query failed for {meeting_id}: {e}")
        yield {"step": 0, "total": 1, "label": f"DB error: {e}", "status": "error"}
        return

    if not meeting:
        logger.error(f"Meeting not found in DB: {meeting_id}")
        yield {"step": 0, "total": 1, "label": f"Meeting not found: {meeting_id}", "status": "error"}
        return

    logger.info(f"Found meeting: {meeting.venue} on {meeting.date}")
    venue = meeting.venue
    race_date = meeting.date
    errors = []
    total_steps = 5

    # Acquire scrape lock
    from punty.scrapers.playwright_base import _scrape_lock, _current_scrape
    import punty.scrapers.playwright_base as pw_base

    try:
        # Try to acquire lock (non-blocking check already done above)
        await _scrape_lock.acquire()
        pw_base._current_scrape = venue
        logger.info(f"Acquired scrape lock for {venue}")
    except Exception as e:
        yield {"step": 0, "total": 1, "label": f"Failed to acquire scrape lock: {e}", "status": "error"}
        return

    try:
        pf_scraper = None

        # Step 1: Primary API — runner/race data
        yield {"step": 0, "total": total_steps, "label": "Scraping race fields data...", "status": "running"}
        try:
            from punty.scrapers.punting_form import PuntingFormScraper
            pf_scraper = await PuntingFormScraper.from_settings(db)
            data = await pf_scraper.scrape_meeting_data(venue, race_date)
            race_count = len(data.get("races", []))
            runner_count = len(data.get("runners", []))
            await _upsert_meeting_data(db, meeting, data)
            yield {"step": 1, "total": total_steps,
                   "label": f"Fields complete — {race_count} races, {runner_count} runners", "status": "done"}
        except Exception as e:
            logger.error(f"Primary scrape failed: {e}")
            errors.append(f"primary: {e}")
            yield {"step": 1, "total": total_steps, "label": f"Fields failed: {e}, trying RA fallback...", "status": "error"}
            # Fallback 1: Racing Australia free fields (httpx, fast)
            try:
                from punty.scrapers.ra_fields import scrape_ra_fields
                data = await scrape_ra_fields(venue, race_date, meeting_id)
                if data:
                    race_count = len(data.get("races", []))
                    runner_count = len(data.get("runners", []))
                    await _upsert_meeting_data(db, meeting, data)
                    yield {"step": 1, "total": total_steps,
                           "label": f"RA fields fallback — {race_count} races, {runner_count} runners", "status": "done"}
                else:
                    raise ValueError("RA fields returned no data")
            except Exception as e_ra:
                logger.error(f"RA fields fallback failed for {venue}: {e_ra}")
                errors.append(f"ra_fields_fallback: {e_ra}")
                # Fallback 2: racing.com (Playwright, slower)
                try:
                    from punty.scrapers.racing_com import RacingComScraper
                    scraper = RacingComScraper()
                    try:
                        data = await scraper.scrape_meeting(venue, race_date)
                        race_count = len(data.get("races", []))
                        runner_count = len(data.get("runners", []))
                        await _upsert_meeting_data(db, meeting, data)
                        yield {"step": 1, "total": total_steps,
                               "label": f"racing.com fallback — {race_count} races, {runner_count} runners", "status": "done"}
                    finally:
                        await scraper.close()
                except Exception as e2:
                    logger.error(f"racing.com fallback also failed: {e2}")
                    errors.append(f"racing.com_fallback: {e2}")

        # Step 2: Conditions + weather
        yield {"step": 1, "total": total_steps, "label": "Fetching conditions/weather...", "status": "running"}
        try:
            if meeting.track_condition_locked:
                logger.info(f"Track condition locked for {venue}: {meeting.track_condition!r} (manual override)")
                yield {"step": 2, "total": total_steps, "label": f"Track conditions: {meeting.track_condition} (locked)", "status": "done"}
            else:
                if pf_scraper is None:
                    from punty.scrapers.punting_form import PuntingFormScraper
                    pf_scraper = await PuntingFormScraper.from_settings(db)
                cond = await pf_scraper.get_conditions_for_venue(venue)
                if cond:
                    _apply_pf_conditions(meeting, cond)
                    yield {"step": 2, "total": total_steps,
                           "label": f"Conditions: {meeting.track_condition or 'N/A'} | Rain: {cond.get('rainfall', 'N/A')}mm",
                           "status": "done"}
                elif meeting.track_condition:
                    yield {"step": 2, "total": total_steps, "label": f"Conditions: {meeting.track_condition} (from fields)", "status": "done"}
                else:
                    yield {"step": 2, "total": total_steps, "label": "Conditions: not available", "status": "done"}
        except Exception as e:
            logger.error(f"Conditions failed: {e}")
            errors.append(f"conditions: {e}")
            yield {"step": 2, "total": total_steps, "label": f"Conditions failed: {e}", "status": "error"}

        # Racing Australia — authoritative conditions override
        if not meeting.track_condition_locked:
            try:
                from punty.scrapers.track_conditions import get_conditions_for_meeting
                ra_cond = await get_conditions_for_meeting(venue)
                if ra_cond:
                    _apply_ra_conditions(meeting, ra_cond)
            except Exception as e:
                logger.error(f"RA conditions failed for {venue}: {e}")

        # Step 3: RA Free Fields cross-check
        pf_ok = not any("primary" in e or "fields" in e for e in errors)
        if pf_ok:
            yield {"step": 2, "total": total_steps, "label": "Cross-checking RA fields...", "status": "running"}
            try:
                ra_xcheck = await _cross_check_ra_fields(db, meeting, meeting_id)
                n = ra_xcheck.get("mismatches", 0)
                if n:
                    yield {"step": 3, "total": total_steps,
                           "label": f"RA cross-check: {n} corrections applied", "status": "done"}
                else:
                    yield {"step": 3, "total": total_steps,
                           "label": "RA cross-check: data consistent", "status": "done"}
            except Exception as e:
                logger.warning(f"RA cross-check failed for {venue}: {e}")
                yield {"step": 3, "total": total_steps,
                       "label": f"RA cross-check failed: {e}", "status": "error"}
        else:
            yield {"step": 3, "total": total_steps,
                   "label": "RA cross-check: skipped (PF failed)", "status": "done"}

        # Step 4: racing.com — supplementary odds + comments
        yield {"step": 3, "total": total_steps, "label": "Scraping racing.com odds + comments...", "status": "running"}
        try:
            from punty.scrapers.racing_com import RacingComScraper
            scraper = RacingComScraper()
            try:
                data = await scraper.scrape_meeting(venue, race_date)
                await _merge_racing_com_supplement(db, meeting_id, data)
            finally:
                await scraper.close()
            yield {"step": 4, "total": total_steps, "label": "racing.com odds/comments complete", "status": "done"}
        except Exception as e:
            logger.error(f"racing.com supplement failed: {e}")
            errors.append(f"racing.com_supplement: {e}")
            yield {"step": 4, "total": total_steps, "label": f"racing.com supplement failed: {e}", "status": "error"}

        # Step 5: TAB odds (optional supplementary — racing.com is primary)
        yield {"step": 4, "total": total_steps, "label": "Scraping TAB odds...", "status": "running"}
        try:
            from punty.scrapers.tab import TabScraper
            scraper = TabScraper()
            try:
                data = await scraper.scrape_meeting(venue, race_date)
                await _merge_odds(db, meeting_id, data.get("runners_odds", []))
            finally:
                await scraper.close()
            yield {"step": 5, "total": total_steps, "label": "TAB odds complete", "status": "done"}
        except Exception as e:
            logger.debug(f"TAB scrape skipped (optional): {e}")
            yield {"step": 5, "total": total_steps, "label": "TAB odds skipped (optional)", "status": "done"}

        # Close scraper
        if pf_scraper:
            await pf_scraper.close()

        # Fill derived fields (days_since_last_run, class_stats) from form_history
        await _fill_derived_fields(db, meeting_id)

        await db.commit()
        error_count = len(errors)
        # Use "meeting_done" instead of "complete" to avoid bulk scrape JS thinking entire operation is done
        if error_count:
            yield {"step": total_steps, "total": total_steps, "label": f"Complete with {error_count} error(s)", "status": "meeting_done", "errors": errors}
        else:
            yield {"step": total_steps, "total": total_steps, "label": "All scrapers complete!", "status": "meeting_done", "errors": []}
    finally:
        # Always release the scrape lock
        pw_base._current_scrape = None
        _scrape_lock.release()
        logger.info(f"Released scrape lock for {venue}")


async def scrape_speed_maps_stream(meeting_id: str, db: AsyncSession) -> AsyncGenerator[dict, None]:
    """Scrape speed maps for all races in a meeting, yielding progress events.

    Uses primary API as source (has speed rank, settle, map factor, jockey factor).
    Falls back to racing.com if primary source fails or has no data.
    """
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        yield {"step": 0, "total": 1, "label": "Meeting not found", "status": "error"}
        return

    result = await db.execute(
        select(Race).where(Race.meeting_id == meeting_id)
    )
    races = result.scalars().all()
    race_count = len(races)

    if race_count == 0:
        yield {"step": 0, "total": 1, "label": "No races found — scrape form data first", "status": "error"}
        return

    pos_map = {
        "leader": "leader",
        "on pace": "on_pace",
        "on_pace": "on_pace",
        "midfield": "midfield",
        "backmarker": "backmarker",
        "off pace": "backmarker",
        "off_pace": "backmarker",
    }

    # Track how many positions we found
    total_positions_found = 0

    # Helper to update runners with positions and pace insights
    async def update_runner_positions(event: dict, include_pf_insights: bool = False) -> int:
        """Update runners with speed map positions and optionally pace insights. Returns count of positions set."""
        count = 0
        if event.get("positions"):
            race_num = event["race_number"]
            race_id = f"{meeting_id}-r{race_num}"
            for pos in event["positions"]:
                horse_name = pos.get("horse_name", "")
                raw_pos = pos.get("position", "").lower()
                norm_pos = pos_map.get(raw_pos)
                if not norm_pos:
                    continue
                runner_result = await db.execute(
                    select(Runner).where(
                        Runner.race_id == race_id,
                        Runner.horse_name == horse_name,
                    )
                )
                runner = runner_result.scalar_one_or_none()
                if runner:
                    runner.speed_map_position = norm_pos
                    count += 1

                    # Store pace analysis insights if available
                    if include_pf_insights:
                        if pos.get("pf_speed_rank"):
                            try:
                                runner.pf_speed_rank = int(pos["pf_speed_rank"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_settle"):
                            try:
                                runner.pf_settle = float(pos["pf_settle"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_map_factor"):
                            try:
                                runner.pf_map_factor = float(pos["pf_map_factor"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_jockey_factor"):
                            try:
                                runner.pf_jockey_factor = float(pos["pf_jockey_factor"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_ai_score"):
                            try:
                                runner.pf_ai_score = float(pos["pf_ai_score"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_ai_price"):
                            try:
                                runner.pf_ai_price = float(pos["pf_ai_price"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_ai_rank"):
                            try:
                                runner.pf_ai_rank = int(pos["pf_ai_rank"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_assessed_price"):
                            try:
                                runner.pf_assessed_price = float(pos["pf_assessed_price"])
                            except (ValueError, TypeError):
                                pass
        return count

    # PRIMARY: Try primary API first (has richer data)
    pf_failed = False
    try:
        from punty.scrapers.punting_form import PuntingFormScraper
        pf_scraper = await PuntingFormScraper.from_settings(db)

        async for event in pf_scraper.scrape_speed_maps(meeting.venue, meeting.date, race_count):
            positions_set = await update_runner_positions(event, include_pf_insights=True)
            total_positions_found += positions_set
            yield {k: v for k, v in event.items() if k != "positions"}

    except Exception as e:
        logger.warning(f"Speed map scrape failed: {e}")
        pf_failed = True
        yield {"step": 0, "total": race_count + 1, "label": f"Speed map source unavailable: {e}", "status": "running"}

    # FALLBACK: Use racing.com if primary source failed or found nothing
    if pf_failed or total_positions_found == 0:
        if not pf_failed:
            logger.info(f"No speed map data for {meeting.venue} (0/{race_count} races had positions), trying racing.com fallback...")
            yield {"step": 0, "total": race_count + 1, "label": "Trying racing.com fallback...", "status": "running"}

        try:
            from punty.scrapers.racing_com import RacingComScraper
            scraper = RacingComScraper()
            try:
                async for event in scraper.scrape_speed_maps(meeting.venue, meeting.date, race_count):
                    positions_set = await update_runner_positions(event, include_pf_insights=False)
                    total_positions_found += positions_set
                    yield {k: v for k, v in event.items() if k != "positions"}
            finally:
                await scraper.close()
        except Exception as e:
            logger.error(f"Racing.com speed map scrape failed: {e}")
            yield {"step": 0, "total": 1, "label": f"Racing.com fallback failed: {e}", "status": "error"}

    # Calculate completeness - count active (non-scratched) runners vs those with speed maps
    total_active_runners = 0
    runners_with_speedmap = 0
    for race in races:
        runner_result = await db.execute(
            select(Runner).where(Runner.race_id == race.id, Runner.scratched == False)
        )
        race_runners = runner_result.scalars().all()
        total_active_runners += len(race_runners)
        runners_with_speedmap += sum(1 for r in race_runners if r.speed_map_position)

    # Consider complete if at least 50% of runners have speed map data
    # (some horses genuinely don't have sectional history)
    completeness_ratio = runners_with_speedmap / total_active_runners if total_active_runners > 0 else 0
    is_complete = completeness_ratio >= 0.5

    # Update meeting status
    meeting.speed_map_complete = is_complete

    if total_positions_found == 0:
        logger.warning(f"No speed map data found for {meeting.venue} from any source")
        meeting.speed_map_complete = False
        # Unselect meetings with no data
        if meeting.selected:
            meeting.selected = False
            logger.info(f"Auto-unselected {meeting.venue} due to missing speed map data")
            yield {
                "step": race_count + 1,
                "total": race_count + 1,
                "label": f"WARNING: No speed map data available - meeting unselected",
                "status": "warning",
                "incomplete": True,
            }
    elif not is_complete:
        logger.warning(f"Incomplete speed map data for {meeting.venue}: {runners_with_speedmap}/{total_active_runners} ({completeness_ratio:.0%})")
        # Unselect meetings with very incomplete data (less than 30%)
        if completeness_ratio < 0.3 and meeting.selected:
            meeting.selected = False
            logger.info(f"Auto-unselected {meeting.venue} due to very incomplete speed map data ({completeness_ratio:.0%})")
            yield {
                "step": race_count + 1,
                "total": race_count + 1,
                "label": f"WARNING: Only {completeness_ratio:.0%} speed map coverage - meeting unselected",
                "status": "warning",
                "incomplete": True,
            }
    else:
        logger.info(f"Set {total_positions_found} speed map positions for {meeting.venue} ({completeness_ratio:.0%} coverage)")

    await db.commit()


async def refresh_odds(meeting_id: str, db: AsyncSession) -> dict:
    """Quick refresh of conditions and scratchings for a meeting.

    Uses PF API for lightweight HTTP calls (no Playwright needed).
    Odds are already captured by the main racing.com scrape.
    """
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    try:
        from punty.scrapers.punting_form import PuntingFormScraper

        pf = await PuntingFormScraper.from_settings(db)
        if pf:
            try:
                # Refresh conditions (track, rail, weather, penetrometer)
                cond = await pf.get_conditions_for_venue(meeting.venue)
                if cond:
                    _apply_pf_conditions(meeting, cond)

                # Refresh scratchings
                pf_meeting_id = await pf.resolve_meeting_id(meeting.venue, meeting.date)
                if pf_meeting_id:
                    scratchings = await pf.get_scratchings_for_meeting(pf_meeting_id)
                    if scratchings:
                        for s in scratchings:
                            race_num = s.get("raceNo") or s.get("race_number")
                            tab_no = s.get("tabNo") or s.get("saddlecloth")
                            if race_num and tab_no:
                                race_id = f"{meeting_id}-r{race_num}"
                                runner = await db.execute(
                                    select(Runner).where(
                                        Runner.race_id == race_id,
                                        Runner.saddlecloth == tab_no,
                                    )
                                )
                                r = runner.scalar_one_or_none()
                                if r and not r.scratched:
                                    r.scratched = True
                                    logger.info(f"Scratched {r.horse_name} (R{race_num} #{tab_no}) via PF API")
            finally:
                await pf.close()

        await db.commit()
        return {"meeting_id": meeting_id, "status": "ok"}
    except Exception as e:
        logger.error(f"Odds refresh failed: {e}")
        return {"meeting_id": meeting_id, "status": "error", "error": str(e)}


# --- Internal helpers ---

async def _upsert_meeting_data(db: AsyncSession, meeting: Meeting, data: dict) -> None:
    """Upsert meeting, races, and runners from scraper output."""
    m = data["meeting"]

    # Update all meeting fields
    for field in MEETING_FIELDS:
        val = m.get(field)
        if val is not None:
            if field == "track_condition" and meeting.track_condition:
                if _is_more_specific(val, meeting.track_condition):
                    logger.info(f"Track condition update via upsert: {meeting.track_condition!r} → {val!r}")
                    meeting.track_condition = val
                else:
                    logger.debug(f"Skipping track condition {val!r} (existing {meeting.track_condition!r} is better)")
            elif field == "irrigation" and isinstance(val, str):
                meeting.irrigation = bool(val and "nil" not in val.lower() and val.strip() != "0")
            else:
                setattr(meeting, field, val)

    for r in data.get("races", []):
        existing_race = await db.get(Race, r["id"])
        if existing_race:
            for field in RACE_FIELDS:
                val = r.get(field)
                if val is not None:
                    setattr(existing_race, field, val)
        else:
            race_kwargs = {
                "id": r["id"],
                "meeting_id": r["meeting_id"],
                "race_number": r["race_number"],
                "name": r.get("name", f"Race {r['race_number']}"),
                "distance": r.get("distance", 1200),
                "class_": r.get("class_"),
                "prize_money": r.get("prize_money"),
                "start_time": r.get("start_time"),
                "status": r.get("status", "scheduled"),
                "track_condition": r.get("track_condition"),
                "race_type": r.get("race_type"),
                "age_restriction": r.get("age_restriction"),
                "weight_type": r.get("weight_type"),
                "field_size": r.get("field_size"),
            }
            db.add(Race(**race_kwargs))

    for runner in data.get("runners", []):
        # Match by race_id + saddlecloth first (stable key), then fall back to ID
        existing_runner = None
        if runner.get("saddlecloth") and runner.get("race_id"):
            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == runner["race_id"],
                    Runner.saddlecloth == runner["saddlecloth"],
                ).limit(1)
            )
            existing_runner = result.scalar_one_or_none()
        if not existing_runner:
            existing_runner = await db.get(Runner, runner["id"])

        if existing_runner:
            # Update ID if it changed (barrier-based → saddlecloth-based)
            if existing_runner.id != runner["id"]:
                existing_runner.id = runner["id"]
            for field in RUNNER_FIELDS:
                val = runner.get(field)
                if val is not None:
                    setattr(existing_runner, field, val)
            if runner.get("career_record"):
                existing_runner.career_record = runner["career_record"]
            if runner.get("speed_map_position"):
                existing_runner.speed_map_position = runner["speed_map_position"]
        else:
            runner_kwargs = {
                "id": runner["id"],
                "race_id": runner["race_id"],
                "horse_name": runner["horse_name"],
                "barrier": runner.get("barrier"),
                "career_record": runner.get("career_record"),
                "speed_map_position": runner.get("speed_map_position"),
            }
            for field in RUNNER_FIELDS:
                runner_kwargs[field] = runner.get(field)
            db.add(Runner(**runner_kwargs))

    await db.flush()


def _apply_pf_conditions(meeting: Meeting, cond: dict) -> None:
    """Apply conditions data to a Meeting object."""
    new_cond = cond.get("condition")
    if new_cond and _is_more_specific(new_cond, meeting.track_condition):
        logger.info(f"Condition for {meeting.venue}: {meeting.track_condition!r} → {new_cond!r}")
        meeting.track_condition = new_cond
    meeting.rail_position = cond.get("rail") or meeting.rail_position
    meeting.weather = cond.get("weather") or meeting.weather
    if cond.get("penetrometer") is not None:
        meeting.penetrometer = cond["penetrometer"]
    if cond.get("wind_speed") is not None:
        meeting.weather_wind_speed = cond["wind_speed"]
    if cond.get("wind_direction"):
        meeting.weather_wind_dir = cond["wind_direction"]
    if cond.get("humidity") is not None:
        meeting.weather_humidity = cond["humidity"]
    if cond.get("rainfall") is not None:
        meeting.rainfall = cond["rainfall"]
    if cond.get("irrigation") is not None:
        irr = cond["irrigation"]
        if isinstance(irr, str):
            meeting.irrigation = bool(irr and "nil" not in irr.lower() and irr.strip() != "0")
        else:
            meeting.irrigation = bool(irr)
    if cond.get("going_stick") is not None:
        meeting.going_stick = cond["going_stick"]


def _apply_ra_conditions(meeting: Meeting, cond: dict) -> None:
    """Apply Racing Australia conditions — authoritative source, always overwrites."""
    new_cond = cond.get("condition")
    if new_cond:
        logger.info(
            f"RA condition for {meeting.venue}: "
            f"{meeting.track_condition!r} → {new_cond!r}"
        )
        meeting.track_condition = new_cond
    if cond.get("rail"):
        meeting.rail_position = cond["rail"]
    if cond.get("weather"):
        meeting.weather = cond["weather"]
    if cond.get("penetrometer") is not None:
        meeting.penetrometer = cond["penetrometer"]
    if cond.get("rainfall") is not None:
        meeting.rainfall = cond["rainfall"]
    if cond.get("irrigation") is not None:
        irr = cond["irrigation"]
        if isinstance(irr, str):
            meeting.irrigation = bool(irr and "nil" not in irr.lower() and irr.strip() != "0")
        else:
            meeting.irrigation = bool(irr)


import re as _re

def _normalise_jockey(name: str) -> str:
    """Normalise jockey name for comparison: strip Ms/Mr prefix, claim weights, (late alt)."""
    s = name.strip()
    s = _re.sub(r"^(Ms|Mr)\s+", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r"\s*\(a\d*/\d+kg\)", "", s)  # claim weight e.g. (a0/53kg)
    s = _re.sub(r"\s*\(late\s+alt\)", "", s, flags=_re.IGNORECASE)
    return s.strip()


def _xcheck_equal(field: str, pf_val, ra_val) -> bool:
    """Compare PF and RA values with field-specific normalisation."""
    if field == "class_":
        return str(pf_val).strip().upper() == str(ra_val).strip().upper()
    if field == "jockey":
        return _normalise_jockey(str(pf_val)) == _normalise_jockey(str(ra_val))
    return str(pf_val) == str(ra_val)


async def _cross_check_ra_fields(
    db: AsyncSession, meeting: Meeting, meeting_id: str,
) -> dict:
    """Cross-check PF data against Racing Australia Free Fields.

    RA is authoritative for: scratched, barrier, weight, jockey (runner-level)
    and distance, class_, field_size, start_time (race-level).
    Only overwrites RA-authoritative fields; PF keeps richer data (form, stats, odds).
    """
    from punty.scrapers.ra_fields import scrape_ra_fields

    venue = meeting.venue
    race_date = meeting.date
    mismatches: list[str] = []

    try:
        ra_data = await scrape_ra_fields(venue, race_date, meeting_id)
    except Exception as e:
        logger.warning(f"RA cross-check failed for {venue}: {e}")
        return {"status": "error", "error": str(e), "mismatches": 0}

    if not ra_data:
        logger.info(f"RA cross-check: no data for {venue}")
        return {"status": "no_data", "mismatches": 0}

    # --- Race-level cross-check ---
    for ra_race in ra_data.get("races", []):
        db_race = await db.get(Race, ra_race["id"])
        if not db_race:
            continue
        for field in _RA_AUTH_RACE_FIELDS:
            ra_val = ra_race.get(field)
            if ra_val is None:
                continue
            pf_val = getattr(db_race, field, None)
            if pf_val is not None and not _xcheck_equal(field, pf_val, ra_val):
                mismatches.append(f"R{db_race.race_number} {field}: PF={pf_val} RA={ra_val}")
                logger.warning(
                    f"PF↔RA mismatch R{db_race.race_number}: {field} "
                    f"PF={pf_val} RA={ra_val} → using RA"
                )
            setattr(db_race, field, ra_val)

    # --- Runner-level cross-check ---
    for ra_runner in ra_data.get("runners", []):
        race_id = ra_runner.get("race_id")
        saddlecloth = ra_runner.get("saddlecloth")
        if not (race_id and saddlecloth):
            continue

        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.saddlecloth == saddlecloth,
            ).limit(1)
        )
        db_runner = result.scalar_one_or_none()
        if not db_runner:
            continue

        horse = db_runner.horse_name or f"#{saddlecloth}"

        for field in _RA_AUTH_RUNNER_FIELDS:
            ra_val = ra_runner.get(field)
            if ra_val is None:
                continue
            pf_val = getattr(db_runner, field, None)

            # Scratched: RA can scratch but never un-scratch
            if field == "scratched":
                if ra_val and not pf_val:
                    mismatches.append(f"{horse}: scratched by RA")
                    logger.warning(f"PF↔RA: {horse} not scratched in PF → scratched in RA")
                    db_runner.scratched = True
                continue

            # Other fields: compare and overwrite with RA
            if pf_val is not None and not _xcheck_equal(field, pf_val, ra_val):
                mismatches.append(f"{horse}: {field} PF={pf_val} RA={ra_val}")
                logger.warning(
                    f"PF↔RA mismatch {horse}: {field} PF={pf_val} RA={ra_val} → using RA"
                )
            setattr(db_runner, field, ra_val)

    await db.flush()

    n = len(mismatches)
    if n:
        logger.info(f"RA cross-check for {venue}: {n} mismatches corrected")
    else:
        logger.info(f"RA cross-check for {venue}: all data consistent")

    return {"status": "ok", "mismatches": n, "details": mismatches}


# Fields from racing.com that supplement primary data (odds, comments only)
_SUPPLEMENT_FIELDS = [
    "current_odds", "opening_odds", "place_odds",
    "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes", "odds_betfair",
    "odds_flucs", "comment_long", "comment_short", "comments", "stewards_comment",
]


async def _merge_racing_com_supplement(db: AsyncSession, meeting_id: str, data: dict) -> None:
    """Merge racing.com odds and comments into existing runners.

    Only updates odds and comment fields — does NOT overwrite primary runner data.
    """
    for runner_data in data.get("runners", []):
        horse_name = runner_data.get("horse_name", "")
        race_id = runner_data.get("race_id", "")
        if not (horse_name and race_id):
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

        for field in _SUPPLEMENT_FIELDS:
            val = runner_data.get(field)
            if val is not None:
                # Don't overwrite existing odds with None or 0
                if field.startswith("odds_") or field in ("current_odds", "opening_odds", "place_odds"):
                    if not val:
                        continue
                setattr(runner, field, val)

    await db.flush()


async def _fill_derived_fields(db: AsyncSession, meeting_id: str) -> None:
    """Fill fields that can be derived from form_history when primary sources fail.

    Derives: days_since_last_run, class_stats (from form_history JSON).
    Called after all scraper data is merged, before commit.
    """
    from datetime import datetime
    from punty.config import melb_now

    result = await db.execute(
        select(Runner).join(Race, Runner.race_id == Race.id).where(
            Race.meeting_id == meeting_id,
            Runner.scratched == False,
        )
    )
    runners = result.scalars().all()
    now = melb_now().replace(tzinfo=None)
    filled = 0

    for runner in runners:
        fh_raw = runner.form_history
        if not fh_raw:
            continue

        try:
            fh = _json.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
        except (ValueError, TypeError):
            continue

        if not isinstance(fh, list) or not fh:
            continue

        # Derive days_since_last_run from most recent form_history entry
        if not runner.days_since_last_run:
            latest_date = fh[0].get("date") if isinstance(fh[0], dict) else None
            if latest_date:
                try:
                    last_dt = datetime.strptime(str(latest_date)[:10], "%Y-%m-%d")
                    runner.days_since_last_run = (now - last_dt).days
                    filled += 1
                except (ValueError, TypeError):
                    pass

        # Derive class_stats from form_history when atThisClassStats is missing
        if not runner.class_stats:
            race = await db.get(Race, runner.race_id)
            if race:
                race_class = getattr(race, "class_", None) or ""
                _derive_class_stats(runner, fh, race_class)

    await db.flush()
    if filled:
        logger.info(f"Filled derived fields for {meeting_id}: {filled} days_since_last_run")


def _derive_class_stats(runner: "Runner", form_history: list, race_class: str) -> None:
    """Derive class_stats from form_history by matching class buckets."""
    from punty.context.combo_form import _bucket_class, _parse_position

    today_bucket = _bucket_class(race_class)
    if not today_bucket:
        return

    starts = 0
    wins = 0
    seconds = 0
    thirds = 0

    for start in form_history:
        if not isinstance(start, dict):
            continue
        hist_class = start.get("class", "")
        if not hist_class:
            continue
        if _bucket_class(hist_class) == today_bucket:
            starts += 1
            pos = _parse_position(start.get("position"))
            if pos == 1:
                wins += 1
            elif pos == 2:
                seconds += 1
            elif pos == 3:
                thirds += 1

    if starts >= 1:
        runner.class_stats = _json.dumps({
            "starts": starts, "wins": wins, "seconds": seconds, "thirds": thirds
        })


async def _merge_odds(db: AsyncSession, meeting_id: str, odds_list: list[dict]) -> None:
    """Merge odds data into existing runners."""
    for odds in odds_list:
        race_num = odds.get("race_number")
        horse_name = odds.get("horse_name")
        if not (race_num and horse_name):
            continue

        race_id = f"{meeting_id}-r{race_num}"
        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.horse_name == horse_name,
            ).limit(1)
        )
        runner = result.scalar_one_or_none()
        if runner:
            if odds.get("current_odds") is not None:
                runner.current_odds = odds["current_odds"]
            if odds.get("opening_odds") is not None and not runner.opening_odds:
                runner.opening_odds = odds["opening_odds"]
            if odds.get("place_odds") is not None:
                runner.place_odds = odds["place_odds"]
            if odds.get("scratched"):
                runner.scratched = True
                runner.scratching_reason = odds.get("scratching_reason") or runner.scratching_reason


def _validate_result(result: dict, race_id: str) -> dict | None:
    """Validate and sanitize a single result entry. Returns None if invalid."""
    import re as _re
    # Validate position (handles dead heats like "=1", "DH1", "D1")
    position = result.get("position")
    if position is not None:
        try:
            position = int(position)
        except (ValueError, TypeError):
            # Try to extract numeric part from dead heat strings
            pos_str = str(position).strip()
            match = _re.search(r"\d+", pos_str)
            if match:
                position = int(match.group())
                result["dead_heat"] = True
                logger.info(f"{race_id}: Dead heat detected — position '{pos_str}' → {position}")
            else:
                logger.warning(f"{race_id}: Non-integer position '{position}', skipping")
                return None
        if position < 1 or position > 30:
            logger.warning(f"{race_id}: Invalid position {position}, skipping")
            return None
        result["position"] = position

    # Validate dividends (must be positive if present)
    for div_field in ("win_dividend", "place_dividend"):
        val = result.get(div_field)
        if val is not None:
            try:
                val = float(val)
                if val < 0:
                    logger.warning(f"{race_id}: Negative {div_field} {val}, setting to None")
                    result[div_field] = None
                else:
                    result[div_field] = val
            except (ValueError, TypeError):
                logger.warning(f"{race_id}: Invalid {div_field} '{val}', setting to None")
                result[div_field] = None

    # Validate saddlecloth
    saddlecloth = result.get("saddlecloth")
    if saddlecloth is not None:
        try:
            saddlecloth = int(saddlecloth)
            if saddlecloth < 1 or saddlecloth > 30:
                result["saddlecloth"] = None
            else:
                result["saddlecloth"] = saddlecloth
        except (ValueError, TypeError):
            result["saddlecloth"] = None

    return result


async def upsert_race_results(db: AsyncSession, meeting_id: str, race_number: int, results_data: dict) -> None:
    """Update Runner result fields and Race status from scraped results."""
    race_id = f"{meeting_id}-r{race_number}"
    race = await db.get(Race, race_id)
    if not race:
        logger.warning(f"Race not found for results: {race_id}")
        return

    # Validate: check for duplicate winners
    positions = [r.get("position") for r in results_data.get("results", []) if r.get("position") is not None]
    if positions.count(1) > 1:
        logger.info(f"{race_id}: Dead heat — {positions.count(1)} runners share 1st place")

    # Update runner result fields
    matched = 0
    for result in results_data.get("results", []):
        # Validate result before processing
        result = _validate_result(result, race_id)
        if result is None:
            continue
        horse_name = result.get("horse_name", "")
        saddlecloth = result.get("saddlecloth")

        runner = None
        # Try horse name first
        if horse_name:
            runner_result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.horse_name == horse_name,
                ).limit(1)
            )
            runner = runner_result.scalar_one_or_none()

        # Fallback to saddlecloth
        if not runner and saddlecloth is not None:
            runner_result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.saddlecloth == int(saddlecloth),
                ).limit(1)
            )
            runner = runner_result.scalar_one_or_none()

        if not runner:
            continue

        matched += 1
        if result.get("position") is not None:
            runner.finish_position = result["position"]
        if result.get("margin") is not None:
            runner.result_margin = result["margin"]
        if result.get("starting_price") is not None:
            runner.starting_price = result["starting_price"]
        if result.get("win_dividend") is not None:
            runner.win_dividend = result["win_dividend"]
        elif result.get("starting_price") and result.get("position") in (1, 2, 3):
            # Use starting price as fallback when dividends aren't available
            try:
                sp_val = float(str(result["starting_price"]).replace("$", "").replace(",", ""))
                if result["position"] == 1 and not runner.win_dividend:
                    runner.win_dividend = sp_val
            except (ValueError, TypeError):
                pass
        if result.get("place_dividend") is not None:
            runner.place_dividend = result["place_dividend"]
        if result.get("sectional_400"):
            runner.sectional_400 = result["sectional_400"]
        if result.get("sectional_800"):
            runner.sectional_800 = result["sectional_800"]

    scraped_count = len(results_data.get("results", []))
    if scraped_count > 0 and matched == 0:
        logger.warning(f"Results for {race_id}: 0/{scraped_count} runners matched — wrong race data? Skipping status update.")
    else:
        # Only update race-level fields if runners actually matched
        race.results_status = "Paying"
        if results_data.get("winning_time"):
            race.winning_time = results_data["winning_time"]
        if results_data.get("exotics"):
            race.exotic_results = _json.dumps(results_data["exotics"])

    await db.flush()
    logger.info(f"Upserted results for {race_id}: {matched}/{scraped_count} runners matched")


def _classify_meeting_type(venue: str) -> str:
    """Classify whether a meeting is a race, trial, or jump out based on venue name."""
    v = venue.lower()
    # Jump out venues typically have a prefix like "Southside", "Inside", "Course Proper"
    # or explicitly say "jump" / "jumpout" / "barrier trial"
    jumpout_keywords = ("jump out", "jumpout", "jump-out")
    trial_keywords = ("trial", "barrier trial")
    # Common jump out venue prefixes used by Racing Victoria, NSW, etc.
    jumpout_prefixes = ("southside", "inside", "course proper", "lakeside")

    for kw in jumpout_keywords:
        if kw in v:
            return "jumpout"
    for kw in trial_keywords:
        if kw in v:
            return "trial"
    for prefix in jumpout_prefixes:
        if v.startswith(prefix):
            return "jumpout"
    return "race"


# Complete Australian racetracks by state (from racingaustralia.horse)
_STATE_TRACKS = {
    "NSW": [
        "adaminaby", "albury", "ardlethan", "armidale", "ballina", "bathurst",
        "beaumont newcastle", "bingara", "binnaway", "boorowa", "bourke", "bowraville",
        "braidwood", "brewarrina", "broken hill", "canberra", "canterbury", "canterbury park",
        "carinda", "carrathool", "casino", "cessnock", "cobar", "coffs harbour",
        "collarenebri", "come-by-chance", "condobolin", "coolabah", "cooma", "coonabarabran",
        "coonamble", "cootamundra", "corowa", "cowra", "crookwell", "deepwater", "deniliquin",
        "enngonia", "fernhill", "forbes", "geurie", "gilgandra", "glen innes", "gosford",
        "goulburn", "grafton", "grenfell", "griffith", "gulargambone", "gulgong", "gundagai",
        "gunnedah", "harden", "hawkesbury", "hay", "hillston", "holbrook", "jerilderie",
        "kembla grange", "kempsey", "kensington", "lakelands", "leeton", "lightning ridge",
        "lismore", "lockhart", "louth", "mallawa", "mendooran", "merriwa", "moama", "moree",
        "moruya", "moulamein", "mudgee", "mungery", "mungindi", "murwillumbah", "muswellbrook",
        "narrabri", "narrandera", "narromine", "newcastle", "nowra", "nyngan", "orange",
        "parkes", "pooncarie", "quambone", "queanbeyan", "quirindi", "randwick", "rosehill",
        "rosehill gardens", "royal randwick", "sapphire coast", "scone", "tabulam", "talmoi",
        "tamworth", "taree", "tocumwal", "tomingley", "tottenham", "trangie", "trundle",
        "tullamore", "tullibigeal", "tumbarumba", "tumut", "tuncurry", "wagga", "wagga riverside",
        "walcha", "walgett", "wallabadah", "wamboyne", "warialda", "warren", "warwick farm",
        "wauchope", "wean", "wellington", "wentworth", "wyong", "yass", "young",
    ],
    "VIC": [
        "alexandra", "ararat", "avoca", "bairnsdale", "ballan", "ballarat", "balnarring",
        "benalla", "bendigo", "burrumbeet", "caulfield", "colac", "coleraine", "cranbourne",
        "donald", "drouin", "dunkeld", "echuca", "edenhope", "flemington", "geelong",
        "great western", "gunbower", "hanging rock", "healesville", "hinnomunjie", "horsham",
        "kerang", "kilmore", "kyneton", "manangatang", "mansfield", "merton", "mildura",
        "moe", "moonee valley", "the valley", "mornington", "mortlake", "murtoa", "nhill",
        "oak park", "pakenham", "penshurst", "sale", "sandown", "seymour", "st arnaud",
        "stawell", "stony creek", "swan hill", "swifts creek", "tatura", "towong", "traralgon",
        "warracknabeal", "warrnambool", "werribee", "werribee park", "wodonga", "wycheproof",
        "yarra glen", "yarra valley", "yea",
    ],
    "QLD": [
        "almaden", "alpha", "aramac", "augathella", "beaudesert", "bedourie", "bell",
        "betoota", "birdsville", "blackall", "bluff", "boulia", "bowen", "bundaberg",
        "burketown", "burrandowan", "cairns", "calliope", "camooweal", "capella",
        "charleville", "charters towers", "chillagoe", "chinchilla", "clifton", "cloncurry",
        "coen", "cooktown", "corfield", "cunnamulla", "dalby", "deagon", "dingo", "doomben",
        "duaringa", "eagle farm", "eidsvold", "einasleigh", "emerald", "eromanga", "esk",
        "ewan", "flinton", "gatton", "gayndah", "georgetown", "gladstone", "gold coast",
        "goondiwindi", "gordonvale", "gregory downs", "gympie", "hebel", "home hill",
        "hughenden", "ilfracombe", "ingham", "injune", "innisfail", "ipswich", "isisford",
        "jandowae", "jericho", "julia creek", "jundah", "kilcoy", "kumbia", "laura",
        "longreach", "mackay", "mareeba", "maxwelton", "mckinlay", "middlemount", "miles",
        "mingela", "mitchell", "monto", "moranbah", "morven", "mount garnet", "mount isa",
        "mount perry", "muttaburra", "nanango", "noccundra", "normanton", "oakey", "oakley",
        "prairie", "quamby", "quilpie", "richmond", "ridgelands", "rockhampton", "roma",
        "springsure", "stamford", "stanthorpe", "st george", "stonehenge", "sunshine coast",
        "surat", "tambo", "tara", "taroom", "thangool", "theodore", "toowoomba", "townsville",
        "tower hill", "twin hills", "wandoan", "warra", "warwick", "wilpeena", "windorah",
        "winton", "wondai", "wyandra",
    ],
    "SA": [
        "balaklava", "bordertown", "ceduna", "cheltenham park", "clare", "gawler", "hawker",
        "jamestown", "kingscote", "kimba", "lock", "mindarie-halidon", "morphettville",
        "morphettville parks", "mount gambier", "murray bridge", "naracoorte", "oakbank",
        "penola", "penong", "port augusta", "port lincoln", "port pirie", "quorn",
        "roxby downs", "strathalbyn", "streaky bay", "tumby bay", "victoria park",
    ],
    "WA": [
        "albany", "ascot", "ashburton", "belmont", "beverley", "broome", "bunbury",
        "carnarvon", "collie", "derby", "dongara", "esperance", "exmouth", "fitzroy",
        "geraldton", "junction", "kalgoorlie", "kimberley", "kojonup", "kununurra", "landor",
        "lark hill", "laverton", "leinster", "leonora", "meekatharra", "mingenew", "moora",
        "mount barker", "mount magnet", "narrogin", "newman", "norseman", "northam", "perth",
        "pingrup", "pinjarra", "pinjarra park", "port hedland", "roebourne", "toodyay",
        "wiluna", "wyndham", "yalgoo", "york",
    ],
    "TAS": [
        "deloraine", "devonport", "hobart", "king island", "launceston", "longford", "spreyton",
    ],
    "NT": [
        "adelaide river", "alice springs", "barrow creek", "darwin", "katherine", "larrimah",
        "mataranka", "pine creek", "pioneer park", "renner", "tennant creek", "timber creek",
    ],
    "ACT": ["canberra", "canberra acton"],
}

# Build reverse lookup: venue -> state
_VENUE_TO_STATE = {}
for state, tracks in _STATE_TRACKS.items():
    for track in tracks:
        _VENUE_TO_STATE[track] = state


def _guess_state(venue: str) -> str:
    """Guess state from venue name using complete Australian track database."""
    v = venue.lower()

    # Direct match
    if v in _VENUE_TO_STATE:
        return _VENUE_TO_STATE[v]

    # Try partial match (venue name contains track name or vice versa)
    for track, state in _VENUE_TO_STATE.items():
        if track in v or v in track:
            return state

    # Default to VIC if unknown
    return "VIC"


# ============ TRAINER STATS ============
# Cache trainer premiership data (refreshed once per session/day)
_trainer_premiership_cache: list[dict] = []
_trainer_cache_loaded: bool = False


async def fetch_trainer_premiership(force_refresh: bool = False) -> list[dict]:
    """Fetch trainer ranking data, using cache if available.

    Tries TRC (Thoroughbred Racing) global rankings first, falls back to Racing Australia.
    """
    global _trainer_premiership_cache, _trainer_cache_loaded

    if _trainer_cache_loaded and not force_refresh:
        return _trainer_premiership_cache

    # Try TRC first (better data with global rankings, group wins, etc.)
    try:
        from punty.scrapers.racing_australia import TRCTrainerScraper

        scraper = TRCTrainerScraper()
        try:
            trainers = await scraper.scrape_trainer_rankings(country="AUS", pages=2)
            if trainers:
                _trainer_premiership_cache = trainers
                _trainer_cache_loaded = True
                logger.info(f"Loaded {len(trainers)} trainers from TRC global rankings")
                return trainers
        finally:
            await scraper.close()
    except Exception as e:
        logger.warning(f"TRC trainer scrape failed, trying Racing Australia: {e}")

    # Fallback to Racing Australia premiership
    try:
        from punty.scrapers.racing_australia import RacingAustraliaScraper

        scraper = RacingAustraliaScraper()
        try:
            trainers = await scraper.scrape_trainer_premiership(season="2025")
            _trainer_premiership_cache = trainers
            _trainer_cache_loaded = True
            logger.info(f"Loaded {len(trainers)} trainers from Racing Australia premiership")
            return trainers
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"Failed to fetch trainer data from both sources: {e}")
        return _trainer_premiership_cache  # Return stale cache if available


async def populate_trainer_stats(db: AsyncSession, meeting_id: str) -> dict:
    """Populate trainer_stats for all runners in a meeting.

    Fetches trainer premiership data and matches to each runner's trainer.
    Returns dict with count of matches.
    """
    from punty.scrapers.racing_australia import match_trainer_name, format_trainer_stats

    # Get trainer premiership data
    trainers = await fetch_trainer_premiership()
    if not trainers:
        return {"matched": 0, "total": 0, "error": "No trainer data available"}

    # Get all runners for this meeting
    result = await db.execute(
        select(Runner)
        .join(Race)
        .where(Race.meeting_id == meeting_id)
    )
    runners = result.scalars().all()

    matched = 0
    for runner in runners:
        if not runner.trainer:
            continue

        # Try to match trainer
        trainer_data = match_trainer_name(runner.trainer, trainers)
        if trainer_data:
            runner.trainer_stats = format_trainer_stats(trainer_data)
            matched += 1

    await db.commit()
    logger.info(f"Trainer stats: matched {matched}/{len(runners)} runners for {meeting_id}")
    return {"matched": matched, "total": len(runners)}
