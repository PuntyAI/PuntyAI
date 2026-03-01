"""Scraping orchestrator — coordinates scrapers and stores results."""

import json as _json
import logging
from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.meeting import Meeting, Race, Runner
from punty.venues import guess_state

logger = logging.getLogger(__name__)


def _normalise_track(cond: str | None) -> str:
    """Normalise track condition for comparison (prevents false change alerts)."""
    if not cond:
        return ""
    import re
    return re.sub(r"\s+", " ", cond.strip().lower().replace("(", "").replace(")", ""))


def _should_update_condition(new_cond: str | None, old_cond: str | None) -> bool:
    """Return True if track condition should be updated.

    Allows all updates EXCEPT losing specificity within the same base category
    (e.g. blocks "Good 4" → "Good", but allows "Soft 5" → "Good 4").

    Returns True for:
    - No existing condition (old is None/empty)
    - Different base category ("Soft 5" → "Good 4") — real condition change
    - More specific ("Good" → "Good 4")
    - Same specificity, different rating ("Good 3" → "Good 4")

    Returns False for:
    - New is None/empty
    - Same normalised value ("Good 4" vs "Good 4")
    - Less specific within same base ("Good 4" → "Good")
    """
    import re
    if not new_cond:
        return False
    if not old_cond:
        return True
    # Same normalised value — no update needed
    norm_new = re.sub(r"\s+", " ", new_cond.strip().lower())
    norm_old = re.sub(r"\s+", " ", old_cond.strip().lower())
    if norm_new == norm_old:
        return False
    # Only block: same base category, old has rating, new doesn't
    new_has_rating = bool(re.search(r"\d", new_cond))
    old_has_rating = bool(re.search(r"\d", old_cond))
    if not new_has_rating and old_has_rating:
        new_base = new_cond.strip().split()[0].lower()
        old_base = old_cond.strip().split()[0].lower()
        if new_base == old_base:
            return False  # "Good 4" → "Good" blocked
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
        from punty.venues import venue_slug as _venue_slug
        venue_slug = _venue_slug(venue) or venue.lower().replace(" ", "-")
        meeting_id = f"{venue_slug}-{today.isoformat()}"

        existing = await db.get(Meeting, meeting_id)
        if existing:
            if not existing.source:
                existing.source = "racing.com/calendar"
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

    # Conditions — single gatekeeper (RA authoritative, PF supplementary)
    try:
        pf_cond = await pf_scraper.get_conditions_for_venue(venue)
    except Exception as e:
        logger.error(f"PF conditions failed for {venue}: {e}")
        errors.append(f"conditions: {e}")
        pf_cond = None
    try:
        await refresh_track_conditions(meeting, pf_cond=pf_cond, source="fields_only")
    except Exception as e:
        logger.error(f"Conditions gatekeeper failed for {venue}: {e}")

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

    # Reclassify trial→race if scrape found actual races (calendar may have been wrong)
    if not pf_failed:
        race_count = await db.execute(
            select(Race).where(Race.meeting_id == meeting_id).limit(1)
        )
        has_races = race_count.scalar_one_or_none() is not None
        if has_races and meeting.meeting_type == "trial":
            meeting.meeting_type = "race"
            logger.info(f"Races found for {venue} — reclassified from trial to race")

    # Fill derived fields (days_since_last_run, class_stats) from form_history
    await _fill_derived_fields(db, meeting_id)

    # Commit with retry — SQLite can transiently lock when monitor writes concurrently
    import asyncio as _aio
    for _attempt in range(3):
        try:
            await db.commit()
            break
        except Exception as commit_err:
            if _attempt < 2:
                logger.warning(f"Commit attempt {_attempt + 1} failed: {commit_err}, retrying...")
                await db.rollback()
                await _aio.sleep(1 + _attempt)
            else:
                logger.error(f"Final commit failed after 3 attempts: {commit_err}")
                errors.append(f"commit: {commit_err}")

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

        # Step 2: Conditions — single gatekeeper (RA authoritative, PF supplementary)
        pf_cond = None
        try:
            if pf_scraper is None:
                from punty.scrapers.punting_form import PuntingFormScraper
                pf_scraper = await PuntingFormScraper.from_settings(db)
            pf_cond = await pf_scraper.get_conditions_for_venue(venue)
        except Exception as e:
            logger.error(f"PF conditions failed: {e}")
            errors.append(f"conditions: {e}")
        try:
            await refresh_track_conditions(meeting, pf_cond=pf_cond, source="scrape_meeting")
        except Exception as e:
            logger.error(f"Conditions gatekeeper failed for {venue}: {e}")

        # Step 2c: HKJC track info for HK venues (RA/PF have no HK data)
        from punty.venues import is_international_venue, guess_state
        if is_international_venue(venue) and guess_state(venue) == "HK":
            try:
                from punty.scrapers.tab_playwright import HKJCTrackInfoScraper
                hkjc_track = HKJCTrackInfoScraper()
                track_info = await hkjc_track.scrape_track_info(race_date)
                if track_info:
                    _apply_hkjc_conditions(meeting, track_info)
                    logger.info(f"HKJC track info applied for {venue}: {track_info}")
            except Exception as e:
                logger.warning(f"HKJC track info failed for {venue}: {e}")

        # Step 2d: RA Free Fields cross-check (only when PF succeeded)
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

        # Step 4: Betfair exchange odds (fast API, no browser — fills current_odds gaps)
        try:
            from punty.scrapers.betfair import BetfairScraper
            bf = await BetfairScraper.from_settings(db)
            if bf:
                bf_odds = await bf.get_odds_for_meeting(venue, race_date, meeting_id)
                await _merge_betfair_odds(db, meeting_id, bf_odds)
                logger.info(f"Betfair odds merged: {len(bf_odds)} runners for {venue}")
        except Exception as e:
            logger.warning(f"Betfair odds failed for {venue}: {e}")
            errors.append(f"betfair: {e}")

        # Step 5: International venue odds (TAB Playwright → HKJC fallback)
        from punty.venues import get_tab_mnemonic
        if get_tab_mnemonic(venue):
            race_result = await db.execute(
                select(Race).where(Race.meeting_id == meeting_id)
            )
            intl_race_count = len(race_result.scalars().all())
            intl_odds = []

            # Try TAB first
            try:
                from punty.scrapers.tab_playwright import TabPlaywrightScraper
                tab_pw = TabPlaywrightScraper()
                intl_odds = await tab_pw.scrape_odds_for_meeting(
                    venue, race_date, meeting_id, intl_race_count
                )
                if intl_odds:
                    logger.info(f"TAB Playwright odds: {len(intl_odds)} runners for {venue}")
            except Exception as e:
                logger.warning(f"TAB Playwright odds failed for {venue}: {e}")

            # Fallback to PointsBet if TAB returned nothing
            if not intl_odds:
                try:
                    from punty.scrapers.pointsbet import PointsBetScraper
                    pb = PointsBetScraper()
                    intl_odds = await pb.scrape_odds_for_meeting(
                        venue, race_date, meeting_id, intl_race_count
                    )
                    if intl_odds:
                        logger.info(f"PointsBet odds: {len(intl_odds)} runners for {venue}")
                except Exception as e:
                    logger.warning(f"PointsBet odds failed for {venue}: {e}")

            # Fallback to HKJC if still nothing (HK venues only)
            if not intl_odds and guess_state(venue) == "HK":
                try:
                    from punty.scrapers.tab_playwright import HKJCOddsScraper
                    hkjc_odds = HKJCOddsScraper()
                    intl_odds = await hkjc_odds.scrape_odds_for_meeting(
                        venue, race_date, intl_race_count
                    )
                    if intl_odds:
                        logger.info(f"HKJC odds fallback: {len(intl_odds)} runners for {venue}")
                except Exception as e:
                    logger.warning(f"HKJC odds fallback failed for {venue}: {e}")
                    errors.append(f"hkjc_odds: {e}")

            if intl_odds:
                await _merge_tab_odds(db, meeting_id, intl_odds)
            else:
                logger.warning(f"International odds: no odds captured for {venue}")
                errors.append("intl_odds: no odds captured")

        # Close PF scraper only if we own it
        if owns_pf and pf_scraper:
            await pf_scraper.close()

        # Reclassify trial→race if scrape found actual races (calendar may have been wrong)
        pf_failed = any("primary" in e for e in errors)
        if not pf_failed:
            race_count = await db.execute(
                select(Race).where(Race.meeting_id == meeting_id).limit(1)
            )
            has_races = race_count.scalar_one_or_none() is not None
            if has_races and meeting.meeting_type == "trial":
                meeting.meeting_type = "race"
                logger.info(f"Races found for {venue} — reclassified from trial to race")

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
    # Check if this is an international venue (needs extra TAB Playwright step)
    from punty.venues import get_tab_mnemonic
    is_international = get_tab_mnemonic(venue) is not None
    total_steps = 6 if is_international else 5

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

        # Step 2: Conditions + weather — single gatekeeper (RA authoritative)
        yield {"step": 1, "total": total_steps, "label": "Fetching conditions/weather...", "status": "running"}
        try:
            pf_cond = None
            if not meeting.track_condition_locked:
                if pf_scraper is None:
                    from punty.scrapers.punting_form import PuntingFormScraper
                    pf_scraper = await PuntingFormScraper.from_settings(db)
                pf_cond = await pf_scraper.get_conditions_for_venue(venue)
            await refresh_track_conditions(meeting, pf_cond=pf_cond, source="full_scrape")
            if meeting.track_condition:
                yield {"step": 2, "total": total_steps,
                       "label": f"Conditions: {meeting.track_condition}" + (f" | Rain: {pf_cond.get('rainfall', 'N/A')}mm" if pf_cond else ""),
                       "status": "done"}
            else:
                yield {"step": 2, "total": total_steps, "label": "Conditions: not available", "status": "done"}
        except Exception as e:
            logger.error(f"Conditions failed: {e}")
            errors.append(f"conditions: {e}")
            yield {"step": 2, "total": total_steps, "label": f"Conditions failed: {e}", "status": "error"}

        # HKJC track info for HK venues
        from punty.venues import is_international_venue, guess_state
        if is_international_venue(venue) and guess_state(venue) == "HK":
            try:
                from punty.scrapers.tab_playwright import HKJCTrackInfoScraper
                hkjc_track = HKJCTrackInfoScraper()
                track_info = await hkjc_track.scrape_track_info(race_date)
                if track_info:
                    _apply_hkjc_conditions(meeting, track_info)
                    yield {"step": 2, "total": total_steps,
                           "label": f"HKJC track: wind {track_info.get('weather_wind_speed', '?')}km/h {track_info.get('weather_wind_dir', '?')}",
                           "status": "done"}
            except Exception as e:
                logger.warning(f"HKJC track info failed for {venue}: {e}")

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

        # Step 5: Betfair exchange odds (fast API, no browser)
        yield {"step": 5, "total": total_steps, "label": "Fetching Betfair exchange odds...", "status": "running"}
        try:
            from punty.scrapers.betfair import BetfairScraper
            bf = await BetfairScraper.from_settings(db)
            if bf:
                bf_odds = await bf.get_odds_for_meeting(venue, race_date, meeting_id)
                await _merge_betfair_odds(db, meeting_id, bf_odds)
                # Also fetch PLACE market odds
                try:
                    bf_place = await bf.fetch_place_odds(venue, race_date, meeting_id)
                    await _merge_betfair_place_odds(db, bf_place)
                except Exception as e:
                    logger.debug(f"Betfair PLACE market: {e}")
                yield {"step": 5, "total": total_steps,
                       "label": f"Betfair odds: {len(bf_odds)} runners", "status": "done"}
            else:
                yield {"step": 5, "total": total_steps,
                       "label": "Betfair: not configured, skipped", "status": "done"}
        except Exception as e:
            logger.warning(f"Betfair odds failed for {venue}: {e}")
            errors.append(f"betfair: {e}")
            yield {"step": 5, "total": total_steps,
                   "label": f"Betfair failed: {e}", "status": "error"}

        # Step 6: International venue odds (TAB Playwright → HKJC fallback)
        if is_international:
            yield {"step": 5, "total": total_steps, "label": "Scraping international odds...", "status": "running"}
            race_result = await db.execute(
                select(Race).where(Race.meeting_id == meeting_id)
            )
            intl_race_count = len(race_result.scalars().all())
            intl_odds = []

            # Try TAB first
            try:
                from punty.scrapers.tab_playwright import TabPlaywrightScraper
                tab_pw = TabPlaywrightScraper()
                intl_odds = await tab_pw.scrape_odds_for_meeting(
                    venue, race_date, meeting_id, intl_race_count
                )
                if intl_odds:
                    logger.info(f"TAB Playwright odds: {len(intl_odds)} runners for {venue}")
            except Exception as e:
                logger.warning(f"TAB Playwright odds failed for {venue}: {e}")

            # Fallback to PointsBet if TAB returned nothing
            if not intl_odds:
                try:
                    from punty.scrapers.pointsbet import PointsBetScraper
                    pb = PointsBetScraper()
                    intl_odds = await pb.scrape_odds_for_meeting(
                        venue, race_date, meeting_id, intl_race_count
                    )
                    if intl_odds:
                        logger.info(f"PointsBet odds: {len(intl_odds)} runners for {venue}")
                except Exception as e:
                    logger.warning(f"PointsBet odds failed for {venue}: {e}")

            # Fallback to HKJC if still nothing (HK venues only)
            if not intl_odds and guess_state(venue) == "HK":
                try:
                    from punty.scrapers.tab_playwright import HKJCOddsScraper
                    hkjc_odds_scraper = HKJCOddsScraper()
                    intl_odds = await hkjc_odds_scraper.scrape_odds_for_meeting(
                        venue, race_date, intl_race_count
                    )
                    if intl_odds:
                        logger.info(f"HKJC odds fallback: {len(intl_odds)} runners for {venue}")
                except Exception as e:
                    logger.warning(f"HKJC odds fallback failed for {venue}: {e}")
                    errors.append(f"hkjc_odds: {e}")

            if intl_odds:
                await _merge_tab_odds(db, meeting_id, intl_odds)
                yield {"step": 6, "total": total_steps,
                       "label": f"International odds: {len(intl_odds)} runners", "status": "done"}
            else:
                errors.append("intl_odds: no odds captured")
                yield {"step": 6, "total": total_steps,
                       "label": "International odds: no odds captured", "status": "error"}

        # Close scraper
        if pf_scraper:
            await pf_scraper.close()

        # Fill derived fields (days_since_last_run, class_stats) from form_history
        await _fill_derived_fields(db, meeting_id)

        # Commit with retry — SQLite can transiently lock when monitor writes concurrently
        import asyncio as _aio
        for _attempt in range(3):
            try:
                await db.commit()
                break
            except Exception as commit_err:
                if _attempt < 2:
                    logger.warning(f"Commit attempt {_attempt + 1} failed: {commit_err}, retrying...")
                    await db.rollback()
                    await _aio.sleep(1 + _attempt)
                else:
                    logger.error(f"Final commit failed after 3 attempts: {commit_err}")
                    errors.append(f"commit: {commit_err}")

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
            if positions_set > 0:
                await db.commit()  # Persist each race immediately (SSE may disconnect)
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
                    if positions_set > 0:
                        await db.commit()  # Persist each race immediately (SSE may disconnect)
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
    """Quick refresh of odds, conditions and scratchings for a meeting.

    Fetches fresh odds from racing.com via check_race_fields(), plus
    conditions/scratchings from PuntingForm and Racing Australia.
    """
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    odds_updated = 0

    try:
        from punty.scrapers.punting_form import PuntingFormScraper

        pf = await PuntingFormScraper.from_settings(db)
        if pf:
            try:
                # Refresh conditions — single gatekeeper (RA authoritative)
                pf_cond = await pf.get_conditions_for_venue(meeting.venue)
                await refresh_track_conditions(meeting, pf_cond=pf_cond, source="refresh_odds")

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

        # Commit conditions/scratchings before odds fetch
        await db.commit()

        # Refresh odds from Betfair Exchange API (fast HTTP, no Playwright)
        try:
            from punty.scrapers.betfair import BetfairScraper
            bf = await BetfairScraper.from_settings(db)
            if bf:
                bf_odds = await bf.get_odds_for_meeting(
                    meeting.venue, meeting.date, meeting_id
                )
                if bf_odds:
                    await _merge_betfair_odds(db, meeting_id, bf_odds)
                    # Also fetch PLACE market odds for PVR accuracy
                    bf_place_odds = {}
                    try:
                        bf_place = await bf.fetch_place_odds(
                            meeting.venue, meeting.date, meeting_id
                        )
                        for po in bf_place:
                            key = (po["race_id"], po["horse_name"])
                            bf_place_odds[key] = po["place_odds_betfair"]
                    except Exception as e:
                        logger.debug(f"Betfair PLACE market fetch failed: {e}")

                    # Update current_odds for runners where Betfair is
                    # the freshest/only source
                    for od in bf_odds:
                        result = await db.execute(
                            select(Runner).where(Runner.race_id == od["race_id"])
                            .where(Runner.horse_name == od["horse_name"])
                            .limit(1)
                        )
                        runner = result.scalar_one_or_none()
                        if not runner:
                            continue
                        # Apply Betfair scratching detection
                        if od.get("scratched"):
                            if not runner.scratched:
                                runner.scratched = True
                                logger.info(
                                    f"Scratched {runner.horse_name} via Betfair "
                                    f"REMOVED status ({od['race_id']})"
                                )
                            continue
                        if runner:
                            bf_price = od["odds_betfair"]
                            # Reject suspiciously low Betfair odds (thin liquidity)
                            if bf_price < 1.10:
                                logger.warning(
                                    f"Betfair odds rejected: {runner.horse_name} "
                                    f"({od['race_id']}) at ${bf_price:.2f} — below $1.10 floor"
                                )
                                continue
                            # Reject extreme divergence from opening odds (>5x ratio)
                            if runner.opening_odds and runner.opening_odds > 1.5:
                                ratio = runner.opening_odds / bf_price
                                if ratio > 5.0:
                                    logger.warning(
                                        f"Betfair odds rejected: {runner.horse_name} "
                                        f"({od['race_id']}) ${bf_price:.2f} vs opening "
                                        f"${runner.opening_odds:.2f} — {ratio:.1f}x divergence"
                                    )
                                    continue
                            runner.current_odds = bf_price
                            odds_updated += 1
                            if not runner.opening_odds:
                                runner.opening_odds = bf_price
                            # Use real Betfair PLACE odds if available, else estimate
                            place_key = (od["race_id"], od["horse_name"])
                            real_place = bf_place_odds.get(place_key)
                            if real_place and real_place > 1.0:
                                runner.place_odds = real_place
                            else:
                                runner.place_odds = round((bf_price - 1) / 3 + 1, 2)
                    logger.info(
                        f"Odds refresh for {meeting_id}: updated {odds_updated} "
                        f"runners from Betfair ({len(bf_place_odds)} with real place odds)"
                    )
            else:
                logger.info(f"Betfair not configured — skipping odds refresh for {meeting_id}")
        except Exception as e:
            logger.warning(f"Betfair odds refresh failed for {meeting_id}: {e}")

        # International venues (HK etc): TAB Playwright → HKJC fallback
        # Betfair is AU-only so never returns odds for these venues.
        from punty.venues import get_tab_mnemonic, guess_state
        tab_info = get_tab_mnemonic(meeting.venue)
        if tab_info and odds_updated == 0:
            race_result = await db.execute(
                select(Race).where(Race.meeting_id == meeting_id)
            )
            intl_race_count = len(race_result.scalars().all())
            intl_odds = []

            try:
                from punty.scrapers.tab_playwright import TabPlaywrightScraper
                tab_pw = TabPlaywrightScraper()
                intl_odds = await tab_pw.scrape_odds_for_meeting(
                    meeting.venue, meeting.date, meeting_id, intl_race_count
                )
                if intl_odds:
                    logger.info(f"[refresh_odds] TAB Playwright: {len(intl_odds)} runners for {meeting.venue}")
            except Exception as e:
                logger.warning(f"[refresh_odds] TAB Playwright failed for {meeting.venue}: {e}")

            # Fallback to PointsBet if TAB returned nothing
            if not intl_odds:
                try:
                    from punty.scrapers.pointsbet import PointsBetScraper
                    pb = PointsBetScraper()
                    intl_odds = await pb.scrape_odds_for_meeting(
                        meeting.venue, meeting.date, meeting_id, intl_race_count
                    )
                    if intl_odds:
                        logger.info(f"[refresh_odds] PointsBet: {len(intl_odds)} runners for {meeting.venue}")
                except Exception as e:
                    logger.warning(f"[refresh_odds] PointsBet failed for {meeting.venue}: {e}")

            # Fallback to HKJC if still nothing (HK venues only)
            if not intl_odds and guess_state(meeting.venue) == "HK":
                try:
                    from punty.scrapers.tab_playwright import HKJCOddsScraper
                    hkjc_odds_scraper = HKJCOddsScraper()
                    intl_odds = await hkjc_odds_scraper.scrape_odds_for_meeting(
                        meeting.venue, meeting.date, intl_race_count
                    )
                    if intl_odds:
                        logger.info(f"[refresh_odds] HKJC odds fallback: {len(intl_odds)} runners for {meeting.venue}")
                except Exception as e:
                    logger.warning(f"[refresh_odds] HKJC odds fallback failed for {meeting.venue}: {e}")

            if intl_odds:
                await _merge_tab_odds(db, meeting_id, intl_odds)
                odds_updated = len(intl_odds)

        # Fallback: if Betfair had no markets, use PF assessed price for
        # runners missing current_odds (e.g. Sunshine Coast, small regionals)
        if odds_updated == 0:
            pf_fallback = 0
            all_runners = await db.execute(
                select(Runner).where(
                    Runner.race_id.like(f"{meeting_id}-r%"),
                    Runner.scratched == False,
                    Runner.current_odds.is_(None),
                )
            )
            for runner in all_runners.scalars().all():
                price = runner.pf_assessed_price or runner.pf_ai_price
                if price and price > 1.0:
                    runner.current_odds = round(price, 2)
                    if not runner.opening_odds:
                        runner.opening_odds = round(price, 2)
                    runner.place_odds = round((price - 1) / 3 + 1, 2)
                    pf_fallback += 1
            if pf_fallback:
                logger.info(
                    f"No Betfair markets for {meeting_id} — "
                    f"used PF assessed price for {pf_fallback} runners"
                )
                odds_updated = pf_fallback

        await db.commit()
        return {"meeting_id": meeting_id, "status": "ok", "odds_updated": odds_updated}
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
                if _should_update_condition(val, meeting.track_condition):
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


async def refresh_track_conditions(
    meeting: Meeting,
    pf_cond: dict | None = None,
    source: str = "unknown",
) -> str | None:
    """Single gatekeeper for all track condition updates.

    Racing Australia is the single source of truth. PuntingForm provides
    supplementary fields (weather details, going stick) but RA always wins
    for track_condition, rail, and weather when available.

    Args:
        meeting: Meeting object to update (mutated in place).
        pf_cond: Optional pre-fetched PuntingForm conditions dict.
        source: Caller label for logging (e.g. "pre_race", "monitor").

    Returns:
        Final track_condition value or None if locked/unchanged.
    """
    if meeting.track_condition_locked:
        logger.info(f"[{source}] Track condition locked for {meeting.venue}: {meeting.track_condition!r}")
        return meeting.track_condition

    old_tc = meeting.track_condition

    # Step 1: Apply PF supplementary fields (weather details, going stick, etc.)
    if pf_cond:
        _apply_supplementary_fields(meeting, pf_cond)

    # Step 2: Try Racing Australia (authoritative for track_condition, rail, weather)
    ra_cond = None
    try:
        from punty.scrapers.track_conditions import get_conditions_for_meeting
        ra_cond = await get_conditions_for_meeting(meeting.venue)
    except Exception as e:
        logger.warning(f"[{source}] RA fetch failed for {meeting.venue}: {e}")

    if ra_cond:
        # RA is authoritative — apply track_condition, rail, weather, penetrometer
        new_cond = ra_cond.get("condition")
        if new_cond:
            meeting.track_condition = new_cond
        if ra_cond.get("rail"):
            meeting.rail_position = ra_cond["rail"]
        if ra_cond.get("weather"):
            meeting.weather = ra_cond["weather"]
        if ra_cond.get("penetrometer") is not None:
            meeting.penetrometer = ra_cond["penetrometer"]
        _apply_irrigation(meeting, ra_cond)
        if ra_cond.get("rainfall") is not None:
            meeting.rainfall = ra_cond["rainfall"]
        logger.info(f"[{source}] RA conditions for {meeting.venue}: {old_tc!r} → {meeting.track_condition!r}")
    elif pf_cond:
        # RA unavailable — fall back to PF for track_condition
        new_cond = pf_cond.get("condition")
        if new_cond and _should_update_condition(new_cond, meeting.track_condition):
            meeting.track_condition = new_cond
        meeting.rail_position = pf_cond.get("rail") or meeting.rail_position
        meeting.weather = pf_cond.get("weather") or meeting.weather
        if pf_cond.get("penetrometer") is not None:
            meeting.penetrometer = pf_cond["penetrometer"]
        _apply_irrigation(meeting, pf_cond)
        if pf_cond.get("rainfall") is not None:
            meeting.rainfall = pf_cond["rainfall"]
        logger.info(f"[{source}] PF conditions for {meeting.venue} (RA unavailable): {old_tc!r} → {meeting.track_condition!r}")

    return meeting.track_condition


def _apply_supplementary_fields(meeting: Meeting, cond: dict) -> None:
    """Apply PuntingForm supplementary weather fields (not track_condition)."""
    if cond.get("wind_speed") is not None:
        meeting.weather_wind_speed = cond["wind_speed"]
    if cond.get("wind_direction"):
        meeting.weather_wind_dir = cond["wind_direction"]
    if cond.get("humidity") is not None:
        meeting.weather_humidity = cond["humidity"]
    if cond.get("going_stick") is not None:
        meeting.going_stick = cond["going_stick"]


def _apply_irrigation(meeting: Meeting, cond: dict) -> None:
    """Apply irrigation field from any source."""
    if cond.get("irrigation") is not None:
        irr = cond["irrigation"]
        if isinstance(irr, str):
            meeting.irrigation = bool(irr and "nil" not in irr.lower() and irr.strip() != "0")
        else:
            meeting.irrigation = bool(irr)


def _apply_pf_conditions(meeting: Meeting, cond: dict) -> None:
    """Legacy: Apply PF conditions directly. Prefer refresh_track_conditions()."""
    new_cond = cond.get("condition")
    if new_cond and _should_update_condition(new_cond, meeting.track_condition):
        logger.info(f"Condition for {meeting.venue}: {meeting.track_condition!r} → {new_cond!r}")
        meeting.track_condition = new_cond
    meeting.rail_position = cond.get("rail") or meeting.rail_position
    meeting.weather = cond.get("weather") or meeting.weather
    if cond.get("penetrometer") is not None:
        meeting.penetrometer = cond["penetrometer"]
    _apply_supplementary_fields(meeting, cond)
    _apply_irrigation(meeting, cond)
    if cond.get("rainfall") is not None:
        meeting.rainfall = cond["rainfall"]


def _apply_ra_conditions(meeting: Meeting, cond: dict) -> None:
    """Legacy: Apply RA conditions directly. Prefer refresh_track_conditions()."""
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
    _apply_irrigation(meeting, cond)


def _apply_hkjc_conditions(meeting: Meeting, info: dict) -> None:
    """Apply HKJC wind tracker / track info data to a Meeting object."""
    if info.get("weather_wind_speed") is not None:
        meeting.weather_wind_speed = info["weather_wind_speed"]
    if info.get("weather_wind_dir"):
        meeting.weather_wind_dir = info["weather_wind_dir"]
    if info.get("weather_condition"):
        meeting.weather_condition = info["weather_condition"]
        if not meeting.weather:
            meeting.weather = info["weather_condition"]
    if info.get("weather_temp") is not None:
        meeting.weather_temp = info["weather_temp"]
    if info.get("weather_humidity") is not None:
        meeting.weather_humidity = info["weather_humidity"]
    # HKJC is the primary source for HK track conditions (RA returns 500)
    if info.get("track_condition"):
        meeting.track_condition = info["track_condition"]
    if info.get("penetrometer") is not None:
        meeting.penetrometer = info["penetrometer"]
    if info.get("rainfall") is not None:
        meeting.rainfall = f"{info['rainfall']}mm"
    # Store course config and soil moisture in rail_position if available
    course = info.get("course_config")
    soil = info.get("soil_moisture")
    if course:
        meeting.rail_position = f'"{course}" Course'
        if soil is not None:
            meeting.rail_position += f" (Soil {soil}%)"


import re as _re

def _normalise_jockey(name: str) -> str:
    """Normalise jockey name for comparison: strip Ms/Mr prefix, claim weights, (late alt), Mc spacing."""
    s = name.strip()
    s = _re.sub(r"^(Ms|Mr)\s+", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r"\s*\(a[\d.]*/\d+kg\)", "", s)  # claim weight e.g. (a0/53kg), (a1.5/52kg)
    s = _re.sub(r",?\s*\(late\s+alt\)", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r"\bMc\s+", "Mc", s)  # "Mc Dougall" → "McDougall"
    return s.strip()


def _normalise_class(val: str) -> str:
    """Normalise race class for comparison: uppercase, strip trailing semicolons/punctuation."""
    return val.strip().rstrip(";").strip().upper()


def _xcheck_equal(field: str, pf_val, ra_val) -> bool:
    """Compare PF and RA values with field-specific normalisation."""
    if field == "class_":
        return _normalise_class(str(pf_val)) == _normalise_class(str(ra_val))
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
    """Merge racing.com odds, comments, and expert tips into existing data.

    Only updates odds and comment fields — does NOT overwrite primary runner data.
    Also stores expert tips on Race records.
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

    # Store expert tips on Race records
    for race_data in data.get("races", []):
        tips = race_data.get("expert_tips")
        if tips:
            race_id = race_data.get("id")
            if race_id:
                result = await db.execute(
                    select(Race).where(Race.id == race_id).limit(1)
                )
                race = result.scalar_one_or_none()
                if race:
                    race.expert_tips = _json.dumps(tips)

    await db.flush()


async def _merge_betfair_odds(db: AsyncSession, meeting_id: str, odds_data: list[dict]) -> None:
    """Merge Betfair exchange odds into existing runner records.

    Sets odds_betfair, and fills current_odds as fallback when no other provider has odds.
    """
    import re

    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())

    if not odds_data:
        return

    matched = 0
    filled_current = 0

    for item in odds_data:
        race_id = item.get("race_id", "")
        horse_name = item.get("horse_name", "")
        price = item.get("odds_betfair")
        if not (race_id and horse_name and price):
            continue

        # Try exact match first
        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.horse_name == horse_name,
            ).limit(1)
        )
        runner = result.scalar_one_or_none()

        # Fuzzy match if exact fails
        if not runner:
            norm_bf = _normalize(horse_name)
            result = await db.execute(
                select(Runner).where(Runner.race_id == race_id)
            )
            candidates = result.scalars().all()
            for c in candidates:
                if _normalize(c.horse_name) == norm_bf:
                    runner = c
                    break

        if not runner:
            continue

        # Reject suspiciously low Betfair odds (thin liquidity)
        if price < 1.10:
            logger.warning(
                f"Betfair merge rejected: {horse_name} ({race_id}) "
                f"at ${price:.2f} — below $1.10 floor"
            )
            continue
        runner.odds_betfair = price
        matched += 1

        # Fill current_odds if no other provider has odds
        if not runner.current_odds or runner.current_odds <= 1.0:
            runner.current_odds = price
            filled_current += 1

    await db.flush()
    logger.info(
        f"Betfair merge: {matched}/{len(odds_data)} matched, "
        f"{filled_current} filled current_odds"
    )


async def _merge_betfair_place_odds(db: AsyncSession, place_data: list[dict]) -> None:
    """Merge Betfair PLACE market odds into runner records.

    Overwrites estimated place_odds with real exchange place prices.
    """
    import re

    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())

    if not place_data:
        return

    matched = 0
    for item in place_data:
        race_id = item.get("race_id", "")
        horse_name = item.get("horse_name", "")
        price = item.get("place_odds_betfair")
        if not (race_id and horse_name and price and price > 1.0):
            continue

        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.horse_name == horse_name,
            ).limit(1)
        )
        runner = result.scalar_one_or_none()

        if not runner:
            norm_bf = _normalize(horse_name)
            result = await db.execute(
                select(Runner).where(Runner.race_id == race_id)
            )
            for c in result.scalars().all():
                if _normalize(c.horse_name) == norm_bf:
                    runner = c
                    break

        if runner:
            runner.place_odds = price
            matched += 1

    if matched:
        await db.flush()
        logger.info(f"Betfair PLACE merge: {matched}/{len(place_data)} matched")


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


async def _merge_tab_odds(db: AsyncSession, meeting_id: str, odds_data: list[dict]) -> None:
    """Merge TAB Playwright odds into existing runner records.

    Matches by saddlecloth number first (reliable for HK), then fuzzy horse name.
    Sets current_odds, opening_odds, place_odds, odds_tab.
    """
    import re

    MAX_VALID_ODDS = 501.0

    def _normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", name.strip().lower())

    if not odds_data:
        return

    matched = 0
    filled = 0

    for item in odds_data:
        race_num = item.get("race_number")
        horse_name = item.get("horse_name", "")
        saddlecloth = item.get("saddlecloth")
        win_odds = item.get("current_odds")
        place_odds = item.get("place_odds")
        opening_odds = item.get("opening_odds")

        if not race_num:
            continue

        race_id = f"{meeting_id}-r{race_num}"

        runner = None

        # Strategy 1: Match by saddlecloth (most reliable for HK)
        if saddlecloth:
            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.saddlecloth == saddlecloth,
                ).limit(1)
            )
            runner = result.scalar_one_or_none()

        # Strategy 2: Exact horse name match
        if not runner and horse_name:
            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.horse_name == horse_name,
                ).limit(1)
            )
            runner = result.scalar_one_or_none()

        # Strategy 3: Fuzzy horse name match
        if not runner and horse_name:
            norm_tab = _normalize(horse_name)
            result = await db.execute(
                select(Runner).where(Runner.race_id == race_id)
            )
            candidates = result.scalars().all()
            for c in candidates:
                if _normalize(c.horse_name) == norm_tab:
                    runner = c
                    break

        if not runner:
            continue

        matched += 1

        # Apply odds (with MAX_VALID_ODDS guard)
        # For international venues, TAB/PointsBet/HKJC are the only real
        # odds sources — always overwrite current_odds (may be stale/wrong)
        if win_odds and 1.0 < win_odds <= MAX_VALID_ODDS:
            runner.odds_tab = win_odds
            if not runner.current_odds or runner.current_odds <= 1.0:
                filled += 1
            runner.current_odds = win_odds

        if opening_odds and 1.0 < opening_odds <= MAX_VALID_ODDS:
            if not runner.opening_odds:
                runner.opening_odds = opening_odds

        if place_odds and 1.0 < place_odds <= MAX_VALID_ODDS:
            runner.place_odds = place_odds

        if item.get("scratched"):
            runner.scratched = True

    await db.flush()
    logger.info(
        f"TAB odds merge: {matched}/{len(odds_data)} matched, "
        f"{filled} filled current_odds"
    )


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
        elif result.get("position") in (1, 2, 3) and not runner.win_dividend:
            # Fallback chain for win dividend when tote unavailable
            # (racing.com doesn't serve tote dividends for some NSW venues)
            # Priority: Betfair back odds > SP
            fallback_win = None
            if runner.odds_betfair and runner.odds_betfair > 1.0:
                fallback_win = runner.odds_betfair
            elif result.get("starting_price"):
                try:
                    fallback_win = float(str(result["starting_price"]).replace("$", "").replace(",", ""))
                except (ValueError, TypeError):
                    pass
            if fallback_win and fallback_win > 1.0 and result["position"] == 1:
                runner.win_dividend = fallback_win
                source = "Betfair" if (runner.odds_betfair and runner.odds_betfair > 1.0) else "SP"
                logger.info(f"{race_id}: win_dividend from {source} ${fallback_win:.2f}")

        if result.get("place_dividend") is not None:
            runner.place_dividend = result["place_dividend"]
        elif not runner.place_dividend and result.get("position") in (1, 2, 3):
            # Determine paying places from field size (TAB rules)
            # Count runners with positions in this result set
            active_count = sum(
                1 for r in results_data.get("results", [])
                if r.get("position") is not None
            )
            paying_places = 2 if active_count <= 7 else 3
            if active_count <= 4:
                paying_places = 0

            # Only estimate place dividend if this position actually pays
            if result["position"] <= paying_places:
                # Fallback chain for place dividend: Betfair > SP > current_odds
                # Estimate: place_div ≈ (win_odds - 1) / paying_places + 1
                fallback_odds = None
                source = None
                if runner.odds_betfair and runner.odds_betfair > 1.0:
                    fallback_odds = runner.odds_betfair
                    source = "Betfair"
                elif result.get("starting_price"):
                    try:
                        fallback_odds = float(str(result["starting_price"]).replace("$", "").replace(",", ""))
                        source = "SP"
                    except (ValueError, TypeError):
                        pass
                if not fallback_odds and runner.current_odds and runner.current_odds > 1.0:
                    fallback_odds = runner.current_odds
                    source = "current_odds"

                if fallback_odds and fallback_odds > 1.0:
                    est_place = round((fallback_odds - 1.0) / paying_places + 1.0, 2)
                    runner.place_dividend = est_place
                    logger.info(
                        f"{race_id}: place_dividend from {source} ${fallback_odds:.2f} "
                        f"→ est ${est_place:.2f} (pos {result['position']})"
                    )
            else:
                logger.info(
                    f"{race_id}: No place dividend for pos {result['position']} "
                    f"({active_count}-runner field, {paying_places} places paid)"
                )
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


async def _check_pf_trial(pf_scraper, venue: str, race_date) -> bool:
    """Check PF's isBarrierTrial flag for a meeting. Returns True if PF says it's a trial."""
    if not pf_scraper:
        return True  # Can't check — don't reclassify
    try:
        meetings = await pf_scraper.get_meetings(race_date)
        # Strip sponsor prefixes for matching
        clean = venue.lower().strip()
        for prefix in ("sportsbet ", "ladbrokes ", "bet365 ", "aquis ", "picklebet park ", "southside ", "tab "):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        for m in meetings:
            track_name = m.get("track", {}).get("name", "").lower().strip()
            if track_name == clean or track_name in clean or clean in track_name:
                return m.get("isBarrierTrial", False)
    except Exception as e:
        logger.warning(f"Could not check PF trial flag for {venue}: {e}")
    return True  # Can't determine — don't reclassify


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
