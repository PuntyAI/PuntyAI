"""Scraping orchestrator — coordinates scrapers and stores results."""

import logging
from datetime import date
from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.meeting import Meeting, Race, Runner

logger = logging.getLogger(__name__)

# All Runner fields that come from the scraper
RUNNER_FIELDS = [
    "weight", "jockey", "trainer", "form", "current_odds", "opening_odds",
    "scratched", "comments", "horse_age", "horse_sex", "horse_colour",
    "sire", "dam", "dam_sire", "career_prize_money", "last_five",
    "days_since_last_run", "handicap_rating", "speed_value",
    "track_dist_stats", "track_stats", "distance_stats",
    "first_up_stats", "second_up_stats", "good_track_stats",
    "soft_track_stats", "heavy_track_stats", "jockey_stats", "class_stats",
    "gear", "gear_changes", "stewards_comment", "comment_long", "comment_short",
    "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes",
    "odds_betfair", "odds_flucs", "trainer_location",
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
    "weather_wind_speed", "weather_wind_dir", "rail_bias_comment",
]


async def scrape_calendar(db: AsyncSession) -> list[dict]:
    """Scrape today's calendar and populate meetings in DB."""
    from punty.scrapers.calendar import scrape_calendar as _scrape

    today = date.today()
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
            results.append(existing.to_dict())
            continue

        meeting = Meeting(
            id=meeting_id,
            venue=venue,
            date=today,
            selected=False,
            source="racing.com/calendar",
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
        })

    await db.commit()
    logger.info(f"Calendar scrape complete: {len(results)} meetings")
    return results


async def scrape_meeting_full(meeting_id: str, db: AsyncSession) -> dict:
    """Run all scrapers for a selected meeting and merge data into DB."""
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    venue = meeting.venue
    race_date = meeting.date
    errors = []

    # Primary form data — racing.com GraphQL
    try:
        from punty.scrapers.racing_com import RacingComScraper
        scraper = RacingComScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _upsert_meeting_data(db, meeting, data)
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racing.com scrape failed: {e}")
        errors.append(f"racing.com: {e}")

    # Track conditions (supplementary)
    try:
        from punty.scrapers.track_conditions import scrape_track_conditions
        state = _guess_state(venue)
        if state:
            conditions = await scrape_track_conditions(state)
            for cond in conditions:
                if cond["venue"] and cond["venue"].lower() in venue.lower():
                    meeting.track_condition = cond.get("condition") or meeting.track_condition
                    meeting.rail_position = cond.get("rail") or meeting.rail_position
                    meeting.weather = cond.get("weather") or meeting.weather
                    break
    except Exception as e:
        logger.error(f"track conditions scrape failed: {e}")
        errors.append(f"track_conditions: {e}")

    await db.commit()
    return {"meeting_id": meeting_id, "errors": errors}


async def scrape_meeting_full_stream(meeting_id: str, db: AsyncSession) -> AsyncGenerator[dict, None]:
    """Run all scrapers for a meeting, yielding progress events."""
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        yield {"step": 0, "total": 1, "label": "Meeting not found", "status": "error"}
        return

    venue = meeting.venue
    race_date = meeting.date
    errors = []
    total_steps = 3

    # Step 1: racing.com GraphQL data (form + speed maps + odds in one pass)
    yield {"step": 0, "total": total_steps, "label": "Scraping racing.com GraphQL data...", "status": "running"}
    try:
        from punty.scrapers.racing_com import RacingComScraper
        scraper = RacingComScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            race_count = len(data.get("races", []))
            runner_count = len(data.get("runners", []))
            await _upsert_meeting_data(db, meeting, data)
            yield {"step": 1, "total": total_steps,
                   "label": f"GraphQL complete — {race_count} races, {runner_count} runners", "status": "done"}
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racing.com scrape failed: {e}")
        errors.append(f"racing.com: {e}")
        yield {"step": 1, "total": total_steps, "label": f"racing.com failed: {e}", "status": "error"}

    # Step 2: Track conditions (supplementary)
    yield {"step": 1, "total": total_steps, "label": "Scraping track conditions...", "status": "running"}
    try:
        from punty.scrapers.track_conditions import scrape_track_conditions
        state = _guess_state(venue)
        if state:
            conditions = await scrape_track_conditions(state)
            found = False
            for cond in conditions:
                if cond["venue"] and cond["venue"].lower() in venue.lower():
                    meeting.track_condition = cond.get("condition") or meeting.track_condition
                    meeting.rail_position = cond.get("rail") or meeting.rail_position
                    meeting.weather = cond.get("weather") or meeting.weather
                    found = True
                    break
            if found:
                yield {"step": 2, "total": total_steps, "label": f"Track conditions: {meeting.track_condition or 'N/A'}", "status": "done"}
            else:
                yield {"step": 2, "total": total_steps, "label": "Track conditions: venue not found in data", "status": "done"}
        else:
            yield {"step": 2, "total": total_steps, "label": "Track conditions: unknown state", "status": "done"}
    except Exception as e:
        logger.error(f"track conditions scrape failed: {e}")
        errors.append(f"track_conditions: {e}")
        yield {"step": 2, "total": total_steps, "label": f"Track conditions failed: {e}", "status": "error"}

    # Step 3: TAB odds (supplementary — GraphQL already has multi-provider odds)
    yield {"step": 2, "total": total_steps, "label": "Scraping TAB odds...", "status": "running"}
    try:
        from punty.scrapers.tab import TabScraper
        scraper = TabScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _merge_odds(db, meeting_id, data.get("runners_odds", []))
        finally:
            await scraper.close()
        yield {"step": 3, "total": total_steps, "label": "TAB odds complete", "status": "done"}
    except Exception as e:
        logger.error(f"TAB scrape failed: {e}")
        errors.append(f"tab: {e}")
        yield {"step": 3, "total": total_steps, "label": f"TAB odds failed: {e}", "status": "error"}

    await db.commit()
    error_count = len(errors)
    if error_count:
        yield {"step": total_steps, "total": total_steps, "label": f"Complete with {error_count} error(s)", "status": "complete", "errors": errors}
    else:
        yield {"step": total_steps, "total": total_steps, "label": "All scrapers complete!", "status": "complete", "errors": []}


async def scrape_speed_maps_stream(meeting_id: str, db: AsyncSession) -> AsyncGenerator[dict, None]:
    """Scrape speed maps for all races in a meeting, yielding progress events."""
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

    try:
        from punty.scrapers.racing_com import RacingComScraper
        scraper = RacingComScraper()
        try:
            async for event in scraper.scrape_speed_maps(meeting.venue, meeting.date, race_count):
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

                yield {k: v for k, v in event.items() if k != "positions"}
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"Speed map scrape failed: {e}")
        yield {"step": 0, "total": 1, "label": f"Speed map scrape failed: {e}", "status": "error"}

    await db.commit()


async def refresh_odds(meeting_id: str, db: AsyncSession) -> dict:
    """Quick refresh of odds and scratchings for a meeting."""
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    try:
        from punty.scrapers.tab import TabScraper
        scraper = TabScraper()
        try:
            data = await scraper.scrape_meeting(meeting.venue, meeting.date)
            await _merge_odds(db, meeting_id, data.get("runners_odds", []))
        finally:
            await scraper.close()
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
        existing_runner = await db.get(Runner, runner["id"])
        if existing_runner:
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
            )
        )
        runner = result.scalar_one_or_none()
        if runner:
            if odds.get("current_odds") is not None:
                runner.current_odds = odds["current_odds"]
            if odds.get("opening_odds") is not None and not runner.opening_odds:
                runner.opening_odds = odds["opening_odds"]
            if odds.get("scratched"):
                runner.scratched = True
                runner.scratching_reason = odds.get("scratching_reason") or runner.scratching_reason


def _guess_state(venue: str) -> str:
    """Guess state from venue name."""
    mapping = {
        "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC",
        "sandown": "VIC", "cranbourne": "VIC", "pakenham": "VIC",
        "southside cranbourne": "VIC",
        "randwick": "NSW", "rosehill": "NSW", "warwick farm": "NSW",
        "canterbury": "NSW", "canterbury park": "NSW", "newcastle": "NSW",
        "doomben": "QLD", "eagle farm": "QLD", "gold coast": "QLD",
        "dalby": "QLD",
        "morphettville": "SA",
        "ascot": "WA", "belmont": "WA",
        "launceston": "TAS", "hobart": "TAS",
    }
    return mapping.get(venue.lower(), "VIC")
