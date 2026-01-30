"""Scraping orchestrator — coordinates scrapers and stores results."""

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.meeting import Meeting, Race, Runner

logger = logging.getLogger(__name__)


async def scrape_calendar(db: AsyncSession) -> list[dict]:
    """Scrape today's calendar and populate meetings in DB.

    Returns the list of meeting dicts created/updated.
    """
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
            # Update source if needed
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
    """Run all scrapers for a selected meeting and merge data into DB.

    Scrapes: racing.com (form), racingandsports, racenet, track conditions, TAB odds.
    """
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    venue = meeting.venue
    race_date = meeting.date
    errors = []

    # 1. Primary form data — racing.com
    try:
        from punty.scrapers.racing_com import RacingComScraper
        scraper = RacingComScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _upsert_meeting_data(db, data)
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racing.com scrape failed: {e}")
        errors.append(f"racing.com: {e}")

    # 2. Racing & Sports form guide
    try:
        from punty.scrapers.racing_sports import RacingSportsScraper
        scraper = RacingSportsScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _merge_runner_data(db, meeting_id, data.get("runners", []))
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racingandsports scrape failed: {e}")
        errors.append(f"racingandsports: {e}")

    # 3. Racenet form guide
    try:
        from punty.scrapers.racenet import RacenetScraper
        scraper = RacenetScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _merge_runner_data(db, meeting_id, data.get("runners", []))
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racenet scrape failed: {e}")
        errors.append(f"racenet: {e}")

    # 4. Track conditions
    try:
        from punty.scrapers.track_conditions import scrape_track_conditions
        state = _guess_state(venue)
        if state:
            conditions = await scrape_track_conditions(state)
            for cond in conditions:
                if cond["venue"] and cond["venue"].lower() == venue.lower():
                    meeting.track_condition = cond.get("condition") or meeting.track_condition
                    meeting.rail_position = cond.get("rail") or meeting.rail_position
                    meeting.weather = cond.get("weather") or meeting.weather
                    break
    except Exception as e:
        logger.error(f"track conditions scrape failed: {e}")
        errors.append(f"track_conditions: {e}")

    # 5. TAB odds
    try:
        from punty.scrapers.tab import TabScraper
        scraper = TabScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _merge_odds(db, meeting_id, data.get("runners_odds", []))
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"TAB scrape failed: {e}")
        errors.append(f"tab: {e}")

    await db.commit()
    return {"meeting_id": meeting_id, "errors": errors}


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

async def _upsert_meeting_data(db: AsyncSession, data: dict) -> None:
    """Upsert meeting, races, and runners from scraper output."""
    m = data["meeting"]
    existing = await db.get(Meeting, m["id"])
    if existing:
        existing.track_condition = m.get("track_condition") or existing.track_condition
        existing.weather = m.get("weather") or existing.weather
        existing.rail_position = m.get("rail_position") or existing.rail_position
    else:
        db.add(Meeting(
            id=m["id"],
            venue=m["venue"],
            date=m["date"],
            track_condition=m.get("track_condition"),
            weather=m.get("weather"),
            rail_position=m.get("rail_position"),
            source="racing.com",
        ))

    for r in data.get("races", []):
        existing_race = await db.get(Race, r["id"])
        if not existing_race:
            db.add(Race(**{k: v for k, v in r.items() if k != "class_" and k != "class"},
                        class_=r.get("class_")))

    for runner in data.get("runners", []):
        existing_runner = await db.get(Runner, runner["id"])
        if not existing_runner:
            db.add(Runner(**runner))


async def _merge_runner_data(db: AsyncSession, meeting_id: str, runners: list[dict]) -> None:
    """Merge supplementary runner data (fill blanks, don't overwrite)."""
    for r in runners:
        existing = await db.get(Runner, r.get("id", ""))
        if not existing:
            # Try to match by race + horse name
            race_id = r.get("race_id", "")
            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.horse_name == r.get("horse_name"),
                )
            )
            existing = result.scalar_one_or_none()

        if existing:
            # Fill blanks only
            for field in ["jockey", "trainer", "form", "weight", "comments", "barrier"]:
                if not getattr(existing, field, None) and r.get(field):
                    setattr(existing, field, r[field])


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
        "randwick": "NSW", "rosehill": "NSW", "warwick farm": "NSW",
        "canterbury": "NSW", "newcastle": "NSW",
        "doomben": "QLD", "eagle farm": "QLD", "gold coast": "QLD",
        "morphettville": "SA",
        "ascot": "WA", "belmont": "WA",
    }
    return mapping.get(venue.lower(), "VIC")
