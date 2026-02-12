"""Future Group race discovery and nomination scraping via PuntingForm API."""

import logging
import re
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today, melb_now_naive
from punty.models.future_race import FutureRace, FutureNomination

logger = logging.getLogger(__name__)

# Group race detection patterns
GROUP_PATTERNS = [
    (r"group\s*1\b", "Group 1"),
    (r"gr\.?\s*1\b", "Group 1"),
    (r"\(g1\)", "Group 1"),
    (r"group\s*2\b", "Group 2"),
    (r"gr\.?\s*2\b", "Group 2"),
    (r"\(g2\)", "Group 2"),
    (r"group\s*3\b", "Group 3"),
    (r"gr\.?\s*3\b", "Group 3"),
    (r"\(g3\)", "Group 3"),
    (r"\blisted\b", "Listed"),
]

# Prize money threshold for catching unlabelled Group races (AUD)
GROUP_PRIZE_THRESHOLD = 150_000


def _detect_group_level(race_name: str, prize_money: int = 0) -> Optional[str]:
    """Detect Group level from race name or prize money."""
    name_lower = (race_name or "").lower()
    for pattern, level in GROUP_PATTERNS:
        if re.search(pattern, name_lower):
            return level
    # High prize money but no explicit group label
    if prize_money >= GROUP_PRIZE_THRESHOLD:
        return "Stakes"
    return None


async def scrape_future_group_races(
    db: AsyncSession,
    weeks_ahead: int = 4,
) -> dict:
    """Discover upcoming Group 1/2/3 races and store nominations.

    Uses PuntingForm API to scan future dates for meetings with Group races.
    Returns summary dict.
    """
    from punty.scrapers.punting_form import PuntingFormScraper

    pf = await PuntingFormScraper.from_settings(db)
    if not pf:
        logger.warning("PuntingForm API not configured — skipping future race scrape")
        return {"status": "skipped", "reason": "no_api_key"}

    today = melb_today()
    end_date = today + timedelta(days=weeks_ahead * 7)
    races_found = 0
    nominations_found = 0
    errors = 0

    try:
        # Scan each date for meetings
        current = today + timedelta(days=1)  # start from tomorrow
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            try:
                meetings = await pf.get_future_meetings(date_str)
                if meetings:
                    for meeting_data in meetings:
                        found, noms = await _process_future_meeting(
                            db, pf, meeting_data, current,
                        )
                        races_found += found
                        nominations_found += noms
            except Exception as e:
                logger.debug(f"Future race scan for {date_str} failed: {e}")
                errors += 1

            current += timedelta(days=1)

        await db.commit()

    finally:
        await pf.close()

    result = {
        "status": "ok",
        "dates_scanned": (end_date - today).days,
        "group_races_found": races_found,
        "nominations_found": nominations_found,
        "errors": errors,
    }
    logger.info(f"Future race scan complete: {result}")
    return result


async def _process_future_meeting(
    db: AsyncSession,
    pf,
    meeting_data: dict,
    race_date: date,
) -> tuple[int, int]:
    """Process a single future meeting — find Group races and store nominations."""
    venue = meeting_data.get("venue") or meeting_data.get("meetingName", "")
    pf_meeting_id = meeting_data.get("meetingId") or meeting_data.get("id")

    if not pf_meeting_id:
        return 0, 0

    races_found = 0
    noms_found = 0

    try:
        fields = await pf.get_future_fields(pf_meeting_id)
        if not fields:
            return 0, 0

        races = fields.get("races") or fields.get("Races") or []
        for race_data in races:
            race_name = race_data.get("raceName") or race_data.get("name") or ""
            race_num = race_data.get("raceNumber") or race_data.get("number")
            distance = race_data.get("distance")
            prize = race_data.get("prizeMoney") or race_data.get("prize") or 0

            group_level = _detect_group_level(race_name, prize)
            if not group_level:
                continue

            # Generate stable ID
            venue_slug = re.sub(r"[^a-z0-9]+", "-", venue.lower()).strip("-")
            race_id = f"{venue_slug}-{race_date.isoformat()}-r{race_num}"

            # Upsert FutureRace
            existing = await db.get(FutureRace, race_id)
            if existing:
                existing.race_name = race_name
                existing.group_level = group_level
                existing.distance = distance
                existing.prize_money = prize
                existing.scraped_at = melb_now_naive()
                # Clear old nominations for refresh
                await db.execute(
                    delete(FutureNomination).where(
                        FutureNomination.future_race_id == race_id
                    )
                )
            else:
                state = meeting_data.get("state") or meeting_data.get("jurisdiction", "")
                db.add(FutureRace(
                    id=race_id,
                    venue=venue,
                    date=race_date,
                    race_number=race_num,
                    race_name=race_name,
                    group_level=group_level,
                    distance=distance,
                    prize_money=prize,
                    state=state,
                ))

            # Store nominations
            runners = race_data.get("runners") or race_data.get("items") or []
            for runner in runners:
                horse = runner.get("horseName") or runner.get("name") or ""
                if not horse:
                    continue
                db.add(FutureNomination(
                    future_race_id=race_id,
                    horse_name=horse,
                    trainer=runner.get("trainerName") or runner.get("trainer"),
                    jockey=runner.get("jockeyName") or runner.get("jockey"),
                    barrier=runner.get("barrierNumber") or runner.get("barrier"),
                    weight=runner.get("weight"),
                    last_start=runner.get("lastStart"),
                    career_record=runner.get("careerRecord"),
                ))
                noms_found += 1

            races_found += 1

    except Exception as e:
        logger.debug(f"Failed to process future meeting {venue}: {e}")

    return races_found, noms_found
