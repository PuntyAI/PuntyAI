"""Job definitions for scheduled tasks."""

import logging
from datetime import date, datetime
from enum import Enum
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


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
