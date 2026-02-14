"""Backfill win_probability, value_rating, and factors_json on settled picks.

Loads all settled selection picks that are missing probability data,
recalculates from the existing runner/race data, and updates in place.

Usage: cd /opt/puntyai && source venv/bin/activate && python scripts/backfill_probabilities.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from punty.models.database import async_session, init_db
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick
from punty.probability import calculate_race_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill():
    await init_db()

    async with async_session() as db:
        # Count picks needing backfill
        total_result = await db.execute(
            select(func.count(Pick.id)).where(
                Pick.pick_type == "selection",
                Pick.settled == True,
            )
        )
        total = total_result.scalar() or 0

        missing_result = await db.execute(
            select(func.count(Pick.id)).where(
                Pick.pick_type == "selection",
                Pick.settled == True,
                Pick.win_probability.is_(None),
            )
        )
        missing = missing_result.scalar() or 0

        logger.info(f"Total settled selections: {total}, missing probability: {missing}")

        if missing == 0:
            logger.info("Nothing to backfill!")
            return

        # Get all unique meeting IDs from picks needing backfill
        meeting_result = await db.execute(
            select(Pick.meeting_id).where(
                Pick.pick_type == "selection",
                Pick.settled == True,
                Pick.win_probability.is_(None),
            ).distinct()
        )
        meeting_ids = [r[0] for r in meeting_result.all()]
        logger.info(f"Processing {len(meeting_ids)} meetings...")

        updated = 0
        skipped = 0

        for mid in meeting_ids:
            # Load meeting with races and runners
            result = await db.execute(
                select(Meeting).where(Meeting.id == mid)
                .options(
                    selectinload(Meeting.races).selectinload(Race.runners)
                )
            )
            meeting = result.scalar_one_or_none()
            if not meeting:
                logger.warning(f"Meeting {mid} not found, skipping")
                skipped += 1
                continue

            # Build probability map for all races
            race_probs = {}
            for race in meeting.races:
                active = [r for r in race.runners if not r.scratched]
                if not active:
                    continue
                try:
                    probs = calculate_race_probabilities(active, race, meeting)
                    for runner in active:
                        if runner.id in probs:
                            rp = probs[runner.id]
                            race_probs[(race.race_number, runner.saddlecloth)] = rp
                except Exception as e:
                    logger.warning(f"Failed to calc probs for {mid} R{race.race_number}: {e}")

            # Load picks for this meeting
            picks_result = await db.execute(
                select(Pick).where(
                    Pick.meeting_id == mid,
                    Pick.pick_type == "selection",
                    Pick.settled == True,
                    Pick.win_probability.is_(None),
                )
            )
            picks = picks_result.scalars().all()

            for pick in picks:
                key = (pick.race_number, pick.saddlecloth)
                rp = race_probs.get(key)
                if rp:
                    pick.win_probability = rp.win_probability
                    pick.place_probability = rp.place_probability
                    pick.value_rating = rp.value_rating
                    pick.recommended_stake = rp.recommended_stake
                    if rp.factors:
                        pick.factors_json = json.dumps(rp.factors)
                    updated += 1
                else:
                    skipped += 1

            await db.commit()
            logger.info(f"  {mid}: {len(picks)} picks processed")

        logger.info(f"Done! Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    asyncio.run(backfill())
