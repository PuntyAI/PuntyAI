"""Re-scrape results for races with missing finish positions and re-settle."""
import asyncio
import sys
import json
sys.path.insert(0, "/opt/puntyai")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sqlalchemy import select, and_
from punty.models.database import async_session, init_db
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick
from punty.scrapers.racing_com import RacingComScraper
from punty.scrapers.orchestrator import upsert_race_results
from punty.results.picks import settle_picks_for_race

PROBLEM_RACES = [
    ("aquis-beaudesert-2026-02-06", 8),
    ("bet365-colac-2026-02-06", 8),
    ("walcha-2026-02-06", 8),
]

async def fix():
    await init_db()

    for meeting_id, race_num in PROBLEM_RACES:
        race_id = f"{meeting_id}-r{race_num}"
        print(f"\n=== {race_id} ===")

        async with async_session() as db:
            meeting = await db.get(Meeting, meeting_id)
            if not meeting:
                print(f"  Meeting not found")
                continue

            # Scrape results from racing.com
            scraper = RacingComScraper()
            try:
                results_data = await scraper.scrape_race_result(
                    meeting.venue, meeting.date, race_num
                )
                if not results_data:
                    print(f"  No results data from racing.com")
                    continue

                print(f"  Got results: {len(results_data.get('runners', []))} runners")

                # Upsert results into DB
                await upsert_race_results(db, meeting_id, race_num, results_data)
                await db.commit()

                # Verify positions were set
                check = await db.execute(
                    select(Runner).where(
                        Runner.race_id == race_id,
                        Runner.finish_position.isnot(None),
                    )
                )
                with_pos = len(check.scalars().all())
                print(f"  {with_pos} runners now have finish positions")

                if with_pos == 0:
                    print(f"  Still no positions — skipping settlement")
                    continue

                # Un-settle existing picks so they can be re-settled correctly
                picks_result = await db.execute(
                    select(Pick).where(
                        Pick.meeting_id == meeting_id,
                        Pick.race_number == race_num,
                        Pick.settled == True,
                    )
                )
                picks = picks_result.scalars().all()
                for pick in picks:
                    pick.settled = False
                    pick.hit = None
                    pick.pnl = None
                    pick.settled_at = None
                await db.commit()
                print(f"  Reset {len(picks)} picks for re-settlement")

                # Re-settle
                settled = await settle_picks_for_race(db, meeting_id, race_num)
                await db.commit()
                print(f"  Re-settled {settled} picks")

                # Show results
                picks_result = await db.execute(
                    select(Pick).where(
                        Pick.meeting_id == meeting_id,
                        Pick.race_number == race_num,
                    ).order_by(Pick.tip_rank)
                )
                for p in picks_result.scalars().all():
                    status = "WIN" if p.hit else ("LOSS" if p.settled else "UNSETTLED")
                    name = p.horse_name or p.exotic_type or "?"
                    print(f"    {p.pick_type:12s} {name:20s} {status} pnl=${p.pnl or 0:.2f}")

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                await scraper.close()

    # Fix Wyong empty sequences
    print("\n=== WYONG EMPTY SEQUENCES ===")
    async with async_session() as db:
        result = await db.execute(
            select(Pick).where(
                and_(
                    Pick.meeting_id == "wyong-2026-02-10",
                    Pick.pick_type == "sequence",
                    Pick.settled == False,
                )
            )
        )
        for pick in result.scalars().all():
            legs = json.loads(pick.sequence_legs) if pick.sequence_legs else []
            if not legs:
                stake = pick.exotic_stake or pick.bet_stake or 0
                pick.settled = True
                pick.hit = False
                pick.pnl = round(-stake, 2)
                print(f"  {pick.sequence_type} {pick.sequence_variant}: empty legs -> LOSS pnl=${pick.pnl}")
            else:
                print(f"  {pick.sequence_type} {pick.sequence_variant}: {len(legs)} legs, start={pick.sequence_start_race}")
        await db.commit()

asyncio.run(fix())
