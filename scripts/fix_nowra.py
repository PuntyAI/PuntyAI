"""Re-scrape results for Nowra 2026-02-08 and settle all picks."""
import asyncio
import sys
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

MEETING_ID = "nowra-2026-02-08"

async def fix():
    await init_db()

    async with async_session() as db:
        meeting = await db.get(Meeting, MEETING_ID)
        if not meeting:
            print("Meeting not found")
            return

        # Get all races
        races = (await db.execute(
            select(Race).where(Race.meeting_id == MEETING_ID).order_by(Race.race_number)
        )).scalars().all()

        print(f"Meeting: {meeting.venue} {meeting.date}")
        print(f"Races: {len(races)}")

        scraper = RacingComScraper()
        try:
            for race in races:
                print(f"\n=== Race {race.race_number}: {race.name} ===")
                try:
                    results_data = await scraper.scrape_race_result(
                        meeting.venue, meeting.date, race.race_number
                    )
                    if not results_data:
                        print("  No results data from racing.com")
                        continue

                    runners = results_data.get("runners", [])
                    print(f"  Got results: {len(runners)} runners")

                    # Upsert results
                    await upsert_race_results(db, MEETING_ID, race.race_number, results_data)
                    await db.commit()

                    # Verify
                    check = (await db.execute(
                        select(Runner).where(
                            Runner.race_id == race.id,
                            Runner.finish_position.isnot(None),
                        )
                    )).scalars().all()
                    print(f"  {len(check)} runners now have finish positions")

                    if check:
                        top3 = sorted([r for r in check if r.finish_position <= 3], key=lambda r: r.finish_position)
                        for r in top3:
                            print(f"    {r.finish_position}. {r.horse_name} (No.{r.saddlecloth}) div=${r.win_dividend or 0}/{r.place_dividend or 0}")

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

            # Now settle all picks
            print("\n=== SETTLING PICKS ===")
            for race in races:
                settled = await settle_picks_for_race(db, MEETING_ID, race.race_number)
                await db.commit()
                if settled:
                    print(f"  R{race.race_number}: settled {settled} picks")

            # Settle sequences
            from punty.results.picks import settle_sequence_picks
            seq_settled = await settle_sequence_picks(db, MEETING_ID)
            await db.commit()
            print(f"  Sequences: settled {seq_settled}")

            # Summary
            picks = (await db.execute(
                select(Pick).where(Pick.meeting_id == MEETING_ID)
            )).scalars().all()
            total = len(picks)
            settled = sum(1 for p in picks if p.settled)
            winners = sum(1 for p in picks if p.hit)
            pnl = sum(p.pnl or 0 for p in picks)
            print(f"\n=== FINAL: {settled}/{total} settled, {winners} winners, P&L: ${pnl:.2f} ===")

        finally:
            await scraper.close()

asyncio.run(fix())
