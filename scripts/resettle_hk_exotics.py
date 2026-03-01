"""Re-settle HK exotic/sequence picks that were hit but got $0 pnl.

After deploying HKJC exotic dividend parsing, run this to:
1. Re-scrape exotic dividends for all HK races missing them
2. Un-settle picks that were hit=True, pnl=0.0
3. Re-run settlement with the new dividend data

Usage (on server):
    cd /opt/puntyai && source venv/bin/activate
    python scripts/resettle_hk_exotics.py
"""

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main():
    from sqlalchemy import select, update
    from punty.models.database import async_session, init_db
    from punty.models.meeting import Meeting, Race, Runner
    from punty.models.pick import Pick

    await init_db()

    async with async_session() as db:
        # 1. Find all HK meetings
        from punty.venues import guess_state
        all_meetings = (await db.execute(select(Meeting))).scalars().all()
        hk_meetings = [m for m in all_meetings if guess_state(m.venue) == "HK"]
        logger.info(f"Found {len(hk_meetings)} HK meetings")

        # 2. Find HK races missing exotic_results
        missing_exotics = []
        for meeting in hk_meetings:
            races = (await db.execute(
                select(Race).where(
                    Race.meeting_id == meeting.id,
                    Race.results_status.in_(["Paying", "Closed"]),
                )
            )).scalars().all()
            for race in races:
                if not race.exotic_results:
                    missing_exotics.append((meeting, race))

        logger.info(f"Found {len(missing_exotics)} HK races missing exotic dividends")

        # 3. Re-scrape exotics via HKJC
        if missing_exotics:
            from punty.scrapers.tab_playwright import HKJCResultsScraper
            hkjc = HKJCResultsScraper()

            backfilled = 0
            for meeting, race in missing_exotics:
                try:
                    results_data = await hkjc.scrape_race_result(
                        meeting.venue, meeting.date, race.race_number
                    )
                    exotics = results_data.get("exotics")
                    if exotics:
                        race.exotic_results = json.dumps(exotics)
                        backfilled += 1
                        logger.info(
                            f"  {meeting.venue} R{race.race_number}: "
                            f"{exotics}"
                        )
                    else:
                        logger.info(
                            f"  {meeting.venue} R{race.race_number}: "
                            f"no exotics on HKJC page"
                        )
                except Exception as e:
                    logger.warning(
                        f"  {meeting.venue} R{race.race_number} failed: {e}"
                    )

            if backfilled:
                await db.flush()
                logger.info(f"Backfilled exotic dividends for {backfilled} races")

        # 4. Find affected picks (hit but $0 pnl)
        affected = (await db.execute(
            select(Pick).where(
                Pick.meeting_id.in_([m.id for m in hk_meetings]),
                Pick.pick_type.in_(["exotic", "sequence"]),
                Pick.hit == True,
                Pick.pnl == 0.0,
                Pick.settled == True,
            )
        )).scalars().all()

        logger.info(f"Found {len(affected)} HK exotic/sequence picks with hit=True, pnl=0.0")

        if not affected:
            # Also show all HK exotic/sequence picks for audit
            all_hk_picks = (await db.execute(
                select(Pick).where(
                    Pick.meeting_id.in_([m.id for m in hk_meetings]),
                    Pick.pick_type.in_(["exotic", "sequence"]),
                )
            )).scalars().all()
            logger.info(f"Total HK exotic/sequence picks: {len(all_hk_picks)}")
            for p in all_hk_picks:
                logger.info(
                    f"  {p.meeting_id} R{p.race_number} "
                    f"{p.pick_type}/{p.exotic_type or p.sequence_type} "
                    f"hit={p.hit} pnl={p.pnl} settled={p.settled}"
                )
            await db.commit()
            return

        # 5. Un-settle affected picks
        for p in affected:
            logger.info(
                f"  Un-settling: {p.meeting_id} R{p.race_number} "
                f"{p.pick_type}/{p.exotic_type or p.sequence_type}"
            )
        await db.execute(
            update(Pick).where(
                Pick.id.in_([p.id for p in affected]),
            ).values(settled=False, settled_at=None)
        )
        await db.flush()

        # 6. Re-settle
        from punty.results.picks import settle_picks_for_race
        resettled_races = set()
        for p in affected:
            key = (p.meeting_id, p.race_number)
            if key not in resettled_races:
                resettled_races.add(key)
                try:
                    await settle_picks_for_race(db, p.meeting_id, p.race_number)
                    logger.info(f"  Re-settled {p.meeting_id} R{p.race_number}")
                except Exception as e:
                    logger.error(f"  Failed to re-settle {p.meeting_id} R{p.race_number}: {e}")

        await db.commit()

        # 7. Report final state
        final = (await db.execute(
            select(Pick).where(
                Pick.id.in_([p.id for p in affected]),
            )
        )).scalars().all()
        total_pnl = sum(p.pnl or 0 for p in final)
        logger.info(f"\nRe-settlement complete:")
        for p in final:
            logger.info(
                f"  {p.meeting_id} R{p.race_number} "
                f"{p.pick_type}/{p.exotic_type or p.sequence_type}: "
                f"hit={p.hit} pnl=${p.pnl:.2f} settled={p.settled}"
            )
        logger.info(f"Total recovered P&L: ${total_pnl:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
