"""Resettle all picks from the last N days using updated settlement logic.

Usage: python scripts/resettle_recent.py [days]
Default: 3 days
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta


async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cutoff = date.today() - timedelta(days=days)

    from punty.models.database import async_session, init_db
    from punty.models.meeting import Meeting, Race
    from punty.models.pick import Pick
    from punty.results.picks import settle_picks_for_race
    from sqlalchemy import select, update

    await init_db()

    async with async_session() as db:
        # Find all meetings in date range
        result = await db.execute(
            select(Meeting).where(Meeting.date >= cutoff).order_by(Meeting.date)
        )
        meetings = result.scalars().all()
        print(f"Found {len(meetings)} meetings since {cutoff}")

        total_resettled = 0
        total_changed = 0

        for meeting in meetings:
            # Find settled picks for this meeting
            pick_result = await db.execute(
                select(Pick).where(
                    Pick.meeting_id == meeting.id,
                    Pick.settled == True,
                    Pick.pick_type.in_(["selection", "big3"]),
                )
            )
            picks = pick_result.scalars().all()
            if not picks:
                continue

            # Collect old P&L values
            old_pnl = {p.id: p.pnl for p in picks}

            # Get unique race numbers
            race_nums = sorted(set(p.race_number for p in picks if p.race_number))

            # Unsettled all picks for this meeting
            await db.execute(
                update(Pick).where(
                    Pick.meeting_id == meeting.id,
                    Pick.settled == True,
                    Pick.pick_type.in_(["selection", "big3", "exotic"]),
                ).values(settled=False)
            )
            await db.flush()

            # Re-settle each race
            for race_num in race_nums:
                try:
                    count = await settle_picks_for_race(db, meeting.id, race_num)
                    total_resettled += count
                except Exception as e:
                    print(f"  ERROR settling {meeting.id} R{race_num}: {e}")

            # Also resettle exotics
            exotic_result = await db.execute(
                select(Pick).where(
                    Pick.meeting_id == meeting.id,
                    Pick.pick_type == "exotic",
                    Pick.settled == False,
                )
            )
            exotic_picks = exotic_result.scalars().all()
            exotic_races = sorted(set(p.race_number for p in exotic_picks if p.race_number))
            for race_num in exotic_races:
                try:
                    count = await settle_picks_for_race(db, meeting.id, race_num)
                    total_resettled += count
                except Exception as e:
                    print(f"  ERROR settling exotic {meeting.id} R{race_num}: {e}")

            # Check for P&L changes
            refetch = await db.execute(
                select(Pick).where(
                    Pick.meeting_id == meeting.id,
                    Pick.settled == True,
                    Pick.pick_type.in_(["selection", "big3"]),
                )
            )
            new_picks = refetch.scalars().all()
            changes = []
            for p in new_picks:
                old = old_pnl.get(p.id)
                if old is not None and old != p.pnl:
                    changes.append(f"    {p.horse_name} R{p.race_number}: ${old:+.2f} → ${p.pnl:+.2f}")
                    total_changed += 1

            if changes:
                print(f"\n{meeting.venue} ({meeting.date}):")
                for c in changes:
                    print(c)

        await db.commit()
        print(f"\nDone. Resettled {total_resettled} picks, {total_changed} P&L values changed.")


if __name__ == "__main__":
    asyncio.run(main())
