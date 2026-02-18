"""Check server state: meetings, scheduler, monitor."""
import asyncio
from sqlalchemy import select, func
from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner
from punty.models.content import Content
from punty.models.pick import Pick
from punty.config import melb_now


async def main():
    print(f"Server time: {melb_now()}")
    async with async_session() as db:
        q = await db.execute(select(Meeting).where(Meeting.date == "2026-02-18"))
        meetings = q.scalars().all()
        print(f"\nToday meetings: {len(meetings)}")
        for m in meetings:
            q2 = await db.execute(select(func.count()).select_from(Race).where(Race.meeting_id == m.id))
            race_count = q2.scalar()
            q3 = await db.execute(
                select(func.count()).select_from(Runner).where(Runner.race_id.like(f"{m.id}%"))
            )
            runner_count = q3.scalar()
            q4 = await db.execute(
                select(Race.results_status, func.count())
                .where(Race.meeting_id == m.id)
                .group_by(Race.results_status)
            )
            statuses = dict(q4.all())
            q5 = await db.execute(
                select(func.count()).select_from(Content).where(Content.meeting_id == m.id)
            )
            content_count = q5.scalar()
            q6 = await db.execute(
                select(func.count())
                .select_from(Pick)
                .where(Pick.meeting_id == m.id, Pick.settled == True)
            )
            settled = q6.scalar()
            q7 = await db.execute(
                select(func.count()).select_from(Pick).where(Pick.meeting_id == m.id)
            )
            total_picks = q7.scalar()
            print(
                f"  {m.id}: sel={m.selected} {race_count}R {runner_count} runners "
                f"| statuses={statuses} | content={content_count} "
                f"| picks={settled}/{total_picks} settled"
            )

        # Tomorrow
        q = await db.execute(select(Meeting).where(Meeting.date == "2026-02-19"))
        meetings_tmr = q.scalars().all()
        print(f"\nTomorrow meetings: {len(meetings_tmr)}")
        for m in meetings_tmr:
            q2 = await db.execute(select(func.count()).select_from(Race).where(Race.meeting_id == m.id))
            rc = q2.scalar()
            print(f"  {m.id}: sel={m.selected} {rc} races type={m.meeting_type}")


asyncio.run(main())
