"""Quick script to check strike rates by tip rank."""
import asyncio
from punty.models.database import async_session
from punty.models.pick import Pick
from sqlalchemy import select, func, case, and_


async def main():
    async with async_session() as db:
        for rank in [1, 2, 3, 4]:
            result = await db.execute(
                select(
                    func.count(Pick.id),
                    func.sum(case((Pick.hit == True, 1), else_=0)),
                    func.sum(Pick.pnl),
                ).where(and_(
                    Pick.settled == True,
                    Pick.pick_type == "selection",
                    Pick.tip_rank == rank,
                ))
            )
            row = result.one()
            total = row[0] or 0
            hits = int(row[1] or 0)
            pnl = row[2] or 0
            rate = round(hits / total * 100, 1) if total > 0 else 0
            labels = {1: "Top Pick", 2: "2nd Pick", 3: "3rd Pick", 4: "Roughie"}
            label = labels.get(rank, f"Rank {rank}")
            print(f"{label}: {hits}/{total} = {rate}% strike rate | PnL: ${pnl:+.2f}")

        # All selections combined
        result = await db.execute(
            select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
            ).where(and_(
                Pick.settled == True,
                Pick.pick_type == "selection",
            ))
        )
        row = result.one()
        total = row[0] or 0
        hits = int(row[1] or 0)
        pnl = row[2] or 0
        rate = round(hits / total * 100, 1) if total > 0 else 0
        print(f"All Selections: {hits}/{total} = {rate}% | PnL: ${pnl:+.2f}")

        # Exotics
        result = await db.execute(
            select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
            ).where(and_(
                Pick.settled == True,
                Pick.pick_type == "exotic",
            ))
        )
        row = result.one()
        total = row[0] or 0
        hits = int(row[1] or 0)
        pnl = row[2] or 0
        rate = round(hits / total * 100, 1) if total > 0 else 0
        print(f"Exotics: {hits}/{total} = {rate}% | PnL: ${pnl:+.2f}")

        # Overall
        result = await db.execute(
            select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
            ).where(Pick.settled == True)
        )
        row = result.one()
        total = row[0] or 0
        hits = int(row[1] or 0)
        pnl = row[2] or 0
        rate = round(hits / total * 100, 1) if total > 0 else 0
        print(f"OVERALL: {hits}/{total} = {rate}% | PnL: ${pnl:+.2f}")


asyncio.run(main())
