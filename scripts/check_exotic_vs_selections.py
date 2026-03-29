"""Check exotic picks vs selections for consistency."""
import asyncio
from sqlalchemy import select
from punty.models.database import async_session
from punty.models.content import Content
from punty.models.pick import Pick

async def check():
    async with async_session() as db:
        result = await db.execute(
            select(Content).where(Content.content_type == 'early_mail').order_by(Content.created_at.desc()).limit(6)
        )
        contents = result.scalars().all()
        for c in contents:
            print(f"\n=== {c.meeting_id} ({c.status}) ===")
            picks_result = await db.execute(
                select(Pick).where(Pick.content_id == c.id).order_by(Pick.race_number, Pick.tip_rank)
            )
            all_picks = picks_result.scalars().all()

            races = {}
            for p in all_picks:
                if p.race_number:
                    races.setdefault(p.race_number, {"selections": [], "exotics": []})
                    if p.pick_type == "selection":
                        races[p.race_number]["selections"].append(p)
                    elif p.pick_type == "exotic":
                        races[p.race_number]["exotics"].append(p)

            for rnum in sorted(races):
                sels = races[rnum]["selections"]
                exos = races[rnum]["exotics"]
                if not exos:
                    continue
                sel_nums = {p.saddlecloth for p in sels}
                for e in exos:
                    import json
                    runners = json.loads(e.exotic_runners) if e.exotic_runners else []
                    if isinstance(runners, list) and runners and isinstance(runners[0], list):
                        exotic_nums = set()
                        for leg in runners:
                            exotic_nums.update(leg)
                    else:
                        exotic_nums = set(runners)

                    mismatch = exotic_nums - sel_nums
                    missing = sel_nums - exotic_nums
                    flag = " *** MISMATCH" if mismatch else ""
                    print(f"  R{rnum}: Selections={sorted(sel_nums)} | {e.exotic_type} {sorted(exotic_nums)}{flag}")
                    if mismatch:
                        print(f"         Extra in exotic: {sorted(mismatch)}, Missing from exotic: {sorted(missing)}")

asyncio.run(check())
