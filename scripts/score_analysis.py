"""Compare different scoring formulas for pre-selections across today's races."""
import asyncio
from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner
from punty.probability import calculate_race_probabilities
from sqlalchemy import select
from sqlalchemy.orm import selectinload


async def analyze():
    async with async_session() as db:
        meetings = [
            "flemington-2026-02-14",
            "eagle-farm-2026-02-14",
            "royal-randwick-2026-02-14",
        ]

        all_races = []

        for mid in meetings:
            mresult = await db.execute(
                select(Meeting).where(Meeting.id == mid)
                .options(selectinload(Meeting.races).selectinload(Race.runners))
            )
            meeting = mresult.scalar_one()

            for race in sorted(meeting.races, key=lambda r: r.race_number):
                runners = [r for r in race.runners if not r.scratched]
                if not runners:
                    continue
                probs = calculate_race_probabilities(runners, race, meeting)

                candidates = []
                for r in runners:
                    p = probs.get(r.id)
                    if not p:
                        continue
                    wp = getattr(p, "win_probability", 0)
                    vr = getattr(p, "value_rating", 0)
                    odds = r.current_odds or 0
                    if wp <= 0 or odds <= 0:
                        continue

                    # Current: prob * capped_value(1.0-2.0)
                    capped = max(1.0, min(vr, 2.0))
                    current_score = wp * capped

                    # A: Additive bonus — prob + small_value_bonus
                    # Value bonus maxes at +0.05 (5% prob boost for 1.5x value)
                    score_a = wp + 0.10 * max(0, min(vr - 1.0, 1.0))

                    # B: Tight cap — prob * capped_value(1.0-1.3)
                    score_b = wp * max(1.0, min(vr, 1.3))

                    # C: Blended — 70% prob + 30% value_score
                    value_score = wp * vr if vr > 1.0 else wp
                    score_c = 0.7 * wp + 0.3 * value_score

                    # D: Log value — prob * (1 + ln(max(1, value)) * 0.3)
                    import math
                    score_d = wp * (1 + math.log(max(1.0, vr)) * 0.3)

                    candidates.append({
                        "horse": r.horse_name,
                        "sc": r.saddlecloth,
                        "odds": odds,
                        "wp": wp,
                        "vr": vr,
                        "current": current_score,
                        "A": score_a,
                        "B": score_b,
                        "C": score_c,
                        "D": score_d,
                    })

                if not candidates:
                    continue

                venue = mid.split("-2026")[0].replace("-", " ").title()
                label = f"{venue} R{race.race_number}"

                top_cur = sorted(candidates, key=lambda c: c["current"], reverse=True)[:4]
                top_a = sorted(candidates, key=lambda c: c["A"], reverse=True)[:4]
                top_b = sorted(candidates, key=lambda c: c["B"], reverse=True)[:4]
                top_c = sorted(candidates, key=lambda c: c["C"], reverse=True)[:4]
                top_d = sorted(candidates, key=lambda c: c["D"], reverse=True)[:4]

                def fmt(lst):
                    parts = []
                    for x in lst:
                        parts.append(f"{x['horse']}(${x['odds']:.1f})")
                    return ", ".join(parts)

                cur_names = [c["horse"] for c in top_cur]
                a_names = [c["horse"] for c in top_a]
                b_names = [c["horse"] for c in top_b]

                # Always print to see full picture
                print(f"\n{'='*80}")
                print(f"{label} ({len(candidates)} runners)")
                print(f"  Current(p*v2.0): {fmt(top_cur)}")
                print(f"  A(p+0.1v):       {fmt(top_a)}")
                print(f"  B(p*v1.3):       {fmt(top_b)}")
                print(f"  C(70/30):        {fmt(top_c)}")
                print(f"  D(log_v):        {fmt(top_d)}")

                # Show full table for key races
                if race.race_number in (8, 9) and "flemington" in mid:
                    print(f"\n  Full breakdown:")
                    print(f"  {'Horse':25s} {'Odds':>6s} {'WinP':>6s} {'Value':>6s} {'Cur':>7s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s}")
                    for c in sorted(candidates, key=lambda c: c["wp"], reverse=True):
                        print(f"  {c['horse']:25s} {c['odds']:6.1f} {c['wp']:6.3f} {c['vr']:6.2f} {c['current']:7.4f} {c['A']:7.4f} {c['B']:7.4f} {c['C']:7.4f} {c['D']:7.4f}")


asyncio.run(analyze())
