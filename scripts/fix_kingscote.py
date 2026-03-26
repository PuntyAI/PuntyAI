"""Fix Kingscote meeting settlement — all picks had garbage odds from venue mismatch.

Racing.com listed "Kingscote" for what was actually King Island (TAS).
TAB odds from racing.com returned garbage values. Re-settle all picks
using actual tote dividends (win_dividend, place_dividend).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick


async def fix():
    async with async_session() as db:
        q = await db.execute(select(Meeting).where(Meeting.venue.like("%ingscote%")))
        meeting = q.scalars().first()
        if not meeting:
            print("No Kingscote meeting found")
            return

        print(f"Meeting: {meeting.id} venue={meeting.venue}")

        # Load all runners keyed by (race_number, saddlecloth)
        runners = {}
        for rn in range(1, 10):
            race_id = f"{meeting.id}-r{rn}"
            q = await db.execute(select(Runner).where(Runner.race_id == race_id))
            for r in q.scalars().all():
                runners[(rn, r.saddlecloth)] = r

        # Load all selection picks for this meeting
        q = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting.id,
                Pick.pick_type == "selection",
                Pick.settled == True,
            )
        )
        picks = q.scalars().all()

        total_old_pnl = 0.0
        total_new_pnl = 0.0
        fixes = 0

        for pick in picks:
            if not pick.race_number or not pick.saddlecloth:
                continue

            runner = runners.get((pick.race_number, pick.saddlecloth))
            if not runner:
                continue

            old_pnl = float(pick.pnl or 0)
            total_old_pnl += old_pnl
            stake = float(pick.bet_stake or 0)
            if stake <= 0:
                continue

            bt = (pick.bet_type or "").lower().replace(" ", "_")
            fp = runner.finish_position
            win_div = runner.win_dividend
            place_div = runner.place_dividend

            # Recalculate using actual tote dividends
            new_pnl = None
            new_hit = False

            if bt in ("win", "saver_win"):
                if fp == 1 and win_div:
                    new_pnl = round(win_div * stake - stake, 2)
                    new_hit = True
                elif fp and fp > 0:
                    new_pnl = round(-stake, 2)
            elif bt == "place":
                if fp and 1 <= fp <= 3 and place_div:
                    new_pnl = round(place_div * stake - stake, 2)
                    new_hit = True
                elif fp and fp > 3:
                    new_pnl = round(-stake, 2)
            elif bt == "each_way":
                half = stake / 2
                if fp == 1 and win_div and place_div:
                    new_pnl = round((win_div * half - half) + (place_div * half - half), 2)
                    new_hit = True
                elif fp and 2 <= fp <= 3 and place_div:
                    new_pnl = round(-half + (place_div * half - half), 2)
                    new_hit = True
                elif fp and fp > 3:
                    new_pnl = round(-stake, 2)

            if new_pnl is not None:
                total_new_pnl += new_pnl
                diff = abs(old_pnl - new_pnl)
                if diff > 0.05:
                    fixes += 1
                    print(
                        f"  FIX R{pick.race_number} {pick.horse_name} #{pick.saddlecloth} "
                        f"({bt}, ${stake:.2f}): "
                        f"pnl ${old_pnl:+.2f} → ${new_pnl:+.2f} "
                        f"(odds_at_tip={pick.odds_at_tip} → win_div={win_div}, "
                        f"place_odds_at_tip={pick.place_odds_at_tip} → place_div={place_div})"
                    )
                    # Update the pick
                    pick.pnl = new_pnl
                    pick.hit = new_hit
                    # Clear garbage odds, store actual dividends
                    if win_div:
                        pick.odds_at_tip = win_div
                    if place_div:
                        pick.place_odds_at_tip = place_div
                else:
                    total_new_pnl += 0  # already counted

        # Also fix runners with garbage current_odds
        runner_fixes = 0
        for (rn, sc), r in runners.items():
            if r.current_odds and r.starting_price:
                try:
                    sp_val = float(r.starting_price.replace("$", ""))
                except (ValueError, AttributeError):
                    continue
                ratio = r.current_odds / sp_val if sp_val > 0 else 0
                if ratio > 5 or ratio < 0.2:
                    print(
                        f"  FIX RUNNER R{rn} #{sc} {r.horse_name}: "
                        f"current_odds={r.current_odds} → SP={sp_val}"
                    )
                    r.current_odds = sp_val
                    runner_fixes += 1

        print(f"\n  Summary:")
        print(f"    Picks fixed: {fixes}")
        print(f"    Runners fixed: {runner_fixes}")
        print(f"    Old total P&L: ${total_old_pnl:+.2f}")
        print(f"    New total P&L: ${total_new_pnl:+.2f}")
        print(f"    Difference: ${total_new_pnl - total_old_pnl:+.2f}")

        await db.commit()
        print("\n  Changes committed.")


asyncio.run(fix())
