"""One-off script to recalculate exotic settlements with flexi formula."""
import asyncio
import json
from datetime import date

from sqlalchemy import select, and_

from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick


def _find_dividend(exotic_divs: dict, exotic_type: str) -> float:
    """Find dividend for exotic type, handling various key formats."""
    for key in [exotic_type, exotic_type.replace("_", ""), exotic_type.replace(" ", "")]:
        if key in exotic_divs:
            try:
                return float(str(exotic_divs[key]).replace(",", "").replace("$", ""))
            except ValueError:
                pass
    return 0.0


async def recalc_exotics():
    async with async_session() as db:
        # Get today's meetings
        result = await db.execute(
            select(Meeting).where(Meeting.date == date(2026, 2, 7))
        )
        meetings = result.scalars().all()
        print(f"Found {len(meetings)} meetings for today")

        total_fixed = 0
        for meeting in meetings:
            # Get settled exotic picks for this meeting
            picks_result = await db.execute(
                select(Pick).where(
                    and_(
                        Pick.meeting_id == meeting.id,
                        Pick.pick_type == "exotic",
                        Pick.settled == True,
                    )
                )
            )
            picks = picks_result.scalars().all()

            for pick in picks:
                if not pick.hit or pick.pnl <= 0:
                    continue  # Only recalculate winning picks

                # Get race and exotic dividends
                race_result = await db.execute(
                    select(Race).where(
                        and_(
                            Race.meeting_id == meeting.id,
                            Race.race_number == pick.race_number,
                        )
                    )
                )
                race = race_result.scalar_one_or_none()
                if not race or not race.exotic_results:
                    continue

                try:
                    exotic_divs = json.loads(race.exotic_results)
                except (json.JSONDecodeError, TypeError):
                    continue

                exotic_type = (pick.exotic_type or "").lower()
                stake = pick.exotic_stake or 20.0

                # Find dividend
                div_key = "first4" if "first" in exotic_type else exotic_type.split()[0]
                dividend = _find_dividend(exotic_divs, div_key)
                if dividend <= 0:
                    continue

                # Parse runners
                exotic_runners = json.loads(pick.exotic_runners) if pick.exotic_runners else []
                is_legs_format = exotic_runners and isinstance(exotic_runners[0], list)

                # Calculate combos
                if is_legs_format:
                    combos = 1
                    for leg in exotic_runners:
                        combos *= len(leg)
                else:
                    n = len(set(exotic_runners))
                    if "trifecta" in exotic_type:
                        combos = n * (n - 1) * (n - 2) if n >= 3 else 1
                    elif "exacta" in exotic_type:
                        combos = n * (n - 1) if n >= 2 else 1
                    elif "quinella" in exotic_type:
                        combos = n * (n - 1) // 2 if n >= 2 else 1
                    elif "first" in exotic_type:
                        combos = n * (n - 1) * (n - 2) * (n - 3) if n >= 4 else 1
                    else:
                        combos = 1

                # Recalculate with flexi formula
                flexi_pct = stake / combos if combos > 0 else stake
                return_amount = dividend * flexi_pct
                new_pnl = round(return_amount - stake, 2)

                if abs(new_pnl - pick.pnl) > 0.01:
                    print(
                        f"  {meeting.venue} R{pick.race_number} {pick.exotic_type}: "
                        f"dividend=${dividend}, stake=${stake}, combos={combos}, "
                        f"PNL ${pick.pnl} -> ${new_pnl}"
                    )
                    pick.pnl = new_pnl
                    total_fixed += 1

        await db.commit()
        print(f"\nFixed {total_fixed} exotic picks")


if __name__ == "__main__":
    asyncio.run(recalc_exotics())
