"""Re-parse exotic_runners to preserve leg structure."""
import asyncio
import json
import re
from datetime import date

from sqlalchemy import select, and_

from punty.models.database import async_session
from punty.models.content import Content
from punty.models.pick import Pick
from punty.models.meeting import Meeting


async def fix_exotic_runners():
    async with async_session() as db:
        # Get today's meetings
        result = await db.execute(
            select(Meeting).where(Meeting.date == date(2026, 2, 7))
        )
        meetings = result.scalars().all()
        print(f"Found {len(meetings)} meetings")

        fixed = 0
        for meeting in meetings:
            # Get exotic picks for this meeting
            picks_result = await db.execute(
                select(Pick).where(
                    and_(
                        Pick.meeting_id == meeting.id,
                        Pick.pick_type == "exotic",
                    )
                )
            )
            picks = picks_result.scalars().all()

            for pick in picks:
                if not pick.content_id:
                    continue

                # Get content
                content_result = await db.execute(
                    select(Content).where(Content.id == pick.content_id)
                )
                content = content_result.scalar_one_or_none()
                if not content or not content.raw_content:
                    continue

                # Find this race's exotic section
                exotic_type = (pick.exotic_type or "").split()[0].lower()
                pattern = (
                    rf"Race\s+{pick.race_number}.*?"
                    rf"({exotic_type})"
                    rf":\s*(.+?)\s*(?:[–\-—]\s*\$|\(\$)"
                )
                m = re.search(pattern, content.raw_content, re.DOTALL | re.IGNORECASE)
                if not m:
                    continue

                runners_str = m.group(2).strip()

                # Check if it has "/" separators
                if "/" not in runners_str:
                    continue  # No legs to parse

                # Parse legs
                legs = []
                for leg in runners_str.split("/"):
                    leg_runners = [int(x) for x in re.findall(r'\d+', leg)]
                    if leg_runners:
                        legs.append(leg_runners)

                if not legs:
                    continue

                old_runners = pick.exotic_runners
                new_runners = json.dumps(legs)

                if old_runners != new_runners:
                    print(
                        f"  {meeting.venue} R{pick.race_number} {pick.exotic_type}: "
                        f"{old_runners} -> {new_runners}"
                    )
                    pick.exotic_runners = new_runners
                    fixed += 1

        await db.commit()
        print(f"\nFixed {fixed} exotic picks")


if __name__ == "__main__":
    asyncio.run(fix_exotic_runners())
