"""Re-parse picks for a meeting, properly handling existing picks."""
import asyncio
import sys
from datetime import date

from sqlalchemy import select, and_, delete

from punty.models.database import async_session
from punty.models.content import Content
from punty.models.pick import Pick
from punty.results.parser import parse_early_mail


async def reparse_meeting(meeting_pattern: str):
    async with async_session() as db:
        # Get content
        result = await db.execute(
            select(Content).where(
                and_(
                    Content.meeting_id.like(f"%{meeting_pattern}%"),
                    Content.content_type == "early_mail",
                    Content.status.in_(["approved", "sent"]),
                )
            ).order_by(Content.created_at.desc())
        )
        content = result.scalars().first()
        if not content:
            print(f"No content found for {meeting_pattern}")
            return

        print(f"Found content {content.id} for {content.meeting_id}")

        # Get existing settled picks to preserve
        settled_result = await db.execute(
            select(Pick).where(
                and_(
                    Pick.content_id == content.id,
                    Pick.settled == True,
                )
            )
        )
        settled_picks = {
            (p.race_number, p.pick_type, p.saddlecloth, p.exotic_type): p
            for p in settled_result.scalars().all()
        }
        print(f"Found {len(settled_picks)} settled picks to preserve")

        # Delete all unsettled picks
        await db.execute(
            delete(Pick).where(
                and_(
                    Pick.content_id == content.id,
                    Pick.settled == False,
                )
            )
        )
        print("Deleted unsettled picks")

        # Re-parse
        picks = parse_early_mail(content.raw_content, content.id, content.meeting_id)
        print(f"Parsed {len(picks)} picks")

        # Count by type
        by_type = {}
        for p in picks:
            t = p["pick_type"]
            by_type[t] = by_type.get(t, 0) + 1
        print(f"By type: {by_type}")

        # Insert new picks, skipping settled ones
        added = 0
        skipped = 0
        for p in picks:
            key = (p["race_number"], p["pick_type"], p.get("saddlecloth"), p.get("exotic_type"))
            if key in settled_picks:
                skipped += 1
                continue

            # Generate new ID to avoid conflicts
            import uuid
            p["id"] = f"pk-{uuid.uuid4().hex[:8]}-{added:03d}"
            pick = Pick(**p)
            db.add(pick)
            added += 1

        await db.commit()
        print(f"Added {added} picks, skipped {skipped} (already settled)")


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "doomben"
    asyncio.run(reparse_meeting(pattern))
