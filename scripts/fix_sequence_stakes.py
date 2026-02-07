"""One-off script to fix sequence stakes for today's meetings."""
import asyncio
import json
import re
from datetime import date

from sqlalchemy import select, and_

from punty.models.database import async_session
from punty.models.meeting import Meeting
from punty.models.content import Content
from punty.models.pick import Pick

# Regex to parse sequence costing - updated version
SEQ_COSTING = re.compile(
    r"\((?:[\d×x]+\s*=\s*)?(\d+)\s*combos?\s*[×x]\s*\$(\d+\.?\d*)\s*=\s*\$(\d+\.?\d*)\)\s*(?:[–\-—]\s*est\.\s*return:\s*(\d+\.?\d*)%)?",
    re.IGNORECASE,
)

SEQ_HEADER = re.compile(
    r"(EARLY\s+QUADDIE|MAIN\s+QUADDIE|QUADDIE|BIG\s*6)\s*\(R(\d+)[–\-—]R(\d+)\)",
    re.IGNORECASE,
)

SEQ_VARIANT = re.compile(
    r"(Skinny|Balanced|Wide)\s*(?:\(\$[\d.]+\))?:\s*(.+)",
    re.IGNORECASE,
)


async def fix_sequences():
    async with async_session() as db:
        # Get today's meetings
        result = await db.execute(
            select(Meeting).where(Meeting.date == date(2026, 2, 7))
        )
        meetings = result.scalars().all()
        print(f"Found {len(meetings)} meetings for today")

        for meeting in meetings:
            # Get approved/sent early mail
            content_result = await db.execute(
                select(Content).where(
                    and_(
                        Content.meeting_id == meeting.id,
                        Content.content_type == "early_mail",
                        Content.status.in_(["approved", "sent"]),
                    )
                ).order_by(Content.created_at.desc())
            )
            content = content_result.scalars().first()
            if not content or not content.raw_content:
                continue

            # Parse sequence info from content
            seq_info = {}  # (seq_type, variant) -> total_outlay

            # Find sequence section
            seq_match = re.search(
                r"SEQUENCE\s+LANES.*?(?=\n\*[A-Z]|\n###|\Z)",
                content.raw_content,
                re.DOTALL | re.IGNORECASE,
            )
            if not seq_match:
                continue

            seq_text = seq_match.group(0)

            # Find each sequence block
            headers = list(SEQ_HEADER.finditer(seq_text))
            for idx, hdr in enumerate(headers):
                seq_name = hdr.group(1).strip().upper()

                # Normalize type
                if "EARLY" in seq_name:
                    seq_type = "early_quaddie"
                elif "BIG" in seq_name:
                    seq_type = "big6"
                else:
                    seq_type = "quaddie"

                # Get text until next header or end
                start_pos = hdr.end()
                end_pos = headers[idx + 1].start() if idx + 1 < len(headers) else len(seq_text)
                block = seq_text[start_pos:end_pos]

                # Find variants
                for var_match in SEQ_VARIANT.finditer(block):
                    variant = var_match.group(1).lower()
                    legs_raw = var_match.group(2)

                    # Parse costing
                    cost_match = SEQ_COSTING.search(legs_raw)
                    if cost_match:
                        total_outlay = float(cost_match.group(3))
                        seq_info[(seq_type, variant)] = total_outlay

            if not seq_info:
                continue

            # Get sequence picks for this meeting
            picks_result = await db.execute(
                select(Pick).where(
                    and_(
                        Pick.meeting_id == meeting.id,
                        Pick.pick_type == "sequence",
                    )
                )
            )
            picks = picks_result.scalars().all()

            updated = 0
            for pick in picks:
                key = (pick.sequence_type, pick.sequence_variant)
                if key in seq_info:
                    old_stake = pick.exotic_stake
                    new_stake = seq_info[key]
                    if old_stake != new_stake:
                        pick.exotic_stake = new_stake
                        # Recalculate PNL if settled (all missed, so PNL = -stake)
                        if pick.settled and not pick.hit:
                            old_pnl = pick.pnl
                            pick.pnl = -new_stake
                            print(
                                f"  {meeting.venue} {pick.sequence_type}/{pick.sequence_variant}: "
                                f"stake ${old_stake} -> ${new_stake}, PNL ${old_pnl} -> ${-new_stake}"
                            )
                        else:
                            print(
                                f"  {meeting.venue} {pick.sequence_type}/{pick.sequence_variant}: "
                                f"stake ${old_stake} -> ${new_stake}"
                            )
                        updated += 1

            if updated > 0:
                print(f"{meeting.venue}: Updated {updated} sequence picks")

        await db.commit()
        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(fix_sequences())
