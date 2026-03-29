"""Full scrape + regenerate early mails for meetings that should already be out.

Run from /opt/puntyai with venv activated:
    python3 scripts/redo_early_mails.py
"""
import asyncio
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from punty.models.database import async_session, init_db
    from punty.models.meeting import Meeting
    from punty.models.content import Content
    from punty.models.pick import Pick
    from punty.config import melb_now
    from sqlalchemy import select, delete

    await init_db()

    # Meetings that should already be out (first race before ~14:00)
    REDO_MEETINGS = [
        "tamworth-2026-02-13",
        "bet365-park-kilmore-2026-02-13",
        "taree-2026-02-13",
        "mackay-2026-02-13",
    ]

    now = melb_now()
    print(f"Melbourne time: {now.strftime('%H:%M:%S')}")
    print()

    # ─── Step 1: Full morning scrape ───
    print("=" * 60)
    print("STEP 1: Full morning scrape (all meetings)")
    print("=" * 60)
    from punty.scheduler.jobs import daily_morning_scrape
    try:
        result = await daily_morning_scrape()
        print(f"Scraped: {result.get('meetings_scraped', [])}")
        print(f"Speed maps: {result.get('speed_maps_done', [])}")
        if result.get("errors"):
            print(f"Errors: {result['errors']}")
        print()
    except Exception as e:
        print(f"ERROR in morning scrape: {e}")
        print("Continuing with generation anyway...")
        print()

    # ─── Step 2: Delete old early mail content + picks for redo meetings ───
    print("=" * 60)
    print("STEP 2: Delete old early mail content + associated picks")
    print("=" * 60)
    async with async_session() as db:
        for mid in REDO_MEETINGS:
            result = await db.execute(
                select(Content).where(
                    Content.meeting_id == mid,
                    Content.content_type == "early_mail",
                )
            )
            old_content = result.scalars().all()
            for c in old_content:
                # Delete picks linked to this content first (FK constraint)
                pick_result = await db.execute(
                    select(Pick).where(Pick.content_id == c.id)
                )
                old_picks = pick_result.scalars().all()
                for p in old_picks:
                    print(f"  Deleting pick: {p.id} ({p.horse_name or p.exotic_type or p.sequence_type}) settled={p.settled}")
                    await db.delete(p)
                print(f"  Deleting old EM for {mid}: {c.id} (status={c.status}, {len(old_picks)} picks)")
                await db.delete(c)
            await db.commit()
    print()

    # ─── Step 3: Run pre-race job for each meeting ───
    print("=" * 60)
    print("STEP 3: Generate + approve + post for each meeting")
    print("=" * 60)
    from punty.scheduler.jobs import meeting_pre_race_job
    for mid in REDO_MEETINGS:
        print(f"\n--- {mid} ---")
        try:
            result = await meeting_pre_race_job(mid)
            steps = result.get("steps", [])
            errors = result.get("errors", [])
            content_id = result.get("content_id")
            post_result = result.get("post_result", {})
            print(f"  Steps: {steps}")
            print(f"  Content ID: {content_id}")
            print(f"  Post result: {post_result}")
            if errors:
                print(f"  ERRORS: {errors}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
