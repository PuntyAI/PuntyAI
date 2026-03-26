"""Regenerate early mails for meetings that should already be out.

Skips heavy Playwright scrape (data already scraped at 5am).
Does lightweight PF API refresh → delete old content → generate → approve → post.

Run from /opt/puntyai with venv activated:
    python3 scripts/redo_generate.py
"""
import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from punty.models.database import async_session, init_db
    from punty.models.meeting import Meeting
    from punty.models.content import Content
    from punty.models.pick import Pick
    from punty.config import melb_now
    from sqlalchemy import select

    await init_db()

    REDO_MEETINGS = [
        "tamworth-2026-02-13",
        "bet365-park-kilmore-2026-02-13",
        "taree-2026-02-13",
        "mackay-2026-02-13",
    ]

    now = melb_now()
    print(f"\nMelbourne time: {now.strftime('%H:%M:%S')}")

    # Step 1: Lightweight PF API refresh (odds + conditions, no Playwright)
    print("\n=== STEP 1: Lightweight PF API refresh ===")
    async with async_session() as db:
        from punty.scrapers.punting_form import PuntingFormScraper
        from punty.scrapers.orchestrator import refresh_odds
        pf = await PuntingFormScraper.from_settings(db)
        try:
            for mid in REDO_MEETINGS:
                meeting = await db.get(Meeting, mid)
                if not meeting:
                    print(f"  {mid}: NOT FOUND")
                    continue
                # Refresh odds via TAB
                try:
                    await refresh_odds(mid, db)
                    print(f"  {meeting.venue}: odds refreshed")
                except Exception as e:
                    print(f"  {meeting.venue}: odds refresh failed: {e}")

                # Refresh conditions via PF
                if not meeting.track_condition_locked:
                    try:
                        from punty.scrapers.orchestrator import _apply_pf_conditions
                        cond = await pf.get_conditions_for_venue(meeting.venue)
                        if cond:
                            _apply_pf_conditions(meeting, cond)
                            await db.commit()
                            print(f"  {meeting.venue}: conditions updated: {meeting.track_condition}")
                    except Exception as e:
                        print(f"  {meeting.venue}: conditions failed: {e}")

                # Refresh speed maps via PF API (no Playwright)
                try:
                    from punty.scrapers.orchestrator import scrape_speed_maps_stream
                    count = 0
                    async for event in scrape_speed_maps_stream(mid, db):
                        if event.get("status") == "done":
                            count += 1
                    print(f"  {meeting.venue}: speed maps refreshed ({count} races)")
                except Exception as e:
                    print(f"  {meeting.venue}: speed maps failed: {e}")
        finally:
            await pf.close()

    # Step 2: Delete old content + picks
    print("\n=== STEP 2: Delete old content + picks ===")
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
                pick_result = await db.execute(
                    select(Pick).where(Pick.content_id == c.id)
                )
                old_picks = pick_result.scalars().all()
                for p in old_picks:
                    print(f"  Deleting pick: {p.id} ({p.horse_name or p.exotic_type or p.sequence_type}) settled={p.settled}")
                    await db.delete(p)
                print(f"  Deleting EM: {c.id[:12]}... status={c.status} ({len(old_picks)} picks)")
                await db.delete(c)
            if not old_content:
                print(f"  {mid}: no existing EM to delete")
            await db.commit()

    # Step 3: Generate + approve + post
    print("\n=== STEP 3: Generate + approve + post ===")
    for mid in REDO_MEETINGS:
        print(f"\n--- {mid} ---")
        async with async_session() as db:
            meeting = await db.get(Meeting, mid)
            venue = meeting.venue if meeting else mid

            # Generate early mail
            try:
                from punty.ai.generator import ContentGenerator
                generator = ContentGenerator(db)
                content_id = None
                print(f"  Generating early mail for {venue}...")
                async for event in generator.generate_early_mail_stream(mid):
                    if event.get("status") == "error":
                        raise Exception(event.get("label", "Unknown error"))
                    if event.get("content_id"):
                        content_id = event.get("content_id")
                    elif event.get("result", {}).get("content_id"):
                        content_id = event["result"]["content_id"]
                    if event.get("status") in ("done", "generation_done"):
                        print(f"  {event.get('label', '')}")
                print(f"  Content ID: {content_id}")

                # Auto-approve and post
                if content_id:
                    from punty.scheduler.automation import auto_approve_and_post
                    post_result = await auto_approve_and_post(content_id, db)
                    status = post_result.get("status")
                    twitter = post_result.get("twitter", {}).get("status", "n/a")
                    facebook = post_result.get("facebook", {}).get("status", "n/a")
                    print(f"  Result: {status} | Twitter: {twitter} | Facebook: {facebook}")
                else:
                    print(f"  WARNING: No content_id returned")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n=== DONE at {melb_now().strftime('%H:%M:%S')} ===")


if __name__ == "__main__":
    asyncio.run(main())
