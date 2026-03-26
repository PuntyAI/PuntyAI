"""Regenerate all early mails with updated scoring, deleting old social posts."""
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("regen")

MEETINGS = [
    "flemington-2026-02-14",
    "eagle-farm-2026-02-14",
    "royal-randwick-2026-02-14",
    "morphettville-parks-2026-02-14",
]


async def regenerate_all():
    from punty.models.database import async_session
    from punty.models.content import Content
    from punty.models.pick import Pick
    from punty.ai.generator import ContentGenerator
    from punty.scheduler.automation import auto_approve_and_post, _delete_social_posts
    from sqlalchemy import select, delete as sa_delete

    for meeting_id in MEETINGS:
        logger.info(f"=== Regenerating {meeting_id} ===")

        async with async_session() as db:
            # 1. Supersede old content + delete social posts
            old_content = await db.execute(
                select(Content).where(
                    Content.meeting_id == meeting_id,
                    Content.content_type == "early_mail",
                    Content.status.notin_(["superseded", "rejected"]),
                )
            )
            old = old_content.scalars().all()
            for c in old:
                logger.info(
                    f"  Superseding: {c.id} (status={c.status}, "
                    f"twitter={c.twitter_id}, facebook={c.facebook_id})"
                )
                # Delete social posts first
                await _delete_social_posts(c, db)
                c.status = "superseded"
                # Delete unsettled picks
                await db.execute(
                    sa_delete(Pick).where(
                        Pick.content_id == c.id, Pick.settled == False
                    )
                )
            await db.commit()

            # 2. Generate new content
            generator = ContentGenerator(db)
            content_id = None
            try:
                async for event in generator.generate_early_mail_stream(meeting_id):
                    if event.get("status") == "error":
                        logger.error(f"  Generation error: {event}")
                        break
                    if event.get("content_id"):
                        content_id = event["content_id"]
                    elif event.get("result", {}).get("content_id"):
                        content_id = event["result"]["content_id"]
            except Exception as e:
                logger.error(f"  Generation failed: {e}")
                continue

            if not content_id:
                logger.error(f"  No content generated for {meeting_id}")
                continue

            logger.info(f"  Generated: {content_id}")

            # 3. Auto-approve and post
            try:
                post_result = await auto_approve_and_post(content_id, db)
                logger.info(f"  Posted: {post_result.get('status')}")
                twitter = post_result.get("twitter", {})
                facebook = post_result.get("facebook", {})
                if twitter.get("url"):
                    logger.info(f"  Twitter: {twitter['url']}")
                if facebook.get("url"):
                    logger.info(f"  Facebook: {facebook['url']}")
            except Exception as e:
                logger.error(f"  Post failed: {e}")

        logger.info(f"  Done: {meeting_id}")

    logger.info("=== All regeneration complete ===")


asyncio.run(regenerate_all())
