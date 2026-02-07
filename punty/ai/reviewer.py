"""AI-assisted content review and fixes."""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.ai.client import AIClient
from punty.ai.generator import load_prompt
from punty.models.content import Content, ContentStatus

logger = logging.getLogger(__name__)


class ContentReviewer:
    """Handles AI-assisted content review and fixes."""

    def __init__(self, db: AsyncSession, model: str = "gpt-4o"):
        self.db = db
        self.ai_client = AIClient(model=model)

    async def fix_content(
        self,
        content_id: str,
        issue_type: str,
        notes: Optional[str] = None,
    ) -> dict:
        """Request AI fix for content issues.

        Args:
            content_id: ID of content to fix
            issue_type: Type of issue (tone_wrong, too_long, etc.)
            notes: Additional feedback

        Returns:
            Dict with new content and metadata
        """
        # Get original content
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        if not content.raw_content:
            raise ValueError("Content has no raw_content to fix")

        # Load personality prompt
        personality = load_prompt("personality")

        # Generate fixed content
        fixed_content = await self.ai_client.review_and_fix(
            original_content=content.raw_content,
            issue_type=issue_type,
            notes=notes,
            system_prompt=personality,
        )

        # Update content
        content.raw_content = fixed_content
        content.status = ContentStatus.PENDING_REVIEW.value
        content.review_notes = f"AI Fix applied: {issue_type}" + (f" - {notes}" if notes else "")

        # Re-format for Twitter
        from punty.formatters.twitter import format_twitter
        content.twitter_formatted = format_twitter(content.raw_content, content.content_type)

        await self.db.commit()

        logger.info(f"Applied AI fix to {content_id}: {issue_type}")

        return {
            "content_id": content_id,
            "issue_type": issue_type,
            "new_content": fixed_content,
            "status": content.status,
        }

    async def regenerate_content(
        self,
        content_id: str,
        additional_instructions: Optional[str] = None,
    ) -> dict:
        """Completely regenerate content with optional new instructions."""
        from punty.ai.generator import ContentGenerator

        # Get original content
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        # Create new generator
        generator = ContentGenerator(self.db)

        # Regenerate based on content type
        if content.content_type == "early_mail":
            new_result = await generator.generate_early_mail(
                meeting_id=content.meeting_id,
                save=False,
            )
        elif content.content_type == "race_preview":
            # Get race number from race_id
            from punty.models.meeting import Race
            race_result = await self.db.execute(
                select(Race).where(Race.id == content.race_id)
            )
            race = race_result.scalar_one_or_none()
            if not race:
                raise ValueError("Associated race not found")

            new_result = await generator.generate_race_preview(
                meeting_id=content.meeting_id,
                race_number=race.race_number,
                save=False,
            )
        elif content.content_type == "results":
            from punty.models.meeting import Race
            race_result = await self.db.execute(
                select(Race).where(Race.id == content.race_id)
            )
            race = race_result.scalar_one_or_none()
            if not race:
                raise ValueError("Associated race not found")

            new_result = await generator.generate_results(
                meeting_id=content.meeting_id,
                race_number=race.race_number,
                save=False,
            )
        elif content.content_type == "meeting_wrapup":
            new_result = await generator.generate_meeting_wrapup(
                meeting_id=content.meeting_id,
                save=False,
            )
        else:
            raise ValueError(f"Cannot regenerate content type: {content.content_type}")

        # Apply additional instructions if provided
        if additional_instructions:
            personality = load_prompt("personality")
            adjusted_content = await self.ai_client.generate(
                system_prompt=personality,
                user_prompt=f"""Here's Punty content that needs adjustment:

{new_result['raw_content']}

Instructions for adjustment:
{additional_instructions}

Generate the adjusted version.""",
                temperature=0.7,
            )
            new_result["raw_content"] = adjusted_content

        # Update the content record
        content.raw_content = new_result["raw_content"]
        content.status = ContentStatus.PENDING_REVIEW.value
        content.review_notes = "Regenerated" + (f": {additional_instructions}" if additional_instructions else "")

        # Re-format for Twitter
        from punty.formatters.twitter import format_twitter
        content.twitter_formatted = format_twitter(content.raw_content, content.content_type)

        await self.db.commit()

        logger.info(f"Regenerated content {content_id}")

        return {
            "content_id": content_id,
            "new_content": new_result["raw_content"],
            "status": content.status,
        }

    async def approve_content(self, content_id: str) -> dict:
        """Approve content for delivery."""
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        content.status = ContentStatus.APPROVED.value
        await self.db.commit()

        logger.info(f"Approved content {content_id}")

        return {
            "content_id": content_id,
            "status": content.status,
        }

    async def reject_content(
        self,
        content_id: str,
        reason: Optional[str] = None,
    ) -> dict:
        """Reject content."""
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        content.status = ContentStatus.REJECTED.value
        content.review_notes = reason

        await self.db.commit()

        logger.info(f"Rejected content {content_id}: {reason}")

        return {
            "content_id": content_id,
            "status": content.status,
            "reason": reason,
        }
