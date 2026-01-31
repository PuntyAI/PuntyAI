"""Console output for previewing content."""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.formatters.whatsapp import format_whatsapp
from punty.formatters.twitter import format_twitter

logger = logging.getLogger(__name__)


class ConsoleDelivery:
    """Output content to console for preview/copy-paste."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def preview(
        self,
        content_id: str,
        platform: str = "whatsapp",
    ) -> dict:
        """Preview formatted content for a platform.

        Args:
            content_id: ID of content to preview
            platform: Target platform (whatsapp, twitter)

        Returns:
            Dict with formatted content and metadata
        """
        from punty.models.content import Content
        from punty.models.meeting import Meeting

        # Get content
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        # Get meeting for venue
        meeting_result = await self.db.execute(
            select(Meeting).where(Meeting.id == content.meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        venue = meeting.venue if meeting else None

        # Format based on platform
        if platform == "whatsapp":
            formatted = format_whatsapp(content.raw_content, content.content_type)
        elif platform == "twitter":
            formatted = format_twitter(content.raw_content, content.content_type, venue)
        else:
            formatted = content.raw_content

        return {
            "content_id": content_id,
            "platform": platform,
            "formatted": formatted,
            "character_count": len(formatted),
            "content_type": content.content_type,
            "meeting_id": content.meeting_id,
            "status": content.status,
        }

    async def preview_twitter_thread(
        self,
        content_id: str,
    ) -> dict:
        """Preview content as Twitter thread."""
        from punty.models.content import Content
        from punty.models.meeting import Meeting

        # Get content
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        # Get meeting for venue
        meeting_result = await self.db.execute(
            select(Meeting).where(Meeting.id == content.meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        venue = meeting.venue if meeting else None

        # Format as single long-form post
        tweet_text = format_twitter(content.raw_content, content.content_type, venue)

        return {
            "content_id": content_id,
            "platform": "twitter",
            "tweet_count": 1,
            "tweets": [tweet_text],
            "total_characters": len(tweet_text),
        }

    def print_preview(
        self,
        formatted: str,
        platform: str,
        content_type: str,
    ) -> None:
        """Print formatted preview to console."""
        separator = "=" * 50

        print(f"\n{separator}")
        print(f"  {platform.upper()} PREVIEW - {content_type.replace('_', ' ').upper()}")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(separator)
        print()
        print(formatted)
        print()
        print(separator)
        print(f"  Character count: {len(formatted)}")
        print(separator)

    def print_thread_preview(
        self,
        tweets: list[str],
        content_type: str,
    ) -> None:
        """Print Twitter thread preview to console."""
        separator = "=" * 50

        print(f"\n{separator}")
        print(f"  TWITTER THREAD PREVIEW - {content_type.replace('_', ' ').upper()}")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Tweets: {len(tweets)}")
        print(separator)

        for i, tweet in enumerate(tweets, 1):
            print(f"\n--- Tweet {i}/{len(tweets)} ({len(tweet)} chars) ---")
            print(tweet)

        print(f"\n{separator}")
        print(f"  Total characters: {sum(len(t) for t in tweets)}")
        print(separator)
