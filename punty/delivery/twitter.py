"""Twitter/X delivery via Tweepy (Twitter API v2 with OAuth 1.0a)."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import settings, melb_now_naive
from punty.formatters.twitter import format_twitter

logger = logging.getLogger(__name__)


class TwitterDelivery:
    """Send content via Twitter API v2 using Tweepy.

    Requires Twitter API setup with:
    - TWITTER_API_KEY (Consumer Key)
    - TWITTER_API_SECRET (Consumer Secret)
    - TWITTER_ACCESS_TOKEN
    - TWITTER_ACCESS_SECRET
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._api_key = None
        self._api_secret = None
        self._access_token = None
        self._access_secret = None
        self._client = None
        self._keys_loaded = False

    async def _load_keys(self):
        """Load API keys from DB, falling back to config."""
        if self._keys_loaded:
            return
        from punty.models.settings import get_api_key
        self._api_key = await get_api_key(self.db, "twitter_api_key", settings.twitter_api_key)
        self._api_secret = await get_api_key(self.db, "twitter_api_secret", settings.twitter_api_secret)
        self._access_token = await get_api_key(self.db, "twitter_access_token", settings.twitter_access_token)
        self._access_secret = await get_api_key(self.db, "twitter_access_secret", settings.twitter_access_secret)
        self._keys_loaded = True

    def _get_client(self):
        """Get authenticated Tweepy client."""
        if self._client is None:
            try:
                import tweepy

                self._client = tweepy.Client(
                    consumer_key=self._api_key,
                    consumer_secret=self._api_secret,
                    access_token=self._access_token,
                    access_token_secret=self._access_secret,
                )
            except ImportError:
                raise ValueError("tweepy not installed. Run: pip install tweepy")
        return self._client

    async def is_configured(self) -> bool:
        """Check if Twitter delivery is configured."""
        await self._load_keys()
        return bool(
            self._api_key
            and self._api_secret
            and self._access_token
            and self._access_secret
        )

    async def send(self, content_id: str) -> dict:
        """Post content as a single tweet.

        Args:
            content_id: ID of content to post

        Returns:
            Dict with post status and tweet ID
        """
        from punty.models.content import Content, ContentStatus
        from punty.models.meeting import Meeting

        if not await self.is_configured():
            raise ValueError("Twitter API not configured. Set keys via Settings page or .env file")

        # Get content
        result = await self.db.execute(select(Content).where(Content.id == content_id))
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        if content.status != ContentStatus.APPROVED.value:
            raise ValueError(f"Content not approved: {content.status}")

        # Get meeting for venue
        meeting_result = await self.db.execute(
            select(Meeting).where(Meeting.id == content.meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        venue = meeting.venue if meeting else None

        # Format content
        formatted = content.twitter_formatted
        if not formatted:
            formatted = format_twitter(content.raw_content, content.content_type, venue)
            content.twitter_formatted = formatted

        # Check length
        if len(formatted) > 280:
            raise ValueError(
                f"Tweet too long: {len(formatted)} chars. Use send_thread() instead."
            )

        # Post tweet (blocking Tweepy call → run in thread)
        try:
            client = self._get_client()
            response = await asyncio.to_thread(client.create_tweet, text=formatted)
            tweet_id = response.data["id"]

            # Update content status
            content.sent_at = melb_now_naive()
            content.status = ContentStatus.SENT.value
            await self.db.commit()

            logger.info(f"Posted tweet: {content_id} -> {tweet_id}")

            return {
                "status": "posted",
                "content_id": content_id,
                "tweet_id": tweet_id,
                "url": f"https://twitter.com/i/status/{tweet_id}",
            }

        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            raise ValueError(f"Twitter API error: {e}")

    async def send_thread(self, content_id: str) -> dict:
        """Post content as a thread of tweets.

        Args:
            content_id: ID of content to post

        Returns:
            Dict with thread status and tweet IDs
        """
        from punty.models.content import Content, ContentStatus
        from punty.models.meeting import Meeting

        if not await self.is_configured():
            raise ValueError("Twitter API not configured")

        # Get content
        result = await self.db.execute(select(Content).where(Content.id == content_id))
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        if content.status != ContentStatus.APPROVED.value:
            raise ValueError(f"Content not approved: {content.status}")

        # Get meeting for venue
        meeting_result = await self.db.execute(
            select(Meeting).where(Meeting.id == content.meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        venue = meeting.venue if meeting else None

        # Format as single long-form post (X Premium supports long posts)
        tweet_text = format_twitter(content.raw_content, content.content_type, venue)

        # Post single tweet (blocking → thread)
        try:
            client = self._get_client()
            response = await asyncio.to_thread(client.create_tweet, text=tweet_text)
            tweet_id = response.data["id"]
            logger.info(f"Posted tweet: {tweet_id}")
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            raise ValueError(f"Twitter API error: {e}")

        # Update content status
        content.sent_at = melb_now_naive()
        content.status = ContentStatus.SENT.value
        await self.db.commit()

        return {
            "status": "posted",
            "content_id": content_id,
            "tweet_count": 1,
            "tweet_ids": [tweet_id],
            "tweet_url": f"https://twitter.com/i/status/{tweet_id}",
        }

    async def post_tweet(self, text: str, long_form: bool = False) -> dict:
        """Post a standalone tweet (not linked to content).

        Args:
            text: Tweet text (max 280 chars for regular, 25000 for verified/long_form)
            long_form: If True, allows longer posts for verified accounts

        Returns:
            Dict with tweet ID and URL
        """
        if not await self.is_configured():
            raise ValueError("Twitter API not configured")

        max_length = 25000 if long_form else 280
        if len(text) > max_length:
            raise ValueError(f"Tweet too long: {len(text)} chars (max {max_length})")

        try:
            client = self._get_client()
            response = await asyncio.to_thread(client.create_tweet, text=text)
            tweet_id = response.data["id"]

            logger.info(f"Posted {'long-form' if long_form else 'standard'} tweet: {tweet_id}")

            return {
                "status": "posted",
                "tweet_id": tweet_id,
                "url": f"https://twitter.com/i/status/{tweet_id}",
                "length": len(text),
            }
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            raise ValueError(f"Twitter API error: {e}")

    async def send_long_post(self, content_id: str) -> dict:
        """Post content as a single long-form post (for verified accounts).

        Args:
            content_id: ID of content to post

        Returns:
            Dict with post status and tweet ID
        """
        from punty.models.content import Content, ContentStatus
        from punty.models.meeting import Meeting

        if not await self.is_configured():
            raise ValueError("Twitter API not configured")

        # Get content
        result = await self.db.execute(select(Content).where(Content.id == content_id))
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        if content.status != ContentStatus.APPROVED.value:
            raise ValueError(f"Content not approved: {content.status}")

        # Get meeting for venue
        meeting_result = await self.db.execute(
            select(Meeting).where(Meeting.id == content.meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        venue = meeting.venue if meeting else "Racing"

        # Format for long-form post
        post_text = self._format_long_post(content.raw_content, content.content_type, venue)

        # Post as long-form (blocking → thread)
        try:
            client = self._get_client()
            response = await asyncio.to_thread(client.create_tweet, text=post_text)
            tweet_id = response.data["id"]

            # Update content status
            content.sent_at = melb_now_naive()
            content.status = ContentStatus.SENT.value
            await self.db.commit()

            logger.info(f"Posted long-form post: {content_id} -> {tweet_id} ({len(post_text)} chars)")

            return {
                "status": "posted",
                "content_id": content_id,
                "tweet_id": tweet_id,
                "url": f"https://twitter.com/i/status/{tweet_id}",
                "length": len(post_text),
            }

        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            raise ValueError(f"Twitter API error: {e}")

    def _format_long_post(self, raw_content: str, content_type: str, venue: str) -> str:
        """Format content for a single long-form X post."""
        from punty.formatters.twitter import TwitterFormatter

        # Add header
        header = f"🏇 PUNTY'S {venue.upper()} TIPS\n\n"

        # Clean markdown - X doesn't render markdown, shows raw asterisks/hashes
        text = TwitterFormatter._clean_markdown(raw_content)

        # Add footer with hashtags
        footer = f"\n\n#AusRacing #HorseRacing #{venue.replace(' ', '')}Racing\n\nGamble Responsibly. gamblinghelponline.org.au | 1800 858 858"

        full_post = header + text + footer

        # Ensure under 25000 chars
        if len(full_post) > 25000:
            # Truncate if needed
            full_post = full_post[:24900] + "\n\n... [continued on punty.ai]"

        return full_post

    async def delete_tweet(self, tweet_id: str) -> dict:
        """Delete a tweet."""
        if not await self.is_configured():
            raise ValueError("Twitter API not configured")

        try:
            client = self._get_client()
            await asyncio.to_thread(client.delete_tweet, tweet_id)
            logger.info(f"Deleted tweet: {tweet_id}")
            return {"status": "deleted", "tweet_id": tweet_id}
        except Exception as e:
            logger.error(f"Error deleting tweet: {e}")
            raise ValueError(f"Twitter API error: {e}")

    async def get_me(self) -> dict:
        """Get authenticated user info."""
        if not await self.is_configured():
            return {"status": "not_configured"}

        try:
            client = self._get_client()
            user = await asyncio.to_thread(client.get_me)
            return {
                "status": "ok",
                "id": user.data.id,
                "username": user.data.username,
                "name": user.data.name,
            }
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return {"status": "error", "error": str(e)}

    async def preview_thread(self, content_id: str) -> dict:
        """Preview how content would be split into a thread (without posting).

        Args:
            content_id: ID of content to preview

        Returns:
            Dict with tweets preview
        """
        from punty.models.content import Content
        from punty.models.meeting import Meeting

        # Get content
        result = await self.db.execute(select(Content).where(Content.id == content_id))
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
            "tweet_count": 1,
            "tweets": [
                {"index": 1, "text": tweet_text, "length": len(tweet_text)},
            ],
        }
