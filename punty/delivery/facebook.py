"""Facebook Page delivery via Graph API."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.formatters.facebook import format_facebook

logger = logging.getLogger(__name__)

MAX_DELIVERY_RETRIES = 2
RETRY_DELAYS = [5, 15]  # seconds

GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


class FacebookDelivery:
    """Post content to a Facebook Page via the Graph API.

    Requires:
    - facebook_page_id: The Facebook Page ID
    - facebook_page_access_token: A long-lived Page Access Token
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._page_id: Optional[str] = None
        self._access_token: Optional[str] = None
        self._keys_loaded = False

    async def _load_keys(self):
        """Load API keys from DB."""
        if self._keys_loaded:
            return
        from punty.models.settings import get_api_key
        self._page_id = await get_api_key(self.db, "facebook_page_id")
        self._access_token = await get_api_key(self.db, "facebook_page_access_token")
        self._keys_loaded = True

    async def is_configured(self) -> bool:
        """Check if Facebook delivery is configured."""
        await self._load_keys()
        return bool(self._page_id and self._access_token)

    async def send(self, content_id: str, image_path: Path | None = None) -> dict:
        """Post content to the Facebook Page.

        Args:
            content_id: ID of content to post
            image_path: Optional path to an image to attach

        Returns:
            Dict with post status and Facebook post ID
        """
        from punty.config import settings
        if settings.mock_external:
            logger.info(f"[MOCK] Would post to Facebook: {content_id} with image={image_path}")
            return {"status": "mock", "post_id": f"mock_{content_id}"}

        from punty.models.content import Content, ContentStatus
        from punty.models.meeting import Meeting

        if not await self.is_configured():
            raise ValueError("Facebook not configured. Set Page ID and Access Token in Settings.")

        result = await self.db.execute(select(Content).where(Content.id == content_id))
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        if content.status not in (ContentStatus.APPROVED.value, ContentStatus.SENT.value):
            raise ValueError(f"Content not approved: {content.status}")

        # Get meeting for venue
        meeting_result = await self.db.execute(
            select(Meeting).where(Meeting.id == content.meeting_id)
        )
        meeting = meeting_result.scalar_one_or_none()
        venue = meeting.venue if meeting else None

        # Format content (truncate early mails for social — full content on website)
        formatted = content.facebook_formatted
        if not formatted:
            raw = content.raw_content
            if content.content_type == "early_mail":
                from punty.formatters.truncate import truncate_for_socials
                raw = truncate_for_socials(raw, venue=venue or "", meeting_id=content.meeting_id)
            formatted = format_facebook(raw, content.content_type, venue)
            content.facebook_formatted = formatted

        # Post to Facebook Graph API with retry
        last_error = None
        for attempt in range(MAX_DELIVERY_RETRIES + 1):
            try:
                if image_path and image_path.exists():
                    # Photo post — better reach than text-only
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        with open(image_path, "rb") as f:
                            resp = await client.post(
                                f"{GRAPH_API_BASE}/{self._page_id}/photos",
                                data={
                                    "message": formatted,
                                    "access_token": self._access_token,
                                },
                                files={"source": (image_path.name, f, "image/png")},
                            )
                else:
                    # Text-only fallback
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        resp = await client.post(
                            f"{GRAPH_API_BASE}/{self._page_id}/feed",
                            data={
                                "message": formatted,
                                "access_token": self._access_token,
                            },
                        )

                if resp.status_code != 200:
                    error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    error_msg = error_data.get("error", {}).get("message", resp.text)
                    # Don't retry auth errors (4xx) — only retry network/server errors (5xx)
                    if resp.status_code < 500:
                        content.status = "delivery_failed"
                        await self.db.commit()
                        raise ValueError(f"Facebook API error ({resp.status_code}): {error_msg}")
                    raise httpx.HTTPStatusError(
                        f"Server error {resp.status_code}: {error_msg}",
                        request=resp.request, response=resp,
                    )

                post_data = resp.json()
                post_id = post_data.get("id", "")

                # Update content
                content.sent_at = melb_now_naive()
                content.status = ContentStatus.SENT.value
                content.facebook_id = post_id
                content.sent_to_facebook = True
                await self.db.commit()

                logger.info(f"Posted to Facebook: {content_id} -> {post_id}")

                return {
                    "status": "posted",
                    "content_id": content_id,
                    "post_id": post_id,
                    "url": f"https://facebook.com/{post_id}" if post_id else None,
                }

            except (httpx.HTTPError, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < MAX_DELIVERY_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Facebook send attempt {attempt + 1} failed: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)

        logger.error(f"Facebook API error after {MAX_DELIVERY_RETRIES + 1} attempts: {last_error}")
        content.status = "delivery_failed"
        await self.db.commit()
        raise ValueError(f"Facebook API error: {last_error}")

    async def post_update(self, message: str) -> dict:
        """Post a standalone live update to the Facebook Page.

        Used for celebrations and pace analysis since commenting requires
        pages_manage_engagement (App Review). Posts as a new page post instead.

        Args:
            message: Update text

        Returns:
            Dict with post_id
        """
        if not await self.is_configured():
            raise ValueError("Facebook not configured")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{GRAPH_API_BASE}/{self._page_id}/feed",
                    data={
                        "message": message,
                        "access_token": self._access_token,
                    },
                )

            if resp.status_code != 200:
                error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                error_msg = error_data.get("error", {}).get("message", resp.text)
                raise ValueError(f"Facebook post error ({resp.status_code}): {error_msg}")

            post_data = resp.json()
            post_id = post_data.get("id", "")

            logger.info(f"Posted Facebook update: {post_id}")
            return {
                "status": "posted",
                "post_id": post_id,
            }

        except httpx.HTTPError as e:
            logger.error(f"Facebook update HTTP error: {e}")
            raise ValueError(f"Facebook API error: {e}")

    async def post_comment(self, parent_post_id: str, message: str) -> dict:
        """Post a comment on an existing Facebook post.

        NOTE: Requires pages_manage_engagement permission (App Review needed).
        Use post_update() as a workaround until App Review is approved.

        Args:
            parent_post_id: The Facebook post ID to comment on
            message: Comment text

        Returns:
            Dict with comment_id and parent_post_id
        """
        if not await self.is_configured():
            raise ValueError("Facebook not configured")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{GRAPH_API_BASE}/{parent_post_id}/comments",
                    data={
                        "message": message,
                        "access_token": self._access_token,
                    },
                )

            if resp.status_code != 200:
                error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                error_msg = error_data.get("error", {}).get("message", resp.text)
                raise ValueError(f"Facebook comment error ({resp.status_code}): {error_msg}")

            comment_data = resp.json()
            comment_id = comment_data.get("id", "")

            logger.info(f"Posted Facebook comment on {parent_post_id}: {comment_id}")
            return {
                "status": "commented",
                "comment_id": comment_id,
                "parent_post_id": parent_post_id,
            }

        except httpx.HTTPError as e:
            logger.error(f"Facebook comment HTTP error: {e}")
            raise ValueError(f"Facebook API error: {e}")

    async def delete_post(self, post_id: str) -> dict:
        """Delete a Facebook post."""
        if not await self.is_configured():
            raise ValueError("Facebook not configured")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(
                    f"{GRAPH_API_BASE}/{post_id}",
                    params={"access_token": self._access_token},
                )

            if resp.status_code != 200:
                raise ValueError(f"Facebook delete error ({resp.status_code}): {resp.text}")

            logger.info(f"Deleted Facebook post: {post_id}")
            return {"status": "deleted", "post_id": post_id}

        except httpx.HTTPError as e:
            logger.error(f"Facebook delete HTTP error: {e}")
            raise ValueError(f"Facebook API error: {e}")

    async def get_page_info(self) -> dict:
        """Get Facebook Page info for status display."""
        if not await self.is_configured():
            return {"status": "not_configured"}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{GRAPH_API_BASE}/{self._page_id}",
                    params={
                        "fields": "name,fan_count",
                        "access_token": self._access_token,
                    },
                )

            if resp.status_code != 200:
                return {"status": "error", "error": resp.text}

            data = resp.json()
            return {
                "status": "ok",
                "page_id": self._page_id,
                "name": data.get("name", "Unknown"),
                "followers": data.get("fan_count", 0),
            }

        except Exception as e:
            logger.error(f"Facebook page info error: {e}")
            return {"status": "error", "error": str(e)}
