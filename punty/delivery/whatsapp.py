"""WhatsApp delivery via WhatsApp Business API."""

import logging
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import settings
from punty.formatters.whatsapp import format_whatsapp

logger = logging.getLogger(__name__)


class WhatsAppDelivery:
    """Send content via WhatsApp Business API.

    Requires WhatsApp Business API setup with:
    - WHATSAPP_API_TOKEN
    - WHATSAPP_PHONE_NUMBER_ID
    """

    API_BASE = "https://graph.facebook.com/v18.0"

    def __init__(self, db: AsyncSession):
        self.db = db
        self.token = settings.whatsapp_api_token
        self.phone_number_id = settings.whatsapp_phone_number_id
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def is_configured(self) -> bool:
        """Check if WhatsApp delivery is configured."""
        return bool(self.token and self.phone_number_id)

    async def send(
        self,
        content_id: str,
        recipient_phone: str,
    ) -> dict:
        """Send content to a WhatsApp number.

        Args:
            content_id: ID of content to send
            recipient_phone: Phone number with country code (e.g., "61400000000")

        Returns:
            Dict with send status and message ID
        """
        from punty.models.content import Content, ContentStatus

        if not self.is_configured():
            raise ValueError("WhatsApp API not configured")

        # Get content
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()

        if not content:
            raise ValueError(f"Content not found: {content_id}")

        if content.status != ContentStatus.APPROVED.value:
            raise ValueError(f"Content not approved: {content.status}")

        # Format content
        formatted = content.whatsapp_formatted
        if not formatted:
            formatted = format_whatsapp(content.raw_content, content.content_type)
            content.whatsapp_formatted = formatted

        # Send via API
        url = f"{self.API_BASE}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_phone,
            "type": "text",
            "text": {"body": formatted},
        }

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result_data = response.json()

            # Update content status
            content.sent_to_whatsapp = True
            content.sent_at = datetime.utcnow()
            content.status = ContentStatus.SENT.value
            await self.db.commit()

            logger.info(f"Sent WhatsApp message: {content_id} to {recipient_phone}")

            return {
                "status": "sent",
                "content_id": content_id,
                "recipient": recipient_phone,
                "message_id": result_data.get("messages", [{}])[0].get("id"),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"WhatsApp API error: {e.response.text}")
            raise ValueError(f"WhatsApp API error: {e.response.status_code}")

    async def send_to_group(
        self,
        content_id: str,
        group_id: str,
    ) -> dict:
        """Send content to a WhatsApp group.

        Note: WhatsApp Business API has limited group support.
        This may require additional setup.
        """
        # Similar to send() but with group endpoint
        raise NotImplementedError("Group messaging requires additional API setup")

    async def get_template_status(self) -> dict:
        """Check status of message templates (for template messages)."""
        if not self.is_configured():
            return {"status": "not_configured"}

        url = f"{self.API_BASE}/{self.phone_number_id}/message_templates"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error checking templates: {e}")
            return {"status": "error", "error": str(e)}
