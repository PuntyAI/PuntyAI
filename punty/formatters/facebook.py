"""Facebook Page post formatter."""

import re
from typing import Optional


class FacebookFormatter:
    """Format content for Facebook Page posts.

    Facebook supports up to 63,206 characters per post.
    No character splitting needed. Basic text formatting only.
    """

    MAX_LENGTH = 63206

    @classmethod
    def format(
        cls,
        raw_content: str,
        content_type: str = "early_mail",
        venue: Optional[str] = None,
    ) -> str:
        """Format raw content for a Facebook Page post.

        Args:
            raw_content: The raw markdown content
            content_type: Type of content (early_mail, meeting_wrapup, etc.)
            venue: Venue name

        Returns:
            Formatted plain text for Facebook
        """
        text = cls._clean_markdown(raw_content)

        # Add responsible gambling footer
        if "Gamble Responsibly" not in text:
            text += "\n\nGamble Responsibly. gamblinghelponline.org.au | 1800 858 858"

        # Truncate if somehow over limit
        if len(text) > cls.MAX_LENGTH:
            text = text[: cls.MAX_LENGTH - 50] + "\n\n... Full tips at punty.ai"

        return text

    @classmethod
    def _clean_markdown(cls, content: str) -> str:
        """Clean markdown for Facebook plain text."""
        # Remove section number prefixes like "### 2) MEET SNAPSHOT" -> "MEET SNAPSHOT"
        content = re.sub(r"^#{1,3}\s+\d+\)\s*", "", content, flags=re.MULTILINE)
        # Remove remaining markdown headers -> keep text as uppercase
        content = re.sub(r"^#{1,3}\s+(.+)$", lambda m: m.group(1).upper(), content, flags=re.MULTILINE)
        # Convert **bold** and *bold* markers to plain text (Facebook doesn't render markdown)
        content = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", content)
        # Remove __ italic/bold markers
        content = re.sub(r"__(.+?)__", r"\1", content)
        content = re.sub(r"_(.+?)_", r"\1", content)
        # Clean up triple+ newlines
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()


def format_facebook(
    raw_content: str,
    content_type: str = "early_mail",
    venue: Optional[str] = None,
) -> str:
    """Convenience function for formatting Facebook content."""
    return FacebookFormatter.format(raw_content, content_type, venue)
