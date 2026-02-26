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
        from punty.formatters import strip_json_block
        text = cls._clean_markdown(strip_json_block(raw_content))

        # Add emoji prefix (matches Twitter styling)
        text = "🏇 " + text

        # Add hashtag + responsible gambling footer (matches Twitter styling)
        venue_tag = f"#{venue.replace(' ', '')}Racing" if venue else ""
        footer_parts = ["#AusRacing #HorseRacing"]
        if venue_tag:
            footer_parts.append(venue_tag)
        footer = " ".join(footer_parts)

        if "Gamble Responsibly" not in text:
            text += f"\n\n{footer}\n\nGamble Responsibly. gamblinghelponline.org.au | 1800 858 858"
        else:
            # Insert hashtags before the existing Gamble Responsibly line
            text = text.replace(
                "Gamble Responsibly",
                f"{footer}\n\nGamble Responsibly",
                1,
            )

        # Truncate if somehow over limit
        if len(text) > cls.MAX_LENGTH:
            text = text[: cls.MAX_LENGTH - 50] + "\n\n... Full tips at punty.ai"

        return text

    # Unicode bold sans-serif mapping (A-Z, a-z, 0-9)
    _BOLD_MAP = str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵",
    )

    @classmethod
    def to_bold(cls, text: str) -> str:
        """Convert text to Unicode bold sans-serif characters."""
        return text.translate(cls._BOLD_MAP)

    @classmethod
    def _clean_markdown(cls, content: str) -> str:
        """Clean markdown for Facebook plain text with Unicode bold headings."""
        # Convert title line to Unicode bold heading
        content = re.sub(
            r'^\*PUNTY EARLY MAIL([^*]*)\*',
            lambda m: cls.to_bold(f"PUNTY EARLY MAIL{m.group(1)}"),
            content,
            flags=re.IGNORECASE,
        )
        # Convert markdown headers to Unicode bold (handle optional "2) " numbering)
        content = re.sub(
            r"^#{1,3}\s+(?:\d+\)\s*)?(.+)$",
            lambda m: cls.to_bold(m.group(1).upper()),
            content,
            flags=re.MULTILINE,
        )
        # Convert markdown links [text](url) -> "text — url" or just url if same
        def _clean_link(m):
            text, url = m.group(1), m.group(2)
            # If link text is the URL itself (or very similar), just show the URL
            if text.strip().rstrip("/") == url.strip().rstrip("/") or text.startswith("http"):
                return url
            return f"{text} — {url}"
        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _clean_link, content)
        # Convert **bold** to Unicode bold, *italic* to plain
        content = re.sub(r"\*\*(.+?)\*\*", lambda m: cls.to_bold(m.group(1)), content)
        content = re.sub(r"\*(.+?)\*", r"\1", content)
        # Remove __ italic/bold markers
        content = re.sub(r"__(.+?)__", r"\1", content)
        content = re.sub(r"_(.+?)_", r"\1", content)
        # Bold race structural lines (after markdown markers are stripped)
        content = re.sub(
            r"^(Race \d+\s*[–—-].+)$",
            lambda m: cls.to_bold(m.group(1)),
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"^(Top \d+.+\(\$\d+ pool\).*)$",
            lambda m: cls.to_bold(m.group(1)),
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"^(Punty.s Pick:)",
            lambda m: cls.to_bold(m.group(1)),
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"^(Roughie:)",
            lambda m: cls.to_bold(m.group(1)),
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"^(Degenerate Exotic.+)$",
            lambda m: cls.to_bold(m.group(1)),
            content,
            flags=re.MULTILINE,
        )
        # Clean up triple+ newlines
        content = re.sub(r"\n{3,}", "\n\n", content)
        # Break Facebook auto-emoticon sequences (e.g. 8) → 😎, :) → 😊)
        # Insert zero-width space between trigger characters and parentheses
        ZWS = "\u200b"
        content = re.sub(r"(\d)([)\]])", rf"\1{ZWS}\2", content)
        content = re.sub(r"([:;B])([)(])", rf"\1{ZWS}\2", content)
        return content.strip()


def format_facebook(
    raw_content: str,
    content_type: str = "early_mail",
    venue: Optional[str] = None,
) -> str:
    """Convenience function for formatting Facebook content."""
    return FacebookFormatter.format(raw_content, content_type, venue)
