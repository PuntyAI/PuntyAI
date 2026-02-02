"""WhatsApp message formatter."""

import re
from typing import Optional


class WhatsAppFormatter:
    """Format content for WhatsApp messages.

    WhatsApp formatting:
    - *bold*
    - _italic_
    - ~strikethrough~
    - ```monospace```
    - Emojis supported
    - Max message length: 65,536 characters (but keep under 4096 for readability)
    - Line breaks preserved
    """

    MAX_LENGTH = 4000  # Keep messages readable

    # Racing-related emojis
    EMOJIS = {
        "horse": "\U0001F40E",      # horse
        "racing": "\U0001F3C7",      # horse racing
        "fire": "\U0001F525",        # fire (hot tip)
        "money": "\U0001F4B0",       # money bag
        "trophy": "\U0001F3C6",      # trophy
        "star": "\u2B50",            # star
        "check": "\u2705",           # green check
        "cross": "\u274C",           # red cross
        "alert": "\U0001F6A8",       # alert
        "chart": "\U0001F4C8",       # chart up
        "dice": "\U0001F3B2",        # dice
        "pin": "\U0001F4CD",         # pin/location
        "clock": "\u23F0",           # clock
        "sun": "\u2600\uFE0F",       # sun
        "rain": "\U0001F327\uFE0F",  # rain
        "thumbs_up": "\U0001F44D",   # thumbs up
    }

    @classmethod
    def format(cls, raw_content: str, content_type: str = "early_mail") -> str:
        """Format raw content for WhatsApp.

        Args:
            raw_content: The raw content to format
            content_type: Type of content (affects formatting choices)

        Returns:
            WhatsApp-formatted content
        """
        # Start with the raw content
        content = raw_content

        # Convert markdown headers to bold
        content = cls._convert_headers(content)

        # Convert markdown bold/italic
        content = cls._convert_markdown(content)

        # Add appropriate emojis
        content = cls._add_emojis(content, content_type)

        # Format race numbers nicely
        content = cls._format_race_numbers(content)

        # Ensure responsible gambling footer
        content = cls._ensure_footer(content)

        # Truncate if too long
        if len(content) > cls.MAX_LENGTH:
            content = cls._truncate(content)

        return content

    @classmethod
    def _convert_headers(cls, content: str) -> str:
        """Convert markdown headers to WhatsApp bold."""
        def _header_to_bold(m):
            text = m.group(1).strip()
            # Strip existing * wrapping to avoid double-bold
            text = text.strip('*').strip()
            return f'*{text}*'
        content = re.sub(r'^#{1,3}\s+(.+)$', _header_to_bold, content, flags=re.MULTILINE)
        return content

    @classmethod
    def _convert_markdown(cls, content: str) -> str:
        """Convert markdown bold/italic to WhatsApp format."""
        # **bold** -> *bold*
        content = re.sub(r'\*\*(.+?)\*\*', r'*\1*', content)
        # __bold__ -> *bold*
        content = re.sub(r'__(.+?)__', r'*\1*', content)
        # _italic_ stays the same
        # Clean up any accidental double asterisks from mixed formatting
        content = re.sub(r'\*{2,}', '*', content)
        return content

    @classmethod
    def _add_emojis(cls, content: str, content_type: str) -> str:
        """Add appropriate emojis based on content type and keywords."""
        e = cls.EMOJIS

        # Add header emoji based on content type
        if content_type == "early_mail":
            if not content.startswith(e["racing"]):
                content = f"{e['racing']} *PUNTY'S EARLY MAIL* {e['racing']}\n\n" + content.lstrip()
        elif content_type == "race_preview":
            if not content.startswith(e["horse"]):
                content = f"{e['horse']} " + content
        elif content_type == "results":
            if "won" in content.lower() or "winner" in content.lower():
                content = f"{e['trophy']} " + content
        elif content_type == "update_alert":
            if not content.startswith(e["alert"]):
                content = f"{e['alert']} *UPDATE ALERT* {e['alert']}\n\n" + content.lstrip()

        # Add emojis for key phrases
        replacements = [
            (r'\bBEST BET\b', f"{e['fire']} *BEST BET*"),
            (r'\bNEXT BEST\b', f"{e['star']} *NEXT BEST*"),
            (r'\bROUGHIE\b', f"{e['dice']} *ROUGHIE*"),
            (r'\bVALUE\b', f"{e['money']} *VALUE*"),
            (r'\bWINNER\b', f"{e['trophy']} *WINNER*"),
            (r'\bTIP\b', f"{e['thumbs_up']} TIP"),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE, count=1)

        return content

    @classmethod
    def _format_race_numbers(cls, content: str) -> str:
        """Format race numbers consistently."""
        # R1, Race 1, Race 1: all become *Race 1*
        # But skip if already inside bold markers
        content = re.sub(r'(?<!\*)\bRace\s*(\d+)(?:[:.])?(?!\*)', r'*Race \1*', content, flags=re.IGNORECASE)
        content = re.sub(r'(?<!\*)\bR(\d+)\b(?!\*)', r'*Race \1*', content, flags=re.IGNORECASE)
        # Clean up any double bold from overlapping replacements
        content = re.sub(r'\*{2,}', '*', content)
        return content

    @classmethod
    def _ensure_footer(cls, content: str) -> str:
        """Ensure content has responsible gambling footer."""
        gambling_keywords = ["gamble responsibly", "1800 858 858", "know your limits"]

        has_footer = any(kw.lower() in content.lower() for kw in gambling_keywords)

        if not has_footer:
            content = content.rstrip() + f"\n\n{cls.EMOJIS['dice']} Gamble responsibly | 1800 858 858"

        return content

    @classmethod
    def _truncate(cls, content: str, suffix: str = "\n\n...[continued]") -> str:
        """Truncate content to max length."""
        if len(content) <= cls.MAX_LENGTH:
            return content

        # Find last complete paragraph within limit
        limit = cls.MAX_LENGTH - len(suffix)
        truncated = content[:limit]

        # Try to break at paragraph
        last_para = truncated.rfind("\n\n")
        if last_para > limit * 0.7:  # At least 70% of content
            truncated = truncated[:last_para]

        return truncated + suffix


def format_whatsapp(raw_content: str, content_type: str = "early_mail") -> str:
    """Convenience function for formatting WhatsApp content."""
    return WhatsAppFormatter.format(raw_content, content_type)
