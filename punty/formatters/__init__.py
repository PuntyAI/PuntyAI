"""Platform-specific content formatters."""

import re

from punty.formatters.twitter import TwitterFormatter

__all__ = ["TwitterFormatter", "strip_json_block"]

_JSON_BLOCK_RE = re.compile(r'\n*```json\s*\n.*?\n```\s*$', re.DOTALL)


def strip_json_block(content: str) -> str:
    """Remove the structured JSON data block from content (parser-only, not for display)."""
    return _JSON_BLOCK_RE.sub('', content)
