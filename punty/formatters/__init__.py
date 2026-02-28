"""Platform-specific content formatters."""

import re

from punty.formatters.twitter import TwitterFormatter

__all__ = ["TwitterFormatter", "strip_json_block"]

# Fenced JSON block: ```json ... ```
_JSON_FENCED_RE = re.compile(r'\n*```json\s*\n.*?\n```\s*$', re.DOTALL)
# Unclosed fence: ```json ... (no closing ```) — AI sometimes forgets to close
_JSON_UNCLOSED_RE = re.compile(r'\n*```json\s*\n.*$', re.DOTALL)
# Raw JSON object at end of content (no fences at all)
_JSON_RAW_TAIL_RE = re.compile(r'\n*\{\s*\n\s*"(?:big3|races|selections)".*$', re.DOTALL)


def strip_json_block(content: str) -> str:
    """Remove the structured JSON data block from content (parser-only, not for display)."""
    result = _JSON_FENCED_RE.sub('', content)
    if result != content:
        return result
    # Fallback: unclosed fence
    result = _JSON_UNCLOSED_RE.sub('', content)
    if result != content:
        return result
    # Fallback: raw JSON tail
    return _JSON_RAW_TAIL_RE.sub('', content)
