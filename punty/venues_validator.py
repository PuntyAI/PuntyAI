"""AI-powered venue validation for unknown racecourses.

When the calendar returns a venue not in our 400+ venue registry, this module
uses a Claude Haiku call to check if it's a real Australian racecourse and
return canonical info. This catches cases like "Kingscote" being used for
King Island, or completely made-up venue names from scraper errors.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def validate_unknown_venue(venue: str) -> dict:
    """Validate an unknown venue name using Claude Haiku.

    Returns dict with:
        valid: bool — is this a real Australian racecourse?
        canonical_name: str — the correct/canonical name
        state: str — state code (NSW, VIC, QLD, SA, WA, TAS, NT, ACT)
        notes: str — any disambiguation info
    """
    try:
        import anthropic
        from punty.models.database import async_session
        from punty.models.settings import get_api_key

        async with async_session() as db:
            api_key = await get_api_key(db, "anthropic_api_key")

        if not api_key:
            logger.warning("No Anthropic API key — skipping venue validation")
            return {"valid": True, "canonical_name": venue, "state": "", "notes": "no API key"}

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Is '{venue}' a real Australian horse racing venue (thoroughbred racecourse)? "
                    f"Reply in exactly this format:\n"
                    f"VALID: yes or no\n"
                    f"NAME: canonical venue name\n"
                    f"STATE: state code (NSW/VIC/QLD/SA/WA/TAS/NT/ACT)\n"
                    f"NOTES: any disambiguation (e.g. if name is commonly confused with another venue)\n"
                    f"Be concise. If unsure, say no."
                ),
            }],
        )

        text = response.content[0].text.strip()
        result = _parse_validation_response(text, venue)
        logger.info(f"Venue validation for '{venue}': {result}")
        return result

    except Exception as e:
        logger.warning(f"Venue validation failed for '{venue}': {e}")
        # Fail open — don't block meetings on validation errors
        return {"valid": True, "canonical_name": venue, "state": "", "notes": f"validation error: {e}"}


def _parse_validation_response(text: str, original_venue: str) -> dict:
    """Parse the structured response from Claude."""
    result = {
        "valid": True,
        "canonical_name": original_venue,
        "state": "",
        "notes": "",
    }

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("VALID:"):
            val = line.split(":", 1)[1].strip().lower()
            result["valid"] = val in ("yes", "true", "y")
        elif line.upper().startswith("NAME:"):
            result["canonical_name"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("STATE:"):
            result["state"] = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("NOTES:"):
            result["notes"] = line.split(":", 1)[1].strip()

    return result


async def validate_calendar_venues(meetings: list[dict]) -> list[dict]:
    """Filter calendar meetings, validating any unknown venues.

    Known venues pass through immediately. Unknown venues get an AI check.
    Invalid venues are removed with a warning.

    Returns filtered list of meetings.
    """
    from punty.venues import is_known_venue, normalize_venue

    valid_meetings = []
    for m in meetings:
        venue = m.get("venue", "")
        normalized = normalize_venue(venue)

        if is_known_venue(normalized):
            valid_meetings.append(m)
            continue

        # Unknown venue — validate with AI
        logger.info(f"Unknown venue '{venue}' (normalized: '{normalized}') — validating with AI")
        result = await validate_unknown_venue(venue)

        if result["valid"]:
            logger.info(
                f"AI validated '{venue}' as real: {result['canonical_name']} "
                f"({result['state']}). {result['notes']}"
            )
            valid_meetings.append(m)
        else:
            logger.warning(
                f"AI rejected venue '{venue}': {result['notes']}. "
                f"Skipping this meeting."
            )

    return valid_meetings
