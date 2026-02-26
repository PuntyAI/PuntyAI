"""Truncate early mail content for social media posts.

Shows full detail for Race 1-2 only, replaces R3+ with a teaser wrap
and CTA driving traffic to punty.ai. Preserves sequence lanes and
responsible gambling sections.
"""

import re

# First race number to truncate (show R1 and R2 in full)
CUTOFF_RACE = 3

# Regex patterns for detecting sections in raw early mail content
RACE_HEADING_RE = re.compile(
    r"^\*{1,2}Race\s+(\d+)\s*[–—-](.+?)[\*]*\s*$", re.MULTILINE
)
SEQUENCE_HEADING_RE = re.compile(
    r"^#{0,3}\s*\d*\)?\s*\*?SEQUENCE LANES", re.MULTILINE | re.IGNORECASE
)
NUGGETS_HEADING_RE = re.compile(
    r"^#{0,3}\s*\d*\)?\s*\*?NUGGETS FROM", re.MULTILINE | re.IGNORECASE
)
FIND_OUT_RE = re.compile(
    r"^#{0,3}\s*\d*\)?\s*\*?FIND OUT MORE", re.MULTILINE | re.IGNORECASE
)
FINAL_WORD_RE = re.compile(
    r"^#{0,3}\s*\d*\)?\s*\*?FINAL WORD", re.MULTILINE | re.IGNORECASE
)

# Number words for teaser text
_NUM_WORDS = {
    1: "One more race", 2: "Two more races", 3: "Three more races",
    4: "Four more races", 5: "Five more races", 6: "Six more races",
    7: "Seven more races", 8: "Eight more races", 9: "Nine more races",
    10: "Ten more races",
}


def truncate_for_socials(
    raw_content: str,
    venue: str = "",
    meeting_id: str = "",
) -> str:
    """Truncate early mail for social media, keeping R1-R2 in full.

    Everything before Race 3 is kept (title, snapshot, Big 3, R1, R2).
    Races 3+ are replaced with a teaser wrap and CTA to punty.ai.
    Sequence lanes and final word sections are preserved.

    Args:
        raw_content: Full early mail raw content (markdown)
        venue: Venue name for display
        meeting_id: Meeting ID for deep link (e.g. "cranbourne-2026-02-20")

    Returns:
        Truncated content with teaser and CTA
    """
    # Find all race headings
    matches = list(RACE_HEADING_RE.finditer(raw_content))
    if not matches:
        return raw_content

    # Find race numbers and their positions
    race_positions = []
    for m in matches:
        race_num = int(m.group(1))
        race_positions.append((race_num, m.start(), m.group(0).strip()))

    # If fewer than CUTOFF_RACE races, nothing to truncate
    max_race = max(rn for rn, _, _ in race_positions)
    if max_race < CUTOFF_RACE:
        return raw_content

    # Find where Race 3 starts
    cutoff_start = None
    for rn, pos, _ in race_positions:
        if rn >= CUTOFF_RACE:
            cutoff_start = pos
            break

    if cutoff_start is None:
        return raw_content

    # Keep everything before the cutoff race
    kept = raw_content[:cutoff_start].rstrip()

    # Count remaining races and last race number
    remaining_races = [rn for rn, _, _ in race_positions if rn >= CUTOFF_RACE]
    num_remaining = len(remaining_races)
    last_race = max_race

    # Extract sequence lanes section (if present)
    sequences_section = _extract_section(
        raw_content, SEQUENCE_HEADING_RE, [NUGGETS_HEADING_RE, FIND_OUT_RE, FINAL_WORD_RE]
    )

    # Extract nuggets from the track section (if present)
    nuggets_section = _extract_section(
        raw_content, NUGGETS_HEADING_RE, [FIND_OUT_RE, FINAL_WORD_RE]
    )

    # Extract final word section (responsible gambling)
    final_word_section = _extract_section(
        raw_content, FINAL_WORD_RE, []
    )

    # Build teaser
    teaser = _build_teaser(
        num_remaining=num_remaining,
        first_remaining=CUTOFF_RACE,
        last_race=last_race,
        venue=venue,
        meeting_id=meeting_id,
    )

    # Assemble truncated content
    parts = [kept, "", teaser]
    if sequences_section:
        parts.append(sequences_section.strip())
    if nuggets_section:
        parts.append(nuggets_section.strip())
    if final_word_section:
        parts.append(final_word_section.strip())

    return "\n\n".join(parts)


def _extract_section(content: str, start_re, end_patterns: list) -> str | None:
    """Extract a section from content between start pattern and the first end pattern."""
    start_match = start_re.search(content)
    if not start_match:
        return None

    start_pos = start_match.start()

    # Find the earliest end pattern after start
    end_pos = len(content)
    for end_re in end_patterns:
        end_match = end_re.search(content, start_match.end())
        if end_match and end_match.start() < end_pos:
            end_pos = end_match.start()

    return content[start_pos:end_pos].strip()


def _build_teaser(
    num_remaining: int,
    first_remaining: int,
    last_race: int,
    venue: str,
    meeting_id: str,
) -> str:
    """Build the teaser wrap text for truncated social posts."""
    count_text = _NUM_WORDS.get(num_remaining, f"{num_remaining} more races")

    # Build the deep link
    if meeting_id:
        link = f"https://punty.ai/tips/{meeting_id}"
    else:
        link = "https://punty.ai"

    range_str = f"R{first_remaining}–R{last_race}"

    teaser = (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"*{count_text} still to come ({range_str})...*\n"
        f"\n"
        f"Short-priced favourites, roughies at overs, sketchy exotics, "
        f"and at least one race that'll make you question every life decision "
        f"you've ever made. The full breakdowns are waiting.\n"
        f"\n"
        f"Full race-by-race analysis, speed maps, exotics, and Punty's Picks:\n"
        f"\U0001F449 {link}"
    )

    return teaser
