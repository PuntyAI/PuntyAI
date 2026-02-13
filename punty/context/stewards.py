"""Parse stewards report comments to extract standardized excuse flags.

Identifies valid reasons for poor performance from race stewards reports,
helping the AI distinguish between horses that ran poorly and those that
had legitimate excuses (held up, raced wide, slow start, etc.).
"""

from __future__ import annotations

import re

# Excuse categories with compiled regex patterns.
# Order matters: checked before the result is returned sorted by severity.
_EXCUSE_PATTERNS: dict[str, list[re.Pattern]] = {}

_RAW_PATTERNS = {
    "held_up": [
        r"held\s+up",
        r"\bchecked\b",
        r"\bsteadied\b",
        r"blocked\s+for\s+(?:a\s+)?run",
        r"couldn'?t\s+obtain\s+clear",
        r"shift(?:ed)?\s+out\s+for\s+clear",
        r"unable\s+to\s+improve",
        r"tight(?:ened)?\s+for\s+room",
        r"\bno\s+room\b",
        r"hemmed\s+in",
        r"couldn'?t\s+(?:get|find)\s+(?:a\s+)?clear",
        r"shuffled\s+back",
    ],
    "wide": [
        r"raced?\s+wide",
        r"three[\s-]?wide",
        r"four[\s-]?wide",
        r"five[\s-]?wide",
        r"without\s+cover",
        r"wide\s+throughout",
        r"wide\s+(?:on|entering|rounding)\s+the\s+(?:turn|bend|home)",
        r"covered\s+extra\s+ground",
        r"wide\s+and\s+without\s+cover",
    ],
    "slow_start": [
        r"slow\s+to\s+begin",
        r"beg(?:a|u)n\s+awkwardly",
        r"\bdwelt\b",
        r"slow\s+into\s+stride",
        r"slow\s+away",
        r"missed\s+the\s+(?:start|break|jump|kick)",
        r"slow\s+out",
    ],
    "bumped": [
        r"\bbumped\b",
        r"\bhampered\b",
        r"\bcrowded\b",
        r"\bsqueezed\b",
        r"clipped\s+heels",
        r"\bcontacted\b",
        r"\bknuckled\b",
    ],
    "interference": [
        r"shifted?\s+(?:in|out)\s+(?:abruptly|sharply)",
        r"\bhung\s+(?:in|out|badly)",
        r"\blugged\s+(?:in|out)",
        r"\blaid\s+(?:in|out)",
        r"over[\s-]?raced",
        r"raced\s+keenly",
        r"raced\s+fiercely",
    ],
    "ran_on": [
        r"ran\s+on\s+well",
        r"finishing\s+(?:off\s+)?strongly",
        r"\blate\s+(?:run|burst|dash)",
        r"closing\s+stages",
        r"finishing\s+fast",
        r"best\s+work\s+late",
        r"hit\s+the\s+line\s+(?:well|strongly|hard)",
    ],
    "eased": [
        r"not\s+(?:fully\s+)?(?:tested|extended)",
        r"rider\s+eased",
        r"eased\s+(?:down|in\s+the\s+final)",
        r"not\s+(?:knocked\s+about|persevered\s+with)",
    ],
    "medical": [
        r"\bbled\b",
        r"\bbleeding\b",
        r"blood\s+at\s+(?:the\s+)?nostrils?",
        r"\blame\b",
        r"\blameness\b",
        r"went\s+amiss",
        r"lost\s+(?:a\s+|the\s+)?(?:near|off)?\s*(?:fore|hind)?\s*shoe",
        r"lost\s+(?:a\s+|the\s+)?plate",
        r"cast\s+(?:a\s+|the\s+)?(?:near|off)?\s*(?:fore|hind)?\s*(?:shoe|plate)",
        r"\binjur(?:y|ed|ies)\b",
        r"pulled\s+up",
        r"did\s+not\s+finish",
        r"\bdnf\b",
        r"eased\s+out\s+of\s+the\s+race",
        r"failed\s+to\s+finish",
        r"broke\s+down",
        r"cardiac",
        r"respiratory",
        r"laboured\s+(?:breathing|in\s+the\s+latter)",
        r"dislodged\s+(?:the\s+)?rider",
        r"\bfell\b",
        r"pulled\s+(?:himself|herself|itself)\s+up",
    ],
}

# Severity ordering (most impactful excuse first)
_SEVERITY = ["medical", "held_up", "wide", "bumped", "slow_start", "interference", "ran_on", "eased"]

# Compile patterns once
for category, patterns in _RAW_PATTERNS.items():
    _EXCUSE_PATTERNS[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

# Human-readable labels
_LABELS = {
    "medical": "medical/physical issue",
    "held_up": "held up",
    "wide": "raced wide",
    "slow_start": "slow start",
    "bumped": "bumped/hampered",
    "eased": "not tested",
    "ran_on": "ran on late",
    "interference": "interference",
}


def parse_stewards_excuses(comment: str) -> list[str]:
    """Extract standardized excuse flags from a stewards comment.

    Returns a list of excuse category keys sorted by severity,
    e.g. ["held_up", "wide"].
    """
    if not comment or not isinstance(comment, str):
        return []

    found = set()
    for category, patterns in _EXCUSE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(comment):
                found.add(category)
                break  # one match per category is enough

    # Sort by severity
    return [e for e in _SEVERITY if e in found]


def format_excuse_summary(excuses: list[str]) -> str:
    """Convert excuse flags to human-readable text.

    e.g. ["held_up", "wide"] -> "held up, raced wide"
    """
    return ", ".join(_LABELS.get(e, e.replace("_", " ")) for e in excuses)


def extract_form_excuses(form_history: list[dict], max_starts: int = 5) -> list[dict]:
    """Extract excuses from a runner's form history.

    Only flags excuses for starts where the horse finished outside the top 3,
    since a horse that won despite racing wide doesn't need an excuse.

    Args:
        form_history: List of past start dicts (most recent first).
        max_starts: Maximum number of starts to analyse.

    Returns:
        List of dicts with excuse details for starts that had valid excuses.
    """
    if not form_history or not isinstance(form_history, list):
        return []

    results = []
    for i, start in enumerate(form_history[:max_starts]):
        comment = start.get("comment") or ""
        if not comment:
            continue

        # Parse finishing position
        pos = start.get("position") or start.get("pos")
        try:
            pos_int = int(pos)
        except (TypeError, ValueError):
            continue

        # Only flag excuses for poor finishes (outside top 3)
        if pos_int <= 3:
            continue

        excuses = parse_stewards_excuses(comment)
        if not excuses:
            continue

        results.append({
            "run_index": i,
            "venue": start.get("venue", ""),
            "distance": start.get("distance"),
            "position": pos_int,
            "excuses": excuses,
            "excuse_text": format_excuse_summary(excuses),
        })

    return results
