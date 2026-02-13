"""Standard time comparison for race form analysis.

Compares each runner's recent race times against pre-computed
venue/distance/condition standard times from deep learning data.
Identifies horses that have raced in genuinely fast or slow races.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Cache standard times per process (loaded once from DB)
_standard_times_cache: dict | None = None


def parse_race_time(time_str: str | None) -> float | None:
    """Parse a race time string to seconds.

    Handles formats like:
    - "1:35.20" (MM:SS.ss) -> 95.20
    - "0:58.40" -> 58.40
    - "2:05.10" -> 125.10
    - "58.40" (SS.ss, no colon) -> 58.40
    """
    if not time_str or not isinstance(time_str, str):
        return None

    time_str = time_str.strip()

    try:
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        else:
            # Might be just seconds (e.g., "58.40")
            val = float(time_str)
            if 30 <= val <= 300:  # Sanity: between 30s and 5min
                return val
    except (ValueError, TypeError):
        pass

    return None


def _distance_bucket(distance: int | float) -> str:
    """Classify distance into standard buckets."""
    d = int(distance)
    if d <= 1200:
        return "sprint"
    elif d <= 1400:
        return "short"
    elif d <= 1800:
        return "middle"
    else:
        return "staying"


def _condition_group(condition: str) -> str:
    """Classify track condition into groups."""
    if not condition:
        return "Good"
    c = condition.lower()
    if "heavy" in c:
        return "Heavy"
    elif "soft" in c:
        return "Soft"
    elif "firm" in c or "synthetic" in c:
        return "Firm"
    return "Good"


def _venue_key(venue: str) -> str:
    """Normalise venue name for lookup."""
    if not venue:
        return ""
    return re.sub(r"[^a-z]", "", venue.lower())


async def load_standard_times(db) -> dict:
    """Load pre-computed standard times from the deep learning PatternInsight.

    Returns a lookup dict keyed by "venuekey_distbucket_condgroup".
    """
    global _standard_times_cache
    if _standard_times_cache is not None:
        return _standard_times_cache

    try:
        from sqlalchemy import select
        from punty.memory.models import PatternInsight

        result = await db.execute(
            select(PatternInsight.conditions_json).where(
                PatternInsight.pattern_type == "deep_learning_standard_times"
            ).limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            data = json.loads(row) if isinstance(row, str) else row
            _standard_times_cache = data.get("lookup", {})
        else:
            _standard_times_cache = {}
    except Exception as e:
        logger.debug(f"Failed to load standard times: {e}")
        _standard_times_cache = {}

    return _standard_times_cache


def clear_cache():
    """Clear the standard times cache (for testing)."""
    global _standard_times_cache
    _standard_times_cache = None


def rate_form_times(
    form_history: list[dict],
    standard_times: dict,
    max_starts: int = 5,
) -> list[dict]:
    """Rate each start's race time against the venue/distance standard.

    Args:
        form_history: Past starts (most recent first).
        standard_times: Lookup dict from load_standard_times().
        max_starts: Number of starts to rate.

    Returns:
        List of rated starts (only those with valid time + matching standard).
    """
    if not form_history or not standard_times:
        return []

    results = []
    for i, start in enumerate(form_history[:max_starts]):
        time_str = start.get("time")
        time_secs = parse_race_time(time_str)
        if time_secs is None:
            continue

        venue = start.get("venue", "")
        distance = start.get("distance")
        condition = start.get("track", "")

        if not venue or not distance:
            continue

        try:
            dist = int(distance)
        except (ValueError, TypeError):
            continue

        # Build lookup key
        vk = _venue_key(venue)
        db = _distance_bucket(dist)
        cg = _condition_group(condition)
        key = f"{vk}_{db}_{cg}"

        standard = standard_times.get(key)
        if not standard:
            # Try without condition (fallback to Good)
            key_fallback = f"{vk}_{db}_Good"
            standard = standard_times.get(key_fallback)
        if not standard:
            continue

        std_time = standard.get("median") or standard.get("avg")
        if not std_time or std_time <= 0:
            continue

        diff_pct = (time_secs - std_time) / std_time * 100

        if diff_pct < -2:
            rating = "FAST"
        elif diff_pct > 2:
            rating = "SLOW"
        else:
            rating = "STANDARD"

        results.append({
            "run_index": i,
            "venue": venue,
            "distance": dist,
            "condition": condition,
            "time_secs": round(time_secs, 2),
            "standard_secs": round(std_time, 2),
            "diff_pct": round(diff_pct, 1),
            "rating": rating,
        })

    return results
