"""Cross-referenced combo form analysis.

Derives combo performance signals from form_history by matching multiple
dimensions against today's race conditions (venue + condition, distance +
condition, jockey partnership, class level, run spacing).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ComboStats:
    """Win/place record for a combo filter."""

    starts: int = 0
    wins: int = 0
    seconds: int = 0
    thirds: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.starts if self.starts else 0.0

    @property
    def place_rate(self) -> float:
        return (self.wins + self.seconds + self.thirds) / self.starts if self.starts else 0.0


@dataclass
class SpacingResult:
    """Performance at a similar rest interval."""

    bucket: str
    stats: ComboStats
    pattern_score: float  # 0.0-1.0


# ──────────────────────────────────────────────
# Bucketing helpers
# ──────────────────────────────────────────────

_VENUE_ALIASES: dict[str, str] = {
    "the valley": "moonee_valley",
    "moonee valley": "moonee_valley",
    "royal randwick": "randwick",
    "randwick": "randwick",
    "pinjarra park": "pinjarra",
}


def _normalise_venue(venue: str) -> str:
    """Normalise venue name for matching."""
    if not venue:
        return ""
    v = venue.lower().strip()
    return _VENUE_ALIASES.get(v, v.replace(" ", "_").replace("'", ""))


def _bucket_condition(track_str: str) -> str:
    """Bucket track condition string to good/soft/heavy."""
    if not track_str:
        return "good"
    tc = track_str.lower()
    if any(k in tc for k in ("heavy", "hvy")):
        return "heavy"
    if any(k in tc for k in ("soft", "sft")):
        return "soft"
    return "good"


_BM_RE = re.compile(r"(?:bm|benchmark)\s*(\d+)", re.IGNORECASE)


def _bucket_class(race_class: str) -> str:
    """Bucket race class for combo matching."""
    if not race_class:
        return "unknown"
    rc = race_class.lower().strip().rstrip(";")

    if "maiden" in rc or "mdn" in rc:
        return "maiden"
    if "class 1" in rc or "cl1" in rc:
        return "class1"
    if "class 2" in rc or "cl2" in rc:
        return "class2"
    if "class 3" in rc or "cl3" in rc:
        return "class3"
    if "restricted" in rc or "rst " in rc:
        return "restricted"

    bm = _BM_RE.search(rc)
    if bm:
        rating = int(bm.group(1))
        if rating <= 58:
            return "bm_low"
        if rating <= 68:
            return "bm_mid"
        if rating <= 78:
            return "bm_high"
        return "open"

    if any(kw in rc for kw in ("group", "listed", "stakes", "quality", "open")):
        return "open"
    if "class 4" in rc or "class 5" in rc or "class 6" in rc:
        return "bm_high"

    return "bm_mid"


def _distance_matches(hist_dist: int, today_dist: int) -> bool:
    """Check if a historical distance is close enough to today's."""
    if not hist_dist or not today_dist:
        return False
    diff = abs(hist_dist - today_dist)
    if today_dist <= 1100:
        return diff <= 100
    if today_dist <= 1399:
        return diff <= 100
    if today_dist <= 1799:
        return diff <= 200
    if today_dist <= 2199:
        return diff <= 200
    return diff <= 300


def _venue_matches(hist_venue: str, today_venue: str) -> bool:
    """Check if venues match after normalisation."""
    return _normalise_venue(hist_venue) == _normalise_venue(today_venue)


def _parse_position(pos) -> Optional[int]:
    """Parse finish position from various formats."""
    if pos is None:
        return None
    if isinstance(pos, int):
        return pos if pos > 0 else None
    if isinstance(pos, float):
        return int(pos) if pos > 0 else None
    s = str(pos).strip()
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _parse_date(date_str) -> Optional[datetime]:
    """Parse a date string from form history."""
    if not date_str:
        return None
    if isinstance(date_str, datetime):
        return date_str
    s = str(date_str).strip()
    if "T" in s:
        s = s.split("T")[0]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ──────────────────────────────────────────────
# Stats builder
# ──────────────────────────────────────────────

def _build_stats(starts: list[dict]) -> ComboStats:
    """Build a ComboStats from a list of matching form entries."""
    wins = seconds = thirds = 0
    for s in starts:
        pos = _parse_position(s.get("position"))
        if pos == 1:
            wins += 1
        elif pos == 2:
            seconds += 1
        elif pos == 3:
            thirds += 1
    return ComboStats(starts=len(starts), wins=wins, seconds=seconds, thirds=thirds)


# ──────────────────────────────────────────────
# Spacing analysis
# ──────────────────────────────────────────────

def _gap_to_bucket(days: int) -> str:
    """Bucket days between runs."""
    if days <= 14:
        return "quick_backup"
    if days <= 28:
        return "normal"
    if days <= 60:
        return "freshened"
    if days <= 120:
        return "spell"
    return "long_spell"


def _analyse_spacing(
    form_history: list[dict],
    days_since_last_run: Optional[int],
) -> Optional[SpacingResult]:
    """Analyse performance at similar rest intervals from form history.

    Looks at gaps between consecutive starts and how the horse performed
    after a similar layoff to today's.
    """
    if not form_history or len(form_history) < 2 or not days_since_last_run:
        return None

    today_bucket = _gap_to_bucket(days_since_last_run)

    # Calculate gaps between consecutive starts
    gap_results: list[dict] = []
    for i in range(len(form_history) - 1):
        curr = form_history[i]
        prev = form_history[i + 1]  # form_history is most-recent-first

        curr_date = _parse_date(curr.get("date"))
        prev_date = _parse_date(prev.get("date"))

        if not curr_date or not prev_date:
            continue

        gap_days = (curr_date - prev_date).days
        if gap_days <= 0:
            continue

        pos = _parse_position(curr.get("position"))
        if pos is None:
            continue

        gap_results.append({"gap": gap_days, "position": pos})

    if not gap_results:
        return None

    # Filter to similar spacing bucket
    similar = [g for g in gap_results if _gap_to_bucket(g["gap"]) == today_bucket]
    if len(similar) < 2:
        return None

    stats = ComboStats(
        starts=len(similar),
        wins=sum(1 for g in similar if g["position"] == 1),
        seconds=sum(1 for g in similar if g["position"] == 2),
        thirds=sum(1 for g in similar if g["position"] == 3),
    )

    # Score: normalise win_rate around a ~15% baseline
    raw = 0.5 + (stats.win_rate - 0.15) * 2.0
    pattern_score = max(0.05, min(0.95, raw))

    return SpacingResult(bucket=today_bucket, stats=stats, pattern_score=pattern_score)


# ──────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────

def analyse_combo_form(
    form_history: list[dict],
    today_venue: str,
    today_distance: int,
    today_condition: str,
    today_class: str,
    today_jockey: str,
    days_since_last_run: Optional[int] = None,
) -> dict:
    """Analyse form history for combo signals matching today's race.

    Returns dict with keys:
        track_cond      — venue + condition combo stats
        dist_cond       — distance + condition combo stats
        track_dist_cond — triple combo stats
        jockey_horse    — today's jockey on this horse
        class_perf      — class-level performance
        spacing         — SpacingResult or None
    """
    if not form_history:
        return {}

    today_venue_key = _normalise_venue(today_venue)
    today_cond_bucket = _bucket_condition(today_condition)
    today_class_bucket = _bucket_class(today_class)
    today_jockey_lower = (today_jockey or "").lower().strip()

    # Filter out trials
    valid_starts = [s for s in form_history if not s.get("is_trial")]

    # Single pass: classify each start into matching combos
    track_cond_starts: list[dict] = []
    dist_cond_starts: list[dict] = []
    track_dist_cond_starts: list[dict] = []
    jockey_starts: list[dict] = []
    class_starts: list[dict] = []

    for start in valid_starts:
        pos = _parse_position(start.get("position"))
        if pos is None:
            continue

        hist_venue_key = _normalise_venue(start.get("venue", ""))
        hist_cond = _bucket_condition(start.get("track", ""))
        hist_class = _bucket_class(start.get("class", ""))
        hist_dist = start.get("distance")
        hist_jockey = (start.get("jockey") or "").lower().strip()

        venue_match = hist_venue_key == today_venue_key and today_venue_key
        cond_match = hist_cond == today_cond_bucket
        dist_match = _distance_matches(hist_dist, today_distance) if hist_dist else False

        # Track + condition
        if venue_match and cond_match:
            track_cond_starts.append(start)

        # Distance + condition
        if dist_match and cond_match:
            dist_cond_starts.append(start)

        # Triple: track + distance + condition
        if venue_match and dist_match and cond_match:
            track_dist_cond_starts.append(start)

        # Jockey on this horse
        if hist_jockey and hist_jockey == today_jockey_lower:
            jockey_starts.append(start)

        # Class performance
        if hist_class == today_class_bucket and today_class_bucket != "unknown":
            class_starts.append(start)

    result: dict = {}

    if track_cond_starts:
        result["track_cond"] = _build_stats(track_cond_starts)
    if dist_cond_starts:
        result["dist_cond"] = _build_stats(dist_cond_starts)
    if track_dist_cond_starts:
        result["track_dist_cond"] = _build_stats(track_dist_cond_starts)
    if jockey_starts:
        result["jockey_horse"] = _build_stats(jockey_starts)
    if class_starts:
        result["class_perf"] = _build_stats(class_starts)

    # Spacing analysis
    spacing = _analyse_spacing(valid_starts, days_since_last_run)
    if spacing:
        result["spacing"] = spacing

    return result
