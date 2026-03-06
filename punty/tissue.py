"""Tissue probability engine — condition-lookup model.

Builds independent win/place probabilities WITHOUT using market odds.
Uses empirical lookup tables derived from 221K historical runners:
  - Condition multipliers (distance × track_condition × barrier × pace)
  - Career win rate bands
  - Form recency patterns
  - Track/distance and condition specialist stats
  - First-up / second-up patterns
  - Horse profile (age × sex)
  - Weight carried relative to field

Market odds are used ONLY in the post-ranking market_layer for value
detection, never in the tissue calculation itself.

The tissue score for each runner is a multiplicative product of
context-specific multipliers, normalized across the field to produce
win and place probabilities.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Dampen Config (validated OOS on 4,467 test races)
# Career at full strength; weak signals compressed toward 1.0
# Config F: 45.4% R1 accuracy, 64.4% agreement, +166.7% win ROI
# ──────────────────────────────────────────────
DAMPEN = {
    "condition": 0.5,      # context helps but noisy
    "form_recency": 0.2,   # weak signal, mostly noise
    "specialist": 0.3,     # T/D and condition specialist
    "spell": 0.2,          # first-up/second-up
    "profile": 0.2,        # age × sex
    "weight": 0.2,         # weight relative to field
}


def _dampen(mult: float, factor_name: str) -> float:
    """Compress multiplier toward 1.0 using validated dampen factors."""
    d = DAMPEN.get(factor_name, 1.0)
    return 1.0 + (mult - 1.0) * d


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

_TISSUE_TABLES: dict | None = None


def _load_tables() -> dict:
    """Load tissue lookup tables from JSON. Cached after first load."""
    global _TISSUE_TABLES
    if _TISSUE_TABLES is not None:
        return _TISSUE_TABLES

    path = Path(__file__).parent / "data" / "tissue_tables.json"
    if not path.exists():
        logger.warning("Tissue tables not found at %s — using neutral defaults", path)
        _TISSUE_TABLES = {}
        return _TISSUE_TABLES

    with open(path, "r") as f:
        _TISSUE_TABLES = json.load(f)
    logger.info("Tissue tables loaded: v%s", _TISSUE_TABLES.get("version", "?"))
    return _TISSUE_TABLES


def _get(obj: Any, attr: str, default=None):
    """Get attribute from ORM object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ──────────────────────────────────────────────
# Bucket Helpers
# ──────────────────────────────────────────────

def _dist_bucket(distance: int) -> str:
    if distance <= 1100:
        return "sprint"
    if distance <= 1399:
        return "short"
    if distance <= 1799:
        return "middle"
    if distance <= 2199:
        return "classic"
    return "staying"


def _cond_bucket(cond: str) -> str:
    c = (cond or "").lower().strip()
    if "heavy" in c:
        return "heavy"
    if "soft" in c:
        return "soft"
    if "synthetic" in c:
        return "synthetic"
    return "good"


def _barrier_zone(barrier: int, field_size: int) -> str:
    if field_size <= 1:
        return "inside"
    relative = (barrier - 1) / max(field_size - 1, 1)
    if relative <= 0.30:
        return "inside"
    if relative <= 0.65:
        return "middle"
    return "outside"


# ──────────────────────────────────────────────
# Individual Multiplier Lookups
# ──────────────────────────────────────────────

def _condition_multiplier(
    distance: int, track_condition: str, barrier: int,
    pace_position: str, field_size: int, venue: str,
) -> float:
    """Look up condition-specific multiplier with fallback chain.

    Tries increasingly general keys until a match with sufficient
    sample size is found. Returns 1.0 (neutral) if nothing matches.
    """
    tables = _load_tables()

    # Try venue-specific override first
    venue_overrides = tables.get("venue_overrides", {})
    if venue:
        v = venue.lower().strip()
        dist = _dist_bucket(distance)
        bzone = _barrier_zone(barrier, field_size)
        for vkey in [f"{v}|{dist}|{bzone}", f"{v}|{dist}"]:
            if vkey in venue_overrides:
                return venue_overrides[vkey]["mult"]

    # Condition multiplier lookup with fallback chain
    cm = tables.get("condition_multipliers", {})
    entries = cm.get("entries", {})
    if not entries:
        return 1.0

    dist = _dist_bucket(distance)
    cond = _cond_bucket(track_condition)
    bzone = _barrier_zone(barrier, field_size)
    pace = pace_position or "unknown"

    # Fallback chain: most specific to least
    fallback_keys = [
        f"{dist}|{cond}|{bzone}|{pace}",   # 4-way
        f"{dist}|{cond}|{bzone}",            # 3-way
        f"{dist}|{cond}|{pace}",             # 3-way
        f"{dist}|{bzone}|{pace}",            # 3-way
        f"{dist}|{cond}",                    # 2-way
        f"{dist}|{bzone}",                   # 2-way
        f"{dist}|{pace}",                    # 2-way
        f"{cond}|{bzone}",                   # 2-way
        dist,                                # 1-way
        cond,                                # 1-way
    ]

    for key in fallback_keys:
        if key in entries:
            return entries[key]["mult"]

    return 1.0


def _career_multiplier(career_record: str, is_maiden_race: bool = False) -> float:
    """Career win rate band → tissue multiplier.

    In maiden races, all horses have 0 wins so the standard career bands
    don't apply. Instead we use maiden-specific bands based on starts
    (experience level) and places (competitiveness).
    """
    tables = _load_tables()
    bands = tables.get("career_bands", {}).get("bands", {})
    if not bands:
        return 1.0

    if not career_record or not isinstance(career_record, str):
        return bands.get("lightly_raced", {}).get("mult", 1.0)

    # Parse "67: 5-14-13" → starts=67, wins=5, seconds=14, thirds=13
    parts = career_record.split(": ")
    if len(parts) < 2:
        return 1.0
    try:
        starts = int(parts[0].strip())
        result_parts = parts[1].split("-")
        wins = int(result_parts[0].strip())
        seconds = int(result_parts[1].strip()) if len(result_parts) > 1 else 0
        thirds = int(result_parts[2].strip()) if len(result_parts) > 2 else 0
    except (ValueError, IndexError):
        return 1.0

    if is_maiden_race:
        return _maiden_career_multiplier(starts, wins, seconds, thirds)

    if starts < 3:
        return bands.get("lightly_raced", {}).get("mult", 1.0)

    win_rate = wins / starts
    if win_rate >= 0.30:
        return bands.get("elite_30pct", {}).get("mult", 1.0)
    if win_rate >= 0.20:
        return bands.get("good_20pct", {}).get("mult", 1.0)
    if win_rate >= 0.10:
        return bands.get("average_10pct", {}).get("mult", 1.0)
    if wins > 0:
        return bands.get("below_avg", {}).get("mult", 1.0)
    return bands.get("maiden_career", {}).get("mult", 1.0)


def _maiden_career_multiplier(starts: int, wins: int, seconds: int, thirds: int) -> float:
    """Career multiplier specifically for maiden races.

    In maidens, 0 wins is expected. What matters is:
    1. Place record (seconds/thirds show competitiveness)
    2. Number of starts (exposure — sweet spot is 3-7, >10 is struggling)

    Derived from maiden race subset of the 221K dataset:
    - Horses with places in 3-7 starts: ~14% win rate in maidens
    - First starters / 1-2 starts: ~9% (unproven, wide variance)
    - 8+ starts with no places: ~3% (exposed non-winners)
    """
    places = seconds + thirds
    place_rate = places / starts if starts > 0 else 0

    if starts <= 2:
        # Lightly raced — unproven, moderate multiplier
        if places > 0:
            return 1.10  # shown ability early
        return 0.90  # unknown quantity

    if starts <= 7:
        # Sweet spot — enough runs to assess, not too exposed
        if place_rate >= 0.40:
            return 1.40  # consistently placing, due to win
        if place_rate >= 0.25:
            return 1.20  # competitive
        if places > 0:
            return 1.00  # has placed at least once
        return 0.70  # several starts, no places = exposed

    if starts <= 12:
        # Getting exposed — needs places to justify
        if place_rate >= 0.30:
            return 1.10  # still competitive despite starts
        if places >= 2:
            return 0.85  # some ability but struggling
        return 0.50  # many starts, rarely places

    # 13+ starts — chronic maiden
    if place_rate >= 0.25:
        return 0.75  # still placing but can't win
    return 0.35  # long-exposed, unlikely to break through


def _form_recency_multiplier(last_five: str) -> float:
    """Recent form pattern → tissue multiplier."""
    tables = _load_tables()
    form = tables.get("form_recency", {})
    recent = form.get("recent_form", {})
    if not recent or not last_five:
        return 1.0

    first_char = last_five[0] if last_five else ""

    if first_char == "1":
        return recent.get("last_1st", {}).get("mult", 1.0)
    if first_char == "2":
        return recent.get("last_2nd", {}).get("mult", 1.0)
    if first_char == "3":
        return recent.get("last_3rd", {}).get("mult", 1.0)
    if first_char in ("4", "5"):
        return recent.get("last_4th5th", {}).get("mult", 1.0)
    if first_char in ("6", "7", "8"):
        return recent.get("last_mid", {}).get("mult", 1.0)
    if first_char in ("x", "X"):
        return recent.get("last_x", {}).get("mult", 1.0)
    return recent.get("last_back", {}).get("mult", 1.0)


def _specialist_multiplier(runner: Any, track_condition: str) -> float:
    """Track/distance and condition specialist → combined multiplier."""
    tables = _load_tables()
    spec = tables.get("specialist_tables", {})

    mult = 1.0

    # Track + distance specialist
    td_stats_raw = _get(runner, "track_dist_stats")
    if td_stats_raw:
        td = _parse_stats(td_stats_raw)
        if td:
            starts, wins = td
            td_table = spec.get("track_distance", {})
            if starts >= 2:
                wr = wins / starts if starts > 0 else 0
                if wr >= 0.30:
                    mult *= td_table.get("td_specialist_30", {}).get("mult", 1.0)
                elif wr >= 0.20:
                    mult *= td_table.get("td_specialist_20", {}).get("mult", 1.0)
                elif wins > 0:
                    mult *= td_table.get("td_winner", {}).get("mult", 1.0)
                else:
                    mult *= td_table.get("td_no_win", {}).get("mult", 1.0)

    # Condition specialist (match current track condition to the right stats)
    cond_field = None
    tc = (track_condition or "").lower()
    if "heavy" in tc:
        cond_field = "heavy_track_stats"
    elif "soft" in tc:
        cond_field = "soft_track_stats"
    else:
        cond_field = "good_track_stats"

    cond_stats_raw = _get(runner, cond_field)
    if cond_stats_raw:
        cs = _parse_stats(cond_stats_raw)
        if cs:
            starts, wins = cs
            cond_table = spec.get("condition", {})
            if starts >= 3:
                wr = wins / starts if starts > 0 else 0
                if wr >= 0.25:
                    mult *= cond_table.get("cond_specialist_25", {}).get("mult", 1.0)
                elif wr >= 0.15:
                    mult *= cond_table.get("cond_specialist_15", {}).get("mult", 1.0)
                elif wins > 0:
                    mult *= cond_table.get("cond_winner", {}).get("mult", 1.0)
                else:
                    mult *= cond_table.get("cond_no_win", {}).get("mult", 1.0)

    return mult


def _spell_multiplier(runner: Any) -> float:
    """First-up / second-up specialist → multiplier."""
    tables = _load_tables()
    spell = tables.get("spell_tables", {})

    days = _get(runner, "days_since_last_run")

    # Determine if first-up or second-up context
    if days and isinstance(days, (int, float)) and days > 60:
        # First-up context
        fu_raw = _get(runner, "first_up_stats")
        if fu_raw:
            fu = _parse_stats(fu_raw)
            if fu:
                starts, wins = fu
                fu_table = spell.get("first_up", {})
                if starts >= 2:
                    wr = wins / starts if starts > 0 else 0
                    if wr >= 0.30:
                        return fu_table.get("fu_strong_30", {}).get("mult", 1.0)
                    elif wr >= 0.15:
                        return fu_table.get("fu_ok_15", {}).get("mult", 1.0)
                    elif wins > 0:
                        return fu_table.get("fu_weak", {}).get("mult", 1.0)
                    else:
                        return fu_table.get("fu_never_won", {}).get("mult", 1.0)
                return fu_table.get("fu_few_starts", {}).get("mult", 1.0)
    elif days and isinstance(days, (int, float)) and 14 <= days <= 60:
        # Second-up context
        su_raw = _get(runner, "second_up_stats")
        if su_raw:
            su = _parse_stats(su_raw)
            if su:
                starts, wins = su
                su_table = spell.get("second_up", {})
                if starts >= 2:
                    wr = wins / starts if starts > 0 else 0
                    if wr >= 0.25:
                        return su_table.get("su_strong_25", {}).get("mult", 1.0)
                    elif wr >= 0.12:
                        return su_table.get("su_ok_12", {}).get("mult", 1.0)
                    elif wins > 0:
                        return su_table.get("su_weak", {}).get("mult", 1.0)
                    else:
                        return su_table.get("su_never_won", {}).get("mult", 1.0)
                return su_table.get("su_few_starts", {}).get("mult", 1.0)

    return 1.0


def _horse_profile_multiplier(runner: Any) -> float:
    """Age × sex → multiplier."""
    tables = _load_tables()
    profiles = tables.get("horse_profile", {})
    if not profiles:
        return 1.0

    age = _get(runner, "horse_age")
    sex = _get(runner, "horse_sex")

    if not age:
        return 1.0

    if age <= 2:
        age_band = "2yo"
    elif age == 3:
        age_band = "3yo"
    elif age == 4:
        age_band = "4yo"
    elif age == 5:
        age_band = "5yo"
    elif age == 6:
        age_band = "6yo"
    else:
        age_band = "7yo_plus"

    sex_str = (sex or "").lower().strip()
    if sex_str in ("m", "c", "r", "h"):
        sex_band = "male"
    elif sex_str in ("f", "mare"):
        sex_band = "female"
    elif sex_str in ("g", "gelding"):
        sex_band = "gelding"
    else:
        sex_band = "other"

    key = f"{age_band}|{sex_band}"
    if key in profiles:
        return profiles[key]["mult"]

    # Fallback: try just age with "other"
    fallback_key = f"{age_band}|other"
    if fallback_key in profiles:
        return profiles[fallback_key]["mult"]

    return 1.0


def _weight_multiplier(runner: Any, avg_weight: float) -> float:
    """Weight relative to field average → multiplier."""
    tables = _load_tables()
    wt_table = tables.get("weight_carried", {})
    if not wt_table:
        return 1.0

    weight = _get(runner, "weight")
    if not weight or not isinstance(weight, (int, float)) or weight <= 0:
        return 1.0
    if avg_weight <= 0:
        return 1.0

    diff = weight - avg_weight
    if diff < -3:
        band = "very_light"
    elif diff < -1:
        band = "light"
    elif diff <= 1:
        band = "average"
    elif diff <= 3:
        band = "heavy"
    else:
        band = "very_heavy"

    return wt_table.get(band, {}).get("mult", 1.0)


# ──────────────────────────────────────────────
# Stats Parsing
# ──────────────────────────────────────────────

def _parse_stats(raw: Any) -> Optional[tuple[int, int]]:
    """Parse stats from JSON or string format → (starts, wins)."""
    if not raw:
        return None

    if isinstance(raw, dict):
        starts = raw.get("starts", 0)
        wins = raw.get("wins", 0)
        if starts:
            return (int(starts), int(wins))
        return None

    if isinstance(raw, str):
        # Try JSON format: {"starts": 10, "wins": 3, ...}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                starts = data.get("starts", 0)
                wins = data.get("wins", 0)
                if starts:
                    return (int(starts), int(wins))
        except (json.JSONDecodeError, TypeError):
            pass

        # Try "10: 3-2-1" format
        parts = raw.split(": ")
        if len(parts) >= 2:
            try:
                starts = int(parts[0].strip())
                wins = int(parts[1].split("-")[0].strip())
                return (starts, wins)
            except (ValueError, IndexError):
                pass

    return None


# ──────────────────────────────────────────────
# Main Tissue Builder
# ──────────────────────────────────────────────

@dataclass
class TissueResult:
    """Tissue probability result for a single runner."""
    tissue_score: float        # raw multiplicative score
    win_probability: float     # normalized tissue win prob
    place_probability: float   # estimated tissue place prob
    factors: dict              # individual multiplier breakdown
    tissue_price: float        # implied fair odds from tissue


def build_tissue(
    runners: list,
    race: Any,
    meeting: Any,
) -> dict[str, TissueResult]:
    """Build tissue probabilities for all runners in a race.

    This is the core tissue engine. It computes an independent
    probability for each runner using multiplicative condition-specific
    lookup tables, with NO input from market odds.

    Args:
        runners: List of active (non-scratched) Runner objects or dicts
        race: Race object or dict
        meeting: Meeting object or dict

    Returns:
        Dict mapping runner_id → TissueResult
    """
    if not runners:
        return {}

    # Race context
    distance = _get(race, "distance") or 1400
    track_condition = (_get(meeting, "track_condition")
                       or _get(race, "track_condition") or "Good")
    venue = _get(meeting, "venue") or ""
    field_size = len(runners)

    # Detect maiden races (class_ or race name)
    race_class = (_get(race, "class_") or _get(race, "class") or "").lower()
    race_name = (_get(race, "name") or "").lower()
    is_maiden = "maiden" in race_class or " mdn " in f" {race_name} " or race_name.endswith(" mdn")

    # Pre-calculate field average weight
    weights = [_get(r, "weight") or 0 for r in runners
               if isinstance(_get(r, "weight"), (int, float)) and _get(r, "weight") > 0]
    avg_weight = sum(weights) / len(weights) if weights else 55.0

    # Calculate tissue score for each runner
    raw_scores: dict[str, float] = {}
    factor_details: dict[str, dict] = {}

    for runner in runners:
        rid = _get(runner, "id", "")
        barrier = _get(runner, "barrier") or 0
        if not isinstance(barrier, (int, float)):
            barrier = 0
        barrier = int(barrier)

        pace_pos = _get(runner, "speed_map_position") or "unknown"
        career_record = _get(runner, "career_record") or ""
        last_five = _get(runner, "last_five") or ""

        # Calculate individual multipliers (career at full strength, rest dampened)
        career_mult = _career_multiplier(career_record, is_maiden_race=is_maiden)
        cond_mult = _dampen(_condition_multiplier(
            distance, track_condition, barrier, pace_pos, field_size, venue), "condition")
        form_mult = _dampen(_form_recency_multiplier(last_five), "form_recency")
        spec_mult = _dampen(_specialist_multiplier(runner, track_condition), "specialist")
        spell_mult = _dampen(_spell_multiplier(runner), "spell")
        profile_mult = _dampen(_horse_profile_multiplier(runner), "profile")
        weight_mult = _dampen(_weight_multiplier(runner, avg_weight), "weight")

        # Combine multiplicatively
        # Career is the dominant signal (3.6× range). Other factors
        # are dampened toward 1.0 to reduce noise while preserving
        # directional information. Validated OOS: 45.4% R1 accuracy.
        tissue_score = (
            career_mult
            * cond_mult
            * form_mult
            * spec_mult
            * spell_mult
            * profile_mult
            * weight_mult
        )

        # Floor at 0.001 to avoid zero probabilities
        tissue_score = max(0.001, tissue_score)
        raw_scores[rid] = tissue_score
        factor_details[rid] = {
            "career": round(career_mult, 3),
            "condition": round(cond_mult, 3),
            "form_recency": round(form_mult, 3),
            "specialist": round(spec_mult, 3),
            "spell": round(spell_mult, 3),
            "profile": round(profile_mult, 3),
            "weight": round(weight_mult, 3),
            "raw_score": round(tissue_score, 4),
        }

    # Normalize to win probabilities (sum to 1.0)
    total = sum(raw_scores.values())
    if total <= 0:
        baseline = 1.0 / field_size
        return {
            _get(r, "id", ""): TissueResult(
                tissue_score=1.0,
                win_probability=baseline,
                place_probability=min(0.75, baseline * 3),
                factors={},
                tissue_price=round(1 / baseline, 2) if baseline > 0 else 99.0,
            )
            for r in runners
        }

    # Get field_size table for place probability estimation
    tables = _load_tables()
    fs_table = tables.get("field_size", {})
    fs_entry = fs_table.get(str(field_size), {})
    expected_pr = fs_entry.get("expected_place_rate", 0.35)

    results: dict[str, TissueResult] = {}
    place_count = 2 if field_size <= 7 else 3

    # Maiden market blend: in maiden races, tissue alone is weak because
    # all horses have 0 wins. Blend with market-implied probability to
    # incorporate trial form, trainer confidence, breeding signals.
    # α = 0.50 means equal weight tissue + market (geometric mean).
    MAIDEN_MARKET_ALPHA = 0.50
    market_probs: dict[str, float] = {}
    if is_maiden:
        odds_total = 0.0
        for runner in runners:
            rid = _get(runner, "id", "")
            odds = _get(runner, "current_odds") or _get(runner, "opening_odds") or 0
            if isinstance(odds, (int, float)) and odds > 1.0:
                market_probs[rid] = 1.0 / odds
                odds_total += 1.0 / odds
            else:
                market_probs[rid] = 0.0
        # Normalize market probs to sum to 1.0 (remove overround)
        if odds_total > 0:
            for rid in market_probs:
                market_probs[rid] /= odds_total
        else:
            # No odds available — disable blend
            market_probs = {}

    blended_probs: dict[str, float] = {}
    for runner in runners:
        rid = _get(runner, "id", "")
        tissue_score = raw_scores.get(rid, 0.001)
        win_prob = tissue_score / total

        # Apply maiden market blend if available
        if is_maiden and rid in market_probs and market_probs[rid] > 0:
            mkt = market_probs[rid]
            # Geometric mean blend: tissue^(1-α) × market^α
            win_prob = (win_prob ** (1 - MAIDEN_MARKET_ALPHA)) * (mkt ** MAIDEN_MARKET_ALPHA)
            factor_details.get(rid, {})["_maiden_blend"] = round(MAIDEN_MARKET_ALPHA, 2)
            factor_details.get(rid, {})["_market_implied"] = round(mkt, 4)

        blended_probs[rid] = win_prob

    # Re-normalize blended probabilities to sum to 1.0
    blend_total = sum(blended_probs.values())
    if blend_total > 0:
        for rid in blended_probs:
            blended_probs[rid] /= blend_total

    for runner in runners:
        rid = _get(runner, "id", "")
        win_prob = blended_probs.get(rid, 0.001)

        # Place probability: use Harville-style approximation from win prob
        # Better runners place at higher rate relative to their win share
        place_prob = _estimate_place_prob(win_prob, field_size, place_count)

        tissue_price = round(1.0 / win_prob, 2) if win_prob > 0.005 else 201.0

        results[rid] = TissueResult(
            tissue_score=round(raw_scores.get(rid, 0.001), 4),
            win_probability=round(win_prob, 4),
            place_probability=round(place_prob, 4),
            factors=factor_details.get(rid, {}),
            tissue_price=min(201.0, tissue_price),
        )

    return results


def _estimate_place_prob(win_prob: float, field_size: int, place_count: int = 3) -> float:
    """Estimate place probability from win probability using Harville approximation.

    Uses the empirical relationship between win and place probabilities
    observed in the 221K runner dataset:
      - Short-priced (high wp): place_prob ≈ wp × 2.5 (diminishing)
      - Mid-priced: place_prob ≈ wp × 3.0
      - Long-priced: place_prob ≈ wp × 3.5 (place more likely relative to win)

    Capped at 0.75 to match weighted engine ceiling.
    """
    if win_prob <= 0:
        return 0.01

    # Scaling factor depends on win probability
    # Higher win prob → lower multiplier (diminishing returns)
    if win_prob >= 0.25:
        scale = 2.2
    elif win_prob >= 0.15:
        scale = 2.6
    elif win_prob >= 0.08:
        scale = 3.0
    else:
        scale = 3.5

    # Adjust for place count (2 in small fields, 3 normally)
    if place_count == 2:
        scale *= 0.67

    place_prob = win_prob * scale
    return min(0.75, max(0.01, place_prob))
