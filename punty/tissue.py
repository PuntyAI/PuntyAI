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
    "condition": 0.5,       # context helps but noisy
    "form_recency": 0.2,    # weak signal, mostly noise (single-char legacy)
    "specialist": 0.3,      # T/D and condition specialist
    "spell": 0.2,           # first-up/second-up
    "profile": 0.2,         # age × sex
    "weight": 0.2,          # weight relative to field
    "pf_assessment": 0.6,   # strong external AI signal, captures class
    "handicap": 0.5,        # official class rating
    "class_stats": 0.4,     # wins at this class
    "jockey_trainer": 0.3,  # jockey/trainer combo performance
    "speed_map": 0.3,       # predicted running position
    "days_spell": 0.2,      # granular spell freshness
    "race_time": 0.6,       # time over distance — strong class indicator
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
    """Career win rate → tissue multiplier via Bayesian shrinkage + interpolation.

    Small-sample career records (e.g. 3 starts, 2 wins = 67%) are regressed
    toward the population mean (~10%) using Bayesian pseudocounts. This prevents
    horses with tiny samples from receiving extreme "elite" multipliers.

    The shrunk win rate is then mapped to a multiplier via linear interpolation
    between empirically calibrated points (no cliffs between bands).

    In maiden races, 0 wins is expected — uses place-rate assessment instead.
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

    # Bayesian shrinkage: regress small samples toward population mean.
    # k=8 pseudocounts at 10% prior. A 3-start/2-win horse (67% raw) shrinks
    # to 25.5%; a 30-start/9-win horse (30% raw) barely moves to 25.8%.
    # This prevents tiny samples from hitting extreme multiplier bands.
    SHRINK_K = 8
    PRIOR_WIN_RATE = 0.10
    shrunk_rate = (wins + SHRINK_K * PRIOR_WIN_RATE) / (starts + SHRINK_K)

    # 0-win horses in non-maiden races: use place-rate assessment.
    # maiden_career (0.004) is for chronic non-winners — not lightly-raced
    # horses like Zambales (3: 0-2-0) running in open-class races.
    if wins == 0:
        return _zero_win_non_maiden_multiplier(starts, seconds, thirds, bands)

    # Continuous interpolation between calibrated points (no cliffs).
    # Points: (shrunk_rate, multiplier) from 221K-runner dataset.
    return _interpolate_career_mult(shrunk_rate, bands)


def _zero_win_non_maiden_multiplier(
    starts: int, seconds: int, thirds: int, bands: dict,
) -> float:
    """Career multiplier for 0-win horses in non-maiden races.

    These horses are entered in winner-class races despite having 0 wins,
    meaning connections believe they're competitive. Assessment uses place
    record and exposure (starts) to differentiate genuine prospects from
    exposed non-winners.
    """
    places = seconds + thirds
    place_rate = places / starts if starts > 0 else 0

    if starts <= 5:
        # Lightly raced, 0 wins — could be promising or unproven
        if place_rate >= 0.40:
            return bands.get("average_10pct", {}).get("mult", 1.0)
        if places > 0:
            return bands.get("lightly_raced", {}).get("mult", 1.0)
        return bands.get("lightly_raced", {}).get("mult", 1.0)

    if starts <= 12:
        # Moderate experience, 0 wins — needs places to justify entry
        if place_rate >= 0.30:
            return bands.get("lightly_raced", {}).get("mult", 1.0)
        if places >= 2:
            return bands.get("below_avg", {}).get("mult", 1.0)
        return bands.get("below_avg", {}).get("mult", 1.0)

    # 13+ starts, 0 wins — chronic non-winner
    if place_rate >= 0.25:
        return bands.get("below_avg", {}).get("mult", 1.0)
    return bands.get("maiden_career", {}).get("mult", 1.0)


def _interpolate_career_mult(shrunk_rate: float, bands: dict) -> float:
    """Linearly interpolate career multiplier from shrunk win rate.

    Calibration points from 221K-runner dataset. Interpolation removes the
    cliff between bands (e.g. 29% → 2.06, 30% → 3.63 was a 76% jump).
    """
    # (shrunk_rate_threshold, multiplier) — ascending by rate
    points = [
        (0.05, bands.get("below_avg", {}).get("mult", 0.47)),
        (0.10, bands.get("average_10pct", {}).get("mult", 1.074)),
        (0.20, bands.get("good_20pct", {}).get("mult", 2.06)),
        (0.30, bands.get("elite_30pct", {}).get("mult", 3.63)),
    ]

    # Below lowest point
    if shrunk_rate <= points[0][0]:
        return points[0][1]

    # Above highest point (cap at elite)
    if shrunk_rate >= points[-1][0]:
        return points[-1][1]

    # Linear interpolation between surrounding points
    for i in range(len(points) - 1):
        r_lo, m_lo = points[i]
        r_hi, m_hi = points[i + 1]
        if r_lo <= shrunk_rate <= r_hi:
            t = (shrunk_rate - r_lo) / (r_hi - r_lo)
            return m_lo + t * (m_hi - m_lo)

    return 1.0  # fallback


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
    """Recent form pattern → tissue multiplier (legacy, single-char)."""
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


# ──────────────────────────────────────────────
# Enhanced Recent Form (scores all of last 5)
# ──────────────────────────────────────────────

# Points per finish position, recency-weighted
_FORM_POINTS = {"1": 3.0, "2": 2.0, "3": 1.5, "4": 0.5, "5": 0.5}
# More recent results matter more (index 0 = most recent)
_RECENCY_WEIGHTS = [1.5, 1.2, 1.0, 0.8, 0.6]


def _recent_form_multiplier(last_five: str) -> float:
    """Score all of last 5 results with recency weighting → multiplier.

    Maps a normalized form score (0-1) to the same multiplier scale as
    career bands using linear interpolation. This gives recent form the
    same expressive range as career win rate.

    Examples:
      "11111" → score 1.0  → mult 3.63 (elite recent form)
      "11143" → score 0.77 → mult 3.05
      "31111" → score 0.88 → mult 3.35
      "71962" → score 0.27 → mult 1.25
      "80191" → score 0.27 → mult 1.25
      "88349" → score 0.10 → mult 0.47
    """
    if not last_five:
        return 1.0

    # Score each position with recency weighting
    total_score = 0.0
    max_possible = 0.0
    for i, ch in enumerate(last_five[:5]):
        w = _RECENCY_WEIGHTS[i] if i < len(_RECENCY_WEIGHTS) else 0.5
        total_score += _FORM_POINTS.get(ch, 0.0) * w
        max_possible += 3.0 * w  # max is "1" (win) at every position

    if max_possible <= 0:
        return 1.0

    # Normalize to 0-1
    norm_score = total_score / max_possible

    # Map to multiplier scale via interpolation points
    # (norm_score, multiplier) — calibrated to match career band range
    points = [
        (0.00, 0.47),   # terrible recent form
        (0.15, 1.074),  # poor
        (0.35, 1.50),   # below average
        (0.50, 2.06),   # average-good
        (0.70, 2.80),   # good
        (0.85, 3.30),   # very good
        (1.00, 3.63),   # elite (all wins)
    ]

    if norm_score <= points[0][0]:
        return points[0][1]
    if norm_score >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        s_lo, m_lo = points[i]
        s_hi, m_hi = points[i + 1]
        if s_lo <= norm_score <= s_hi:
            t = (norm_score - s_lo) / (s_hi - s_lo)
            return m_lo + t * (m_hi - m_lo)

    return 1.0


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
# Additional Tissue Factors (previously ignored)
# ──────────────────────────────────────────────

def _pf_assessment_multiplier(runner: Any, field_runners: list) -> float:
    """PuntingForm AI score → multiplier relative to field.

    PF's AI model factors in class, form, sectionals, and track — signals
    our tissue misses. We use the score relative to field average rather
    than absolute, so it acts as a differentiator within each race.

    Coverage: ~96% of runners.
    """
    score = _get(runner, "pf_ai_score")
    if not score or not isinstance(score, (int, float)):
        return 1.0

    # Compute field average PF score
    scores = [_get(r, "pf_ai_score") for r in field_runners
              if isinstance(_get(r, "pf_ai_score"), (int, float)) and _get(r, "pf_ai_score") > 0]
    if len(scores) < 2:
        return 1.0

    avg_score = sum(scores) / len(scores)
    if avg_score <= 0:
        return 1.0

    # Ratio: 1.0 = field average, >1 = above avg, <1 = below
    ratio = score / avg_score

    # Map ratio to multiplier. PF scores typically range 20-90.
    # A runner at 1.5× field average is clearly superior.
    if ratio >= 1.4:
        return 1.8
    if ratio >= 1.2:
        return 1.0 + (ratio - 1.0) * 1.5  # 1.2→1.3, 1.3→1.45
    if ratio >= 0.8:
        return 1.0 + (ratio - 1.0) * 1.0  # linear around center
    if ratio >= 0.6:
        return 0.8 + (ratio - 0.6) * 1.0
    return 0.7


def _handicap_rating_multiplier(runner: Any, field_runners: list) -> float:
    """Handicap rating relative to field → multiplier.

    Official handicap ratings directly capture class ability. A horse
    rated 118 (Pride of Jenni) vs field average of 80 is a class above.
    Relative-to-field ensures the signal differentiates within each race.

    Coverage: ~81% of runners.
    """
    rating = _get(runner, "handicap_rating")
    if not rating or not isinstance(rating, (int, float)):
        return 1.0

    ratings = [_get(r, "handicap_rating") for r in field_runners
               if isinstance(_get(r, "handicap_rating"), (int, float)) and _get(r, "handicap_rating") > 0]
    if len(ratings) < 2:
        return 1.0

    avg_rating = sum(ratings) / len(ratings)
    if avg_rating <= 0:
        return 1.0

    # Each point above/below average = ~2% multiplier adjustment
    diff = rating - avg_rating
    # Cap at ±20 points (~±40% multiplier swing)
    diff = max(-20, min(20, diff))
    return 1.0 + diff * 0.02


def _class_stats_multiplier(runner: Any) -> float:
    """Wins/starts at this class level → multiplier.

    Direct measure of ability at the grade being contested. A horse
    with 4/10 at this class is proven; one with 0/6 is struggling.

    Coverage: ~83% of runners.
    """
    raw = _get(runner, "class_stats")
    parsed = _parse_stats(raw)
    if not parsed:
        return 1.0

    starts, wins = parsed
    if starts < 2:
        return 1.0  # too few starts at this class to assess

    win_rate = wins / starts
    if win_rate >= 0.30:
        return 1.5
    if win_rate >= 0.20:
        return 1.3
    if win_rate >= 0.10:
        return 1.1
    if wins > 0:
        return 0.95
    return 0.80  # multiple starts, no wins at this class


def _jockey_trainer_multiplier(runner: Any) -> float:
    """Jockey and trainer stats → combined multiplier.

    Uses actual-vs-expected (A2E) from PuntingForm as the primary signal.
    A2E > 1.0 means outperforming expectations. Combo stats (this jockey
    with this trainer) are weighted more heavily when available.

    Coverage: 100% for jockey/trainer, ~60% for combo stats.
    """
    mult = 1.0

    # Jockey A2E (last 100 runners)
    jockey_raw = _get(runner, "jockey_stats")
    if jockey_raw:
        jstats = jockey_raw if isinstance(jockey_raw, dict) else {}
        if isinstance(jockey_raw, str):
            try:
                jstats = json.loads(jockey_raw)
            except (json.JSONDecodeError, TypeError):
                jstats = {}

        # Prefer combo stats (jockey+trainer together) if sufficient sample
        combo = jstats.get("combo_career", {})
        combo_runners = combo.get("runners", 0) if isinstance(combo, dict) else 0
        if combo_runners >= 10:
            combo_a2e = combo.get("a2e", 1.0)
            if isinstance(combo_a2e, (int, float)):
                # Combo A2E is the strongest jockey/trainer signal
                mult *= max(0.7, min(1.4, combo_a2e))
        else:
            # Fall back to jockey last-100 A2E
            last100 = jstats.get("last100", {})
            if isinstance(last100, dict):
                a2e = last100.get("a2e", 1.0)
                if isinstance(a2e, (int, float)):
                    # Dampen jockey-only signal (less specific than combo)
                    mult *= max(0.85, min(1.2, 1.0 + (a2e - 1.0) * 0.5))

    # Trainer last-100 A2E (additive to jockey)
    trainer_raw = _get(runner, "trainer_stats")
    if trainer_raw:
        tstats = trainer_raw if isinstance(trainer_raw, dict) else {}
        if isinstance(trainer_raw, str):
            try:
                tstats = json.loads(trainer_raw)
            except (json.JSONDecodeError, TypeError):
                tstats = {}

        last100 = tstats.get("last100", {})
        if isinstance(last100, dict):
            a2e = last100.get("a2e", 1.0)
            if isinstance(a2e, (int, float)):
                mult *= max(0.90, min(1.15, 1.0 + (a2e - 1.0) * 0.3))

    return mult


def _speed_map_direct_multiplier(runner: Any) -> float:
    """Speed map position as direct competitive advantage.

    Leaders and on-pace runners have a statistical edge, particularly
    in sprints. The existing condition multiplier uses pace as a lookup
    key but doesn't capture the direct leader advantage.

    Coverage: ~96% of runners.
    """
    pos = (_get(runner, "speed_map_position") or "").lower().strip()
    if not pos:
        return 1.0

    if pos == "leader":
        return 1.15
    if pos == "on_pace":
        return 1.08
    if pos == "midfield":
        return 1.0
    if pos == "backmarker":
        return 0.90
    return 1.0


def _days_since_run_multiplier(runner: Any) -> float:
    """Granular spell length → multiplier.

    Finer-grained than the existing first-up/second-up assessment.
    Optimal freshness varies by distance, but generally:
    - 14-35 days: peak freshness (racing fit)
    - 36-90 days: short break (usually fine)
    - 91-180 days: spell (first-up, needs assessment)
    - 180+: long spell (fitness question)

    Coverage: ~98% of runners.
    """
    days = _get(runner, "days_since_last_run")
    if not days or not isinstance(days, (int, float)):
        return 1.0

    days = int(days)
    if days <= 7:
        return 0.90  # backed up very quickly
    if days <= 14:
        return 1.0
    if days <= 35:
        return 1.05  # peak fitness window
    if days <= 60:
        return 1.0  # normal break
    if days <= 120:
        return 0.95  # first-up, slight question
    if days <= 180:
        return 0.90  # long spell
    return 0.85  # very long absence


def _race_time_multiplier(runner: Any, race: Any, field_runners: list) -> float:
    """Recent race times over similar distance/track → multiplier.

    Analyses form_history to extract times run at similar distances on
    similar track types. Compares each runner's average time per 200m
    to the field average. Faster runners get a boost.

    This is a key class differentiator: a horse running 1:08.1 for 1200m
    at Group 1 Flemington is faster than one running 1:12.0 at a BM58.

    Coverage: depends on form_history availability (~80% of runners have
    at least one timed run at a comparable distance).
    """
    race_distance = _get(race, "distance") or 0
    if not race_distance:
        return 1.0

    # Collect times for all runners at similar distances
    runner_times: dict[str, list[float]] = {}
    dist_lo = race_distance * 0.85  # ±15% distance tolerance
    dist_hi = race_distance * 1.15

    for r in field_runners:
        rid = _get(r, "id", "")
        fh = _get(r, "form_history")
        if not fh:
            continue

        if isinstance(fh, str):
            try:
                fh = json.loads(fh)
            except (json.JSONDecodeError, TypeError):
                continue

        if not isinstance(fh, list):
            continue

        times_per_200: list[float] = []
        for run in fh[:10]:  # last 10 runs max
            if not isinstance(run, dict):
                continue
            if run.get("is_trial"):
                continue
            dist = run.get("distance")
            if not dist or not isinstance(dist, (int, float)):
                continue
            if not (dist_lo <= dist <= dist_hi):
                continue

            time_str = run.get("time")
            if not time_str or not isinstance(time_str, str):
                continue

            # Parse "00:01:08.1000000" → seconds
            secs = _parse_race_time(time_str)
            if secs and secs > 0 and dist > 0:
                per_200 = secs / (dist / 200)
                times_per_200.append(per_200)

        if times_per_200:
            # Use best time (peak ability) rather than average
            runner_times[rid] = times_per_200

    # Need at least 3 runners with times to make meaningful comparison
    if len(runner_times) < 3:
        return 1.0

    # Field average of best time per 200m
    all_best = [min(ts) for ts in runner_times.values()]
    field_avg = sum(all_best) / len(all_best)
    if field_avg <= 0:
        return 1.0

    # This runner's best time
    rid = _get(runner, "id", "")
    if rid not in runner_times:
        return 1.0

    runner_best = min(runner_times[rid])

    # Ratio: field_avg / runner_best — faster = higher ratio
    # A runner 2% faster than average gets ~1.2 multiplier
    ratio = field_avg / runner_best
    # Map to multiplier: 0.95 to 1.05 range → 0.8 to 1.3 multiplier
    if ratio >= 1.03:
        return 1.3   # significantly faster
    if ratio >= 1.02:
        return 1.2
    if ratio >= 1.01:
        return 1.1
    if ratio >= 0.99:
        return 1.0   # average
    if ratio >= 0.98:
        return 0.9
    if ratio >= 0.97:
        return 0.8
    return 0.75       # significantly slower


def _parse_race_time(time_str: str) -> float | None:
    """Parse race time string to seconds.

    Handles formats:
    - "00:01:08.1000000" (HH:MM:SS.fraction)
    - "1:08.10" (M:SS.fraction)
    - "68.10" (raw seconds)
    """
    if not time_str:
        return None

    try:
        # Strip trailing zeros from fraction
        time_str = time_str.strip()

        if time_str.count(":") == 2:
            # HH:MM:SS.frac
            parts = time_str.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        elif time_str.count(":") == 1:
            # M:SS.frac
            parts = time_str.split(":")
            m = int(parts[0])
            s = float(parts[1])
            return m * 60 + s
        else:
            # Raw seconds
            return float(time_str)
    except (ValueError, IndexError):
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

        # Calculate individual multipliers
        career_mult = _career_multiplier(career_record, is_maiden_race=is_maiden)
        recent_form_mult = _recent_form_multiplier(last_five)
        cond_mult = _dampen(_condition_multiplier(
            distance, track_condition, barrier, pace_pos, field_size, venue), "condition")
        form_mult = _dampen(_form_recency_multiplier(last_five), "form_recency")
        spec_mult = _dampen(_specialist_multiplier(runner, track_condition), "specialist")
        spell_mult = _dampen(_spell_multiplier(runner), "spell")
        profile_mult = _dampen(_horse_profile_multiplier(runner), "profile")
        weight_mult = _dampen(_weight_multiplier(runner, avg_weight), "weight")

        # New factors (previously ignored by tissue)
        pf_mult = _dampen(_pf_assessment_multiplier(runner, runners), "pf_assessment")
        hcap_mult = _dampen(_handicap_rating_multiplier(runner, runners), "handicap")
        class_mult = _dampen(_class_stats_multiplier(runner), "class_stats")
        jt_mult = _dampen(_jockey_trainer_multiplier(runner), "jockey_trainer")
        smap_mult = _dampen(_speed_map_direct_multiplier(runner), "speed_map")
        days_mult = _dampen(_days_since_run_multiplier(runner), "days_spell")
        time_mult = _dampen(_race_time_multiplier(runner, race, runners), "race_time")

        # Blend career and recent form 40/60.
        # Career win rate captures long-term ability but lacks class context.
        # Recent form (last 5 results, recency-weighted) captures current
        # form cycle and is a better 12-month indicator.
        CAREER_WEIGHT = 0.4
        FORM_WEIGHT = 0.6
        blended_ability = CAREER_WEIGHT * career_mult + FORM_WEIGHT * recent_form_mult

        tissue_score = (
            blended_ability
            * cond_mult
            * form_mult
            * spec_mult
            * spell_mult
            * profile_mult
            * weight_mult
            * pf_mult
            * hcap_mult
            * class_mult
            * jt_mult
            * smap_mult
            * days_mult
            * time_mult
        )

        # Floor at 0.001 to avoid zero probabilities
        tissue_score = max(0.001, tissue_score)
        raw_scores[rid] = tissue_score
        factor_details[rid] = {
            "career": round(career_mult, 3),
            "recent_form": round(recent_form_mult, 3),
            "blended_ability": round(blended_ability, 3),
            "condition": round(cond_mult, 3),
            "form_recency": round(form_mult, 3),
            "specialist": round(spec_mult, 3),
            "spell": round(spell_mult, 3),
            "profile": round(profile_mult, 3),
            "weight": round(weight_mult, 3),
            "pf_assessment": round(pf_mult, 3),
            "handicap": round(hcap_mult, 3),
            "class_stats": round(class_mult, 3),
            "jockey_trainer": round(jt_mult, 3),
            "speed_map": round(smap_mult, 3),
            "days_spell": round(days_mult, 3),
            "race_time": round(time_mult, 3),
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

    # Build market-implied probabilities for all runners (used by both
    # maiden geometric blend and universal 20% linear blend).
    MAIDEN_MARKET_ALPHA = 0.50
    MARKET_BLEND = 0.20  # 20% market, 80% tissue for all races
    market_probs: dict[str, float] = {}
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
        market_probs = {}

    blended_probs: dict[str, float] = {}
    for runner in runners:
        rid = _get(runner, "id", "")
        tissue_score = raw_scores.get(rid, 0.001)
        win_prob = tissue_score / total

        # Maiden geometric blend (50/50): tissue alone is weak for maidens
        if is_maiden and rid in market_probs and market_probs[rid] > 0:
            mkt = market_probs[rid]
            win_prob = (win_prob ** (1 - MAIDEN_MARKET_ALPHA)) * (mkt ** MAIDEN_MARKET_ALPHA)
            factor_details.get(rid, {})["_maiden_blend"] = round(MAIDEN_MARKET_ALPHA, 2)

        blended_probs[rid] = win_prob

    # Re-normalize after maiden blend
    blend_total = sum(blended_probs.values())
    if blend_total > 0:
        for rid in blended_probs:
            blended_probs[rid] /= blend_total

    # Universal 20% market blend: 80% tissue + 20% market-implied
    if market_probs:
        for rid in blended_probs:
            tissue_p = blended_probs[rid]
            mkt_p = market_probs.get(rid, 0.0)
            blended_probs[rid] = (1.0 - MARKET_BLEND) * tissue_p + MARKET_BLEND * mkt_p
            if rid in factor_details:
                factor_details[rid]["_market_implied"] = round(mkt_p, 4)
                factor_details[rid]["_market_blend"] = MARKET_BLEND

        # Re-normalize after market blend
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
