"""Probability calculation engine for horse racing picks.

Calculates win/place probability, value detection, and recommended stakes
for each runner in a race using multi-factor analysis across 10 factors
grouped into 6 categories:
  - Market Intelligence: Market consensus + market movement
  - Form & Fitness: Form rating + class/fitness
  - Race Dynamics: Pace factor + barrier draw
  - Connections: Jockey & trainer stats
  - Physical: Weight carried + horse profile
  - Pattern Intelligence: Deep learning patterns from historical analysis
"""

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Deep Learning Pattern Cache (module-level, sync)
# ──────────────────────────────────────────────
_dl_pattern_cache: list[dict] | None = None


def set_dl_pattern_cache(patterns: list[dict]) -> None:
    """Store DL patterns in module-level cache for sync access."""
    global _dl_pattern_cache
    _dl_pattern_cache = patterns
    logger.info("DL pattern cache set: %d patterns", len(patterns))


def get_dl_pattern_cache() -> list[dict] | None:
    """Get cached DL patterns. Falls back to JSON file if DB cache empty."""
    if _dl_pattern_cache is not None:
        return _dl_pattern_cache
    # Fallback: load from exported JSON file (for backtesting / offline use)
    return _load_dl_patterns_from_file()


def _load_dl_patterns_from_file() -> list[dict] | None:
    """Load DL patterns from JSON file as fallback when DB not available."""
    global _dl_pattern_cache
    dl_path = Path(__file__).parent / "data" / "dl_patterns.json"
    if not dl_path.exists():
        return None
    try:
        with open(dl_path, "r") as f:
            patterns = json.load(f)
        _dl_pattern_cache = patterns
        logger.info("DL patterns loaded from file: %d patterns", len(patterns))
        return patterns
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load DL patterns from file: %s", e)
        return None


# ──────────────────────────────────────────────
# Context Profiles (pre-computed factor multipliers)
# ──────────────────────────────────────────────
_CONTEXT_PROFILES: dict | None = None


def _load_context_profiles() -> dict | None:
    """Load pre-computed context profiles from JSON file.

    Profiles contain factor multipliers for different racing contexts
    (venue type × distance × class × condition), computed from historical
    analysis of 190K+ runners. Loaded once at first use.
    """
    global _CONTEXT_PROFILES
    if _CONTEXT_PROFILES is not None:
        return _CONTEXT_PROFILES

    profiles_path = Path(__file__).parent / "data" / "context_profiles.json"
    if not profiles_path.exists():
        logger.debug("No context profiles file found at %s", profiles_path)
        _CONTEXT_PROFILES = {}
        return _CONTEXT_PROFILES

    try:
        with open(profiles_path, "r") as f:
            data = json.load(f)
        _CONTEXT_PROFILES = data
        n_profiles = len(data.get("profiles", {}))
        n_fallbacks = len(data.get("fallbacks", {}))
        logger.info("Context profiles loaded: %d profiles, %d fallbacks", n_profiles, n_fallbacks)
        return _CONTEXT_PROFILES
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load context profiles: %s", e)
        _CONTEXT_PROFILES = {}
        return _CONTEXT_PROFILES


def _track_key(venue: str) -> str:
    """Normalize venue name to a stable track key for per-track profiles."""
    v = venue.lower().strip()
    # Normalize common aliases
    if v in ("the valley", "moonee valley"):
        return "moonee_valley"
    if v in ("royal randwick", "randwick"):
        return "randwick"
    return v.replace(" ", "_").replace("'", "")


def _get_context_multipliers(race: Any, meeting: Any, outcome_type: str = "win") -> dict[str, float]:
    """Get factor multipliers for the current race context.

    Uses hierarchical fallback (per-track first, then venue-type):
    1. track|distance|class      (e.g. flemington|sprint|open)
    2. venue_type|distance|class (e.g. metro_vic|sprint|open)
    3. track|distance            (e.g. flemington|sprint)
    4. venue_type|distance       (e.g. metro_vic|sprint)
    5. distance|class            (e.g. sprint|open)
    6. Default: empty (no adjustment)

    Args:
        outcome_type: "win", "place", or "top4" — selects which outcome's
            multipliers to use from the nested profile structure.
    """
    data = _load_context_profiles()
    if not data:
        return {}

    profiles = data.get("profiles", {})
    fallbacks = data.get("fallbacks", {})

    # Build context keys
    venue = _get(meeting, "venue") or ""
    state = _get_state_for_venue(venue)
    distance = _get(race, "distance") or 1400
    race_class = _get(race, "class_") or ""

    track = _track_key(venue)
    vtype = _context_venue_type(venue, state)
    dbucket = _get_dist_bucket(distance)
    cbucket = _context_class_bucket(race_class)

    # Hierarchical lookup: per-track first, then venue-type category
    lookup_order = [
        (profiles, f"{track}|{dbucket}|{cbucket}"),
        (profiles, f"{vtype}|{dbucket}|{cbucket}"),
        (fallbacks, f"{track}|{dbucket}"),
        (fallbacks, f"{vtype}|{dbucket}"),
        (fallbacks, f"{dbucket}|{cbucket}"),
    ]

    for source, key in lookup_order:
        if key in source:
            profile = source[key]
            # New nested structure: profile["win"], profile["place"], etc.
            if outcome_type in profile and isinstance(profile[outcome_type], dict):
                return dict(profile[outcome_type])
            # Legacy flat structure (backward compat)
            if "market" in profile:
                return {k: v for k, v in profile.items() if k != "_n"}
            return {}

    return {}


def _context_venue_type(venue: str, state: str) -> str:
    """Classify venue into type for context profiles."""
    v = venue.lower().strip()
    _METRO_VIC = {"flemington", "caulfield", "moonee valley", "sandown", "the valley"}
    _METRO_NSW = {"randwick", "rosehill", "royal randwick", "canterbury", "warwick farm"}
    _METRO_QLD = {"eagle farm", "doomben"}
    _METRO_SA = {"morphettville"}
    _METRO_WA = {"ascot", "belmont"}

    if v in _METRO_VIC:
        return "metro_vic"
    if v in _METRO_NSW:
        return "metro_nsw"
    if v in _METRO_QLD:
        return "metro_qld"
    if v in _METRO_SA or v in _METRO_WA:
        return "metro_other"
    s = (state or "").upper().strip()
    if s in ("VIC", "NSW", "QLD"):
        return "provincial"
    return "country"


def _context_class_bucket(race_class: str) -> str:
    """Classify race class for context profiles."""
    rc = race_class.lower().strip().rstrip(";")
    if "maiden" in rc or "mdn" in rc:
        return "maiden"
    if "class 1" in rc or "cl1" in rc:
        return "class1"
    if "restricted" in rc or "rst " in rc:
        return "restricted"
    bm = re.search(r"(?:bm|benchmark)\s*(\d+)", rc)
    if bm:
        rating = int(bm.group(1))
        if rating <= 58:
            return "bm58"
        if rating <= 68:
            return "bm64"
        if rating <= 76:
            return "bm72"
        return "open"
    if any(kw in rc for kw in ("group", "listed", "stakes", "quality")):
        return "open"
    if "open" in rc:
        return "open"
    if "class 2" in rc or "cl2" in rc:
        return "class2"
    if "class 3" in rc or "cl3" in rc:
        return "class3"
    if "class 4" in rc or "class 5" in rc or "class 6" in rc:
        return "bm72"
    return "bm64"


# ──────────────────────────────────────────────
# Calibrated Parameters (fitted from 190K historical runners)
# ──────────────────────────────────────────────
_CALIBRATION: dict | None = None


def _load_calibration() -> dict | None:
    """Load calibrated scoring parameters from historical analysis.

    Contains empirical scoring curves (signal value -> win rate -> 0.05-0.95 score)
    and optimal factor weights fitted from 190K+ runners with known results.
    Falls back to hand-coded logic if not found.
    """
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION

    cal_path = Path(__file__).parent / "data" / "calibrated_params.json"
    if not cal_path.exists():
        logger.debug("No calibration file found at %s", cal_path)
        _CALIBRATION = {}
        return _CALIBRATION

    try:
        with open(cal_path, "r") as f:
            data = json.load(f)
        _CALIBRATION = data
        n_curves = len(data.get("scoring_curves", {}))
        logger.info("Calibration loaded: %d scoring curves, weights=%s",
                     n_curves, data.get("weights", {}))
        return _CALIBRATION
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load calibration: %s", e)
        _CALIBRATION = {}
        return _CALIBRATION


def _calibrated_score(signal_name: str, raw_value: float,
                      fallback: float = 0.5) -> float:
    """Look up empirical score from calibrated bin curve.

    Interpolates between bins to produce a smooth score. The scoring curves
    map raw signal values to 0.05-0.95 scores based on actual historical win rates.

    Args:
        signal_name: Name of the signal (e.g., "jockey_career_sr")
        raw_value: Raw signal value to score
        fallback: Score to return if no calibration available

    Returns:
        Score in [0.05, 0.95] range
    """
    cal = _load_calibration()
    if not cal:
        return fallback

    curves = cal.get("scoring_curves", {})
    curve = curves.get(signal_name)
    if not curve:
        return fallback

    bins = curve.get("bins", [])
    scores = curve.get("scores", [])
    if not bins or not scores or len(bins) < 2:
        return fallback

    # Binary search for the right bin
    if raw_value <= bins[0]:
        return scores[0]
    if raw_value >= bins[-1]:
        return scores[-1]

    # Find bin index and interpolate
    for i in range(len(bins) - 1):
        if raw_value < bins[i + 1]:
            # Interpolate within bin
            bin_start = bins[i]
            bin_end = bins[i + 1]
            if i < len(scores):
                score_start = scores[i]
                score_end = scores[min(i + 1, len(scores) - 1)]
                if bin_end > bin_start:
                    t = (raw_value - bin_start) / (bin_end - bin_start)
                    return score_start + t * (score_end - score_start)
                return score_start
            return fallback

    return scores[-1] if scores else fallback


def _get_calibrated_weights() -> dict[str, float] | None:
    """Get optimal factor weights from calibration, or None to use defaults."""
    cal = _load_calibration()
    if not cal:
        return None
    weights = cal.get("weights")
    if not weights:
        return None
    # Calibration doesn't include deep_learning — add it with residual weight
    if "deep_learning" not in weights:
        # Give DL the average of the smaller factor weights
        non_market_form = {k: v for k, v in weights.items()
                          if k not in ("market", "form")}
        dl_weight = sum(non_market_form.values()) / max(len(non_market_form), 1) * 0.8
        weights["deep_learning"] = round(dl_weight, 4)
    # Normalise to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    return weights


# Factor registry — defines all probability factors with metadata
FACTOR_REGISTRY = {
    "market":         {"label": "Market Consensus", "category": "Market Intelligence",
                       "description": "Multi-bookmaker median odds stripped of overround"},
    "movement":       {"label": "Market Movement",  "category": "Market Intelligence",
                       "description": "Odds shift direction and magnitude — smart money signals"},
    "form":           {"label": "Form Rating",      "category": "Form & Fitness",
                       "description": "Recent results, track/distance/condition win rates"},
    "class_fitness":  {"label": "Class & Fitness",   "category": "Form & Fitness",
                       "description": "Class level suitability and fitness from spell length"},
    "pace":           {"label": "Pace Factor",       "category": "Race Dynamics",
                       "description": "Speed map position, pace scenario, and map factor"},
    "barrier":        {"label": "Barrier Draw",      "category": "Race Dynamics",
                       "description": "Gate position advantage relative to field size and distance"},
    "jockey_trainer": {"label": "Jockey & Trainer",  "category": "Connections",
                       "description": "Jockey and trainer win rate statistics"},
    "weight_carried": {"label": "Weight Carried",    "category": "Physical",
                       "description": "Carried weight relative to race average"},
    "horse_profile":  {"label": "Horse Profile",     "category": "Physical",
                       "description": "Age and sex peak performance assessment"},
    "deep_learning":  {"label": "Deep Learning",     "category": "Pattern Intelligence",
                       "description": "Historical pattern edges from 280K+ runners analysis"},
}

# Default weights (must sum to 1.0)
# Calibrated from 190K+ historical runners via scripts/run_calibration.py
# Previous hand-tuned weights: market=0.22, form=0.15, pace=0.11, jt=0.11, barrier=0.09, dl=0.10, movement=0.07, class=0.05, weight=0.05, profile=0.05
DEFAULT_WEIGHTS = {
    "market": 0.40,
    "form": 0.32,
    "deep_learning": 0.08,
    "jockey_trainer": 0.07,
    "weight_carried": 0.04,
    "horse_profile": 0.03,
    "class_fitness": 0.03,
    "barrier": 0.02,
    "movement": 0.00,
    "pace": 0.01,
}  # sums to 1.0 — movement zeroed (100% neutral), pace near-zero (anti-predictive)

# Legacy aliases
WEIGHT_MARKET = DEFAULT_WEIGHTS["market"]
WEIGHT_FORM = DEFAULT_WEIGHTS["form"]
WEIGHT_PACE = DEFAULT_WEIGHTS["pace"]
WEIGHT_MOVEMENT = DEFAULT_WEIGHTS["movement"]
WEIGHT_CLASS = DEFAULT_WEIGHTS["class_fitness"]

# Baseline win rate for an average runner (1/field_size fallback)
DEFAULT_BASELINE = 0.10

# Value detection thresholds
VALUE_THRESHOLD = 0.90       # include near-fair-value horses
STRONG_VALUE_THRESHOLD = 1.20  # 20% edge = strong value

# Kelly fraction (quarter-Kelly for conservative sizing)
KELLY_FRACTION = 0.25
DEFAULT_POOL = 20.0  # $20 per race pool


@dataclass
class RunnerProbability:
    """Calculated probability data for a single runner."""

    win_probability: float = 0.0       # 0.0 - 1.0 (displayed as percentage)
    place_probability: float = 0.0     # 0.0 - 1.0
    market_implied: float = 0.0        # raw market probability (before our adjustments)
    value_rating: float = 1.0          # our_prob / market_prob (>1.0 = value)
    place_value_rating: float = 1.0    # place_prob / market_implied_place (>1.0 = value)
    edge: float = 0.0                  # our_prob - market_prob
    recommended_stake: float = 0.0     # quarter-Kelly from $20 pool
    factors: dict = field(default_factory=dict)  # breakdown per factor


@dataclass
class StatsRecord:
    """Parsed racing stats (starts: wins-seconds-thirds)."""

    starts: int = 0
    wins: int = 0
    seconds: int = 0
    thirds: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.starts if self.starts > 0 else 0.0

    @property
    def place_rate(self) -> float:
        return (self.wins + self.seconds + self.thirds) / self.starts if self.starts > 0 else 0.0


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def calculate_race_probabilities(
    runners: list,
    race: Any,
    meeting: Any,
    pool: float = DEFAULT_POOL,
    weights: dict[str, float] | None = None,
    dl_patterns: list[dict] | None = None,
) -> dict[str, "RunnerProbability"]:
    """Calculate probabilities for all active runners in a race.

    Args:
        runners: List of Runner ORM objects (or dicts with runner fields)
        race: Race ORM object (or dict)
        meeting: Meeting ORM object (or dict)
        pool: Total stake pool per race (default $20)
        weights: Custom factor weights (key → 0.0-1.0). Defaults to DEFAULT_WEIGHTS.
        dl_patterns: Deep learning patterns from PatternInsight table.

    Returns:
        Dict mapping runner_id → RunnerProbability
    """
    active = [r for r in runners if not _get(r, "scratched", False)]
    if not active:
        return {}

    # Use cached DL patterns if not explicitly passed
    if dl_patterns is None:
        dl_patterns = get_dl_pattern_cache()

    w = weights if weights else (_get_calibrated_weights() or DEFAULT_WEIGHTS)
    field_size = len(active)
    baseline = 1.0 / field_size if field_size > 0 else DEFAULT_BASELINE
    track_condition = _get(meeting, "track_condition") or _get(race, "track_condition") or ""
    race_distance = _get(race, "distance") or 1400

    # Determine pace scenario from race analysis or speed map positions
    pace_scenario = _determine_pace_scenario(active)

    # Pre-calculate shared values
    overround = _calculate_overround(active)
    avg_weight = _average_weight(active)

    # Step 1: Calculate raw factor scores for each runner
    raw_scores: dict[str, float] = {}
    market_implied: dict[str, float] = {}
    runner_odds: dict[str, float] = {}
    factor_details: dict[str, dict] = {}
    all_factor_scores: dict[str, dict] = {}

    for runner in active:
        rid = _get(runner, "id", "")

        # Calculate all factor scores
        mkt_raw = _market_consensus(runner, overround)
        # Use calibrated market curve if available (maps raw prob → better score)
        mkt_score = _calibrated_score("market_prob", mkt_raw, fallback=mkt_raw) if mkt_raw > 0 else 0.0
        scores = {
            "market":         mkt_score,
            "movement":       _market_movement_factor(runner),
            "form":           _form_rating(runner, track_condition, baseline),
            "class_fitness":  _class_factor(runner, baseline, race),
            "pace":           _pace_factor(runner, pace_scenario),
            "barrier":        _barrier_draw_factor(runner, field_size, race_distance),
            "jockey_trainer": _jockey_trainer_factor(runner, baseline),
            "weight_carried": _weight_factor(runner, avg_weight, race_distance, race),
            "horse_profile":  _horse_profile_factor(runner, race),
            "deep_learning":  _deep_learning_factor(
                runner, meeting, race_distance, track_condition,
                field_size, dl_patterns,
            ),
        }

        all_factor_scores[rid] = scores
        market_implied[rid] = mkt_raw  # raw market probability for value detection
        odds = _get_median_odds(runner)
        runner_odds[rid] = odds or 0.0
        factor_details[rid] = {k: round(v, 4) for k, v in scores.items()}

    # Step 1a: Apply context multipliers (amplify/dampen factors by racing context)
    ctx_mults = _get_context_multipliers(race, meeting)
    if ctx_mults:
        for rid, scores in all_factor_scores.items():
            for factor, mult in ctx_mults.items():
                if factor in scores and factor != "deep_learning":  # DL already context-aware
                    original = scores[factor]
                    adjusted = 0.5 + (original - 0.5) * mult
                    scores[factor] = max(0.05, min(0.95, adjusted))
        # Update factor_details with context-adjusted scores
        for rid, scores in all_factor_scores.items():
            factor_details[rid] = {k: round(v, 4) for k, v in scores.items()}

    # Step 1b: Dynamic market weight boost for low-information races
    eff_w = _boost_market_weight(w, all_factor_scores)

    # Step 1c: Compute weighted sums with effective weights
    for rid, scores in all_factor_scores.items():
        raw = sum(eff_w.get(k, 0.0) * v for k, v in scores.items())
        raw_scores[rid] = max(0.001, raw)

    # Step 2: Normalize to sum to 1.0
    total = sum(raw_scores.values())
    win_probs: dict[str, float] = {}
    for rid in raw_scores:
        win_probs[rid] = raw_scores[rid] / total if total > 0 else baseline

    # Step 2a: Sharpen distribution to fix favorite-longshot bias
    # Model undershoots strong picks (predicts 22% when actual is 31%).
    # Power > 1 concentrates probability on favorites, reducing longshot overestimate.
    SHARPEN = 1.45
    sharpened = {rid: p ** SHARPEN for rid, p in win_probs.items()}
    sharp_total = sum(sharpened.values())
    if sharp_total > 0:
        win_probs = {rid: p / sharp_total for rid, p in sharpened.items()}

    # Step 2b: Market floor — prevent extreme disagreement with market
    win_probs = _apply_market_floor(win_probs, market_implied)

    results: dict[str, RunnerProbability] = {}

    for runner in active:
        rid = _get(runner, "id", "")
        win_prob = win_probs.get(rid, baseline)

        # Place probability
        place_prob = _place_probability(win_prob, field_size)

        # Value detection (win)
        mkt_prob = market_implied.get(rid, baseline)
        value = win_prob / mkt_prob if mkt_prob > 0 else 1.0
        edge = win_prob - mkt_prob

        # Place value detection — use place odds if available
        place_odds = _get(runner, "place_odds", None)
        if place_odds and place_odds > 1.0:
            market_implied_place = 1.0 / place_odds
            place_value = place_prob / market_implied_place if market_implied_place > 0 else 1.0
        else:
            # Approximate: place market implied ≈ place_probability from market consensus
            mkt_place = _place_probability(mkt_prob, field_size)
            place_value = place_prob / mkt_place if mkt_place > 0 else 1.0

        # Recommended stake (quarter-Kelly)
        odds = runner_odds.get(rid, 0.0)
        stake = _recommended_stake(win_prob, odds, pool)

        results[rid] = RunnerProbability(
            win_probability=round(win_prob, 4),
            place_probability=round(place_prob, 4),
            market_implied=round(mkt_prob, 4),
            value_rating=round(value, 3),
            place_value_rating=round(place_value, 3),
            edge=round(edge, 4),
            recommended_stake=round(stake, 2),
            factors=factor_details.get(rid, {}),
        )

    return results


# ──────────────────────────────────────────────
# Factor: Market Consensus
# ──────────────────────────────────────────────

def _calculate_overround(runners: list) -> float:
    """Calculate total market overround from all runners' median odds."""
    total = 0.0
    for runner in runners:
        odds = _get_median_odds(runner)
        if odds and odds > 1.0:
            total += 1.0 / odds
    return total if total > 0 else 1.0


def _market_consensus(runner: Any, overround: float) -> float:
    """Convert multi-bookmaker odds to normalized implied probability."""
    odds = _get_median_odds(runner)
    if not odds or odds <= 1.0:
        return 0.0
    raw_implied = 1.0 / odds
    return raw_implied / overround if overround > 0 else raw_implied


def _get_median_odds(runner: Any) -> Optional[float]:
    """Get median odds across all available bookmakers."""
    sources = [
        _get(runner, "current_odds"),
        _get(runner, "odds_tab"),
        _get(runner, "odds_sportsbet"),
        _get(runner, "odds_bet365"),
        _get(runner, "odds_ladbrokes"),
        _get(runner, "odds_betfair"),
    ]
    valid = [o for o in sources if o and isinstance(o, (int, float)) and o > 1.0]
    if not valid:
        return None
    return statistics.median(valid)


# ──────────────────────────────────────────────
# Factor: Form Rating
# ──────────────────────────────────────────────

def _form_rating(runner: Any, track_condition: str, baseline: float) -> float:
    """Rate recent form strength based on results and stats."""
    score = 0.5  # neutral baseline
    signals = 0
    signal_sum = 0.0

    # Recent form (last_five: "12x34" = recent finish positions)
    last_five = _get(runner, "last_five")
    if last_five and isinstance(last_five, str):
        lf_score = _score_last_five(last_five)
        signal_sum += lf_score
        signals += 1

    # Track + distance stats (strongest form indicator)
    td = parse_stats_string(_get(runner, "track_dist_stats"))
    if td and td.starts >= 4:
        td_score = _stat_to_score(td.win_rate, baseline, "track_dist_sr")
        signal_sum += td_score * 1.5  # higher weight for track+distance
        signals += 1.5

    # Condition-appropriate stats
    cond_stats_field = _condition_stats_field(track_condition)
    if cond_stats_field:
        cond = parse_stats_string(_get(runner, cond_stats_field))
        if cond and cond.starts >= 3:
            cond_signal = {"good_track_stats": "cond_good_sr", "soft_track_stats": "cond_soft_sr",
                           "heavy_track_stats": "cond_heavy_sr"}.get(cond_stats_field)
            signal_sum += _stat_to_score(cond.win_rate, baseline, cond_signal)
            signals += 1

    # Distance stats (broader than track+distance)
    dist = parse_stats_string(_get(runner, "distance_stats"))
    if dist and dist.starts >= 3:
        signal_sum += _stat_to_score(dist.win_rate, baseline, "distance_sr") * 0.8
        signals += 0.8

    # First-up / second-up stats
    days_since = _get(runner, "days_since_last_run")
    if days_since and days_since > 60:
        fu = parse_stats_string(_get(runner, "first_up_stats"))
        if fu and fu.starts >= 2:
            signal_sum += _stat_to_score(fu.win_rate, baseline, "first_up_sr") * 0.7
            signals += 0.7
    elif days_since and 21 <= days_since <= 60:
        su = parse_stats_string(_get(runner, "second_up_stats"))
        if su and su.starts >= 2:
            signal_sum += _stat_to_score(su.win_rate, baseline, "second_up_sr") * 0.5
            signals += 0.5

    # Career win/place percentage (Q5-Q1: 26.6% win, 18.8% place)
    career = parse_stats_string(_get(runner, "career_record"))
    if career and career.starts >= 10:
        career_win_score = _stat_to_score(career.win_rate, baseline, "career_win_pct")
        career_place_score = _stat_to_score(career.place_rate, baseline * 3, "career_place_pct")
        career_score = career_win_score * 0.7 + career_place_score * 0.3
        signal_sum += career_score * 0.6
        signals += 0.6

    # Average condition score — aggregate across all track conditions (Q5-Q1: 10.6%)
    cond_total_starts = 0
    cond_total_wins = 0
    for cond_field in ("good_track_stats", "soft_track_stats", "heavy_track_stats"):
        cond_s = parse_stats_string(_get(runner, cond_field))
        if cond_s:
            cond_total_starts += cond_s.starts
            cond_total_wins += cond_s.wins
    if cond_total_starts >= 5:
        avg_cond_wr = cond_total_wins / cond_total_starts
        signal_sum += _stat_to_score(avg_cond_wr, baseline) * 0.4
        signals += 0.4

    if signals > 0:
        score = signal_sum / signals

    return max(0.02, min(0.95, score))


def _score_last_five(last_five: str) -> float:
    """Score a last_five string (e.g. '12x34') with recency weighting."""
    score = 0.5
    # Filter to digit/x characters only
    positions = [c for c in last_five.strip() if c.isdigit() or c.lower() == "x"]

    for i, pos in enumerate(positions[:5]):
        weight = 1.0 - (i * 0.15)  # most recent = highest weight
        if pos == "1":
            score += 0.08 * weight
        elif pos == "2":
            score += 0.04 * weight
        elif pos == "3":
            score += 0.02 * weight
        elif pos.lower() == "x":
            score -= 0.02 * weight
        elif pos.isdigit() and int(pos) >= 7:
            score -= 0.03 * weight

    return max(0.05, min(0.95, score))


def _stat_to_score(win_rate: float, baseline: float,
                    signal_name: str | None = None) -> float:
    """Convert a win rate to a 0.0-1.0 score relative to baseline.

    Uses calibrated empirical curves when available (from 190K+ historical runners),
    falls back to piecewise linear mapping.
    """
    # Try calibrated curve first
    if signal_name and win_rate > 0:
        cal_score = _calibrated_score(signal_name, win_rate)
        if cal_score != 0.5:  # 0.5 is the fallback default — means no curve found
            return cal_score

    # Fallback: piecewise linear
    if baseline <= 0:
        baseline = 0.10
    ratio = win_rate / baseline
    if ratio <= 1.0:
        score = 0.2 + 0.3 * ratio
    else:
        score = 0.5 + 0.175 * min(ratio - 1.0, 2.0)
    return max(0.05, min(0.90, score))


def _condition_stats_field(track_condition: str) -> Optional[str]:
    """Map track condition to the appropriate stats field name."""
    if not track_condition:
        return None
    tc = track_condition.lower()
    if any(k in tc for k in ("heavy", "hvy")):
        return "heavy_track_stats"
    if any(k in tc for k in ("soft", "sft")):
        return "soft_track_stats"
    if any(k in tc for k in ("good", "firm", "gd")):
        return "good_track_stats"
    return None


# ──────────────────────────────────────────────
# Factor: Pace/Speed
# ──────────────────────────────────────────────

def _pace_factor(runner: Any, pace_scenario: str) -> float:
    """Calculate pace advantage/disadvantage score."""
    score = 0.5  # neutral

    # Map factor is the primary signal
    map_factor = _get(runner, "pf_map_factor")
    if map_factor and isinstance(map_factor, (int, float)):
        # map_factor > 1.0 = advantage, < 1.0 = disadvantage
        adjustment = (map_factor - 1.0) * 0.5
        score += max(-0.25, min(0.25, adjustment))

    # Speed rank + pace scenario interaction
    speed_rank = _get(runner, "pf_speed_rank")
    position = _get(runner, "speed_map_position") or ""

    if speed_rank and isinstance(speed_rank, int):
        if speed_rank <= 5:  # early speed horse
            if pace_scenario in ("slow_pace", "genuine_pace"):
                score += 0.05  # leaders benefit
            elif pace_scenario == "hot_pace":
                score -= 0.04  # too much pace competition
        elif speed_rank >= 15:  # backmarker
            if pace_scenario == "hot_pace":
                score += 0.04  # pace collapses, closers benefit
            elif pace_scenario == "slow_pace":
                score -= 0.04  # hard to run down slow leaders

    # Speed map position confirmation
    if position == "leader":
        if pace_scenario in ("slow_pace", "genuine_pace"):
            score += 0.03
        elif pace_scenario == "hot_pace":
            score -= 0.03
    elif position == "backmarker":
        if pace_scenario == "hot_pace":
            score += 0.03
        elif pace_scenario == "slow_pace":
            score -= 0.03

    # Jockey factor (small supporting signal)
    jockey_factor = _get(runner, "pf_jockey_factor")
    if jockey_factor and isinstance(jockey_factor, (int, float)):
        adj = (jockey_factor - 1.0) * 0.1
        score += max(-0.05, min(0.05, adj))

    return max(0.05, min(0.95, score))


def _determine_pace_scenario(runners: list) -> str:
    """Determine pace scenario from speed map positions."""
    leaders = sum(1 for r in runners if _get(r, "speed_map_position") == "leader")
    on_pace = sum(1 for r in runners if _get(r, "speed_map_position") == "on_pace")

    if leaders >= 3:
        return "hot_pace"
    elif leaders == 0 and on_pace <= 2:
        return "slow_pace"
    elif leaders == 1:
        return "genuine_pace"
    else:
        return "moderate_pace"


# ──────────────────────────────────────────────
# Factor: Market Movement
# ──────────────────────────────────────────────

def _market_movement_factor(runner: Any) -> float:
    """Score based on odds movement direction and magnitude."""
    score = 0.5  # neutral

    flucs_json = _get(runner, "odds_flucs")
    if flucs_json and isinstance(flucs_json, str):
        try:
            flucs = json.loads(flucs_json)
            if isinstance(flucs, list) and len(flucs) >= 2:
                opening = _parse_odds_value(flucs[0].get("odds"))
                latest = _parse_odds_value(flucs[-1].get("odds"))

                if opening and latest and opening > 0:
                    pct_change = ((latest - opening) / opening) * 100

                    if pct_change <= -20:  # heavy support
                        score += 0.10
                    elif pct_change <= -10:  # firming
                        score += 0.05
                    elif pct_change >= 30:  # big drift
                        score -= 0.08
                    elif pct_change >= 15:  # drifting
                        score -= 0.04
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    else:
        # Fallback to opening vs current odds
        opening = _get(runner, "opening_odds")
        current = _get(runner, "current_odds")
        if opening and current and opening > 0:
            pct_change = ((current - opening) / opening) * 100
            if pct_change <= -20:
                score += 0.08
            elif pct_change <= -10:
                score += 0.04
            elif pct_change >= 20:
                score -= 0.06
            elif pct_change >= 10:
                score -= 0.03

    return max(0.05, min(0.95, score))


# ──────────────────────────────────────────────
# Factor: Class/Fitness
# ──────────────────────────────────────────────

def _class_factor(runner: Any, baseline: float, race: Any = None) -> float:
    """Score based on class suitability and fitness."""
    score = 0.5
    has_signal = False

    # Class stats (from parse_stats_string — handles both JSON and string formats)
    cls = parse_stats_string(_get(runner, "class_stats"))
    if cls and cls.starts >= 2:
        ratio = cls.win_rate / baseline if baseline > 0 else cls.win_rate / 0.10
        adjustment = (ratio - 1.0) * 0.15
        score += max(-0.15, min(0.15, adjustment))
        has_signal = True

    # Handicap rating as class proxy (higher = better class)
    # Neutral point ~70, range ~45-115 for Australian racing
    handicap = _get(runner, "handicap_rating")
    if handicap and isinstance(handicap, (int, float)) and handicap > 0:
        # Normalize around 70: below = class drop concern, above = class edge
        deviation = (handicap - 70.0) / 30.0  # 30-point range = full swing
        adj = max(-0.10, min(0.10, deviation * 0.10))
        score += adj
        has_signal = True

    # Prize money per start (Q5-Q1: 16.1%)
    career = parse_stats_string(_get(runner, "career_record"))
    prize_money = _get(runner, "career_prize_money")
    if career and career.starts >= 5 and prize_money and isinstance(prize_money, (int, float)) and prize_money > 0:
        prize_per_start = prize_money / career.starts
        # Class-appropriate benchmark
        race_class = (_get(race, "class_") or "") if race else ""
        rc_lower = race_class.lower()
        if any(kw in rc_lower for kw in ("group", "listed", "stakes")):
            benchmark = 50000
        elif "bm" in rc_lower:
            bm_match = re.search(r"bm\s*(\d+)", rc_lower)
            rating = int(bm_match.group(1)) if bm_match else 64
            benchmark = max(rating * 500, 15000)
        else:
            benchmark = 15000  # Maiden/Class 1/Restricted
        ratio = prize_per_start / benchmark if benchmark > 0 else 1.0
        prize_adj = max(-0.10, min(0.10, (ratio - 1.0) * 0.10))
        score += prize_adj
        has_signal = True

    # Average margin from recent starts (Q5-Q1: 10.6%)
    form_hist = _get(runner, "form_history")
    if form_hist:
        try:
            if isinstance(form_hist, str):
                fh = json.loads(form_hist)
            else:
                fh = form_hist
            if isinstance(fh, list):
                margins = []
                for start in fh[:5]:
                    pos = start.get("position") or start.get("pos")
                    margin_raw = start.get("margin")
                    if pos == 1 or str(pos) == "1":
                        margins.append(0.0)
                    elif margin_raw is not None:
                        mv = _parse_margin_value(margin_raw)
                        if mv is not None:
                            margins.append(mv)
                if len(margins) >= 3:
                    avg_margin = sum(margins) / len(margins)
                    if avg_margin <= 2.0:
                        score += 0.06
                    elif avg_margin <= 4.0:
                        score += 0.03
                    elif avg_margin <= 6.0:
                        pass  # neutral
                    elif avg_margin <= 10.0:
                        score -= 0.03
                    else:
                        score -= 0.06
                    has_signal = True
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    # Days since last run — fitness curve
    days = _get(runner, "days_since_last_run")
    if days and isinstance(days, int):
        if 14 <= days <= 28:
            score += 0.05  # sweet spot
        elif 7 <= days <= 13:
            score += 0.03  # backed up quickly, okay
        elif 29 <= days <= 42:
            score += 0.01  # still fresh enough
        elif days > 90:
            score -= 0.05  # long layoff concern
        elif days > 60:
            score -= 0.03  # moderate layoff

    return max(0.05, min(0.95, score))


# ──────────────────────────────────────────────
# Factor: Barrier Draw
# ──────────────────────────────────────────────

def _barrier_draw_factor(runner: Any, field_size: int, distance: int = 1400) -> float:
    """Score based on barrier position relative to field size and distance.

    Inside barriers get a slight boost; wide gates are penalized more at
    shorter distances where there's less time to recover.
    """
    score = 0.5  # neutral
    barrier = _get(runner, "barrier")
    if not barrier or not isinstance(barrier, (int, float)) or barrier < 1:
        return score

    barrier = int(barrier)

    # Relative position (0.0 = rail, 1.0 = widest)
    if field_size <= 1:
        return score
    relative = (barrier - 1) / max(field_size - 1, 1)

    # Distance multiplier — barriers matter more in sprints
    if distance <= 1200:
        dist_mult = 1.3
    elif distance <= 1600:
        dist_mult = 1.0
    else:
        dist_mult = 0.7

    # Inside (relative < 0.3) = advantage; outside (relative > 0.7) = disadvantage
    if relative <= 0.15:
        score += 0.06 * dist_mult  # inside 2-3 gates
    elif relative <= 0.30:
        score += 0.03 * dist_mult  # inner quarter
    elif relative >= 0.85:
        score -= 0.08 * dist_mult  # widest gates
    elif relative >= 0.70:
        score -= 0.04 * dist_mult  # outer quarter

    return max(0.05, min(0.95, score))


# ──────────────────────────────────────────────
# Factor: Jockey & Trainer
# ──────────────────────────────────────────────

def _parse_a2e_json(stats_str: Any) -> dict | None:
    """Parse PF A2E JSON format: {"career": {"a2e": 1.07, ...}, "last100": {...}, ...}.

    Returns parsed dict if valid A2E format, None otherwise.
    """
    if isinstance(stats_str, dict):
        data = stats_str
    elif isinstance(stats_str, str):
        try:
            data = json.loads(stats_str)
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        return None

    if not isinstance(data, dict):
        return None
    # A2E format has "career" key with nested stats
    if "career" in data and isinstance(data["career"], dict):
        return data
    return None


def _a2e_to_score(a2e_data: dict, baseline: float) -> float:
    """Score a jockey/trainer from PF A2E data.

    Uses strike_rate vs baseline, A2E ratio (>1.0 = outperforming earnings),
    and Profit on Turnover for a composite score.
    """
    career = a2e_data.get("career", {})
    runners = career.get("runners", 0) or 0
    if runners < 10:
        return 0.5  # not enough data

    score = 0.5
    adjustments = 0.0
    signals = 0

    # Strike rate vs baseline (primary signal)
    sr = career.get("strike_rate", 0) or 0
    sr_decimal = sr / 100.0 if sr > 1 else sr  # handle both % and decimal
    if sr_decimal > 0 and baseline > 0:
        ratio = sr_decimal / baseline
        # Map ratio: 0.5→-0.12, 1.0→0, 1.5→+0.12, 2.0→+0.20
        adj = max(-0.15, min(0.20, (ratio - 1.0) * 0.15))
        adjustments += adj * 0.5
        signals += 0.5

    # A2E ratio (>1.0 means earning more than expected)
    a2e = career.get("a2e", 0) or 0
    if a2e > 0:
        # Map a2e: 0.8→-0.08, 1.0→0, 1.2→+0.08, 1.5→+0.15
        adj = max(-0.12, min(0.15, (a2e - 1.0) * 0.40))
        adjustments += adj * 0.3
        signals += 0.3

    # Profit on Turnover
    pot = career.get("pot", 0) or 0
    if pot != 0:
        # pot is percentage: -20 bad, 0 break-even, +10 good
        adj = max(-0.08, min(0.10, pot / 100.0))
        adjustments += adj * 0.2
        signals += 0.2

    # Use last100 for recency signal if available
    last100 = a2e_data.get("last100", {})
    if last100 and (last100.get("runners", 0) or 0) >= 20:
        l100_sr = (last100.get("strike_rate", 0) or 0)
        l100_decimal = l100_sr / 100.0 if l100_sr > 1 else l100_sr
        if l100_decimal > 0 and sr_decimal > 0:
            # Recent form vs career: trending up = bonus
            trend = l100_decimal / sr_decimal if sr_decimal > 0 else 1.0
            adj = max(-0.05, min(0.05, (trend - 1.0) * 0.10))
            adjustments += adj
            signals += 0.1 if adj != 0 else 0

    if signals > 0:
        score += adjustments / signals * signals  # preserve signal weighting

    return max(0.05, min(0.95, score))


def _jockey_trainer_factor(runner: Any, baseline: float) -> float:
    """Score based on jockey and trainer statistics.

    Jockey weighted 60%, trainer 40% — jockey has more direct race impact.
    Supports PF A2E JSON format (rich career stats) and racing.com string format.
    """
    score = 0.5
    signals = 0
    signal_sum = 0.0

    # --- Jockey (60% weight) ---
    j_raw = _get(runner, "jockey_stats")
    j_a2e = _parse_a2e_json(j_raw)
    if j_a2e:
        j_score = _a2e_to_score(j_a2e, baseline)
        signal_sum += j_score * 0.6
        signals += 0.6
    else:
        jockey = parse_stats_string(j_raw)
        if jockey and jockey.starts >= 3:
            j_score = _stat_to_score(jockey.win_rate, baseline, "jockey_career_sr")
            signal_sum += j_score * 0.6
            signals += 0.6

    # --- Trainer (40% weight) ---
    t_raw = _get(runner, "trainer_stats")
    t_a2e = _parse_a2e_json(t_raw)
    if t_a2e:
        t_score = _a2e_to_score(t_a2e, baseline)
        signal_sum += t_score * 0.4
        signals += 0.4
    else:
        trainer = parse_stats_string(t_raw)
        if trainer and trainer.starts >= 3:
            t_score = _stat_to_score(trainer.win_rate, baseline, "trainer_career_sr")
            signal_sum += t_score * 0.4
            signals += 0.4

    # --- Combo bonus: jockey+trainer combination (20% extra signal) ---
    if j_a2e and "combo_career" in j_a2e:
        combo = j_a2e["combo_career"]
        combo_runners = combo.get("runners", 0) or 0
        if combo_runners >= 5:
            combo_sr = (combo.get("strike_rate", 0) or 0)
            combo_decimal = combo_sr / 100.0 if combo_sr > 1 else combo_sr
            if combo_decimal > 0 and baseline > 0:
                ratio = combo_decimal / baseline
                combo_score = 0.5 + max(-0.15, min(0.20, (ratio - 1.0) * 0.15))
                signal_sum += combo_score * 0.2
                signals += 0.2

    # --- Combo last 100 rides (Q5-Q1: 13.1% — recency premium) ---
    if j_a2e and "combo_last100" in j_a2e:
        combo_l100 = j_a2e["combo_last100"]
        combo_l100_runners = combo_l100.get("runners", 0) or 0
        if combo_l100_runners >= 20:
            l100_sr = (combo_l100.get("strike_rate", 0) or 0)
            l100_decimal = l100_sr / 100.0 if l100_sr > 1 else l100_sr
            if l100_decimal > 0 and baseline > 0:
                ratio = l100_decimal / baseline
                l100_score = 0.5 + max(-0.12, min(0.18, (ratio - 1.0) * 0.15))
                signal_sum += l100_score * 0.25
                signals += 0.25

    if signals > 0:
        score = signal_sum / signals

    return max(0.05, min(0.95, score))


# ──────────────────────────────────────────────
# Factor: Weight Carried
# ──────────────────────────────────────────────

def _weight_factor(runner: Any, avg_weight: float, race_distance: int = 1400, race: Any = None) -> float:
    """Score based on carried weight relative to race average.

    In Australian handicap racing, heavier weight = handicapper rates horse higher.
    Two competing effects:
      - Class signal: heavier weight → better horse (positive, dominant effect)
      - Physical burden: extra kgs slow the horse (negative, secondary effect)

    Net effect: slight positive for heavier horses (class proxy dominates),
    with diminishing returns at extreme weights where physical burden catches up.
    Weight effect is amplified in low-class races (9.2% spread vs 4.1% open).
    """
    score = 0.5
    weight = _get(runner, "weight")
    if not weight or not isinstance(weight, (int, float)) or weight <= 0:
        return score
    if avg_weight <= 0:
        return score

    diff = weight - avg_weight  # positive = heavier than average

    # Class proxy: heavier = better horse (small positive)
    # +3kg above avg → +0.04, -3kg below avg → -0.04
    class_adj = (diff / 3.0) * 0.04

    # Physical burden: only at extremes (>3kg above avg), mild negative
    # This captures the diminishing returns of very heavy weights
    burden_adj = 0.0
    if diff > 3.0:
        excess = diff - 3.0
        burden_adj = -(excess / 3.0) * 0.03  # mild penalty for extreme weight

    # Distance scaling: weight matters more in sprints
    dist = max(800, min(3200, race_distance))
    distance_mult = 1.0 + (1400 - dist) * 0.0003

    # Class-dependent amplification: weight matters more in low-class (9.2% vs 4.1%)
    race_class = (_get(race, "class_") or "") if race else ""
    if race_class and any(
        kw in race_class.lower()
        for kw in ("maiden", "class 1", "restricted", "mdn")
    ):
        distance_mult *= 1.5

    adjustment = (class_adj + burden_adj) * distance_mult
    score += max(-0.10, min(0.10, adjustment))

    return max(0.05, min(0.95, score))


def _average_weight(runners: list) -> float:
    """Calculate average carried weight across active runners."""
    weights = []
    for r in runners:
        w = _get(r, "weight")
        if w and isinstance(w, (int, float)) and w > 0:
            weights.append(float(w))
    return statistics.mean(weights) if weights else 0.0


# ──────────────────────────────────────────────
# Factor: Horse Profile
# ──────────────────────────────────────────────

def _horse_profile_factor(runner: Any, race: Any = None) -> float:
    """Score based on horse age and sex with context-dependent adjustments.

    Peak racing age is 4-5yo for gallopers. Young (2yo) and older (8+)
    horses get a mild penalty. Colts get a boost in low-class races (22% SR
    in Maidens/Class 1 vs 8.6% for geldings — backtested across 267K runners).
    """
    score = 0.5

    age = _get(runner, "horse_age")
    if age and isinstance(age, (int, float)):
        age = int(age)
        if 4 <= age <= 5:
            score += 0.05  # peak age
        elif age == 3 or age == 6:
            score += 0.02  # near peak
        elif age == 2:
            score -= 0.03  # immature, less predictable
        elif age == 7:
            score -= 0.01  # slight decline
        elif age >= 8:
            score -= 0.04  # declining

    # Determine race class context for sex adjustment
    race_class = (_get(race, "class_") or "") if race else ""
    is_low_class = any(
        kw in race_class.lower()
        for kw in ("maiden", "class 1", "restricted", "mdn")
    ) if race_class else False

    sex = _get(runner, "horse_sex")
    if sex and isinstance(sex, str):
        sex_lower = sex.lower().strip()
        # Geldings are slightly more consistent runners
        if sex_lower in ("gelding", "g"):
            score += 0.01
        # Mares can be inconsistent in certain conditions
        elif sex_lower in ("mare", "m", "filly", "f"):
            score -= 0.01
        # Colts: context-dependent (22% SR in low-class vs 8.6% in open)
        elif sex_lower in ("colt", "c"):
            if is_low_class:
                score += 0.08  # strong advantage in maidens/low-class
            else:
                score -= 0.02  # slight penalty in open company

    return max(0.05, min(0.95, score))


# ──────────────────────────────────────────────
# Market Weight Boost & Floor
# ──────────────────────────────────────────────

def _boost_market_weight(
    weights: dict[str, float],
    all_scores: dict[str, dict],
) -> dict[str, float]:
    """Boost market weight when other factors lack information.

    When most non-market factors return near-neutral (0.5), it means
    there's limited form/stats data (e.g. maiden races). In these cases,
    market consensus is the strongest signal and should be weighted higher.

    Thresholds:
      - neutral_ratio <= 0.4 → no boost (enough data)
      - neutral_ratio 0.8+ → full boost (market weight up to ~50%)
    """
    if not all_scores:
        return weights

    non_market_keys = [k for k in weights if k != "market"]

    # Count how many non-market factors are near-neutral across all runners
    total_checks = 0
    neutral_count = 0
    for scores in all_scores.values():
        for k in non_market_keys:
            total_checks += 1
            if abs(scores.get(k, 0.5) - 0.5) <= 0.05:
                neutral_count += 1

    neutral_ratio = neutral_count / total_checks if total_checks > 0 else 0

    if neutral_ratio <= 0.4:
        return weights  # Enough information, use default weights

    # Boost market weight proportionally to how little info we have
    # neutral_ratio 0.4 → no boost, 0.8+ → full boost
    boost_factor = min(1.0, (neutral_ratio - 0.4) / 0.4)
    max_boost = 0.43  # Market weight can increase from 22% up to ~65%
    boost = boost_factor * max_boost

    effective = dict(weights)
    effective["market"] = weights["market"] + boost

    # Scale non-market weights down to maintain sum = 1.0
    non_market_total = sum(weights[k] for k in non_market_keys)
    if non_market_total > 0:
        scale = (1.0 - effective["market"]) / non_market_total
        for k in non_market_keys:
            effective[k] = weights[k] * scale

    return effective


def _apply_market_floor(
    win_probs: dict[str, float],
    market_implied: dict[str, float],
) -> dict[str, float]:
    """Blend probabilities towards market when model strongly disagrees.

    Prevents the model from giving a runner less than 35% of what the
    market implies. For a $1.40 favourite (market 59%), this ensures the
    model gives at least ~21% instead of the compressed 17% that happens
    when most factors are neutral.

    Only applies to runners with meaningful market presence (>5% implied).
    Uses 60/40 blend (model-heavy) to preserve model signal while
    anchoring closer to market for strong favourites.
    Re-normalizes after adjustments to maintain probabilities summing to 1.0.
    """
    adjusted = dict(win_probs)
    needs_renorm = False

    for rid in adjusted:
        mkt = market_implied.get(rid, 0)
        if mkt > 0.05:  # Only for runners with real market presence
            ratio = adjusted[rid] / mkt if mkt > 0 else 1.0
            if ratio < 0.50:
                # Blend 60/40 model/market — anchors toward market for favourites
                adjusted[rid] = adjusted[rid] * 0.6 + mkt * 0.4
                needs_renorm = True

    if needs_renorm:
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {rid: p / total for rid, p in adjusted.items()}

    return adjusted


# ──────────────────────────────────────────────
# Factor: Deep Learning Patterns
# ──────────────────────────────────────────────

def _deep_learning_factor(
    runner: Any,
    meeting: Any,
    race_distance: int,
    track_condition: str,
    field_size: int,
    dl_patterns: list[dict] | None,
) -> float:
    """Score based on matching deep learning patterns from historical analysis.

    Matches runner attributes against 844+ statistical patterns discovered
    from 280K+ historical runners. Each matching pattern's edge contributes
    to the score, weighted by confidence and capped to prevent dominance.
    """
    if not dl_patterns:
        return 0.5  # neutral when no patterns available

    # Skip non-discriminative pattern types — these fire for ALL runners equally
    # in a race, so they cancel out after normalization and just add noise.
    # condition_specialist: matches every runner on that condition
    # market: duplicates the market factor (already 40% weight)
    # seasonal: matches every runner at that time of year
    # track_dist_cond: matches every runner in the same race
    # standard_times: lookup table, not a per-runner pattern
    _SKIP_TYPES = {
        "deep_learning_condition_specialist",
        "deep_learning_market",
        "deep_learning_seasonal",
        "deep_learning_track_dist_cond",
        "deep_learning_standard_times",
    }

    score = 0.5
    total_adjustment = 0.0

    # Pre-compute runner attributes for matching
    venue = _get(meeting, "venue") or ""
    dist_bucket = _get_dist_bucket(race_distance)
    condition = _normalize_condition(track_condition)
    position = _get(runner, "speed_map_position") or ""
    pace_style = _position_to_style(position)
    barrier = _get(runner, "barrier")
    barrier_bucket = _get_barrier_bucket(barrier, field_size) if barrier else ""
    jockey = _get(runner, "jockey") or ""
    trainer = _get(runner, "trainer") or ""
    move_type = _get_move_type(runner)

    # Derive state from venue using assessment mapping
    state = _get_state_for_venue(venue)

    for pattern in dl_patterns:
        p_type = pattern.get("type", "")
        conds = pattern.get("conditions", {})
        edge = pattern.get("edge", 0.0)
        confidence = pattern.get("confidence", "LOW")

        if not edge or confidence == "LOW":
            continue

        if p_type in _SKIP_TYPES:
            continue

        # Confidence multiplier: HIGH patterns count more
        conf_mult = 1.0 if confidence == "HIGH" else 0.6

        matched = False

        if p_type == "deep_learning_pace" and pace_style:
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
                and conds.get("style") == pace_style
            )

        elif p_type == "deep_learning_barrier_bias" and barrier_bucket:
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and conds.get("barrier_bucket") == barrier_bucket
            )

        elif p_type == "deep_learning_jockey_trainer" and jockey and trainer:
            matched = (
                conds.get("jockey", "").lower() == jockey.lower()
                and conds.get("trainer", "").lower() == trainer.lower()
                and (not state or conds.get("state") == state)
            )

        elif p_type == "deep_learning_track_dist_cond":
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
            )

        elif p_type == "deep_learning_acceleration" and move_type:
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
                and conds.get("move_type") == move_type
            )

        elif p_type == "deep_learning_pace_collapse" and pace_style == "leader":
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
            )

        elif p_type == "deep_learning_condition_specialist":
            matched = _condition_matches(
                conds.get("condition", ""), condition
            )

        elif p_type == "deep_learning_market" and state:
            odds = _get_median_odds(runner)
            if odds:
                sp_range = _odds_to_sp_range(odds)
                matched = (
                    conds.get("state") == state
                    and conds.get("sp_range") == sp_range
                )

        elif p_type == "deep_learning_class_mover" and state:
            # Match class movers (upgrade/downgrade runners backed/drifting)
            direction = conds.get("direction", "")
            indicator = conds.get("indicator", "")
            # Derive class movement from runner data
            runner_class_move = _get(runner, "class_move") or ""
            runner_mkt_move = _get_market_direction(runner)
            matched = (
                conds.get("state") == state
                and direction == runner_class_move
                and indicator == runner_mkt_move
            )

        elif p_type == "deep_learning_weight_impact":
            # Match weight change impact by distance
            change_class = conds.get("change_class", "")
            runner_wt_change = _get_weight_change_class(runner)
            matched = (
                conds.get("dist_bucket") == dist_bucket
                and conds.get("state") == state
                and change_class == runner_wt_change
            )

        elif p_type == "deep_learning_form_trend" and state:
            # Match form trend (improving/declining)
            trend = conds.get("trend", "")
            runner_trend = _get_form_trend(runner)
            matched = (
                conds.get("state") == state
                and trend == runner_trend
            )

        elif p_type == "deep_learning_bounceback":
            # Match bounceback candidates (excuse for last poor run)
            excuse_type = conds.get("excuse_type", "")
            runner_excuse = _get_excuse_type(runner)
            matched = (
                conds.get("dist_bucket") == dist_bucket
                and conds.get("state") == state
                and excuse_type == runner_excuse
            )

        elif p_type == "deep_learning_form_cycle":
            # Match form cycle (runs this prep)
            prep_runs = conds.get("prep_runs")
            runner_prep = _get(runner, "prep_runs") or _get(runner, "runs_since_spell")
            if prep_runs is not None and runner_prep is not None:
                matched = (int(runner_prep) == int(prep_runs))

        elif p_type == "deep_learning_class_transition":
            # Match class transition patterns
            transition = conds.get("transition", "")
            runner_transition = _get(runner, "class_move") or ""
            trans_map = {"class_drop": "downgrade", "class_rise": "upgrade"}
            matched = (trans_map.get(transition, transition) == runner_transition)

        elif p_type == "deep_learning_track_bias":
            # Match track bias (wide/inside runners at specific venues)
            width = conds.get("width", "")
            pattern_venue = conds.get("venue", "")
            runner_barrier = barrier or 0
            # wide = barrier > 60% of field, inside = barrier <= 30%
            if field_size > 0 and runner_barrier:
                ratio = runner_barrier / field_size
                runner_width = "wide" if ratio > 0.6 else "inside" if ratio <= 0.3 else "mid"
            else:
                runner_width = ""
            matched = (
                _venue_matches(pattern_venue, venue)
                and conds.get("dist_bucket") == dist_bucket
                and width == runner_width
            )

        elif p_type == "deep_learning_seasonal" and state:
            # Match seasonal patterns
            import datetime
            month = conds.get("month")
            matched = (
                conds.get("state") == state
                and month == datetime.date.today().month
            )

        if matched:
            # Cap individual pattern contribution
            capped_edge = max(-0.15, min(0.15, edge))
            total_adjustment += capped_edge * conf_mult

    # Cap total adjustment to prevent extreme swings
    total_adjustment = max(-0.25, min(0.25, total_adjustment))
    score += total_adjustment

    return max(0.05, min(0.95, score))


def _get_dist_bucket(distance: int) -> str:
    """Map distance to bucket matching deep learning and context patterns."""
    if distance <= 1100:
        return "sprint"
    elif distance <= 1399:
        return "short"
    elif distance <= 1799:
        return "middle"
    elif distance <= 2199:
        return "classic"
    else:
        return "staying"


def _normalize_condition(condition: str) -> str:
    """Normalize track condition to match pattern keys."""
    if not condition:
        return ""
    c = condition.lower().strip()
    if "heavy" in c or "hvy" in c:
        return "Heavy"
    if "soft" in c or "sft" in c:
        return "Soft"
    if "synthetic" in c or "all weather" in c:
        return "Synthetic"
    if "firm" in c:
        return "Firm"
    # Default to Good (includes Good 3, Good 4, etc.)
    if "good" in c or "gd" in c:
        return "Good"
    return "Good"


def _position_to_style(position: str) -> str:
    """Map speed_map_position to pattern style key."""
    mapping = {
        "leader": "leader",
        "on_pace": "on_pace",
        "midfield": "midfield",
        "backmarker": "backmarker",
    }
    return mapping.get(position, "")


def _get_barrier_bucket(barrier: Any, field_size: int) -> str:
    """Map barrier number to inside/middle/outside bucket."""
    if not barrier or not isinstance(barrier, (int, float)) or field_size < 2:
        return ""
    b = int(barrier)
    third = max(1, field_size / 3)
    if b <= third:
        return "inside"
    elif b <= third * 2:
        return "middle"
    else:
        return "outside"


def _get_move_type(runner: Any) -> str:
    """Determine market movement type for acceleration pattern matching."""
    opening = _get(runner, "opening_odds")
    current = _get(runner, "current_odds")
    if not opening or not current or opening <= 0:
        return ""
    pct = ((current - opening) / opening) * 100
    if pct <= -20:
        return "big_mover"
    elif pct <= -10:
        return "improver"
    elif pct >= 20:
        return "fader"
    elif abs(pct) < 10:
        return "steady"
    return ""


def _odds_to_sp_range(odds: float) -> str:
    """Map odds to SP range bucket matching market patterns."""
    if odds < 3:
        return "$1-$3"
    elif odds < 5:
        return "$3-$5"
    elif odds < 8:
        return "$5-$8"
    elif odds < 12:
        return "$8-$12"
    elif odds < 20:
        return "$12-$20"
    else:
        return "$20+"


def _get_market_direction(runner: Any) -> str:
    """Determine if runner is 'backed' or 'drifting' from market movement."""
    opening = _get(runner, "opening_odds")
    current = _get(runner, "current_odds")
    if not opening or not current or opening <= 0:
        return ""
    pct = ((current - opening) / opening) * 100
    if pct <= -5:
        return "backed"
    elif pct >= 5:
        return "drifting"
    return ""


def _get_weight_change_class(runner: Any) -> str:
    """Classify weight change for weight_impact pattern matching."""
    wt = _get(runner, "weight")
    last_wt = _get(runner, "last_start_weight")
    if not wt or not last_wt:
        return ""
    diff = float(wt) - float(last_wt)
    if diff > 2.0:
        return "weight_up_big"
    elif diff > 0:
        return "weight_up_small"
    elif diff < -2.0:
        return "weight_down_big"
    elif diff < 0:
        return "weight_down_small"
    return "weight_same"


def _get_form_trend(runner: Any) -> str:
    """Determine form trend from last_five or recent positions."""
    last5 = _get(runner, "last_five") or ""
    if not last5:
        return ""
    positions = []
    for ch in str(last5).replace(" ", ""):
        if ch.isdigit():
            positions.append(int(ch))
        elif ch.lower() == "x":
            positions.append(9)
    if len(positions) < 3:
        return ""
    recent = sum(positions[:2]) / 2  # last 2 starts
    older = sum(positions[2:min(4, len(positions))]) / max(1, min(2, len(positions) - 2))
    if recent < older - 1:
        return "improving"
    elif recent > older + 1:
        return "declining"
    return "steady"


def _get_excuse_type(runner: Any) -> str:
    """Determine bounceback excuse type from last start data."""
    last_pos = _get(runner, "last_start_position")
    margin = _get(runner, "last_start_margin")
    if not last_pos:
        return ""
    pos = int(last_pos) if str(last_pos).isdigit() else 0
    if pos == 0:
        return ""
    if margin and float(margin) > 10:
        return "heavy_loss"
    if pos >= 8:
        return "bad_run"
    return ""


def _venue_matches(pattern_venue: str, meeting_venue: str) -> bool:
    """Check if a pattern venue matches the meeting venue (partial match)."""
    if not pattern_venue or not meeting_venue:
        return False
    pv = pattern_venue.lower().strip()
    mv = meeting_venue.lower().strip()
    return pv in mv or mv in pv


def _condition_matches(pattern_cond: str, normalized_cond: str) -> bool:
    """Check if pattern condition matches normalized condition."""
    if not pattern_cond or not normalized_cond:
        return False
    # Pattern conditions can be "G", "S5", "H10", "Good", "Soft", etc.
    pc = pattern_cond.lower().strip()
    nc = normalized_cond.lower().strip()
    # Direct match
    if pc == nc:
        return True
    # Short code matches: G→Good, S→Soft, H→Heavy
    if pc.startswith("g") and nc == "good":
        return True
    if pc.startswith("s") and nc == "soft":
        return True
    if pc.startswith("h") and nc == "heavy":
        return True
    if pc == "synthetic" and nc == "synthetic":
        return True
    return False


# Inline venue-to-state mapping (subset of assessment.py)
_VENUE_STATE = {
    "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC",
    "sandown": "VIC", "pakenham": "VIC", "cranbourne": "VIC",
    "mornington": "VIC", "ballarat": "VIC", "bendigo": "VIC",
    "geelong": "VIC", "sale": "VIC", "warrnambool": "VIC",
    "kilmore": "VIC", "wangaratta": "VIC",
    "randwick": "NSW", "rosehill": "NSW", "canterbury": "NSW",
    "warwick farm": "NSW", "newcastle": "NSW", "kembla grange": "NSW",
    "gosford": "NSW", "wyong": "NSW", "hawkesbury": "NSW",
    "tamworth": "NSW", "taree": "NSW", "nowra": "NSW",
    "eagle farm": "QLD", "doomben": "QLD", "gold coast": "QLD",
    "sunshine coast": "QLD", "toowoomba": "QLD", "ipswich": "QLD",
    "mackay": "QLD", "rockhampton": "QLD", "cairns": "QLD",
    "morphettville": "SA", "murray bridge": "SA", "gawler": "SA",
    "port augusta": "SA", "port lincoln": "SA",
    "ascot": "WA", "belmont": "WA", "bunbury": "WA", "albany": "WA",
    "hobart": "TAS", "launceston": "TAS", "devonport": "TAS",
    "fannie bay": "NT", "darwin": "NT", "alice springs": "NT",
}


def _get_state_for_venue(venue: str) -> str:
    """Look up state from venue name."""
    if not venue:
        return ""
    v = venue.lower().strip()
    if v in _VENUE_STATE:
        return _VENUE_STATE[v]
    # Partial match
    for known, state in _VENUE_STATE.items():
        if known in v or v in known:
            return state
    return ""


# ──────────────────────────────────────────────
# Place Probability
# ──────────────────────────────────────────────

def _place_probability(win_prob: float, field_size: int) -> float:
    """Estimate place probability from win probability and field size.

    Uses empirical multipliers calibrated to Australian racing place terms:
    - Fields <= 7: places 1-2
    - Fields 8+: places 1-3
    """
    if field_size <= 5:
        factor = 2.0
    elif field_size <= 8:
        factor = 2.5
    elif field_size <= 12:
        factor = 3.0
    else:
        factor = 3.3

    return min(0.95, win_prob * factor)


# ──────────────────────────────────────────────
# Recommended Stake (Quarter-Kelly)
# ──────────────────────────────────────────────

def _recommended_stake(
    prob: float, odds: float, pool: float = DEFAULT_POOL,
) -> float:
    """Calculate recommended stake using quarter-Kelly criterion.

    Args:
        prob: Calculated win probability (0.0 - 1.0)
        odds: Decimal odds (e.g. 5.0 for $5.00)
        pool: Total available pool (default $20)

    Returns:
        Recommended stake in dollars (0.0 if no edge)
    """
    if not odds or odds <= 1.0 or prob <= 0:
        return 0.0

    b = odds - 1  # net odds (profit per $1 if win)
    q = 1 - prob

    # Kelly fraction: (bp - q) / b
    kelly = (b * prob - q) / b

    if kelly <= 0:
        return 0.0  # no edge, don't bet

    stake = kelly * KELLY_FRACTION * pool
    # Round to nearest 50 cents, minimum $1
    stake = round(stake * 2) / 2
    return max(0.0, min(pool, stake))


# ──────────────────────────────────────────────
# Stats string parser
# ──────────────────────────────────────────────

# Common formats from racing.com:
#   "5: 2-1-0"  (starts: wins-seconds-thirds)
#   "5-2-1-0"   (starts-wins-seconds-thirds)
#   "2-1-0 (5)" (wins-seconds-thirds (starts))
_STATS_COLON = re.compile(r"(\d+)\s*:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)")
_STATS_DASH4 = re.compile(r"^(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)$")
_STATS_PAREN = re.compile(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\((\d+)\)")


def parse_stats_string(stats_str: Any) -> Optional[StatsRecord]:
    """Parse a racing stats string into a StatsRecord.

    Handles multiple formats:
      "5: 2-1-0" → StatsRecord(starts=5, wins=2, seconds=1, thirds=0)
      "5-2-1-0"  → StatsRecord(starts=5, wins=2, seconds=1, thirds=0)
      '{"starts": 5, "wins": 2, ...}' → StatsRecord (PF JSON format)
      dict with starts/wins keys → StatsRecord (direct dict input)
    """
    if not stats_str:
        return None

    # Handle dict passed directly (from internal callers)
    if isinstance(stats_str, dict):
        starts = stats_str.get("starts", 0) or 0
        if starts > 0:
            return StatsRecord(
                starts=int(starts),
                wins=int(stats_str.get("wins", 0) or 0),
                seconds=int(stats_str.get("seconds", 0) or 0),
                thirds=int(stats_str.get("thirds", 0) or 0),
            )
        # Also handle "firsts" key (PF uses firsts, not wins in some records)
        firsts = stats_str.get("firsts", 0) or 0
        starts = stats_str.get("Starts", stats_str.get("starts", 0)) or 0
        if starts and int(starts) > 0:
            return StatsRecord(
                starts=int(starts),
                wins=int(firsts),
                seconds=int(stats_str.get("seconds", stats_str.get("Seconds", 0)) or 0),
                thirds=int(stats_str.get("thirds", stats_str.get("Thirds", 0)) or 0),
            )
        return None

    if not isinstance(stats_str, str):
        return None

    stats_str = stats_str.strip()
    if not stats_str:
        return None

    # Try JSON parse first (PF format: {"starts": 10, "wins": 2, ...})
    if stats_str.startswith("{"):
        try:
            data = json.loads(stats_str)
            if isinstance(data, dict):
                starts = data.get("starts", data.get("Starts", 0)) or 0
                wins = data.get("wins", data.get("firsts", data.get("Firsts", 0))) or 0
                if int(starts) > 0:
                    return StatsRecord(
                        starts=int(starts),
                        wins=int(wins),
                        seconds=int(data.get("seconds", data.get("Seconds", 0)) or 0),
                        thirds=int(data.get("thirds", data.get("Thirds", 0)) or 0),
                    )
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass

    # Format: "5: 2-1-0"
    m = _STATS_COLON.search(stats_str)
    if m:
        return StatsRecord(
            starts=int(m.group(1)),
            wins=int(m.group(2)),
            seconds=int(m.group(3)),
            thirds=int(m.group(4)),
        )

    # Format: "5-2-1-0"
    m = _STATS_DASH4.match(stats_str)
    if m:
        return StatsRecord(
            starts=int(m.group(1)),
            wins=int(m.group(2)),
            seconds=int(m.group(3)),
            thirds=int(m.group(4)),
        )

    # Format: "2-1-0 (5)"
    m = _STATS_PAREN.search(stats_str)
    if m:
        return StatsRecord(
            starts=int(m.group(4)),
            wins=int(m.group(1)),
            seconds=int(m.group(2)),
            thirds=int(m.group(3)),
        )

    return None


# ──────────────────────────────────────────────
# Margin Parsing
# ──────────────────────────────────────────────

def _parse_margin_value(margin: Any) -> Optional[float]:
    """Convert a margin value to lengths (float). Returns None if unparseable.

    Handles numeric values, string lengths ("1.5L"), and abbreviations
    (NK, HD, LNG, DST) from Punting Form data.
    """
    if margin is None:
        return None
    if isinstance(margin, (int, float)):
        return float(margin) if margin >= 0 else None
    if not isinstance(margin, str):
        return None
    ms = margin.strip().upper()
    if not ms:
        return None

    # Winner (position 1) — no margin
    if ms in ("0", "0.0", "0L", "WIN", "WON", "DH"):
        return 0.0

    # Standard abbreviations
    _ABBREVS = {
        "NK": 0.05, "NOSE": 0.05, "SH-NK": 0.05, "N": 0.05,
        "HD": 0.1, "HEAD": 0.1, "SHD": 0.1,
        "SNK": 0.2, "SHORT-NECK": 0.2,
        "LNG": 15.0, "LONG": 15.0,
        "DST": 25.0, "DIST": 25.0,
    }
    if ms in _ABBREVS:
        return _ABBREVS[ms]

    # Length format: "1.5L", "3L", "1.5", "3"
    m = re.match(r"^([\d.]+)\s*L?$", ms)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from ORM object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _parse_odds_value(val: Any) -> Optional[float]:
    """Parse an odds value that may be a string with $ prefix or a number."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace("$", "").strip())
        except ValueError:
            return None
    return None


# ──────────────────────────────────────────────
# Exotic Probability Calculations
# ──────────────────────────────────────────────

# TAB takeout rates by exotic type
TAB_TAKEOUT = {
    "Quinella": 0.15,
    "Exacta": 0.20,
    "Trifecta Box": 0.20,
    "First4 Box": 0.25,
}


@dataclass
class ExoticCombination:
    """Pre-calculated exotic bet combination with value analysis."""

    exotic_type: str        # "Quinella", "Exacta", "Trifecta Box", "First4 Box"
    runners: list[int]      # saddlecloth numbers
    runner_names: list[str]  # horse names for display
    estimated_probability: float  # our Harville model probability
    market_probability: float     # market-implied Harville probability
    value_ratio: float      # our_prob / market_prob (>1.0 = value)
    cost: float             # always $20
    num_combos: int         # 1 for flat, 6/24/etc for boxed
    format: str             # "flat" or "boxed"


@dataclass
class SequenceLegAnalysis:
    """Probability-based analysis for a single sequence bet leg."""

    race_number: int
    top_runners: list[dict]  # [{saddlecloth, horse_name, win_prob, value_rating}]
    leg_confidence: str      # "HIGH" / "MED" / "LOW"
    suggested_width: int     # how many runners to include in this leg


def _harville_probability(probs_ordered: list[float]) -> float:
    """Calculate Harville model probability for an ordered finishing sequence.

    For an ordered sequence [P(A), P(B), P(C)], calculates the probability that
    A finishes 1st, B finishes 2nd, C finishes 3rd using conditional probabilities.
    """
    if not probs_ordered:
        return 0.0
    result = 1.0
    remaining = 1.0
    for p in probs_ordered:
        if remaining <= 0 or p <= 0:
            return 0.0
        result *= p / remaining
        remaining -= p
    return result


def _quinella_probability(prob_a: float, prob_b: float) -> float:
    """Probability of two runners filling top 2 in any order (Harville)."""
    if prob_a <= 0 or prob_b <= 0:
        return 0.0
    remaining_a = 1.0 - prob_a
    remaining_b = 1.0 - prob_b
    if remaining_a <= 0 or remaining_b <= 0:
        return 0.0
    return prob_a * (prob_b / remaining_a) + prob_b * (prob_a / remaining_b)


def _box_probability(probs: list[float], positions: int) -> float:
    """Probability of runners filling top-N positions in any order.

    For N runners in `positions` places: sums Harville probability over all
    C(N, positions) subsets × positions! permutations.

    Examples:
        - Trifecta box 3 runners: C(3,3)×3! = 6 permutations
        - Trifecta box 4 runners: C(4,3)×3! = 24 permutations
        - First4 box 4 runners: C(4,4)×4! = 24 permutations
        - First4 box 5 runners: C(5,4)×4! = 120 permutations
    """
    n = len(probs)
    if n < positions:
        return 0.0

    total = 0.0
    for subset_indices in combinations(range(n), positions):
        subset_probs = [probs[i] for i in subset_indices]
        for perm in permutations(subset_probs):
            total += _harville_probability(list(perm))

    return total


def calculate_exotic_combinations(
    runners_data: list[dict],
    stake: float = 20.0,
) -> list[ExoticCombination]:
    """Pre-calculate best exotic combinations for a race.

    Args:
        runners_data: List of dicts, each with keys:
            - saddlecloth (int)
            - horse_name (str)
            - win_prob (float) — our model probability
            - market_implied (float) — market probability
        stake: Total stake per exotic (default $20)

    Returns:
        Top 12 value combinations sorted by value_ratio descending,
        with per-type value filters based on historical performance.
    """
    if len(runners_data) < 2:
        return []

    # Sort by our win probability descending
    sorted_runners = sorted(
        runners_data,
        key=lambda r: r.get("win_prob", 0),
        reverse=True,
    )

    # Restrict to top 4 to align with selections (top 3 + roughie).
    # Only First4 4th position extends to top 5.
    top4 = sorted_runners[:min(4, len(sorted_runners))]
    top5 = sorted_runners[:min(5, len(sorted_runners))]

    # Per-type value thresholds based on actual P&L data
    VALUE_THRESHOLDS = {
        "Quinella": 1.2,           # High-probability play, both runners in selections
        "Exacta": 1.2,
        "Trifecta Box": 1.2,      # Best performer (+3.5% ROI)
        "Trifecta Standout": 1.2,
        "First4": 1.2,            # Positional legs format — targeted, fewer combos
        "First4 Box": 1.5,        # Rare, extreme value only (0/50 all-time)
    }

    # Minimum win probability for lead runner — prevents degenerate exotics
    # where value is high but absolute probability is negligible.
    # Exacta: first runner must realistically win (≥20%)
    # Quinella: at least one runner must be a genuine contender (≥20%)
    MIN_LEAD_PROB = 0.20

    results: list[ExoticCombination] = []

    # --- Quinella: pairs from top 4 ---
    for combo in combinations(top4, 2):
        # At least one runner must be a genuine contender
        if max(combo[0]["win_prob"], combo[1]["win_prob"]) < MIN_LEAD_PROB:
            continue

        our_prob = _quinella_probability(
            combo[0]["win_prob"], combo[1]["win_prob"]
        )
        mkt_prob = _quinella_probability(
            combo[0]["market_implied"], combo[1]["market_implied"]
        )
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

        if value >= VALUE_THRESHOLDS["Quinella"]:
            results.append(ExoticCombination(
                exotic_type="Quinella",
                runners=[r["saddlecloth"] for r in combo],
                runner_names=[r.get("horse_name", "") for r in combo],
                estimated_probability=round(our_prob, 6),
                market_probability=round(mkt_prob, 6),
                value_ratio=round(value, 3),
                cost=stake,
                num_combos=1,
                format="flat",
            ))

    # --- Exacta: ordered pairs from top 4 ---
    for combo in permutations(top4, 2):
        # First runner must realistically win — no degenerate roughie exactas
        if combo[0]["win_prob"] < MIN_LEAD_PROB:
            continue

        our_prob = _harville_probability([r["win_prob"] for r in combo])
        mkt_prob = _harville_probability([r["market_implied"] for r in combo])
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

        if value >= VALUE_THRESHOLDS["Exacta"]:
            results.append(ExoticCombination(
                exotic_type="Exacta",
                runners=[r["saddlecloth"] for r in combo],
                runner_names=[r.get("horse_name", "") for r in combo],
                estimated_probability=round(our_prob, 6),
                market_probability=round(mkt_prob, 6),
                value_ratio=round(value, 3),
                cost=stake,
                num_combos=1,
                format="flat",
            ))

    # --- Trifecta Box: 3 runners from top 4 ---
    for combo in combinations(top4, 3):
        our_probs = [r["win_prob"] for r in combo]
        mkt_probs = [r["market_implied"] for r in combo]
        our_prob = _box_probability(our_probs, 3)
        mkt_prob = _box_probability(mkt_probs, 3)
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

        if value >= VALUE_THRESHOLDS["Trifecta Box"]:
            results.append(ExoticCombination(
                exotic_type="Trifecta Box",
                runners=[r["saddlecloth"] for r in combo],
                runner_names=[r.get("horse_name", "") for r in combo],
                estimated_probability=round(our_prob, 6),
                market_probability=round(mkt_prob, 6),
                value_ratio=round(value, 3),
                cost=stake,
                num_combos=6,
                format="boxed",
            ))

    # --- Trifecta Box: 4 runners from top 4 ---
    if len(top4) >= 4:
        our_probs = [r["win_prob"] for r in top4]
        mkt_probs = [r["market_implied"] for r in top4]
        our_prob = _box_probability(our_probs, 3)
        mkt_prob = _box_probability(mkt_probs, 3)
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

        if value >= VALUE_THRESHOLDS["Trifecta Box"]:
            results.append(ExoticCombination(
                exotic_type="Trifecta Box",
                runners=[r["saddlecloth"] for r in top4],
                runner_names=[r.get("horse_name", "") for r in top4],
                estimated_probability=round(our_prob, 6),
                market_probability=round(mkt_prob, 6),
                value_ratio=round(value, 3),
                cost=stake,
                num_combos=24,
                format="boxed",
            ))

    # --- Trifecta Standout: top pick anchored 1st, others fill 2nd/3rd ---
    if len(top4) >= 3:
        standout = top4[0]
        others = top4[1:4]
        for combo in combinations(others, 2):
            all_runners = [standout] + list(combo)
            # Standout must win, other two fill 2nd/3rd in any order
            our_prob = 0.0
            mkt_prob = 0.0
            for perm in permutations(combo):
                our_prob += _harville_probability(
                    [standout["win_prob"]] + [r["win_prob"] for r in perm]
                )
                mkt_prob += _harville_probability(
                    [standout["market_implied"]] + [r["market_implied"] for r in perm]
                )
            value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

            if value >= VALUE_THRESHOLDS["Trifecta Standout"]:
                results.append(ExoticCombination(
                    exotic_type="Trifecta Standout",
                    runners=[r["saddlecloth"] for r in all_runners],
                    runner_names=[r.get("horse_name", "") for r in all_runners],
                    estimated_probability=round(our_prob, 6),
                    market_probability=round(mkt_prob, 6),
                    value_ratio=round(value, 3),
                    cost=stake,
                    num_combos=2,
                    format="flat",
                ))

    # --- First4 positional (legs format): targeted positions ---
    # 1st: [top1] / 2nd: [top1,top2] / 3rd: [top1,top2,top3] / 4th: [top3,top4,top5]
    if len(top4) >= 4:
        # Build positional legs using probability rankings
        leg1 = top4[:1]    # anchor: best runner
        leg2 = top4[:2]    # top 2
        leg3 = top4[:3]    # top 3
        leg4 = top5[2:5] if len(top5) >= 5 else top4[2:4]  # runners 3-5 for 4th place

        # Calculate positional probability using Harville
        our_prob = 0.0
        mkt_prob = 0.0
        for r1 in leg1:
            for r2 in leg2:
                if r2["saddlecloth"] == r1["saddlecloth"]:
                    continue
                for r3 in leg3:
                    if r3["saddlecloth"] in (r1["saddlecloth"], r2["saddlecloth"]):
                        continue
                    for r4 in leg4:
                        if r4["saddlecloth"] in (r1["saddlecloth"], r2["saddlecloth"], r3["saddlecloth"]):
                            continue
                        our_prob += _harville_probability([
                            r1["win_prob"], r2["win_prob"],
                            r3["win_prob"], r4["win_prob"],
                        ])
                        mkt_prob += _harville_probability([
                            r1["market_implied"], r2["market_implied"],
                            r3["market_implied"], r4["market_implied"],
                        ])

        if our_prob > 0:
            value = our_prob / mkt_prob if mkt_prob > 0 else 1.0
            num_combos = len(leg1) * len(leg2) * len(leg3) * len(leg4)
            # Subtract combos where same runner appears in multiple legs
            # (already handled by the skip logic above, num_combos is approximate)
            all_sc = [r["saddlecloth"] for r in leg1 + leg2 + leg3 + leg4]
            all_names = [r.get("horse_name", "") for r in leg1 + leg2 + leg3 + leg4]
            # Deduplicate for display
            seen = set()
            display_sc, display_names = [], []
            for sc, nm in zip(all_sc, all_names):
                if sc not in seen:
                    display_sc.append(sc)
                    display_names.append(nm)
                    seen.add(sc)

            if value >= VALUE_THRESHOLDS["First4"]:
                results.append(ExoticCombination(
                    exotic_type="First4",
                    runners=display_sc,
                    runner_names=display_names,
                    estimated_probability=round(our_prob, 6),
                    market_probability=round(mkt_prob, 6),
                    value_ratio=round(value, 3),
                    cost=stake,
                    num_combos=max(1, num_combos - len(seen)),  # approximate
                    format="legs",
                ))

    # --- First4 Box: RARE — only when genuine 5-way contention ---
    # Only generate when 5+ runners each have >12% win prob (strong field)
    strong_runners = [r for r in sorted_runners if r.get("win_prob", 0) >= 0.12]
    if len(strong_runners) >= 5:
        for combo in combinations(strong_runners[:5], 4):
            our_probs = [r["win_prob"] for r in combo]
            mkt_probs = [r["market_implied"] for r in combo]
            our_prob = _box_probability(our_probs, 4)
            mkt_prob = _box_probability(mkt_probs, 4)
            value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

            if value >= VALUE_THRESHOLDS["First4 Box"]:
                results.append(ExoticCombination(
                    exotic_type="First4 Box",
                    runners=[r["saddlecloth"] for r in combo],
                    runner_names=[r.get("horse_name", "") for r in combo],
                    estimated_probability=round(our_prob, 6),
                    market_probability=round(mkt_prob, 6),
                    value_ratio=round(value, 3),
                    cost=stake,
                    num_combos=24,
                    format="boxed",
                ))

    # Sort by value ratio descending, then probability
    results.sort(key=lambda x: (-x.value_ratio, -x.estimated_probability))

    return results[:12]


def calculate_sequence_leg_confidence(
    races_data: list[dict],
) -> list[SequenceLegAnalysis]:
    """Analyse each leg for sequence bet construction.

    Args:
        races_data: List of dicts, each with keys:
            - race_number (int)
            - runners: list of dicts with saddlecloth, horse_name, win_prob, value_rating

    Returns:
        List of SequenceLegAnalysis, one per race.
    """
    results = []

    for race in races_data:
        runners = sorted(
            race.get("runners", []),
            key=lambda r: r.get("win_prob", 0),
            reverse=True,
        )
        if not runners:
            continue

        top_prob = runners[0].get("win_prob", 0)
        second_prob = runners[1].get("win_prob", 0) if len(runners) > 1 else 0
        top2_combined = top_prob + second_prob

        # Determine confidence level
        if top_prob > 0.30 and (top_prob - second_prob) > 0.10:
            confidence = "HIGH"
            width = 1
        elif top2_combined > 0.45:
            confidence = "MED"
            width = 2
        else:
            confidence = "LOW"
            width = 3 if top_prob > 0.15 else 4

        # Include top runners (width + 1 for context)
        top_runners = []
        for r in runners[:width + 1]:
            top_runners.append({
                "saddlecloth": r.get("saddlecloth", 0),
                "horse_name": r.get("horse_name", ""),
                "win_prob": round(r.get("win_prob", 0), 4),
                "value_rating": round(r.get("value_rating", 1.0), 3),
            })

        results.append(SequenceLegAnalysis(
            race_number=race["race_number"],
            top_runners=top_runners,
            leg_confidence=confidence,
            suggested_width=width,
        ))

    return results


# ──────────────────────────────────────────────
# Deep Learning Pattern Loader (async)
# ──────────────────────────────────────────────

async def load_dl_patterns_for_probability(db) -> list[dict]:
    """Load deep learning patterns with structured conditions for probability matching.

    Returns all HIGH and MEDIUM confidence patterns with their full conditions
    dict so the probability model can match them against runner attributes.
    """
    from sqlalchemy import select
    from punty.memory.models import PatternInsight

    result = await db.execute(
        select(PatternInsight)
        .where(PatternInsight.pattern_type.like("deep_learning_%"))
    )
    rows = result.scalars().all()

    patterns = []
    for r in rows:
        try:
            conds = json.loads(r.conditions_json) if r.conditions_json else {}
        except (json.JSONDecodeError, TypeError):
            conds = {}
        confidence = conds.get("confidence", "LOW")
        if confidence not in ("HIGH", "MEDIUM"):
            continue
        patterns.append({
            "type": r.pattern_type,
            "key": r.pattern_key,
            "conditions": conds,
            "confidence": confidence,
            "sample_size": r.sample_count,
            "edge": r.avg_pnl,
        })

    # Populate module-level cache so sync callers can access patterns
    set_dl_pattern_cache(patterns)

    return patterns
