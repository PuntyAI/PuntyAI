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
import re
import statistics
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Optional

logger = logging.getLogger(__name__)

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
DEFAULT_WEIGHTS = {
    "market": 0.22,
    "movement": 0.07,
    "form": 0.15,
    "class_fitness": 0.05,
    "pace": 0.11,
    "barrier": 0.09,
    "jockey_trainer": 0.11,
    "weight_carried": 0.05,
    "horse_profile": 0.05,
    "deep_learning": 0.10,
}

# Legacy aliases for backward compatibility
WEIGHT_MARKET = DEFAULT_WEIGHTS["market"]
WEIGHT_FORM = DEFAULT_WEIGHTS["form"]
WEIGHT_PACE = DEFAULT_WEIGHTS["pace"]
WEIGHT_MOVEMENT = DEFAULT_WEIGHTS["movement"]
WEIGHT_CLASS = DEFAULT_WEIGHTS["class_fitness"]

# Baseline win rate for an average runner (1/field_size fallback)
DEFAULT_BASELINE = 0.10

# Value detection thresholds
VALUE_THRESHOLD = 1.05       # 5% edge minimum
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

    w = weights if weights else DEFAULT_WEIGHTS
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
        scores = {
            "market":         _market_consensus(runner, overround),
            "movement":       _market_movement_factor(runner),
            "form":           _form_rating(runner, track_condition, baseline),
            "class_fitness":  _class_factor(runner, baseline),
            "pace":           _pace_factor(runner, pace_scenario),
            "barrier":        _barrier_draw_factor(runner, field_size, race_distance),
            "jockey_trainer": _jockey_trainer_factor(runner, baseline),
            "weight_carried": _weight_factor(runner, avg_weight, race_distance),
            "horse_profile":  _horse_profile_factor(runner),
            "deep_learning":  _deep_learning_factor(
                runner, meeting, race_distance, track_condition,
                field_size, dl_patterns,
            ),
        }

        all_factor_scores[rid] = scores
        market_implied[rid] = scores["market"]
        odds = _get_median_odds(runner)
        runner_odds[rid] = odds or 0.0
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
        td_score = _stat_to_score(td.win_rate, baseline)
        signal_sum += td_score * 1.5  # higher weight for track+distance
        signals += 1.5

    # Condition-appropriate stats
    cond_stats_field = _condition_stats_field(track_condition)
    if cond_stats_field:
        cond = parse_stats_string(_get(runner, cond_stats_field))
        if cond and cond.starts >= 3:
            signal_sum += _stat_to_score(cond.win_rate, baseline)
            signals += 1

    # Distance stats (broader than track+distance)
    dist = parse_stats_string(_get(runner, "distance_stats"))
    if dist and dist.starts >= 3:
        signal_sum += _stat_to_score(dist.win_rate, baseline) * 0.8
        signals += 0.8

    # First-up / second-up stats
    days_since = _get(runner, "days_since_last_run")
    if days_since and days_since > 60:
        fu = parse_stats_string(_get(runner, "first_up_stats"))
        if fu and fu.starts >= 2:
            signal_sum += _stat_to_score(fu.win_rate, baseline) * 0.7
            signals += 0.7
    elif days_since and 21 <= days_since <= 60:
        su = parse_stats_string(_get(runner, "second_up_stats"))
        if su and su.starts >= 2:
            signal_sum += _stat_to_score(su.win_rate, baseline) * 0.5
            signals += 0.5

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


def _stat_to_score(win_rate: float, baseline: float) -> float:
    """Convert a win rate to a 0.0-1.0 score relative to baseline."""
    # If win_rate is double the baseline, score should be ~0.7
    # If win_rate is zero, score should be ~0.2
    if baseline <= 0:
        baseline = 0.10
    ratio = win_rate / baseline
    # Map ratio to score: 0→0.2, 1→0.5, 2→0.7, 3+→0.85
    score = 0.2 + 0.3 * min(ratio, 3.0) / 3.0 * 2.0
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

def _class_factor(runner: Any, baseline: float) -> float:
    """Score based on class suitability and fitness."""
    score = 0.5

    # Class stats
    cls = parse_stats_string(_get(runner, "class_stats"))
    if cls and cls.starts >= 2:
        ratio = cls.win_rate / baseline if baseline > 0 else cls.win_rate / 0.10
        adjustment = (ratio - 1.0) * 0.15
        score += max(-0.15, min(0.15, adjustment))

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

def _jockey_trainer_factor(runner: Any, baseline: float) -> float:
    """Score based on jockey and trainer win rate statistics.

    Jockey weighted 60%, trainer 40% — jockey has more direct race impact.
    """
    score = 0.5
    signals = 0
    signal_sum = 0.0

    # Jockey stats (60% weight)
    jockey = parse_stats_string(_get(runner, "jockey_stats"))
    if jockey and jockey.starts >= 5:
        j_score = _stat_to_score(jockey.win_rate, baseline)
        signal_sum += j_score * 0.6
        signals += 0.6

    # Trainer stats (40% weight)
    trainer = parse_stats_string(_get(runner, "trainer_stats"))
    if trainer and trainer.starts >= 5:
        t_score = _stat_to_score(trainer.win_rate, baseline)
        signal_sum += t_score * 0.4
        signals += 0.4

    if signals > 0:
        score = signal_sum / signals

    return max(0.05, min(0.95, score))


# ──────────────────────────────────────────────
# Factor: Weight Carried
# ──────────────────────────────────────────────

def _weight_factor(runner: Any, avg_weight: float, race_distance: int = 1400) -> float:
    """Score based on carried weight relative to race average.

    Lighter weight = advantage. Impact scales with distance:
    sprints (<1200m) weight matters most, stayers (>2000m) less so.
    """
    score = 0.5
    weight = _get(runner, "weight")
    if not weight or not isinstance(weight, (int, float)) or weight <= 0:
        return score
    if avg_weight <= 0:
        return score

    diff = weight - avg_weight  # positive = heavier than average

    # Distance multiplier: sprints amplify weight impact, staying races diminish it
    # 1000m → 1.3x, 1200m → 1.15x, 1400m → 1.0x, 1600m → 0.9x, 2000m → 0.75x, 2400m → 0.65x
    dist = max(800, min(3200, race_distance))
    distance_mult = 1.0 + (1400 - dist) * 0.0005

    # Map weight difference to score adjustment
    # ±3kg from average → ±0.10 score change (at 1400m baseline)
    adjustment = -(diff / 3.0) * 0.10 * distance_mult
    score += max(-0.20, min(0.20, adjustment))

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

def _horse_profile_factor(runner: Any) -> float:
    """Score based on horse age and sex.

    Peak racing age is 4-5yo for gallopers. Young (2yo) and older (8+)
    horses get a mild penalty.
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

    sex = _get(runner, "horse_sex")
    if sex and isinstance(sex, str):
        sex_lower = sex.lower().strip()
        # Geldings are slightly more consistent runners
        if sex_lower in ("gelding", "g"):
            score += 0.01
        # Mares can be inconsistent in certain conditions
        elif sex_lower in ("mare", "m", "filly", "f"):
            score -= 0.01

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
    max_boost = 0.28  # Market weight can increase from 22% up to ~50%
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
    Uses 70/30 blend (model-heavy) to preserve model signal.
    Re-normalizes after adjustments to maintain probabilities summing to 1.0.
    """
    adjusted = dict(win_probs)
    needs_renorm = False

    for rid in adjusted:
        mkt = market_implied.get(rid, 0)
        if mkt > 0.05:  # Only for runners with real market presence
            ratio = adjusted[rid] / mkt if mkt > 0 else 1.0
            if ratio < 0.35:
                # Blend 70/30 model/market to preserve model signal
                adjusted[rid] = adjusted[rid] * 0.7 + mkt * 0.3
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

        if matched:
            # Cap individual pattern contribution
            capped_edge = max(-0.15, min(0.15, edge))
            total_adjustment += capped_edge * conf_mult

    # Cap total adjustment to prevent extreme swings
    total_adjustment = max(-0.25, min(0.25, total_adjustment))
    score += total_adjustment

    return max(0.05, min(0.95, score))


def _get_dist_bucket(distance: int) -> str:
    """Map distance to bucket matching deep learning patterns."""
    if distance <= 1100:
        return "sprint"
    elif distance <= 1300:
        return "short"
    elif distance <= 1800:
        return "middle"
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
    """
    if not stats_str or not isinstance(stats_str, str):
        return None

    stats_str = stats_str.strip()
    if not stats_str:
        return None

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
        Top 10 value combinations sorted by value_ratio descending,
        filtered to value_ratio >= 1.2.
    """
    if len(runners_data) < 2:
        return []

    # Sort by our win probability descending
    sorted_runners = sorted(
        runners_data,
        key=lambda r: r.get("win_prob", 0),
        reverse=True,
    )

    # Only consider top N runners (keep it computationally manageable)
    top_n = sorted_runners[:min(6, len(sorted_runners))]

    results: list[ExoticCombination] = []

    # --- Quinella: pairs from top 4 ---
    for combo in combinations(top_n[:4], 2):
        our_prob = _quinella_probability(
            combo[0]["win_prob"], combo[1]["win_prob"]
        )
        mkt_prob = _quinella_probability(
            combo[0]["market_implied"], combo[1]["market_implied"]
        )
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

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
    for combo in permutations(top_n[:4], 2):
        our_prob = _harville_probability([r["win_prob"] for r in combo])
        mkt_prob = _harville_probability([r["market_implied"] for r in combo])
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

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

    # --- Trifecta Box: 3 runners from top 5 ---
    for combo in combinations(top_n[:5], 3):
        our_probs = [r["win_prob"] for r in combo]
        mkt_probs = [r["market_implied"] for r in combo]
        our_prob = _box_probability(our_probs, 3)
        mkt_prob = _box_probability(mkt_probs, 3)
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

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

    # --- Trifecta Box: 4 runners from top 5 ---
    for combo in combinations(top_n[:5], 4):
        our_probs = [r["win_prob"] for r in combo]
        mkt_probs = [r["market_implied"] for r in combo]
        our_prob = _box_probability(our_probs, 3)
        mkt_prob = _box_probability(mkt_probs, 3)
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

        results.append(ExoticCombination(
            exotic_type="Trifecta Box",
            runners=[r["saddlecloth"] for r in combo],
            runner_names=[r.get("horse_name", "") for r in combo],
            estimated_probability=round(our_prob, 6),
            market_probability=round(mkt_prob, 6),
            value_ratio=round(value, 3),
            cost=stake,
            num_combos=24,
            format="boxed",
        ))

    # --- First4 Box: 4 runners from top 6 ---
    for combo in combinations(top_n[:6], 4):
        our_probs = [r["win_prob"] for r in combo]
        mkt_probs = [r["market_implied"] for r in combo]
        our_prob = _box_probability(our_probs, 4)
        mkt_prob = _box_probability(mkt_probs, 4)
        value = our_prob / mkt_prob if mkt_prob > 0 else 1.0

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

    # Filter to value combinations only and sort by value
    results = [r for r in results if r.value_ratio >= 1.2]
    results.sort(key=lambda x: (-x.value_ratio, -x.estimated_probability))

    return results[:10]


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
    return patterns
