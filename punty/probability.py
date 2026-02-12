"""Probability calculation engine for horse racing picks.

Calculates win/place probability, value detection, and recommended stakes
for each runner in a race using multi-factor analysis across 9 factors
grouped into 5 categories:
  - Market Intelligence: Market consensus + market movement
  - Form & Fitness: Form rating + class/fitness
  - Race Dynamics: Pace factor + barrier draw
  - Connections: Jockey & trainer stats
  - Physical: Weight carried + horse profile
"""

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
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
}

# Default weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "market": 0.25,
    "movement": 0.08,
    "form": 0.18,
    "class_fitness": 0.05,
    "pace": 0.12,
    "barrier": 0.10,
    "jockey_trainer": 0.12,
    "weight_carried": 0.05,
    "horse_profile": 0.05,
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
) -> dict[str, "RunnerProbability"]:
    """Calculate probabilities for all active runners in a race.

    Args:
        runners: List of Runner ORM objects (or dicts with runner fields)
        race: Race ORM object (or dict)
        meeting: Meeting ORM object (or dict)
        pool: Total stake pool per race (default $20)
        weights: Custom factor weights (key → 0.0-1.0). Defaults to DEFAULT_WEIGHTS.

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

    # Step 1: Calculate raw scores for each runner
    raw_scores: dict[str, float] = {}
    market_implied: dict[str, float] = {}
    runner_odds: dict[str, float] = {}
    factor_details: dict[str, dict] = {}

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
            "weight_carried": _weight_factor(runner, avg_weight),
            "horse_profile":  _horse_profile_factor(runner),
        }

        market_implied[rid] = scores["market"]
        odds = _get_median_odds(runner)
        runner_odds[rid] = odds or 0.0

        # Composite raw score — weighted sum of all factors
        raw = sum(w.get(k, 0.0) * v for k, v in scores.items())

        raw_scores[rid] = max(0.001, raw)  # floor to prevent zero
        factor_details[rid] = {k: round(v, 4) for k, v in scores.items()}

    # Step 2: Normalize to sum to 1.0
    total = sum(raw_scores.values())
    results: dict[str, RunnerProbability] = {}

    for runner in active:
        rid = _get(runner, "id", "")
        win_prob = raw_scores[rid] / total if total > 0 else baseline

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
    if td and td.starts >= 2:
        td_score = _stat_to_score(td.win_rate, baseline)
        signal_sum += td_score * 1.5  # higher weight for track+distance
        signals += 1.5

    # Condition-appropriate stats
    cond_stats_field = _condition_stats_field(track_condition)
    if cond_stats_field:
        cond = parse_stats_string(_get(runner, cond_stats_field))
        if cond and cond.starts >= 2:
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

def _weight_factor(runner: Any, avg_weight: float) -> float:
    """Score based on carried weight relative to race average.

    Lighter weight = advantage (approx 0.5kg per length at 1600m).
    """
    score = 0.5
    weight = _get(runner, "weight")
    if not weight or not isinstance(weight, (int, float)) or weight <= 0:
        return score
    if avg_weight <= 0:
        return score

    diff = weight - avg_weight  # positive = heavier than average

    # Map weight difference to score adjustment
    # ±3kg from average → ±0.10 score change
    adjustment = -(diff / 3.0) * 0.10
    score += max(-0.15, min(0.15, adjustment))

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
