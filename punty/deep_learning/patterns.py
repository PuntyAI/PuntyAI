"""Pattern analysis engine for Proform historical data.

Runs statistical analyses against the deep_learning.db to discover
significant betting patterns. Results written as PatternInsight rows
to the main production database.

15 core analyses covering:
- Track/distance/condition biases
- Barrier draw advantages
- Pace/running style patterns
- Form cycles and spell recovery
- Jockey/trainer combinations
- Class transitions (drop/rise)
- Condition specialists
- Seasonal patterns
- Market efficiency
- Sectional acceleration profiles
- Track bias (rail vs wide)
- Pace collapse detection
- Coming into form (improving trend)
- Class droppers (grade reduction advantage)
- Race speed quality (next-start performance)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select, text, and_, case

from .models import (
    HistoricalRace,
    HistoricalRunner,
    HistoricalSectional,
    get_session,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES = 20
SIGNIFICANCE_LEVEL = 0.05


@dataclass
class Pattern:
    """A discovered betting pattern with statistical backing."""

    pattern_type: str  # e.g. "track_distance_condition"
    dimension: str  # e.g. "Flemington_1600_Good"
    description: str  # Human-readable description
    sample_size: int
    win_rate: float  # Observed win rate for the filtered group
    base_rate: float  # Expected win rate (overall or market-implied)
    edge: float  # win_rate - base_rate
    p_value: float  # Statistical significance
    confidence: str  # HIGH/MEDIUM/LOW
    metadata: dict = field(default_factory=dict)


def _binomial_p_value(successes: int, trials: int, expected_rate: float) -> float:
    """Approximate binomial test p-value using normal approximation.

    H0: true rate = expected_rate
    H1: true rate != expected_rate (two-sided)
    """
    if trials == 0 or expected_rate <= 0 or expected_rate >= 1:
        return 1.0
    observed_rate = successes / trials
    std_err = math.sqrt(expected_rate * (1 - expected_rate) / trials)
    if std_err == 0:
        return 1.0
    z = abs(observed_rate - expected_rate) / std_err
    # Normal CDF approximation (Abramowitz & Stegun)
    p = math.erfc(z / math.sqrt(2))
    return p


def _confidence_label(p_value: float, sample_size: int) -> str:
    if p_value < 0.01 and sample_size >= 50:
        return "HIGH"
    elif p_value < 0.05 and sample_size >= 30:
        return "MEDIUM"
    return "LOW"


def _distance_bucket(distance: int | None) -> str:
    """Bucket distances: sprint/short/middle/staying/extreme."""
    if not distance:
        return "unknown"
    if distance < 1200:
        return "sprint"
    elif distance < 1400:
        return "short"
    elif distance < 1800:
        return "middle"
    elif distance < 2400:
        return "staying"
    return "extreme"


def _barrier_bucket(barrier: int | None, field_size: int | None) -> str:
    """Bucket barriers relative to field size."""
    if not barrier or not field_size or field_size == 0:
        return "unknown"
    ratio = barrier / field_size
    if ratio <= 0.33:
        return "inside"
    elif ratio <= 0.66:
        return "middle"
    return "outside"


def _condition_group(condition: str | None) -> str:
    """Group track conditions into Firm/Good/Soft/Heavy."""
    if not condition:
        return "unknown"
    c = condition.upper().strip()
    if "SYN" in c:
        return "Synthetic"
    elif c.startswith("F") or c in ("1", "2"):
        return "Firm"
    elif c.startswith("G") or c in ("3", "4"):
        return "Good"
    elif c.startswith("S") or c in ("5", "6"):
        return "Soft"
    elif c.startswith("H") or c in ("7", "8", "9", "10"):
        return "Heavy"
    return "unknown"


# ==============================================================
# Pattern Analyses
# ==============================================================


def analyse_track_distance_condition(session) -> list[Pattern]:
    """Pattern 1: Win rates by venue + distance bucket + condition group.

    Answers: "Do leaders win more at Flemington 1600m Good?"
    """
    patterns = []

    # Get overall win rate
    total = session.query(HistoricalRunner).filter(
        HistoricalRunner.finish_position.isnot(None)
    ).count()
    total_wins = session.query(HistoricalRunner).filter(
        HistoricalRunner.won == True
    ).count()
    if total == 0:
        return patterns
    overall_win_rate = total_wins / total

    # Query grouped stats
    rows = session.execute(text("""
        SELECT r.venue, r.distance, r.track_condition,
               COUNT(*) as runners,
               SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
        GROUP BY r.venue, r.distance, r.track_condition
        HAVING COUNT(*) >= :min_samples
    """), {"min_samples": MIN_SAMPLES}).fetchall()

    for venue, distance, condition, count, wins in rows:
        if not venue or not distance:
            continue
        dist_bucket = _distance_bucket(distance)
        cond_group = _condition_group(condition)
        win_rate = wins / count
        edge = win_rate - overall_win_rate
        p_val = _binomial_p_value(wins, count, overall_win_rate)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.02:
            dim = f"{venue}_{dist_bucket}_{cond_group}"
            direction = "outperform" if edge > 0 else "underperform"
            patterns.append(Pattern(
                pattern_type="deep_learning_track_dist_cond",
                dimension=dim,
                description=(
                    f"At {venue} {distance}m on {cond_group}, "
                    f"winners {direction} by {abs(edge)*100:.1f}% "
                    f"(win rate {win_rate*100:.1f}% vs {overall_win_rate*100:.1f}% base)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=overall_win_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={"venue": venue, "distance": distance,
                          "condition": condition, "dist_bucket": dist_bucket},
            ))

    return patterns


def analyse_barrier_bias(session) -> list[Pattern]:
    """Pattern 2: Barrier draw advantage by venue + distance bucket."""
    patterns = []

    rows = session.execute(text("""
        SELECT r.venue, r.distance, hr.barrier, r.field_size,
               COUNT(*) as runners,
               SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.barrier IS NOT NULL
          AND r.field_size >= 6
        GROUP BY r.venue, r.distance, hr.barrier
        HAVING COUNT(*) >= :min_samples
    """), {"min_samples": MIN_SAMPLES}).fetchall()

    # Aggregate by venue + distance + barrier bucket
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    for venue, distance, barrier, field_size, count, wins in rows:
        bucket = _barrier_bucket(barrier, field_size)
        dist_bucket = _distance_bucket(distance)
        key = f"{venue}_{dist_bucket}_{bucket}"
        agg[key]["count"] += count
        agg[key]["wins"] += wins
        agg[key]["venue"] = venue
        agg[key]["dist_bucket"] = dist_bucket
        agg[key]["barrier_bucket"] = bucket

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        # Expected: 1/field_size average (approx 10% for 10-horse fields)
        expected_rate = 0.10
        win_rate = wins / count
        edge = win_rate - expected_rate
        p_val = _binomial_p_value(wins, count, expected_rate)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.02:
            patterns.append(Pattern(
                pattern_type="deep_learning_barrier_bias",
                dimension=key,
                description=(
                    f"{data['barrier_bucket'].title()} barriers at "
                    f"{data['venue']} {data['dist_bucket']} "
                    f"win at {win_rate*100:.1f}% ({'+' if edge > 0 else ''}{edge*100:.1f}%)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=expected_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata=data,
            ))

    return patterns


def analyse_pace_patterns(session) -> list[Pattern]:
    """Pattern 3: Running style win rates by venue + distance + condition.

    Uses settle_pos to classify: leader (1), on_pace (2-3), midfield (4-6), back (7+).
    """
    patterns = []

    rows = session.execute(text("""
        SELECT r.venue, r.distance, r.track_condition,
               hr.settle_pos, r.field_size,
               COUNT(*) as runners,
               SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.settle_pos IS NOT NULL
          AND r.field_size >= 6
        GROUP BY r.venue, r.distance, r.track_condition, hr.settle_pos
    """)).fetchall()

    # Classify and aggregate
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    for venue, distance, condition, settle, field_size, count, wins in rows:
        if not settle or not field_size:
            continue
        if settle == 1:
            style = "leader"
        elif settle <= 3:
            style = "on_pace"
        elif settle <= max(6, field_size * 0.5):
            style = "midfield"
        else:
            style = "backmarker"
        dist_bucket = _distance_bucket(distance)
        cond_group = _condition_group(condition)
        key = f"{venue}_{dist_bucket}_{cond_group}_{style}"
        agg[key]["count"] += count
        agg[key]["wins"] += wins
        agg[key]["venue"] = venue
        agg[key]["dist_bucket"] = dist_bucket
        agg[key]["condition"] = cond_group
        agg[key]["style"] = style

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        expected_rate = 0.10
        win_rate = wins / count
        edge = win_rate - expected_rate
        p_val = _binomial_p_value(wins, count, expected_rate)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.03:
            patterns.append(Pattern(
                pattern_type="deep_learning_pace",
                dimension=key,
                description=(
                    f"{data['style'].title()}s at {data['venue']} "
                    f"{data['dist_bucket']} on {data['condition']} "
                    f"win at {win_rate*100:.1f}%"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=expected_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata=data,
            ))

    return patterns


def analyse_form_cycles(session) -> list[Pattern]:
    """Pattern 4: First-up, second-up, third-up win rates by prep stage."""
    patterns = []

    rows = session.execute(text("""
        SELECT hr.prep_runs,
               COUNT(*) as runners,
               SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN hr.placed = 1 THEN 1 ELSE 0 END) as places
        FROM historical_runners hr
        WHERE hr.finish_position IS NOT NULL
          AND hr.prep_runs IS NOT NULL
        GROUP BY hr.prep_runs
        HAVING COUNT(*) >= :min_samples
    """), {"min_samples": MIN_SAMPLES}).fetchall()

    total = sum(r[1] for r in rows) or 1
    total_wins = sum(r[2] for r in rows)
    base_rate = total_wins / total if total else 0.10

    for prep, count, wins, places in rows:
        win_rate = wins / count
        place_rate = places / count
        edge = win_rate - base_rate
        p_val = _binomial_p_value(wins, count, base_rate)

        label = {0: "First-up", 1: "Second-up", 2: "Third-up"}.get(
            prep, f"Run {prep+1} of prep"
        )

        if p_val < SIGNIFICANCE_LEVEL:
            patterns.append(Pattern(
                pattern_type="deep_learning_form_cycle",
                dimension=f"prep_{prep}",
                description=(
                    f"{label}: win {win_rate*100:.1f}%, "
                    f"place {place_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% vs base)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=base_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={"prep_runs": prep, "place_rate": place_rate},
            ))

    return patterns


def analyse_jockey_trainer_combos(session) -> list[Pattern]:
    """Pattern 5: Jockey/trainer combination win rates."""
    patterns = []

    rows = session.execute(text("""
        SELECT hr.jockey, hr.trainer, r.state,
               COUNT(*) as runners,
               SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins,
               AVG(hr.starting_price) as avg_sp
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.jockey IS NOT NULL
          AND hr.trainer IS NOT NULL
        GROUP BY hr.jockey, hr.trainer, r.state
        HAVING COUNT(*) >= :min_samples
    """), {"min_samples": MIN_SAMPLES}).fetchall()

    for jockey, trainer, state, count, wins, avg_sp in rows:
        # Expected based on average SP
        expected_rate = 1.0 / (avg_sp or 10)
        win_rate = wins / count
        edge = win_rate - expected_rate
        p_val = _binomial_p_value(wins, count, min(expected_rate, 0.5))

        if p_val < SIGNIFICANCE_LEVEL and edge > 0.03:
            patterns.append(Pattern(
                pattern_type="deep_learning_jockey_trainer",
                dimension=f"{jockey}_{trainer}_{state or 'ALL'}",
                description=(
                    f"{jockey} / {trainer} ({state}): "
                    f"{win_rate*100:.1f}% SR from {count} rides "
                    f"(market expected {expected_rate*100:.1f}%)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=expected_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={"jockey": jockey, "trainer": trainer,
                          "state": state, "avg_sp": avg_sp},
            ))

    return patterns


def analyse_class_transitions(session) -> list[Pattern]:
    """Pattern 6: Class drop/rise performance.

    Compares current race class to previous race class from form history.
    """
    patterns = []

    runners = session.execute(text("""
        SELECT hr.id, hr.form_history, hr.won, hr.placed,
               hr.starting_price, hr.finish_position,
               r.race_class, r.prize_money
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.form_history IS NOT NULL
    """)).fetchall()

    class_data: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "places": 0})

    for runner_id, form_json, won, placed, sp, pos, current_class, current_prize in runners:
        try:
            history = json.loads(form_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not history:
            continue

        # Compare prize money as proxy for class level
        last_prize = history[0].get("prize_money") if isinstance(history[0], dict) else None
        if not current_prize or not last_prize:
            continue

        if current_prize < last_prize * 0.7:
            transition = "class_drop"
        elif current_prize > last_prize * 1.3:
            transition = "class_rise"
        else:
            transition = "same_class"

        class_data[transition]["count"] += 1
        class_data[transition]["wins"] += 1 if won else 0
        class_data[transition]["places"] += 1 if placed else 0

    total = sum(d["count"] for d in class_data.values()) or 1
    total_wins = sum(d["wins"] for d in class_data.values())
    base_rate = total_wins / total

    for transition, data in class_data.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        win_rate = wins / count
        place_rate = data["places"] / count
        edge = win_rate - base_rate
        p_val = _binomial_p_value(wins, count, base_rate)

        if p_val < SIGNIFICANCE_LEVEL:
            label = {
                "class_drop": "Dropping in class (>30% prize drop)",
                "class_rise": "Rising in class (>30% prize rise)",
                "same_class": "Same class level",
            }[transition]
            patterns.append(Pattern(
                pattern_type="deep_learning_class_transition",
                dimension=transition,
                description=(
                    f"{label}: win {win_rate*100:.1f}%, place {place_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% edge)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=base_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={"transition": transition, "place_rate": place_rate},
            ))

    return patterns


def analyse_condition_specialists(session) -> list[Pattern]:
    """Pattern 7: Horses with strong condition records outperforming."""
    patterns = []

    # For each condition type, compare horses with good records vs poor records
    for cond_field, cond_label in [
        ("good_record", "Good"), ("soft_record", "Soft"),
        ("heavy_record", "Heavy"), ("firm_record", "Firm"),
    ]:
        rows = session.execute(text(f"""
            SELECT hr.{cond_field}, hr.won, hr.placed, hr.starting_price
            FROM historical_runners hr
            JOIN historical_races r ON hr.race_fk = r.id
            WHERE hr.finish_position IS NOT NULL
              AND hr.{cond_field} IS NOT NULL
              AND r.track_condition LIKE :cond_prefix
        """), {"cond_prefix": f"{cond_label[0]}%"}).fetchall()

        specialists = {"count": 0, "wins": 0}
        non_specialists = {"count": 0, "wins": 0}

        for record_json, won, placed, sp in rows:
            try:
                record = json.loads(record_json)
            except (json.JSONDecodeError, TypeError):
                continue
            starts = record.get("Starts", 0)
            firsts = record.get("Firsts", 0)
            if starts < 3:
                continue  # Not enough data

            cond_sr = firsts / starts if starts else 0
            if cond_sr >= 0.25:
                specialists["count"] += 1
                specialists["wins"] += 1 if won else 0
            else:
                non_specialists["count"] += 1
                non_specialists["wins"] += 1 if won else 0

        if specialists["count"] < MIN_SAMPLES:
            continue
        spec_wr = specialists["wins"] / specialists["count"]
        non_wr = (
            non_specialists["wins"] / non_specialists["count"]
            if non_specialists["count"] else 0.10
        )
        edge = spec_wr - non_wr
        p_val = _binomial_p_value(
            specialists["wins"], specialists["count"], non_wr or 0.10
        )

        if p_val < SIGNIFICANCE_LEVEL and edge > 0:
            patterns.append(Pattern(
                pattern_type="deep_learning_condition_specialist",
                dimension=f"{cond_label}_specialist",
                description=(
                    f"{cond_label} track specialists (25%+ SR on {cond_label}): "
                    f"win {spec_wr*100:.1f}% vs {non_wr*100:.1f}% non-specialists "
                    f"(+{edge*100:.1f}% edge)"
                ),
                sample_size=specialists["count"],
                win_rate=spec_wr,
                base_rate=non_wr,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, specialists["count"]),
                metadata={"condition": cond_label,
                          "specialist_count": specialists["count"],
                          "non_specialist_count": non_specialists["count"]},
            ))

    return patterns


def analyse_seasonal_patterns(session) -> list[Pattern]:
    """Pattern 8: Seasonal win rate variations by state/month."""
    patterns = []

    rows = session.execute(text("""
        SELECT r.state,
               CAST(strftime('%m', r.meeting_date) AS INTEGER) as month,
               COUNT(*) as runners,
               SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND r.state IS NOT NULL
        GROUP BY r.state, month
        HAVING COUNT(*) >= :min_samples
    """), {"min_samples": MIN_SAMPLES}).fetchall()

    # Calculate state-level base rates
    state_totals: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    for state, month, count, wins in rows:
        state_totals[state]["count"] += count
        state_totals[state]["wins"] += wins

    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    for state, month, count, wins in rows:
        st = state_totals[state]
        base_rate = st["wins"] / st["count"] if st["count"] else 0.10
        win_rate = wins / count
        edge = win_rate - base_rate
        p_val = _binomial_p_value(wins, count, base_rate)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.02:
            patterns.append(Pattern(
                pattern_type="deep_learning_seasonal",
                dimension=f"{state}_{month_names[month]}",
                description=(
                    f"{state} in {month_names[month]}: "
                    f"favourites win at {win_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% vs state avg)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=base_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={"state": state, "month": month},
            ))

    return patterns


def analyse_market_efficiency(session) -> list[Pattern]:
    """Pattern 9: Market efficiency — SP ranges where actual win rate
    differs significantly from market-implied probability."""
    patterns = []

    sp_buckets = [
        (1.0, 3.0, "$1-$3"),
        (3.0, 5.0, "$3-$5"),
        (5.0, 8.0, "$5-$8"),
        (8.0, 12.0, "$8-$12"),
        (12.0, 20.0, "$12-$20"),
        (20.0, 50.0, "$20-$50"),
    ]

    for sp_min, sp_max, label in sp_buckets:
        rows = session.execute(text("""
            SELECT r.state, r.location_type,
                   COUNT(*) as runners,
                   SUM(CASE WHEN hr.won = 1 THEN 1 ELSE 0 END) as wins,
                   AVG(hr.starting_price) as avg_sp
            FROM historical_runners hr
            JOIN historical_races r ON hr.race_fk = r.id
            WHERE hr.finish_position IS NOT NULL
              AND hr.starting_price >= :sp_min
              AND hr.starting_price < :sp_max
            GROUP BY r.state, r.location_type
            HAVING COUNT(*) >= :min_samples
        """), {"sp_min": sp_min, "sp_max": sp_max,
               "min_samples": MIN_SAMPLES}).fetchall()

        for state, loc_type, count, wins, avg_sp in rows:
            # Market-implied probability
            market_prob = 1.0 / avg_sp if avg_sp else 0.10
            win_rate = wins / count
            edge = win_rate - market_prob
            p_val = _binomial_p_value(wins, count, market_prob)

            if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.02:
                loc_label = {"M": "Metro", "P": "Provincial", "C": "Country"}.get(
                    loc_type, loc_type or "All"
                )
                patterns.append(Pattern(
                    pattern_type="deep_learning_market",
                    dimension=f"{state}_{loc_label}_{label}",
                    description=(
                        f"{label} runners at {state} {loc_label}: "
                        f"actual {win_rate*100:.1f}% vs market {market_prob*100:.1f}% "
                        f"({'+' if edge > 0 else ''}{edge*100:.1f}%)"
                    ),
                    sample_size=count,
                    win_rate=win_rate,
                    base_rate=market_prob,
                    edge=edge,
                    p_value=p_val,
                    confidence=_confidence_label(p_val, count),
                    metadata={"state": state, "location": loc_type,
                              "sp_range": label, "avg_sp": avg_sp},
                ))

    return patterns


def analyse_sectional_acceleration(session) -> list[Pattern]:
    """Pattern 10: Horses that gain positions in the last 400m.

    Measures position change from pos_800 to pos_fin. Significant movers
    who finish 3+ positions better than their 800m position.
    """
    patterns = []

    rows = session.execute(text("""
        SELECT r.venue, r.distance, r.track_condition,
               hs.pos_800, hs.pos_fin, r.field_size,
               CASE WHEN hr.won = 1 THEN 1 ELSE 0 END as won
        FROM historical_sectionals hs
        JOIN historical_races r ON hs.race_fk = r.id
        JOIN historical_runners hr ON hr.form_id = hs.form_id AND hr.race_id = r.race_id
        WHERE hs.pos_800 IS NOT NULL
          AND hs.pos_fin IS NOT NULL
          AND r.field_size >= 6
    """)).fetchall()

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    for venue, distance, condition, pos800, pos_fin, field_size, won in rows:
        gain = pos800 - pos_fin  # Positive = improved (moved forward)
        if gain >= 3:
            move_type = "big_mover"
        elif gain >= 1:
            move_type = "improver"
        elif gain <= -3:
            move_type = "fader"
        else:
            move_type = "steady"

        dist_bucket = _distance_bucket(distance)
        cond_group = _condition_group(condition)
        key = f"{venue}_{dist_bucket}_{cond_group}_{move_type}"
        agg[key]["count"] += 1
        agg[key]["wins"] += won
        agg[key]["venue"] = venue
        agg[key]["dist_bucket"] = dist_bucket
        agg[key]["condition"] = cond_group
        agg[key]["move_type"] = move_type

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        expected = 0.10
        win_rate = wins / count
        edge = win_rate - expected
        p_val = _binomial_p_value(wins, count, expected)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.03:
            patterns.append(Pattern(
                pattern_type="deep_learning_acceleration",
                dimension=key,
                description=(
                    f"{data['move_type'].replace('_', ' ').title()} "
                    f"at {data['venue']} {data['dist_bucket']} "
                    f"on {data['condition']}: "
                    f"win rate {win_rate*100:.1f}%"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=expected,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata=data,
            ))

    return patterns


def analyse_track_bias_wides(session) -> list[Pattern]:
    """Pattern 11: Rail runners vs wide runners.

    Uses wides_400 to classify: rail (<=4), middle (5-6), wide (7+).
    """
    patterns = []

    rows = session.execute(text("""
        SELECT r.venue, r.distance,
               hs.wides_400,
               CASE WHEN hr.won = 1 THEN 1 ELSE 0 END as won
        FROM historical_sectionals hs
        JOIN historical_races r ON hs.race_fk = r.id
        JOIN historical_runners hr ON hr.form_id = hs.form_id AND hr.race_id = r.race_id
        WHERE hs.wides_400 IS NOT NULL
          AND hs.wides_400 > 0
    """)).fetchall()

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})
    for venue, distance, wides, won in rows:
        if wides <= 4:
            width = "rail"
        elif wides <= 6:
            width = "middle"
        else:
            width = "wide"

        dist_bucket = _distance_bucket(distance)
        key = f"{venue}_{dist_bucket}_{width}"
        agg[key]["count"] += 1
        agg[key]["wins"] += won
        agg[key]["venue"] = venue
        agg[key]["dist_bucket"] = dist_bucket
        agg[key]["width"] = width

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        expected = 0.10
        win_rate = wins / count
        edge = win_rate - expected
        p_val = _binomial_p_value(wins, count, expected)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.03:
            patterns.append(Pattern(
                pattern_type="deep_learning_track_bias",
                dimension=key,
                description=(
                    f"{data['width'].title()} runners at {data['venue']} "
                    f"{data['dist_bucket']}: "
                    f"win rate {win_rate*100:.1f}%"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=expected,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata=data,
            ))

    return patterns


def analyse_pace_collapse(session) -> list[Pattern]:
    """Pattern 12: Leaders that fade — early pace vs sustainability.

    When leaders run fast early (top-3 split to 600m), do they hold on?
    """
    patterns = []

    rows = session.execute(text("""
        SELECT r.venue, r.distance, r.track_condition,
               hs.pos_800, hs.pos_fin, hs.last_600,
               r.last_600 as race_last_600,
               CASE WHEN hr.won = 1 THEN 1 ELSE 0 END as won
        FROM historical_sectionals hs
        JOIN historical_races r ON hs.race_fk = r.id
        JOIN historical_runners hr ON hr.form_id = hs.form_id AND hr.race_id = r.race_id
        WHERE hs.pos_800 IS NOT NULL AND hs.pos_800 <= 2
          AND hs.pos_fin IS NOT NULL
          AND hs.last_600 IS NOT NULL
          AND r.last_600 IS NOT NULL
    """)).fetchall()

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "faded": 0})
    for venue, distance, condition, pos800, pos_fin, runner_l600, race_l600, won in rows:
        dist_bucket = _distance_bucket(distance)
        cond_group = _condition_group(condition)
        # Runner ran slower last 600 than race average = faded
        faded = pos_fin > pos800 + 2
        key = f"{venue}_{dist_bucket}_{cond_group}_leader"
        agg[key]["count"] += 1
        agg[key]["wins"] += won
        agg[key]["faded"] += 1 if faded else 0
        agg[key]["venue"] = venue
        agg[key]["dist_bucket"] = dist_bucket
        agg[key]["condition"] = cond_group

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        faded = data["faded"]
        if count < MIN_SAMPLES:
            continue
        win_rate = wins / count
        fade_rate = faded / count
        expected = 0.20  # Leaders expected to win ~20%
        edge = win_rate - expected
        p_val = _binomial_p_value(wins, count, expected)

        if p_val < SIGNIFICANCE_LEVEL:
            patterns.append(Pattern(
                pattern_type="deep_learning_pace_collapse",
                dimension=key,
                description=(
                    f"Leaders at {data['venue']} {data['dist_bucket']} "
                    f"on {data['condition']}: "
                    f"win {win_rate*100:.1f}%, fade {fade_rate*100:.1f}%"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=expected,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={**data, "fade_rate": fade_rate},
            ))

    return patterns


def analyse_coming_into_form(session) -> list[Pattern]:
    """Pattern 13: Horses showing improving form trend.

    Looks at last 3 positions from form_history — if positions are
    trending downward (improving), flags as 'coming into form'.
    """
    patterns = []

    runners = session.execute(text("""
        SELECT hr.form_history, hr.won, hr.placed, hr.starting_price,
               r.state, r.location_type
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.form_history IS NOT NULL
    """)).fetchall()

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "places": 0})

    for form_json, won, placed, sp, state, loc in runners:
        try:
            history = json.loads(form_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if len(history) < 3:
            continue

        # Get last 3 finishing positions (most recent first)
        positions = []
        for h in history[:3]:
            pos = h.get("pos") if isinstance(h, dict) else None
            if pos and isinstance(pos, (int, float)) and pos > 0:
                positions.append(int(pos))

        if len(positions) < 3:
            continue

        # Check if improving: each run better than previous
        # positions[0] is most recent, positions[2] is oldest
        improving = positions[0] < positions[1] < positions[2]
        declining = positions[0] > positions[1] > positions[2]

        if improving:
            trend = "improving"
        elif declining:
            trend = "declining"
        else:
            trend = "mixed"

        key = f"{trend}_{state or 'ALL'}_{loc or 'ALL'}"
        agg[key]["count"] += 1
        agg[key]["wins"] += 1 if won else 0
        agg[key]["places"] += 1 if placed else 0
        agg[key]["trend"] = trend
        agg[key]["state"] = state
        agg[key]["location"] = loc

    total = sum(d["count"] for d in agg.values()) or 1
    total_wins = sum(d["wins"] for d in agg.values())
    base_rate = total_wins / total

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        win_rate = wins / count
        place_rate = data["places"] / count
        edge = win_rate - base_rate
        p_val = _binomial_p_value(wins, count, base_rate)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.01:
            loc_label = {"M": "Metro", "P": "Provincial", "C": "Country"}.get(
                data.get("location"), "All"
            )
            patterns.append(Pattern(
                pattern_type="deep_learning_form_trend",
                dimension=key,
                description=(
                    f"{data['trend'].title()} form at {data['state'] or 'ALL'} "
                    f"{loc_label}: "
                    f"win {win_rate*100:.1f}%, place {place_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% edge)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=base_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={**data, "place_rate": place_rate},
            ))

    return patterns


def analyse_class_movers(session) -> list[Pattern]:
    """Pattern 14: Class upgrades and downgrades — winning indicators.

    Analyses horses moving up or down in class (by prize money).
    For each direction, identifies which indicators correlate with winning:
    - Was competitive at previous level (top 4 last start)
    - Improving form trend (last 3 runs getting better)
    - Market confidence (SP shortened from opening)
    - Fitness (3+ runs this prep)
    """
    patterns = []

    runners = session.execute(text("""
        SELECT hr.form_history, hr.won, hr.placed, hr.starting_price,
               hr.opening_odds, hr.prep_runs, hr.last_10,
               r.prize_money as current_prize, r.state, r.location_type
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.form_history IS NOT NULL
          AND r.prize_money IS NOT NULL
          AND r.prize_money > 0
    """)).fetchall()

    agg: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "wins": 0, "places": 0,
    })

    for row in runners:
        (form_json, won, placed, sp, opening, prep_runs, last_10,
         current_prize, state, loc) = row
        try:
            history = json.loads(form_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not history:
            continue

        # Average prize money from last 2-3 starts
        recent_prizes = []
        for h in history[:3]:
            prize = h.get("prize_money") if isinstance(h, dict) else None
            if prize and isinstance(prize, (int, float)) and prize > 0:
                recent_prizes.append(prize)

        if not recent_prizes:
            continue

        avg_recent_prize = sum(recent_prizes) / len(recent_prizes)
        if avg_recent_prize == 0:
            continue

        ratio = current_prize / avg_recent_prize

        # Classify direction
        if ratio < 0.5:
            direction = "big_downgrade"
        elif ratio < 0.75:
            direction = "downgrade"
        elif ratio > 1.5:
            direction = "big_upgrade"
        elif ratio > 1.2:
            direction = "upgrade"
        else:
            continue  # Similar class

        # --- Winning indicators ---
        last_pos = history[0].get("pos") if isinstance(history[0], dict) else None
        was_competitive = last_pos is not None and last_pos <= 4

        # Improving form: last 3 positions getting better
        positions = []
        for h in history[:3]:
            p = h.get("pos") if isinstance(h, dict) else None
            if p and isinstance(p, (int, float)) and p > 0:
                positions.append(int(p))
        improving_form = (
            len(positions) >= 3
            and positions[0] < positions[1] < positions[2]
        )

        # Market confidence: SP shorter than opening (backed in)
        market_confidence = (
            sp is not None and opening is not None
            and opening > 0 and sp < opening * 0.85
        )

        # Fitness: 3+ runs this preparation
        fit = prep_runs is not None and prep_runs >= 3

        # Build sub-keys with indicators
        indicators = []
        if was_competitive:
            indicators.append("competitive")
        if improving_form:
            indicators.append("improving")
        if market_confidence:
            indicators.append("backed")
        if fit:
            indicators.append("fit")

        # Store base direction
        base_key = f"{direction}_{state or 'ALL'}"
        agg[base_key]["count"] += 1
        agg[base_key]["wins"] += 1 if won else 0
        agg[base_key]["places"] += 1 if placed else 0
        agg[base_key]["direction"] = direction
        agg[base_key]["state"] = state

        # Store each indicator combo (direction + indicator)
        for ind in indicators:
            ind_key = f"{direction}_{ind}_{state or 'ALL'}"
            agg[ind_key]["count"] += 1
            agg[ind_key]["wins"] += 1 if won else 0
            agg[ind_key]["places"] += 1 if placed else 0
            agg[ind_key]["direction"] = direction
            agg[ind_key]["indicator"] = ind
            agg[ind_key]["state"] = state

        # Store multi-indicator (2+ positive indicators)
        if len(indicators) >= 2:
            multi_key = f"{direction}_multi_indicator_{state or 'ALL'}"
            agg[multi_key]["count"] += 1
            agg[multi_key]["wins"] += 1 if won else 0
            agg[multi_key]["places"] += 1 if placed else 0
            agg[multi_key]["direction"] = direction
            agg[multi_key]["indicator"] = "multi"
            agg[multi_key]["state"] = state

    total = sum(
        d["count"] for k, d in agg.items() if "indicator" not in k
    ) or 1
    total_wins = sum(
        d["wins"] for k, d in agg.items() if "indicator" not in k
    )
    base_rate = total_wins / total

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        win_rate = wins / count
        place_rate = data["places"] / count
        edge = win_rate - base_rate
        p_val = _binomial_p_value(wins, count, base_rate)

        if p_val < SIGNIFICANCE_LEVEL:
            direction = data.get("direction", "")
            indicator = data.get("indicator", "")
            dir_label = direction.replace("_", " ").title()
            if indicator:
                ind_label = {
                    "competitive": "was top 4 last start",
                    "improving": "3-run improving trend",
                    "backed": "SP firmed >15% from open",
                    "fit": "3+ runs this prep",
                    "multi": "2+ positive indicators",
                }.get(indicator, indicator)
                desc = (
                    f"{dir_label} + {ind_label} ({data.get('state', 'ALL')}): "
                    f"win {win_rate*100:.1f}%, place {place_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% edge)"
                )
            else:
                desc = (
                    f"{dir_label} ({data.get('state', 'ALL')}): "
                    f"win {win_rate*100:.1f}%, place {place_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% edge)"
                )

            patterns.append(Pattern(
                pattern_type="deep_learning_class_mover",
                dimension=key,
                description=desc,
                sample_size=count,
                win_rate=win_rate,
                base_rate=base_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata={**data, "place_rate": place_rate},
            ))

    return patterns


def analyse_race_speed_quality(session) -> list[Pattern]:
    """Pattern 15: Runners from fast-run races vs slow races.

    Compares race time to track/distance average. Horses coming out of
    a genuinely fast race may be fitter/better than the bare result suggests.
    """
    patterns = []

    # Get average race times by venue+distance
    avg_times = {}
    rows = session.execute(text("""
        SELECT venue, distance, AVG(official_time_secs) as avg_time,
               COUNT(*) as race_count
        FROM historical_races
        WHERE official_time_secs IS NOT NULL
          AND official_time_secs > 0
        GROUP BY venue, distance
        HAVING COUNT(*) >= 5
    """)).fetchall()

    for venue, distance, avg_time, count in rows:
        avg_times[(venue, distance)] = avg_time

    # Now check runners' previous race speed
    runners = session.execute(text("""
        SELECT hr.form_history, hr.won, hr.placed, hr.starting_price,
               r.venue, r.distance, r.state
        FROM historical_runners hr
        JOIN historical_races r ON hr.race_fk = r.id
        WHERE hr.finish_position IS NOT NULL
          AND hr.form_history IS NOT NULL
    """)).fetchall()

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0})

    for form_json, won, placed, sp, venue, dist, state in runners:
        try:
            history = json.loads(form_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not history:
            continue

        # Check last race time quality
        last = history[0]
        if not isinstance(last, dict):
            continue
        last_time_str = last.get("time")
        last_venue = last.get("venue", "")
        last_dist = last.get("dist")
        if not last_time_str or not last_venue or not last_dist:
            continue

        # Parse time
        last_secs = None
        try:
            parts = last_time_str.split(":")
            if len(parts) == 3:
                last_secs = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                last_secs = float(parts[0]) * 60 + float(parts[1])
        except (ValueError, TypeError):
            continue

        if not last_secs:
            continue

        avg = avg_times.get((last_venue, last_dist))
        if not avg:
            continue

        # Classify speed
        diff_pct = (last_secs - avg) / avg * 100
        if diff_pct < -2:
            speed_class = "fast_race"
        elif diff_pct > 2:
            speed_class = "slow_race"
        else:
            continue  # Normal

        key = f"{speed_class}_{state or 'ALL'}"
        agg[key]["count"] += 1
        agg[key]["wins"] += 1 if won else 0
        agg[key]["speed_class"] = speed_class
        agg[key]["state"] = state

    total = sum(d["count"] for d in agg.values()) or 1
    total_wins = sum(d["wins"] for d in agg.values())
    base_rate = total_wins / total

    for key, data in agg.items():
        count = data["count"]
        wins = data["wins"]
        if count < MIN_SAMPLES:
            continue
        win_rate = wins / count
        edge = win_rate - base_rate
        p_val = _binomial_p_value(wins, count, base_rate)

        if p_val < SIGNIFICANCE_LEVEL and abs(edge) > 0.02:
            patterns.append(Pattern(
                pattern_type="deep_learning_race_speed",
                dimension=key,
                description=(
                    f"Runners from {data['speed_class'].replace('_', ' ')} "
                    f"({data['state'] or 'ALL'}): "
                    f"win {win_rate*100:.1f}% "
                    f"({'+' if edge > 0 else ''}{edge*100:.1f}% edge)"
                ),
                sample_size=count,
                win_rate=win_rate,
                base_rate=base_rate,
                edge=edge,
                p_value=p_val,
                confidence=_confidence_label(p_val, count),
                metadata=data,
            ))

    return patterns


# ==============================================================
# Orchestrator
# ==============================================================

ALL_ANALYSES = [
    ("Track+Distance+Condition", analyse_track_distance_condition),
    ("Barrier Bias", analyse_barrier_bias),
    ("Pace Patterns", analyse_pace_patterns),
    ("Form Cycles", analyse_form_cycles),
    ("Jockey/Trainer Combos", analyse_jockey_trainer_combos),
    ("Class Transitions", analyse_class_transitions),
    ("Condition Specialists", analyse_condition_specialists),
    ("Seasonal Patterns", analyse_seasonal_patterns),
    ("Market Efficiency", analyse_market_efficiency),
    ("Sectional Acceleration", analyse_sectional_acceleration),
    ("Track Bias (Wides)", analyse_track_bias_wides),
    ("Pace Collapse", analyse_pace_collapse),
    ("Coming Into Form", analyse_coming_into_form),
    ("Class Movers (Up/Down)", analyse_class_movers),
    ("Race Speed Quality", analyse_race_speed_quality),
]


def run_all_analyses(db_path=None) -> list[Pattern]:
    """Run all 15 pattern analyses and return significant patterns."""
    session = get_session(db_path)
    all_patterns: list[Pattern] = []

    for name, func in ALL_ANALYSES:
        try:
            patterns = func(session)
            significant = [p for p in patterns if p.p_value < SIGNIFICANCE_LEVEL]
            all_patterns.extend(significant)
            print(f"  {name}: {len(significant)} significant patterns "
                  f"(of {len(patterns)} total)")
        except Exception as e:
            logger.error(f"Analysis '{name}' failed: {e}", exc_info=True)
            print(f"  {name}: ERROR - {e}")

    session.close()

    # Sort by confidence and edge size
    all_patterns.sort(key=lambda p: (-{"HIGH": 3, "MEDIUM": 2, "LOW": 1}[p.confidence],
                                     -abs(p.edge)))

    print(f"\nTotal: {len(all_patterns)} significant patterns")
    return all_patterns
