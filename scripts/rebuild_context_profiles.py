#!/usr/bin/env python3
"""Rebuild context profiles from production SQLite data.

Usage:
    python scripts/rebuild_context_profiles.py [--db path/to/punty.db]

Derives venue|distance|class context multipliers for each probability factor
from historical race results. Replaces the one-off Proform-derived profiles.

Each profile key = "venue|dist_bucket|class_bucket"
Each profile has win/place/top4 multipliers per factor:
  multiplier > 1.0 = factor is MORE predictive in this context
  multiplier < 1.0 = factor is LESS predictive
  multiplier = 0.5 = neutral (insufficient data)

Outputs: punty/data/context_profiles.json
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from punty.probability import _get_dist_bucket

# Minimum sample size for a profile to be valid
# Lower than ideal (30) because current DB has ~2K runners
MIN_PROFILE_SAMPLES = 15

# Factor data we can derive from DB columns
# Maps factor name → (runner column, how to bucket it)
FACTORS = [
    "market",
    "form",
    "class_fitness",
    "pace",
    "barrier",
    "jockey_trainer",
    "weight_carried",
    "horse_profile",
    "movement",
]


def _classify_class(race_class: str) -> str:
    """Bucket race class into categories matching profile keys."""
    if not race_class:
        return "other"
    cl = race_class.lower().strip()

    if "maiden" in cl:
        return "maiden"
    if "group 1" in cl or "group1" in cl:
        return "group1"
    if "group 2" in cl or "group2" in cl:
        return "group2"
    if "group 3" in cl or "group3" in cl:
        return "group3"
    if "listed" in cl:
        return "listed"
    if "open" in cl:
        return "open"
    if "stakes" in cl:
        return "stakes"

    # BM (benchmark) races
    import re
    bm_match = re.search(r"bm\s*(\d+)", cl)
    if bm_match:
        rating = int(bm_match.group(1))
        if rating <= 58:
            return "class1"
        elif rating <= 68:
            return "class2"
        elif rating <= 78:
            return "class3"
        else:
            return "open"

    # Class N
    class_match = re.search(r"class\s*(\d+)", cl)
    if class_match:
        n = int(class_match.group(1))
        return f"class{min(n, 4)}"

    # Restricted/conditional
    if any(kw in cl for kw in ("restricted", "conditional", "3yo", "4yo")):
        return "class1"

    return "other"


def _normalize_venue(venue: str) -> str:
    """Normalize venue name for profile key."""
    if not venue:
        return ""
    v = venue.lower().strip()
    # Common aliases
    aliases = {
        "royal randwick": "randwick",
        "rosehill gardens": "rosehill",
        "pinjarra park": "pinjarra",
        "moonee valley": "moonee_valley",
        "the valley": "moonee_valley",
        "eagle farm": "eagle_farm",
        "doomben racecourse": "doomben",
        "morphettville parks": "morphettville",
        "sandown lakeside": "sandown",
        "sandown hillside": "sandown",
    }
    return aliases.get(v, v.replace(" ", "_"))


def _is_market_winner(runner: dict, all_runners: list[dict]) -> bool:
    """Did the market favourite win?"""
    race_runners = [r for r in all_runners if r["race_id"] == runner["race_id"]]
    if not race_runners:
        return False
    # Find favourite (lowest odds)
    with_odds = [(r, r["current_odds"]) for r in race_runners
                 if r["current_odds"] and r["current_odds"] > 0]
    if not with_odds:
        return False
    fav = min(with_odds, key=lambda x: x[1])
    return fav[0]["finish_position"] == 1


def _compute_factor_multipliers(
    context_runners: list[dict],
    all_runners_by_race: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Compute factor multipliers for a set of runners in a context.

    For each factor, compare the win/place rate when the factor
    "predicted" correctly vs overall baseline.
    """
    if len(context_runners) < MIN_PROFILE_SAMPLES:
        return {}

    results = {}
    for outcome in ["win", "place", "top4"]:
        factor_mults = {}

        for factor in FACTORS:
            mult = _compute_single_factor(context_runners, all_runners_by_race, factor, outcome)
            factor_mults[factor] = mult

        results[outcome] = factor_mults

    return results


def _compute_single_factor(
    runners: list[dict],
    all_by_race: dict[str, list[dict]],
    factor: str,
    outcome: str,
) -> float:
    """Compute multiplier for a single factor in a context.

    Multiplier = success_rate_when_factor_favours / baseline_success_rate
    """
    if not runners:
        return 0.5

    # Define success
    def _is_success(r: dict) -> bool:
        pos = r.get("finish_position", 99)
        if outcome == "win":
            return pos == 1
        elif outcome == "place":
            return pos <= 3
        else:  # top4
            return pos <= 4

    baseline_successes = sum(1 for r in runners if _is_success(r))
    baseline_rate = baseline_successes / len(runners) if runners else 0

    if baseline_rate == 0:
        return 0.5  # No winners → can't measure predictiveness

    # For each factor, identify "favoured" runners (those the factor would rank highly)
    favoured = []
    for r in runners:
        if _is_factor_favoured(r, all_by_race, factor):
            favoured.append(r)

    if len(favoured) < 10:
        return 0.5  # Insufficient data

    favoured_successes = sum(1 for r in favoured if _is_success(r))
    favoured_rate = favoured_successes / len(favoured)

    # Multiplier: how much better does the factor predict in this context?
    mult = favoured_rate / baseline_rate if baseline_rate > 0 else 0.5

    # Cap to reasonable range [0.2, 3.0]
    return round(max(0.2, min(3.0, mult)), 3)


def _is_factor_favoured(runner: dict, all_by_race: dict, factor: str) -> bool:
    """Is this runner 'favoured' by the given factor?"""
    race_runners = all_by_race.get(runner["race_id"], [])
    if not race_runners:
        return False

    if factor == "market":
        # Favoured = in top 3 by odds (shortest)
        with_odds = [(r, r.get("current_odds") or 999) for r in race_runners]
        with_odds.sort(key=lambda x: x[1])
        top3_ids = {r["runner_id"] for r, _ in with_odds[:3]}
        return runner["runner_id"] in top3_ids

    elif factor == "form":
        # Favoured = last_five starts with "1" or "2"
        lf = runner.get("last_five", "") or ""
        return bool(lf) and lf[0:1] in ("1", "2")

    elif factor == "barrier":
        # Favoured = barrier in 1-4 (inside)
        b = runner.get("barrier")
        return b is not None and 1 <= b <= 4

    elif factor == "pace":
        # Favoured = leader or on_pace
        smp = (runner.get("speed_map_position") or "").lower()
        return smp in ("leader", "on_pace")

    elif factor == "jockey_trainer":
        # Favoured = has a "top" jockey (proxy: runner has lowest odds in race = best J/T)
        # Simplified: in top 2 by odds
        with_odds = [(r, r.get("current_odds") or 999) for r in race_runners]
        with_odds.sort(key=lambda x: x[1])
        top2_ids = {r["runner_id"] for r, _ in with_odds[:2]}
        return runner["runner_id"] in top2_ids

    elif factor == "weight_carried":
        # Favoured = lighter than average weight
        weights = [r.get("weight") for r in race_runners if r.get("weight") and r["weight"] > 0]
        if not weights:
            return False
        avg = sum(weights) / len(weights)
        rw = runner.get("weight")
        return rw is not None and rw < avg

    elif factor == "class_fitness":
        # Favoured = had recent runs (days_since < 30)
        days = runner.get("days_since_last_run")
        return days is not None and 7 <= days <= 35

    elif factor == "horse_profile":
        # Favoured = career win rate above average
        # Proxy: has winning form (last_five contains "1")
        lf = runner.get("last_five", "") or ""
        return "1" in lf

    elif factor == "movement":
        # Favoured = odds shortened (opening > current)
        opening = runner.get("opening_odds")
        current = runner.get("current_odds")
        if opening and current and opening > 0 and current > 0:
            return current < opening * 0.85  # firmed by 15%+
        return False

    return False


def rebuild_profiles(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            r.id as runner_id, r.race_id, r.horse_name, r.jockey, r.trainer,
            r.barrier, r.weight, r.current_odds, r.opening_odds,
            r.speed_map_position, r.finish_position, r.last_five,
            r.days_since_last_run,
            ra.distance, ra.class AS race_class,
            m.venue, m.track_condition
        FROM runners r
        JOIN races ra ON r.race_id = ra.id
        JOIN meetings m ON ra.meeting_id = m.id
        WHERE r.finish_position IS NOT NULL
          AND r.finish_position > 0
          AND r.scratched = 0
    """)

    all_runners = [dict(row) for row in cursor.fetchall()]
    conn.close()

    print(f"Loaded {len(all_runners)} runners with results")

    # Group by race for factor calculations
    all_by_race = defaultdict(list)
    for r in all_runners:
        all_by_race[r["race_id"]].append(r)

    # Group by context key: venue|dist_bucket|class_bucket
    contexts = defaultdict(list)
    for r in all_runners:
        venue = _normalize_venue(r.get("venue", ""))
        distance = r.get("distance", 0)
        race_class = r.get("race_class", "")

        if not venue or not distance:
            continue

        dist_bucket = _get_dist_bucket(distance)
        class_bucket = _classify_class(race_class)
        key = f"{venue}|{dist_bucket}|{class_bucket}"
        contexts[key].append(r)

    print(f"Context groups: {len(contexts)}")
    print(f"Groups with >={MIN_PROFILE_SAMPLES} runners: "
          f"{sum(1 for v in contexts.values() if len(v) >= MIN_PROFILE_SAMPLES)}")

    # Build profiles
    profiles = {}
    for key, runners in contexts.items():
        if len(runners) < MIN_PROFILE_SAMPLES:
            continue

        multipliers = _compute_factor_multipliers(runners, all_by_race)
        if multipliers:
            profiles[key] = {"_n": len(runners), **multipliers}

    print(f"\nBuilt {len(profiles)} profiles")

    # Summary stats
    if profiles:
        all_n = [p["_n"] for p in profiles.values()]
        print(f"  Min samples: {min(all_n)}")
        print(f"  Max samples: {max(all_n)}")
        print(f"  Median samples: {sorted(all_n)[len(all_n)//2]}")

    # Output to separate file (don't overwrite Proform-derived profiles)
    # The live profiles will only replace the originals once we have enough data
    output = {"profiles": profiles}
    output_path = Path(__file__).parent.parent / "punty" / "data" / "context_profiles_live.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten {len(profiles)} profiles to {output_path}")

    # Compare with existing Proform profiles
    existing_path = Path(__file__).parent.parent / "punty" / "data" / "context_profiles.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        existing_count = len(existing.get("profiles", {}))
        print(f"Existing Proform profiles: {existing_count}")
        print(f"New live profiles: {len(profiles)}")
        if len(profiles) >= existing_count * 0.5:
            print("Live data covers 50%+ of Proform profiles — consider switching.")
        else:
            print(f"Live data only covers {len(profiles)}/{existing_count} "
                  f"({len(profiles)/max(existing_count,1)*100:.0f}%) — keep Proform profiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/punty.db")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: DB not found at {args.db}")
        print("Copy: scp root@app.punty.ai:/opt/puntyai/data/punty.db data/punty.db")
        sys.exit(1)

    rebuild_profiles(args.db)
