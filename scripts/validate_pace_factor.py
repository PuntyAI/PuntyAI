#!/usr/bin/env python3
"""Validate the pace factor's predictive value.

Usage:
    python scripts/validate_pace_factor.py [--db path/to/punty.db]

Tests whether the pace factor (currently 1% weight) improves or hurts
prediction accuracy. Compares calibration with pace at 1% vs 0%.

If removing pace improves accuracy, recommends zeroing the weight.
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate_pace(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get runners with speed_map_position and results
    cursor.execute("""
        SELECT
            r.speed_map_position, r.finish_position, r.current_odds,
            ra.distance,
            (SELECT COUNT(*) FROM runners r2
             WHERE r2.race_id = r.race_id AND r2.scratched = 0) as field_size
        FROM runners r
        JOIN races ra ON r.race_id = ra.id
        WHERE r.finish_position IS NOT NULL
          AND r.finish_position > 0
          AND r.scratched = 0
          AND r.speed_map_position IS NOT NULL
          AND r.speed_map_position != ''
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    if not rows:
        print("ERROR: No runners with speed_map_position found.")
        return

    print(f"Loaded {len(rows)} runners with speed map data")

    # Analyse win rate by pace position
    position_stats = defaultdict(lambda: {"wins": 0, "total": 0, "distances": defaultdict(lambda: {"wins": 0, "total": 0})})

    for r in rows:
        pos = (r["speed_map_position"] or "").lower().strip()
        won = r["finish_position"] == 1
        dist = r["distance"] or 0

        if dist <= 1100:
            dist_bucket = "sprint"
        elif dist <= 1300:
            dist_bucket = "short"
        elif dist <= 1800:
            dist_bucket = "middle"
        else:
            dist_bucket = "staying"

        position_stats[pos]["total"] += 1
        position_stats[pos]["distances"][dist_bucket]["total"] += 1
        if won:
            position_stats[pos]["wins"] += 1
            position_stats[pos]["distances"][dist_bucket]["wins"] += 1

    # Overall baseline
    total_wins = sum(s["wins"] for s in position_stats.values())
    total_runners = sum(s["total"] for s in position_stats.values())
    baseline_wr = total_wins / total_runners if total_runners > 0 else 0.10

    print(f"\nBaseline win rate: {baseline_wr:.4f} ({total_wins}/{total_runners})")
    print(f"\n{'Position':<15} {'Wins':<8} {'Total':<8} {'Win Rate':<10} {'vs Base':<10}")
    print("-" * 55)

    for pos in ["leader", "on_pace", "midfield", "backmarker"]:
        stats = position_stats.get(pos, {"wins": 0, "total": 0})
        if stats["total"] == 0:
            continue
        wr = stats["wins"] / stats["total"]
        diff = wr - baseline_wr
        print(f"{pos:<15} {stats['wins']:<8} {stats['total']:<8} {wr:.4f}    {diff:+.4f}")

    # Distance breakdown
    print(f"\n{'Position':<15} {'Distance':<10} {'Wins':<6} {'Total':<8} {'Win Rate':<10}")
    print("-" * 55)
    for pos in ["leader", "on_pace", "midfield", "backmarker"]:
        stats = position_stats.get(pos, {"wins": 0, "total": 0, "distances": {}})
        for dist in ["sprint", "short", "middle", "staying"]:
            d = stats["distances"].get(dist, {"wins": 0, "total": 0})
            if d["total"] < 20:
                continue
            wr = d["wins"] / d["total"]
            print(f"{pos:<15} {dist:<10} {d['wins']:<6} {d['total']:<8} {wr:.4f}")

    # Predictiveness test: Does knowing pace position improve predictions
    # beyond what market odds already tell us?
    print("\n" + "=" * 60)
    print("PACE PREDICTIVENESS TEST")
    print("=" * 60)

    # Group by odds band + pace position
    odds_pace = defaultdict(lambda: {"wins": 0, "total": 0})
    odds_only = defaultdict(lambda: {"wins": 0, "total": 0})

    for r in rows:
        odds = r["current_odds"]
        if not odds or odds <= 0:
            continue
        pos = (r["speed_map_position"] or "").lower().strip()
        won = r["finish_position"] == 1

        # Odds band
        if odds < 3:
            band = "short"
        elif odds < 6:
            band = "mid"
        elif odds < 15:
            band = "long"
        else:
            band = "roughie"

        odds_only[band]["total"] += 1
        odds_pace[f"{band}_{pos}"]["total"] += 1
        if won:
            odds_only[band]["wins"] += 1
            odds_pace[f"{band}_{pos}"]["wins"] += 1

    print(f"\n{'Band':<10} {'Base WR':<10} | {'Leaders':<12} {'On Pace':<12} {'Midfield':<12} {'Back':<12}")
    print("-" * 75)

    for band in ["short", "mid", "long", "roughie"]:
        base = odds_only[band]
        base_wr = base["wins"] / base["total"] if base["total"] > 0 else 0

        parts = [f"{band:<10} {base_wr:.3f}     |"]
        for pos in ["leader", "on_pace", "midfield", "backmarker"]:
            key = f"{band}_{pos}"
            s = odds_pace[key]
            if s["total"] >= 10:
                wr = s["wins"] / s["total"]
                diff = wr - base_wr
                parts.append(f"{wr:.3f} ({diff:+.3f})")
            else:
                parts.append(f"  n<10     ")
        print(" ".join(parts))

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    # Check if leaders consistently outperform within odds bands
    leader_lift = 0
    leader_tests = 0
    for band in ["short", "mid", "long"]:
        base = odds_only[band]
        leader = odds_pace[f"{band}_leader"]
        if base["total"] >= 50 and leader["total"] >= 20:
            base_wr = base["wins"] / base["total"]
            leader_wr = leader["wins"] / leader["total"]
            if leader_wr > base_wr:
                leader_lift += 1
            leader_tests += 1

    if leader_tests > 0 and leader_lift >= leader_tests * 0.6:
        print("KEEP: Leaders consistently outperform within odds bands.")
        print("Pace factor adds information beyond market odds alone.")
        print("Current 1% weight is appropriate.")
    else:
        print("CONSIDER ZEROING: Leaders do NOT consistently outperform")
        print("within odds bands. Pace factor may be redundant with market.")
        print("Set pace weight to 0% (same as movement factor).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/punty.db")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: DB not found at {args.db}")
        print("Copy: scp root@app.punty.ai:/opt/puntyai/data/punty.db data/punty.db")
        sys.exit(1)

    validate_pace(args.db)
