#!/usr/bin/env python3
"""Validate market movement factor predictive value.

Usage:
    python scripts/validate_market_movement.py [--db path/to/punty.db]

The market movement factor is currently zeroed (0% weight).
Tests whether specific movement thresholds (e.g., >20% firming)
have predictive value that justifies re-enabling at 2-3% weight.
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate_movement(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            r.current_odds, r.opening_odds, r.finish_position,
            (SELECT COUNT(*) FROM runners r2
             WHERE r2.race_id = r.race_id AND r2.scratched = 0) as field_size
        FROM runners r
        JOIN races ra ON r.race_id = ra.id
        WHERE r.finish_position IS NOT NULL
          AND r.finish_position > 0
          AND r.scratched = 0
          AND r.current_odds IS NOT NULL
          AND r.current_odds > 0
          AND r.opening_odds IS NOT NULL
          AND r.opening_odds > 0
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    if not rows:
        print("ERROR: No runners with both opening and current odds found.")
        return

    print(f"Loaded {len(rows)} runners with opening + current odds")

    # Calculate movement for each runner
    # Firming = current < opening (odds shortened = money came in)
    # Drifting = current > opening (odds lengthened = money went out)
    movement_bands = {
        "firm_30+": {"min_firm": 0.30, "max_firm": 1.0},
        "firm_20-30": {"min_firm": 0.20, "max_firm": 0.30},
        "firm_10-20": {"min_firm": 0.10, "max_firm": 0.20},
        "stable": {"min_firm": -0.10, "max_firm": 0.10},
        "drift_10-20": {"min_firm": -0.20, "max_firm": -0.10},
        "drift_20-30": {"min_firm": -0.30, "max_firm": -0.20},
        "drift_30+": {"min_firm": -1.0, "max_firm": -0.30},
    }

    stats = defaultdict(lambda: {"wins": 0, "places": 0, "total": 0, "avg_odds": 0.0})
    total_wins = 0
    total = 0

    for r in rows:
        opening = r["opening_odds"]
        current = r["current_odds"]
        won = r["finish_position"] == 1
        placed = r["finish_position"] <= 3

        # Movement as fraction of opening odds
        # Positive = firming (odds shortened), negative = drifting
        movement = (opening - current) / opening if opening > 0 else 0

        total += 1
        if won:
            total_wins += 1

        for band_name, band in movement_bands.items():
            if band["min_firm"] <= movement < band["max_firm"] or \
               (band_name == "firm_30+" and movement >= 0.30) or \
               (band_name == "drift_30+" and movement <= -0.30):
                stats[band_name]["total"] += 1
                stats[band_name]["avg_odds"] += current
                if won:
                    stats[band_name]["wins"] += 1
                if placed:
                    stats[band_name]["places"] += 1
                break

    baseline_wr = total_wins / total if total > 0 else 0.10

    print(f"\nBaseline win rate: {baseline_wr:.4f}")
    print(f"\n{'Movement':<15} {'Wins':<6} {'Total':<8} {'Win Rate':<10} {'vs Base':<10} {'Place Rate':<10} {'Avg Odds':<10}")
    print("-" * 80)

    band_order = ["firm_30+", "firm_20-30", "firm_10-20", "stable", "drift_10-20", "drift_20-30", "drift_30+"]
    for band_name in band_order:
        s = stats[band_name]
        if s["total"] == 0:
            continue
        wr = s["wins"] / s["total"]
        pr = s["places"] / s["total"]
        diff = wr - baseline_wr
        avg_odds = s["avg_odds"] / s["total"]
        print(f"{band_name:<15} {s['wins']:<6} {s['total']:<8} {wr:.4f}    {diff:+.4f}    {pr:.4f}     ${avg_odds:.1f}")

    # ROI analysis by movement band
    print(f"\n{'='*60}")
    print("ROI ANALYSIS (assuming $1 win bet at current odds)")
    print(f"{'='*60}")
    print(f"\n{'Movement':<15} {'Wins':<6} {'Total':<8} {'ROI':<10} {'$1 P&L':<10}")
    print("-" * 55)

    for band_name in band_order:
        s = stats[band_name]
        if s["total"] == 0:
            continue
        avg_odds = s["avg_odds"] / s["total"]
        # ROI = (wins × avg_odds - total) / total
        revenue = s["wins"] * avg_odds
        roi = (revenue - s["total"]) / s["total"] * 100
        pnl = revenue - s["total"]
        print(f"{band_name:<15} {s['wins']:<6} {s['total']:<8} {roi:+.1f}%    ${pnl:+.0f}")

    # Smart money test: Do strong firmers (>20%) predict better than odds imply?
    print(f"\n{'='*60}")
    print("SMART MONEY TEST")
    print(f"{'='*60}")

    # Compare firm_20+ win rate against what their odds imply
    firm_runners = []
    drift_runners = []

    for r in rows:
        opening = r["opening_odds"]
        current = r["current_odds"]
        movement = (opening - current) / opening if opening > 0 else 0

        if movement >= 0.20:
            firm_runners.append(r)
        elif movement <= -0.20:
            drift_runners.append(r)

    if firm_runners:
        firm_wins = sum(1 for r in firm_runners if r["finish_position"] == 1)
        firm_wr = firm_wins / len(firm_runners)
        implied_wr = sum(1/r["current_odds"] for r in firm_runners) / len(firm_runners)
        print(f"\nStrong firmers (>20%): {len(firm_runners)} runners")
        print(f"  Actual win rate: {firm_wr:.4f}")
        print(f"  Market-implied:  {implied_wr:.4f}")
        print(f"  Edge vs market:  {(firm_wr - implied_wr):+.4f}")

    if drift_runners:
        drift_wins = sum(1 for r in drift_runners if r["finish_position"] == 1)
        drift_wr = drift_wins / len(drift_runners)
        implied_wr = sum(1/r["current_odds"] for r in drift_runners) / len(drift_runners)
        print(f"\nStrong drifters (>20%): {len(drift_runners)} runners")
        print(f"  Actual win rate: {drift_wr:.4f}")
        print(f"  Market-implied:  {implied_wr:.4f}")
        print(f"  Edge vs market:  {(drift_wr - implied_wr):+.4f}")

    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    if firm_runners:
        firm_wins = sum(1 for r in firm_runners if r["finish_position"] == 1)
        firm_wr = firm_wins / len(firm_runners)
        implied_wr = sum(1/r["current_odds"] for r in firm_runners) / len(firm_runners)
        edge = firm_wr - implied_wr

        if edge > 0.02 and len(firm_runners) >= 100:
            print(f"RE-ENABLE at 2-3% weight: Strong firmers show {edge:+.3f} edge vs market.")
            print("Filter: Only apply movement factor for >20% firming.")
            print("Suggested implementation:")
            print("  if movement > 0.20: score = 0.55 + min(0.10, movement * 0.2)")
            print("  elif movement < -0.20: score = 0.45 - min(0.10, abs(movement) * 0.2)")
            print("  else: score = 0.50 (neutral)")
        elif edge > 0:
            print(f"MARGINAL: Edge of {edge:+.3f} is positive but small.")
            print("Keep at 0% or test at 1% weight with tight thresholds.")
        else:
            print(f"KEEP ZEROED: No edge found ({edge:+.3f}). Movement is already")
            print("priced into market odds. The 40% market weight captures this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/punty.db")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: DB not found at {args.db}")
        print("Copy: scp root@app.punty.ai:/opt/puntyai/data/punty.db data/punty.db")
        sys.exit(1)

    validate_movement(args.db)
