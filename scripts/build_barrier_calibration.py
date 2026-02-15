#!/usr/bin/env python3
"""Build venue-specific barrier calibration data from historical results.

Usage:
    python scripts/build_barrier_calibration.py [--db path/to/punty.db]

Analyses historical win rates by barrier × venue × distance bucket.
Outputs: punty/data/barrier_calibration.json
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from punty.probability import _get_barrier_bucket, _get_dist_bucket


def build_calibration(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            r.barrier, r.finish_position, r.scratched,
            ra.distance,
            m.venue,
            (SELECT COUNT(*) FROM runners r2
             WHERE r2.race_id = r.race_id AND r2.scratched = 0) as field_size
        FROM runners r
        JOIN races ra ON r.race_id = ra.id
        JOIN meetings m ON ra.meeting_id = m.id
        WHERE r.finish_position IS NOT NULL
          AND r.finish_position > 0
          AND r.scratched = 0
          AND r.barrier IS NOT NULL
          AND r.barrier > 0
    """)

    # Aggregate: venue → dist_bucket → barrier_bucket → {wins, starts}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"wins": 0, "starts": 0})))
    total_wins = 0
    total_starts = 0

    for row in cursor.fetchall():
        venue = row["venue"] or ""
        distance = row["distance"] or 0
        barrier = row["barrier"]
        field_size = row["field_size"] or 8
        won = row["finish_position"] == 1

        if not venue or not distance or not barrier:
            continue

        dist_bucket = _get_dist_bucket(distance)
        barrier_bucket = _get_barrier_bucket(int(barrier), field_size)

        venue_key = venue.lower().strip()
        data[venue_key][dist_bucket][barrier_bucket]["starts"] += 1
        total_starts += 1
        if won:
            data[venue_key][dist_bucket][barrier_bucket]["wins"] += 1
            total_wins += 1

    conn.close()

    baseline_wr = total_wins / total_starts if total_starts > 0 else 0.10
    print(f"Total: {total_starts} starts, {total_wins} wins ({baseline_wr:.3f} baseline)")
    print(f"Venues: {len(data)}")

    # Build calibration: multiplier = actual_wr / baseline_wr
    calibration = {}
    for venue, dist_buckets in data.items():
        calibration[venue] = {}
        for dist_bucket, barrier_buckets in dist_buckets.items():
            calibration[venue][dist_bucket] = {}
            # Get venue+dist baseline
            vd_wins = sum(bb["wins"] for bb in barrier_buckets.values())
            vd_starts = sum(bb["starts"] for bb in barrier_buckets.values())
            vd_baseline = vd_wins / vd_starts if vd_starts > 0 else baseline_wr

            for bb, stats in barrier_buckets.items():
                if stats["starts"] >= 10:  # minimum sample (lower threshold for small DBs)
                    wr = stats["wins"] / stats["starts"]
                    multiplier = wr / vd_baseline if vd_baseline > 0 else 1.0
                    calibration[venue][dist_bucket][bb] = {
                        "multiplier": round(multiplier, 3),
                        "win_rate": round(wr, 4),
                        "sample": stats["starts"],
                    }

    # Clean empty entries
    calibration = {
        v: {d: bbs for d, bbs in dists.items() if bbs}
        for v, dists in calibration.items()
    }
    calibration = {v: dists for v, dists in calibration.items() if dists}

    output_path = Path(__file__).parent.parent / "punty" / "data" / "barrier_calibration.json"
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration for {len(calibration)} venues written to {output_path}")

    # Show top biases
    biases = []
    for venue, dists in calibration.items():
        for dist, bbs in dists.items():
            for bb, stats in bbs.items():
                if stats["sample"] >= 10:
                    biases.append((venue, dist, bb, stats["multiplier"], stats["sample"]))

    biases.sort(key=lambda x: abs(x[3] - 1.0), reverse=True)
    print("\nTop 15 barrier biases (furthest from neutral):")
    for venue, dist, bb, mult, n in biases[:15]:
        direction = "FAVOURS" if mult > 1.0 else "AGAINST"
        print(f"  {venue} {dist} {bb}: {mult:.3f}x ({direction}, n={n})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/punty.db")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: DB not found at {args.db}")
        print("Copy: scp root@app.punty.ai:/opt/puntyai/data/punty.db data/punty.db")
        sys.exit(1)

    build_calibration(args.db)
