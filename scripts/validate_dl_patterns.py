#!/usr/bin/env python3
"""Validate DL patterns against historical race data.

Usage:
    python scripts/validate_dl_patterns.py [--db path/to/punty.db]

Loads dl_patterns.json, queries the production DB for runners matching each
pattern's conditions, splits into 80/20 train/test, calculates actual win
rates on the test set, and filters to keep only validated patterns.

Outputs: punty/data/dl_patterns_validated.json
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the probability engine helpers for consistent matching
from punty.probability import (
    _get_barrier_bucket,
    _get_dist_bucket,
    _normalize_condition,
    _position_to_style,
)

# Skip types that don't discriminate between runners
_SKIP_TYPES = {
    "deep_learning_condition_specialist",
    "deep_learning_market",
    "deep_learning_seasonal",
    "deep_learning_track_dist_cond",
    "deep_learning_standard_times",
}

# State mapping for venues (simplified)
_VENUE_STATE = {}


def _load_venue_states(cursor: sqlite3.Cursor) -> dict[str, str]:
    """Build venue→state mapping from meetings table."""
    cursor.execute(
        "SELECT DISTINCT venue, state FROM meetings WHERE state IS NOT NULL AND state != ''"
    )
    mapping = {}
    for venue, state in cursor.fetchall():
        if venue:
            mapping[venue.lower().strip()] = state.upper().strip()
    return mapping


def _get_state(venue: str, venue_states: dict) -> str:
    """Get state for a venue."""
    return venue_states.get(venue.lower().strip(), "")


def _matches_pattern(pattern: dict, runner: dict, venue_states: dict) -> bool:
    """Check if a runner matches a pattern's conditions."""
    p_type = pattern.get("type", "")
    conds = pattern.get("conditions", {})

    venue = runner.get("venue", "")
    distance = runner.get("distance", 0)
    track_condition = runner.get("track_condition", "")
    field_size = runner.get("field_size", 8)
    barrier = runner.get("barrier")
    speed_map_position = runner.get("speed_map_position", "")
    jockey = runner.get("jockey", "")
    trainer = runner.get("trainer", "")

    dist_bucket = _get_dist_bucket(distance) if distance else ""
    condition = _normalize_condition(track_condition) if track_condition else ""
    pace_style = _position_to_style(speed_map_position) if speed_map_position else ""
    barrier_bucket = (
        _get_barrier_bucket(barrier, field_size)
        if barrier and isinstance(barrier, (int, float)) and barrier >= 1
        else ""
    )
    state = _get_state(venue, venue_states)

    def _venue_match(pattern_venue: str, actual_venue: str) -> bool:
        if not pattern_venue or not actual_venue:
            return False
        return pattern_venue.lower().strip() == actual_venue.lower().strip()

    def _cond_match(pattern_cond: str, actual_cond: str) -> bool:
        if not pattern_cond or not actual_cond:
            return False
        return pattern_cond.lower().strip() == actual_cond.lower().strip()

    if p_type == "deep_learning_pace" and pace_style:
        return (
            _venue_match(conds.get("venue", ""), venue)
            and conds.get("dist_bucket") == dist_bucket
            and _cond_match(conds.get("condition", ""), condition)
            and conds.get("style") == pace_style
        )

    elif p_type == "deep_learning_barrier_bias" and barrier_bucket:
        return (
            _venue_match(conds.get("venue", ""), venue)
            and conds.get("dist_bucket") == dist_bucket
            and conds.get("barrier_bucket") == barrier_bucket
        )

    elif p_type == "deep_learning_jockey_trainer" and jockey and trainer:
        return (
            conds.get("jockey", "").lower() == jockey.lower()
            and conds.get("trainer", "").lower() == trainer.lower()
            and (not state or conds.get("state") == state)
        )

    elif p_type == "deep_learning_acceleration":
        # Would need market movement data — skip for now
        return False

    elif p_type == "deep_learning_pace_collapse" and pace_style == "leader":
        return (
            _venue_match(conds.get("venue", ""), venue)
            and conds.get("dist_bucket") == dist_bucket
            and _cond_match(conds.get("condition", ""), condition)
        )

    elif p_type == "deep_learning_class_mover" and state:
        # Need class_move and market direction — complex, skip
        return False

    elif p_type == "deep_learning_weight_impact":
        # Need weight change class — complex, skip
        return False

    elif p_type == "deep_learning_form_trend" and state:
        # Need form trend analysis — complex, skip
        return False

    elif p_type == "deep_learning_bounceback":
        # Need excuse type analysis — complex, skip
        return False

    elif p_type == "deep_learning_form_cycle":
        # Need prep runs data — complex, skip
        return False

    elif p_type == "deep_learning_class_transition":
        # Need class transition data — complex, skip
        return False

    elif p_type == "deep_learning_track_bias":
        # Need track bias data
        return False

    elif p_type == "deep_learning_race_speed":
        # Need race speed data
        return False

    return False


def validate_patterns(db_path: str) -> None:
    """Main validation pipeline."""
    # Load patterns
    patterns_path = Path(__file__).parent.parent / "punty" / "data" / "dl_patterns.json"
    with open(patterns_path) as f:
        patterns = json.load(f)

    print(f"Loaded {len(patterns)} patterns")

    # Filter out skip types
    active_patterns = [p for p in patterns if p.get("type") not in _SKIP_TYPES]
    skipped = len(patterns) - len(active_patterns)
    print(f"Active patterns (after skipping non-discriminative): {len(active_patterns)}")
    print(f"Skipped types: {skipped}")

    # Connect to DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build venue→state mapping
    venue_states = _load_venue_states(cursor)
    print(f"Venue→state mappings: {len(venue_states)}")

    # Load all runners with results
    cursor.execute("""
        SELECT
            r.horse_name, r.jockey, r.trainer, r.barrier,
            r.speed_map_position, r.finish_position,
            r.weight, r.current_odds,
            ra.distance, ra.class AS race_class,
            m.venue, m.track_condition, m.state,
            (SELECT COUNT(*) FROM runners r2
             WHERE r2.race_id = r.race_id AND r2.scratched = 0) as field_size
        FROM runners r
        JOIN races ra ON r.race_id = ra.id
        JOIN meetings m ON ra.meeting_id = m.id
        WHERE r.finish_position IS NOT NULL
          AND r.finish_position > 0
          AND r.scratched = 0
        ORDER BY m.date
    """)

    all_runners = [dict(row) for row in cursor.fetchall()]
    conn.close()

    print(f"Loaded {len(all_runners)} runners with results")

    if not all_runners:
        print("ERROR: No runners found in database. Exiting.")
        return

    # Split 80/20 by chronological order (already ordered by date)
    split_idx = int(len(all_runners) * 0.8)
    train_runners = all_runners[:split_idx]
    test_runners = all_runners[split_idx:]
    print(f"Train: {len(train_runners)}, Test: {len(test_runners)}")

    # Calculate baseline win rate on test set
    test_wins = sum(1 for r in test_runners if r["finish_position"] == 1)
    baseline_wr = test_wins / len(test_runners) if test_runners else 0
    print(f"Baseline test win rate: {baseline_wr:.4f} ({test_wins}/{len(test_runners)})")

    # Validate each pattern on test set
    validated = []
    stats = defaultdict(int)

    for i, pattern in enumerate(active_patterns):
        p_type = pattern.get("type", "")
        confidence = pattern.get("confidence", "LOW")
        original_edge = pattern.get("edge", 0)
        original_sample = pattern.get("sample_size", 0)

        # Match against test runners
        matched_test = [
            r for r in test_runners
            if _matches_pattern(pattern, r, venue_states)
        ]

        if len(matched_test) < 10:
            stats["too_few_matches"] += 1
            continue

        # Calculate actual win rate for matched runners
        wins = sum(1 for r in matched_test if r["finish_position"] == 1)
        actual_wr = wins / len(matched_test)
        actual_edge = actual_wr - baseline_wr

        # Validation criteria
        edge_threshold = 0.01  # Must have >1% edge
        min_confidence = 0.6   # Roughly maps to MEDIUM+
        min_sample = 20        # At least 20 test matches

        conf_value = 1.0 if confidence == "HIGH" else 0.6 if confidence == "MEDIUM" else 0.3

        if actual_edge > edge_threshold and conf_value >= min_confidence and len(matched_test) >= min_sample:
            # Pattern validated — update with actual test performance
            validated_pattern = pattern.copy()
            validated_pattern["validated"] = True
            validated_pattern["test_sample"] = len(matched_test)
            validated_pattern["test_win_rate"] = round(actual_wr, 4)
            validated_pattern["test_edge"] = round(actual_edge, 4)
            validated_pattern["original_edge"] = original_edge
            validated.append(validated_pattern)
            stats["validated"] += 1
        elif actual_edge > -edge_threshold:
            # Neutral — edge too small but not harmful
            stats["neutral"] += 1
        else:
            # Anti-predictive on test set
            stats["anti_predictive"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(active_patterns)}...")

    # Also keep skip-type patterns as-is (they're skipped in the engine anyway)
    skip_patterns = [p for p in patterns if p.get("type") in _SKIP_TYPES]

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Patterns tested:     {len(active_patterns)}")
    print(f"  Validated (edge>1%): {stats['validated']}")
    print(f"  Neutral:             {stats['neutral']}")
    print(f"  Anti-predictive:     {stats['anti_predictive']}")
    print(f"  Too few matches:     {stats['too_few_matches']}")
    print(f"Skip-type (kept):      {len(skip_patterns)}")
    print(f"Total output:          {len(validated) + len(skip_patterns)}")

    # Type breakdown of validated
    type_counts = defaultdict(int)
    for p in validated:
        type_counts[p["type"]] += 1
    print(f"\nValidated by type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    # Top edges
    validated.sort(key=lambda x: x.get("test_edge", 0), reverse=True)
    print(f"\nTop 10 validated patterns by test edge:")
    for p in validated[:10]:
        print(
            f"  {p['type']} | edge={p['test_edge']:.3f} | "
            f"test_n={p['test_sample']} | wr={p['test_win_rate']:.3f} | "
            f"conf={p['confidence']}"
        )

    # Output validated patterns
    output = validated + skip_patterns
    output_path = Path(__file__).parent.parent / "punty" / "data" / "dl_patterns_validated.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten {len(output)} patterns to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DL patterns against historical data")
    parser.add_argument("--db", default="data/punty.db", help="Path to SQLite database")
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Copy production DB locally: scp root@app.punty.ai:/opt/puntyai/data/punty.db data/punty.db")
        sys.exit(1)

    validate_patterns(db_path)
