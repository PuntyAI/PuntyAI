"""Test dampened tissue model: career-dominant with compressed weak signals.

Option B: compress specialist/weight/form/spell multipliers toward 1.0
so career dominates but other factors contribute mildly.

Dampen formula: dampened = 1.0 + (raw_mult - 1.0) * dampen_factor

Tests multiple dampen levels to find optimal balance.

Usage:
    python scripts/validate_tissue_damped.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import time
from collections import defaultdict

from punty.tissue import (
    _load_tables, _dist_bucket, _cond_bucket, _barrier_zone,
    _career_multiplier, _form_recency_multiplier,
    _condition_multiplier, _specialist_multiplier,
    _spell_multiplier, _horse_profile_multiplier,
    _weight_multiplier, _estimate_place_prob,
)
import punty.tissue as tissue_module

# Import the OOS table builder
from validate_tissue_oos import build_tables_from_subset

DB_PATH = "data/backtest.db"


def _dampen(mult: float, factor: float) -> float:
    """Compress multiplier toward 1.0 by factor (0=neutral, 1=full)."""
    return 1.0 + (mult - 1.0) * factor


def build_tissue_dampened(runners, race, meeting, dampen_config):
    """Build tissue with dampened weak factors."""
    if not runners:
        return {}

    distance = race.get("distance", 1400) if isinstance(race, dict) else getattr(race, "distance", 1400)
    track_cond = (meeting.get("track_condition", "Good") if isinstance(meeting, dict)
                  else getattr(meeting, "track_condition", "Good"))
    venue = (meeting.get("venue", "") if isinstance(meeting, dict) else getattr(meeting, "venue", ""))
    field_size = len(runners)

    weights = [r.get("weight", 0) for r in runners if isinstance(r.get("weight"), (int, float)) and r.get("weight", 0) > 0]
    avg_weight = sum(weights) / len(weights) if weights else 55.0

    raw_scores = {}
    factor_details = {}

    for runner in runners:
        rid = runner.get("id", "")
        barrier = int(runner.get("barrier", 0) or 0)
        pace_pos = runner.get("speed_map_position", "unknown") or "unknown"

        # Career: full strength (strongest signal)
        career_mult = _career_multiplier(runner.get("career_record", ""))

        # Condition: dampened
        cond_mult = _condition_multiplier(distance, track_cond, barrier, pace_pos, field_size, venue)
        cond_mult = _dampen(cond_mult, dampen_config["condition"])

        # Form recency: dampened
        form_mult = _form_recency_multiplier(runner.get("last_five", ""))
        form_mult = _dampen(form_mult, dampen_config["form_recency"])

        # Specialist: dampened
        spec_mult = _specialist_multiplier(runner, track_cond)
        spec_mult = _dampen(spec_mult, dampen_config["specialist"])

        # Spell: dampened
        spell_mult = _spell_multiplier(runner)
        spell_mult = _dampen(spell_mult, dampen_config["spell"])

        # Profile: dampened
        profile_mult = _horse_profile_multiplier(runner)
        profile_mult = _dampen(profile_mult, dampen_config["profile"])

        # Weight: dampened
        weight_mult = _weight_multiplier(runner, avg_weight)
        weight_mult = _dampen(weight_mult, dampen_config["weight"])

        tissue_score = max(0.001, career_mult * cond_mult * form_mult *
                          spec_mult * spell_mult * profile_mult * weight_mult)
        raw_scores[rid] = tissue_score
        factor_details[rid] = {
            "career": round(career_mult, 3),
            "condition": round(cond_mult, 3),
            "form_recency": round(form_mult, 3),
            "specialist": round(spec_mult, 3),
            "spell": round(spell_mult, 3),
            "profile": round(profile_mult, 3),
            "weight": round(weight_mult, 3),
        }

    total = sum(raw_scores.values())
    if total <= 0:
        return {}

    place_count = 2 if field_size <= 7 else 3
    result = {}
    for runner in runners:
        rid = runner.get("id", "")
        score = raw_scores.get(rid, 0.001)
        wp = score / total
        pp = _estimate_place_prob(wp, field_size, place_count)
        result[rid] = type('T', (), {
            'win_probability': wp, 'place_probability': pp,
            'tissue_score': score, 'factors': factor_details.get(rid, {}),
            'tissue_price': min(201, round(1/wp, 2)) if wp > 0.005 else 201
        })()
    return result


def run_validation(valid_races, rd_map, dampen_config, label=""):
    """Run tissue validation with given dampen config."""
    total = 0
    t_r1_wins = 0
    t_r1_places = 0
    mf_wins = 0
    agree_n = 0
    agree_wins = 0
    t_r1_win_pnl = 0.0
    t_r1_place_pnl = 0.0

    for race_id, (runners, race_dict, meet_dict) in valid_races.items():
        tissue = build_tissue_dampened(runners, race_dict, meet_dict, dampen_config)
        if not tissue:
            continue

        winner_id = None
        for rd in runners:
            if rd["finish_position"] == 1:
                winner_id = rd["id"]
                break
        if not winner_id:
            continue

        total += 1
        tissue_ranked = sorted(tissue.items(), key=lambda x: x[1].win_probability, reverse=True)
        t_r1_id = tissue_ranked[0][0]
        t_r1 = rd_map[race_id][t_r1_id]
        mf_id = min(runners, key=lambda r: r["current_odds"])["id"]

        if t_r1["finish_position"] == 1:
            t_r1_wins += 1
        if t_r1["finish_position"] <= 3:
            t_r1_places += 1
        if rd_map[race_id][mf_id]["finish_position"] == 1:
            mf_wins += 1

        if t_r1_id == mf_id:
            agree_n += 1
            if t_r1["finish_position"] == 1:
                agree_wins += 1

        if t_r1["finish_position"] == 1:
            t_r1_win_pnl += (t_r1.get("win_dividend") or t_r1["current_odds"]) * 10 - 10
        else:
            t_r1_win_pnl -= 10

        if t_r1["finish_position"] <= 3:
            pd = t_r1.get("place_dividend")
            t_r1_place_pnl += (pd * 10 - 10) if pd and pd > 0 else 10
        else:
            t_r1_place_pnl -= 10

    outlay = total * 10
    t1_wr = t_r1_wins / total * 100 if total > 0 else 0
    t1_pr = t_r1_places / total * 100 if total > 0 else 0
    mf_wr = mf_wins / total * 100 if total > 0 else 0
    ag_wr = agree_wins / agree_n * 100 if agree_n > 0 else 0
    win_roi = t_r1_win_pnl / outlay * 100 if outlay > 0 else 0
    place_roi = t_r1_place_pnl / outlay * 100 if outlay > 0 else 0

    return {
        "label": label,
        "total": total,
        "t1_wr": t1_wr, "t1_pr": t1_pr,
        "mf_wr": mf_wr,
        "agree_n": agree_n, "ag_wr": ag_wr,
        "win_roi": win_roi, "place_roi": place_roi,
        "win_pnl": t_r1_win_pnl, "place_pnl": t_r1_place_pnl,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("DAMPENED TISSUE VALIDATION — Option B Variants")
    print("=" * 70)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # 80/20 split
    all_race_ids = [r[0] for r in conn.execute("SELECT id FROM races ORDER BY id").fetchall()]
    split_idx = int(len(all_race_ids) * 0.80)
    train_max_id = all_race_ids[split_idx]

    print(f"Train: {split_idx} races, Test: {len(all_race_ids) - split_idx} races")
    print()

    # Build training tables
    print("Building training tables...")
    train_tables = build_tables_from_subset(conn, train_max_id)
    tissue_module._TISSUE_TABLES = train_tables

    # Load test data
    print("Loading test data...")
    test_runners = conn.execute("""
        SELECT
            r.id, r.race_id, r.barrier, r.weight, r.speed_map_position,
            r.career_record, r.last_five, r.days_since_last_run,
            r.horse_age, r.horse_sex, r.current_odds, r.opening_odds,
            r.finish_position, r.win_dividend, r.place_dividend,
            r.track_dist_stats, r.good_track_stats, r.soft_track_stats,
            r.heavy_track_stats, r.first_up_stats, r.second_up_stats,
            ra.distance, COALESCE(ra.track_condition, m.track_condition, 'Good'),
            m.venue, ra.field_size
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        JOIN meetings m ON m.id = ra.meeting_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND ra.id > ?
        ORDER BY r.race_id, r.current_odds
    """, [train_max_id]).fetchall()
    conn.close()

    # Build race dicts
    races_raw = defaultdict(list)
    for row in test_runners:
        races_raw[row[1]].append(row)

    valid_races = {}
    rd_map = {}
    for race_id, rows in races_raw.items():
        if len(rows) < 4:
            continue
        runners = []
        for row in rows:
            runners.append({
                "id": row[0], "barrier": row[2], "weight": row[3],
                "speed_map_position": row[4], "career_record": row[5],
                "last_five": row[6], "days_since_last_run": row[7],
                "horse_age": row[8], "horse_sex": row[9],
                "current_odds": row[10], "opening_odds": row[11],
                "finish_position": row[12], "win_dividend": row[13],
                "place_dividend": row[14], "track_dist_stats": row[15],
                "good_track_stats": row[16], "soft_track_stats": row[17],
                "heavy_track_stats": row[18], "first_up_stats": row[19],
                "second_up_stats": row[20],
            })
        row0 = rows[0]
        race_dict = {"distance": row0[21] or 1400, "track_condition": row0[22] or "Good", "field_size": row0[24] or len(runners)}
        meet_dict = {"venue": row0[23] or "", "track_condition": row0[22] or "Good"}
        valid_races[race_id] = (runners, race_dict, meet_dict)
        rd_map[race_id] = {r["id"]: r for r in runners}

    print(f"Test races: {len(valid_races)}")
    print()

    # Test configurations
    configs = [
        {
            "label": "A: Career ONLY (others=0)",
            "config": {"condition": 0.0, "form_recency": 0.0, "specialist": 0.0,
                       "spell": 0.0, "profile": 0.0, "weight": 0.0},
        },
        {
            "label": "B1: Light dampen (0.3)",
            "config": {"condition": 0.3, "form_recency": 0.3, "specialist": 0.3,
                       "spell": 0.3, "profile": 0.3, "weight": 0.3},
        },
        {
            "label": "B2: Moderate dampen (0.5)",
            "config": {"condition": 0.5, "form_recency": 0.5, "specialist": 0.5,
                       "spell": 0.5, "profile": 0.5, "weight": 0.5},
        },
        {
            "label": "B3: Heavy dampen (0.7)",
            "config": {"condition": 0.7, "form_recency": 0.7, "specialist": 0.7,
                       "spell": 0.7, "profile": 0.7, "weight": 0.7},
        },
        {
            "label": "C: Full (no dampen, 1.0)",
            "config": {"condition": 1.0, "form_recency": 1.0, "specialist": 1.0,
                       "spell": 1.0, "profile": 1.0, "weight": 1.0},
        },
        {
            "label": "D: Career+Cond only",
            "config": {"condition": 1.0, "form_recency": 0.0, "specialist": 0.0,
                       "spell": 0.0, "profile": 0.0, "weight": 0.0},
        },
        {
            "label": "E: Career+Specialist only",
            "config": {"condition": 0.0, "form_recency": 0.0, "specialist": 1.0,
                       "spell": 0.0, "profile": 0.0, "weight": 0.0},
        },
        {
            "label": "F: Smart (career=full, cond=0.5, spec=0.3, rest=0.2)",
            "config": {"condition": 0.5, "form_recency": 0.2, "specialist": 0.3,
                       "spell": 0.2, "profile": 0.2, "weight": 0.2},
        },
    ]

    results = []
    for cfg in configs:
        t1 = time.time()
        r = run_validation(valid_races, rd_map, cfg["config"], cfg["label"])
        elapsed = time.time() - t1
        results.append(r)
        print(f"  {cfg['label']:45s}: R1={r['t1_wr']:5.1f}%  place={r['t1_pr']:5.1f}%  "
              f"agree={r['ag_wr']:5.1f}%  winROI={r['win_roi']:+6.1f}%  plROI={r['place_roi']:+5.1f}%  ({elapsed:.1f}s)")

    # Summary table
    print()
    print("=" * 90)
    print(f"{'Config':45s} {'R1 Win%':>8s} {'R1 Pl%':>8s} {'Agree%':>8s} {'Win ROI':>9s} {'Pl ROI':>8s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x["t1_wr"], reverse=True):
        print(f"{r['label']:45s} {r['t1_wr']:7.1f}% {r['t1_pr']:7.1f}% {r['ag_wr']:7.1f}% "
              f"{r['win_roi']:+8.1f}% {r['place_roi']:+7.1f}%")

    print()
    best = max(results, key=lambda x: x["t1_wr"])
    print(f"BEST: {best['label']} — {best['t1_wr']:.1f}% R1 win accuracy")
    print(f"  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
