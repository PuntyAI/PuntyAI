"""Out-of-sample tissue validation with time-based train/test split.

Builds tissue tables from first 80% of races (chronologically),
validates on final 20%. This eliminates in-sample bias.

Usage:
    python scripts/validate_tissue_oos.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import json
import time
from collections import defaultdict
from pathlib import Path

from punty.tissue import (
    _load_tables, _dist_bucket, _cond_bucket, _barrier_zone,
    _career_multiplier, _form_recency_multiplier,
    _condition_multiplier, _specialist_multiplier,
    _spell_multiplier, _horse_profile_multiplier,
    _weight_multiplier, _parse_stats,
    build_tissue, _TISSUE_TABLES,
)
import punty.tissue as tissue_module


DB_PATH = "data/backtest.db"


def build_tables_from_subset(conn, max_race_id: str) -> dict:
    """Build tissue tables from races up to max_race_id (training set)."""

    MIN_N = 80

    # Baseline win rate (training set only)
    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1
          AND ra.id <= ?
    """, [max_race_id]).fetchone()[0]

    # Build condition multipliers
    rows = conn.execute("""
        SELECT
            ra.distance,
            COALESCE(ra.track_condition, m.track_condition, 'Good') as track_cond,
            r.barrier, ra.field_size, r.speed_map_position, r.finish_position
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        JOIN meetings m ON m.id = ra.meeting_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND r.barrier > 0 AND ra.distance IS NOT NULL
          AND ra.id <= ?
    """, [max_race_id]).fetchall()

    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "places": 0})
    for distance, track_cond, barrier, field_size, pace_pos, finish_pos in rows:
        dist = _dist_bucket(distance)
        cond = _cond_bucket(track_cond)
        bzone = _barrier_zone(barrier, field_size or 10)
        pace = pace_pos or "unknown"

        keys = [
            f"{dist}|{cond}|{bzone}|{pace}",
            f"{dist}|{cond}|{bzone}", f"{dist}|{cond}|{pace}", f"{dist}|{bzone}|{pace}",
            f"{dist}|{cond}", f"{dist}|{bzone}", f"{dist}|{pace}", f"{cond}|{bzone}",
            f"{cond}|{pace}", dist, cond, bzone, pace,
        ]
        for key in keys:
            buckets[key]["n"] += 1
            if finish_pos == 1:
                buckets[key]["wins"] += 1
            if finish_pos <= 3:
                buckets[key]["places"] += 1

    cm_entries = {}
    for key, data in buckets.items():
        if data["n"] < MIN_N:
            continue
        wr = data["wins"] / data["n"]
        pr = data["places"] / data["n"]
        mult = wr / baseline if baseline > 0 else 1.0
        cm_entries[key] = {"mult": round(mult, 3), "win_rate": round(wr, 4),
                           "place_rate": round(pr, 4), "n": data["n"]}

    # Career bands
    career_rows = conn.execute("""
        WITH parsed AS (
            SELECT r.finish_position,
                TRY_CAST(SPLIT_PART(r.career_record, ': ', 1) AS INT) as starts,
                TRY_CAST(SPLIT_PART(SPLIT_PART(r.career_record, ': ', 2), '-', 1) AS INT) as wins_career
            FROM runners r
            JOIN races ra ON ra.id = r.race_id
            WHERE r.finish_position IS NOT NULL AND r.scratched = 0
              AND r.current_odds > 1 AND r.career_record IS NOT NULL
              AND ra.id <= ?
        )
        SELECT
            CASE
                WHEN starts < 3 THEN 'lightly_raced'
                WHEN 1.0 * wins_career / starts >= 0.30 THEN 'elite_30pct'
                WHEN 1.0 * wins_career / starts >= 0.20 THEN 'good_20pct'
                WHEN 1.0 * wins_career / starts >= 0.10 THEN 'average_10pct'
                WHEN wins_career > 0 THEN 'below_avg'
                ELSE 'maiden_career'
            END as band,
            COUNT(*) n, SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed WHERE starts IS NOT NULL AND starts > 0
        GROUP BY band
    """, [max_race_id]).fetchall()

    career_bands = {}
    for band, n, wins, places in career_rows:
        wr = wins / n
        career_bands[band] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                              "place_rate": round(places / n, 4), "n": n}

    # Form recency
    form_rows = conn.execute("""
        SELECT
            CASE
                WHEN last_five LIKE '1%%' THEN 'last_1st'
                WHEN last_five LIKE '2%%' THEN 'last_2nd'
                WHEN last_five LIKE '3%%' THEN 'last_3rd'
                WHEN last_five LIKE '4%%' OR last_five LIKE '5%%' THEN 'last_4th5th'
                WHEN last_five LIKE '6%%' OR last_five LIKE '7%%' OR last_five LIKE '8%%' THEN 'last_mid'
                WHEN last_five LIKE 'x%%' OR last_five LIKE 'X%%' THEN 'last_x'
                ELSE 'last_back'
            END as form_band,
            COUNT(*) n, SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND r.last_five IS NOT NULL AND LENGTH(r.last_five) >= 1
          AND ra.id <= ?
        GROUP BY form_band
    """, [max_race_id]).fetchall()

    form_recent = {}
    for band, n, wins, places in form_rows:
        wr = wins / n
        form_recent[band] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                             "place_rate": round(places / n, 4), "n": n}

    # Specialist (simplified — track_dist only)
    td_rows = conn.execute("""
        WITH parsed_td AS (
            SELECT r.finish_position,
                TRY_CAST(r.track_dist_stats->>'starts' AS INT) as td_starts,
                TRY_CAST(r.track_dist_stats->>'wins' AS INT) as td_wins
            FROM runners r
            JOIN races ra ON ra.id = r.race_id
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1
              AND r.scratched = 0 AND r.track_dist_stats IS NOT NULL
              AND ra.id <= ?
        )
        SELECT
            CASE
                WHEN td_starts IS NULL OR td_starts < 2 THEN 'td_no_form'
                WHEN 1.0 * td_wins / td_starts >= 0.30 THEN 'td_specialist_30'
                WHEN 1.0 * td_wins / td_starts >= 0.20 THEN 'td_specialist_20'
                WHEN td_wins > 0 THEN 'td_winner'
                ELSE 'td_no_win'
            END as td_band,
            COUNT(*) n, SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed_td GROUP BY td_band HAVING COUNT(*) >= 50
    """, [max_race_id]).fetchall()

    td_specialist = {}
    for band, n, wins, places in td_rows:
        wr = wins / n
        td_specialist[band] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                               "place_rate": round(places / n, 4), "n": n}

    # Condition specialist
    cond_rows = conn.execute("""
        WITH cond AS (
            SELECT r.finish_position,
                COALESCE(ra.track_condition, m.track_condition) as race_cond,
                r.good_track_stats, r.soft_track_stats, r.heavy_track_stats
            FROM runners r
            JOIN races ra ON ra.id = r.race_id
            JOIN meetings m ON m.id = ra.meeting_id
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1 AND r.scratched = 0
              AND ra.id <= ?
        ),
        parsed AS (
            SELECT finish_position,
                CASE
                    WHEN LOWER(race_cond) LIKE '%%heavy%%' THEN TRY_CAST(heavy_track_stats->>'starts' AS INT)
                    WHEN LOWER(race_cond) LIKE '%%soft%%' THEN TRY_CAST(soft_track_stats->>'starts' AS INT)
                    ELSE TRY_CAST(good_track_stats->>'starts' AS INT)
                END as cond_starts,
                CASE
                    WHEN LOWER(race_cond) LIKE '%%heavy%%' THEN TRY_CAST(heavy_track_stats->>'wins' AS INT)
                    WHEN LOWER(race_cond) LIKE '%%soft%%' THEN TRY_CAST(soft_track_stats->>'wins' AS INT)
                    ELSE TRY_CAST(good_track_stats->>'wins' AS INT)
                END as cond_wins
            FROM cond
        )
        SELECT
            CASE
                WHEN cond_starts IS NULL OR cond_starts < 3 THEN 'cond_no_form'
                WHEN 1.0 * cond_wins / cond_starts >= 0.25 THEN 'cond_specialist_25'
                WHEN 1.0 * cond_wins / cond_starts >= 0.15 THEN 'cond_specialist_15'
                WHEN cond_wins > 0 THEN 'cond_winner'
                ELSE 'cond_no_win'
            END as cond_band,
            COUNT(*) n, SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed GROUP BY cond_band HAVING COUNT(*) >= 50
    """, [max_race_id]).fetchall()

    cond_specialist = {}
    for band, n, wins, places in cond_rows:
        wr = wins / n
        cond_specialist[band] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                                 "place_rate": round(places / n, 4), "n": n}

    # Spell tables (first_up / second_up)
    fu_rows = conn.execute("""
        WITH parsed_fu AS (
            SELECT r.finish_position,
                TRY_CAST(r.first_up_stats->>'starts' AS INT) as fu_starts,
                TRY_CAST(r.first_up_stats->>'wins' AS INT) as fu_wins
            FROM runners r JOIN races ra ON ra.id = r.race_id
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1
              AND r.scratched = 0 AND r.first_up_stats IS NOT NULL AND ra.id <= ?
        )
        SELECT
            CASE
                WHEN fu_starts IS NULL OR fu_starts < 2 THEN 'fu_few_starts'
                WHEN 1.0 * fu_wins / fu_starts >= 0.30 THEN 'fu_strong_30'
                WHEN 1.0 * fu_wins / fu_starts >= 0.15 THEN 'fu_ok_15'
                WHEN fu_wins > 0 THEN 'fu_weak'
                ELSE 'fu_never_won'
            END as fu_band,
            COUNT(*) n, SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed_fu WHERE fu_starts IS NOT NULL AND fu_starts > 0
        GROUP BY fu_band HAVING COUNT(*) >= 50
    """, [max_race_id]).fetchall()

    first_up = {}
    for band, n, wins, places in fu_rows:
        wr = wins / n
        first_up[band] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                          "place_rate": round(places / n, 4), "n": n}

    su_rows = conn.execute("""
        WITH parsed_su AS (
            SELECT r.finish_position,
                TRY_CAST(r.second_up_stats->>'starts' AS INT) as su_starts,
                TRY_CAST(r.second_up_stats->>'wins' AS INT) as su_wins
            FROM runners r JOIN races ra ON ra.id = r.race_id
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1
              AND r.scratched = 0 AND r.second_up_stats IS NOT NULL AND ra.id <= ?
        )
        SELECT
            CASE
                WHEN su_starts IS NULL OR su_starts < 2 THEN 'su_few_starts'
                WHEN 1.0 * su_wins / su_starts >= 0.25 THEN 'su_strong_25'
                WHEN 1.0 * su_wins / su_starts >= 0.12 THEN 'su_ok_12'
                WHEN su_wins > 0 THEN 'su_weak'
                ELSE 'su_never_won'
            END as su_band,
            COUNT(*) n, SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed_su WHERE su_starts IS NOT NULL AND su_starts > 0
        GROUP BY su_band HAVING COUNT(*) >= 50
    """, [max_race_id]).fetchall()

    second_up = {}
    for band, n, wins, places in su_rows:
        wr = wins / n
        second_up[band] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                           "place_rate": round(places / n, 4), "n": n}

    # Horse profile (age × sex)
    prof_rows = conn.execute("""
        SELECT
            CASE WHEN horse_age <= 2 THEN '2yo' WHEN horse_age = 3 THEN '3yo'
                 WHEN horse_age = 4 THEN '4yo' WHEN horse_age = 5 THEN '5yo'
                 WHEN horse_age = 6 THEN '6yo' WHEN horse_age >= 7 THEN '7yo_plus'
                 ELSE 'unknown' END as age_band,
            CASE WHEN LOWER(horse_sex) IN ('m','c','r','h') THEN 'male'
                 WHEN LOWER(horse_sex) IN ('f','mare') THEN 'female'
                 WHEN LOWER(horse_sex) IN ('g','gelding') THEN 'gelding'
                 ELSE 'other' END as sex_band,
            COUNT(*) n, SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners r JOIN races ra ON ra.id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0 AND r.current_odds > 1
          AND r.horse_age IS NOT NULL AND ra.id <= ?
        GROUP BY age_band, sex_band HAVING COUNT(*) >= 50
    """, [max_race_id]).fetchall()

    horse_profile = {}
    for age, sex, n, wins, places in prof_rows:
        key = f"{age}|{sex}"
        wr = wins / n
        horse_profile[key] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                              "place_rate": round(places / n, 4), "n": n}

    # Weight carried
    wt_rows = conn.execute("""
        SELECT
            CASE WHEN r.weight < field.avg_wt - 3 THEN 'very_light'
                 WHEN r.weight < field.avg_wt - 1 THEN 'light'
                 WHEN r.weight <= field.avg_wt + 1 THEN 'average'
                 WHEN r.weight <= field.avg_wt + 3 THEN 'heavy'
                 ELSE 'very_heavy' END as wt_class,
            COUNT(*) n, SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        JOIN (SELECT race_id, AVG(weight) avg_wt FROM runners
              WHERE weight > 0 AND scratched = 0 GROUP BY race_id) field
          ON field.race_id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.weight > 0
          AND r.current_odds > 1 AND r.scratched = 0 AND ra.id <= ?
        GROUP BY wt_class
    """, [max_race_id]).fetchall()

    weight_table = {}
    for wt, n, wins, places in wt_rows:
        wr = wins / n
        weight_table[wt] = {"mult": round(wr / baseline, 3), "win_rate": round(wr, 4),
                            "place_rate": round(places / n, 4), "n": n}

    # Field size
    fs_rows = conn.execute("""
        SELECT ra.field_size, COUNT(*) n,
            SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners r JOIN races ra ON ra.id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND ra.field_size IS NOT NULL AND ra.field_size > 0
          AND ra.id <= ?
        GROUP BY ra.field_size HAVING COUNT(*) >= 30 ORDER BY ra.field_size
    """, [max_race_id]).fetchall()

    field_size_table = {}
    for fs, n, wins, places in fs_rows:
        field_size_table[str(fs)] = {
            "expected_win_rate": round(wins / n, 4),
            "expected_place_rate": round(places / n, 4),
            "n": n,
        }

    return {
        "version": "1.0-oos-train",
        "condition_multipliers": {"entries": cm_entries, "baseline_win_rate": round(baseline, 5)},
        "career_bands": {"bands": career_bands, "experience": {}, "baseline": round(baseline, 5)},
        "form_recency": {"recent_form": form_recent, "sequences": {}, "trends": {}},
        "specialist_tables": {"track_distance": td_specialist, "condition": cond_specialist},
        "spell_tables": {"first_up": first_up, "second_up": second_up},
        "venue_overrides": {},  # skip for OOS (would need separate train split)
        "field_size": field_size_table,
        "horse_profile": horse_profile,
        "weight_carried": weight_table,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("OUT-OF-SAMPLE TISSUE VALIDATION")
    print("=" * 70)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get chronological race ID split point (80/20)
    all_race_ids = conn.execute("""
        SELECT id FROM races ORDER BY id
    """).fetchall()
    all_race_ids = [r[0] for r in all_race_ids]
    split_idx = int(len(all_race_ids) * 0.80)
    train_max_id = all_race_ids[split_idx]
    test_min_id = all_race_ids[split_idx + 1]

    print(f"Total races: {len(all_race_ids)}")
    print(f"Train: first {split_idx} races (up to {train_max_id})")
    print(f"Test:  last {len(all_race_ids) - split_idx} races (from {test_min_id})")
    print()

    # Build tables from training set only
    print("Building tissue tables from training set...")
    train_tables = build_tables_from_subset(conn, train_max_id)
    print(f"  Condition entries: {len(train_tables['condition_multipliers']['entries'])}")
    print(f"  Career bands: {len(train_tables['career_bands']['bands'])}")
    print()

    # Inject training tables into tissue module (replace the loaded tables)
    tissue_module._TISSUE_TABLES = train_tables

    # Load test set runners
    print("Loading test set runners...")
    test_runners = conn.execute("""
        SELECT
            r.id, r.race_id, r.barrier, r.weight, r.speed_map_position,
            r.career_record, r.last_five, r.days_since_last_run,
            r.horse_age, r.horse_sex, r.current_odds, r.opening_odds,
            r.finish_position, r.win_dividend, r.place_dividend,
            r.track_dist_stats, r.good_track_stats, r.soft_track_stats,
            r.heavy_track_stats, r.first_up_stats, r.second_up_stats,
            ra.distance, COALESCE(ra.track_condition, m.track_condition, 'Good') as track_cond,
            m.venue, ra.field_size
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        JOIN meetings m ON m.id = ra.meeting_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1
          AND ra.id > ?
        ORDER BY r.race_id, r.current_odds
    """, [train_max_id]).fetchall()
    conn.close()

    print(f"Test runners: {len(test_runners)}")

    # Group by race
    races = defaultdict(list)
    for row in test_runners:
        races[row[1]].append(row)
    valid_races = {rid: runners for rid, runners in races.items() if len(runners) >= 4}
    print(f"Test races (4+ runners): {len(valid_races)}")
    print()

    # Counters
    total = 0
    tissue_r1_wins = 0
    tissue_r1_places = 0
    market_fav_wins = 0
    market_fav_places = 0
    agree_n = 0
    agree_wins = 0
    disagree_n = 0
    disagree_wins = 0

    tissue_rank_wins = defaultdict(int)
    tissue_rank_places = defaultdict(int)
    tissue_rank_n = defaultdict(int)

    t_r1_win_pnl = 0.0
    t_r1_place_pnl = 0.0
    t_r2_place_pnl = 0.0
    mf_win_pnl = 0.0

    factor_w = defaultdict(float)
    factor_l = defaultdict(float)
    n_w = 0
    n_l = 0

    t1 = time.time()
    for race_idx, (race_id, runners) in enumerate(valid_races.items()):
        if race_idx % 1000 == 0 and race_idx > 0:
            elapsed = time.time() - t1
            print(f"  {race_idx}/{len(valid_races)}  "
                  f"tissue={tissue_r1_wins}/{total} ({tissue_r1_wins/total*100:.1f}%)  "
                  f"mkt={market_fav_wins}/{total} ({market_fav_wins/total*100:.1f}%)")

        row0 = runners[0]
        distance = row0[21] or 1400
        track_cond = row0[22] or "Good"
        venue = row0[23] or ""
        field_size = row0[24] or len(runners)

        runner_dicts = []
        for row in runners:
            runner_dicts.append({
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

        race_dict = {"distance": distance, "track_condition": track_cond, "field_size": field_size}
        meet_dict = {"venue": venue, "track_condition": track_cond}

        tissue = build_tissue(runner_dicts, race_dict, meet_dict)
        if not tissue:
            continue

        winner_id = None
        for rd in runner_dicts:
            if rd["finish_position"] == 1:
                winner_id = rd["id"]
                break
        if not winner_id:
            continue

        total += 1
        rd_by_id = {rd["id"]: rd for rd in runner_dicts}

        tissue_ranked = sorted(tissue.items(), key=lambda x: x[1].win_probability, reverse=True)
        tissue_r1_id = tissue_ranked[0][0]
        tissue_r1 = rd_by_id[tissue_r1_id]
        mf_id = min(runner_dicts, key=lambda r: r["current_odds"])["id"]
        mf = rd_by_id[mf_id]

        for rank_idx, (rid, tres) in enumerate(tissue_ranked[:6]):
            rank = rank_idx + 1
            tissue_rank_n[rank] += 1
            r = rd_by_id[rid]
            if r["finish_position"] == 1:
                tissue_rank_wins[rank] += 1
            if r["finish_position"] <= 3:
                tissue_rank_places[rank] += 1

        if tissue_r1["finish_position"] == 1:
            tissue_r1_wins += 1
        if tissue_r1["finish_position"] <= 3:
            tissue_r1_places += 1

        if mf["finish_position"] == 1:
            market_fav_wins += 1
        if mf["finish_position"] <= 3:
            market_fav_places += 1

        if tissue_r1_id == mf_id:
            agree_n += 1
            if tissue_r1["finish_position"] == 1:
                agree_wins += 1
        else:
            disagree_n += 1
            if tissue_r1["finish_position"] == 1:
                disagree_wins += 1

        r1_odds = tissue_r1["current_odds"]
        if tissue_r1["finish_position"] == 1:
            t_r1_win_pnl += (tissue_r1.get("win_dividend") or r1_odds) * 10 - 10
        else:
            t_r1_win_pnl -= 10

        if tissue_r1["finish_position"] <= 3:
            pd = tissue_r1.get("place_dividend")
            t_r1_place_pnl += (pd * 10 - 10) if pd and pd > 0 else 10
        else:
            t_r1_place_pnl -= 10

        if len(tissue_ranked) >= 2:
            r2 = rd_by_id[tissue_ranked[1][0]]
            if r2["finish_position"] <= 3:
                pd2 = r2.get("place_dividend")
                t_r2_place_pnl += (pd2 * 10 - 10) if pd2 and pd2 > 0 else 10
            else:
                t_r2_place_pnl -= 10

        if mf["finish_position"] == 1:
            mf_win_pnl += (mf.get("win_dividend") or mf["current_odds"]) * 10 - 10
        else:
            mf_win_pnl -= 10

        r1_factors = tissue[tissue_r1_id].factors
        if tissue_r1["finish_position"] == 1:
            n_w += 1
            for k, v in r1_factors.items():
                if k != "raw_score":
                    factor_w[k] += v
        else:
            n_l += 1
            for k, v in r1_factors.items():
                if k != "raw_score":
                    factor_l[k] += v

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"OUT-OF-SAMPLE RESULTS — {total} test races ({elapsed:.0f}s)")
    print("=" * 70)

    t1_wr = tissue_r1_wins / total * 100
    t1_pr = tissue_r1_places / total * 100
    mf_wr = market_fav_wins / total * 100
    mf_pr = market_fav_places / total * 100

    print()
    print("--- RANK-1 ACCURACY (OUT-OF-SAMPLE) ---")
    print(f"  Tissue Rank 1:    win={t1_wr:.1f}%  place={t1_pr:.1f}%")
    print(f"  Market Favourite: win={mf_wr:.1f}%  place={mf_pr:.1f}%")
    delta = t1_wr - mf_wr
    print(f"  Delta: {delta:+.1f}%  {'*** TISSUE BEATS MARKET ***' if delta > 0 else 'Market still ahead'}")

    print()
    print("--- AGREEMENT ---")
    if agree_n > 0:
        print(f"  Aligned (n={agree_n:,}):    win={agree_wins/agree_n*100:.1f}%")
    if disagree_n > 0:
        print(f"  Disagreed (n={disagree_n:,}): tissue win={disagree_wins/disagree_n*100:.1f}%")

    print()
    print("--- RANK ACCURACY (Top 6) ---")
    for rank in range(1, 7):
        n = tissue_rank_n[rank]
        w = tissue_rank_wins[rank]
        p = tissue_rank_places[rank]
        if n > 0:
            print(f"  Rank {rank}: n={n:6d}  win={w/n*100:5.1f}%  place={p/n*100:5.1f}%")

    print()
    print("--- FLAT BET ROI ($10/race, OUT-OF-SAMPLE) ---")
    outlay = total * 10
    print(f"  Tissue R1 Win:   PnL=${t_r1_win_pnl:+,.0f}  ROI={t_r1_win_pnl/outlay*100:+.1f}%")
    print(f"  Tissue R1 Place: PnL=${t_r1_place_pnl:+,.0f}  ROI={t_r1_place_pnl/outlay*100:+.1f}%")
    print(f"  Tissue R2 Place: PnL=${t_r2_place_pnl:+,.0f}  ROI={t_r2_place_pnl/outlay*100:+.1f}%")
    print(f"  Mkt Fav Win:     PnL=${mf_win_pnl:+,.0f}  ROI={mf_win_pnl/outlay*100:+.1f}%")

    print()
    print("--- FACTOR DELTA (Winners vs Losers, Tissue R1) ---")
    if n_w > 0 and n_l > 0:
        for factor in sorted(factor_w.keys()):
            avg_w = factor_w[factor] / n_w
            avg_l = factor_l[factor] / n_l
            diff = avg_w - avg_l
            signal = "+++" if diff > 0.1 else "++" if diff > 0.05 else "+" if diff > 0.02 else "." if diff > 0 else "-"
            print(f"  {factor:15s}: winners={avg_w:.3f}  losers={avg_l:.3f}  delta={diff:+.3f} {signal}")

    print()
    print("=" * 70)
    print("VERDICT (OUT-OF-SAMPLE)")
    print("=" * 70)
    print(f"  Tissue R1 accuracy:  {t1_wr:.1f}%  (old engine: 27.4%, target: >30%)")
    print(f"  Market fav accuracy: {mf_wr:.1f}%  (benchmark: 32-35%)")
    if agree_n > 0:
        print(f"  Agreement accuracy:  {agree_wins/agree_n*100:.1f}%  (target: >40%)")
    print(f"  Tissue R1 Win ROI:   {t_r1_win_pnl/outlay*100:+.1f}%")
    print(f"  Tissue R1 Place ROI: {t_r1_place_pnl/outlay*100:+.1f}%")


if __name__ == "__main__":
    main()
