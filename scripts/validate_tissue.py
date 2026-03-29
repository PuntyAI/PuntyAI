"""Validate tissue engine accuracy against 221K historical runners.

Optimized: loads all data into memory first, then processes batch.

Usage:
    python scripts/validate_tissue.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import time
from collections import defaultdict

from punty.tissue import (
    build_tissue, _load_tables,
    _career_multiplier, _form_recency_multiplier,
    _condition_multiplier, _specialist_multiplier,
    _spell_multiplier, _horse_profile_multiplier,
    _weight_multiplier, _barrier_zone, _parse_stats,
)


DB_PATH = "data/backtest.db"


def main():
    t0 = time.time()
    print("=" * 70)
    print("TISSUE ENGINE VALIDATION — 221K Runners")
    print("=" * 70)

    tables = _load_tables()
    if not tables:
        print("ERROR: tissue_tables.json not found.")
        return
    print(f"Tables loaded: v{tables.get('version', '?')}")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Load ALL data at once into memory
    print("Loading all race data...")
    all_runners = conn.execute("""
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
        ORDER BY r.race_id, r.current_odds
    """).fetchall()
    conn.close()

    print(f"Loaded {len(all_runners)} runners in {time.time()-t0:.1f}s")

    # Group by race
    races = defaultdict(list)
    for row in all_runners:
        races[row[1]].append(row)

    # Filter to 4+ runner races
    valid_races = {rid: runners for rid, runners in races.items() if len(runners) >= 4}
    print(f"Valid races (4+ runners): {len(valid_races)}")
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

    # ROI
    t_r1_win_pnl = 0.0
    t_r1_place_pnl = 0.0
    t_r2_place_pnl = 0.0
    mf_win_pnl = 0.0

    # Factor deltas
    factor_w = defaultdict(float)
    factor_l = defaultdict(float)
    n_w = 0
    n_l = 0

    # Process each race
    t1 = time.time()
    for race_idx, (race_id, runners) in enumerate(valid_races.items()):
        if race_idx % 5000 == 0 and race_idx > 0:
            elapsed = time.time() - t1
            rate = race_idx / elapsed
            print(f"  {race_idx}/{len(valid_races)} races ({rate:.0f}/s)  "
                  f"tissue_r1={tissue_r1_wins}/{total} ({tissue_r1_wins/total*100:.1f}%)  "
                  f"mkt_fav={market_fav_wins}/{total} ({market_fav_wins/total*100:.1f}%)")

        # Extract race context from first runner row
        row0 = runners[0]
        distance = row0[21] or 1400
        track_cond = row0[22] or "Good"
        venue = row0[23] or ""
        field_size = row0[24] or len(runners)

        # Build runner dicts for tissue engine
        runner_dicts = []
        for row in runners:
            runner_dicts.append({
                "id": row[0],
                "barrier": row[2],
                "weight": row[3],
                "speed_map_position": row[4],
                "career_record": row[5],
                "last_five": row[6],
                "days_since_last_run": row[7],
                "horse_age": row[8],
                "horse_sex": row[9],
                "current_odds": row[10],
                "opening_odds": row[11],
                "finish_position": row[12],
                "win_dividend": row[13],
                "place_dividend": row[14],
                "track_dist_stats": row[15],
                "good_track_stats": row[16],
                "soft_track_stats": row[17],
                "heavy_track_stats": row[18],
                "first_up_stats": row[19],
                "second_up_stats": row[20],
            })

        race_dict = {"distance": distance, "track_condition": track_cond, "field_size": field_size}
        meet_dict = {"venue": venue, "track_condition": track_cond}

        # Build tissue
        tissue = build_tissue(runner_dicts, race_dict, meet_dict)
        if not tissue:
            continue

        # Find winner
        winner_id = None
        for rd in runner_dicts:
            if rd["finish_position"] == 1:
                winner_id = rd["id"]
                break
        if not winner_id:
            continue

        total += 1
        rd_by_id = {rd["id"]: rd for rd in runner_dicts}

        # Tissue ranking
        tissue_ranked = sorted(tissue.items(), key=lambda x: x[1].win_probability, reverse=True)
        tissue_r1_id = tissue_ranked[0][0]
        tissue_r1 = rd_by_id[tissue_r1_id]

        # Market favourite
        mf_id = min(runner_dicts, key=lambda r: r["current_odds"])["id"]
        mf = rd_by_id[mf_id]

        # Rank accuracy
        for rank_idx, (rid, tres) in enumerate(tissue_ranked[:6]):
            rank = rank_idx + 1
            tissue_rank_n[rank] += 1
            r = rd_by_id[rid]
            if r["finish_position"] == 1:
                tissue_rank_wins[rank] += 1
            if r["finish_position"] <= 3:
                tissue_rank_places[rank] += 1

        # Tissue R1
        if tissue_r1["finish_position"] == 1:
            tissue_r1_wins += 1
        if tissue_r1["finish_position"] <= 3:
            tissue_r1_places += 1

        # Market fav
        if mf["finish_position"] == 1:
            market_fav_wins += 1
        if mf["finish_position"] <= 3:
            market_fav_places += 1

        # Agreement
        if tissue_r1_id == mf_id:
            agree_n += 1
            if tissue_r1["finish_position"] == 1:
                agree_wins += 1
        else:
            disagree_n += 1
            if tissue_r1["finish_position"] == 1:
                disagree_wins += 1

        # ROI
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

        # Factor analysis
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
    print(f"RESULTS — {total} races validated in {elapsed:.0f}s")
    print("=" * 70)

    print()
    print("--- RANK-1 ACCURACY ---")
    t1_wr = tissue_r1_wins / total * 100
    t1_pr = tissue_r1_places / total * 100
    mf_wr = market_fav_wins / total * 100
    mf_pr = market_fav_places / total * 100
    print(f"  Tissue Rank 1:    win={t1_wr:.1f}%  place={t1_pr:.1f}%")
    print(f"  Market Favourite: win={mf_wr:.1f}%  place={mf_pr:.1f}%")
    delta = t1_wr - mf_wr
    print(f"  Delta: {delta:+.1f}%  {'TISSUE WINS' if delta > 0 else 'Market still ahead'}")

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
    print("--- FLAT BET ROI ($10/race) ---")
    outlay = total * 10
    print(f"  Tissue R1 Win:   PnL=${t_r1_win_pnl:+,.0f}  ROI={t_r1_win_pnl/outlay*100:+.1f}%")
    print(f"  Tissue R1 Place: PnL=${t_r1_place_pnl:+,.0f}  ROI={t_r1_place_pnl/outlay*100:+.1f}%")
    print(f"  Tissue R2 Place: PnL=${t_r2_place_pnl:+,.0f}  ROI={t_r2_place_pnl/outlay*100:+.1f}%")
    print(f"  Mkt Fav Win:     PnL=${mf_win_pnl:+,.0f}  ROI={mf_win_pnl/outlay*100:+.1f}%")

    print()
    print("--- FACTOR ANALYSIS (Tissue R1: Winners vs Losers) ---")
    if n_w > 0 and n_l > 0:
        for factor in sorted(factor_w.keys()):
            avg_w = factor_w[factor] / n_w
            avg_l = factor_l[factor] / n_l
            diff = avg_w - avg_l
            signal = "+++" if diff > 0.1 else "++" if diff > 0.05 else "+" if diff > 0.02 else "." if diff > 0 else "-"
            print(f"  {factor:15s}: winners={avg_w:.3f}  losers={avg_l:.3f}  delta={diff:+.3f} {signal}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Tissue R1 accuracy:  {t1_wr:.1f}%  (old engine: 27.4%, target: >30%)")
    print(f"  Market fav accuracy: {mf_wr:.1f}%  (benchmark)")
    if agree_n > 0:
        print(f"  Agreement accuracy:  {agree_wins/agree_n*100:.1f}%  (target: >40%)")


if __name__ == "__main__":
    main()
