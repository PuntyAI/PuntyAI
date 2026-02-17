"""Grid search weight optimizer using backtest.db historical data.

Usage (local):
    python scripts/tune_weights.py

Searches over factor weight combinations and evaluates each against
the full backtest dataset. Outputs data/optimal_weights.json.
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["PUNTY_DB_PATH"] = "data/backtest.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def get_output_path():
    return Path(f"data/optimal_weights_batch{BATCH}.json")


def safe_div(a, b):
    return a / b if b else 0


# Batch config — change BATCH and SEED for each overnight run
# --- BATCH CONFIG --- change these for each overnight run
BATCH = 2          # Batch 2 = 2026-02-18
SEED = 123          # Different seed per batch for different race samples
SAMPLE_SIZE = 1000  # 1K races: good balance of reliability vs speed

# Win weight search space
WIN_SEARCH_SPACE = {
    "market":        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "form":          [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "jockey_trainer": [0.03, 0.05, 0.08, 0.10, 0.15],
    "horse_profile":  [0.02, 0.04, 0.06, 0.08],
    "class_fitness":  [0.02, 0.04, 0.06],
    "weight_carried": [0.02, 0.04, 0.06],
    "barrier":        [0.01, 0.02, 0.03],
    "pace":           [0.00, 0.01, 0.02],
    "movement":       [0.00, 0.02],
}

# Place weight search space — different emphasis for consistency
PLACE_SEARCH_SPACE = {
    "market":        [0.25, 0.30, 0.35, 0.40],
    "form":          [0.15, 0.20, 0.25, 0.30, 0.35],
    "class_fitness":  [0.05, 0.08, 0.10, 0.15],
    "jockey_trainer": [0.05, 0.08, 0.10, 0.15],
    "barrier":        [0.03, 0.05, 0.08],
    "weight_carried": [0.03, 0.05, 0.08],
    "horse_profile":  [0.03, 0.05, 0.08],
    "pace":           [0.00, 0.02, 0.04],
    "movement":       [0.00, 0.02],
}

# Factors to hold fixed at 0 (deep_learning gets auto-added by probability engine)
FIXED_ZERO = ["deep_learning"]

WIN_MAX_COMBOS = 1500   # Phase 1: win weights (~5 hours)
PLACE_MAX_COMBOS = 1500  # Phase 2: place weights (~5 hours)

def generate_weight_combos(search_space, max_combos, seed_offset=1000):
    """Generate weight combinations — random sample from the full grid."""
    import random
    factors = list(search_space.keys())
    value_lists = [search_space[f] for f in factors]

    # Calculate total grid size
    total_grid = 1
    for vl in value_lists:
        total_grid *= len(vl)
    logger.info(f"Full grid has {total_grid:,} combos, sampling {min(max_combos, total_grid):,}")

    if total_grid <= max_combos:
        # Small enough to enumerate
        all_combos = []
        for vals in product(*value_lists):
            weights = dict(zip(factors, vals))
            total = sum(weights.values())
            if total > 0:
                weights = {k: round(v / total, 4) for k, v in weights.items()}
            all_combos.append(weights)
        return all_combos

    # Random sample from the grid
    rng = random.Random(SEED + seed_offset)
    combos = []
    seen = set()
    while len(combos) < max_combos:
        vals = tuple(rng.choice(vl) for vl in value_lists)
        if vals in seen:
            continue
        seen.add(vals)
        weights = dict(zip(factors, vals))
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        combos.append(weights)
    return combos


async def evaluate_weights(weights, races_data, place_weights=None):
    """Evaluate a weight configuration against pre-loaded race data.

    Returns dict with metrics including win/place, exotics, and sequence leg hit rates.
    place_weights: separate weights for place probability calculation.
    """
    from punty.probability import calculate_race_probabilities

    STAKE = 5
    top1_wins = 0
    top3_wins = 0
    win_pnl = 0.0
    place_pnl = 0.0
    total_races = 0

    # Exotic counters
    exacta_hits = 0
    quinella_hits = 0
    trifecta_hits = 0
    first4_hits = 0
    exacta_eligible = 0  # races with 2+ ranked runners
    trifecta_eligible = 0  # races with 3+ ranked runners
    first4_eligible = 0  # races with 4+ ranked runners

    # Sequence leg hit rates: did winner fall in our top N?
    leg_hits_top3 = 0   # skinny quaddie (~3 per leg)
    leg_hits_top4 = 0   # balanced quaddie (~4 per leg)
    leg_hits_top6 = 0   # wide quaddie (~6 per leg)

    for meeting, race, active, winners, placers in races_data:
        try:
            probs = calculate_race_probabilities(active, race, meeting, weights=weights, place_weights=place_weights)
        except Exception:
            continue

        if not probs:
            continue

        total_races += 1
        ranked = sorted(probs.items(), key=lambda x: -x[1].win_probability)

        # Build runner lookup by id
        runner_by_id = {r.id: r for r in active}

        # Top 1 accuracy
        winner_id = winners[0].id
        top1_won = ranked[0][0] == winner_id
        if top1_won:
            top1_wins += 1

        # Sequence leg hit rates (did winner fall in our top N ranked?)
        top3_ids = set(rid for rid, _ in ranked[:3])
        top4_ids = set(rid for rid, _ in ranked[:4])
        top6_ids = set(rid for rid, _ in ranked[:6])
        if winner_id in top3_ids:
            leg_hits_top3 += 1
        if winner_id in top4_ids:
            leg_hits_top4 += 1
        if winner_id in top6_ids:
            leg_hits_top6 += 1

        # Top 3 accuracy
        top3_ids = set(rid for rid, _ in ranked[:3])
        if winner_id in top3_ids:
            top3_wins += 1

        # --- Exotic metrics ---
        # Get actual finish positions for our top-ranked runners
        def get_pos(runner_id):
            r = runner_by_id.get(runner_id)
            return r.finish_position if r else 999

        # Exacta: our top 2 finish 1st-2nd in exact order
        if len(ranked) >= 2:
            exacta_eligible += 1
            r1_pos = get_pos(ranked[0][0])
            r2_pos = get_pos(ranked[1][0])
            if r1_pos == 1 and r2_pos == 2:
                exacta_hits += 1

            # Quinella: our top 2 finish in top 2 (any order)
            if {r1_pos, r2_pos} == {1, 2}:
                quinella_hits += 1

        # Trifecta: our top 3 fill the top 3 placings (any order)
        if len(ranked) >= 3:
            trifecta_eligible += 1
            top3_positions = {get_pos(ranked[j][0]) for j in range(3)}
            if top3_positions == {1, 2, 3}:
                trifecta_hits += 1

        # First4: our top 4 fill the top 4 placings (any order)
        if len(ranked) >= 4:
            first4_eligible += 1
            top4_positions = {get_pos(ranked[j][0]) for j in range(4)}
            if top4_positions == {1, 2, 3, 4}:
                first4_hits += 1

        # Win P&L: bet on top 1 by win probability ranking
        top1_runner = runner_by_id.get(ranked[0][0])
        if top1_runner:
            is_winner = top1_runner.finish_position == 1
            win_div = top1_runner.win_dividend or 0

            if is_winner and win_div > 0:
                win_pnl += (win_div - 1) * STAKE
            else:
                win_pnl -= STAKE

        # Place P&L: bet on top 1 by PLACE probability ranking
        place_ranked = sorted(probs.items(), key=lambda x: -x[1].place_probability)
        place_top1_runner = runner_by_id.get(place_ranked[0][0])
        if place_top1_runner:
            is_placed = place_top1_runner.id in placers
            place_div = place_top1_runner.place_dividend or 0

            if is_placed and place_div > 0:
                place_pnl += (place_div - 1) * STAKE
            else:
                place_pnl -= STAKE

    if total_races == 0:
        return None

    top1_rate = safe_div(top1_wins, total_races)
    top3_rate = safe_div(top3_wins, total_races)
    win_roi = safe_div(win_pnl, total_races * STAKE)
    place_roi = safe_div(place_pnl, total_races * STAKE)

    exacta_rate = safe_div(exacta_hits, exacta_eligible)
    quinella_rate = safe_div(quinella_hits, exacta_eligible)
    trifecta_rate = safe_div(trifecta_hits, trifecta_eligible)
    first4_rate = safe_div(first4_hits, first4_eligible)

    # Sequence per-leg hit rates
    leg_rate_skinny = safe_div(leg_hits_top3, total_races)    # top 3 per leg
    leg_rate_balanced = safe_div(leg_hits_top4, total_races)  # top 4 per leg
    leg_rate_wide = safe_div(leg_hits_top6, total_races)      # top 6 per leg

    # Estimated quaddie hit rate = leg_rate ^ 4 (independent legs)
    quaddie_skinny = leg_rate_skinny ** 4
    quaddie_balanced = leg_rate_balanced ** 4
    quaddie_wide = leg_rate_wide ** 4

    # Combined score: balance accuracy, profitability, and exotic hit rates
    score = (
        0.20 * top1_rate
        + 0.10 * top3_rate
        + 0.20 * (win_roi + 1)
        + 0.20 * (place_roi + 1)
        + 0.05 * quinella_rate
        + 0.05 * trifecta_rate
        + 0.05 * first4_rate
        + 0.05 * leg_rate_skinny
        + 0.05 * leg_rate_balanced
        + 0.05 * leg_rate_wide
    )

    return {
        "weights": weights,
        "races": total_races,
        "top1_win_rate": round(top1_rate * 100, 2),
        "top3_win_rate": round(top3_rate * 100, 2),
        "win_roi": round(win_roi * 100, 2),
        "place_roi": round(place_roi * 100, 2),
        "exacta_rate": round(exacta_rate * 100, 2),
        "quinella_rate": round(quinella_rate * 100, 2),
        "trifecta_rate": round(trifecta_rate * 100, 2),
        "first4_rate": round(first4_rate * 100, 2),
        "leg_skinny": round(leg_rate_skinny * 100, 2),
        "leg_balanced": round(leg_rate_balanced * 100, 2),
        "leg_wide": round(leg_rate_wide * 100, 2),
        "quaddie_skinny": round(quaddie_skinny * 100, 2),
        "quaddie_balanced": round(quaddie_balanced * 100, 2),
        "quaddie_wide": round(quaddie_wide * 100, 2),
        "score": round(score, 6),
    }


def print_result(label, result, current=None):
    """Print a result summary. If current is provided, show diffs."""
    print(f"\n  Score: {result['score']:.6f}", end="")
    if current:
        print(f" ({result['score'] - current['score']:+.6f})")
    else:
        print()
    print(f"  Top1: {result['top1_win_rate']}%, Top3: {result['top3_win_rate']}%")
    print(f"  Win ROI: {result['win_roi']}%, Place ROI: {result['place_roi']}%")
    print(f"  Exacta: {result['exacta_rate']}%, Quinella: {result['quinella_rate']}%")
    print(f"  Trifecta: {result['trifecta_rate']}%, First4: {result['first4_rate']}%")
    print(f"  Leg hit: top3={result['leg_skinny']}%, top4={result['leg_balanced']}%, top6={result['leg_wide']}%")
    print(f"  Quaddie est: {result['quaddie_skinny']}/{result['quaddie_balanced']}/{result['quaddie_wide']}%")


async def main():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy.orm import selectinload
    from punty.models.meeting import Meeting, Race, Runner

    start_time = time.time()

    # Create engine for backtest DB
    bt_engine = create_async_engine(
        "sqlite+aiosqlite:///data/backtest.db",
        echo=False,
        connect_args={"timeout": 30},
    )
    session_factory = async_sessionmaker(bt_engine, class_=AsyncSession, expire_on_commit=False)

    # Pre-load all race data into memory
    logger.info("Loading race data into memory...")
    races_data = []

    async with session_factory() as db:
        result = await db.execute(
            select(Meeting)
            .options(selectinload(Meeting.races).selectinload(Race.runners))
            .order_by(Meeting.date)
        )
        meetings = result.scalars().all()

    for meeting in meetings:
        for race in meeting.races:
            if not race.results_status:
                continue
            active = [r for r in race.runners if not r.scratched and r.finish_position and r.finish_position > 0]
            if len(active) < 3:
                continue
            winners = [r for r in active if r.finish_position == 1]
            if not winners:
                continue
            field_size = len(active)
            place_count = 2 if field_size <= 7 else 3
            placers = set(r.id for r in active if r.finish_position <= place_count)
            races_data.append((meeting, race, active, winners, placers))

    logger.info(f"Loaded {len(races_data)} races into memory ({time.time() - start_time:.0f}s)")

    # Sample races
    import random
    random.seed(SEED)
    sample_size = min(SAMPLE_SIZE, len(races_data))
    sample = random.sample(races_data, sample_size)

    # Log batch info
    sample_race_ids = sorted(set(race.id for _, race, _, _, _ in sample))
    logger.info(f"BATCH {BATCH} | seed={SEED} | {sample_size} races from {len(races_data)} total")
    batch_meta = {
        "batch": BATCH, "seed": SEED,
        "sample_size": sample_size, "total_races": len(races_data),
        "race_ids": sample_race_ids,
    }
    with open(Path(f"data/tuner_batch{BATCH}_races.json"), "w") as f:
        json.dump(batch_meta, f, indent=2)

    # Get current defaults
    from punty.probability import DEFAULT_WEIGHTS, DEFAULT_PLACE_WEIGHTS
    current_win_w = dict(DEFAULT_WEIGHTS)
    current_place_w = dict(DEFAULT_PLACE_WEIGHTS)

    # ================================================================
    # PHASE 1: Optimize WIN weights (place weights held at current)
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: Optimizing WIN weights...")
    logger.info(f"{'='*60}")

    win_combos = generate_weight_combos(WIN_SEARCH_SPACE, WIN_MAX_COMBOS, seed_offset=1000)
    logger.info(f"Testing {len(win_combos)} win weight combinations...")

    # Baseline with current weights
    baseline_result = await evaluate_weights(current_win_w, sample, place_weights=current_place_w)
    logger.info(f"Baseline: score={baseline_result['score']:.6f}, "
                f"top1={baseline_result['top1_win_rate']}%, "
                f"win_roi={baseline_result['win_roi']}%, place_roi={baseline_result['place_roi']}%")

    best_win_result = baseline_result
    win_results = []

    for i, weights in enumerate(win_combos):
        r = await evaluate_weights(weights, sample, place_weights=current_place_w)
        if r:
            win_results.append(r)
            if r["score"] > best_win_result["score"]:
                best_win_result = r
                logger.info(f"  Win best at {i}: score={r['score']:.6f}, "
                            f"top1={r['top1_win_rate']}%, win_roi={r['win_roi']}%, "
                            f"place_roi={r['place_roi']}%")
        if (i + 1) % 100 == 0:
            logger.info(f"  Win phase: {i + 1}/{len(win_combos)} ({time.time() - start_time:.0f}s)")

    win_results.sort(key=lambda x: -x["score"])
    best_win_weights = best_win_result["weights"]
    phase1_time = time.time() - start_time

    logger.info(f"\nPhase 1 complete: {len(win_combos)} combos in {phase1_time:.0f}s")
    logger.info(f"Best win weights: {best_win_weights}")

    # ================================================================
    # PHASE 2: Optimize PLACE weights (win weights held at best from phase 1)
    # ================================================================
    phase2_start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: Optimizing PLACE weights...")
    logger.info(f"{'='*60}")

    place_combos = generate_weight_combos(PLACE_SEARCH_SPACE, PLACE_MAX_COMBOS, seed_offset=2000)
    logger.info(f"Testing {len(place_combos)} place weight combinations...")

    # Baseline: best win weights + current place weights
    place_baseline = await evaluate_weights(best_win_weights, sample, place_weights=current_place_w)
    logger.info(f"Place baseline (current place weights): "
                f"place_roi={place_baseline['place_roi']}%")

    best_place_result = place_baseline
    place_results = []

    for i, pw in enumerate(place_combos):
        r = await evaluate_weights(best_win_weights, sample, place_weights=pw)
        if r:
            # For place phase, optimize primarily on place_roi
            r["_place_score"] = r["place_roi"]
            r["place_weights"] = pw
            place_results.append(r)
            if r["place_roi"] > best_place_result.get("place_roi", -999):
                best_place_result = r
                logger.info(f"  Place best at {i}: place_roi={r['place_roi']}%, "
                            f"win_roi={r['win_roi']}%, top1={r['top1_win_rate']}%")
        if (i + 1) % 100 == 0:
            logger.info(f"  Place phase: {i + 1}/{len(place_combos)} ({time.time() - phase2_start:.0f}s)")

    place_results.sort(key=lambda x: -x.get("_place_score", -999))
    best_place_weights = best_place_result.get("place_weights", current_place_w)
    total_time = time.time() - start_time

    # ================================================================
    # Save combined results
    # ================================================================
    output = {
        "meta": {
            "batch": BATCH, "seed": SEED,
            "win_combos": len(win_combos), "place_combos": len(place_combos),
            "sample_size": sample_size, "total_races": len(races_data),
            "elapsed_seconds": round(total_time, 1),
        },
        "baseline": baseline_result,
        "best_win": {
            "weights": best_win_weights,
            "result": best_win_result,
        },
        "best_place": {
            "weights": best_place_weights,
            "result": best_place_result,
        },
        "win_top_10": win_results[:10],
        "place_top_10": [{k: v for k, v in r.items() if k != "_place_score"} for r in place_results[:10]],
    }

    output_path = get_output_path()
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # ================================================================
    # Print summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"WEIGHT OPTIMIZATION RESULTS - BATCH {BATCH}")
    print(f"{'='*70}")
    print(f"Win combos: {len(win_combos)}, Place combos: {len(place_combos)}")
    print(f"Sample: {sample_size} races, Elapsed: {total_time:.0f}s")

    print(f"\n{'='*70}")
    print(f"CURRENT WIN WEIGHTS")
    print(f"{'='*70}")
    for k, v in sorted(current_win_w.items()):
        print(f"  {k:<20} {v:.4f}")
    print_result("current", baseline_result)

    print(f"\n{'='*70}")
    print(f"BEST WIN WEIGHTS (Phase 1)")
    print(f"{'='*70}")
    for k, v in sorted(best_win_weights.items()):
        curr = current_win_w.get(k, 0)
        diff = v - curr
        if abs(diff) > 0.001 or v > 0.001:
            print(f"  {k:<20} {v:.4f}  (was {curr:.4f}, {diff:>+.4f})")
    print_result("best_win", best_win_result, baseline_result)

    print(f"\n{'='*70}")
    print(f"CURRENT PLACE WEIGHTS")
    print(f"{'='*70}")
    for k, v in sorted(current_place_w.items()):
        print(f"  {k:<20} {v:.4f}")

    print(f"\n{'='*70}")
    print(f"BEST PLACE WEIGHTS (Phase 2)")
    print(f"{'='*70}")
    for k, v in sorted(best_place_weights.items()):
        curr = current_place_w.get(k, 0)
        diff = v - curr
        if abs(diff) > 0.001 or v > 0.001:
            print(f"  {k:<20} {v:.4f}  (was {curr:.4f}, {diff:>+.4f})")
    print(f"\n  Place ROI: {best_place_result['place_roi']}% (was {baseline_result['place_roi']}%)")

    print(f"\n{'='*70}")
    print(f"TOP 5 WIN CONFIGS")
    print(f"{'='*70}")
    for i, r in enumerate(win_results[:5]):
        print(f"\n  #{i+1}: top1={r['top1_win_rate']}%, win_roi={r['win_roi']}%, place_roi={r['place_roi']}%")
        for k, v in sorted(r["weights"].items()):
            if v > 0.001:
                print(f"    {k}: {v:.4f}")

    print(f"\n{'='*70}")
    print(f"TOP 5 PLACE CONFIGS")
    print(f"{'='*70}")
    for i, r in enumerate(place_results[:5]):
        print(f"\n  #{i+1}: place_roi={r['place_roi']}%, win_roi={r['win_roi']}%, top1={r['top1_win_rate']}%")
        pw = r.get("place_weights") or r.get("weights", {})
        for k, v in sorted(pw.items()):
            if v > 0.001:
                print(f"    {k}: {v:.4f}")

    await bt_engine.dispose()


asyncio.run(main())
