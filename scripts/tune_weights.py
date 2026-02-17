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

OUTPUT_PATH = Path("data/optimal_weights.json")


def safe_div(a, b):
    return a / b if b else 0


# Weight search space — each factor gets a list of values to try.
# We normalize weights after, so absolute values don't matter — ratios do.
SEARCH_SPACE = {
    "market":        [0.30, 0.40, 0.50],
    "form":          [0.20, 0.30, 0.40],
    "jockey_trainer": [0.05, 0.08, 0.12],
    "horse_profile":  [0.02, 0.05],
    "class_fitness":  [0.02, 0.04],
    "weight_carried": [0.02, 0.04],
    "barrier":        [0.01, 0.03],
    "pace":           [0.01, 0.02],
    "movement":       [0.00, 0.03],
}

# Factors to hold fixed at 0 (deep_learning gets auto-added by probability engine)
FIXED_ZERO = ["deep_learning"]


def generate_weight_combos():
    """Generate all weight combinations from the search space."""
    factors = list(SEARCH_SPACE.keys())
    value_lists = [SEARCH_SPACE[f] for f in factors]
    combos = []
    for vals in product(*value_lists):
        weights = dict(zip(factors, vals))
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        combos.append(weights)
    return combos


async def evaluate_weights(weights, races_data):
    """Evaluate a weight configuration against pre-loaded race data.

    Returns dict with metrics: top1_win_rate, top3_win_rate, win_roi, place_roi, score.
    """
    from punty.probability import calculate_race_probabilities

    STAKE = 5
    top1_wins = 0
    top3_wins = 0
    win_pnl = 0.0
    place_pnl = 0.0
    total_races = 0

    for meeting, race, active, winners, placers in races_data:
        try:
            probs = calculate_race_probabilities(active, race, meeting, weights=weights)
        except Exception:
            continue

        if not probs:
            continue

        total_races += 1
        ranked = sorted(probs.items(), key=lambda x: -x[1].win_probability)

        # Top 1 accuracy
        winner_id = winners[0].id
        if ranked[0][0] == winner_id:
            top1_wins += 1

        # Top 3 accuracy
        top3_ids = set(rid for rid, _ in ranked[:3])
        if winner_id in top3_ids:
            top3_wins += 1

        # Win/place P&L on top 1 pick
        top1_runner = next((r for r in active if r.id == ranked[0][0]), None)
        if top1_runner:
            is_winner = top1_runner.finish_position == 1
            is_placed = top1_runner.id in placers
            win_div = top1_runner.win_dividend or 0
            place_div = top1_runner.place_dividend or 0

            if is_winner and win_div > 0:
                win_pnl += (win_div - 1) * STAKE
            else:
                win_pnl -= STAKE

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

    # Combined score: balance accuracy and profitability
    score = (0.3 * top1_rate) + (0.2 * top3_rate) + (0.25 * (win_roi + 1)) + (0.25 * (place_roi + 1))

    return {
        "weights": weights,
        "races": total_races,
        "top1_win_rate": round(top1_rate * 100, 2),
        "top3_win_rate": round(top3_rate * 100, 2),
        "win_roi": round(win_roi * 100, 2),
        "place_roi": round(place_roi * 100, 2),
        "score": round(score, 6),
    }


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

    # Pre-load all race data into memory to avoid repeated DB queries
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

    # Use a random sample for speed (full dataset would be too slow for grid search)
    import random
    random.seed(42)
    sample_size = min(3000, len(races_data))
    sample = random.sample(races_data, sample_size)
    logger.info(f"Using {sample_size} race sample for tuning")

    # Generate all weight combos
    combos = generate_weight_combos()
    logger.info(f"Testing {len(combos)} weight combinations...")

    # Get current default weights for comparison
    from punty.probability import DEFAULT_WEIGHTS
    current_weights = dict(DEFAULT_WEIGHTS)

    # Evaluate current weights first
    current_result = await evaluate_weights(current_weights, sample)
    logger.info(f"Current weights: score={current_result['score']:.6f}, "
                f"top1={current_result['top1_win_rate']}%, "
                f"win_roi={current_result['win_roi']}%, "
                f"place_roi={current_result['place_roi']}%")

    # Evaluate all combos
    best_result = current_result
    best_idx = -1
    results = []

    for i, weights in enumerate(combos):
        result = await evaluate_weights(weights, sample)
        if result:
            results.append(result)
            if result["score"] > best_result["score"]:
                best_result = result
                best_idx = i
                logger.info(f"  New best at combo {i}: score={result['score']:.6f}, "
                            f"top1={result['top1_win_rate']}%, "
                            f"win_roi={result['win_roi']}%, "
                            f"place_roi={result['place_roi']}%")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Tested {i + 1}/{len(combos)} combos ({elapsed:.0f}s)")

    # Sort results by score
    results.sort(key=lambda x: -x["score"])

    # Save results
    output = {
        "meta": {
            "total_combos": len(combos),
            "sample_size": sample_size,
            "total_races": len(races_data),
            "elapsed_seconds": round(time.time() - start_time, 1),
        },
        "current": current_result,
        "best": best_result,
        "improvement": {
            "top1_win_rate": round(best_result["top1_win_rate"] - current_result["top1_win_rate"], 2),
            "top3_win_rate": round(best_result["top3_win_rate"] - current_result["top3_win_rate"], 2),
            "win_roi": round(best_result["win_roi"] - current_result["win_roi"], 2),
            "place_roi": round(best_result["place_roi"] - current_result["place_roi"], 2),
            "score": round(best_result["score"] - current_result["score"], 6),
        },
        "top_10": results[:10],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"WEIGHT OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print(f"Tested {len(combos)} combos on {sample_size} race sample")
    print(f"Elapsed: {time.time() - start_time:.0f}s")

    print(f"\n{'='*70}")
    print(f"CURRENT WEIGHTS")
    print(f"{'='*70}")
    for k, v in sorted(current_weights.items()):
        print(f"  {k:<20} {v:.4f}")
    print(f"\n  Score: {current_result['score']:.6f}")
    print(f"  Top1: {current_result['top1_win_rate']}%, Top3: {current_result['top3_win_rate']}%")
    print(f"  Win ROI: {current_result['win_roi']}%, Place ROI: {current_result['place_roi']}%")

    print(f"\n{'='*70}")
    print(f"BEST WEIGHTS")
    print(f"{'='*70}")
    for k, v in sorted(best_result["weights"].items()):
        curr = current_weights.get(k, 0)
        diff = v - curr
        print(f"  {k:<20} {v:.4f}  (was {curr:.4f}, {diff:>+.4f})")
    print(f"\n  Score: {best_result['score']:.6f} ({output['improvement']['score']:+.6f})")
    print(f"  Top1: {best_result['top1_win_rate']}% ({output['improvement']['top1_win_rate']:+.2f}pp)")
    print(f"  Top3: {best_result['top3_win_rate']}% ({output['improvement']['top3_win_rate']:+.2f}pp)")
    print(f"  Win ROI: {best_result['win_roi']}% ({output['improvement']['win_roi']:+.2f}pp)")
    print(f"  Place ROI: {best_result['place_roi']}% ({output['improvement']['place_roi']:+.2f}pp)")

    print(f"\n{'='*70}")
    print(f"TOP 5 CONFIGURATIONS")
    print(f"{'='*70}")
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: score={r['score']:.6f}, top1={r['top1_win_rate']}%, "
              f"win_roi={r['win_roi']}%, place_roi={r['place_roi']}%")
        for k, v in sorted(r["weights"].items()):
            if v > 0.001:
                print(f"    {k}: {v:.4f}")

    await bt_engine.dispose()


asyncio.run(main())
