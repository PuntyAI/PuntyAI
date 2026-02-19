"""Build output-level isotonic calibration for win probabilities.

Recalculates win probabilities for ALL runners in settled races,
compares predicted vs actual win rates, and builds a monotonic
calibration curve using Pool Adjacent Violators (isotonic regression).

Usage:
    python scripts/calibrate_output.py

Output is saved to punty/data/calibrated_params.json under
the "output_calibration" key.
"""

import asyncio
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_BINS = 20  # vigintiles


def isotonic_regression(values: list[float]) -> list[float]:
    """Pool Adjacent Violators algorithm for isotonic (non-decreasing) regression."""
    n = len(values)
    if n <= 1:
        return list(values)
    result = list(values)
    # Iteratively merge adjacent blocks that violate monotonicity
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 1:
            if result[i] > result[i + 1]:
                # Pool: replace both with their average
                avg = (result[i] + result[i + 1]) / 2
                result[i] = avg
                result[i + 1] = avg
                changed = True
            i += 1
    return result


async def main():
    from punty.models.database import async_session, init_db
    from punty.models.meeting import Meeting, Race, Runner
    from punty.probability import calculate_race_probabilities
    from sqlalchemy import select

    await init_db()

    all_predictions = []  # (predicted_win_prob, actually_won)

    async with async_session() as db:
        # Get all races with final results
        races = await db.execute(
            select(Race).where(
                Race.results_status.in_(["Paying", "Closed", "Final"])
            )
        )
        races = races.scalars().all()
        print(f"Found {len(races)} settled races")

        # Group races by meeting
        race_by_meeting = defaultdict(list)
        for race in races:
            race_by_meeting[race.meeting_id].append(race)

        meetings_result = await db.execute(select(Meeting))
        meetings = {m.id: m for m in meetings_result.scalars().all()}

        processed = 0
        for meeting_id, meeting_races in race_by_meeting.items():
            meeting = meetings.get(meeting_id)
            if not meeting:
                continue

            for race in meeting_races:
                runners_result = await db.execute(
                    select(Runner).where(Runner.race_id == race.id)
                )
                runners = runners_result.scalars().all()
                active = [r for r in runners if not r.scratched]
                if len(active) < 3:
                    continue

                # Check we have a winner
                winner = [r for r in active if r.finish_position == 1]
                if not winner:
                    continue

                # Calculate probabilities
                try:
                    probs = calculate_race_probabilities(runners, race, meeting)
                except Exception as e:
                    continue

                for runner in active:
                    rp = probs.get(runner.id)
                    if not rp:
                        continue
                    won = runner.finish_position == 1
                    all_predictions.append((rp.win_probability, won))

                processed += 1
                if processed % 50 == 0:
                    print(f"  Processed {processed} races...")

    print(f"\nTotal predictions: {len(all_predictions)}")
    print(f"Total winners: {sum(1 for _, w in all_predictions if w)}")

    if len(all_predictions) < 100:
        print("ERROR: Not enough data for calibration")
        return

    # Sort by predicted probability
    all_predictions.sort(key=lambda x: x[0])

    # Bin into N_BINS groups
    bin_size = len(all_predictions) // N_BINS
    bins = []
    win_rates = []

    for i in range(N_BINS):
        start = i * bin_size
        end = start + bin_size if i < N_BINS - 1 else len(all_predictions)
        group = all_predictions[start:end]
        if not group:
            continue

        avg_pred = sum(p for p, _ in group) / len(group)
        actual_wr = sum(1 for _, w in group if w) / len(group)
        bins.append(avg_pred)
        win_rates.append(actual_wr)

        print(f"  Bin {i+1:2d}: pred={avg_pred:.4f} actual={actual_wr:.4f} n={len(group)}")

    # Apply isotonic regression for monotonicity
    calibrated = isotonic_regression(win_rates)

    print("\n=== CALIBRATION CURVE ===")
    print(f"{'Bin':>4} {'Predicted':>10} {'Actual':>10} {'Calibrated':>12}")
    for i in range(len(bins)):
        marker = " *" if abs(win_rates[i] - calibrated[i]) > 0.01 else ""
        print(f"  {i+1:2d}   {bins[i]:10.4f}  {win_rates[i]:10.4f}  {calibrated[i]:12.4f}{marker}")

    # Check monotonicity
    is_monotonic = all(calibrated[i] <= calibrated[i+1] for i in range(len(calibrated)-1))
    print(f"\nMonotonic: {'YES' if is_monotonic else 'NO'}")

    # Save to calibrated_params.json
    params_path = Path(__file__).parent.parent / "punty" / "data" / "calibrated_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        params = {}

    params["output_calibration"] = {
        "bins": [round(b, 6) for b in bins],
        "calibrated": [round(c, 6) for c in calibrated],
        "n_predictions": len(all_predictions),
        "n_races": processed,
    }

    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {params_path}")


if __name__ == "__main__":
    asyncio.run(main())
