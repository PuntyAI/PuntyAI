"""Calibrate probability engine from historical race results.

Loads the calibration dataset (190K+ runners with actual results) and computes:
1. Empirical scoring curves — for each signal, what score should map to what actual win rate?
2. Optimal factor weights — which signals actually predict winners?
3. Per-context scoring curves — how does each signal's predictive power vary by track/distance/class?

The output replaces hand-coded scoring curves and guessed weights with data-driven parameters.

Usage:
    python scripts/run_calibration.py
    python scripts/run_calibration.py --dataset path/to/calibration_dataset.json
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Minimum runners required for a scoring curve bin
MIN_BIN_SIZE = 200
# Number of bins for scoring curves (vigintiles = 20)
N_BINS = 20
# Minimum context size for per-context curves
MIN_CONTEXT_SIZE = 300
# Minimum bin size within a context
MIN_CONTEXT_BIN = 30


# ── Core statistical functions ───────────────────────────────────────────────

def point_biserial_r(values: list[float], outcomes: list[bool]) -> float:
    """Point-biserial correlation between continuous signal and binary outcome.

    Returns correlation coefficient in [-1, 1]. Higher = more predictive of wins.
    Equivalent to Pearson r between the signal and a 0/1 indicator.
    """
    n = len(values)
    if n < 30:
        return 0.0

    n1 = sum(1 for o in outcomes if o)  # count of wins
    n0 = n - n1

    if n0 == 0 or n1 == 0:
        return 0.0

    # Mean of signal for winners vs non-winners
    sum1, sum0 = 0.0, 0.0
    for v, o in zip(values, outcomes):
        if o:
            sum1 += v
        else:
            sum0 += v

    mean1 = sum1 / n1
    mean0 = sum0 / n0

    # Overall standard deviation
    mean_all = (sum1 + sum0) / n
    ss = sum((v - mean_all) ** 2 for v in values)
    sd = math.sqrt(ss / n) if ss > 0 else 0.0

    if sd == 0:
        return 0.0

    # Point-biserial formula
    rpb = (mean1 - mean0) / sd * math.sqrt(n1 * n0 / (n * n))
    return max(-1.0, min(1.0, rpb))


def compute_scoring_curve(values_with_wins: list[tuple[float, bool]],
                          n_bins: int = N_BINS,
                          min_bin: int = MIN_BIN_SIZE) -> dict | None:
    """Compute empirical scoring curve from signal values and win outcomes.

    Bins runners into equal-size groups by signal value, computes actual win rate
    per bin, then normalises to a 0.05-0.95 score range.

    Returns:
        dict with "bins" (edges), "win_rates" (per bin), "scores" (normalised),
        "n_per_bin" (count), "total" (total runners).
        None if insufficient data.
    """
    if len(values_with_wins) < min_bin * n_bins // 2:
        return None

    # Sort by signal value
    sorted_vw = sorted(values_with_wins, key=lambda x: x[0])
    n = len(sorted_vw)

    # Adaptive binning: equal-count bins
    bin_size = n // n_bins
    if bin_size < min_bin // 2:
        # Reduce number of bins to maintain minimum bin size
        n_bins = max(5, n // min_bin)
        bin_size = n // n_bins

    bins = []
    win_rates = []
    n_per_bin = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n  # last bin gets remainder
        bin_data = sorted_vw[start:end]
        if not bin_data:
            continue

        bin_edge = bin_data[0][0]
        wins = sum(1 for _, w in bin_data if w)
        wr = wins / len(bin_data)

        bins.append(round(bin_edge, 6))
        win_rates.append(round(wr, 6))
        n_per_bin.append(len(bin_data))

    if len(bins) < 3:
        return None

    # Monotonic smoothing — enforce monotonically increasing win rates
    # Uses isotonic regression (pool adjacent violators algorithm)
    smoothed = _isotonic_regression(win_rates)

    # Normalise to 0.05-0.95 score range
    min_wr = min(smoothed)
    max_wr = max(smoothed)
    wr_range = max_wr - min_wr

    if wr_range < 0.005:
        # Signal has almost no predictive power — return flat curve
        scores = [0.50] * len(smoothed)
    else:
        scores = [round(0.05 + 0.90 * (wr - min_wr) / wr_range, 4) for wr in smoothed]

    # Add upper bound for interpolation
    upper_edge = sorted_vw[-1][0]
    bins.append(round(upper_edge, 6))

    return {
        "bins": bins,
        "win_rates": win_rates,
        "scores": scores,
        "n_per_bin": n_per_bin,
        "total": n,
        "min_wr": round(min_wr, 6),
        "max_wr": round(max_wr, 6),
        "spread": round(max_wr - min_wr, 6),
    }


def _isotonic_regression(values: list[float]) -> list[float]:
    """Pool Adjacent Violators Algorithm for monotonic non-decreasing fit.

    Ensures the output win rates are monotonically increasing, which is required
    for a valid scoring curve (higher signal ->higher win probability).
    """
    n = len(values)
    result = list(values)
    # Weighted pool adjacent violators
    weights = [1.0] * n

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Pool: merge i and i+1
            total_w = weights[i] + weights[i + 1]
            pooled = (result[i] * weights[i] + result[i + 1] * weights[i + 1]) / total_w
            result[i] = pooled
            result[i + 1] = pooled
            weights[i] = total_w
            weights[i + 1] = total_w
            # Back-track to check previous elements
            while i > 0 and result[i - 1] > result[i]:
                total_w = weights[i - 1] + weights[i]
                pooled = (result[i - 1] * weights[i - 1] + result[i] * weights[i]) / total_w
                result[i - 1] = pooled
                result[i] = pooled
                weights[i - 1] = total_w
                weights[i] = total_w
                i -= 1
        i += 1

    return [round(v, 6) for v in result]


def compute_optimal_weights(signal_correlations: dict[str, float],
                            temperature: float = 3.0) -> dict[str, float]:
    """Compute optimal factor weights from signal correlations using softmax.

    Higher correlation ->higher weight. Temperature controls spread:
    - Low temp (1.0) = very concentrated on best signals
    - High temp (5.0) = more even distribution
    - Default 3.0 balances predictive power with diversification
    """
    if not signal_correlations:
        return {}

    # Use absolute correlation (some signals are inverted)
    abs_corrs = {k: abs(v) for k, v in signal_correlations.items() if v != 0}
    if not abs_corrs:
        return {}

    # Softmax normalisation
    max_c = max(abs_corrs.values())
    exp_vals = {k: math.exp((v / max_c) * temperature) for k, v in abs_corrs.items()}
    total = sum(exp_vals.values())

    weights = {k: round(v / total, 4) for k, v in exp_vals.items()}
    return weights


# ── Main calibration ─────────────────────────────────────────────────────────

def run_calibration(dataset: list[dict]) -> dict:
    """Run full calibration analysis on the dataset."""

    print(f"Calibrating from {len(dataset):,} runners...")
    start = time.time()

    # ── STEP 1: Global scoring curves ────────────────────────────────────

    # Define which signals to calibrate and how they map to factors
    SIGNAL_DEFS = {
        # Factor: market
        "market_prob": {"factor": "market", "direction": "higher_better"},

        # Factor: form (multiple sub-signals)
        "career_win_pct": {"factor": "form", "direction": "higher_better"},
        "career_place_pct": {"factor": "form", "direction": "higher_better"},
        "track_dist_sr": {"factor": "form", "direction": "higher_better", "min_field": "track_dist_starts", "min_val": 3},
        "distance_sr": {"factor": "form", "direction": "higher_better", "min_field": "distance_starts", "min_val": 3},
        "track_sr": {"factor": "form", "direction": "higher_better", "min_field": "track_starts", "min_val": 3},
        "cond_good_sr": {"factor": "form", "direction": "higher_better", "min_field": "cond_good_starts", "min_val": 3},
        "cond_soft_sr": {"factor": "form", "direction": "higher_better", "min_field": "cond_soft_starts", "min_val": 3},
        "cond_heavy_sr": {"factor": "form", "direction": "higher_better", "min_field": "cond_heavy_starts", "min_val": 3},
        "first_up_sr": {"factor": "form", "direction": "higher_better", "min_field": "first_up_starts", "min_val": 3},
        "second_up_sr": {"factor": "form", "direction": "higher_better", "min_field": "second_up_starts", "min_val": 3},
        "last5_score": {"factor": "form", "direction": "higher_better"},
        "last5_wins": {"factor": "form", "direction": "higher_better"},

        # Factor: class_fitness
        "prize_per_start": {"factor": "class_fitness", "direction": "higher_better"},
        "avg_margin": {"factor": "class_fitness", "direction": "lower_better"},
        "days_since_last": {"factor": "class_fitness", "direction": "custom"},

        # Factor: pace
        "settle_pos": {"factor": "pace", "direction": "lower_better"},

        # Factor: barrier
        "barrier_relative": {"factor": "barrier", "direction": "lower_better"},

        # Factor: jockey_trainer
        "jockey_career_sr": {"factor": "jockey_trainer", "direction": "higher_better"},
        "jockey_career_a2e": {"factor": "jockey_trainer", "direction": "higher_better"},
        "jockey_l100_sr": {"factor": "jockey_trainer", "direction": "higher_better"},
        "trainer_career_sr": {"factor": "jockey_trainer", "direction": "higher_better"},
        "trainer_l100_sr": {"factor": "jockey_trainer", "direction": "higher_better"},
        "combo_career_sr": {"factor": "jockey_trainer", "direction": "higher_better"},
        "combo_l100_sr": {"factor": "jockey_trainer", "direction": "higher_better"},

        # Factor: weight_carried
        "weight_diff": {"factor": "weight_carried", "direction": "custom"},

        # Factor: horse_profile
        "age": {"factor": "horse_profile", "direction": "custom"},

        # Factor: movement
        "price_move_pct": {"factor": "movement", "direction": "higher_better"},
    }

    print("\n" + "=" * 80)
    print("STEP 1: GLOBAL SCORING CURVES")
    print("=" * 80)

    scoring_curves = {}
    signal_stats = {}

    for signal_name, defn in SIGNAL_DEFS.items():
        min_field = defn.get("min_field")
        min_val = defn.get("min_val", 0)

        # Collect signal values paired with win outcomes
        vw = []
        for r in dataset:
            val = r.get(signal_name)
            if val is None:
                continue
            # Apply minimum starts filter
            if min_field and r.get(min_field, 0) < min_val:
                continue
            vw.append((float(val), r["won"]))

        if len(vw) < 1000:
            print(f"  {signal_name:25s}: SKIPPED ({len(vw):,} runners, need 1000+)")
            continue

        # Compute correlation
        vals, wins = zip(*vw)
        corr = point_biserial_r(list(vals), list(wins))

        # Compute scoring curve
        curve = compute_scoring_curve(vw, n_bins=N_BINS, min_bin=MIN_BIN_SIZE)
        if curve:
            scoring_curves[signal_name] = curve
            signal_stats[signal_name] = {
                "correlation": round(corr, 4),
                "n": len(vw),
                "factor": defn["factor"],
                "spread": curve["spread"],
                "min_wr": curve["min_wr"],
                "max_wr": curve["max_wr"],
            }

            direction_char = "+" if corr > 0 else "-"
            print(f"  {signal_name:25s}: r={corr:+.4f} {direction_char}  "
                  f"spread={curve['spread']*100:5.1f}%  "
                  f"WR range [{curve['min_wr']*100:.1f}%-{curve['max_wr']*100:.1f}%]  "
                  f"n={len(vw):,}")
        else:
            print(f"  {signal_name:25s}: r={corr:+.4f}  INSUFFICIENT BINS  n={len(vw):,}")

    # ── STEP 2: Factor-level correlations and optimal weights ────────────

    print("\n" + "=" * 80)
    print("STEP 2: OPTIMAL FACTOR WEIGHTS")
    print("=" * 80)

    # For each factor, compute a composite correlation
    # Use the best sub-signal correlation as the factor's representative
    factor_signals = defaultdict(list)
    for sig_name, stats in signal_stats.items():
        factor_signals[stats["factor"]].append((sig_name, stats["correlation"], stats["spread"]))

    factor_correlations = {}
    print("\nBest signal per factor:")
    for factor, sigs in sorted(factor_signals.items()):
        # Weight by both correlation strength and win rate spread
        # This gives a balanced view — a signal that's both correlated AND has wide spread
        best = max(sigs, key=lambda x: abs(x[1]) * (1 + x[2] * 10))
        factor_correlations[factor] = best[1]
        print(f"  {factor:20s}: best={best[0]:25s}  r={best[1]:+.4f}  "
              f"spread={best[2]*100:.1f}%")
        for sig_name, corr, spread in sorted(sigs, key=lambda x: -abs(x[1])):
            if sig_name != best[0]:
                print(f"  {'':20s}  also={sig_name:25s}  r={corr:+.4f}  spread={spread*100:.1f}%")

    # Compute optimal weights using softmax at different temperatures
    print("\nOptimal weights (softmax temperatures):")
    for temp in [2.0, 3.0, 5.0]:
        weights = compute_optimal_weights(factor_correlations, temperature=temp)
        print(f"  temp={temp}: {json.dumps({k: round(v*100, 1) for k, v in sorted(weights.items(), key=lambda x: -x[1])})}")

    # Use temperature 3.0 as default (balanced)
    optimal_weights = compute_optimal_weights(factor_correlations, temperature=3.0)
    print(f"\n  Selected (temp=3.0): {json.dumps({k: round(v*100, 1) for k, v in sorted(optimal_weights.items(), key=lambda x: -x[1])})}")

    # ── STEP 3: Per-context scoring curves ───────────────────────────────

    print("\n" + "=" * 80)
    print("STEP 3: PER-CONTEXT CALIBRATION")
    print("=" * 80)

    # Group by context
    context_groups = defaultdict(list)
    for r in dataset:
        # Per-track context
        track_key = f"{r['track']}|{r['dist_bucket']}|{r['class_bucket']}"
        context_groups[track_key].append(r)
        # Venue-type context
        vtype_key = f"{r['venue_type']}|{r['dist_bucket']}|{r['class_bucket']}"
        context_groups[vtype_key].append(r)
        # Distance+class fallback
        dc_key = f"{r['dist_bucket']}|{r['class_bucket']}"
        context_groups[dc_key].append(r)

    # Key signals to calibrate per context (the most impactful ones)
    CONTEXT_SIGNALS = [
        "market_prob", "career_win_pct", "jockey_career_sr", "trainer_career_sr",
        "barrier_relative", "settle_pos", "weight_diff", "prize_per_start",
        "last5_score", "price_move_pct",
    ]

    context_curves = {}
    context_weights = {}
    context_count = 0

    for ctx_key, ctx_records in sorted(context_groups.items(), key=lambda x: -len(x[1])):
        if len(ctx_records) < MIN_CONTEXT_SIZE:
            continue

        ctx_scoring = {}
        ctx_corrs = {}

        for signal_name in CONTEXT_SIGNALS:
            vw = [(float(r[signal_name]), r["won"])
                  for r in ctx_records
                  if r.get(signal_name) is not None]

            if len(vw) < MIN_CONTEXT_BIN * 5:
                continue

            # Smaller bin count for context-specific curves
            ctx_n_bins = min(10, len(vw) // MIN_CONTEXT_BIN)
            if ctx_n_bins < 3:
                continue

            curve = compute_scoring_curve(vw, n_bins=ctx_n_bins, min_bin=MIN_CONTEXT_BIN)
            if curve and curve["spread"] > 0.005:
                ctx_scoring[signal_name] = curve

            # Context-specific correlation
            vals, wins = zip(*vw)
            corr = point_biserial_r(list(vals), list(wins))
            if abs(corr) > 0.01:
                # Map signal to factor
                factor = SIGNAL_DEFS.get(signal_name, {}).get("factor", signal_name)
                if factor not in ctx_corrs or abs(corr) > abs(ctx_corrs[factor]):
                    ctx_corrs[factor] = corr

        if ctx_scoring:
            context_curves[ctx_key] = ctx_scoring
            context_count += 1

        if ctx_corrs:
            ctx_weights = compute_optimal_weights(ctx_corrs, temperature=3.0)
            if ctx_weights:
                context_weights[ctx_key] = ctx_weights

    print(f"\nGenerated context-specific curves: {context_count}")
    print(f"Generated context-specific weights: {len(context_weights)}")

    # Show sample contexts
    sample_contexts = [k for k in context_weights.keys()
                       if k.count("|") == 2 and not k.startswith(("metro_", "provincial", "country"))]
    print(f"\nSample per-track calibrations (of {len(sample_contexts)} total):")
    for ctx in sorted(sample_contexts)[:15]:
        n = len(context_groups[ctx])
        wr = sum(1 for r in context_groups[ctx] if r["won"]) / n * 100
        w = context_weights.get(ctx, {})
        top3 = sorted(w.items(), key=lambda x: -x[1])[:3]
        w_str = ", ".join(f"{k}={v*100:.0f}%" for k, v in top3)
        print(f"  {ctx:40s} (n={n:>5,}, WR={wr:4.1f}%): {w_str}")

    # ── STEP 4: Compile results ──────────────────────────────────────────

    print("\n" + "=" * 80)
    print("STEP 4: FINAL PARAMETERS")
    print("=" * 80)

    # Determine the primary signal for each factor (used in probability.py)
    primary_signals = {}
    for factor, sigs in factor_signals.items():
        best = max(sigs, key=lambda x: abs(x[1]) * (1 + x[2] * 10))
        primary_signals[factor] = best[0]

    print(f"\nPrimary signals per factor:")
    for factor, sig in sorted(primary_signals.items()):
        print(f"  {factor:20s} ->{sig}")

    # Compare current vs calibrated weights
    CURRENT_WEIGHTS = {
        "market": 0.22, "movement": 0.07, "form": 0.15, "class_fitness": 0.05,
        "pace": 0.11, "barrier": 0.09, "jockey_trainer": 0.11,
        "weight_carried": 0.05, "horse_profile": 0.05, "deep_learning": 0.10,
    }
    print(f"\nWeight comparison (current ->calibrated):")
    for factor in sorted(CURRENT_WEIGHTS.keys()):
        curr = CURRENT_WEIGHTS[factor] * 100
        cal = optimal_weights.get(factor, 0) * 100
        delta = cal - curr
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"  {factor:20s}: {curr:5.1f}% ->{cal:5.1f}% ({arrow}{abs(delta):+.1f}%)")

    # Show scoring curve spread comparison
    print(f"\nScoring curve spreads (win rate range per signal):")
    for sig_name, curve in sorted(scoring_curves.items(), key=lambda x: -x[1]["spread"]):
        stats = signal_stats[sig_name]
        print(f"  {sig_name:25s}: {curve['min_wr']*100:5.1f}% ->{curve['max_wr']*100:5.1f}%  "
              f"(spread={curve['spread']*100:.1f}%,  score range 0.05-0.95)  "
              f"r={stats['correlation']:+.4f}")

    elapsed = time.time() - start
    print(f"\nCalibration completed in {elapsed:.1f}s")

    result = {
        "weights": optimal_weights,
        "scoring_curves": scoring_curves,
        "signal_stats": signal_stats,
        "primary_signals": primary_signals,
        "factor_correlations": {k: round(v, 4) for k, v in factor_correlations.items()},
        "context_weights": context_weights,
        "context_curves": context_curves,
        "metadata": {
            "runners": len(dataset),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "win_rate": round(sum(1 for r in dataset if r["won"]) / len(dataset), 6),
            "n_bins": N_BINS,
            "min_bin": MIN_BIN_SIZE,
            "min_context": MIN_CONTEXT_SIZE,
            "n_scoring_curves": len(scoring_curves),
            "n_context_curves": context_count,
            "n_context_weights": len(context_weights),
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Run calibration analysis on historical dataset")
    parser.add_argument("--dataset", default=str(
        Path(__file__).resolve().parent.parent / "punty" / "data" / "calibration_dataset.json"
    ))
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parent.parent / "punty" / "data" / "calibrated_params.json"
    ))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run build_calibration.py first.")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    load_start = time.time()
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset):,} records in {time.time() - load_start:.1f}s")

    result = run_calibration(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Don't include context_curves in the main output (too large)
    # Save separately if needed
    main_output = {k: v for k, v in result.items() if k != "context_curves"}

    with open(output_path, "w") as f:
        json.dump(main_output, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nSaved to {output_path} ({size_kb:.0f} KB)")

    # Save context curves separately
    ctx_output_path = output_path.parent / "calibrated_context_curves.json"
    with open(ctx_output_path, "w") as f:
        json.dump(result["context_curves"], f)
    ctx_size = ctx_output_path.stat().st_size / (1024 * 1024)
    print(f"Context curves saved to {ctx_output_path} ({ctx_size:.1f} MB)")


if __name__ == "__main__":
    main()
