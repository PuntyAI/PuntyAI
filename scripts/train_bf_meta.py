#!/usr/bin/env python3
"""Train Betfair meta-model — a learned bet selector for LGBM rank 1 picks.

Loads Proform historical data, runs the main LGBM to get WP predictions for
each race, picks the rank 1 runner, and trains a binary classifier on whether
that pick actually placed. Sample weights = place_dividend for placed runners
so the model optimises ROI, not just strike rate.

Usage:
    python scripts/train_bf_meta.py
    python scripts/train_bf_meta.py --data-dir D:\\Punty\\DatafromProform
    python scripts/train_bf_meta.py --threshold 0.65
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_lgbm_v2 import (
    load_proform_data,
    extract_features_proform,
    DEFAULT_DATA_DIR,
    MONTH_DIRS,
    _score_last5,
    _parse_last10,
    _form_trend_proform,
    _sf,
    _parse_settling,
    _parse_flucs,
)
from punty.ml.features import (
    FEATURE_NAMES, NUM_FEATURES,
    _distance_bucket, _track_cond_bucket, _class_bucket, _venue_type_code,
)
from punty.betting.meta_model import (
    META_FEATURE_NAMES, NUM_META_FEATURES, META_MODEL_PATH, META_METADATA_PATH,
)

MODEL_DIR = ROOT / "punty" / "data"

# Meta-model LightGBM params — small, fast binary classifier
META_LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "is_unbalanced": False,  # We use sample weights instead
}

EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 500  # Small model — 50-100 trees typical


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Convert LambdaRank scores to probabilities."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def load_proform_races_with_dividends(data_dir: Path) -> list[dict]:
    """Load Proform data grouped by race, including dividends for weighting.

    Returns list of race dicts, each with:
        - meta: race metadata
        - runners: list of runner dicts with Proform data
        - features: list of feature vectors (aligned with runners)
    """
    print(f"Loading Proform data with dividends from {data_dir}...")
    start = time.time()

    races = []
    total_runners = 0
    total_files = 0
    skipped = 0

    # Process year directories (same dedup logic as train_lgbm_v2)
    year_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if len(year_dirs) > 1:
        dir_file_counts = {}
        for yd in year_dirs:
            count = sum(1 for m in range(1, 13)
                        if (yd / MONTH_DIRS[m] / "Form").exists()
                        for _ in (yd / MONTH_DIRS[m] / "Form").glob("*.json"))
            dir_file_counts[yd.name] = count

        seen_counts = set()
        deduped = []
        priority = sorted(year_dirs, key=lambda d: (d.name != "2025", d.name))
        for yd in priority:
            cnt = dir_file_counts[yd.name]
            if cnt in seen_counts and cnt > 0:
                continue
            seen_counts.add(cnt)
            deduped.append(yd)
        year_dirs = deduped

    for year_dir in year_dirs:
        year = year_dir.name
        print(f"\n  Processing {year}...")

        for month_num in range(1, 13):
            month_name = MONTH_DIRS[month_num]
            meetings_path = year_dir / month_name / "meetings.json"
            race_meta = {}
            if meetings_path.exists():
                with open(meetings_path, "r", encoding="utf-8") as f:
                    meetings = json.load(f)
                for m in meetings:
                    md = m.get("MeetingDate", "")
                    venue = m.get("Track", {}).get("Name", "")
                    state = m.get("Track", {}).get("State", "")
                    for race in m.get("Races", []):
                        rid = race.get("RaceId")
                        try:
                            rid = int(rid)
                        except (ValueError, TypeError):
                            continue
                        race_meta[rid] = {
                            "distance": race.get("Distance", 1400),
                            "race_class": (race.get("RaceClass", "") or "").rstrip(";").strip(),
                            "venue": venue,
                            "state": state,
                            "condition": race.get("TrackCondition", "") or "",
                            "field_size": race.get("Starters", 12),
                            "date": md[:10] if md else "",
                            "prize_money": race.get("PrizeMoney", 0),
                        }

            form_dir = year_dir / month_name / "Form"
            if not form_dir.exists():
                continue

            month_races = 0
            for fpath in sorted(form_dir.glob("*.json")):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        runners = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(runners, list):
                    continue
                total_files += 1

                # Group by race
                race_groups = defaultdict(list)
                for r in runners:
                    rid = r.get("RaceId")
                    if rid:
                        try:
                            race_groups[int(rid)].append(r)
                        except (ValueError, TypeError):
                            pass

                for race_id, race_runners in race_groups.items():
                    meta = race_meta.get(race_id, {})
                    if not meta:
                        continue

                    field_size = len(race_runners)
                    meta["field_size"] = field_size

                    # Need at least 5 runners for place betting
                    if field_size < 5:
                        continue

                    weights = [r.get("Weight", 0) for r in race_runners
                               if r.get("Weight", 0) and r["Weight"] > 40]
                    avg_weight = statistics.mean(weights) if weights else 56.0

                    # Infer TrackCondition from runner Forms if not in meetings.json
                    if not meta.get("condition"):
                        for r in race_runners:
                            forms = [f for f in r.get("Forms", []) if not f.get("IsBarrierTrial")]
                            if forms:
                                tc = forms[0].get("TrackCondition", "")
                                if tc:
                                    meta["condition"] = tc
                                    break

                    # Filter: all runners need position and SP
                    valid_runners = []
                    valid_features = []
                    for r in race_runners:
                        pos = r.get("Position")
                        sp = r.get("PriceSP", 0)
                        if pos is None or pos == 0 or not sp or sp <= 1.0:
                            skipped += 1
                            continue

                        features = extract_features_proform(r, meta, field_size, avg_weight)
                        if len(features) != NUM_FEATURES:
                            continue

                        valid_runners.append(r)
                        valid_features.append(features)

                    if len(valid_runners) < 3:
                        continue  # Need enough runners for meaningful ranking

                    races.append({
                        "meta": meta,
                        "runners": valid_runners,
                        "features": valid_features,
                        "field_size": field_size,
                        "avg_weight": avg_weight,
                    })
                    total_runners += len(valid_runners)
                    month_races += 1

            if month_races > 0:
                print(f"    {month_name}: {month_races:,} races")

    elapsed = time.time() - start
    print(f"\nTotal: {len(races):,} races, {total_runners:,} runners in {elapsed:.1f}s")
    print(f"  Skipped (no position/SP): {skipped:,}")
    return races


def extract_meta_features_from_proform(
    runner: dict,
    meta: dict,
    wp: float,
    wp_margin: float,
    field_size: int,
    avg_weight: float,
) -> list[float]:
    """Extract the 18 meta-features for a Proform runner."""
    nan = float("nan")

    # Market odds
    sp = runner.get("PriceSP", 0)
    odds = float(sp) if sp and sp > 1.0 else nan
    market_implied = 1.0 / odds if odds and odds > 1.0 else 0.0

    # Distance/class/condition buckets
    distance = meta.get("distance", 1400)
    dist_bucket = _distance_bucket(distance)
    cls_bucket = _class_bucket(meta.get("race_class", ""))
    tc_bucket = _track_cond_bucket(meta.get("condition", ""))
    venue_type = _venue_type_code(meta.get("venue", ""))

    # Barrier
    barrier = runner.get("Barrier", 0) or 0
    barrier_relative = (barrier - 1) / (field_size - 1) if barrier and field_size > 1 else nan

    # Age
    age = _sf(runner.get("Age"))

    # Days since
    forms = runner.get("Forms", [])
    real_forms = [f for f in forms if not f.get("IsBarrierTrial")]
    days_since = nan
    if real_forms:
        try:
            from datetime import datetime
            last_date = real_forms[0].get("MeetingDate", "")
            race_date = meta.get("date", "")
            if last_date and race_date:
                ld = datetime.fromisoformat(last_date.replace("T00:00:00", ""))
                rd = datetime.strptime(race_date[:10], "%Y-%m-%d")
                days_since = float((rd - ld).days)
        except (ValueError, TypeError):
            pass

    # Form score and trend
    last10 = runner.get("Last10", "")
    positions = _parse_last10(last10)
    form_score = _sf(_score_last5(positions))
    form_trend = _sf(_form_trend_proform(positions))

    # Value rating = WP / market implied probability
    value_rating = wp / market_implied if market_implied > 0 else nan

    # Speed map position from settle
    settle_vals = []
    for f in real_forms[:5]:
        s = _parse_settling(f.get("InRun", ""), field_size)
        if s is not None:
            settle_vals.append(s)
    settle_pos = statistics.mean(settle_vals) if settle_vals else nan
    if not math.isnan(settle_pos):
        if settle_pos <= 0.15:
            speed_map_pos = 1.0  # leader
        elif settle_pos <= 0.35:
            speed_map_pos = 2.0  # on_pace
        elif settle_pos <= 0.65:
            speed_map_pos = 3.0  # midfield
        else:
            speed_map_pos = 4.0  # backmarker
    else:
        speed_map_pos = nan

    # Weight diff
    weight = runner.get("Weight", 0) or 0
    weight_diff = float(weight) - avg_weight if weight and weight > 40 and avg_weight else nan

    # Career stats
    win_pct = runner.get("WinPct", 0)
    place_pct = runner.get("PlacePct", 0)
    career_win_pct = win_pct / 100 if win_pct else nan
    career_place_pct = place_pct / 100 if place_pct else nan

    return [
        _sf(wp),
        _sf(wp_margin),
        _sf(odds),
        float(field_size),
        dist_bucket,
        cls_bucket,
        tc_bucket,
        venue_type,
        _sf(barrier_relative),
        age,
        _sf(days_since),
        form_score,
        form_trend,
        _sf(value_rating),
        _sf(speed_map_pos),
        _sf(weight_diff),
        career_win_pct,
        career_place_pct,
    ]


def build_meta_dataset(races: list[dict], rank_model) -> tuple:
    """Run LGBM rank model on each race, pick rank 1, extract meta-features.

    Returns:
        X_meta: np.ndarray of shape (n_races, NUM_META_FEATURES)
        y_meta: np.ndarray of shape (n_races,) — 1 if placed, 0 if not
        weights: np.ndarray of shape (n_races,) — place_dividend for placed, 1.0 for not
        dates: list of race dates
        odds_list: list of SP odds for rank 1 pick
    """
    print("\nBuilding meta-dataset from LGBM rank 1 picks...")
    start = time.time()

    X_meta = []
    y_meta = []
    sample_weights = []
    dates = []
    odds_list = []
    skipped_small_field = 0

    for race in races:
        meta = race["meta"]
        runners = race["runners"]
        features = race["features"]
        field_size = race["field_size"]
        avg_weight = race["avg_weight"]

        if len(features) < 3:
            continue

        # Run LGBM rank model to get scores
        X = np.array(features, dtype=np.float64)
        model_n_features = rank_model.num_feature()
        if X.shape[1] > model_n_features:
            X = X[:, :model_n_features]

        raw_scores = rank_model.predict(X)

        # Softmax to get win probabilities
        shifted = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(shifted)
        win_probs = exp_scores / np.sum(exp_scores)

        # Find rank 1 (highest WP)
        rank1_idx = np.argmax(win_probs)
        rank1_wp = float(win_probs[rank1_idx])

        # WP margin (gap to rank 2)
        sorted_probs = sorted(win_probs, reverse=True)
        wp_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        # Rank 1 runner's data
        rank1_runner = runners[rank1_idx]

        # Label: did rank 1 place?
        pos = rank1_runner.get("Position", 99)
        place_cutoff = 2 if field_size <= 7 else 3
        placed = 1 if pos <= place_cutoff else 0

        # Sample weight: place dividend for placed, 1.0 for not
        # This optimises ROI — high-dividend places get higher weight
        place_div = rank1_runner.get("PlaceDividend", 0) or 0
        if placed and place_div and place_div > 0:
            weight = float(place_div)
        else:
            weight = 1.0

        # Extract meta-features
        mf = extract_meta_features_from_proform(
            rank1_runner, meta, rank1_wp, wp_margin,
            field_size, avg_weight,
        )

        if len(mf) != NUM_META_FEATURES:
            continue

        X_meta.append(mf)
        y_meta.append(placed)
        sample_weights.append(weight)
        dates.append(meta.get("date", ""))
        odds_list.append(rank1_runner.get("PriceSP", 0) or 0)

    elapsed = time.time() - start
    placed_count = sum(y_meta)
    print(f"  Meta-dataset: {len(y_meta):,} races in {elapsed:.1f}s")
    print(f"  Placed: {placed_count:,} ({placed_count/len(y_meta)*100:.1f}%)")
    print(f"  Not placed: {len(y_meta) - placed_count:,}")

    return (
        np.array(X_meta, dtype=np.float64),
        np.array(y_meta, dtype=np.int32),
        np.array(sample_weights, dtype=np.float64),
        dates,
        odds_list,
    )


def train_meta_model(
    X_train, y_train, w_train,
    X_val, y_val, w_val,
) -> lgb.Booster:
    """Train the meta-model binary classifier."""
    print(f"\n{'='*60}")
    print("Training Betfair meta-model")
    n_pos = y_train.sum()
    print(f"  Train: {len(y_train):,} races, {n_pos:,} placed ({n_pos/len(y_train)*100:.1f}%)")
    n_vpos = y_val.sum()
    print(f"  Val:   {len(y_val):,} races, {n_vpos:,} placed ({n_vpos/len(y_val)*100:.1f}%)")
    print(f"{'='*60}")

    train_data = lgb.Dataset(
        X_train, label=y_train, weight=w_train,
        feature_name=META_FEATURE_NAMES, free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, weight=w_val,
        feature_name=META_FEATURE_NAMES, free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        META_LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Trees: {model.num_trees()}")
    return model


def evaluate_meta_model(
    model, X, y, odds, label: str,
    thresholds: list[float] | None = None,
) -> dict:
    """Evaluate meta-model at various probability thresholds.

    Reports: place SR, ROI, bet count for each threshold.
    Also compares with flat WP thresholds.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    probs = model.predict(X)
    odds_arr = np.array(odds, dtype=np.float64)

    print(f"\n  {label} — Meta-model threshold analysis:")
    print(f"  {'Threshold':>10s} {'Bets':>6s} {'Placed':>7s} {'SR%':>6s} {'ROI%':>7s} {'P&L':>10s}")
    print(f"  {'-'*50}")

    best_result = {}
    best_roi = -999.0

    for threshold in thresholds:
        mask = probs >= threshold
        n_bets = mask.sum()
        if n_bets == 0:
            print(f"  {threshold:>10.2f} {0:>6d} {0:>7d} {'N/A':>6s} {'N/A':>7s} {'N/A':>10s}")
            continue

        placed = y[mask].sum()
        sr = placed / n_bets * 100

        # ROI calculation: $1 flat stake, place dividend return
        # Simple model: odds / 3 as approximate place odds
        bet_odds = odds_arr[mask]
        placed_mask = y[mask] == 1
        returns = np.where(placed_mask, bet_odds / 3.0, 0.0)
        pnl = returns.sum() - n_bets
        roi = pnl / n_bets * 100

        print(f"  {threshold:>10.2f} {n_bets:>6d} {placed:>7d} {sr:>5.1f}% {roi:>+6.1f}% {pnl:>+9.1f}")

        if roi > best_roi:
            best_roi = roi
            best_result = {
                "threshold": threshold,
                "bets": int(n_bets),
                "placed": int(placed),
                "sr": round(sr, 1),
                "roi": round(roi, 1),
            }

    return best_result


def evaluate_wp_baseline(X_meta, y, odds, wp_thresholds: list[float], label: str):
    """Evaluate flat WP threshold baselines for comparison."""
    # WP is the first meta-feature (index 0)
    wps = X_meta[:, 0]
    odds_arr = np.array(odds, dtype=np.float64)

    print(f"\n  {label} — Flat WP threshold baselines:")
    print(f"  {'WP Thresh':>10s} {'Bets':>6s} {'Placed':>7s} {'SR%':>6s} {'ROI%':>7s} {'P&L':>10s}")
    print(f"  {'-'*50}")

    for wp_thresh in wp_thresholds:
        mask = wps >= wp_thresh
        n_bets = mask.sum()
        if n_bets == 0:
            print(f"  {f'WP>={wp_thresh:.0%}':>10s} {0:>6d}")
            continue

        placed = y[mask].sum()
        sr = placed / n_bets * 100

        bet_odds = odds_arr[mask]
        placed_mask = y[mask] == 1
        returns = np.where(placed_mask, bet_odds / 3.0, 0.0)
        pnl = returns.sum() - n_bets
        roi = pnl / n_bets * 100

        print(f"  {'WP>=' + f'{wp_thresh:.0%}':>10s} {n_bets:>6d} {placed:>7d} {sr:>5.1f}% {roi:>+6.1f}% {pnl:>+9.1f}")


def print_feature_importance(model: lgb.Booster):
    """Print meta-model feature importance."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(META_FEATURE_NAMES, importance), key=lambda x: -x[1])

    print(f"\n  Meta-model Feature Importance (gain):")
    max_imp = pairs[0][1] if pairs[0][1] > 0 else 1
    for name, imp in pairs:
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {name:20s} {imp:10.1f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train Betfair meta-model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to Proform data directory")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Default threshold for should_bet (saved in metadata)")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Load the main LGBM rank model
    rank_model_path = MODEL_DIR / "lgbm_rank_model.txt"
    if not rank_model_path.exists():
        print(f"ERROR: Main LGBM rank model not found at {rank_model_path}")
        print("  Train the main model first: python scripts/train_lgbm_v2.py")
        sys.exit(1)

    print("Loading main LGBM rank model...")
    rank_model = lgb.Booster(model_file=str(rank_model_path))
    print(f"  Loaded: {rank_model.num_trees()} trees, {rank_model.num_feature()} features")

    # Load Proform data with full race context
    races = load_proform_races_with_dividends(data_dir)
    if not races:
        print("ERROR: No races loaded from Proform data")
        sys.exit(1)

    # Build meta-dataset
    X_meta, y_meta, weights, dates, odds_list = build_meta_dataset(races, rank_model)

    # Temporal split: 75% train / 12.5% val / 12.5% test
    all_dates = sorted(set(d for d in dates if d))
    n_dates = len(all_dates)
    val_idx = int(n_dates * 0.75)
    test_idx = int(n_dates * 0.875)
    val_cutoff = all_dates[val_idx]
    test_cutoff = all_dates[test_idx]

    train_mask = np.array([d < val_cutoff for d in dates])
    val_mask = np.array([(d >= val_cutoff and d < test_cutoff) for d in dates])
    test_mask = np.array([d >= test_cutoff for d in dates])

    print(f"\n  Temporal split: train < {val_cutoff}, val {val_cutoff}–{test_cutoff}, test >= {test_cutoff}")
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n = mask.sum()
        placed = y_meta[mask].sum()
        print(f"  {name:6s}: {n:,} races, {placed:,} placed ({placed/n*100:.1f}%)")

    X_tr = X_meta[train_mask]
    y_tr = y_meta[train_mask]
    w_tr = weights[train_mask]

    X_val = X_meta[val_mask]
    y_val = y_meta[val_mask]
    w_val = weights[val_mask]

    X_test = X_meta[test_mask]
    y_test = y_meta[test_mask]
    w_test = weights[test_mask]

    odds_tr = [o for o, m in zip(odds_list, train_mask) if m]
    odds_val = [o for o, m in zip(odds_list, val_mask) if m]
    odds_test = [o for o, m in zip(odds_list, test_mask) if m]

    # Train meta-model
    meta_model = train_meta_model(X_tr, y_tr, w_tr, X_val, y_val, w_val)

    # Feature importance
    print_feature_importance(meta_model)

    # Evaluate at various thresholds
    val_best = evaluate_meta_model(meta_model, X_val, y_val, odds_val, "Validation")
    test_best = evaluate_meta_model(meta_model, X_test, y_test, odds_test, "Test (OOS)")

    # Compare with flat WP baselines
    evaluate_wp_baseline(X_val, y_val, odds_val, [0.18, 0.20, 0.22, 0.25, 0.30], "Validation")
    evaluate_wp_baseline(X_test, y_test, odds_test, [0.18, 0.20, 0.22, 0.25, 0.30], "Test (OOS)")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta_model.save_model(str(META_MODEL_PATH))

    metadata = {
        "version": 1,
        "model_type": "binary_classifier",
        "purpose": "Betfair bet selector — predicts if LGBM rank 1 pick will place",
        "feature_names": META_FEATURE_NAMES,
        "num_features": NUM_META_FEATURES,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "train_races": int(train_mask.sum()),
        "val_races": int(val_mask.sum()),
        "test_races": int(test_mask.sum()),
        "temporal_split": {
            "train": f"< {val_cutoff}",
            "val": f"{val_cutoff} to {test_cutoff}",
            "test": f">= {test_cutoff}",
        },
        "lgbm_params": {k: v for k, v in META_LGBM_PARAMS.items() if k != "verbose"},
        "best_iteration": meta_model.best_iteration,
        "num_trees": meta_model.num_trees(),
        "default_threshold": args.threshold,
        "val_best": val_best,
        "test_best": test_best,
        "train_place_rate": round(y_tr.mean() * 100, 1),
    }

    with open(META_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Meta-model saved!")
    print(f"  Model:    {META_MODEL_PATH} ({META_MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  Metadata: {META_METADATA_PATH}")
    print(f"  Trees:    {meta_model.num_trees()}")
    print(f"  Default threshold: {args.threshold}")
    if val_best:
        print(f"  Val best: threshold={val_best.get('threshold', 'N/A')}, "
              f"SR={val_best.get('sr', 'N/A')}%, ROI={val_best.get('roi', 'N/A')}%")
    if test_best:
        print(f"  Test best: threshold={test_best.get('threshold', 'N/A')}, "
              f"SR={test_best.get('sr', 'N/A')}%, ROI={test_best.get('roi', 'N/A')}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
