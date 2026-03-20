#!/usr/bin/env python3
"""Train bet type meta-model — learned Win/Each Way/Place selector.

For each Proform race:
  1. Run LGBM rank model → rank runners → top 4 picks
  2. For each pick, simulate Win / Each Way / Place returns
  3. Label: which bet type was most profitable
  4. Extract context features

Multi-class LightGBM: 3 classes (Win=0, Each Way=1, Place=2).

Usage:
    python scripts/train_bettype_meta.py
    python scripts/train_bettype_meta.py --data-dir D:\\Punty\\DatafromProform
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
    extract_features_proform,
    DEFAULT_DATA_DIR,
    MONTH_DIRS,
    _sf,
)
from punty.ml.features import (
    FEATURE_NAMES, NUM_FEATURES,
    _distance_bucket, _track_cond_bucket, _class_bucket, _venue_type_code,
)
from punty.betting.bettype_model import (
    BETTYPE_FEATURE_NAMES, NUM_BETTYPE_FEATURES,
    BETTYPE_MODEL_PATH, BETTYPE_METADATA_PATH,
    BET_TYPE_CLASSES,
)

MODEL_DIR = ROOT / "punty" / "data"
RANK_MODEL_PATH = MODEL_DIR / "lgbm_rank_model.txt"

# LightGBM params — multi-class
BETTYPE_LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": ["multi_logloss"],
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}

EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 500

# Simulated stake for P&L calculations
SIM_STAKE = 10.0


def _simulate_bet_pnl(
    finish_pos: int, sp_odds: float, place_div: float,
    field_size: int,
) -> dict[str, float]:
    """Simulate P&L for each bet type given actual result.

    Args:
        finish_pos: Actual finish position (1=won, 2=2nd, etc.)
        sp_odds: Starting price odds
        place_div: Estimated place dividend (SP/3 approximation if not available)
        field_size: Number of runners

    Returns:
        Dict of {"Win": pnl, "Each Way": pnl, "Place": pnl}
    """
    # Place cutoff: top 3 for 8+ runners, top 2 for 5-7
    place_cutoff = 3 if field_size >= 8 else 2
    won = finish_pos == 1
    placed = finish_pos <= place_cutoff

    # Estimate place dividend if not available
    if not place_div or place_div <= 1.0:
        place_div = max(1.10, sp_odds / 3.0)

    stake = SIM_STAKE

    # Win: full stake on win
    win_pnl = (sp_odds * stake - stake) if won else -stake

    # Each Way: half on win, half on place
    ew_half = stake / 2
    if won:
        ew_pnl = (sp_odds * ew_half - ew_half) + (place_div * ew_half - ew_half)
    elif placed:
        ew_pnl = -ew_half + (place_div * ew_half - ew_half)
    else:
        ew_pnl = -stake

    # Place: full stake on place
    place_pnl = (place_div * stake - stake) if placed else -stake

    return {"Win": win_pnl, "Each Way": ew_pnl, "Place": place_pnl}


def _best_bet_type(pnls: dict[str, float], finish_pos: int, odds: float) -> int:
    """Return class index of the best bet type for this outcome.

    Key insight: E/W is best when the horse places but doesn't win AND
    the odds are in the mid range ($3-$8) where Win bleeds on non-winners
    but the place dividend is meaningful. We can't know in advance whether
    the horse will win or place, so we assign E/W to the "placed but
    didn't win" outcomes at mid odds — this teaches the model that
    mid-odds horses in competitive races should be E/W.

    - Won at short odds (<$3): Win (collecting at short odds is reliable)
    - Won at any odds: Win (maximises return)
    - Placed (not won) at mid odds ($3-$8): Each Way (hedge was optimal)
    - Placed (not won) at short odds (<$3): Place (thin win margin, place safer)
    - Placed (not won) at long odds ($8+): Place (win too unlikely)
    - Missed: Place (least damage, or Win if odds very short)
    """
    if finish_pos == 1:
        # Won — Win was best (highest return on winner)
        return 0  # Win
    elif finish_pos <= 3:
        # Placed but didn't win — which hedge was optimal?
        if 3.0 <= odds <= 8.0:
            return 1  # Each Way — the sweet spot for hedging
        else:
            return 2  # Place — either too short (thin win margin) or too long
    else:
        # Missed entirely
        if odds < 3.0:
            return 2  # Place — short odds should collect more often
        return 2  # Place — minimises loss


def load_proform_races(data_dir: Path) -> list[dict]:
    """Load Proform data grouped by race."""
    print(f"Loading Proform data from {data_dir}...")
    start = time.time()

    races = []
    total_runners = 0

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
                            "condition": race.get("TrackCondition", "") or "",
                            "field_size": race.get("Starters", 12),
                            "date": md[:10] if md else "",
                            "prize_money": race.get("PrizeMoney", 0) or 0,
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
                    if field_size < 5:
                        continue

                    weights = [r.get("Weight", 0) for r in race_runners
                               if r.get("Weight", 0) and r["Weight"] > 40]
                    avg_weight = statistics.mean(weights) if weights else 56.0

                    if not meta.get("condition"):
                        for r in race_runners:
                            forms = [f for f in r.get("Forms", []) if not f.get("IsBarrierTrial")]
                            if forms:
                                tc = forms[0].get("TrackCondition", "")
                                if tc:
                                    meta["condition"] = tc
                                    break

                    valid_runners = []
                    valid_features = []
                    for r in race_runners:
                        pos = r.get("Position")
                        sp = r.get("PriceSP", 0)
                        if pos is None or pos == 0 or not sp or sp <= 1.0:
                            continue
                        features = extract_features_proform(r, meta, field_size, avg_weight)
                        if len(features) != NUM_FEATURES:
                            continue
                        valid_runners.append(r)
                        valid_features.append(features)

                    if len(valid_runners) < 4:
                        continue

                    races.append({
                        "meta": meta,
                        "runners": valid_runners,
                        "features": valid_features,
                        "field_size": field_size,
                    })
                    total_runners += len(valid_runners)
                    month_races += 1

            if month_races > 0:
                print(f"    {month_name}: {month_races:,} races")

    elapsed = time.time() - start
    print(f"\nTotal: {len(races):,} races, {total_runners:,} runners in {elapsed:.1f}s")
    return races


def build_bettype_dataset(races: list[dict], rank_model) -> tuple:
    """For each race, simulate Win/EW/Place for top 4 picks."""
    print("\nBuilding bet type dataset...")
    start = time.time()

    X_all = []
    y_all = []
    dates = []

    rank_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "pnl": 0.0}))

    for race in races:
        meta = race["meta"]
        runners = race["runners"]
        features = race["features"]
        field_size = race["field_size"]

        if len(features) < 4:
            continue

        # Run LGBM
        X = np.array(features, dtype=np.float64)
        model_n = rank_model.num_feature()
        if X.shape[1] > model_n:
            X = X[:, :model_n]

        raw_scores = rank_model.predict(X)
        shifted = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(shifted)
        win_probs = exp_scores / np.sum(exp_scores)

        ranked_indices = np.argsort(-win_probs)
        top4_indices = ranked_indices[:min(4, len(ranked_indices))]

        # Race context
        distance = meta.get("distance", 1400)
        dist_bucket = _distance_bucket(distance)
        cls_bucket = _class_bucket(meta.get("race_class", ""))
        tc_bucket = _track_cond_bucket(meta.get("condition", ""))
        venue_type = _venue_type_code(meta.get("venue", ""))
        prize_money = float(meta.get("prize_money", 0) or 0)

        # Top pick info for race shape
        rank1_wp = float(win_probs[top4_indices[0]])
        rank3_wp = float(win_probs[top4_indices[2]]) if len(top4_indices) > 2 else 0
        wp_spread = rank1_wp - rank3_wp

        # Fav odds
        fav_sp = min(r.get("PriceSP", 99) for r in runners if r.get("PriceSP", 0) > 1.0)

        race_date = meta.get("date", "")

        # For each of top 4 picks
        for rank_idx, runner_idx in enumerate(top4_indices):
            rank = rank_idx + 1  # 1-4
            runner = runners[int(runner_idx)]
            wp = float(win_probs[runner_idx])
            sp = runner.get("PriceSP", 0) or 0
            if sp <= 1.0:
                continue
            pos = runner.get("Position", 99)

            # Market implied
            mkt_implied = 1.0 / sp if sp > 1.0 else 0
            value_rating = wp / mkt_implied if mkt_implied > 0 else 1.0

            # Place prob (Harville approximation)
            place_prob = min(0.95, wp * 2.5)  # rough approximation

            # Place dividend approximation
            place_div = sp / 3.0 if sp > 1.0 else 1.10

            # Gap to next rank
            if rank_idx + 1 < len(top4_indices):
                next_wp = float(win_probs[top4_indices[rank_idx + 1]])
                gap_to_next = wp - next_wp
            else:
                gap_to_next = 0.0

            # Simulate all 3 bet types
            pnls = _simulate_bet_pnl(pos, sp, place_div, field_size)

            # Label: best bet type based on outcome + context
            label = _best_bet_type(pnls, pos, sp)

            # Track stats
            best_type = BET_TYPE_CLASSES[label]
            rank_stats[rank][best_type]["count"] += 1
            rank_stats[rank][best_type]["pnl"] += pnls[best_type]

            # Build feature vector
            from punty.betting.bettype_model import extract_bettype_features
            feat = extract_bettype_features(
                tip_rank=rank,
                is_roughie=(rank == 4),
                win_prob=wp,
                place_prob=place_prob,
                value_rating=value_rating,
                odds=sp,
                field_size=field_size,
                distance_bucket=dist_bucket,
                class_bucket=cls_bucket,
                track_cond_bucket=tc_bucket,
                venue_type=venue_type,
                prize_money=prize_money,
                rank1_wp=rank1_wp,
                wp_spread=wp_spread,
                gap_to_next=gap_to_next,
                fav_odds=fav_sp,
                place_odds=place_div,
            )

            if len(feat) != NUM_BETTYPE_FEATURES:
                continue

            X_all.append(feat)
            y_all.append(label)
            dates.append(race_date)

    elapsed = time.time() - start
    print(f"  Dataset: {len(y_all):,} picks in {elapsed:.1f}s")

    # Distribution
    from collections import Counter
    dist = Counter(y_all)
    for i, name in enumerate(BET_TYPE_CLASSES):
        n = dist.get(i, 0)
        print(f"  {name}: {n:,} ({n/len(y_all)*100:.1f}%)")

    print(f"\n  Best bet type by rank (most profitable in hindsight):")
    print(f"  {'Rank':<6s} {'Win':>8s} {'E/W':>8s} {'Place':>8s}")
    print(f"  {'-'*32}")
    for rank in sorted(rank_stats.keys()):
        parts = []
        for bt in BET_TYPE_CLASSES:
            n = rank_stats[rank][bt]["count"]
            parts.append(f"{n:>7,d}")
        print(f"  R{rank:<5d} {'  '.join(parts)}")

    return (
        np.array(X_all, dtype=np.float64),
        np.array(y_all, dtype=np.int32),
        dates,
    )


def train_bettype_model(X_train, y_train, X_val, y_val) -> lgb.Booster:
    """Train the bet type multi-class model."""
    print(f"\n{'='*60}")
    print("Training Bet Type meta-model")
    print(f"  Train: {len(y_train):,} picks")
    print(f"  Val:   {len(y_val):,} picks")
    print(f"{'='*60}")

    train_data = lgb.Dataset(
        X_train, label=y_train,
        feature_name=BETTYPE_FEATURE_NAMES, free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val, label=y_val,
        feature_name=BETTYPE_FEATURE_NAMES, free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        BETTYPE_LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Trees: {model.num_trees()}")
    return model


def evaluate_model(model, X, y, label: str):
    """Evaluate model predictions by rank."""
    raw_preds = model.predict(X)  # shape: (n, 3)
    pred_classes = np.argmax(raw_preds, axis=1)

    # Overall accuracy
    correct = (pred_classes == y).sum()
    print(f"\n  {label} — Overall accuracy: {correct}/{len(y)} ({correct/len(y)*100:.1f}%)")

    # Per-class
    print(f"  {'Class':<12s} {'Actual':>8s} {'Predicted':>10s} {'Correct':>8s}")
    print(f"  {'-'*42}")
    for i, name in enumerate(BET_TYPE_CLASSES):
        actual = (y == i).sum()
        predicted = (pred_classes == i).sum()
        correct_i = ((pred_classes == i) & (y == i)).sum()
        print(f"  {name:<12s} {actual:>8,d} {predicted:>10,d} {correct_i:>8,d}")

    # By rank (rank is feature index 0)
    print(f"\n  {label} — Selection distribution by rank:")
    print(f"  {'Rank':<6s} {'Win':>8s} {'E/W':>8s} {'Place':>8s}")
    print(f"  {'-'*32}")
    for rank in [1, 2, 3, 4]:
        mask = X[:, 0] == rank
        if mask.sum() == 0:
            continue
        rank_preds = pred_classes[mask]
        parts = []
        for i, name in enumerate(BET_TYPE_CLASSES):
            n = (rank_preds == i).sum()
            pct = n / len(rank_preds) * 100
            parts.append(f"{pct:>6.1f}%")
        print(f"  R{rank:<5d} {'  '.join(parts)}")


def print_feature_importance(model: lgb.Booster):
    """Print feature importance."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(BETTYPE_FEATURE_NAMES, importance), key=lambda x: -x[1])

    print(f"\n  Feature Importance (gain):")
    max_imp = pairs[0][1] if pairs[0][1] > 0 else 1
    for name, imp in pairs:
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {name:22s} {imp:10.1f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train bet type meta-model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    if not RANK_MODEL_PATH.exists():
        print(f"ERROR: LGBM rank model not found at {RANK_MODEL_PATH}")
        sys.exit(1)

    print("Loading main LGBM rank model...")
    rank_model = lgb.Booster(model_file=str(RANK_MODEL_PATH))
    print(f"  Loaded: {rank_model.num_trees()} trees, {rank_model.num_feature()} features")

    races = load_proform_races(data_dir)
    if not races:
        print("ERROR: No races loaded")
        sys.exit(1)

    X, y, dates = build_bettype_dataset(races, rank_model)

    # Temporal split
    all_dates = sorted(set(d for d in dates if d))
    n_dates = len(all_dates)
    val_idx = int(n_dates * 0.75)
    test_idx = int(n_dates * 0.875)
    val_cutoff = all_dates[val_idx]
    test_cutoff = all_dates[test_idx]

    train_mask = np.array([d < val_cutoff for d in dates])
    val_mask = np.array([(d >= val_cutoff and d < test_cutoff) for d in dates])
    test_mask = np.array([d >= test_cutoff for d in dates])

    print(f"\n  Temporal split: train < {val_cutoff}, val {val_cutoff}-{test_cutoff}, test >= {test_cutoff}")

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Train
    model = train_bettype_model(X_tr, y_tr, X_val, y_val)

    # Feature importance
    print_feature_importance(model)

    # Evaluate
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test (OOS)")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(BETTYPE_MODEL_PATH))

    metadata = {
        "version": 1,
        "model_type": "multiclass_classifier",
        "purpose": "Bet type selector — predicts Win/EW/Place per selection",
        "classes": BET_TYPE_CLASSES,
        "feature_names": BETTYPE_FEATURE_NAMES,
        "num_features": NUM_BETTYPE_FEATURES,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "total_races": len(races),
        "train_picks": int(train_mask.sum()),
        "val_picks": int(val_mask.sum()),
        "test_picks": int(test_mask.sum()),
        "temporal_split": {
            "train": f"< {val_cutoff}",
            "val": f"{val_cutoff} to {test_cutoff}",
            "test": f">= {test_cutoff}",
        },
        "lgbm_params": {k: v for k, v in BETTYPE_LGBM_PARAMS.items() if k != "verbose"},
        "best_iteration": model.best_iteration,
        "num_trees": model.num_trees(),
    }

    with open(BETTYPE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Bet type meta-model saved!")
    print(f"  Model:    {BETTYPE_MODEL_PATH}")
    print(f"  Trees:    {model.num_trees()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
