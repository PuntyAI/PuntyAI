#!/usr/bin/env python3
"""Train LightGBM win and place probability models from backtest.db.

Usage:
    python scripts/train_lightgbm.py

Loads 222K runners from data/backtest.db, extracts 58 features,
trains separate win and place binary classifiers with temporal split,
and saves models to punty/data/.

Output:
    punty/data/lgbm_win_model.txt    - Win probability model
    punty/data/lgbm_place_model.txt  - Place probability model
    punty/data/lgbm_metadata.json    - Feature names, metrics, training info
"""

import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from punty.ml.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    _f,
    _parse_stats,
    _safe_float,
    _score_last_five,
    _count_last5,
    _sr_from_stats,
    extract_features_from_db_row,
)

DB_PATH = ROOT / "data" / "backtest.db"
MODEL_DIR = ROOT / "punty" / "data"

# LightGBM parameters — tuned for racing data
LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,
    "max_depth": 7,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "is_unbalanced": True,  # Win rate ~8% — heavily imbalanced
}

EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 2000


def load_data() -> tuple[list[dict], list[dict], list[dict]]:
    """Load runners, races, meetings from backtest.db."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Load meetings
    meetings_by_id = {}
    for row in conn.execute("SELECT * FROM meetings"):
        meetings_by_id[row["id"]] = dict(row)

    # Load races
    races_by_id = {}
    for row in conn.execute("SELECT * FROM races"):
        races_by_id[row["id"]] = dict(row)

    # Load runners (only non-scratched with finish positions)
    runners = []
    for row in conn.execute(
        "SELECT * FROM runners WHERE scratched = 0 AND finish_position IS NOT NULL"
    ):
        runners.append(dict(row))

    conn.close()
    return runners, races_by_id, meetings_by_id


def prepare_dataset(
    runners: list[dict], races: dict, meetings: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract features and targets from all runners.

    Returns:
        X: feature matrix (n_runners, 58)
        y_win: win target (1 if finished 1st)
        y_place: place target (1 if finished top-3 or top-2 for small fields)
        race_ids: race_id per runner (for race-level eval)
        dates: meeting date per runner (for temporal split)
    """
    print(f"Extracting features from {len(runners)} runners...")
    start = time.time()

    # Pre-compute per-race field sizes and avg weights
    race_runners = defaultdict(list)
    for r in runners:
        race_runners[r["race_id"]].append(r)

    field_sizes = {}
    avg_weights = {}
    for race_id, race_group in race_runners.items():
        field_sizes[race_id] = len(race_group)
        weights = [r["weight"] for r in race_group if r.get("weight") and r["weight"] > 40]
        avg_weights[race_id] = statistics.mean(weights) if weights else 0.0

    X_list = []
    y_win = []
    y_place = []
    race_ids = []
    dates = []

    skipped = 0
    for runner in runners:
        race_id = runner["race_id"]
        race = races.get(race_id, {})
        meeting_id = race.get("meeting_id", "")
        meeting = meetings.get(meeting_id, {})

        fs = field_sizes.get(race_id, 12)
        aw = avg_weights.get(race_id, 0.0)

        features = extract_features_from_db_row(runner, race, meeting, fs, aw)
        if len(features) != NUM_FEATURES:
            skipped += 1
            continue

        X_list.append(features)

        pos = runner.get("finish_position")
        y_win.append(1 if pos == 1 else 0)
        # Place: top-3 for 8+ field, top-2 for ≤7
        place_cutoff = 2 if fs <= 7 else 3
        y_place.append(1 if pos and pos <= place_cutoff else 0)

        race_ids.append(race_id)
        dates.append(meeting.get("date", "") or "")

    elapsed = time.time() - start
    print(f"  Extracted {len(X_list)} samples in {elapsed:.1f}s ({skipped} skipped)")

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_win, dtype=np.int32),
        np.array(y_place, dtype=np.int32),
        race_ids,
        dates,
    )


def temporal_split(
    X: np.ndarray, y_win: np.ndarray, y_place: np.ndarray,
    race_ids: list[str], dates: list[str],
) -> dict:
    """Split data temporally: train Jan-Oct, val Nov, test Dec."""
    train_mask = np.array([d[:7] <= "2025-10" for d in dates])
    val_mask = np.array([d[:7] == "2025-11" for d in dates])
    test_mask = np.array([d[:7] == "2025-12" for d in dates])

    return {
        "train": (X[train_mask], y_win[train_mask], y_place[train_mask],
                  [r for r, m in zip(race_ids, train_mask) if m]),
        "val": (X[val_mask], y_win[val_mask], y_place[val_mask],
                [r for r, m in zip(race_ids, val_mask) if m]),
        "test": (X[test_mask], y_win[test_mask], y_place[test_mask],
                 [r for r, m in zip(race_ids, test_mask) if m]),
    }


def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    label: str,
) -> lgb.Booster:
    """Train a LightGBM binary classifier."""
    print(f"\n{'='*60}")
    print(f"Training {label} model")
    print(f"  Train: {len(y_train)} samples, {y_train.sum()} positives ({y_train.mean()*100:.1f}%)")
    print(f"  Val:   {len(y_val)} samples, {y_val.sum()} positives ({y_val.mean()*100:.1f}%)")
    print(f"{'='*60}")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(
    model: lgb.Booster, X: np.ndarray, y: np.ndarray,
    race_ids: list[str], label: str,
) -> dict:
    """Evaluate model on a dataset, printing race-level metrics."""
    probs = model.predict(X)

    # Standard metrics
    ll = log_loss(y, probs)
    auc = roc_auc_score(y, probs)

    print(f"\n  {label} Results:")
    print(f"    Log Loss: {ll:.4f}")
    print(f"    AUC:      {auc:.4f}")

    # Race-level accuracy (for win: did highest-prob runner win?)
    race_groups = defaultdict(list)
    for i, rid in enumerate(race_ids):
        race_groups[rid].append((probs[i], y[i], i))

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_races = 0

    for rid, entries in race_groups.items():
        entries.sort(key=lambda x: -x[0])  # Sort by predicted prob descending
        total_races += 1

        # Top-1: did highest-prob runner actually win/place?
        if entries[0][1] == 1:
            top1_correct += 1
        # Top-3: was the actual winner in our top 3?
        if any(e[1] == 1 for e in entries[:3]):
            top3_correct += 1
        # Top-5
        if any(e[1] == 1 for e in entries[:5]):
            top5_correct += 1

    top1_acc = top1_correct / total_races if total_races else 0
    top3_acc = top3_correct / total_races if total_races else 0
    top5_acc = top5_correct / total_races if total_races else 0

    print(f"    Races:    {total_races}")
    print(f"    Top-1:    {top1_acc:.1%} ({top1_correct}/{total_races})")
    print(f"    Top-3:    {top3_acc:.1%} ({top3_correct}/{total_races})")
    print(f"    Top-5:    {top5_acc:.1%} ({top5_correct}/{total_races})")

    # Calibration bands
    print(f"\n    Calibration:")
    bands = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]
    for lo, hi in bands:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            predicted = probs[mask].mean()
            actual = y[mask].mean()
            print(f"      [{lo:.2f}-{hi:.2f}): predicted={predicted:.3f}, actual={actual:.3f}, "
                  f"n={mask.sum()}, gap={actual - predicted:+.3f}")

    # Simulated flat-bet ROI (for win model only)
    roi_data = _simulate_roi(probs, y, race_ids, race_groups)

    return {
        "log_loss": round(ll, 4),
        "auc": round(auc, 4),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "n_races": total_races,
        "n_samples": len(y),
        **roi_data,
    }


def _simulate_roi(
    probs: np.ndarray, y: np.ndarray, race_ids: list[str],
    race_groups: dict,
) -> dict:
    """Simulate flat-bet ROI on top-1 picks (very rough estimate)."""
    # This is approximate — real ROI needs actual odds
    return {}


def print_feature_importance(model: lgb.Booster, label: str):
    """Print top features by importance."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])

    print(f"\n  {label} Top-15 Feature Importance (gain):")
    for name, imp in pairs[:15]:
        bar = "#" * int(imp / pairs[0][1] * 30)
        print(f"    {name:25s} {imp:10.1f}  {bar}")


def main():
    if not DB_PATH.exists():
        print(f"ERROR: backtest.db not found at {DB_PATH}")
        print("Run the backtest data pipeline first.")
        sys.exit(1)

    print(f"Loading data from {DB_PATH}...")
    runners, races, meetings = load_data()
    print(f"  {len(runners)} runners, {len(races)} races, {len(meetings)} meetings")

    X, y_win, y_place, race_ids, dates = prepare_dataset(runners, races, meetings)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Win rate:   {y_win.mean()*100:.1f}%")
    print(f"  Place rate: {y_place.mean()*100:.1f}%")

    splits = temporal_split(X, y_win, y_place, race_ids, dates)
    for name, (Xs, yw, yp, rids) in splits.items():
        print(f"  {name:6s}: {len(yw)} samples, {len(set(rids))} races")

    X_tr, yw_tr, yp_tr, rids_tr = splits["train"]
    X_val, yw_val, yp_val, rids_val = splits["val"]
    X_test, yw_test, yp_test, rids_test = splits["test"]

    # ── Train Win Model ──────────────────────────────────────────
    win_model = train_model(X_tr, yw_tr, X_val, yw_val, "Win")
    print_feature_importance(win_model, "Win")

    print("\n--- Win Model Evaluation ---")
    val_metrics_win = evaluate_model(win_model, X_val, yw_val, rids_val, "Validation")
    test_metrics_win = evaluate_model(win_model, X_test, yw_test, rids_test, "Test")

    # ── Train Place Model ────────────────────────────────────────
    place_model = train_model(X_tr, yp_tr, X_val, yp_val, "Place")
    print_feature_importance(place_model, "Place")

    print("\n--- Place Model Evaluation ---")
    val_metrics_place = evaluate_model(place_model, X_val, yp_val, rids_val, "Validation")
    test_metrics_place = evaluate_model(place_model, X_test, yp_test, rids_test, "Test")

    # ── Save Models ──────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    win_path = MODEL_DIR / "lgbm_win_model.txt"
    place_path = MODEL_DIR / "lgbm_place_model.txt"
    meta_path = MODEL_DIR / "lgbm_metadata.json"

    win_model.save_model(str(win_path))
    place_model.save_model(str(place_path))

    metadata = {
        "feature_names": FEATURE_NAMES,
        "num_features": NUM_FEATURES,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": str(DB_PATH),
        "train_samples": len(yw_tr),
        "val_samples": len(yw_val),
        "test_samples": len(yw_test),
        "temporal_split": {"train": "2025-01 to 2025-10", "val": "2025-11", "test": "2025-12"},
        "lgbm_params": LGBM_PARAMS,
        "win_model": {
            "best_iteration": win_model.best_iteration,
            "val_metrics": val_metrics_win,
            "test_metrics": test_metrics_win,
        },
        "place_model": {
            "best_iteration": place_model.best_iteration,
            "val_metrics": val_metrics_place,
            "test_metrics": test_metrics_place,
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Models saved:")
    print(f"  Win:      {win_path} ({win_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Place:    {place_path} ({place_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}")

    # Summary comparison with baseline
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON (weighted engine)")
    print(f"  Baseline top-1: 36.3%  |  LightGBM test top-1: {test_metrics_win['top1_accuracy']*100:.1f}%")
    print(f"  Baseline AUC:   ~0.72  |  LightGBM test AUC:   {test_metrics_win['auc']:.3f}")
    delta = test_metrics_win["top1_accuracy"] * 100 - 36.3
    print(f"  Delta: {delta:+.1f}% top-1 accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
