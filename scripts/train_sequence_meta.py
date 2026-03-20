#!/usr/bin/env python3
"""Train sequence meta-model — learned play/skip gate for multi-race bets.

For each Proform meeting:
  1. Run LGBM rank model → get WP for all runners per race
  2. Simulate sequences (Early Quaddie, Quaddie, Big 6, Big3 Multi)
  3. Label: did the sequence hit? Weight: estimated dividend
  4. Extract meeting-level context features

Usage:
    python scripts/train_sequence_meta.py
    python scripts/train_sequence_meta.py --data-dir D:\\Punty\\DatafromProform
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
    _track_cond_bucket, _venue_type_code,
)
from punty.betting.sequence_model import (
    SEQ_FEATURE_NAMES, NUM_SEQ_FEATURES,
    SEQ_MODEL_PATH, SEQ_METADATA_PATH,
    SEQ_TYPE_CODES,
)

MODEL_DIR = ROOT / "punty" / "data"
RANK_MODEL_PATH = MODEL_DIR / "lgbm_rank_model.txt"

# LightGBM params
SEQ_LGBM_PARAMS = {
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
}

EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 500


def _class_bucket_val(race_class: str) -> float:
    """Quick class bucket for training."""
    c = (race_class or "").lower().strip().rstrip(";")
    if "maiden" in c:
        return 1.0
    if "benchmark" in c or c.startswith("bm"):
        return 3.0
    if "handicap" in c or "hcp" in c:
        return 4.0
    if any(x in c for x in ("open", "wfa")):
        return 5.0
    if any(x in c for x in ("group", "listed", "stakes")):
        return 6.0
    return 2.0


def _distance_bucket_val(distance) -> float:
    """Quick distance bucket for training."""
    d = float(distance or 1400)
    if d < 1200:
        return 1.0
    if d < 1400:
        return 2.0
    if d < 1800:
        return 3.0
    if d < 2200:
        return 4.0
    return 5.0


def load_proform_meetings(data_dir: Path) -> list[dict]:
    """Load Proform data grouped by meeting → races → runners."""
    print(f"Loading Proform meetings from {data_dir}...")
    start = time.time()

    meetings = []
    total_races = 0

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
            meeting_meta = {}
            if meetings_path.exists():
                with open(meetings_path, "r", encoding="utf-8") as f:
                    meetings_data = json.load(f)
                for m in meetings_data:
                    md = m.get("MeetingDate", "")
                    venue = m.get("Track", {}).get("Name", "")
                    state = m.get("Track", {}).get("State", "")
                    mid = f"{venue}-{md[:10]}" if venue and md else ""
                    if mid:
                        meeting_meta[mid] = {
                            "venue": venue,
                            "state": state,
                            "date": md[:10],
                        }
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
                            "prize_money": race.get("PrizeMoney", 0) or 0,
                            "race_number": race.get("Number", 0),
                            "meeting_id": mid,
                        }

            form_dir = year_dir / month_name / "Form"
            if not form_dir.exists():
                continue

            # Group races by meeting
            meeting_races = defaultdict(dict)  # mid → {race_num: {meta, runners, features}}
            month_meetings = 0

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
                    if field_size < 3:
                        continue

                    weights = [r.get("Weight", 0) for r in race_runners
                               if r.get("Weight", 0) and r["Weight"] > 40]
                    avg_weight = statistics.mean(weights) if weights else 56.0

                    # Infer track condition
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

                    if len(valid_runners) < 3:
                        continue

                    mid = meta.get("meeting_id", "")
                    rn = meta.get("race_number", 0)
                    if mid and rn:
                        meeting_races[mid][rn] = {
                            "meta": meta,
                            "runners": valid_runners,
                            "features": valid_features,
                            "field_size": field_size,
                            "avg_weight": avg_weight,
                        }

            for mid, races_dict in meeting_races.items():
                if len(races_dict) < 4:
                    continue  # Need at least 4 races for a quaddie
                mm = meeting_meta.get(mid, {})
                meetings.append({
                    "meeting_id": mid,
                    "venue": mm.get("venue", ""),
                    "date": mm.get("date", ""),
                    "races": races_dict,
                })
                total_races += len(races_dict)
                month_meetings += 1

            if month_meetings > 0:
                print(f"    {month_name}: {month_meetings:,} meetings")

    elapsed = time.time() - start
    print(f"\nTotal: {len(meetings):,} meetings, {total_races:,} races in {elapsed:.1f}s")
    return meetings


def _run_lgbm_on_race(race_data: dict, rank_model) -> dict | None:
    """Run LGBM on a single race, return top 3 picks info."""
    runners = race_data["runners"]
    features = race_data["features"]
    if len(features) < 3:
        return None

    X = np.array(features, dtype=np.float64)
    model_n = rank_model.num_feature()
    if X.shape[1] > model_n:
        X = X[:, :model_n]

    raw_scores = rank_model.predict(X)
    shifted = raw_scores - np.max(raw_scores)
    exp_scores = np.exp(shifted)
    win_probs = exp_scores / np.sum(exp_scores)

    ranked_indices = np.argsort(-win_probs)
    top3_indices = ranked_indices[:min(3, len(ranked_indices))]

    rank1_idx = int(top3_indices[0])
    rank1_runner = runners[rank1_idx]

    # Check if ANY of our top 3 picks won (matches production multi-runner legs)
    top3_positions = [runners[int(idx)].get("Position", 99) for idx in top3_indices]
    any_top3_won = any(p == 1 for p in top3_positions)

    return {
        "rank1_wp": float(win_probs[rank1_idx]),
        "rank1_pos": rank1_runner.get("Position", 99),
        "rank1_sp": float(rank1_runner.get("PriceSP", 0) or 0),
        "field_size": race_data["field_size"],
        "any_top3_won": any_top3_won,
        "top3_wps": [float(win_probs[int(idx)]) for idx in top3_indices],
    }


def _check_sequence_hit(leg_results: list[dict], strict: bool = False) -> bool:
    """Did our picks win every leg?

    strict=True: rank 1 must win (Big3 Multi - single pick per leg)
    strict=False: any of top 3 must win (Quaddie/Big6 - multi-runner legs)
    """
    if strict:
        return all(r["rank1_pos"] == 1 for r in leg_results)
    return all(r["any_top3_won"] for r in leg_results)


# Typical outlays by sequence type
TYPICAL_OUTLAYS = {
    "Early Quaddie": 30.0,
    "Quaddie": 30.0,
    "Big 6": 5.0,
    "Big3 Multi": 10.0,
}

# Typical combos by type (with multi-runner legs)
TYPICAL_COMBOS = {
    "Early Quaddie": 81,   # ~3 per leg × 4 legs = 3^4
    "Quaddie": 81,
    "Big 6": 1,            # 1 pick per leg
    "Big3 Multi": 1,       # straight multi
}


def _estimate_sequence_dividend(leg_results: list[dict]) -> float:
    """Estimate sequence dividend from SP odds of winners."""
    product = 1.0
    for r in leg_results:
        sp = r["rank1_sp"]
        if sp > 1.0:
            product *= sp
    # Rough pool-based dividend estimate
    return product * 0.65  # pool takeout


def _is_profitable(leg_results: list[dict], seq_type: str) -> tuple[bool, float]:
    """Would this sequence have been profitable?

    Returns (profitable: bool, estimated_roi: float).
    A sequence is profitable if estimated payout > outlay.
    """
    outlay = TYPICAL_OUTLAYS.get(seq_type, 30.0)
    combos = TYPICAL_COMBOS.get(seq_type, 1)
    flexi = outlay / combos if combos > 0 else outlay

    est_div = _estimate_sequence_dividend(leg_results)
    payout = est_div * flexi  # flexi % of full dividend
    roi = (payout - outlay) / outlay if outlay > 0 else 0
    return payout > outlay, roi


def build_sequence_dataset(meetings: list[dict], rank_model) -> tuple:
    """For each meeting, simulate sequence bets and build training rows."""
    print("\nBuilding sequence dataset...")
    start = time.time()

    X_all = []
    y_all = []
    w_all = []
    dates = []
    types_list = []

    type_stats = defaultdict(lambda: {"total": 0, "hits": 0})

    for meeting in meetings:
        venue = meeting["venue"]
        date = meeting["date"]
        races = meeting["races"]
        race_numbers = sorted(races.keys())
        total_races = len(race_numbers)

        if total_races < 4:
            continue

        # Run LGBM on each race
        race_results = {}
        for rn in race_numbers:
            result = _run_lgbm_on_race(races[rn], rank_model)
            if result:
                race_results[rn] = result

        if len(race_results) < 4:
            continue

        # Get track condition from first race
        first_race = races[race_numbers[0]]
        tc = first_race["meta"].get("condition", "")
        vt = _venue_type_code(venue)

        # Define sequence windows (same as production rules)
        sequence_defs = []
        if total_races >= 6:
            # Early Quaddie: first 4 races
            eq_start = race_numbers[0]
            eq_end = race_numbers[3]
            sequence_defs.append(("Early Quaddie", list(range(eq_start, eq_end + 1))))

            # Quaddie: last 4 races
            q_start = race_numbers[-4]
            q_end = race_numbers[-1]
            sequence_defs.append(("Quaddie", list(range(q_start, q_end + 1))))

            # Big 6: last 6 races (or first 6 if exactly 6)
            if total_races >= 6:
                b6_start = race_numbers[-6] if total_races > 6 else race_numbers[0]
                b6_end = race_numbers[-1]
                b6_legs = list(range(b6_start, b6_end + 1))[:6]
                sequence_defs.append(("Big 6", b6_legs))
        elif total_races >= 4:
            # Only quaddie for smaller meetings
            q_start = race_numbers[-4]
            q_end = race_numbers[-1]
            sequence_defs.append(("Quaddie", list(range(q_start, q_end + 1))))

        # Big3 Multi: best 3 races by rank 1 WP
        if len(race_results) >= 3:
            sorted_by_wp = sorted(race_results.items(), key=lambda x: x[1]["rank1_wp"], reverse=True)
            b3_races = [rn for rn, _ in sorted_by_wp[:3]]
            sequence_defs.append(("Big3 Multi", b3_races))

        # Simulate each sequence
        for seq_type, leg_race_numbers in sequence_defs:
            leg_results = []
            for rn in leg_race_numbers:
                if rn in race_results:
                    leg_results.append(race_results[rn])

            if len(leg_results) < len(leg_race_numbers):
                continue  # Missing race data

            # Label by PROFITABILITY, not just hit/miss
            # A sequence that hits but pays less than the outlay is a loss
            is_multi = seq_type == "Big3 Multi"
            hit = _check_sequence_hit(leg_results, strict=is_multi)
            profitable, est_roi = _is_profitable(leg_results, seq_type)

            # Label: 1 = profitable (hit AND payout > outlay), 0 = loss
            label = 1 if (hit and profitable) else 0

            # Equal weights — let the model learn from the natural distribution
            weight = 1.0

            # Extract features
            leg_wps = [r["rank1_wp"] for r in leg_results]
            leg_fields = [float(r["field_size"]) for r in leg_results]

            # Extract per-leg class and distance context
            leg_classes = []
            leg_distances = []
            leg_prizes = []
            for rn in leg_race_numbers:
                if rn in races:
                    race_meta = races[rn]["meta"]
                    leg_classes.append(_class_bucket_val(race_meta.get("race_class", "")))
                    leg_distances.append(_distance_bucket_val(race_meta.get("distance", 1400)))
                    leg_prizes.append(float(race_meta.get("prize_money", 0) or 0))

            avg_cls = sum(leg_classes) / len(leg_classes) if leg_classes else 0
            avg_dist = sum(leg_distances) / len(leg_distances) if leg_distances else 0
            avg_prize = sum(leg_prizes) / len(leg_prizes) if leg_prizes else 0
            pm_bucket = 4.0 if avg_prize >= 100000 else (3.0 if avg_prize >= 50000 else (2.0 if avg_prize >= 25000 else 1.0))

            from punty.betting.sequence_model import extract_sequence_features
            feat = extract_sequence_features(
                sequence_type=seq_type,
                leg_wps=leg_wps,
                leg_field_sizes=leg_fields,
                track_condition=tc,
                venue_type=vt,
                total_combos=1 if is_multi else 3 ** len(leg_results),
                estimated_return_pct=0.0,
                hit_probability=0.0,
                avg_class_bucket=avg_cls,
                avg_distance_bucket=avg_dist,
                prize_money_bucket=pm_bucket,
            )

            if len(feat) != NUM_SEQ_FEATURES:
                continue

            X_all.append(feat)
            y_all.append(label)
            w_all.append(weight)
            dates.append(date)
            types_list.append(seq_type)

            type_stats[seq_type]["total"] += 1
            if label:
                type_stats[seq_type]["hits"] += 1

    elapsed = time.time() - start
    total_hits = sum(y_all)
    print(f"  Dataset: {len(y_all):,} rows")
    print(f"  Total hits: {total_hits:,} ({total_hits/len(y_all)*100:.1f}%)" if y_all else "  No data")
    print(f"  Built in {elapsed:.1f}s")

    print(f"\n  Per-type hit rates:")
    print(f"  {'Type':<20s} {'Total':>8s} {'Hits':>7s} {'SR%':>7s}")
    print(f"  {'-'*45}")
    for stype in SEQ_TYPE_CODES:
        stats = type_stats[stype]
        if stats["total"] > 0:
            sr = stats["hits"] / stats["total"] * 100
            print(f"  {stype:<20s} {stats['total']:>8,d} {stats['hits']:>7,d} {sr:>6.1f}%")

    return (
        np.array(X_all, dtype=np.float64),
        np.array(y_all, dtype=np.int32),
        np.array(w_all, dtype=np.float64),
        dates,
        types_list,
    )


def train_sequence_model(X_train, y_train, w_train, X_val, y_val, w_val) -> lgb.Booster:
    """Train the sequence meta-model."""
    print(f"\n{'='*60}")
    print("Training Sequence meta-model")
    n_pos = y_train.sum()
    print(f"  Train: {len(y_train):,} rows, {n_pos:,} hits ({n_pos/len(y_train)*100:.1f}%)")
    n_vpos = y_val.sum()
    print(f"  Val:   {len(y_val):,} rows, {n_vpos:,} hits ({n_vpos/len(y_val)*100:.1f}%)")
    print(f"{'='*60}")

    train_data = lgb.Dataset(
        X_train, label=y_train, weight=w_train,
        feature_name=SEQ_FEATURE_NAMES, free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, weight=w_val,
        feature_name=SEQ_FEATURE_NAMES, free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        SEQ_LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Trees: {model.num_trees()}")
    return model


def evaluate_per_type(model, X, y, types_list, label: str):
    """Evaluate model at various thresholds per type."""
    probs = model.predict(X)

    print(f"\n  {label} — Per-type threshold analysis:")
    for stype in SEQ_TYPE_CODES:
        mask = np.array([t == stype for t in types_list])
        if mask.sum() == 0:
            continue

        n = mask.sum()
        hits = y[mask].sum()
        sr = hits / n * 100

        print(f"\n  {stype} (n={n:,}, SR={sr:.1f}%):")
        print(f"  {'Threshold':>10s} {'Play':>6s} {'Hits':>6s} {'SR%':>6s} {'Skip':>6s}")
        print(f"  {'-'*40}")

        type_probs = probs[mask]
        type_y = y[mask]

        for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            play_mask = type_probs >= thresh
            n_play = play_mask.sum()
            n_skip = n - n_play
            play_hits = type_y[play_mask].sum() if n_play > 0 else 0
            play_sr = play_hits / n_play * 100 if n_play > 0 else 0
            print(f"  {thresh:>10.2f} {n_play:>6d} {play_hits:>6d} {play_sr:>5.1f}% {n_skip:>6d}")


def print_feature_importance(model: lgb.Booster):
    """Print feature importance."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(SEQ_FEATURE_NAMES, importance), key=lambda x: -x[1])

    print(f"\n  Feature Importance (gain):")
    max_imp = pairs[0][1] if pairs[0][1] > 0 else 1
    for name, imp in pairs:
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {name:24s} {imp:10.1f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train sequence meta-model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to Proform data directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    if not RANK_MODEL_PATH.exists():
        print(f"ERROR: Main LGBM rank model not found at {RANK_MODEL_PATH}")
        sys.exit(1)

    print("Loading main LGBM rank model...")
    rank_model = lgb.Booster(model_file=str(RANK_MODEL_PATH))
    print(f"  Loaded: {rank_model.num_trees()} trees, {rank_model.num_feature()} features")

    # Load meetings
    meetings = load_proform_meetings(data_dir)
    if not meetings:
        print("ERROR: No meetings loaded")
        sys.exit(1)

    # Build dataset
    X, y, weights, dates, types_list = build_sequence_dataset(meetings, rank_model)
    if len(X) == 0:
        print("ERROR: Empty dataset")
        sys.exit(1)

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

    print(f"\n  Temporal split: train < {val_cutoff}, val {val_cutoff}–{test_cutoff}, test >= {test_cutoff}")
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n = mask.sum()
        hits = y[mask].sum()
        print(f"  {name:6s}: {n:,} rows, {hits:,} hits ({hits/n*100:.1f}%)" if n > 0 else f"  {name:6s}: 0 rows")

    X_tr, y_tr, w_tr = X[train_mask], y[train_mask], weights[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], weights[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], weights[test_mask]

    types_val = [t for t, m in zip(types_list, val_mask) if m]
    types_test = [t for t, m in zip(types_list, test_mask) if m]

    # Train
    model = train_sequence_model(X_tr, y_tr, w_tr, X_val, y_val, w_val)

    # Feature importance
    print_feature_importance(model)

    # Per-type evaluation
    evaluate_per_type(model, X_val, y_val, types_val, "Validation")
    evaluate_per_type(model, X_test, y_test, types_test, "Test (OOS)")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(SEQ_MODEL_PATH))

    metadata = {
        "version": 1,
        "model_type": "binary_classifier",
        "purpose": "Sequence play/skip gate — predicts P(hit) for sequence bets",
        "feature_names": SEQ_FEATURE_NAMES,
        "num_features": NUM_SEQ_FEATURES,
        "seq_type_codes": SEQ_TYPE_CODES,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "total_meetings": len(meetings),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "temporal_split": {
            "train": f"< {val_cutoff}",
            "val": f"{val_cutoff} to {test_cutoff}",
            "test": f">= {test_cutoff}",
        },
        "lgbm_params": {k: v for k, v in SEQ_LGBM_PARAMS.items() if k != "verbose"},
        "best_iteration": model.best_iteration,
        "num_trees": model.num_trees(),
        "train_hit_rate": round(y_tr.mean() * 100, 2),
    }

    with open(SEQ_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sequence meta-model saved!")
    print(f"  Model:    {SEQ_MODEL_PATH}")
    print(f"  Metadata: {SEQ_METADATA_PATH}")
    print(f"  Trees:    {model.num_trees()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
