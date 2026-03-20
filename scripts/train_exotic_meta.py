#!/usr/bin/env python3
"""Train exotic meta-model — a learned exotic type selector for race context.

For each Proform race:
  1. Run LGBM rank model → get WP for all runners → rank to picks (top 4)
  2. Simulate every exotic type against actual finish positions
  3. Label: did the exotic hit? Weight: estimated dividend (SP-based)
  4. Extract context features: race shape, pick quality, exotic type

Trains a single LightGBM binary classifier with exotic_type as a feature.
At inference, score each candidate exotic type → pick highest score.

Usage:
    python scripts/train_exotic_meta.py
    python scripts/train_exotic_meta.py --data-dir D:\\Punty\\DatafromProform
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations, permutations
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
from punty.betting.exotic_model import (
    EXOTIC_FEATURE_NAMES, NUM_EXOTIC_FEATURES,
    EXOTIC_MODEL_PATH, EXOTIC_METADATA_PATH,
    EXOTIC_TYPE_CODES, _prize_money_bucket,
)

MODEL_DIR = ROOT / "punty" / "data"
RANK_MODEL_PATH = MODEL_DIR / "lgbm_rank_model.txt"

# LightGBM params — small, fast binary classifier
EXOTIC_LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
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
    "is_unbalanced": False,  # We use sample weights instead
}

EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 500


# ── Exotic simulation: check if each type would have hit ──

def _check_quinella(top_positions: list[int], n_runners: int) -> bool:
    """2-runner quinella: both finish 1st-2nd (any order)."""
    if n_runners < 2:
        return False
    return set(top_positions[:2]) <= {1, 2}


def _check_quinella_box(top_positions: list[int], n_runners: int) -> bool:
    """3-runner quinella box: any 2 of 3 finish 1st-2nd."""
    if n_runners < 3:
        return False
    placed_top2 = sum(1 for p in top_positions[:3] if p <= 2)
    return placed_top2 >= 2


def _check_exacta(top_positions: list[int], n_runners: int) -> bool:
    """Straight exacta: rank 1 wins, rank 2 runs 2nd."""
    if n_runners < 2:
        return False
    return top_positions[0] == 1 and top_positions[1] == 2


def _check_exacta_standout(top_positions: list[int], n_runners: int) -> bool:
    """Exacta standout: rank 1 wins, any of ranks 2-4 runs 2nd."""
    if n_runners < 2:
        return False
    if top_positions[0] != 1:
        return False
    return any(p == 2 for p in top_positions[1:4])


def _check_trifecta_standout(top_positions: list[int], n_runners: int) -> bool:
    """Trifecta standout: rank 1 wins, ranks 2-4 fill 2nd+3rd."""
    if n_runners < 3:
        return False
    if top_positions[0] != 1:
        return False
    placed_23 = sum(1 for p in top_positions[1:4] if p in (2, 3))
    return placed_23 >= 2


def _check_trifecta_box3(top_positions: list[int], n_runners: int) -> bool:
    """3-runner trifecta box: top 3 picks all finish 1st-3rd."""
    if n_runners < 3:
        return False
    return all(p <= 3 for p in top_positions[:3])


def _check_trifecta_box4(top_positions: list[int], n_runners: int) -> bool:
    """4-runner trifecta box: any 3 of top 4 picks finish 1st-3rd."""
    if n_runners < 4:
        return False
    placed_top3 = sum(1 for p in top_positions[:4] if p <= 3)
    return placed_top3 >= 3


def _check_first4(top_positions: list[int], n_runners: int) -> bool:
    """Positional First4: top 4 in predicted order (1st, 2nd, 3rd, 4th)."""
    if n_runners < 4:
        return False
    return (top_positions[0] == 1 and top_positions[1] == 2
            and top_positions[2] == 3 and top_positions[3] == 4)


def _check_first4_box(top_positions: list[int], n_runners: int) -> bool:
    """4-runner First4 box: top 4 picks all finish 1st-4th (any order)."""
    if n_runners < 4:
        return False
    return all(p <= 4 for p in top_positions[:4])


# Each entry: (checker_fn, min_runners_needed, num_combos, combo_ranks)
# combo_ranks tells the training which of our ranked picks form this combo
EXOTIC_CHECKERS = {
    "Quinella":            (_check_quinella, 2, 1, [1, 2]),
    "Quinella Box":        (_check_quinella_box, 3, 3, [1, 2, 3]),
    "Exacta":              (_check_exacta, 2, 1, [1, 2]),
    "Exacta Standout":     (_check_exacta_standout, 2, 3, [1, 2, 3]),
    "Trifecta Standout":   (_check_trifecta_standout, 3, 6, [1, 2, 3]),
    "Trifecta Box":        (_check_trifecta_box3, 3, 6, [1, 2, 3]),      # 3-runner
    "Trifecta Box 4":      (_check_trifecta_box4, 4, 24, [1, 2, 3, 4]),  # 4-runner (wider)
    "First4":              (_check_first4, 4, 1, [1, 2, 3, 4]),
    "First4 Box":          (_check_first4_box, 4, 24, [1, 2, 3, 4]),
}


def _estimate_exotic_dividend(exotic_type: str, sp_odds: list[float], field_size: int) -> float:
    """Estimate exotic dividend from SP odds.

    Uses simplified models based on typical dividend structures:
    - Quinella ≈ win_div1 * win_div2 / 2 (rough approximation)
    - Exacta ≈ win_div1 * win_div2 * 0.8
    - Trifecta ≈ product of top 3 SPs * field_factor
    - First4 ≈ product of top 4 SPs * field_factor
    """
    if not sp_odds or len(sp_odds) < 2:
        return 1.0

    # Normalize: dividends are approximately SP-based
    sorted_odds = sorted(sp_odds[:4])
    field_factor = max(1.0, field_size / 10.0)

    if "quinella" in exotic_type.lower():
        # Quinella dividend ≈ (SP1 + SP2) * 0.85 (pool-based, any order discount)
        return (sorted_odds[0] + sorted_odds[1]) * 0.85

    if "exacta" in exotic_type.lower():
        # Exacta ≈ SP1 * SP2 * 0.65
        return sorted_odds[0] * sorted_odds[1] * 0.65

    if "trifecta" in exotic_type.lower():
        if len(sorted_odds) >= 3:
            return sorted_odds[0] * sorted_odds[1] * sorted_odds[2] * 0.15 * field_factor
        return 5.0

    if "first4" in exotic_type.lower():
        if len(sorted_odds) >= 4:
            return (sorted_odds[0] * sorted_odds[1] * sorted_odds[2]
                    * sorted_odds[3] * 0.03 * field_factor)
        return 10.0

    return 1.0


# ── Data loading (reuse Betfair meta-model infrastructure) ──

def load_proform_races(data_dir: Path) -> list[dict]:
    """Load Proform data grouped by race with full runner details."""
    print(f"Loading Proform data from {data_dir}...")
    start = time.time()

    races = []
    total_runners = 0
    total_files = 0

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
                total_files += 1

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

                    # Need enough runners for exotics
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
                            continue

                        features = extract_features_proform(r, meta, field_size, avg_weight)
                        if len(features) != NUM_FEATURES:
                            continue

                        valid_runners.append(r)
                        valid_features.append(features)

                    if len(valid_runners) < 4:
                        continue  # Need 4+ for meaningful exotic simulation

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
    return races


def build_exotic_dataset(races: list[dict], rank_model) -> tuple:
    """For each race, simulate all exotic types and build training rows.

    Each row = one (race, exotic_type) pair with features and hit/miss label.

    Returns:
        X: np.ndarray of shape (n_samples, NUM_EXOTIC_FEATURES)
        y: np.ndarray of shape (n_samples,) — 1 if exotic hit, 0 if not
        weights: np.ndarray — estimated dividend for hits, 1.0 for misses
        dates: list of date strings
        exotic_types: list of exotic type names
    """
    print("\nBuilding exotic dataset from LGBM rankings...")
    start = time.time()

    X_all = []
    y_all = []
    w_all = []
    dates = []
    types_list = []

    type_stats = defaultdict(lambda: {"total": 0, "hits": 0})

    for race in races:
        meta = race["meta"]
        runners = race["runners"]
        features = race["features"]
        field_size = race["field_size"]

        if len(features) < 4:
            continue

        # Run LGBM rank model
        X = np.array(features, dtype=np.float64)
        model_n_features = rank_model.num_feature()
        if X.shape[1] > model_n_features:
            X = X[:, :model_n_features]

        raw_scores = rank_model.predict(X)

        # Softmax → win probabilities
        shifted = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(shifted)
        win_probs = exp_scores / np.sum(exp_scores)

        # Rank by WP → our picks
        ranked_indices = np.argsort(-win_probs)
        top4_indices = ranked_indices[:min(4, len(ranked_indices))]

        # Get finish positions for our top 4 picks
        top_positions = []
        top_wps = []
        top_odds = []
        for idx in top4_indices:
            pos = runners[idx].get("Position", 99)
            wp = float(win_probs[idx])
            sp = runners[idx].get("PriceSP", 0) or 0
            top_positions.append(pos)
            top_wps.append(wp)
            top_odds.append(float(sp) if sp > 1.0 else float("nan"))

        n_picks = len(top_positions)

        # Race context features
        distance = meta.get("distance", 1400)
        dist_bucket = _distance_bucket(distance)
        cls_bucket = _class_bucket(meta.get("race_class", ""))
        tc_bucket = _track_cond_bucket(meta.get("condition", ""))
        venue_type = _venue_type_code(meta.get("venue", ""))
        prize_money = float(meta.get("prize_money", 0) or 0)

        race_date = meta.get("date", "")

        # Simulate each exotic type
        for exotic_type, (checker, min_runners, num_combos, combo_ranks) in EXOTIC_CHECKERS.items():
            if n_picks < min_runners:
                continue

            # Field-size gates (same as production)
            if "Trifecta" in exotic_type and field_size < 7:
                continue
            if "First4" in exotic_type and field_size < 8:
                continue

            hit = checker(top_positions, n_picks)

            # Confidence-appropriate weighting: reward hits that match the
            # exotic type's intended use case, penalise lucky flukes.
            # This teaches the model to select exotics based on confidence
            # profile, not just value overlay.
            gap_12 = top_wps[0] - top_wps[1] if len(top_wps) >= 2 else 0
            gap_23 = top_wps[1] - top_wps[2] if len(top_wps) >= 3 else 0
            gap_34 = top_wps[2] - top_wps[3] if len(top_wps) >= 4 else 0

            if hit:
                est_div = _estimate_exotic_dividend(exotic_type, top_odds, field_size)
                base_weight = max(1.0, est_div)

                # Confidence multiplier: reward hits in appropriate contexts
                conf_mult = 1.0
                if exotic_type in ("Exacta",):
                    # Exacta should hit when gap_12 is large (high confidence in order)
                    conf_mult = 2.0 if gap_12 > 0.06 and top_wps[0] > 0.22 else 0.5
                elif exotic_type in ("Exacta Standout",):
                    # Exacta standout: rank 1 strong, moderate confidence in 2nd
                    conf_mult = 1.5 if top_wps[0] > 0.20 else 0.7
                elif exotic_type in ("Quinella",):
                    # Quinella: strong top 2, clear gap to 3rd
                    conf_mult = 1.5 if gap_23 > 0.04 else 0.8
                elif exotic_type in ("Quinella Box",):
                    # Quinella Box: open race, small gaps — this IS the right choice
                    conf_mult = 1.5 if gap_12 < 0.05 else 0.8
                elif exotic_type in ("Trifecta Standout",):
                    # Tri standout: rank 1 must be strong
                    conf_mult = 1.5 if top_wps[0] > 0.22 else 0.6
                elif exotic_type == "Trifecta Box":
                    # 3-runner: top 3 separated from field
                    conf_mult = 1.3 if gap_34 > 0.03 else 0.7
                elif exotic_type == "Trifecta Box 4":
                    # 4-runner: roughie must be live (genuine 4th contender)
                    conf_mult = 1.5 if top_wps[3] > 0.10 else 0.5
                elif exotic_type in ("First4",):
                    conf_mult = 1.0  # Rare, let data speak
                elif exotic_type in ("First4 Box",):
                    # Need all picks genuinely live
                    conf_mult = 1.3 if top_wps[3] > 0.08 else 0.5

                weight = base_weight * conf_mult
            else:
                # Misses: slight upweight when the exotic was appropriate but unlucky
                # (teaches model not to avoid appropriate types just because SR < 50%)
                weight = 1.0

            # Use "Trifecta Box" as the type code for both 3 and 4 runner variants
            # (production generates both under the same type name)
            model_type = exotic_type.replace(" 4", "")  # "Trifecta Box 4" → "Trifecta Box"

            combo_wps = [top_wps[r - 1] for r in combo_ranks if r - 1 < len(top_wps)]

            # Build feature vector
            from punty.betting.exotic_model import extract_exotic_features
            feat = extract_exotic_features(
                field_size=field_size,
                distance_bucket=dist_bucket,
                class_bucket=cls_bucket,
                track_cond_bucket=tc_bucket,
                venue_type=venue_type,
                prize_money=prize_money,
                rank_wps=top_wps,
                rank_odds=top_odds,
                exotic_type=model_type,
                num_combo_runners=len(combo_ranks),
                num_combos=num_combos,
                combo_runner_ranks=combo_ranks,
                combo_runner_wps=combo_wps,
            )

            if len(feat) != NUM_EXOTIC_FEATURES:
                continue

            X_all.append(feat)
            y_all.append(1 if hit else 0)
            w_all.append(weight)
            dates.append(race_date)
            types_list.append(model_type)

            type_stats[exotic_type]["total"] += 1  # Track original name for display
            if hit:
                type_stats[exotic_type]["hits"] += 1

    elapsed = time.time() - start
    total_hits = sum(y_all)
    print(f"  Dataset: {len(y_all):,} rows ({len(y_all)//len(EXOTIC_CHECKERS):,} races × {len(EXOTIC_CHECKERS)} types approx)")
    print(f"  Total hits: {total_hits:,} ({total_hits/len(y_all)*100:.1f}%)")
    print(f"  Built in {elapsed:.1f}s")

    print(f"\n  Per-type hit rates:")
    print(f"  {'Type':<20s} {'Total':>8s} {'Hits':>7s} {'SR%':>7s}")
    print(f"  {'-'*45}")
    for etype in EXOTIC_TYPE_CODES:
        stats = type_stats[etype]
        if stats["total"] > 0:
            sr = stats["hits"] / stats["total"] * 100
            print(f"  {etype:<20s} {stats['total']:>8,d} {stats['hits']:>7,d} {sr:>6.1f}%")

    return (
        np.array(X_all, dtype=np.float64),
        np.array(y_all, dtype=np.int32),
        np.array(w_all, dtype=np.float64),
        dates,
        types_list,
    )


def train_exotic_model(X_train, y_train, w_train, X_val, y_val, w_val) -> lgb.Booster:
    """Train the exotic meta-model binary classifier."""
    print(f"\n{'='*60}")
    print("Training Exotic meta-model")
    n_pos = y_train.sum()
    print(f"  Train: {len(y_train):,} rows, {n_pos:,} hits ({n_pos/len(y_train)*100:.1f}%)")
    n_vpos = y_val.sum()
    print(f"  Val:   {len(y_val):,} rows, {n_vpos:,} hits ({n_vpos/len(y_val)*100:.1f}%)")
    print(f"{'='*60}")

    train_data = lgb.Dataset(
        X_train, label=y_train, weight=w_train,
        feature_name=EXOTIC_FEATURE_NAMES, free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, weight=w_val,
        feature_name=EXOTIC_FEATURE_NAMES, free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        EXOTIC_LGBM_PARAMS,
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
    """Evaluate the model's per-type discrimination ability.

    For each race (set of rows with same index), pick the highest-scored
    exotic type and check if it hit. Compare to baselines.
    """
    probs = model.predict(X)

    # Group by exotic type
    type_results = defaultdict(lambda: {"total": 0, "hits": 0, "pred_sum": 0.0})
    for i, (etype, prob, actual) in enumerate(zip(types_list, probs, y)):
        type_results[etype]["total"] += 1
        type_results[etype]["hits"] += actual
        type_results[etype]["pred_sum"] += prob

    print(f"\n  {label} — Per-type model predictions:")
    print(f"  {'Type':<20s} {'N':>7s} {'Hits':>6s} {'SR%':>6s} {'AvgPred':>8s} {'Cal':>6s}")
    print(f"  {'-'*56}")
    for etype in EXOTIC_TYPE_CODES:
        r = type_results[etype]
        if r["total"] > 0:
            sr = r["hits"] / r["total"] * 100
            avg_pred = r["pred_sum"] / r["total"] * 100
            cal = f"{sr/avg_pred:.2f}x" if avg_pred > 0 else "N/A"
            print(f"  {etype:<20s} {r['total']:>7,d} {r['hits']:>6,d} {sr:>5.1f}% {avg_pred:>7.1f}% {cal:>6s}")


def evaluate_selection_accuracy(model, X, y, types_list, dates, label: str):
    """Simulate race-by-race exotic selection: did the model's top pick hit?

    Groups rows by date to approximate race grouping (since we don't have
    explicit race IDs, consecutive rows of 8 types form one race).
    """
    probs = model.predict(X)
    n_types = len(EXOTIC_TYPE_CODES)

    # Group into races (each race = n_types consecutive rows)
    n_races = len(X) // n_types
    model_hits = 0
    model_total = 0
    any_hit_total = 0

    # Also track which types the model selects
    selection_counts = defaultdict(int)
    selection_hits = defaultdict(int)

    for race_idx in range(n_races):
        start = race_idx * n_types
        end = start + n_types
        if end > len(X):
            break

        race_probs = probs[start:end]
        race_labels = y[start:end]
        race_types = types_list[start:end]

        # Model picks the highest predicted probability
        best_idx = np.argmax(race_probs)
        selected_type = race_types[best_idx]
        selected_hit = race_labels[best_idx]

        selection_counts[selected_type] += 1
        if selected_hit:
            selection_hits[selected_type] += 1
            model_hits += 1
        model_total += 1

        if any(race_labels):
            any_hit_total += 1

    model_sr = model_hits / model_total * 100 if model_total > 0 else 0
    any_sr = any_hit_total / model_total * 100 if model_total > 0 else 0

    print(f"\n  {label} — Race-by-race selection simulation ({model_total:,} races):")
    print(f"  Model's top pick hit: {model_hits:,}/{model_total:,} ({model_sr:.1f}%)")
    print(f"  Any exotic hit:       {any_hit_total:,}/{model_total:,} ({any_sr:.1f}%)")
    print(f"  Selection efficiency:  {model_sr/any_sr*100:.0f}% of possible hits captured" if any_sr > 0 else "")

    print(f"\n  Model selection distribution:")
    print(f"  {'Type':<20s} {'Selected':>9s} {'Hits':>6s} {'SR%':>6s}")
    print(f"  {'-'*44}")
    for etype in EXOTIC_TYPE_CODES:
        n = selection_counts[etype]
        h = selection_hits[etype]
        sr = h / n * 100 if n > 0 else 0
        print(f"  {etype:<20s} {n:>9,d} {h:>6,d} {sr:>5.1f}%")


def print_feature_importance(model: lgb.Booster):
    """Print feature importance."""
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(EXOTIC_FEATURE_NAMES, importance), key=lambda x: -x[1])

    print(f"\n  Feature Importance (gain):")
    max_imp = pairs[0][1] if pairs[0][1] > 0 else 1
    for name, imp in pairs:
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {name:22s} {imp:10.1f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train exotic meta-model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to Proform data directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Load main LGBM rank model
    if not RANK_MODEL_PATH.exists():
        print(f"ERROR: Main LGBM rank model not found at {RANK_MODEL_PATH}")
        print("  Train the main model first: python scripts/train_lgbm_v2.py")
        sys.exit(1)

    print("Loading main LGBM rank model...")
    rank_model = lgb.Booster(model_file=str(RANK_MODEL_PATH))
    print(f"  Loaded: {rank_model.num_trees()} trees, {rank_model.num_feature()} features")

    # Load Proform data
    races = load_proform_races(data_dir)
    if not races:
        print("ERROR: No races loaded")
        sys.exit(1)

    # Build exotic dataset
    X, y, weights, dates, types_list = build_exotic_dataset(races, rank_model)

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
        hits = y[mask].sum()
        print(f"  {name:6s}: {n:,} rows, {hits:,} hits ({hits/n*100:.1f}%)" if n > 0 else f"  {name:6s}: 0 rows")

    X_tr, y_tr, w_tr = X[train_mask], y[train_mask], weights[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], weights[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], weights[test_mask]

    types_tr = [t for t, m in zip(types_list, train_mask) if m]
    types_val = [t for t, m in zip(types_list, val_mask) if m]
    types_test = [t for t, m in zip(types_list, test_mask) if m]
    dates_val = [d for d, m in zip(dates, val_mask) if m]
    dates_test = [d for d, m in zip(dates, test_mask) if m]

    # Train
    model = train_exotic_model(X_tr, y_tr, w_tr, X_val, y_val, w_val)

    # Feature importance
    print_feature_importance(model)

    # Per-type evaluation
    evaluate_per_type(model, X_val, y_val, types_val, "Validation")
    evaluate_per_type(model, X_test, y_test, types_test, "Test (OOS)")

    # Race-by-race selection simulation
    evaluate_selection_accuracy(model, X_val, y_val, types_val, dates_val, "Validation")
    evaluate_selection_accuracy(model, X_test, y_test, types_test, dates_test, "Test (OOS)")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(EXOTIC_MODEL_PATH))

    metadata = {
        "version": 1,
        "model_type": "binary_classifier",
        "purpose": "Exotic type selector — predicts P(hit) for each exotic type given race context",
        "feature_names": EXOTIC_FEATURE_NAMES,
        "num_features": NUM_EXOTIC_FEATURES,
        "exotic_type_codes": EXOTIC_TYPE_CODES,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "total_races": len(races),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "temporal_split": {
            "train": f"< {val_cutoff}",
            "val": f"{val_cutoff} to {test_cutoff}",
            "test": f">= {test_cutoff}",
        },
        "lgbm_params": {k: v for k, v in EXOTIC_LGBM_PARAMS.items() if k != "verbose"},
        "best_iteration": model.best_iteration,
        "num_trees": model.num_trees(),
        "train_hit_rate": round(y_tr.mean() * 100, 2),
    }

    with open(EXOTIC_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exotic meta-model saved!")
    print(f"  Model:    {EXOTIC_MODEL_PATH}")
    print(f"  Metadata: {EXOTIC_METADATA_PATH}")
    print(f"  Trees:    {model.num_trees()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
