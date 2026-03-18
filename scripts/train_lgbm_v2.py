#!/usr/bin/env python3
"""Train LightGBM v2 — from Proform data with rich features.

Unlike v1 (trained on backtest.db with 20+ NaN features), this version loads
directly from Proform JSON files which have full A2E/PoT jockey/trainer stats,
combo career/last100, group/stakes records, avg margins, and settling positions.

Usage:
    python scripts/train_lgbm_v2.py
    python scripts/train_lgbm_v2.py --data-dir D:\\Punty\\DatafromProform
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from punty.ml.features import FEATURE_NAMES, NUM_FEATURES

DEFAULT_DATA_DIR = Path(r"D:\Punty\DatafromProform")

MONTH_DIRS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

MODEL_DIR = ROOT / "punty" / "data"

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
    "is_unbalanced": True,
}

EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 2000


# ── Proform helpers ─────────────────────────────────────────────────


def _record_sr(record: dict) -> float | None:
    starts = record.get("Starts", 0)
    if not starts or starts < 1:
        return None
    return record.get("Firsts", 0) / starts


def _parse_last10(last10: str) -> list[int | None]:
    if not last10:
        return []
    results = []
    for ch in last10.strip()[:10]:
        if ch == 'x':
            results.append(99)
        elif ch == '0':
            results.append(None)
        elif ch.isdigit():
            results.append(int(ch))
    return results


def _score_last5(positions: list[int | None]) -> float | None:
    """Score last 5 — mirrors features.py _score_last_five."""
    recent = [p for p in positions[:5] if p is not None]
    if not recent:
        return None
    scores = []
    for p in recent:
        if p == 1: scores.append(1.0)
        elif p == 2: scores.append(0.7)
        elif p == 3: scores.append(0.5)
        elif p == 4: scores.append(0.3)
        elif p <= 9: scores.append(0.1)
        else: scores.append(0.0)
    weights = [1.0, 0.9, 0.8, 0.7, 0.6][:len(scores)]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def _parse_flucs(flucs_str: str) -> tuple[float | None, float | None]:
    if not flucs_str:
        return None, None
    prices = {}
    for seg in flucs_str.split(";"):
        seg = seg.strip()
        if "," in seg:
            parts = seg.split(",", 1)
            try:
                prices[parts[0].strip()] = float(parts[1].strip())
            except (ValueError, IndexError):
                pass
    return prices.get("opening"), prices.get("starting")


def _parse_settling(in_run: str, field_size: int) -> float | None:
    if not in_run:
        return None
    for part in in_run.strip(";").split(";"):
        if "," in part:
            k, v = part.split(",", 1)
            if k.strip() == "settling_down":
                try:
                    pos = int(v.strip())
                    return pos / field_size if field_size > 1 else pos / 12
                except (ValueError, TypeError):
                    pass
    return None


def _avg_margin(forms: list[dict], n: int = 5) -> float | None:
    margins = []
    for f in forms[:n]:
        if f.get("IsBarrierTrial"):
            continue
        m = f.get("Margin")
        if m is not None and isinstance(m, (int, float)):
            margins.append(abs(m))
    return statistics.mean(margins) if margins else None


def _a2e_field(a2e_dict: dict, field: str, min_runners: int = 20) -> float | None:
    if not a2e_dict or not isinstance(a2e_dict, dict):
        return None
    runners = a2e_dict.get("Runners", 0)
    if not runners or runners < min_runners:
        return None
    val = a2e_dict.get(field)
    return float(val) if val is not None else None


def _sf(val) -> float:
    """Safe float — return NaN for None."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _record_place_rate(record: dict) -> float | None:
    """Place rate (1st+2nd+3rd / starts) from a Proform record dict."""
    starts = record.get("Starts", 0)
    if not starts or starts < 1:
        return None
    firsts = record.get("Firsts", 0)
    seconds = record.get("Seconds", 0)
    thirds = record.get("Thirds", 0)
    return (firsts + seconds + thirds) / starts


def _form_trend_proform(positions: list[int | None]) -> float | None:
    """Linear regression slope of L5 finishing positions (negative = improving)."""
    vals = []
    for p in positions[:5]:
        if p is None:
            continue
        vals.append(p if p < 99 else 10)
    if len(vals) < 3:
        return None
    n = len(vals)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(vals) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, vals))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    return num / den if den != 0 else 0.0


# ── Feature extraction from Proform runner ──────────────────────────


def extract_features_proform(pf_runner: dict, race_meta: dict,
                              field_size: int, avg_weight: float) -> list[float]:
    """Extract 61-feature vector from Proform runner dict.

    Uses the same FEATURE_NAMES order as features.py but fills in ALL fields
    that Proform provides (A2E, combo, groups, margins, etc).
    """
    nan = float("nan")
    forms = pf_runner.get("Forms", [])
    real_forms = [f for f in forms if not f.get("IsBarrierTrial")]
    distance = race_meta.get("distance", 1400)

    # ── Market ──
    sp = pf_runner.get("PriceSP", 0)
    market_prob = 1.0 / sp if sp and sp > 1.0 else nan

    # ── Career ──
    win_pct = pf_runner.get("WinPct", 0)
    place_pct = pf_runner.get("PlacePct", 0)
    career_starts = pf_runner.get("CareerStarts", 0)
    if not career_starts and win_pct and pf_runner.get("CareerWins", 0):
        career_starts = round(pf_runner["CareerWins"] / (win_pct / 100))
    career_win_pct = win_pct / 100 if win_pct else nan
    career_place_pct = place_pct / 100 if place_pct else nan

    # ── Track/distance records ──
    td = pf_runner.get("TrackDistRecord", {})
    td_sr = _record_sr(td)
    td_starts = td.get("Starts", 0)

    dr = pf_runner.get("DistanceRecord", {})
    dist_sr = _record_sr(dr)
    dist_starts = dr.get("Starts", 0)

    tr = pf_runner.get("TrackRecord", {})
    trk_sr = _record_sr(tr)
    trk_starts = tr.get("Starts", 0)

    # ── Condition records ──
    cond_data = {}
    for cond_name in ("Good", "Soft", "Heavy", "Firm", "Synthetic"):
        rec = pf_runner.get(f"{cond_name}Record", {})
        cond_data[cond_name.lower()] = (_record_sr(rec), rec.get("Starts", 0))

    # ── Freshness ──
    fu = pf_runner.get("FirstUpRecord", {})
    fu_sr = _record_sr(fu)
    fu_starts = fu.get("Starts", 0)
    su = pf_runner.get("SecondUpRecord", {})
    su_sr = _record_sr(su)
    su_starts = su.get("Starts", 0)

    # ── Last 5 form ──
    last10 = pf_runner.get("Last10", "")
    positions = _parse_last10(last10)
    l5_score = _score_last5(positions)
    recent5 = [p for p in positions[:5] if p is not None]
    l5_wins = sum(1 for p in recent5 if p == 1) if recent5 else nan
    l5_places = sum(1 for p in recent5 if p and p <= 3) if recent5 else nan

    # ── New: form trend (slope of L5 positions) ──
    form_trend_val = _sf(_form_trend_proform(positions))

    # ── New: place_vs_market (career place rate - field implied place rate) ──
    place_count = 2 if field_size <= 7 else 3
    market_implied_place = (place_count / field_size) if field_size > 0 else nan
    place_vs_market = (
        career_place_pct - market_implied_place
        if not math.isnan(career_place_pct) and not math.isnan(market_implied_place)
        else nan
    )

    # ── Class & fitness ──
    prize = pf_runner.get("PrizeMoney", 0)
    prize_per_start = prize / career_starts if prize and career_starts and career_starts > 0 else nan
    handicap = _sf(pf_runner.get("HandicapRating"))
    avg_marg = _sf(_avg_margin(real_forms))

    # ── New: class differential (race prize / career prize per start) ──
    try:
        race_prize = float(race_meta.get("prize_money", 0) or 0)
    except (ValueError, TypeError):
        race_prize = 0.0
    class_diff = (
        race_prize / prize_per_start
        if race_prize and not math.isnan(prize_per_start) and prize_per_start > 0
        else nan
    )

    # ── Pace ──
    # Days since last run
    days_since = nan
    if real_forms:
        try:
            last_date = real_forms[0].get("MeetingDate", "")
            race_date = race_meta.get("date", "")
            if last_date and race_date:
                ld = datetime.fromisoformat(last_date.replace("T00:00:00", ""))
                rd = datetime.strptime(race_date[:10], "%Y-%m-%d")
                days_since = float((rd - ld).days)
        except (ValueError, TypeError):
            pass

    # Settle position
    settle_vals = []
    for f in real_forms[:5]:
        s = _parse_settling(f.get("InRun", ""), field_size)
        if s is not None:
            settle_vals.append(s)
    settle_pos = statistics.mean(settle_vals) if settle_vals else nan

    # ── Barrier ──
    barrier = pf_runner.get("Barrier", 0) or 0
    barrier_relative = (barrier - 1) / (field_size - 1) if barrier and field_size > 1 else nan
    barrier_raw = float(barrier) if barrier else nan

    # ── Jockey stats (RICH — A2E, PoT, L100) ──
    jc = pf_runner.get("JockeyA2E_Career", {})
    jockey_career_sr = _sf(_a2e_field(jc, "StrikeRate", 50))
    jockey_career_a2e = _sf(_a2e_field(jc, "A2E", 50))
    jockey_career_pot = _sf(_a2e_field(jc, "PoT", 50))
    jockey_career_runners = _sf(jc.get("Runners", 0) if jc else 0)

    jl = pf_runner.get("JockeyA2E_Last100", {})
    jockey_l100_sr = _sf(_a2e_field(jl, "StrikeRate", 20))

    # ── Trainer stats (RICH) ──
    tc = pf_runner.get("TrainerA2E_Career", {})
    trainer_career_sr = _sf(_a2e_field(tc, "StrikeRate", 50))
    trainer_career_a2e = _sf(_a2e_field(tc, "A2E", 50))
    trainer_career_pot = _sf(_a2e_field(tc, "PoT", 50))

    tl = pf_runner.get("TrainerA2E_Last100", {})
    trainer_l100_sr = _sf(_a2e_field(tl, "StrikeRate", 20))

    # ── Combo stats (RICH) ──
    cc = pf_runner.get("TrainerJockeyA2E_Career", {})
    combo_career_sr = _sf(_a2e_field(cc, "StrikeRate", 10))
    combo_career_runners = _sf(cc.get("Runners", 0) if cc else 0)

    cl = pf_runner.get("TrainerJockeyA2E_Last100", {})
    combo_l100_sr = _sf(_a2e_field(cl, "StrikeRate", 10))

    # ── Physical ──
    weight = _sf(pf_runner.get("Weight", 0) or None)
    weight_diff = weight - avg_weight if not math.isnan(weight) and avg_weight else nan
    age = _sf(pf_runner.get("Age"))
    sex = (pf_runner.get("Sex") or "").lower()
    is_gelding = 1.0 if "gelding" in sex else 0.0
    is_mare = 1.0 if ("mare" in sex or "filly" in sex) else 0.0

    # ── Movement ──
    price_move_pct = nan
    if real_forms:
        opening, starting = _parse_flucs(real_forms[0].get("Flucs", ""))
        if opening and starting and opening > 1 and starting > 1:
            price_move_pct = (opening - starting) / opening

    # ── Group/stakes records (RICH) ──
    g1 = pf_runner.get("Group1Record", {})
    g2 = pf_runner.get("Group2Record", {})
    g3 = pf_runner.get("Group3Record", {})
    grp_starts = sum(r.get("Starts", 0) for r in [g1, g2, g3])
    grp_wins = sum(r.get("Firsts", 0) for r in [g1, g2, g3])
    group_sr = grp_wins / grp_starts if grp_starts >= 3 else nan

    # ── New: place specialist rates ──
    dist_place = _sf(_record_place_rate(dr))
    trk_place = _sf(_record_place_rate(tr))

    # ── v4 features ──
    # Leader signal — from settle position
    is_leader = 1.0 if (settle_pos and not math.isnan(settle_pos) and settle_pos <= 2) else 0.0

    # Staying race
    is_staying = 1.0 if distance and distance >= 2000 else 0.0

    # Last start won
    l5_str = str(pf_runner.get("Last5Starts", "") or "")
    last_start_won = 1.0 if l5_str and l5_str.lstrip("0").startswith("1") else 0.0

    # Country — from Proform state/venue
    pf_state = race_meta.get("state", "")
    pf_venue = (race_meta.get("venue", "") or "").lower()
    if pf_state == "HK" or pf_venue in ("sha tin", "happy valley"):
        is_australia, is_hong_kong, is_new_zealand = 0.0, 1.0, 0.0
    elif pf_state == "NZ" or pf_venue in ("ellerslie", "trentham", "riccarton", "te rapa", "otaki", "wingatui", "tauranga", "wanganui", "hastings", "awapuni", "pukekohe"):
        is_australia, is_hong_kong, is_new_zealand = 0.0, 0.0, 1.0
    else:
        is_australia, is_hong_kong, is_new_zealand = 1.0, 0.0, 0.0

    # ── v5 features: pace/speed map ──
    # Proform doesn't have pf_map_factor/speed_rank/jockey_factor directly in form JSONs
    # but InRun settling is available. Map factor and speed rank are NaN in training.
    pf_mf_val = nan  # Not in Proform training data
    pf_sr_val = nan  # Not in Proform training data
    pf_jf_val = nan  # Not in Proform training data
    # Speed map encoded from settling position
    if not math.isnan(settle_pos) and settle_pos > 0:
        if settle_pos <= 0.15:
            sme = 1.0   # leader
        elif settle_pos <= 0.35:
            sme = 2.0   # on_pace
        elif settle_pos <= 0.65:
            sme = 3.0   # midfield
        else:
            sme = 4.0   # backmarker
    else:
        sme = 0.0

    # ── v5: weather ──
    # Proform has track condition but not live weather — use NaN
    weather_rain = nan
    weather_wind = nan
    weather_temp_val = nan
    weather_hum = nan

    # ── v5: gear ──
    gear_str = (pf_runner.get("Gear", "") or "").lower()
    gear_change_str = (pf_runner.get("GearChange", "") or "")
    has_gc = 1.0 if gear_change_str.strip() else 0.0
    blinkers = 1.0 if "blinker" in gear_str or "blink" in gear_str else 0.0

    # ── v5: campaign / fatigue ──
    runs_prep = 0
    prep_wins = 0
    for i, f in enumerate(real_forms):
        if i > 0:
            try:
                d1 = datetime.fromisoformat(f.get("MeetingDate", "").replace("T00:00:00", ""))
                d0 = datetime.fromisoformat(real_forms[i-1].get("MeetingDate", "").replace("T00:00:00", ""))
                if abs((d0 - d1).days) > 60:
                    break
            except (ValueError, TypeError):
                pass
        runs_prep += 1
        if f.get("Position") == 1:
            prep_wins += 1
    runs_this_prep_val = float(runs_prep) if runs_prep > 0 else nan
    campaign_wr_val = prep_wins / runs_prep if runs_prep > 0 else nan

    # ── v5: class specialist ──
    # Use the current race class to look for matching class in form history
    race_class = (race_meta.get("race_class", "") or "").lower()
    class_starts = 0
    class_wins = 0
    for f in real_forms:
        fc = (f.get("RaceClass", "") or "").rstrip(";").strip().lower()
        if fc and race_class and fc == race_class:
            class_starts += 1
            if f.get("Position") == 1:
                class_wins += 1
    class_wr = class_wins / class_starts if class_starts >= 2 else nan

    # ── v5: track specialist ──
    trk_specialist = 1.0 if trk_starts >= 3 and trk_sr is not None and trk_sr >= 0.25 else 0.0

    # ── v5: stewards / excuses ──
    has_stew = 0.0
    has_exc = 0.0
    if real_forms:
        stew_txt = real_forms[0].get("StewardsComment", "") or ""
        if stew_txt.strip():
            has_stew = 1.0
        # Check for common excuse keywords in last start
        comment = (real_forms[0].get("Comment", "") or "").lower()
        excuse_keywords = ("checked", "held up", "blocked", "bumped", "hampered",
                          "steadied", "crowded", "shifted", "lost rider", "eased")
        if any(kw in comment for kw in excuse_keywords):
            has_exc = 1.0

    # ── v5: age×sex peak ──
    age_val = _sf(pf_runner.get("Age"))
    if not math.isnan(age_val):
        if 4 <= age_val <= 5:
            peak = 1.0
        elif age_val == 3:
            peak = 0.85
        elif age_val == 6:
            peak = 0.90
        elif age_val == 7:
            peak = 0.75
        elif age_val >= 8:
            peak = 0.60
        elif age_val <= 2:
            peak = 0.70
        else:
            peak = 0.80
        if is_mare == 1.0 and age_val == 3:
            peak *= 0.85
    else:
        peak = nan

    # ── v5: flucs direction ──
    if real_forms:
        opening_f, starting_f = _parse_flucs(real_forms[0].get("Flucs", ""))
        if opening_f and starting_f and opening_f > 1 and starting_f > 1:
            pct_chg = (opening_f - starting_f) / opening_f
            if pct_chg > 0.10:
                flucs_dir = 1.0   # firming
            elif pct_chg < -0.10:
                flucs_dir = -1.0  # drifting
            else:
                flucs_dir = 0.0
        else:
            flucs_dir = nan
    else:
        flucs_dir = nan

    # ── v5: head-to-head and field beaten (need field context — NaN in training) ──
    h2h_wins = nan
    # Field beaten % from form history
    fb_pcts = []
    for f in real_forms[:5]:
        pos = f.get("Position")
        fld = f.get("Starters") or f.get("FieldSize", 0)
        if pos and fld and fld > 1:
            fb_pcts.append((fld - pos) / (fld - 1))
    field_beaten = statistics.mean(fb_pcts) if fb_pcts else nan

    # ── v5: rail bias ── (not available in Proform training data)
    rail_bias = nan

    # ── Build feature vector (MUST match FEATURE_NAMES order) ──
    return [
        market_prob,
        career_win_pct, career_place_pct, _sf(career_starts),
        _sf(td_sr), float(td_starts),
        _sf(dist_sr), float(dist_starts),
        _sf(trk_sr), float(trk_starts),
        _sf(cond_data["good"][0]), float(cond_data["good"][1]),
        _sf(cond_data["soft"][0]), float(cond_data["soft"][1]),
        _sf(cond_data["heavy"][0]), float(cond_data["heavy"][1]),
        _sf(cond_data["firm"][0]), float(cond_data["firm"][1]),
        _sf(cond_data["synthetic"][0]), float(cond_data["synthetic"][1]),
        _sf(fu_sr), float(fu_starts),
        _sf(su_sr), float(su_starts),
        _sf(l5_score), _sf(l5_wins), _sf(l5_places),
        form_trend_val, _sf(place_vs_market),
        _sf(prize_per_start), handicap, avg_marg,
        _sf(class_diff),
        _sf(days_since), settle_pos,
        _sf(barrier_relative), barrier_raw,
        jockey_career_sr, jockey_career_a2e, jockey_career_pot,
        jockey_career_runners, jockey_l100_sr,
        trainer_career_sr, trainer_career_a2e,
        trainer_career_pot, trainer_l100_sr,
        combo_career_sr, combo_career_runners, combo_l100_sr,
        weight, weight_diff, _sf(age),
        is_gelding, is_mare,
        price_move_pct,
        float(grp_starts), _sf(group_sr),
        dist_place, trk_place,
        float(field_size), float(distance),
        # ── v4 features ──
        is_leader, is_staying, last_start_won,
        is_australia, is_hong_kong, is_new_zealand,
        _sf(h2h_wins), _sf(field_beaten),
        _sf(rail_bias),
        # ── v5 features ──
        _sf(pf_mf_val), _sf(pf_sr_val), _sf(pf_jf_val), sme,
        _sf(weather_rain), _sf(weather_wind), _sf(weather_temp_val), _sf(weather_hum),
        has_gc, blinkers,
        _sf(runs_this_prep_val), _sf(campaign_wr_val),
        _sf(class_wr), trk_specialist,
        has_stew, has_exc,
        _sf(peak), _sf(flucs_dir),
    ]


# ── Data loading ────────────────────────────────────────────────────


def load_proform_data(data_dir: Path) -> tuple[list[list[float]], list[int], list[int], list[str], list[str]]:
    """Load all Proform data, extract features, return X, y_win, y_place, race_ids, dates."""
    print(f"Loading Proform data from {data_dir}...")
    start = time.time()

    X_list = []
    y_win = []
    y_place = []
    race_ids = []
    dates = []
    total_runners = 0
    total_files = 0
    skipped_no_pos = 0
    skipped_no_sp = 0

    # Process year directories — deduplicate by using only '2025' if both exist
    # (2026 directory contains the same data with 2025 dates)
    year_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if len(year_dirs) > 1:
        # Check if multiple dirs contain same data by looking at file prefixes
        # Use only the dir matching the actual year in meeting dates
        year_dirs = [year_dirs[0]]
        print(f"  Note: Using only {year_dirs[0].name}/ directory (others contain duplicates)")
    for year_dir in year_dirs:
        year = year_dir.name
        print(f"\n  Processing {year}...")

        for month_num in range(1, 13):
            month_name = MONTH_DIRS[month_num]

            # Load race metadata
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

            # Load form files
            form_dir = year_dir / month_name / "Form"
            if not form_dir.exists():
                continue

            month_count = 0
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
                races = defaultdict(list)
                for r in runners:
                    rid = r.get("RaceId")
                    if rid:
                        try:
                            races[int(rid)].append(r)
                        except (ValueError, TypeError):
                            pass

                for race_id, race_runners in races.items():
                    meta = race_meta.get(race_id, {})
                    if not meta:
                        continue

                    field_size = len(race_runners)
                    meta["field_size"] = field_size

                    # Compute avg weight
                    weights = [r.get("Weight", 0) for r in race_runners if r.get("Weight", 0) and r["Weight"] > 40]
                    avg_weight = statistics.mean(weights) if weights else 56.0

                    race_date = meta.get("date", "")
                    race_id_str = f"{meta.get('venue', 'unk')}-{race_date}-{race_id}"

                    for r in race_runners:
                        pos = r.get("Position")
                        if pos is None or pos == 0:
                            skipped_no_pos += 1
                            continue
                        sp = r.get("PriceSP", 0)
                        if not sp or sp <= 1.0:
                            skipped_no_sp += 1
                            continue

                        features = extract_features_proform(r, meta, field_size, avg_weight)
                        if len(features) != NUM_FEATURES:
                            continue

                        X_list.append(features)
                        y_win.append(1 if pos == 1 else 0)
                        place_cutoff = 2 if field_size <= 7 else 3
                        y_place.append(1 if pos <= place_cutoff else 0)
                        race_ids.append(race_id_str)
                        dates.append(race_date)
                        total_runners += 1
                        month_count += 1

            if month_count > 0:
                print(f"    {month_name}: {month_count:,} runners from {total_files} files")

    elapsed = time.time() - start
    print(f"\nTotal: {total_runners:,} runners from {total_files} files in {elapsed:.1f}s")
    print(f"  Skipped (no position): {skipped_no_pos:,}")
    print(f"  Skipped (no SP): {skipped_no_sp:,}")

    return X_list, y_win, y_place, race_ids, dates


# ── Training & evaluation (reused from v1) ──────────────────────────


def train_model(X_train, y_train, X_val, y_val, label: str) -> lgb.Booster:
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


def evaluate_model(model, X, y, race_ids, label: str) -> dict:
    probs = model.predict(X)

    ll = log_loss(y, probs)
    auc = roc_auc_score(y, probs)

    print(f"\n  {label} Results:")
    print(f"    Log Loss: {ll:.4f}")
    print(f"    AUC:      {auc:.4f}")

    # Race-level accuracy
    race_groups = defaultdict(list)
    for i, rid in enumerate(race_ids):
        race_groups[rid].append((probs[i], y[i], i))

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_races = 0

    for rid, entries in race_groups.items():
        entries.sort(key=lambda x: -x[0])
        total_races += 1
        if entries[0][1] == 1:
            top1_correct += 1
        if any(e[1] == 1 for e in entries[:3]):
            top3_correct += 1
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

    return {
        "log_loss": round(ll, 4),
        "auc": round(auc, 4),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "n_races": total_races,
        "n_samples": len(y),
    }


def print_feature_importance(model: lgb.Booster, label: str):
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])

    print(f"\n  {label} Top-20 Feature Importance (gain):")
    for name, imp in pairs[:20]:
        bar = "#" * int(imp / pairs[0][1] * 30)
        print(f"    {name:25s} {imp:10.1f}  {bar}")

    # Also show previously-NaN features that now have signal
    rich_features = {
        "jockey_career_a2e", "jockey_career_pot", "jockey_l100_sr",
        "trainer_career_a2e", "trainer_career_pot", "trainer_l100_sr",
        "combo_career_sr", "combo_career_runners", "combo_l100_sr",
        "group_starts", "group_sr", "avg_margin",
        "cond_firm_sr", "cond_synthetic_sr",
        "form_trend", "place_vs_market", "class_differential",
        "distance_place_rate", "track_place_rate",
    }
    print(f"\n  {label} Rich Feature Signal (previously NaN in v1):")
    for name, imp in pairs:
        if name in rich_features:
            bar = "#" * int(imp / pairs[0][1] * 30) if pairs[0][1] > 0 else ""
            print(f"    {name:25s} {imp:10.1f}  {bar}")


def _quick_train(params, X_tr, y_tr, X_val, y_val):
    """Train a model silently, return (booster, best_iteration, val_logloss, val_auc)."""
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, free_raw_data=False)
    callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)]
    model = lgb.train(params, train_data, num_boost_round=NUM_BOOST_ROUND,
                      valid_sets=[val_data], valid_names=["val"], callbacks=callbacks)
    probs = model.predict(X_val)
    ll = log_loss(y_val, probs)
    auc = roc_auc_score(y_val, probs)
    return model, model.best_iteration, ll, auc


def _race_top1(model, X, y, race_ids):
    """Compute race-level top-1 accuracy."""
    probs = model.predict(X)
    groups = defaultdict(list)
    for i, rid in enumerate(race_ids):
        groups[rid].append((probs[i], y[i]))
    correct = sum(1 for entries in groups.values() if sorted(entries, key=lambda x: -x[0])[0][1] == 1)
    return correct / len(groups) if groups else 0


def hyperparameter_search(X_tr, yw_tr, yp_tr, X_val, yw_val, yp_val, rids_val):
    """Grid search over key LGBM hyperparams. Returns best params for win and place."""
    param_grid = {
        "num_leaves": [31, 63, 127],
        "max_depth": [5, 7, 10, -1],
        "learning_rate": [0.02, 0.05, 0.1],
        "min_data_in_leaf": [20, 50, 100],
        "feature_fraction": [0.6, 0.8, 1.0],
        "bagging_fraction": [0.7, 0.8, 0.9],
        "lambda_l1": [0.0, 0.1, 1.0],
        "lambda_l2": [0.0, 1.0, 5.0],
    }

    # Phase 1: tree structure (num_leaves × max_depth × min_data_in_leaf)
    # Phase 2: sampling (feature_fraction × bagging_fraction)
    # Phase 3: regularisation (lambda_l1 × lambda_l2)
    # Phase 4: learning rate with best of above
    # This staged approach = 3×4×3 + 3×3 + 3×3 + 3 = 57 combos (not 3^8 = 6561)

    base = dict(LGBM_PARAMS)
    best_results = {}

    for model_label, y_tr, y_val in [("Win", yw_tr, yw_val), ("Place", yp_tr, yp_val)]:
        print(f"\n{'='*60}")
        print(f"  Hyperparameter search: {model_label} model")
        print(f"{'='*60}")

        best_ll = float("inf")
        best_params = dict(base)

        # Phase 1: tree structure
        print(f"\n  Phase 1: Tree structure ({3*4*3} combos)...")
        p1_results = []
        for nl in param_grid["num_leaves"]:
            for md in param_grid["max_depth"]:
                for mdl in param_grid["min_data_in_leaf"]:
                    p = dict(base)
                    p["num_leaves"] = nl
                    p["max_depth"] = md
                    p["min_data_in_leaf"] = mdl
                    _, itr, ll, auc = _quick_train(p, X_tr, y_tr, X_val, y_val)
                    p1_results.append((ll, auc, itr, nl, md, mdl))
                    if ll < best_ll:
                        best_ll = ll
                        best_params.update({"num_leaves": nl, "max_depth": md, "min_data_in_leaf": mdl})

        p1_results.sort()
        print(f"    Top 5 tree combos:")
        for ll, auc, itr, nl, md, mdl in p1_results[:5]:
            print(f"      leaves={nl:3d} depth={md:2d} min_leaf={mdl:3d} -> logloss={ll:.5f} auc={auc:.4f} itr={itr}")

        # Phase 2: sampling
        print(f"\n  Phase 2: Sampling ({3*3} combos)...")
        p2_results = []
        for ff in param_grid["feature_fraction"]:
            for bf in param_grid["bagging_fraction"]:
                p = dict(best_params)
                p["feature_fraction"] = ff
                p["bagging_fraction"] = bf
                _, itr, ll, auc = _quick_train(p, X_tr, y_tr, X_val, y_val)
                p2_results.append((ll, auc, itr, ff, bf))
                if ll < best_ll:
                    best_ll = ll
                    best_params.update({"feature_fraction": ff, "bagging_fraction": bf})

        p2_results.sort()
        print(f"    Top 3 sampling combos:")
        for ll, auc, itr, ff, bf in p2_results[:3]:
            print(f"      feat_frac={ff:.1f} bag_frac={bf:.1f} -> logloss={ll:.5f} auc={auc:.4f}")

        # Phase 3: regularisation
        print(f"\n  Phase 3: Regularisation ({3*3} combos)...")
        p3_results = []
        for l1 in param_grid["lambda_l1"]:
            for l2 in param_grid["lambda_l2"]:
                p = dict(best_params)
                p["lambda_l1"] = l1
                p["lambda_l2"] = l2
                _, itr, ll, auc = _quick_train(p, X_tr, y_tr, X_val, y_val)
                p3_results.append((ll, auc, itr, l1, l2))
                if ll < best_ll:
                    best_ll = ll
                    best_params.update({"lambda_l1": l1, "lambda_l2": l2})

        p3_results.sort()
        print(f"    Top 3 reg combos:")
        for ll, auc, itr, l1, l2 in p3_results[:3]:
            print(f"      L1={l1:.1f} L2={l2:.1f} -> logloss={ll:.5f} auc={auc:.4f}")

        # Phase 4: learning rate (lower LR + more trees)
        print(f"\n  Phase 4: Learning rate refinement...")
        p4_results = []
        for lr in param_grid["learning_rate"]:
            p = dict(best_params)
            p["learning_rate"] = lr
            _, itr, ll, auc = _quick_train(p, X_tr, y_tr, X_val, y_val)
            p4_results.append((ll, auc, itr, lr))
            if ll < best_ll:
                best_ll = ll
                best_params["learning_rate"] = lr

        p4_results.sort()
        for ll, auc, itr, lr in p4_results:
            print(f"      lr={lr:.3f} -> logloss={ll:.5f} auc={auc:.4f} itr={itr}")

        # Summary
        print(f"\n  Best {model_label} params (logloss={best_ll:.5f}):")
        for k in ["num_leaves", "max_depth", "min_data_in_leaf", "feature_fraction",
                   "bagging_fraction", "lambda_l1", "lambda_l2", "learning_rate"]:
            print(f"    {k}: {best_params[k]}")

        best_results[model_label] = best_params

    return best_results


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM v2 from Proform data")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to Proform data directory")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter search before final training")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    X_list, y_win_list, y_place_list, race_ids, dates = load_proform_data(data_dir)

    X = np.array(X_list, dtype=np.float64)
    y_win = np.array(y_win_list, dtype=np.int32)
    y_place = np.array(y_place_list, dtype=np.int32)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Win rate:   {y_win.mean()*100:.1f}%")
    print(f"  Place rate: {y_place.mean()*100:.1f}%")

    # Check NaN rates for previously-all-NaN features
    print(f"\n  Feature coverage (non-NaN %):")
    rich_features = [
        "jockey_career_a2e", "jockey_career_pot", "jockey_l100_sr",
        "trainer_career_a2e", "trainer_career_pot", "trainer_l100_sr",
        "combo_career_sr", "combo_l100_sr", "group_sr", "avg_margin",
        "form_trend", "place_vs_market", "class_differential",
        "distance_place_rate", "track_place_rate",
    ]
    for feat in rich_features:
        idx = FEATURE_NAMES.index(feat)
        non_nan = np.sum(~np.isnan(X[:, idx]))
        pct = non_nan / X.shape[0] * 100
        print(f"    {feat:25s}: {pct:5.1f}% ({non_nan:,} / {X.shape[0]:,})")

    # ── Temporal split ──
    # 2025 Jan-Oct: train, 2025 Nov: val, 2025 Dec + 2026: test
    train_mask = np.array([d[:7] <= "2025-10" for d in dates])
    val_mask = np.array([d[:7] == "2025-11" for d in dates])
    test_mask = np.array([d[:7] >= "2025-12" for d in dates])

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n = mask.sum()
        races = len(set(r for r, m in zip(race_ids, mask) if m))
        print(f"  {name:6s}: {n:,} samples, {races:,} races")

    X_tr, yw_tr, yp_tr = X[train_mask], y_win[train_mask], y_place[train_mask]
    X_val, yw_val, yp_val = X[val_mask], y_win[val_mask], y_place[val_mask]
    X_test, yw_test, yp_test = X[test_mask], y_win[test_mask], y_place[test_mask]
    rids_tr = [r for r, m in zip(race_ids, train_mask) if m]
    rids_val = [r for r, m in zip(race_ids, val_mask) if m]
    rids_test = [r for r, m in zip(race_ids, test_mask) if m]

    # ── Hyperparameter search (optional) ──
    if args.tune:
        best_params = hyperparameter_search(X_tr, yw_tr, yp_tr, X_val, yw_val, yp_val, rids_val)
        # Apply best params globally for final training
        global LGBM_PARAMS
        # Use win params as base, override per model in train calls
        print(f"\n{'='*60}")
        print("  Applying tuned params for final training")
        print(f"{'='*60}")
    else:
        best_params = None

    # ── Train models ──
    if best_params and "Win" in best_params:
        win_lgbm = best_params["Win"]
    else:
        win_lgbm = LGBM_PARAMS

    if best_params and "Place" in best_params:
        place_lgbm = best_params["Place"]
    else:
        place_lgbm = LGBM_PARAMS

    win_model = train_model(X_tr, yw_tr, X_val, yw_val, "Win")
    if best_params:
        # Retrain with tuned params
        print("\n  Re-training Win with tuned params...")
        train_data = lgb.Dataset(X_tr, label=yw_tr, feature_name=FEATURE_NAMES, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=yw_val, feature_name=FEATURE_NAMES, free_raw_data=False)
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(period=100)]
        win_model = lgb.train(win_lgbm, train_data, num_boost_round=NUM_BOOST_ROUND,
                              valid_sets=[train_data, val_data], valid_names=["train", "val"],
                              callbacks=callbacks)
    print_feature_importance(win_model, "Win")

    print("\n--- Win Model Evaluation ---")
    val_metrics_win = evaluate_model(win_model, X_val, yw_val, rids_val, "Validation")
    test_metrics_win = evaluate_model(win_model, X_test, yw_test, rids_test, "Test")

    place_model = train_model(X_tr, yp_tr, X_val, yp_val, "Place")
    if best_params:
        print("\n  Re-training Place with tuned params...")
        train_data = lgb.Dataset(X_tr, label=yp_tr, feature_name=FEATURE_NAMES, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=yp_val, feature_name=FEATURE_NAMES, free_raw_data=False)
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True), lgb.log_evaluation(period=100)]
        place_model = lgb.train(place_lgbm, train_data, num_boost_round=NUM_BOOST_ROUND,
                                valid_sets=[train_data, val_data], valid_names=["train", "val"],
                                callbacks=callbacks)
    print_feature_importance(place_model, "Place")

    print("\n--- Place Model Evaluation ---")
    val_metrics_place = evaluate_model(place_model, X_val, yp_val, rids_val, "Validation")
    test_metrics_place = evaluate_model(place_model, X_test, yp_test, rids_test, "Test")

    # ── Save models ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    win_path = MODEL_DIR / "lgbm_win_model.txt"
    place_path = MODEL_DIR / "lgbm_place_model.txt"
    meta_path = MODEL_DIR / "lgbm_metadata.json"

    win_model.save_model(str(win_path))
    place_model.save_model(str(place_path))

    metadata = {
        "version": 2,
        "feature_names": FEATURE_NAMES,
        "num_features": NUM_FEATURES,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "Proform JSON",
        "data_dir": str(data_dir),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "temporal_split": {"train": "2025-01 to 2025-10", "val": "2025-11", "test": "2025-12+"},
        "lgbm_params_default": LGBM_PARAMS,
        "lgbm_params_win": {k: v for k, v in win_lgbm.items() if k != "verbose"},
        "lgbm_params_place": {k: v for k, v in place_lgbm.items() if k != "verbose"},
        "tuned": args.tune,
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
    print(f"Models saved!")
    print(f"  Win:      {win_path} ({win_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Place:    {place_path} ({place_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
