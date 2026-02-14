"""Build context profiles for the probability engine.

Processes 267K runners from PF historical form data to compute how each
factor's predictive power varies across racing contexts (venue type, distance,
class, condition). Outputs multipliers to punty/data/context_profiles.json.

Usage:
    python scripts/build_context_profiles.py
    python scripts/build_context_profiles.py --min-sample 300
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_DATA_DIR = Path(r"D:\Punty\DatafromProform\2026")
MIN_SAMPLE = 200  # minimum runners per context for a valid profile

MONTH_DIRS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

# ── Venue classification ─────────────────────────────────────────────────────

METRO_VIC = {"flemington", "caulfield", "moonee valley", "sandown", "the valley"}
METRO_NSW = {"randwick", "rosehill", "royal randwick", "canterbury", "warwick farm"}
METRO_QLD = {"eagle farm", "doomben"}
METRO_SA = {"morphettville"}
METRO_WA = {"ascot", "belmont"}

def _venue_type(venue: str, state: str) -> str:
    v = venue.lower().strip()
    if v in METRO_VIC or v in METRO_NSW or v in METRO_QLD or v in METRO_SA or v in METRO_WA:
        s = state.upper().strip() if state else ""
        if v in METRO_VIC:
            return "metro_vic"
        elif v in METRO_NSW:
            return "metro_nsw"
        elif v in METRO_QLD:
            return "metro_qld"
        else:
            return "metro_other"
    # Provincial vs country heuristic
    s = (state or "").upper().strip()
    if s in ("VIC", "NSW", "QLD"):
        return "provincial"
    return "country"

def _dist_bucket(distance: int) -> str:
    if distance <= 1100:
        return "sprint"
    elif distance <= 1399:
        return "short"
    elif distance <= 1799:
        return "middle"
    elif distance <= 2199:
        return "classic"
    else:
        return "staying"

def _class_bucket(race_class: str) -> str:
    rc = race_class.lower().strip().rstrip(";")
    if "maiden" in rc or "mdn" in rc:
        return "maiden"
    if "class 1" in rc or "restricted" in rc or "cl1" in rc:
        return "restricted"
    # BM rating parsing
    bm = re.search(r"bm\s*(\d+)", rc)
    if bm:
        rating = int(bm.group(1))
        if rating <= 72:
            return "mid_bm"
        return "open"
    if any(kw in rc for kw in ("group", "listed", "stakes", "open", "quality")):
        return "open"
    if "class 2" in rc or "class 3" in rc or "cl2" in rc or "cl3" in rc:
        return "mid_bm"
    return "mid_bm"  # default fallback

def _cond_bucket(condition: str) -> str:
    c = condition.lower().strip()
    if "heavy" in c or "hvy" in c:
        return "heavy"
    if "soft" in c or "sft" in c:
        return "soft"
    return "good"

# ── Signal derivation ─────────────────────────────────────────────────────────

def _derive_signals(pf_runner: dict, race_meta: dict) -> dict:
    """Derive per-factor proxy signals from raw PF data.

    Returns dict of signal_name -> float value (or None if unavailable).
    Each signal corresponds to a probability engine factor.
    """
    forms = pf_runner.get("Forms", [])
    real_forms = [f for f in forms if not f.get("IsBarrierTrial")]
    signals = {}

    # market: use PriceSP as market consensus proxy
    sp = pf_runner.get("PriceSP", 0)
    signals["market"] = 1.0 / sp if sp and sp > 1.0 else None

    # form: career win rate
    career_starts = pf_runner.get("CareerStarts", 0)
    career_wins = pf_runner.get("CareerWins", 0)
    if career_starts and career_starts >= 5:
        signals["form"] = career_wins / career_starts
    else:
        signals["form"] = None

    # class_fitness: prize money per start
    prize = pf_runner.get("PrizeMoney", 0)
    if prize and career_starts and career_starts >= 5:
        signals["class_fitness"] = prize / career_starts
    else:
        signals["class_fitness"] = None

    # pace: settling position from InRun
    settle_vals = []
    for f in real_forms[:5]:
        in_run = f.get("InRun", "") or ""
        for part in in_run.strip(";").split(";"):
            if "," in part:
                k, v = part.split(",", 1)
                if k.strip() == "settling_down":
                    try:
                        settle_vals.append(int(v.strip()))
                    except (ValueError, TypeError):
                        pass
                    break
    if settle_vals:
        field_size = race_meta.get("field_size", 12) or 12
        avg_settle = sum(settle_vals) / len(settle_vals)
        signals["pace"] = 1.0 - (avg_settle / max(field_size, 1))  # 1.0=leader, 0=last
    else:
        signals["pace"] = None

    # barrier: normalized barrier position
    barrier = pf_runner.get("Barrier", 0)
    field_size = race_meta.get("field_size", 12) or 12
    if barrier and barrier > 0:
        signals["barrier"] = 1.0 - (barrier / max(field_size, 1))  # 1.0=inside, 0=outside
    else:
        signals["barrier"] = None

    # jockey_trainer: jockey career strike rate
    jockey_a2e = pf_runner.get("JockeyA2E_Career", {})
    if jockey_a2e and jockey_a2e.get("Runners", 0) >= 50:
        signals["jockey_trainer"] = (jockey_a2e.get("StrikeRate", 0) or 0) / 100.0
    else:
        signals["jockey_trainer"] = None

    # weight_carried: raw weight (higher = heavier = handicapper rates higher)
    weight = pf_runner.get("Weight", 0) or pf_runner.get("WeightTotal", 0)
    if weight and weight > 40:
        signals["weight_carried"] = weight
    else:
        signals["weight_carried"] = None

    # horse_profile: age (peak 4-5)
    age = pf_runner.get("Age", 0)
    if age and age >= 2:
        signals["horse_profile"] = age
    else:
        signals["horse_profile"] = None

    # movement: price movement (firming/drifting from historical flucs)
    if real_forms:
        flucs_str = real_forms[0].get("Flucs", "") or ""
        prices = {}
        for seg in flucs_str.split(";"):
            seg = seg.strip()
            if "," in seg:
                parts = seg.split(",", 1)
                try:
                    prices[parts[0].strip()] = float(parts[1].strip())
                except (ValueError, IndexError):
                    pass
        opening = prices.get("opening")
        starting = prices.get("starting")
        if opening and starting and opening > 1 and starting > 1:
            signals["movement"] = (opening - starting) / opening  # positive = firmed
        else:
            signals["movement"] = None
    else:
        signals["movement"] = None

    return signals


# ── Quintile spread computation ───────────────────────────────────────────────

def _quintile_spread(values_with_wins: list[tuple[float, bool]]) -> float | None:
    """Compute Q5-Q1 strike rate spread for a signal.

    values_with_wins: list of (signal_value, did_win)
    Returns Q5_SR - Q1_SR (positive = higher signal predicts more wins).
    """
    if len(values_with_wins) < 50:
        return None

    sorted_vw = sorted(values_with_wins, key=lambda x: x[0])
    n = len(sorted_vw)
    q_size = n // 5

    if q_size < 10:
        return None

    q1 = sorted_vw[:q_size]
    q5 = sorted_vw[-q_size:]

    q1_sr = sum(1 for _, w in q1 if w) / len(q1)
    q5_sr = sum(1 for _, w in q5 if w) / len(q5)

    return q5_sr - q1_sr


# ── Main processing ──────────────────────────────────────────────────────────

def build_profiles(data_dir: Path, min_sample: int = MIN_SAMPLE) -> dict:
    """Build context profiles from all PF historical data."""

    print(f"Loading data from {data_dir}...")
    start = time.time()

    # Collect all runners with context + signals
    all_entries = []  # (context_key, signals_dict, won)

    total_runners = 0
    total_files = 0

    for month_num in range(1, 13):
        month_name = MONTH_DIRS[month_num]

        # Load race metadata
        meetings_path = data_dir / month_name / "meetings.json"
        race_meta = {}
        if meetings_path.exists():
            with open(meetings_path, "r", encoding="utf-8") as f:
                meetings = json.load(f)
            for m in meetings:
                md = m.get("MeetingDate", "")
                if md and not md.startswith("2025"):
                    continue
                venue = m.get("Track", {}).get("Name", "")
                state = m.get("Track", {}).get("State", "")
                for race in m.get("Races", []):
                    rid = race.get("RaceId")
                    try:
                        rid = int(rid)
                    except (ValueError, TypeError):
                        continue
                    raw_class = (race.get("RaceClass", "") or "").rstrip(";").strip()
                    cond = race.get("TrackCondition", "") or ""
                    race_meta[rid] = {
                        "distance": race.get("Distance", 1400),
                        "race_class": raw_class,
                        "venue": venue,
                        "state": state,
                        "condition": cond,
                        "field_size": race.get("Starters", 12),
                    }

        # Load form files
        form_dir = data_dir / month_name / "Form"
        if not form_dir.exists() and month_num == 1:
            form_dir = data_dir / "Form"
        if not form_dir.exists():
            continue

        for fpath in sorted(form_dir.glob("*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    runners = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(runners, list):
                continue

            total_files += 1

            # Group by RaceId
            races = defaultdict(list)
            for r in runners:
                rid = r.get("RaceId")
                if rid:
                    races[int(rid)].append(r)

            for race_id, race_runners in races.items():
                meta = race_meta.get(race_id, {})
                if not meta:
                    continue

                venue = meta.get("venue", "")
                state = meta.get("state", "")
                distance = meta.get("distance", 1400)
                rc = meta.get("race_class", "")
                cond = meta.get("condition", "")

                vtype = _venue_type(venue, state)
                dbucket = _dist_bucket(distance)
                cbucket = _class_bucket(rc)
                condbucket = _cond_bucket(cond)

                # Build context keys at multiple levels
                ctx_full = f"{vtype}|{dbucket}|{cbucket}|{condbucket}"
                ctx_no_cond = f"{vtype}|{dbucket}|{cbucket}"
                ctx_dist_class = f"{dbucket}|{cbucket}"

                for r in race_runners:
                    pos = r.get("Position", 99)
                    if pos is None or pos == 0:
                        continue
                    won = pos == 1

                    sigs = _derive_signals(r, meta)
                    total_runners += 1

                    all_entries.append({
                        "ctx_full": ctx_full,
                        "ctx_no_cond": ctx_no_cond,
                        "ctx_dist_class": ctx_dist_class,
                        "signals": sigs,
                        "won": won,
                    })

        print(f"  {month_name}: {total_runners:,} runners loaded")

    elapsed = time.time() - start
    print(f"\nLoaded {total_runners:,} runners from {total_files} files in {elapsed:.1f}s")

    # ── Compute overall signal spreads ────────────────────────────────────────

    print("\nComputing overall signal spreads...")

    factor_names = ["market", "form", "class_fitness", "pace", "barrier",
                    "jockey_trainer", "weight_carried", "horse_profile", "movement"]

    overall_spreads = {}
    for factor in factor_names:
        vals = [(e["signals"][factor], e["won"])
                for e in all_entries
                if e["signals"].get(factor) is not None]
        spread = _quintile_spread(vals)
        overall_spreads[factor] = spread
        if spread is not None:
            print(f"  {factor:20s}: {spread*100:+6.1f}% spread ({len(vals):,} runners)")
        else:
            print(f"  {factor:20s}: insufficient data ({len(vals):,} runners)")

    # ── Compute per-context spreads ───────────────────────────────────────────

    print("\nComputing per-context spreads...")

    # Group entries by context keys
    def _group_by(entries, key_field):
        groups = defaultdict(list)
        for e in entries:
            groups[e[key_field]].append(e)
        return groups

    profiles = {}
    fallbacks = {}

    # Level 1: Full context (venue_type|distance|class|condition)
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_full").items():
        if len(ctx_entries) < min_sample:
            continue
        mults = {}
        for factor in factor_names:
            vals = [(e["signals"][factor], e["won"])
                    for e in ctx_entries
                    if e["signals"].get(factor) is not None]
            ctx_spread = _quintile_spread(vals)
            overall = overall_spreads.get(factor)
            if ctx_spread is not None and overall and abs(overall) > 0.005:
                mult = ctx_spread / overall
                mults[factor] = round(max(0.3, min(2.5, mult)), 3)
        if mults:
            profiles[ctx_key] = mults

    # Level 2: Without condition (venue_type|distance|class)
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_no_cond").items():
        if len(ctx_entries) < min_sample:
            continue
        mults = {}
        for factor in factor_names:
            vals = [(e["signals"][factor], e["won"])
                    for e in ctx_entries
                    if e["signals"].get(factor) is not None]
            ctx_spread = _quintile_spread(vals)
            overall = overall_spreads.get(factor)
            if ctx_spread is not None and overall and abs(overall) > 0.005:
                mult = ctx_spread / overall
                mults[factor] = round(max(0.3, min(2.5, mult)), 3)
        if mults:
            fallbacks[ctx_key] = mults

    # Level 3: Distance + class only
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_dist_class").items():
        if ctx_key in fallbacks:
            continue  # don't overwrite venue-specific
        if len(ctx_entries) < min_sample:
            continue
        mults = {}
        for factor in factor_names:
            vals = [(e["signals"][factor], e["won"])
                    for e in ctx_entries
                    if e["signals"].get(factor) is not None]
            ctx_spread = _quintile_spread(vals)
            overall = overall_spreads.get(factor)
            if ctx_spread is not None and overall and abs(overall) > 0.005:
                mult = ctx_spread / overall
                mults[factor] = round(max(0.3, min(2.5, mult)), 3)
        if mults:
            fallbacks[ctx_key] = mults

    result = {
        "profiles": profiles,
        "fallbacks": fallbacks,
        "metadata": {
            "built_from": total_runners,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_sample": min_sample,
            "overall_spreads": {k: round(v * 100, 2) if v else None
                                for k, v in overall_spreads.items()},
        },
    }

    print(f"\nGenerated {len(profiles)} full profiles, {len(fallbacks)} fallback profiles")

    # Print some interesting examples
    if profiles:
        print("\nSample profiles:")
        for ctx, mults in sorted(profiles.items())[:5]:
            max_factor = max(mults, key=mults.get) if mults else "?"
            min_factor = min(mults, key=mults.get) if mults else "?"
            print(f"  {ctx:45s}: strongest={max_factor}({mults.get(max_factor, 0):.2f}x) "
                  f"weakest={min_factor}({mults.get(min_factor, 0):.2f}x)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build context profiles from PF historical data")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--min-sample", type=int, default=MIN_SAMPLE)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "punty" / "data" / "context_profiles.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    result = build_profiles(data_dir, args.min_sample)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
