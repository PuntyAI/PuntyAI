"""Build context profiles for the probability engine.

Processes 267K runners from PF historical form data to compute how each
factor's predictive power varies across racing contexts. Now supports
per-track profiles (e.g., "flemington|sprint|open") with venue-type fallback.

Hierarchy:
  1. track|distance|class    (per-track, e.g., "flemington|sprint|open")
  2. venue_type|distance|class (category, e.g., "metro_vic|sprint|open")
  3. track|distance or venue_type|distance (fallback)
  4. distance|class (fallback)

Usage:
    python scripts/build_context_profiles.py
    python scripts/build_context_profiles.py --min-sample 200
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

DEFAULT_DATA_DIRS = [
    Path(r"D:\Punty\DatafromProform\2025"),
    Path(r"D:\Punty\DatafromProform\2026"),
]
MIN_SAMPLE = 50   # 50 runners minimum per context for quintile analysis
MULT_MIN = 0.5    # raised from 0.3 to prevent near-elimination of factors
MULT_MAX = 2.5

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
    if v in METRO_VIC:
        return "metro_vic"
    elif v in METRO_NSW:
        return "metro_nsw"
    elif v in METRO_QLD:
        return "metro_qld"
    elif v in METRO_SA or v in METRO_WA:
        return "metro_other"
    # Provincial vs country heuristic
    s = (state or "").upper().strip()
    if s in ("VIC", "NSW", "QLD"):
        return "provincial"
    return "country"

def _track_key(venue: str) -> str:
    """Normalize venue name to a stable track key."""
    v = venue.lower().strip()
    # Normalize common aliases
    if v in ("the valley", "moonee valley"):
        return "moonee_valley"
    if v in ("royal randwick", "randwick"):
        return "randwick"
    return v.replace(" ", "_").replace("'", "")

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
    if "class 1" in rc or "cl1" in rc:
        return "class1"
    if "restricted" in rc or "rst " in rc:
        return "restricted"
    bm = re.search(r"(?:bm|benchmark)\s*(\d+)", rc)
    if bm:
        rating = int(bm.group(1))
        if rating <= 58:
            return "bm58"
        if rating <= 68:
            return "bm64"
        if rating <= 76:
            return "bm72"
        return "open"
    if any(kw in rc for kw in ("group", "listed", "stakes", "quality")):
        return "open"
    if "open" in rc:
        return "open"
    if "class 2" in rc or "cl2" in rc:
        return "class2"
    if "class 3" in rc or "cl3" in rc:
        return "class3"
    if "class 4" in rc or "class 5" in rc or "class 6" in rc:
        return "bm72"
    return "bm64"

# ── Signal derivation ─────────────────────────────────────────────────────────

def _derive_signals(pf_runner: dict, race_meta: dict) -> dict:
    """Derive per-factor proxy signals from raw PF data."""
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
        signals["pace"] = 1.0 - (avg_settle / max(field_size, 1))
    else:
        signals["pace"] = None

    # barrier: normalized barrier position
    barrier = pf_runner.get("Barrier", 0)
    field_size = race_meta.get("field_size", 12) or 12
    if barrier and barrier > 0:
        signals["barrier"] = 1.0 - (barrier / max(field_size, 1))
    else:
        signals["barrier"] = None

    # jockey_trainer: jockey career strike rate
    jockey_a2e = pf_runner.get("JockeyA2E_Career", {})
    if jockey_a2e and jockey_a2e.get("Runners", 0) >= 50:
        signals["jockey_trainer"] = (jockey_a2e.get("StrikeRate", 0) or 0) / 100.0
    else:
        signals["jockey_trainer"] = None

    # weight_carried: raw weight
    weight = pf_runner.get("Weight", 0) or pf_runner.get("WeightTotal", 0)
    if weight and weight > 40:
        signals["weight_carried"] = weight
    else:
        signals["weight_carried"] = None

    # horse_profile: age
    age = pf_runner.get("Age", 0)
    if age and age >= 2:
        signals["horse_profile"] = age
    else:
        signals["horse_profile"] = None

    # movement: price movement
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
            signals["movement"] = (opening - starting) / opening
        else:
            signals["movement"] = None
    else:
        signals["movement"] = None

    return signals


# ── Quintile spread computation ───────────────────────────────────────────────

def _quintile_spread(values_with_wins: list[tuple[float, bool]], min_sample: int = 100) -> float | None:
    """Compute Q5-Q1 strike rate spread for a signal."""
    if len(values_with_wins) < min_sample:
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

def build_profiles(data_dirs: list[Path], min_sample: int = MIN_SAMPLE) -> dict:
    """Build context profiles from all PF historical data across multiple years."""

    # Support single Path for backwards compat
    if isinstance(data_dirs, Path):
        data_dirs = [data_dirs]

    print(f"Loading data from {len(data_dirs)} directories:")
    for d in data_dirs:
        print(f"  {d}")
    print(f"Settings: min_sample={min_sample}, mult_range=[{MULT_MIN}, {MULT_MAX}]")
    start = time.time()

    all_entries = []
    total_runners = 0
    total_files = 0
    track_runner_counts = defaultdict(int)

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  SKIPPING {data_dir} (not found)")
            continue

        year_label = data_dir.name
        dir_runners_start = total_runners

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

                    track = _track_key(venue)
                    vtype = _venue_type(venue, state)
                    dbucket = _dist_bucket(distance)
                    cbucket = _class_bucket(rc)

                    # Build context keys at multiple levels
                    ctx_track = f"{track}|{dbucket}|{cbucket}"
                    ctx_vtype = f"{vtype}|{dbucket}|{cbucket}"
                    ctx_track_dist = f"{track}|{dbucket}"
                    ctx_vtype_dist = f"{vtype}|{dbucket}"
                    ctx_dist_class = f"{dbucket}|{cbucket}"

                    for r in race_runners:
                        pos = r.get("Position", 99)
                        if pos is None or pos == 0:
                            continue
                        won = pos == 1

                        sigs = _derive_signals(r, meta)
                        total_runners += 1
                        track_runner_counts[track] += 1

                        all_entries.append({
                            "ctx_track": ctx_track,
                            "ctx_vtype": ctx_vtype,
                            "ctx_track_dist": ctx_track_dist,
                            "ctx_vtype_dist": ctx_vtype_dist,
                            "ctx_dist_class": ctx_dist_class,
                            "signals": sigs,
                            "won": won,
                        })

            print(f"  {year_label}/{month_name}: {total_runners:,} runners loaded")

        dir_runners = total_runners - dir_runners_start
        print(f"  {year_label} subtotal: {dir_runners:,} runners\n")

    elapsed = time.time() - start
    print(f"\nLoaded {total_runners:,} runners from {total_files} files in {elapsed:.1f}s")

    # Show track distribution
    print(f"\nTop 20 tracks by runner count:")
    for track, count in sorted(track_runner_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {track:25s}: {count:,} runners")

    # ── Compute overall signal spreads ────────────────────────────────────────

    print("\nComputing overall signal spreads...")

    factor_names = ["market", "form", "class_fitness", "pace", "barrier",
                    "jockey_trainer", "weight_carried", "horse_profile", "movement"]

    overall_spreads = {}
    for factor in factor_names:
        vals = [(e["signals"][factor], e["won"])
                for e in all_entries
                if e["signals"].get(factor) is not None]
        spread = _quintile_spread(vals, min_sample=min_sample)
        overall_spreads[factor] = spread
        if spread is not None:
            print(f"  {factor:20s}: {spread*100:+6.1f}% spread ({len(vals):,} runners)")
        else:
            print(f"  {factor:20s}: insufficient data ({len(vals):,} runners)")

    # ── Compute per-context spreads ───────────────────────────────────────────

    print("\nComputing per-context spreads...")

    def _group_by(entries, key_field):
        groups = defaultdict(list)
        for e in entries:
            groups[e[key_field]].append(e)
        return groups

    profiles = {}
    fallbacks = {}

    def _compute_mults(ctx_entries, min_n=MIN_SAMPLE):
        if len(ctx_entries) < min_n:
            return {}
        mults = {}
        for factor in factor_names:
            vals = [(e["signals"][factor], e["won"])
                    for e in ctx_entries
                    if e["signals"].get(factor) is not None]
            ctx_spread = _quintile_spread(vals, min_sample=min_n)
            overall = overall_spreads.get(factor)
            if ctx_spread is not None and overall and abs(overall) > 0.005:
                mult = ctx_spread / overall
                mults[factor] = round(max(MULT_MIN, min(MULT_MAX, mult)), 3)
        return mults

    # Level 1: Per-track profiles (e.g., "flemington|sprint|open")
    track_profile_count = 0
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_track").items():
        mults = _compute_mults(ctx_entries)
        if mults:
            profiles[ctx_key] = mults
            profiles[ctx_key]["_n"] = len(ctx_entries)
            track_profile_count += 1

    # Level 2: Venue-type profiles (e.g., "metro_vic|sprint|open")
    vtype_profile_count = 0
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_vtype").items():
        if ctx_key not in profiles:  # don't overwrite per-track
            mults = _compute_mults(ctx_entries)
            if mults:
                profiles[ctx_key] = mults
                profiles[ctx_key]["_n"] = len(ctx_entries)
                vtype_profile_count += 1

    # Level 3: Track + distance fallback
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_track_dist").items():
        mults = _compute_mults(ctx_entries)
        if mults:
            fallbacks[ctx_key] = mults
            fallbacks[ctx_key]["_n"] = len(ctx_entries)

    # Level 4: Venue-type + distance fallback
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_vtype_dist").items():
        if ctx_key not in fallbacks:
            mults = _compute_mults(ctx_entries)
            if mults:
                fallbacks[ctx_key] = mults
                fallbacks[ctx_key]["_n"] = len(ctx_entries)

    # Level 5: Distance + class fallback
    for ctx_key, ctx_entries in _group_by(all_entries, "ctx_dist_class").items():
        if ctx_key not in fallbacks:
            mults = _compute_mults(ctx_entries)
            if mults:
                fallbacks[ctx_key] = mults
                fallbacks[ctx_key]["_n"] = len(ctx_entries)

    result = {
        "profiles": profiles,
        "fallbacks": fallbacks,
        "metadata": {
            "built_from": total_runners,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_sample": min_sample,
            "mult_range": [MULT_MIN, MULT_MAX],
            "track_profiles": track_profile_count,
            "vtype_profiles": vtype_profile_count,
            "overall_spreads": {k: round(v * 100, 2) if v else None
                                for k, v in overall_spreads.items()},
        },
    }

    print(f"\nGenerated profiles:")
    print(f"  Per-track: {track_profile_count}")
    print(f"  Venue-type: {vtype_profile_count}")
    print(f"  Total profiles: {len(profiles)}")
    print(f"  Fallbacks: {len(fallbacks)}")

    # Sample per-track profiles
    track_profiles = {k: v for k, v in profiles.items()
                      if k.count("|") == 2 and not k.startswith("metro_") and not k.startswith("provincial") and not k.startswith("country")}
    if track_profiles:
        print("\nSample per-track profiles:")
        for ctx in sorted(track_profiles.keys())[:10]:
            mults = track_profiles[ctx]
            n = mults.get("_n", 0)
            factor_strs = [f"{f}={mults.get(f, 1.0):.2f}" for f in factor_names if f in mults]
            print(f"  {ctx:45s} (n={n:,}): {', '.join(factor_strs[:5])}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build context profiles from PF historical data")
    parser.add_argument("--data-dir", nargs="*", default=None,
                        help="One or more data directories (default: 2025 + 2026)")
    parser.add_argument("--min-sample", type=int, default=MIN_SAMPLE)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "punty" / "data" / "context_profiles.json"))
    args = parser.parse_args()

    if args.data_dir:
        data_dirs = [Path(d) for d in args.data_dir]
    else:
        data_dirs = [d for d in DEFAULT_DATA_DIRS if d.exists()]

    if not data_dirs:
        print("ERROR: No data directories found")
        sys.exit(1)

    result = build_profiles(data_dirs, args.min_sample)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
