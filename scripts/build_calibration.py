"""Build calibration dataset from 190K+ historical PF runners.

Extracts 25+ raw signals per runner, records actual race results,
and outputs a dataset for calibrating the probability engine.

Usage:
    python scripts/build_calibration.py
    python scripts/build_calibration.py --data-dir D:\Punty\DatafromProform\2026
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_DATA_DIR = Path(r"D:\Punty\DatafromProform\2026")

MONTH_DIRS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

# ── Venue classification (mirror probability.py) ────────────────────────────

METRO_VIC = {"flemington", "caulfield", "moonee valley", "sandown", "the valley"}
METRO_NSW = {"randwick", "rosehill", "royal randwick", "canterbury", "warwick farm"}
METRO_QLD = {"eagle farm", "doomben"}
METRO_SA = {"morphettville"}
METRO_WA = {"ascot", "belmont"}


def _venue_type(venue: str, state: str) -> str:
    v = venue.lower().strip()
    if v in METRO_VIC: return "metro_vic"
    if v in METRO_NSW: return "metro_nsw"
    if v in METRO_QLD: return "metro_qld"
    if v in METRO_SA or v in METRO_WA: return "metro_other"
    s = (state or "").upper().strip()
    if s in ("VIC", "NSW", "QLD"): return "provincial"
    return "country"


def _track_key(venue: str) -> str:
    v = venue.lower().strip()
    if v in ("the valley", "moonee valley"): return "moonee_valley"
    if v in ("royal randwick", "randwick"): return "randwick"
    return v.replace(" ", "_").replace("'", "")


def _dist_bucket(distance: int) -> str:
    if distance <= 1100: return "sprint"
    elif distance <= 1399: return "short"
    elif distance <= 1799: return "middle"
    elif distance <= 2199: return "classic"
    return "staying"


def _class_bucket(race_class: str) -> str:
    rc = race_class.lower().strip().rstrip(";")
    if "maiden" in rc or "mdn" in rc: return "maiden"
    if "class 1" in rc or "cl1" in rc: return "class1"
    if "restricted" in rc or "rst " in rc: return "restricted"
    bm = re.search(r"(?:bm|benchmark)\s*(\d+)", rc)
    if bm:
        rating = int(bm.group(1))
        if rating <= 58: return "bm58"
        if rating <= 68: return "bm64"
        if rating <= 76: return "bm72"
        return "open"
    if any(kw in rc for kw in ("group", "listed", "stakes", "quality")): return "open"
    if "open" in rc: return "open"
    if "class 2" in rc or "cl2" in rc: return "class2"
    if "class 3" in rc or "cl3" in rc: return "class3"
    if "class 4" in rc or "class 5" in rc or "class 6" in rc: return "bm72"
    return "bm64"


def _cond_bucket(condition: str) -> str:
    c = (condition or "").lower().strip()
    if "heavy" in c or "h" in c.split()[-1:]: return "heavy"
    if "soft" in c or "s" in c.split()[-1:]: return "soft"
    return "good"


def _safe_div(a, b, default=None):
    if b and b > 0:
        return a / b
    return default


def _record_sr(record: dict) -> float | None:
    """Extract strike rate from a record dict."""
    starts = record.get("Starts", 0)
    if not starts or starts < 1:
        return None
    return record.get("Firsts", 0) / starts


def _record_place_rate(record: dict) -> float | None:
    starts = record.get("Starts", 0)
    if not starts or starts < 1:
        return None
    return (record.get("Firsts", 0) + record.get("Seconds", 0) + record.get("Thirds", 0)) / starts


def _parse_last10(last10: str) -> list[int | None]:
    """Parse Last10 string into list of positions. 'x'=unplaced, '0'=unknown."""
    if not last10:
        return []
    results = []
    for ch in last10.strip()[:10]:
        if ch == 'x':
            results.append(99)  # unplaced
        elif ch == '0':
            results.append(None)
        elif ch.isdigit():
            results.append(int(ch))
        # else skip
    return results


def _score_last5(positions: list[int | None]) -> float | None:
    """Score last 5 results (mirrors probability.py _score_last_five)."""
    recent = [p for p in positions[:5] if p is not None]
    if not recent:
        return None
    score = 0.5
    weights = [1.0, 0.9, 0.8, 0.7, 0.6]
    total_weight = 0
    adj_sum = 0
    for i, pos in enumerate(recent):
        w = weights[i] if i < len(weights) else 0.5
        if pos == 1:
            adj = 0.08
        elif pos == 2:
            adj = 0.04
        elif pos == 3:
            adj = 0.02
        elif pos <= 6:
            adj = -0.01
        elif pos == 99:
            adj = -0.02
        else:
            adj = -0.03
        adj_sum += adj * w
        total_weight += w
    if total_weight > 0:
        score += adj_sum / total_weight * 5
    return max(0.05, min(0.95, score))


def _parse_flucs(flucs_str: str) -> tuple[float | None, float | None]:
    """Parse opening and starting prices from Flucs string."""
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
    """Parse settling position from InRun string, normalize by field size."""
    if not in_run:
        return None
    for part in in_run.strip(";").split(";"):
        if "," in part:
            k, v = part.split(",", 1)
            if k.strip() == "settling_down":
                try:
                    pos = int(v.strip())
                    if field_size and field_size > 1:
                        return pos / field_size
                    return pos / 12
                except (ValueError, TypeError):
                    pass
    return None


def _avg_margin(forms: list[dict], n: int = 5) -> float | None:
    """Average margin from last N forms."""
    margins = []
    for f in forms[:n]:
        if f.get("IsBarrierTrial"):
            continue
        m = f.get("Margin")
        if m is not None and isinstance(m, (int, float)):
            margins.append(abs(m))
    return statistics.mean(margins) if margins else None


def _a2e_field(a2e_dict: dict, field: str, min_runners: int = 20) -> float | None:
    """Safely extract a field from A2E dict, require minimum runners."""
    if not a2e_dict or not isinstance(a2e_dict, dict):
        return None
    runners = a2e_dict.get("Runners", 0)
    if not runners or runners < min_runners:
        return None
    val = a2e_dict.get(field)
    if val is None:
        return None
    return float(val)


def extract_signals(pf_runner: dict, race_meta: dict) -> dict:
    """Extract all calibration signals from a PF runner.

    Returns dict of signal_name → float value (or None if missing).
    """
    forms = pf_runner.get("Forms", [])
    real_forms = [f for f in forms if not f.get("IsBarrierTrial")]
    field_size = race_meta.get("field_size", 12) or 12
    race_class = race_meta.get("race_class", "")
    distance = race_meta.get("distance", 1400)

    signals = {}

    # ── Market ───────────────────────────────────────────────────────────
    sp = pf_runner.get("PriceSP", 0)
    signals["market_prob"] = 1.0 / sp if sp and sp > 1.0 else None

    # ── Form ─────────────────────────────────────────────────────────────
    # Career win/place rate
    win_pct = pf_runner.get("WinPct", 0)
    place_pct = pf_runner.get("PlacePct", 0)
    career_starts = pf_runner.get("CareerStarts", 0)

    # Derive starts from WinPct if CareerStarts is 0
    if not career_starts and win_pct and pf_runner.get("CareerWins", 0):
        career_starts = round(pf_runner["CareerWins"] / (win_pct / 100))

    # WinPct and PlacePct are always available from PF
    signals["career_win_pct"] = win_pct / 100 if win_pct else None
    signals["career_place_pct"] = place_pct / 100 if place_pct else None
    signals["career_starts"] = career_starts

    # Track+distance record
    td = pf_runner.get("TrackDistRecord", {})
    signals["track_dist_sr"] = _record_sr(td)
    signals["track_dist_starts"] = td.get("Starts", 0)

    # Distance record
    dr = pf_runner.get("DistanceRecord", {})
    signals["distance_sr"] = _record_sr(dr)
    signals["distance_starts"] = dr.get("Starts", 0)

    # Track record
    tr = pf_runner.get("TrackRecord", {})
    signals["track_sr"] = _record_sr(tr)
    signals["track_starts"] = tr.get("Starts", 0)

    # Condition records
    for cond_name in ("Good", "Soft", "Heavy", "Firm", "Synthetic"):
        rec = pf_runner.get(f"{cond_name}Record", {})
        signals[f"cond_{cond_name.lower()}_sr"] = _record_sr(rec)
        signals[f"cond_{cond_name.lower()}_starts"] = rec.get("Starts", 0)

    # First-up / second-up
    fu = pf_runner.get("FirstUpRecord", {})
    signals["first_up_sr"] = _record_sr(fu)
    signals["first_up_starts"] = fu.get("Starts", 0)
    su = pf_runner.get("SecondUpRecord", {})
    signals["second_up_sr"] = _record_sr(su)
    signals["second_up_starts"] = su.get("Starts", 0)

    # Last 5 form score
    last10 = pf_runner.get("Last10", "")
    positions = _parse_last10(last10)
    signals["last5_score"] = _score_last5(positions)
    # Also raw: count of wins in last 5
    recent5 = [p for p in positions[:5] if p is not None]
    signals["last5_wins"] = sum(1 for p in recent5 if p == 1) if recent5 else None
    signals["last5_places"] = sum(1 for p in recent5 if p and p <= 3) if recent5 else None

    # ── Class & Fitness ──────────────────────────────────────────────────
    prize = pf_runner.get("PrizeMoney", 0)
    if career_starts and career_starts > 0 and prize:
        signals["prize_per_start"] = prize / career_starts
    else:
        signals["prize_per_start"] = None

    signals["handicap_rating"] = pf_runner.get("HandicapRating") or None
    signals["avg_margin"] = _avg_margin(real_forms)

    # Days since last run
    if real_forms:
        try:
            last_date = real_forms[0].get("MeetingDate", "")
            race_date = race_meta.get("date", "")
            if last_date and race_date:
                ld = datetime.fromisoformat(last_date.replace("T00:00:00", ""))
                rd = datetime.strptime(race_date[:10], "%Y-%m-%d")
                signals["days_since_last"] = (rd - ld).days
            else:
                signals["days_since_last"] = None
        except (ValueError, TypeError):
            signals["days_since_last"] = None
    else:
        signals["days_since_last"] = None

    # ── Pace ─────────────────────────────────────────────────────────────
    # Settling position from most recent forms
    settle_vals = []
    for f in real_forms[:5]:
        s = _parse_settling(f.get("InRun", ""), field_size)
        if s is not None:
            settle_vals.append(s)
    signals["settle_pos"] = statistics.mean(settle_vals) if settle_vals else None

    # ── Barrier ──────────────────────────────────────────────────────────
    barrier = pf_runner.get("Barrier", 0)
    if barrier and barrier > 0 and field_size > 1:
        signals["barrier_relative"] = (barrier - 1) / (field_size - 1)
    else:
        signals["barrier_relative"] = None
    signals["barrier_raw"] = barrier

    # ── Jockey / Trainer ─────────────────────────────────────────────────
    jc = pf_runner.get("JockeyA2E_Career", {})
    signals["jockey_career_sr"] = _a2e_field(jc, "StrikeRate", 50)
    signals["jockey_career_a2e"] = _a2e_field(jc, "A2E", 50)
    signals["jockey_career_pot"] = _a2e_field(jc, "PoT", 50)
    signals["jockey_career_runners"] = jc.get("Runners", 0) if jc else 0

    jl = pf_runner.get("JockeyA2E_Last100", {})
    signals["jockey_l100_sr"] = _a2e_field(jl, "StrikeRate", 20)
    signals["jockey_l100_a2e"] = _a2e_field(jl, "A2E", 20)

    tc = pf_runner.get("TrainerA2E_Career", {})
    signals["trainer_career_sr"] = _a2e_field(tc, "StrikeRate", 50)
    signals["trainer_career_a2e"] = _a2e_field(tc, "A2E", 50)
    signals["trainer_career_pot"] = _a2e_field(tc, "PoT", 50)

    tl = pf_runner.get("TrainerA2E_Last100", {})
    signals["trainer_l100_sr"] = _a2e_field(tl, "StrikeRate", 20)

    # Combo stats
    cc = pf_runner.get("TrainerJockeyA2E_Career", {})
    signals["combo_career_sr"] = _a2e_field(cc, "StrikeRate", 10)
    signals["combo_career_runners"] = cc.get("Runners", 0) if cc else 0

    cl = pf_runner.get("TrainerJockeyA2E_Last100", {})
    signals["combo_l100_sr"] = _a2e_field(cl, "StrikeRate", 10)

    # ── Weight ───────────────────────────────────────────────────────────
    weight = pf_runner.get("Weight", 0)
    signals["weight"] = weight if weight and weight > 40 else None

    # ── Horse Profile ────────────────────────────────────────────────────
    signals["age"] = pf_runner.get("Age", 0) or None
    sex = pf_runner.get("Sex", "")
    signals["sex"] = sex
    signals["is_gelding"] = 1 if sex and "gelding" in sex.lower() else 0
    signals["is_mare"] = 1 if sex and ("mare" in sex.lower() or "filly" in sex.lower()) else 0
    signals["is_colt"] = 1 if sex and ("colt" in sex.lower() or "entire" in sex.lower() or "stallion" in sex.lower() or "rig" in sex.lower()) else 0

    # ── Movement ─────────────────────────────────────────────────────────
    if real_forms:
        opening, starting = _parse_flucs(real_forms[0].get("Flucs", ""))
        if opening and starting and opening > 1 and starting > 1:
            signals["price_move_pct"] = (opening - starting) / opening
        else:
            signals["price_move_pct"] = None
    else:
        signals["price_move_pct"] = None

    # ── Group/stakes record ──────────────────────────────────────────────
    g1 = pf_runner.get("Group1Record", {})
    g2 = pf_runner.get("Group2Record", {})
    g3 = pf_runner.get("Group3Record", {})
    grp_starts = sum(r.get("Starts", 0) for r in [g1, g2, g3])
    grp_wins = sum(r.get("Firsts", 0) for r in [g1, g2, g3])
    signals["group_starts"] = grp_starts
    signals["group_sr"] = grp_wins / grp_starts if grp_starts >= 3 else None

    return signals


def build_dataset(data_dir: Path) -> list[dict]:
    """Build calibration dataset from all PF historical data."""

    print(f"Loading data from {data_dir}...")
    start = time.time()

    all_records = []
    total_runners = 0
    total_files = 0
    skipped_no_pos = 0
    skipped_no_sp = 0

    for month_num in range(1, 13):
        month_name = MONTH_DIRS[month_num]

        # Load race metadata from meetings.json
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
                        "date": md[:10] if md else "",
                        "prize_money": race.get("PrizeMoney", 0),
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

            # Group by race to compute per-race fields (avg weight)
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
                field_size = len(race_runners)  # actual field size
                meta["field_size"] = field_size

                # Compute race-level averages
                weights = [r.get("Weight", 0) for r in race_runners if r.get("Weight", 0) > 40]
                avg_weight = statistics.mean(weights) if weights else 56.0

                track = _track_key(venue)
                vtype = _venue_type(venue, state)
                dbucket = _dist_bucket(distance)
                cbucket = _class_bucket(rc)
                condbucket = _cond_bucket(meta.get("condition", ""))

                for r in race_runners:
                    pos = r.get("Position")
                    if pos is None or pos == 0:
                        skipped_no_pos += 1
                        continue

                    sp = r.get("PriceSP", 0)
                    if not sp or sp <= 1.0:
                        skipped_no_sp += 1
                        continue

                    total_runners += 1
                    signals = extract_signals(r, meta)

                    # Add weight diff (needs race-level avg)
                    w = r.get("Weight", 0)
                    signals["weight_diff"] = w - avg_weight if w and w > 40 else None

                    record = {
                        # Result
                        "won": pos == 1,
                        "placed": pos <= 3,
                        "position": pos,
                        "sp": sp,
                        "margin": r.get("Margin"),
                        # Context
                        "track": track,
                        "venue_type": vtype,
                        "distance": distance,
                        "dist_bucket": dbucket,
                        "class_bucket": cbucket,
                        "condition": condbucket,
                        "field_size": field_size,
                        "race_prize": meta.get("prize_money", 0),
                        "race_date": meta.get("date", ""),
                        # Signals
                        **signals,
                    }
                    all_records.append(record)

        print(f"  {month_name}: {total_runners:,} total runners")

    elapsed = time.time() - start
    print(f"\nLoaded {total_runners:,} runners from {total_files} files in {elapsed:.1f}s")
    print(f"Skipped: {skipped_no_pos:,} no position, {skipped_no_sp:,} no SP")

    # Summary stats
    wins = sum(1 for r in all_records if r["won"])
    print(f"Win rate: {wins:,}/{total_runners:,} = {wins/total_runners*100:.1f}%")

    # Signal coverage
    signal_keys = [k for k in all_records[0].keys()
                   if k not in ("won", "placed", "position", "sp", "margin",
                                "track", "venue_type", "distance", "dist_bucket",
                                "class_bucket", "condition", "field_size",
                                "race_prize", "race_date", "sex",
                                "career_starts", "track_dist_starts",
                                "distance_starts", "track_starts",
                                "jockey_career_runners", "combo_career_runners",
                                "group_starts", "is_gelding", "is_mare", "is_colt",
                                "barrier_raw") +
                                tuple(f"cond_{c}_starts" for c in ("good", "soft", "heavy", "firm", "synthetic")) +
                                ("first_up_starts", "second_up_starts")]
    print("\nSignal coverage:")
    for key in sorted(signal_keys):
        non_null = sum(1 for r in all_records if r.get(key) is not None)
        pct = non_null / total_runners * 100
        print(f"  {key:25s}: {non_null:>7,} ({pct:.0f}%)")

    return all_records


def main():
    parser = argparse.ArgumentParser(description="Build calibration dataset from PF historical data")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parent.parent / "punty" / "data" / "calibration_dataset.json"
    ))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    records = build_dataset(data_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(records):,} records to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(records, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
