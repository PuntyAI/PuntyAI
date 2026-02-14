"""Historical backtester — runs probability model against PF form data.

Processes 12 months of PuntiForm historical data (~267K runners across ~3,459 meetings)
through our probability calculation engine and measures predictive accuracy.

Usage:
    python scripts/backtest_probability.py
    python scripts/backtest_probability.py --months 2,3,4  # Feb, Mar, Apr only
    python scripts/backtest_probability.py --output results.json
"""

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from punty.probability import calculate_race_probabilities, DEFAULT_WEIGHTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(r"D:\Punty\DatafromProform\2026")
DATA_DIR = DEFAULT_DATA_DIR  # can be overridden by --data-dir

MONTH_DIRS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ── Data loading ────────────────────────────────────────────────────────────


def _load_race_metadata(month_num: int) -> dict[int, dict]:
    """Load race-level metadata (distance, class, condition) from meetings.json.

    Returns dict mapping RaceId -> {distance, race_class, track_condition, venue, ...}.
    """
    month_name = MONTH_DIRS[month_num]
    meetings_path = DATA_DIR / month_name / "meetings.json"
    if not meetings_path.exists():
        return {}

    with open(meetings_path, "r", encoding="utf-8") as f:
        meetings = json.load(f)

    race_meta = {}
    for meeting in meetings:
        venue = meeting.get("Track", {}).get("Name", "")
        state = meeting.get("Track", {}).get("State", "")
        meeting_date = meeting.get("MeetingDate", "")
        rail = meeting.get("RailPosition", "")

        # Skip non-2025 data (January has 2009 meetings)
        if meeting_date and not meeting_date.startswith("2025"):
            continue

        for race in meeting.get("Races", []):
            rid = race.get("RaceId")
            if not rid:
                continue
            # Normalize to int — meetings.json may have string IDs
            try:
                rid = int(rid)
            except (ValueError, TypeError):
                continue
            # Strip trailing semicolons from RaceClass (PF quirk)
            raw_class = race.get("RaceClass", "") or ""
            race_meta[rid] = {
                "distance": race.get("Distance", 1400),
                "race_class": raw_class.rstrip(";").strip(),
                "race_name": race.get("Name", ""),
                "prize_money": race.get("PrizeMoney", 0),
                "venue": venue,
                "state": state,
                "meeting_date": meeting_date,
                "rail_position": rail,
                "race_number": race.get("Number", 0),
            }

    return race_meta


def _load_form_files(month_num: int) -> list[Path]:
    """Get all form file paths for a month."""
    month_name = MONTH_DIRS[month_num]

    # January: form files in top-level Form/ or January/Form/
    form_dir = DATA_DIR / month_name / "Form"
    if not form_dir.exists() and month_num == 1:
        form_dir = DATA_DIR / "Form"

    if not form_dir.exists():
        return []

    return sorted(form_dir.glob("*.json"))


def _build_a2e(career_data, last100_data, combo_career=None):
    """Build A2E JSON from PF career/last100 data."""
    result = {}
    if career_data and career_data.get("Runners"):
        result["career"] = {
            "a2e": career_data.get("A2E", 0),
            "pot": career_data.get("PoT", 0),
            "strike_rate": career_data.get("StrikeRate", 0),
            "wins": career_data.get("Wins", 0),
            "runners": career_data.get("Runners", 0),
        }
    if last100_data and last100_data.get("Runners"):
        result["last100"] = {
            "a2e": last100_data.get("A2E", 0),
            "pot": last100_data.get("PoT", 0),
            "strike_rate": last100_data.get("StrikeRate", 0),
            "wins": last100_data.get("Wins", 0),
            "runners": last100_data.get("Runners", 0),
        }
    if combo_career and combo_career.get("Runners"):
        result["combo_career"] = {
            "a2e": combo_career.get("A2E", 0),
            "pot": combo_career.get("PoT", 0),
            "strike_rate": combo_career.get("StrikeRate", 0),
            "wins": combo_career.get("Wins", 0),
            "runners": combo_career.get("Runners", 0),
        }
    return json.dumps(result) if result else None


def _record_to_stats(record):
    """Convert PF record dict to our JSON stats format."""
    if not record or not record.get("Starts", 0):
        return None
    return json.dumps({
        "starts": record.get("Starts", 0),
        "wins": record.get("Firsts", 0),
        "seconds": record.get("Seconds", 0),
        "thirds": record.get("Thirds", 0),
    })


def _derive_days_since_last_run(pf_runner: dict, race_date: str) -> int | None:
    """Compute days between most recent start and this race date."""
    forms = pf_runner.get("Forms", [])
    if not forms or not race_date:
        return None
    # Forms are ordered most recent first
    last_date_str = forms[0].get("MeetingDate", "")
    if not last_date_str:
        return None
    try:
        last_date = datetime.fromisoformat(last_date_str.replace("T00:00:00", ""))
        this_date = datetime.fromisoformat(race_date.split("T")[0])
        days = (this_date - last_date).days
        return days if days > 0 else None
    except (ValueError, TypeError):
        return None


def _derive_class_stats(pf_runner: dict, current_class: str) -> str | None:
    """Compute win/place record at same class from past starts."""
    if not current_class:
        return None
    forms = pf_runner.get("Forms", [])
    if not forms:
        return None

    # Normalize class for matching (e.g., "BM72", "Maiden", "Class 1")
    target = current_class.strip().lower()
    starts = wins = seconds = thirds = 0

    for form in forms:
        rc = (form.get("RaceClass") or "").strip().lower()
        if rc == target:
            starts += 1
            pos = form.get("Position", 99)
            if pos == 1:
                wins += 1
            elif pos == 2:
                seconds += 1
            elif pos == 3:
                thirds += 1

    if starts < 1:
        return None
    return json.dumps({"starts": starts, "wins": wins, "seconds": seconds, "thirds": thirds})


def _derive_settling_position(pf_runner: dict, field_size: int) -> str | None:
    """Derive typical settling position from InRun data in past starts.

    InRun format: "settling_down,6;m800,6;m400,4;finish,3;"
    We average the settling_down position across recent starts to get a pace proxy.
    """
    forms = pf_runner.get("Forms", [])
    if not forms:
        return None

    positions = []
    for form in forms[:10]:  # Last 10 starts
        inrun = form.get("InRun", "")
        if not inrun:
            continue
        # Parse settling_down position
        for segment in inrun.split(";"):
            if segment.startswith("settling_down,"):
                try:
                    pos = int(segment.split(",")[1])
                    positions.append(pos)
                except (ValueError, IndexError):
                    pass
                break

    if not positions:
        return None

    avg_pos = sum(positions) / len(positions)
    # Map average position to speed_map_position categories
    # Use field_size-relative thresholds
    if field_size <= 0:
        field_size = 12
    pct = avg_pos / field_size

    if pct <= 0.20:
        return "leader"
    elif pct <= 0.40:
        return "on_pace"
    elif pct <= 0.65:
        return "midfield"
    else:
        return "backmarker"


def _derive_track_condition(pf_runners: list[dict]) -> str:
    """Derive track condition from runner form data (current race).

    PF doesn't store track condition at the race level in meetings.json,
    but each past start's Forms[] has TrackCondition. Since all runners in
    the same race ran on the same track, we look for condition records
    that have the most data (Good is default if nothing found).
    """
    # Can't determine from this data - return empty (model handles gracefully)
    return ""


def _derive_last_start_weight(pf_runner: dict) -> float | None:
    """Get weight from most recent start."""
    forms = pf_runner.get("Forms", [])
    real = [f for f in forms if not f.get("IsBarrierTrial")]
    if real:
        return real[0].get("Weight")
    return None


def _derive_last_start_position(pf_runner: dict) -> int | None:
    """Get finishing position from most recent start."""
    forms = pf_runner.get("Forms", [])
    real = [f for f in forms if not f.get("IsBarrierTrial")]
    if real:
        return real[0].get("Position")
    return None


def _derive_last_start_margin(pf_runner: dict) -> float | None:
    """Get margin from most recent start."""
    forms = pf_runner.get("Forms", [])
    real = [f for f in forms if not f.get("IsBarrierTrial")]
    if real:
        return real[0].get("Margin")
    return None


def _derive_class_move(pf_runner: dict, current_class: str) -> str:
    """Determine if runner is upgrading/downgrading in class."""
    forms = pf_runner.get("Forms", [])
    real = [f for f in forms if not f.get("IsBarrierTrial")]
    if not real:
        return ""
    last_class = real[0].get("RaceClass", "")
    if not last_class or not current_class:
        return ""
    # Simple ordinal class ranking
    CLASS_ORDER = {
        "maiden": 1, "mdn": 1, "class 1": 2, "cl1": 2, "bm58": 2, "bm54": 2,
        "class 2": 3, "cl2": 3, "bm64": 3, "bm66": 3, "bm68": 3,
        "class 3": 4, "cl3": 4, "bm70": 4, "bm72": 4, "bm74": 4,
        "class 4": 5, "cl4": 5, "bm76": 5, "bm78": 5,
        "class 5": 6, "cl5": 6, "bm80": 6, "bm82": 6, "bm84": 6,
        "class 6": 7, "cl6": 7, "bm86": 7, "bm88": 7,
        "listed": 8, "open": 8, "group 3": 9, "grp3": 9,
        "group 2": 10, "grp2": 10, "group 1": 11, "grp1": 11,
    }
    curr_ord = CLASS_ORDER.get(current_class.lower().strip(), 5)
    last_ord = CLASS_ORDER.get(last_class.lower().strip(), 5)
    if curr_ord > last_ord:
        return "upgrade"
    elif curr_ord < last_ord:
        return "downgrade"
    return ""


def _derive_runs_since_spell(pf_runner: dict) -> int | None:
    """Count runs this preparation (since last spell > 28 days)."""
    forms = pf_runner.get("Forms", [])
    real = [f for f in forms if not f.get("IsBarrierTrial")]
    count = 0
    for i, f in enumerate(real):
        count += 1
        if i + 1 < len(real):
            curr_date = f.get("RaceDate", "")
            prev_date = real[i + 1].get("RaceDate", "")
            if curr_date and prev_date:
                try:
                    from datetime import datetime as _dt
                    d1 = _dt.fromisoformat(curr_date[:10])
                    d2 = _dt.fromisoformat(prev_date[:10])
                    if (d1 - d2).days > 60:
                        break
                except (ValueError, TypeError):
                    pass
    return count if count > 0 else None


def _adapt_runner(pf_runner: dict, race_date: str = "", race_class: str = "",
                  field_size: int = 12) -> dict:
    """Convert PF form runner to our probability engine format."""
    sp = pf_runner.get("PriceSP")

    # Last five from Last10 field
    last10 = pf_runner.get("Last10", "")
    last_five = last10[:5].strip() if last10 else None

    # Determine condition records
    good_stats = _record_to_stats(pf_runner.get("GoodRecord"))
    soft_stats = _record_to_stats(pf_runner.get("SoftRecord"))
    heavy_stats = _record_to_stats(pf_runner.get("HeavyRecord"))

    # Derive missing fields from Forms[]
    days_since = _derive_days_since_last_run(pf_runner, race_date)
    class_stats = _derive_class_stats(pf_runner, race_class)
    settling = _derive_settling_position(pf_runner, field_size)

    return {
        "id": str(pf_runner.get("RunnerId", "")),
        "horse_name": pf_runner.get("Name", ""),
        "saddlecloth": pf_runner.get("TabNo", 0),
        "barrier": pf_runner.get("Barrier", 0),
        "weight": pf_runner.get("Weight", 0),
        "current_odds": sp,
        "opening_odds": None,  # Not available for current race
        "last_five": last_five,
        "jockey": pf_runner.get("Jockey", {}).get("FullName", ""),
        "trainer": pf_runner.get("Trainer", {}).get("FullName", ""),
        "jockey_stats": _build_a2e(
            pf_runner.get("JockeyA2E_Career"),
            pf_runner.get("JockeyA2E_Last100"),
            pf_runner.get("TrainerJockeyA2E_Career"),
        ),
        "trainer_stats": _build_a2e(
            pf_runner.get("TrainerA2E_Career"),
            pf_runner.get("TrainerA2E_Last100"),
        ),
        "track_dist_stats": _record_to_stats(pf_runner.get("TrackDistRecord")),
        "distance_stats": _record_to_stats(pf_runner.get("DistanceRecord")),
        "class_stats": class_stats,
        "good_track_stats": good_stats,
        "soft_track_stats": soft_stats,
        "heavy_track_stats": heavy_stats,
        "first_up_stats": _record_to_stats(pf_runner.get("FirstUpRecord")),
        "second_up_stats": _record_to_stats(pf_runner.get("SecondUpRecord")),
        "handicap_rating": pf_runner.get("HandicapRating", 0) or 0,
        "horse_age": pf_runner.get("Age", 0),
        "horse_sex": pf_runner.get("Sex", ""),
        "scratched": False,
        # Derived from InRun history
        "speed_map_position": settling,
        "days_since_last_run": days_since,
        # Derived fields for DL pattern matching
        "last_start_weight": _derive_last_start_weight(pf_runner),
        "last_start_position": _derive_last_start_position(pf_runner),
        "last_start_margin": _derive_last_start_margin(pf_runner),
        "class_move": _derive_class_move(pf_runner, race_class),
        "runs_since_spell": _derive_runs_since_spell(pf_runner),
        # Not available in historical data
        "odds_flucs": None,
        "place_odds": None,
        "odds_tab": None,
        "odds_sportsbet": None,
        "odds_bet365": None,
        "odds_ladbrokes": None,
        "odds_betfair": None,
        # PF speed map fields (not in Form exports)
        "pf_speed_rank": None,
        "pf_map_factor": None,
        "pf_jockey_factor": None,
        "pf_settle": None,
    }


# ── Processing ──────────────────────────────────────────────────────────────


def _build_backtest_weights() -> dict[str, float]:
    """Build weight overrides for backtesting.

    Zeroes out factors that have no data in historical exports (movement, deep_learning)
    and redistributes their weight proportionally to remaining factors.
    """
    base_weights = dict(DEFAULT_WEIGHTS)
    dead_factors = {"movement"}  # No data in PF Form exports; DL now loads from JSON file

    dead_total = sum(base_weights.get(f, 0) for f in dead_factors)
    live_total = sum(v for k, v in base_weights.items() if k not in dead_factors)

    if live_total <= 0:
        return base_weights

    # Redistribute dead weight proportionally
    scale = (live_total + dead_total) / live_total
    new_weights = {}
    for k, v in base_weights.items():
        if k in dead_factors:
            new_weights[k] = 0.0
        else:
            new_weights[k] = v * scale

    return new_weights


def process_month(month_num: int, weight_overrides: dict | None = None) -> dict:
    """Process all form files for a month, returning aggregated stats."""
    race_meta = _load_race_metadata(month_num)
    form_files = _load_form_files(month_num)

    if not form_files:
        logger.warning("No form files found for month %d", month_num)
        return {"races": 0, "runners": 0, "results": []}

    month_results = []
    total_races = 0
    total_runners = 0
    skipped_no_sp = 0
    skipped_small_field = 0

    for file_path in form_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                runners_raw = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load %s: %s", file_path.name, e)
            continue

        if not isinstance(runners_raw, list) or not runners_raw:
            continue

        # Extract venue from filename: 250101_Flemington.json
        venue = file_path.stem.split("_", 1)[1].replace("_", " ") if "_" in file_path.stem else ""

        # Group by RaceId
        races: dict[int, list] = defaultdict(list)
        for r in runners_raw:
            rid = r.get("RaceId")
            if rid:
                races[rid].append(r)

        for race_id, race_runners in races.items():
            # Filter to runners with SP and position (settled races only)
            valid = [r for r in race_runners if r.get("PriceSP") and r.get("Position")]
            if len(valid) < 3:
                skipped_small_field += 1
                continue

            # Get race metadata
            meta = race_meta.get(race_id, {})
            distance = meta.get("distance", 1400)
            race_class = meta.get("race_class", "")
            race_date = meta.get("meeting_date", "")
            field_size = len(valid)

            # Build race and meeting dicts for probability engine
            race_dict = {
                "id": str(race_id),
                "distance": distance,
                "race_number": meta.get("race_number", 0),
                "class_": race_class,
            }
            meeting_dict = {
                "venue": meta.get("venue") or venue,
                "track_condition": "",  # Not available per-race in meetings.json
            }

            # Adapt runners with derived data
            adapted = [_adapt_runner(r, race_date, race_class, field_size) for r in valid]
            total_runners += len(adapted)

            # Run probability engine
            try:
                probs = calculate_race_probabilities(
                    adapted, race_dict, meeting_dict,
                    weights=weight_overrides,
                )
            except Exception as e:
                logger.debug("Probability calculation failed for race %s: %s", race_id, e)
                continue

            if not probs:
                continue

            total_races += 1

            # Analyze results — map runner to finish position
            for r in valid:
                rid = str(r.get("RunnerId", ""))
                pos = r.get("Position", 99)
                sp = r.get("PriceSP", 0)
                margin = r.get("Margin", 0)

                prob = probs.get(rid)
                if not prob:
                    continue

                won = pos == 1
                placed = pos <= 3

                month_results.append({
                    "race_id": race_id,
                    "venue": meeting_dict["venue"],
                    "distance": distance,
                    "runner_id": rid,
                    "horse": r.get("Name", ""),
                    "sp": sp,
                    "position": pos,
                    "margin": margin,
                    "win_prob": prob.win_probability,
                    "place_prob": prob.place_probability,
                    "value_rating": prob.value_rating,
                    "won": won,
                    "placed": placed,
                    "factors": prob.factors,
                    # Store raw PF runner for experimental signal analysis
                    "_pf": r,
                    "_race_date": race_date,
                    "_race_class": race_class,
                    "_field_size": field_size,
                })
                # Derive experimental signals now and drop raw PF data to save memory
                result_entry = month_results[-1]
                result_entry["_exp_signals"] = _derive_experimental_signals(result_entry)
                del result_entry["_pf"]

    logger.info(
        "Month %d: %d races, %d runners processed (skipped %d small field, %d no SP)",
        month_num, total_races, total_runners, skipped_small_field, skipped_no_sp,
    )

    return {
        "races": total_races,
        "runners": total_runners,
        "results": month_results,
    }


# ── Experimental Signal Derivation ──────────────────────────────────────


def _parse_race_time(time_str: str) -> float | None:
    """Parse OfficialRaceTime '00:01:02.3400000' to seconds."""
    if not time_str or time_str.startswith("00:00:00"):
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            total = h * 3600 + m * 60 + s
            return total if total > 10 else None
    except (ValueError, IndexError):
        pass
    return None


def _condition_speed_adj(cond_num: int) -> float:
    """Adjustment factor for track condition (slower tracks get bonus).

    Scale: 1=firm (fast) ... 10=heavy (slow). Neutral at Good 3-4.
    """
    if not cond_num or cond_num <= 0:
        return 1.0
    return 1.0 + max(0, (cond_num - 4)) * 0.015


def _derive_experimental_signals(result: dict) -> dict[str, float | None]:
    """Derive all experimental signals from raw PF runner data."""
    pf = result.get("_pf", {})
    race_date = result.get("_race_date", "")
    race_class = result.get("_race_class", "")
    race_distance = result.get("distance", 1400)
    field_size = result.get("_field_size", 12)
    forms = pf.get("Forms", [])
    signals = {}

    real_forms = [f for f in forms if not f.get("IsBarrierTrial")]

    # 1. KRI Average (recent 5 starts)
    kri_vals = []
    for f in real_forms[:5]:
        k = f.get("KRI", 0)
        if k and k > 0:
            kri_vals.append(k)
    signals["kri_avg"] = statistics.mean(kri_vals) if len(kri_vals) >= 2 else None

    # 2. KRI Trend (recent 2 vs older 2)
    if len(kri_vals) >= 4:
        recent = statistics.mean(kri_vals[:2])
        older = statistics.mean(kri_vals[2:4])
        signals["kri_trend"] = recent - older
    else:
        signals["kri_trend"] = None

    # 3. Speed Figure (condition-adjusted m/s)
    speed_figs = []
    for f in real_forms[:10]:
        dist = f.get("Distance", 0)
        if not dist or dist <= 0:
            continue
        time_secs = _parse_race_time(f.get("OfficialRaceTime", ""))
        if not time_secs:
            continue
        cond_num = f.get("TrackConditionNumber", 0) or 0
        raw_speed = dist / time_secs
        adj_speed = raw_speed * _condition_speed_adj(cond_num)
        speed_figs.append({"speed": adj_speed, "dist": dist})

    similar = [s["speed"] for s in speed_figs if abs(s["dist"] - race_distance) <= 200]
    signals["speed_best"] = max(similar) if similar else None
    signals["speed_avg"] = statistics.mean(similar) if len(similar) >= 2 else None
    all_speeds = [s["speed"] for s in speed_figs]
    signals["speed_all_avg"] = statistics.mean(all_speeds) if len(all_speeds) >= 2 else None

    # 4. Margin Analysis
    margins = []
    for f in real_forms[:5]:
        pos = f.get("Position", 99)
        m = f.get("Margin", 0)
        if pos and pos <= 20 and m is not None:
            margins.append(0.0 if pos == 1 else -m)
    signals["margin_avg"] = statistics.mean(margins) if len(margins) >= 2 else None

    class_margins = []
    if race_class:
        target_class = race_class.strip().lower()
        for f in real_forms:
            rc = (f.get("RaceClass") or "").strip().lower()
            if rc == target_class:
                pos = f.get("Position", 99)
                m = f.get("Margin", 0)
                if pos <= 20 and m is not None:
                    class_margins.append(0.0 if pos == 1 else -m)
    signals["margin_at_class"] = statistics.mean(class_margins) if len(class_margins) >= 2 else None

    # 5. Career Stats
    career_starts = pf.get("CareerStarts", 0)
    career_wins = pf.get("CareerWins", 0)
    if career_starts and career_starts >= 3:
        signals["career_win_pct"] = career_wins / career_starts * 100
        signals["career_place_pct"] = pf.get("PlacePct", 0) or 0
    else:
        signals["career_win_pct"] = None
        signals["career_place_pct"] = None

    # 6. Prize Money
    prize = pf.get("PrizeMoney", 0)
    signals["prize_money"] = prize if prize and prize > 0 else None
    if prize and career_starts and career_starts >= 3:
        signals["prize_per_start"] = prize / career_starts
    else:
        signals["prize_per_start"] = None

    # 7. Historical Market Support (firming pattern)
    firm_count = 0
    drift_count = 0
    total_moves = 0
    for f in real_forms[:10]:
        flucs_str = f.get("Flucs", "")
        if not flucs_str:
            continue
        prices = {}
        for seg in flucs_str.split(";"):
            seg = seg.strip()
            if "," in seg:
                parts = seg.split(",", 1)
                try:
                    prices[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        opening = prices.get("opening", 0)
        starting = prices.get("starting", 0)
        if opening > 0 and starting > 0:
            total_moves += 1
            pct_change = (starting - opening) / opening
            if pct_change <= -0.10:
                firm_count += 1
            elif pct_change >= 0.15:
                drift_count += 1
    if total_moves >= 3:
        signals["market_support_pct"] = firm_count / total_moves * 100
        signals["market_drift_pct"] = drift_count / total_moves * 100
    else:
        signals["market_support_pct"] = None
        signals["market_drift_pct"] = None

    # 8. Gear Changes
    has_gear_change = False
    for f in real_forms[:1]:
        gc = f.get("GearChanges", "")
        if gc and ("blinker" in gc.lower() or "tongue" in gc.lower() or "visor" in gc.lower()):
            has_gear_change = True
    signals["recent_gear_change"] = 1.0 if has_gear_change else 0.0

    first_time_gear = False
    if has_gear_change and len(real_forms) >= 2:
        prev_gc = real_forms[1].get("GearChanges", "") or ""
        if not prev_gc:
            first_time_gear = True
    signals["first_time_gear"] = 1.0 if first_time_gear else 0.0

    # 9. Track Record
    track_rec = pf.get("TrackRecord", {})
    if track_rec and track_rec.get("Starts", 0) >= 2:
        signals["track_win_pct"] = track_rec.get("Firsts", 0) / track_rec["Starts"] * 100
    else:
        signals["track_win_pct"] = None

    # 10. Trainer+Jockey Combo Last 100
    combo_l100 = pf.get("TrainerJockeyA2E_Last100", {})
    if combo_l100 and combo_l100.get("Runners", 0) >= 10:
        signals["combo_l100_sr"] = combo_l100.get("StrikeRate", 0)
        signals["combo_l100_a2e"] = combo_l100.get("A2E", 0)
    else:
        signals["combo_l100_sr"] = None
        signals["combo_l100_a2e"] = None

    # 11. Apprentice Claim
    claim = pf.get("JockeyClaim", 0)
    signals["jockey_claim"] = claim if claim and claim > 0 else None
    signals["is_apprentice"] = 1.0 if pf.get("Jockey", {}).get("IsApprentice", False) else 0.0

    # 12. Raw Weight
    signals["raw_weight"] = pf.get("Weight", 0) or None

    # 13. Prep Runs (starts within last 90 days)
    if real_forms and race_date:
        try:
            rd = datetime.fromisoformat(race_date.split("T")[0])
            prep_count = 0
            for f in real_forms:
                fd = f.get("MeetingDate", "")
                if fd:
                    form_date = datetime.fromisoformat(fd.replace("T00:00:00", ""))
                    if (rd - form_date).days <= 90:
                        prep_count += 1
                    else:
                        break
            signals["prep_runs"] = prep_count
        except (ValueError, TypeError):
            signals["prep_runs"] = None
    else:
        signals["prep_runs"] = None

    # 14. Days Since Last Run
    signals["days_since"] = _derive_days_since_last_run(pf, race_date)

    # 15. Betfair vs SP ratio (market efficiency)
    bf_ratios = []
    for f in real_forms[:5]:
        bf = f.get("PriceBF", 0)
        sp = f.get("PriceSP", 0)
        if bf and bf > 1 and sp and sp > 1:
            bf_ratios.append(bf / sp)
    signals["bf_sp_ratio"] = statistics.mean(bf_ratios) if len(bf_ratios) >= 2 else None

    # 16. Average Field Size Faced
    field_sizes = []
    for f in real_forms[:10]:
        st = f.get("Starters", 0)
        if st and st > 0:
            field_sizes.append(st)
    signals["avg_field_size"] = statistics.mean(field_sizes) if field_sizes else None

    # 17. Condition Performance (normalized)
    cond_scores = []
    for f in real_forms[:10]:
        cn = f.get("TrackConditionNumber", 0)
        pos = f.get("Position", 99)
        starters = f.get("Starters", 0) or 12
        if cn and cn > 0 and pos and pos <= 20:
            score = max(0, 1 - (pos - 1) / max(1, starters - 1))
            cond_scores.append(score)
    signals["avg_cond_score"] = statistics.mean(cond_scores) if len(cond_scores) >= 2 else None

    # 18. Weight Change from Last Start
    if real_forms and pf.get("Weight"):
        last_weight = real_forms[0].get("Weight", 0)
        current_weight = pf.get("Weight", 0)
        if last_weight and last_weight > 0 and current_weight and current_weight > 0:
            signals["weight_change"] = current_weight - last_weight
        else:
            signals["weight_change"] = None
    else:
        signals["weight_change"] = None

    # 19. Recent Place Strike Rate
    recent_places = 0
    recent_count = 0
    for f in real_forms[:10]:
        pos = f.get("Position", 99)
        if pos and pos <= 20:
            recent_count += 1
            if pos <= 3:
                recent_places += 1
    signals["recent_place_pct"] = (recent_places / recent_count * 100) if recent_count >= 3 else None

    # 20. Win Rate at Similar Distance
    dist_results = []
    for f in real_forms:
        fd = f.get("Distance", 0)
        if fd and abs(fd - race_distance) <= 200:
            pos = f.get("Position", 99)
            if pos <= 20:
                dist_results.append(1 if pos == 1 else 0)
    signals["dist_win_rate"] = (sum(dist_results) / len(dist_results) * 100) if len(dist_results) >= 3 else None

    return signals


def analyze_experimental_signals(all_results: list[dict]) -> dict:
    """Run quintile analysis on all experimental signals."""
    logger.info("Deriving experimental signals for %d runners...", len(all_results))

    signal_data: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    coverage: dict[str, int] = defaultdict(int)
    total = len(all_results)

    for i, result in enumerate(all_results):
        if i % 50000 == 0 and i > 0:
            logger.info("  Signal derivation: %d/%d", i, len(all_results))

        signals = result.get("_exp_signals") or _derive_experimental_signals(result)
        won = result["won"]

        for name, value in signals.items():
            if value is not None:
                signal_data[name].append((value, won))
                coverage[name] += 1

    analysis = {}
    for name, scored in sorted(signal_data.items()):
        if len(scored) < 500:
            continue

        scored.sort(key=lambda x: x[0])
        quintile_size = len(scored) // 5

        quintiles = {}
        for q in range(5):
            start = q * quintile_size
            end = start + quintile_size if q < 4 else len(scored)
            chunk = scored[start:end]
            q_wins = sum(1 for _, w in chunk if w)
            q_count = len(chunk)
            avg_val = statistics.mean(v for v, _ in chunk)
            quintiles[f"Q{q+1}"] = {
                "avg_value": round(avg_val, 4),
                "win_rate": round(q_wins / q_count * 100, 2) if q_count else 0,
                "count": q_count,
            }

        q1_wr = quintiles["Q1"]["win_rate"]
        q5_wr = quintiles["Q5"]["win_rate"]
        spread = q5_wr - q1_wr

        analysis[name] = {
            "spread_q5_q1": round(spread, 2),
            "predictive": abs(spread) > 2.0,
            "coverage_pct": round(coverage[name] / total * 100, 1),
            "q1_win_rate": q1_wr,
            "q5_win_rate": q5_wr,
            "direction": "higher=better" if spread > 0 else "lower=better",
            "quintiles": quintiles,
        }

    logger.info("Experimental signal analysis complete: %d signals evaluated", len(analysis))
    return analysis


# ── Analysis ────────────────────────────────────────────────────────────────


def analyze_results(all_results: list[dict]) -> dict:
    """Analyze backtested results across all months."""
    if not all_results:
        return {"error": "No results to analyze"}

    # ── Overall strike rates ──
    total = len(all_results)
    wins = sum(1 for r in all_results if r["won"])
    places = sum(1 for r in all_results if r["placed"])

    # ── Strike rate by probability bucket ──
    buckets = defaultdict(lambda: {"count": 0, "wins": 0, "places": 0, "total_prob": 0})
    for r in all_results:
        bucket = int(r["win_prob"] * 100 // 5) * 5  # 0, 5, 10, 15, ...
        bucket = min(bucket, 50)  # cap at 50+
        buckets[bucket]["count"] += 1
        buckets[bucket]["wins"] += 1 if r["won"] else 0
        buckets[bucket]["places"] += 1 if r["placed"] else 0
        buckets[bucket]["total_prob"] += r["win_prob"]

    calibration = {}
    for b in sorted(buckets.keys()):
        bd = buckets[b]
        if bd["count"] < 10:
            continue
        actual_sr = bd["wins"] / bd["count"]
        predicted = bd["total_prob"] / bd["count"]
        calibration[f"{b}-{b+5}%"] = {
            "count": bd["count"],
            "predicted_win_pct": round(predicted * 100, 2),
            "actual_win_pct": round(actual_sr * 100, 2),
            "calibration_error": round(abs(actual_sr - predicted) * 100, 2),
            "place_pct": round(bd["places"] / bd["count"] * 100, 2),
        }

    # ── Simulated P&L (top pick per race, $10 win bet) ──
    races: dict[int, list] = defaultdict(list)
    for r in all_results:
        races[r["race_id"]].append(r)

    sim_bets = 0
    sim_pnl = 0.0
    sim_wins = 0
    stake = 10.0
    value_bets = 0
    value_pnl = 0.0
    value_wins = 0

    for race_id, runners in races.items():
        # Sort by win_prob descending
        runners.sort(key=lambda x: x["win_prob"], reverse=True)
        top = runners[0]

        # Standard sim: always bet on top pick
        sim_bets += 1
        if top["won"]:
            sim_pnl += (top["sp"] * stake - stake)
            sim_wins += 1
        else:
            sim_pnl -= stake

        # Value sim: only bet when value_rating >= 1.0
        if top["value_rating"] >= 1.0:
            value_bets += 1
            if top["won"]:
                value_pnl += (top["sp"] * stake - stake)
                value_wins += 1
            else:
                value_pnl -= stake

    # ── Factor analysis: per-factor quintile vs actual win rate ──
    factor_analysis = {}
    sample = all_results[0] if all_results else {}
    factor_names = list(sample.get("factors", {}).keys())

    for fname in factor_names:
        scored = [(r["factors"].get(fname, 0.5), r["won"]) for r in all_results if r["factors"].get(fname) is not None]
        if len(scored) < 100:
            continue

        scored.sort(key=lambda x: x[0])
        quintile_size = len(scored) // 5

        quintiles = {}
        for q in range(5):
            start = q * quintile_size
            end = start + quintile_size if q < 4 else len(scored)
            chunk = scored[start:end]
            q_wins = sum(1 for _, w in chunk if w)
            q_count = len(chunk)
            avg_score = statistics.mean(s for s, _ in chunk)
            quintiles[f"Q{q+1}"] = {
                "avg_score": round(avg_score, 4),
                "win_rate": round(q_wins / q_count * 100, 2) if q_count else 0,
                "count": q_count,
            }

        # Is the factor predictive? Q5 win rate should be > Q1 win rate
        q1_wr = quintiles.get("Q1", {}).get("win_rate", 0)
        q5_wr = quintiles.get("Q5", {}).get("win_rate", 0)
        spread = q5_wr - q1_wr

        factor_analysis[fname] = {
            "quintiles": quintiles,
            "spread_q5_q1": round(spread, 2),
            "predictive": spread > 2.0,  # >2% spread = meaningful signal
            "neutral_pct": round(sum(1 for s, _ in scored if 0.49 <= s <= 0.51) / len(scored) * 100, 1),
        }

    # ── Experimental signal analysis ──
    exp_analysis = analyze_experimental_signals(all_results)

    # ── Brier score (calibration measure) ──
    brier_sum = sum((r["win_prob"] - (1.0 if r["won"] else 0.0)) ** 2 for r in all_results)
    brier_score = brier_sum / total

    # Market Brier for comparison
    market_brier_sum = 0
    market_count = 0
    for r in all_results:
        if r["sp"] and r["sp"] > 0:
            mkt_prob = 1.0 / r["sp"]
            market_brier_sum += (mkt_prob - (1.0 if r["won"] else 0.0)) ** 2
            market_count += 1
    market_brier = market_brier_sum / market_count if market_count else 0

    return {
        "summary": {
            "total_runners": total,
            "total_races": len(races),
            "total_wins": wins,
            "overall_win_rate": round(wins / total * 100, 2),
            "overall_place_rate": round(places / total * 100, 2),
            "brier_score": round(brier_score, 6),
            "market_brier_score": round(market_brier, 6),
            "brier_vs_market": "BETTER" if brier_score < market_brier else "WORSE",
        },
        "calibration": calibration,
        "simulation": {
            "always_top_pick": {
                "bets": sim_bets,
                "wins": sim_wins,
                "strike_rate": round(sim_wins / sim_bets * 100, 2) if sim_bets else 0,
                "total_pnl": round(sim_pnl, 2),
                "roi": round(sim_pnl / (sim_bets * stake) * 100, 2) if sim_bets else 0,
            },
            "value_bets_only": {
                "bets": value_bets,
                "wins": value_wins,
                "strike_rate": round(value_wins / value_bets * 100, 2) if value_bets else 0,
                "total_pnl": round(value_pnl, 2),
                "roi": round(value_pnl / (value_bets * stake) * 100, 2) if value_bets else 0,
            },
        },
        "factor_analysis": factor_analysis,
        "experimental_signals": exp_analysis,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest probability model against historical PF data")
    parser.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12",
                        help="Comma-separated month numbers (default: all)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--no-redistribute", action="store_true",
                        help="Don't redistribute dead factor weights (use default weights)")
    args = parser.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    months = [int(m.strip()) for m in args.months.split(",")]
    logger.info("Backtesting months: %s", months)

    # Build weight overrides: zero out factors with no historical data
    if args.no_redistribute:
        bt_weights = None
        logger.info("Using default weights (no redistribution)")
    else:
        bt_weights = _build_backtest_weights()
        dead = {k: v for k, v in bt_weights.items() if v == 0.0}
        live = {k: round(v * 100, 1) for k, v in bt_weights.items() if v > 0.0}
        logger.info("Dead factors (zeroed): %s", list(dead.keys()))
        logger.info("Redistributed weights: %s", live)

    all_results = []
    t0 = time.time()

    for month_num in months:
        logger.info("Processing month %d (%s)...", month_num, MONTH_DIRS[month_num])
        try:
            month_data = process_month(month_num, weight_overrides=bt_weights)
            all_results.extend(month_data["results"])
            del month_data  # free month data immediately
            gc.collect()
            logger.info("  -> %d results so far", len(all_results))
        except MemoryError:
            logger.warning("MemoryError on month %d — continuing with %d results", month_num, len(all_results))
            gc.collect()
            break

    elapsed = time.time() - t0
    logger.info("Processing complete: %d results in %.1fs", len(all_results), elapsed)

    # Analyze
    analysis = analyze_results(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    if "error" in analysis:
        print(f"\nError: {analysis['error']}")
        print("Check that the data directory exists and files are locally synced (not OneDrive cloud-only).")
        return

    s = analysis["summary"]
    print(f"\nRunners analyzed: {s['total_runners']:,}")
    print(f"Races: {s['total_races']:,}")
    print(f"Overall win rate: {s['overall_win_rate']}%")
    print(f"Overall place rate: {s['overall_place_rate']}%")
    print(f"\nBrier Score: {s['brier_score']:.6f} (lower = better)")
    print(f"Market Brier: {s['market_brier_score']:.6f}")
    print(f"Model vs Market: {s['brier_vs_market']}")

    print("\n-- Calibration by Probability Bucket --")
    print(f"{'Bucket':<12} {'Count':>7} {'Predicted':>10} {'Actual':>8} {'Error':>7} {'Place%':>7}")
    for bucket, data in analysis["calibration"].items():
        print(f"{bucket:<12} {data['count']:>7,} {data['predicted_win_pct']:>9.1f}% {data['actual_win_pct']:>7.1f}% {data['calibration_error']:>6.1f}% {data['place_pct']:>6.1f}%")

    print("\n-- Simulation: Always Bet Top Pick ($10) --")
    sim = analysis["simulation"]["always_top_pick"]
    print(f"Bets: {sim['bets']:,}  Wins: {sim['wins']:,}  SR: {sim['strike_rate']}%  P&L: ${sim['total_pnl']:,.2f}  ROI: {sim['roi']}%")

    sim_v = analysis["simulation"]["value_bets_only"]
    print(f"\n-- Simulation: Value Bets Only (value >= 1.0x) --")
    print(f"Bets: {sim_v['bets']:,}  Wins: {sim_v['wins']:,}  SR: {sim_v['strike_rate']}%  P&L: ${sim_v['total_pnl']:,.2f}  ROI: {sim_v['roi']}%")

    print("\n-- Factor Analysis (Q5-Q1 spread, higher = more predictive) --")
    print(f"{'Factor':<20} {'Spread':>8} {'Predictive':>12} {'Neutral%':>10}")
    for fname, fdata in sorted(analysis["factor_analysis"].items(), key=lambda x: x[1]["spread_q5_q1"], reverse=True):
        marker = "YES" if fdata["predictive"] else "NO"
        print(f"{fname:<20} {fdata['spread_q5_q1']:>7.1f}% {marker:>12} {fdata['neutral_pct']:>9.1f}%")

    # Experimental signals
    if "experimental_signals" in analysis and analysis["experimental_signals"]:
        exp = analysis["experimental_signals"]
        print("\n-- Experimental Signals (Q5-Q1 spread, sorted by predictive power) --")
        print(f"{'Signal':<25} {'Spread':>8} {'Coverage':>10} {'Q1 WR':>7} {'Q5 WR':>7} {'Direction':<16} {'Useful':>6}")
        for name, data in sorted(exp.items(), key=lambda x: abs(x[1]["spread_q5_q1"]), reverse=True):
            marker = "YES" if data["predictive"] else "no"
            print(f"{name:<25} {data['spread_q5_q1']:>7.1f}% {data['coverage_pct']:>9.1f}% {data['q1_win_rate']:>6.1f}% {data['q5_win_rate']:>6.1f}% {data['direction']:<16} {marker:>6}")

    # Save to file (strip _pf raw data from results before saving)
    if args.output:
        save_analysis = {k: v for k, v in analysis.items()}
        with open(args.output, "w") as f:
            json.dump(save_analysis, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
