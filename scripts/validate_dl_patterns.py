"""DL Pattern Hold-Out Validation.

Tests 662 deep learning patterns (discovered from Jan-Sep 2025 data)
against an Oct-Dec 2025 hold-out set to determine signal vs noise.

Time-series split prevents data leakage — patterns were mined from
earlier data, validated on future races they never saw.

Usage:
    python scripts/validate_dl_patterns.py
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from scipy import stats as scipy_stats

# ── Paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PATTERNS_PATH = ROOT / "punty" / "data" / "dl_patterns.json"
BACKTEST_DB = ROOT / "data" / "backtest.db"
RESULTS_PATH = ROOT / "data" / "dl_validation_results.json"

# ── Hold-out split ──────────────────────────────────────────────────
HOLDOUT_START = "2025-10-01"  # Oct-Dec = 25% of 2025

# ── Decision thresholds ────────────────────────────────────────────
MIN_HOLDOUT_EDGE = 0.01        # 1% edge minimum
MIN_EDGE_RETENTION = 0.30      # 30% of discovery edge retained
MIN_HOLDOUT_SAMPLES = 20       # at least 20 matches
MAX_P_VALUE = 0.10             # 10% significance

# ── Skip types (non-discriminative, already disabled) ──────────────
SKIP_TYPES = {
    "deep_learning_condition_specialist",
    "deep_learning_market",
    "deep_learning_seasonal",
    "deep_learning_track_dist_cond",
    "deep_learning_standard_times",
}

# ── Import venue registry for state lookup ──────────────────────────
sys.path.insert(0, str(ROOT))
from punty.venues import guess_state  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# Helper functions (replicated from probability.py for standalone use)
# ══════════════════════════════════════════════════════════════════════

def _get_dist_bucket(distance: int) -> str:
    if distance <= 1100:
        return "sprint"
    elif distance <= 1399:
        return "short"
    elif distance <= 1799:
        return "middle"
    elif distance <= 2199:
        return "classic"
    return "staying"


def _normalize_condition(condition: str) -> str:
    if not condition:
        return ""
    c = condition.lower().strip()
    if "heavy" in c or "hvy" in c:
        return "Heavy"
    if "soft" in c or "sft" in c:
        return "Soft"
    if "synthetic" in c or "all weather" in c:
        return "Synthetic"
    if "firm" in c:
        return "Firm"
    if "good" in c or "gd" in c:
        return "Good"
    return "Good"


def _position_to_style(position: str) -> str:
    return {"leader": "leader", "on_pace": "on_pace",
            "midfield": "midfield", "backmarker": "backmarker"}.get(position or "", "")


def _get_barrier_bucket(barrier, field_size: int) -> str:
    if not barrier or field_size < 2:
        return ""
    b = int(barrier)
    third = max(1, field_size / 3)
    if b <= third:
        return "inside"
    elif b <= third * 2:
        return "middle"
    return "outside"


def _get_move_type(runner: dict) -> str:
    opening = runner.get("opening_odds")
    current = runner.get("current_odds")
    if not opening or not current or opening <= 0:
        return ""
    pct = ((current - opening) / opening) * 100
    if pct <= -20:
        return "big_mover"
    elif pct <= -10:
        return "improver"
    elif pct >= 20:
        return "fader"
    elif abs(pct) < 10:
        return "steady"
    return ""


def _get_market_direction(runner: dict) -> str:
    opening = runner.get("opening_odds")
    current = runner.get("current_odds")
    if not opening or not current or opening <= 0:
        return ""
    pct = ((current - opening) / opening) * 100
    if pct <= -5:
        return "backed"
    elif pct >= 5:
        return "drifting"
    return ""


def _get_weight_change_class(runner: dict) -> str:
    wt = runner.get("weight")
    if not wt:
        return ""
    last_wt = None
    fh_raw = runner.get("form_history")
    if fh_raw:
        try:
            fh = json.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
            if isinstance(fh, list) and fh:
                last_wt = fh[0].get("weight") if isinstance(fh[0], dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
    if not last_wt:
        return ""
    try:
        diff = float(wt) - float(last_wt)
    except (ValueError, TypeError):
        return ""
    if diff > 2.0:
        return "weight_up_big"
    elif diff > 0:
        return "weight_up_small"
    elif diff < -2.0:
        return "weight_down_big"
    elif diff < 0:
        return "weight_down_small"
    return "weight_same"


def _get_form_trend(runner: dict) -> str:
    last5 = runner.get("last_five") or ""
    if not last5:
        return ""
    positions = []
    for ch in str(last5).replace(" ", ""):
        if ch.isdigit():
            positions.append(int(ch))
        elif ch.lower() == "x":
            positions.append(9)
    if len(positions) < 3:
        return ""
    recent = sum(positions[:2]) / 2
    older = sum(positions[2:min(4, len(positions))]) / max(1, min(2, len(positions) - 2))
    if recent < older - 1:
        return "improving"
    elif recent > older + 1:
        return "declining"
    return "steady"


def _get_excuse_type(runner: dict) -> str:
    last_pos = runner.get("last_start_position")
    margin = runner.get("last_start_margin")
    if not last_pos:
        return ""
    pos = int(last_pos) if str(last_pos).isdigit() else 0
    if pos == 0:
        return ""
    if margin and float(margin) > 10:
        return "heavy_loss"
    if pos >= 8:
        return "bad_run"
    return ""


def _venue_matches(pattern_venue: str, meeting_venue: str) -> bool:
    if not pattern_venue or not meeting_venue:
        return False
    pv = pattern_venue.lower().strip()
    mv = meeting_venue.lower().strip()
    return pv in mv or mv in pv


def _condition_matches(pattern_cond: str, normalized_cond: str) -> bool:
    if not pattern_cond or not normalized_cond:
        return False
    pc = pattern_cond.lower().strip()
    nc = normalized_cond.lower().strip()
    if pc == nc:
        return True
    if pc.startswith("g") and nc == "good":
        return True
    if pc.startswith("s") and nc == "soft":
        return True
    if pc.startswith("h") and nc == "heavy":
        return True
    if pc == "synthetic" and nc == "synthetic":
        return True
    return False


def _odds_to_sp_range(odds: float) -> str:
    if odds < 3:
        return "$1-$3"
    elif odds < 5:
        return "$3-$5"
    elif odds < 8:
        return "$5-$8"
    elif odds < 12:
        return "$8-$12"
    elif odds < 20:
        return "$12-$20"
    return "$20+"


# ══════════════════════════════════════════════════════════════════════
# Pattern matching — mirrors _deep_learning_factor() but tracks
# individual pattern matches instead of aggregate score
# ══════════════════════════════════════════════════════════════════════

def match_patterns(runner: dict, venue: str, dist_bucket: str,
                   condition: str, field_size: int, state: str,
                   patterns: list[dict]) -> list[str]:
    """Return list of pattern keys that match this runner."""
    matched_keys = []

    pace_style = _position_to_style(runner.get("speed_map_position") or "")
    barrier = runner.get("barrier")
    barrier_bucket = _get_barrier_bucket(barrier, field_size) if barrier else ""
    jockey = runner.get("jockey") or ""
    trainer = runner.get("trainer") or ""
    move_type = _get_move_type(runner)

    for pattern in patterns:
        p_type = pattern.get("type", "")
        conds = pattern.get("conditions", {})
        edge = pattern.get("edge", 0.0)
        confidence = pattern.get("confidence", "LOW")
        key = pattern.get("key", "")

        if not edge or confidence == "LOW":
            continue
        if p_type in SKIP_TYPES:
            continue

        matched = False

        if p_type == "deep_learning_pace" and pace_style:
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
                and conds.get("style") == pace_style
            )

        elif p_type == "deep_learning_barrier_bias" and barrier_bucket:
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and conds.get("barrier_bucket") == barrier_bucket
            )

        elif p_type == "deep_learning_jockey_trainer" and jockey and trainer:
            matched = (
                conds.get("jockey", "").lower() == jockey.lower()
                and conds.get("trainer", "").lower() == trainer.lower()
                and (not state or conds.get("state") == state)
            )

        elif p_type == "deep_learning_acceleration" and move_type:
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
                and conds.get("move_type") == move_type
            )

        elif p_type == "deep_learning_pace_collapse" and pace_style == "leader":
            matched = (
                _venue_matches(conds.get("venue", ""), venue)
                and conds.get("dist_bucket") == dist_bucket
                and _condition_matches(conds.get("condition", ""), condition)
            )

        elif p_type == "deep_learning_class_mover" and state:
            direction = conds.get("direction", "")
            indicator = conds.get("indicator", "")
            runner_class_move = runner.get("class_move") or ""
            runner_mkt_move = _get_market_direction(runner)
            matched = (
                conds.get("state") == state
                and direction == runner_class_move
                and indicator == runner_mkt_move
            )

        elif p_type == "deep_learning_weight_impact":
            change_class = conds.get("change_class", "")
            runner_wt_change = _get_weight_change_class(runner)
            matched = (
                conds.get("dist_bucket") == dist_bucket
                and conds.get("state") == state
                and change_class == runner_wt_change
            )

        elif p_type == "deep_learning_form_trend" and state:
            trend = conds.get("trend", "")
            runner_trend = _get_form_trend(runner)
            matched = (
                conds.get("state") == state
                and trend == runner_trend
            )

        elif p_type == "deep_learning_bounceback":
            excuse_type = conds.get("excuse_type", "")
            runner_excuse = _get_excuse_type(runner)
            matched = (
                conds.get("dist_bucket") == dist_bucket
                and conds.get("state") == state
                and excuse_type == runner_excuse
            )

        elif p_type == "deep_learning_form_cycle":
            prep_runs = conds.get("prep_runs")
            runner_prep = runner.get("prep_runs") or runner.get("runs_since_spell")
            if prep_runs is not None and runner_prep is not None:
                matched = (int(runner_prep) == int(prep_runs))

        elif p_type == "deep_learning_class_transition":
            transition = conds.get("transition", "")
            runner_transition = runner.get("class_move") or ""
            trans_map = {"class_drop": "downgrade", "class_rise": "upgrade"}
            matched = (trans_map.get(transition, transition) == runner_transition)

        elif p_type == "deep_learning_track_bias":
            width = conds.get("width", "")
            pattern_venue = conds.get("venue", "")
            runner_barrier = barrier or 0
            if field_size > 0 and runner_barrier:
                ratio = runner_barrier / field_size
                runner_width = "wide" if ratio > 0.6 else "inside" if ratio <= 0.3 else "mid"
            else:
                runner_width = ""
            matched = (
                _venue_matches(pattern_venue, venue)
                and conds.get("dist_bucket") == dist_bucket
                and width == runner_width
            )

        elif p_type == "deep_learning_race_speed":
            # race_speed patterns don't match individual runners well
            # (they're about the whole race pace), skip for validation
            pass

        if matched:
            matched_keys.append(key)

    return matched_keys


# ══════════════════════════════════════════════════════════════════════
# Main validation
# ══════════════════════════════════════════════════════════════════════

def load_patterns() -> tuple[list[dict], dict[str, dict]]:
    """Load patterns and build lookup by key."""
    with open(PATTERNS_PATH) as f:
        patterns = json.load(f)

    by_key = {}
    for p in patterns:
        key = p.get("key", "")
        if key:
            by_key[key] = p
    return patterns, by_key


def load_holdout_data(conn: sqlite3.Connection) -> list[dict]:
    """Load races with runners from the hold-out period."""
    conn.row_factory = sqlite3.Row
    races = conn.execute("""
        SELECT r.id as race_id, r.distance, r.field_size,
               r.track_condition as race_condition,
               m.venue, m.track_condition as meet_condition
        FROM races r
        JOIN meetings m ON r.meeting_id = m.id
        WHERE m.date >= ?
        ORDER BY m.date, r.race_number
    """, (HOLDOUT_START,)).fetchall()

    result = []
    for race in races:
        race_dict = dict(race)
        runners = conn.execute("""
            SELECT * FROM runners WHERE race_id = ?
        """, (race_dict["race_id"],)).fetchall()
        race_dict["runners"] = [dict(r) for r in runners]
        result.append(race_dict)
    return result


def compute_baseline_win_rate(races: list[dict]) -> float:
    """Compute overall baseline win rate across hold-out races."""
    total = 0
    wins = 0
    for race in races:
        for runner in race["runners"]:
            total += 1
            if runner.get("finish_position") == 1:
                wins += 1
    return wins / total if total > 0 else 0.0


def validate():
    """Run the full validation."""
    print("=" * 70)
    print("DL PATTERN HOLD-OUT VALIDATION")
    print("=" * 70)

    # Load patterns
    patterns, by_key = load_patterns()
    active_patterns = [p for p in patterns
                       if p.get("type") not in SKIP_TYPES
                       and p.get("confidence") != "LOW"
                       and p.get("edge", 0)]
    print(f"\nPatterns loaded: {len(patterns)} total, {len(active_patterns)} active")

    # Load hold-out data
    conn = sqlite3.connect(str(BACKTEST_DB))
    races = load_holdout_data(conn)
    baseline = compute_baseline_win_rate(races)
    total_runners = sum(len(r["runners"]) for r in races)
    print(f"Hold-out: {len(races)} races, {total_runners} runners (from {HOLDOUT_START})")
    print(f"Baseline win rate: {baseline:.4f} ({baseline * 100:.1f}%)")

    # Track per-pattern results
    # pattern_key -> {"matches": int, "wins": int}
    pattern_results = defaultdict(lambda: {"matches": 0, "wins": 0})
    type_results = defaultdict(lambda: {"matches": 0, "wins": 0, "patterns_seen": set()})

    # Process each race
    print(f"\nProcessing {len(races)} races...")
    for i, race in enumerate(races):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(races)}...")

        venue = race["venue"] or ""
        distance = race["distance"] or 1200
        field_size = race["field_size"] or len(race["runners"])
        condition = _normalize_condition(race["race_condition"] or race["meet_condition"] or "")
        dist_bucket = _get_dist_bucket(distance)
        state = guess_state(venue) if venue else "VIC"

        for runner in race["runners"]:
            won = runner.get("finish_position") == 1
            matched_keys = match_patterns(
                runner, venue, dist_bucket, condition, field_size, state,
                active_patterns
            )

            for key in matched_keys:
                pattern_results[key]["matches"] += 1
                if won:
                    pattern_results[key]["wins"] += 1

                # Track by type
                p = by_key.get(key, {})
                p_type = p.get("type", "unknown")
                type_results[p_type]["matches"] += 1
                if won:
                    type_results[p_type]["wins"] += 1
                type_results[p_type]["patterns_seen"].add(key)

    # ── Per-Type Summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-TYPE RESULTS")
    print("=" * 70)

    type_verdicts = {}
    header = f"{'Type':<30} {'Matches':>8} {'WinRate':>8} {'Edge':>8} {'Retain':>8} {'p-val':>8} {'Verdict':>10}"
    print(header)
    print("-" * len(header))

    for p_type in sorted(type_results.keys()):
        data = type_results[p_type]
        matches = data["matches"]
        wins = data["wins"]
        holdout_wr = wins / matches if matches > 0 else 0
        holdout_edge = holdout_wr - baseline

        # Compute discovery edge (average across patterns of this type)
        discovery_edges = []
        for key in data["patterns_seen"]:
            p = by_key.get(key, {})
            discovery_edges.append(p.get("edge", 0))
        avg_discovery_edge = sum(discovery_edges) / len(discovery_edges) if discovery_edges else 0

        # Edge retention
        retention = holdout_edge / avg_discovery_edge if avg_discovery_edge > 0 else 0

        # Binomial test: is hold-out win rate significantly above baseline?
        if matches >= 5:
            p_value = scipy_stats.binomtest(wins, matches, baseline, alternative="greater").pvalue
        else:
            p_value = 1.0

        # Verdict
        short_type = p_type.replace("deep_learning_", "")
        if matches < MIN_HOLDOUT_SAMPLES:
            verdict = "INSUFFICIENT"
        elif holdout_edge > MIN_HOLDOUT_EDGE and retention > MIN_EDGE_RETENTION and p_value < MAX_P_VALUE:
            verdict = "VALID"
        elif holdout_edge <= 0:
            verdict = "NOISE"
        elif retention < 0.20:
            verdict = "NOISE"
        else:
            verdict = "WEAK"

        type_verdicts[p_type] = {
            "verdict": verdict,
            "matches": matches,
            "wins": wins,
            "holdout_win_rate": round(holdout_wr, 4),
            "holdout_edge": round(holdout_edge, 4),
            "discovery_edge": round(avg_discovery_edge, 4),
            "retention": round(retention, 4),
            "p_value": round(p_value, 4),
            "pattern_count": len(data["patterns_seen"]),
        }

        print(f"{short_type:<30} {matches:>8} {holdout_wr:>7.1%} {holdout_edge:>+7.1%} "
              f"{retention:>7.0%} {p_value:>8.4f} {verdict:>10}")

    # ── Per-Pattern Detail (for valid types) ────────────────────────
    print("\n" + "=" * 70)
    print("TOP INDIVIDUAL PATTERNS (by hold-out edge)")
    print("=" * 70)

    all_pattern_stats = []
    for key, data in pattern_results.items():
        p = by_key.get(key, {})
        matches = data["matches"]
        wins = data["wins"]
        holdout_wr = wins / matches if matches > 0 else 0
        holdout_edge = holdout_wr - baseline
        discovery_edge = p.get("edge", 0)
        retention = holdout_edge / discovery_edge if discovery_edge > 0 else 0

        all_pattern_stats.append({
            "key": key,
            "type": p.get("type", "").replace("deep_learning_", ""),
            "confidence": p.get("confidence", ""),
            "discovery_edge": round(discovery_edge, 4),
            "holdout_matches": matches,
            "holdout_wins": wins,
            "holdout_edge": round(holdout_edge, 4),
            "retention": round(retention, 4),
        })

    # Sort by hold-out edge descending
    all_pattern_stats.sort(key=lambda x: x["holdout_edge"], reverse=True)

    print(f"\n{'Key':<45} {'Matches':>7} {'H-Edge':>8} {'D-Edge':>8} {'Retain':>8}")
    print("-" * 80)
    for ps in all_pattern_stats[:20]:
        print(f"{ps['key'][:44]:<45} {ps['holdout_matches']:>7} {ps['holdout_edge']:>+7.1%} "
              f"{ps['discovery_edge']:>+7.1%} {ps['retention']:>7.0%}")

    print(f"\n... and {len(all_pattern_stats) - 20} more patterns")

    # ── Unmatched patterns ──────────────────────────────────────────
    unmatched = [p for p in active_patterns if p.get("key") not in pattern_results]
    print(f"\nUnmatched patterns (0 hold-out matches): {len(unmatched)}/{len(active_patterns)}")
    if unmatched:
        type_counts = defaultdict(int)
        for p in unmatched:
            type_counts[p.get("type", "").replace("deep_learning_", "")] += 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    # ── Overall Verdict ─────────────────────────────────────────────
    valid_types = [t for t, v in type_verdicts.items() if v["verdict"] == "VALID"]
    noise_types = [t for t, v in type_verdicts.items() if v["verdict"] == "NOISE"]
    weak_types = [t for t, v in type_verdicts.items() if v["verdict"] == "WEAK"]
    insuff_types = [t for t, v in type_verdicts.items() if v["verdict"] == "INSUFFICIENT"]

    tested = len(type_verdicts) - len(insuff_types)
    valid_pct = len(valid_types) / tested * 100 if tested > 0 else 0

    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)
    print(f"Valid types:        {len(valid_types)}/{tested} ({valid_pct:.0f}%)")
    print(f"Noise types:        {len(noise_types)}/{tested}")
    print(f"Weak types:         {len(weak_types)}/{tested}")
    print(f"Insufficient data:  {len(insuff_types)}")

    if valid_pct >= 50:
        print("\n>> RECOMMENDATION: Prune noise types, enable DL weight at 0.02-0.05")
        print(f"  Keep: {[t.replace('deep_learning_', '') for t in valid_types]}")
        print(f"  Prune: {[t.replace('deep_learning_', '') for t in noise_types]}")
    else:
        print("\n>> RECOMMENDATION: Disable DL entirely")
        print("  - Remove patterns from AI context")
        print("  - Simplify _deep_learning_factor() to return (0.5, [])")
        if valid_types:
            print(f"  - Exception: consider keeping {[t.replace('deep_learning_', '') for t in valid_types]} for AI context only")

    # ── Save results ────────────────────────────────────────────────
    output = {
        "holdout_start": HOLDOUT_START,
        "holdout_races": len(races),
        "holdout_runners": total_runners,
        "baseline_win_rate": round(baseline, 4),
        "type_verdicts": type_verdicts,
        "pattern_stats": all_pattern_stats,
        "recommendation": "enable" if valid_pct >= 50 else "disable",
        "valid_types": [t.replace("deep_learning_", "") for t in valid_types],
        "noise_types": [t.replace("deep_learning_", "") for t in noise_types],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    conn.close()
    return output


if __name__ == "__main__":
    validate()
