"""Feature extraction for LightGBM probability models.

Three extraction paths producing an identical 61-feature vector:
  - extract_features_from_db_row(): training from backtest.db rows
  - extract_features_from_runner(): live inference from Runner ORM objects
  - extract_features_batch(): batch inference for a full race

LightGBM handles NaN natively — missing data is passed as NaN, not imputed.
"""

import math
import re
import statistics
from typing import Any, Optional

import numpy as np

# ──────────────────────────────────────────────
# Feature names — ORDER MUST MATCH extraction
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    # Market (1)
    "market_prob",
    # Career form (3)
    "career_win_pct", "career_place_pct", "career_starts",
    # Track/distance records (6)
    "track_dist_sr", "track_dist_starts",
    "distance_sr", "distance_starts",
    "track_sr", "track_starts",
    # Condition records (10)
    "cond_good_sr", "cond_good_starts",
    "cond_soft_sr", "cond_soft_starts",
    "cond_heavy_sr", "cond_heavy_starts",
    "cond_firm_sr", "cond_firm_starts",
    "cond_synthetic_sr", "cond_synthetic_starts",
    # Freshness (4)
    "first_up_sr", "first_up_starts",
    "second_up_sr", "second_up_starts",
    # Recent form (5)
    "last5_score", "last5_wins", "last5_places",
    "form_trend",           # slope of L5 positions (negative = improving)
    "place_vs_market",      # career_place_pct minus market implied place rate
    # Class & fitness (4)
    "prize_per_start", "handicap_rating", "avg_margin",
    "class_differential",   # race prize money / career prize per start
    # Pace (2)
    "days_since_last", "settle_pos",
    # Barrier (2)
    "barrier_relative", "barrier_raw",
    # Jockey (5)
    "jockey_career_sr", "jockey_career_a2e", "jockey_career_pot",
    "jockey_career_runners", "jockey_l100_sr",
    # Trainer (4)
    "trainer_career_sr", "trainer_career_a2e",
    "trainer_career_pot", "trainer_l100_sr",
    # Combo (3)
    "combo_career_sr", "combo_career_runners", "combo_l100_sr",
    # Physical (5)
    "weight", "weight_diff", "age",
    "is_gelding", "is_mare",
    # Movement (1)
    "price_move_pct",
    # Group (2)
    "group_starts", "group_sr",
    # Place specialist (2)
    "distance_place_rate",  # place rate at this distance
    "track_place_rate",     # place rate at this track
    # Race context (2)
    "field_size", "distance",
    # ── New features (v4) ── added after 61-feature v3 model ──
    # These are extracted but only used after model retrain.
    # Pace signal (1)
    "is_leader",              # 1.0 if speed_map_position == leader
    # Distance signal (1)
    "is_staying",             # 1.0 if distance >= 2000m
    # Form signal (1)
    "last_start_won",         # 1.0 if last-five starts with '1'
    # Country (3) — one-hot for AU/HK/NZ
    "is_australia",           # 1.0 if Australian venue
    "is_hong_kong",           # 1.0 if HK venue (Sha Tin, Happy Valley)
    "is_new_zealand",         # 1.0 if NZ venue
    # Head-to-head & field beaten (2) — S21
    "head_to_head_wins",      # count of horses in today's field beaten in past meetings
    "field_beaten_pct",       # avg % of field beaten in last 5 starts
    # Track bias (1) — S23
    "rail_bias_score",        # barrier advantage/disadvantage from rail position
    # ── Pace / speed map (4) — missing from v3, major gap ──
    "pf_map_factor",          # PuntingForm map factor (>1 = pace advantage)
    "pf_speed_rank",          # predicted early speed rank in field
    "pf_jockey_factor",       # jockey factor from PF speed map
    "speed_map_encoded",      # leader=1, on_pace=2, midfield=3, backmarker=4, unknown=0
    # ── Weather (4) — not in any prior version ──
    "weather_rain_prob",      # hourly rain probability %
    "weather_wind_speed",     # wind speed km/h
    "weather_temp",           # temperature celsius
    "weather_humidity",       # humidity %
    # ── Gear (2) — strong form reversal signals ──
    "has_gear_change",        # 1.0 if any gear change this start
    "blinkers_on",            # 1.0 if wearing blinkers (from gear field)
    # ── Campaign / fatigue (2) ──
    "runs_this_prep",         # count of runs since last spell (>60 days gap)
    "campaign_win_rate",      # win rate this preparation
    # ── Class specialist (1) ──
    "class_win_rate",         # win rate at this exact class level
    # ── Track specialist (1) ──
    "track_specialist",       # 1.0 if 3+ wins AND 25%+ SR at this track
    # ── Stewards / excuses (2) ──
    "has_stewards_issue",     # 1.0 if stewards comment present on last start
    "has_excuse",             # 1.0 if form excuse flagged (checked, held up, etc.)
    # ── Age×sex interaction (1) ──
    "peak_age_sex_score",     # peak performance curve: 4-5yo gelding=1.0, 3yo filly=0.7, 7+yo=0.6
    # ── Odds flucs / smart money (1) ──
    "flucs_direction",        # -1=drifting, 0=stable, 1=firming (from odds_flucs array)
    # ── v6: Context buckets (4) — ordinal-encoded for LightGBM tree splits ──
    "distance_bucket",        # 1=sprint, 2=short, 3=middle, 4=classic, 5=staying
    "track_cond_bucket",      # 1=firm, 2=good, 3=soft, 4=heavy, 5=synthetic
    "class_bucket",           # 1=maiden, 2=restricted, 3=benchmark, 4=handicap, 5=open, 6=group
    "venue_type_code",        # 1=metro, 2=provincial, 3=country
    # ── v6: Interaction features (10) — context-dependent factor weighting ──
    "jockey_sr_x_cond",       # jockey_career_sr * horse's SR on today's track condition
    "barrier_rel_x_dist",    # barrier_relative * distance decay (barrier matters more in sprints)
    "form_score_x_cond",     # last5_score * horse's condition SR (form relevance by going)
    "weight_diff_x_dist",    # weight_diff * distance/1400 (weight penalty grows with distance)
    "settle_pos_x_dist",     # settle_pos * distance decay (pace advantage by distance)
    "jockey_sr_x_class",     # jockey_career_sr * class_bucket/6 (jockey importance by class)
    "barrier_rel_x_field",   # barrier_relative * field_size/16 (barrier in big fields)
    "trainer_sr_x_venue",    # trainer_career_sr * venue_type/3 (trainer metro/country spec)
    "form_trend_x_class",    # form_trend * class_bucket/6 (improving form by class)
    "days_since_x_class",    # (days_since/365) * class_bucket/6 (freshness by class)
    # ── v8: KASH model features (4) — DEPRECATED: set to NaN at inference ──
    # Kept for backward compat with trained model indices. Retrain will learn to ignore.
    "kash_wp_implied",       # DEPRECATED — external model prediction
    "kash_early_speed",      # DEPRECATED — external model
    "kash_late_speed",       # DEPRECATED — external model
    "kash_consensus",        # DEPRECATED — external model
    # ── v9: Signal-driven features (8) — from 381K runner audit ──
    "kri_score",             # KRI from most recent start (0-100) — 10x discriminator
    "kri_trend",             # Recent 2 KRI avg vs older 2 (positive = improving)
    "position_change",       # Settle→finish from last start (positive = gained ground)
    "weight_change",         # Weight today vs last start (positive = gained = class up)
    "distance_change",       # Distance today vs last start (metres, 0 = same)
    "margin_last",           # Margin beaten last start (lengths, 0 = won)
    "condition_record_sr",   # SR on today's specific going (good/soft/heavy)
    "combo_a2e",             # Trainer+jockey combo A2E value
]

NUM_FEATURES = len(FEATURE_NAMES)  # 102
# Features the current trained model knows (v5). New features appended after this.
NUM_FEATURES_V5 = 88
NUM_FEATURES_V3 = 61


# ──────────────────────────────────────────────
# Stats string parsers
# ──────────────────────────────────────────────

_STATS_COLON = re.compile(r"(\d+)\s*:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)")
_STATS_DASH4 = re.compile(r"^(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)$")


def _parse_stats(s: Any) -> tuple[int, int, int, int] | None:
    """Parse stats string → (starts, wins, seconds, thirds). Returns None if unparseable."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    m = _STATS_COLON.search(s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    m = _STATS_DASH4.match(s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


def _sr_from_stats(s: Any) -> tuple[Optional[float], int]:
    """Return (strike_rate, starts) from a stats string or JSON dict."""
    # Handle JSON dict format from live DB: {"starts": 27, "wins": 5, ...}
    if isinstance(s, str) and s.strip().startswith("{"):
        try:
            import json
            d = json.loads(s)
            if isinstance(d, dict):
                starts = d.get("starts", 0) or 0
                wins = d.get("wins", 0) or 0
                return (wins / starts, starts) if starts > 0 else (None, 0)
        except (json.JSONDecodeError, ValueError):
            pass
    parsed = _parse_stats(s)
    if not parsed:
        return None, 0
    starts, wins, seconds, thirds = parsed
    if starts == 0:
        return None, 0
    return wins / starts, starts


def _place_rate_from_stats(s: Any) -> Optional[float]:
    """Return place rate from a stats string or JSON dict."""
    if isinstance(s, str) and s.strip().startswith("{"):
        try:
            import json
            d = json.loads(s)
            if isinstance(d, dict):
                starts = d.get("starts", 0) or 0
                wins = d.get("wins", 0) or 0
                seconds = d.get("seconds", 0) or 0
                thirds = d.get("thirds", 0) or 0
                return (wins + seconds + thirds) / starts if starts > 0 else None
        except (json.JSONDecodeError, ValueError):
            pass
    parsed = _parse_stats(s)
    if not parsed:
        return None
    starts, wins, seconds, thirds = parsed
    if starts == 0:
        return None
    return (wins + seconds + thirds) / starts


def _score_last_five(last_five: Any) -> Optional[float]:
    """Score last-5 form string (e.g. '12x34') on 0-1 scale."""
    if not last_five:
        return None
    s = str(last_five).strip()
    if not s:
        return None
    scores = []
    for ch in s[:5]:
        if ch == '1':
            scores.append(1.0)
        elif ch == '2':
            scores.append(0.7)
        elif ch == '3':
            scores.append(0.5)
        elif ch == '4':
            scores.append(0.3)
        elif ch in ('5', '6', '7', '8', '9'):
            scores.append(0.1)
        elif ch in ('x', 'f', '0'):
            scores.append(0.0)
        # Skip non-digit/non-form chars
    if not scores:
        return None
    # Weight recent starts higher
    weights = [1.0, 0.9, 0.8, 0.7, 0.6][:len(scores)]
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def _form_trend(last_five: Any) -> Optional[float]:
    """Compute form trend from L5 string. Negative = improving (positions decreasing).

    Uses finish positions as raw numbers. Returns slope of linear regression.
    E.g. "56321" → positions [5,6,3,2,1] → negative slope (improving).
    """
    if not last_five:
        return None
    s = str(last_five).strip()
    positions = []
    for ch in s[:5]:
        if ch.isdigit() and ch != '0':
            positions.append(int(ch))
        elif ch in ('x', 'f', '0'):
            positions.append(10)
    if len(positions) < 3:
        return None
    n = len(positions)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(positions) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, positions))
    den = sum((xi - x_mean) ** 2 for xi in xs)
    if den == 0:
        return 0.0
    return num / den


def _count_last5(last_five: Any, positions: set) -> Optional[int]:
    """Count finishes in given positions from last-5 string."""
    if not last_five:
        return None
    s = str(last_five).strip()[:5]
    count = 0
    for ch in s:
        if ch.isdigit() and int(ch) in positions:
            count += 1
    return count


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from ORM object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _try_parse_json(val: Any) -> Optional[dict]:
    """Try to parse a JSON string into a dict. Returns None if not JSON."""
    if not val:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            import json
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def capped_f(d: dict, key: str, min_runners: int = 20) -> Optional[float]:
    """Extract float from dict with minimum runner threshold."""
    if not d or not isinstance(d, dict):
        return None
    runners = d.get("runners", 0) or d.get("Runners", 0)
    if runners < min_runners:
        return None
    val = d.get(key) or d.get(key.title())
    return float(val) if val is not None else None


def _extract_head_to_head_wins(form_history_raw: Any, field_horse_names: set[str]) -> float:
    """Count how many horses in today's field this horse has beaten in past races.

    Scans form_history JSON for past races where both this horse and another
    horse in today's field were present, and this horse finished ahead.
    """
    parsed = _try_parse_json(form_history_raw)
    if not parsed or not isinstance(parsed, list) or not field_horse_names:
        return float("nan")
    beaten = set()
    for race_entry in parsed[:10]:  # Check last 10 starts
        if not isinstance(race_entry, dict):
            continue
        finish = race_entry.get("FinishPosition") or race_entry.get("finish_position")
        if not finish:
            continue
        try:
            my_pos = int(finish)
        except (ValueError, TypeError):
            continue
        # Check other runners in this past race
        other_runners = race_entry.get("other_runners") or race_entry.get("OtherRunners") or []
        if not isinstance(other_runners, list):
            continue
        for other in other_runners:
            if not isinstance(other, dict):
                continue
            other_name = (other.get("horse_name") or other.get("HorseName") or "").strip().upper()
            other_pos = other.get("finish_position") or other.get("FinishPosition")
            if not other_name or not other_pos:
                continue
            try:
                other_pos = int(other_pos)
            except (ValueError, TypeError):
                continue
            if other_name in field_horse_names and my_pos < other_pos:
                beaten.add(other_name)
    return float(len(beaten)) if beaten else 0.0


def _extract_field_beaten_pct(form_history_raw: Any) -> float:
    """Average percentage of the field beaten in last 5 starts.

    E.g., finished 3rd in 12 runners = beat 9/11 = 81.8%.
    """
    parsed = _try_parse_json(form_history_raw)
    if not parsed or not isinstance(parsed, list):
        return float("nan")
    pcts = []
    for race_entry in parsed[:5]:
        if not isinstance(race_entry, dict):
            continue
        finish = race_entry.get("FinishPosition") or race_entry.get("finish_position")
        field = race_entry.get("Starters") or race_entry.get("starters") or race_entry.get("field_size")
        if not finish or not field:
            continue
        try:
            pos = int(finish)
            total = int(field)
        except (ValueError, TypeError):
            continue
        if total <= 1 or pos < 1:
            continue
        beaten_count = total - pos
        pcts.append(beaten_count / (total - 1))  # % of other runners beaten
    if pcts:
        return statistics.mean(pcts)
    return float("nan")


def _compute_rail_bias_score(rail_position: Any, barrier: int, field_size: int) -> float:
    """Compute barrier advantage/disadvantage from rail position.

    - True rail + inside barrier (<=4): advantage = 0.1
    - Rail out 3m+ and outside barrier (> field_size - 3): advantage = 0.1
    - Otherwise: 0.0
    """
    if not rail_position or not barrier or field_size < 2:
        return 0.0
    rail = str(rail_position).lower().strip()
    if "true" in rail and barrier <= 4:
        return 0.1
    # Parse "out Xm" pattern
    out_match = re.search(r"out\s+(\d+)", rail)
    if out_match:
        out_metres = int(out_match.group(1))
        if out_metres >= 3 and barrier > field_size - 3:
            return 0.1
    return 0.0


def _extract_campaign_stats(form_history_raw: Any, days_since: Any) -> tuple[float, float]:
    """Extract runs this preparation and campaign win rate.

    A preparation ends when there's a gap of > 60 days between runs.
    Returns (runs_this_prep, campaign_win_rate).
    """
    nan = float("nan")
    parsed = _try_parse_json(form_history_raw)
    if not parsed or not isinstance(parsed, list):
        return (nan, nan)
    runs = 0
    wins = 0
    for i, start in enumerate(parsed):
        if i > 0:
            # Check gap to previous start
            days_gap = None
            try:
                from datetime import datetime
                d1 = start.get("date") or start.get("meeting_date")
                d0 = parsed[i - 1].get("date") or parsed[i - 1].get("meeting_date")
                if d1 and d0:
                    dt1 = datetime.strptime(str(d1)[:10], "%Y-%m-%d")
                    dt0 = datetime.strptime(str(d0)[:10], "%Y-%m-%d")
                    days_gap = abs((dt0 - dt1).days)
            except (ValueError, TypeError):
                pass
            if days_gap and days_gap > 60:
                break  # End of this preparation
        runs += 1
        pos = start.get("finish_position") or start.get("position")
        try:
            if int(pos) == 1:
                wins += 1
        except (ValueError, TypeError):
            pass
    if runs == 0:
        return (nan, nan)
    return (float(runs), wins / runs if runs > 0 else 0.0)


def _class_sr_from_stats(stats_raw: Any) -> float:
    """Extract strike rate from a stats JSON dict (single float, not tuple)."""
    if not stats_raw:
        return float("nan")
    parsed = stats_raw
    if isinstance(stats_raw, str):
        parsed = _try_parse_json(stats_raw)
    if isinstance(parsed, dict):
        starts = parsed.get("starts", 0)
        wins = parsed.get("wins", 0)
        if starts and starts > 0:
            return wins / starts
    return float("nan")


def _wins_from_stats(stats_raw: Any) -> int:
    """Extract win count from a stats JSON dict."""
    if not stats_raw:
        return 0
    parsed = stats_raw
    if isinstance(stats_raw, str):
        parsed = _try_parse_json(stats_raw)
    if isinstance(parsed, dict):
        return int(parsed.get("wins", 0))
    return 0


def _compute_peak_age_sex(age: float, is_gelding: float, is_mare: float) -> float:
    """Peak performance curve based on age and sex.

    Returns 0.0-1.0 score where 1.0 = peak performance.
    Peak: 4-5yo geldings. 3yo fillies penalised. 7+ declining.
    """
    nan = float("nan")
    if age is None or age != age:  # NaN check
        return nan
    a = float(age)
    # Base age curve
    if 4.0 <= a <= 5.0:
        score = 1.0
    elif a == 3.0:
        score = 0.85
    elif a == 6.0:
        score = 0.90
    elif a == 7.0:
        score = 0.75
    elif a >= 8.0:
        score = 0.60
    elif a <= 2.0:
        score = 0.70
    else:
        score = 0.80
    # Sex adjustment
    if is_mare == 1.0 and a == 3.0:
        score *= 0.85  # 3yo fillies/mares weaker in open
    return score


# ──────────────────────────────────────────────
# Context bucket encoders (v6 interaction features)
# ──────────────────────────────────────────────

def _distance_bucket(distance: Any) -> float:
    """Ordinal encode distance into racing context buckets."""
    d = _safe_float(distance)
    if d is None:
        return 0.0
    if d < 1200:
        return 1.0   # sprint
    if d < 1400:
        return 2.0   # short
    if d < 1800:
        return 3.0   # middle
    if d < 2200:
        return 4.0   # classic
    return 5.0        # staying


def _track_cond_bucket(condition: Any) -> float:
    """Ordinal encode track condition (wetness scale).

    Handles both full names ("Good 4", "Soft 5", "Heavy 8") and
    Proform short codes ("G4", "S5", "H8", "F", "Syn").
    """
    if not condition:
        return 0.0
    c = str(condition).lower().strip()
    # Proform short codes: G/G3/G4 = good, S5/S6/S7 = soft, H8/H9/H10 = heavy, F = firm
    if c.startswith("f") and (len(c) <= 2 or "firm" in c):
        return 1.0
    if c.startswith("g") or "good" in c:
        return 2.0
    if c.startswith("s") and c[0:1] == "s" and (len(c) <= 2 or c[1:2].isdigit()) or "soft" in c or "dead" in c:
        return 3.0
    if c.startswith("h") and (len(c) <= 3 or "heavy" in c):
        return 4.0
    if "synth" in c or "syn" in c or "all weather" in c:
        return 5.0
    return 0.0


def _class_bucket(race_class: Any) -> float:
    """Ordinal encode race class.

    Handles both live DB formats ("Maiden", "BM64") and
    Proform formats ("Maiden;", "Benchmark 58;", "Class 1;").
    """
    if not race_class:
        return 0.0
    c = str(race_class).lower().strip().rstrip(";").strip()
    if "maiden" in c or "mdn" in c:
        return 1.0
    if any(x in c for x in ("restricted", "cg&e", "cg ", "c,g", "f&m")):
        return 2.0
    if "benchmark" in c or c.startswith("bm"):
        return 3.0
    if "handicap" in c or "hcp" in c:
        return 4.0
    if any(x in c for x in ("open", "wfa", "weight for age", "set weight")):
        return 5.0
    if any(x in c for x in ("group", "listed", "stakes")):
        return 6.0
    if "class" in c:
        return 2.0
    return 0.0


def _venue_type_code(venue: Any) -> float:
    """1=metro, 2=provincial, 3=country."""
    if not venue:
        return 0.0
    try:
        from punty.venues import is_metro
        if is_metro(str(venue)):
            return 1.0
        return 3.0  # Default country (provincial detection would need separate list)
    except Exception:
        return 0.0


def _condition_sr_for_today(
    track_cond: float, good_sr: Any, soft_sr: Any, heavy_sr: Any, firm_sr: Any,
) -> float:
    """Select the horse's strike rate matching today's track condition.

    Falls back to any available condition SR when the primary is unavailable,
    preferring conditions closest on the wetness scale.
    """
    # Map condition bucket to ordered preference list
    preference = {
        1.0: [firm_sr, good_sr, soft_sr, heavy_sr],    # Firm → try good, soft, heavy
        2.0: [good_sr, firm_sr, soft_sr, heavy_sr],    # Good → try firm, soft, heavy
        3.0: [soft_sr, good_sr, heavy_sr, firm_sr],    # Soft → try good, heavy, firm
        4.0: [heavy_sr, soft_sr, good_sr, firm_sr],    # Heavy → try soft, good, firm
    }
    candidates = preference.get(track_cond, [good_sr, soft_sr, heavy_sr, firm_sr])

    for sr in candidates:
        v = _f(sr)
        if v == v:  # not NaN
            return v
    return float("nan")


def _dist_decay(distance_bucket: float) -> float:
    """Distance decay factor — barrier/pace matters less as distance increases."""
    return {1.0: 1.0, 2.0: 0.85, 3.0: 0.65, 4.0: 0.4, 5.0: 0.2}.get(distance_bucket, 0.5)


def _compute_interaction_features(
    distance_bucket: float, track_cond_bucket: float, class_bucket: float,
    venue_type: float, jockey_sr: float, barrier_rel: float, l5_score: float,
    weight_diff: float, settle: float, trainer_sr: float, form_trend: float,
    days_since: float, field_size: float, cond_sr: float,
) -> list[float]:
    """Compute 10 multiplicative interaction features."""
    nan = float("nan")
    dist_decay = _dist_decay(distance_bucket)
    class_norm = class_bucket / 6.0 if class_bucket > 0 else nan

    return [
        _f(jockey_sr) * _f(cond_sr) if _f(jockey_sr) == _f(jockey_sr) and _f(cond_sr) == _f(cond_sr) else nan,
        _f(barrier_rel) * dist_decay if _f(barrier_rel) == _f(barrier_rel) else nan,
        _f(l5_score) * _f(cond_sr) if _f(l5_score) == _f(l5_score) and _f(cond_sr) == _f(cond_sr) else nan,
        _f(weight_diff) * (_f(float(distance_bucket)) * 400 / 1400) if _f(weight_diff) == _f(weight_diff) else nan,
        _f(settle) * dist_decay if _f(settle) == _f(settle) else nan,
        _f(jockey_sr) * class_norm if _f(jockey_sr) == _f(jockey_sr) and class_norm == class_norm else nan,
        _f(barrier_rel) * (field_size / 16.0) if _f(barrier_rel) == _f(barrier_rel) and field_size > 0 else nan,
        _f(trainer_sr) * (venue_type / 3.0) if _f(trainer_sr) == _f(trainer_sr) and venue_type > 0 else nan,
        _f(form_trend) * class_norm if _f(form_trend) == _f(form_trend) and class_norm == class_norm else nan,
        (_f(days_since) / 365.0) * class_norm if _f(days_since) == _f(days_since) and class_norm == class_norm else nan,
    ]


def _extract_flucs_direction(odds_flucs_raw: Any) -> float:
    """Extract overall odds movement direction from flucs array.

    Returns: -1.0 (drifting), 0.0 (stable), 1.0 (firming).
    """
    nan = float("nan")
    if not odds_flucs_raw:
        return nan
    parsed = odds_flucs_raw
    if isinstance(odds_flucs_raw, str):
        parsed = _try_parse_json(odds_flucs_raw)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return nan
    try:
        first = float(parsed[0].get("odds", 0) if isinstance(parsed[0], dict) else parsed[0])
        last = float(parsed[-1].get("odds", 0) if isinstance(parsed[-1], dict) else parsed[-1])
        if first <= 0 or last <= 0:
            return nan
        pct_change = (first - last) / first
        if pct_change > 0.10:
            return 1.0   # Firming (shortened)
        elif pct_change < -0.10:
            return -1.0  # Drifting
        return 0.0  # Stable
    except (ValueError, TypeError, IndexError):
        return nan


def _extract_avg_margin(form_history_raw: Any) -> float:
    """Extract average margin from last 5 starts in form_history JSON."""
    parsed = _try_parse_json(form_history_raw)
    if not parsed or not isinstance(parsed, list):
        return float("nan")
    margins = []
    for f in parsed[:5]:
        if not isinstance(f, dict):
            continue
        m = f.get("Margin") or f.get("margin")
        if m is not None:
            try:
                margins.append(abs(float(m)))
            except (ValueError, TypeError):
                pass
    if margins:
        return statistics.mean(margins)
    return float("nan")


def capped_sr(val: Optional[float]) -> Optional[float]:
    """Convert percentage strike rate to decimal (Proform stores as %, live DB as %).
    Returns as decimal fraction matching Proform training format."""
    if val is None:
        return None
    return val


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float or return None."""
    if val is None:
        return None
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _f(val: Any) -> float:
    """Convert to float, NaN if None."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


# ──────────────────────────────────────────────
# Extraction from backtest.db rows (training)
# ──────────────────────────────────────────────

def extract_features_from_db_row(
    runner: dict, race: dict, meeting: dict, field_size: int, avg_weight: float,
) -> list:
    """Extract feature vector from backtest.db row dicts."""
    nan = float("nan")

    # Market
    odds = runner.get("current_odds") or runner.get("odds_tab")
    market_prob = 1.0 / odds if odds and odds > 1.0 else nan

    # Career
    career = _parse_stats(runner.get("career_record"))
    if career and career[0] > 0:
        career_starts, career_wins, career_secs, career_thirds = career
        career_win_pct = career_wins / career_starts
        career_place_pct = (career_wins + career_secs + career_thirds) / career_starts
    else:
        career_starts = nan
        career_win_pct = nan
        career_place_pct = nan

    # Track/distance/condition records
    td_sr, td_starts = _sr_from_stats(runner.get("track_dist_stats"))
    dist_sr, dist_starts = _sr_from_stats(runner.get("distance_stats"))
    trk_sr, trk_starts = _sr_from_stats(runner.get("track_stats"))

    good_sr, good_starts = _sr_from_stats(runner.get("good_track_stats"))
    soft_sr, soft_starts = _sr_from_stats(runner.get("soft_track_stats"))
    heavy_sr, heavy_starts = _sr_from_stats(runner.get("heavy_track_stats"))
    firm_sr, firm_starts = nan, 0
    synth_sr, synth_starts = nan, 0

    fu_sr, fu_starts = _sr_from_stats(runner.get("first_up_stats"))
    su_sr, su_starts = _sr_from_stats(runner.get("second_up_stats"))

    # Last 5 form
    last_five = runner.get("last_five") or runner.get("form", "")
    l5_score = _score_last_five(last_five)
    l5_wins = _count_last5(last_five, {1})
    l5_places = _count_last5(last_five, {1, 2, 3})
    form_trend_val = _form_trend(last_five)

    # Place vs market residual
    place_count = 2 if field_size <= 7 else 3
    market_implied_place = (place_count / field_size) if field_size > 0 else nan
    place_vs_market = (career_place_pct - market_implied_place
                       if not math.isnan(career_place_pct) and not math.isnan(market_implied_place)
                       else nan)

    # Class & fitness
    prize = runner.get("career_prize_money")
    c_starts = career[0] if career else None
    prize_per_start = prize / c_starts if prize and c_starts and c_starts > 0 else nan
    race_prize = race.get("prize_money") or 0
    class_diff = race_prize / prize_per_start if race_prize and not math.isnan(prize_per_start) and prize_per_start > 0 else nan

    handicap = _safe_float(runner.get("handicap_rating"))
    avg_margin = nan

    # Pace
    days_since = _safe_float(runner.get("days_since_last_run"))
    settle = _safe_float(runner.get("pf_settle"))

    # Barrier
    barrier = runner.get("barrier") or 0
    barrier_relative = (barrier - 1) / (field_size - 1) if barrier and field_size > 1 else nan

    # Jockey/trainer stats
    j_sr, j_starts = _sr_from_stats(runner.get("jockey_stats"))
    t_sr, t_starts = _sr_from_stats(runner.get("trainer_stats"))
    jockey_career_sr = j_sr if j_sr is not None else nan
    jockey_career_a2e = nan
    jockey_career_pot = nan
    jockey_career_runners = j_starts if j_starts else nan
    jockey_l100_sr = nan

    trainer_career_sr = t_sr if t_sr is not None else nan
    trainer_career_a2e = nan
    trainer_career_pot = nan
    trainer_l100_sr = nan

    combo_career_sr = nan
    combo_career_runners = nan
    combo_l100_sr = nan

    # Physical
    weight = _safe_float(runner.get("weight"))
    weight_diff = weight - avg_weight if weight and avg_weight else nan
    age = _safe_float(runner.get("horse_age"))
    sex = (runner.get("horse_sex") or "").lower()
    is_gelding = 1.0 if "gelding" in sex else 0.0
    is_mare = 1.0 if ("mare" in sex or "filly" in sex) else 0.0

    # Movement
    opening = _safe_float(runner.get("opening_odds"))
    current = _safe_float(runner.get("current_odds"))
    if opening and current and opening > 1 and current > 1:
        price_move_pct = (opening - current) / opening
    else:
        price_move_pct = nan

    # Group stats — not in backtest.db
    group_starts = nan
    group_sr = nan

    # Place specialist rates
    dist_place = _place_rate_from_stats(runner.get("distance_stats"))
    trk_place = _place_rate_from_stats(runner.get("track_stats"))

    # Race context
    distance = race.get("distance") or 1400

    # ── v4 features ──
    # Leader signal — from Proform speed map settle position
    settle_raw = runner.get("pf_settle") or runner.get("speed_map_position") or ""
    is_leader = 1.0 if (isinstance(settle_raw, str) and settle_raw == "leader") or (isinstance(settle_raw, (int, float)) and settle_raw and settle_raw <= 2) else 0.0

    # Staying race
    is_staying = 1.0 if distance and distance >= 2000 else 0.0

    # Last start won
    l5_raw = runner.get("last_five") or runner.get("form", "")
    last_start_won = 1.0 if l5_raw and str(l5_raw).lstrip("0").startswith("1") else 0.0

    # Country — derive from meeting venue
    venue = (meeting.get("venue") or "").lower() if isinstance(meeting, dict) else ""
    from punty.venues import guess_state
    state = guess_state(venue)
    is_australia = 1.0 if state not in ("HK", "SGP", "NZ", "JP", "UK", "") else 0.0
    is_hong_kong = 1.0 if state == "HK" else 0.0
    is_new_zealand = 1.0 if state == "NZ" else 0.0

    # Head-to-head & field beaten (S21) — not available in backtest.db context
    head_to_head_wins = nan
    field_beaten_pct = _extract_field_beaten_pct(runner.get("form_history"))

    # Rail bias score (S23)
    rail_pos = meeting.get("rail_position") if isinstance(meeting, dict) else None
    rail_bias_score = _compute_rail_bias_score(rail_pos, barrier, field_size)

    # ── Pace / speed map (4) ──
    pf_mf = _safe_float(runner.get("pf_map_factor"))
    pf_sr = _safe_float(runner.get("pf_speed_rank"))
    pf_jf = _safe_float(runner.get("pf_jockey_factor"))
    smp = str(runner.get("speed_map_position") or "").lower()
    speed_map_encoded = {"leader": 1.0, "on_pace": 2.0, "midfield": 3.0, "backmarker": 4.0}.get(smp, 0.0)

    # ── Weather (4) ──
    m = meeting if isinstance(meeting, dict) else {}
    weather_rain_prob = _safe_float(m.get("hourly_rain_prob") or m.get("rain_probability"))
    weather_wind_speed = _safe_float(m.get("weather_wind_speed") or m.get("wind_speed"))
    weather_temp = _safe_float(m.get("weather_temp") or m.get("temperature"))
    weather_humidity = _safe_float(m.get("weather_humidity") or m.get("humidity"))

    # ── Gear (2) ──
    gear_changes_raw = str(runner.get("gear_changes") or "")
    has_gear_change = 1.0 if gear_changes_raw.strip() and gear_changes_raw.strip() != "None" else 0.0
    gear_raw = str(runner.get("gear") or "").lower()
    blinkers_on = 1.0 if "blinker" in gear_raw or "blink" in gear_raw else 0.0

    # ── Campaign / fatigue (2) ──
    runs_this_prep, campaign_wr = _extract_campaign_stats(runner.get("form_history"), days_since)

    # ── Class specialist (1) ──
    class_win_rate = _class_sr_from_stats(runner.get("class_stats"))

    # ── Track specialist (1) ──
    trk_wins_raw = _wins_from_stats(runner.get("track_stats"))
    track_specialist = 1.0 if trk_wins_raw >= 3 and trk_sr and trk_sr >= 0.25 else 0.0

    # ── Stewards / excuses (2) ──
    stew = str(runner.get("stewards_comment") or "")
    has_stewards_issue = 1.0 if stew.strip() and stew.strip() != "None" else 0.0
    excuses = runner.get("form_excuses") or ""
    has_excuse = 1.0 if excuses and str(excuses).strip() and str(excuses).strip() != "None" else 0.0

    # ── Age×sex interaction (1) ──
    peak_age_sex_score = _compute_peak_age_sex(age, is_gelding, is_mare)

    # ── Odds flucs / smart money (1) ──
    flucs_direction = _extract_flucs_direction(runner.get("odds_flucs"))

    # ── v6: Context buckets + interaction features (14) ──
    dist_bkt = _distance_bucket(distance)
    tc_raw = (meeting.get("track_condition") or "") if isinstance(meeting, dict) else ""
    tc_bkt = _track_cond_bucket(tc_raw)
    cls_bkt = _class_bucket(race.get("class") if isinstance(race, dict) else "")
    vt_code = _venue_type_code(meeting.get("venue") if isinstance(meeting, dict) else "")
    cond_sr = _condition_sr_for_today(tc_bkt, good_sr, soft_sr, heavy_sr, firm_sr)
    interactions = _compute_interaction_features(
        dist_bkt, tc_bkt, cls_bkt, vt_code,
        jockey_career_sr, barrier_relative, l5_score, weight_diff,
        settle, trainer_career_sr, form_trend_val, days_since,
        field_size, cond_sr,
    )

    return [
        market_prob,
        career_win_pct, career_place_pct, _f(career_starts),
        _f(td_sr), float(td_starts),
        _f(dist_sr), float(dist_starts),
        _f(trk_sr), float(trk_starts),
        _f(good_sr), float(good_starts),
        _f(soft_sr), float(soft_starts),
        _f(heavy_sr), float(heavy_starts),
        _f(firm_sr), float(firm_starts),
        _f(synth_sr), float(synth_starts),
        _f(fu_sr), float(fu_starts),
        _f(su_sr), float(su_starts),
        _f(l5_score), _f(l5_wins), _f(l5_places),
        _f(form_trend_val), _f(place_vs_market),
        _f(prize_per_start), _f(handicap), _f(avg_margin),
        _f(class_diff),
        _f(days_since), _f(settle),
        _f(barrier_relative), float(barrier) if barrier else nan,
        _f(jockey_career_sr), _f(jockey_career_a2e), _f(jockey_career_pot),
        _f(jockey_career_runners), _f(jockey_l100_sr),
        _f(trainer_career_sr), _f(trainer_career_a2e),
        _f(trainer_career_pot), _f(trainer_l100_sr),
        _f(combo_career_sr), _f(combo_career_runners), _f(combo_l100_sr),
        _f(weight), _f(weight_diff), _f(age),
        is_gelding, is_mare,
        _f(price_move_pct),
        _f(group_starts), _f(group_sr),
        _f(dist_place), _f(trk_place),
        float(field_size), float(distance),
        # ── v4 features ──
        is_leader, is_staying, last_start_won,
        is_australia, is_hong_kong, is_new_zealand,
        _f(head_to_head_wins), _f(field_beaten_pct),
        rail_bias_score,
        # Pace / speed map (4)
        _f(pf_mf), _f(pf_sr), _f(pf_jf), speed_map_encoded,
        # Weather (4)
        _f(weather_rain_prob), _f(weather_wind_speed), _f(weather_temp), _f(weather_humidity),
        # Gear (2)
        has_gear_change, blinkers_on,
        # Campaign (2)
        _f(runs_this_prep), _f(campaign_wr),
        # Class/track specialist (2)
        _f(class_win_rate), track_specialist,
        # Stewards/excuses (2)
        has_stewards_issue, has_excuse,
        # Age×sex + flucs (2)
        _f(peak_age_sex_score), _f(flucs_direction),
        # ── v6: Context buckets (4) + interactions (10) ──
        dist_bkt, tc_bkt, cls_bkt, vt_code,
    ] + interactions


# ──────────────────────────────────────────────
# Extraction from live Runner ORM (inference)
# ──────────────────────────────────────────────

def extract_features_from_runner(
    runner: Any, race: Any, meeting: Any,
    field_size: int, avg_weight: float, overround: float = 1.0,
    **kwargs: Any,
) -> list:
    """Extract feature vector from live Runner ORM object."""
    nan = float("nan")

    # Market — median odds across bookmakers
    odds_vals = []
    for attr in ("odds_betfair", "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes"):
        v = _safe_float(_get(runner, attr))
        if v and v > 1.0:
            odds_vals.append(v)
    if not odds_vals:
        v = _safe_float(_get(runner, "current_odds"))
        if v and v > 1.0:
            odds_vals = [v]

    if odds_vals:
        median_odds = sorted(odds_vals)[len(odds_vals) // 2]
        raw_implied = 1.0 / median_odds
        market_prob = raw_implied / overround if overround > 0 else raw_implied
    else:
        median_odds = nan
        market_prob = nan

    # Career record from ORM
    career = _parse_stats(_get(runner, "career_record"))
    if career and career[0] > 0:
        career_starts, career_wins, career_secs, career_thirds = career
        career_win_pct = career_wins / career_starts
        career_place_pct = (career_wins + career_secs + career_thirds) / career_starts
    else:
        career_starts = nan
        career_win_pct = nan
        career_place_pct = nan

    # Track/distance records
    td_sr, td_starts = _sr_from_stats(_get(runner, "track_dist_stats"))
    dist_sr, dist_starts = _sr_from_stats(_get(runner, "distance_stats"))
    trk_sr, trk_starts = _sr_from_stats(_get(runner, "track_stats"))

    good_sr, good_starts = _sr_from_stats(_get(runner, "good_track_stats"))
    soft_sr, soft_starts = _sr_from_stats(_get(runner, "soft_track_stats"))
    heavy_sr, heavy_starts = _sr_from_stats(_get(runner, "heavy_track_stats"))
    firm_sr, firm_starts = nan, 0
    synth_sr, synth_starts = nan, 0

    fu_sr, fu_starts = _sr_from_stats(_get(runner, "first_up_stats"))
    su_sr, su_starts = _sr_from_stats(_get(runner, "second_up_stats"))

    # Last 5 form
    last_five = _get(runner, "last_five") or _get(runner, "form", "")
    l5_score = _score_last_five(last_five)
    l5_wins = _count_last5(last_five, {1})
    l5_places = _count_last5(last_five, {1, 2, 3})
    form_trend_val = _form_trend(last_five)

    # Place vs market residual
    place_count = 2 if field_size <= 7 else 3
    market_implied_place = (place_count / field_size) if field_size > 0 else nan
    place_vs_market = (career_place_pct - market_implied_place
                       if not math.isnan(career_place_pct) and not math.isnan(market_implied_place)
                       else nan)

    # Class & fitness
    prize = _safe_float(_get(runner, "career_prize_money"))
    c_starts = career[0] if career else None
    prize_per_start = prize / c_starts if prize and c_starts and c_starts > 0 else nan
    race_prize = _safe_float(_get(race, "prize_money")) or 0
    class_diff = race_prize / prize_per_start if race_prize and not math.isnan(prize_per_start) and prize_per_start > 0 else nan

    handicap = _safe_float(_get(runner, "handicap_rating"))
    avg_margin = _extract_avg_margin(_get(runner, "form_history"))

    # Pace
    days_since = _safe_float(_get(runner, "days_since_last_run"))
    settle = _safe_float(_get(runner, "pf_settle"))

    # Barrier
    barrier = _get(runner, "barrier") or 0
    barrier_relative = (barrier - 1) / (field_size - 1) if barrier and field_size > 1 else nan

    # Jockey/trainer — live ORM stores as JSON or string
    jockey_raw = _get(runner, "jockey_stats")
    jockey_json = _try_parse_json(jockey_raw)
    if jockey_json:
        jc = jockey_json.get("career", {})
        jl = jockey_json.get("last100", {})
        jcombo_c = jockey_json.get("combo_career", {})
        jcombo_l = jockey_json.get("combo_last100", {})
        jockey_career_sr = _f(capped_sr(capped_f(jc, "strike_rate", 50)))
        jockey_career_a2e = _f(capped_f(jc, "a2e", 50))
        jockey_career_pot = _f(capped_f(jc, "pot", 50))
        jockey_career_runners = _f(jc.get("runners", 0))
        jockey_l100_sr = _f(capped_sr(capped_f(jl, "strike_rate", 0)))
        combo_career_sr = _f(capped_sr(capped_f(jcombo_c, "strike_rate", 10)))
        combo_career_runners = _f(jcombo_c.get("runners", 0))
        combo_l100_sr = _f(capped_sr(capped_f(jcombo_l, "strike_rate", 10)))
    else:
        j_sr, j_starts = _sr_from_stats(jockey_raw)
        jockey_career_sr = j_sr if j_sr is not None else nan
        jockey_career_runners = j_starts if j_starts else nan
        jockey_career_a2e = nan
        jockey_career_pot = nan
        jockey_l100_sr = nan
        combo_career_sr = nan
        combo_career_runners = nan
        combo_l100_sr = nan

    trainer_raw = _get(runner, "trainer_stats")
    trainer_json = _try_parse_json(trainer_raw)
    if trainer_json:
        tc = trainer_json.get("career", {})
        tl = trainer_json.get("last100", {})
        trainer_career_sr = _f(capped_sr(capped_f(tc, "strike_rate", 50)))
        trainer_career_a2e = _f(capped_f(tc, "a2e", 50))
        trainer_career_pot = _f(capped_f(tc, "pot", 50))
        trainer_l100_sr = _f(capped_sr(capped_f(tl, "strike_rate", 0)))
    else:
        t_sr, t_starts = _sr_from_stats(trainer_raw)
        trainer_career_sr = t_sr if t_sr is not None else nan
        trainer_career_a2e = nan
        trainer_career_pot = nan
        trainer_l100_sr = nan

    # Physical
    weight = _safe_float(_get(runner, "weight"))
    weight_diff = weight - avg_weight if weight and avg_weight else nan
    age = _safe_float(_get(runner, "horse_age"))
    sex = (_get(runner, "horse_sex") or "").lower()
    is_gelding = 1.0 if "gelding" in sex else 0.0
    is_mare = 1.0 if ("mare" in sex or "filly" in sex) else 0.0

    # Movement
    opening = _safe_float(_get(runner, "opening_odds"))
    current = _safe_float(_get(runner, "current_odds"))
    if opening and current and opening > 1 and current > 1:
        price_move_pct = (opening - current) / opening
    else:
        price_move_pct = nan

    # Group stats — not available in live ORM
    group_starts = nan
    group_sr = nan

    # Place specialist rates
    dist_place = _place_rate_from_stats(_get(runner, "distance_stats"))
    trk_place = _place_rate_from_stats(_get(runner, "track_stats"))

    # Race context
    distance = _get(race, "distance") or 1400

    # ── v4 features ──
    # Leader signal — from speed map position
    smp = (_get(runner, "speed_map_position") or "").lower() if isinstance(_get(runner, "speed_map_position"), str) else ""
    settle_num = _safe_float(_get(runner, "pf_settle"))
    is_leader = 1.0 if smp == "leader" or (settle_num and settle_num <= 2) else 0.0

    # Staying race
    is_staying = 1.0 if distance and distance >= 2000 else 0.0

    # Last start won
    l5_raw = str(last_five or "")
    last_start_won = 1.0 if l5_raw and l5_raw.lstrip("0").startswith("1") else 0.0

    # Country — derive from meeting venue
    venue_str = str(_get(meeting, "venue") or "").strip()
    from punty.venues import guess_state
    state = guess_state(venue_str)
    is_australia = 1.0 if state not in ("HK", "SGP", "NZ", "JP", "UK", "") else 0.0
    is_hong_kong = 1.0 if state == "HK" else 0.0
    is_new_zealand = 1.0 if state == "NZ" else 0.0

    # Head-to-head & field beaten (S21)
    # field_horse_names populated by extract_features_batch via _field_names kwarg
    field_names = kwargs.get("_field_names", set())
    form_history_raw = _get(runner, "form_history")
    head_to_head_wins = _extract_head_to_head_wins(form_history_raw, field_names)
    field_beaten_pct = _extract_field_beaten_pct(form_history_raw)

    # Rail bias score (S23)
    rail_pos = _get(meeting, "rail_position")
    rail_bias_score = _compute_rail_bias_score(rail_pos, barrier, field_size)

    # ── Pace / speed map (4) ──
    pf_mf = _safe_float(_get(runner, "pf_map_factor"))
    pf_sr = _safe_float(_get(runner, "pf_speed_rank"))
    pf_jf = _safe_float(_get(runner, "pf_jockey_factor"))
    smp = str(_get(runner, "speed_map_position") or "").lower()
    speed_map_encoded = {"leader": 1.0, "on_pace": 2.0, "midfield": 3.0, "backmarker": 4.0}.get(smp, 0.0)

    # ── Weather (4) ──
    weather_rain_prob = _safe_float(_get(meeting, "hourly_rain_prob") or _get(meeting, "rain_probability"))
    weather_wind_speed = _safe_float(_get(meeting, "weather_wind_speed") or _get(meeting, "wind_speed"))
    weather_temp = _safe_float(_get(meeting, "weather_temp") or _get(meeting, "temperature"))
    weather_humidity = _safe_float(_get(meeting, "weather_humidity") or _get(meeting, "humidity"))

    # ── Gear (2) ──
    gear_changes_raw = str(_get(runner, "gear_changes") or "")
    has_gear_change = 1.0 if gear_changes_raw.strip() and gear_changes_raw.strip() != "None" else 0.0
    gear_raw = str(_get(runner, "gear") or "").lower()
    blinkers_on = 1.0 if "blinker" in gear_raw or "blink" in gear_raw else 0.0

    # ── Campaign / fatigue (2) ──
    runs_this_prep, campaign_wr = _extract_campaign_stats(form_history_raw, days_since)

    # ── Class specialist (1) ──
    class_win_rate = _class_sr_from_stats(_get(runner, "class_stats"))

    # ── Track specialist (1) ──
    trk_wins_raw = _wins_from_stats(_get(runner, "track_stats"))
    track_specialist = 1.0 if trk_wins_raw >= 3 and trk_sr and trk_sr >= 0.25 else 0.0

    # ── Stewards / excuses (2) ──
    stew = str(_get(runner, "stewards_comment") or "")
    has_stewards_issue = 1.0 if stew.strip() and stew.strip() != "None" else 0.0
    excuses = _get(runner, "form_excuses") or ""
    has_excuse = 1.0 if excuses and str(excuses).strip() and str(excuses).strip() != "None" else 0.0

    # ── Age×sex interaction (1) ──
    peak_age_sex_score = _compute_peak_age_sex(age, is_gelding, is_mare)

    # ── Odds flucs / smart money (1) ──
    flucs_direction = _extract_flucs_direction(_get(runner, "odds_flucs"))

    # ── v6: Context buckets + interaction features (14) ──
    dist_bkt = _distance_bucket(distance)
    tc_raw = str(_get(meeting, "track_condition") or "")
    tc_bkt = _track_cond_bucket(tc_raw)
    cls_bkt = _class_bucket(str(_get(race, "class_") or _get(race, "class") or ""))
    vt_code = _venue_type_code(str(_get(meeting, "venue") or ""))
    cond_sr = _condition_sr_for_today(tc_bkt, good_sr, soft_sr, heavy_sr, firm_sr)
    interactions = _compute_interaction_features(
        dist_bkt, tc_bkt, cls_bkt, vt_code,
        jockey_career_sr, barrier_relative, l5_score, weight_diff,
        settle, trainer_career_sr, form_trend_val, days_since,
        field_size, cond_sr,
    )

    # ── INDEPENDENCE: Override external model features to NaN ──
    # market_prob is feature #1 (index 0) — set to NaN so LGBM ignores it.
    # Model is 100% independent — no market, KASH, or PF predictions in features.
    market_prob = nan  # Override — was computed above for other uses but NOT for LGBM
    flucs_direction = nan  # Odds-derived — override
    price_move_pct = nan  # Odds-derived — override
    place_vs_market = nan  # Market comparison — override

    # ── v8: KASH features — DEPRECATED (external model, set to NaN) ──
    kash_wp_implied = nan
    kash_early_spd = nan
    kash_late_spd = nan
    kash_consensus = nan

    # ── v9: Signal-driven features (8) ──
    import json as _json_v9

    # KRI score from form_history
    kri_score = nan
    kri_trend = nan
    position_change_val = nan
    margin_last_val = nan
    fh_raw = _get(runner, "form_history")
    if fh_raw:
        try:
            fh = _json_v9.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
            if isinstance(fh, list):
                # KRI from recent starts
                kri_vals = []
                for s in fh[:5]:
                    if isinstance(s, dict) and not s.get("is_trial"):
                        k = s.get("kri")
                        if k is not None:
                            try:
                                kri_vals.append(float(k))
                            except (ValueError, TypeError):
                                pass
                if kri_vals:
                    kri_score = kri_vals[0]
                    if len(kri_vals) >= 3:
                        recent = sum(kri_vals[:2]) / 2
                        older = sum(kri_vals[2:min(4, len(kri_vals))]) / len(kri_vals[2:min(4, len(kri_vals))])
                        kri_trend = recent - older

                # Position change (settle → finish) from last start
                if fh and isinstance(fh[0], dict):
                    settle = fh[0].get("settled")
                    finish = fh[0].get("position") or fh[0].get("pos")
                    if settle is not None and finish is not None:
                        try:
                            position_change_val = float(int(settle) - int(finish))
                        except (ValueError, TypeError):
                            pass

                    # Margin last start
                    margin_raw = fh[0].get("margin")
                    if margin_raw is not None:
                        try:
                            margin_last_val = float(margin_raw)
                        except (ValueError, TypeError):
                            pass
                    if fh[0].get("position") == 1 or fh[0].get("pos") == 1:
                        margin_last_val = 0.0
        except (ValueError, TypeError):
            pass

    # Weight change (today vs last start)
    weight_change_val = nan
    if weight and fh_raw:
        try:
            fh2 = _json_v9.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
            if isinstance(fh2, list) and fh2 and isinstance(fh2[0], dict):
                last_wt = fh2[0].get("weight")
                if last_wt is not None:
                    weight_change_val = weight - float(last_wt)
        except (ValueError, TypeError):
            pass

    # Distance change (today vs last start)
    distance_change_val = nan
    if distance and fh_raw:
        try:
            fh3 = _json_v9.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
            if isinstance(fh3, list) and fh3 and isinstance(fh3[0], dict):
                last_dist = fh3[0].get("distance")
                if last_dist is not None:
                    distance_change_val = float(distance) - float(last_dist)
        except (ValueError, TypeError):
            pass

    # Condition-specific career SR (on today's going)
    condition_record_sr = nan
    tc_raw_v9 = str(_get(meeting, "track_condition") or "").upper()
    if tc_raw_v9.startswith("G"):
        cond_rec = _get(runner, "good_track_stats") or _get(runner, "GoodRecord")
    elif tc_raw_v9.startswith("S"):
        cond_rec = _get(runner, "soft_track_stats") or _get(runner, "SoftRecord")
    elif tc_raw_v9.startswith("H"):
        cond_rec = _get(runner, "heavy_track_stats") or _get(runner, "HeavyRecord")
    else:
        cond_rec = None
    if cond_rec:
        csr, cstarts = _sr_from_stats(cond_rec) if isinstance(cond_rec, str) else (None, 0)
        if isinstance(cond_rec, dict):
            cstarts = cond_rec.get("Starts", 0)
            if cstarts and cstarts > 0:
                csr = cond_rec.get("Firsts", 0) / cstarts
        if csr is not None and csr > 0:
            condition_record_sr = float(csr)

    # Combo A2E (trainer+jockey together)
    combo_a2e_val = nan
    combo_raw = _get(runner, "TrainerJockeyA2E_Career")
    if not combo_raw:
        # Try from jockey_stats JSON
        jstats = _get(runner, "jockey_stats")
        if jstats:
            try:
                jj = _json_v9.loads(jstats) if isinstance(jstats, str) else jstats
                if isinstance(jj, dict) and "combo_career" in jj:
                    combo_raw = jj["combo_career"]
            except (ValueError, TypeError):
                pass
    if isinstance(combo_raw, dict):
        a2e = combo_raw.get("a2e") or combo_raw.get("A2E")
        if a2e is not None:
            try:
                combo_a2e_val = float(a2e)
            except (ValueError, TypeError):
                pass

    fvec = [
        market_prob,
        career_win_pct, career_place_pct, _f(career_starts),
        _f(td_sr), float(td_starts),
        _f(dist_sr), float(dist_starts),
        _f(trk_sr), float(trk_starts),
        _f(good_sr), float(good_starts),
        _f(soft_sr), float(soft_starts),
        _f(heavy_sr), float(heavy_starts),
        _f(firm_sr), float(firm_starts),
        _f(synth_sr), float(synth_starts),
        _f(fu_sr), float(fu_starts),
        _f(su_sr), float(su_starts),
        _f(l5_score), _f(l5_wins), _f(l5_places),
        _f(form_trend_val), _f(place_vs_market),
        _f(prize_per_start), _f(handicap), _f(avg_margin),
        _f(class_diff),
        _f(days_since), _f(settle),
        _f(barrier_relative), float(barrier) if barrier else nan,
        _f(jockey_career_sr), _f(jockey_career_a2e), _f(jockey_career_pot),
        _f(jockey_career_runners), _f(jockey_l100_sr),
        _f(trainer_career_sr), _f(trainer_career_a2e),
        _f(trainer_career_pot), _f(trainer_l100_sr),
        _f(combo_career_sr), _f(combo_career_runners), _f(combo_l100_sr),
        _f(weight), _f(weight_diff), _f(age),
        is_gelding, is_mare,
        _f(price_move_pct),
        _f(group_starts), _f(group_sr),
        _f(dist_place), _f(trk_place),
        float(field_size), float(distance),
        # ── v4 features ──
        is_leader, is_staying, last_start_won,
        is_australia, is_hong_kong, is_new_zealand,
        _f(head_to_head_wins), _f(field_beaten_pct),
        rail_bias_score,
        # Pace / speed map (4)
        _f(pf_mf), _f(pf_sr), _f(pf_jf), speed_map_encoded,
        # Weather (4)
        _f(weather_rain_prob), _f(weather_wind_speed), _f(weather_temp), _f(weather_humidity),
        # Gear (2)
        has_gear_change, blinkers_on,
        # Campaign (2)
        _f(runs_this_prep), _f(campaign_wr),
        # Class/track specialist (2)
        _f(class_win_rate), track_specialist,
        # Stewards/excuses (2)
        has_stewards_issue, has_excuse,
        # Age×sex + flucs (2)
        _f(peak_age_sex_score), _f(flucs_direction),
        # ── v6: Context buckets (4) + interactions (10) ──
        dist_bkt, tc_bkt, cls_bkt, vt_code,
    ] + interactions + [
        # ── v8: KASH (DEPRECATED — NaN) ──
        kash_wp_implied, kash_early_spd, kash_late_spd, kash_consensus,
        # ── v9: Signal-driven features (8) ──
        _f(kri_score), _f(kri_trend), _f(position_change_val), _f(weight_change_val),
        _f(distance_change_val), _f(margin_last_val), _f(condition_record_sr), _f(combo_a2e_val),
    ]

    # S1: NaN overrides removed — v5 model trained with these features populated.

    return fvec


def extract_features_batch(
    runners: list, race: Any, meeting: Any,
) -> np.ndarray:
    """Extract features for all active runners in a race.

    Returns:
        np.ndarray of shape (n_runners, NUM_FEATURES) with NaN for missing values.
    """
    active = [r for r in runners if not _get(r, "scratched", False)]
    if not active:
        return np.empty((0, NUM_FEATURES))

    field_size = len(active)

    # Calculate avg weight for weight_diff feature
    weights = []
    for r in active:
        w = _safe_float(_get(r, "weight"))
        if w and w > 40:
            weights.append(w)
    avg_weight = statistics.mean(weights) if weights else 0.0

    # Calculate overround for market_prob normalization
    overround = 0.0
    for r in active:
        odds_vals = []
        for attr in ("odds_betfair", "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes"):
            v = _safe_float(_get(r, attr))
            if v and v > 1.0:
                odds_vals.append(v)
        if not odds_vals:
            v = _safe_float(_get(r, "current_odds"))
            if v and v > 1.0:
                odds_vals = [v]
        if odds_vals:
            median = sorted(odds_vals)[len(odds_vals) // 2]
            overround += 1.0 / median
    overround = overround if overround > 0 else 1.0

    # Collect field horse names for head-to-head feature (S21)
    field_names = set()
    for r in active:
        name = _get(r, "horse_name")
        if name:
            field_names.add(str(name).strip().upper())

    rows = []
    for r in active:
        row = extract_features_from_runner(
            r, race, meeting, field_size, avg_weight, overround,
            _field_names=field_names,
        )
        rows.append(row)

    return np.array(rows, dtype=np.float64)
