"""Feature extraction for LightGBM probability models.

Two extraction paths producing an identical 58-feature vector:
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
    # Recent form (3)
    "last5_score", "last5_wins", "last5_places",
    # Class & fitness (3)
    "prize_per_start", "handicap_rating", "avg_margin",
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
    # Race context (2)
    "field_size", "distance",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 56


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
    """Return place rate from a stats string."""
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
    # Proform StrikeRate is already a percentage (e.g. 14.15 = 14.15%)
    # Live DB strike_rate is also a percentage (e.g. 14.15 = 14.15%)
    # Training extract_features_proform passes raw StrikeRate through _a2e_field
    # which returns the raw value — so keep consistent: return the raw percentage value
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


# ──────────────────────────────────────────────
# Extraction from backtest.db rows (training)
# ──────────────────────────────────────────────

def extract_features_from_db_row(
    runner: dict, race: dict, meeting: dict, field_size: int, avg_weight: float,
) -> list:
    """Extract 58-feature vector from backtest.db row dicts.

    Args:
        runner: Dict from runners table
        race: Dict from races table
        meeting: Dict from meetings table
        field_size: Number of non-scratched runners in the race
        avg_weight: Average weight carried in the race

    Returns:
        List of 58 float/NaN values in FEATURE_NAMES order
    """
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
    # firm/synthetic not in backtest.db — NaN
    firm_sr, firm_starts = nan, 0
    synth_sr, synth_starts = nan, 0

    fu_sr, fu_starts = _sr_from_stats(runner.get("first_up_stats"))
    su_sr, su_starts = _sr_from_stats(runner.get("second_up_stats"))

    # Last 5 form
    last_five = runner.get("last_five") or runner.get("form", "")
    l5_score = _score_last_five(last_five)
    l5_wins = _count_last5(last_five, {1})
    l5_places = _count_last5(last_five, {1, 2, 3})

    # Class & fitness
    prize = runner.get("career_prize_money")
    c_starts = career[0] if career else None
    prize_per_start = prize / c_starts if prize and c_starts and c_starts > 0 else nan

    handicap = _safe_float(runner.get("handicap_rating"))
    avg_margin = nan  # Not directly available in backtest.db

    # Pace
    days_since = _safe_float(runner.get("days_since_last_run"))
    settle = _safe_float(runner.get("pf_settle"))

    # Barrier
    barrier = runner.get("barrier") or 0
    barrier_relative = (barrier - 1) / (field_size - 1) if barrier and field_size > 1 else nan

    # Jockey/trainer stats — backtest.db has aggregated stats strings
    j_sr, j_starts = _sr_from_stats(runner.get("jockey_stats"))
    t_sr, t_starts = _sr_from_stats(runner.get("trainer_stats"))
    # A2E/PoT/L100/combo not in backtest.db
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

    # Race context
    distance = race.get("distance") or 1400

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
        _f(prize_per_start), _f(handicap), _f(avg_margin),
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
        float(field_size), float(distance),
    ]


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
# Extraction from live Runner ORM (inference)
# ──────────────────────────────────────────────

def extract_features_from_runner(
    runner: Any, race: Any, meeting: Any,
    field_size: int, avg_weight: float, overround: float = 1.0,
) -> list:
    """Extract 58-feature vector from live Runner ORM object.

    Mirrors extract_features_from_db_row() but reads ORM attributes.
    Missing data (e.g. Proform-specific stats) passed as NaN.
    """
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

    # Class & fitness
    prize = _safe_float(_get(runner, "career_prize_money"))
    c_starts = career[0] if career else None
    prize_per_start = prize / c_starts if prize and c_starts and c_starts > 0 else nan

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

    # Race context
    distance = _get(race, "distance") or 1400

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
        _f(prize_per_start), _f(handicap), _f(avg_margin),
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
        float(field_size), float(distance),
    ]


def extract_features_batch(
    runners: list, race: Any, meeting: Any,
) -> np.ndarray:
    """Extract features for all active runners in a race.

    Returns:
        np.ndarray of shape (n_runners, 58) with NaN for missing values.
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

    rows = []
    for r in active:
        row = extract_features_from_runner(r, race, meeting, field_size, avg_weight, overround)
        rows.append(row)

    return np.array(rows, dtype=np.float64)
