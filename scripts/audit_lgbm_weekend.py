#!/usr/bin/env python3
"""Audit LightGBM vs Weighted Engine on weekend race data.

Compares LightGBM predictions against actual results for settled weekend
races, alongside what the weighted engine produced (released to public).

Usage:
    python scripts/audit_lgbm_weekend.py

Run on the server where punty.db lives:
    cd /opt/puntyai && source venv/bin/activate && python3 scripts/audit_lgbm_weekend.py
"""

import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from punty.ml.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    _f,
    _safe_float,
    _score_last_five,
    _count_last5,
    _parse_stats,
    _sr_from_stats,
)

DB_PATH = ROOT / "data" / "punty.db"
DATE_FROM = "2026-02-21"
DATE_TO = "2026-02-22"  # Sat + Sun


def load_weekend_data(conn):
    """Load all settled weekend races with runners."""
    meetings = {}
    for row in conn.execute(
        "SELECT * FROM meetings WHERE date >= ? AND date <= ? ORDER BY date, venue",
        (DATE_FROM, DATE_TO),
    ).fetchall():
        meetings[row["id"]] = dict(row)

    races = {}
    for row in conn.execute(
        """SELECT r.* FROM races r JOIN meetings m ON r.meeting_id = m.id
           WHERE m.date >= ? AND m.date <= ?
           AND r.results_status IN ('Paying', 'Closed', 'Final')
           ORDER BY m.date, m.venue, r.race_number""",
        (DATE_FROM, DATE_TO),
    ).fetchall():
        races[row["id"]] = dict(row)

    runners = defaultdict(list)
    for row in conn.execute(
        """SELECT ru.* FROM runners ru
           JOIN races r ON ru.race_id = r.id
           JOIN meetings m ON r.meeting_id = m.id
           WHERE m.date >= ? AND m.date <= ?
           AND ru.scratched = 0 AND ru.finish_position IS NOT NULL
           ORDER BY r.id, ru.saddlecloth""",
        (DATE_FROM, DATE_TO),
    ).fetchall():
        runners[row["race_id"]].append(dict(row))

    picks = defaultdict(list)
    for row in conn.execute(
        """SELECT p.* FROM picks p
           JOIN content c ON p.content_id = c.id
           JOIN meetings m ON c.meeting_id = m.id
           WHERE m.date >= ? AND m.date <= ?
           AND p.settled = 1
           AND c.status IN ('approved', 'sent')
           ORDER BY m.date, p.race_number""",
        (DATE_FROM, DATE_TO),
    ).fetchall():
        key = f"{row['meeting_id']}-r{row['race_number']}"
        picks[key].append(dict(row))

    return meetings, races, runners, picks


def parse_json_stats(val):
    """Parse stats that might be JSON dict or string format."""
    if not val:
        return None, 0
    if isinstance(val, str):
        try:
            d = json.loads(val)
            if isinstance(d, dict):
                starts = d.get("starts", 0)
                wins = d.get("wins", 0)
                return wins / starts if starts > 0 else None, starts
        except (json.JSONDecodeError, ValueError):
            return _sr_from_stats(val)
    return None, 0


def parse_jockey_stats(val):
    """Parse rich jockey stats JSON."""
    if not val:
        return {}
    try:
        d = json.loads(val) if isinstance(val, str) else val
        if not isinstance(d, dict):
            return {}
        result = {}
        career = d.get("career", {})
        if career:
            result["jockey_career_sr"] = career.get("strike_rate", 0) / 100 if career.get("strike_rate") else None
            result["jockey_career_a2e"] = career.get("a2e")
            result["jockey_career_pot"] = career.get("pot")
            result["jockey_career_runners"] = career.get("runners", 0)
        l100 = d.get("last100", {})
        if l100:
            result["jockey_l100_sr"] = l100.get("strike_rate", 0) / 100 if l100.get("strike_rate") else None
        combo = d.get("combo_career", {})
        if combo:
            result["combo_career_sr"] = combo.get("strike_rate", 0) / 100 if combo.get("strike_rate") else None
            result["combo_career_runners"] = combo.get("runners", 0)
        combo_l100 = d.get("combo_last100", {})
        if combo_l100:
            result["combo_l100_sr"] = combo_l100.get("strike_rate", 0) / 100 if combo_l100.get("strike_rate") else None
        return result
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_trainer_stats(val):
    """Parse rich trainer stats JSON."""
    if not val:
        return {}
    try:
        d = json.loads(val) if isinstance(val, str) else val
        if not isinstance(d, dict):
            return {}
        result = {}
        career = d.get("career", {})
        if career:
            result["trainer_career_sr"] = career.get("strike_rate", 0) / 100 if career.get("strike_rate") else None
            result["trainer_career_a2e"] = career.get("a2e")
            result["trainer_career_pot"] = career.get("pot")
        l100 = d.get("last100", {})
        if l100:
            result["trainer_l100_sr"] = l100.get("strike_rate", 0) / 100 if l100.get("strike_rate") else None
        return result
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_features_live(runner, race, meeting, field_size, avg_weight, overround):
    """Extract features from live DB runner dict (handles JSON stats)."""
    nan = float("nan")

    # Market
    odds_vals = []
    for key in ("odds_betfair", "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes"):
        v = _safe_float(runner.get(key))
        if v and v > 1.0:
            odds_vals.append(v)
    if not odds_vals:
        v = _safe_float(runner.get("current_odds"))
        if v and v > 1.0:
            odds_vals = [v]
    if odds_vals:
        median_odds = sorted(odds_vals)[len(odds_vals) // 2]
        raw_implied = 1.0 / median_odds
        market_prob = raw_implied / overround if overround > 0 else raw_implied
    else:
        market_prob = nan

    # Career
    career = _parse_stats(runner.get("career_record"))
    if career and career[0] > 0:
        cs, cw, c2, c3 = career
        career_win_pct = cw / cs
        career_place_pct = (cw + c2 + c3) / cs
        career_starts = float(cs)
    else:
        career_win_pct = nan
        career_place_pct = nan
        career_starts = nan

    # Stats fields — handle both JSON dict and string formats
    td_sr, td_starts = parse_json_stats(runner.get("track_dist_stats"))
    dist_sr, dist_starts = parse_json_stats(runner.get("distance_stats"))
    trk_sr, trk_starts = parse_json_stats(runner.get("track_stats"))
    good_sr, good_starts = parse_json_stats(runner.get("good_track_stats"))
    soft_sr, soft_starts = parse_json_stats(runner.get("soft_track_stats"))
    heavy_sr, heavy_starts = parse_json_stats(runner.get("heavy_track_stats"))
    fu_sr, fu_starts = parse_json_stats(runner.get("first_up_stats"))
    su_sr, su_starts = parse_json_stats(runner.get("second_up_stats"))
    firm_sr, firm_starts = nan, 0
    synth_sr, synth_starts = nan, 0

    # Last 5
    last_five = runner.get("last_five") or runner.get("form", "")
    l5_score = _score_last_five(last_five)
    l5_wins = _count_last5(last_five, {1})
    l5_places = _count_last5(last_five, {1, 2, 3})

    # Class
    prize = _safe_float(runner.get("career_prize_money"))
    c_starts_raw = career[0] if career else None
    prize_per_start = prize / c_starts_raw if prize and c_starts_raw and c_starts_raw > 0 else nan
    handicap = _safe_float(runner.get("handicap_rating"))

    # Pace
    days_since = _safe_float(runner.get("days_since_last_run"))
    settle = _safe_float(runner.get("pf_settle"))

    # Barrier
    barrier = runner.get("barrier") or 0
    barrier_relative = (barrier - 1) / (field_size - 1) if barrier and field_size > 1 else nan

    # Jockey/trainer — parse rich JSON
    js = parse_jockey_stats(runner.get("jockey_stats"))
    ts = parse_trainer_stats(runner.get("trainer_stats"))

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
    price_move_pct = (opening - current) / opening if opening and current and opening > 1 and current > 1 else nan

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
        _f(prize_per_start), _f(handicap), nan,  # avg_margin not available
        _f(days_since), _f(settle),
        _f(barrier_relative), float(barrier) if barrier else nan,
        _f(js.get("jockey_career_sr")), _f(js.get("jockey_career_a2e")),
        _f(js.get("jockey_career_pot")), _f(js.get("jockey_career_runners")),
        _f(js.get("jockey_l100_sr")),
        _f(ts.get("trainer_career_sr")), _f(ts.get("trainer_career_a2e")),
        _f(ts.get("trainer_career_pot")), _f(ts.get("trainer_l100_sr")),
        _f(js.get("combo_career_sr")), _f(js.get("combo_career_runners")),
        _f(js.get("combo_l100_sr")),
        _f(weight), _f(weight_diff), _f(age),
        is_gelding, is_mare,
        _f(price_move_pct),
        nan, nan,  # group_starts, group_sr not available
        float(field_size), float(race.get("distance") or 1400),
    ]


def run_lgbm_on_race(lgbm_win, lgbm_place, race_runners, race, meeting):
    """Run LightGBM on a race, return runner_id → (win_prob, place_prob)."""
    field_size = len(race_runners)

    weights = [r["weight"] for r in race_runners if r.get("weight") and r["weight"] > 40]
    avg_weight = statistics.mean(weights) if weights else 0.0

    overround = 0.0
    for r in race_runners:
        odds_vals = []
        for key in ("odds_betfair", "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes"):
            v = _safe_float(r.get(key))
            if v and v > 1.0:
                odds_vals.append(v)
        if not odds_vals:
            v = _safe_float(r.get("current_odds"))
            if v and v > 1.0:
                odds_vals = [v]
        if odds_vals:
            overround += 1.0 / sorted(odds_vals)[len(odds_vals) // 2]
    overround = overround if overround > 0 else 1.0

    rows = []
    ids = []
    for r in race_runners:
        feats = extract_features_live(r, race, meeting, field_size, avg_weight, overround)
        rows.append(feats)
        ids.append(r["id"])

    X = np.array(rows, dtype=np.float64)
    win_raw = lgbm_win.predict(X)
    place_raw = lgbm_place.predict(X)

    # Normalize
    win_total = sum(win_raw)
    place_count = 2 if field_size <= 7 else 3
    place_total = sum(place_raw)

    results = {}
    for i, rid in enumerate(ids):
        wp = win_raw[i] / win_total if win_total > 0 else 1.0 / field_size
        pp = min(0.95, place_raw[i] / place_total * place_count) if place_total > 0 else 0.3
        results[rid] = (wp, pp)
    return results


def simulate_picks(lgbm_preds, race_runners, race, meeting):
    """Simulate what LightGBM picks would have been for a race.

    Returns list of simulated pick dicts with hypothetical PnL.
    """
    field_size = len(race_runners)
    # Sort by win prob descending
    ranked = sorted(race_runners, key=lambda r: lgbm_preds.get(r["id"], (0, 0))[0], reverse=True)

    sim_picks = []
    for rank, runner in enumerate(ranked[:4], 1):
        rid = runner["id"]
        wp, pp = lgbm_preds.get(rid, (0, 0))
        odds = _safe_float(runner.get("current_odds")) or 0
        pos = runner.get("finish_position")
        win_div = _safe_float(runner.get("win_dividend")) or 0
        place_div = _safe_float(runner.get("place_dividend")) or 0

        # Assign bet type using same rules as production
        if rank == 1:
            if odds <= 2.50:
                bet_type = "win"
            elif odds <= 4.00:
                bet_type = "each_way"
            else:
                bet_type = "saver_win"
        elif rank == 2:
            if odds <= 3.00:
                bet_type = "place"
            else:
                bet_type = "place"
        elif rank == 3:
            bet_type = "place"
        else:  # rank 4 = roughie
            bet_type = "place"

        # Stake allocation (simplified $20 pool)
        stakes = {1: 8.0, 2: 5.0, 3: 4.0, 4: 3.0}
        stake = stakes[rank]

        # Settlement
        hit = False
        pnl = -stake
        if bet_type == "win" or bet_type == "saver_win":
            if pos == 1:
                pnl = win_div * stake - stake
                hit = True
        elif bet_type == "place":
            place_cutoff = 2 if field_size <= 7 else 3
            if pos and pos <= place_cutoff and place_div > 0:
                pnl = place_div * stake - stake
                hit = True
        elif bet_type == "each_way":
            half = stake / 2
            place_cutoff = 2 if field_size <= 7 else 3
            if pos == 1:
                pnl = (win_div * half - half) + (place_div * half - half) if place_div > 0 else (win_div * half - half) + (-half)
                hit = True
            elif pos and pos <= place_cutoff and place_div > 0:
                pnl = (-half) + (place_div * half - half)
                hit = True

        sim_picks.append({
            "rank": rank,
            "horse": runner.get("horse_name", "?"),
            "saddlecloth": runner.get("saddlecloth"),
            "odds": odds,
            "win_prob": wp,
            "place_prob": pp,
            "finish": pos,
            "bet_type": bet_type,
            "stake": stake,
            "hit": hit,
            "pnl": round(pnl, 2),
        })

    return sim_picks


def main():
    import lightgbm as lgb

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    # Load models
    model_dir = ROOT / "punty" / "data"
    win_model = lgb.Booster(model_file=str(model_dir / "lgbm_win_model.txt"))
    place_model = lgb.Booster(model_file=str(model_dir / "lgbm_place_model.txt"))
    print(f"Loaded LightGBM models: win={win_model.num_trees()} trees, place={place_model.num_trees()} trees")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    meetings, races, runners, picks = load_weekend_data(conn)
    conn.close()

    print(f"\n{'='*80}")
    print(f"WEEKEND AUDIT: {DATE_FROM} to {DATE_TO}")
    print(f"  Meetings: {len(meetings)}")
    print(f"  Settled races: {len(races)}")
    print(f"  Runners with results: {sum(len(v) for v in runners.values())}")
    print(f"  Settled picks (public): {sum(len(v) for v in picks.values())}")
    print(f"{'='*80}")

    # ── Run all engines on every settled race ────────────────────────────
    from punty.probability import calculate_race_probabilities

    # Counters for 3 engines: lgbm, weighted (public), blend
    engines = ["lgbm", "weighted", "blend"]
    top1 = {e: 0 for e in engines}
    top3 = {e: 0 for e in engines}
    sim_pnl = {e: 0.0 for e in engines}
    sim_hits = {e: 0 for e in engines}
    sim_total = {e: 0 for e in engines}
    pub_total_pnl = 0.0
    pub_total_hits = 0
    pub_total_picks = 0
    total_races = 0

    race_details = []

    for race_id, race in races.items():
        race_runners = runners.get(race_id, [])
        if len(race_runners) < 2:
            continue

        meeting = meetings.get(race.get("meeting_id", ""), {})
        total_races += 1
        field_size = len(race_runners)

        # ── LightGBM predictions ──
        lgbm_preds = run_lgbm_on_race(win_model, place_model, race_runners, race, meeting)

        # ── Weighted engine predictions (same as production) ──
        weighted_results = calculate_race_probabilities(race_runners, race, meeting)
        weighted_preds = {}
        for rid, rp in weighted_results.items():
            weighted_preds[rid] = (rp.win_probability, rp.place_probability)

        # ── 50/50 Blend ──
        blend_preds = {}
        all_rids = set(list(lgbm_preds.keys()) + list(weighted_preds.keys()))
        baseline = 1.0 / field_size
        for rid in all_rids:
            lw, lp = lgbm_preds.get(rid, (baseline, baseline * 2))
            ww, wp = weighted_preds.get(rid, (baseline, baseline * 2))
            blend_preds[rid] = (0.5 * lw + 0.5 * ww, 0.5 * lp + 0.5 * wp)

        # Normalize blend win probs to sum to 1.0
        blend_win_total = sum(wp for wp, _ in blend_preds.values())
        if blend_win_total > 0:
            blend_preds = {rid: (wp / blend_win_total, pp) for rid, (wp, pp) in blend_preds.items()}
        # Normalize blend place probs
        place_count = 2 if field_size <= 7 else 3
        blend_place_total = sum(pp for _, pp in blend_preds.values())
        if blend_place_total > 0:
            blend_preds = {rid: (wp, min(0.95, pp / blend_place_total * place_count))
                           for rid, (wp, pp) in blend_preds.items()}

        # Find actual winner
        winner = None
        for r in race_runners:
            if r.get("finish_position") == 1:
                winner = r
                break
        if not winner:
            continue

        winner_id = winner["id"]
        winner_sc = winner.get("saddlecloth")

        # Rank each engine and check accuracy
        all_preds = {"lgbm": lgbm_preds, "weighted": weighted_preds, "blend": blend_preds}
        engine_ranks = {}
        for eng, preds in all_preds.items():
            ranked = sorted(race_runners, key=lambda r: preds.get(r["id"], (0, 0))[0], reverse=True)
            rank = next((i + 1 for i, r in enumerate(ranked) if r["id"] == winner_id), 99)
            engine_ranks[eng] = rank
            if rank == 1:
                top1[eng] += 1
            if rank <= 3:
                top3[eng] += 1

        # Simulate picks for each engine
        for eng, preds in all_preds.items():
            sp = simulate_picks(preds, race_runners, race, meeting)
            for pick in sp:
                sim_pnl[eng] += pick["pnl"]
                sim_hits[eng] += 1 if pick["hit"] else 0
                sim_total[eng] += 1

        # Published picks — actual public PnL
        race_key = f"{race.get('meeting_id', '')}-r{race.get('race_number', 0)}"
        race_picks = picks.get(race_key, [])
        sel_picks = [p for p in race_picks if p["pick_type"] == "selection"]

        pub_top1_won = False
        pub_top1_name = "N/A"
        if sel_picks:
            top_pick = min(sel_picks, key=lambda p: p.get("tip_rank") or 99)
            pub_top1_name = top_pick.get("horse_name", "?")
            pub_top1_won = top_pick.get("saddlecloth") == winner_sc

        for p in race_picks:
            if p.get("pnl") is not None:
                pub_total_pnl += p["pnl"]
                pub_total_hits += 1 if p.get("hit") else 0
                pub_total_picks += 1

        race_details.append({
            "race_id": race_id,
            "venue": meeting.get("venue", "?"),
            "race_num": race.get("race_number", 0),
            "date": meeting.get("date", ""),
            "field_size": field_size,
            "winner": winner.get("horse_name", "?"),
            "winner_odds": _safe_float(winner.get("current_odds")) or 0,
            "lgbm_rank": engine_ranks["lgbm"],
            "weighted_rank": engine_ranks["weighted"],
            "blend_rank": engine_ranks["blend"],
            "pub_top1": pub_top1_name,
            "pub_top1_won": pub_top1_won,
            "lgbm_pnl": sum(p["pnl"] for p in simulate_picks(lgbm_preds, race_runners, race, meeting)),
            "blend_pnl": sum(p["pnl"] for p in simulate_picks(blend_preds, race_runners, race, meeting)),
            "pub_pnl": sum(p["pnl"] for p in race_picks if p.get("pnl") is not None),
        })

    # ── Print Results ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"TOP-PICK ACCURACY (did #1 ranked runner win?)")
    print(f"{'='*80}")
    print(f"  {'Engine':20s} {'Top-1':>14s} {'Top-3':>14s} {'Races':>8s}")
    print(f"  {'─'*58}")
    for eng, label in [("lgbm", "LightGBM"), ("weighted", "Weighted"), ("blend", "50/50 Blend")]:
        pct1 = top1[eng] / total_races * 100 if total_races else 0
        pct3 = top3[eng] / total_races * 100 if total_races else 0
        print(f"  {label:20s} {top1[eng]:>4d}/{total_races} ({pct1:4.1f}%) {top3[eng]:>4d}/{total_races} ({pct3:4.1f}%) {total_races:>6d}")
    # Published picks comparison (uses tip_rank from actual content)
    pub_t1 = sum(1 for rd in race_details if rd["pub_top1_won"])
    pub_t3 = sum(1 for rd in race_details if rd.get("pub_top1_won"))  # approx
    print(f"  {'Published picks':20s} {pub_t1:>4d}/{total_races} ({pub_t1/total_races*100:4.1f}%)")
    blend_pct1 = top1["blend"] / total_races * 100 if total_races else 0
    best_single = max(top1["lgbm"], top1["weighted"])
    print(f"\n  Blend vs best single engine: {top1['blend'] - best_single:+d} top-1 picks")

    print(f"\n{'='*80}")
    print(f"P&L COMPARISON (simulated flat staking)")
    print(f"{'='*80}")
    print(f"  {'Engine':20s} {'Total PnL':>12s} {'Hits':>8s} {'Picks':>8s} {'Hit%':>8s} {'ROI':>8s}")
    print(f"  {'─'*66}")
    for eng, label in [("lgbm", "LightGBM (sim)"), ("weighted", "Weighted (sim)"), ("blend", "Blend 50/50 (sim)")]:
        t = sim_total[eng]
        h = sim_hits[eng]
        p = sim_pnl[eng]
        hp = h / t * 100 if t else 0
        roi = p / (t * 5.0) * 100 if t else 0
        print(f"  {label:20s} ${p:>+10.2f} {h:>6d} {t:>6d} {hp:>6.1f}% {roi:>+6.1f}%")
    pub_hit_pct = pub_total_hits / pub_total_picks * 100 if pub_total_picks else 0
    pub_roi = pub_total_pnl / (pub_total_picks * 5.0) * 100 if pub_total_picks else 0
    print(f"  {'Published (actual)':20s} ${pub_total_pnl:>+10.2f} {pub_total_hits:>6d} {pub_total_picks:>6d} {pub_hit_pct:>6.1f}% {pub_roi:>+6.1f}%")

    # ── Per-venue breakdown ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"PER-VENUE BREAKDOWN")
    print(f"{'='*80}")
    venue_stats = defaultdict(lambda: {"lgbm_top1": 0, "weighted_top1": 0, "blend_top1": 0,
                                         "races": 0, "lgbm_pnl": 0.0, "blend_pnl": 0.0, "pub_pnl": 0.0})
    for rd in race_details:
        v = rd["venue"]
        venue_stats[v]["races"] += 1
        if rd["lgbm_rank"] == 1:
            venue_stats[v]["lgbm_top1"] += 1
        if rd["weighted_rank"] == 1:
            venue_stats[v]["weighted_top1"] += 1
        if rd["blend_rank"] == 1:
            venue_stats[v]["blend_top1"] += 1
        venue_stats[v]["lgbm_pnl"] += rd["lgbm_pnl"]
        venue_stats[v]["blend_pnl"] += rd["blend_pnl"]
        venue_stats[v]["pub_pnl"] += rd["pub_pnl"]

    print(f"  {'Venue':18s} {'R':>3s} {'LGBM':>7s} {'Wt':>7s} {'Blend':>7s} {'LGBM PnL':>10s} {'Blend PnL':>10s} {'Pub PnL':>10s}")
    print(f"  {'─'*76}")
    for venue in sorted(venue_stats, key=lambda v: -venue_stats[v]["races"]):
        vs = venue_stats[venue]
        n = vs["races"]
        print(f"  {venue:18s} {n:>3d} {vs['lgbm_top1']:>3d}/{n:<3d} {vs['weighted_top1']:>3d}/{n:<3d} {vs['blend_top1']:>3d}/{n:<3d}"
              f" ${vs['lgbm_pnl']:>+8.2f} ${vs['blend_pnl']:>+8.2f} ${vs['pub_pnl']:>+8.2f}")

    # ── Per-day breakdown ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"PER-DAY BREAKDOWN")
    print(f"{'='*80}")
    day_stats = defaultdict(lambda: {"lgbm_top1": 0, "weighted_top1": 0, "blend_top1": 0,
                                       "races": 0, "lgbm_pnl": 0.0, "blend_pnl": 0.0, "pub_pnl": 0.0})
    for rd in race_details:
        d = rd["date"]
        day_stats[d]["races"] += 1
        if rd["lgbm_rank"] == 1:
            day_stats[d]["lgbm_top1"] += 1
        if rd["weighted_rank"] == 1:
            day_stats[d]["weighted_top1"] += 1
        if rd["blend_rank"] == 1:
            day_stats[d]["blend_top1"] += 1
        day_stats[d]["lgbm_pnl"] += rd["lgbm_pnl"]
        day_stats[d]["blend_pnl"] += rd["blend_pnl"]
        day_stats[d]["pub_pnl"] += rd["pub_pnl"]

    print(f"  {'Date':12s} {'R':>4s} {'LGBM Top1':>12s} {'Wt Top1':>12s} {'Blend Top1':>12s} {'Blend PnL':>10s} {'Pub PnL':>10s}")
    print(f"  {'─'*74}")
    for day in sorted(day_stats):
        ds = day_stats[day]
        n = ds["races"]
        l1 = ds["lgbm_top1"] / n * 100 if n else 0
        w1 = ds["weighted_top1"] / n * 100 if n else 0
        b1 = ds["blend_top1"] / n * 100 if n else 0
        print(f"  {day:12s} {n:>4d} {ds['lgbm_top1']:>4d} ({l1:4.1f}%) {ds['weighted_top1']:>4d} ({w1:4.1f}%) {ds['blend_top1']:>4d} ({b1:4.1f}%) ${ds['blend_pnl']:>+8.2f} ${ds['pub_pnl']:>+8.2f}")

    # ── Interesting disagreements ──────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"NOTABLE RACES: LightGBM vs Public picks disagreed on winner")
    print(f"{'='*80}")
    disagreements = [rd for rd in race_details if rd["lgbm_rank"] == 1 and not rd["pub_top1_won"]]
    disagreements.sort(key=lambda x: -abs(x["lgbm_pnl"] - x["pub_pnl"]))
    print(f"  {'Date':12s} {'Venue':15s} R#  {'Winner':22s} {'Odds':>6s} LGBM# Pub#1                  {'Sim PnL':>8s} {'Pub PnL':>8s}")
    print(f"  {'─'*105}")
    for rd in disagreements[:20]:
        print(f"  {rd['date']:12s} {rd['venue']:15s} R{rd['race_num']:<2d} {rd['winner']:22s} ${rd['winner_odds']:5.2f} "
              f"#{rd['lgbm_rank']}    {rd['pub_top1']:22s} ${rd['sim_pnl']:>+7.2f} ${rd['pub_pnl']:>+7.2f}")

    # ── Biggest misses ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"BIGGEST LGBM MISSES: Winner ranked low by LightGBM")
    print(f"{'='*80}")
    misses = sorted(race_details, key=lambda x: -x["lgbm_rank"])
    print(f"  {'Date':12s} {'Venue':15s} R#  {'Winner':22s} {'Odds':>6s} LGBM# {'Field':>5s}")
    print(f"  {'─'*75}")
    for rd in misses[:15]:
        if rd["lgbm_rank"] > 3:
            print(f"  {rd['date']:12s} {rd['venue']:15s} R{rd['race_num']:<2d} {rd['winner']:22s} ${rd['winner_odds']:5.2f} "
                  f"#{rd['lgbm_rank']:<3d}  {rd['field_size']:>3d}")

    # ── Odds-band analysis ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"TOP-1 ACCURACY BY WINNER'S ODDS BAND")
    print(f"{'='*80}")
    bands = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 15), (15, 100)]
    print(f"  {'Band':12s} {'Races':>6s} {'LGBM':>12s} {'Weighted':>12s} {'Blend':>12s}")
    print(f"  {'─'*56}")
    for lo, hi in bands:
        band_races = [rd for rd in race_details if lo <= rd["winner_odds"] < hi]
        if not band_races:
            continue
        n = len(band_races)
        l1 = sum(1 for rd in band_races if rd["lgbm_rank"] == 1)
        w1 = sum(1 for rd in band_races if rd["weighted_rank"] == 1)
        b1 = sum(1 for rd in band_races if rd["blend_rank"] == 1)
        print(f"  ${lo:<3d}-${hi:<3d}    {n:>6d} {l1:>4d} ({l1/n*100:4.1f}%) {w1:>4d} ({w1/n*100:4.1f}%) {b1:>4d} ({b1/n*100:4.1f}%)")


if __name__ == "__main__":
    main()
