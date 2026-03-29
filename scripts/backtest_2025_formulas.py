"""Backtest old (pure prob) vs new (prob*value) ranking on 2025 Proform data.

Uses Proform ratings (WeightClassPrice, TimePrice) as independent probability
estimates (like our engine), and SP as market odds.
- Engine probability = 1/PF_rating (normalized)
- Market probability = 1/SP (normalized)
- Value = engine_prob / market_prob (>1 = PF thinks horse is better than market)

For months without ratings data, falls back to meetings.json form-based signals.
"""
import json
import os
import sys
from collections import defaultdict

BASE_2025 = "D:/Punty/DatafromProform/2025"
MONTHS = [
    "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

FORMULAS = {
    "pure_prob": (0, 0),
    "clamp_95_105": (0.95, 1.05),
    "clamp_90_110": (0.90, 1.10),
    "clamp_85_115": (0.85, 1.15),
    "clamp_85_130": (0.85, 1.30),
    "clamp_80_150": (0.80, 1.50),
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def process_month(month):
    results_data = load_json(os.path.join(BASE_2025, month, "results.json"))
    meetings_data = load_json(os.path.join(BASE_2025, month, "meetings.json"))

    if not results_data or not meetings_data:
        return {}, 0

    # Build results lookup: {meeting_id: {race_num: {tab: {position, sp}}}}
    results_map = {}
    for m in results_data:
        mid = m.get("MeetingId")
        if not mid:
            continue
        meeting_results = {}
        for race in m.get("RaceResults", []):
            rn = race.get("RaceNumber")
            if not rn:
                continue
            runners = {}
            for r in race.get("Runners", []):
                tab = r.get("TabNo")
                pos = r.get("Position")
                sp = r.get("Price", 0)
                if tab and pos:
                    runners[tab] = {"position": pos, "sp": sp or 0}
            if runners:
                meeting_results[rn] = runners
        results_map[int(mid)] = meeting_results

    # Build meetings lookup
    stats = {name: defaultdict(float) for name in FORMULAS}

    total_races = 0
    total_with_value = 0

    for meeting in meetings_data:
        mid = int(meeting.get("MeetingId", 0))
        track = meeting.get("Track", {})
        country = track.get("Country", "")
        if country != "AUS":
            continue

        meeting_date = meeting.get("MeetingDate", "")[:10]
        if meeting_date < "2025-01-01":
            continue

        meeting_results = results_map.get(mid, {})

        for race in meeting.get("Races", []):
            rn = race.get("Number")
            if not rn or rn not in meeting_results:
                continue

            race_results = meeting_results[rn]
            runners = race.get("Runners", [])

            candidates = []
            for runner in runners:
                tab = runner.get("TabNo")
                sp = runner.get("PriceSP", 0)
                if not tab or not sp or sp <= 1.0:
                    continue

                result = race_results.get(tab)
                if not result:
                    continue

                # Build independent probability from form data
                # Use career stats, track/distance records as proxy for our engine
                career_starts = runner.get("CareerStarts", 0) or 0
                career_wins = runner.get("CareerWins", 0) or 0
                career_seconds = runner.get("CareerSeconds", 0) or 0
                career_thirds = runner.get("CareerThirds", 0) or 0

                td_rec = runner.get("TrackDistRecord", {}) or {}
                td_starts = td_rec.get("Starts", 0) or 0
                td_wins = td_rec.get("Firsts", 0) or 0

                dist_rec = runner.get("DistanceRecord", {}) or {}
                dist_starts = dist_rec.get("Starts", 0) or 0
                dist_wins = dist_rec.get("Firsts", 0) or 0

                track_rec = runner.get("TrackRecord", {}) or {}
                track_starts = track_rec.get("Starts", 0) or 0
                track_wins = track_rec.get("Firsts", 0) or 0

                # Win rate composite: weight career > distance > track > track+dist
                win_rate = 0.0
                samples = 0
                if career_starts >= 3:
                    win_rate += (career_wins / career_starts) * 3
                    samples += 3
                if dist_starts >= 2:
                    win_rate += (dist_wins / dist_starts) * 2
                    samples += 2
                if track_starts >= 2:
                    win_rate += (track_wins / track_starts) * 1
                    samples += 1
                if samples > 0:
                    win_rate /= samples
                else:
                    win_rate = 0.05  # default for no data

                # Blend with market (SP-implied) — 50% form, 50% market
                market_prob_raw = 1.0 / sp
                form_prob = max(0.01, min(win_rate, 0.8))
                engine_prob = 0.5 * form_prob + 0.5 * market_prob_raw

                candidates.append({
                    "tab": tab,
                    "sp": sp,
                    "position": result["position"],
                    "name": runner.get("Name", ""),
                    "engine_prob": engine_prob,
                    "market_prob_raw": market_prob_raw,
                })

            if len(candidates) < 4:
                continue

            # Normalize engine probabilities
            total_engine = sum(c["engine_prob"] for c in candidates)
            total_market = sum(c["market_prob_raw"] for c in candidates)
            for c in candidates:
                c["engine_prob_norm"] = c["engine_prob"] / total_engine
                c["market_prob_norm"] = c["market_prob_raw"] / total_market

            # Calculate value rating: engine's fair odds / market odds
            # = (1/engine_prob_norm) / sp... but more intuitively:
            # value = engine_prob_norm / market_prob_norm
            for c in candidates:
                c["value_rating"] = c["engine_prob_norm"] / c["market_prob_norm"]

            has_value_diff = any(
                abs(c["value_rating"] - 1.0) > 0.05 for c in candidates
            )
            if has_value_diff:
                total_with_value += 1

            total_races += 1

            for name, (floor, cap) in FORMULAS.items():
                if floor == 0 and cap == 0:
                    ranked = sorted(candidates, key=lambda c: c["engine_prob_norm"], reverse=True)
                else:
                    def _score(c, f=floor, cp=cap):
                        v = max(f, min(c["value_rating"], cp))
                        return c["engine_prob_norm"] * v
                    ranked = sorted(candidates, key=_score, reverse=True)

                top1 = ranked[0]
                top3 = ranked[:3]

                stats[name]["races"] += 1

                if top1["position"] == 1:
                    stats[name]["top1_wins"] += 1
                    stats[name]["top1_win_pnl"] += top1["sp"] * 10 - 10
                else:
                    stats[name]["top1_win_pnl"] -= 10

                if top1["position"] <= 3:
                    stats[name]["top1_places"] += 1
                    place_div = max(1.1, top1["sp"] / 3.0)
                    stats[name]["top1_place_pnl"] += place_div * 10 - 10
                else:
                    stats[name]["top1_place_pnl"] -= 10

                if any(c["position"] == 1 for c in top3):
                    stats[name]["top3_contain_winner"] += 1

                stats[name]["top1_odds_sum"] += top1["sp"]

                # Track by odds band
                sp = top1["sp"]
                if sp < 2.40:
                    band = "<$2.40"
                elif sp < 4:
                    band = "$2.40-$4"
                elif sp < 6:
                    band = "$4-$6"
                elif sp < 10:
                    band = "$6-$10"
                elif sp < 20:
                    band = "$10-$20"
                else:
                    band = "$20+"
                stats[name]["band_%s_cnt" % band] += 1

    return stats, total_races


def main():
    all_stats = {name: defaultdict(float) for name in FORMULAS}
    grand_total = 0

    for month in MONTHS:
        print("Processing %s..." % month, file=sys.stderr)
        stats, total = process_month(month)
        grand_total += total
        for name in FORMULAS:
            for key in stats[name]:
                all_stats[name][key] += stats[name][key]

    print()
    print("=" * 110)
    print("2025 BACKTEST: Formula Comparison (%d races, Feb-Dec 2025 Australian Thoroughbreds)" % grand_total)
    print("Engine = 50%% form-based win rate + 50%% market (SP)")
    print("Value = engine_prob / market_prob (>1 = engine rates higher than market)")
    print("=" * 110)
    print()

    header = "%-20s %7s %7s %7s %7s %7s %10s %10s %10s" % (
        "Formula", "#1 Win%", "#1 Pl%", "Top3 W%", "Avg Odd", "Races",
        "Win PnL", "Place PnL", "Combined"
    )
    print(header)
    print("-" * len(header))

    for name in ["pure_prob", "clamp_95_105", "clamp_90_110", "clamp_85_115", "clamp_85_130", "clamp_80_150"]:
        s = all_stats[name]
        races = s["races"]
        if races == 0:
            continue
        print("%-20s %6.1f%% %6.1f%% %6.1f%% $%5.2f %7d $%+9.0f $%+9.0f $%+9.0f" % (
            name,
            s["top1_wins"] / races * 100,
            s["top1_places"] / races * 100,
            s["top3_contain_winner"] / races * 100,
            s["top1_odds_sum"] / races,
            races,
            s["top1_win_pnl"],
            s["top1_place_pnl"],
            s["top1_win_pnl"] + s["top1_place_pnl"],
        ))

    print()
    print("ODDS BAND DISTRIBUTION OF #1 PICK:")
    bands = ["<$2.40", "$2.40-$4", "$4-$6", "$6-$10", "$10-$20", "$20+"]
    header2 = "%-20s" + "".join(["%10s" % b for b in bands])
    print(header2)
    print("-" * (20 + 10 * len(bands)))
    for name in ["pure_prob", "clamp_85_130", "clamp_80_150"]:
        s = all_stats[name]
        row = "%-20s" % name
        for b in bands:
            cnt = s.get("band_%s_cnt" % b, 0)
            row += "%10d" % cnt
        print(row)


if __name__ == "__main__":
    main()
