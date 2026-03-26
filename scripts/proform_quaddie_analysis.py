"""
Proform Historical Data Analysis for Quaddie/Sequence Betting Patterns
Analyzes 13 months of Australian horse racing data from Proform.
"""

import json
import os
import sys
from collections import defaultdict
import statistics

BASE = "D:/Punty/DatafromProform"

MONTHS_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def get_month_dirs():
    dirs = []
    for year in ["2025", "2026"]:
        ypath = os.path.join(BASE, year)
        if not os.path.isdir(ypath):
            continue
        for month in MONTHS_ORDER:
            mpath = os.path.join(ypath, month)
            rpath = os.path.join(mpath, "results.json")
            mpath2 = os.path.join(mpath, "meetings.json")
            if os.path.isfile(rpath) and os.path.isfile(mpath2):
                dirs.append((year, month, mpath))
    return dirs


def load_all_data():
    all_meetings = []
    month_dirs = get_month_dirs()
    print(f"Found {len(month_dirs)} month directories")

    for year, month, mpath in month_dirs:
        rpath = os.path.join(mpath, "results.json")
        mpath2 = os.path.join(mpath, "meetings.json")

        with open(rpath, "r", encoding="utf-8") as f:
            results = json.load(f)
        with open(mpath2, "r", encoding="utf-8") as f:
            meetings = json.load(f)

        race_class_lookup = {}
        for m in meetings:
            mid = m.get("MeetingId")
            if m.get("Races"):
                for race in m["Races"]:
                    key = (str(mid), race.get("Number"))
                    race_class_lookup[key] = {
                        "class": race.get("RaceClass", ""),
                        "description": race.get("Description", ""),
                        "group": race.get("Group", ""),
                        "distance": race.get("Distance"),
                        "prize_money": race.get("PrizeMoney"),
                    }

        for meeting in results:
            mid = str(meeting.get("MeetingId"))
            track = meeting.get("Track", "")
            location = ""
            for m in meetings:
                if str(m.get("MeetingId")) == mid:
                    location = m.get("Track", {}).get("Location", "") if isinstance(m.get("Track"), dict) else ""
                    break

            meeting_date = meeting.get("MeetingDate", "")
            race_results = meeting.get("RaceResults", [])
            if not race_results:
                continue

            meeting_data = {
                "meeting_id": mid,
                "track": track,
                "location": location,
                "date": meeting_date,
                "races": [],
            }

            for race in race_results:
                race_num = race.get("RaceNumber")
                runners = race.get("Runners", [])
                if not runners:
                    continue

                class_info = race_class_lookup.get((mid, race_num), {})
                valid_runners = [r for r in runners if r.get("Position") and r["Position"] > 0]
                if not valid_runners:
                    continue

                priced_runners = [r for r in runners if r.get("Price") and r["Price"] > 0]
                priced_runners.sort(key=lambda x: x["Price"])
                valid_runners.sort(key=lambda x: x["Position"])

                winner = valid_runners[0] if valid_runners else None
                field_size = len([r for r in runners if r.get("Price") and r["Price"] > 0])

                race_data = {
                    "race_number": race_num,
                    "field_size": field_size,
                    "class": class_info.get("class", ""),
                    "description": class_info.get("description", ""),
                    "group": class_info.get("group", ""),
                    "distance": class_info.get("distance"),
                    "prize_money": class_info.get("prize_money"),
                    "track_condition": race.get("TrackConditionLabel", ""),
                    "winner": winner,
                    "runners_by_price": priced_runners,
                    "runners_by_position": valid_runners,
                }
                meeting_data["races"].append(race_data)

            if meeting_data["races"]:
                all_meetings.append(meeting_data)

    return all_meetings


def classify_race(race_data):
    cls = (race_data.get("class") or "").lower().strip().rstrip(";")
    desc = (race_data.get("description") or "").lower()
    group = (race_data.get("group") or "").strip()

    if group in ["1", "2", "3"]:
        return f"Group {group}"
    if "listed" in desc or group == "LR":
        return "Listed"
    if "maiden" in cls:
        return "Maiden"
    if "benchmark" in cls or "bm" in cls:
        return "Benchmark"
    if "class" in cls:
        return "Class Restricted"
    if "handicap" in desc and "open" in desc:
        return "Open Handicap"
    if "handicap" in desc:
        return "Handicap"
    if "set weight" in desc or "wfa" in desc or "weight for age" in desc:
        return "Set Weight/WFA"
    if cls:
        return "Other"
    return "Unknown"


def analyze_fav_by_race_number(all_meetings):
    print("\n" + "="*80)
    print("1. FAVOURITE WIN RATE BY RACE POSITION IN MEETING")
    print("="*80)

    stats = defaultdict(lambda: {"top1_wins": 0, "top2_wins": 0, "top3_wins": 0, "total": 0})

    for meeting in all_meetings:
        for race in meeting["races"]:
            rn = race["race_number"]
            if rn < 1 or rn > 12:
                continue
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced or len(priced) < 3:
                continue

            winner_tab = winner["TabNo"]
            stats[rn]["total"] += 1

            if priced[0]["TabNo"] == winner_tab:
                stats[rn]["top1_wins"] += 1
            if any(r["TabNo"] == winner_tab for r in priced[:2]):
                stats[rn]["top2_wins"] += 1
            if any(r["TabNo"] == winner_tab for r in priced[:3]):
                stats[rn]["top3_wins"] += 1

    print(f"\n{'Race':>6} {'Races':>7} {'Fav Win%':>10} {'Top2 Win%':>10} {'Top3 Win%':>10}")
    print("-" * 50)

    for rn in sorted(stats.keys()):
        s = stats[rn]
        t = s["total"]
        if t == 0:
            continue
        print(f"R{rn:>4}  {t:>7} {100*s['top1_wins']/t:>9.1f}% {100*s['top2_wins']/t:>9.1f}% {100*s['top3_wins']/t:>9.1f}%")

    total_races = sum(s["total"] for s in stats.values())
    total_fav = sum(s["top1_wins"] for s in stats.values())
    total_t2 = sum(s["top2_wins"] for s in stats.values())
    total_t3 = sum(s["top3_wins"] for s in stats.values())
    print(f"\n{'ALL':>6} {total_races:>7} {100*total_fav/total_races:>9.1f}% {100*total_t2/total_races:>9.1f}% {100*total_t3/total_races:>9.1f}%")

    early = [stats[r] for r in range(1, 5) if r in stats]
    late = [stats[r] for r in range(7, 13) if r in stats]
    early_total = sum(s["total"] for s in early)
    early_fav = sum(s["top1_wins"] for s in early)
    late_total = sum(s["total"] for s in late)
    late_fav = sum(s["top1_wins"] for s in late)
    if early_total and late_total:
        print(f"\nEarly (R1-R4): {100*early_fav/early_total:.1f}% fav win rate ({early_total} races)")
        print(f"Late  (R7-R12): {100*late_fav/late_total:.1f}% fav win rate ({late_total} races)")


def analyze_field_size(all_meetings):
    print("\n" + "="*80)
    print("2. FIELD SIZE IMPACT ON FAVOURITE WIN RATE")
    print("="*80)

    bands = {"5-7": (5, 7), "8-10": (8, 10), "11-13": (11, 13), "14+": (14, 99)}
    stats = {b: {"top1": 0, "top2": 0, "top3": 0, "total": 0} for b in bands}

    for meeting in all_meetings:
        for race in meeting["races"]:
            fs = race["field_size"]
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced or len(priced) < 3:
                continue
            winner_tab = winner["TabNo"]

            for band_name, (lo, hi) in bands.items():
                if lo <= fs <= hi:
                    stats[band_name]["total"] += 1
                    if priced[0]["TabNo"] == winner_tab:
                        stats[band_name]["top1"] += 1
                    if any(r["TabNo"] == winner_tab for r in priced[:2]):
                        stats[band_name]["top2"] += 1
                    if any(r["TabNo"] == winner_tab for r in priced[:3]):
                        stats[band_name]["top3"] += 1
                    break

    print(f"\n{'Field':>8} {'Races':>7} {'Fav Win%':>10} {'Top2 Win%':>10} {'Top3 Win%':>10}")
    print("-" * 50)
    for band_name in bands:
        s = stats[band_name]
        t = s["total"]
        if t == 0:
            continue
        print(f"{band_name:>8} {t:>7} {100*s['top1']/t:>9.1f}% {100*s['top2']/t:>9.1f}% {100*s['top3']/t:>9.1f}%")


def analyze_race_class(all_meetings):
    print("\n" + "="*80)
    print("3. RACE CLASS IMPACT ON FAVOURITE WIN RATE")
    print("="*80)

    stats = defaultdict(lambda: {"top1": 0, "top2": 0, "top3": 0, "total": 0})

    for meeting in all_meetings:
        for race in meeting["races"]:
            cls = classify_race(race)
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced or len(priced) < 3:
                continue
            winner_tab = winner["TabNo"]
            stats[cls]["total"] += 1
            if priced[0]["TabNo"] == winner_tab:
                stats[cls]["top1"] += 1
            if any(r["TabNo"] == winner_tab for r in priced[:2]):
                stats[cls]["top2"] += 1
            if any(r["TabNo"] == winner_tab for r in priced[:3]):
                stats[cls]["top3"] += 1

    print(f"\n{'Class':<20} {'Races':>7} {'Fav Win%':>10} {'Top2 Win%':>10} {'Top3 Win%':>10}")
    print("-" * 60)
    for cls in sorted(stats.keys(), key=lambda x: stats[x]["total"], reverse=True):
        s = stats[cls]
        t = s["total"]
        if t < 50:
            continue
        print(f"{cls:<20} {t:>7} {100*s['top1']/t:>9.1f}% {100*s['top2']/t:>9.1f}% {100*s['top3']/t:>9.1f}%")


def analyze_odds_bands(all_meetings):
    print("\n" + "="*80)
    print("4. FAVOURITE ODDS BAND WIN RATE (SP ODDS OF TOP-1 FAVOURITE)")
    print("="*80)

    bands = [
        ("$1.00-$1.50", 1.00, 1.50),
        ("$1.50-$2.00", 1.50, 2.00),
        ("$2.00-$2.50", 2.00, 2.50),
        ("$2.50-$3.00", 2.50, 3.00),
        ("$3.00-$3.50", 3.00, 3.50),
        ("$3.50-$4.00", 3.50, 4.00),
        ("$4.00-$5.00", 4.00, 5.00),
        ("$5.00-$6.00", 5.00, 6.00),
        ("$6.00-$8.00", 6.00, 8.00),
        ("$8.00-$10.00", 8.00, 10.00),
        ("$10.00+", 10.00, 999.00),
    ]

    stats = {b[0]: {"wins": 0, "total": 0} for b in bands}
    stats2 = {b[0]: {"wins": 0, "total": 0} for b in bands}
    stats3 = {b[0]: {"wins": 0, "total": 0} for b in bands}

    for meeting in all_meetings:
        for race in meeting["races"]:
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced:
                continue
            fav_price = priced[0]["Price"]
            winner_tab = winner["TabNo"]

            for band_name, lo, hi in bands:
                if lo <= fav_price < hi:
                    stats[band_name]["total"] += 1
                    if priced[0]["TabNo"] == winner_tab:
                        stats[band_name]["wins"] += 1
                    if len(priced) >= 2:
                        stats2[band_name]["total"] += 1
                        if any(r["TabNo"] == winner_tab for r in priced[:2]):
                            stats2[band_name]["wins"] += 1
                    if len(priced) >= 3:
                        stats3[band_name]["total"] += 1
                        if any(r["TabNo"] == winner_tab for r in priced[:3]):
                            stats3[band_name]["wins"] += 1
                    break

    print(f"\n{'Fav Odds':>14} {'Races':>7} {'Fav Win%':>10} {'Top2 Win%':>10} {'Top3 Win%':>10}")
    print("-" * 55)
    for band_name, _, _ in bands:
        s = stats[band_name]
        s2 = stats2[band_name]
        s3 = stats3[band_name]
        t = s["total"]
        if t == 0:
            continue
        t2_pct = 100 * s2["wins"] / s2["total"] if s2["total"] else 0
        t3_pct = 100 * s3["wins"] / s3["total"] if s3["total"] else 0
        print(f"{band_name:>14} {t:>7} {100*s['wins']/t:>9.1f}% {t2_pct:>9.1f}% {t3_pct:>9.1f}%")


def analyze_consecutive_winners(all_meetings):
    print("\n" + "="*80)
    print("5. CONSECUTIVE WINNER PATTERNS (QUADDIE SIMULATION)")
    print("="*80)

    total_windows = 0
    hits = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    leg_hits = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    leg_total = 0

    for meeting in all_meetings:
        races = sorted(meeting["races"], key=lambda r: r["race_number"])
        for i in range(len(races) - 3):
            window = races[i:i+4]
            valid = True
            for race in window:
                if not race["winner"] or not race["runners_by_price"] or len(race["runners_by_price"]) < 3:
                    valid = False
                    break
            if not valid:
                continue
            nums = [r["race_number"] for r in window]
            if nums != list(range(nums[0], nums[0]+4)):
                continue

            total_windows += 1
            for top_n in [1, 2, 3, 4, 5]:
                all_legs_hit = True
                for race in window:
                    winner_tab = race["winner"]["TabNo"]
                    top_runners = race["runners_by_price"][:top_n]
                    if not any(r["TabNo"] == winner_tab for r in top_runners):
                        all_legs_hit = False
                        break
                if all_legs_hit:
                    hits[top_n] += 1

    for meeting in all_meetings:
        for race in meeting["races"]:
            if not race["winner"] or not race["runners_by_price"] or len(race["runners_by_price"]) < 3:
                continue
            leg_total += 1
            winner_tab = race["winner"]["TabNo"]
            for top_n in [1, 2, 3, 4, 5]:
                if any(r["TabNo"] == winner_tab for r in race["runners_by_price"][:top_n]):
                    leg_hits[top_n] += 1

    print(f"\nTotal 4-race consecutive windows: {total_windows}")
    print(f"\n{'Top-N':>8} {'Leg Hit%':>10} {'4-Leg Sweep%':>14} {'Expected*':>12}")
    print("-" * 48)
    for n in [1, 2, 3, 4, 5]:
        leg_pct = leg_hits[n] / leg_total if leg_total else 0
        sweep_pct = hits[n] / total_windows if total_windows else 0
        expected = leg_pct ** 4
        print(f"Top-{n:>2}  {100*leg_pct:>9.1f}% {100*sweep_pct:>13.2f}% {100*expected:>11.2f}%")

    print("\n* Expected = (leg hit rate)^4 assuming independence")


def analyze_early_vs_late(all_meetings):
    print("\n" + "="*80)
    print("6. EARLY VS LATE RACE PREDICTABILITY")
    print("="*80)

    groups = {
        "R1-R3 (Early)": (1, 3),
        "R4-R6 (Mid)": (4, 6),
        "R7-R9 (Late)": (7, 9),
        "R10-R12 (Final)": (10, 12),
    }

    stats = {g: {"top1": 0, "top2": 0, "top3": 0, "total": 0, "avg_field": []} for g in groups}

    for meeting in all_meetings:
        for race in meeting["races"]:
            rn = race["race_number"]
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced or len(priced) < 3:
                continue
            winner_tab = winner["TabNo"]

            for gname, (lo, hi) in groups.items():
                if lo <= rn <= hi:
                    stats[gname]["total"] += 1
                    stats[gname]["avg_field"].append(race["field_size"])
                    if priced[0]["TabNo"] == winner_tab:
                        stats[gname]["top1"] += 1
                    if any(r["TabNo"] == winner_tab for r in priced[:2]):
                        stats[gname]["top2"] += 1
                    if any(r["TabNo"] == winner_tab for r in priced[:3]):
                        stats[gname]["top3"] += 1
                    break

    print(f"\n{'Group':<20} {'Races':>7} {'Avg Field':>10} {'Fav Win%':>10} {'Top2 Win%':>10} {'Top3 Win%':>10}")
    print("-" * 70)
    for gname in groups:
        s = stats[gname]
        t = s["total"]
        if t == 0:
            continue
        avg_f = statistics.mean(s["avg_field"]) if s["avg_field"] else 0
        print(f"{gname:<20} {t:>7} {avg_f:>9.1f} {100*s['top1']/t:>9.1f}% {100*s['top2']/t:>9.1f}% {100*s['top3']/t:>9.1f}%")


def analyze_venue_type(all_meetings):
    print("\n" + "="*80)
    print("7. VENUE TYPE PATTERNS (METRO vs PROVINCIAL vs COUNTRY)")
    print("="*80)

    type_map = {"M": "Metro", "P": "Provincial", "C": "Country"}
    stats = defaultdict(lambda: {"top1": 0, "top2": 0, "top3": 0, "total": 0, "avg_field": [], "avg_fav_price": []})
    quad_stats = defaultdict(lambda: {n: 0 for n in [1, 2, 3, 4, 5]})
    quad_total = defaultdict(int)

    for meeting in all_meetings:
        loc = meeting.get("location", "")
        vtype = type_map.get(loc, "Unknown")
        if vtype == "Unknown":
            continue

        for race in meeting["races"]:
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced or len(priced) < 3:
                continue
            winner_tab = winner["TabNo"]
            stats[vtype]["total"] += 1
            stats[vtype]["avg_field"].append(race["field_size"])
            stats[vtype]["avg_fav_price"].append(priced[0]["Price"])
            if priced[0]["TabNo"] == winner_tab:
                stats[vtype]["top1"] += 1
            if any(r["TabNo"] == winner_tab for r in priced[:2]):
                stats[vtype]["top2"] += 1
            if any(r["TabNo"] == winner_tab for r in priced[:3]):
                stats[vtype]["top3"] += 1

        races = sorted(meeting["races"], key=lambda r: r["race_number"])
        for i in range(len(races) - 3):
            window = races[i:i+4]
            nums = [r["race_number"] for r in window]
            if nums != list(range(nums[0], nums[0]+4)):
                continue
            valid = all(r["winner"] and r["runners_by_price"] and len(r["runners_by_price"]) >= 3 for r in window)
            if not valid:
                continue
            quad_total[vtype] += 1
            for top_n in [1, 2, 3, 4, 5]:
                all_hit = all(
                    any(r["TabNo"] == race["winner"]["TabNo"] for r in race["runners_by_price"][:top_n])
                    for race in window
                )
                if all_hit:
                    quad_stats[vtype][top_n] += 1

    print(f"\n{'Type':<12} {'Races':>7} {'Avg Field':>10} {'Avg Fav$':>10} {'Fav Win%':>10} {'Top2 Win%':>10} {'Top3 Win%':>10}")
    print("-" * 75)
    for vtype in ["Metro", "Provincial", "Country"]:
        s = stats[vtype]
        t = s["total"]
        if t == 0:
            continue
        avg_f = statistics.mean(s["avg_field"]) if s["avg_field"] else 0
        avg_p = statistics.mean(s["avg_fav_price"]) if s["avg_fav_price"] else 0
        print(f"{vtype:<12} {t:>7} {avg_f:>9.1f} {avg_p:>9.2f} {100*s['top1']/t:>9.1f}% {100*s['top2']/t:>9.1f}% {100*s['top3']/t:>9.1f}%")

    print(f"\n--- Quaddie Sweep Rates by Venue Type ---")
    print(f"{'Type':<12} {'Windows':>8} {'Top1':>8} {'Top2':>8} {'Top3':>8} {'Top4':>8} {'Top5':>8}")
    print("-" * 60)
    for vtype in ["Metro", "Provincial", "Country"]:
        qt = quad_total[vtype]
        if qt == 0:
            continue
        row = f"{vtype:<12} {qt:>8}"
        for n in [1, 2, 3, 4, 5]:
            row += f" {100*quad_stats[vtype][n]/qt:>7.1f}%"
        print(row)


def analyze_leg_width_sim(all_meetings):
    print("\n" + "="*80)
    print("BONUS: QUADDIE LEG WIDTH SIMULATION (LAST 4 RACES PER MEETING)")
    print("="*80)

    total_quaddies = 0
    width_configs = {
        "1x1x1x1 (Skinny)": [1, 1, 1, 1],
        "2x2x2x2 (Narrow)": [2, 2, 2, 2],
        "3x3x3x3 (Standard)": [3, 3, 3, 3],
        "2x3x3x4 (Graded)": [2, 3, 3, 4],
        "3x3x4x5 (Wide)": [3, 3, 4, 5],
        "4x4x4x4 (Very Wide)": [4, 4, 4, 4],
        "2x2x3x3 (Min2)": [2, 2, 3, 3],
    }

    hits = {k: 0 for k in width_configs}

    for meeting in all_meetings:
        races = sorted(meeting["races"], key=lambda r: r["race_number"])
        if len(races) < 4:
            continue
        window = races[-4:]
        nums = [r["race_number"] for r in window]
        if nums != list(range(nums[0], nums[0]+4)):
            continue
        valid = all(r["winner"] and r["runners_by_price"] and len(r["runners_by_price"]) >= 3 for r in window)
        if not valid:
            continue

        total_quaddies += 1

        for config_name, widths in width_configs.items():
            all_hit = True
            for race, w in zip(window, widths):
                winner_tab = race["winner"]["TabNo"]
                top_runners = race["runners_by_price"][:w]
                if not any(r["TabNo"] == winner_tab for r in top_runners):
                    all_hit = False
                    break
            if all_hit:
                hits[config_name] += 1

    print(f"\nTotal quaddies (last 4 races per meeting): {total_quaddies}")
    print(f"\n{'Config':<25} {'Hit%':>8} {'Combos':>8} {'Cost@$1':>10} {'Implied $div':>14}")
    print("-" * 70)
    for config_name, widths in width_configs.items():
        combos = 1
        for w in widths:
            combos *= w
        h = hits[config_name]
        hit_pct = h / total_quaddies if total_quaddies else 0
        cost = combos
        implied_div = 1 / hit_pct if hit_pct > 0 else float("inf")
        print(f"{config_name:<25} {100*hit_pct:>7.1f}% {combos:>8} {f'${cost}':>10} {f'${implied_div:.0f}':>14}")


def analyze_winner_odds_when_fav_loses(all_meetings):
    print("\n" + "="*80)
    print("EXTRA: WHEN FAVOURITE LOSES - WINNER ODDS DISTRIBUTION")
    print("="*80)

    fav_loses_winner_odds = []
    fav_wins_count = 0
    total = 0

    for meeting in all_meetings:
        for race in meeting["races"]:
            priced = race["runners_by_price"]
            winner = race["winner"]
            if not winner or not priced:
                continue
            total += 1
            winner_tab = winner["TabNo"]
            if priced[0]["TabNo"] == winner_tab:
                fav_wins_count += 1
            else:
                for r in priced:
                    if r["TabNo"] == winner_tab:
                        fav_loses_winner_odds.append(r["Price"])
                        break

    if not fav_loses_winner_odds:
        return

    bands = [
        ("$1-$3", 1, 3), ("$3-$5", 3, 5), ("$5-$8", 5, 8),
        ("$8-$12", 8, 12), ("$12-$20", 12, 20), ("$20-$50", 20, 50), ("$50+", 50, 9999),
    ]

    non_fav_total = len(fav_loses_winner_odds)
    print(f"\nFav wins: {fav_wins_count}/{total} ({100*fav_wins_count/total:.1f}%)")
    print(f"Non-fav winners: {non_fav_total}")
    print(f"\n{'Band':<12} {'Count':>8} {'%':>8} {'Cumulative%':>12}")
    print("-" * 45)
    cumul = 0
    for bname, lo, hi in bands:
        cnt = sum(1 for o in fav_loses_winner_odds if lo <= o < hi)
        pct = cnt / non_fav_total
        cumul += pct
        print(f"{bname:<12} {cnt:>8} {100*pct:>7.1f}% {100*cumul:>11.1f}%")


def analyze_quaddie_by_class_mix(all_meetings):
    print("\n" + "="*80)
    print("EXTRA: QUADDIE HIT RATE BY NUMBER OF MAIDEN LEGS")
    print("="*80)

    maiden_count_stats = defaultdict(lambda: {"total": 0, "top3_hit": 0})

    for meeting in all_meetings:
        races = sorted(meeting["races"], key=lambda r: r["race_number"])
        if len(races) < 4:
            continue
        window = races[-4:]
        nums = [r["race_number"] for r in window]
        if nums != list(range(nums[0], nums[0]+4)):
            continue
        valid = all(r["winner"] and r["runners_by_price"] and len(r["runners_by_price"]) >= 3 for r in window)
        if not valid:
            continue

        maiden_legs = sum(1 for r in window if "maiden" in classify_race(r).lower())
        maiden_count_stats[maiden_legs]["total"] += 1

        all_hit = all(
            any(r["TabNo"] == race["winner"]["TabNo"] for r in race["runners_by_price"][:3])
            for race in window
        )
        if all_hit:
            maiden_count_stats[maiden_legs]["top3_hit"] += 1

    print(f"\n{'Maiden Legs':>12} {'Quaddies':>10} {'Top3 Sweep%':>14}")
    print("-" * 40)
    for n_maiden in sorted(maiden_count_stats.keys()):
        s = maiden_count_stats[n_maiden]
        t = s["total"]
        if t < 10:
            continue
        print(f"{n_maiden:>12} {t:>10} {100*s['top3_hit']/t:>13.1f}%")


def analyze_dynamic_width(all_meetings):
    print("\n" + "="*80)
    print("BONUS: ADAPTIVE WIDTH QUADDIE (WIDTH BASED ON FAV ODDS)")
    print("="*80)
    print("Strategy: fav<$2 -> 2 wide, $2-$4 -> 3 wide, $4+ -> 4 wide")

    total = 0
    adaptive_hits = 0
    fixed3_hits = 0
    adaptive_combos_total = 0

    for meeting in all_meetings:
        races = sorted(meeting["races"], key=lambda r: r["race_number"])
        if len(races) < 4:
            continue
        window = races[-4:]
        nums = [r["race_number"] for r in window]
        if nums != list(range(nums[0], nums[0]+4)):
            continue
        valid = all(r["winner"] and r["runners_by_price"] and len(r["runners_by_price"]) >= 3 for r in window)
        if not valid:
            continue

        total += 1

        adaptive_all_hit = True
        combos = 1
        for race in window:
            fav_price = race["runners_by_price"][0]["Price"]
            if fav_price < 2.0:
                w = 2
            elif fav_price < 4.0:
                w = 3
            else:
                w = 4
            combos *= w
            winner_tab = race["winner"]["TabNo"]
            if not any(r["TabNo"] == winner_tab for r in race["runners_by_price"][:w]):
                adaptive_all_hit = False

        adaptive_combos_total += combos
        if adaptive_all_hit:
            adaptive_hits += 1

        fixed3_all_hit = all(
            any(r["TabNo"] == race["winner"]["TabNo"] for r in race["runners_by_price"][:3])
            for race in window
        )
        if fixed3_all_hit:
            fixed3_hits += 1

    if total == 0:
        return

    avg_adaptive_combos = adaptive_combos_total / total
    print(f"\nTotal quaddies: {total}")
    print(f"\n{'Strategy':<25} {'Hit%':>8} {'Avg Combos':>12} {'Implied Div':>14}")
    print("-" * 62)

    adaptive_pct = adaptive_hits / total
    fixed3_pct = fixed3_hits / total
    print(f"{'Adaptive (2/3/4)':<25} {100*adaptive_pct:>7.1f}% {avg_adaptive_combos:>11.1f} {f'${1/adaptive_pct:.0f}' if adaptive_pct else 'inf':>14}")
    print(f"{'Fixed 3-wide':<25} {100*fixed3_pct:>7.1f}% {81:>11.0f} {f'${1/fixed3_pct:.0f}' if fixed3_pct else 'inf':>14}")


def main():
    print("Loading Proform data...")
    all_meetings = load_all_data()

    total_races = sum(len(m["races"]) for m in all_meetings)
    total_runners = sum(len(r["runners_by_price"]) for m in all_meetings for r in m["races"])
    print(f"\nLoaded {len(all_meetings)} meetings, {total_races} races, {total_runners} priced runners")

    dates = [m["date"] for m in all_meetings if m["date"]]
    dates.sort()
    print(f"Date range: {dates[0][:10] if dates else '?'} to {dates[-1][:10] if dates else '?'}")

    analyze_fav_by_race_number(all_meetings)
    analyze_field_size(all_meetings)
    analyze_race_class(all_meetings)
    analyze_odds_bands(all_meetings)
    analyze_consecutive_winners(all_meetings)
    analyze_early_vs_late(all_meetings)
    analyze_venue_type(all_meetings)
    analyze_leg_width_sim(all_meetings)
    analyze_winner_odds_when_fav_loses(all_meetings)
    analyze_quaddie_by_class_mix(all_meetings)
    analyze_dynamic_width(all_meetings)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
