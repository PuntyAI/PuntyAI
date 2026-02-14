"""Analyze historical payouts across all bet types from Proform data.

Uses PriceSP (starting prices) + finishing positions to simulate what
different bet types would have paid across 381K+ runners (2025+2026).

Analyzes: Win, Place, Exacta, Quinella, Trifecta, First4
Grouped by: venue type, distance bucket, class bucket, field size

Usage:
    python scripts/analyze_historical_payouts.py
    python scripts/analyze_historical_payouts.py --output results.json
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIRS = [
    Path(r"D:\Punty\DatafromProform\2025"),
    Path(r"D:\Punty\DatafromProform\2026"),
]

MONTH_DIRS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

# Venue classification (same as build_context_profiles.py)
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


def _dist_bucket(distance: int) -> str:
    if distance <= 1100: return "sprint"
    if distance <= 1399: return "short"
    if distance <= 1799: return "middle"
    if distance <= 2199: return "classic"
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
    return "bm64"


def _field_size_bucket(n: int) -> str:
    if n <= 6: return "small (<=6)"
    if n <= 10: return "medium (7-10)"
    if n <= 14: return "large (11-14)"
    return "very_large (15+)"


def _estimate_place_dividend(sp: float, field_size: int) -> float:
    """Estimate place dividend from starting price.

    Place typically pays ~1/3 to 1/4 of win odds for top 3.
    Smaller fields (<=7) only pay top 2.
    """
    if sp <= 1.0:
        return 1.0
    # Place fraction depends on field size
    if field_size <= 7:
        # Only 2 places, slightly higher place odds
        place_fraction = 0.35
    elif field_size >= 16:
        # 4 places in very large fields, lower fraction
        place_fraction = 0.22
    else:
        place_fraction = 0.28

    place_odds = 1.0 + (sp - 1.0) * place_fraction
    return round(place_odds, 2)


def load_all_races(data_dirs: list[Path]) -> list[dict]:
    """Load all races with finishing positions and SPs from Proform data."""
    all_races = []
    total_runners = 0

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  SKIPPING {data_dir} (not found)")
            continue

        year = data_dir.name

        for month_num in range(1, 13):
            month_name = MONTH_DIRS[month_num]

            # Load race metadata from meetings.json
            meetings_path = data_dir / month_name / "meetings.json"
            race_meta = {}
            if meetings_path.exists():
                with open(meetings_path, "r", encoding="utf-8") as f:
                    meetings = json.load(f)
                for m in meetings:
                    venue = m.get("Track", {}).get("Name", "")
                    state = m.get("Track", {}).get("State", "")
                    for race in m.get("Races", []):
                        rid = race.get("RaceId")
                        try:
                            rid = int(rid)
                        except (ValueError, TypeError):
                            continue
                        race_meta[rid] = {
                            "venue": venue,
                            "state": state,
                            "distance": race.get("Distance", 1400),
                            "race_class": (race.get("RaceClass", "") or "").rstrip(";").strip(),
                            "prize_money": race.get("PrizeMoney", 0),
                        }

            # Load form files (have PriceSP + Position)
            form_dir = data_dir / month_name / "Form"
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

                # Group by race
                races_in_file = defaultdict(list)
                for r in runners:
                    rid = r.get("RaceId")
                    if rid:
                        races_in_file[int(rid)].append(r)

                for race_id, race_runners in races_in_file.items():
                    meta = race_meta.get(race_id, {})
                    if not meta:
                        continue

                    # Get runners with valid positions and prices
                    valid_runners = []
                    for r in race_runners:
                        pos = r.get("Position")
                        sp = r.get("PriceSP", 0)
                        if pos and pos > 0 and sp and sp > 1.0:
                            valid_runners.append({
                                "position": pos,
                                "sp": sp,
                                "tab_no": r.get("TabNo", 0),
                                "name": r.get("Name", ""),
                            })

                    if len(valid_runners) < 4:
                        continue  # Need at least 4 finishers

                    # Sort by finishing position
                    valid_runners.sort(key=lambda x: x["position"])
                    field_size = len(valid_runners)
                    total_runners += field_size

                    all_races.append({
                        "race_id": race_id,
                        "venue": meta["venue"],
                        "state": meta["state"],
                        "distance": meta["distance"],
                        "race_class": meta["race_class"],
                        "prize_money": meta.get("prize_money", 0),
                        "field_size": field_size,
                        "runners": valid_runners,
                        "year": year,
                    })

            print(f"  {year}/{month_name}: {len(all_races):,} races, {total_runners:,} runners")

    return all_races


def analyze_payouts(races: list[dict]) -> dict:
    """Analyze simulated payouts across all bet types."""

    # Accumulators per context
    contexts = defaultdict(lambda: {
        "win": [], "place": [],
        "exacta": [], "quinella": [], "trifecta": [], "first4": [],
        "win_by_sp_range": defaultdict(list),
        "races": 0, "field_sizes": [],
    })

    # Also track overall
    overall = {
        "win": [], "place_2nd": [], "place_3rd": [],
        "exacta": [], "quinella": [], "trifecta": [], "first4": [],
        "win_by_sp_range": defaultdict(list),
        "place_by_sp_range": defaultdict(list),
        "win_by_field_size": defaultdict(list),
        "trifecta_by_field_size": defaultdict(list),
        "exacta_by_field_size": defaultdict(list),
        "races": 0,
        # Track how often top-N rated runners fill positions
        "top1_wins": 0, "top2_exacta": 0, "top3_trifecta": 0, "top4_first4": 0,
        "fav_wins": 0, "fav_places": 0,
    }

    for race in races:
        runners = race["runners"]
        field_size = race["field_size"]

        if len(runners) < 4:
            continue

        # Get finishing order
        r1 = runners[0]  # Winner
        r2 = runners[1]  # 2nd
        r3 = runners[2]  # 3rd
        r4 = runners[3]  # 4th

        # Context keys
        vtype = _venue_type(race["venue"], race["state"])
        dbucket = _dist_bucket(race["distance"])
        cbucket = _class_bucket(race["race_class"])
        fsbucket = _field_size_bucket(field_size)

        ctx_keys = [
            f"ALL",
            f"vtype:{vtype}",
            f"dist:{dbucket}",
            f"class:{cbucket}",
            f"field:{fsbucket}",
            f"{vtype}|{dbucket}",
            f"{vtype}|{cbucket}",
            f"{dbucket}|{cbucket}",
        ]

        # --- Win dividend ---
        win_div = r1["sp"]

        # --- Place dividends (estimated) ---
        place_div_1st = _estimate_place_dividend(r1["sp"], field_size)
        place_div_2nd = _estimate_place_dividend(r2["sp"], field_size)
        place_div_3rd = _estimate_place_dividend(r3["sp"], field_size)

        # --- Exacta: 1st × 2nd (approximate) ---
        # Exacta ≈ SP1 × SP2 × 0.6 (TAB pool factor)
        exacta_div = r1["sp"] * r2["sp"] * 0.6

        # --- Quinella: any order top 2 ---
        # Quinella ≈ SP1 × SP2 × 0.35
        quinella_div = r1["sp"] * r2["sp"] * 0.35

        # --- Trifecta: 1st, 2nd, 3rd in order ---
        # Trifecta ≈ SP1 × SP2 × SP3 × 0.4
        trifecta_div = r1["sp"] * r2["sp"] * r3["sp"] * 0.4

        # --- Trifecta Box (3 runners): any order = trifecta / 6 ---
        trifecta_box_3 = trifecta_div / 6

        # --- Trifecta Box (4 runners): 24 combos ---
        trifecta_box_4 = trifecta_div / 24

        # --- First4: 1st, 2nd, 3rd, 4th ---
        first4_div = r1["sp"] * r2["sp"] * r3["sp"] * r4["sp"] * 0.3

        # --- First4 Box (4 runners): 24 combos ---
        first4_box_4 = first4_div / 24

        # SP range for winner
        if win_div <= 2.0: sp_range = "$1-$2"
        elif win_div <= 4.0: sp_range = "$2-$4"
        elif win_div <= 8.0: sp_range = "$4-$8"
        elif win_div <= 15.0: sp_range = "$8-$15"
        elif win_div <= 30.0: sp_range = "$15-$30"
        else: sp_range = "$30+"

        # Check if favourite won (lowest SP)
        sps = sorted([r["sp"] for r in runners])
        fav_sp = sps[0]
        fav_won = r1["sp"] == fav_sp
        fav_placed = any(r["sp"] == fav_sp for r in runners[:3])

        # Check if top-N by SP filled positions
        sp_ranked = sorted(runners, key=lambda x: x["sp"])
        top1_sc = {sp_ranked[0]["tab_no"]}
        top2_sc = {sp_ranked[i]["tab_no"] for i in range(min(2, len(sp_ranked)))}
        top3_sc = {sp_ranked[i]["tab_no"] for i in range(min(3, len(sp_ranked)))}
        top4_sc = {sp_ranked[i]["tab_no"] for i in range(min(4, len(sp_ranked)))}

        actual_1st = {r1["tab_no"]}
        actual_top2 = {r1["tab_no"], r2["tab_no"]}
        actual_top3 = {r1["tab_no"], r2["tab_no"], r3["tab_no"]}
        actual_top4 = {r1["tab_no"], r2["tab_no"], r3["tab_no"], r4["tab_no"]}

        overall["races"] += 1
        overall["win"].append(win_div)
        overall["place_2nd"].append(place_div_2nd)
        overall["place_3rd"].append(place_div_3rd)
        overall["exacta"].append(exacta_div)
        overall["quinella"].append(quinella_div)
        overall["trifecta"].append(trifecta_div)
        overall["first4"].append(first4_div)
        overall["win_by_sp_range"][sp_range].append(win_div)
        overall["place_by_sp_range"][sp_range].append(place_div_2nd)
        overall["win_by_field_size"][fsbucket].append(win_div)
        overall["trifecta_by_field_size"][fsbucket].append(trifecta_div)
        overall["exacta_by_field_size"][fsbucket].append(exacta_div)

        if fav_won: overall["fav_wins"] += 1
        if fav_placed: overall["fav_places"] += 1
        if actual_1st <= top1_sc: overall["top1_wins"] += 1
        if actual_top2 <= top2_sc: overall["top2_exacta"] += 1
        if actual_top3 <= top3_sc: overall["top3_trifecta"] += 1
        if actual_top4 <= top4_sc: overall["top4_first4"] += 1

        for key in ctx_keys:
            ctx = contexts[key]
            ctx["races"] += 1
            ctx["win"].append(win_div)
            ctx["place"].append(place_div_2nd)
            ctx["exacta"].append(exacta_div)
            ctx["quinella"].append(quinella_div)
            ctx["trifecta"].append(trifecta_div)
            ctx["first4"].append(first4_div)
            ctx["field_sizes"].append(field_size)
            ctx["win_by_sp_range"][sp_range].append(win_div)

    return {"overall": overall, "contexts": dict(contexts)}


def _percentile(data: list, p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _stats(data: list) -> dict:
    """Compute summary stats for a list of dividends."""
    if not data:
        return {"n": 0}
    return {
        "n": len(data),
        "mean": round(statistics.mean(data), 2),
        "median": round(statistics.median(data), 2),
        "p25": round(_percentile(data, 25), 2),
        "p75": round(_percentile(data, 75), 2),
        "p90": round(_percentile(data, 90), 2),
        "min": round(min(data), 2),
        "max": round(max(data), 2),
    }


def print_report(results: dict):
    """Print formatted analysis report."""
    overall = results["overall"]
    contexts = results["contexts"]
    n = overall["races"]

    print(f"\n{'='*80}")
    print(f"HISTORICAL PAYOUT ANALYSIS — {n:,} races analyzed")
    print(f"{'='*80}")

    # === OVERALL DIVIDENDS ===
    print(f"\n{'-'*60}")
    print("OVERALL DIVIDEND STATISTICS (estimated from SP)")
    print(f"{'-'*60}")

    for bet_type, data in [
        ("Win", overall["win"]),
        ("Place (2nd)", overall["place_2nd"]),
        ("Place (3rd)", overall["place_3rd"]),
        ("Exacta", overall["exacta"]),
        ("Quinella", overall["quinella"]),
        ("Trifecta", overall["trifecta"]),
        ("First4", overall["first4"]),
    ]:
        s = _stats(data)
        if s["n"]:
            print(f"\n  {bet_type:20s} (n={s['n']:,})")
            print(f"    Mean: ${s['mean']:>10.2f}  |  Median: ${s['median']:>10.2f}")
            print(f"    P25:  ${s['p25']:>10.2f}  |  P75:    ${s['p75']:>10.2f}  |  P90: ${s['p90']:>10.2f}")

    # === FAVOURITE PERFORMANCE ===
    print(f"\n{'-'*60}")
    print("FAVOURITE / TOP-RATED PERFORMANCE")
    print(f"{'-'*60}")
    fav_win_pct = overall["fav_wins"] / n * 100 if n else 0
    fav_place_pct = overall["fav_places"] / n * 100 if n else 0
    top1_pct = overall["top1_wins"] / n * 100 if n else 0
    top2_pct = overall["top2_exacta"] / n * 100 if n else 0
    top3_pct = overall["top3_trifecta"] / n * 100 if n else 0
    top4_pct = overall["top4_first4"] / n * 100 if n else 0

    print(f"  Favourite wins:           {fav_win_pct:5.1f}%  ({overall['fav_wins']:,}/{n:,})")
    print(f"  Favourite places (top 3): {fav_place_pct:5.1f}%  ({overall['fav_places']:,}/{n:,})")
    print(f"  Top 1 rated wins:         {top1_pct:5.1f}%")
    print(f"  Top 2 fill exacta:        {top2_pct:5.1f}%")
    print(f"  Top 3 fill trifecta:      {top3_pct:5.1f}%")
    print(f"  Top 4 fill first4:        {top4_pct:5.1f}%")

    # === WIN BY SP RANGE ===
    print(f"\n{'-'*60}")
    print("WIN DIVIDEND BY SP RANGE")
    print(f"{'-'*60}")
    for sp_range in ["$1-$2", "$2-$4", "$4-$8", "$8-$15", "$15-$30", "$30+"]:
        data = overall["win_by_sp_range"].get(sp_range, [])
        if data:
            pct = len(data) / n * 100
            print(f"  {sp_range:10s}: {len(data):>6,} races ({pct:5.1f}%)  avg div ${statistics.mean(data):>8.2f}")

    # === BY FIELD SIZE ===
    print(f"\n{'-'*60}")
    print("EXOTIC DIVIDENDS BY FIELD SIZE")
    print(f"{'-'*60}")
    for fs in ["small (<=6)", "medium (7-10)", "large (11-14)", "very_large (15+)"]:
        tri_data = overall["trifecta_by_field_size"].get(fs, [])
        exa_data = overall["exacta_by_field_size"].get(fs, [])
        win_data = overall["win_by_field_size"].get(fs, [])
        if tri_data:
            print(f"  {fs:20s} ({len(tri_data):>6,} races)")
            print(f"    Win avg:      ${statistics.mean(win_data):>10.2f}  median: ${statistics.median(win_data):>10.2f}")
            print(f"    Exacta avg:   ${statistics.mean(exa_data):>10.2f}  median: ${statistics.median(exa_data):>10.2f}")
            print(f"    Trifecta avg: ${statistics.mean(tri_data):>10.2f}  median: ${statistics.median(tri_data):>10.2f}")

    # === BY VENUE TYPE ===
    print(f"\n{'-'*60}")
    print("DIVIDENDS BY VENUE TYPE")
    print(f"{'-'*60}")
    for vtype in ["metro_vic", "metro_nsw", "metro_qld", "metro_other", "provincial", "country"]:
        key = f"vtype:{vtype}"
        if key in contexts:
            ctx = contexts[key]
            n_ctx = ctx["races"]
            if n_ctx < 50:
                continue
            avg_fs = statistics.mean(ctx["field_sizes"])
            print(f"\n  {vtype:15s} ({n_ctx:,} races, avg field {avg_fs:.1f})")
            for bt, label in [("win", "Win"), ("quinella", "Quinella"), ("exacta", "Exacta"),
                             ("trifecta", "Trifecta"), ("first4", "First4")]:
                s = _stats(ctx[bt])
                if s["n"]:
                    print(f"    {label:12s}: mean ${s['mean']:>10.2f}  median ${s['median']:>10.2f}  P75 ${s['p75']:>10.2f}")

    # === BY DISTANCE ===
    print(f"\n{'-'*60}")
    print("DIVIDENDS BY DISTANCE")
    print(f"{'-'*60}")
    for dist in ["sprint", "short", "middle", "classic", "staying"]:
        key = f"dist:{dist}"
        if key in contexts:
            ctx = contexts[key]
            n_ctx = ctx["races"]
            if n_ctx < 50:
                continue
            print(f"\n  {dist:12s} ({n_ctx:,} races)")
            for bt, label in [("win", "Win"), ("quinella", "Quinella"), ("exacta", "Exacta"),
                             ("trifecta", "Trifecta"), ("first4", "First4")]:
                s = _stats(ctx[bt])
                if s["n"]:
                    print(f"    {label:12s}: mean ${s['mean']:>10.2f}  median ${s['median']:>10.2f}")

    # === BY CLASS ===
    print(f"\n{'-'*60}")
    print("DIVIDENDS BY CLASS")
    print(f"{'-'*60}")
    for cls in ["maiden", "class1", "restricted", "class2", "class3", "bm58", "bm64", "bm72", "open"]:
        key = f"class:{cls}"
        if key in contexts:
            ctx = contexts[key]
            n_ctx = ctx["races"]
            if n_ctx < 50:
                continue
            print(f"\n  {cls:12s} ({n_ctx:,} races)")
            for bt, label in [("win", "Win"), ("trifecta", "Trifecta"), ("first4", "First4")]:
                s = _stats(ctx[bt])
                if s["n"]:
                    print(f"    {label:12s}: mean ${s['mean']:>10.2f}  median ${s['median']:>10.2f}")

    # === CROSS-CONTEXT: VENUE × DISTANCE ===
    print(f"\n{'-'*60}")
    print("CROSS-CONTEXT: VENUE TYPE × DISTANCE (Top Exotic Values)")
    print(f"{'-'*60}")
    cross_results = []
    for key, ctx in contexts.items():
        if "|" in key and "vtype:" not in key and "dist:" not in key and "class:" not in key and "field:" not in key:
            if ctx["races"] >= 100:
                tri_s = _stats(ctx["trifecta"])
                exa_s = _stats(ctx["exacta"])
                cross_results.append({
                    "key": key,
                    "races": ctx["races"],
                    "tri_median": tri_s.get("median", 0),
                    "exa_median": exa_s.get("median", 0),
                    "win_median": _stats(ctx["win"]).get("median", 0),
                })

    # Sort by trifecta median (highest payouts)
    cross_results.sort(key=lambda x: x["tri_median"], reverse=True)
    print(f"\n  {'Context':30s} {'Races':>7s}  {'Win Med':>10s}  {'Exacta Med':>10s}  {'Tri Med':>10s}")
    print(f"  {'-'*30} {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    for cr in cross_results[:20]:
        print(f"  {cr['key']:30s} {cr['races']:>7,}  ${cr['win_median']:>9.2f}  ${cr['exa_median']:>9.2f}  ${cr['tri_median']:>9.2f}")

    # === VALUE ANALYSIS: $20 bet ROI simulation ===
    print(f"\n{'-'*60}")
    print("$20 BET SIMULATION — Expected Returns")
    print(f"{'-'*60}")
    print("If you bet $20 on every race, what would each bet type return?")
    print("(Based on SP-estimated dividends, $20 flat stake)")
    print()

    stake = 20.0
    for bt, label, combos in [
        ("win", "Win ($20 flat)", 1),
        ("place_2nd", "Place 2nd ($20)", 1),
        ("place_3rd", "Place 3rd ($20)", 1),
        ("exacta", "Exacta ($20, 1 combo)", 1),
        ("quinella", "Quinella ($20, 1 combo)", 1),
    ]:
        data = overall.get(bt, [])
        if data:
            # For a $20 bet: you spend $20, you win (div × $20 / combos) if it hits
            # But exotics require hitting the EXACT combo
            # Win: you bet $20 on the favourite — hits fav_win_pct of time
            total_cost = len(data) * stake
            total_returns = sum(d * stake for d in data)
            roi = (total_returns - total_cost) / total_cost * 100
            avg_return = statistics.mean(data) * stake
            med_return = statistics.median(data) * stake
            print(f"  {label:30s}: avg return ${avg_return:>8.2f}  median ${med_return:>8.2f}")

    # Boxed exotics — what they pay per $1 unit
    print()
    print("  Boxed exotic returns PER $1 UNIT (unit = stake/combos):")
    for bt, label, combos in [
        ("trifecta", "Trifecta Box 3 (6 combos)", 6),
        ("trifecta", "Trifecta Box 4 (24 combos)", 24),
        ("first4", "First4 Box 4 (24 combos)", 24),
        ("first4", "First4 Box 5 (120 combos)", 120),
    ]:
        data = overall.get(bt, [])
        if data:
            # Per unit dividend = straight_div / combos
            per_unit = [d / combos for d in data]
            avg_pu = statistics.mean(per_unit)
            med_pu = statistics.median(per_unit)
            # At $20 stake: unit = $20/combos
            unit_price = stake / combos
            avg_return = avg_pu * stake
            med_return = med_pu * stake
            print(f"  {label:35s}: unit ${unit_price:.2f}  avg return ${avg_return:>8.2f}  median ${med_return:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze historical payouts from Proform data")
    parser.add_argument("--output", default=None, help="Save raw results to JSON")
    args = parser.parse_args()

    start = time.time()
    print("Loading races from Proform data (2025 + 2026)...")

    races = load_all_races(DATA_DIRS)
    elapsed = time.time() - start
    print(f"\nLoaded {len(races):,} races in {elapsed:.1f}s")

    print("\nAnalyzing payouts...")
    results = analyze_payouts(races)

    print_report(results)

    if args.output:
        # Serialize for JSON (convert defaultdicts)
        output = {
            "overall": {
                "races": results["overall"]["races"],
                "fav_wins": results["overall"]["fav_wins"],
                "fav_places": results["overall"]["fav_places"],
                "top1_wins": results["overall"]["top1_wins"],
                "top2_exacta": results["overall"]["top2_exacta"],
                "top3_trifecta": results["overall"]["top3_trifecta"],
                "top4_first4": results["overall"]["top4_first4"],
                "win": _stats(results["overall"]["win"]),
                "place_2nd": _stats(results["overall"]["place_2nd"]),
                "place_3rd": _stats(results["overall"]["place_3rd"]),
                "exacta": _stats(results["overall"]["exacta"]),
                "quinella": _stats(results["overall"]["quinella"]),
                "trifecta": _stats(results["overall"]["trifecta"]),
                "first4": _stats(results["overall"]["first4"]),
            },
            "by_context": {},
        }
        for key, ctx in results["contexts"].items():
            if ctx["races"] >= 50:
                output["by_context"][key] = {
                    "races": ctx["races"],
                    "win": _stats(ctx["win"]),
                    "exacta": _stats(ctx["exacta"]),
                    "quinella": _stats(ctx["quinella"]),
                    "trifecta": _stats(ctx["trifecta"]),
                    "first4": _stats(ctx["first4"]),
                }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
