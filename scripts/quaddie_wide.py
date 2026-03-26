"""Quaddie analysis extended to 6+ runners per leg.

$100 is a serious bet. What does it take to ACTUALLY hit?
- Capture rates out to rank 6, 7, 8
- Full quaddie hit rates at all width combinations
- Payout economics: what do quaddies pay vs what they cost?
- The case for going wider on specific legs
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from itertools import product as iterproduct

BASE = Path("D:/Punty/DatafromProform/2025")
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_all_meetings():
    meetings = []
    for month in MONTHS:
        path = BASE / month / "meetings.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        for m in data:
            track = m.get("Track", {})
            if track.get("Country") != "AUS":
                continue
            races = m.get("Races", [])
            if len(races) < 7:
                continue
            valid_races = []
            for r in races:
                runners = r.get("Runners", [])
                has_results = any(
                    rn.get("Position") is not None and rn.get("Position", 0) > 0
                    for rn in runners
                )
                if runners and has_results:
                    valid_races.append(r)
            if len(valid_races) >= 7:
                meetings.append({
                    "track": track.get("Name", ""),
                    "state": track.get("State", ""),
                    "location": track.get("Location", ""),
                    "races": valid_races,
                })
    return meetings


def analyse_race(race):
    runners = race.get("Runners", [])
    active = [r for r in runners if r.get("Position") and r["Position"] > 0]
    if len(active) < 3:
        return None

    for r in active:
        sp = r.get("PriceSP", 0)
        r["impl_prob"] = 1.0 / sp if sp and sp > 0 else 0
    total_prob = sum(r["impl_prob"] for r in active)
    if total_prob <= 0:
        return None
    for r in active:
        r["norm_prob"] = r["impl_prob"] / total_prob

    by_sp = sorted(active, key=lambda r: r.get("PriceSP", 999))

    winner = None
    for r in active:
        if r["Position"] == 1:
            winner = r
            break
    if not winner:
        return None

    winner_rank = None
    for i, r in enumerate(by_sp):
        if r.get("TabNo") == winner.get("TabNo") and r.get("Name") == winner.get("Name"):
            winner_rank = i + 1
            break
    if winner_rank is None:
        return None

    top = by_sp[:min(8, len(by_sp))]
    prices = [r.get("PriceSP", 999) for r in top]
    while len(prices) < 8:
        prices.append(999)

    ratios = []
    for i in range(min(7, len(top) - 1)):
        if prices[i] > 0:
            ratios.append(prices[i + 1] / prices[i])
        else:
            ratios.append(999)
    while len(ratios) < 7:
        ratios.append(999)

    return {
        "field_size": len(active),
        "winner_rank": winner_rank,
        "winner_sp": winner.get("PriceSP", 0),
        "prices": prices,
        "p1": prices[0], "p2": prices[1], "p3": prices[2], "p4": prices[3],
        "p5": prices[4], "p6": prices[5],
        "ratios": ratios,
        "ratio_2_1": ratios[0], "ratio_3_2": ratios[1], "ratio_4_3": ratios[2],
        "race_class": race.get("RaceClass", "") or "",
        "race_number": race.get("Number", 0),
    }


def classify_shape(leg):
    p1, p2, p3, p4 = leg["p1"], leg["p2"], leg["p3"], leg["p4"]
    if p1 <= 1.50:
        return "STANDOUT"       # Autumn Glow territory
    elif p1 <= 2.00 and p2 >= p1 * 2.5:
        return "DOMINANT"
    elif p1 <= 2.00:
        return "SHORT_PAIR"
    elif p1 <= 3.00 and p2 / p1 < 1.5:
        return "TWO_HORSE"
    elif p1 <= 3.50:
        return "CLEAR_FAV"
    elif p1 <= 5.00 and p3 / p1 < 1.8:
        return "TRIO"
    elif p1 <= 5.00:
        return "MID_FAV"
    elif p4 <= p1 * 2.0:
        return "OPEN_BUNCH"
    else:
        return "WIDE_OPEN"


def main():
    print("Loading 2025 Proform data...")
    meetings = load_all_meetings()
    print(f"Loaded {len(meetings)} meetings\n")

    all_legs = []
    quaddies = []
    eq_quaddies = []

    for m in meetings:
        races = m["races"]
        n = len(races)

        # Main quaddie = last 4
        q_legs = []
        for i, r in enumerate(races[n - 4:]):
            a = analyse_race(r)
            if a:
                a["leg_pos"] = i + 1
                all_legs.append(a)
                q_legs.append(a)
        if len(q_legs) == 4:
            quaddies.append(q_legs)

        # Early quaddie = first 4 (if 8+ races)
        if n >= 8:
            eq_legs = []
            for i, r in enumerate(races[:4]):
                a = analyse_race(r)
                if a:
                    a["leg_pos"] = i + 1
                    eq_legs.append(a)
            if len(eq_legs) == 4:
                eq_quaddies.append(eq_legs)

    print(f"Total legs: {len(all_legs)}")
    print(f"Complete quaddies: {len(quaddies)}")
    print(f"Complete early quaddies: {len(eq_quaddies)}")

    # ================================================================
    # 1. FULL CAPTURE RATES OUT TO RANK 8
    # ================================================================
    print("\n" + "=" * 90)
    print("1. LEG CAPTURE RATES: How many runners to catch the winner? (to rank 8)")
    print("=" * 90)

    total = len(all_legs)
    print(f"\n{'Width':>6} {'Captures':>9} {'Rate':>7} {'Cumul':>7} {'Marginal':>9}")
    print("-" * 45)
    cum = 0
    for w in range(1, 9):
        cnt = sum(1 for l in all_legs if l["winner_rank"] == w)
        cum += cnt
        marginal = cnt / total * 100
        print(f"  {w:>4}   {cum:>8}  {cum/total*100:>5.1f}%  {cum/total*100:>5.1f}%  +{marginal:>5.1f}%")

    # ================================================================
    # 2. BY ODDS SHAPE — OUT TO RANK 6
    # ================================================================
    print("\n" + "=" * 90)
    print("2. CAPTURE RATES BY ODDS SHAPE (to rank 6)")
    print("=" * 90)

    shapes_order = ["STANDOUT", "DOMINANT", "SHORT_PAIR", "TWO_HORSE", "CLEAR_FAV", "TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"]
    shape_desc = {
        "STANDOUT": "Fav $1.50 or less",
        "DOMINANT": "Fav $1.51-$2, 2nd 2.5x+ away",
        "SHORT_PAIR": "Fav $1.51-$2, 2nd close",
        "TWO_HORSE": "Fav $2-$3, 2nd within 1.5x",
        "CLEAR_FAV": "Fav $2.50-$3.50, clear of field",
        "TRIO": "Fav $3.50-$5, top3 bunched",
        "MID_FAV": "Fav $3.50-$5, top2 only",
        "OPEN_BUNCH": "Fav $5+, top4 bunched",
        "WIDE_OPEN": "Fav $5+, spread field",
    }

    shape_buckets = defaultdict(list)
    for l in all_legs:
        shape_buckets[classify_shape(l)].append(l)

    print(f"\n{'Shape':<14} {'N':>5} {'Top1':>5} {'Top2':>5} {'Top3':>5} {'Top4':>5} {'Top5':>5} {'Top6':>5} {'AvgPrice':>9}")
    print("-" * 75)
    for shape in shapes_order:
        legs = shape_buckets.get(shape, [])
        n = len(legs)
        if n < 15:
            continue
        rates = []
        for w in range(1, 7):
            rates.append(sum(1 for l in legs if l["winner_rank"] <= w) / n * 100)
        avg_prices = "/".join(f"${sum(l[f'p{i}'] for l in legs)/n:.0f}" for i in range(1, 5))
        print(f"  {shape:<12} {n:>5} {rates[0]:>4.0f}% {rates[1]:>4.0f}% {rates[2]:>4.0f}% {rates[3]:>4.0f}% {rates[4]:>4.0f}% {rates[5]:>4.0f}%  {avg_prices}")

    # ================================================================
    # 3. MARGINAL VALUE OF EACH RUNNER — Is runner 5/6 worth it?
    # ================================================================
    print("\n" + "=" * 90)
    print("3. MARGINAL VALUE: What does runner 5 and 6 actually buy per shape?")
    print("   '+X%' = extra legs captured by adding this runner")
    print("=" * 90)

    print(f"\n{'Shape':<14} {'N':>5} {'R1':>6} {'R2':>6} {'R3':>6} {'R4':>6} {'R5':>6} {'R6':>6}")
    print("-" * 60)
    for shape in shapes_order:
        legs = shape_buckets.get(shape, [])
        n = len(legs)
        if n < 15:
            continue
        marginals = []
        for w in range(1, 7):
            m = sum(1 for l in legs if l["winner_rank"] == w) / n * 100
            marginals.append(m)
        print(f"  {shape:<12} {n:>5} {marginals[0]:>5.1f}% {marginals[1]:>5.1f}% {marginals[2]:>5.1f}% {marginals[3]:>5.1f}% {marginals[4]:>5.1f}% {marginals[5]:>5.1f}%")

    print("\n  Rule of thumb: include a runner if their marginal capture > 7%")
    print("  Below 7% you're spending combos for diminishing returns")

    # ================================================================
    # 4. FULL QUADDIE HIT RATES — All widths 1-6
    # ================================================================
    print("\n" + "=" * 90)
    print("4. QUADDIE HIT RATES: Every width 1-6 across all 4 legs")
    print("   With $100 budget: unit = $100/combos")
    print("=" * 90)

    print(f"\n{'Width':>6} {'Combos':>7} {'$/unit':>7} {'Hits':>5} {'Hit%':>6} {'Avg Payout':>11} {'Est ROI':>8}")
    print("-" * 60)
    for w in range(1, 7):
        combos = w ** 4
        unit = 100.0 / combos
        hits = sum(1 for q in quaddies if all(q[i]["winner_rank"] <= w for i in range(4)))
        hit_pct = hits / len(quaddies) * 100

        # Estimate payout: product of winner SPs * 0.80 (pool return) * unit
        total_return = 0
        for q in quaddies:
            if all(q[i]["winner_rank"] <= w for i in range(4)):
                sp_product = 1
                for leg in q:
                    sp_product *= leg["winner_sp"]
                total_return += sp_product * 0.80 * unit
        avg_payout = total_return / hits if hits else 0
        total_cost = 100.0 * len(quaddies)
        roi = (total_return - total_cost) / total_cost * 100

        print(f"  {w:>4}   {combos:>6}  ${unit:>5.2f}  {hits:>5} {hit_pct:>5.1f}%  ${avg_payout:>9.0f}  {roi:>6.1f}%")

    # ================================================================
    # 5. MIXED WIDTH QUADDIE — The real question
    # ================================================================
    print("\n" + "=" * 90)
    print("5. MIXED WIDTH QUADDIES: Best allocations at $100")
    print("   Testing ALL combinations of 1-6 runners across 4 legs")
    print("   Filtering to combos that make sense ($1+ per unit)")
    print("=" * 90)

    # Test all width combos from 1-6 across 4 legs where combos <= 100
    results = []
    for w1 in range(1, 7):
        for w2 in range(1, 7):
            for w3 in range(1, 7):
                for w4 in range(1, 7):
                    combos = w1 * w2 * w3 * w4
                    if combos > 100:
                        continue
                    unit = 100.0 / combos
                    hits = 0
                    total_return = 0
                    for q in quaddies:
                        widths = [w1, w2, w3, w4]
                        if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                            hits += 1
                            sp_prod = 1
                            for leg in q:
                                sp_prod *= leg["winner_sp"]
                            total_return += sp_prod * 0.80 * unit
                    if hits > 0:
                        avg_payout = total_return / hits
                        total_cost = 100.0 * len(quaddies)
                        roi = (total_return - total_cost) / total_cost * 100
                        results.append({
                            "widths": (w1, w2, w3, w4),
                            "combos": combos,
                            "unit": unit,
                            "hits": hits,
                            "hit_pct": hits / len(quaddies) * 100,
                            "avg_payout": avg_payout,
                            "roi": roi,
                            "total_return": total_return,
                        })

    # Sort by hits descending
    results.sort(key=lambda r: r["hits"], reverse=True)

    print(f"\n  TOP 20 BY HIT RATE (combos <= 100, $1+/unit):")
    print(f"  {'Widths':<15} {'Combos':>6} {'$/unit':>7} {'Hits':>5} {'Hit%':>6} {'AvgPay':>8} {'ROI':>7}")
    print("  " + "-" * 58)
    seen_combos = set()
    shown = 0
    for r in results:
        # Show best at each combo level
        if r["combos"] not in seen_combos or shown < 5:
            seen_combos.add(r["combos"])
            w = r["widths"]
            print(f"  {str(w):<15} {r['combos']:>6} ${r['unit']:>5.2f}  {r['hits']:>5} {r['hit_pct']:>5.1f}% ${r['avg_payout']:>6.0f}  {r['roi']:>5.1f}%")
            shown += 1
            if shown >= 20:
                break

    # Sort by ROI
    results.sort(key=lambda r: r["roi"], reverse=True)
    print(f"\n  TOP 20 BY ROI:")
    print(f"  {'Widths':<15} {'Combos':>6} {'$/unit':>7} {'Hits':>5} {'Hit%':>6} {'AvgPay':>8} {'ROI':>7}")
    print("  " + "-" * 58)
    for r in results[:20]:
        w = r["widths"]
        print(f"  {str(w):<15} {r['combos']:>6} ${r['unit']:>5.2f}  {r['hits']:>5} {r['hit_pct']:>5.1f}% ${r['avg_payout']:>6.0f}  {r['roi']:>5.1f}%")

    # Sort by efficiency (hit% / combos)
    results.sort(key=lambda r: r["hit_pct"] / r["combos"], reverse=True)
    print(f"\n  TOP 20 BY EFFICIENCY (hit% per combo):")
    print(f"  {'Widths':<15} {'Combos':>6} {'$/unit':>7} {'Hits':>5} {'Hit%':>6} {'AvgPay':>8} {'ROI':>7}")
    print("  " + "-" * 58)
    for r in results[:20]:
        w = r["widths"]
        print(f"  {str(w):<15} {r['combos']:>6} ${r['unit']:>5.2f}  {r['hits']:>5} {r['hit_pct']:>5.1f}% ${r['avg_payout']:>6.0f}  {r['roi']:>5.1f}%")

    # ================================================================
    # 6. SMART ALLOCATION: Assign width based on each leg's shape
    # ================================================================
    print("\n" + "=" * 90)
    print("6. SMART SHAPE-BASED ALLOCATION vs FIXED")
    print("   Per-leg width decided by odds shape, constrained to <= 100 combos")
    print("=" * 90)

    # Different width tables to test
    width_tables = {
        "Conservative": {
            "STANDOUT": 1, "DOMINANT": 1, "SHORT_PAIR": 2, "TWO_HORSE": 2,
            "CLEAR_FAV": 2, "TRIO": 3, "MID_FAV": 3, "OPEN_BUNCH": 4, "WIDE_OPEN": 4,
        },
        "Moderate": {
            "STANDOUT": 1, "DOMINANT": 2, "SHORT_PAIR": 2, "TWO_HORSE": 3,
            "CLEAR_FAV": 3, "TRIO": 3, "MID_FAV": 4, "OPEN_BUNCH": 4, "WIDE_OPEN": 5,
        },
        "Aggressive": {
            "STANDOUT": 1, "DOMINANT": 2, "SHORT_PAIR": 3, "TWO_HORSE": 3,
            "CLEAR_FAV": 3, "TRIO": 4, "MID_FAV": 4, "OPEN_BUNCH": 5, "WIDE_OPEN": 6,
        },
        "Ultra-Wide": {
            "STANDOUT": 2, "DOMINANT": 3, "SHORT_PAIR": 3, "TWO_HORSE": 4,
            "CLEAR_FAV": 4, "TRIO": 5, "MID_FAV": 5, "OPEN_BUNCH": 6, "WIDE_OPEN": 6,
        },
    }

    print(f"\n{'Strategy':<20} {'Hits':>5} {'Hit%':>6} {'AvgCombos':>10} {'AvgUnit':>8} {'AvgPay':>8} {'ROI':>7} {'MinUnit':>8}")
    print("-" * 80)

    # Fixed strategies for comparison
    for fw in [2, 3, 4, 5]:
        combos = fw ** 4
        if combos > 100 and fw > 3:
            continue
        unit = 100.0 / combos
        hits = 0
        total_ret = 0
        for q in quaddies:
            if all(q[i]["winner_rank"] <= fw for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * unit
        total_cost = 100.0 * len(quaddies)
        roi = (total_ret - total_cost) / total_cost * 100
        avg_pay = total_ret / hits if hits else 0
        print(f"  Fixed {fw}-{fw}-{fw}-{fw}      {hits:>5} {hits/len(quaddies)*100:>5.1f}%  {combos:>9}  ${unit:>6.2f}  ${avg_pay:>6.0f}  {roi:>5.1f}%  ${unit:>6.2f}")

    # Shape-based strategies
    for name, table in width_tables.items():
        hits = 0
        total_ret = 0
        total_combos = 0
        min_unit = 999
        for q in quaddies:
            widths = [table[classify_shape(leg)] for leg in q]
            combos = 1
            for w in widths:
                combos *= w
            # Cap at 100 combos — tighten easiest leg if over
            while combos > 100:
                easiest_idx = min(range(4), key=lambda i: q[i]["p1"])
                if widths[easiest_idx] > 1:
                    widths[easiest_idx] -= 1
                    combos = 1
                    for w in widths:
                        combos *= w
                else:
                    break
            unit = 100.0 / combos
            min_unit = min(min_unit, unit)
            total_combos += combos
            all_hit = all(q[i]["winner_rank"] <= widths[i] for i in range(4))
            if all_hit:
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * unit
        total_cost = 100.0 * len(quaddies)
        roi = (total_ret - total_cost) / total_cost * 100
        avg_combos = total_combos / len(quaddies)
        avg_pay = total_ret / hits if hits else 0
        avg_unit = 100.0 / avg_combos if avg_combos else 0
        print(f"  {name:<18} {hits:>5} {hits/len(quaddies)*100:>5.1f}%  {avg_combos:>9.1f}  ${avg_unit:>6.2f}  ${avg_pay:>6.0f}  {roi:>5.1f}%  ${min_unit:>6.2f}")

    # ================================================================
    # 7. THE $100 UNIT PRICE SWEET SPOT
    # ================================================================
    print("\n" + "=" * 90)
    print("7. UNIT PRICE vs PROFITABILITY")
    print("   Is it better to have fewer combos (bigger unit) or more combos (more hits)?")
    print("=" * 90)

    # Group all results by combo range
    combo_ranges = [
        ("1-4 combos ($25-100/unit)", 1, 4),
        ("5-8 combos ($12-20/unit)", 5, 8),
        ("9-16 combos ($6-11/unit)", 9, 16),
        ("17-36 combos ($2.78-5.88/unit)", 17, 36),
        ("37-64 combos ($1.56-2.70/unit)", 37, 64),
        ("65-100 combos ($1.00-1.54/unit)", 65, 100),
    ]

    # Collect all fixed-width results for these ranges
    all_results_by_range = defaultdict(list)
    for r in results:
        for label, lo, hi in combo_ranges:
            if lo <= r["combos"] <= hi:
                all_results_by_range[label].append(r)
                break

    print(f"\n{'Combo Range':<40} {'Best Hits':>10} {'Best ROI':>10} {'Best Config':>15}")
    print("-" * 80)
    for label, lo, hi in combo_ranges:
        rng = all_results_by_range.get(label, [])
        if not rng:
            continue
        best_hit = max(rng, key=lambda r: r["hits"])
        best_roi = max(rng, key=lambda r: r["roi"])
        print(f"  {label:<38} {best_hit['hits']:>4} ({best_hit['hit_pct']:.1f}%)  {best_roi['roi']:>7.1f}%  {str(best_roi['widths'])}")

    # ================================================================
    # 8. WHEN DOES RUNNER 5 AND 6 ACTUALLY WIN? — Price profile
    # ================================================================
    print("\n" + "=" * 90)
    print("8. RUNNER 5 AND 6 WINNERS: What do they look like?")
    print("   When the 5th or 6th favourite wins, what's the race profile?")
    print("=" * 90)

    r5_wins = [l for l in all_legs if l["winner_rank"] == 5]
    r6_wins = [l for l in all_legs if l["winner_rank"] == 6]

    for label, wins in [("Rank 5 winners", r5_wins), ("Rank 6 winners", r6_wins)]:
        n = len(wins)
        print(f"\n  {label}: {n} legs ({n/len(all_legs)*100:.1f}% of all legs)")
        if n < 10:
            continue
        avg_p1 = sum(l["p1"] for l in wins) / n
        avg_p2 = sum(l["p2"] for l in wins) / n
        avg_p3 = sum(l["p3"] for l in wins) / n
        avg_p4 = sum(l["p4"] for l in wins) / n
        avg_p5 = sum(l["p5"] for l in wins) / n
        avg_field = sum(l["field_size"] for l in wins) / n
        avg_winner_sp = sum(l["winner_sp"] for l in wins) / n
        print(f"    Avg prices: ${avg_p1:.1f} / ${avg_p2:.1f} / ${avg_p3:.1f} / ${avg_p4:.1f} / ${avg_p5:.1f}")
        print(f"    Avg winner SP: ${avg_winner_sp:.1f}")
        print(f"    Avg field: {avg_field:.1f}")

        # Shape distribution
        shape_counts = defaultdict(int)
        for l in wins:
            shape_counts[classify_shape(l)] += 1
        print(f"    Race shapes when {label}:")
        for shape in shapes_order:
            cnt = shape_counts.get(shape, 0)
            if cnt > 0:
                print(f"      {shape}: {cnt} ({cnt/n*100:.0f}%)")

    # ================================================================
    # 9. CONSECUTIVE PRICE GAPS — Where is the natural cutoff?
    # ================================================================
    print("\n" + "=" * 90)
    print("9. NATURAL CUTOFFS: Where is the biggest price gap?")
    print("   If there's a gap between runner 3 and 4, use 3. If between 4 and 5, use 4.")
    print("=" * 90)

    # For each race, find the biggest ratio gap in the top 6
    gap_at = defaultdict(lambda: {"total": 0, "captured": 0})
    for l in all_legs:
        # Find the biggest ratio jump in top 6
        max_ratio = 0
        max_pos = 1
        for i in range(min(5, len(l["ratios"]))):
            if l["ratios"][i] > max_ratio and l["ratios"][i] < 50:
                max_ratio = l["ratios"][i]
                max_pos = i + 1  # gap is AFTER position i+1

        # Width = position of the gap (include everyone before the gap)
        width = max_pos
        gap_at[width]["total"] += 1
        if l["winner_rank"] <= width:
            gap_at[width]["captured"] += 1

    print(f"\n  {'Biggest gap after':>20} {'Legs':>6} {'Capture if width=gap':>22}")
    print("  " + "-" * 52)
    for pos in sorted(gap_at.keys()):
        d = gap_at[pos]
        if d["total"] < 20:
            continue
        print(f"  After runner {pos:>2}        {d['total']:>5}  {d['captured']}/{d['total']} = {d['captured']/d['total']*100:.1f}%")

    # ================================================================
    # 10. GAP-BASED WIDTH STRATEGY
    # ================================================================
    print("\n" + "=" * 90)
    print("10. GAP-BASED STRATEGY: Use the natural price gap to set width")
    print("    Width = position of biggest price ratio jump (capped 2-6)")
    print("=" * 90)

    def gap_width(leg, min_w=2, max_w=6):
        """Width = biggest price gap position, capped."""
        max_ratio = 0
        max_pos = 2
        for i in range(min(5, len(leg["ratios"]))):
            r = leg["ratios"][i]
            if r > max_ratio and r < 50:
                max_ratio = r
                max_pos = i + 1
        return max(min_w, min(max_pos, max_w))

    def gap_width_with_floor(leg):
        """Gap-based but with shape floor."""
        shape = classify_shape(leg)
        # Minimum widths by shape
        floors = {
            "STANDOUT": 1, "DOMINANT": 1, "SHORT_PAIR": 2, "TWO_HORSE": 2,
            "CLEAR_FAV": 2, "TRIO": 3, "MID_FAV": 2, "OPEN_BUNCH": 3, "WIDE_OPEN": 3,
        }
        floor = floors.get(shape, 2)
        gap_w = gap_width(leg, min_w=1, max_w=6)
        return max(floor, gap_w)

    # Test gap strategies
    gap_strategies = {
        "Gap (2-4 cap)": lambda l: gap_width(l, 2, 4),
        "Gap (2-5 cap)": lambda l: gap_width(l, 2, 5),
        "Gap (2-6 cap)": lambda l: gap_width(l, 2, 6),
        "Gap (1-6 cap)": lambda l: gap_width(l, 1, 6),
        "Gap + Shape floor": gap_width_with_floor,
    }

    print(f"\n{'Strategy':<22} {'Hits':>5} {'Hit%':>6} {'AvgComb':>8} {'AvgUnit':>8} {'AvgPay':>8} {'ROI':>7} {'MinUnit':>8}")
    print("-" * 80)

    for name, fn in gap_strategies.items():
        hits = 0
        total_ret = 0
        total_combos = 0
        min_unit = 999
        for q in quaddies:
            widths = [fn(leg) for leg in q]
            combos = 1
            for w in widths:
                combos *= w
            # Cap at 100
            while combos > 100:
                # Tighten leg with smallest p1 (easiest)
                easiest = min(range(4), key=lambda i: q[i]["p1"])
                if widths[easiest] > 1:
                    widths[easiest] -= 1
                    combos = 1
                    for w in widths:
                        combos *= w
                else:
                    break
            unit = 100.0 / combos
            min_unit = min(min_unit, unit)
            total_combos += combos
            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * unit
        total_cost = 100.0 * len(quaddies)
        roi = (total_ret - total_cost) / total_cost * 100
        avg_combos = total_combos / len(quaddies)
        avg_pay = total_ret / hits if hits else 0
        avg_unit = 100.0 / avg_combos
        print(f"  {name:<20} {hits:>5} {hits/len(quaddies)*100:>5.1f}%  {avg_combos:>7.1f}  ${avg_unit:>6.2f}  ${avg_pay:>6.0f}  {roi:>5.1f}%  ${min_unit:>6.2f}")

    # ================================================================
    # 11. QUADDIE PAYOUT DISTRIBUTION
    # ================================================================
    print("\n" + "=" * 90)
    print("11. WHAT DO QUADDIES ACTUALLY PAY? (estimated from SP product)")
    print("=" * 90)

    # Calculate estimated payout for each quaddie
    payouts = []
    for q in quaddies:
        sp_prod = 1
        for leg in q:
            sp_prod *= leg["winner_sp"]
        est_payout = sp_prod * 0.80  # 80% pool return, per $1 unit
        payouts.append(est_payout)

    payouts.sort()
    n = len(payouts)
    print(f"\n  Based on {n} quaddies (SP product * 80% pool return, per $1 unit):")
    print(f"    Min:      ${payouts[0]:.0f}")
    print(f"    10th pct: ${payouts[int(n*0.10)]:.0f}")
    print(f"    25th pct: ${payouts[int(n*0.25)]:.0f}")
    print(f"    Median:   ${payouts[int(n*0.50)]:.0f}")
    print(f"    75th pct: ${payouts[int(n*0.75)]:.0f}")
    print(f"    90th pct: ${payouts[int(n*0.90)]:.0f}")
    print(f"    Max:      ${payouts[-1]:.0f}")

    # Break into payout ranges
    print(f"\n  Payout distribution (per $1 unit):")
    payout_ranges = [
        ("$1-$50", 1, 50),
        ("$51-$100", 51, 100),
        ("$101-$250", 101, 250),
        ("$251-$500", 251, 500),
        ("$501-$1000", 501, 1000),
        ("$1001-$5000", 1001, 5000),
        ("$5000+", 5001, 999999),
    ]
    for label, lo, hi in payout_ranges:
        cnt = sum(1 for p in payouts if lo <= p <= hi)
        print(f"    {label:<15}: {cnt:>5} ({cnt/n*100:.1f}%)")

    # ================================================================
    # 12. BREAKEVEN ANALYSIS
    # ================================================================
    print("\n" + "=" * 90)
    print("12. BREAKEVEN: How often must we hit to break even at $100?")
    print("=" * 90)

    median_payout = payouts[int(n * 0.50)]
    p25_payout = payouts[int(n * 0.25)]
    p75_payout = payouts[int(n * 0.75)]

    for combos_label, combos in [("12 combos", 12), ("24 combos", 24), ("36 combos", 36), ("48 combos", 48), ("72 combos", 72)]:
        unit = 100.0 / combos
        # Need: hit_rate * avg_payout_per_unit * unit >= $100
        # hit_rate >= $100 / (median_payout * unit) = combos / median_payout
        be_median = combos / median_payout * 100
        be_p25 = combos / p25_payout * 100
        print(f"  {combos_label}: ${unit:.2f}/unit. Need {be_median:.1f}% hit rate (median payout) or {be_p25:.1f}% (if unlucky, 25th pct)")

    # ================================================================
    # 13. EARLY QUADDIE COMPARISON
    # ================================================================
    print("\n" + "=" * 90)
    print("13. EARLY QUADDIE: How does it compare?")
    print("=" * 90)

    if eq_quaddies:
        print(f"\n  Early quaddies: {len(eq_quaddies)}")
        for w in range(1, 7):
            combos = w ** 4
            if combos > 100 and w > 3:
                continue
            unit = 100.0 / combos
            hits = sum(1 for q in eq_quaddies if all(q[i]["winner_rank"] <= w for i in range(4)))
            hit_pct = hits / len(eq_quaddies) * 100
            total_ret = 0
            for q in eq_quaddies:
                if all(q[i]["winner_rank"] <= w for i in range(4)):
                    sp_prod = 1
                    for leg in q:
                        sp_prod *= leg["winner_sp"]
                    total_ret += sp_prod * 0.80 * unit
            avg_pay = total_ret / hits if hits else 0
            print(f"    Width {w}: {hits}/{len(eq_quaddies)} = {hit_pct:.1f}% hit rate, {combos} combos, ${unit:.2f}/unit, avg payout ${avg_pay:.0f}")

        # Shape distribution in EQ vs Q
        eq_shapes = defaultdict(int)
        q_shapes = defaultdict(int)
        for q in eq_quaddies:
            for leg in q:
                eq_shapes[classify_shape(leg)] += 1
        for q in quaddies:
            for leg in q:
                q_shapes[classify_shape(leg)] += 1
        total_eq = sum(eq_shapes.values())
        total_q = sum(q_shapes.values())

        print(f"\n  Shape distribution (EQ vs Main Q):")
        print(f"  {'Shape':<14} {'EQ%':>6} {'Q%':>6}")
        print("  " + "-" * 28)
        for shape in shapes_order:
            eq_pct = eq_shapes.get(shape, 0) / total_eq * 100 if total_eq else 0
            q_pct = q_shapes.get(shape, 0) / total_q * 100 if total_q else 0
            if eq_pct > 0.5 or q_pct > 0.5:
                print(f"  {shape:<14} {eq_pct:>5.1f}% {q_pct:>5.1f}%")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
