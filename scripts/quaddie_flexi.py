"""Quaddie analysis with FLEXI framing.

$100 budget. Flexi% = $100 / combos.
Minimum 30% flexi = max 333 combos.
Higher flexi = bigger payout per combo hit.

Key question: given 333 combo budget, how do we allocate width
across legs to maximise hit rate while keeping flexi >= 30%?
"""
import json
from collections import defaultdict
from pathlib import Path

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
        "p5": prices[4], "p6": prices[5], "p7": prices[6],
        "ratios": ratios,
        "ratio_2_1": ratios[0], "ratio_3_2": ratios[1], "ratio_4_3": ratios[2],
        "ratio_5_4": ratios[3], "ratio_6_5": ratios[4],
        "race_class": race.get("RaceClass", "") or "",
        "race_number": race.get("Number", 0),
    }


def classify_shape(leg):
    p1, p2, p3, p4 = leg["p1"], leg["p2"], leg["p3"], leg["p4"]
    if p1 <= 1.50:
        return "STANDOUT"
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

    quaddies = []
    eq_quaddies = []
    all_legs = []

    for m in meetings:
        races = m["races"]
        n = len(races)

        q_legs = []
        for i, r in enumerate(races[n - 4:]):
            a = analyse_race(r)
            if a:
                a["leg_pos"] = i + 1
                all_legs.append(a)
                q_legs.append(a)
        if len(q_legs) == 4:
            quaddies.append(q_legs)

        if n >= 8:
            eq_legs = []
            for i, r in enumerate(races[:4]):
                a = analyse_race(r)
                if a:
                    a["leg_pos"] = i + 1
                    eq_legs.append(a)
            if len(eq_legs) == 4:
                eq_quaddies.append(eq_legs)

    print(f"Quaddies: {len(quaddies)}, Early Quaddies: {len(eq_quaddies)}")
    print(f"Total legs: {len(all_legs)}")

    # ================================================================
    # 1. FLEXI FRAMING — Fixed widths
    # ================================================================
    print("\n" + "=" * 90)
    print("1. FIXED WIDTH QUADDIES — Flexi at $100")
    print("=" * 90)

    print(f"\n{'Width':<8} {'Combos':>7} {'Flexi%':>8} {'Hits':>5} {'Hit%':>6} {'AvgPayout':>10} {'EstROI':>8}")
    print("-" * 60)
    for w in range(1, 8):
        combos = w ** 4
        flexi = 100.0 / combos * 100
        hits = sum(1 for q in quaddies if all(q[i]["winner_rank"] <= w for i in range(4)))
        hit_pct = hits / len(quaddies) * 100
        # Payout = SP_product * pool_return * flexi
        total_ret = 0
        for q in quaddies:
            if all(q[i]["winner_rank"] <= w for i in range(4)):
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * (100.0 / combos)
        avg_pay = total_ret / hits if hits else 0
        total_cost = 100.0 * len(quaddies)
        roi = (total_ret - total_cost) / total_cost * 100
        marker = " <-- 30% floor" if abs(flexi - 30) < 15 else ""
        print(f"  {w:<6}  {combos:>6}  {flexi:>6.1f}%  {hits:>5} {hit_pct:>5.1f}%  ${avg_pay:>8.0f}  {roi:>6.1f}%{marker}")

    # ================================================================
    # 2. ALL MIXED WIDTHS 1-7, max 333 combos (30% flexi floor)
    # ================================================================
    print("\n" + "=" * 90)
    print("2. BEST MIXED WIDTH QUADDIES (flexi >= 30%, max 333 combos)")
    print("   Exhaustive search of all width combos 1-7 across 4 legs")
    print("=" * 90)

    results = []
    for w1 in range(1, 8):
        for w2 in range(1, 8):
            for w3 in range(1, 8):
                for w4 in range(1, 8):
                    combos = w1 * w2 * w3 * w4
                    if combos > 333:
                        continue
                    flexi = 100.0 / combos * 100
                    hits = 0
                    total_ret = 0
                    for q in quaddies:
                        if all(q[i]["winner_rank"] <= [w1, w2, w3, w4][i] for i in range(4)):
                            hits += 1
                            sp_prod = 1
                            for leg in q:
                                sp_prod *= leg["winner_sp"]
                            total_ret += sp_prod * 0.80 * (100.0 / combos)
                    if hits > 0:
                        results.append({
                            "widths": (w1, w2, w3, w4),
                            "combos": combos,
                            "flexi": flexi,
                            "hits": hits,
                            "hit_pct": hits / len(quaddies) * 100,
                            "avg_payout": total_ret / hits,
                            "roi": (total_ret - 100.0 * len(quaddies)) / (100.0 * len(quaddies)) * 100,
                            "total_ret": total_ret,
                        })

    # Deduplicate by sorted widths (2,3,4,5 same as 3,2,5,4 for fixed analysis)
    # Actually NO - leg position matters. Keep all.

    # Sort by hit rate
    results.sort(key=lambda r: r["hits"], reverse=True)
    print(f"\n  TOP 25 BY HIT RATE (flexi >= 30%):")
    print(f"  {'Widths':<15} {'Combos':>6} {'Flexi':>7} {'Hits':>5} {'Hit%':>6} {'AvgPay':>8} {'ROI':>7}")
    print("  " + "-" * 60)
    for r in results[:25]:
        print(f"  {str(r['widths']):<15} {r['combos']:>6} {r['flexi']:>5.0f}%  {r['hits']:>5} {r['hit_pct']:>5.1f}% ${r['avg_payout']:>6.0f}  {r['roi']:>5.1f}%")

    # Sort by ROI
    results.sort(key=lambda r: r["roi"], reverse=True)
    print(f"\n  TOP 25 BY ROI:")
    print(f"  {'Widths':<15} {'Combos':>6} {'Flexi':>7} {'Hits':>5} {'Hit%':>6} {'AvgPay':>8} {'ROI':>7}")
    print("  " + "-" * 60)
    for r in results[:25]:
        print(f"  {str(r['widths']):<15} {r['combos']:>6} {r['flexi']:>5.0f}%  {r['hits']:>5} {r['hit_pct']:>5.1f}% ${r['avg_payout']:>6.0f}  {r['roi']:>5.1f}%")

    # Best at each combo bucket
    print(f"\n  BEST CONFIG AT EACH FLEXI LEVEL:")
    print(f"  {'Flexi Range':<20} {'BestConfig':<15} {'Combos':>6} {'Hits':>5} {'Hit%':>6} {'AvgPay':>8} {'ROI':>7}")
    print("  " + "-" * 70)
    flexi_ranges = [
        ("100%+ (tight)", 100, 9999),
        ("50-99%", 50, 99.9),
        ("30-49%", 30, 49.9),
    ]
    for label, flo, fhi in flexi_ranges:
        bucket = [r for r in results if flo <= r["flexi"] <= fhi]
        if not bucket:
            continue
        # Best by ROI in this bucket
        best = max(bucket, key=lambda r: r["roi"])
        # Also best by hits
        most_hits = max(bucket, key=lambda r: r["hits"])
        print(f"  {label:<20} {str(best['widths']):<15} {best['combos']:>6} {best['hits']:>5} {best['hit_pct']:>5.1f}% ${best['avg_payout']:>6.0f}  {best['roi']:>5.1f}%  (best ROI)")
        if most_hits["widths"] != best["widths"]:
            print(f"  {'':20} {str(most_hits['widths']):<15} {most_hits['combos']:>6} {most_hits['hits']:>5} {most_hits['hit_pct']:>5.1f}% ${most_hits['avg_payout']:>6.0f}  {most_hits['roi']:>5.1f}%  (most hits)")

    # ================================================================
    # 3. SHAPE-BASED STRATEGIES with 333 combo cap
    # ================================================================
    print("\n" + "=" * 90)
    print("3. SHAPE-BASED SMART STRATEGIES (flexi >= 30%)")
    print("   Width per leg decided by odds shape, total capped at 333 combos")
    print("=" * 90)

    width_tables = {
        "Tight (1-3)": {
            "STANDOUT": 1, "DOMINANT": 1, "SHORT_PAIR": 2, "TWO_HORSE": 2,
            "CLEAR_FAV": 2, "TRIO": 3, "MID_FAV": 3, "OPEN_BUNCH": 3, "WIDE_OPEN": 3,
        },
        "Standard (1-4)": {
            "STANDOUT": 1, "DOMINANT": 1, "SHORT_PAIR": 2, "TWO_HORSE": 3,
            "CLEAR_FAV": 3, "TRIO": 4, "MID_FAV": 3, "OPEN_BUNCH": 4, "WIDE_OPEN": 4,
        },
        "Wide (1-5)": {
            "STANDOUT": 1, "DOMINANT": 2, "SHORT_PAIR": 2, "TWO_HORSE": 3,
            "CLEAR_FAV": 3, "TRIO": 4, "MID_FAV": 4, "OPEN_BUNCH": 5, "WIDE_OPEN": 5,
        },
        "Ultra (1-6)": {
            "STANDOUT": 1, "DOMINANT": 2, "SHORT_PAIR": 3, "TWO_HORSE": 3,
            "CLEAR_FAV": 4, "TRIO": 5, "MID_FAV": 5, "OPEN_BUNCH": 6, "WIDE_OPEN": 6,
        },
        "Marginal >7%": {
            # Include a runner only if marginal capture > 7%
            "STANDOUT": 1, "DOMINANT": 2, "SHORT_PAIR": 2, "TWO_HORSE": 3,
            "CLEAR_FAV": 3, "TRIO": 5, "MID_FAV": 4, "OPEN_BUNCH": 6, "WIDE_OPEN": 5,
        },
    }

    print(f"\n{'Strategy':<18} {'Hits':>5} {'Hit%':>6} {'AvgCmb':>7} {'AvgFlexi':>9} {'MinFlexi':>9} {'AvgPay':>8} {'ROI':>7} {'Over30':>7}")
    print("-" * 85)

    for name, table in width_tables.items():
        hits = 0
        total_ret = 0
        total_combos = 0
        min_flexi = 9999
        over_30_count = 0
        under_30_count = 0
        for q in quaddies:
            widths = [table[classify_shape(leg)] for leg in q]
            combos = 1
            for w in widths:
                combos *= w
            # Cap at 333 combos — tighten easiest leg
            while combos > 333:
                easiest_idx = min(range(4), key=lambda i: q[i]["p1"])
                if widths[easiest_idx] > 1:
                    widths[easiest_idx] -= 1
                    combos = 1
                    for w in widths:
                        combos *= w
                else:
                    # Tighten next easiest
                    sorted_legs = sorted(range(4), key=lambda i: q[i]["p1"])
                    reduced = False
                    for idx in sorted_legs:
                        if widths[idx] > 1:
                            widths[idx] -= 1
                            combos = 1
                            for w in widths:
                                combos *= w
                            reduced = True
                            break
                    if not reduced:
                        break

            flexi = 100.0 / combos * 100
            min_flexi = min(min_flexi, flexi)
            if flexi >= 30:
                over_30_count += 1
            else:
                under_30_count += 1
            total_combos += combos

            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * (100.0 / combos)

        total_cost = 100.0 * len(quaddies)
        roi = (total_ret - total_cost) / total_cost * 100
        avg_combos = total_combos / len(quaddies)
        avg_pay = total_ret / hits if hits else 0
        avg_flexi = 100.0 / avg_combos * 100
        pct_over = over_30_count / len(quaddies) * 100
        print(f"  {name:<16} {hits:>5} {hits/len(quaddies)*100:>5.1f}%  {avg_combos:>6.0f}  {avg_flexi:>7.0f}%  {min_flexi:>7.0f}%  ${avg_pay:>6.0f}  {roi:>5.1f}%  {pct_over:>5.0f}%")

    # ================================================================
    # 4. PER-SHAPE: What width gives best marginal value?
    # ================================================================
    print("\n" + "=" * 90)
    print("4. MARGINAL ANALYSIS: At what width does each shape drop below 7%?")
    print("   This determines the 'natural' width per leg")
    print("=" * 90)

    shapes_order = ["STANDOUT", "DOMINANT", "SHORT_PAIR", "TWO_HORSE", "CLEAR_FAV",
                     "TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"]

    shape_buckets = defaultdict(list)
    for l in all_legs:
        shape_buckets[classify_shape(l)].append(l)

    print(f"\n{'Shape':<14} {'N':>5} | {'R1':>5} {'R2':>5} {'R3':>5} {'R4':>5} {'R5':>5} {'R6':>5} {'R7':>5} | {'Rec':>4} {'Cum@Rec':>8}")
    print("-" * 85)
    shape_recommended = {}
    for shape in shapes_order:
        legs = shape_buckets.get(shape, [])
        n = len(legs)
        if n < 15:
            continue
        marginals = []
        cum = 0
        rec_width = 1
        for w in range(1, 8):
            m = sum(1 for l in legs if l["winner_rank"] == w) / n * 100
            marginals.append(m)
            cum += m
            if m >= 7:
                rec_width = w
        cum_at_rec = sum(marginals[:rec_width])

        shape_recommended[shape] = rec_width
        vals = " ".join(f"{m:>4.0f}%" for m in marginals)
        print(f"  {shape:<12} {n:>5} | {vals} | {rec_width:>3}  {cum_at_rec:>6.0f}%")

    print(f"\n  Recommended widths (marginal >= 7% cutoff):")
    for shape in shapes_order:
        w = shape_recommended.get(shape)
        if w:
            desc = {
                "STANDOUT": "Fav <= $1.50",
                "DOMINANT": "Fav $1.50-$2, big gap to 2nd",
                "SHORT_PAIR": "Fav $1.50-$2, 2nd close",
                "TWO_HORSE": "Fav $2-$3, 2nd within 1.5x",
                "CLEAR_FAV": "Fav $2.50-$3.50",
                "TRIO": "Fav $3.50-$5, top 3 bunched",
                "MID_FAV": "Fav $3.50-$5, 3rd distant",
                "OPEN_BUNCH": "Fav $5+, all bunched",
                "WIDE_OPEN": "Fav $5+, spread",
            }
            print(f"    {shape:<14} -> {w} runners  ({desc.get(shape, '')})")

    # ================================================================
    # 5. SIMULATE WITH RECOMMENDED WIDTHS
    # ================================================================
    print("\n" + "=" * 90)
    print("5. SIMULATION WITH RECOMMENDED WIDTHS (marginal >= 7%)")
    print("   Per-leg width from section 4, capped at 333 combos")
    print("=" * 90)

    combo_dist = defaultdict(int)
    flexi_dist = defaultdict(int)
    hits = 0
    total_ret = 0
    total_combos_all = 0
    quad_details = []

    for q in quaddies:
        shapes = [classify_shape(leg) for leg in q]
        widths = [shape_recommended.get(s, 2) for s in shapes]
        combos = 1
        for w in widths:
            combos *= w

        # Cap at 333 (30% flexi floor)
        while combos > 333:
            easiest_idx = min(range(4), key=lambda i: q[i]["p1"])
            sorted_legs = sorted(range(4), key=lambda i: q[i]["p1"])
            reduced = False
            for idx in sorted_legs:
                if widths[idx] > 1:
                    widths[idx] -= 1
                    combos = 1
                    for w in widths:
                        combos *= w
                    reduced = True
                    break
            if not reduced:
                break

        flexi = 100.0 / combos * 100
        total_combos_all += combos

        all_hit = all(q[i]["winner_rank"] <= widths[i] for i in range(4))
        if all_hit:
            hits += 1
            sp_prod = 1
            for leg in q:
                sp_prod *= leg["winner_sp"]
            payout = sp_prod * 0.80 * (100.0 / combos)
            total_ret += payout
        else:
            payout = 0

        # Track combo distribution
        if combos <= 10:
            combo_dist["1-10"] += 1
        elif combos <= 30:
            combo_dist["11-30"] += 1
        elif combos <= 80:
            combo_dist["31-80"] += 1
        elif combos <= 150:
            combo_dist["81-150"] += 1
        elif combos <= 250:
            combo_dist["151-250"] += 1
        else:
            combo_dist["251-333"] += 1

        # Track flexi distribution
        if flexi >= 200:
            flexi_dist["200%+"] += 1
        elif flexi >= 100:
            flexi_dist["100-199%"] += 1
        elif flexi >= 50:
            flexi_dist["50-99%"] += 1
        elif flexi >= 30:
            flexi_dist["30-49%"] += 1
        else:
            flexi_dist["<30%"] += 1

        quad_details.append({
            "shapes": shapes, "widths": widths, "combos": combos,
            "flexi": flexi, "hit": all_hit, "payout": payout,
        })

    total_cost = 100.0 * len(quaddies)
    roi = (total_ret - total_cost) / total_cost * 100
    avg_combos = total_combos_all / len(quaddies)
    avg_flexi = 100.0 / avg_combos * 100
    avg_pay = total_ret / hits if hits else 0

    print(f"\n  Hit rate: {hits}/{len(quaddies)} = {hits/len(quaddies)*100:.1f}%")
    print(f"  Avg combos: {avg_combos:.0f}  (avg flexi: {avg_flexi:.0f}%)")
    print(f"  Avg payout when hit: ${avg_pay:.0f}")
    print(f"  ROI: {roi:.1f}%")

    print(f"\n  Combo distribution:")
    for bucket in ["1-10", "11-30", "31-80", "81-150", "151-250", "251-333"]:
        cnt = combo_dist.get(bucket, 0)
        print(f"    {bucket:>10}: {cnt:>5} ({cnt/len(quaddies)*100:.1f}%)")

    print(f"\n  Flexi distribution:")
    for bucket in ["200%+", "100-199%", "50-99%", "30-49%", "<30%"]:
        cnt = flexi_dist.get(bucket, 0)
        print(f"    {bucket:>10}: {cnt:>5} ({cnt/len(quaddies)*100:.1f}%)")

    # ================================================================
    # 6. COMPARE: RECOMMENDED vs FIXED vs ALTERNATIVES
    # ================================================================
    print("\n" + "=" * 90)
    print("6. HEAD-TO-HEAD COMPARISON (all capped at 333 combos)")
    print("=" * 90)

    strategies = {}

    # Fixed widths
    for fw in [2, 3, 4, 5]:
        strategies[f"Fixed {fw}x{fw}x{fw}x{fw}"] = lambda q, w=fw: [w, w, w, w]

    # Shape-based recommended (from section 4)
    strategies["Recommended (>7%)"] = lambda q: [shape_recommended.get(classify_shape(leg), 2) for leg in q]

    # Tighter version: only include if marginal >= 10%
    shape_tight = {
        "STANDOUT": 1, "DOMINANT": 1, "SHORT_PAIR": 2, "TWO_HORSE": 2,
        "CLEAR_FAV": 2, "TRIO": 3, "MID_FAV": 2, "OPEN_BUNCH": 3, "WIDE_OPEN": 3,
    }
    strategies["Tight (>10%)"] = lambda q: [shape_tight.get(classify_shape(leg), 2) for leg in q]

    # Gap-based: width = where biggest price jump is
    def gap_strategy(q):
        widths = []
        for leg in q:
            max_ratio = 0
            max_pos = 2
            for i in range(min(6, len(leg["ratios"]))):
                r = leg["ratios"][i]
                if r > max_ratio and r < 50:
                    max_ratio = r
                    max_pos = i + 1
            widths.append(max(2, min(max_pos, 7)))
        return widths
    strategies["Price Gap"] = gap_strategy

    # Hybrid: use shape recommendation but bump up if price gap says wider
    def hybrid_strategy(q):
        widths = []
        for leg in q:
            shape_w = shape_recommended.get(classify_shape(leg), 2)
            # Check price gap
            max_ratio = 0
            max_pos = 2
            for i in range(min(6, len(leg["ratios"]))):
                r = leg["ratios"][i]
                if r > max_ratio and r < 50:
                    max_ratio = r
                    max_pos = i + 1
            gap_w = max(2, min(max_pos, 7))
            # Use the LARGER of shape or gap
            widths.append(max(shape_w, gap_w))
        return widths
    strategies["Hybrid (shape+gap)"] = hybrid_strategy

    print(f"\n{'Strategy':<25} {'Hits':>5} {'Hit%':>6} {'AvgCmb':>7} {'AvgFlexi':>9} {'AvgPay':>8} {'ROI':>7}")
    print("-" * 75)

    for name, fn in strategies.items():
        hits = 0
        total_ret = 0
        total_combos = 0
        for q in quaddies:
            widths = fn(q)
            combos = 1
            for w in widths:
                combos *= w
            # Cap at 333
            while combos > 333:
                sorted_legs = sorted(range(4), key=lambda i: q[i]["p1"])
                reduced = False
                for idx in sorted_legs:
                    if widths[idx] > 1:
                        widths[idx] -= 1
                        combos = 1
                        for w in widths:
                            combos *= w
                        reduced = True
                        break
                if not reduced:
                    break
            total_combos += combos
            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * (100.0 / combos)
        total_cost = 100.0 * len(quaddies)
        roi = (total_ret - total_cost) / total_cost * 100
        avg_combos = total_combos / len(quaddies)
        avg_pay = total_ret / hits if hits else 0
        avg_flexi = 100.0 / avg_combos * 100
        print(f"  {name:<23} {hits:>5} {hits/len(quaddies)*100:>5.1f}%  {avg_combos:>6.0f}  {avg_flexi:>7.0f}%  ${avg_pay:>6.0f}  {roi:>5.1f}%")

    # ================================================================
    # 7. THE SKINNY QUAD BONUS — When all legs are STANDOUT/DOMINANT
    # ================================================================
    print("\n" + "=" * 90)
    print("7. THE SKINNY QUAD: When 2+ legs are standouts")
    print("   These are the high-flexi goldmines")
    print("=" * 90)

    for min_locks in [0, 1, 2, 3, 4]:
        qualifying = []
        for q in quaddies:
            locks = sum(1 for leg in q if classify_shape(leg) in ("STANDOUT", "DOMINANT"))
            if locks >= min_locks:
                qualifying.append(q)
        if not qualifying:
            continue
        # Simulate with recommended widths
        hits = 0
        total_ret = 0
        total_combos = 0
        for q in qualifying:
            widths = [shape_recommended.get(classify_shape(leg), 2) for leg in q]
            combos = 1
            for w in widths:
                combos *= w
            total_combos += combos
            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * (100.0 / combos)
        avg_combos = total_combos / len(qualifying)
        avg_flexi = 100.0 / avg_combos * 100
        hit_pct = hits / len(qualifying) * 100 if qualifying else 0
        total_cost = 100.0 * len(qualifying)
        roi = (total_ret - total_cost) / total_cost * 100 if total_cost else 0
        avg_pay = total_ret / hits if hits else 0
        print(f"  {min_locks}+ locks: {len(qualifying):>5} quaddies, {hits:>4} hits ({hit_pct:.1f}%), avg {avg_combos:.0f} combos ({avg_flexi:.0f}% flexi), avg pay ${avg_pay:.0f}, ROI {roi:.1f}%")

    # ================================================================
    # 8. COMBO SHAPE PATTERNS — What shape combos are common?
    # ================================================================
    print("\n" + "=" * 90)
    print("8. COMMON QUADDIE SHAPE PATTERNS")
    print("   Most frequent combinations of leg shapes")
    print("=" * 90)

    pattern_stats = defaultdict(lambda: {"count": 0, "hits": 0, "combos_sum": 0, "ret_sum": 0})
    for q in quaddies:
        shapes = tuple(sorted(classify_shape(leg) for leg in q))
        widths = [shape_recommended.get(classify_shape(leg), 2) for leg in q]
        combos = 1
        for w in widths:
            combos *= w
        hit = all(q[i]["winner_rank"] <= widths[i] for i in range(4))
        sp_prod = 1
        for leg in q:
            sp_prod *= leg["winner_sp"]
        payout = sp_prod * 0.80 * (100.0 / combos) if hit else 0

        pattern_stats[shapes]["count"] += 1
        pattern_stats[shapes]["hits"] += int(hit)
        pattern_stats[shapes]["combos_sum"] += combos
        pattern_stats[shapes]["ret_sum"] += payout

    # Sort by frequency
    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    print(f"\n{'Pattern':<55} {'N':>4} {'Hits':>4} {'Hit%':>5} {'AvgCmb':>7} {'Flexi':>6}")
    print("-" * 85)
    for pattern, stats in sorted_patterns[:20]:
        n = stats["count"]
        hits = stats["hits"]
        avg_c = stats["combos_sum"] / n
        flexi = 100.0 / avg_c * 100
        hit_pct = hits / n * 100 if n else 0
        pattern_str = " + ".join(pattern)
        if len(pattern_str) > 53:
            pattern_str = pattern_str[:50] + "..."
        print(f"  {pattern_str:<53} {n:>4} {hits:>4} {hit_pct:>4.0f}%  {avg_c:>6.0f}  {flexi:>4.0f}%")

    # ================================================================
    # 9. EARLY QUADDIE vs MAIN QUADDIE
    # ================================================================
    print("\n" + "=" * 90)
    print("9. EARLY QUADDIE vs MAIN QUADDIE (with recommended widths)")
    print("=" * 90)

    for label, qs in [("Main Quaddie", quaddies), ("Early Quaddie", eq_quaddies)]:
        hits = 0
        total_ret = 0
        total_combos = 0
        for q in qs:
            widths = [shape_recommended.get(classify_shape(leg), 2) for leg in q]
            combos = 1
            for w in widths:
                combos *= w
            while combos > 333:
                sorted_legs = sorted(range(4), key=lambda i: q[i]["p1"])
                reduced = False
                for idx in sorted_legs:
                    if widths[idx] > 1:
                        widths[idx] -= 1
                        combos = 1
                        for w in widths:
                            combos *= w
                        reduced = True
                        break
                if not reduced:
                    break
            total_combos += combos
            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * (100.0 / combos)
        avg_combos = total_combos / len(qs)
        avg_flexi = 100.0 / avg_combos * 100
        hit_pct = hits / len(qs) * 100
        total_cost = 100.0 * len(qs)
        roi = (total_ret - total_cost) / total_cost * 100
        avg_pay = total_ret / hits if hits else 0
        print(f"\n  {label}: {len(qs)} quaddies")
        print(f"    Hits: {hits} ({hit_pct:.1f}%), Avg combos: {avg_combos:.0f} ({avg_flexi:.0f}% flexi)")
        print(f"    Avg payout: ${avg_pay:.0f}, ROI: {roi:.1f}%")

    # ================================================================
    # 10. WHAT IF WE ONLY BET FAVOURABLE CARDS?
    # ================================================================
    print("\n" + "=" * 90)
    print("10. SELECTIVE BETTING: Only bet when flexi >= X%")
    print("    What if we skip quaddies that require too many combos?")
    print("=" * 90)

    for min_flexi_pct in [30, 40, 50, 75, 100, 150, 200]:
        qualifying = 0
        hits = 0
        total_ret = 0
        for q in quaddies:
            widths = [shape_recommended.get(classify_shape(leg), 2) for leg in q]
            combos = 1
            for w in widths:
                combos *= w
            flexi = 100.0 / combos * 100
            if flexi < min_flexi_pct:
                continue
            qualifying += 1
            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                hits += 1
                sp_prod = 1
                for leg in q:
                    sp_prod *= leg["winner_sp"]
                total_ret += sp_prod * 0.80 * (100.0 / combos)
        if qualifying == 0:
            continue
        total_cost = 100.0 * qualifying
        roi = (total_ret - total_cost) / total_cost * 100
        avg_pay = total_ret / hits if hits else 0
        print(f"  Flexi >= {min_flexi_pct:>3}%: {qualifying:>5} quaddies ({qualifying/len(quaddies)*100:.0f}%), {hits:>4} hits ({hits/qualifying*100:.1f}%), avg pay ${avg_pay:.0f}, ROI {roi:.1f}%")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
