"""Quaddie leg width analysis: what odds patterns predict how many runners needed.

Answers: given the SHAPE of the odds in a race, how many runners should we use?
- $2 fav with $10 second? Just 1.
- $2 fav with $2.50 second? Need 2.
- 4 horses around $5-$8? Need 3-4.
- When does the roughie (4th pick) actually win?
"""
import json
import os
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
    """Build detailed odds landscape for a race."""
    runners = race.get("Runners", [])
    active = [r for r in runners if r.get("Position") and r["Position"] > 0]
    if len(active) < 3:
        return None

    # Calculate implied probabilities from SP
    for r in active:
        sp = r.get("PriceSP", 0)
        r["impl_prob"] = 1.0 / sp if sp and sp > 0 else 0
    total_prob = sum(r["impl_prob"] for r in active)
    if total_prob <= 0:
        return None
    for r in active:
        r["norm_prob"] = r["impl_prob"] / total_prob

    # Sort by SP (shortest first = most favoured)
    by_sp = sorted(active, key=lambda r: r.get("PriceSP", 999))

    # Find winner
    winner = None
    for r in active:
        if r["Position"] == 1:
            winner = r
            break
    if not winner:
        return None

    # Winner's rank by SP
    winner_rank = None
    for i, r in enumerate(by_sp):
        if r.get("TabNo") == winner.get("TabNo") and r.get("Name") == winner.get("Name"):
            winner_rank = i + 1
            break
    if winner_rank is None:
        return None

    # Get top 6 runners by SP
    top = by_sp[:min(6, len(by_sp))]
    prices = [r.get("PriceSP", 999) for r in top]

    # Pad if fewer than 6
    while len(prices) < 6:
        prices.append(999)

    # Gaps between consecutive runners
    gaps = []
    for i in range(min(5, len(top) - 1)):
        gaps.append(prices[i + 1] - prices[i])
    while len(gaps) < 5:
        gaps.append(999)

    # Ratios between consecutive runners
    ratios = []
    for i in range(min(5, len(top) - 1)):
        if prices[i] > 0:
            ratios.append(prices[i + 1] / prices[i])
        else:
            ratios.append(999)
    while len(ratios) < 5:
        ratios.append(999)

    return {
        "field_size": len(active),
        "winner_rank": winner_rank,
        "winner_sp": winner.get("PriceSP", 0),
        "prices": prices,  # [fav, 2nd, 3rd, 4th, 5th, 6th]
        "gaps": gaps,      # [fav-to-2nd, 2nd-to-3rd, ...]
        "ratios": ratios,  # [2nd/fav, 3rd/2nd, ...]
        "p1": prices[0],   # fav price
        "p2": prices[1],   # 2nd price
        "p3": prices[2],   # 3rd price
        "p4": prices[3],   # 4th price
        "gap_1_2": gaps[0],
        "gap_2_3": gaps[1],
        "gap_3_4": gaps[2],
        "ratio_2_1": ratios[0],  # how much more expensive is 2nd vs fav
        "ratio_3_2": ratios[1],
        "ratio_4_3": ratios[2],
        "race_class": race.get("RaceClass", "") or "",
        "race_number": race.get("Number", 0),
    }


def main():
    print("Loading 2025 Proform data...")
    meetings = load_all_meetings()
    print(f"Loaded {len(meetings)} meetings\n")

    # Collect all quaddie legs
    all_legs = []
    quaddies = []

    for m in meetings:
        races = m["races"]
        n = len(races)
        q_races = races[n - 4:]
        q_legs = []
        for i, r in enumerate(q_races):
            a = analyse_race(r)
            if a:
                a["leg_pos"] = i + 1
                a["seq_type"] = "Q"
                all_legs.append(a)
                q_legs.append(a)

        if len(q_legs) == 4:
            quaddies.append(q_legs)

        # Early quaddie
        if n >= 8:
            eq_legs = []
            for i, r in enumerate(races[:4]):
                a = analyse_race(r)
                if a:
                    a["leg_pos"] = i + 1
                    a["seq_type"] = "EQ"
                    all_legs.append(a)
                    eq_legs.append(a)

    print(f"Total legs analysed: {len(all_legs)}")
    print(f"Complete quaddies: {len(quaddies)}")

    # ================================================================
    # A. THE CORE QUESTION: Fav price vs how many runners needed
    # ================================================================
    print("\n" + "=" * 85)
    print("A. FAVOURITE'S PRICE vs WINNER'S RANK")
    print("   When the fav is $X, how often does rank 1/2/3/4 win?")
    print("=" * 85)

    fav_bands = [
        ("$1.20-$1.50", 1.20, 1.50),
        ("$1.51-$2.00", 1.51, 2.00),
        ("$2.01-$2.50", 2.01, 2.50),
        ("$2.51-$3.00", 2.51, 3.00),
        ("$3.01-$3.50", 3.01, 3.50),
        ("$3.51-$4.00", 3.51, 4.00),
        ("$4.01-$5.00", 4.01, 5.00),
        ("$5.01-$7.00", 5.01, 7.00),
        ("$7.01+", 7.01, 999),
    ]

    print(f"\n{'Fav SP':<14} {'N':>5} {'#1 wins':>8} {'#2 wins':>8} {'#3 wins':>8} {'#4 wins':>8} {'#5+ wins':>9} {'Need':>5}")
    print("-" * 75)
    for label, lo, hi in fav_bands:
        legs = [l for l in all_legs if lo <= l["p1"] <= hi]
        n = len(legs)
        if n < 20:
            continue
        r1 = sum(1 for l in legs if l["winner_rank"] == 1) / n * 100
        r2 = sum(1 for l in legs if l["winner_rank"] == 2) / n * 100
        r3 = sum(1 for l in legs if l["winner_rank"] == 3) / n * 100
        r4 = sum(1 for l in legs if l["winner_rank"] == 4) / n * 100
        r5 = sum(1 for l in legs if l["winner_rank"] >= 5) / n * 100
        # Minimum width for 75% capture
        cum = 0
        need = 1
        for rank in range(1, 10):
            cum += sum(1 for l in legs if l["winner_rank"] == rank)
            if cum / n >= 0.75:
                need = rank
                break
        print(f"  {label:<12} {n:>5} {r1:>7.1f}% {r2:>7.1f}% {r3:>7.1f}% {r4:>7.1f}% {r5:>8.1f}% {need:>4}")

    # ================================================================
    # B. THE GAP: When 2nd horse is close vs far from fav
    # ================================================================
    print("\n" + "=" * 85)
    print("B. PRICE GAP BETWEEN FAV AND 2ND — Should we include the 2nd?")
    print("   Ratio = 2nd_price / fav_price (1.0 = equal, 3.0 = triple)")
    print("=" * 85)

    ratio_bands = [
        ("2nd/1st < 1.3 (close)", 0, 1.3),
        ("2nd/1st 1.3-1.8", 1.3, 1.8),
        ("2nd/1st 1.8-2.5", 1.8, 2.5),
        ("2nd/1st 2.5-3.5", 2.5, 3.5),
        ("2nd/1st 3.5+ (dominant fav)", 3.5, 999),
    ]

    print(f"\n{'Gap Ratio':<30} {'N':>5} {'Fav wins':>9} {'2nd wins':>9} {'Top2':>6} {'Need 2?':>8}")
    print("-" * 75)
    for label, lo, hi in ratio_bands:
        legs = [l for l in all_legs if lo <= l["ratio_2_1"] < hi]
        n = len(legs)
        if n < 20:
            continue
        fav_w = sum(1 for l in legs if l["winner_rank"] == 1) / n * 100
        sec_w = sum(1 for l in legs if l["winner_rank"] == 2) / n * 100
        top2 = fav_w + sec_w
        avg_p1 = sum(l["p1"] for l in legs) / n
        avg_p2 = sum(l["p2"] for l in legs) / n
        verdict = "YES" if sec_w > 15 else ("MAYBE" if sec_w > 10 else "NO")
        print(f"  {label:<28} {n:>5} {fav_w:>8.1f}% {sec_w:>8.1f}% {top2:>5.1f}% {verdict:>7}  (avg ${avg_p1:.1f}/${avg_p2:.1f})")

    # ================================================================
    # C. WHEN TO GO TO 3 RUNNERS — The 3rd horse's value
    # ================================================================
    print("\n" + "=" * 85)
    print("C. WHEN SHOULD WE ADD A 3RD RUNNER?")
    print("   3rd's price relative to 2nd (ratio_3_2)")
    print("=" * 85)

    # Only look at legs where fav is NOT dominant (ratio_2_1 < 3.0)
    non_dom = [l for l in all_legs if l["ratio_2_1"] < 3.0]
    print(f"  (Excluding dominant fav races, N={len(non_dom)})")

    r32_bands = [
        ("3rd/2nd < 1.2 (very close)", 0, 1.2),
        ("3rd/2nd 1.2-1.5 (close)", 1.2, 1.5),
        ("3rd/2nd 1.5-2.0 (gap)", 1.5, 2.0),
        ("3rd/2nd 2.0-3.0 (big gap)", 2.0, 3.0),
        ("3rd/2nd 3.0+ (huge gap)", 3.0, 999),
    ]

    print(f"\n{'3rd/2nd Ratio':<30} {'N':>5} {'Top2':>6} {'3rd wins':>9} {'Top3':>6} {'Add 3rd?':>9}  (avg prices)")
    print("-" * 85)
    for label, lo, hi in r32_bands:
        legs = [l for l in non_dom if lo <= l["ratio_3_2"] < hi]
        n = len(legs)
        if n < 20:
            continue
        top2 = sum(1 for l in legs if l["winner_rank"] <= 2) / n * 100
        third_w = sum(1 for l in legs if l["winner_rank"] == 3) / n * 100
        top3 = top2 + third_w
        avg_p1 = sum(l["p1"] for l in legs) / n
        avg_p2 = sum(l["p2"] for l in legs) / n
        avg_p3 = sum(l["p3"] for l in legs) / n
        verdict = "YES" if third_w > 12 else ("MAYBE" if third_w > 8 else "NO")
        print(f"  {label:<28} {n:>5} {top2:>5.1f}% {third_w:>8.1f}% {top3:>5.1f}% {verdict:>8}  (${avg_p1:.1f}/${avg_p2:.1f}/${avg_p3:.1f})")

    # ================================================================
    # D. WHEN TO PUSH TO 4 / INCLUDE THE ROUGHIE
    # ================================================================
    print("\n" + "=" * 85)
    print("D. WHEN TO PUSH TO 4 RUNNERS / INCLUDE THE ROUGHIE?")
    print("   4th runner's price and gap from 3rd")
    print("=" * 85)

    # Look at races where top 3 are close (all within reasonable range)
    open_races = [l for l in all_legs if l["p3"] <= 10.0]  # 3rd is no longer than $10
    print(f"  (Races where 3rd fav is $10 or less, N={len(open_races)})")

    r43_bands = [
        ("4th/3rd < 1.3 (cluster)", 0, 1.3),
        ("4th/3rd 1.3-1.5", 1.3, 1.5),
        ("4th/3rd 1.5-2.0", 1.5, 2.0),
        ("4th/3rd 2.0+ (clear gap)", 2.0, 999),
    ]

    print(f"\n{'4th/3rd Ratio':<25} {'N':>5} {'Top3':>6} {'4th wins':>9} {'Top4':>6} {'Add 4th?':>9}  (avg prices)")
    print("-" * 85)
    for label, lo, hi in r43_bands:
        legs = [l for l in open_races if lo <= l["ratio_4_3"] < hi]
        n = len(legs)
        if n < 20:
            continue
        top3 = sum(1 for l in legs if l["winner_rank"] <= 3) / n * 100
        fourth_w = sum(1 for l in legs if l["winner_rank"] == 4) / n * 100
        top4 = top3 + fourth_w
        avg_p1 = sum(l["p1"] for l in legs) / n
        avg_p2 = sum(l["p2"] for l in legs) / n
        avg_p3 = sum(l["p3"] for l in legs) / n
        avg_p4 = sum(l["p4"] for l in legs) / n
        verdict = "YES" if fourth_w > 10 else ("MAYBE" if fourth_w > 7 else "NO")
        print(f"  {label:<23} {n:>5} {top3:>5.1f}% {fourth_w:>8.1f}% {top4:>5.1f}% {verdict:>8}  (${avg_p1:.1f}/${avg_p2:.1f}/${avg_p3:.1f}/${avg_p4:.1f})")

    # ================================================================
    # E. ODDS SHAPE PATTERNS — Common race types
    # ================================================================
    print("\n" + "=" * 85)
    print("E. RACE SHAPE CLASSIFICATION — Which pattern is this race?")
    print("=" * 85)

    shapes = classify_shapes(all_legs)
    print(f"\n{'Shape':<35} {'N':>5} {'%':>5} {'FavW':>5} {'Top2':>5} {'Top3':>5} {'Top4':>5} {'Rec Width':>10}")
    print("-" * 85)
    for shape_name, legs, rec in shapes:
        n = len(legs)
        if n < 20:
            continue
        pct = n / len(all_legs) * 100
        t1 = sum(1 for l in legs if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs if l["winner_rank"] <= 4) / n * 100
        print(f"  {shape_name:<33} {n:>5} {pct:>4.1f}% {t1:>4.1f}% {t2:>4.1f}% {t3:>4.1f}% {t4:>4.1f}% {rec:>9}")

    # ================================================================
    # F. SIMULATE SINGLE SMART QUADDIE using shape-based widths
    # ================================================================
    print("\n" + "=" * 85)
    print("F. QUADDIE SIMULATION: SINGLE SMART PICK vs ALTERNATIVES")
    print("   One intelligently-sized quaddie per meeting")
    print("=" * 85)

    strategies = {
        "Shape-Based Smart": shape_width,
        "Fixed 2-2-2-2": lambda l: 2,
        "Fixed 2-2-3-3": lambda l: 3 if l["leg_pos"] >= 3 else 2,
        "Always 3": lambda l: 3,
        "SP-Ratio Smart": sp_ratio_width,
    }

    print(f"\n{'Strategy':<25} {'Hits':>5} {'Hit%':>6} {'AvgCombos':>10} {'Eff':>8} {'$1 unit ROI':>12}")
    print("-" * 75)
    for name, fn in strategies.items():
        hits = 0
        total_combos = 0
        total_payout_est = 0
        for q in quaddies:
            widths = [fn(leg) for leg in q]
            combos = 1
            all_hit = True
            payout_runners = 1
            for i, leg in enumerate(q):
                w = widths[i]
                combos *= w
                if leg["winner_rank"] > w:
                    all_hit = False
                # Estimate payout contribution from winner's SP
                payout_runners *= leg["winner_sp"]
            total_combos += combos
            if all_hit:
                hits += 1
                total_payout_est += payout_runners * 0.85  # 85% pool return
        avg_combos = total_combos / len(quaddies)
        hit_pct = hits / len(quaddies) * 100
        eff = hit_pct / avg_combos if avg_combos > 0 else 0
        avg_cost = avg_combos  # $1 per unit
        avg_payout = total_payout_est / hits if hits else 0
        roi = (total_payout_est - total_combos) / total_combos * 100 if total_combos else 0
        print(f"  {name:<23} {hits:>5} {hit_pct:>5.1f}% {avg_combos:>9.1f} {eff:>7.3f} {roi:>10.1f}%")

    # ================================================================
    # G. DRILL INTO SHAPE-BASED: per-leg width distribution
    # ================================================================
    print("\n" + "=" * 85)
    print("G. SHAPE-BASED STRATEGY: Width distribution per leg")
    print("=" * 85)

    width_dist = defaultdict(lambda: defaultdict(int))
    width_hit = defaultdict(lambda: defaultdict(int))
    for leg in all_legs:
        w = shape_width(leg)
        hit = 1 if leg["winner_rank"] <= w else 0
        width_dist[w]["total"] += 1
        width_dist[w]["hit"] += hit

    print(f"\n{'Width':>6} {'Legs':>6} {'Hits':>6} {'Hit%':>7}")
    print("-" * 30)
    for w in sorted(width_dist.keys()):
        t = width_dist[w]["total"]
        h = width_dist[w]["hit"]
        print(f"  {w:>4}   {t:>5}  {h:>5}  {h/t*100:>5.1f}%")

    # ================================================================
    # H. MARGINAL VALUE OF EACH ADDITIONAL RUNNER
    # ================================================================
    print("\n" + "=" * 85)
    print("H. MARGINAL VALUE: What does each extra runner buy us?")
    print("   Per odds-shape category")
    print("=" * 85)

    for shape_name, legs, rec in shapes:
        n = len(legs)
        if n < 50:
            continue
        print(f"\n  {shape_name} (N={n}):")
        cum = 0
        for width in range(1, 6):
            new_captures = sum(1 for l in legs if l["winner_rank"] == width)
            cum += new_captures
            marginal_pct = new_captures / n * 100
            cum_pct = cum / n * 100
            bar = "#" * int(marginal_pct)
            extra = f"  +{marginal_pct:.1f}% marginal" if width > 1 else ""
            print(f"    Width {width}: {cum_pct:>5.1f}% cumulative ({new_captures:>4} legs){extra}  {bar}")

    # ================================================================
    # I. CROSS-LEG ANALYSIS: Best combos given a budget
    # ================================================================
    print("\n" + "=" * 85)
    print("I. BUDGET OPTIMISATION: Given X total combos, best allocation?")
    print("   Test different total combo budgets across 4 legs")
    print("=" * 85)

    budgets = [8, 12, 16, 24, 36, 48, 64, 81]

    for budget in budgets:
        best_hits = 0
        best_alloc = None

        # Try all allocations of widths 1-5 across 4 legs where product <= budget
        for w1 in range(1, 6):
            for w2 in range(1, 6):
                for w3 in range(1, 6):
                    for w4 in range(1, 6):
                        if w1 * w2 * w3 * w4 > budget:
                            continue
                        hits = 0
                        for q in quaddies:
                            widths = [w1, w2, w3, w4]
                            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                                hits += 1
                        if hits > best_hits:
                            best_hits = hits
                            best_alloc = (w1, w2, w3, w4)

        # Also test smart allocation for this budget
        smart_hits = 0
        for q in quaddies:
            widths = allocate_budget_smart(q, budget)
            if all(q[i]["winner_rank"] <= widths[i] for i in range(4)):
                smart_hits += 1

        combos = best_alloc[0] * best_alloc[1] * best_alloc[2] * best_alloc[3] if best_alloc else 0
        print(f"  Budget <={budget:>3} combos: Best fixed={best_alloc} ({combos} combos, {best_hits} hits, {best_hits/len(quaddies)*100:.1f}%)  Smart={smart_hits} hits ({smart_hits/len(quaddies)*100:.1f}%)")

    # ================================================================
    # J. THE $2 FAV QUESTION — Deep dive
    # ================================================================
    print("\n" + "=" * 85)
    print("J. THE $2 FAVOURITE DEEP DIVE")
    print("   When fav is $1.50-$2.50, what determines if we need 1 or 2 runners?")
    print("=" * 85)

    short_favs = [l for l in all_legs if 1.50 <= l["p1"] <= 2.50]
    print(f"\n  Races with $1.50-$2.50 fav: {len(short_favs)}")

    # Split by 2nd horse's price
    print(f"\n{'2nd Horse Price':<20} {'N':>5} {'Fav wins':>9} {'2nd wins':>9} {'Top2':>6} {'VERDICT':>10}")
    print("-" * 65)
    second_bands = [
        ("$2.50-$3.50", 2.50, 3.50),
        ("$3.51-$5.00", 3.51, 5.00),
        ("$5.01-$7.00", 5.01, 7.00),
        ("$7.01-$10.00", 7.01, 10.00),
        ("$10.01+", 10.01, 999),
    ]
    for label, lo, hi in second_bands:
        legs = [l for l in short_favs if lo <= l["p2"] <= hi]
        n = len(legs)
        if n < 10:
            continue
        fav_w = sum(1 for l in legs if l["winner_rank"] == 1) / n * 100
        sec_w = sum(1 for l in legs if l["winner_rank"] == 2) / n * 100
        top2 = fav_w + sec_w
        if sec_w < 10:
            verdict = "1 RUNNER"
        elif sec_w < 18:
            verdict = "1-2 DEPENDS"
        else:
            verdict = "2 RUNNERS"
        print(f"  {label:<18} {n:>5} {fav_w:>8.1f}% {sec_w:>8.1f}% {top2:>5.1f}% {verdict:>10}")

    # ================================================================
    # K. THE OPEN RACE QUESTION — 4 horses bunched around $5-$10
    # ================================================================
    print("\n" + "=" * 85)
    print("K. THE OPEN RACE: 4+ Horses Bunched Together")
    print("   When top 4 are all within 2x of each other's price")
    print("=" * 85)

    bunched = [l for l in all_legs if l["p1"] >= 3.0 and l["p4"] <= l["p1"] * 2.5 and l["p4"] < 20]
    not_bunched = [l for l in all_legs if l["p1"] >= 3.0 and (l["p4"] > l["p1"] * 2.5 or l["p4"] >= 20)]

    for label, legs in [("Bunched (top4 within 2.5x)", bunched), ("Spread (top4 NOT within 2.5x)", not_bunched)]:
        n = len(legs)
        if n < 20:
            continue
        t1 = sum(1 for l in legs if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs if l["winner_rank"] <= 4) / n * 100
        t5 = sum(1 for l in legs if l["winner_rank"] >= 5) / n * 100
        avg_p = sum(l["p1"] for l in legs) / n
        print(f"\n  {label} (N={n}, avg fav ${avg_p:.1f}):")
        print(f"    Top-1: {t1:.1f}%, Top-2: {t2:.1f}%, Top-3: {t3:.1f}%, Top-4: {t4:.1f}%, Outside top4: {t5:.1f}%")

    # ================================================================
    # L. ROUGHIE ANALYSIS — When does rank 4+ win?
    # ================================================================
    print("\n" + "=" * 85)
    print("L. ROUGHIE WINS: When does the 4th+ favourite actually win?")
    print("=" * 85)

    roughie_wins = [l for l in all_legs if l["winner_rank"] >= 4]
    print(f"\n  Roughie wins (rank 4+): {len(roughie_wins)} / {len(all_legs)} = {len(roughie_wins)/len(all_legs)*100:.1f}%")

    # What price was the fav in roughie-winning races?
    print(f"\n  Favourite's price when roughie wins:")
    for label, lo, hi in fav_bands:
        legs = [l for l in roughie_wins if lo <= l["p1"] <= hi]
        all_in_band = [l for l in all_legs if lo <= l["p1"] <= hi]
        n_all = len(all_in_band)
        n_rough = len(legs)
        if n_all < 20:
            continue
        pct = n_rough / n_all * 100
        print(f"    Fav {label:<14}: {n_rough:>4}/{n_all:>5} = {pct:>5.1f}% roughie win rate")

    # What characterises roughie-winning races?
    print(f"\n  Roughie-winning race characteristics:")
    avg_field_r = sum(l["field_size"] for l in roughie_wins) / len(roughie_wins)
    avg_field_a = sum(l["field_size"] for l in all_legs) / len(all_legs)
    avg_p1_r = sum(l["p1"] for l in roughie_wins) / len(roughie_wins)
    avg_p1_a = sum(l["p1"] for l in all_legs) / len(all_legs)
    avg_gap_r = sum(l["gap_1_2"] for l in roughie_wins) / len(roughie_wins)
    avg_gap_a = sum(l["gap_1_2"] for l in all_legs) / len(all_legs)
    print(f"    Avg field size: {avg_field_r:.1f} (vs {avg_field_a:.1f} overall)")
    print(f"    Avg fav price:  ${avg_p1_r:.2f} (vs ${avg_p1_a:.2f} overall)")
    print(f"    Avg fav-2nd gap: ${avg_gap_r:.2f} (vs ${avg_gap_a:.2f} overall)")

    # ================================================================
    # M. COMBINED DECISION TREE — The actual rules
    # ================================================================
    print("\n" + "=" * 85)
    print("M. DECISION TREE: Exact rules for per-leg width")
    print("=" * 85)

    # Test the proposed decision tree
    tree_hits = defaultdict(lambda: {"total": 0, "correct": 0})

    for leg in all_legs:
        w = decision_tree_width(leg)
        hit = leg["winner_rank"] <= w
        key = f"width={w}"
        tree_hits[key]["total"] += 1
        tree_hits[key]["correct"] += int(hit)

    print(f"\n  Decision tree leg capture rates:")
    total_correct = 0
    total_all = 0
    for key in sorted(tree_hits.keys()):
        t = tree_hits[key]["total"]
        c = tree_hits[key]["correct"]
        total_correct += c
        total_all += t
        print(f"    {key}: {c}/{t} = {c/t*100:.1f}% capture rate")
    print(f"    OVERALL: {total_correct}/{total_all} = {total_correct/total_all*100:.1f}%")

    # Simulate quaddies with decision tree
    print(f"\n  Quaddie simulation with decision tree:")
    dt_hits = 0
    dt_combos_total = 0
    for q in quaddies:
        widths = [decision_tree_width(leg) for leg in q]
        combos = 1
        all_hit = True
        for i in range(4):
            combos *= widths[i]
            if q[i]["winner_rank"] > widths[i]:
                all_hit = False
        dt_combos_total += combos
        if all_hit:
            dt_hits += 1

    avg_c = dt_combos_total / len(quaddies)
    print(f"    Hits: {dt_hits}/{len(quaddies)} = {dt_hits/len(quaddies)*100:.1f}%")
    print(f"    Avg combos: {avg_c:.1f}")
    print(f"    Efficiency: {dt_hits/len(quaddies)*100/avg_c:.3f}")

    # Compare to fixed
    for label, fixed_w in [("Fixed 2-2-2-2", 2), ("Fixed 3-3-3-3", 3)]:
        fhits = sum(1 for q in quaddies if all(q[i]["winner_rank"] <= fixed_w for i in range(4)))
        fcombos = fixed_w ** 4
        print(f"    vs {label}: {fhits}/{len(quaddies)} = {fhits/len(quaddies)*100:.1f}%, combos={fcombos}, eff={fhits/len(quaddies)*100/fcombos:.3f}")

    # ================================================================
    # N. FINAL: Show example legs with the decision
    # ================================================================
    print("\n" + "=" * 85)
    print("N. EXAMPLE LEGS: How the decision tree would work")
    print("=" * 85)

    import random
    random.seed(42)
    samples = random.sample(quaddies, min(5, len(quaddies)))
    for qi, q in enumerate(samples):
        print(f"\n  --- Quaddie Example {qi+1} ---")
        total_combos = 1
        all_hit = True
        for leg in q:
            w = decision_tree_width(leg)
            total_combos *= w
            hit = leg["winner_rank"] <= w
            if not hit:
                all_hit = False
            shape = classify_single(leg)
            prices_str = "/".join(f"${p:.1f}" for p in leg["prices"][:4])
            hit_str = "HIT" if hit else f"MISS (winner was rank {leg['winner_rank']})"
            print(f"    Leg {leg['leg_pos']}: Prices [{prices_str}] -> {shape} -> Width {w} -> {hit_str}")
        print(f"    Total: {total_combos} combos, {'QUADDIE HIT' if all_hit else 'MISSED'}")


def classify_single(leg):
    """Classify a single leg's odds shape."""
    p1, p2, p3, p4 = leg["p1"], leg["p2"], leg["p3"], leg["p4"]

    if p1 <= 2.0 and p2 >= p1 * 2.5:
        return "DOMINANT"
    elif p1 <= 2.0 and p2 < p1 * 2.5:
        return "SHORT_PAIR"
    elif p1 <= 3.50 and p2 / p1 < 1.5:
        return "TWO_HORSE"
    elif p1 <= 3.50:
        return "CLEAR_FAV"
    elif p1 <= 5.0 and p3 / p1 < 1.8:
        return "TRIO"
    elif p1 <= 5.0:
        return "MID_FAV"
    elif p4 <= p1 * 2.0:
        return "OPEN_BUNCH"
    else:
        return "WIDE_OPEN"


def classify_shapes(all_legs):
    """Classify all legs into shape categories."""
    buckets = defaultdict(list)
    for l in all_legs:
        shape = classify_single(l)
        buckets[shape].append(l)

    recs = {
        "DOMINANT": "1",
        "SHORT_PAIR": "2",
        "TWO_HORSE": "2",
        "CLEAR_FAV": "2",
        "TRIO": "3",
        "MID_FAV": "2-3",
        "OPEN_BUNCH": "3-4",
        "WIDE_OPEN": "3-4",
    }

    result = []
    order = ["DOMINANT", "SHORT_PAIR", "TWO_HORSE", "CLEAR_FAV", "TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"]
    for shape in order:
        legs = buckets.get(shape, [])
        desc = {
            "DOMINANT": f"Fav <=$2, 2nd 2.5x+ away",
            "SHORT_PAIR": f"Fav <=$2, 2nd within 2.5x",
            "TWO_HORSE": f"Fav <=$3.50, 2nd within 1.5x",
            "CLEAR_FAV": f"Fav <=$3.50, 2nd further out",
            "TRIO": f"Fav $3.50-$5, top 3 within 1.8x",
            "MID_FAV": f"Fav $3.50-$5, 3rd further out",
            "OPEN_BUNCH": f"Fav $5+, top 4 within 2x",
            "WIDE_OPEN": f"Fav $5+, spread field",
        }
        result.append((f"{shape}: {desc.get(shape, '')}", legs, recs.get(shape, "?")))

    return result


def shape_width(leg):
    """Determine width from odds shape."""
    shape = classify_single(leg)
    return {
        "DOMINANT": 1,
        "SHORT_PAIR": 2,
        "TWO_HORSE": 2,
        "CLEAR_FAV": 2,
        "TRIO": 3,
        "MID_FAV": 2,
        "OPEN_BUNCH": 3,
        "WIDE_OPEN": 3,
    }.get(shape, 2)


def sp_ratio_width(leg):
    """Width based on price ratios between runners."""
    p1, p2, p3, p4 = leg["p1"], leg["p2"], leg["p3"], leg["p4"]
    width = 1

    # Always include fav
    # Include 2nd if within 2x of fav
    if p2 <= p1 * 2.0:
        width = 2

    # Include 3rd if within 1.8x of 2nd AND 3rd is $12 or less
    if width >= 2 and p3 <= p2 * 1.8 and p3 <= 12:
        width = 3

    # Include 4th if within 1.5x of 3rd AND 4th is $15 or less
    if width >= 3 and p4 <= p3 * 1.5 and p4 <= 15:
        width = 4

    return width


def decision_tree_width(leg):
    """The proposed decision tree for leg width."""
    p1, p2, p3, p4 = leg["p1"], leg["p2"], leg["p3"], leg["p4"]
    ratio_21 = leg["ratio_2_1"]
    ratio_32 = leg["ratio_3_2"]
    ratio_43 = leg["ratio_4_3"]

    # Rule 1: Dominant favourite — odds-on with big gap
    if p1 <= 2.0 and ratio_21 >= 2.5:
        return 1

    # Rule 2: Short fav but 2nd is close — need both
    if p1 <= 2.50 and ratio_21 < 2.0:
        return 2

    # Rule 3: Short fav, 2nd is mid-range — just the fav
    if p1 <= 2.50:
        return 1

    # Rule 4: Mid fav, two-horse race
    if p1 <= 3.50 and ratio_21 < 1.5:
        return 2

    # Rule 5: Mid fav with clear gap to field
    if p1 <= 3.50 and ratio_21 >= 1.5:
        return 2  # Still include a saver

    # Rule 6: Fav $3.50-$5, top 3 bunched
    if p1 <= 5.0 and p3 <= p1 * 1.8:
        return 3

    # Rule 7: Fav $3.50-$5, top 2 close but 3rd far
    if p1 <= 5.0:
        return 2

    # Rule 8: Open race, top 4 bunched
    if p4 <= p1 * 2.0 and p4 <= 15:
        return 4

    # Rule 9: Open race, top 3 bunched
    if p3 <= p1 * 2.0:
        return 3

    # Rule 10: Wide open
    return 3


def allocate_budget_smart(quaddie_legs, max_combos):
    """Allocate width across 4 legs to maximise coverage within combo budget.

    Sort legs by difficulty (openness), give more width to open legs.
    """
    # Score each leg by openness (lower HHI = more open = needs more width)
    scored = [(i, leg["p1"], leg["ratio_2_1"]) for i, leg in enumerate(quaddie_legs)]
    # Sort by fav price descending (most open first)
    scored.sort(key=lambda x: -x[1])

    widths = [2, 2, 2, 2]  # Start with 2 each (16 combos)

    # If budget allows, expand the most open legs
    while True:
        current_combos = widths[0] * widths[1] * widths[2] * widths[3]
        if current_combos >= max_combos:
            break
        # Find the most open leg that hasn't been expanded too much
        best_idx = None
        best_score = -1
        for idx, fav_p, ratio in scored:
            if widths[idx] < 5 and fav_p > best_score:
                # Check if expanding would exceed budget
                test = widths.copy()
                test[idx] += 1
                if test[0] * test[1] * test[2] * test[3] <= max_combos:
                    best_score = fav_p
                    best_idx = idx
        if best_idx is None:
            break
        widths[best_idx] += 1

    # If over budget, shrink the tightest legs
    while widths[0] * widths[1] * widths[2] * widths[3] > max_combos:
        # Find easiest leg (highest ratio_21 = most dominant fav)
        easiest = max(scored, key=lambda x: x[2])
        idx = easiest[0]
        if widths[idx] > 1:
            widths[idx] -= 1
        else:
            break

    return widths


if __name__ == "__main__":
    main()
