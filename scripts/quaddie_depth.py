"""Deep quaddie leg analysis: how many runners needed to win every leg.

Analyses 2025 Proform data to determine optimal leg width based on:
- Winner's SP rank (favourite, 2nd fav, etc.)
- Odds bands (short-priced fav, mid-range, open)
- Field size impact
- Race class impact
- HHI (market concentration)
- Value overlay (prob × value proxy)
- Leg position within quaddie
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
    """Load all 2025 Australian meetings with 7+ races."""
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
            # Filter out races with no runners or no results
            valid_races = []
            for r in races:
                runners = r.get("Runners", [])
                # Must have results (Position field)
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
                    "location": track.get("Location", ""),  # M=Metro, C=Country, P=Provincial
                    "races": valid_races,
                })
    return meetings


def analyse_race(race):
    """Analyse a single race leg. Returns dict of metrics."""
    runners = race.get("Runners", [])
    # Filter scratched (Position=0 or no Position)
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

    # Sort by probability (favourite first)
    by_prob = sorted(active, key=lambda r: r["norm_prob"], reverse=True)

    # Find the winner
    winner = None
    for r in active:
        if r["Position"] == 1:
            winner = r
            break
    if not winner:
        return None

    # Winner's rank by SP probability
    winner_rank = None
    for i, r in enumerate(by_prob):
        if r.get("TabNo") == winner.get("TabNo") and r.get("Name") == winner.get("Name"):
            winner_rank = i + 1  # 1-indexed
            break
    if winner_rank is None:
        return None

    # HHI
    hhi = sum(r["norm_prob"] ** 2 for r in active)

    # Top runner stats
    top_prob = by_prob[0]["norm_prob"]
    second_prob = by_prob[1]["norm_prob"] if len(by_prob) > 1 else 0
    gap = top_prob - second_prob
    top2 = top_prob + second_prob
    top3 = top2 + (by_prob[2]["norm_prob"] if len(by_prob) > 2 else 0)
    top4 = top3 + (by_prob[3]["norm_prob"] if len(by_prob) > 3 else 0)

    # Winner's SP and implied prob
    winner_sp = winner.get("PriceSP", 0)
    winner_prob = winner.get("norm_prob", 0)

    # Fav SP
    fav_sp = by_prob[0].get("PriceSP", 0)

    # Value proxy for winner: actual_return / fair_odds
    # If winner was paying $5 and their fair prob was 25% (fair odds $4), value = 5/4 = 1.25
    fair_odds = 1.0 / winner["norm_prob"] if winner["norm_prob"] > 0 else 999
    value_proxy = winner_sp / fair_odds if fair_odds > 0 else 1.0

    # Race class
    race_class = race.get("RaceClass", "") or ""

    return {
        "field_size": len(active),
        "winner_rank": winner_rank,
        "winner_sp": winner_sp,
        "winner_prob": winner_prob,
        "fav_sp": fav_sp,
        "fav_prob": top_prob,
        "gap_to_2nd": gap,
        "top2_combined": top2,
        "top3_combined": top3,
        "top4_combined": top4,
        "hhi": hhi,
        "value_proxy": value_proxy,
        "race_class": race_class,
        "race_number": race.get("Number", 0),
    }


def main():
    print("Loading 2025 Proform data...")
    meetings = load_all_meetings()
    print(f"Loaded {len(meetings)} Australian meetings with 7+ races\n")

    # Collect all quaddie legs
    all_legs = []  # Each: {meeting_info, leg_position (1-4), race_analysis}
    all_eq_legs = []  # Early quaddie
    all_b6_legs = []  # Big 6

    for m in meetings:
        races = m["races"]
        n = len(races)

        # Quaddie = last 4 races
        q_races = races[n - 4:]
        q_analyses = []
        for i, r in enumerate(q_races):
            a = analyse_race(r)
            if a:
                a["leg_pos"] = i + 1  # 1=first leg, 4=last leg
                a["track"] = m["track"]
                a["state"] = m["state"]
                a["location"] = m["location"]
                q_analyses.append(a)
                all_legs.append(a)

        # Early Quaddie = first 4 races (if 8+ races)
        if n >= 8:
            for i, r in enumerate(races[:4]):
                a = analyse_race(r)
                if a:
                    a["leg_pos"] = i + 1
                    a["track"] = m["track"]
                    a["state"] = m["state"]
                    a["location"] = m["location"]
                    all_eq_legs.append(a)

            # Big 6 = last 6 races
            for i, r in enumerate(races[n - 6:]):
                a = analyse_race(r)
                if a:
                    a["leg_pos"] = i + 1
                    a["track"] = m["track"]
                    a["state"] = m["state"]
                    a["location"] = m["location"]
                    all_b6_legs.append(a)

    print(f"Quaddie legs analysed: {len(all_legs)}")
    print(f"Early Quaddie legs: {len(all_eq_legs)}")
    print(f"Big 6 legs: {len(all_b6_legs)}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 1. WINNER'S RANK DISTRIBUTION — How many runners needed?
    # ═══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("1. WINNER'S SP RANK DISTRIBUTION (all quaddie legs)")
    print("   'How many runners do we need to capture the winner?'")
    print("=" * 80)

    rank_counts = defaultdict(int)
    for leg in all_legs:
        rank_counts[leg["winner_rank"]] += 1

    total = len(all_legs)
    cumulative = 0
    print(f"\n{'Rank':>6} {'Count':>7} {'%':>7} {'Cumul%':>8}  Visual")
    print("-" * 60)
    for rank in sorted(rank_counts.keys()):
        count = rank_counts[rank]
        pct = count / total * 100
        cumulative += count
        cum_pct = cumulative / total * 100
        bar = "#" * int(pct)
        print(f"  #{rank:<4} {count:>7} {pct:>6.1f}% {cum_pct:>7.1f}%  {bar}")
        if rank >= 10:
            remaining = total - cumulative
            print(f"  >10   {remaining:>7} {remaining/total*100:>6.1f}%  100.0%")
            break

    print(f"\n  TOTAL: {total} legs")
    print(f"  Top-1 captures: {cum_pct_at(rank_counts, 1, total):.1f}%")
    print(f"  Top-2 captures: {cum_pct_at(rank_counts, 2, total):.1f}%")
    print(f"  Top-3 captures: {cum_pct_at(rank_counts, 3, total):.1f}%")
    print(f"  Top-4 captures: {cum_pct_at(rank_counts, 4, total):.1f}%")
    print(f"  Top-5 captures: {cum_pct_at(rank_counts, 5, total):.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 2. WINNER RANK BY FAVOURITE'S ODDS BAND
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. WINNER CAPTURE RATE BY FAVOURITE'S SP ODDS")
    print("   'When the fav is short/mid/long, how many runners needed?'")
    print("=" * 80)

    odds_bands = [
        ("$1.01-$1.80 (odds-on)", 1.01, 1.80),
        ("$1.81-$2.50 (short)", 1.81, 2.50),
        ("$2.51-$3.50 (mid-short)", 2.51, 3.50),
        ("$3.51-$5.00 (mid)", 3.51, 5.00),
        ("$5.01-$7.00 (mid-long)", 5.01, 7.00),
        ("$7.01-$10.00 (long)", 7.01, 10.00),
        ("$10.01+ (open)", 10.01, 999),
    ]

    print(f"\n{'Fav Odds Band':<25} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'Top5':>6} {'Avg Rank':>9}")
    print("-" * 80)
    for label, lo, hi in odds_bands:
        legs_in_band = [l for l in all_legs if lo <= l["fav_sp"] <= hi]
        n = len(legs_in_band)
        if n < 10:
            continue
        t1 = sum(1 for l in legs_in_band if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in_band if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in_band if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in_band if l["winner_rank"] <= 4) / n * 100
        t5 = sum(1 for l in legs_in_band if l["winner_rank"] <= 5) / n * 100
        avg_rank = sum(l["winner_rank"] for l in legs_in_band) / n
        print(f"  {label:<23} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}% {t5:>5.1f}% {avg_rank:>8.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 3. WINNER RANK BY HHI (Market Concentration)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. WINNER CAPTURE RATE BY HHI (Market Concentration)")
    print("   'How does market concentration predict leg difficulty?'")
    print("=" * 80)

    hhi_bands = [
        ("HHI > 0.30 (dominated)", 0.30, 1.0),
        ("HHI 0.25-0.30", 0.25, 0.30),
        ("HHI 0.20-0.25", 0.20, 0.25),
        ("HHI 0.15-0.20", 0.15, 0.20),
        ("HHI 0.12-0.15", 0.12, 0.15),
        ("HHI 0.10-0.12", 0.10, 0.12),
        ("HHI < 0.10 (wide open)", 0.0, 0.10),
    ]

    print(f"\n{'HHI Band':<25} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'Suggested':>10}")
    print("-" * 80)
    for label, lo, hi in hhi_bands:
        legs_in_band = [l for l in all_legs if lo <= l["hhi"] < hi]
        if label.startswith("HHI > 0.30"):
            legs_in_band = [l for l in all_legs if l["hhi"] >= 0.30]
        n = len(legs_in_band)
        if n < 10:
            continue
        t1 = sum(1 for l in legs_in_band if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in_band if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in_band if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in_band if l["winner_rank"] <= 4) / n * 100
        # Suggest width: need 70%+ capture
        if t1 >= 70:
            sug = "1 runner"
        elif t2 >= 70:
            sug = "2 runners"
        elif t3 >= 70:
            sug = "3 runners"
        elif t4 >= 70:
            sug = "4 runners"
        else:
            sug = "5+ runners"
        print(f"  {label:<23} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}% {sug:>10}")

    # ═══════════════════════════════════════════════════════════════
    # 4. FIELD SIZE IMPACT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. WINNER CAPTURE RATE BY FIELD SIZE")
    print("=" * 80)

    field_bands = [
        ("3-5 runners", 3, 5),
        ("6-7 runners", 6, 7),
        ("8-9 runners", 8, 9),
        ("10-11 runners", 10, 11),
        ("12-14 runners", 12, 14),
        ("15+ runners", 15, 30),
    ]

    print(f"\n{'Field Size':<15} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'AvgWinRank':>11}")
    print("-" * 65)
    for label, lo, hi in field_bands:
        legs_in_band = [l for l in all_legs if lo <= l["field_size"] <= hi]
        n = len(legs_in_band)
        if n < 10:
            continue
        t1 = sum(1 for l in legs_in_band if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in_band if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in_band if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in_band if l["winner_rank"] <= 4) / n * 100
        avg = sum(l["winner_rank"] for l in legs_in_band) / n
        print(f"  {label:<13} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}% {avg:>10.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 5. LEG POSITION WITHIN QUADDIE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. WINNER CAPTURE RATE BY LEG POSITION (Leg 1=4th last, Leg 4=last)")
    print("=" * 80)

    print(f"\n{'Leg Pos':>8} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'AvgField':>9} {'AvgRank':>8}")
    print("-" * 60)
    for pos in range(1, 5):
        legs_in_pos = [l for l in all_legs if l["leg_pos"] == pos]
        n = len(legs_in_pos)
        if n == 0:
            continue
        t1 = sum(1 for l in legs_in_pos if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in_pos if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in_pos if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in_pos if l["winner_rank"] <= 4) / n * 100
        avg_field = sum(l["field_size"] for l in legs_in_pos) / n
        avg_rank = sum(l["winner_rank"] for l in legs_in_pos) / n
        print(f"  Leg {pos}   {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}% {avg_field:>8.1f} {avg_rank:>7.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 6. METRO vs PROVINCIAL vs COUNTRY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("6. WINNER CAPTURE RATE BY VENUE TYPE")
    print("=" * 80)

    loc_labels = {"M": "Metro", "P": "Provincial", "C": "Country"}
    print(f"\n{'Venue Type':<12} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'AvgField':>9}")
    print("-" * 55)
    for loc_code, loc_name in loc_labels.items():
        legs_in = [l for l in all_legs if l["location"] == loc_code]
        n = len(legs_in)
        if n < 10:
            continue
        t1 = sum(1 for l in legs_in if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in if l["winner_rank"] <= 4) / n * 100
        avg_f = sum(l["field_size"] for l in legs_in) / n
        print(f"  {loc_name:<10} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}% {avg_f:>8.1f}")

    # ═══════════════════════════════════════════════════════════════
    # 7. STATE BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("7. WINNER CAPTURE RATE BY STATE")
    print("=" * 80)

    states = sorted(set(l["state"] for l in all_legs))
    print(f"\n{'State':<6} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6}")
    print("-" * 42)
    for st in states:
        legs_in = [l for l in all_legs if l["state"] == st]
        n = len(legs_in)
        if n < 50:
            continue
        t1 = sum(1 for l in legs_in if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in if l["winner_rank"] <= 4) / n * 100
        print(f"  {st:<4} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 8. FULL QUADDIE SIMULATION — How many legs needed at each width
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("8. FULL QUADDIE HIT SIMULATION")
    print("   'If we picked top-N in every leg, how many quaddies hit?'")
    print("=" * 80)

    # Group legs back into quaddies (4 consecutive legs per meeting)
    quaddies = []
    meeting_legs = defaultdict(list)
    for m_idx, m in enumerate(meetings):
        races = m["races"]
        n = len(races)
        q_races = races[n - 4:]
        q_legs = []
        for i, r in enumerate(q_races):
            a = analyse_race(r)
            if a:
                a["leg_pos"] = i + 1
                q_legs.append(a)
        if len(q_legs) == 4:
            quaddies.append(q_legs)

    print(f"\n  Complete 4-leg quaddies: {len(quaddies)}")

    for width in [1, 2, 3, 4, 5]:
        hits = 0
        for q in quaddies:
            all_captured = all(leg["winner_rank"] <= width for leg in q)
            if all_captured:
                hits += 1
        combos = width ** 4
        pct = hits / len(quaddies) * 100
        print(f"  Width={width} (all legs): {hits}/{len(quaddies)} quaddies hit ({pct:.1f}%), combos={combos}")

    # ═══════════════════════════════════════════════════════════════
    # 9. ADAPTIVE WIDTH SIMULATION — Use HHI per leg
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("9. ADAPTIVE WIDTH STRATEGIES")
    print("   'Dynamically set width per leg based on race characteristics'")
    print("=" * 80)

    strategies = {
        "Fixed 1-1-1-1": lambda legs: [1, 1, 1, 1],
        "Fixed 2-2-2-2": lambda legs: [2, 2, 2, 2],
        "Fixed 3-3-3-3": lambda legs: [3, 3, 3, 3],
        "Fixed 1-2-2-3": lambda legs: [1, 2, 2, 3],
        "Fixed 2-2-2-3": lambda legs: [2, 2, 2, 3],
        "Fixed 2-2-3-3": lambda legs: [2, 2, 3, 3],
        "HHI-Skinny": lambda legs: [hhi_width_skinny(l) for l in legs],
        "HHI-Balanced": lambda legs: [hhi_width_balanced(l) for l in legs],
        "HHI-Wide": lambda legs: [hhi_width_wide(l) for l in legs],
        "Odds-Guided": lambda legs: [odds_width(l) for l in legs],
        "Combined (HHI+Field)": lambda legs: [combined_width(l) for l in legs],
    }

    print(f"\n{'Strategy':<25} {'Hits':>5} {'Hit%':>6} {'AvgCombos':>10} {'Cost@$1unit':>12} {'Eff Score':>10}")
    print("-" * 75)
    for name, fn in strategies.items():
        hits = 0
        total_combos = 0
        for q in quaddies:
            widths = fn(q)
            combos = 1
            all_hit = True
            for i, leg in enumerate(q):
                w = widths[i]
                combos *= w
                if leg["winner_rank"] > w:
                    all_hit = False
            total_combos += combos
            if all_hit:
                hits += 1
        avg_combos = total_combos / len(quaddies)
        hit_pct = hits / len(quaddies) * 100
        cost = avg_combos  # $1 per combo unit
        # Efficiency: hit_rate / cost (higher = better value)
        eff = hit_pct / avg_combos if avg_combos > 0 else 0
        print(f"  {name:<23} {hits:>5} {hit_pct:>5.1f}% {avg_combos:>9.1f} {cost:>11.1f} {eff:>9.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 10. WHERE DO VALUE WINNERS COME FROM?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("10. VALUE ANALYSIS — When winners pay overs")
    print("    'Do value overlays change which runners we should include?'")
    print("=" * 80)

    # Split legs by winner's value proxy
    val_bands = [
        ("Unders (val<0.8)", 0, 0.8),
        ("Fair (0.8-1.2)", 0.8, 1.2),
        ("Overs (1.2-2.0)", 1.2, 2.0),
        ("Big value (2.0+)", 2.0, 100),
    ]
    print(f"\n{'Value Band':<22} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'AvgRank':>8} {'AvgWinSP':>9}")
    print("-" * 65)
    for label, lo, hi in val_bands:
        legs_in = [l for l in all_legs if lo <= l["value_proxy"] < hi]
        n = len(legs_in)
        if n < 10:
            continue
        t1 = sum(1 for l in legs_in if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in if l["winner_rank"] <= 3) / n * 100
        avg_rank = sum(l["winner_rank"] for l in legs_in) / n
        avg_sp = sum(l["winner_sp"] for l in legs_in) / n
        print(f"  {label:<20} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {avg_rank:>7.2f} {avg_sp:>8.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 11. RACE CLASS IMPACT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("11. WINNER CAPTURE RATE BY RACE CLASS")
    print("=" * 80)

    # Simplify classes
    class_map = defaultdict(list)
    for l in all_legs:
        rc = l["race_class"].lower().strip().rstrip(";")
        if "maiden" in rc:
            class_map["Maiden"].append(l)
        elif "group 1" in rc or "group1" in rc:
            class_map["Group 1"].append(l)
        elif "group 2" in rc or "group2" in rc:
            class_map["Group 2"].append(l)
        elif "group 3" in rc or "group3" in rc:
            class_map["Group 3"].append(l)
        elif "listed" in rc:
            class_map["Listed"].append(l)
        elif "benchmark" in rc or "bm" in rc:
            class_map["Benchmark"].append(l)
        elif "class" in rc or "cls" in rc:
            class_map["Class (1-6)"].append(l)
        elif "handicap" in rc:
            class_map["Handicap"].append(l)
        elif "open" in rc:
            class_map["Open"].append(l)
        else:
            class_map["Other"].append(l)

    print(f"\n{'Class':<15} {'Legs':>5} {'Top1':>6} {'Top2':>6} {'Top3':>6} {'Top4':>6} {'AvgField':>9}")
    print("-" * 60)
    for cls_name in ["Group 1", "Group 2", "Group 3", "Listed", "Open", "Benchmark", "Handicap", "Class (1-6)", "Maiden", "Other"]:
        legs_in = class_map.get(cls_name, [])
        n = len(legs_in)
        if n < 20:
            continue
        t1 = sum(1 for l in legs_in if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in legs_in if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in legs_in if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in legs_in if l["winner_rank"] <= 4) / n * 100
        avg_f = sum(l["field_size"] for l in legs_in) / n
        print(f"  {cls_name:<13} {n:>5} {t1:>5.1f}% {t2:>5.1f}% {t3:>5.1f}% {t4:>5.1f}% {avg_f:>8.1f}")

    # ═══════════════════════════════════════════════════════════════
    # 12. WHAT MAKES QUADDIES FAIL? — Analyse losing legs
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("12. QUADDIE FAILURE ANALYSIS (Top-3 width)")
    print("    'When a quaddie with top-3 per leg fails, what went wrong?'")
    print("=" * 80)

    fail_legs = []  # Legs where winner_rank > 3
    for q in quaddies:
        for leg in q:
            if leg["winner_rank"] > 3:
                fail_legs.append(leg)

    n_fail = len(fail_legs)
    print(f"\n  Failed legs (winner outside top 3): {n_fail}")
    if n_fail > 0:
        # What's the average winner SP when we miss?
        avg_sp = sum(l["winner_sp"] for l in fail_legs) / n_fail
        avg_rank = sum(l["winner_rank"] for l in fail_legs) / n_fail
        avg_field = sum(l["field_size"] for l in fail_legs) / n_fail
        avg_hhi = sum(l["hhi"] for l in fail_legs) / n_fail
        print(f"  Avg winner SP when we miss: ${avg_sp:.2f}")
        print(f"  Avg winner rank: {avg_rank:.1f}")
        print(f"  Avg field size: {avg_field:.1f}")
        print(f"  Avg HHI: {avg_hhi:.3f}")

        # Winner SP distribution when we miss
        print("\n  Winner's SP when outside top 3:")
        sp_bands = [(1, 5), (5, 10), (10, 20), (20, 50), (50, 999)]
        for lo, hi in sp_bands:
            cnt = sum(1 for l in fail_legs if lo <= l["winner_sp"] < hi)
            pct = cnt / n_fail * 100
            print(f"    ${lo}-${hi}: {cnt} ({pct:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # 13. MULTI-LEG CORRELATION — Do open legs cluster?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("13. LEG DIFFICULTY CLUSTERING WITHIN QUADDIES")
    print("    'Do hard legs tend to cluster together?'")
    print("=" * 80)

    difficulty_patterns = defaultdict(lambda: {"count": 0, "hit_top3": 0})
    for q in quaddies:
        # Classify each leg as E(asy, fav wins), M(edium, top3 wins), H(ard, outside top3)
        pattern = ""
        for leg in q:
            if leg["winner_rank"] == 1:
                pattern += "E"
            elif leg["winner_rank"] <= 3:
                pattern += "M"
            else:
                pattern += "H"
        difficulty_patterns[pattern]["count"] += 1

    # Count by number of hard legs
    hard_counts = defaultdict(int)
    for pattern, data in difficulty_patterns.items():
        n_hard = pattern.count("H")
        hard_counts[n_hard] += data["count"]

    total_q = len(quaddies)
    print(f"\n{'Hard Legs':>10} {'Count':>6} {'%':>6}  Meaning")
    print("-" * 60)
    for n_hard in range(5):
        cnt = hard_counts[n_hard]
        pct = cnt / total_q * 100
        meanings = [
            "All legs winnable with top-3",
            "1 upset — need top-4+ in one leg",
            "2 upsets — very hard to hit",
            "3 upsets — near impossible",
            "All legs upset — unwinnable",
        ]
        print(f"  {n_hard:>8} {cnt:>6} {pct:>5.1f}%  {meanings[n_hard]}")

    # ═══════════════════════════════════════════════════════════════
    # 14. EARLY QUADDIE vs MAIN QUADDIE vs BIG 6
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("14. EARLY QUADDIE vs MAIN QUADDIE vs BIG 6 COMPARISON")
    print("=" * 80)

    for seq_name, seq_legs in [("Main Quaddie", all_legs), ("Early Quaddie", all_eq_legs), ("Big 6", all_b6_legs)]:
        n = len(seq_legs)
        if n == 0:
            continue
        t1 = sum(1 for l in seq_legs if l["winner_rank"] <= 1) / n * 100
        t2 = sum(1 for l in seq_legs if l["winner_rank"] <= 2) / n * 100
        t3 = sum(1 for l in seq_legs if l["winner_rank"] <= 3) / n * 100
        t4 = sum(1 for l in seq_legs if l["winner_rank"] <= 4) / n * 100
        avg_f = sum(l["field_size"] for l in seq_legs) / n
        avg_hhi = sum(l["hhi"] for l in seq_legs) / n
        print(f"\n  {seq_name}: {n} legs")
        print(f"    Top-1: {t1:.1f}%, Top-2: {t2:.1f}%, Top-3: {t3:.1f}%, Top-4: {t4:.1f}%")
        print(f"    Avg field: {avg_f:.1f}, Avg HHI: {avg_hhi:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 15. RECOMMENDED RULES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("15. RECOMMENDED LEG WIDTH RULES")
    print("=" * 80)

    print("""
  Based on the analysis above, here are the recommended rules:

  CLASSIFIER: Use HHI (Herfindahl-Hirschman Index) as primary signal,
  with field size as secondary adjustment.

  HHI = sum of squared implied probabilities from SP odds.
  Higher HHI = more concentrated market = more predictable.

  SKINNY VARIANT ($10 total):
  ┌──────────────────┬─────────┬────────────────────────┐
  │ Condition        │ Width   │ Rationale              │
  ├──────────────────┼─────────┼────────────────────────┤
  │ HHI >= 0.25      │ 1       │ Strong fav dominates   │
  │ HHI 0.15-0.25    │ 2       │ Top 2 cover 60%+       │
  │ HHI < 0.15       │ 2       │ Cap at 2 to keep cost  │
  └──────────────────┴─────────┴────────────────────────┘

  BALANCED VARIANT ($50 total):
  ┌──────────────────┬─────────┬────────────────────────┐
  │ Condition        │ Width   │ Rationale              │
  ├──────────────────┼─────────┼────────────────────────┤
  │ HHI >= 0.25      │ 2       │ Strong fav + saver     │
  │ HHI 0.15-0.25    │ 2-3     │ Core contenders        │
  │ HHI < 0.15       │ 3       │ Need coverage          │
  └──────────────────┴─────────┴────────────────────────┘

  WIDE VARIANT ($100 total):
  ┌──────────────────┬─────────┬────────────────────────┐
  │ Condition        │ Width   │ Rationale              │
  ├──────────────────┼─────────┼────────────────────────┤
  │ HHI >= 0.25      │ 2       │ Don't waste combos     │
  │ HHI 0.15-0.25    │ 3       │ Solid coverage         │
  │ HHI 0.12-0.15    │ 4       │ Open field needs width │
  │ HHI < 0.12       │ 4-5     │ Wide open, max cover   │
  │ Field 15+        │ +1      │ Big fields add 1       │
  └──────────────────┴─────────┴────────────────────────┘

  ADDITIONAL RULES:
  - Last leg of quaddie: always add +1 width (later races more open)
  - Maiden races: add +1 width (most unpredictable class)
  - If ALL 4 legs are HHI < 0.15: flag as RISKY, consider skip
  """)


def cum_pct_at(rank_counts, max_rank, total):
    """Cumulative percentage at a given rank."""
    return sum(rank_counts.get(r, 0) for r in range(1, max_rank + 1)) / total * 100


def hhi_width_skinny(leg):
    """Skinny variant: tight selections."""
    hhi = leg["hhi"]
    if hhi >= 0.25:
        return 1
    return 2


def hhi_width_balanced(leg):
    """Balanced variant: moderate selections."""
    hhi = leg["hhi"]
    if hhi >= 0.25:
        return 2
    elif hhi >= 0.15:
        return 2
    return 3


def hhi_width_wide(leg):
    """Wide variant: broad selections."""
    hhi = leg["hhi"]
    field = leg["field_size"]
    if hhi >= 0.25:
        return 2
    elif hhi >= 0.15:
        return 3
    elif hhi >= 0.12:
        w = 4
    else:
        w = 4
    if field >= 15:
        w += 1
    return w


def odds_width(leg):
    """Width based on favourite's SP odds."""
    fav_sp = leg["fav_sp"]
    if fav_sp <= 2.0:
        return 1
    elif fav_sp <= 3.5:
        return 2
    elif fav_sp <= 6.0:
        return 3
    return 4


def combined_width(leg):
    """Combined HHI + field size width."""
    hhi = leg["hhi"]
    field = leg["field_size"]
    base = 2
    if hhi >= 0.25:
        base = 1
    elif hhi >= 0.20:
        base = 2
    elif hhi >= 0.15:
        base = 2
    else:
        base = 3
    # Field adjustment
    if field >= 14:
        base += 1
    elif field <= 6:
        base = max(1, base - 1)
    # Cap
    return min(base, 5)


if __name__ == "__main__":
    main()
