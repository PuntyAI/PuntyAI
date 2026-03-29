"""
Deep Quaddie Analysis from Proform Historical Data
===================================================
Analyzes 13+ months of Australian horse racing results to understand
quaddie (last-4-race) dynamics, optimal widths, and strategy performance.

Data source: D:/Punty/DatafromProform
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
import math

DATA_ROOT = Path("D:/Punty/DatafromProform")

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_all_results():
    """Load all results.json files from Proform data."""
    all_meetings = []
    for year_dir in sorted(DATA_ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            results_file = month_dir / "results.json"
            if results_file.is_file():
                with open(results_file, "r") as f:
                    data = json.load(f)
                    all_meetings.extend(data)
    return all_meetings


def extract_quaddie_races(meeting):
    """
    Extract the last 4 races of a meeting (the typical quaddie window).
    Returns list of 4 race dicts, each with sorted runners by SP.
    Only includes meetings with at least 4 races where all have results.
    """
    races = meeting.get("RaceResults", [])
    if len(races) < 4:
        return None

    # Sort by race number
    races_sorted = sorted(races, key=lambda r: r.get("RaceNumber", 0))

    # Take last 4
    last4 = races_sorted[-4:]

    # Validate: each race must have runners with positions and prices
    processed = []
    for race in last4:
        runners = race.get("Runners", [])
        # Filter to runners that actually finished and have a price
        valid = []
        for r in runners:
            pos = r.get("Position")
            price = r.get("Price")
            if pos is not None and pos > 0 and price is not None and price > 0:
                valid.append({
                    "position": pos,
                    "sp": price,
                    "tab_no": r.get("TabNo", 0),
                    "name": r.get("Runner", "Unknown"),
                    "barrier": r.get("Barrier", 0),
                })
            elif pos is not None and pos > 0:
                # Runner finished but no price (scratching handling)
                valid.append({
                    "position": pos,
                    "sp": 999.0,  # No price available
                    "tab_no": r.get("TabNo", 0),
                    "name": r.get("Runner", "Unknown"),
                    "barrier": r.get("Barrier", 0),
                })

        if not valid:
            return None

        # Sort by SP (shortest first = favourite)
        by_sp = sorted(valid, key=lambda x: x["sp"])

        # Find winner
        winners = [r for r in valid if r["position"] == 1]
        if not winners:
            return None

        processed.append({
            "race_number": race.get("RaceNumber", 0),
            "runners": valid,
            "runners_by_sp": by_sp,
            "winner": winners[0],
            "field_size": len(valid),
            "track_condition": race.get("TrackConditionLabel", ""),
        })

    return processed


# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def sp_rank_of_winner(race):
    """What SP rank was the winner? 1=favourite, 2=2nd fav, etc."""
    by_sp = race["runners_by_sp"]
    winner_tab = race["winner"]["tab_no"]
    for i, r in enumerate(by_sp):
        if r["tab_no"] == winner_tab:
            return i + 1
    return len(by_sp)


def is_favourite(race):
    """Did the favourite (shortest SP) win?"""
    return sp_rank_of_winner(race) == 1


def fav_sp(race):
    """SP of the favourite."""
    return race["runners_by_sp"][0]["sp"] if race["runners_by_sp"] else 0


def winner_sp(race):
    """SP of the winner."""
    return race["winner"]["sp"]


def categorize_sp(sp):
    """Categorize an SP into a band."""
    if sp <= 2.0:
        return "$1-$2"
    elif sp <= 3.0:
        return "$2-$3"
    elif sp <= 4.0:
        return "$3-$4"
    elif sp <= 6.0:
        return "$4-$6"
    elif sp <= 8.0:
        return "$6-$8"
    elif sp <= 12.0:
        return "$8-$12"
    elif sp <= 20.0:
        return "$12-$20"
    else:
        return "$20+"


# ---------------------------------------------------------------------------
# 3. STRATEGY SIMULATION
# ---------------------------------------------------------------------------

def strategy_top_n(race, n):
    """Select top N runners by SP (shortest priced)."""
    return [r["tab_no"] for r in race["runners_by_sp"][:n]]


def strategy_graded(race):
    """Fav<$2 → 2 wide, $2-$4 → 3 wide, $4+ → 4 wide."""
    fsp = fav_sp(race)
    if fsp < 2.0:
        return strategy_top_n(race, 2)
    elif fsp <= 4.0:
        return strategy_top_n(race, 3)
    else:
        return strategy_top_n(race, 4)


def strategy_must_include_fav(race):
    """Always include #1 + next 1-2 overlay runners (total 2-3)."""
    by_sp = race["runners_by_sp"]
    fsp = by_sp[0]["sp"] if by_sp else 99
    selections = [by_sp[0]["tab_no"]] if by_sp else []
    # Add next 1-2
    if fsp < 2.0:
        # Strong fav, add 1 more
        if len(by_sp) > 1:
            selections.append(by_sp[1]["tab_no"])
    else:
        # Weaker fav, add 2 more
        for r in by_sp[1:3]:
            selections.append(r["tab_no"])
    return selections


def strategy_min3_adaptive(race):
    """Minimum 3 runners, 4 if field > 12."""
    if race["field_size"] > 12:
        return strategy_top_n(race, 4)
    else:
        return strategy_top_n(race, 3)


def check_leg_hit(race, selections):
    """Did the winner appear in our selections?"""
    return race["winner"]["tab_no"] in selections


def simulate_strategy(quaddies, strategy_fn, name):
    """Run a strategy across all quaddies and return stats."""
    total = 0
    hits = 0
    total_combos = 0
    leg_misses = Counter()  # Which leg (0-3) missed most
    miss_winner_sps = []  # SP of winner that killed it
    miss_field_sizes = []  # Field size of killing leg
    leg_miss_details = defaultdict(list)  # leg_idx -> list of winner SPs

    for q in quaddies:
        legs = []
        all_hit = True
        for i, race in enumerate(q):
            sels = strategy_fn(race)
            legs.append(sels)
            hit = check_leg_hit(race, sels)
            if not hit:
                all_hit = False
                leg_misses[i] += 1
                miss_winner_sps.append(winner_sp(race))
                miss_field_sizes.append(race["field_size"])
                leg_miss_details[i].append(winner_sp(race))

        combos = 1
        for l in legs:
            combos *= len(l)
        total_combos += combos
        total += 1
        if all_hit:
            hits += 1

    hit_rate = hits / total * 100 if total else 0
    avg_combos = total_combos / total if total else 0

    return {
        "name": name,
        "total": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "avg_combos": avg_combos,
        "leg_misses": dict(leg_misses),
        "miss_winner_sp_mean": mean(miss_winner_sps) if miss_winner_sps else 0,
        "miss_winner_sp_median": median(miss_winner_sps) if miss_winner_sps else 0,
        "miss_field_size_mean": mean(miss_field_sizes) if miss_field_sizes else 0,
        "leg_miss_details": {k: mean(v) for k, v in leg_miss_details.items()},
    }


# ---------------------------------------------------------------------------
# 4. ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------

def analyze_winner_profiles(quaddies):
    """Section 1: Actual quaddie winner profiles."""
    print("\n" + "=" * 80)
    print("SECTION 1: ACTUAL QUADDIE WINNER PROFILES")
    print("=" * 80)

    all_winner_sps = []
    leg_winner_sps = [[], [], [], []]
    fav_counts = Counter()  # How many favs won in each quaddie
    profile_counter = Counter()  # e.g., "2 short + 1 mid + 1 long"

    for q in quaddies:
        favs_won = 0
        quad_sps = []
        for i, race in enumerate(q):
            wsp = winner_sp(race)
            all_winner_sps.append(wsp)
            leg_winner_sps[i].append(wsp)
            quad_sps.append(wsp)
            if is_favourite(race):
                favs_won += 1
        fav_counts[favs_won] += 1

        # Categorize profile
        short = sum(1 for sp in quad_sps if sp <= 3.0)
        mid = sum(1 for sp in quad_sps if 3.0 < sp <= 8.0)
        long_ = sum(1 for sp in quad_sps if sp > 8.0)
        profile_counter[f"{short}s/{mid}m/{long_}l"] += 1

    n = len(quaddies)
    print(f"\nTotal quaddies analyzed: {n}")

    print(f"\nOverall winner SP: mean=${mean(all_winner_sps):.2f}, median=${median(all_winner_sps):.2f}")

    for i in range(4):
        sps = leg_winner_sps[i]
        print(f"  Leg {i+1} (race {i+1} of last 4): mean=${mean(sps):.2f}, median=${median(sps):.2f}")

    print(f"\nFavourites winning per quaddie:")
    for fav_count in sorted(fav_counts.keys()):
        count = fav_counts[fav_count]
        pct = count / n * 100
        print(f"  {fav_count} favourites won: {count} ({pct:.1f}%)")

    print(f"\n  ALL 4 favs won (skinny scenario): {fav_counts[4] / n * 100:.1f}%")
    print(f"  3 favs + 1 non-fav: {fav_counts[3] / n * 100:.1f}%")
    print(f"  2 favs + 2 non-favs: {fav_counts[2] / n * 100:.1f}%")
    print(f"  1 fav + 3 non-favs: {fav_counts[1] / n * 100:.1f}%")
    print(f"  0 favs (all upsets): {fav_counts[0] / n * 100:.1f}%")

    print(f"\nWinner profile distribution (short=$1-$3, mid=$3-$8, long=$8+):")
    for profile, count in sorted(profile_counter.items(), key=lambda x: -x[1])[:15]:
        pct = count / n * 100
        print(f"  {profile}: {count} ({pct:.1f}%)")

    # SP band of winners
    print(f"\nWinner SP band distribution (across all legs):")
    sp_bands = Counter(categorize_sp(sp) for sp in all_winner_sps)
    band_order = ["$1-$2", "$2-$3", "$3-$4", "$4-$6", "$6-$8", "$8-$12", "$12-$20", "$20+"]
    for band in band_order:
        count = sp_bands.get(band, 0)
        pct = count / len(all_winner_sps) * 100
        print(f"  {band}: {count} ({pct:.1f}%)")


def analyze_estimated_dividends(quaddies):
    """Section 2: Estimate quaddie dividends from runner SPs."""
    print("\n" + "=" * 80)
    print("SECTION 2: ESTIMATED QUADDIE DIVIDEND ANALYSIS")
    print("=" * 80)
    print("(No actual dividend data - estimating from combined winner SPs)")
    print("(Estimate: product of winner SPs, with tote takeout ~17.5%)")

    divs = []
    fav_count_divs = defaultdict(list)

    for q in quaddies:
        combined_sp = 1.0
        favs_won = 0
        skip = False
        for race in q:
            wsp = winner_sp(race)
            if wsp >= 500:  # No real SP data, skip this quaddie
                skip = True
                break
            combined_sp *= wsp
            if is_favourite(race):
                favs_won += 1
        if skip:
            divs.append(None)
            continue
        # Apply approximate tote takeout (reduce by ~17.5% for quaddie pool)
        # Cap at reasonable max for estimation purposes
        estimated_div = min(combined_sp * 0.825, 500000)
        divs.append(estimated_div)
        fav_count_divs[favs_won].append(estimated_div)

    valid_divs = [d for d in divs if d is not None]
    if not valid_divs:
        print("No data.")
        return divs

    print(f"\nEstimated quaddie dividends (n={len(valid_divs)}, skipped {sum(1 for d in divs if d is None)} with missing SP):")
    print(f"  Mean: ${mean(valid_divs):.0f}")
    print(f"  Median: ${median(valid_divs):.0f}")
    print(f"  Std dev: ${stdev(valid_divs):.0f}" if len(valid_divs) > 1 else "")
    print(f"  Min: ${min(valid_divs):.0f}")
    print(f"  Max: ${max(valid_divs):.0f}")

    # Distribution bands
    bands = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000),
             (1000, 2000), (2000, 5000), (5000, 10000), (10000, float("inf"))]
    print(f"\nDividend distribution:")
    for lo, hi in bands:
        count = sum(1 for d in valid_divs if lo <= d < hi)
        pct = count / len(valid_divs) * 100
        label = f"${lo}-${hi}" if hi != float("inf") else f"${lo}+"
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\nBy favourites winning:")
    for fav_count in sorted(fav_count_divs.keys()):
        ds = fav_count_divs[fav_count]
        print(f"  {fav_count} favs won: n={len(ds)}, mean=${mean(ds):.0f}, median=${median(ds):.0f}")

    return divs


def analyze_strategies(quaddies):
    """Section 3: Strategy simulation."""
    print("\n" + "=" * 80)
    print("SECTION 3: STRATEGY SIMULATION (2-WIDE vs 3-WIDE vs 4-WIDE etc)")
    print("=" * 80)

    strategies = [
        ("A: Always Top 2", lambda r: strategy_top_n(r, 2)),
        ("B: Always Top 3", lambda r: strategy_top_n(r, 3)),
        ("C: Always Top 4", lambda r: strategy_top_n(r, 4)),
        ("D: Always Top 5", lambda r: strategy_top_n(r, 5)),
        ("E: Graded (fav<$2=2, $2-$4=3, $4+=4)", strategy_graded),
        ("F: Must-include fav + overlays", strategy_must_include_fav),
        ("G: Min 3, 4 if field>12", strategy_min3_adaptive),
    ]

    results = []
    for name, fn in strategies:
        result = simulate_strategy(quaddies, fn, name)
        results.append(result)

    # Print results table
    print(f"\n{'Strategy':<50} {'Hit%':>6} {'Hits':>6} {'AvgComb':>8} {'$1/comb ROI':>12}")
    print("-" * 90)

    for r in results:
        # Estimate ROI: assume average $1 per combo outlay
        # If hit, collect median estimated dividend
        # ROI = (hits * median_div) / (total * avg_combos) - 1
        # We'll calculate this properly later with estimated divs
        print(f"{r['name']:<50} {r['hit_rate']:>5.1f}% {r['hits']:>5}/{r['total']} {r['avg_combos']:>8.1f}")

    return results


def analyze_killing_legs(quaddies, strategy_results):
    """Section 4: Which leg kills the quaddie?"""
    print("\n" + "=" * 80)
    print("SECTION 4: WHICH LEG KILLS THE QUADDIE?")
    print("=" * 80)

    # Use the "Always Top 3" strategy as the primary analysis
    # But also run for all strategies
    strategies = [
        ("Top 2", lambda r: strategy_top_n(r, 2)),
        ("Top 3", lambda r: strategy_top_n(r, 3)),
        ("Top 4", lambda r: strategy_top_n(r, 4)),
        ("Graded", strategy_graded),
    ]

    for strat_name, strat_fn in strategies:
        print(f"\n--- Strategy: {strat_name} ---")

        leg_miss_count = Counter()
        leg_miss_winner_sps = defaultdict(list)
        leg_miss_field_sizes = defaultdict(list)
        first_miss_leg = Counter()  # Which leg was the FIRST miss?

        for q in quaddies:
            first_missed = None
            for i, race in enumerate(q):
                sels = strat_fn(race)
                if not check_leg_hit(race, sels):
                    leg_miss_count[i] += 1
                    leg_miss_winner_sps[i].append(winner_sp(race))
                    leg_miss_field_sizes[i].append(race["field_size"])
                    if first_missed is None:
                        first_missed = i
            if first_missed is not None:
                first_miss_leg[first_missed] += 1

        total_misses = sum(leg_miss_count.values())
        print(f"  Total leg misses: {total_misses}")
        for leg in range(4):
            count = leg_miss_count.get(leg, 0)
            pct = count / total_misses * 100 if total_misses else 0
            avg_sp = mean(leg_miss_winner_sps[leg]) if leg_miss_winner_sps[leg] else 0
            avg_fs = mean(leg_miss_field_sizes[leg]) if leg_miss_field_sizes[leg] else 0
            first_pct = first_miss_leg.get(leg, 0) / sum(first_miss_leg.values()) * 100 if first_miss_leg else 0
            print(f"  Leg {leg+1}: {count} misses ({pct:.1f}%), avg winner SP=${avg_sp:.1f}, avg field={avg_fs:.1f}, first-miss={first_pct:.1f}%")


def analyze_fav_heavy_vs_mixed(quaddies, estimated_divs):
    """Section 5: Favourite-heavy vs mixed quaddies."""
    print("\n" + "=" * 80)
    print("SECTION 5: FAVOURITE-HEAVY vs MIXED QUADDIES")
    print("=" * 80)

    # Categorize each quaddie
    categories = defaultdict(list)

    for i, q in enumerate(quaddies):
        favs = sum(1 for race in q if is_favourite(race))
        upsets = sum(1 for race in q if winner_sp(race) >= 8.0)
        big_upsets = sum(1 for race in q if winner_sp(race) >= 15.0)

        div = estimated_divs[i] if i < len(estimated_divs) and estimated_divs[i] is not None else None

        if div is None:
            continue

        if favs >= 3:
            categories["3+ favs (skinny)"].append(div)
        if favs == 4:
            categories["4 favs (ultra skinny)"].append(div)
        if upsets == 0 and favs >= 2:
            categories["No upsets, 2+ favs (solid)"].append(div)
        if upsets == 1:
            categories["Exactly 1 upset ($8+)"].append(div)
        if upsets >= 2:
            categories["2+ upsets ($8+)"].append(div)
        if big_upsets >= 1:
            categories["1+ big upset ($15+)"].append(div)

        # Mixed: 1-2 favs, 1-2 mid-price, 0-1 upset
        short = sum(1 for race in q if winner_sp(race) <= 3.0)
        mid = sum(1 for race in q if 3.0 < winner_sp(race) <= 8.0)
        if short == 2 and mid >= 1 and upsets <= 1:
            categories["Mixed (2 short + 1-2 mid + 0-1 upset)"].append(div)

    print(f"\n{'Category':<50} {'Count':>6} {'Mean$':>8} {'Med$':>8} {'Min$':>8} {'Max$':>8}")
    print("-" * 90)

    for cat in ["4 favs (ultra skinny)", "3+ favs (skinny)", "No upsets, 2+ favs (solid)",
                "Mixed (2 short + 1-2 mid + 0-1 upset)", "Exactly 1 upset ($8+)",
                "2+ upsets ($8+)", "1+ big upset ($15+)"]:
        ds = categories.get(cat, [])
        if ds:
            print(f"  {cat:<48} {len(ds):>6} ${mean(ds):>7.0f} ${median(ds):>7.0f} ${min(ds):>7.0f} ${max(ds):>7.0f}")

    # ROI scenarios (assume $1/combo unit)
    print(f"\n--- ROI SCENARIOS (assume top-3 strategy, avg ~81 combos, $1/combo = $81 outlay) ---")
    top3_combos = 81  # 3^4
    for cat in ["4 favs (ultra skinny)", "3+ favs (skinny)",
                "Mixed (2 short + 1-2 mid + 0-1 upset)", "Exactly 1 upset ($8+)",
                "2+ upsets ($8+)"]:
        ds = categories.get(cat, [])
        if ds:
            # For $1/combo (total $81 outlay), dividend is already the raw number
            # But we'd only collect if we HIT - and top-3 has ~50% hit rate
            avg_div = mean(ds)
            med_div = median(ds)
            print(f"  {cat}: mean div ${avg_div:.0f}, median ${med_div:.0f}")


def analyze_second_favourite(quaddies):
    """Section 6: Second favourite analysis."""
    print("\n" + "=" * 80)
    print("SECTION 6: SECOND FAVOURITE (AND DEPTH) ANALYSIS")
    print("=" * 80)

    total_races = 0
    fav_wins = 0
    # When fav loses, what rank was the winner?
    non_fav_winner_ranks = []
    # Overall winner rank distribution
    all_winner_ranks = []

    # When fav loses, how often does 2nd fav win?
    fav_loses_2nd_wins = 0
    fav_loses_total = 0

    for q in quaddies:
        for race in q:
            total_races += 1
            rank = sp_rank_of_winner(race)
            all_winner_ranks.append(rank)

            if rank == 1:
                fav_wins += 1
            else:
                fav_loses_total += 1
                non_fav_winner_ranks.append(rank)
                if rank == 2:
                    fav_loses_2nd_wins += 1

    print(f"\nTotal races analyzed: {total_races}")
    print(f"Favourite win rate: {fav_wins}/{total_races} ({fav_wins/total_races*100:.1f}%)")

    print(f"\nWhen favourite LOSES ({fav_loses_total} times):")
    print(f"  2nd fav wins: {fav_loses_2nd_wins} ({fav_loses_2nd_wins/fav_loses_total*100:.1f}%)")

    # Cumulative capture rate
    print(f"\n  Depth needed to capture winner (cumulative):")
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        captured = sum(1 for r in non_fav_winner_ranks if r <= depth)
        pct = captured / fav_loses_total * 100
        print(f"    Top {depth} by SP: captures {captured}/{fav_loses_total} ({pct:.1f}%)")

    # Overall winner rank distribution
    print(f"\n  Overall winner SP rank distribution:")
    rank_counts = Counter(all_winner_ranks)
    for rank in range(1, 11):
        count = rank_counts.get(rank, 0)
        pct = count / total_races * 100
        cumulative = sum(rank_counts.get(r, 0) for r in range(1, rank + 1))
        cum_pct = cumulative / total_races * 100
        print(f"    Rank {rank}: {count} ({pct:.1f}%), cumulative top-{rank}: {cum_pct:.1f}%")

    beyond10 = sum(count for rank, count in rank_counts.items() if rank > 10)
    print(f"    Rank 11+: {beyond10} ({beyond10/total_races*100:.1f}%)")

    # Minimum width needed per leg to guarantee capture
    print(f"\n  MINIMUM WIDTH TO CAPTURE 90% OF WINNERS:")
    for depth in range(1, 15):
        captured = sum(1 for r in all_winner_ranks if r <= depth)
        pct = captured / total_races * 100
        if pct >= 90:
            print(f"    Need top-{depth} to capture 90% (actual: {pct:.1f}%)")
            break

    for target in [80, 85, 90, 95]:
        for depth in range(1, 20):
            captured = sum(1 for r in all_winner_ranks if r <= depth)
            pct = captured / total_races * 100
            if pct >= target:
                print(f"    Top-{depth} captures {target}% of winners (actual: {pct:.1f}%)")
                break


def analyze_consecutive_favs(quaddies):
    """Section 7: Consecutive favourite wins."""
    print("\n" + "=" * 80)
    print("SECTION 7: CONSECUTIVE FAVOURITE WINS")
    print("=" * 80)

    # For full meetings (not just quaddies), we need meeting-level data
    # But we can use our quaddie data for the last-4 races
    total = len(quaddies)

    # Consecutive fav wins in quaddie window
    consec_counts = Counter()
    top2_consec_counts = Counter()

    four_fav_streak = 0
    four_top2_streak = 0

    for q in quaddies:
        # Consecutive favs
        max_streak = 0
        current = 0
        for race in q:
            if is_favourite(race):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        consec_counts[max_streak] += 1
        if max_streak == 4:
            four_fav_streak += 1

        # Consecutive top-2 in market wins
        max_streak2 = 0
        current2 = 0
        for race in q:
            if sp_rank_of_winner(race) <= 2:
                current2 += 1
                max_streak2 = max(max_streak2, current2)
            else:
                current2 = 0
        top2_consec_counts[max_streak2] += 1
        if max_streak2 == 4:
            four_top2_streak += 1

    print(f"\nLongest consecutive FAVOURITE wins in quaddie window:")
    for streak in sorted(consec_counts.keys()):
        count = consec_counts[streak]
        pct = count / total * 100
        print(f"  {streak} consecutive: {count} ({pct:.1f}%)")

    print(f"\n  4 consecutive favs (natural skinny quaddie): {four_fav_streak}/{total} ({four_fav_streak/total*100:.1f}%)")

    print(f"\nLongest consecutive TOP-2 IN MARKET wins in quaddie window:")
    for streak in sorted(top2_consec_counts.keys()):
        count = top2_consec_counts[streak]
        pct = count / total * 100
        print(f"  {streak} consecutive: {count} ({pct:.1f}%)")

    print(f"\n  4 consecutive top-2 winners: {four_top2_streak}/{total} ({four_top2_streak/total*100:.1f}%)")


def analyze_field_size_impact(quaddies):
    """Bonus: How does field size affect quaddie difficulty?"""
    print("\n" + "=" * 80)
    print("BONUS: FIELD SIZE IMPACT ON QUADDIE DIFFICULTY")
    print("=" * 80)

    # Categorize by average field size of the 4 legs
    small_field = []  # avg < 8
    medium_field = []  # 8-12
    large_field = []  # 12+

    for q in quaddies:
        avg_field = mean(r["field_size"] for r in q)
        favs_won = sum(1 for r in q if is_favourite(r))
        avg_winner_sp = mean(winner_sp(r) for r in q)

        entry = {"avg_field": avg_field, "favs_won": favs_won, "avg_winner_sp": avg_winner_sp}
        if avg_field < 8:
            small_field.append(entry)
        elif avg_field < 12:
            medium_field.append(entry)
        else:
            large_field.append(entry)

    for label, entries in [("Small (<8 avg field)", small_field),
                           ("Medium (8-12 avg field)", medium_field),
                           ("Large (12+ avg field)", large_field)]:
        if entries:
            avg_fav = mean(e["favs_won"] for e in entries)
            avg_sp = mean(e["avg_winner_sp"] for e in entries)
            print(f"\n  {label}: n={len(entries)}")
            print(f"    Avg favs winning: {avg_fav:.2f}")
            print(f"    Avg winner SP: ${avg_sp:.2f}")


def analyze_roi_by_strategy(quaddies, estimated_divs):
    """Detailed ROI calculation for each strategy."""
    print("\n" + "=" * 80)
    print("SECTION 8: DETAILED ROI BY STRATEGY")
    print("=" * 80)
    print("(Assumes $1 per combination unit stake)")

    strategies = [
        ("Top 2 (16 combos)", lambda r: strategy_top_n(r, 2)),
        ("Top 3 (81 combos)", lambda r: strategy_top_n(r, 3)),
        ("Top 4 (256 combos)", lambda r: strategy_top_n(r, 4)),
        ("Top 5 (625 combos)", lambda r: strategy_top_n(r, 5)),
        ("Graded", strategy_graded),
        ("Fav + overlays", strategy_must_include_fav),
        ("Min 3, 4 if big field", strategy_min3_adaptive),
    ]

    for strat_name, strat_fn in strategies:
        total_outlay = 0
        total_return = 0
        hits = 0
        total = 0

        for qi, q in enumerate(quaddies):
            total += 1
            legs = [strat_fn(race) for race in q]
            combos = 1
            for l in legs:
                combos *= len(l)
            outlay = combos  # $1 per combo

            all_hit = all(check_leg_hit(q[i], legs[i]) for i in range(4))

            total_outlay += outlay
            if all_hit:
                hits += 1
                # Estimated dividend
                div = estimated_divs[qi] if qi < len(estimated_divs) and estimated_divs[qi] is not None else 0
                if div:
                    total_return += div

        roi = (total_return - total_outlay) / total_outlay * 100 if total_outlay else 0
        avg_outlay = total_outlay / total if total else 0
        avg_return_per_quad = total_return / total if total else 0

        print(f"\n  {strat_name}:")
        print(f"    Quaddies: {total}, Hits: {hits} ({hits/total*100:.1f}%)")
        print(f"    Avg outlay: ${avg_outlay:.1f}")
        print(f"    Total outlay: ${total_outlay:,.0f}, Total return: ${total_return:,.0f}")
        print(f"    ROI: {roi:+.1f}%")
        print(f"    Avg return per quaddie: ${avg_return_per_quad:.1f}")


def analyze_fav_price_bands(quaddies):
    """Analyse how favourite's SP correlates with fav win rate per leg."""
    print("\n" + "=" * 80)
    print("BONUS: FAVOURITE WIN RATE BY FAVOURITE SP BAND")
    print("=" * 80)

    bands = defaultdict(lambda: {"total": 0, "wins": 0, "top2_wins": 0, "top3_wins": 0})

    for q in quaddies:
        for race in q:
            fsp = fav_sp(race)
            band = categorize_sp(fsp)
            bands[band]["total"] += 1
            rank = sp_rank_of_winner(race)
            if rank == 1:
                bands[band]["wins"] += 1
            if rank <= 2:
                bands[band]["top2_wins"] += 1
            if rank <= 3:
                bands[band]["top3_wins"] += 1

    band_order = ["$1-$2", "$2-$3", "$3-$4", "$4-$6", "$6-$8", "$8-$12", "$12-$20", "$20+"]
    print(f"\n{'Fav SP Band':<15} {'Races':>7} {'Fav Wins':>10} {'Top2 Wins':>10} {'Top3 Wins':>10}")
    print("-" * 60)
    for band in band_order:
        b = bands.get(band)
        if b and b["total"] > 0:
            print(f"  {band:<13} {b['total']:>7} {b['wins']/b['total']*100:>8.1f}% {b['top2_wins']/b['total']*100:>8.1f}% {b['top3_wins']/b['total']*100:>8.1f}%")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading Proform data...")
    meetings = load_all_results()
    print(f"Loaded {len(meetings)} meetings")

    # De-duplicate meetings by MeetingId (2025 and 2026 folders appear to have same data)
    seen_ids = set()
    unique_meetings = []
    for m in meetings:
        mid = m.get("MeetingId")
        if mid not in seen_ids:
            seen_ids.add(mid)
            unique_meetings.append(m)
    meetings = unique_meetings
    print(f"After de-duplication: {len(meetings)} unique meetings")

    # Filter to Australian meetings with 4+ races
    aus_meetings = [m for m in meetings if len(m.get("RaceResults", [])) >= 4]
    print(f"Meetings with 4+ races: {len(aus_meetings)}")

    # Extract quaddie windows
    quaddies = []
    for m in aus_meetings:
        q = extract_quaddie_races(m)
        if q:
            quaddies.append(q)

    print(f"Valid quaddies extracted: {len(quaddies)}")
    print(f"Total races in quaddies: {len(quaddies) * 4}")

    # Run all analyses
    analyze_winner_profiles(quaddies)
    estimated_divs = analyze_estimated_dividends(quaddies)
    analyze_strategies(quaddies)
    analyze_killing_legs(quaddies, None)
    analyze_fav_heavy_vs_mixed(quaddies, estimated_divs)
    analyze_second_favourite(quaddies)
    analyze_consecutive_favs(quaddies)
    analyze_field_size_impact(quaddies)
    analyze_fav_price_bands(quaddies)
    analyze_roi_by_strategy(quaddies, estimated_divs)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
