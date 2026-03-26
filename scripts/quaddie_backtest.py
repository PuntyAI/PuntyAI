#!/usr/bin/env python3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Quaddie / Sequence Betting Backtest -- 2025 Proform Data
========================================================
Analyses optimal leg widths for Quaddie, Early Quaddie, and Big 6 bets
using SP-derived implied probabilities and actual race results.

Usage: python scripts/quaddie_backtest.py
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import math

# --- Configuration -----------------------------------------------------------

DATA_ROOT = Path(r"D:\Punty\DatafromProform\2025")
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Minimum races for a valid meeting
MIN_RACES_QUADDIE = 7     # Need at least 7 races for a proper quaddie
MIN_RACES_EARLY_Q = 8     # Need 8+ for early quaddie
MIN_RACES_BIG6 = 8        # Need 8+ for big 6

# Target capture rates for strategy recommendations
TARGET_SKINNY = 0.70      # 70% leg capture for skinny
TARGET_BALANCED = 0.80    # 80% for balanced
TARGET_WIDE = 0.85        # 85% for wide


# --- Data Structures ---------------------------------------------------------

@dataclass
class RunnerResult:
    tab_no: int
    name: str
    position: int           # 0=scratched, 99=DNF
    price_sp: float
    barrier: int
    implied_prob: float = 0.0
    sp_rank: int = 0        # 1=favourite, 2=2nd fav, etc.

@dataclass
class RaceResult:
    race_number: int
    distance: int
    race_class: str
    field_size: int         # After removing scratchings
    runners: list           # List[RunnerResult] sorted by implied_prob desc
    # Metrics
    top_prob: float = 0.0
    gap_to_2nd: float = 0.0
    top2_combined: float = 0.0
    top3_combined: float = 0.0
    herfindahl: float = 0.0
    winner_sp_rank: int = 0 # What rank was the winner? (1=fav won)

@dataclass
class MeetingResult:
    venue: str
    date: str
    location: str           # M/P/C
    state: str
    races: list             # List[RaceResult]


# --- Data Loading -------------------------------------------------------------

def load_all_meetings() -> list:
    """Load all 2025 Australian meetings from Proform data."""
    all_meetings = []
    total_loaded = 0

    for month in MONTHS:
        path = DATA_ROOT / month / "meetings.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        month_count = 0
        for m in data:
            track = m.get("Track", {})
            if track.get("Country") != "AUS":
                continue
            if m.get("IsBarrierTrial"):
                continue

            races_data = m.get("Races", [])
            races = []

            for r in races_data:
                runners_data = r.get("Runners", [])
                # Filter out scratchings (Position==0) and runners with no SP
                valid_runners = []
                for ru in runners_data:
                    pos = ru.get("Position", 0)
                    sp = ru.get("PriceSP", 0.0)
                    if pos == 0 or sp is None or sp <= 0:
                        continue
                    valid_runners.append(RunnerResult(
                        tab_no=ru.get("TabNo", 0),
                        name=ru.get("Name", ""),
                        position=pos,
                        price_sp=sp,
                        barrier=ru.get("Barrier", 0),
                    ))

                if len(valid_runners) < 2:
                    continue

                # Calculate implied probabilities
                raw_probs = [1.0 / ru.price_sp for ru in valid_runners]
                prob_sum = sum(raw_probs)
                for i, ru in enumerate(valid_runners):
                    ru.implied_prob = raw_probs[i] / prob_sum if prob_sum > 0 else 0

                # Sort by implied probability descending (favourite first)
                valid_runners.sort(key=lambda x: x.implied_prob, reverse=True)
                for rank, ru in enumerate(valid_runners, 1):
                    ru.sp_rank = rank

                # Find the winner
                winner_rank = 0
                for ru in valid_runners:
                    if ru.position == 1:
                        winner_rank = ru.sp_rank
                        break

                # Calculate race metrics
                probs = [ru.implied_prob for ru in valid_runners]
                top_prob = probs[0] if probs else 0
                gap = (probs[0] - probs[1]) if len(probs) >= 2 else 0
                top2 = sum(probs[:2])
                top3 = sum(probs[:3])
                hhi = sum(p * p for p in probs)

                race = RaceResult(
                    race_number=r.get("Number", 0),
                    distance=r.get("Distance", 0),
                    race_class=r.get("RaceClass", ""),
                    field_size=len(valid_runners),
                    runners=valid_runners,
                    top_prob=top_prob,
                    gap_to_2nd=gap,
                    top2_combined=top2,
                    top3_combined=top3,
                    herfindahl=hhi,
                    winner_sp_rank=winner_rank,
                )
                races.append(race)

            if not races:
                continue

            # Sort races by race_number
            races.sort(key=lambda x: x.race_number)

            meeting = MeetingResult(
                venue=track.get("Name", ""),
                date=m.get("MeetingDate", ""),
                location=track.get("Location", ""),
                state=track.get("State", ""),
                races=races,
            )
            all_meetings.append(meeting)
            month_count += 1

        total_loaded += month_count
        print(f"  {month}: {month_count} AUS meetings loaded")

    print(f"\nTotal: {total_loaded} Australian meetings loaded")
    return all_meetings


# --- Confidence Classification ------------------------------------------------

def classify_confidence(race: RaceResult) -> str:
    """Classify a race leg as HIGH/MED/LOW confidence (current system logic)."""
    if race.top_prob > 0.30 and race.gap_to_2nd > 0.10:
        return "HIGH"
    elif race.top2_combined > 0.45:
        return "MED"
    else:
        return "LOW"


def classify_hhi(race: RaceResult) -> str:
    """Classify by Herfindahl-Hirschman Index."""
    if race.herfindahl > 0.20:
        return "HIGH"  # Concentrated -- one or two dominant runners
    elif race.herfindahl > 0.12:
        return "MED"
    else:
        return "LOW"   # Competitive/open field


# --- Analysis Functions -------------------------------------------------------

def analyze_winner_capture_rates(all_races: list):
    """Q4a: Width vs Win Rate by Confidence Level."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: WINNER CAPTURE RATES BY CONFIDENCE BAND & WIDTH")
    print("=" * 80)

    # Group by confidence
    buckets = defaultdict(list)
    for race in all_races:
        conf = classify_confidence(race)
        buckets[conf].append(race)

    for conf in ["HIGH", "MED", "LOW"]:
        races = buckets[conf]
        if not races:
            continue

        n = len(races)
        # Count how often winner is in top-N
        capture = {w: 0 for w in range(1, 7)}
        for race in races:
            for w in range(1, 7):
                if race.winner_sp_rank <= w and race.winner_sp_rank > 0:
                    capture[w] += 1

        print(f"\n  {conf} confidence ({n} races):")
        print(f"    Avg top_prob: {sum(r.top_prob for r in races)/n:.3f}")
        print(f"    Avg gap_to_2nd: {sum(r.gap_to_2nd for r in races)/n:.3f}")
        print(f"    Avg HHI: {sum(r.herfindahl for r in races)/n:.3f}")
        print(f"    Avg field_size: {sum(r.field_size for r in races)/n:.1f}")
        print(f"    Winner capture rates:")
        for w in range(1, 7):
            pct = capture[w] / n * 100 if n else 0
            bar = "#" * int(pct / 2)
            marker = " <-- 70%" if abs(pct - 70) < 5 else (" <-- 85%" if abs(pct - 85) < 5 else "")
            print(f"      Width {w}: {capture[w]:>5}/{n} = {pct:5.1f}%  {bar}{marker}")

    # Also do HHI-based classification
    print("\n  --- HHI-Based Classification ---")
    hhi_buckets = defaultdict(list)
    for race in all_races:
        hhi_buckets[classify_hhi(race)].append(race)

    for conf in ["HIGH", "MED", "LOW"]:
        races = hhi_buckets[conf]
        if not races:
            continue
        n = len(races)
        capture = {w: 0 for w in range(1, 7)}
        for race in races:
            for w in range(1, 7):
                if race.winner_sp_rank <= w and race.winner_sp_rank > 0:
                    capture[w] += 1

        print(f"\n  HHI-{conf} ({n} races, avg HHI={sum(r.herfindahl for r in races)/n:.3f}):")
        for w in range(1, 7):
            pct = capture[w] / n * 100 if n else 0
            print(f"      Width {w}: {capture[w]:>5}/{n} = {pct:5.1f}%")


def analyze_optimal_widths(all_races: list):
    """Q4b: Find minimum width for target capture rates in each confidence band."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: OPTIMAL WIDTH PER CONFIDENCE BAND")
    print("=" * 80)
    print(f"  Targets: Skinny={TARGET_SKINNY:.0%}, Balanced={TARGET_BALANCED:.0%}, Wide={TARGET_WIDE:.0%}")

    # Create fine-grained probability bands
    bands = [
        ("Dominant (top>35%, gap>12%)", lambda r: r.top_prob > 0.35 and r.gap_to_2nd > 0.12),
        ("Strong fav (top>30%, gap>10%)", lambda r: r.top_prob > 0.30 and r.gap_to_2nd > 0.10),
        ("Clear top2 (top2>50%)", lambda r: r.top2_combined > 0.50 and not (r.top_prob > 0.30 and r.gap_to_2nd > 0.10)),
        ("Moderate top2 (top2>45%)", lambda r: 0.45 < r.top2_combined <= 0.50 and not (r.top_prob > 0.30 and r.gap_to_2nd > 0.10)),
        ("Open (top3>50%)", lambda r: r.top3_combined > 0.50 and r.top2_combined <= 0.45),
        ("Wide open (top3<=50%)", lambda r: r.top3_combined <= 0.50),
    ]

    for name, filt in bands:
        races = [r for r in all_races if filt(r)]
        if not races:
            continue
        n = len(races)

        # Find minimum width for each target
        results = {}
        for target_name, target in [("skinny", TARGET_SKINNY), ("balanced", TARGET_BALANCED), ("wide", TARGET_WIDE)]:
            for w in range(1, 8):
                captured = sum(1 for r in races if 0 < r.winner_sp_rank <= w)
                rate = captured / n
                if rate >= target:
                    results[target_name] = (w, rate)
                    break
            else:
                results[target_name] = (7, sum(1 for r in races if 0 < r.winner_sp_rank <= 7) / n)

        print(f"\n  {name} ({n} races):")
        for tname in ["skinny", "balanced", "wide"]:
            w, rate = results[tname]
            print(f"    {tname:>10}: width={w}, capture={rate:.1%}")


def analyze_winner_distribution(all_races: list):
    """Q4d: Where do winners come from?"""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: WINNER SP RANK DISTRIBUTION")
    print("=" * 80)

    rank_counts = defaultdict(int)
    total = 0
    for race in all_races:
        if race.winner_sp_rank > 0:
            rank_counts[race.winner_sp_rank] += 1
            total += 1

    print(f"\n  Total races with valid winners: {total}")
    cumulative = 0
    for rank in range(1, 16):
        count = rank_counts.get(rank, 0)
        cumulative += count
        pct = count / total * 100 if total else 0
        cum_pct = cumulative / total * 100 if total else 0
        bar = "#" * int(pct)
        print(f"    Rank {rank:>2}: {count:>5} ({pct:5.1f}%)  cum: {cum_pct:5.1f}%  {bar}")

    beyond = sum(v for k, v in rank_counts.items() if k > 15)
    cumulative += beyond
    print(f"    Rank 16+: {beyond:>4} ({beyond/total*100:5.1f}%)  cum: {cumulative/total*100:5.1f}%")

    # By venue type
    print("\n  Winner rank by venue type (M=Metro, P=Provincial, C=Country):")
    # We need race+meeting info, so we'll do this in a separate pass


def analyze_winner_by_venue_and_field(meetings: list):
    """Winner rank distribution split by venue type and field size."""
    print("\n  --- Winner Rank by Venue Type ---")

    loc_races = defaultdict(list)
    for m in meetings:
        for r in m.races:
            loc_races[m.location].append(r)

    for loc_name, loc_code in [("Metro", "M"), ("Provincial", "P"), ("Country", "C")]:
        races = loc_races.get(loc_code, [])
        if not races:
            continue
        n = len(races)
        rank_cum = defaultdict(int)
        for r in races:
            if r.winner_sp_rank > 0:
                for w in range(r.winner_sp_rank, 8):
                    rank_cum[w] += 1
        valid = sum(1 for r in races if r.winner_sp_rank > 0)
        print(f"\n    {loc_name} ({valid} races):")
        for w in range(1, 7):
            pct = rank_cum.get(w, 0) / valid * 100 if valid else 0
            print(f"      Top-{w}: {pct:5.1f}%")

    # By field size buckets
    print("\n  --- Winner Rank by Field Size ---")
    size_buckets = {
        "Small (2-6)": lambda r: r.field_size <= 6,
        "Medium (7-10)": lambda r: 7 <= r.field_size <= 10,
        "Large (11-14)": lambda r: 11 <= r.field_size <= 14,
        "Very Large (15+)": lambda r: r.field_size >= 15,
    }
    all_races = [r for m in meetings for r in m.races]
    for name, filt in size_buckets.items():
        races = [r for r in all_races if filt(r)]
        valid = sum(1 for r in races if r.winner_sp_rank > 0)
        if not valid:
            continue
        avg_rank = sum(r.winner_sp_rank for r in races if r.winner_sp_rank > 0) / valid
        top1 = sum(1 for r in races if r.winner_sp_rank == 1) / valid * 100
        top2 = sum(1 for r in races if 0 < r.winner_sp_rank <= 2) / valid * 100
        top3 = sum(1 for r in races if 0 < r.winner_sp_rank <= 3) / valid * 100
        top4 = sum(1 for r in races if 0 < r.winner_sp_rank <= 4) / valid * 100
        print(f"    {name:>20} ({valid:>5} races): avg_rank={avg_rank:.1f}  top1={top1:.0f}%  top2={top2:.0f}%  top3={top3:.0f}%  top4={top4:.0f}%")


def simulate_quaddies(meetings: list):
    """Q4c: Simulate different quaddie strategies across all meetings."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: QUADDIE STRATEGY SIMULATION")
    print("=" * 80)

    # Collect valid quaddie sequences
    quaddies = []        # Last 4 races
    early_quaddies = []  # First 4 races
    big6s = []           # Last 6 races

    for m in meetings:
        n_races = len(m.races)
        if n_races >= MIN_RACES_QUADDIE:
            quaddies.append((m, m.races[-4:]))
        if n_races >= MIN_RACES_EARLY_Q:
            early_quaddies.append((m, m.races[:4]))
        if n_races >= MIN_RACES_BIG6:
            big6s.append((m, m.races[-6:]))

    print(f"\n  Valid Quaddies: {len(quaddies)}")
    print(f"  Valid Early Quaddies: {len(early_quaddies)}")
    print(f"  Valid Big 6s: {len(big6s)}")

    # -- Strategy Definitions --
    # Each strategy returns the WIDTH (number of runners) for a given race/leg
    def strategy_current(race: RaceResult) -> int:
        """Current system: HIGH=1, MED=2, LOW=3-4."""
        conf = classify_confidence(race)
        if conf == "HIGH":
            return 1
        elif conf == "MED":
            return 2
        else:
            return 3

    def strategy_current_skinny(race: RaceResult) -> int:
        """Current skinny: HIGH=1, MED=1, LOW=2."""
        conf = classify_confidence(race)
        if conf == "HIGH":
            return 1
        elif conf == "MED":
            return 1
        else:
            return 2

    def strategy_current_wide(race: RaceResult) -> int:
        """Current wide: HIGH=2, MED=3, LOW=4."""
        conf = classify_confidence(race)
        if conf == "HIGH":
            return 2
        elif conf == "MED":
            return 3
        else:
            return 4

    def strategy_sp_guided(race: RaceResult) -> int:
        """SP-guided: use actual probability concentration."""
        if race.top_prob > 0.35 and race.gap_to_2nd > 0.12:
            return 1
        elif race.top_prob > 0.28 and race.gap_to_2nd > 0.08:
            return 2
        elif race.top2_combined > 0.45:
            return 2
        elif race.top3_combined > 0.55:
            return 3
        else:
            return 4

    def strategy_sp_guided_skinny(race: RaceResult) -> int:
        """SP-guided skinny: tighter version."""
        if race.top_prob > 0.30 and race.gap_to_2nd > 0.10:
            return 1
        elif race.top2_combined > 0.50:
            return 1
        elif race.top2_combined > 0.40:
            return 2
        else:
            return 3

    def strategy_fixed_2222(race: RaceResult) -> int:
        return 2

    def strategy_fixed_1223(race: RaceResult) -> int:
        """1 in strongest leg, 2 in middle two, 3 in weakest."""
        # This needs context of other legs, so we'll handle it differently
        return 2  # placeholder

    def strategy_hhi(race: RaceResult) -> int:
        """HHI-based: continuous."""
        if race.herfindahl > 0.20:
            return 1
        elif race.herfindahl > 0.15:
            return 2
        elif race.herfindahl > 0.10:
            return 3
        else:
            return 4

    def strategy_hhi_skinny(race: RaceResult) -> int:
        if race.herfindahl > 0.18:
            return 1
        elif race.herfindahl > 0.12:
            return 2
        else:
            return 3

    def strategy_fixed_1_2_2_3(legs: list) -> list:
        """Rank legs by openness, assign 1 to tightest, 3 to most open."""
        # Sort legs by HHI descending (most concentrated first)
        indexed = sorted(enumerate(legs), key=lambda x: x[1].herfindahl, reverse=True)
        widths = [0] * len(legs)
        if len(legs) == 4:
            assignment = [1, 2, 2, 3]  # tightest=1, most open=3
        elif len(legs) == 6:
            assignment = [1, 2, 2, 3, 3, 4]
        else:
            assignment = [2] * len(legs)
        for i, (orig_idx, _) in enumerate(indexed):
            widths[orig_idx] = assignment[i] if i < len(assignment) else 3
        return widths

    # -- Run Simulations --
    strategies = {
        "A: Current (balanced)": lambda race: strategy_current(race),
        "A-skinny: Current skinny": lambda race: strategy_current_skinny(race),
        "A-wide: Current wide": lambda race: strategy_current_wide(race),
        "B: SP-Guided": lambda race: strategy_sp_guided(race),
        "B-skinny: SP-Guided skinny": lambda race: strategy_sp_guided_skinny(race),
        "C: Fixed 2-2-2-2": lambda race: strategy_fixed_2222(race),
        "D: Fixed 1-2-2-3": None,  # Special handling
        "E: HHI-based": lambda race: strategy_hhi(race),
        "E-skinny: HHI skinny": lambda race: strategy_hhi_skinny(race),
    }

    for seq_name, sequences in [("QUADDIE (last 4)", quaddies), ("EARLY QUADDIE (first 4)", early_quaddies), ("BIG 6 (last 6)", big6s)]:
        if not sequences:
            continue

        print(f"\n  -- {seq_name} ({len(sequences)} meetings) --")
        print(f"  {'Strategy':<30} {'Hits':>5} {'Rate':>7} {'AvgCombos':>10} {'AvgLegs':>8} {'AvgCost($1)':>12}")
        print(f"  {'-'*28:<30} {'-----':>5} {'------':>7} {'---------':>10} {'-------':>8} {'-----------':>12}")

        for strat_name, strat_fn in strategies.items():
            hits = 0
            total_combos = 0
            total_legs_hit = 0
            total_legs = 0

            for meeting, legs in sequences:
                # Determine widths
                if strat_name == "D: Fixed 1-2-2-3":
                    widths = strategy_fixed_1_2_2_3(legs)
                else:
                    widths = [strat_fn(leg) for leg in legs]

                # Calculate combos
                combos = 1
                for w in widths:
                    combos *= w
                total_combos += combos

                # Check if all legs hit
                all_hit = True
                legs_hit = 0
                for leg, width in zip(legs, widths):
                    if leg.winner_sp_rank > 0 and leg.winner_sp_rank <= width:
                        legs_hit += 1
                    else:
                        all_hit = False

                total_legs_hit += legs_hit
                total_legs += len(legs)

                if all_hit:
                    hits += 1

            n = len(sequences)
            hit_rate = hits / n if n else 0
            avg_combos = total_combos / n if n else 0
            avg_legs = total_legs_hit / n if n else 0
            avg_cost = avg_combos  # At $1 per combo

            print(f"  {strat_name:<30} {hits:>5} {hit_rate:>6.1%} {avg_combos:>10.1f} {avg_legs:>8.2f} {avg_cost:>11.1f}")


def analyze_quaddie_economics(meetings: list):
    """Q4f: Dividend analysis and ROI estimation."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: QUADDIE ECONOMICS -- COST vs HIT RATE TRADEOFF")
    print("=" * 80)

    quaddies = []
    for m in meetings:
        if len(m.races) >= MIN_RACES_QUADDIE:
            quaddies.append((m, m.races[-4:]))

    if not quaddies:
        print("  No valid quaddies found")
        return

    # Estimate quaddie dividends from SP prices
    # Actual quaddie dividend ~ product of winner SPs scaled by pool factor
    # We'll use a simplified model: dividend = product(winner_SP) * pool_factor
    # Pool factor varies but typically 0.7-0.85 (after takeout)
    POOL_FACTOR = 0.80  # Typical tote takeout leaves ~80%

    dividends = []
    for meeting, legs in quaddies:
        winner_sps = []
        for leg in legs:
            for ru in leg.runners:
                if ru.position == 1:
                    winner_sps.append(ru.price_sp)
                    break
        if len(winner_sps) == 4:
            # Rough dividend estimate
            div = 1.0
            for sp in winner_sps:
                div *= sp
            div *= POOL_FACTOR
            dividends.append(div)

    if not dividends:
        print("  No complete quaddie results found")
        return

    dividends.sort()
    n = len(dividends)
    avg_div = sum(dividends) / n
    median_div = dividends[n // 2]
    p25 = dividends[n // 4]
    p75 = dividends[3 * n // 4]

    print(f"\n  Estimated quaddie dividends (n={n}):")
    print(f"    Mean:   ${avg_div:>10,.0f}")
    print(f"    Median: ${median_div:>10,.0f}")
    print(f"    P25:    ${p25:>10,.0f}")
    print(f"    P75:    ${p75:>10,.0f}")
    print(f"    Min:    ${min(dividends):>10,.0f}")
    print(f"    Max:    ${max(dividends):>10,.0f}")

    # Distribution
    buckets = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]
    print(f"\n  Dividend distribution:")
    prev = 0
    for b in buckets:
        count = sum(1 for d in dividends if prev <= d < b)
        pct = count / n * 100
        print(f"    ${prev:>6} - ${b:>6}: {count:>4} ({pct:5.1f}%)")
        prev = b
    count = sum(1 for d in dividends if d >= buckets[-1])
    print(f"    ${buckets[-1]:>6}+       : {count:>4} ({count/n*100:5.1f}%)")

    # ROI simulation for different strategies
    print(f"\n  -- ROI Estimation (per $1 unit) --")
    print(f"  Assumes: pool_factor={POOL_FACTOR}, dividends estimated from SP product")
    print(f"  {'Strategy':<30} {'HitRate':>8} {'AvgCost':>8} {'AvgDiv':>8} {'ExpReturn':>10} {'ROI':>8}")
    print(f"  {'-'*28:<30} {'------':>8} {'------':>8} {'------':>8} {'---------':>10} {'------':>8}")

    strategies = {
        "Current (balanced)": lambda r: (2 if classify_confidence(r) == "MED" else (1 if classify_confidence(r) == "HIGH" else 3)),
        "SP-Guided": lambda r: (1 if r.top_prob > 0.35 and r.gap_to_2nd > 0.12 else (2 if r.top_prob > 0.28 and r.gap_to_2nd > 0.08 else (2 if r.top2_combined > 0.45 else (3 if r.top3_combined > 0.55 else 4)))),
        "Fixed 2-2-2-2": lambda r: 2,
        "HHI-based": lambda r: (1 if r.herfindahl > 0.20 else (2 if r.herfindahl > 0.15 else (3 if r.herfindahl > 0.10 else 4))),
        "Tight 1-1-2-2": lambda r: (1 if r.herfindahl > 0.15 else 2),
        "Wide 2-3-3-4": lambda r: (2 if r.herfindahl > 0.15 else (3 if r.herfindahl > 0.10 else 4)),
    }

    for strat_name, strat_fn in strategies.items():
        hits = 0
        total_cost = 0
        total_payout = 0
        n_valid = 0

        for meeting, legs in quaddies:
            widths = [strat_fn(leg) for leg in legs]
            combos = 1
            for w in widths:
                combos *= w
            cost = combos  # $1 per combo

            # Check hit
            all_hit = True
            for leg, width in zip(legs, widths):
                if leg.winner_sp_rank <= 0 or leg.winner_sp_rank > width:
                    all_hit = False
                    break

            # Get dividend
            winner_sps = []
            for leg in legs:
                for ru in leg.runners:
                    if ru.position == 1:
                        winner_sps.append(ru.price_sp)
                        break
            if len(winner_sps) != 4:
                continue

            div = 1.0
            for sp in winner_sps:
                div *= sp
            div *= POOL_FACTOR

            n_valid += 1
            total_cost += cost

            if all_hit:
                hits += 1
                total_payout += div  # $1 unit, payout = div

        if n_valid:
            hit_rate = hits / n_valid
            avg_cost = total_cost / n_valid
            avg_div = total_payout / hits if hits else 0
            exp_return = total_payout / n_valid
            roi = (exp_return - avg_cost) / avg_cost * 100 if avg_cost else 0
            print(f"  {strat_name:<30} {hit_rate:>7.1%} {avg_cost:>7.0f} {avg_div:>7.0f} {exp_return:>9.1f} {roi:>7.1f}%")


def analyze_leg_position_effect(meetings: list):
    """Analyze if leg position (1st/2nd/3rd/4th) affects predictability."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: LEG POSITION EFFECT (Does leg order matter?)")
    print("=" * 80)

    quaddies = []
    for m in meetings:
        if len(m.races) >= MIN_RACES_QUADDIE:
            quaddies.append(m.races[-4:])

    if not quaddies:
        return

    for leg_idx in range(4):
        leg_name = f"Leg {leg_idx+1} (R{['N-3','N-2','N-1','N'][leg_idx]})"
        races = [q[leg_idx] for q in quaddies]
        n = len(races)
        valid = [r for r in races if r.winner_sp_rank > 0]
        nv = len(valid)
        if not nv:
            continue

        fav_wins = sum(1 for r in valid if r.winner_sp_rank == 1)
        top2 = sum(1 for r in valid if r.winner_sp_rank <= 2)
        top3 = sum(1 for r in valid if r.winner_sp_rank <= 3)
        top4 = sum(1 for r in valid if r.winner_sp_rank <= 4)
        avg_rank = sum(r.winner_sp_rank for r in valid) / nv
        avg_field = sum(r.field_size for r in valid) / nv
        avg_hhi = sum(r.herfindahl for r in valid) / nv

        print(f"\n  {leg_name} ({nv} races, avg_field={avg_field:.1f}, avg_HHI={avg_hhi:.3f}):")
        print(f"    Fav wins: {fav_wins/nv:.1%}  Top-2: {top2/nv:.1%}  Top-3: {top3/nv:.1%}  Top-4: {top4/nv:.1%}  AvgRank: {avg_rank:.2f}")


def analyze_confidence_granular(all_races: list):
    """Deep dive into probability bands for precise width rules."""
    print("\n" + "=" * 80)
    print("ANALYSIS 7: GRANULAR PROBABILITY BANDS FOR WIDTH RULES")
    print("=" * 80)

    # Create fine-grained bands based on top_prob
    prob_bands = [
        (0.00, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30),
        (0.30, 0.35), (0.35, 0.40), (0.40, 0.50), (0.50, 1.00),
    ]

    print(f"\n  {'Top Prob Band':<18} {'N':>6} {'Fav%':>6} {'Top2%':>6} {'Top3%':>6} {'Top4%':>6} {'AvgRank':>8} {'Rec Width':>10}")
    print(f"  {'-'*17:<18} {'----':>6} {'----':>6} {'----':>6} {'----':>6} {'----':>6} {'-------':>8} {'---------':>10}")

    for lo, hi in prob_bands:
        races = [r for r in all_races if lo <= r.top_prob < hi and r.winner_sp_rank > 0]
        if not races:
            continue
        n = len(races)
        fav = sum(1 for r in races if r.winner_sp_rank == 1) / n
        t2 = sum(1 for r in races if r.winner_sp_rank <= 2) / n
        t3 = sum(1 for r in races if r.winner_sp_rank <= 3) / n
        t4 = sum(1 for r in races if r.winner_sp_rank <= 4) / n
        avg_rank = sum(r.winner_sp_rank for r in races) / n

        # Recommend width for 75% capture
        rec = 1
        for w in range(1, 7):
            captured = sum(1 for r in races if r.winner_sp_rank <= w) / n
            if captured >= 0.75:
                rec = w
                break
        else:
            rec = 6

        print(f"  {lo:.2f} - {hi:.2f}       {n:>6} {fav:>5.0%} {t2:>5.0%} {t3:>5.0%} {t4:>5.0%} {avg_rank:>8.2f} {rec:>10}")

    # Gap analysis
    print(f"\n  --- Gap to 2nd Analysis ---")
    print(f"  {'Gap Band':<18} {'N':>6} {'Fav%':>6} {'Top2%':>6} {'Top3%':>6}")
    gap_bands = [(0.00, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 1.0)]
    for lo, hi in gap_bands:
        races = [r for r in all_races if lo <= r.gap_to_2nd < hi and r.winner_sp_rank > 0]
        if not races:
            continue
        n = len(races)
        fav = sum(1 for r in races if r.winner_sp_rank == 1) / n
        t2 = sum(1 for r in races if r.winner_sp_rank <= 2) / n
        t3 = sum(1 for r in races if r.winner_sp_rank <= 3) / n
        print(f"  {lo:.2f} - {hi:.2f}       {n:>6} {fav:>5.0%} {t2:>5.0%} {t3:>5.0%}")

    # HHI analysis
    print(f"\n  --- Herfindahl Index Analysis ---")
    print(f"  {'HHI Band':<18} {'N':>6} {'Fav%':>6} {'Top2%':>6} {'Top3%':>6} {'Top4%':>6} {'AvgField':>9}")
    hhi_bands = [(0.04, 0.08), (0.08, 0.10), (0.10, 0.12), (0.12, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    for lo, hi in hhi_bands:
        races = [r for r in all_races if lo <= r.herfindahl < hi and r.winner_sp_rank > 0]
        if not races:
            continue
        n = len(races)
        fav = sum(1 for r in races if r.winner_sp_rank == 1) / n
        t2 = sum(1 for r in races if r.winner_sp_rank <= 2) / n
        t3 = sum(1 for r in races if r.winner_sp_rank <= 3) / n
        t4 = sum(1 for r in races if r.winner_sp_rank <= 4) / n
        avg_field = sum(r.field_size for r in races) / n
        print(f"  {lo:.2f} - {hi:.2f}       {n:>6} {fav:>5.0%} {t2:>5.0%} {t3:>5.0%} {t4:>5.0%} {avg_field:>9.1f}")


def analyze_combo_count_vs_value(meetings: list):
    """Does fewer combos mean higher payout per unit? Cost/benefit of width."""
    print("\n" + "=" * 80)
    print("ANALYSIS 8: COMBO COUNT vs VALUE -- NARROW vs WIDE TRADEOFF")
    print("=" * 80)

    quaddies = []
    for m in meetings:
        if len(m.races) >= MIN_RACES_QUADDIE:
            quaddies.append((m, m.races[-4:]))

    if not quaddies:
        return

    POOL_FACTOR = 0.80

    # For different fixed combo counts, calculate hit rate and expected value
    width_configs = [
        ("1-1-1-1 (1 combo)", [1, 1, 1, 1]),
        ("1-1-1-2 (2 combos)", [1, 1, 1, 2]),
        ("1-1-2-2 (4 combos)", [1, 1, 2, 2]),
        ("1-2-2-2 (8 combos)", [1, 2, 2, 2]),
        ("2-2-2-2 (16 combos)", [2, 2, 2, 2]),
        ("2-2-2-3 (24 combos)", [2, 2, 2, 3]),
        ("2-2-3-3 (36 combos)", [2, 2, 3, 3]),
        ("2-3-3-3 (54 combos)", [2, 3, 3, 3]),
        ("3-3-3-3 (81 combos)", [3, 3, 3, 3]),
        ("3-3-3-4 (108 combos)", [3, 3, 3, 4]),
        ("3-3-4-4 (144 combos)", [3, 3, 4, 4]),
        ("4-4-4-4 (256 combos)", [4, 4, 4, 4]),
    ]

    # Smart allocation: assign widest to most open legs
    print(f"\n  Fixed width configs (widest assigned to most open leg by HHI):")
    print(f"  {'Config':<25} {'Combos':>6} {'Hits':>5} {'HitRate':>8} {'$50 Outlay':>10} {'AvgPayoff':>10} {'ExpVal/meet':>12} {'ROI':>8}")
    print(f"  {'-'*23:<25} {'-----':>6} {'----':>5} {'------':>8} {'---------':>10} {'---------':>10} {'-----------':>12} {'------':>8}")

    for config_name, base_widths in width_configs:
        hits = 0
        total_payout = 0
        n_valid = 0
        combos = 1
        for w in base_widths:
            combos *= w

        for meeting, legs in quaddies:
            # Sort legs by openness (lowest HHI = most open)
            # Assign widest width to most open leg
            sorted_indices = sorted(range(4), key=lambda i: legs[i].herfindahl)
            sorted_widths = sorted(base_widths, reverse=True)  # Widest first
            widths = [0] * 4
            for i, leg_idx in enumerate(sorted_indices):
                widths[leg_idx] = sorted_widths[i]

            # Check hit
            all_hit = True
            for leg, width in zip(legs, widths):
                if leg.winner_sp_rank <= 0 or leg.winner_sp_rank > width:
                    all_hit = False
                    break

            # Get dividend
            winner_sps = []
            for leg in legs:
                for ru in leg.runners:
                    if ru.position == 1:
                        winner_sps.append(ru.price_sp)
                        break
            if len(winner_sps) != 4:
                continue

            n_valid += 1
            div = 1.0
            for sp in winner_sps:
                div *= sp
            div *= POOL_FACTOR

            if all_hit:
                hits += 1
                total_payout += div

        if n_valid:
            hit_rate = hits / n_valid
            unit_stake = 50.0 / combos  # $50 total outlay spread across combos
            avg_payoff = (total_payout / hits * unit_stake) if hits else 0
            exp_val = total_payout / n_valid * unit_stake
            roi = (exp_val - 50.0) / 50.0 * 100

            print(f"  {config_name:<25} {combos:>6} {hits:>5} {hit_rate:>7.1%} {'$50':>10} {f'${avg_payoff:,.0f}':>10} {f'${exp_val:,.1f}':>12} {roi:>7.1f}%")


def generate_recommendations(all_races: list, meetings: list):
    """Generate final actionable recommendations."""
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    # Calculate key stats for recommendations
    total = sum(1 for r in all_races if r.winner_sp_rank > 0)

    # What HHI thresholds give clean breakpoints?
    print(f"\n  Based on {total} race results across {len(meetings)} meetings:")

    # Find optimal HHI thresholds
    hhi_threshold_results = {}
    for thresh in [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]:
        high_races = [r for r in all_races if r.herfindahl >= thresh and r.winner_sp_rank > 0]
        low_races = [r for r in all_races if r.herfindahl < thresh and r.winner_sp_rank > 0]
        if high_races and low_races:
            high_fav = sum(1 for r in high_races if r.winner_sp_rank == 1) / len(high_races)
            low_fav = sum(1 for r in low_races if r.winner_sp_rank == 1) / len(low_races)
            hhi_threshold_results[thresh] = {
                "n_high": len(high_races),
                "n_low": len(low_races),
                "high_fav": high_fav,
                "low_fav": low_fav,
                "separation": high_fav - low_fav,
            }

    print(f"\n  HHI Threshold Analysis (finding best split point):")
    print(f"  {'Threshold':>10} {'N_high':>7} {'N_low':>7} {'High_fav%':>10} {'Low_fav%':>10} {'Separation':>11}")
    for thresh, res in sorted(hhi_threshold_results.items()):
        print(f"  {thresh:>10.2f} {res['n_high']:>7} {res['n_low']:>7} {res['high_fav']:>9.1%} {res['low_fav']:>9.1%} {res['separation']:>10.1%}")

    # Confidence distribution for current system
    conf_dist = defaultdict(int)
    for r in all_races:
        conf_dist[classify_confidence(r)] += 1
    total_r = sum(conf_dist.values())
    print(f"\n  Current confidence distribution:")
    for c in ["HIGH", "MED", "LOW"]:
        print(f"    {c}: {conf_dist[c]} ({conf_dist[c]/total_r:.1%})")

    # The money question: what's the optimal approach?
    print("""
  ===================================================================
  DATA-DRIVEN FINDINGS & RECOMMENDED WIDTH RULES
  ===================================================================

  1. CURRENT SYSTEM IS POORLY CALIBRATED
     - HIGH confidence (top>30%, gap>10%): Fav only wins 46% of the time.
       Width=1 captures just 46%, NOT the ~70% you'd want for skinny.
       You need width=3 to reach 77% capture even in HIGH legs.
     - The current system over-trusts "strong" favourites.

  2. FAVOURITES WIN 35% OF THE TIME (OVERALL)
     - Top-2 covers 55%, Top-3 covers 69%, Top-4 covers 79%.
     - Even in "dominant" races (HHI>0.30): fav wins only 59%.
     - You ALWAYS need at least width=2 in any leg for safety.

  3. HHI IS THE BEST SINGLE PREDICTOR
     HHI > 0.30: Fav=59%, Top2=80%, Top3=89%, Top4=94% (n=2010)
     HHI 0.20-0.30: Fav=40%, Top2=63%, Top3=78%, Top4=87% (n=6837)
     HHI 0.15-0.20: Fav=31%, Top2=51%, Top3=66%, Top4=77% (n=7161)
     HHI 0.12-0.15: Fav=25%, Top2=43%, Top3=55%, Top4=67% (n=3557)
     HHI < 0.12: Fav=21%, Top2=35%, Top3=46%, Top4=55% (n=1149)

  4. BEST STRATEGY BY THE NUMBERS: Fixed 2-2-2-2 or 2-2-2-3
     - Fixed 2-2-2-2: 7.7% hit rate, 16 combos ($3.13/combo at $50)
     - 2-2-2-3 (smart): 10.6% hit rate, 24 combos ($2.08/combo)
     - Current balanced: 7.3% hit rate but 20 combos -- WORSE than fixed!
     - The current system's dynamic widths don't outperform fixed widths.
     - SP-Guided gets 9.9% but at 40 combos -- not cost efficient.

  5. LAST LEG IS HARDEST -- Leg 4 has lowest capture rates
     - Leg 1: fav=35.3%, top4=77.4%
     - Leg 4: fav=29.1%, top4=72.7%
     - Later races have bigger fields and more open markets.
     - --> Always give extra width to the last leg.

  6. METRO IS HARDER THAN PROVINCIAL/COUNTRY
     - Metro top-3 capture: 67% vs Provincial 70% vs Country 70%
     - Add +1 width for metro meetings.

  7. ALL STRATEGIES SHOW NEGATIVE ROI (-57% to -64%)
     - SP-product dividend estimate is rough, but the pattern is clear.
     - Quaddies are a VOLUME game: you need the occasional huge payout.
     - Skinny strategies hit 2-4% but when they hit, dividends are 5-10x bigger.
     - The 2-2-2-3 sweet spot: 10.6% hit, $208 avg div per $1 unit.

  ===================================================================
  RECOMMENDED LEG WIDTH RULES (for probability.py)
  ===================================================================

  Use HHI as primary classifier. Calculate from SP-implied probabilities.
  Assign widest width to the MOST OPEN leg (lowest HHI).

  SKINNY variant (~$10 outlay, 4-8 combos):
    HHI > 0.30  -> Width 1 (true standout, 59% fav win rate)
    HHI > 0.20  -> Width 1 (strong market, 40% fav)
    HHI > 0.15  -> Width 2 (need backup, fav only 31%)
    HHI <= 0.15 -> Width 2 (open, but keep cost down)
    Typical: 1x1x2x2 = 4 combos, or 1x2x2x2 = 8

  BALANCED variant (~$50 outlay, 16-24 combos):
    HHI > 0.30  -> Width 2 (cover danger even in strong races)
    HHI > 0.20  -> Width 2 (top2 = 63%)
    HHI > 0.15  -> Width 2 (top2 = 51%)
    HHI <= 0.15 -> Width 3 (must go wider, top2 only 43%)
    Typical: 2x2x2x3 = 24 combos -- BEST hit-rate/cost ratio

  WIDE variant (~$100 outlay, 36-72 combos):
    HHI > 0.30  -> Width 2 (top2 = 80%, no need for 3)
    HHI > 0.20  -> Width 3 (top3 = 78%)
    HHI > 0.15  -> Width 3 (top3 = 66%)
    HHI <= 0.15 -> Width 4 (must cast wide net)
    Typical: 2x3x3x4 = 72 combos, or 3x3x3x3 = 81

  BIG 6: Same rules but cap max width at 3 (3^6 = 729).
  Use skinny rules for Big 6 to keep combos manageable.

  EARLY QUADDIE: Fixed 2-2-2-2 works best (11.3% hit rate).
  Early races are more predictable -- simpler strategies win.

  KEY IMPLEMENTATION NOTES:
  - Always sort legs by HHI and assign widest to most open
  - Current system's HIGH/MED/LOW is outperformed by HHI-based
  - The biggest win is going from width=1 to width=2 for 'standout' legs
  - Width=1 should ONLY be used for skinny when HHI > 0.30
  """)


# --- Main ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("QUADDIE / SEQUENCE BETTING BACKTEST -- 2025 Proform Data")
    print("=" * 80)

    # Load data
    print("\nLoading meetings...")
    meetings = load_all_meetings()

    if not meetings:
        print("ERROR: No meetings loaded. Check data paths.")
        sys.exit(1)

    # Filter to meetings with enough races
    valid_meetings = [m for m in meetings if len(m.races) >= MIN_RACES_QUADDIE]
    print(f"\nMeetings with {MIN_RACES_QUADDIE}+ races: {len(valid_meetings)}")

    # Flatten all races for per-race analysis
    all_races = [r for m in meetings for r in m.races]
    print(f"Total races: {len(all_races)}")
    valid_races = [r for r in all_races if r.winner_sp_rank > 0]
    print(f"Races with valid winners: {len(valid_races)}")

    # Exclude races where winner wasn't in field (position 99, DNF, etc.)
    clean_races = [r for r in valid_races if r.winner_sp_rank <= r.field_size]
    print(f"Clean races (winner in field): {len(clean_races)}")

    # Run analyses
    analyze_winner_capture_rates(clean_races)
    analyze_optimal_widths(clean_races)
    analyze_winner_distribution(clean_races)
    analyze_winner_by_venue_and_field(meetings)
    analyze_confidence_granular(clean_races)
    simulate_quaddies(valid_meetings)
    analyze_quaddie_economics(valid_meetings)
    analyze_leg_position_effect(valid_meetings)
    analyze_combo_count_vs_value(valid_meetings)
    generate_recommendations(clean_races, valid_meetings)

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
