"""Algorithmic sequence lane construction for Quaddie / Early Quaddie / Big 6.

Builds Skinny, Balanced, and Wide lanes from probability rankings,
calculating exact combo counts and unit prices targeting fixed outlays.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Fixed outlays per variant
SKINNY_OUTLAY = 10.0
BALANCED_OUTLAY = 50.0
WIDE_OUTLAY = 100.0

# Maximum combos before a lane becomes impractical
MAX_COMBOS_SKINNY = 12
MAX_COMBOS_BALANCED = 100
MAX_COMBOS_WIDE = 500


@dataclass
class LaneLeg:
    """A single leg within a sequence lane."""

    race_number: int
    runners: list[int]       # saddlecloth numbers
    runner_names: list[str]  # horse names for display
    confidence: str          # HIGH / MED / LOW from leg analysis


@dataclass
class SequenceLane:
    """A fully constructed sequence lane (Skinny/Balanced/Wide)."""

    variant: str             # "Skinny", "Balanced", "Wide"
    legs: list[LaneLeg]
    total_combos: int
    unit_price: float        # outlay / combos (dollars per combo)
    total_outlay: float
    flexi_pct: float         # unit_price as percentage (unit_price * 100)


@dataclass
class SequenceBlock:
    """Complete sequence block with all three lanes."""

    sequence_type: str       # "Early Quaddie", "Quaddie", "Big 6"
    race_start: int
    race_end: int
    skinny: SequenceLane
    balanced: SequenceLane
    wide: SequenceLane
    recommended: str         # "Skinny" / "Balanced" / "Wide"
    recommend_reason: str


def build_sequence_lanes(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> SequenceBlock | None:
    """Build Skinny/Balanced/Wide lanes for a sequence bet.

    Args:
        sequence_type: "Early Quaddie", "Quaddie", or "Big 6"
        race_range: (start_race, end_race) inclusive
        leg_analysis: List of sequence leg analysis dicts from builder
            (race_number, confidence, suggested_width, top_runners)
        race_contexts: All race context dicts for the meeting

    Returns:
        SequenceBlock with all three lanes, or None if insufficient data.
    """
    start, end = race_range
    num_legs = end - start + 1

    # Map race number → leg analysis and race context
    leg_map = {la["race_number"]: la for la in leg_analysis}
    race_map = {rc["race_number"]: rc for rc in race_contexts}

    legs_data = []
    for rn in range(start, end + 1):
        la = leg_map.get(rn)
        rc = race_map.get(rn)
        if not la or not rc:
            logger.debug(f"Missing data for R{rn} in {sequence_type}")
            return None

        # Get top runners sorted by win probability
        top_runners = la.get("top_runners", [])

        # Extend with runners from race context if needed
        if len(top_runners) < 4:
            probs = rc.get("probabilities", {})
            ranked = probs.get("probability_ranked", [])
            existing_sc = {r.get("saddlecloth", 0) for r in top_runners}
            for entry in ranked:
                if len(top_runners) >= 6:
                    break
                sc = entry.get("saddlecloth")
                if sc and sc not in existing_sc:
                    top_runners.append({
                        "saddlecloth": sc,
                        "horse_name": entry.get("horse", ""),
                        "win_prob": float(entry.get("win_prob", "0").rstrip("%")) / 100
                        if isinstance(entry.get("win_prob"), str)
                        else entry.get("win_prob", 0),
                    })
                    existing_sc.add(sc)

        legs_data.append({
            "race_number": rn,
            "confidence": la.get("confidence", "LOW"),
            "suggested_width": la.get("suggested_width", 3),
            "top_runners": top_runners,
        })

    if len(legs_data) != num_legs:
        return None

    # Build three lane variants
    skinny = _build_lane("Skinny", legs_data, SKINNY_OUTLAY)
    balanced = _build_lane("Balanced", legs_data, BALANCED_OUTLAY)
    wide = _build_lane("Wide", legs_data, WIDE_OUTLAY)

    # Determine recommendation based on confidence distribution
    recommended, reason = _recommend_variant(legs_data)

    return SequenceBlock(
        sequence_type=sequence_type,
        race_start=start,
        race_end=end,
        skinny=skinny,
        balanced=balanced,
        wide=wide,
        recommended=recommended,
        recommend_reason=reason,
    )


def _build_lane(
    variant: str,
    legs_data: list[dict],
    outlay: float,
) -> SequenceLane:
    """Build a single lane variant from leg data."""
    lane_legs = []

    for ld in legs_data:
        rn = ld["race_number"]
        confidence = ld["confidence"]
        suggested = ld["suggested_width"]
        top = ld["top_runners"]

        # Determine width based on variant
        width = _width_for_variant(variant, confidence, suggested)

        # Select runners (top N by probability)
        selected = top[:width]

        lane_legs.append(LaneLeg(
            race_number=rn,
            runners=[r.get("saddlecloth", 0) for r in selected],
            runner_names=[r.get("horse_name", "") for r in selected],
            confidence=confidence,
        ))

    # Calculate combos = product of runners per leg
    total_combos = 1
    for leg in lane_legs:
        total_combos *= max(1, len(leg.runners))

    # Cap combos to prevent absurdly expensive bets
    max_combos = {
        "Skinny": MAX_COMBOS_SKINNY,
        "Balanced": MAX_COMBOS_BALANCED,
        "Wide": MAX_COMBOS_WIDE,
    }.get(variant, MAX_COMBOS_WIDE)

    if total_combos > max_combos:
        # Trim widest legs until under limit
        lane_legs = _trim_to_combo_limit(lane_legs, max_combos)
        total_combos = 1
        for leg in lane_legs:
            total_combos *= max(1, len(leg.runners))

    # Unit price = outlay / combos
    unit_price = round(outlay / total_combos, 2) if total_combos > 0 else outlay

    return SequenceLane(
        variant=variant,
        legs=lane_legs,
        total_combos=total_combos,
        unit_price=unit_price,
        total_outlay=outlay,
        flexi_pct=round(unit_price * 100, 1),
    )


def _width_for_variant(variant: str, confidence: str, suggested: int) -> int:
    """Determine runner width for a leg based on variant and confidence."""
    if variant == "Skinny":
        if confidence == "HIGH":
            return 1
        elif confidence == "MED":
            return min(2, suggested)
        else:  # LOW
            return min(2, suggested)

    elif variant == "Balanced":
        if confidence == "HIGH":
            return 2
        elif confidence == "MED":
            return 3
        else:  # LOW
            return min(3, suggested + 1)

    else:  # Wide
        if confidence == "HIGH":
            return 3
        elif confidence == "MED":
            return 4
        else:  # LOW
            return min(4, suggested + 1)


def _trim_to_combo_limit(
    legs: list[LaneLeg],
    max_combos: int,
) -> list[LaneLeg]:
    """Trim widest legs to bring total combos under max_combos."""
    while True:
        total = 1
        for leg in legs:
            total *= max(1, len(leg.runners))
        if total <= max_combos:
            break

        # Find widest leg and trim by 1
        widest = max(legs, key=lambda l: len(l.runners))
        if len(widest.runners) <= 1:
            break  # can't trim further
        widest.runners = widest.runners[:-1]
        widest.runner_names = widest.runner_names[:-1]

    return legs


def _recommend_variant(legs_data: list[dict]) -> tuple[str, str]:
    """Recommend Skinny/Balanced/Wide based on leg confidence distribution."""
    high = sum(1 for ld in legs_data if ld["confidence"] == "HIGH")
    med = sum(1 for ld in legs_data if ld["confidence"] == "MED")
    low = sum(1 for ld in legs_data if ld["confidence"] == "LOW")
    total = len(legs_data)

    if high >= total * 0.75:
        return "Skinny", f"{high}/{total} legs are HIGH confidence — trust the standouts."
    elif high >= total * 0.5:
        return "Balanced", f"{high} HIGH + {med} MED legs — mix of confidence levels."
    elif low >= total * 0.5:
        return "Wide", f"{low}/{total} legs are LOW confidence — need coverage."
    else:
        return "Balanced", f"Mixed confidence ({high}H/{med}M/{low}L) — Balanced covers the spread."


def build_all_sequence_lanes(
    total_races: int,
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> list[SequenceBlock]:
    """Build sequence lanes for all applicable sequence types.

    Args:
        total_races: Number of races in the meeting
        leg_analysis: sequence_leg_analysis from context
        race_contexts: All race contexts from the meeting

    Returns:
        List of SequenceBlock (one per applicable sequence type).
    """
    # Sequence range rules (same as generator._get_sequence_lanes)
    rules = {
        7:  {"early_quad": (1, 4), "quaddie": (4, 7), "big6": None},
        8:  {"early_quad": (1, 4), "quaddie": (5, 8), "big6": (3, 8)},
        9:  {"early_quad": (2, 5), "quaddie": (6, 9), "big6": (4, 9)},
        10: {"early_quad": (3, 6), "quaddie": (7, 10), "big6": (5, 10)},
        11: {"early_quad": (4, 7), "quaddie": (8, 11), "big6": (6, 11)},
        12: {"early_quad": (5, 8), "quaddie": (9, 12), "big6": (7, 12)},
    }
    sequences = rules.get(
        total_races,
        rules.get(min(rules.keys(), key=lambda k: abs(k - total_races)), {}),
    )
    if not sequences:
        return []

    results = []

    type_map = {
        "early_quad": "Early Quaddie",
        "quaddie": "Quaddie",
        "big6": "Big 6",
    }

    for key, label in type_map.items():
        race_range = sequences.get(key)
        if not race_range:
            continue

        block = build_sequence_lanes(label, race_range, leg_analysis, race_contexts)
        if block:
            results.append(block)

    return results


def format_sequence_lanes(blocks: list[SequenceBlock]) -> str:
    """Format pre-built sequence lanes for injection into AI prompt context."""
    if not blocks:
        return ""

    lines = ["\n**PRE-BUILT SEQUENCE LANES (use these exact selections):**"]

    for block in blocks:
        lines.append(f"\n{block.sequence_type} (R{block.race_start}–R{block.race_end}):")

        for lane in [block.skinny, block.balanced, block.wide]:
            legs_str = " / ".join(
                ", ".join(str(r) for r in leg.runners)
                for leg in lane.legs
            )
            lines.append(
                f"  {lane.variant} (${lane.total_outlay:.0f}): "
                f"{legs_str} "
                f"({lane.total_combos} combos x ${lane.unit_price:.2f} "
                f"= ${lane.total_outlay:.0f}) "
                f"— est. return: {lane.flexi_pct:.0f}%"
            )

        lines.append(
            f"  Recommended: {block.recommended} — {block.recommend_reason}"
        )

    return "\n".join(lines)
