"""Algorithmic sequence lane construction for Quaddie / Early Quaddie / Big 6.

Builds a SINGLE smart quaddie per sequence type with per-leg widths
determined by odds-shape classification (validated on 14,246 legs from
2025 Proform data). Width = include runners whose marginal capture rate >= 7%.

$50 outlay, minimum 30% flexi (max 166 combos).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

OUTLAY = 50.0
MIN_FLEXI_PCT = 30.0

# Main quaddie: 0/12 hits, -$750, -100% ROI. Early quaddie: 5/14, 35.7%.
# Suppress main quaddie until evidence shows it can be profitable.
ENABLE_MAIN_QUADDIE = False

# Minimum estimated return % thresholds — skip sequences below these
MIN_RETURN_PCT = {
    "early_quaddie": 20.0,
    "quaddie": 20.0,
    "big6": 5.0,
}


@dataclass
class LaneLeg:
    """A single leg within a sequence lane."""

    race_number: int
    runners: list[int]       # saddlecloth numbers
    runner_names: list[str]  # horse names for display
    odds_shape: str          # STANDOUT/DOMINANT/etc.
    shape_width: int         # data-driven width before constraint


@dataclass
class SmartSequence:
    """A single smart sequence bet."""

    sequence_type: str       # "Early Quaddie", "Quaddie", "Big 6"
    race_start: int
    race_end: int
    legs: list[LaneLeg]
    total_combos: int
    flexi_pct: float         # $100 / combos * 100
    total_outlay: float
    constrained: bool        # True if widths were tightened to fit flexi floor
    risk_note: str           # warning if many open legs
    estimated_return_pct: float = 0.0  # estimated return % based on leg capture probs


# Keep old classes for backward compatibility during transition
@dataclass
class SequenceLane:
    """Legacy: a single lane variant."""
    variant: str
    legs: list[LaneLeg]
    total_combos: int
    unit_price: float
    total_outlay: float
    flexi_pct: float


@dataclass
class SequenceBlock:
    """Legacy wrapper — now contains a single smart sequence."""
    sequence_type: str
    race_start: int
    race_end: int
    skinny: SequenceLane | None
    balanced: SequenceLane | None
    wide: SequenceLane | None
    recommended: str
    recommend_reason: str
    smart: SmartSequence | None = None


def build_smart_sequence(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> SmartSequence | None:
    """Build a single smart sequence bet with per-leg widths from odds shape.

    Args:
        sequence_type: "Early Quaddie", "Quaddie", or "Big 6"
        race_range: (start_race, end_race) inclusive
        leg_analysis: List of sequence leg analysis dicts from builder
        race_contexts: All race context dicts for the meeting

    Returns:
        SmartSequence or None if insufficient data.
    """
    from punty.probability import calculate_smart_quaddie_widths

    start, end = race_range
    num_legs = end - start + 1
    is_big6 = "big" in sequence_type.lower() or "6" in sequence_type

    # Map race number -> leg analysis and race context
    leg_map = {la["race_number"]: la for la in leg_analysis}
    race_map = {rc["race_number"]: rc for rc in race_contexts}

    # Build leg data with extended runner lists
    from punty.probability import SequenceLegAnalysis
    legs_data = []

    for rn in range(start, end + 1):
        la = leg_map.get(rn)
        rc = race_map.get(rn)
        if not la or not rc:
            logger.debug(f"Missing data for R{rn} in {sequence_type}")
            return None

        top_runners = list(la.get("top_runners", []))

        # Extend with runners from race context if needed (up to 8)
        if len(top_runners) < 8:
            probs = rc.get("probabilities", {})
            ranked = probs.get("probability_ranked", [])
            existing_sc = {r.get("saddlecloth", 0) for r in top_runners}
            for entry in ranked:
                if len(top_runners) >= 8:
                    break
                sc = entry.get("saddlecloth")
                if sc and sc not in existing_sc:
                    wp = entry.get("win_prob", 0)
                    if isinstance(wp, str):
                        wp = float(wp.rstrip("%")) / 100
                    top_runners.append({
                        "saddlecloth": sc,
                        "horse_name": entry.get("horse", ""),
                        "win_prob": float(wp),
                        "value_rating": 1.0,
                    })
                    existing_sc.add(sc)

        odds_shape = la.get("odds_shape", "CLEAR_FAV")
        shape_width = la.get("shape_width", 3)

        legs_data.append(SequenceLegAnalysis(
            race_number=rn,
            top_runners=top_runners,
            leg_confidence=la.get("confidence", "LOW"),
            suggested_width=la.get("suggested_width", 3),
            odds_shape=odds_shape,
            shape_width=shape_width,
        ))

    if len(legs_data) != num_legs:
        return None

    # Calculate optimal widths
    widths = calculate_smart_quaddie_widths(
        legs_data,
        budget=OUTLAY,
        min_flexi_pct=MIN_FLEXI_PCT,
        is_big6=is_big6,
    )

    # Check if we had to constrain
    original_combos = 1
    for leg in legs_data:
        original_combos *= max(leg.shape_width, 1)
    max_combos = int(OUTLAY / (MIN_FLEXI_PCT / 100.0))
    constrained = original_combos > max_combos

    # Build legs
    lane_legs = []
    for i, leg in enumerate(legs_data):
        w = widths[i]
        selected = leg.top_runners[:w]
        lane_legs.append(LaneLeg(
            race_number=leg.race_number,
            runners=[r.get("saddlecloth", 0) for r in selected],
            runner_names=[r.get("horse_name", "") for r in selected],
            odds_shape=leg.odds_shape,
            shape_width=leg.shape_width,
        ))

    total_combos = 1
    for leg in lane_legs:
        total_combos *= max(1, len(leg.runners))

    flexi_pct = round(OUTLAY / total_combos * 100, 1) if total_combos > 0 else 0

    # Risk assessment
    open_legs = sum(1 for leg in legs_data if leg.odds_shape in ("TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"))
    if open_legs >= num_legs - 1:
        risk_note = f"RISKY: {open_legs}/{num_legs} legs are open races."
    elif open_legs >= num_legs // 2:
        risk_note = f"Moderate risk: {open_legs}/{num_legs} open legs."
    else:
        risk_note = ""

    # Calculate estimated return % from leg capture probabilities
    # Each leg's capture prob = sum of win_probs for selected runners
    # Sequence hit prob = product of leg captures
    # Estimated return = hit_prob * average_dividend / outlay * 100
    leg_captures = []
    for i, leg in enumerate(lane_legs):
        selected_sc = set(leg.runners)
        cap = 0.0
        for r in legs_data[i].top_runners:
            if r.get("saddlecloth", 0) in selected_sc:
                wp = r.get("win_prob", 0)
                if isinstance(wp, str):
                    wp = float(str(wp).rstrip("%")) / 100
                cap += float(wp)
        leg_captures.append(min(cap, 1.0))

    hit_prob = 1.0
    for cap in leg_captures:
        hit_prob *= cap

    # Estimated return: if it hits, payout ~ outlay / hit_prob (fair odds)
    # But actual pools pay less due to takeout (~15%), so discount
    pool_takeout = 0.85
    est_return_pct = round(hit_prob * (1.0 / hit_prob * pool_takeout) * 100, 1) if hit_prob > 0 else 0.0
    # Simpler: est_return_pct ≈ pool_takeout * 100 = 85% for fair-value bets
    # More useful: use leg capture probs to estimate edge
    # If our captures are better than random, return > 100%
    # Use: est_return = (hit_prob * average_pool_payout) / outlay * 100
    # average_pool_payout ≈ outlay / (random_hit_prob) * pool_takeout
    random_hit = 1.0
    for leg in lane_legs:
        # Random prob = width / field_size (approx from legs_data)
        field = max(len(legs_data[lane_legs.index(leg)].top_runners), len(leg.runners))
        random_hit *= len(leg.runners) / max(field, 1)
    if random_hit > 0 and hit_prob > 0:
        est_return_pct = round((hit_prob / random_hit) * pool_takeout * 100, 1)
    else:
        est_return_pct = 0.0

    return SmartSequence(
        sequence_type=sequence_type,
        race_start=start,
        race_end=end,
        legs=lane_legs,
        total_combos=total_combos,
        flexi_pct=flexi_pct,
        total_outlay=OUTLAY,
        constrained=constrained,
        risk_note=risk_note,
        estimated_return_pct=est_return_pct,
    )


def build_all_sequence_lanes(
    total_races: int,
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> list[SequenceBlock]:
    """Build smart sequence for all applicable sequence types.

    Returns SequenceBlock for backward compatibility but the smart
    field contains the new single-quaddie output.
    """
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

        # Skip main quaddie if disabled (0/12, -100% ROI)
        if key == "quaddie" and not ENABLE_MAIN_QUADDIE:
            logger.info("Main Quaddie suppressed (ENABLE_MAIN_QUADDIE=False)")
            continue

        smart = build_smart_sequence(label, race_range, leg_analysis, race_contexts)
        if not smart:
            continue

        # Check minimum estimated return threshold
        min_ret = MIN_RETURN_PCT.get(key, 0.0)
        if min_ret > 0 and smart.estimated_return_pct < min_ret:
            logger.info(
                f"Skipping {label}: est. return {smart.estimated_return_pct:.1f}% "
                f"< minimum {min_ret:.0f}%"
            )
            continue

        # Build legacy SequenceBlock wrapper with the smart sequence as "Balanced"
        # so existing parser can still pick it up as a variant
        legacy_legs = [
            LaneLeg(
                race_number=leg.race_number,
                runners=leg.runners,
                runner_names=leg.runner_names,
                odds_shape=leg.odds_shape,
                shape_width=leg.shape_width,
            )
            for leg in smart.legs
        ]

        balanced_lane = SequenceLane(
            variant="Smart",
            legs=legacy_legs,
            total_combos=smart.total_combos,
            unit_price=round(OUTLAY / smart.total_combos, 2),
            total_outlay=OUTLAY,
            flexi_pct=smart.flexi_pct,
        )

        results.append(SequenceBlock(
            sequence_type=label,
            race_start=smart.race_start,
            race_end=smart.race_end,
            skinny=None,
            balanced=balanced_lane,
            wide=None,
            recommended="Smart",
            recommend_reason=_build_reason(smart),
            smart=smart,
        ))

    return results


def _build_reason(smart: SmartSequence) -> str:
    """Build explanation string for the smart quaddie."""
    shapes = [leg.odds_shape for leg in smart.legs]
    widths = [len(leg.runners) for leg in smart.legs]
    width_str = "x".join(str(w) for w in widths)

    locks = sum(1 for s in shapes if s in ("STANDOUT", "DOMINANT", "SHORT_PAIR"))
    open_count = sum(1 for s in shapes if s in ("TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"))

    parts = [f"{width_str} ({smart.total_combos} combos, {smart.flexi_pct:.0f}% flexi)"]

    if locks >= 2:
        parts.append(f"{locks} standout legs locked tight")
    if open_count >= 2:
        parts.append(f"{open_count} open legs need coverage")
    if smart.constrained:
        parts.append("widths constrained to maintain 30%+ flexi")
    if smart.risk_note:
        parts.append(smart.risk_note)

    return ". ".join(parts) + "."


def format_sequence_lanes(blocks: list[SequenceBlock]) -> str:
    """Format pre-built sequence lanes for injection into AI prompt context."""
    if not blocks:
        return ""

    lines = ["\n**PRE-BUILT SEQUENCE BETS (use these exact selections):**"]
    lines.append(f"One smart quaddie per sequence type. ${OUTLAY:.0f} outlay, flexi >= {MIN_FLEXI_PCT:.0f}%.")
    lines.append("Widths are set per-leg based on odds shape (validated on 14,246 legs from 2025).")

    for block in blocks:
        smart = block.smart
        if not smart:
            continue

        lines.append(f"\n{smart.sequence_type} (R{smart.race_start}--R{smart.race_end}):")

        # Per-leg detail
        for leg in smart.legs:
            runners_str = ", ".join(str(r) for r in leg.runners)
            names_str = ", ".join(leg.runner_names[:len(leg.runners)])
            lines.append(
                f"  R{leg.race_number} [{leg.odds_shape}]: "
                f"{runners_str} "
                f"({len(leg.runners)} runners) "
                f"-- {names_str}"
            )

        # Summary line in parser-compatible format
        legs_str = " / ".join(
            ", ".join(str(r) for r in leg.runners)
            for leg in smart.legs
        )
        lines.append(
            f"  Smart (${OUTLAY:.0f}): {legs_str} "
            f"({smart.total_combos} combos x "
            f"${OUTLAY / smart.total_combos:.2f} "
            f"= ${OUTLAY:.0f}) "
            f"-- {smart.flexi_pct:.0f}% flexi"
            f" -- est. return: {smart.estimated_return_pct:.0f}%"
        )

        if smart.risk_note:
            lines.append(f"  WARNING: {smart.risk_note}")

        lines.append(f"  Rationale: {block.recommend_reason}")

    return "\n".join(lines)
