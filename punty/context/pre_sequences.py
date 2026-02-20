"""Single optimised sequence construction for Quaddie / Early Quaddie / Big 6.

Builds ONE ticket per sequence type with per-leg widths determined by
odds-shape classification (validated on 14,246 legs from 2025 Proform data).

Outlay scales $30-$60 based on chaos ratio:
- Tight (banker-heavy) → low outlay, fewer combos, higher payout per hit
- Wide (chaos-heavy) → high outlay, more combos, higher hit rate
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BASE_OUTLAY = 50.0
MIN_OUTLAY = 30.0
MAX_OUTLAY = 60.0
MIN_FLEXI_PCT = 30.0

# Main quaddie re-enabled with wider legs and edge-based construction.
ENABLE_MAIN_QUADDIE = True

# Minimum estimated return % thresholds — skip sequences below these
MIN_RETURN_PCT = {
    "early_quaddie": 20.0,
    "quaddie": 20.0,
    "big6": 5.0,
}

# Odds shapes classified as chaos / dividend-decider legs
CHAOS_SHAPES = {"TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"}

# Banker shapes — tight, predictable
BANKER_SHAPES = {"STANDOUT", "DOMINANT", "SHORT_PAIR"}


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
    estimated_return_pct: float = 0.0
    hit_probability: float = 0.0     # probability all legs hit
    chaos_ratio: float = 0.0         # fraction of legs that are chaos shapes


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
    """Wrapper for sequence data."""
    sequence_type: str
    race_start: int
    race_end: int
    skinny: SequenceLane | None
    balanced: SequenceLane | None
    wide: SequenceLane | None
    recommended: str
    recommend_reason: str
    smart: SmartSequence | None = None


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

def _prepare_legs_data(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> tuple[list, float, bool] | None:
    """Prepare leg data and compute dynamic outlay.

    Returns (legs_data, outlay, is_big6) or None if insufficient data.
    """
    from punty.probability import SequenceLegAnalysis

    start, end = race_range
    num_legs = end - start + 1
    is_big6 = "big" in sequence_type.lower() or "6" in sequence_type

    leg_map = {la["race_number"]: la for la in leg_analysis}
    race_map = {rc["race_number"]: rc for rc in race_contexts}

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
                    vr = 1.0
                    for ctx_r in rc.get("runners", []):
                        if ctx_r.get("saddlecloth") == sc:
                            vr = ctx_r.get("punty_value_rating", 1.0)
                            break
                    top_runners.append({
                        "saddlecloth": sc,
                        "horse_name": entry.get("horse", ""),
                        "win_prob": float(wp),
                        "value_rating": vr,
                        "edge": 0,
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

    # Dynamic outlay: scale $30-$60 based on chaos ratio
    # More chaos legs → wider ticket → need more budget
    # More banker legs → tighter ticket → less budget needed
    chaos_count = sum(1 for leg in legs_data if leg.odds_shape in CHAOS_SHAPES)
    chaos_ratio = chaos_count / num_legs
    outlay = round(MIN_OUTLAY + chaos_ratio * (MAX_OUTLAY - MIN_OUTLAY), 0)
    outlay = max(MIN_OUTLAY, min(MAX_OUTLAY, outlay))

    return (legs_data, outlay, is_big6)


def _select_runners_for_leg(leg, width: int, value_swap: bool = True) -> list[dict]:
    """Select runners for a leg with optional value swaps.

    Runners in leg.top_runners are already sorted by composite score
    (win_prob * value blend) from the probability engine.
    """
    candidates = list(leg.top_runners)
    selected = candidates[:width]

    if value_swap and width < len(candidates):
        for extra_idx in range(width, min(width + 2, len(candidates))):
            extra = candidates[extra_idx]
            extra_vr = extra.get("value_rating", 1.0)
            if extra_vr < 1.15:
                continue
            worst_idx = None
            worst_vr = 999
            for j in range(1, len(selected)):
                vr = selected[j].get("value_rating", 1.0)
                if vr < worst_vr:
                    worst_vr = vr
                    worst_idx = j
            if worst_idx is not None and worst_vr < 0.90 and extra_vr > worst_vr + 0.25:
                selected[worst_idx] = extra

    return selected


def _make_lane_leg(leg, selected: list[dict]) -> LaneLeg:
    """Create a LaneLeg from selected runner dicts."""
    return LaneLeg(
        race_number=leg.race_number,
        runners=[r.get("saddlecloth", 0) for r in selected],
        runner_names=[r.get("horse_name", "") for r in selected],
        odds_shape=leg.odds_shape,
        shape_width=leg.shape_width,
    )


def _calc_combos(lane_legs: list[LaneLeg]) -> int:
    """Calculate total combinations from lane legs."""
    combos = 1
    for leg in lane_legs:
        combos *= max(1, len(leg.runners))
    return combos


def _calc_estimated_return(lane_legs: list[LaneLeg], legs_data: list) -> tuple[float, float]:
    """Calculate estimated return % and hit probability from leg capture probabilities.

    Returns (estimated_return_pct, hit_probability).
    """
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

    pool_takeout = 0.85
    random_hit = 1.0
    for i, leg in enumerate(lane_legs):
        field = max(len(legs_data[i].top_runners), len(leg.runners))
        random_hit *= len(leg.runners) / max(field, 1)

    if random_hit > 0 and hit_prob > 0:
        est_return = round((hit_prob / random_hit) * pool_takeout * 100, 1)
    else:
        est_return = 0.0

    return (est_return, round(hit_prob * 100, 1))


def _risk_note(legs_data: list) -> str:
    """Generate risk note from leg shapes."""
    num_legs = len(legs_data)
    open_legs = sum(1 for leg in legs_data if leg.odds_shape in CHAOS_SHAPES)
    if open_legs >= num_legs - 1:
        return f"RISKY: {open_legs}/{num_legs} legs are open races."
    elif open_legs >= num_legs // 2:
        return f"Moderate risk: {open_legs}/{num_legs} open legs."
    return ""


# ──────────────────────────────────────────────
# Smart Sequence Builder (single optimised ticket)
# ──────────────────────────────────────────────

def build_smart_sequence(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> SmartSequence | None:
    """Build a single optimised sequence bet.

    Uses shape-driven widths constrained to a 30% flexi floor.
    Outlay scales $30-$60 based on chaos ratio:
    - Banker-heavy → $30 (tight, fewer combos, bigger payout per hit)
    - Chaos-heavy → $60 (wide, more combos, higher hit rate)
    """
    from punty.probability import calculate_smart_quaddie_widths

    prep = _prepare_legs_data(sequence_type, race_range, leg_analysis, race_contexts)
    if not prep:
        return None
    legs_data, outlay, is_big6 = prep

    widths = calculate_smart_quaddie_widths(
        legs_data, budget=outlay, min_flexi_pct=MIN_FLEXI_PCT, is_big6=is_big6,
    )

    original_combos = 1
    for leg in legs_data:
        original_combos *= max(leg.shape_width, 1)
    max_combos = int(outlay / (MIN_FLEXI_PCT / 100.0))
    constrained = original_combos > max_combos

    lane_legs = []
    for i, leg in enumerate(legs_data):
        selected = _select_runners_for_leg(leg, widths[i], value_swap=True)
        lane_legs.append(_make_lane_leg(leg, selected))

    total_combos = _calc_combos(lane_legs)
    flexi_pct = round(outlay / total_combos * 100, 1) if total_combos > 0 else 0
    risk = _risk_note(legs_data)
    est_return, hit_prob = _calc_estimated_return(lane_legs, legs_data)

    num_legs = race_range[1] - race_range[0] + 1
    chaos_count = sum(1 for leg in legs_data if leg.odds_shape in CHAOS_SHAPES)
    chaos_ratio = round(chaos_count / num_legs, 2)

    return SmartSequence(
        sequence_type=sequence_type,
        race_start=race_range[0],
        race_end=race_range[1],
        legs=lane_legs,
        total_combos=total_combos,
        flexi_pct=flexi_pct,
        total_outlay=outlay,
        constrained=constrained,
        risk_note=risk,
        estimated_return_pct=est_return,
        hit_probability=hit_prob,
        chaos_ratio=chaos_ratio,
    )


# ──────────────────────────────────────────────
# Sequence lane orchestrator
# ──────────────────────────────────────────────

def build_all_sequence_lanes(
    total_races: int,
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> list[SequenceBlock]:
    """Build single optimised sequence for each applicable sequence type.

    Returns SequenceBlock with smart field containing the optimised ticket.
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

        # Build legacy SequenceLane for backward compat
        balanced_lane = SequenceLane(
            variant="Smart",
            legs=smart.legs,
            total_combos=smart.total_combos,
            unit_price=round(smart.total_outlay / smart.total_combos, 2),
            total_outlay=smart.total_outlay,
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
    """Build explanation string for smart sequence."""
    widths = [len(leg.runners) for leg in smart.legs]
    width_str = "x".join(str(w) for w in widths)

    parts = [f"{width_str} ({smart.total_combos} combos)"]
    parts.append(f"Hit prob {smart.hit_probability:.0f}%")
    parts.append(f"Est return {smart.estimated_return_pct:.0f}%")

    if smart.chaos_ratio >= 0.5:
        chaos_count = int(smart.chaos_ratio * len(smart.legs))
        parts.append(f"{chaos_count}/{len(smart.legs)} chaos legs → wide & higher outlay")
    elif smart.chaos_ratio == 0:
        parts.append("All banker legs → tight & low outlay")

    return ". ".join(parts) + "."


# ──────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────

def format_sequence_lanes(blocks: list[SequenceBlock]) -> str:
    """Format pre-built sequence lanes for injection into AI prompt context."""
    if not blocks:
        return ""

    lines = ["\n**PRE-BUILT SEQUENCE BETS (single optimised ticket per type):**"]
    lines.append(
        f"One ticket per sequence. "
        f"Outlay ${MIN_OUTLAY:.0f}-${MAX_OUTLAY:.0f} scaled by chaos ratio "
        f"(banker-heavy = low outlay, chaos-heavy = high outlay). "
        f"Minimum {MIN_FLEXI_PCT:.0f}% flexi."
    )

    for block in blocks:
        smart = block.smart
        if not smart:
            continue

        lines.append(f"\n{smart.sequence_type} (R{smart.race_start}--R{smart.race_end}):")

        # Per-leg detail with shape and runners
        for leg in smart.legs:
            runners_str = ", ".join(str(r) for r in leg.runners)
            names_str = ", ".join(leg.runner_names[:len(leg.runners)])
            chaos_marker = " [CHAOS]" if leg.odds_shape in CHAOS_SHAPES else ""
            banker_marker = " [BANKER]" if leg.odds_shape in BANKER_SHAPES else ""
            lines.append(
                f"  R{leg.race_number} [{leg.odds_shape}]{chaos_marker}{banker_marker}: "
                f"{runners_str} "
                f"({len(leg.runners)} runners) "
                f"-- {names_str}"
            )

        # Single ticket line in parser-compatible format
        legs_str = " / ".join(
            ", ".join(str(r) for r in leg.runners)
            for leg in smart.legs
        )
        unit_price = smart.total_outlay / smart.total_combos if smart.total_combos > 0 else 0
        lines.append(
            f"  Smart: {legs_str} "
            f"({smart.total_combos} combos x "
            f"${unit_price:.2f} "
            f"= ${smart.total_outlay:.0f}) "
            f"-- {smart.flexi_pct:.0f}% flexi"
        )

        # Metrics line
        lines.append(
            f"  Metrics: Hit prob {smart.hit_probability:.0f}% | "
            f"Est return {smart.estimated_return_pct:.0f}% | "
            f"Chaos ratio {smart.chaos_ratio:.0%} | "
            f"Risk: {'HIGH' if 'RISKY' in smart.risk_note else 'MODERATE' if smart.risk_note else 'LOW'}"
        )

        if smart.risk_note:
            lines.append(f"  WARNING: {smart.risk_note}")

        lines.append(f"  Rationale: {block.recommend_reason}")

    return "\n".join(lines)
