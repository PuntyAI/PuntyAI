"""ABC multi-ticket sequence construction for Quaddie / Early Quaddie / Big 6.

Builds THREE tickets (A/B/C) per sequence type with per-leg widths
determined by odds-shape classification (validated on 14,246 legs from
2025 Proform data). Width = include runners whose marginal capture rate >= 7%.

Ticket A (Banker Press): 50% budget, 2 runners/leg, highest flexi
Ticket B (Value Spread): 30% budget, shape-driven widths, value swaps
Ticket C (Chaos Saver):  20% budget, wider on chaos legs, low flexi big dividend
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BASE_OUTLAY = 50.0  # baseline, scaled $30-$60 by edge strength
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

# ABC ticket budget split percentages
TICKET_A_PCT = 0.50   # Banker Press: concentrated on strongest overlays
TICKET_B_PCT = 0.30   # Value Spread: shape-driven widths, value swaps
TICKET_C_PCT = 0.20   # Chaos Saver: wide on chaos legs, longshot value

# Odds shapes classified as chaos / dividend-decider legs
CHAOS_SHAPES = {"TRIO", "MID_FAV", "OPEN_BUNCH", "WIDE_OPEN"}

# Chaos leg expansion for Ticket C
CHAOS_EXTRA_RUNNERS = 2
TICKET_C_MIN_FLEXI_PCT = 10.0  # relaxed — intentionally low flexi for big dividend


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


@dataclass
class ABCTicket:
    """A single ticket within the ABC multi-ticket structure."""
    ticket_label: str        # "A", "B", "C"
    legs: list[LaneLeg]
    total_combos: int
    flexi_pct: float
    total_outlay: float
    description: str         # "Banker Press" / "Value Spread" / "Chaos Saver"


@dataclass
class ABCSequence:
    """Three-ticket ABC structure for a single sequence type."""
    sequence_type: str
    race_start: int
    race_end: int
    tickets: list[ABCTicket]  # [A, B, C]
    total_outlay: float       # sum across all 3 tickets
    risk_note: str
    estimated_return_pct: float
    chaos_legs: list[int]     # race numbers of chaos/dividend-decider legs


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
    """Legacy wrapper — now contains ABC sequence."""
    sequence_type: str
    race_start: int
    race_end: int
    skinny: SequenceLane | None
    balanced: SequenceLane | None
    wide: SequenceLane | None
    recommended: str
    recommend_reason: str
    smart: SmartSequence | None = None
    abc: ABCSequence | None = None


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

    # Dynamic outlay: scale $30-$60 based on average edge
    all_edges = []
    for leg in legs_data:
        for r in leg.top_runners[:3]:
            e = r.get("edge", 0)
            if e > 0:
                all_edges.append(e)
    avg_positive_edge = sum(all_edges) / len(all_edges) if all_edges else 0
    edge_ratio = min(avg_positive_edge / 0.05, 1.0)
    outlay = round(MIN_OUTLAY + edge_ratio * (MAX_OUTLAY - MIN_OUTLAY), 0)
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


def _calc_estimated_return(lane_legs: list[LaneLeg], legs_data: list) -> float:
    """Calculate estimated return % from leg capture probabilities."""
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
        return round((hit_prob / random_hit) * pool_takeout * 100, 1)
    return 0.0


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
# ABC Ticket Builder
# ──────────────────────────────────────────────

def build_abc_tickets(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> ABCSequence | None:
    """Build 3-ticket ABC quaddie structure for a sequence type.

    Ticket A (Banker Press): 2 runners/leg, highest flexi, concentrated on overlays
    Ticket B (Value Spread): shape-driven widths, value swaps, standard play
    Ticket C (Chaos Saver): B widths + expanded chaos legs, low flexi, big dividend
    """
    from punty.probability import calculate_smart_quaddie_widths

    prep = _prepare_legs_data(sequence_type, race_range, leg_analysis, race_contexts)
    if not prep:
        return None
    legs_data, total_outlay, is_big6 = prep

    # Identify chaos legs
    chaos_legs = [
        leg.race_number for leg in legs_data
        if leg.odds_shape in CHAOS_SHAPES
    ]

    # Budget split
    budget_a = round(total_outlay * TICKET_A_PCT)
    budget_b = round(total_outlay * TICKET_B_PCT)
    budget_c = total_outlay - budget_a - budget_b  # remainder to avoid rounding loss

    MIN_LEG_WIDTH = 2

    # ── Ticket A: Banker Press ──
    # Fixed width of 2 per leg, top 2 runners by composite score
    widths_a = [MIN_LEG_WIDTH] * len(legs_data)
    legs_a = []
    for i, leg in enumerate(legs_data):
        selected = _select_runners_for_leg(leg, widths_a[i], value_swap=False)
        legs_a.append(_make_lane_leg(leg, selected))

    combos_a = _calc_combos(legs_a)
    flexi_a = round(budget_a / combos_a * 100, 1) if combos_a > 0 else 0

    ticket_a = ABCTicket(
        ticket_label="A",
        legs=legs_a,
        total_combos=combos_a,
        flexi_pct=flexi_a,
        total_outlay=budget_a,
        description="Banker Press",
    )

    # ── Ticket B: Value Spread ──
    # Shape-driven widths with value swaps (same as old smart quaddie)
    widths_b = calculate_smart_quaddie_widths(
        legs_data, budget=budget_b, min_flexi_pct=MIN_FLEXI_PCT, is_big6=is_big6,
    )
    legs_b = []
    for i, leg in enumerate(legs_data):
        selected = _select_runners_for_leg(leg, widths_b[i], value_swap=True)
        legs_b.append(_make_lane_leg(leg, selected))

    combos_b = _calc_combos(legs_b)
    flexi_b = round(budget_b / combos_b * 100, 1) if combos_b > 0 else 0

    ticket_b = ABCTicket(
        ticket_label="B",
        legs=legs_b,
        total_combos=combos_b,
        flexi_pct=flexi_b,
        total_outlay=budget_b,
        description="Value Spread",
    )

    # ── Ticket C: Chaos Saver ──
    # Start with B's widths, expand chaos legs by +CHAOS_EXTRA_RUNNERS
    widths_c = list(widths_b)
    for i, leg in enumerate(legs_data):
        if leg.odds_shape in CHAOS_SHAPES:
            max_available = len(leg.top_runners)
            widths_c[i] = min(widths_c[i] + CHAOS_EXTRA_RUNNERS, max_available)

    # Constrain to budget_c with relaxed flexi floor
    max_combos_c = int(budget_c / (TICKET_C_MIN_FLEXI_PCT / 100.0)) if budget_c > 0 else 1

    combos_c = 1
    for w in widths_c:
        combos_c *= max(w, 1)

    # If over budget, tighten NON-chaos legs first (preserve chaos width)
    while combos_c > max_combos_c:
        best_idx = None
        best_prob = -1
        for i, leg in enumerate(legs_data):
            if widths_c[i] > MIN_LEG_WIDTH and leg.odds_shape not in CHAOS_SHAPES:
                top_prob = leg.top_runners[0].get("win_prob", 0) if leg.top_runners else 0
                if top_prob > best_prob:
                    best_prob = top_prob
                    best_idx = i
        # If no non-chaos legs to tighten, tighten chaos legs
        if best_idx is None:
            for i, leg in enumerate(legs_data):
                if widths_c[i] > MIN_LEG_WIDTH:
                    top_prob = leg.top_runners[0].get("win_prob", 0) if leg.top_runners else 0
                    if top_prob > best_prob:
                        best_prob = top_prob
                        best_idx = i
        if best_idx is None:
            break
        widths_c[best_idx] -= 1
        combos_c = 1
        for w in widths_c:
            combos_c *= max(w, 1)

    legs_c = []
    for i, leg in enumerate(legs_data):
        selected = _select_runners_for_leg(leg, widths_c[i], value_swap=True)
        legs_c.append(_make_lane_leg(leg, selected))

    combos_c = _calc_combos(legs_c)
    flexi_c = round(budget_c / combos_c * 100, 1) if combos_c > 0 else 0

    ticket_c = ABCTicket(
        ticket_label="C",
        legs=legs_c,
        total_combos=combos_c,
        flexi_pct=flexi_c,
        total_outlay=budget_c,
        description="Chaos Saver",
    )

    # Estimated return from Ticket B (the standard play)
    est_return = _calc_estimated_return(legs_b, legs_data)
    risk = _risk_note(legs_data)

    return ABCSequence(
        sequence_type=sequence_type,
        race_start=race_range[0],
        race_end=race_range[1],
        tickets=[ticket_a, ticket_b, ticket_c],
        total_outlay=total_outlay,
        risk_note=risk,
        estimated_return_pct=est_return,
        chaos_legs=chaos_legs,
    )


# ──────────────────────────────────────────────
# Legacy: build_smart_sequence (kept for backward compat)
# ──────────────────────────────────────────────

def build_smart_sequence(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> SmartSequence | None:
    """Build a single smart sequence bet (legacy, used by tests)."""
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
    est_return = _calc_estimated_return(lane_legs, legs_data)

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
    )


# ──────────────────────────────────────────────
# Sequence lane orchestrator
# ──────────────────────────────────────────────

def build_all_sequence_lanes(
    total_races: int,
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> list[SequenceBlock]:
    """Build ABC ticket sequences for all applicable sequence types.

    Returns SequenceBlock for backward compatibility with abc field
    containing the new 3-ticket structure.
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

        abc = build_abc_tickets(label, race_range, leg_analysis, race_contexts)
        if not abc:
            continue

        # Check minimum estimated return threshold
        min_ret = MIN_RETURN_PCT.get(key, 0.0)
        if min_ret > 0 and abc.estimated_return_pct < min_ret:
            logger.info(
                f"Skipping {label}: est. return {abc.estimated_return_pct:.1f}% "
                f"< minimum {min_ret:.0f}%"
            )
            continue

        # Build legacy SequenceLane from Ticket B for backward compat
        ticket_b = abc.tickets[1]  # Value Spread
        balanced_lane = SequenceLane(
            variant="Smart",
            legs=ticket_b.legs,
            total_combos=ticket_b.total_combos,
            unit_price=round(ticket_b.total_outlay / ticket_b.total_combos, 2),
            total_outlay=ticket_b.total_outlay,
            flexi_pct=ticket_b.flexi_pct,
        )

        results.append(SequenceBlock(
            sequence_type=label,
            race_start=abc.race_start,
            race_end=abc.race_end,
            skinny=None,
            balanced=balanced_lane,
            wide=None,
            recommended="ABC",
            recommend_reason=_build_reason_abc(abc),
            smart=None,
            abc=abc,
        ))

    return results


def _build_reason_abc(abc: ABCSequence) -> str:
    """Build explanation string for ABC sequence."""
    ticket_b = abc.tickets[1]
    widths = [len(leg.runners) for leg in ticket_b.legs]
    width_str = "x".join(str(w) for w in widths)

    parts = [f"B={width_str} ({ticket_b.total_combos} combos)"]

    if abc.chaos_legs:
        parts.append(f"{len(abc.chaos_legs)} chaos legs expanded in C")
    parts.append(f"${abc.total_outlay:.0f} total across A/B/C")

    return ". ".join(parts) + "."


# ──────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────

def format_sequence_lanes(blocks: list[SequenceBlock]) -> str:
    """Format pre-built sequence lanes for injection into AI prompt context."""
    if not blocks:
        return ""

    lines = ["\n**PRE-BUILT SEQUENCE BETS (ABC multi-ticket structure):**"]
    lines.append(
        f"Three tickets per sequence: "
        f"A (Banker Press, {TICKET_A_PCT*100:.0f}%), "
        f"B (Value Spread, {TICKET_B_PCT*100:.0f}%), "
        f"C (Chaos Saver, {TICKET_C_PCT*100:.0f}%). "
        f"${MIN_OUTLAY:.0f}-${MAX_OUTLAY:.0f} total outlay (edge-scaled)."
    )

    for block in blocks:
        abc = block.abc
        if not abc:
            continue

        lines.append(f"\n{abc.sequence_type} (R{abc.race_start}--R{abc.race_end}):")

        # Per-leg detail using Ticket B as reference
        ticket_b = abc.tickets[1]
        for leg in ticket_b.legs:
            runners_str = ", ".join(str(r) for r in leg.runners)
            names_str = ", ".join(leg.runner_names[:len(leg.runners)])
            chaos_marker = " [CHAOS]" if leg.race_number in abc.chaos_legs else ""
            lines.append(
                f"  R{leg.race_number} [{leg.odds_shape}]{chaos_marker}: "
                f"{runners_str} "
                f"({len(leg.runners)} runners) "
                f"-- {names_str}"
            )

        # Three ticket lines in parser-compatible format
        for ticket in abc.tickets:
            legs_str = " / ".join(
                ", ".join(str(r) for r in leg.runners)
                for leg in ticket.legs
            )
            lines.append(
                f"  Ticket {ticket.ticket_label} (${ticket.total_outlay:.0f}): {legs_str} "
                f"({ticket.total_combos} combos x "
                f"${ticket.total_outlay / ticket.total_combos:.2f} "
                f"= ${ticket.total_outlay:.0f}) "
                f"-- {ticket.flexi_pct:.0f}% flexi"
            )

        if abc.risk_note:
            lines.append(f"  WARNING: {abc.risk_note}")

        lines.append(f"  Rationale: {block.recommend_reason}")

    return "\n".join(lines)
