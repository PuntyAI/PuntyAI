"""Single optimised sequence construction for Quaddie / Early Quaddie / Big 6.

Builds ONE ticket per sequence type using edge-driven overlay-only selection.
Legs classified as anchor (single) / chaos (wide) / normal based on
probability distribution and favourite odds.

Outlay range $40-$60, budget-optimised to trim/add runners by edge.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BASE_OUTLAY = 50.0
MIN_OUTLAY = 40.0
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
    field_size: int = 8      # actual number of non-scratched runners


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
                    odds = None
                    edge = 0
                    for ctx_r in rc.get("runners", []):
                        if ctx_r.get("saddlecloth") == sc:
                            vr = ctx_r.get("punty_value_rating", 1.0)
                            odds = ctx_r.get("current_odds")
                            edge = ctx_r.get("_edge_raw", 0)
                            break
                    top_runners.append({
                        "saddlecloth": sc,
                        "horse_name": entry.get("horse", ""),
                        "win_prob": float(wp),
                        "value_rating": vr,
                        "edge": edge,
                        "current_odds": odds,
                    })
                    existing_sc.add(sc)

        odds_shape = la.get("odds_shape", "CLEAR_FAV")
        shape_width = la.get("shape_width", 3)

        # Count actual non-scratched runners for accurate return estimates
        all_runners = rc.get("runners", [])
        actual_field = sum(
            1 for r in all_runners
            if not r.get("scratched") and r.get("saddlecloth")
        )
        if actual_field < 2:
            actual_field = max(len(top_runners), 8)

        legs_data.append(SequenceLegAnalysis(
            race_number=rn,
            top_runners=top_runners,
            leg_confidence=la.get("confidence", "LOW"),
            suggested_width=la.get("suggested_width", 3),
            odds_shape=odds_shape,
            shape_width=shape_width,
        ))
        # Stash field size on the leg for use in return estimates
        legs_data[-1]._field_size = actual_field

    if len(legs_data) != num_legs:
        return None

    # Dynamic outlay: scale $40-$60 based on chaos ratio
    # More chaos legs → wider ticket → need more budget
    # More banker legs → tighter ticket → less budget needed
    chaos_count = sum(1 for leg in legs_data if _is_chaos_leg(leg))
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
            extra_wp = float(extra.get("win_prob", 0))
            if extra_vr < 1.20:
                continue
            worst_idx = None
            worst_vr = 999
            for j in range(1, len(selected)):
                vr = selected[j].get("value_rating", 1.0)
                if vr < worst_vr:
                    worst_vr = vr
                    worst_idx = j
            if worst_idx is not None and worst_vr < 0.85:
                worst_wp = float(selected[worst_idx].get("win_prob", 0))
                # Only swap if replacement has reasonable win probability (>60% of displaced)
                if extra_vr > worst_vr + 0.30 and extra_wp > worst_wp * 0.6:
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
        field_size=getattr(leg, "_field_size", 8),
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
        # Use actual field size (not capped top_runners) for accurate random baseline
        field = max(leg.field_size, len(leg.runners))
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
# Leg classification helpers
# ──────────────────────────────────────────────

def _get_fav_odds(leg) -> float:
    """Get favourite odds for a leg (lowest odds among top runners)."""
    best = 999.0
    for r in leg.top_runners:
        odds = r.get("current_odds")
        if odds and float(odds) > 1.0:
            best = min(best, float(odds))
    # Fallback: derive from win prob
    if best >= 999.0:
        top_wp = max((float(r.get("win_prob", 0)) for r in leg.top_runners), default=0)
        if top_wp > 0:
            best = 1.0 / top_wp
    return best


def _is_chaos_leg(leg) -> bool:
    """True if ANY: field_size>=14, fav_odds>=5.0, p_top1<=0.22, (p_top1-p_top3)<=0.06."""
    field_size = getattr(leg, "_field_size", len(leg.top_runners))
    if field_size >= 14:
        return True

    fav_odds = _get_fav_odds(leg)
    if fav_odds >= 5.0:
        return True

    probs = sorted(
        (float(r.get("win_prob", 0)) for r in leg.top_runners),
        reverse=True,
    )
    p_top1 = probs[0] if probs else 0
    p_top3 = probs[2] if len(probs) > 2 else 0

    if p_top1 <= 0.22:
        return True
    if len(probs) >= 3 and (p_top1 - p_top3) <= 0.06:
        return True

    return False


def _is_anchor_leg(leg) -> bool:
    """True if ALL: p_top1>=0.30, fav_odds<=4.0, gap>=0.10, NOT chaos."""
    if _is_chaos_leg(leg):
        return False

    probs = sorted(
        (float(r.get("win_prob", 0)) for r in leg.top_runners),
        reverse=True,
    )
    p_top1 = probs[0] if probs else 0
    p_top2 = probs[1] if len(probs) > 1 else 0

    if p_top1 < 0.30:
        return False
    if (p_top1 - p_top2) < 0.10:
        return False

    fav_odds = _get_fav_odds(leg)
    if fav_odds > 4.0:
        return False

    return True


# ──────────────────────────────────────────────
# EV proxy calculation (from strategy spec)
# ──────────────────────────────────────────────

def _calc_ev_proxy(selected: list[list[dict]], legs_data: list) -> float:
    """Calculate EV proxy = ticket_hit_prob × payout_proxy.

    ticket_hit_prob = product of (sum of p_true_win for selected runners) per leg
    payout_proxy = product of (average market_win_odds of selected runners) per leg
    """
    hit_prob = 1.0
    payout = 1.0
    for i, sel in enumerate(selected):
        if not sel:
            return 0.0
        leg_prob = sum(float(r.get("win_prob", 0)) for r in sel)
        leg_prob = min(leg_prob, 1.0)
        hit_prob *= leg_prob

        odds_vals = [float(r.get("current_odds", 0)) for r in sel if float(r.get("current_odds", 0)) > 1.0]
        if odds_vals:
            avg_odds = sum(odds_vals) / len(odds_vals)
        else:
            # Derive from win prob as fallback
            avg_prob = sum(float(r.get("win_prob", 0)) for r in sel) / len(sel)
            avg_odds = (1.0 / avg_prob) if avg_prob > 0 else 10.0
        payout *= avg_odds

    return hit_prob * payout


# ──────────────────────────────────────────────
# Core optimiser: edge-driven selection
# ──────────────────────────────────────────────

def _optimiser_select(
    legs_data: list,
    budget: float,
    is_big6: bool = False,
) -> list[list[dict]] | None:
    """Edge-driven overlay-only selection for sequence legs.

    Returns list of selected runner dicts per leg, or None to PASS.
    """
    num_legs = len(legs_data)

    # Step 1: Build overlay pools per leg
    overlay_pools = []
    leg_types = []  # "anchor", "chaos", "normal"

    for leg in legs_data:
        runners = list(leg.top_runners)
        # Sort by edge descending, tie-break by win_prob descending
        pool = sorted(
            runners,
            key=lambda r: (float(r.get("edge", 0)), float(r.get("win_prob", 0))),
            reverse=True,
        )
        # Filter to overlays (edge > 0)
        overlays = [r for r in pool if float(r.get("edge", 0)) > 0]
        # Allow up to ONE neutral runner per leg (edge >= -0.01) if needed
        neutrals = [r for r in pool if -0.01 <= float(r.get("edge", 0)) <= 0]
        # Attach neutrals separately — used only when overlays can't fill min width
        overlay_pools.append({"overlays": overlays, "neutrals": neutrals[:1]})

        # Classify leg
        if _is_anchor_leg(leg):
            leg_types.append("anchor")
        elif _is_chaos_leg(leg):
            leg_types.append("chaos")
        else:
            leg_types.append("normal")

    # Step 2: Viability check — need at least one anchor OR one strong runner
    has_anchor = "anchor" in leg_types
    has_strong = any(
        float(r.get("win_prob", 0)) >= 0.32
        for leg in legs_data
        for r in leg.top_runners[:1]  # top runner only
    )
    if not has_anchor and not has_strong:
        logger.info("PASS: no anchor leg and no strong runner (>=32%)")
        return None

    # Step 3: Set initial targets per leg type
    targets = []
    for i, lt in enumerate(leg_types):
        field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
        pool_depth = len(overlay_pools[i]["overlays"]) + len(overlay_pools[i]["neutrals"])
        if lt == "anchor":
            targets.append(2)  # Min 2-wide even for anchors (single legs = 30% SR vs 64%+ wider)
        elif lt == "chaos":
            targets.append(min(5, pool_depth))
        else:
            # Normal: 2-4 based on field size and overlay depth
            if field_size >= 12:
                targets.append(min(4, pool_depth))
            else:
                targets.append(min(3, pool_depth))

    # Step 4: Apply field-size caps
    for i in range(num_legs):
        field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
        if field_size <= 10:
            targets[i] = min(targets[i], 3)
        elif field_size <= 13:
            targets[i] = min(targets[i], 4)
        elif field_size <= 16:
            targets[i] = min(targets[i], 6)
        # Never >= 50% of field
        max_half = max(1, field_size // 2 - 1) if field_size > 2 else 1
        targets[i] = min(targets[i], max_half)
        # Ensure at least 2 (min 2-wide rule from backtest)
        targets[i] = max(targets[i], 2)

    # Big6 tighter caps
    if is_big6:
        for i in range(num_legs):
            targets[i] = min(targets[i], 3)

    # Step 5: Populate legs
    # Anchor legs: top win_prob runners (favorites = short-priced, predictable)
    # Normal/chaos legs: overlay-only selection (edge-driven)
    selected = []
    for i in range(num_legs):
        if leg_types[i] == "anchor":
            # Anchor = most probable runners, sorted by win_prob descending
            by_prob = sorted(
                legs_data[i].top_runners,
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )
            sel = by_prob[:targets[i]]
        else:
            # Normal/chaos: overlays first, then ONE neutral, then probability padding
            overlays = overlay_pools[i]["overlays"]
            neutrals = overlay_pools[i]["neutrals"]
            sel = overlays[:targets[i]]
            # If overlays can't fill target, allow ONE neutral runner (edge >= -0.01)
            if len(sel) < targets[i] and neutrals:
                existing_sc = {r.get("saddlecloth") for r in sel}
                for r in neutrals:
                    if len(sel) >= targets[i]:
                        break
                    if r.get("saddlecloth") not in existing_sc:
                        sel.append(r)
                        existing_sc.add(r.get("saddlecloth"))
        # If still under min 2, pad with best win_prob runner (last resort)
        if len(sel) < 2:
            existing_sc = {r.get("saddlecloth") for r in sel}
            by_prob = sorted(
                legs_data[i].top_runners,
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )
            for r in by_prob:
                if len(sel) >= 2:
                    break
                if r.get("saddlecloth") not in existing_sc:
                    sel.append(r)
                    existing_sc.add(r.get("saddlecloth"))
        selected.append(sel)

    # Step 6: Budget optimisation loop
    def _total_combos():
        c = 1
        for s in selected:
            c *= max(1, len(s))
        return c

    budget_hi = budget
    budget_lo = budget * 0.85
    budget_floor = budget * 0.70

    # Trim phase: remove runner causing smallest EV_proxy drop, respecting min 2-wide
    for _ in range(20):
        combos = _total_combos()
        cost = combos * (MIN_FLEXI_PCT / 100.0)
        if cost <= budget_hi:
            break
        current_ev = _calc_ev_proxy(selected, legs_data)
        best_trim_leg = -1
        best_trim_j = -1
        best_ev_loss = 999
        for i in range(num_legs):
            if len(selected[i]) <= 2:  # Never trim below min 2-wide
                continue
            for j in range(len(selected[i])):
                test = [list(s) for s in selected]
                test[i].pop(j)
                new_ev = _calc_ev_proxy(test, legs_data)
                ev_loss = current_ev - new_ev
                if ev_loss < best_ev_loss:
                    best_ev_loss = ev_loss
                    best_trim_leg = i
                    best_trim_j = j
        if best_trim_leg < 0:
            break
        selected[best_trim_leg].pop(best_trim_j)

    # Add phase: add overlays that give best EV_proxy increase per combo cost
    for _ in range(20):
        combos = _total_combos()
        cost = combos * (MIN_FLEXI_PCT / 100.0)
        if cost >= budget_lo:
            break
        # Find best candidate to add (overlays only, by EV impact)
        best_add = None
        best_ev_gain = -999
        best_leg_idx = -1
        current_ev = _calc_ev_proxy(selected, legs_data)
        for i in range(num_legs):
            field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
            # Respect field-size caps
            max_sel = 6
            if field_size <= 10:
                max_sel = 3
            elif field_size <= 13:
                max_sel = 4
            max_half = max(1, field_size // 2 - 1) if field_size > 2 else 1
            max_sel = min(max_sel, max_half)
            if is_big6:
                max_sel = min(max_sel, 3)
            if len(selected[i]) >= max_sel:
                continue
            existing_sc = {r.get("saddlecloth") for r in selected[i]}
            for r in overlay_pools[i]["overlays"]:
                if r.get("saddlecloth") not in existing_sc:
                    edge = float(r.get("edge", 0))
                    if edge <= 0:
                        break  # sorted by edge desc, no more overlays
                    # Estimate EV gain from adding this runner
                    test_selected = [list(s) for s in selected]
                    test_selected[i].append(r)
                    new_ev = _calc_ev_proxy(test_selected, legs_data)
                    new_combos = _total_combos() // max(1, len(selected[i])) * (len(selected[i]) + 1)
                    combo_cost_delta = (new_combos - combos) * (MIN_FLEXI_PCT / 100.0)
                    ev_per_cost = (new_ev - current_ev) / max(combo_cost_delta, 0.01)
                    if ev_per_cost > best_ev_gain:
                        best_ev_gain = ev_per_cost
                        best_add = r
                        best_leg_idx = i
                    break  # pools are sorted, first non-selected is best
        if best_add is None:
            break
        # Check adding wouldn't exceed budget
        new_combos = _total_combos() // max(1, len(selected[best_leg_idx])) * (len(selected[best_leg_idx]) + 1)
        new_cost = new_combos * (MIN_FLEXI_PCT / 100.0)
        if new_cost > budget_hi:
            break
        selected[best_leg_idx].append(best_add)

    # Step 7: Sanity checks
    # Enforce minimum 2 runners per leg (single legs = 30% SR vs 64%+ wider)
    for i in range(num_legs):
        if len(selected[i]) < 2:
            existing_sc = {r.get("saddlecloth") for r in selected[i]}
            # Add next best runner from top_runners
            for r in legs_data[i].top_runners:
                if r.get("saddlecloth") not in existing_sc:
                    selected[i].append(r)
                    break

    # Max one chaos leg >5 selections
    wide_chaos = [i for i in range(num_legs) if leg_types[i] == "chaos" and len(selected[i]) > 5]
    if len(wide_chaos) > 1:
        # Keep only the one with best top-runner edge, trim others to 5
        wide_chaos.sort(
            key=lambda i: float(selected[i][0].get("edge", 0)) if selected[i] else 0,
            reverse=True,
        )
        for idx in wide_chaos[1:]:
            selected[idx] = selected[idx][:5]

    return selected


# ──────────────────────────────────────────────
# Smart Sequence Builder (single optimised ticket)
# ──────────────────────────────────────────────

def build_smart_sequence(
    sequence_type: str,
    race_range: tuple[int, int],
    leg_analysis: list[dict],
    race_contexts: list[dict],
) -> SmartSequence | None:
    """Build a single optimised sequence bet using edge-driven overlay selection.

    Legs classified as anchor (single) / chaos (wide) / normal.
    Outlay $40-$60, budget-optimised by trimming/adding runners by edge.
    """
    prep = _prepare_legs_data(sequence_type, race_range, leg_analysis, race_contexts)
    if not prep:
        return None
    legs_data, outlay, is_big6 = prep

    # Clamp outlay to $40-60
    outlay = max(MIN_OUTLAY, min(MAX_OUTLAY, outlay))

    # Run optimiser
    optimised = _optimiser_select(legs_data, budget=outlay, is_big6=is_big6)
    if optimised is None:
        logger.info(f"Optimiser returned PASS for {sequence_type}")
        return None

    lane_legs = []
    for i, leg in enumerate(legs_data):
        lane_legs.append(_make_lane_leg(leg, optimised[i]))

    total_combos = _calc_combos(lane_legs)
    flexi_pct = round(outlay / total_combos * 100, 1) if total_combos > 0 else 0
    risk = _risk_note(legs_data)
    est_return, hit_prob = _calc_estimated_return(lane_legs, legs_data)

    num_legs = race_range[1] - race_range[0] + 1
    chaos_count = sum(1 for leg in legs_data if _is_chaos_leg(leg))
    chaos_ratio = round(chaos_count / num_legs, 2)

    return SmartSequence(
        sequence_type=sequence_type,
        race_start=race_range[0],
        race_end=race_range[1],
        legs=lane_legs,
        total_combos=total_combos,
        flexi_pct=flexi_pct,
        total_outlay=outlay,
        constrained=False,
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

        # Per-leg detail with classification and runners
        for leg in smart.legs:
            runners_str = ", ".join(str(r) for r in leg.runners)
            names_str = ", ".join(leg.runner_names[:len(leg.runners)])
            if len(leg.runners) <= 2 and leg.odds_shape in BANKER_SHAPES:
                type_marker = " [ANCHOR]"
            elif len(leg.runners) >= 5:
                type_marker = " [CHAOS]"
            elif leg.odds_shape in BANKER_SHAPES:
                type_marker = " [BANKER]"
            else:
                type_marker = ""
            lines.append(
                f"  R{leg.race_number} [{leg.odds_shape}]{type_marker}: "
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
