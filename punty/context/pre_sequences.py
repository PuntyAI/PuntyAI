"""Single optimised sequence construction for Quaddie / Early Quaddie / Big 6.

Builds ONE ticket per sequence type using edge-driven overlay-only selection.
Legs classified as anchor (single) / chaos (wide) / normal based on
probability distribution and favourite odds.

Outlay range $40-$60, budget-optimised to trim/add runners by edge.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BASE_OUTLAY = 30.0
MIN_OUTLAY = 20.0
MAX_OUTLAY = 80.0  # Shape-driven legs need wider budget (was $60, $40 before that)
BIG6_MIN_OUTLAY = 25.0
BIG6_MAX_OUTLAY = 25.0
BIG6_OUTLAY = 5.0   # Hail-mary: 1 pick per leg, fixed $5 ticket
MIN_FLEXI_PCT = 20.0  # Lowered from 30% — allows 400 combos at $80 budget (5×4×5×4)

    # Hard-coded gates removed — sequence meta-model handles play/skip (PR #238)

# LGBM-driven floor: exclude runners with win_prob < 5% from legs
MIN_LEG_RUNNER_WP = 0.05

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

        # Build set of actual tip saddlecloths from pre_selections
        pre_sel = rc.get("pre_selections")
        tip_saddlecloths = {}  # saddlecloth → rank
        if pre_sel and hasattr(pre_sel, "picks"):
            for pick in pre_sel.picks:
                tip_saddlecloths[pick.saddlecloth] = pick.rank

        # Tag actual tip selections vs probability-pool runners
        for r in top_runners:
            sc = r.get("saddlecloth", 0)
            if sc in tip_saddlecloths:
                r["_is_pick"] = True
                r["_pick_rank"] = tip_saddlecloths[sc]
            else:
                r["_is_pick"] = False
                r["_pick_rank"] = 99

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
                        "_is_pick": False,
                        "_pick_rank": 99,
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

    # Dynamic outlay scaled by chaos ratio
    chaos_count = sum(1 for leg in legs_data if _is_chaos_leg(leg))
    chaos_ratio = chaos_count / num_legs
    if is_big6:
        outlay = round(BIG6_MIN_OUTLAY + chaos_ratio * (BIG6_MAX_OUTLAY - BIG6_MIN_OUTLAY), 0)
        outlay = max(BIG6_MIN_OUTLAY, min(BIG6_MAX_OUTLAY, outlay))
    else:
        outlay = round(MIN_OUTLAY + chaos_ratio * (MAX_OUTLAY - MIN_OUTLAY), 0)
        outlay = max(MIN_OUTLAY, min(MAX_OUTLAY, outlay))

    return (legs_data, outlay, is_big6)


def _select_runners_for_leg(leg, width: int, value_swap: bool = True) -> list[dict]:
    """Select runners for a leg with optional value swaps.

    Runners in leg.top_runners are already sorted by composite score
    (win_prob * value blend) from the probability engine.

    Width is raised to include all our ranked picks (tip_rank 1-4) that pass
    the win_prob floor, so we never exclude our own selections.
    """
    candidates = list(leg.top_runners)

    # Ensure width is at least large enough to include all our picks above floor
    our_picks = [r for r in candidates
                 if r.get("_is_pick") and float(r.get("win_prob", 0)) >= MIN_LEG_RUNNER_WP]
    width = max(width, len(our_picks))

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

    # Bonus for legs that include all of our picks: wider legs with all picks
    # have lower raw payout but significantly higher hit rate. Apply 1.1x per leg
    # where all 4 picks are present to reflect the improved sequence hit probability.
    for i, leg in enumerate(lane_legs):
        selected_sc = set(leg.runners)
        pick_runners = [r for r in legs_data[i].top_runners if r.get("_is_pick")]
        if len(pick_runners) >= 4:
            picks_in_leg = sum(1 for r in pick_runners if r.get("saddlecloth", 0) in selected_sc)
            if picks_in_leg >= 4:
                est_return = round(est_return * 1.1, 1)

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

    # Step 2: Classify anchor presence (informational — no longer a hard skip)
    has_anchor = "anchor" in leg_types
    has_strong = any(
        float(r.get("win_prob", 0)) >= 0.32
        for leg in legs_data
        for r in leg.top_runners[:1]  # top runner only
    )
    if not has_anchor and not has_strong:
        logger.info("No anchor leg and no strong runner (>=32%%) — building wider legs")

    # Step 3: Set initial targets per leg type
    # Check which legs have a short-priced fav (≤$2.50) that is one of our picks
    has_short_fav_pick = []
    for leg in legs_data:
        found = False
        for r in leg.top_runners:
            odds = r.get("current_odds")
            if odds and float(odds) <= 2.50 and r.get("_is_pick"):
                found = True
                break
        has_short_fav_pick.append(found)

    targets = []
    for i, lt in enumerate(leg_types):
        field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
        pool_depth = len(overlay_pools[i]["overlays"]) + len(overlay_pools[i]["neutrals"])
        # Use odds-shape-driven width as the target — diagnosis showed 45% of
        # missed legs had winners within shape_width but outside our leg width.
        # Shape width comes from 14,246-leg backtest marginal capture >=7%.
        shape_w = legs_data[i].shape_width if hasattr(legs_data[i], "shape_width") else 3
        if lt == "anchor":
            # Anchor: trust the favourite, allow 2-wide with short fav
            if has_short_fav_pick[i]:
                targets.append(2)
            else:
                targets.append(min(3, shape_w))
        elif lt == "chaos":
            targets.append(min(shape_w, pool_depth))
        else:
            # Normal: use shape width, capped by pool depth
            targets.append(min(shape_w, pool_depth))

    # Step 4: Apply field-size caps and minimums
    # Count our picks per leg (above 5% win_prob floor) to ensure we never
    # exclude our own ranked selections (rank 1-4). Data shows 59% of
    # "had the winner but didn't include them" misses are rank 3/4.
    picks_per_leg = []
    for i in range(num_legs):
        runners = legs_data[i].top_runners
        pick_count = sum(1 for r in runners
                         if r.get("_is_pick") and float(r.get("win_prob", 0)) >= MIN_LEG_RUNNER_WP)
        picks_per_leg.append(pick_count)

    for i in range(num_legs):
        field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
        shape_w = legs_data[i].shape_width if hasattr(legs_data[i], "shape_width") else 3
        # Cap at shape width or field-size-based maximum, whichever is lower
        if field_size <= 5:
            targets[i] = min(targets[i], 3)
        elif field_size <= 7:
            targets[i] = min(targets[i], 4)
        elif field_size <= 10:
            targets[i] = min(targets[i], min(shape_w, 5))
        elif field_size <= 13:
            targets[i] = min(targets[i], min(shape_w, 6))
        else:
            targets[i] = min(targets[i], min(shape_w, 7))
        # Never > 50% of field
        max_half = max(1, field_size // 2) if field_size > 2 else 1
        targets[i] = min(targets[i], max_half)
        # Field-size-driven minimums — ensure we always include all our picks
        if field_size >= 14:
            min_width = 4
        elif is_big6:
            # Big6 needs wider legs to compensate for 6 legs
            if leg_types[i] == "chaos":
                min_width = min(5, max_half)  # Chaos legs in Big6: min 5
            else:
                min_width = min(4, max_half)  # Normal/anchor Big6: min 4
        else:
            min_width = 3  # Quaddie: always 3-wide minimum
        # Never go below the number of our picks in this race
        min_width = max(min_width, min(picks_per_leg[i], max_half))
        targets[i] = max(targets[i], min_width)

    # Big6 wider caps — need wider legs to compensate for 6 legs
    # (was 2-wide max, now 4-5 to improve hit rate)
    if is_big6:
        for i in range(num_legs):
            field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
            max_half = max(1, field_size // 2) if field_size > 2 else 1
            if leg_types[i] == "chaos":
                targets[i] = min(targets[i], min(5, max_half))
            else:
                targets[i] = min(targets[i], min(4, max_half))

    # Step 5: Populate legs
    # Anchor legs: top win_prob runners (favorites = short-priced, predictable)
    # Normal/chaos legs: tip-first selection — picks form the backbone,
    #   extended pool runners only supplement for width when picks can't fill.
    # LGBM-driven floor: exclude runners with win_prob < 5% from legs.
    # Production data: 2-wide legs hit 45.3% but 3-wide hit 54.5% — wider is
    # better, but only if the extra runners have genuine win chances.
    selected = []
    for i in range(num_legs):
        runners = legs_data[i].top_runners
        target = targets[i]

        # Filter to runners above LGBM probability floor
        viable = [r for r in runners if float(r.get("win_prob", 0)) >= MIN_LEG_RUNNER_WP]
        # Ensure we always have at least min runners from the full pool
        if len(viable) < 2:
            viable = runners

        if leg_types[i] == "anchor":
            # Anchor = most probable runners, sorted by win_prob descending
            by_prob = sorted(
                viable,
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )
            sel = by_prob[:target]
        else:
            # Normal/chaos: picks-first selection
            # Quaddies need WINNERS — our picks sorted by win_prob always come
            # first, regardless of edge. Edge only matters for non-pick width.
            # 1) All our picks, sorted by win_prob descending (best picks first)
            all_picks = sorted(
                [r for r in viable if r.get("_is_pick")],
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )
            # 2) Extended pool: top win_prob non-picks (need coverage, not just edge)
            extended_pool = sorted(
                [r for r in viable if not r.get("_is_pick")],
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )

            sel = []
            existing_sc = set()

            def _add(runner_list, limit):
                for r in runner_list:
                    if len(sel) >= limit:
                        break
                    sc = r.get("saddlecloth")
                    if sc not in existing_sc:
                        sel.append(r)
                        existing_sc.add(sc)

            # Fill with our picks first (sorted by win_prob — strongest picks anchor)
            _add(all_picks, target)
            # Then extend with highest win_prob non-picks for width coverage
            _add(extended_pool, target)

        # If still under min 2, pad with best win_prob pick (last resort)
        if len(sel) < 2:
            existing_sc = {r.get("saddlecloth") for r in sel}
            by_prob = sorted(
                [r for r in runners if r.get("_is_pick")],
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )
            for r in by_prob:
                if len(sel) >= 2:
                    break
                if r.get("saddlecloth") not in existing_sc:
                    sel.append(r)
                    existing_sc.add(r.get("saddlecloth"))
            # Absolute last resort: any runner by probability
            if len(sel) < 2:
                by_prob_all = sorted(
                    runners,
                    key=lambda r: float(r.get("win_prob", 0)),
                    reverse=True,
                )
                for r in by_prob_all:
                    if len(sel) >= 2:
                        break
                    if r.get("saddlecloth") not in existing_sc:
                        sel.append(r)
                        existing_sc.add(r.get("saddlecloth"))
        selected.append(sel)

    # Step 5b: Mandatory favourite inclusion — ≤$2.50 must be in leg
    for i in range(num_legs):
        runners = legs_data[i].top_runners
        existing_sc = {r.get("saddlecloth") for r in selected[i]}
        # Find short-priced favourite (lowest odds ≤ $2.50)
        short_fav = None
        for r in runners:
            odds = r.get("current_odds")
            if odds and float(odds) <= 2.50 and r.get("saddlecloth") not in existing_sc:
                if short_fav is None or float(odds) < float(short_fav.get("current_odds", 999)):
                    short_fav = r
        if short_fav:
            # Force-add: replace weakest runner if at capacity
            if len(selected[i]) >= targets[i]:
                # Prefer replacing non-pick runners first
                worst_idx = None
                worst_score = 999
                for j, r in enumerate(selected[i]):
                    if not r.get("_is_pick"):
                        score = float(r.get("edge", 0))
                        if score < worst_score:
                            worst_score = score
                            worst_idx = j
                # If all runners are picks, replace the weakest pick by win_prob.
                # Short-priced favs (≤$2.50) should ALWAYS be in our legs —
                # the market is pricing them at 40-75% chance. Trust market here.
                if worst_idx is None:
                    worst_wp = 999
                    for j, r in enumerate(selected[i]):
                        wp = float(r.get("win_prob", 0))
                        if wp < worst_wp:
                            worst_wp = wp
                            worst_idx = j
                if worst_idx is not None:
                    replaced = selected[i][worst_idx]
                    selected[i][worst_idx] = short_fav
                    logger.info(
                        f"Leg {i+1} R{legs_data[i].race_number}: forced fav "
                        f"{short_fav.get('horse_name')} (${short_fav.get('current_odds')}) "
                        f"replacing {replaced.get('horse_name')}"
                    )
            else:
                selected[i].append(short_fav)
                logger.info(
                    f"Leg {i+1} R{legs_data[i].race_number}: added mandatory fav "
                    f"{short_fav.get('horse_name')} (${short_fav.get('current_odds')})"
                )

    # Debug: log what was selected per leg
    for i in range(num_legs):
        rn = legs_data[i].race_number
        picks_in = [r for r in legs_data[i].top_runners if r.get("_is_pick")]
        sel_names = [(r.get("saddlecloth"), r.get("horse_name", "?"), r.get("_is_pick")) for r in selected[i]]
        logger.info(
            f"Leg {i+1} R{rn} ({leg_types[i]}, target={targets[i]}): "
            f"tips={[p.get('saddlecloth') for p in picks_in]} "
            f"selected={sel_names}"
        )

    # Step 6: Budget optimisation loop
    def _total_combos():
        c = 1
        for s in selected:
            c *= max(1, len(s))
        return c

    budget_hi = budget
    budget_lo = budget * 0.85
    budget_floor = budget * 0.70

    # Trim phase: remove runner causing smallest EV_proxy drop, respecting min width.
    # CRITICAL: Never trim a runner that is one of our pre-selected picks (tip_rank 1-4).
    # Only trim non-pick extension runners. This preserves our core selections while
    # allowing the optimiser to control width of extension runners.
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
            field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
            max_half_i = max(1, field_size // 2) if field_size > 2 else 1
            if field_size >= 14:
                leg_min = 4
            elif is_big6:
                if leg_types[i] == "chaos":
                    leg_min = min(5, max_half_i)
                else:
                    leg_min = min(4, max_half_i)
            else:
                leg_min = 3
            # Also enforce picks minimum
            leg_min = max(leg_min, min(picks_per_leg[i], max_half_i))
            if len(selected[i]) <= leg_min:
                continue
            for j in range(len(selected[i])):
                # Never trim our own picks — only trim non-pick extension runners
                if selected[i][j].get("_is_pick"):
                    continue
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

    # Safety net: after trim, force-add any missing picks back into their legs.
    # The trim phase only removes non-picks, but the initial selection may not have
    # included all picks if width was tight. This ensures all rank 1-4 picks are present.
    for i in range(num_legs):
        existing_sc = {r.get("saddlecloth") for r in selected[i]}
        field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
        max_half_i = max(1, field_size // 2) if field_size > 2 else 1
        for r in legs_data[i].top_runners:
            if (r.get("_is_pick")
                    and r.get("saddlecloth") not in existing_sc
                    and float(r.get("win_prob", 0)) >= MIN_LEG_RUNNER_WP
                    and len(selected[i]) < max_half_i):
                selected[i].append(r)
                existing_sc.add(r.get("saddlecloth"))
                logger.info(
                    f"Safety net: force-added pick {r.get('horse_name')} (rank {r.get('_pick_rank')}) "
                    f"back into leg {i+1} R{legs_data[i].race_number}"
                )

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
            shape_w = legs_data[i].shape_width if hasattr(legs_data[i], "shape_width") else 3
            # Match Step 4 field-size caps — but respect shape_width
            if field_size <= 5:
                max_sel = 3
            elif field_size <= 7:
                max_sel = min(shape_w, 4)
            elif field_size <= 10:
                max_sel = min(shape_w, 5)
            elif field_size <= 13:
                max_sel = min(shape_w, 6)
            else:
                max_sel = min(shape_w, 7)
            max_half = max(1, field_size // 2) if field_size > 2 else 1
            max_sel = min(max_sel, max_half)
            if is_big6:
                if leg_types[i] == "chaos":
                    max_sel = min(max_sel, 5)
                else:
                    max_sel = min(max_sel, 4)
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

    # Step 7: Sanity checks — enforce per-leg minimum width
    for i in range(num_legs):
        field_size = getattr(legs_data[i], "_field_size", len(legs_data[i].top_runners))
        max_half_i = max(1, field_size // 2) if field_size > 2 else 1
        if field_size >= 14:
            leg_min = 4
        elif is_big6:
            if leg_types[i] == "chaos":
                leg_min = min(5, max_half_i)
            else:
                leg_min = min(4, max_half_i)
        else:
            leg_min = 3
        # Also enforce picks minimum
        leg_min = max(leg_min, min(picks_per_leg[i], max_half_i))
        while len(selected[i]) < leg_min:
            existing_sc = {r.get("saddlecloth") for r in selected[i]}
            added = False
            for r in legs_data[i].top_runners:
                if r.get("saddlecloth") not in existing_sc:
                    selected[i].append(r)
                    added = True
                    break
            if not added:
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
    track_condition: str = "",
) -> SmartSequence | None:
    """Build a single optimised sequence bet using edge-driven overlay selection.

    Legs classified as anchor (single) / chaos (wide) / normal.
    Outlay $40-$60, budget-optimised by trimming/adding runners by edge.
    """
    # Big6 hail-mary: 1 pick per leg, fixed $5 ticket (1x1x1x1x1x1 = 1 combo, 500% flexi)
    # No optimiser, no trim/add loop. Just our rank-1 pick per leg.
    if "big" in sequence_type.lower() and "6" in sequence_type:
        prep = _prepare_legs_data(sequence_type, race_range, leg_analysis, race_contexts)
        if not prep:
            return None
        legs_data, _outlay, _is_big6 = prep
        lane_legs = []
        for leg in legs_data:
            # Pick rank-1 runner: highest win_prob in top_runners
            ranked = sorted(
                leg.top_runners,
                key=lambda r: float(r.get("win_prob", 0)),
                reverse=True,
            )
            rank1 = ranked[:1] if ranked else []
            if not rank1:
                logger.info(f"Big6 hail-mary: no runner found for R{leg.race_number}")
                return None
            lane_legs.append(_make_lane_leg(leg, rank1))
        total_combos = 1  # 1x1x1x1x1x1
        outlay = BIG6_OUTLAY
        flexi_pct = round(outlay / total_combos * 100, 1)  # 500%
        est_return, hit_prob = _calc_estimated_return(lane_legs, legs_data)
        num_legs_b6 = race_range[1] - race_range[0] + 1
        chaos_count_b6 = sum(1 for leg in legs_data if _is_chaos_leg(leg))
        chaos_ratio_b6 = round(chaos_count_b6 / num_legs_b6, 2)
        risk = _risk_note(legs_data)
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
            chaos_ratio=chaos_ratio_b6,
        )

    # Track condition and confidence gates removed — sequence meta-model
    # now learns when to play/skip from historical data (PR #238).

    prep = _prepare_legs_data(sequence_type, race_range, leg_analysis, race_contexts)
    if not prep:
        return None
    legs_data, outlay, is_big6 = prep

    # Clamp outlay to appropriate range
    if is_big6:
        outlay = max(BIG6_MIN_OUTLAY, min(BIG6_MAX_OUTLAY, outlay))
    else:
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
    sequence_override: dict | None = None,
    track_condition: str = "",
) -> list[SequenceBlock]:
    """Build single optimised sequence for each applicable sequence type.

    Returns SequenceBlock with smart field containing the optimised ticket.
    If sequence_override is provided, use those lanes instead of standard rules.
    """
    if sequence_override is not None:
        sequences = sequence_override
    elif total_races < 6:
        # TAB doesn't offer quaddie pools at meets with fewer than 6 races
        logger.info(f"Skipping sequences: only {total_races} races (minimum 6)")
        return []
    else:
        rules = {
            6:  {"early_quad": (1, 4), "quaddie": (3, 6), "big6": (1, 6)},
            7:  {"early_quad": (1, 4), "quaddie": (4, 7), "big6": (2, 7)},
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

        smart = build_smart_sequence(label, race_range, leg_analysis, race_contexts,
                                     track_condition=track_condition)
        if not smart:
            continue

        # Meta-model play/skip gate (replaces hard-coded return % floors)
        try:
            from punty.betting.sequence_model import sequence_model_available, should_play_sequence
            if sequence_model_available():
                # Extract per-leg features for the model
                leg_map = {la["race_number"]: la for la in leg_analysis}
                leg_wps = []
                leg_fields = []
                for rn in range(smart.race_start, smart.race_end + 1):
                    la = leg_map.get(rn, {})
                    top_runners = la.get("top_runners", [])
                    top_wp = max((float(r.get("win_prob", 0)) for r in top_runners), default=0)
                    leg_wps.append(top_wp)
                    leg_fields.append(la.get("field_size", 10))

                play, prob, reason = should_play_sequence(
                    sequence_type=label,
                    leg_wps=leg_wps,
                    leg_field_sizes=leg_fields,
                    track_condition=track_condition,
                    total_combos=smart.total_combos,
                    estimated_return_pct=smart.estimated_return_pct,
                    hit_probability=smart.hit_probability,
                )
                if not play:
                    logger.info(f"Sequence model skipped {label}: {reason}")
                    continue
                logger.info(
                    f"Sequence model approved {label} (prob={prob:.1%}): {reason}"
                )
        except ImportError:
            pass  # Module not yet available — play all sequences
        except Exception as e:
            logger.debug(f"Sequence model error, playing {label}: {e}")

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
            if len(leg.runners) <= 3 and leg.odds_shape in BANKER_SHAPES:
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
