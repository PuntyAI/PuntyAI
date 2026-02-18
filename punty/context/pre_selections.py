"""Deterministic pre-selection engine for race picks.

Pre-calculates bet types, stakes, pick order, Punty's Pick, and
recommended exotic for each race based on probability model output.
The LLM receives these as RECOMMENDED defaults and can override
with justification.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Pool constraints
RACE_POOL = 20.0       # $20 per race
MIN_STAKE = 1.0        # minimum bet
STAKE_STEP = 0.50      # round to nearest 50c

# Bet type thresholds (defaults — can be overridden by tuned thresholds)
WIN_MIN_PROB = 0.18          # 18% win prob minimum for Win bet
WIN_MIN_VALUE = 0.90         # allow Win bets at near-fair odds
SAVER_WIN_MIN_PROB = 0.14    # 14% for secondary Win bet
EACH_WAY_MIN_PROB = 0.15     # 15% win prob for Each Way
EACH_WAY_MAX_PROB = 0.40     # above 40% just go Win
EACH_WAY_MIN_ODDS = 4.0      # $4+ for Each Way to make sense
EACH_WAY_MAX_ODDS = 20.0     # $20 cap for Each Way
PLACE_MIN_PROB = 0.35         # 35% place prob for Place bet
PLACE_MIN_VALUE = 0.95        # accept slight undervalue for Place safety

# Roughie thresholds
ROUGHIE_MIN_ODDS = 8.0
ROUGHIE_MIN_VALUE = 1.10

# Punty's Pick exotic threshold
EXOTIC_PUNTYS_PICK_VALUE = 1.5  # exotic needs 1.5x value to be Punty's Pick


def _load_thresholds(overrides: dict | None = None) -> dict:
    """Build thresholds dict from module defaults, with optional overrides."""
    t = {
        "win_min_prob": WIN_MIN_PROB,
        "win_min_value": WIN_MIN_VALUE,
        "saver_win_min_prob": SAVER_WIN_MIN_PROB,
        "each_way_min_prob": EACH_WAY_MIN_PROB,
        "each_way_max_prob": EACH_WAY_MAX_PROB,
        "each_way_min_odds": EACH_WAY_MIN_ODDS,
        "each_way_max_odds": EACH_WAY_MAX_ODDS,
        "place_min_prob": PLACE_MIN_PROB,
        "place_min_value": PLACE_MIN_VALUE,
    }
    if overrides:
        t.update(overrides)
    return t


@dataclass
class RecommendedPick:
    """A single recommended pick for a race."""

    rank: int                   # 1-4 (1=top pick, 4=roughie)
    saddlecloth: int
    horse_name: str
    bet_type: str               # "Win", "Saver Win", "Place", "Each Way"
    stake: float                # dollars from $20 pool
    odds: float                 # win odds
    place_odds: float | None    # place odds if available
    win_prob: float             # 0.0-1.0
    place_prob: float           # 0.0-1.0
    value_rating: float         # win value
    place_value_rating: float   # place value
    expected_return: float      # estimated return on this bet
    is_roughie: bool = False


@dataclass
class RecommendedExotic:
    """Recommended exotic bet for a race."""

    exotic_type: str            # "Trifecta Box", "Exacta", etc.
    runners: list[int]          # saddlecloth numbers
    runner_names: list[str]
    probability: float
    value_ratio: float
    num_combos: int
    format: str                 # "flat" or "boxed"


@dataclass
class PuntysPick:
    """The single best bet recommendation for a race."""

    pick_type: str              # "selection" or "exotic"
    # For selections:
    saddlecloth: int | None = None
    horse_name: str | None = None
    bet_type: str | None = None
    stake: float | None = None
    odds: float | None = None
    expected_value: float = 0.0
    reason: str = ""
    # For exotics:
    exotic_type: str | None = None
    exotic_runners: list[int] = field(default_factory=list)
    exotic_value: float = 0.0
    # Optional secondary bet:
    secondary_saddlecloth: int | None = None
    secondary_horse: str | None = None
    secondary_bet_type: str | None = None
    secondary_stake: float | None = None
    secondary_odds: float | None = None


@dataclass
class RacePreSelections:
    """Complete pre-calculated selections for one race."""

    race_number: int
    picks: list[RecommendedPick]
    exotic: RecommendedExotic | None
    puntys_pick: PuntysPick | None
    total_stake: float          # sum of pick stakes (should be <= $20)
    notes: list[str] = field(default_factory=list)


def calculate_pre_selections(
    race_context: dict[str, Any],
    pool: float = RACE_POOL,
    selection_thresholds: dict | None = None,
    place_context_multipliers: dict[str, float] | None = None,
) -> RacePreSelections:
    """Calculate deterministic pre-selections for a single race.

    Args:
        race_context: Race dict from ContextBuilder with runners and probabilities.
        pool: Total stake pool (default $20).
        selection_thresholds: Optional tuned thresholds from bet_type_tuning.
        place_context_multipliers: Optional dict of place context multiplier values.
            When the average multiplier is strong (>1.15), the PLACE_MIN_PROB
            threshold is lowered to encourage more place bets.

    Returns:
        RacePreSelections with ordered picks, exotic, and Punty's Pick.
    """
    race_number = race_context.get("race_number", 0)
    runners = race_context.get("runners", [])
    probs = race_context.get("probabilities", {})
    exotic_combos = probs.get("exotic_combinations", [])

    thresholds = _load_thresholds(selection_thresholds)

    # Adjust place threshold based on place context multipliers.
    # Only tighten (raise threshold) when context is weak — never loosen.
    # Lowering the place threshold was systematically pushing favourites to Place
    # even when Win would be more profitable (costing ~$64/day in missed profit).
    if place_context_multipliers:
        mults = [v for v in place_context_multipliers.values() if isinstance(v, (int, float))]
        if mults:
            avg_place_strength = sum(mults) / len(mults)
            if avg_place_strength < 0.85:
                thresholds["place_min_prob"] = min(0.45, thresholds["place_min_prob"] + 0.05)

    # Build runner data with probability info
    candidates = _build_candidates(runners)
    if not candidates:
        return RacePreSelections(
            race_number=race_number, picks=[], exotic=None,
            puntys_pick=None, total_stake=0.0,
        )

    # Split-formula ranking: different strategies per pick position.
    # Validated on 314 races (Feb 2026 production DB) + 18,888 Proform 2025 races:
    #   #1-2: Pure probability is best. Value weighting hurts #2 PnL.
    #   #3: prob * clamp(value, 0.85, 1.30) = +$281 vs pure prob.
    #       Reduces win losses significantly while place PnL drops only slightly.
    #   #4 roughie: best value from $8+ pool (unchanged).

    # Sort by pure probability (for #1 and #2 pick selection)
    candidates.sort(key=lambda c: c["win_prob"], reverse=True)

    # Separate roughie candidates (odds >= $8, value >= 1.1)
    roughie_pool = [
        c for c in candidates
        if c["odds"] >= ROUGHIE_MIN_ODDS and c["value_rating"] >= ROUGHIE_MIN_VALUE
    ]

    # Select top 3 picks + roughie
    picks: list[RecommendedPick] = []
    used_saddlecloths: set[int] = set()

    # Pick roughie first (so we can exclude from main pool)
    roughie = None
    if roughie_pool:
        # Best value roughie
        roughie_pool.sort(key=lambda c: c["value_rating"], reverse=True)
        roughie = roughie_pool[0]

    # #1 and #2 picks: pure probability (validated best for Win/Saver Win bets)
    for c in candidates:
        if len(picks) >= 2:
            break
        if c["saddlecloth"] in used_saddlecloths:
            continue
        if roughie and c["saddlecloth"] == roughie["saddlecloth"]:
            continue
        picks.append(_make_pick(c, len(picks) + 1, is_roughie=False, thresholds=thresholds))
        used_saddlecloths.add(c["saddlecloth"])

    # #3 pick: re-rank remaining by prob * clamped value (validated +$281 improvement)
    remaining = [c for c in candidates if c["saddlecloth"] not in used_saddlecloths
                 and not (roughie and c["saddlecloth"] == roughie["saddlecloth"])]
    remaining.sort(
        key=lambda c: c["win_prob"] * max(0.85, min(c["value_rating"], 1.30)),
        reverse=True,
    )

    if remaining:
        picks.append(_make_pick(remaining[0], 3, is_roughie=False, thresholds=thresholds))
        used_saddlecloths.add(remaining[0]["saddlecloth"])

    # Add roughie as pick #4
    if roughie and roughie["saddlecloth"] not in used_saddlecloths:
        picks.append(_make_pick(roughie, 4, is_roughie=True, thresholds=thresholds))
        used_saddlecloths.add(roughie["saddlecloth"])
    elif len(picks) < 4:
        # No qualifying roughie — fill 4th from remaining
        for c in candidates:
            if c["saddlecloth"] not in used_saddlecloths:
                picks.append(_make_pick(c, len(picks) + 1, is_roughie=False, thresholds=thresholds))
                used_saddlecloths.add(c["saddlecloth"])
                break

    # Ensure at least one Win/Each Way/Saver Win bet (mandatory rule)
    _ensure_win_bet(picks)

    # Allocate stakes from pool
    _allocate_stakes(picks, pool)

    # Select recommended exotic
    exotic = _select_exotic(exotic_combos, used_saddlecloths)

    # Calculate Punty's Pick
    puntys_pick = _calculate_puntys_pick(picks, exotic)

    total_stake = sum(p.stake for p in picks)
    notes = _generate_notes(picks, exotic, candidates)

    return RacePreSelections(
        race_number=race_number,
        picks=picks,
        exotic=exotic,
        puntys_pick=puntys_pick,
        total_stake=round(total_stake, 2),
        notes=notes,
    )


def _build_candidates(runners: list[dict]) -> list[dict]:
    """Build candidate list from runner context data."""
    candidates = []
    for r in runners:
        if r.get("scratched"):
            continue
        sc = r.get("saddlecloth")
        if not sc:
            continue

        odds = r.get("current_odds") or 0
        place_odds = r.get("place_odds")
        if not odds or odds <= 1.0:
            continue

        win_prob = r.get("_win_prob_raw", 0)
        place_prob = r.get("_place_prob_raw", 0)
        value_rating = r.get("punty_value_rating", 1.0)
        place_value = r.get("punty_place_value_rating", 1.0)
        rec_stake = r.get("punty_recommended_stake", 0)

        # Expected value = prob * odds - 1 (per $1 bet)
        ev = win_prob * odds - 1 if odds > 0 else -1

        candidates.append({
            "saddlecloth": sc,
            "horse_name": r.get("horse_name", ""),
            "odds": odds,
            "place_odds": place_odds,
            "win_prob": win_prob,
            "place_prob": place_prob,
            "value_rating": value_rating,
            "place_value_rating": place_value,
            "rec_stake": rec_stake,
            "ev": ev,
        })

    return candidates


def _determine_bet_type(c: dict, rank: int, is_roughie: bool, thresholds: dict | None = None) -> str:
    """Determine optimal bet type for a runner based on probability profile.

    Edge-aware logic validated on historical performance data:
    - Win sweet spot $4-$6: +60.8% ROI — strongly prefer Win
    - Short-priced favs <$2 on Win: -38.9% ROI — prefer Place
    - $2-$4 Win: moderate, use for top picks only
    - Roughie $10-$20: +53% ROI sweet spot
    - Roughie $50+: -100% ROI — avoid Win entirely
    """
    t = thresholds or _load_thresholds()
    win_prob = c["win_prob"]
    place_prob = c["place_prob"]
    odds = c["odds"]
    value = c["value_rating"]
    place_value = c["place_value_rating"]

    # --- Edge-aware odds-band rules (apply to all ranks) ---

    # Short-priced favourites (<$2): Place is almost always better (-38.9% Win ROI)
    if odds < 2.0 and not is_roughie:
        if place_prob >= 0.50:
            return "Place"
        # Only Win if extraordinary value (shouldn't happen at short prices)
        if win_prob >= 0.45 and value >= 1.15:
            return "Win"
        return "Place"

    # Win sweet spot ($4-$6): +60.8% ROI — prefer Win for top pick
    if 4.0 <= odds <= 6.0 and win_prob >= t["win_min_prob"] and value >= 0.95:
        if rank == 1:
            return "Win"
        # Rank 2: Each Way for upside + place protection (most #2 picks place, not win)
        if rank == 2:
            return "Each Way"
        # Rank 3 gets Win only with strong value
        if value >= 1.05:
            return "Saver Win"

    # --- Roughie logic ---
    if is_roughie:
        # $10-$20 sweet spot: +53% ROI — upgrade to Win if value is there
        if 10.0 <= odds <= 20.0 and win_prob >= t["win_min_prob"] and value >= 1.15:
            return "Win"
        if win_prob >= t["win_min_prob"] and value >= 1.25:
            return "Win"  # strong roughie value at any odds
        return "Place"

    # --- Standard rank-based logic (for $2-$4 and $6+ non-roughie) ---

    # Top pick (#1): prefer Win or Each Way
    if rank == 1:
        if win_prob >= t["win_min_prob"] and value >= t["win_min_value"]:
            # Each Way if in the sweet spot
            if (t["each_way_min_odds"] <= odds <= t["each_way_max_odds"]
                    and t["each_way_min_prob"] <= win_prob <= t["each_way_max_prob"]):
                return "Each Way"
            return "Win"
        if (t["each_way_min_odds"] <= odds <= t["each_way_max_odds"]
                and win_prob >= t["each_way_min_prob"]):
            return "Each Way"
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Win"  # fallback — at least try

    # Second pick (#2): Place-focused — most #2 picks place, not win
    if rank == 2:
        # Only upgrade to Saver Win with very strong win signal
        if win_prob >= 0.25 and value >= 1.10:
            return "Saver Win"
        if (t["each_way_min_odds"] <= odds <= t["each_way_max_odds"]
                and win_prob >= t["each_way_min_prob"] and value >= 1.0):
            return "Each Way"
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Place"

    # Third pick (#3): Place-focused, but Win if in sweet spot (handled above)
    if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
        return "Place"
    if win_prob >= t["win_min_prob"] and value >= 1.10:
        return "Saver Win"
    return "Place"


def _make_pick(c: dict, rank: int, is_roughie: bool, thresholds: dict | None = None) -> RecommendedPick:
    """Create a RecommendedPick from candidate data."""
    bet_type = _determine_bet_type(c, rank, is_roughie, thresholds)
    expected_return = _expected_return(c, bet_type)

    return RecommendedPick(
        rank=rank,
        saddlecloth=c["saddlecloth"],
        horse_name=c["horse_name"],
        bet_type=bet_type,
        stake=0.0,  # allocated later
        odds=c["odds"],
        place_odds=c.get("place_odds"),
        win_prob=c["win_prob"],
        place_prob=c["place_prob"],
        value_rating=c["value_rating"],
        place_value_rating=c["place_value_rating"],
        expected_return=round(expected_return, 2),
        is_roughie=is_roughie,
    )


def _expected_return(c: dict, bet_type: str) -> float:
    """Calculate expected return per $1 for a given bet type."""
    odds = c["odds"]
    place_odds = c.get("place_odds") or _estimate_place_odds(odds)
    win_prob = c["win_prob"]
    place_prob = c["place_prob"]

    if bet_type in ("Win", "Saver Win"):
        return win_prob * odds - 1
    elif bet_type == "Place":
        return place_prob * place_odds - 1
    elif bet_type == "Each Way":
        # Half win, half place
        win_er = win_prob * odds - 1
        place_er = place_prob * place_odds - 1
        return (win_er + place_er) / 2
    return 0.0


def _estimate_place_odds(win_odds: float) -> float:
    """Estimate place odds when not provided (approx 1/3 of win profit + 1)."""
    if win_odds <= 1.0:
        return 1.0
    return round((win_odds - 1) / 3 + 1, 2)


def _ensure_win_bet(picks: list[RecommendedPick]) -> None:
    """Ensure at least one pick is Win, Saver Win, or Each Way (mandatory rule)."""
    has_win = any(p.bet_type in ("Win", "Saver Win", "Each Way") for p in picks)
    if has_win or not picks:
        return

    # Upgrade the top pick to Win
    picks[0].bet_type = "Win"
    picks[0].expected_return = round(
        picks[0].win_prob * picks[0].odds - 1, 2,
    )


def _allocate_stakes(picks: list[RecommendedPick], pool: float) -> None:
    """Allocate stakes from pool using edge-weighted sizing.

    Instead of static rank weights, scale allocation by expected return
    so more capital flows to high-edge bets. Historical edges:
    - Win $4-$6: +60.8% ROI → deserves larger stake
    - Place (any rank): +13.8% ROI → solid base
    - Roughie $10-$20: +53% ROI → bump up from default 15%
    """
    if not picks:
        return

    # Base allocation weights by rank
    rank_weights = {1: 0.35, 2: 0.28, 3: 0.22, 4: 0.15}

    # Edge-aware multipliers on top of rank weights
    for pick in picks:
        base = rank_weights.get(pick.rank, 0.15)

        # Win sweet spot $4-$6 bonus (our best edge at +60.8% ROI)
        if pick.bet_type in ("Win", "Saver Win") and 4.0 <= pick.odds <= 6.0:
            base *= 1.25  # 25% stake boost in sweet spot

        # Roughie $10-$20 bonus (+53% ROI sweet spot)
        if pick.is_roughie and 10.0 <= pick.odds <= 20.0:
            base *= 1.20  # 20% stake boost

        # Positive expected return bonus — tilt more to profitable bets
        if pick.expected_return > 0.10:
            base *= 1.15  # 15% bonus for strong +EV

        # Short-priced Place penalty — these are safe but low-returning
        if pick.bet_type == "Place" and pick.odds < 2.5:
            base *= 0.85  # reduce stake on low-odds Place

        rank_weights[pick.rank] = base

    total_weight = sum(rank_weights.get(p.rank, 0.15) for p in picks)

    for pick in picks:
        weight = rank_weights.get(pick.rank, 0.15)
        raw_stake = pool * (weight / total_weight)

        # Each Way costs double (half win + half place)
        if pick.bet_type == "Each Way":
            # The displayed stake is per-part, total cost = 2x
            raw_stake = raw_stake / 2  # show per-part amount

        # Round to nearest 50c, minimum $1
        stake = max(MIN_STAKE, round(raw_stake / STAKE_STEP) * STAKE_STEP)
        pick.stake = stake

    # Verify total doesn't exceed pool (account for Each Way doubling)
    total = sum(
        p.stake * 2 if p.bet_type == "Each Way" else p.stake
        for p in picks
    )
    if total > pool + 0.01:
        # Scale down proportionally
        scale = pool / total
        for p in picks:
            p.stake = max(MIN_STAKE, round(p.stake * scale / STAKE_STEP) * STAKE_STEP)


def _select_exotic(
    exotic_combos: list[dict],
    selection_saddlecloths: set[int],
) -> RecommendedExotic | None:
    """Select the best exotic bet, enforcing consistency with selections.

    Overlap rules: ALL exotic runners MUST be from our selections.

    Type preferences (validated on 65 settled exotics):
    - Exacta / Exacta Standout: 22.2% hit rate, +0.2% ROI — BEST
    - Trifecta Standout: ~16% hit rate when #1 wins
    - Trifecta Box: 4.5% hit rate, -19.2% ROI — avoid
    - Quinella: 0/3 hits — avoid
    - First4 Box: 0% hit rate — banned

    Strategy: Anchor on #1 pick (39% win rate). Exacta Standout
    leverages this directly with fewer combos and higher hit rate.
    """
    if not exotic_combos:
        return None

    # Score exotics: heavily favour formats that anchor on #1 pick
    scored = []
    for ec in exotic_combos:
        runners = set(ec.get("runners", []))
        n_runners = len(runners)
        overlap = len(runners & selection_saddlecloths)
        overlap_ratio = overlap / n_runners if n_runners else 0

        # ALL exotic runners must be from selections
        if overlap_ratio < 1.0:
            continue

        exotic_type = ec.get("type", "").lower()

        # Ban underperforming types
        if "first4" in exotic_type and "box" in exotic_type:
            continue  # 0% hit rate historically
        if "quinella" in exotic_type:
            continue  # 0/3, no edge over exacta

        # Blend value with probability
        raw_prob = ec.get("probability", 0)
        if isinstance(raw_prob, str):
            raw_prob = float(raw_prob.rstrip("%")) / 100
        prob_weight = min(raw_prob / 0.05, 1.0)

        # Type bonuses based on validated performance data
        type_bonus = 0.0
        if "exacta" in exotic_type:
            type_bonus = 1.5  # 22.2% hit rate, all exotic wins were exactas
            if "standout" in exotic_type:
                type_bonus = 2.0  # standout format is ideal (anchored on #1)
        elif "trifecta" in exotic_type and "standout" in exotic_type:
            type_bonus = 0.8  # 15.8% hit rate when #1 wins
        elif "trifecta" in exotic_type and "box" in exotic_type:
            type_bonus = -0.5  # 4.5% hit rate, -19.2% ROI — penalise

        # Score = value * probability_weight + overlap bonus + type bonus
        score = ec.get("value", 1.0) * prob_weight + overlap_ratio * 1.0 + type_bonus

        scored.append((score, ec))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return None

    best = scored[0][1]
    return RecommendedExotic(
        exotic_type=best.get("type", ""),
        runners=best.get("runners", []),
        runner_names=best.get("runner_names", []),
        probability=float(best.get("probability", "0").rstrip("%")) / 100
        if isinstance(best.get("probability"), str)
        else best.get("probability", 0),
        value_ratio=best.get("value", 1.0),
        num_combos=best.get("combos", 1),
        format=best.get("format", "boxed"),
    )


def _calculate_puntys_pick(
    picks: list[RecommendedPick],
    exotic: RecommendedExotic | None,
) -> PuntysPick | None:
    """Determine the single best bet for Punty's Pick.

    Compares the best selection EV against the exotic value ratio.
    Exotic wins only if value >= 1.5x (higher bar than regular exotics).
    """
    if not picks:
        return None

    # Best selection by expected return
    best_sel = max(picks, key=lambda p: p.expected_return)

    # Check if exotic beats selections
    if exotic and exotic.value_ratio >= EXOTIC_PUNTYS_PICK_VALUE:
        # Compare exotic EV to selection EV
        # Exotic EV proxy: (value_ratio - 1) * probability * 15
        exotic_edge = exotic.value_ratio - 1.0
        sel_edge = best_sel.expected_return

        if exotic_edge > sel_edge * 1.2:  # exotic must be clearly better
            return PuntysPick(
                pick_type="exotic",
                exotic_type=exotic.exotic_type,
                exotic_runners=exotic.runners,
                exotic_value=exotic.value_ratio,
                expected_value=exotic_edge,
                reason=(
                    f"{exotic.exotic_type} at {exotic.value_ratio:.1f}x value "
                    f"— the model loves this combination."
                ),
            )

    # Selection-based Punty's Pick
    pp = PuntysPick(
        pick_type="selection",
        saddlecloth=best_sel.saddlecloth,
        horse_name=best_sel.horse_name,
        bet_type=best_sel.bet_type,
        stake=best_sel.stake,
        odds=best_sel.place_odds if best_sel.bet_type == "Place" and best_sel.place_odds else best_sel.odds,
        expected_value=best_sel.expected_return,
        reason=_pick_reason(best_sel),
    )

    # Add secondary bet if there's a clear #2
    if len(picks) >= 2:
        second = sorted(
            [p for p in picks if p.saddlecloth != best_sel.saddlecloth],
            key=lambda p: p.expected_return,
            reverse=True,
        )
        if second and second[0].expected_return > 0:
            s = second[0]
            pp.secondary_saddlecloth = s.saddlecloth
            pp.secondary_horse = s.horse_name
            pp.secondary_bet_type = s.bet_type
            pp.secondary_stake = s.stake
            pp.secondary_odds = s.place_odds if s.bet_type == "Place" and s.place_odds else s.odds

    return pp


def _pick_reason(pick: RecommendedPick) -> str:
    """Generate a short reason string for Punty's Pick."""
    prob_pct = pick.win_prob * 100
    if pick.bet_type in ("Win", "Saver Win"):
        return (
            f"{prob_pct:.0f}% chance at ${pick.odds:.2f} "
            f"({pick.value_rating:.2f}x value)"
        )
    elif pick.bet_type == "Each Way":
        place_pct = pick.place_prob * 100
        return (
            f"{prob_pct:.0f}% win / {place_pct:.0f}% place at ${pick.odds:.2f} "
            f"— Each Way covers both angles"
        )
    else:  # Place
        place_pct = pick.place_prob * 100
        return (
            f"{place_pct:.0f}% place chance at "
            f"${pick.place_odds or _estimate_place_odds(pick.odds):.2f} "
            f"({pick.place_value_rating:.2f}x place value)"
        )


def _generate_notes(
    picks: list[RecommendedPick],
    exotic: RecommendedExotic | None,
    candidates: list[dict],
) -> list[str]:
    """Generate advisory notes about the selections."""
    notes = []

    # Flag if no clear value in this race
    if picks and all(p.value_rating < 1.05 for p in picks):
        notes.append("No clear value in this race — consider reducing exposure.")

    # Flag strong value
    strong = [p for p in picks if p.value_rating >= 1.20]
    if strong:
        names = ", ".join(p.horse_name for p in strong)
        notes.append(f"Strong value detected: {names}")

    # Flag wide-open race
    if candidates and candidates[0]["win_prob"] < 0.15:
        notes.append("Wide-open race — no runner above 15% probability.")

    return notes


def format_pre_selections(pre_sel: RacePreSelections) -> str:
    """Format pre-selections for injection into AI prompt context."""
    lines = [f"\n**RECOMMENDED SELECTIONS (Race {pre_sel.race_number}):**"]

    for pick in pre_sel.picks:
        rank_label = "Roughie" if pick.is_roughie else f"Pick #{pick.rank}"
        prob_label = (
            f"{pick.win_prob * 100:.1f}%"
            if pick.bet_type in ("Win", "Saver Win")
            else f"{pick.place_prob * 100:.1f}%"
        )
        value_label = (
            f"{pick.value_rating:.2f}x"
            if pick.bet_type in ("Win", "Saver Win")
            else f"{pick.place_value_rating:.2f}x"
        )

        # Calculate display return
        if pick.bet_type in ("Win", "Saver Win"):
            ret = round(pick.stake * pick.odds, 2)
        elif pick.bet_type == "Place":
            po = pick.place_odds or _estimate_place_odds(pick.odds)
            ret = round(pick.stake * po, 2)
        elif pick.bet_type == "Each Way":
            po = pick.place_odds or _estimate_place_odds(pick.odds)
            ret = round(pick.stake * pick.odds + pick.stake * po, 2)
        else:
            ret = 0

        lines.append(
            f"  {rank_label}: {pick.horse_name} (No.{pick.saddlecloth}) "
            f"— ${pick.odds:.2f} | {pick.bet_type} ${pick.stake:.2f} "
            f"| Prob: {prob_label} | Value: {value_label} "
            f"| Return: ${ret:.2f}"
        )

    if pre_sel.exotic:
        ex = pre_sel.exotic
        runners_str = ", ".join(str(r) for r in ex.runners)
        names_str = ", ".join(ex.runner_names)
        lines.append(
            f"  Exotic: {ex.exotic_type} [{runners_str}] {names_str} "
            f"— $20 | Prob: {ex.probability * 100:.1f}% "
            f"| Value: {ex.value_ratio:.2f}x | {ex.num_combos} combos"
        )

    if pre_sel.puntys_pick:
        pp = pre_sel.puntys_pick
        if pp.pick_type == "exotic":
            runners_str = ", ".join(str(r) for r in pp.exotic_runners)
            lines.append(
                f"  Punty's Pick: {pp.exotic_type} [{runners_str}] "
                f"— $20 (Value: {pp.exotic_value:.1f}x) | {pp.reason}"
            )
        else:
            line = (
                f"  Punty's Pick: {pp.horse_name} (No.{pp.saddlecloth}) "
                f"${pp.odds:.2f} {pp.bet_type}"
            )
            if pp.secondary_horse:
                line += (
                    f" + {pp.secondary_horse} (No.{pp.secondary_saddlecloth}) "
                    f"${pp.secondary_odds:.2f} {pp.secondary_bet_type}"
                )
            lines.append(f"  {line} | {pp.reason}")

    if pre_sel.notes:
        for note in pre_sel.notes:
            lines.append(f"  NOTE: {note}")

    lines.append(f"  Total stake: ${pre_sel.total_stake:.2f} / $20.00")

    return "\n".join(lines)
