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

    # Adjust place threshold based on place context multipliers
    if place_context_multipliers:
        mults = [v for v in place_context_multipliers.values() if isinstance(v, (int, float))]
        if mults:
            avg_place_strength = sum(mults) / len(mults)
            if avg_place_strength > 1.15:
                thresholds["place_min_prob"] = max(0.25, thresholds["place_min_prob"] - 0.05)
            elif avg_place_strength < 0.85:
                thresholds["place_min_prob"] = min(0.45, thresholds["place_min_prob"] + 0.05)

    # Build runner data with probability info
    candidates = _build_candidates(runners)
    if not candidates:
        return RacePreSelections(
            race_number=race_number, picks=[], exotic=None,
            puntys_pick=None, total_stake=0.0,
        )

    # Sort by win probability — pick the most likely winners first.
    # Value is used for bet type decisions and the roughie slot, not pick order.
    candidates.sort(key=lambda c: c["win_prob"], reverse=True)

    # Separate roughie candidates (odds >= $8, value >= 1.1)
    roughie_pool = [
        c for c in candidates
        if c["odds"] >= ROUGHIE_MIN_ODDS and c["value_rating"] >= ROUGHIE_MIN_VALUE
    ]

    # Top 3 from main pool (exclude roughie if selected)
    main_pool = candidates[:]

    # Select top 3 picks + roughie
    picks: list[RecommendedPick] = []
    used_saddlecloths: set[int] = set()

    # Pick roughie first (so we can exclude from main pool)
    roughie = None
    if roughie_pool:
        # Best value roughie
        roughie_pool.sort(key=lambda c: c["value_rating"], reverse=True)
        roughie = roughie_pool[0]

    # Top 3 from remaining candidates
    for c in main_pool:
        if len(picks) >= 3:
            break
        if c["saddlecloth"] in used_saddlecloths:
            continue
        if roughie and c["saddlecloth"] == roughie["saddlecloth"]:
            continue  # reserve for roughie slot
        picks.append(_make_pick(c, len(picks) + 1, is_roughie=False, thresholds=thresholds))
        used_saddlecloths.add(c["saddlecloth"])

    # Add roughie as pick #4
    if roughie and roughie["saddlecloth"] not in used_saddlecloths:
        picks.append(_make_pick(roughie, 4, is_roughie=True, thresholds=thresholds))
        used_saddlecloths.add(roughie["saddlecloth"])
    elif len(picks) < 4:
        # No qualifying roughie — fill 4th from remaining
        for c in main_pool:
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
    """Determine optimal bet type for a runner based on probability profile."""
    t = thresholds or _load_thresholds()
    win_prob = c["win_prob"]
    place_prob = c["place_prob"]
    odds = c["odds"]
    value = c["value_rating"]
    place_value = c["place_value_rating"]

    # Roughie: usually Place (safer) unless strong value
    if is_roughie:
        if win_prob >= t["win_min_prob"] and value >= 1.20:
            return "Win"  # strong roughie value
        if place_prob >= t["place_min_prob"]:
            return "Place"
        return "Place"  # default roughie to Place

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

    # Second pick (#2): Win if strong, Saver Win, or Place
    if rank == 2:
        if win_prob >= t["win_min_prob"] and value >= t["win_min_value"]:
            return "Saver Win"
        if (t["each_way_min_odds"] <= odds <= t["each_way_max_odds"]
                and win_prob >= t["each_way_min_prob"] and value >= 1.0):
            return "Each Way"
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Place"

    # Third pick (#3): Place-focused
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
    """Allocate stakes from pool based on Kelly recommendations and rank."""
    if not picks:
        return

    # Base allocation weights by rank
    rank_weights = {1: 0.35, 2: 0.28, 3: 0.22, 4: 0.15}
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

    Overlap rules by exotic size:
    - 2 runners (Quinella/Exacta): both must be in selections (100%)
    - 3 runners (Trifecta): at least 2 of 3 in selections (67%)
    - 4+ runners (First4/Trifecta Box 4): at least 3 of 4 in selections (75%)

    Falls back to Trifecta Box from selections if no combo meets threshold.
    """
    if not exotic_combos:
        return None

    # Score exotics: value ratio + strong bonus for using our selected runners
    scored = []
    for ec in exotic_combos:
        runners = set(ec.get("runners", []))
        n_runners = len(runners)
        overlap = len(runners & selection_saddlecloths)
        overlap_ratio = overlap / n_runners if n_runners else 0

        # Size-based minimum overlap
        if n_runners <= 2:
            min_overlap = 1.0    # Quinella/Exacta: all runners must be our picks
        elif n_runners == 3:
            min_overlap = 0.66   # Trifecta 3: at least 2 of 3
        else:
            min_overlap = 0.75   # First4/Trifecta 4: at least 3 of 4

        if overlap_ratio < min_overlap:
            continue

        # Blend value with probability — value alone is meaningless if the
        # absolute probability is negligible (e.g. two roughies in an Exacta).
        # prob_weight caps at 1.0 so strong-probability combos aren't over-boosted.
        raw_prob = ec.get("probability", 0)
        if isinstance(raw_prob, str):
            raw_prob = float(raw_prob.rstrip("%")) / 100
        prob_weight = min(raw_prob / 0.05, 1.0)  # full weight at ≥5% combined prob

        # Score = value * probability_weight + overlap bonus
        score = ec.get("value", 1.0) * prob_weight + overlap_ratio * 1.0

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
        # Exotic EV proxy: (value_ratio - 1) * probability * 20
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
        odds=best_sel.odds,
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
            pp.secondary_odds = s.odds

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
