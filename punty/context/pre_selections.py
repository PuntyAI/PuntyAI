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
    venue_type: str = "",
) -> RacePreSelections:
    """Calculate deterministic pre-selections for a single race.

    Args:
        race_context: Race dict from ContextBuilder with runners and probabilities.
        pool: Total stake pool (default $20).
        selection_thresholds: Optional tuned thresholds from bet_type_tuning.
        place_context_multipliers: Optional dict of place context multiplier values.
            When the average multiplier is weak (<0.85), PLACE_MIN_PROB is raised
            to require higher place probability (tighter threshold, fewer Place bets).

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

    # Compute field size (active runners) for bet type and exotic decisions
    field_size = sum(1 for r in runners if not r.get("scratched"))

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
        picks.append(_make_pick(c, len(picks) + 1, is_roughie=False, thresholds=thresholds, field_size=field_size))
        used_saddlecloths.add(c["saddlecloth"])

    # #3 pick: re-rank remaining by prob * clamped value (validated +$281 improvement)
    remaining = [c for c in candidates if c["saddlecloth"] not in used_saddlecloths
                 and not (roughie and c["saddlecloth"] == roughie["saddlecloth"])]
    remaining.sort(
        key=lambda c: c["win_prob"] * max(0.85, min(c["value_rating"], 1.30)),
        reverse=True,
    )

    if remaining:
        picks.append(_make_pick(remaining[0], 3, is_roughie=False, thresholds=thresholds, field_size=field_size))
        used_saddlecloths.add(remaining[0]["saddlecloth"])

    # Add roughie as pick #4
    if roughie and roughie["saddlecloth"] not in used_saddlecloths:
        picks.append(_make_pick(roughie, 4, is_roughie=True, thresholds=thresholds, field_size=field_size))
        used_saddlecloths.add(roughie["saddlecloth"])
    elif len(picks) < 4:
        # No qualifying roughie — fill 4th from remaining
        for c in candidates:
            if c["saddlecloth"] not in used_saddlecloths:
                picks.append(_make_pick(c, len(picks) + 1, is_roughie=False, thresholds=thresholds, field_size=field_size))
                used_saddlecloths.add(c["saddlecloth"])
                break

    # Ensure at least one Win/Each Way/Saver Win bet (mandatory rule)
    _ensure_win_bet(picks)

    # Cap win-exposed bets to avoid spreading win risk across too many horses
    win_capped = _cap_win_exposure(picks)

    # Allocate stakes from pool
    _allocate_stakes(picks, pool)

    # Select recommended exotic
    anchor_odds = picks[0].odds if picks else 0.0
    exotic = _select_exotic(
        exotic_combos, used_saddlecloths,
        field_size=field_size, anchor_odds=anchor_odds,
        venue_type=venue_type, picks=picks,
    )

    # Calculate Punty's Pick
    puntys_pick = _calculate_puntys_pick(picks, exotic)

    total_stake = sum(p.stake for p in picks)
    notes = _generate_notes(picks, exotic, candidates, field_size=field_size,
                            win_capped=win_capped)

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


def _determine_bet_type(c: dict, rank: int, is_roughie: bool, thresholds: dict | None = None, field_size: int = 12) -> str:
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

    # Field size affects place payouts:
    # ≤4 runners: no place betting, ≤7 runners: only 2 places paid
    # Prefer Win/Each Way over Place in small fields
    num_places = 0 if field_size <= 4 else (2 if field_size <= 7 else 3)

    if num_places == 0:
        # No place betting available — everything must be Win/Saver Win
        return "Win"

    if num_places == 2:
        # Only 2 places paid — Place is much harder to collect.
        # Prefer Win or Each Way over straight Place.
        if rank <= 2 and win_prob >= 0.25:
            return "Win"
        if rank <= 2:
            return "Each Way"
        # Lower ranks: only Place if very high place probability
        if place_prob >= 0.55 and place_value >= 1.0:
            return "Place"
        return "Win"  # default to Win in small fields

    # --- Edge-aware odds-band rules (validated on all settled bets) ---
    #
    # Win ROI by band: <$2 = -38.9%, $2.40-$3 = +21.3%, $3-$4 = -30.8%,
    #   $4-$5 = +144.8%, $5-$6 = -100%, $6+ = -42%
    # Place ROI by band: <$2 = +24%, $2-$4 = +20%, $4-$6 = +35%, $6-$10 = -0.3%
    # EW collect rate: $2.40-$3 = 94%, $3-$4 = 62%

    # Very short-priced favourites (<$1.80): Win or skip
    # Place odds ≈ $1.10-$1.25 — a Place bet here is more worthless than Win.
    # Small Win bet if value is there; otherwise minimal stake Win.
    # Does NOT affect exotics or sequences — only the selection bet.
    if odds < 1.80 and not is_roughie:
        return "Win"  # flagged for minimal stake in _allocate_stakes

    # Short-priced favourites ($1.80-$2.00): Win preferred over Place
    # At these odds, Place returns ~$1.25 — not worth it.
    if odds < 2.00 and not is_roughie:
        return "Win"

    # $2.00-$2.40: Win is viable with value edge
    if odds < 2.40 and not is_roughie:
        if rank <= 2 and win_prob >= 0.35 and value >= 1.05:
            return "Win"  # solid value overlay — don't force Place
        if rank == 1 and win_prob >= 0.30 and value >= 1.00:
            return "Each Way"  # protect with EW instead of straight Place
        return "Place"

    # $2.40-$3.00: Win sweet spot #1 (+21.3% ROI, 47% win rate)
    # #2 gets Each Way (94% collect rate at these odds)
    if 2.40 <= odds < 3.0:
        if win_prob >= t["win_min_prob"]:
            if rank == 1 and value >= 0.95:
                return "Win"
            if rank == 2:
                return "Each Way"
        # Low-prob fallback: still competitive odds, Each Way or Place
        if rank <= 2:
            return "Each Way"
        return "Place"

    # $3.00-$4.00: Dead zone for Win (-30.8% ROI)
    # #1 gets Each Way (62% collect rate), #2 gets Place
    if 3.0 <= odds < 4.0 and not is_roughie:
        if rank == 1:
            if win_prob >= 0.25 and value >= 1.10:
                return "Win"  # only with strong conviction
            if win_prob >= t["each_way_min_prob"]:
                return "Each Way"
            return "Place"
        if rank == 2:
            if win_prob >= 0.25 and value >= 1.10:
                return "Each Way"
            return "Place"
        return "Place"

    # $4.00-$5.00: THE profit engine (+144.8% ROI, 54% win rate)
    # #1 must be Win. #2 gets Each Way for upside + place protection.
    if 4.0 <= odds < 5.0:
        if win_prob >= t["win_min_prob"] and value >= 0.95:
            if rank == 1:
                return "Win"
            if rank == 2:
                return "Each Way"
            if value >= 1.05:
                return "Saver Win"
        # Low-prob fallback: good odds range, still worth Each Way/Place
        if rank <= 2 and place_prob >= t["place_min_prob"]:
            return "Each Way"
        return "Place"

    # $5.00-$6.00: Mixed — Win data is thin (0/6), lean Each Way
    if 5.0 <= odds <= 6.0:
        if win_prob >= t["win_min_prob"]:
            if rank == 1:
                if value >= 1.05:
                    return "Win"  # only with clear value edge
                return "Each Way"
            if rank == 2:
                return "Each Way"
        # Low-prob fallback
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Place"

    # --- Roughie logic ---
    if is_roughie:
        if 10.0 <= odds <= 20.0 and win_prob >= t["win_min_prob"] and value >= 1.15:
            return "Win"
        if win_prob >= t["win_min_prob"] and value >= 1.25:
            return "Win"
        return "Place"

    # --- $6+ non-roughie: Place territory ---
    if rank == 1:
        if win_prob >= t["win_min_prob"] and value >= t["win_min_value"]:
            if (t["each_way_min_odds"] <= odds <= t["each_way_max_odds"]
                    and t["each_way_min_prob"] <= win_prob <= t["each_way_max_prob"]):
                return "Each Way"
            return "Place"  # $6+ Win is -42% ROI, prefer Place
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Place"

    if rank == 2:
        if win_prob >= 0.25 and value >= 1.10:
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


def _make_pick(c: dict, rank: int, is_roughie: bool, thresholds: dict | None = None, field_size: int = 12) -> RecommendedPick:
    """Create a RecommendedPick from candidate data."""
    bet_type = _determine_bet_type(c, rank, is_roughie, thresholds, field_size=field_size)
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


MAX_WIN_EXPOSED = 2  # Max bets with win exposure (Win/EW/Saver Win) per race


def _cap_win_exposure(picks: list[RecommendedPick]) -> int:
    """Cap win-exposed bets at MAX_WIN_EXPOSED per race.

    With 4 picks all needing different horses to win, you lose ~50% of the time.
    Capping at 2 win-exposed bets and pushing picks 3-4 to Place improves
    overall race ROI by covering place scenarios instead of spreading win risk.

    Returns:
        Number of picks downgraded to Place.
    """
    _WIN_TYPES = {"Win", "Saver Win", "Each Way"}
    win_exposed = [p for p in picks if p.bet_type in _WIN_TYPES]
    if len(win_exposed) <= MAX_WIN_EXPOSED:
        return 0

    # Keep win exposure on highest-ranked picks, downgrade the rest to Place
    # Sort by rank so we keep the best picks' bet types
    win_exposed.sort(key=lambda p: p.rank)
    downgraded = 0
    for pick in win_exposed[MAX_WIN_EXPOSED:]:
        pick.bet_type = "Place"
        # Recalculate expected return for Place
        place_odds = pick.place_odds if pick.place_odds else _estimate_place_odds(pick.odds)
        pick.expected_return = round(pick.place_prob * place_odds - 1, 2)
        downgraded += 1
    return downgraded


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
    base_rank_weights = {1: 0.35, 2: 0.28, 3: 0.22, 4: 0.15}

    # Edge-aware multipliers on top of rank weights — per pick, not shared dict
    pick_weights: list[float] = []
    for pick in picks:
        base = base_rank_weights.get(pick.rank, 0.15)

        # Win sweet spot $4-$6 bonus (our best edge at +60.8% ROI)
        if pick.bet_type in ("Win", "Saver Win") and 4.0 <= pick.odds <= 6.0:
            base *= 1.25  # 25% stake boost in sweet spot

        # Roughie $10-$20 bonus (+53% ROI sweet spot)
        if pick.is_roughie and 10.0 <= pick.odds <= 20.0:
            base *= 1.20  # 20% stake boost

        # Positive expected return bonus — tilt more to profitable bets
        if pick.expected_return > 0.10:
            base *= 1.15  # 15% bonus for strong +EV

        # Very short-priced Win (<$1.80): minimal stake — odds-on favs
        # Still included so AI can reference them, but don't burn budget
        if pick.odds < 1.80 and not pick.is_roughie:
            base *= 0.25  # near-minimum stake

        # Short-priced Win ($1.80-$2.00): reduced stake — -38.9% ROI historically
        elif pick.odds < 2.00 and pick.bet_type in ("Win", "Saver Win") and not pick.is_roughie:
            base *= 0.50  # half stake — poor Win ROI at these odds

        # Short-priced Place penalty — these are safe but low-returning
        elif pick.bet_type == "Place" and pick.odds < 2.5:
            base *= 0.75  # reduce stake on low-odds Place

        pick_weights.append(base)

    total_weight = sum(pick_weights)

    for i, pick in enumerate(picks):
        weight = pick_weights[i]
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
    field_size: int = 0,
    anchor_odds: float = 0.0,
    venue_type: str = "",
    picks: list[RecommendedPick] | None = None,
) -> RecommendedExotic | None:
    """Select the best exotic bet based on probability × value (EV).

    All exotic types compete on equal footing — Quinella, Exacta,
    Trifecta Box, Trifecta Standout, First4, First4 Box.
    The Harville model probabilities and value ratios drive the selection.

    When top picks are tightly clustered in probability, box exotics get
    a scoring boost (any ordering is equally likely, so covering all
    orderings de-risks the bet).

    Overlap rules: ALL exotic runners MUST be from our selections.
    """
    if not exotic_combos:
        return None

    # --- Tight cluster detection ---
    # When top 3 non-roughie picks are within 8% win probability,
    # box exotics become much more attractive (any order equally likely).
    cluster_boost = 1.0
    if picks:
        top_probs = [p.win_prob for p in picks if not p.is_roughie][:3]
        if len(top_probs) >= 3:
            spread = max(top_probs) - min(top_probs)
            if spread <= 0.05:
                cluster_boost = 1.5   # very tight — strongly prefer boxes
            elif spread <= 0.08:
                cluster_boost = 1.25  # tight — moderate box preference

    # Score all combos by expected value: probability × value_ratio
    # Higher EV = better mix of probability and payout
    scored = []
    for ec in exotic_combos:
        runners = set(ec.get("runners", []))
        n_runners = len(runners)
        overlap = len(runners & selection_saddlecloths)
        overlap_ratio = overlap / n_runners if n_runners else 0

        # ALL exotic runners must be from selections
        if overlap_ratio < 1.0:
            continue

        # Parse probability
        raw_prob = ec.get("probability", 0)
        if isinstance(raw_prob, str):
            raw_prob = float(raw_prob.rstrip("%")) / 100

        value = ec.get("value", 1.0)

        # EV score: probability × value_ratio
        # This naturally favours combos where both probability AND value are good
        # A 10% prob × 2.0 value (EV 0.20) beats 5% prob × 3.0 value (EV 0.15)
        ev_score = raw_prob * value

        # Small combo efficiency bonus: fewer combos = higher unit stake = bigger payout
        # Normalise: 1 combo = +10%, 24 combos = +0%
        combos = max(1, ec.get("combos", 1))
        efficiency_bonus = max(0, (1 - combos / 24)) * 0.1 * ev_score

        score = ev_score + efficiency_bonus

        # Tight cluster boost: prefer box exotics when picks are bunched
        if cluster_boost > 1.0:
            ec_format = ec.get("format", "")
            ec_type = ec.get("type", "")
            if ec_format == "boxed":
                score *= cluster_boost
            elif ec_format in ("flat", "standout") and ec_type != "Quinella":
                # Quinella is inherently unordered (a "box" of 2) — no penalty
                score *= 0.85  # slight penalty for directional in tight fields

        scored.append((score, ec))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return None

    best = scored[0][1]
    best_prob = best.get("probability", 0)
    if isinstance(best_prob, str):
        best_prob = float(best_prob.rstrip("%")) / 100

    return RecommendedExotic(
        exotic_type=best.get("type", ""),
        runners=best.get("runners", []),
        runner_names=best.get("runner_names", []),
        probability=best_prob,
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
        # Compare exotic EV to selection EV using consistent metric:
        # Both as expected return per $1 (prob * payout - 1)
        exotic_ev = exotic.probability * exotic.value_ratio - 1.0  # expected return per $1
        sel_ev = best_sel.expected_return  # also expected return per $1

        if exotic_ev > sel_ev * 1.2:  # exotic must be clearly better
            return PuntysPick(
                pick_type="exotic",
                exotic_type=exotic.exotic_type,
                exotic_runners=exotic.runners,
                exotic_value=exotic.value_ratio,
                expected_value=exotic_ev,
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
    field_size: int = 12,
    win_capped: int = 0,
) -> list[str]:
    """Generate advisory notes about the selections."""
    notes = []

    # Win exposure cap triggered
    if win_capped > 0:
        notes.append(
            f"Win exposure capped — {win_capped} pick(s) moved to Place "
            f"to protect race ROI (max {MAX_WIN_EXPOSED} win-exposed bets)."
        )

    # Small field warnings
    if field_size <= 4:
        notes.append(f"Small field ({field_size} runners) — no place betting available. Win bets only.")
    elif field_size <= 7:
        notes.append(f"Small field ({field_size} runners) — only 2 places paid. Win/Each Way preferred.")

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

    # Flag tight cluster (box exotic territory)
    top_probs = [p.win_prob for p in picks if not p.is_roughie][:3]
    if len(top_probs) >= 3:
        spread = max(top_probs) - min(top_probs)
        if spread <= 0.08:
            probs_str = "/".join(f"{p*100:.0f}%" for p in top_probs)
            notes.append(
                f"Tight top 3 ({probs_str}, {spread*100:.0f}% spread) "
                f"— box exotic preferred over directional."
            )

    return notes


def format_pre_selections(pre_sel: RacePreSelections) -> str:
    """Format pre-selections for injection into AI prompt context."""
    lines = [f"\n**LOCKED SELECTIONS (Race {pre_sel.race_number}) — DO NOT REORDER:**"]

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

        # Build bet description with correct return for bet type
        if pick.bet_type == "Each Way":
            half = pick.stake / 2
            po = pick.place_odds or _estimate_place_odds(pick.odds)
            bet_desc = (
                f"${pick.stake:.2f} Each Way "
                f"(=${half:.2f}W + ${half:.2f}P), "
                f"return ${round(pick.stake * pick.odds, 2):.2f} (wins) / "
                f"${round(pick.stake * po, 2):.2f} (places)"
            )
        elif pick.bet_type == "Place":
            po = pick.place_odds or _estimate_place_odds(pick.odds)
            bet_desc = f"${pick.stake:.2f} Place, return ${ret:.2f}"
        else:
            bet_desc = f"${pick.stake:.2f} {pick.bet_type}, return ${ret:.2f}"

        lines.append(
            f"  {rank_label}: {pick.horse_name} (No.{pick.saddlecloth}) "
            f"— ${pick.odds:.2f} / ${(pick.place_odds or _estimate_place_odds(pick.odds)):.2f}"
        )
        lines.append(
            f"    STATS: {prob_label} | Value: {value_label}"
        )
        lines.append(
            f"    BET: {bet_desc}"
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
