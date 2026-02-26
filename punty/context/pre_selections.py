"""Deterministic pre-selection engine for race picks.

Pre-calculates bet types, stakes, pick order, Punty's Pick, and
recommended exotic for each race based on probability model output.
The LLM receives these as RECOMMENDED defaults and can override
with justification.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Pool constraints — 3-tier confidence staking
POOL_HIGH = 25.0       # strong edge detected (was 35, reduced to control total outlay)
POOL_STANDARD = 20.0   # normal
POOL_LOW = 12.0        # marginal edge only
RACE_POOL = POOL_STANDARD  # default (for backward compat)
MIN_STAKE = 1.0        # minimum bet
STAKE_STEP = 0.50      # round to nearest 50c
SINGLE_PICK_CAP = 15.0 # max stake when only 1 pick passes edge gate

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
ROUGHIE_MAX_ODDS = 20.0   # $20+ roughies historically -100% ROI
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
    tracked_only: bool = False  # True = displayed but not staked (edge gate failed)
    no_bet_reason: str | None = None  # Human-readable reason when tracked_only=True


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


@dataclass
class RacePreSelections:
    """Complete pre-calculated selections for one race."""

    race_number: int
    picks: list[RecommendedPick]
    exotic: RecommendedExotic | None
    puntys_pick: PuntysPick | None
    total_stake: float          # sum of pick stakes (should be <= $20)
    notes: list[str] = field(default_factory=list)
    classification: Any = None  # RaceClassification from bet_optimizer


def calculate_pre_selections(
    race_context: dict[str, Any],
    pool: float = RACE_POOL,
    selection_thresholds: dict | None = None,
    place_context_multipliers: dict[str, float] | None = None,
    venue_type: str = "",
    meeting_hit_count: int | None = None,
    meeting_race_count: int | None = None,
) -> RacePreSelections:
    """Calculate deterministic pre-selections for a single race.

    Uses the race-level bet optimizer to classify races and determine
    bet types based on EV, edge, odds movement, and venue confidence.

    Args:
        race_context: Race dict from ContextBuilder with runners and probabilities.
        pool: Total stake pool (default $20).
        selection_thresholds: Optional tuned thresholds from bet_type_tuning.
        place_context_multipliers: Optional dict of place context multiplier values.
            When the average multiplier is weak (<0.85), PLACE_MIN_PROB is raised
            to require higher place probability (tighter threshold, fewer Place bets).
            When strong (>1.2), threshold is lowered for wider Place coverage.
        venue_type: Venue tier (metro_vic, provincial, country, etc.)
        meeting_hit_count: Running count of winning selections at this meeting.
        meeting_race_count: How many races processed so far at this meeting.

    Returns:
        RacePreSelections with ordered picks, exotic, and Punty's Pick.
    """
    from punty.context.bet_optimizer import optimize_race

    race_number = race_context.get("race_number", 0)
    runners = race_context.get("runners", [])
    probs = race_context.get("probabilities", {})
    exotic_combos = probs.get("exotic_combinations", [])

    thresholds = _load_thresholds(selection_thresholds)

    # Adjust place threshold based on place context multipliers.
    # Tighten when context is weak (<0.85), loosen when strong (>1.2).
    if place_context_multipliers:
        mults = [v for v in place_context_multipliers.values() if isinstance(v, (int, float))]
        if mults:
            avg_place_strength = sum(mults) / len(mults)
            if avg_place_strength < 0.85:
                thresholds["place_min_prob"] = min(0.45, thresholds["place_min_prob"] + 0.05)
            elif avg_place_strength > 1.2:
                # Favourable conditions — lower threshold to capture more place bets
                thresholds["place_min_prob"] = max(0.25, thresholds["place_min_prob"] - 0.05)

    # Compute field size (active runners) for bet type and exotic decisions
    field_size = sum(1 for r in runners if not r.get("scratched"))

    # Run race-level optimizer for classification and bet type recommendations
    optimization = optimize_race(
        race_context, pool=pool, venue_type=venue_type,
        meeting_hit_count=meeting_hit_count,
        meeting_race_count=meeting_race_count,
    )
    classification = optimization.classification

    # Handle No Bet — 2yo maiden first starters with zero form
    if classification.no_bet:
        return RacePreSelections(
            race_number=race_number, picks=[], exotic=None,
            puntys_pick=None, total_stake=0.0,
            notes=["NO BET — 2yo maiden first starters, zero form data"],
            classification=classification,
        )

    # Build runner data with probability info
    candidates = _build_candidates(runners)
    if not candidates:
        return RacePreSelections(
            race_number=race_number, picks=[], exotic=None,
            puntys_pick=None, total_stake=0.0,
            classification=classification,
        )

    # Build lookup from optimizer recommendations: saddlecloth -> BetRecommendation
    rec_lookup = {r.saddlecloth: r for r in optimization.recommendations}

    # Split-formula ranking: different strategies per pick position.
    # Validated on 314 races (Feb 2026 production DB) + 18,888 Proform 2025 races:
    #   #1-2: Pure probability is best. Value weighting hurts #2 PnL.
    #   #3: prob * clamp(value, 0.85, 1.30) = +$281 vs pure prob.
    #       Reduces win losses significantly while place PnL drops only slightly.
    #   #4 roughie: best value from $8+ pool (unchanged).

    # NTD fields (5-7 runners): only 2 places paid, so select top 2 by place_prob.
    # Validated +29.3% ROI vs +21.0% baseline on 85 NTD races.
    # Remaining picks still tracked (displayed, not staked) for accuracy tracking.
    is_ntd = 5 <= field_size <= 7

    # Sort by pure probability (for #1 and #2 pick selection)
    candidates.sort(key=lambda c: c["win_prob"], reverse=True)

    # Separate roughie candidates (odds >= $8, value >= 1.1)
    roughie_pool = [
        c for c in candidates
        if ROUGHIE_MIN_ODDS <= c["odds"] <= ROUGHIE_MAX_ODDS and c["value_rating"] >= ROUGHIE_MIN_VALUE
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

    if is_ntd:
        # NTD path: select top 2 by place_prob from ALL candidates (not roughie)
        ntd_candidates = [c for c in candidates
                          if not (roughie and c["saddlecloth"] == roughie["saddlecloth"])]
        ntd_candidates.sort(key=lambda c: c["place_prob"], reverse=True)

        # Top 2 staked picks: #1 Win, #2 Place
        for c in ntd_candidates[:2]:
            picks.append(_make_pick_from_optimizer(
                c, len(picks) + 1, is_roughie=False, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size,
            ))
            used_saddlecloths.add(c["saddlecloth"])

        # Force bet types: pick #1 = Win, pick #2 = Place
        if len(picks) >= 1:
            picks[0].bet_type = "Win"
            picks[0].expected_return = round(
                picks[0].win_prob * picks[0].odds - 1, 2)
        if len(picks) >= 2:
            picks[1].bet_type = "Place"
            place_odds = picks[1].place_odds or _estimate_place_odds(picks[1].odds)
            picks[1].expected_return = round(
                picks[1].place_prob * place_odds - 1, 2)

        # Remaining non-roughie candidates become tracked_only (pick #3+)
        for c in ntd_candidates[2:]:
            if len(picks) >= 3:
                break
            pick = _make_pick_from_optimizer(
                c, len(picks) + 1, is_roughie=False, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size,
            )
            pick.tracked_only = True
            pick.no_bet_reason = "NTD field — only 2 staked picks"
            pick.stake = 0.0
            picks.append(pick)
            used_saddlecloths.add(c["saddlecloth"])

        # Roughie always tracked_only in NTD
        if roughie and roughie["saddlecloth"] not in used_saddlecloths:
            pick = _make_pick_from_optimizer(
                roughie, 4, is_roughie=True, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size,
            )
            pick.tracked_only = True
            pick.no_bet_reason = "NTD field — only 2 staked picks"
            pick.stake = 0.0
            picks.append(pick)
            used_saddlecloths.add(roughie["saddlecloth"])
    else:
        # Standard path: #1-2 by win_prob, #3 by prob*value, #4 roughie
        # #1 and #2 picks: pure probability (validated best for Win/Saver Win bets)
        for c in candidates:
            if len(picks) >= 2:
                break
            if c["saddlecloth"] in used_saddlecloths:
                continue
            if roughie and c["saddlecloth"] == roughie["saddlecloth"]:
                continue
            picks.append(_make_pick_from_optimizer(
                c, len(picks) + 1, is_roughie=False, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size,
            ))
            used_saddlecloths.add(c["saddlecloth"])

        # #3 pick: re-rank remaining by prob * clamped value (validated +$281 improvement)
        remaining = [c for c in candidates if c["saddlecloth"] not in used_saddlecloths
                     and not (roughie and c["saddlecloth"] == roughie["saddlecloth"])]
        remaining.sort(
            key=lambda c: c["win_prob"] * max(0.85, min(c["value_rating"], 1.30)),
            reverse=True,
        )

        if remaining:
            picks.append(_make_pick_from_optimizer(
                remaining[0], 3, is_roughie=False, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size,
            ))
            used_saddlecloths.add(remaining[0]["saddlecloth"])

        # Add roughie as pick #4
        if roughie and roughie["saddlecloth"] not in used_saddlecloths:
            picks.append(_make_pick_from_optimizer(
                roughie, 4, is_roughie=True, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size,
            ))
            used_saddlecloths.add(roughie["saddlecloth"])
        elif len(picks) < 4:
            # No qualifying roughie — fill 4th from remaining
            for c in candidates:
                if c["saddlecloth"] not in used_saddlecloths:
                    picks.append(_make_pick_from_optimizer(
                        c, len(picks) + 1, is_roughie=False, rec_lookup=rec_lookup,
                        thresholds=thresholds, field_size=field_size,
                    ))
                    used_saddlecloths.add(c["saddlecloth"])
                    break

    # Ensure at least one Win/Saver Win bet (mandatory rule)
    # Skip for PLACE_LEVERAGE — this classification can go all-Place
    if classification.race_type != "PLACE_LEVERAGE":
        _ensure_win_bet(picks)

    # Cap win-exposed bets to avoid spreading win risk across too many horses
    win_capped = _cap_win_exposure(picks)

    # Determine race pool using 3-tier confidence system
    race_pool = _determine_race_pool(picks, classification)

    # Allocate stakes with edge gating
    _allocate_stakes(picks, race_pool)

    # Select recommended exotic
    anchor_odds = picks[0].odds if picks else 0.0
    active_odds = [r.get("current_odds", 0) for r in runners
                   if not r.get("scratched") and r.get("current_odds", 0) > 0]
    fav_price = min(active_odds) if active_odds else 0.0
    exotic = _select_exotic(
        exotic_combos, used_saddlecloths,
        field_size=field_size, anchor_odds=anchor_odds,
        venue_type=venue_type, picks=picks,
        track_condition=race_context.get("track_condition", ""),
        race_class=race_context.get("class", ""),
        is_hk=(race_context.get("state") == "HK"),
        fav_price=fav_price,
        distance=race_context.get("distance", 0),
    )

    # Calculate Punty's Pick
    puntys_pick = _calculate_puntys_pick(picks, exotic)

    total_stake = sum(p.stake for p in picks)
    notes = _generate_notes(picks, exotic, candidates, field_size=field_size,
                            win_capped=win_capped, is_ntd=is_ntd)

    # Add classification note
    if classification.watch_only:
        notes.insert(0, f"WATCH ONLY — {classification.reasoning}")
    elif classification.race_type != "COMPRESSED_VALUE":
        notes.insert(0, f"Race type: {classification.race_type} — {classification.reasoning}")

    return RacePreSelections(
        race_number=race_number,
        picks=picks,
        exotic=exotic,
        puntys_pick=puntys_pick,
        total_stake=round(total_stake, 2),
        notes=notes,
        classification=classification,
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

        # Guard against AI-hallucinated place odds (HK races have no fixed place market)
        # Place odds should be roughly (win-1)/3+1; if way off, use estimate
        if place_odds:
            estimated = _estimate_place_odds(odds)
            if place_odds > estimated * 3:
                logger.warning(
                    "Hallucinated place odds: %s win=$%.1f place=$%.1f → $%.2f",
                    r.get("horse_name", "?"), odds, place_odds, estimated,
                )
                place_odds = estimated

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
    # Prefer Win over Place in small fields
    num_places = 0 if field_size <= 4 else (2 if field_size <= 7 else 3)

    if num_places == 0:
        # No place betting available — everything must be Win/Saver Win
        return "Win"

    if num_places == 2:
        # Only 2 places paid — Place is much harder to collect.
        # Prefer Win over straight Place.
        if rank <= 2 and win_prob >= 0.25:
            return "Win"
        if rank <= 2:
            return "Win"  # small fields (≤7) still need Win exposure
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

    # Very short-priced favourites (<$1.80): Place (edge gate will likely track it)
    # Place odds ≈ $1.10-$1.25 — neither Win nor Place is worth staking here.
    # Set to Place so the edge gate can properly evaluate place_prob.
    if odds < 1.80 and not is_roughie:
        return "Place"  # edge gate evaluates — likely tracked_only

    # Short-priced favourites ($1.80-$2.00): Place only
    # Win ROI at these odds is -38.9% historically.
    if odds < 2.00 and not is_roughie:
        return "Place"

    # $2.00-$2.40: Outside $4-$6 win zone — Place only
    if odds < 2.40 and not is_roughie:
        return "Place"

    # $2.40-$3.00: Outside $4-$6 win zone — Place only
    if 2.40 <= odds < 3.0:
        return "Place"

    # $3.00-$4.00: Outside $4-$6 win zone — Place only
    if 3.0 <= odds < 4.0 and not is_roughie:
        return "Place"

    # $4.00-$5.00: THE profit engine (+144.8% ROI, 54% win rate)
    # #1 must be Win. #2 gets Each Way for upside + place protection.
    if 4.0 <= odds < 5.0:
        if win_prob >= t["win_min_prob"] and value >= 0.95:
            if rank == 1:
                return "Win"
            if rank == 2:
                return "Place"  # E/W killed — rank 2 $4-5 → Place
            if value >= 1.05:
                return "Saver Win"
        # Low-prob fallback: good odds but not enough conviction → Place
        return "Place"

    # $5.00-$6.00: Mixed — Win data is thin (0/6), Place preferred
    if 5.0 <= odds <= 6.0:
        if win_prob >= t["win_min_prob"]:
            if rank == 1:
                if value >= 1.05:
                    return "Win"  # only with clear value edge
                return "Place"  # E/W killed — rank 1 $5-6 → Place
            if rank == 2:
                return "Place"  # was E/W — tighten to Place for rank 2
        # Low-prob fallback
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Place"

    # --- Roughie logic ---
    # Rank 4 Win: 0/43 = -98.7% ROI. All roughies → Place.
    if is_roughie:
        return "Place"

    # --- $6+ non-roughie: Place territory ---
    if rank == 1:
        if win_prob >= t["win_min_prob"] and value >= t["win_min_value"]:
            return "Place"  # $6+ Win is -42% ROI, E/W too risky — prefer Place
        if place_prob >= t["place_min_prob"] and place_value >= t["place_min_value"]:
            return "Place"
        return "Place"

    if rank == 2:
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
    """Create a RecommendedPick from candidate data (legacy path)."""
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


def _make_pick_from_optimizer(
    c: dict,
    rank: int,
    is_roughie: bool,
    rec_lookup: dict,
    thresholds: dict | None = None,
    field_size: int = 12,
) -> RecommendedPick:
    """Create a RecommendedPick using optimizer recommendation when available.

    Falls back to legacy _determine_bet_type if no optimizer recommendation
    exists for this saddlecloth (shouldn't happen in practice).
    """
    rec = rec_lookup.get(c["saddlecloth"])
    if rec:
        bet_type = rec.bet_type
    else:
        # Fallback to legacy
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
    """Ensure at least one pick is Win or Saver Win (mandatory rule)."""
    has_win = any(p.bet_type in ("Win", "Saver Win") for p in picks)
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
    _WIN_TYPES = {"Win", "Saver Win"}
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


def _determine_race_pool(
    picks: list[RecommendedPick],
    classification: Any = None,
) -> float:
    """Determine pool size based on edge confidence.

    3-tier system replaces flat $20 and watch-only half-pool:
    - HIGH ($25): Best pick has ev_win > 0.15 OR (place_prob > 0.50 AND place_value > 1.05)
    - STANDARD ($20): Default — at least one pick would pass edge gate
    - LOW ($12): Watch-only race OR only marginal edge detected

    Args:
        picks: Pre-computed picks (before staking, after bet type assignment).
        classification: RaceClassification from bet_optimizer.

    Returns:
        Pool amount in dollars.
    """
    from punty.context.bet_optimizer import DOMINANT_EDGE, NO_EDGE

    # Watch-only or no-edge → low pool
    if classification:
        if classification.watch_only or classification.no_bet:
            return POOL_LOW
        if classification.race_type == NO_EDGE:
            return POOL_LOW

    if not picks:
        return POOL_LOW

    # Check for HIGH confidence signals
    best = picks[0]  # rank 1
    ev_win = best.win_prob * best.odds - 1 if best.odds > 0 else -1
    has_high_ev = ev_win > 0.15
    has_strong_place = best.place_prob > 0.50 and best.place_value_rating > 1.05

    # DOMINANT_EDGE classification also qualifies for high pool
    is_dominant = classification and classification.race_type == DOMINANT_EDGE

    if has_high_ev or has_strong_place or is_dominant:
        return POOL_HIGH

    # Check if multiple picks have positive edge (standard confidence)
    positive_ev_count = sum(
        1 for p in picks
        if (p.win_prob * p.odds - 1) > 0 or
           (p.place_prob * (p.place_odds or _estimate_place_odds(p.odds)) - 1) > 0
    )

    if positive_ev_count >= 2:
        return POOL_STANDARD

    # Only 1 pick with marginal edge → low pool
    if positive_ev_count <= 1:
        return POOL_LOW

    return POOL_STANDARD


def _passes_edge_gate(pick: RecommendedPick, live_profile: dict | None = None) -> tuple[bool, str | None]:
    """Check if a pick passes the edge gate (proven-profitable criteria).

    Returns (True, None) if the pick should be staked,
    or (False, reason) with a human-readable explanation if tracked-only.
    Uses live edge profile when available (sample >= 50), otherwise hardcoded.
    """
    odds = pick.odds
    bt = pick.bet_type
    win_prob = pick.win_prob
    place_prob = pick.place_prob
    value = pick.value_rating
    place_value = pick.place_value_rating

    # Compute EV metrics for the negative-EV gate
    place_odds_est = pick.place_odds or _estimate_place_odds(odds)
    ev_win = win_prob * odds - 1
    ev_place = place_prob * place_odds_est - 1

    # Check live profile override: if live ROI for this bet_type+band is
    # strongly negative (< -15%) with sufficient sample, reject
    if live_profile:
        bt_key = bt.lower().replace(" ", "_")
        for label, lo, hi in _EDGE_ODDS_BANDS:
            if lo <= odds < hi:
                cell = live_profile.get((bt_key, label))
                if cell and cell["bets"] >= 50:
                    if cell["roi"] < -15.0:
                        return (False, f"Losing odds band (live ROI {cell['roi']:.0f}%)")
                break

    # --- Win-first staking criteria ---

    # 1. Win/Saver at $2.00-$10.00 — wide Win zone (optimizer sets bet type)
    if bt in ("Win", "Saver Win") and 2.0 <= odds <= 10.0:
        # $2.00-$4.00: needs reasonable conviction
        if odds < 4.0 and win_prob >= 0.20:
            return (True, None)
        # $4.00-$6.00: proven sweet spot
        if 4.0 <= odds <= 6.0:
            return (True, None)
        # $6.00-$10.00: needs some model confidence
        if odds > 6.0 and win_prob >= 0.15:
            return (True, None)

    # 2. Place with graduated prob threshold by odds band
    if bt == "Place":
        # Short odds ($1.50-$3): needs high collection prob
        if odds < 3.0 and place_prob >= 0.55:
            return (True, None)
        # Medium odds ($3-$6): standard threshold
        if 3.0 <= odds <= 6.0 and place_prob >= 0.40:
            return (True, None)
        # Longer odds ($6+): lower threshold acceptable (bigger payoff compensates)
        if odds > 6.0 and place_prob >= 0.35:
            return (True, None)

    # --- No Bet (tracked) zones ---

    # Win < $2.00: -38.9% ROI historically
    if bt in ("Win", "Saver Win") and odds < 2.0:
        return (False, f"Too short to back (Win < $2.00)")

    # Win $2.00-$4.00 without sufficient conviction
    if bt in ("Win", "Saver Win") and 2.0 <= odds < 4.0 and win_prob < 0.20:
        return (False, f"Not enough conviction (Win ${odds:.0f}, {win_prob * 100:.0f}% win prob)")

    # Place with low collection probability — graduated by odds
    if bt == "Place":
        place_floor = 0.55 if odds < 3.0 else (0.40 if odds <= 6.0 else 0.35)
        if place_prob < place_floor:
            return (False, f"Place prob too low ({place_prob * 100:.0f}% < {place_floor * 100:.0f}%)")

    # Negative expected value both ways — no edge
    if ev_win < -0.10 and ev_place < -0.05:
        return (False, "Negative expected value")

    # Default: pass through (don't over-filter)
    return (True, None)


# Odds bands matching strategy.py's _ODDS_BANDS for live profile lookup
_EDGE_ODDS_BANDS = [
    ("$1-$2", 1.0, 2.0),
    ("$2-$3", 2.0, 3.0),
    ("$3-$4", 3.0, 4.0),
    ("$4-$6", 4.0, 6.0),
    ("$6-$10", 6.0, 10.0),
    ("$10-$20", 10.0, 20.0),
    ("$20+", 20.0, 999.0),
]


def _allocate_stakes(picks: list[RecommendedPick], pool: float) -> None:
    """Allocate stakes from pool using edge-weighted sizing.

    Two-pass approach:
    1. Edge gate: mark picks as tracked_only if they don't pass proven-profitable criteria
    2. Distribute pool only among staked picks

    Historical edges (data-driven):
    - Win $4-$6: +60.8% ROI → deserves larger stake
    - Place (any rank): +13.8% ROI → solid base
    - Roughie $10-$20: +53% ROI → bump up from default 15%
    """
    if not picks:
        return

    # VR cap removed — let the optimizer's bet type stand.
    # Previously VR 1.2+ forced Win→Place, but we now trust the optimizer's
    # wider Win range ($2-$10) and don't want double-filtering.

    # --- Pass 1: Edge gate ---
    from punty.memory.strategy import get_cached_edge_profile
    live_profile = get_cached_edge_profile()

    for pick in picks:
        if pick.tracked_only:
            continue  # already marked (e.g. NTD path) — preserve original reason
        passed, reason = _passes_edge_gate(pick, live_profile)
        if not passed:
            pick.tracked_only = True
            pick.no_bet_reason = reason or "No proven edge in this odds band"
            pick.stake = 0.0

    # Count staked picks
    staked_picks = [p for p in picks if not p.tracked_only]

    # Fallback: if ALL picks fail edge gate, stake the best single Place bet
    if not staked_picks:
        # Find best place candidate
        best = max(picks, key=lambda p: p.place_prob)
        best.tracked_only = False
        best.bet_type = "Place"
        place_odds = best.place_odds or _estimate_place_odds(best.odds)
        best.expected_return = round(best.place_prob * place_odds - 1, 2)
        staked_picks = [best]

    # --- Pass 2: Allocate pool among staked picks ---
    # Cap pool when only 1 pick passes — prevents concentration blowups
    effective_pool = pool
    if len(staked_picks) == 1:
        effective_pool = min(pool, SINGLE_PICK_CAP)

    # Base allocation weights by rank
    base_rank_weights = {1: 0.35, 2: 0.28, 3: 0.22, 4: 0.15}

    pick_weights: list[float] = []
    for pick in staked_picks:
        base = base_rank_weights.get(pick.rank, 0.15)

        # Win bonus — broader range now that Win is default
        if pick.bet_type in ("Win", "Saver Win") and 2.5 <= pick.odds <= 8.0:
            base *= 1.15  # 15% stake boost for Win bets in range

        # Roughie $10-$20 bonus (+53% ROI sweet spot)
        if pick.is_roughie and 10.0 <= pick.odds <= 20.0:
            base *= 1.20  # 20% stake boost

        # Positive expected return bonus — tilt more to profitable bets
        if pick.expected_return > 0.10:
            base *= 1.15  # 15% bonus for strong +EV

        # High-confidence Place bonus — Place is our profit engine (+7.55% ROI)
        if pick.bet_type == "Place" and pick.place_prob >= 0.45:
            base *= 1.15  # 15% stake boost for high-confidence Place

        # Place overlay stake modifier — reduce stake on thin-edge bets
        # Market place prob estimated as 3/odds (place pays ~1/3 of field)
        if pick.bet_type == "Place" and pick.place_prob and pick.odds and pick.odds > 1.0:
            market_place_prob = min(1.0, 3.0 / pick.odds)
            overlay = pick.place_prob / market_place_prob if market_place_prob > 0 else 1.0
            if overlay < 1.05:
                base *= 0.60  # Thin edge: reduce 40%
            elif overlay < 1.10:
                base *= 0.80  # Moderate edge: reduce 20%

        pick_weights.append(base)

    total_weight = sum(pick_weights)
    if total_weight <= 0:
        return

    for i, pick in enumerate(staked_picks):
        weight = pick_weights[i]
        raw_stake = effective_pool * (weight / total_weight)

        # Round to nearest 50c, minimum $1
        stake = max(MIN_STAKE, round(raw_stake / STAKE_STEP) * STAKE_STEP)
        pick.stake = stake

    # Verify total doesn't exceed effective pool
    total = sum(p.stake for p in staked_picks)
    if total > effective_pool + 0.01:
        # Scale down proportionally
        scale = effective_pool / total
        for p in staked_picks:
            p.stake = max(MIN_STAKE, round(p.stake * scale / STAKE_STEP) * STAKE_STEP)


def _select_exotic(
    exotic_combos: list[dict],
    selection_saddlecloths: set[int],
    field_size: int = 0,
    anchor_odds: float = 0.0,
    venue_type: str = "",
    picks: list[RecommendedPick] | None = None,
    track_condition: str = "",
    race_class: str = "",
    is_hk: bool = False,
    fav_price: float = 0.0,
    distance: int = 0,
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

    # Data-driven filters (Feb 24 audit)
    tc = (track_condition or "").lower()
    if "heavy" in tc:
        return None  # 0/21 exotic hits on Heavy tracks
    soft_match = re.search(r'soft\s*(\d+)', tc)
    if soft_match and int(soft_match.group(1)) >= 7:
        return None  # 0% strike on Soft 7+
    if is_hk:
        return None  # 0/11 exotic hits on HK races
    if field_size and field_size <= 6:
        return None  # All types losing in ≤6 fields

    # --- Tight cluster detection ---
    # When top 3 non-roughie picks are within 8% win probability,
    # box exotics become much more attractive (any order equally likely).
    cluster_boost = 1.0
    if picks:
        top_probs = [p.win_prob for p in picks if not p.is_roughie][:3]
        if len(top_probs) >= 3:
            spread = max(top_probs) - min(top_probs)
            if spread <= 0.05:
                cluster_boost = 1.2   # very tight — prefer boxes (dampened from 1.5)
            elif spread <= 0.08:
                cluster_boost = 1.1   # tight — slight box preference (dampened from 1.25)

    # Build rank map: saddlecloth → tip_rank (1=best, 4=roughie)
    rank_map: dict[int, int] = {}
    if picks:
        for p in picks:
            rank_map[p.saddlecloth] = p.rank

    # Score all combos by expected value: probability × value_ratio
    # Higher EV = better mix of probability and payout
    scored = []
    for ec in exotic_combos:
        runners = set(ec.get("runners", []))
        n_runners = len(runners)
        overlap = len(runners & selection_saddlecloths)
        overlap_ratio = overlap / n_runners if n_runners else 0

        ec_type = ec.get("type", "")

        # Trifecta Standout: only when #1 pick is dominant (win_prob >= 0.30)
        # AND field size 8-12 (not too open, not too small)
        if ec_type == "Trifecta Standout":
            top_pick_prob = max((p.win_prob for p in picks if p.rank == 1), default=0)
            if top_pick_prob < 0.30 or not (8 <= field_size <= 12):
                continue

        # Trifecta Box: profitable only in narrow scenario (field 11-13,
        # non-sprint, fav $2-$3.50, Good/Soft track). All other combos lose.
        if ec_type == "Trifecta Box":
            if not (11 <= field_size <= 13):
                continue
            if distance and distance <= 1200:
                continue
            if fav_price and not (2.01 <= fav_price <= 3.50):
                continue

        # Exacta/Quinella fav_price guard: only profitable when fav $2-$3.50
        if fav_price and ec_type in ("Exacta", "Quinella"):
            if fav_price <= 2.0 or fav_price > 3.50:
                continue

        # Overlap rules by exotic type:
        # Quinella/Exacta: ALL runners must be from our picks (strict 2-runner bets)
        # Trifecta/First4: at least 2 runners from picks (allow ranking runners
        # in trailing positions — 3rd for tri, 4th+ for First4)
        if ec_type in ("Quinella", "Exacta", "Exacta Standout"):
            if overlap_ratio < 1.0:
                continue
        else:
            # Trifecta, First4, etc: require at least 2 from picks
            if overlap < min(2, n_runners):
                continue

        # Quinella 2-runner in small fields: -73% ROI, skip
        if ec_type == "Quinella" and n_runners == 2 and field_size and field_size <= 10:
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

        # Top-pick bonus: exotics should anchor on our best selections.
        # Combos with pick #1 get a 30% boost, pick #2 gets 15%.
        # This prevents the exotic from skipping our strongest horse
        # in favour of a roughie-heavy combo with marginally higher raw EV.
        if rank_map:
            runner_ranks = [rank_map.get(r, 99) for r in runners]
            best_rank = min(runner_ranks)
            if best_rank == 1:
                score *= 1.30
            elif best_rank == 2:
                score *= 1.15

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

    Picks the highest-confidence selection (most likely to collect),
    excluding roughies. Exotic wins only if value >= 1.5x and EV
    clearly beats the best selection.
    """
    if not picks:
        return None

    # Exclude roughies — they're speculative, not best-bet material
    candidates = [p for p in picks if not p.is_roughie]
    if not candidates:
        candidates = picks  # fallback if all are roughies (shouldn't happen)

    # Best selection by confidence (probability of collecting)
    def _collect_prob(p: RecommendedPick) -> float:
        if p.bet_type in ("Win", "Saver Win"):
            return p.win_prob
        elif p.bet_type == "Place":
            return p.place_prob
        elif p.bet_type == "Each Way":
            return p.place_prob  # EW collects if places
        return p.win_prob

    best_sel = max(candidates, key=_collect_prob)

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
    is_ntd: bool = False,
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
    elif is_ntd:
        notes.append(
            f"NTD field ({field_size} runners) — only 2 places paid. "
            f"2 staked picks (1 Win + 1 Place) selected by place_prob."
        )

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
    # Classification-aware header
    cls = pre_sel.classification
    if cls and cls.no_bet:
        lines = [f"\n**NO BET (Race {pre_sel.race_number}) — 2yo maiden first starters, zero form:**"]
        if pre_sel.notes:
            for note in pre_sel.notes:
                lines.append(f"  NOTE: {note}")
        return "\n".join(lines)

    if cls and cls.watch_only:
        lines = [
            f"\n**WATCH ONLY (Race {pre_sel.race_number}) — {cls.race_type}:**",
            f"  {cls.reasoning}",
        ]
    elif cls and cls.race_type != "COMPRESSED_VALUE":
        lines = [
            f"\n**LOCKED SELECTIONS (Race {pre_sel.race_number}) — {cls.race_type}:**",
            f"  Race type: {cls.race_type} ({cls.confidence:.0%}) — {cls.reasoning}",
        ]
    else:
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

        lines.append(
            f"  {rank_label}: {pick.horse_name} (No.{pick.saddlecloth}) "
            f"— ${pick.odds:.2f} / ${(pick.place_odds or _estimate_place_odds(pick.odds)):.2f}"
        )
        lines.append(
            f"    STATS: {prob_label} | Value: {value_label}"
        )

        # Tracked picks: no stake, displayed for accuracy tracking
        if pick.tracked_only:
            reason = pick.no_bet_reason or "edge gate failed"
            lines.append(
                f"    BET: No Bet — {reason}"
            )
        else:
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

            lines.append(f"    BET: {bet_desc}")

    if pre_sel.exotic:
        ex = pre_sel.exotic
        runners_str = ", ".join(str(r) for r in ex.runners)
        names_str = ", ".join(ex.runner_names)
        lines.append(
            f"  Exotic: {ex.exotic_type} [{runners_str}] {names_str} "
            f"— $20 | Prob: {ex.probability * 100:.1f}% "
            f"| Value: {ex.value_ratio:.2f}x | {ex.num_combos} combos"
        )

    if pre_sel.notes:
        for note in pre_sel.notes:
            lines.append(f"  NOTE: {note}")

    lines.append(f"  Total stake: ${pre_sel.total_stake:.2f}")

    # Punty's Pick — explicit instruction for the AI
    pp = pre_sel.puntys_pick
    if pp:
        if pp.pick_type == "exotic" and pp.exotic_type:
            runners_str = ", ".join(str(r) for r in pp.exotic_runners)
            val_str = f" (Value: {pp.exotic_value:.1f}x)" if pp.exotic_value else ""
            lines.append(
                f"  PUNTY'S PICK: {pp.exotic_type} [{runners_str}] — $20{val_str}"
            )
        elif pp.horse_name and pp.saddlecloth:
            lines.append(
                f"  PUNTY'S PICK: {pp.horse_name} (No.{pp.saddlecloth}) "
                f"${pp.odds:.2f} {pp.bet_type}"
            )

    return "\n".join(lines)
