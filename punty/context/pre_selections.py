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
MAX_STAKED_PICKS = 3   # max picks with real stakes per race (picks #4+ tracked only)

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
ROUGHIE_MAX_ODDS = 51.0   # Include longshots — tracked_only, no stake risk
ROUGHIE_MIN_VALUE = 1.10

# Punty's Pick exotic threshold
EXOTIC_PUNTYS_PICK_VALUE = 1.5  # exotic needs 1.5x value to be Punty's Pick

# Exotic budget — default fallback (dynamic budget via _get_exotic_budget)
EXOTIC_BUDGET = 8.0


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
    budget: float = EXOTIC_BUDGET  # dynamic budget based on meet quality
    legs: list[list[int]] | None = None  # positional legs for structured exotics (e.g. [[1], [2,3,4], [2,3,4,5]])


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
    race_class = race_context.get("class", "")
    race_distance = race_context.get("distance", 0)

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

    # Ultra-small field (≤2 runners): no bet — no value in any market
    if field_size <= 2:
        return RacePreSelections(
            race_number=race_number, picks=[], exotic=None,
            puntys_pick=None, total_stake=0.0,
            notes=[f"NO BET — only {field_size} runner(s) after scratchings, no betting value"],
        )

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
        logger.info(f"R{race_number}: no candidates (all filtered)")
        return RacePreSelections(
            race_number=race_number, picks=[], exotic=None,
            puntys_pick=None, total_stake=0.0,
            classification=classification,
        )
    logger.info(f"R{race_number}: {len(candidates)} candidates, field={field_size}, class={classification.race_type}")

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

    # Sort by pure win probability — R1 must always be the highest WP horse.
    # VR tiebreaker removed — it promoted losers (VR 1.20+ wins 14.5% vs VR<0.90 at 29.7%).
    # Confidence boost is preserved in candidate data for staking decisions but does NOT
    # affect ranking order. Previously boost up to 0.20 could override 1-2% WP differences,
    # causing R1 to display lower WP than R2 (user-reported Rosehill R7 bug).
    candidates.sort(
        key=lambda c: c["win_prob"],
        reverse=True,
    )

    # Log top candidates after sort
    for i, c in enumerate(candidates[:6]):
        lgbm_rank = c.get("lgbm_rank", "?")
        logger.info(
            f"R{race_number} candidate #{i+1}: {c['horse_name']} (#{c['saddlecloth']}) "
            f"WP={c['win_prob']:.1%} PP={c['place_prob']:.1%} "
            f"odds=${c['odds']:.2f} VR={c['value_rating']:.2f} "
            f"LGBM={lgbm_rank} boost={c.get('confidence_boost', 0):.2f}"
        )

    # Determine favourite price (lowest odds) for dominant fav logic
    fav_price = min((c["odds"] for c in candidates), default=None)

    # Protect top-3 candidates from being demoted to roughie.
    # Candidates are already sorted by win_prob — top 3 should always be R1/R2/R3.
    top3_scs = {c["saddlecloth"] for c in candidates[:3]} if len(candidates) >= 3 else set()

    # Roughie: best probability from $9+ shots, outside top 3
    roughie_pool = [
        c for c in candidates
        if c["odds"] >= 9.0
        and c["odds"] <= ROUGHIE_MAX_ODDS
        and c["saddlecloth"] not in top3_scs
    ]

    # Select top 3 picks + roughie
    picks: list[RecommendedPick] = []
    used_saddlecloths: set[int] = set()

    # Pick roughie first (so we can exclude from main pool)
    roughie = None
    if roughie_pool:
        # Best probability roughie from $9+ shots
        roughie_pool.sort(key=lambda c: c["win_prob"], reverse=True)
        roughie = roughie_pool[0]
        logger.info(
            f"R{race_number} roughie: {roughie['horse_name']} (#{roughie['saddlecloth']}) "
            f"odds=${roughie['odds']:.2f} VR={roughie['value_rating']:.2f}"
        )
    else:
        logger.info(f"R{race_number} no qualifying roughie (pool={len(roughie_pool)})")

    if is_ntd:
        # NTD path: select top 2 by place_prob from ALL candidates (not roughie)
        ntd_candidates = [c for c in candidates
                          if not (roughie and c["saddlecloth"] == roughie["saddlecloth"])]
        ntd_candidates.sort(key=lambda c: c["place_prob"], reverse=True)

        # Top 2 staked picks: #1 Win, #2 Place
        for c in ntd_candidates[:2]:
            picks.append(_make_pick_from_optimizer(
                c, len(picks) + 1, is_roughie=False, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
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
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
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
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
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
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
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
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
            ))
            used_saddlecloths.add(remaining[0]["saddlecloth"])

        # Add roughie as pick #4
        if roughie and roughie["saddlecloth"] not in used_saddlecloths:
            picks.append(_make_pick_from_optimizer(
                roughie, 4, is_roughie=True, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
            ))
            used_saddlecloths.add(roughie["saddlecloth"])
        elif len(picks) < 4:
            # No qualifying roughie — fill 4th from remaining
            for c in candidates:
                if c["saddlecloth"] not in used_saddlecloths:
                    picks.append(_make_pick_from_optimizer(
                        c, len(picks) + 1, is_roughie=False, rec_lookup=rec_lookup,
                        thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                        race_class=race_class, distance=race_distance,
                    ))
                    used_saddlecloths.add(c["saddlecloth"])
                    break

        # Add picks #5 and #6 (tracked_only) for wider exotic coverage
        # These feed into F4 Standout legs: R1/R2-R4/R2-R5/R2-R6
        remaining_for_5_6 = [c for c in candidates if c["saddlecloth"] not in used_saddlecloths]
        remaining_for_5_6.sort(key=lambda c: c["place_prob"], reverse=True)
        for c in remaining_for_5_6[:2]:
            rank = len(picks) + 1
            pick = _make_pick_from_optimizer(
                c, rank, is_roughie=False, rec_lookup=rec_lookup,
                thresholds=thresholds, field_size=field_size, fav_price=fav_price,
                race_class=race_class, distance=race_distance,
            )
            pick.tracked_only = True
            pick.no_bet_reason = "Exotic coverage only (R5/R6)"
            pick.stake = 0.0
            pick.bet_type = "Place"
            picks.append(pick)
            used_saddlecloths.add(c["saddlecloth"])

    # Ensure at least one Win/Saver Win bet — but only when the conditional
    # bet type logic assigned Win to rank 1. If rank 1 was shifted to Place
    # (Class/Open race, staying, or long odds), respect that decision.
    # Skip for PLACE_LEVERAGE — this classification can go all-Place.
    rank1_shifted_to_place = any(
        p.rank == 1 and not p.is_roughie and p.bet_type == "Place"
        for p in picks
    )
    if classification.race_type != "PLACE_LEVERAGE" and not rank1_shifted_to_place:
        _ensure_win_bet(picks)

    # Cap win-exposed bets to avoid spreading win risk across too many horses
    win_capped = _cap_win_exposure(picks)

    # Ultra-small fields (≤4): force ALL staked picks to Win — place divs too thin
    if field_size <= 4:
        for pick in picks:
            if pick.bet_type in ("Place", "Each Way") and not pick.tracked_only:
                pick.bet_type = "Win"
                pick.expected_return = round(pick.win_prob * pick.odds - 1, 2)

    # Concentrate stakes: only top MAX_STAKED_PICKS get real money.
    # Picks #3+ become tracked_only (shown for exotics/AI context, no stake).
    # This prevents thin-margin 4-way splits that historically lose.
    if not is_ntd:  # NTD already handles its own tracked_only logic
        for pick in picks[MAX_STAKED_PICKS:]:
            if not pick.tracked_only:
                pick.tracked_only = True
                pick.no_bet_reason = "Stake concentrated on top 3 picks"
                pick.stake = 0.0

    # Determine race pool using 3-tier confidence system
    race_pool = _determine_race_pool(picks, classification)

    # Allocate stakes with edge gating
    _allocate_stakes(picks, race_pool, field_size=field_size)

    # Log final pick summary
    for p in picks:
        status = "TRACKED" if p.tracked_only else f"${p.stake:.1f}"
        reason = f" [{p.no_bet_reason}]" if p.no_bet_reason else ""
        logger.info(
            f"R{race_number} pick #{p.rank}: {p.horse_name} (#{p.saddlecloth}) "
            f"{p.bet_type} {status} WP={p.win_prob:.1%} PP={p.place_prob:.1%} "
            f"odds=${p.odds:.2f}{reason}"
        )

    # Recalculate exotic combos using our picks + top 6 by probability.
    # Our 4 picks anchor positions 1-2 (Exacta/First4). Positions 2 (Exacta)
    # and 3-4 (First4) draw from the wider top-6 pool for better coverage.
    # Diagnosis showed 57% of exotic misses were "winner not in picks" —
    # the wider pool for trailing positions addresses this.
    if picks and len(picks) >= 2:
        from punty.probability import calculate_exotic_combinations
        pick_scs = {p.saddlecloth for p in picks}

        # Build runner data: our 4 picks + next 2 highest-probability runners
        # that aren't already in our picks (top 6 total)
        pick_runners_data = []
        for p in picks:
            market_implied = 0.0
            for r in runners:
                if r.get("saddlecloth") == p.saddlecloth:
                    odds = r.get("current_odds") or p.odds
                    if odds and odds > 0:
                        market_implied = 1.0 / odds
                    break
            pick_runners_data.append({
                "saddlecloth": p.saddlecloth,
                "horse_name": p.horse_name,
                "win_prob": p.win_prob,
                "market_implied": market_implied or (1.0 / p.odds if p.odds > 0 else 0),
                "value_rating": p.value_rating,
            })

        # Add next-best runners outside our picks for wider exotic coverage
        non_pick_runners = sorted(
            [r for r in runners
             if not r.get("scratched") and r.get("saddlecloth") not in pick_scs
             and r.get("_win_prob_raw", 0) > 0],
            key=lambda r: r.get("_win_prob_raw", 0),
            reverse=True,
        )[:2]  # Next 2 best runners by probability
        for r in non_pick_runners:
            odds = r.get("current_odds", 0)
            market_implied = 1.0 / odds if odds and odds > 0 else 0
            pick_runners_data.append({
                "saddlecloth": r.get("saddlecloth"),
                "horse_name": r.get("horse_name", ""),
                "win_prob": r.get("_win_prob_raw", 0),
                "market_implied": market_implied,
                "value_rating": r.get("punty_value_rating", 1.0),
            })

        try:
            pick_combos = calculate_exotic_combinations(pick_runners_data)
            if pick_combos:
                exotic_combos = [
                    {
                        "type": c.exotic_type,
                        "runners": c.runners,
                        "runner_names": c.runner_names,
                        "probability": f"{c.estimated_probability * 100:.1f}%"
                            if isinstance(c.estimated_probability, float) else str(c.estimated_probability),
                        "value": c.value_ratio,
                        "combos": c.num_combos,
                        "format": c.format,
                    }
                    for c in pick_combos
                ]
        except Exception:
            pass  # Fall back to builder's combos

    # Select recommended exotic
    # Anchor odds = best price among our rank 1 pick OR market favourite.
    # When tissue diverges heavily from market (e.g. tissue rank 1 is $12
    # but market fav is $1.65), using only tissue rank 1's odds would
    # incorrectly kill exotics via the anchor > $5 gate.
    active_odds = [r.get("current_odds", 0) for r in runners
                   if not r.get("scratched") and r.get("current_odds", 0) > 0]
    fav_price = min(active_odds) if active_odds else 0.0
    rank1_odds = picks[0].odds if picks else 0.0
    anchor_odds = min(rank1_odds, fav_price) if fav_price > 0 else rank1_odds
    exotic = _select_exotic(
        exotic_combos, used_saddlecloths,
        field_size=field_size, anchor_odds=anchor_odds,
        venue_type=venue_type, picks=picks,
        track_condition=race_context.get("track_condition", ""),
        race_class=race_context.get("class", ""),
        is_hk=(race_context.get("state") == "HK"),
        fav_price=fav_price,
        distance=race_context.get("distance", 0),
        prize_money=race_context.get("prize_money", 0),
        runners=race_context.get("runners", []),
    )

    # Guaranteed fallback: quinella on top 2 picks if no exotic was selected.
    # Ensures every race gets an exotic recommendation (required for auto-approval).
    if exotic is None and len(picks) >= 2:
        top2 = sorted(picks[:2], key=lambda p: p.rank)
        combined_prob = top2[0].win_prob * top2[1].win_prob * 2  # rough quinella prob
        logger.info(
            "Exotic fallback: Quinella %s for R%d (no combos passed routing)",
            [p.saddlecloth for p in top2], race_number,
        )
        exotic = RecommendedExotic(
            exotic_type="Quinella",
            runners=[p.saddlecloth for p in top2],
            runner_names=[p.horse_name for p in top2],
            probability=min(combined_prob, 0.99),
            value_ratio=1.0,
            num_combos=1,
            format="flat",
            budget=EXOTIC_BUDGET,
        )

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
        puntys_pick=None,
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

        # KASH consensus: boost confidence when independent model agrees
        kash_rp = r.get("kash_rated_price")
        kash_boost = 0.0
        if kash_rp and kash_rp > 0 and win_prob > 0:
            kash_wp = 1.0 / kash_rp
            divergence = abs(win_prob - kash_wp)
            if divergence < 0.08:
                kash_boost = 0.02  # Strong agreement — small but meaningful
            elif divergence > 0.15:
                kash_boost = -0.01  # Strong disagreement — slight dampen

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
            "confidence_boost": r.get("_confidence_boost", 0.0) + kash_boost,
            "kash_rated_price": kash_rp,
            "kash_speed_cat": r.get("kash_speed_cat"),
            "kash_early_speed": r.get("kash_early_speed"),
        })

    return candidates


def _determine_bet_type(
    c: dict, rank: int, is_roughie: bool, thresholds: dict | None = None,
    field_size: int = 12, fav_price: float | None = None,
    race_class: str = "", distance: int = 0,
) -> str:
    """Determine bet type using explicit business rules.

    The LightGBM meta-model was disabled because it NEVER recommends Win
    (trained on Proform data where Win ROI was negative). These rules
    implement the user's explicit bet type policy.

    Rules:
    - R1: Win (odds < $7, not Class/Open, not 2000m+)
    - R1: Each Way ($4-$10, good value)
    - R1: Place (long odds $10+ or Class/Open/staying)
    - R2: Place (always)
    - R3: Place (always, may be tracked_only by staking cap)
    - Roughie: tracked_only (handled by MAX_STAKED_PICKS)
    - ≤4 runners: Win for all (no place market)
    """
    # ≤4 runners: no place betting available
    if field_size <= 4:
        return "Win"

    odds = c.get("odds", 0) or 0
    wp = c.get("win_prob", 0) or 0
    rc = (race_class or "").lower()
    is_class_open = "class" in rc or "open" in rc or "group" in rc or "listed" in rc
    is_staying = distance >= 2000

    if is_roughie:
        return "Place"  # Will be tracked_only via MAX_STAKED_PICKS

    if rank == 1:
        # R1 bet type decision — our top pick always gets a bet
        # Under $3.00 → always Win, regardless of class/distance
        if 0 < odds < 3.0:
            return "Win"
        if odds <= 0 or odds > 10.0:
            # Long odds or no odds — Place is safer
            return "Place"
        if is_class_open or is_staying:
            # Class/Open and staying at mid odds — Each Way for protection
            if 4.0 <= odds <= 10.0:
                return "Each Way"
            return "Place"
        if odds < 7.0:
            # Mid odds, standard race — Win
            return "Win"
        # $7-$10 — Each Way
        return "Each Way"

    if rank == 2:
        return "Place"

    if rank == 3:
        return "Place"

    return "Place"


def _make_pick(c: dict, rank: int, is_roughie: bool, thresholds: dict | None = None, field_size: int = 12, fav_price: float | None = None, race_class: str = "", distance: int = 0) -> RecommendedPick:
    """Create a RecommendedPick from candidate data (legacy path)."""
    bet_type = _determine_bet_type(c, rank, is_roughie, thresholds, field_size=field_size, fav_price=fav_price, race_class=race_class, distance=distance)
    tracked_only = False
    no_bet_reason = None
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
        tracked_only=tracked_only,
        no_bet_reason=no_bet_reason,
    )


def _make_pick_from_optimizer(
    c: dict,
    rank: int,
    is_roughie: bool,
    rec_lookup: dict,
    thresholds: dict | None = None,
    field_size: int = 12,
    fav_price: float | None = None,
    race_class: str = "",
    distance: int = 0,
) -> RecommendedPick:
    """Create a RecommendedPick from optimizer context.

    Note: rec_lookup (optimizer bet recommendations) is not used — bet type
    is determined by the bettype meta-model via _determine_bet_type() which
    supersedes the optimizer's per-runner recommendations. The optimizer's
    race classification (DOMINANT_EDGE etc.) is used upstream for pool sizing.
    """
    bet_type = _determine_bet_type(
        c, rank, is_roughie, thresholds, field_size=field_size,
        fav_price=fav_price, race_class=race_class, distance=distance,
    )
    tracked_only = False
    no_bet_reason = None

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
        tracked_only=tracked_only,
        no_bet_reason=no_bet_reason,
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


def _estimate_place_odds(win_odds: float, field_size: int = 12) -> float:
    """Estimate place odds when not provided.

    Uses /3 divisor for 3-place fields (8+ runners) and /2 for NTD fields (5-7).
    """
    if win_odds <= 1.0:
        return 1.0
    divisor = 2 if 5 <= field_size <= 7 else 3
    return round((win_odds - 1) / divisor + 1, 2)


def _ensure_win_bet(picks: list[RecommendedPick]) -> None:
    """Ensure at least one pick is Win or Saver Win if viable.

    Only force Win on picks within the proven-profitable $2-$10 range.
    Below $2.00 is historically -38.9% ROI on Win — leave as Place.
    """
    has_win = any(p.bet_type in ("Win", "Saver Win", "Each Way") for p in picks)
    if has_win or not picks:
        return

    # Find best Win candidate: within $2-$10 odds range
    for pick in picks:
        if 2.0 <= pick.odds <= 10.0:
            pick.bet_type = "Win"
            pick.expected_return = round(
                pick.win_prob * pick.odds - 1, 2,
            )
            return

    # No viable Win candidate — all too short or too long. Stay all-Place.


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
    return POOL_LOW


def _passes_edge_gate(pick: RecommendedPick) -> tuple[bool, str | None]:
    """Check if a pick passes the edge gate (probability-based criteria).

    Returns (True, None) if the pick should be staked,
    or (False, reason) with a human-readable explanation if tracked-only.
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

    # Live ROI gate REMOVED — probability engine already prices edge.
    # Historical ROI by odds band is noisy and backwards-looking; it was
    # blanket-killing all Rank 1 Win bets (e.g. entire Matamata card).
    # Strike rate via calibrated probability is the right filter.

    # --- High-probability universal override ---
    # Any Place bet with place_prob >= 0.75 passes regardless of odds band.
    # Catches short-priced favourites (e.g. $1.35-$2.60) with 0.80+ place_prob
    # that were incorrectly getting no_bet under stricter thresholds.
    # Only applies to Place — Win at short odds is still blocked below.
    if bt == "Place" and place_prob >= 0.75:
        return (True, None)

    # --- Win staking criteria ---
    # Rank 1 always gets Win — Punty's top tip backs the best horse to win.
    # Minimum win_prob threshold prevents backing no-hopers.
    if bt in ("Win", "Saver Win"):
        if win_prob >= 0.15:
            return (True, None)
        return (False, f"Win prob too low ({win_prob * 100:.0f}% < 15%)")

    # 2. Place with graduated prob threshold by odds band
    if bt == "Place":
        # High collection confidence auto-pass (pp >= 0.70)
        if place_prob >= 0.70:
            return (True, None)
        # Strong win probability override: if wp >= 0.30, the runner is a genuine
        # contender — place prob may be underestimated by model. Production data
        # showed runners with wp=0.34 at $3.50 getting no_bet despite winning.
        if win_prob >= 0.30 and 2.0 <= odds <= 8.0:
            return (True, None)
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



def _allocate_stakes(picks: list[RecommendedPick], pool: float, field_size: int = 0) -> None:
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
    for pick in picks:
        if pick.tracked_only:
            continue  # already marked (e.g. NTD path) — preserve original reason
        # R1 always gets a bet — never edge-gate our top pick
        if pick.rank == 1:
            logger.info(
                f"  edge gate PASS (R1 override): {pick.horse_name} (#{pick.saddlecloth}) "
                f"rank=1 {pick.bet_type} ${pick.odds:.2f}"
            )
            continue
        passed, reason = _passes_edge_gate(pick)
        if not passed:
            pick.tracked_only = True
            pick.no_bet_reason = reason or "No proven edge in this odds band"
            pick.stake = 0.0
            logger.info(
                f"  edge gate FAIL: {pick.horse_name} (#{pick.saddlecloth}) "
                f"rank={pick.rank} {pick.bet_type} ${pick.odds:.2f} — {pick.no_bet_reason}"
            )
        # Place odds cap — rank-dependent (check place_odds, not win odds)
        # Rank 1: $10 cap (strong anchor). Rank 2+: $8 cap (yesterday data: +$33.50)
        elif pick.bet_type == "Place" and (pick.place_odds or pick.odds):
            place_cap = 10.0 if pick.rank == 1 else 8.0
            check_odds = pick.place_odds or pick.odds
            if check_odds > place_cap:
                pick.tracked_only = True
                pick.no_bet_reason = f"Place odds > ${place_cap:.0f} (rank {pick.rank})"
                pick.stake = 0.0
                logger.info(
                    f"  edge gate FAIL: {pick.horse_name} (#{pick.saddlecloth}) "
                    f"rank={pick.rank} — {pick.no_bet_reason}"
                )
            else:
                logger.info(
                    f"  edge gate PASS: {pick.horse_name} (#{pick.saddlecloth}) "
                    f"rank={pick.rank} {pick.bet_type} ${pick.odds:.2f}"
                )
        else:
            logger.info(
                f"  edge gate PASS: {pick.horse_name} (#{pick.saddlecloth}) "
                f"rank={pick.rank} {pick.bet_type} ${pick.odds:.2f}"
            )

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

    # Base allocation weights by rank — top 2 picks get most capital
    # With MAX_STAKED_PICKS=2, only ranks 1-2 normally get stakes
    base_rank_weights = {1: 0.58, 2: 0.42, 3: 0.22, 4: 0.15}

    pick_weights: list[float] = []
    for pick in staked_picks:
        base = base_rank_weights.get(pick.rank, 0.15)

        # Win bonus — rank 1 always gets Win, slight boost for conviction
        if pick.bet_type in ("Win", "Saver Win") and pick.win_prob and pick.win_prob >= 0.20:
            base *= 1.10  # 10% stake boost for high-conviction Win

        # Saver Win: reduced stake — 60% of full Win allocation
        if pick.bet_type == "Saver Win":
            base *= 0.60

        # Roughie $10-$20 bonus (+53% ROI sweet spot)
        if pick.is_roughie and 10.0 <= pick.odds <= 20.0:
            base *= 1.20  # 20% stake boost

        # Positive expected return bonus — tilt more to profitable bets
        if pick.expected_return > 0.10:
            base *= 1.15  # 15% bonus for strong +EV

        # High-confidence Place bonus — Place is our profit engine (+7.55% ROI)
        if pick.bet_type == "Place" and pick.place_prob >= 0.45:
            base *= 1.15  # 15% stake boost for high-confidence Place

        # VR-based stake reduction — high VR = overvalued by market
        # Data: VR 1.5-2.0: -28% ROI. VR 2.0+: -34%.
        if pick.value_rating and pick.value_rating >= 2.0:
            base *= 0.40  # 60% reduction
        elif pick.value_rating and pick.value_rating >= 1.5:
            base *= 0.60  # 40% reduction

        # 9-10 runner defensive stake — Data: -$807, -20% ROI
        if 9 <= field_size <= 10:
            base *= 0.80  # 20% reduction

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


def _midweek_multiplier() -> float:
    """Return 0.5 on Mon/Tue/Thu (thin cards, worse exotic ROI), 1.0 otherwise."""
    from punty.config import melb_today
    dow = melb_today().weekday()  # 0=Mon, 1=Tue, ..., 6=Sun
    if dow in (0, 1, 3):  # Mon, Tue, Thu
        return 0.5
    return 1.0


def _get_exotic_budget(meet_quality: float) -> float:
    """Scale exotic budget by meet quality and day of week.

    Halved from original to reduce bleed — exotics are -9.4% ROI on $27K
    staked. Lower stakes let the model learn while limiting damage.

    Country meets (quality 0): $5 — thin pools, smaller dividends
    Provincial meets (quality 1): $8 — standard
    Metro meets (quality >= 2): $10 — deeper pools, bigger dividends

    Mon/Tue/Thu: halved again (thin cards, worse exotic ROI historically).
    """
    if meet_quality >= 2:
        base = 10.0
    elif meet_quality >= 1:
        base = 8.0
    else:
        base = 5.0
    return base * _midweek_multiplier()


# ── V3 Context-Aware Exotic Cascade ──
# Backtested on v3 model (78.2% place SR): +302% ROI on Exacta R1/R2,R3.
# Config maps distance×condition → optimal exotic type.
# v6 cascade config: used for MID-FIELD (9-11 runners) routing only.
# Small fields (≤8) always go F4 Standout. Big fields (12+) always Quin Box.
# This config determines the exotic type for mid-field races by context.
# Based on actual R1-R4 finish positions from our settled picks.
_EXOTIC_CASCADE_CONFIG = {
    # Sprint mid-field: Tri strong (15% maiden, 15% benchmark)
    "sprint|good": "tri",            # Tri 14.8% maiden, 6.7% benchmark
    "sprint|soft": "tri",            # Tri 15% benchmark
    "sprint|heavy": "f4_standout",   # F4 17.6% (heavy = our strength)
    # Short mid-field: Exacta dominant
    "short|good": "exacta",         # Exacta 19.2% other, 17.6% handicap
    "short|soft": "exacta",         # Exacta 20% maiden, 21.4% benchmark
    "short|heavy": "f4_standout",   # F4 16% (heavy)
    # Middle mid-field: mixed — Exacta for soft, Tri for good
    "middle|good": "tri",           # Tri 8% benchmark, 7.4% overall
    "middle|soft": "exacta",        # Exacta 28.6% other, 15% benchmark
    "middle|heavy": "f4_standout",  # F4 10.7% (heavy dominant)
    # Classic: Tri strong at 21% benchmark
    "classic|good": "tri",          # Tri 21.4% benchmark
    "classic|soft": "exacta",       # Exacta 19.6%
    "classic|heavy": "quin_box",    # tiny sample
    # Staying
    "staying|good": "exacta",       # Exacta 7.7%
    "staying|soft": "f4_standout",  # F4 22.2%
    "staying|heavy": "quin_box",    # tiny sample
}


def _v3_exotic_cascade(
    distance: int = 0,
    track_condition: str = "",
    picks: list | None = None,
    field_size: int = 0,
) -> RecommendedExotic | None:
    """V3 context-aware exotic routing.

    Builds the exotic directly from our ranked picks. No AI involvement.
    - Exacta: R1 over R2,R3 (2 combos)
    - Tri structured: R1,R2 / R2,R3 / R3,R4 (8 combos)
    - Quinella: R1 with R2 (1 combo)
    - Quinella Box: R1,R2,R3 (3 combos)

    R1 is ALWAYS anchored. Roughie NEVER in position 1.
    """
    from punty.calibration_engine import _distance_bucket, _condition_bucket

    if not picks or len(picks) < 2:
        return None

    sorted_picks = sorted(picks, key=lambda p: p.rank)
    r1 = sorted_picks[0]
    r2 = sorted_picks[1] if len(sorted_picks) > 1 else None
    r3 = sorted_picks[2] if len(sorted_picks) > 2 else None
    r4 = sorted_picks[3] if len(sorted_picks) > 3 else None

    if not r1 or not r2:
        return None

    dist_b = _distance_bucket(distance or 1400)
    cond_b = _condition_bucket(track_condition or "Good 4")
    ctx_key = f"{dist_b}|{cond_b}"

    # CASCADE: F4 → Tri → Exacta → Quin → Quin Box
    # Field size is the primary router (data shows small fields = F4 goldmine):
    #   Small (≤8): F4 Standout (20-43% hit rate in our data)
    #   Mid (9-11): Tri/Exacta based on context
    #   Big (12+): Quin Box (F4/Tri near 0% in big fields)
    if field_size <= 0:
        field_size = 10  # assume mid if unknown

    if field_size < 4:
        return None  # Too few runners

    if field_size <= 8 and r4 and r3:
        # Small fields: F4 Standout is dominant (our picks fill top 4 at 20-43%)
        exotic_type = "f4_standout"
    elif field_size <= 11 and r3:
        # Mid fields: use context cascade config
        exotic_type = _EXOTIC_CASCADE_CONFIG.get(ctx_key, "tri")
        # Ensure minimum runners for chosen type
        if exotic_type == "f4_standout" and not r4:
            exotic_type = "tri"
    else:
        # Big fields (12+): our picks spread thin, default to Quin Box
        exotic_type = "quin_box"

    # Final minimum field size guards
    if exotic_type == "f4_standout" and (field_size < 5 or not r4):
        exotic_type = "tri"
    if exotic_type == "tri" and (field_size < 5 or not r3):
        exotic_type = "exacta"
    if exotic_type == "exacta" and field_size < 4:
        exotic_type = "quin_box"
    if exotic_type in ("quin", "quin_box") and field_size < 4:
        return None

    BUDGET = 15.0

    if exotic_type == "f4_standout":
        # F4 Standout structured legs:
        # 1st: R1 (anchor must win)
        # 2nd: R2, R3, R4 (3 options)
        # 3rd: R2, R3, R4, R5 (4 options)
        # 4th: R2, R3, R4, R5, R6 (5 options — widest coverage)
        # Combos: 1 × 3 × 4 × 5 = 60
        r5 = sorted_picks[4] if len(sorted_picks) > 4 else None
        r6 = sorted_picks[5] if len(sorted_picks) > 5 else None

        sc_to_name = {p.saddlecloth: p.horse_name for p in sorted_picks}
        all_scs = [r1.saddlecloth, r2.saddlecloth, r3.saddlecloth, r4.saddlecloth]
        if r5:
            all_scs.append(r5.saddlecloth)
        if r6:
            all_scs.append(r6.saddlecloth)
        all_scs = list(set(all_scs))
        names = [sc_to_name.get(sc, f"#{sc}") for sc in all_scs]

        # Build structured legs
        leg1_scs = [r1.saddlecloth]
        leg2_scs = [r2.saddlecloth, r3.saddlecloth, r4.saddlecloth]
        leg3_scs = [r2.saddlecloth, r3.saddlecloth, r4.saddlecloth]
        if r5:
            leg3_scs.append(r5.saddlecloth)
        leg4_scs = [r2.saddlecloth, r3.saddlecloth, r4.saddlecloth]
        if r5:
            leg4_scs.append(r5.saddlecloth)
        if r6:
            leg4_scs.append(r6.saddlecloth)

        combos = len(leg1_scs) * len(leg2_scs) * len(leg3_scs) * len(leg4_scs)

        return RecommendedExotic(
            exotic_type="First4",
            runners=all_scs,
            runner_names=names,
            probability=r1.win_prob * r2.win_prob * r3.win_prob * (r4.win_prob if r4 else 0.1),
            value_ratio=3.0,
            num_combos=combos,
            format="flat",
            budget=BUDGET,
            legs=[leg1_scs, leg2_scs, leg3_scs, leg4_scs],
        )

    elif exotic_type == "exacta":
        # R1 over R2,R3 — 2 combos
        firsts = [r1.saddlecloth]
        seconds = [r2.saddlecloth]
        if r3:
            seconds.append(r3.saddlecloth)
        runner_scs = list(set(firsts + seconds))
        runner_names = []
        sc_to_name = {p.saddlecloth: p.horse_name for p in sorted_picks}
        for sc in runner_scs:
            runner_names.append(sc_to_name.get(sc, f"#{sc}"))
        combos = len(seconds)  # R1 over each of the seconds
        return RecommendedExotic(
            exotic_type="Exacta",
            runners=runner_scs,
            runner_names=runner_names,
            probability=r1.win_prob * (r2.win_prob + (r3.win_prob if r3 else 0)),
            value_ratio=1.5,
            num_combos=combos,
            format="flat",
            budget=BUDGET,
            legs=[firsts, seconds],
        )

    elif exotic_type == "tri":
        # Trifecta structured: R1,R2 / R2,R3 / R3,R4 — 8 combos
        if not r3:
            # Fall back to exacta
            return _v3_exotic_cascade(distance, track_condition, picks, field_size)
        firsts = [r1.saddlecloth, r2.saddlecloth]
        seconds = [r2.saddlecloth, r3.saddlecloth]
        thirds = [r3.saddlecloth]
        if r4:
            thirds.append(r4.saddlecloth)
        all_scs = list(set(firsts + seconds + thirds))
        sc_to_name = {p.saddlecloth: p.horse_name for p in sorted_picks}
        names = [sc_to_name.get(sc, f"#{sc}") for sc in all_scs]
        combos = len(firsts) * len(seconds) * len(thirds)
        return RecommendedExotic(
            exotic_type="Trifecta Standout",
            runners=all_scs,
            runner_names=names,
            probability=r1.win_prob * r2.win_prob * (r3.win_prob if r3 else 0.1),
            value_ratio=2.0,
            num_combos=combos,
            format="flat",
            budget=BUDGET,
            legs=[firsts, seconds, thirds],
        )

    elif exotic_type == "quin":
        # Quinella: R1 with R2 — 1 combo, full $15
        return RecommendedExotic(
            exotic_type="Quinella",
            runners=[r1.saddlecloth, r2.saddlecloth],
            runner_names=[r1.horse_name, r2.horse_name],
            probability=r1.win_prob + r2.win_prob,
            value_ratio=1.5,
            num_combos=1,
            format="flat",
            budget=BUDGET,
        )

    else:  # quin_box
        # Quinella Box: R1, R2, R3 — 3 combos
        scs = [r1.saddlecloth, r2.saddlecloth]
        names = [r1.horse_name, r2.horse_name]
        if r3:
            scs.append(r3.saddlecloth)
            names.append(r3.horse_name)
        combos = len(scs) * (len(scs) - 1) // 2
        return RecommendedExotic(
            exotic_type="Quinella Box",
            runners=scs,
            runner_names=names,
            probability=(r1.win_prob + r2.win_prob + (r3.win_prob if r3 else 0)) * 0.5,
            value_ratio=1.2,
            num_combos=combos,
            format="boxed",
            budget=BUDGET,
        )


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
    prize_money: int | float = 0,
    runners: list[dict] | None = None,
) -> RecommendedExotic | None:
    """Select the best exotic bet for this race.

    Uses a trained LightGBM meta-model when available (context-aware learned
    selector). Falls back to hand-tuned race-shape + meet-quality routing.

    Budget: dynamic ($10 country / $15 provincial / $20 metro).
    """
    if not exotic_combos:
        return None

    # ── V3 Context-Aware Cascade: bypass complex routing when we know
    #    the optimal exotic type from backtested data (78.2% place SR model).
    #    Cascade config maps dist×cond → best exotic type.
    #    Exacta R1/R2,R3 dominates in 8/12 contexts (+302% ROI overall).
    #    NO box trifecta or first4 box — structured only. ──
    try:
        _cascade_result = _v3_exotic_cascade(
            distance=distance, track_condition=track_condition,
            picks=picks, field_size=field_size,
        )
        if _cascade_result:
            return _cascade_result
    except Exception as _e:
        logger.debug("V3 cascade fallback: %s", _e)

    # ── Legacy routing (fallback if cascade doesn't apply) ──

    # ── Try to load meta-model for scoring (used after filtering below) ──
    _use_meta_model = False
    _meta_sc_to_rank: dict[int, int] = {}
    _meta_sc_to_wp: dict[int, float] = {}
    _meta_rank_wps: list[float] = []
    _meta_rank_odds: list[float] = []
    try:
        from punty.betting.exotic_model import exotic_model_available, predict_exotic_hit_probability, extract_exotic_features
        from punty.ml.features import _distance_bucket, _track_cond_bucket, _class_bucket, _venue_type_code

        if exotic_model_available() and picks:
            _use_meta_model = True
            sorted_picks = sorted(picks, key=lambda p: p.rank)
            _meta_rank_wps = [p.win_prob for p in sorted_picks]
            _meta_rank_odds = [p.odds for p in sorted_picks]
            _meta_sc_to_rank = {p.saddlecloth: p.rank for p in sorted_picks}
            _meta_sc_to_wp = {p.saddlecloth: p.win_prob for p in sorted_picks}
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Exotic meta-model unavailable: %s", e)

    # ── Hard gates ──
    tc = (track_condition or "").lower()

    # ── Anchor: our rank 1 pick must be in every exotic ──
    fav_saddlecloth = None
    if picks:
        rank1 = min(picks, key=lambda p: p.rank)
        fav_saddlecloth = rank1.saddlecloth

    # ── Classify race shape by win-probability spread ──
    top_wps = sorted([p.win_prob for p in picks if not p.is_roughie], reverse=True)[:3] if picks else []
    if len(top_wps) >= 3:
        wp_spread = top_wps[0] - top_wps[2]
    elif len(top_wps) >= 2:
        wp_spread = top_wps[0] - top_wps[1]
    else:
        wp_spread = 0.10

    if wp_spread > 0.12:
        race_shape = "dominant"
    elif wp_spread >= 0.05:
        race_shape = "structured"
    else:
        race_shape = "open"

    # ── KASH speed refinement ──
    # KASH speed_cat (Leader/On-Pace/Midfield/Backmarker) and early_speed
    # can upgrade/downgrade race shape when pace analysis diverges from
    # probability spread. Fast leader in a slow race = more dominant.
    if runners:
        kash_speeds = []
        for r in runners:
            sc = (r.get("kash_speed_cat") or "").lower()
            es = r.get("kash_early_speed")
            if sc and not r.get("scratched"):
                kash_speeds.append({"sc": sc, "es": es or 0})
        if kash_speeds:
            leaders = [s for s in kash_speeds if s["sc"] in ("leader", "on pace")]
            # If <25% of field are leaders/on-pace, pace is thin — upgrade shape
            # (standout frontrunner advantage = more dominant outcome)
            if leaders and len(leaders) / len(kash_speeds) < 0.25:
                if race_shape == "structured":
                    race_shape = "dominant"  # Thin pace → frontrunner advantage
            # If >50% are leaders/on-pace, pace is hot — downgrade shape
            # (hot pace = form reversals, less predictable)
            elif leaders and len(leaders) / len(kash_speeds) > 0.50:
                if race_shape == "dominant":
                    race_shape = "structured"  # Hot pace erodes standout edge

    # ── Meet quality score (0-3) ──
    # Drives exotic ambition: higher quality = more ambitious bet types
    meet_quality = 0
    vt = (venue_type or "").lower()
    if "metro" in vt:
        meet_quality += 2
    elif "provincial" in vt:
        meet_quality += 1
    # else country = 0

    if field_size and field_size >= 12:
        meet_quality += 1  # Big field → bigger dividends worth chasing

    pm = float(prize_money or 0)
    if pm >= 100_000:
        meet_quality += 1  # High-stakes race
    elif pm >= 50_000:
        meet_quality += 0.5

    # ── Dynamic budget by meet quality ──
    budget = _get_exotic_budget(meet_quality)

    # ── Minimum field sizes by type, relative to meet quality ──
    # Higher quality meets tolerate smaller fields (better horses, deeper pools)
    # Country needs bigger fields for meaningful dividends
    if meet_quality >= 2:
        min_field = {"Quinella": 4, "Quinella Box": 4, "Exacta": 5, "Exacta Standout": 5,
                     "Trifecta Standout": 7, "Trifecta Box": 7,
                     "First4": 8, "First4 Box": 9}
    elif meet_quality >= 1:
        min_field = {"Quinella": 5, "Quinella Box": 5, "Exacta": 6, "Exacta Standout": 6,
                     "Trifecta Standout": 8, "Trifecta Box": 8,
                     "First4": 9, "First4 Box": 10}
    else:
        min_field = {"Quinella": 6, "Quinella Box": 6, "Exacta": 7, "Exacta Standout": 7,
                     "Trifecta Standout": 9, "Trifecta Box": 9,
                     "First4": 10, "First4 Box": 11}

    # ── Build rank map ──
    rank_map: dict[int, int] = {}
    if picks:
        for p in picks:
            rank_map[p.saddlecloth] = p.rank

    # Soft track penalty — Data: Good/Firm +21% ROI, any Soft -37% ROI
    soft_penalty = 0.65 if "soft" in tc else 1.0

    # ── Field-size routing: only allow types in their profitable zones ──
    # Data-backed from 1,797 settled exotics. Each type has field-size
    # sweet spots where ROI is positive and dead zones where it hemorrhages.
    # The meta-model still picks the best option, but from a pre-filtered
    # set that excludes proven losing segments.
    #
    # Field 5-7:   Quinella Box -2%, Quinella -8% (predictable small fields)
    # Field 8-9:   Exacta Standout +59%, Quinella Box -5% (balanced)
    # Field 10-11: Exacta Standout +127%, Quinella -3% (standout picks shine)
    # Field 12-13: Quinella -7%, Trifecta Box -7% (near break-even only)
    # Field 14+:   Exacta +364%, Quinella +6% (big dividends)
    top_pick_wp = max((p.win_prob for p in picks), default=0) if picks else 0
    roughie_wp = min((p.win_prob for p in picks), default=0) if picks else 0
    roughie_is_live = roughie_wp >= 0.08

    # Build allowed types from field-size profitability data
    # All 8 exotic types — used when field_size unknown (0)
    ALL_EXOTIC_TYPES = {
        "Quinella", "Quinella Box", "Exacta", "Exacta Standout",
        "Trifecta Standout", "Trifecta Box", "First4", "First4 Box",
    }
    allowed_types: set[str] = set()

    if field_size == 0:
        # Unknown field size — allow all types, let model decide
        allowed_types = ALL_EXOTIC_TYPES
    elif field_size <= 7:
        # Small fields: our picks cover high % of field
        allowed_types = {"Quinella", "Quinella Box"}
        if top_pick_wp >= 0.20:
            allowed_types.add("Trifecta Standout")  # +354% in small fields
            allowed_types.add("Exacta Standout")
        if roughie_is_live:
            allowed_types.add("First4 Box")  # +102% in small fields
    elif field_size <= 9:
        # Mid-small: Exacta Standout sweet spot (+59%)
        allowed_types = {"Exacta Standout", "Quinella Box", "Quinella"}
        if top_pick_wp >= 0.20:
            allowed_types.add("Trifecta Standout")  # +2% break-even
        if top_pick_wp >= 0.25:
            allowed_types.add("Exacta")
    elif field_size <= 11:
        # Mid-large: Exacta Standout dominant (+127%)
        allowed_types = {"Exacta Standout", "Quinella", "Quinella Box"}
        if top_pick_wp >= 0.25:
            allowed_types.add("Exacta")
    elif field_size <= 13:
        # Large: most types struggle, Quinella/Tri Box near break-even
        allowed_types = {"Quinella", "Quinella Box"}
        if roughie_is_live and meet_quality >= 1:
            allowed_types.add("Trifecta Box")  # -7% near break-even
        if top_pick_wp >= 0.25:
            allowed_types.add("Exacta")
    else:
        # 14+: Exacta dominates (+364%), Quinella profitable (+6%)
        allowed_types = {"Exacta", "Quinella"}
        if top_pick_wp >= 0.25:
            allowed_types.add("Exacta Standout")

    # Race shape still nudges preferred types within the allowed set
    if race_shape == "dominant":
        preferred_types = allowed_types & {"Exacta Standout", "Exacta", "Quinella", "Trifecta Standout"}
    elif race_shape == "open":
        preferred_types = allowed_types & {"Quinella", "Quinella Box", "Trifecta Box", "First4 Box"}
    else:  # structured
        preferred_types = allowed_types & {"Quinella", "Quinella Box", "Exacta Standout", "Trifecta Standout"}

    # Fallback: if shape intersection is empty, use all allowed types
    if not preferred_types:
        preferred_types = allowed_types

    # ── Score and filter combos ──
    scored = []
    for ec in exotic_combos:
        runners = set(ec.get("runners", []))
        n_runners = len(runners)
        overlap = len(runners & selection_saddlecloths)
        overlap_ratio = overlap / n_runners if n_runners else 0
        ec_type = ec.get("type", "")

        # Field-size routing: skip types outside their profitable zone
        if ec_type and ec_type not in allowed_types:
            continue

        # Anchor enforcement
        if fav_saddlecloth and fav_saddlecloth not in runners:
            continue

        # ALL exotic runners must come from our selections (top 3 + roughie)
        runner_list = ec.get("runners", [])
        if not all(r in selection_saddlecloths for r in runner_list):
            continue

        # Parse probability and value
        raw_prob = ec.get("probability", 0)
        if isinstance(raw_prob, str):
            raw_prob = float(raw_prob.rstrip("%")) / 100
        value = ec.get("value", 1.0)
        combos = max(1, ec.get("combos", 1))

        # Budget feasibility: flexi must be >= 10% to be meaningful
        flexi_pct = budget / combos * 100 if combos > 0 else 0
        if flexi_pct < 10:
            continue

        # Trifecta minimum value ratio: 1.5x (higher bar than other exotics)
        # Trifectas are harder to hit, so only play when value clearly justifies it
        if ec_type in ("Trifecta Standout", "Trifecta Box") and value < 1.5:
            continue

        # Exacta certainty gate: lead runner must have >= 25% WP.
        # Exactas = "we know the winner, options for 2nd". Without a genuine
        # standout anchoring 1st, quinella boxes are the better structure.
        if ec_type in ("Exacta", "Exacta Standout") and picks and top_pick_wp < 0.25:
            continue

        # ── Scoring: meta-model or hand-tuned ──
        if _use_meta_model:
            # Meta-model: learned score from race context + combo composition
            combo_ranks = [_meta_sc_to_rank.get(r, 4) for r in runner_list]
            combo_wps_list = [_meta_sc_to_wp.get(r, 0.05) for r in runner_list]
            features = extract_exotic_features(
                field_size=field_size,
                distance_bucket=_distance_bucket(distance),
                class_bucket=_class_bucket(race_class),
                track_cond_bucket=_track_cond_bucket(track_condition),
                venue_type=_venue_type_code(venue_type),
                prize_money=float(prize_money or 0),
                rank_wps=_meta_rank_wps,
                rank_odds=_meta_rank_odds,
                exotic_type=ec_type,
                num_combo_runners=n_runners,
                num_combos=combos,
                combo_runner_ranks=combo_ranks,
                combo_runner_wps=combo_wps_list,
            )
            hit_prob = predict_exotic_hit_probability(features)
            if hit_prob < 0:
                # Model prediction failed — fall back to hand-tuned
                capped_value = min(value, 2.5)
                score = raw_prob * (capped_value ** 2)
            else:
                # Score = P(hit)² × √value — probability dominates, value is a
                # mild tiebreaker. This prevents high-value roughie combos from
                # beating higher-probability plays.
                capped_value = min(value, 2.5)
                score = (hit_prob ** 2) * (capped_value ** 0.5)
        else:
            # Hand-tuned scoring fallback
            capped_value = min(value, 2.5)
            score = raw_prob * (capped_value ** 2)

            # Preferred type bonus — gentle nudge, not hard exclusion
            if ec_type in preferred_types:
                score *= 1.15
            else:
                score *= 0.90

            # Wide box penalty — 4-runner Trifecta Box and 5-runner First4 Box
            # need a genuine roughie to justify the combo explosion
            if ec_type == "Trifecta Box" and n_runners >= 4 and not roughie_is_live:
                score *= 0.50
            if ec_type == "First4 Box" and n_runners >= 5 and not roughie_is_live:
                score *= 0.40

            # Anchor rank bonus
            anchor_bonus = 1.0
            if rank_map:
                best_rank = min(rank_map.get(r, 99) for r in runners)
                if best_rank == 1:
                    anchor_bonus *= 1.20
                elif best_rank == 2:
                    anchor_bonus *= 1.10
                if all(r in selection_saddlecloths for r in runners):
                    anchor_bonus *= 1.10
            score *= min(anchor_bonus, 1.40)

            # Field-size scaling
            if field_size:
                if ec_type in ("Quinella", "Quinella Box"):
                    score *= max(0.90, min(1.10, 1.18 - field_size * 0.02))
                elif ec_type in ("Exacta", "Exacta Standout"):
                    score *= max(0.95, min(1.10, 0.85 + field_size * 0.02))
                elif ec_type in ("Trifecta Standout", "Trifecta Box"):
                    score *= max(0.80, min(1.15, 0.55 + field_size * 0.05))
                elif ec_type in ("First4", "First4 Box"):
                    score *= max(0.70, min(1.20, 0.40 + field_size * 0.06))

        # Soft track penalty (applied to both paths)
        score *= soft_penalty

        scored.append((score, ec))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        # Fallback: pick the highest-probability combo from the original list
        # so every race gets an exotic recommendation
        fallback = sorted(
            exotic_combos,
            key=lambda e: float(str(e.get("probability", 0)).rstrip("%")) / (100 if "%" in str(e.get("probability", 0)) else 1),
            reverse=True,
        )
        if not fallback:
            return None
        best = fallback[0]
        fb_prob = best.get("probability", 0)
        if isinstance(fb_prob, str):
            fb_prob = float(fb_prob.rstrip("%")) / 100
        return RecommendedExotic(
            exotic_type=best.get("type", ""),
            runners=best.get("runners", []),
            runner_names=best.get("runner_names", []),
            probability=fb_prob,
            value_ratio=best.get("value", 1.0),
            num_combos=best.get("combos", 1),
            format=best.get("format", "boxed"),
            budget=budget,
        )

    best_score, best = scored[0]
    best_prob = best.get("probability", 0)
    if isinstance(best_prob, str):
        best_prob = float(best_prob.rstrip("%")) / 100

    scoring_method = "meta-model" if _use_meta_model else "hand-tuned"
    logger.info(
        "Exotic selected (%s): %s %s (score=%.4f, value=%.2fx) for field=%d",
        scoring_method, best.get("type", "?"), best.get("runners", []),
        best_score, best.get("value", 1.0), field_size,
    )

    return RecommendedExotic(
        exotic_type=best.get("type", ""),
        runners=best.get("runners", []),
        runner_names=best.get("runner_names", []),
        probability=best_prob,
        value_ratio=best.get("value", 1.0),
        num_combos=best.get("combos", 1),
        format=best.get("format", "boxed"),
        budget=budget,
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
        prob_label = f"Win: {pick.win_prob * 100:.1f}% | Place: {pick.place_prob * 100:.1f}%"
        value_label = f"{pick.value_rating:.2f}x"

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
                half = pick.stake / 2
                ret = round(half * pick.odds + half * po, 2)
            else:
                ret = 0

            # Build bet description with correct return for bet type
            if pick.bet_type == "Each Way":
                half = pick.stake / 2
                po = pick.place_odds or _estimate_place_odds(pick.odds)
                bet_desc = (
                    f"${pick.stake:.2f} Each Way "
                    f"(${half:.2f}W + ${half:.2f}P), "
                    f"return ${round(half * pick.odds, 2):.2f} (wins) / "
                    f"${round(half * po, 2):.2f} (places)"
                )
            elif pick.bet_type == "Place":
                po = pick.place_odds or _estimate_place_odds(pick.odds)
                bet_desc = f"${pick.stake:.2f} Place, return ${ret:.2f}"
            else:
                bet_desc = f"${pick.stake:.2f} {pick.bet_type}, return ${ret:.2f}"

            lines.append(f"    BET: {bet_desc}")

    if pre_sel.exotic:
        ex = pre_sel.exotic
        if ex.legs:
            runners_str = " / ".join(
                ", ".join(str(r) for r in leg) for leg in ex.legs
            )
        else:
            runners_str = ", ".join(str(r) for r in ex.runners)
        names_str = ", ".join(ex.runner_names)
        budget = ex.budget
        combos = max(1, ex.num_combos)
        unit_cost = round(budget / combos, 2)
        flexi_pct = round(budget / combos * 100, 1) if combos > 1 else 100.0
        if combos == 1:
            stake_str = f"${budget:.0f}"
        else:
            stake_str = f"${budget:.0f} ({flexi_pct:.0f}% x {combos} combos = ${unit_cost:.2f}/combo)"
        lines.append(
            f"  Exotic: {ex.exotic_type} [{runners_str}] {names_str} "
            f"— {stake_str} | Prob: {ex.probability * 100:.1f}% "
            f"| Value: {ex.value_ratio:.2f}x"
        )
    else:
        lines.append("  Exotic: SKIP — no exotic recommended for this race")

    if pre_sel.notes:
        for note in pre_sel.notes:
            lines.append(f"  NOTE: {note}")

    lines.append(f"  Total stake: ${pre_sel.total_stake:.2f}")

    return "\n".join(lines)
