"""Race-level bet type optimizer (v1.0).

Classifies each race into one of 5 types, then determines per-runner
bet type and stake allocation based on EV calculations, edge detection,
odds movement, venue confidence, and circuit breaker signals.

Replaces the per-runner _determine_bet_type() approach in pre_selections.py
with a holistic race-level analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Race types
DOMINANT_EDGE = "DOMINANT_EDGE"
COMPRESSED_VALUE = "COMPRESSED_VALUE"
PLACE_LEVERAGE = "PLACE_LEVERAGE"
CHAOS_HANDICAP = "CHAOS_HANDICAP"
NO_EDGE = "NO_EDGE"

# Edge thresholds (base — adjusted by venue confidence + circuit breaker)
WIN_EDGE_MIN = 0.03           # minimum overlay for any win bet
WIN_EDGE_STRONG = 0.04        # strong overlay for dominant edge
PLACE_EDGE_MIN = 0.05         # minimum overlay for place edge
DOMINANT_GAP = 0.08           # probability gap #1 vs #2 for dominant
DOMINANT_P_WIN = 0.30         # min p_win for dominant classification

# Roughie limits
ROUGHIE_MAX_WIN_ODDS = 30.0   # no win bets above this
ROUGHIE_MIN_P_WIN = 0.06      # minimum model probability for roughie win
ROUGHIE_MIN_ODDS = 15.0       # minimum odds to qualify as roughie

# EW filters
EW_MIN_ODDS = 2.50
EW_MAX_ODDS = 15.00

# Capital efficiency
MAX_WIN_BETS = 2              # max primary win bets per race

# Venue confidence tiers
VENUE_CONFIDENCE = {
    "metro_vic": 1.0,
    "metro_nsw": 1.0,
    "metro_qld": 1.0,
    "metro_other": 0.95,
    "provincial": 0.90,
    "country": 0.80,
}

# Odds movement thresholds
STRONG_SHORTEN = 0.20
MODERATE_SHORTEN = 0.10
MODERATE_DRIFT = -0.10
STRONG_DRIFT = -0.20

# Circuit breaker
CB_MIN_RACES = 4
CB_EDGE_MULTIPLIER = 1.2      # 20% tighter when active


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class RaceClassification:
    """Race-level classification result."""

    race_type: str          # DOMINANT_EDGE | COMPRESSED_VALUE | PLACE_LEVERAGE | CHAOS_HANDICAP | NO_EDGE
    confidence: float       # 0.0-1.0
    reasoning: str          # one-liner for AI prompt context
    watch_only: bool        # True = show analysis, $0 stakes
    no_bet: bool            # True = skip entirely (2yo maiden first starters)


@dataclass
class BetRecommendation:
    """Per-runner bet type and stake recommendation."""

    saddlecloth: int
    horse_name: str
    bet_type: str           # Win | Place | Each Way | Saver Win
    stake_pct: float        # fraction of pool (0.0-1.0)
    ev_win: float
    ev_place: float
    win_edge: float
    place_edge: float
    reasoning: str


@dataclass
class RaceOptimization:
    """Complete race optimization result."""

    classification: RaceClassification
    recommendations: list[BetRecommendation]
    venue_confidence: float
    circuit_breaker_active: bool


# ──────────────────────────────────────────────
# Odds Movement
# ──────────────────────────────────────────────

def calculate_odds_movement(opening_odds: float | None, current_odds: float | None) -> float:
    """Return a confidence multiplier based on market movement.

    Strong shortening (>20%): 1.15 (boost edge confidence)
    Moderate shortening (10-20%): 1.07
    Stable: 1.0
    Moderate drift (10-20%): 0.93
    Strong drift (>20%): 0.85
    """
    if not opening_odds or not current_odds or opening_odds <= 1.0:
        return 1.0
    movement = (opening_odds - current_odds) / opening_odds
    if movement > STRONG_SHORTEN:
        return 1.15
    if movement > MODERATE_SHORTEN:
        return 1.07
    if movement < STRONG_DRIFT:
        return 0.85
    if movement < MODERATE_DRIFT:
        return 0.93
    return 1.0


# ──────────────────────────────────────────────
# Venue Confidence
# ──────────────────────────────────────────────

def get_venue_confidence(venue_type: str) -> float:
    """Return venue confidence tier multiplier."""
    return VENUE_CONFIDENCE.get(venue_type, 0.85)


# ──────────────────────────────────────────────
# Circuit Breaker
# ──────────────────────────────────────────────

def check_circuit_breaker(
    meeting_hit_count: int | None,
    meeting_race_count: int | None,
) -> bool:
    """Activate circuit breaker if 0 hits from 4+ races at a meeting."""
    if meeting_hit_count is None or meeting_race_count is None:
        return False
    return meeting_race_count >= CB_MIN_RACES and meeting_hit_count == 0


# ──────────────────────────────────────────────
# Field-Size-Scaled Place Baseline
# ──────────────────────────────────────────────

def place_baseline(field_size: int) -> float:
    """Random baseline place probability given field size.

    Accounts for places paid: 2 for ≤7, 3 for 8+.
    """
    place_count = 2 if field_size <= 7 else 3
    return min(place_count, field_size) / max(field_size, 1)


# ──────────────────────────────────────────────
# Candidate Builder
# ──────────────────────────────────────────────

def _build_optimizer_candidates(runners: list[dict]) -> list[dict]:
    """Build candidate list with EV and edge calculations from runner context."""
    candidates = []
    for r in runners:
        if r.get("scratched"):
            continue
        sc = r.get("saddlecloth")
        if not sc:
            continue
        odds = r.get("current_odds") or 0
        if not odds or odds <= 1.0:
            continue

        win_prob = r.get("_win_prob_raw", 0)
        place_prob = r.get("_place_prob_raw", 0)

        # Place odds — real from TAB or estimated
        place_odds = r.get("place_odds")
        if not place_odds or place_odds <= 1.0:
            place_odds = _estimate_place_odds(odds)

        # Implied probabilities (market)
        implied_win = 1.0 / odds
        implied_place = 1.0 / place_odds if place_odds > 1.0 else 0.5

        # EV per $1
        ev_win = win_prob * odds - 1
        ev_place = place_prob * place_odds - 1

        # Edge = our probability - market probability
        win_edge = win_prob - implied_win
        place_edge = place_prob - implied_place

        candidates.append({
            "saddlecloth": sc,
            "horse_name": r.get("horse_name", ""),
            "odds": odds,
            "place_odds": place_odds,
            "opening_odds": r.get("opening_odds"),
            "win_prob": win_prob,
            "place_prob": place_prob,
            "value_rating": r.get("punty_value_rating", 1.0),
            "place_value_rating": r.get("punty_place_value_rating", 1.0),
            "ev_win": ev_win,
            "ev_place": ev_place,
            "win_edge": win_edge,
            "place_edge": place_edge,
            "implied_win": implied_win,
            "implied_place": implied_place,
            "career_record": r.get("career_record"),
            "rec_stake": r.get("punty_recommended_stake", 0),
        })

    # Sort by win probability descending
    candidates.sort(key=lambda c: c["win_prob"], reverse=True)
    return candidates


def _estimate_place_odds(win_odds: float) -> float:
    """Estimate place odds when not provided (approx 1/3 of win profit + 1)."""
    if win_odds <= 1.0:
        return 1.0
    return round((win_odds - 1) / 3 + 1, 2)


# ──────────────────────────────────────────────
# Race Classification
# ──────────────────────────────────────────────

def _is_2yo_maiden_first_starters(
    race_class: str,
    age_restriction: str | None,
    candidates: list[dict],
) -> bool:
    """Check if race is a 2yo maiden with ALL first starters (zero form)."""
    rc = (race_class or "").lower()
    ar = (age_restriction or "").lower()

    # Must be a maiden race
    if "maiden" not in rc and "mdn" not in rc:
        return False

    # Must be 2yo restricted
    has_2yo = "2yo" in rc or "2yo" in ar or "2 y" in rc or "2 y" in ar
    if not has_2yo:
        return False

    # ALL runners must be first starters (no career record)
    if not candidates:
        return True
    for c in candidates:
        cr = c.get("career_record") or ""
        # "0: 0-0-0" or empty or None = first starter
        cr_clean = cr.strip().replace(" ", "")
        if cr_clean and cr_clean != "0:0-0-0" and not cr_clean.startswith("0:"):
            return False

    return True


def classify_race(
    candidates: list[dict],
    field_size: int,
    race_class: str = "",
    age_restriction: str | None = None,
    venue_confidence: float = 1.0,
    circuit_breaker_active: bool = False,
) -> RaceClassification:
    """Classify race into type using priority order.

    Priority: NO_BET → DOMINANT_EDGE → CHAOS_HANDICAP → PLACE_LEVERAGE → COMPRESSED_VALUE
    Watch Only is applied as an overlay on any classification.
    """
    # 1. No Bet — 2yo maiden first starters only
    if _is_2yo_maiden_first_starters(race_class, age_restriction, candidates):
        return RaceClassification(
            race_type=NO_EDGE,
            confidence=1.0,
            reasoning="2yo maiden, all first starters — zero form data",
            watch_only=False,
            no_bet=True,
        )

    if not candidates:
        return RaceClassification(
            race_type=NO_EDGE, confidence=1.0,
            reasoning="No active runners",
            watch_only=True, no_bet=False,
        )

    # Scale edge thresholds by venue confidence and circuit breaker
    edge_scale = 1.0 / venue_confidence
    if circuit_breaker_active:
        edge_scale *= CB_EDGE_MULTIPLIER

    win_edge_min = WIN_EDGE_MIN * edge_scale
    win_edge_strong = WIN_EDGE_STRONG * edge_scale
    place_edge_min = PLACE_EDGE_MIN * edge_scale

    # Apply odds movement to edges before classification
    adjusted_candidates = []
    for c in candidates:
        ca = dict(c)
        movement = calculate_odds_movement(c.get("opening_odds"), c["odds"])
        ca["adj_win_edge"] = c["win_edge"] * movement
        ca["adj_place_edge"] = c["place_edge"] * movement
        ca["odds_movement"] = movement
        adjusted_candidates.append(ca)

    # Sort by win_prob for classification
    adjusted_candidates.sort(key=lambda c: c["win_prob"], reverse=True)

    top1 = adjusted_candidates[0]
    top2 = adjusted_candidates[1] if len(adjusted_candidates) > 1 else None

    # Watch Only check — no meaningful edge anywhere
    has_win_edge = any(c["adj_win_edge"] >= win_edge_min for c in adjusted_candidates)
    has_place_edge = any(c["adj_place_edge"] >= place_edge_min for c in adjusted_candidates)
    watch_only = not has_win_edge and not has_place_edge

    # 2. DOMINANT_EDGE
    gap = (top1["win_prob"] - top2["win_prob"]) if top2 else 1.0
    if (top1["win_prob"] >= DOMINANT_P_WIN
            and top1["adj_win_edge"] >= win_edge_strong
            and gap >= DOMINANT_GAP):
        return RaceClassification(
            race_type=DOMINANT_EDGE,
            confidence=min(1.0, top1["adj_win_edge"] / 0.10),
            reasoning=f"Clear standout {top1['horse_name']}, {top1['adj_win_edge']:.1%} edge, {gap:.1%} gap",
            watch_only=watch_only,
            no_bet=False,
        )

    # 3. CHAOS_HANDICAP
    fav_odds = top1["odds"]
    spread_top4 = 0
    if len(adjusted_candidates) >= 4:
        spread_top4 = adjusted_candidates[0]["win_prob"] - adjusted_candidates[3]["win_prob"]
    is_chaos = (
        field_size >= 14
        or (fav_odds >= 5.0 and top1["win_prob"] <= 0.22)
    )
    if is_chaos:
        return RaceClassification(
            race_type=CHAOS_HANDICAP,
            confidence=0.8 if field_size >= 14 else 0.6,
            reasoning=f"Open race, {field_size} runners, fav ${fav_odds:.1f}",
            watch_only=watch_only,
            no_bet=False,
        )

    # 4. PLACE_LEVERAGE
    pb = place_baseline(field_size)
    place_threshold = pb + 0.15
    for c in adjusted_candidates[:3]:
        if (0.15 <= c["win_prob"] <= 0.22
                and c["place_prob"] >= place_threshold
                and c["adj_place_edge"] >= place_edge_min
                and 5.0 <= c["odds"] <= 12.0
                and c["ev_win"] >= 0
                and c["ev_place"] >= 0):
            return RaceClassification(
                race_type=PLACE_LEVERAGE,
                confidence=min(1.0, c["adj_place_edge"] / 0.10),
                reasoning=f"Place leverage on {c['horse_name']}, {c['adj_place_edge']:.1%} place edge",
                watch_only=watch_only,
                no_bet=False,
            )

    # 5. COMPRESSED_VALUE (default)
    overlay_count = sum(1 for c in adjusted_candidates[:3] if c["adj_win_edge"] > 0)
    return RaceClassification(
        race_type=COMPRESSED_VALUE,
        confidence=min(1.0, overlay_count / 3),
        reasoning=f"Standard race, {overlay_count} runners with positive overlay",
        watch_only=watch_only,
        no_bet=False,
    )


# ──────────────────────────────────────────────
# Bet Recommendation
# ──────────────────────────────────────────────

def _recommend_bet_dominant(c: dict, rank: int, field_size: int) -> tuple[str, float, str]:
    """Bet recommendation for DOMINANT_EDGE race."""
    if rank == 1:
        return "Win", 0.40, "Clear standout, strong win overlay"
    if rank == 2:
        if EW_MIN_ODDS <= c["odds"] <= EW_MAX_ODDS and c["ev_place"] > 0:
            return "Each Way", 0.25, "Secondary pick, EW for place protection"
        return "Place", 0.25, "Place fallback, odds outside EW range"
    if rank == 3:
        return "Place", 0.20, "Third pick, place-only"
    # Rank 4 (roughie)
    return "Place", 0.15, "Roughie, place-only in dominant race"


def _recommend_bet_compressed(
    c: dict, rank: int, field_size: int, candidates: list[dict],
) -> tuple[str, float, str]:
    """Bet recommendation for COMPRESSED_VALUE race."""
    # Check if 2+ runners have win overlay
    overlays = [ca for ca in candidates[:3] if ca["win_edge"] >= WIN_EDGE_MIN]

    if rank == 1:
        if len(overlays) >= 2 and 4.0 <= c["odds"] <= 6.0:
            return "Win", 0.35, "Win overlay in sweet spot $4-$6"
        if c["ev_win"] > 0 and EW_MIN_ODDS <= c["odds"] <= EW_MAX_ODDS:
            return "Each Way", 0.30, "Positive EV, each way for coverage"
        if c["ev_place"] > 0.03 and c["place_odds"] >= 2.0:
            return "Place", 0.30, "Place edge, win EV insufficient"
        return "Place", 0.30, "Default place, no clear win edge"

    if rank == 2:
        # RULE 3: Win + Saver Win when 2+ overlays
        if len(overlays) >= 2 and c["win_edge"] >= WIN_EDGE_MIN:
            return "Saver Win", 0.20, "Second overlay, saver win structure"
        if c["ev_place"] > 0 and EW_MIN_ODDS <= c["odds"] <= EW_MAX_ODDS:
            return "Each Way", 0.25, "EW coverage"
        return "Place", 0.25, "Place, modest edge"

    if rank == 3:
        # RULE 4: Place-only edge
        if c["ev_win"] <= 0 and c["ev_place"] > 0.03 and c["place_odds"] >= 2.0:
            return "Place", 0.20, "Place-only edge, EV_win negative"
        return "Place", 0.20, "Third pick, place default"

    # Rank 4 (roughie) — RULE 5
    return _recommend_roughie(c)


def _recommend_bet_place_leverage(c: dict, rank: int, field_size: int) -> tuple[str, float, str]:
    """Bet recommendation for PLACE_LEVERAGE race."""
    # RULE 2: EW when both EVs positive, otherwise Place
    if rank == 1:
        if (c["ev_win"] >= 0 and c["ev_place"] >= 0
                and EW_MIN_ODDS <= c["odds"] <= EW_MAX_ODDS):
            return "Each Way", 0.30, "EW, both EVs positive in place leverage race"
        return "Place", 0.30, "Place, leveraging place edge"
    if rank == 2:
        if c["ev_place"] > 0:
            return "Place", 0.30, "Strong place edge"
        return "Place", 0.25, "Place fallback"
    if rank == 3:
        return "Place", 0.25, "Place, consistent with leverage strategy"
    # Rank 4
    return "Place", 0.15, "Place, roughie in leverage race"


def _recommend_bet_chaos(c: dict, rank: int, field_size: int) -> tuple[str, float, str]:
    """Bet recommendation for CHAOS_HANDICAP race."""
    # RULE 6: Win on best overlay OR Exotics Only. Avoid broad EW.
    if rank == 1:
        if c["win_edge"] > WIN_EDGE_MIN and c["odds"] >= 4.0:
            return "Win", 0.30, "Best overlay in chaos race"
        return "Place", 0.25, "Place, no strong win overlay in chaos"
    if rank == 2:
        return "Place", 0.25, "Place, chaos race — avoid EW spread"
    if rank == 3:
        return "Place", 0.25, "Place, chaos race"
    # Rank 4 — in chaos, roughie is high risk
    return "Place", 0.15, "Place, roughie in chaos — high risk"


def _recommend_roughie(c: dict) -> tuple[str, float, str]:
    """RULE 5: Roughie bet recommendation."""
    odds = c["odds"]

    # Never win above $30
    if odds > ROUGHIE_MAX_WIN_ODDS:
        return "Place", 0.15, f"Place, roughie ${odds:.0f} exceeds win cap"

    # Small Win if p_win >= 0.06 and positive edge
    if (c["win_prob"] >= ROUGHIE_MIN_P_WIN
            and c["win_edge"] > 0
            and odds >= ROUGHIE_MIN_ODDS):
        return "Win", 0.10, f"Small win, roughie ${odds:.0f} with edge"

    # Never EW above $15
    if odds > EW_MAX_ODDS:
        return "Place", 0.15, "Place, odds too high for EW"

    return "Place", 0.15, "Place, roughie default"


def recommend_bet(
    candidate: dict,
    race_type: str,
    rank: int,
    field_size: int,
    candidates: list[dict] | None = None,
) -> BetRecommendation:
    """Determine bet type and stake % for a single runner given race classification."""
    # RULE 7: EW odds filter — applied as post-check
    # RULE 8: Capital efficiency — handled in optimize_race()

    if race_type == DOMINANT_EDGE:
        bt, pct, reason = _recommend_bet_dominant(candidate, rank, field_size)
    elif race_type == PLACE_LEVERAGE:
        bt, pct, reason = _recommend_bet_place_leverage(candidate, rank, field_size)
    elif race_type == CHAOS_HANDICAP:
        bt, pct, reason = _recommend_bet_chaos(candidate, rank, field_size)
    elif race_type == NO_EDGE:
        bt, pct, reason = "Place", 0.0, "No edge detected"
    else:
        # COMPRESSED_VALUE or default
        bt, pct, reason = _recommend_bet_compressed(
            candidate, rank, field_size, candidates or [],
        )

    # RULE 7 post-check: EW odds filter
    if bt == "Each Way":
        if candidate["odds"] < EW_MIN_ODDS or candidate["odds"] > EW_MAX_ODDS:
            if candidate["ev_place"] > 0:
                bt = "Place"
                reason += f" (EW filtered, odds ${candidate['odds']:.1f})"
            else:
                bt = "Win" if candidate["ev_win"] > 0 else "Place"
                reason += f" (EW filtered, odds ${candidate['odds']:.1f})"

    return BetRecommendation(
        saddlecloth=candidate["saddlecloth"],
        horse_name=candidate["horse_name"],
        bet_type=bt,
        stake_pct=pct,
        ev_win=round(candidate["ev_win"], 4),
        ev_place=round(candidate["ev_place"], 4),
        win_edge=round(candidate["win_edge"], 4),
        place_edge=round(candidate["place_edge"], 4),
        reasoning=reason,
    )


# ──────────────────────────────────────────────
# Stake Allocation
# ──────────────────────────────────────────────

def _allocate_stakes(
    recommendations: list[BetRecommendation],
    pool: float,
    watch_only: bool = False,
) -> None:
    """Allocate dollar stakes from pool based on stake_pct, rounding to 50c."""
    if watch_only:
        for rec in recommendations:
            rec.stake_pct = 0.0
        return

    # Normalize stake_pcts to sum to 1.0
    total_pct = sum(r.stake_pct for r in recommendations)
    if total_pct <= 0:
        return

    for rec in recommendations:
        rec.stake_pct = rec.stake_pct / total_pct

    # Convert to dollar amounts with rounding
    for rec in recommendations:
        raw = rec.stake_pct * pool
        # Round to nearest 50c, minimum $1
        rounded = max(1.0, round(raw * 2) / 2)
        rec.stake_pct = rounded / pool  # store as normalized pct

    # Adjust to fit pool exactly
    total = sum(r.stake_pct * pool for r in recommendations)
    if total > pool and recommendations:
        # Trim from lowest-rank (last) recommendation
        excess = total - pool
        recommendations[-1].stake_pct = max(
            1.0 / pool,
            recommendations[-1].stake_pct - excess / pool,
        )


def _enforce_capital_efficiency(recommendations: list[BetRecommendation]) -> None:
    """RULE 8: Max 2 win-exposed bets, max 40% on place component."""
    win_types = {"Win", "Saver Win", "Each Way"}
    win_exposed = [r for r in recommendations if r.bet_type in win_types]

    if len(win_exposed) > MAX_WIN_BETS:
        # Downgrade lowest-ranked win bets to Place
        win_exposed.sort(key=lambda r: r.stake_pct)
        for r in win_exposed[:-MAX_WIN_BETS]:
            r.bet_type = "Place"
            r.reasoning += " (capped: max 2 win bets)"


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def optimize_race(
    race_context: dict,
    pool: float = 20.0,
    venue_type: str = "",
    meeting_hit_count: int | None = None,
    meeting_race_count: int | None = None,
) -> RaceOptimization:
    """Classify race and recommend bets for all candidates.

    Args:
        race_context: Full race context dict from builder.py.
        pool: Stake pool (default $20).
        venue_type: metro_vic/provincial/country etc.
        meeting_hit_count: Running count of winning selections at this meeting.
        meeting_race_count: How many races processed so far at this meeting.

    Returns:
        RaceOptimization with classification and per-runner recommendations.
    """
    runners = race_context.get("runners", [])
    field_size = sum(1 for r in runners if not r.get("scratched"))
    race_class = race_context.get("class", "") or ""
    age_restriction = race_context.get("age_restriction")

    venue_conf = get_venue_confidence(venue_type)
    cb_active = check_circuit_breaker(meeting_hit_count, meeting_race_count)

    candidates = _build_optimizer_candidates(runners)

    classification = classify_race(
        candidates, field_size,
        race_class=race_class,
        age_restriction=age_restriction,
        venue_confidence=venue_conf,
        circuit_breaker_active=cb_active,
    )

    # Build recommendations
    recommendations: list[BetRecommendation] = []

    if classification.no_bet:
        return RaceOptimization(
            classification=classification,
            recommendations=[],
            venue_confidence=venue_conf,
            circuit_breaker_active=cb_active,
        )

    # Select top 4 candidates (3 + roughie)
    # Same ranking as pre_selections: #1-2 by pure prob, #3 by prob*value, #4 roughie
    used: set[int] = set()

    # Find roughie first
    roughie_pool = [
        c for c in candidates
        if c["odds"] >= 8.0 and c["value_rating"] >= 1.10
    ]
    roughie = None
    if roughie_pool:
        roughie_pool.sort(key=lambda c: c["value_rating"], reverse=True)
        roughie = roughie_pool[0]

    # Picks 1-2: pure probability
    selected: list[tuple[dict, int]] = []  # (candidate, rank)
    for c in candidates:
        if len(selected) >= 2:
            break
        if roughie and c["saddlecloth"] == roughie["saddlecloth"]:
            continue
        selected.append((c, len(selected) + 1))
        used.add(c["saddlecloth"])

    # Pick 3: prob * clamped value
    remaining = [
        c for c in candidates
        if c["saddlecloth"] not in used
        and not (roughie and c["saddlecloth"] == roughie["saddlecloth"])
    ]
    remaining.sort(
        key=lambda c: c["win_prob"] * max(0.85, min(c["value_rating"], 1.30)),
        reverse=True,
    )
    if remaining:
        selected.append((remaining[0], 3))
        used.add(remaining[0]["saddlecloth"])

    # Pick 4: roughie or next best
    if roughie and roughie["saddlecloth"] not in used:
        selected.append((roughie, 4))
        used.add(roughie["saddlecloth"])
    elif candidates:
        for c in candidates:
            if c["saddlecloth"] not in used:
                selected.append((c, 4))
                used.add(c["saddlecloth"])
                break

    # Generate recommendations
    for cand, rank in selected:
        rec = recommend_bet(
            cand,
            classification.race_type,
            rank,
            field_size,
            candidates=candidates,
        )
        recommendations.append(rec)

    # RULE 8: Capital efficiency
    _enforce_capital_efficiency(recommendations)

    # Allocate stakes
    _allocate_stakes(recommendations, pool, watch_only=classification.watch_only)

    return RaceOptimization(
        classification=classification,
        recommendations=recommendations,
        venue_confidence=venue_conf,
        circuit_breaker_active=cb_active,
    )
