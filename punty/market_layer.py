"""Post-ranking market comparison layer.

Compares tissue probabilities to market odds AFTER tissue ranking is
complete. Used for:
  1. Value detection (overlay/underlay classification)
  2. Odds movement confirmation signal
  3. Tissue-market agreement confidence boost
  4. Stake sizing (higher confidence = larger bet)

CRITICAL RULE: Market comparison can boost confidence in a pick for
stake sizing, but NEVER changes rank order. The tissue ranking stands.
"""

import logging
import statistics
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get(obj: Any, attr: str, default=None):
    """Get attribute from ORM object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


@dataclass
class MarketComparison:
    """Market comparison result for a single runner."""
    tissue_prob: float         # our independent tissue estimate
    market_implied: float      # from median odds (overround-stripped)
    value_rating: float        # tissue / market (>1.0 = overlay)
    place_value_rating: float  # tissue place / market place
    edge: float                # tissue - market (absolute difference)
    movement_signal: str       # steamer/firmed/steady/drifted/blowout
    movement_mult: float       # multiplier from odds movement table
    agreement: str             # aligned/near/disagreed
    confidence_boost: float    # 0.0-0.15 bonus when tissue+market agree


def _get_median_odds(runner: Any) -> Optional[float]:
    """Get median odds from available bookmaker sources."""
    trusted = [
        _get(runner, "odds_betfair"),
        _get(runner, "odds_sportsbet"),
    ]
    supplementary = [
        _get(runner, "odds_tab"),
        _get(runner, "odds_bet365"),
        _get(runner, "odds_ladbrokes"),
    ]
    trusted_valid = [o for o in trusted if o and isinstance(o, (int, float)) and o > 1.0]
    supp_valid = [o for o in supplementary if o and isinstance(o, (int, float)) and o > 1.0]

    if trusted_valid:
        anchor = statistics.median(trusted_valid)
        all_valid = list(trusted_valid)
        for o in supp_valid:
            if 0.5 * anchor <= o <= 2.0 * anchor:
                all_valid.append(o)
        return statistics.median(all_valid)
    elif supp_valid:
        if len(supp_valid) >= 3:
            med = statistics.median(supp_valid)
            filtered = [o for o in supp_valid if 0.5 * med <= o <= 2.0 * med]
            return statistics.median(filtered) if filtered else med
        return statistics.median(supp_valid)
    else:
        co = _get(runner, "current_odds")
        if co and isinstance(co, (int, float)) and co > 1.0:
            return co
        return None


def _classify_movement(runner: Any) -> tuple[str, float]:
    """Classify odds movement direction and return (signal_name, multiplier).

    Movement multipliers from 221K-runner analysis:
      firmed_5_20:  1.558  (strongest positive signal)
      steamer_20pct: 1.162
      steady:       1.177
      drifted_5_20: 1.120
      blowout_20pct: 0.500  (strong negative signal)
    """
    current = _get(runner, "current_odds")
    opening = _get(runner, "opening_odds")

    if (not current or not opening or not isinstance(current, (int, float))
            or not isinstance(opening, (int, float)) or current <= 1 or opening <= 1):
        return "unknown", 1.0

    ratio = current / opening

    if ratio < 0.80:
        return "steamer", 1.162
    elif ratio < 0.95:
        return "firmed", 1.558
    elif ratio <= 1.05:
        return "steady", 1.177
    elif ratio <= 1.20:
        return "drifted", 1.120
    else:
        return "blowout", 0.500


def compare_to_market(
    tissue_probs: dict,
    runners: list,
) -> dict[str, MarketComparison]:
    """Compare tissue probabilities to market after tissue ranking.

    Args:
        tissue_probs: Dict of runner_id → TissueResult from build_tissue()
        runners: List of active Runner objects/dicts

    Returns:
        Dict of runner_id → MarketComparison
    """
    # Calculate market overround
    total_implied = 0.0
    runner_odds: dict[str, float] = {}
    for runner in runners:
        rid = _get(runner, "id", "")
        odds = _get_median_odds(runner)
        if odds and odds > 1.0:
            runner_odds[rid] = odds
            total_implied += 1.0 / odds

    overround = total_implied if total_implied > 0 else 1.0
    field_size = len(runners)

    # Find market favourite (lowest odds)
    market_fav_id = min(runner_odds, key=runner_odds.get) if runner_odds else None

    # Find tissue rank 1 (highest tissue probability)
    tissue_rank1_id = max(tissue_probs, key=lambda rid: tissue_probs[rid].win_probability) if tissue_probs else None

    results: dict[str, MarketComparison] = {}

    for runner in runners:
        rid = _get(runner, "id", "")
        tissue = tissue_probs.get(rid)
        if not tissue:
            continue

        tp = tissue.win_probability
        tpp = tissue.place_probability
        odds = runner_odds.get(rid, 0)

        # Market implied probability (overround-stripped)
        if odds > 1.0 and overround > 0:
            market_implied = (1.0 / odds) / overround
        else:
            market_implied = 1.0 / field_size if field_size > 0 else 0.1

        # Value rating
        value = tp / market_implied if market_implied > 0 else 1.0
        edge = tp - market_implied

        # Place value
        place_count = 2 if field_size <= 7 else 3
        place_odds = _get(runner, "place_odds")
        if place_odds and isinstance(place_odds, (int, float)) and place_odds > 1.0:
            mkt_place = 1.0 / place_odds
        else:
            mkt_place = market_implied * place_count  # rough estimate
        place_value = tpp / mkt_place if mkt_place > 0 else 1.0

        # Odds movement
        movement_signal, movement_mult = _classify_movement(runner)

        # Agreement classification
        # Tissue and market agree when they both rank this runner similarly
        tissue_rank = sorted(
            tissue_probs.keys(),
            key=lambda r: tissue_probs[r].win_probability,
            reverse=True
        ).index(rid) + 1

        market_rank = sorted(
            [r for r in runner_odds],
            key=lambda r: runner_odds[r]
        ).index(rid) + 1 if rid in runner_odds else field_size

        rank_diff = abs(tissue_rank - market_rank)
        if rank_diff <= 1:
            agreement = "aligned"
        elif rank_diff <= 3:
            agreement = "near"
        else:
            agreement = "disagreed"

        # Confidence boost — tissue + market alignment is the sweet spot
        # 221K analysis: when our rank 1 matches market fav → 43.5% win rate
        confidence_boost = 0.0
        if rid == tissue_rank1_id and rid == market_fav_id:
            confidence_boost = 0.15  # strongest: tissue #1 = market fav
        elif agreement == "aligned":
            confidence_boost = 0.08  # good: ranks within 1
        elif agreement == "near":
            confidence_boost = 0.03  # mild: ranks within 3

        # Movement confirmation/contradiction
        if movement_signal == "firmed" and tissue_rank <= 3:
            confidence_boost += 0.05  # smart money confirms our pick
        elif movement_signal == "blowout" and tissue_rank <= 3:
            confidence_boost -= 0.05  # smart money disagrees

        results[rid] = MarketComparison(
            tissue_prob=round(tp, 4),
            market_implied=round(market_implied, 4),
            value_rating=round(value, 3),
            place_value_rating=round(place_value, 3),
            edge=round(edge, 4),
            movement_signal=movement_signal,
            movement_mult=movement_mult,
            agreement=agreement,
            confidence_boost=round(max(0.0, min(0.20, confidence_boost)), 3),
        )

    return results
