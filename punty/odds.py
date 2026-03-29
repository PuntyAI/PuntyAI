"""Centralised odds service — single source of truth for runner odds updates.

ALL odds writes to Runner.current_odds, Runner.place_odds, and Runner.opening_odds
should go through this module. Provides:
- Consistent priority: PB > Betfair > Sportsbet > Bet365 > Ladbrokes > TAB
- Divergence guards: reject >3x changes
- Place odds sanity: place must be < win
- Logging and issue tracking for bad data

Usage:
    from punty.odds import update_runner_odds
    update_runner_odds(runner, current_odds=3.50, source="pointsbet")
    update_runner_odds(runner, place_odds=1.40, source="betfair")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Priority ranking: lower = more trusted
SOURCE_PRIORITY = {
    "pointsbet": 1,
    "betfair": 2,
    "sportsbet": 3,
    "bet365": 4,
    "ladbrokes": 5,
    "tab": 6,  # TAB tote pools are least reliable for fixed odds
    "racing_com": 6,
    "kash": 7,
    "punting_form": 8,
}

MAX_VALID_ODDS = 501.0
MAX_DIVERGENCE_RATIO = 3.0


def update_runner_odds(
    runner,
    current_odds: Optional[float] = None,
    place_odds: Optional[float] = None,
    opening_odds: Optional[float] = None,
    source: str = "unknown",
) -> bool:
    """Update runner odds with guards and priority checks.

    Returns True if any odds were updated, False if all rejected.
    """
    updated = False
    horse = getattr(runner, "horse_name", "?")

    # ── current_odds ──
    if current_odds is not None and current_odds > 1.0:
        if current_odds > MAX_VALID_ODDS:
            logger.warning(f"Odds rejected (>{MAX_VALID_ODDS}): {horse} ${current_odds} from {source}")
        elif runner.current_odds and runner.current_odds > 1.0:
            ratio = max(current_odds, runner.current_odds) / min(current_odds, runner.current_odds)
            if ratio > MAX_DIVERGENCE_RATIO:
                # Only allow override from higher-priority source
                existing_priority = _guess_source_priority(runner)
                new_priority = SOURCE_PRIORITY.get(source.lower(), 99)
                if new_priority <= existing_priority:
                    logger.info(
                        f"Odds override ({source}): {horse} ${runner.current_odds:.2f} → ${current_odds:.2f} "
                        f"({ratio:.1f}x, {source} priority {new_priority} <= {existing_priority})"
                    )
                    runner.current_odds = current_odds
                    updated = True
                else:
                    logger.debug(
                        f"Odds rejected ({source}): {horse} ${runner.current_odds:.2f} → ${current_odds:.2f} "
                        f"({ratio:.1f}x divergence, lower priority {new_priority} > {existing_priority})"
                    )
            else:
                runner.current_odds = current_odds
                updated = True
        else:
            runner.current_odds = current_odds
            updated = True

    # ── opening_odds ── (only set once)
    if opening_odds is not None and opening_odds > 1.0 and not runner.opening_odds:
        if opening_odds <= MAX_VALID_ODDS:
            runner.opening_odds = opening_odds
            updated = True

    # ── place_odds ── (must be < win odds)
    if place_odds is not None and place_odds > 1.0:
        win = runner.current_odds or current_odds or 0
        if win > 1.0 and place_odds >= win:
            # Estimate correct place odds
            estimated = round((win - 1) / 3 + 1, 2)
            logger.warning(
                f"Place odds rejected: {horse} place ${place_odds:.2f} >= win ${win:.2f} "
                f"(from {source}). Using estimate ${estimated:.2f}"
            )
            runner.place_odds = estimated
            updated = True
        elif place_odds <= MAX_VALID_ODDS:
            runner.place_odds = place_odds
            updated = True

    return updated


def resolve_best_odds(runner) -> Optional[float]:
    """Get the best available odds for a runner using priority chain.

    Returns the highest-priority non-null odds source.
    """
    for attr in ["odds_pointsbet", "odds_betfair", "odds_sportsbet",
                 "odds_bet365", "odds_ladbrokes", "odds_tab"]:
        val = getattr(runner, attr, None)
        if val and isinstance(val, (int, float)) and val > 1.0:
            return float(val)
    return runner.current_odds if runner.current_odds and runner.current_odds > 1.0 else None


def _guess_source_priority(runner) -> int:
    """Guess which source set the current_odds by matching against stored bookie odds."""
    co = runner.current_odds
    if not co or co <= 1.0:
        return 99

    # Check which bookie odds matches current_odds
    for attr, priority in [
        ("odds_pointsbet", 1), ("odds_betfair", 2), ("odds_sportsbet", 3),
        ("odds_bet365", 4), ("odds_ladbrokes", 5), ("odds_tab", 6),
    ]:
        val = getattr(runner, attr, None)
        if val and abs(val - co) < 0.01:
            return priority

    return 5  # Default: assume mid-priority
