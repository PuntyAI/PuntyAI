"""Punty-style celebration phrases for wins, graded by size."""

import random
from typing import Optional

# Celebration tiers based on profit (in units)
# Tier 1: Small wins (pnl < 1.5)
# Tier 2: Medium wins (1.5 <= pnl < 5)
# Tier 3: Big wins (5 <= pnl < 15)
# Tier 4: Huge wins (pnl >= 15)

TIER_1_PHRASES = [
    "Get around it!",
    "That'll do pig!",
    "In the bin!",
    "Money in the bank!",
    "Easy as!",
    "Too easy!",
    "Tick!",
]

TIER_2_PHRASES = [
    "Bang!",
    "Get in there!",
    "We're on the money!",
    "Punters are eating!",
    "Fuck yeah!",
    "That's the stuff!",
    "Absolutely cooked it!",
]

TIER_3_PHRASES = [
    "Fkn BANG!",
    "HOOROO!",
    "She's a beauty!",
    "Punty you bloody legend!",
    "Send it to the pool room!",
    "That's a paddlin'... for the bookies!",
]

TIER_4_PHRASES = [
    "HOLY SHIT!",
    "ABSOLUTE SCENES!",
    "THE EAGLE HAS LANDED!",
    "WE'RE GOING TO BALI BOYS!",
    "CALL THE AMBULANCE... BUT NOT FOR US!",
]

# Track recently used phrases to avoid repetition
_recent_phrases: list[str] = []
_MAX_RECENT = 10


def get_celebration(pnl: float, pick_type: Optional[str] = None) -> str:
    """Get a Punty-style celebration phrase based on win size.

    Args:
        pnl: Profit in units (positive number for wins)
        pick_type: Optional pick type for context (exotic, sequence, etc.)

    Returns:
        A celebration phrase string
    """
    global _recent_phrases

    if pnl < 1.5:
        pool = TIER_1_PHRASES
    elif pnl < 5:
        pool = TIER_2_PHRASES
    elif pnl < 15:
        pool = TIER_3_PHRASES
    else:
        pool = TIER_4_PHRASES

    # Filter out recently used phrases
    available = [p for p in pool if p not in _recent_phrases]
    if not available:
        # Reset if we've used them all
        available = pool
        _recent_phrases = []

    phrase = random.choice(available)

    # Track as recently used
    _recent_phrases.append(phrase)
    if len(_recent_phrases) > _MAX_RECENT:
        _recent_phrases.pop(0)

    return phrase


def compose_celebration_tweet(
    horse_name: str,
    odds: float,
    stake: float,
    collect: float,
    bet_type: str = "Win",
) -> str:
    """Compose a short Punty-style celebration tweet for a big win."""
    pnl = collect - stake
    phrase = get_celebration(pnl)

    # Calculate effective odds (what actually paid)
    effective_odds = collect / stake if stake > 0 else odds
    odds_str = f"${effective_odds:.2f}"
    collect_str = f"${collect:,.2f}"
    stake_str = f"${stake:.0f}"

    # Normalise bet type for display
    bet_display = bet_type.replace("_", " ").title() if bet_type else "Win"
    if bet_display == "Saver Win":
        bet_display = "Win"
    elif bet_display == "Each Way":
        bet_display = "E/W"

    tweet = f"\U0001F3C7 {phrase} {horse_name} salutes at {odds_str}! {stake_str} on {bet_display} \u2192 {collect_str} collect \U0001F4B0"

    if len(tweet) > 270:
        tweet = tweet[:267] + "..."

    return tweet


CLEAN_SWEEP_PHRASES = [
    "CLEAN SWEEP!",
    "WE RAN THE TABLE!",
    "ALL FOUR SALUTE!",
    "FOUR FROM FOUR!",
    "THE PUNTY SPECIAL!",
    "PERFECTION!",
    "THEY ALL GOT UP!",
]


def compose_clean_sweep_tweet(
    venue: str,
    race_number: int,
    winners: list[dict],
    total_pnl: float,
    total_collect: float,
) -> str:
    """Compose a celebration tweet when all selections in a race win.

    Args:
        venue: Meeting venue name
        race_number: Race number
        winners: List of dicts with horse_name, odds keys
        total_pnl: Combined P&L for all selections in the race
        total_collect: Combined collect for all selections
    """
    phrase = random.choice(CLEAN_SWEEP_PHRASES)
    pnl_str = f"${total_pnl:+,.2f}"
    collect_str = f"${total_collect:,.2f}"

    horses = " / ".join(w["horse_name"] for w in winners)

    tweet = (
        f"\U0001F525\U0001F525\U0001F525 {phrase} {venue} R{race_number} — "
        f"all tips placed! {horses}. "
        f"Collect: {collect_str} ({pnl_str}) \U0001F525\U0001F525\U0001F525"
    )

    if len(tweet) > 270:
        tweet = (
            f"\U0001F525\U0001F525\U0001F525 {phrase} {venue} R{race_number} — "
            f"all tips placed! Collect: {collect_str} ({pnl_str}) \U0001F525\U0001F525\U0001F525"
        )

    return tweet[:280]


def compose_exotic_celebration_tweet(
    exotic_type: str,
    pnl: float,
    stake: float,
    collect: float,
    venue: str = "",
    race_number: int = 0,
) -> str:
    """Compose a celebration tweet for an exotic win (trifecta, exacta, etc.)."""
    phrase = get_celebration(pnl)
    collect_str = f"${collect:,.2f}"
    stake_str = f"${stake:.0f}"
    exotic_display = exotic_type.replace("_", " ").title()

    race_info = f" R{race_number}" if race_number else ""
    venue_info = f" {venue}" if venue else ""

    tweet = (
        f"\U0001F4A5 {phrase} {exotic_display} LANDS{venue_info}{race_info}! "
        f"{stake_str} outlay \u2192 {collect_str} collect \U0001F4B0\U0001F4B0"
    )

    if len(tweet) > 270:
        tweet = tweet[:267] + "..."

    return tweet


def get_all_phrases() -> dict:
    """Return all celebration phrases grouped by tier for reference."""
    return {
        "tier_1": {"threshold": "<1.5 units", "phrases": TIER_1_PHRASES},
        "tier_2": {"threshold": "1.5-5 units", "phrases": TIER_2_PHRASES},
        "tier_3": {"threshold": "5-15 units", "phrases": TIER_3_PHRASES},
        "tier_4": {"threshold": "15+ units", "phrases": TIER_4_PHRASES},
    }
