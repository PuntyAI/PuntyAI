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
    """Compose a short Punty-style celebration tweet for a selection win."""
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

    tweet = f"\U0001F3C7 {phrase} {horse_name} salutes at {odds_str}! {stake_str} {bet_display} \u2192 {collect_str} collect \U0001F4B0"

    if len(tweet) > 270:
        tweet = tweet[:267] + "..."

    return tweet


def compose_exotic_celebration(
    exotic_type: str,
    race_number: int,
    venue: str,
    stake: float,
    collect: float,
) -> str:
    """Compose a celebration tweet for an exotic bet win."""
    pnl = collect - stake
    phrase = get_celebration(pnl, "exotic")
    collect_str = f"${collect:,.2f}"
    stake_str = f"${stake:.0f}"
    display_type = exotic_type.title() if exotic_type else "Exotic"

    tweet = f"\U0001F3AF {phrase} {display_type} lands at {venue} R{race_number}! {stake_str} \u2192 {collect_str} collect \U0001F4B0"

    if len(tweet) > 270:
        tweet = tweet[:267] + "..."

    return tweet


def compose_sequence_celebration(
    sequence_type: str,
    variant: Optional[str],
    venue: str,
    stake: float,
    collect: float,
) -> str:
    """Compose a celebration tweet for a sequence/multi win."""
    pnl = collect - stake
    phrase = get_celebration(pnl, "sequence")
    collect_str = f"${collect:,.2f}"
    stake_str = f"${stake:.0f}"

    display_type = (sequence_type or "Sequence").title()
    if variant:
        display_type = f"{display_type} ({variant.title()})"

    emoji = "\U0001F680" if pnl >= 15 else "\U0001F525"
    tweet = f"{emoji} {phrase} {display_type} nails it at {venue}! {stake_str} \u2192 {collect_str} collect \U0001F4B0"

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
