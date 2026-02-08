"""Pace bias analysis - compares speed map predictions to actual race results."""

import logging
import random
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.meeting import Race, Runner

logger = logging.getLogger(__name__)

POSITION_CATEGORIES = ["leader", "on_pace", "midfield", "backmarker"]
MIN_RACES_FOR_ANALYSIS = 3
BIAS_THRESHOLD = 0.6  # 60%+ of winners from one category
COMBINED_BIAS_THRESHOLD = 0.75  # 75%+ from front or back combined


class PaceBiasResult:
    """Result of a pace bias analysis."""

    def __init__(self):
        self.total_races_analyzed = 0
        self.winners_by_position = {pos: 0 for pos in POSITION_CATEGORIES}
        self.bias_detected = False
        self.bias_type: Optional[str] = None  # "speed", "closer", "on_pace"
        self.bias_strength: float = 0.0


async def analyze_pace_bias(
    db: AsyncSession,
    meeting_id: str,
    completed_race_numbers: list[int],
) -> Optional[PaceBiasResult]:
    """Analyze pace bias across completed races at a meeting.

    Compares winner's speed_map_position (pre-race prediction) to actual
    finish positions to detect if the track is favouring a particular
    running style.

    Returns PaceBiasResult if enough data, else None.
    """
    if len(completed_race_numbers) < MIN_RACES_FOR_ANALYSIS:
        return None

    result = PaceBiasResult()

    for race_num in completed_race_numbers:
        race_id = f"{meeting_id}-r{race_num}"

        winner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.finish_position == 1,
            )
        )
        winner = winner_result.scalar_one_or_none()
        if not winner or not winner.speed_map_position:
            continue

        position = winner.speed_map_position.lower().strip()
        if position in result.winners_by_position:
            result.winners_by_position[position] += 1
            result.total_races_analyzed += 1

    if result.total_races_analyzed < MIN_RACES_FOR_ANALYSIS:
        return None

    total = result.total_races_analyzed
    front = result.winners_by_position.get("leader", 0) + result.winners_by_position.get("on_pace", 0)
    back = result.winners_by_position.get("midfield", 0) + result.winners_by_position.get("backmarker", 0)

    # Check single-category dominance
    for pos, count in result.winners_by_position.items():
        ratio = count / total
        if ratio >= BIAS_THRESHOLD:
            result.bias_detected = True
            result.bias_strength = ratio
            if pos == "leader":
                result.bias_type = "speed"
            elif pos == "on_pace":
                result.bias_type = "on_pace"
            else:
                result.bias_type = "closer"
            break

    # Check combined front bias
    if not result.bias_detected and front / total >= COMBINED_BIAS_THRESHOLD:
        result.bias_detected = True
        result.bias_type = "speed"
        result.bias_strength = front / total

    # Check combined closer bias
    if not result.bias_detected and back / total >= COMBINED_BIAS_THRESHOLD:
        result.bias_detected = True
        result.bias_type = "closer"
        result.bias_strength = back / total

    return result


async def find_bias_fits(
    db: AsyncSession,
    meeting_id: str,
    completed_race_numbers: list[int],
    total_races: int,
    bias_type: Optional[str],
) -> list[dict]:
    """Find runners in remaining races that fit the detected bias pattern.

    Returns a list of {horse_name, race_number, odds, position} dicts,
    sorted by odds (shortest first), max 4 runners.
    """
    if not bias_type:
        return []

    # Determine which speed_map_positions to look for
    if bias_type == "speed":
        target_positions = ["leader", "on_pace"]
    elif bias_type == "closer":
        target_positions = ["midfield", "backmarker"]
    elif bias_type == "on_pace":
        target_positions = ["on_pace"]
    else:
        return []

    remaining_nums = [n for n in range(1, total_races + 1) if n not in completed_race_numbers]
    if not remaining_nums:
        return []

    fits = []
    for race_num in remaining_nums:
        race_id = f"{meeting_id}-r{race_num}"
        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.scratched == False,
                Runner.speed_map_position.isnot(None),
            )
        )
        for runner in result.scalars().all():
            pos = runner.speed_map_position.lower().strip()
            if pos in target_positions:
                fits.append({
                    "horse_name": runner.horse_name,
                    "race_number": race_num,
                    "odds": runner.current_odds or 999,
                    "position": pos,
                })

    # Sort by odds (shortest first) and take top 4
    fits.sort(key=lambda x: x["odds"])
    return fits[:4]


def _format_horse_suggestions(fits: list[dict]) -> str:
    """Format horse suggestions into a compact string like 'Horse (R5 $3.50), Horse (R7 $6)'."""
    if not fits:
        return ""
    parts = []
    for f in fits:
        odds_str = f"${f['odds']:.0f}" if f["odds"] >= 10 else f"${f['odds']:.2f}"
        parts.append(f"{f['horse_name']} (R{f['race_number']} {odds_str})")
    return ", ".join(parts)


def compose_pace_tweet(
    bias_result: PaceBiasResult,
    venue: str,
    races_remaining: int,
    horse_suggestions: Optional[list[dict]] = None,
) -> Optional[str]:
    """Compose a short Punty-style pace analysis tweet.

    Includes sequence adjustment suggestions when the track pattern
    is riding differently than the speed maps predicted.
    When no bias is detected, posts a reassuring "maps are tracking" update.
    horse_suggestions is an optional list of {horse_name, race_number, odds, position} dicts.
    """
    total = bias_result.total_races_analyzed
    front = bias_result.winners_by_position.get("leader", 0) + bias_result.winners_by_position.get("on_pace", 0)
    back = bias_result.winners_by_position.get("midfield", 0) + bias_result.winners_by_position.get("backmarker", 0)

    suggestions = _format_horse_suggestions(horse_suggestions or [])

    if not bias_result.bias_detected:
        templates = [
            f"\U0001F3C1 {venue} map check after {total} races: No funny business \u2014 the track's playing honest and the maps are holding up. Trust your tips for the last {races_remaining}, punt away \U0001F91D",
            f"\U0001F3C1 {venue} pace read ({total} in): Had a look at the runs so far and we're tracking nicely. No bias, no dramas \u2014 the speed maps are doing their job. Fire away for the last {races_remaining} \U0001F525",
            f"\U0001F3C1 {venue} update: {total} races done, had a squiz at the patterns \u2014 all square. Leaders and closers both getting their chance. Maps are on the money, stick with the reads \U0001F3AF",
            f"\U0001F3C1 {venue} track check: Punty's reviewed {total} races and the map reads are bang on. No adjustments needed \u2014 back yourself for the last {races_remaining} \U0001F4AA",
        ]
    elif bias_result.bias_type == "speed":
        base = f"\U0001F3C1 {venue} track read: Speed's king \u2014 {front}/{total} winners on-pace or leading."
        if suggestions:
            templates = [
                f"{base} Ones to watch up front: {suggestions} \U0001F525",
                f"{base} The map horses to follow: {suggestions} \U0001F3AF",
            ]
        else:
            templates = [
                f"{base} Stick with the map reads, lean on the speed horses for the last {races_remaining} \U0001F525",
            ]
    elif bias_result.bias_type == "closer":
        base = f"\U0001F3C1 {venue} track read: Closers running riot \u2014 {back}/{total} from behind."
        if suggestions:
            templates = [
                f"{base} Ones sitting off it to watch: {suggestions} \U0001F30A",
                f"{base} Back-runners to follow: {suggestions} \U0001F4E1",
            ]
        else:
            templates = [
                f"{base} Think about adding closers in those sequence legs for the last {races_remaining} \U0001F30A",
            ]
    elif bias_result.bias_type == "on_pace":
        on_pace = bias_result.winners_by_position.get("on_pace", 0)
        base = f"\U0001F3C1 {venue}: Stalkers dominating \u2014 {on_pace}/{total} sat just off the speed and kicked."
        if suggestions:
            templates = [
                f"{base} Sit-and-kick types to watch: {suggestions} \U0001F3AF",
            ]
        else:
            templates = [
                f"{base} The sit-and-kick pattern is the play for the last {races_remaining} \U0001F3AF",
            ]
    else:
        return None

    tweet = random.choice(templates)

    # Trim suggestions if tweet too long
    if len(tweet) > 280 and suggestions:
        # Try with fewer horses
        shorter = _format_horse_suggestions((horse_suggestions or [])[:2])
        tweet = tweet.replace(suggestions, shorter)

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
