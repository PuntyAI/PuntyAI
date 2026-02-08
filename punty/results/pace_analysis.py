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


def compose_pace_tweet(
    bias_result: PaceBiasResult,
    venue: str,
    races_remaining: int,
) -> Optional[str]:
    """Compose a short Punty-style pace analysis tweet.

    Includes sequence adjustment suggestions when the track pattern
    is riding differently than the speed maps predicted.
    """
    if not bias_result.bias_detected:
        return None

    total = bias_result.total_races_analyzed
    front = bias_result.winners_by_position.get("leader", 0) + bias_result.winners_by_position.get("on_pace", 0)
    back = bias_result.winners_by_position.get("midfield", 0) + bias_result.winners_by_position.get("backmarker", 0)

    if bias_result.bias_type == "speed":
        templates = [
            f"\U0001F3C1 {venue} update: Speed's king today \u2014 {front}/{total} winners from on-pace or leading. Stick with the map reads for the last {races_remaining}. If you're in the sequences, lean on the speed horses \U0001F5FA\uFE0F",
            f"\U0001F3C1 {venue}: The front-runners are holding on \u2014 {front}/{total} sat handy and got the job done. Back the map, trust the speed in your quaddie legs \U0001F525",
            f"\U0001F3C1 {venue} track read: {front}/{total} winners raced on pace. The rail's playing fair \u2014 don't overthink the sequences, leaders and on-pacers are the play \U0001F3AF",
        ]
    elif bias_result.bias_type == "closer":
        templates = [
            f"\U0001F3C1 {venue} update: Closers running over them \u2014 {back}/{total} from behind today. Adjust your reads for the last {races_remaining}: look for midfield/back runners in those sequence legs \U0001F30A",
            f"\U0001F3C1 {venue}: Speed's cooked \u2014 {back}/{total} winners sat off the pace. Swap your quaddie legs toward the closers, the map horses are getting rolled \U0001F4E1",
            f"\U0001F3C1 {venue} track read: {back}/{total} from midfield or back. If your sequences had leaders, think about adding closers in the remaining legs \U0001F914",
        ]
    elif bias_result.bias_type == "on_pace":
        on_pace = bias_result.winners_by_position.get("on_pace", 0)
        templates = [
            f"\U0001F3C1 {venue}: Stalkers dominating \u2014 {on_pace}/{total} winners sat just off the leader. The sit-and-kick pattern is the play for the last {races_remaining} \U0001F3AF",
        ]
    else:
        return None

    tweet = random.choice(templates)

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
