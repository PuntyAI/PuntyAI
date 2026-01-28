"""Detect significant changes between context versions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds for significant changes
ODDS_CHANGE_THRESHOLD = 0.20  # 20% change in odds is significant
SCRATCHING_THRESHOLD = 1  # Any scratching is significant


def detect_significant_changes(
    old_context: dict[str, Any],
    new_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect significant changes between two context versions.

    Returns list of change descriptions that might affect tips.
    """
    changes = []

    # Compare races
    old_races = {r["race_number"]: r for r in old_context.get("races", [])}
    new_races = {r["race_number"]: r for r in new_context.get("races", [])}

    for race_num, new_race in new_races.items():
        old_race = old_races.get(race_num)

        if not old_race:
            changes.append({
                "type": "new_race",
                "race": race_num,
                "description": f"New race {race_num} added",
            })
            continue

        # Check for scratchings
        scratching_changes = _detect_scratchings(old_race, new_race, race_num)
        changes.extend(scratching_changes)

        # Check for odds movements
        odds_changes = _detect_odds_changes(old_race, new_race, race_num)
        changes.extend(odds_changes)

        # Check for speed map updates
        speed_map_changes = _detect_speed_map_changes(old_race, new_race, race_num)
        changes.extend(speed_map_changes)

    # Check track condition changes
    old_track = old_context.get("meeting", {}).get("track_condition")
    new_track = new_context.get("meeting", {}).get("track_condition")
    if old_track != new_track and new_track:
        changes.append({
            "type": "track_condition",
            "old": old_track,
            "new": new_track,
            "description": f"Track condition changed: {old_track} -> {new_track}",
        })

    return changes


def _detect_scratchings(
    old_race: dict,
    new_race: dict,
    race_num: int,
) -> list[dict]:
    """Detect new scratchings."""
    changes = []

    old_runners = {r["horse_name"]: r for r in old_race.get("runners", [])}
    new_runners = {r["horse_name"]: r for r in new_race.get("runners", [])}

    for horse_name, new_runner in new_runners.items():
        old_runner = old_runners.get(horse_name)

        if not old_runner:
            continue

        # Check for new scratching
        if new_runner.get("scratched") and not old_runner.get("scratched"):
            # Check if this was a favorite
            is_favorite = (new_runner.get("current_odds") or 99) < 5.0

            changes.append({
                "type": "scratching",
                "race": race_num,
                "horse": horse_name,
                "is_favorite": is_favorite,
                "description": f"R{race_num}: {horse_name} scratched" + (
                    " (FAVORITE)" if is_favorite else ""
                ),
            })

    return changes


def _detect_odds_changes(
    old_race: dict,
    new_race: dict,
    race_num: int,
) -> list[dict]:
    """Detect significant odds movements."""
    changes = []

    old_runners = {r["horse_name"]: r for r in old_race.get("runners", [])}
    new_runners = {r["horse_name"]: r for r in new_race.get("runners", [])}

    for horse_name, new_runner in new_runners.items():
        if new_runner.get("scratched"):
            continue

        old_runner = old_runners.get(horse_name)
        if not old_runner:
            continue

        old_odds = old_runner.get("current_odds")
        new_odds = new_runner.get("current_odds")

        if not old_odds or not new_odds:
            continue

        # Calculate percentage change
        pct_change = abs((new_odds - old_odds) / old_odds)

        if pct_change >= ODDS_CHANGE_THRESHOLD:
            direction = "firmed" if new_odds < old_odds else "drifted"

            changes.append({
                "type": "odds_movement",
                "race": race_num,
                "horse": horse_name,
                "old_odds": old_odds,
                "new_odds": new_odds,
                "direction": direction,
                "pct_change": round(pct_change * 100, 1),
                "description": f"R{race_num}: {horse_name} {direction} ${old_odds:.2f} -> ${new_odds:.2f}",
            })

    return changes


def _detect_speed_map_changes(
    old_race: dict,
    new_race: dict,
    race_num: int,
) -> list[dict]:
    """Detect speed map position updates."""
    changes = []

    old_runners = {r["horse_name"]: r for r in old_race.get("runners", [])}
    new_runners = {r["horse_name"]: r for r in new_race.get("runners", [])}

    # Track if any speed maps were added (they weren't there before)
    old_has_speed_maps = any(
        r.get("speed_map_position") for r in old_race.get("runners", [])
    )
    new_has_speed_maps = any(
        r.get("speed_map_position") for r in new_race.get("runners", [])
    )

    if not old_has_speed_maps and new_has_speed_maps:
        # Speed maps newly available
        leaders = [
            r["horse_name"] for r in new_race.get("runners", [])
            if r.get("speed_map_position") == "leader"
        ]

        changes.append({
            "type": "speed_maps_available",
            "race": race_num,
            "leaders": leaders,
            "description": f"R{race_num}: Speed maps now available. Leaders: {', '.join(leaders) or 'None'}",
        })

    # Check for position changes
    for horse_name, new_runner in new_runners.items():
        if new_runner.get("scratched"):
            continue

        old_runner = old_runners.get(horse_name)
        if not old_runner:
            continue

        old_pos = old_runner.get("speed_map_position")
        new_pos = new_runner.get("speed_map_position")

        # Only flag changes if both have positions
        if old_pos and new_pos and old_pos != new_pos:
            # Significant position changes
            significant = (
                (old_pos == "backmarker" and new_pos == "leader") or
                (old_pos == "leader" and new_pos == "backmarker")
            )

            if significant:
                changes.append({
                    "type": "speed_map_change",
                    "race": race_num,
                    "horse": horse_name,
                    "old_position": old_pos,
                    "new_position": new_pos,
                    "description": f"R{race_num}: {horse_name} position changed {old_pos} -> {new_pos}",
                })

    return changes


def summarize_changes(changes: list[dict]) -> str:
    """Create human-readable summary of changes."""
    if not changes:
        return "No significant changes detected."

    summary_parts = []

    # Group by type
    scratchings = [c for c in changes if c["type"] == "scratching"]
    odds_moves = [c for c in changes if c["type"] == "odds_movement"]
    speed_maps = [c for c in changes if c["type"] in ["speed_maps_available", "speed_map_change"]]
    track = [c for c in changes if c["type"] == "track_condition"]

    if scratchings:
        horses = [c["horse"] for c in scratchings]
        summary_parts.append(f"Scratchings: {', '.join(horses)}")

    if odds_moves:
        movers = [f"{c['horse']} ({c['direction']})" for c in odds_moves]
        summary_parts.append(f"Odds movements: {', '.join(movers)}")

    if speed_maps:
        summary_parts.append(f"Speed map updates in {len(speed_maps)} race(s)")

    if track:
        summary_parts.append(f"Track: {track[0]['new']}")

    return " | ".join(summary_parts)
