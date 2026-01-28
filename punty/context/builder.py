"""Build context for AI prompts."""

import json
import logging
from datetime import date
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds rich context for AI content generation."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def build_meeting_context(
        self,
        meeting_id: str,
        include_odds: bool = True,
        include_speed_maps: bool = True,
        include_form: bool = True,
    ) -> dict[str, Any]:
        """Build full context for a race meeting.

        Returns a dict with all relevant information for content generation.
        """
        from punty.models.meeting import Meeting, Race, Runner

        # Load meeting with races and runners
        result = await self.db.execute(
            select(Meeting)
            .where(Meeting.id == meeting_id)
            .options(
                selectinload(Meeting.races).selectinload(Race.runners)
            )
        )
        meeting = result.scalar_one_or_none()

        if not meeting:
            return {}

        context = {
            "meeting": {
                "id": meeting.id,
                "venue": meeting.venue,
                "date": meeting.date.isoformat(),
                "track_condition": meeting.track_condition,
                "weather": meeting.weather,
                "rail_position": meeting.rail_position,
            },
            "races": [],
            "summary": {
                "total_races": len(meeting.races),
                "total_runners": 0,
                "scratchings": 0,
                "favorites": [],
                "roughies": [],
            },
        }

        for race in sorted(meeting.races, key=lambda r: r.race_number):
            race_context = self._build_race_context(
                race,
                include_odds=include_odds,
                include_speed_maps=include_speed_maps,
                include_form=include_form,
            )
            context["races"].append(race_context)

            # Update summary
            context["summary"]["total_runners"] += len(race.runners)
            context["summary"]["scratchings"] += sum(
                1 for r in race.runners if r.scratched
            )

            # Track favorites and roughies
            active_runners = [r for r in race.runners if not r.scratched and r.current_odds]
            if active_runners:
                sorted_by_odds = sorted(active_runners, key=lambda r: r.current_odds)
                if sorted_by_odds:
                    fav = sorted_by_odds[0]
                    context["summary"]["favorites"].append({
                        "race": race.race_number,
                        "horse": fav.horse_name,
                        "odds": fav.current_odds,
                    })
                # Roughies at good odds with decent form
                for runner in sorted_by_odds:
                    if runner.current_odds >= 10.0 and runner.form:
                        # Check for recent placings in form
                        if any(c in runner.form[:4] for c in "123"):
                            context["summary"]["roughies"].append({
                                "race": race.race_number,
                                "horse": runner.horse_name,
                                "odds": runner.current_odds,
                                "form": runner.form,
                            })
                            break

        return context

    def _build_race_context(
        self,
        race,
        include_odds: bool = True,
        include_speed_maps: bool = True,
        include_form: bool = True,
    ) -> dict[str, Any]:
        """Build context for a single race."""
        race_context = {
            "race_number": race.race_number,
            "name": race.name,
            "distance": race.distance,
            "class": race.class_,
            "prize_money": race.prize_money,
            "start_time": race.start_time.isoformat() if race.start_time else None,
            "status": race.status,
            "runners": [],
            "analysis": {},
        }

        # Process runners
        active_runners = []
        for runner in sorted(race.runners, key=lambda r: r.barrier or 99):
            runner_data = {
                "barrier": runner.barrier,
                "horse_name": runner.horse_name,
                "jockey": runner.jockey,
                "trainer": runner.trainer,
                "weight": runner.weight,
                "scratched": runner.scratched,
            }

            if include_odds and not runner.scratched:
                runner_data["current_odds"] = runner.current_odds
                runner_data["opening_odds"] = runner.opening_odds
                if runner.current_odds and runner.opening_odds:
                    runner_data["odds_movement"] = self._calculate_odds_movement(
                        runner.opening_odds, runner.current_odds
                    )

            if include_form and not runner.scratched:
                runner_data["form"] = runner.form
                runner_data["career_record"] = runner.career_record
                runner_data["comments"] = runner.comments

            if include_speed_maps and not runner.scratched:
                runner_data["speed_map_position"] = runner.speed_map_position

            race_context["runners"].append(runner_data)

            if not runner.scratched:
                active_runners.append(runner)

        # Add race analysis
        race_context["analysis"] = self._analyze_race(active_runners)

        return race_context

    def _calculate_odds_movement(self, opening: float, current: float) -> str:
        """Calculate odds movement description."""
        if not opening or not current:
            return "stable"

        pct_change = ((current - opening) / opening) * 100

        if pct_change <= -20:
            return "heavy_support"
        elif pct_change <= -10:
            return "firming"
        elif pct_change >= 20:
            return "drifting"
        elif pct_change >= 10:
            return "easing"
        else:
            return "stable"

    def _analyze_race(self, runners: list) -> dict[str, Any]:
        """Generate race analysis from runner data."""
        analysis = {
            "pace_scenario": "unknown",
            "likely_leaders": [],
            "backmarkers": [],
            "market_movers": [],
        }

        # Analyze pace
        leaders = [r for r in runners if r.speed_map_position == "leader"]
        on_pace = [r for r in runners if r.speed_map_position == "on_pace"]
        backmarkers = [r for r in runners if r.speed_map_position == "backmarker"]

        if len(leaders) >= 3:
            analysis["pace_scenario"] = "hot_pace"
        elif len(leaders) == 0 and len(on_pace) <= 2:
            analysis["pace_scenario"] = "slow_pace"
        elif len(leaders) == 1:
            analysis["pace_scenario"] = "genuine_pace"
        else:
            analysis["pace_scenario"] = "moderate_pace"

        analysis["likely_leaders"] = [r.horse_name for r in leaders]
        analysis["backmarkers"] = [r.horse_name for r in backmarkers]

        # Find market movers
        for runner in runners:
            if runner.current_odds and runner.opening_odds:
                movement = self._calculate_odds_movement(
                    runner.opening_odds, runner.current_odds
                )
                if movement in ["heavy_support", "firming"]:
                    analysis["market_movers"].append({
                        "horse": runner.horse_name,
                        "direction": "in",
                        "from": runner.opening_odds,
                        "to": runner.current_odds,
                    })
                elif movement == "drifting":
                    analysis["market_movers"].append({
                        "horse": runner.horse_name,
                        "direction": "out",
                        "from": runner.opening_odds,
                        "to": runner.current_odds,
                    })

        return analysis

    async def build_race_context(
        self,
        meeting_id: str,
        race_number: int,
    ) -> dict[str, Any]:
        """Build context for a specific race only."""
        from punty.models.meeting import Meeting, Race, Runner

        result = await self.db.execute(
            select(Race)
            .where(Race.meeting_id == meeting_id, Race.race_number == race_number)
            .options(selectinload(Race.runners), selectinload(Race.meeting))
        )
        race = result.scalar_one_or_none()

        if not race:
            return {}

        # Get meeting context
        meeting_context = {
            "venue": race.meeting.venue,
            "date": race.meeting.date.isoformat(),
            "track_condition": race.meeting.track_condition,
            "weather": race.meeting.weather,
        }

        # Get race context
        race_context = self._build_race_context(race)

        return {
            "meeting": meeting_context,
            "race": race_context,
        }

    def context_to_json(self, context: dict) -> str:
        """Convert context to JSON string for hashing."""
        return json.dumps(context, sort_keys=True, default=str)
