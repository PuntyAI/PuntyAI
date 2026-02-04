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
        """Build full context for a race meeting."""
        from punty.models.meeting import Meeting, Race, Runner

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
                "penetrometer": meeting.penetrometer,
                "weather_condition": meeting.weather_condition,
                "weather_temp": meeting.weather_temp,
                "weather_wind_speed": meeting.weather_wind_speed,
                "weather_wind_dir": meeting.weather_wind_dir,
                "rail_bias_comment": meeting.rail_bias_comment,
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

            context["summary"]["total_runners"] += len(race.runners)
            context["summary"]["scratchings"] += sum(
                1 for r in race.runners if r.scratched
            )

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
                for runner in sorted_by_odds:
                    if runner.current_odds >= 10.0 and runner.form:
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
            "track_condition": race.track_condition,
            "race_type": race.race_type,
            "age_restriction": race.age_restriction,
            "weight_type": race.weight_type,
            "field_size": race.field_size,
            "runners": [],
            "analysis": {},
        }

        active_runners = []
        for runner in sorted(race.runners, key=lambda r: r.saddlecloth or r.barrier or 99):
            runner_data = {
                "saddlecloth": runner.saddlecloth,
                "barrier": runner.barrier,
                "horse_name": runner.horse_name,
                "jockey": runner.jockey,
                "trainer": runner.trainer,
                "trainer_location": runner.trainer_location,
                "weight": runner.weight,
                "scratched": runner.scratched,
            }

            if not runner.scratched:
                # Pedigree
                if runner.sire or runner.dam:
                    runner_data["pedigree"] = {
                        "sire": runner.sire,
                        "dam": runner.dam,
                        "dam_sire": runner.dam_sire,
                    }

                # Horse details
                if runner.horse_age or runner.horse_sex:
                    runner_data["horse_age"] = runner.horse_age
                    runner_data["horse_sex"] = runner.horse_sex
                    runner_data["horse_colour"] = runner.horse_colour

                # Performance
                runner_data["handicap_rating"] = runner.handicap_rating
                runner_data["days_since_last_run"] = runner.days_since_last_run
                runner_data["career_prize_money"] = runner.career_prize_money

                if include_odds:
                    runner_data["current_odds"] = runner.current_odds
                    runner_data["opening_odds"] = runner.opening_odds
                    if runner.current_odds and runner.opening_odds:
                        runner_data["odds_movement"] = self._calculate_odds_movement(
                            runner.opening_odds, runner.current_odds
                        )
                    # Multi-provider odds
                    runner_data["odds_tab"] = runner.odds_tab
                    runner_data["odds_sportsbet"] = runner.odds_sportsbet
                    runner_data["odds_bet365"] = runner.odds_bet365
                    runner_data["odds_ladbrokes"] = runner.odds_ladbrokes
                    runner_data["odds_betfair"] = runner.odds_betfair
                    runner_data["odds_flucs"] = runner.odds_flucs

                if include_form:
                    runner_data["form"] = runner.form
                    runner_data["last_five"] = runner.last_five
                    runner_data["career_record"] = runner.career_record
                    runner_data["form_history"] = runner.form_history
                    runner_data["comments"] = runner.comments
                    runner_data["comment_long"] = runner.comment_long
                    runner_data["comment_short"] = runner.comment_short
                    runner_data["stewards_comment"] = runner.stewards_comment

                    # Stats
                    runner_data["track_dist_stats"] = runner.track_dist_stats
                    runner_data["track_stats"] = runner.track_stats
                    runner_data["distance_stats"] = runner.distance_stats
                    runner_data["first_up_stats"] = runner.first_up_stats
                    runner_data["second_up_stats"] = runner.second_up_stats
                    runner_data["good_track_stats"] = runner.good_track_stats
                    runner_data["soft_track_stats"] = runner.soft_track_stats
                    runner_data["heavy_track_stats"] = runner.heavy_track_stats

                    # Jockey, trainer & class stats
                    runner_data["jockey_stats"] = runner.jockey_stats
                    runner_data["trainer_stats"] = runner.trainer_stats
                    runner_data["class_stats"] = runner.class_stats

                    # Gear
                    runner_data["gear"] = runner.gear
                    runner_data["gear_changes"] = runner.gear_changes

                if include_speed_maps:
                    runner_data["speed_map_position"] = runner.speed_map_position
                    runner_data["speed_value"] = runner.speed_value
                    # Punting Form insights
                    runner_data["pf_speed_rank"] = runner.pf_speed_rank  # 1-25, lower = faster early speed
                    runner_data["pf_settle"] = runner.pf_settle  # Historical avg settling position
                    runner_data["pf_map_factor"] = runner.pf_map_factor  # >1.0 = pace advantage
                    runner_data["pf_jockey_factor"] = runner.pf_jockey_factor  # Jockey effectiveness

            race_context["runners"].append(runner_data)

            if not runner.scratched:
                active_runners.append(runner)

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
            "pace_advantaged": [],  # Runners with pf_map_factor > 1.0
            "pace_disadvantaged": [],  # Runners with pf_map_factor < 1.0
            "early_speed_ranks": [],  # Top 5 by pf_speed_rank
        }

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

        # Punting Form map factor analysis
        for runner in runners:
            if runner.pf_map_factor:
                if runner.pf_map_factor >= 1.1:
                    analysis["pace_advantaged"].append({
                        "horse": runner.horse_name,
                        "map_factor": runner.pf_map_factor,
                    })
                elif runner.pf_map_factor <= 0.9:
                    analysis["pace_disadvantaged"].append({
                        "horse": runner.horse_name,
                        "map_factor": runner.pf_map_factor,
                    })

        # Top early speed runners by PF speed rank
        runners_with_speed = [r for r in runners if r.pf_speed_rank]
        if runners_with_speed:
            sorted_by_speed = sorted(runners_with_speed, key=lambda r: r.pf_speed_rank)[:5]
            analysis["early_speed_ranks"] = [
                {"horse": r.horse_name, "speed_rank": r.pf_speed_rank}
                for r in sorted_by_speed
            ]

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

        meeting_context = {
            "venue": race.meeting.venue,
            "date": race.meeting.date.isoformat(),
            "track_condition": race.meeting.track_condition,
            "weather": race.meeting.weather,
            "penetrometer": race.meeting.penetrometer,
            "weather_condition": race.meeting.weather_condition,
            "weather_temp": race.meeting.weather_temp,
            "rail_bias_comment": race.meeting.rail_bias_comment,
        }

        race_context = self._build_race_context(race)

        return {
            "meeting": meeting_context,
            "race": race_context,
        }

    async def build_results_context(
        self,
        meeting_id: str,
        race_number: int,
    ) -> dict[str, Any]:
        """Build context for results commentary — includes results, picks comparison, P&L."""
        from punty.models.meeting import Meeting, Race, Runner
        from punty.results.tracker import build_race_comparison

        result = await self.db.execute(
            select(Race)
            .where(Race.meeting_id == meeting_id, Race.race_number == race_number)
            .options(selectinload(Race.runners), selectinload(Race.meeting))
        )
        race = result.scalar_one_or_none()
        if not race:
            return {}

        meeting_context = {
            "venue": race.meeting.venue,
            "date": race.meeting.date.isoformat(),
            "track_condition": race.meeting.track_condition,
            "weather": race.meeting.weather,
            "penetrometer": race.meeting.penetrometer,
            "rail_bias_comment": race.meeting.rail_bias_comment,
        }

        # Race info
        race_context = {
            "race_number": race.race_number,
            "name": race.name,
            "distance": race.distance,
            "class": race.class_,
            "prize_money": race.prize_money,
            "track_condition": race.track_condition,
            "winning_time": race.winning_time,
            "results_status": race.results_status,
            "exotic_results": race.exotic_results,
        }

        # Full results with positions, margins, dividends, sectionals
        results_list = []
        for runner in sorted(race.runners, key=lambda r: r.finish_position or 999):
            if runner.finish_position is None:
                continue
            results_list.append({
                "position": runner.finish_position,
                "saddlecloth": runner.saddlecloth,
                "horse_name": runner.horse_name,
                "jockey": runner.jockey,
                "trainer": runner.trainer,
                "margin": runner.result_margin,
                "starting_price": runner.starting_price,
                "win_dividend": runner.win_dividend,
                "place_dividend": runner.place_dividend,
                "sectional_400": runner.sectional_400,
                "sectional_800": runner.sectional_800,
                "barrier": runner.barrier,
                "weight": runner.weight,
                "speed_map_position": runner.speed_map_position,
            })

        race_context["results"] = results_list

        # Picks comparison
        comparison = await build_race_comparison(self.db, meeting_id, race_number)

        # Count remaining races
        total_races_result = await self.db.execute(
            select(Race).where(Race.meeting_id == meeting_id)
        )
        all_races = total_races_result.scalars().all()
        races_remaining = sum(1 for r in all_races if not r.results_status or r.results_status == "Open")

        return {
            "meeting": meeting_context,
            "race": race_context,
            "picks_comparison": comparison,
            "races_remaining": races_remaining,
        }

    async def build_wrapup_context(self, meeting_id: str) -> dict[str, Any]:
        """Build context for meeting wrap-up / punt review — all results with picks ledger."""
        from punty.models.meeting import Meeting, Race
        from punty.models.content import Content, ContentType
        from punty.results.tracker import build_meeting_summary, build_pick_ledger

        result = await self.db.execute(
            select(Meeting)
            .where(Meeting.id == meeting_id)
            .options(selectinload(Meeting.races).selectinload(Race.runners))
        )
        meeting = result.scalar_one_or_none()
        if not meeting:
            return {}

        meeting_context = {
            "venue": meeting.venue,
            "date": meeting.date.isoformat(),
            "track_condition": meeting.track_condition,
            "weather": meeting.weather,
        }

        # Load the original early mail content
        early_mail_content = None
        em_result = await self.db.execute(
            select(Content).where(
                Content.meeting_id == meeting_id,
                Content.content_type == ContentType.EARLY_MAIL.value,
            ).order_by(Content.created_at.desc())
        )
        early_mail = em_result.scalars().first()
        if early_mail:
            early_mail_content = early_mail.raw_content

        # Summarise all races
        race_summaries = []
        for race in sorted(meeting.races, key=lambda r: r.race_number):
            if not race.results_status or race.results_status == "Open":
                continue
            race_data = {
                "race_number": race.race_number,
                "name": race.name,
                "distance": race.distance,
                "class": race.class_,
                "winning_time": race.winning_time,
                "exotic_results": race.exotic_results,
                "results": [],
            }
            for runner in sorted(race.runners, key=lambda r: r.finish_position or 999):
                if runner.finish_position is None:
                    continue
                race_data["results"].append({
                    "position": runner.finish_position,
                    "horse_name": runner.horse_name,
                    "margin": runner.result_margin,
                    "starting_price": runner.starting_price,
                    "win_dividend": runner.win_dividend,
                    "place_dividend": runner.place_dividend,
                })
            race_summaries.append(race_data)

        # Full meeting picks summary
        picks_summary = await build_meeting_summary(self.db, meeting_id)

        # Per-tier P&L ledger for the punt review
        pick_ledger = await build_pick_ledger(self.db, meeting_id)

        return {
            "meeting": meeting_context,
            "early_mail": early_mail_content,
            "race_summaries": race_summaries,
            "picks_summary": picks_summary,
            "pick_ledger": pick_ledger,
        }

    def context_to_json(self, context: dict) -> str:
        """Convert context to JSON string for hashing."""
        return json.dumps(context, sort_keys=True, default=str)
