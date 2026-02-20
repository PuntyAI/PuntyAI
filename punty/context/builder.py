"""Build context for AI prompts."""

import json
import logging
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)
MELB = ZoneInfo("Australia/Melbourne")


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
        from punty.models.settings import AppSettings

        # Load custom probability weights (once for all races)
        self._probability_weights = None
        try:
            wt_result = await self.db.execute(
                select(AppSettings).where(AppSettings.key == "probability_weights")
            )
            wt_setting = wt_result.scalar_one_or_none()
            if wt_setting and wt_setting.value:
                import json as _json
                raw = _json.loads(wt_setting.value)
                # Convert percentages (0-100) to decimals (0.0-1.0)
                self._probability_weights = {k: v / 100.0 for k, v in raw.items()}
        except Exception:
            logger.debug("Failed to load probability weights, using defaults")

        # Load deep learning patterns (once for all races in this meeting)
        self._dl_patterns = []
        try:
            from punty.probability import load_dl_patterns_for_probability
            self._dl_patterns = await load_dl_patterns_for_probability(self.db)
        except Exception as e:
            logger.debug(f"Failed to load DL patterns: {e}")

        # Load tuned bet type thresholds (once for all races)
        self._sel_thresholds = None
        try:
            from punty.bet_type_tuning import load_bet_thresholds
            _tuned = await load_bet_thresholds(self.db)
            self._sel_thresholds = _tuned.get("selection")
        except Exception:
            logger.debug("Failed to load tuned thresholds, using defaults")

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

        # Fetch live WillyWeather data for hourly forecasts and observations
        ww_data = await self._fetch_willyweather(meeting)

        # Load standard times for time rating comparisons (cached per process)
        self._standard_times = {}
        try:
            from punty.context.time_ratings import load_standard_times
            self._standard_times = await load_standard_times(self.db)
        except Exception as e:
            logger.debug(f"Standard times load failed: {e}")

        # Fetch PF strike rates (cached per process, fetched once per day)
        self._jockey_strike_rates, self._trainer_strike_rates = {}, {}
        try:
            from punty.scrapers.punting_form import PuntingFormScraper
            from punty.models.settings import get_api_key
            api_key = await get_api_key(self.db, "punting_form_api_key")
            if api_key:
                pf = PuntingFormScraper(api_key=api_key)
                try:
                    rates = await pf.get_all_strike_rates()
                    self._jockey_strike_rates = rates.get("jockeys", {})
                    self._trainer_strike_rates = rates.get("trainers", {})
                finally:
                    await pf.close()
        except Exception as e:
            logger.debug(f"Strike rate fetch failed: {e}")

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
                "weather_humidity": meeting.weather_humidity,
                "wind_impact": self._get_wind_impact(meeting),
                "rail_bias_comment": meeting.rail_bias_comment,
                "rainfall": meeting.rainfall,
                "irrigation": meeting.irrigation,
                "going_stick": meeting.going_stick,
                "hourly_wind": ww_data.get("hourly_wind") if ww_data else None,
                "hourly_rain_prob": ww_data.get("hourly_rain_prob") if ww_data else None,
                "observation": ww_data.get("observation") if ww_data else None,
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

        # Calculate sequence leg confidence across races
        context["sequence_leg_analysis"] = self._calculate_sequence_legs(context["races"])

        # Pre-build sequence lanes (Skinny/Balanced/Wide) from probability data
        try:
            from punty.context.pre_sequences import build_all_sequence_lanes
            total_races = len(context["races"])
            seq_legs = context.get("sequence_leg_analysis", [])
            context["pre_built_sequences"] = build_all_sequence_lanes(
                total_races, seq_legs, context["races"],
            )
        except Exception as e:
            logger.debug(f"Pre-sequence lane construction failed: {e}")

        # Pre-calculate Big 3 Multi recommendation
        try:
            from punty.context.pre_big3 import calculate_pre_big3
            context["pre_big3"] = calculate_pre_big3(context["races"])
        except Exception as e:
            logger.debug(f"Pre-Big3 calculation failed: {e}")

        return context

    def _calculate_sequence_legs(self, races: list[dict]) -> list[dict]:
        """Calculate sequence leg confidence for quaddie construction."""
        from punty.probability import calculate_sequence_leg_confidence

        try:
            # Build races_data from probability-enriched race contexts
            races_data = []
            for race_ctx in races:
                probs = race_ctx.get("probabilities", {})
                ranked = probs.get("probability_ranked", [])
                if not ranked:
                    continue

                # Build runner list from ranked data + runner context
                runners = []
                for entry in ranked:
                    horse_name = entry.get("horse")
                    # Find matching runner in race context for saddlecloth
                    for r in race_ctx.get("runners", []):
                        if r.get("horse_name") == horse_name and not r.get("scratched"):
                            runners.append({
                                "saddlecloth": r.get("saddlecloth", 0),
                                "horse_name": horse_name,
                                "win_prob": r.get("_win_prob_raw", 0),
                                "value_rating": r.get("punty_value_rating", 1.0),
                                "edge": r.get("_edge_raw", 0),
                                "current_odds": r.get("current_odds"),
                            })
                            break

                races_data.append({
                    "race_number": race_ctx["race_number"],
                    "runners": runners,
                })

            if not races_data:
                return []

            legs = calculate_sequence_leg_confidence(races_data)
            return [
                {
                    "race_number": leg.race_number,
                    "confidence": leg.leg_confidence,
                    "suggested_width": leg.suggested_width,
                    "top_runners": leg.top_runners,
                    "odds_shape": leg.odds_shape,
                    "shape_width": leg.shape_width,
                }
                for leg in legs
            ]
        except Exception as e:
            logger.debug(f"Sequence leg confidence calculation failed: {e}")
            return []

    @staticmethod
    def _get_wind_impact(meeting) -> str | None:
        """Get wind impact analysis for the meeting venue."""
        if not meeting.weather_wind_speed or not meeting.weather_wind_dir:
            return None
        try:
            from punty.scrapers.willyweather import analyse_wind_impact
            result = analyse_wind_impact(
                meeting.venue, meeting.weather_wind_speed, meeting.weather_wind_dir
            )
            return result["description"] if result else None
        except Exception:
            return None

    async def _fetch_willyweather(self, meeting) -> dict | None:
        """Fetch live WillyWeather data for hourly forecasts.

        Returns the full weather dict with hourly_wind, hourly_rain_prob,
        and observation data — or None if unavailable. Uses cached data
        if the scraper has already fetched for this venue today.
        """
        try:
            from punty.scrapers.willyweather import WillyWeatherScraper
            ww = await WillyWeatherScraper.from_settings(self.db)
            if not ww:
                return None
            try:
                return await ww.get_weather(meeting.venue, meeting.date)
            finally:
                await ww.close()
        except Exception:
            return None

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

        # Include expert tips if available (from racing.com tipsters)
        if race.expert_tips:
            try:
                import json as _json
                race_context["expert_tips"] = _json.loads(race.expert_tips)
            except (ValueError, TypeError):
                pass

        active_runners = []
        for runner in sorted(race.runners, key=lambda r: r.saddlecloth or r.barrier or 99):
            # Audit trail for malformed runner data
            missing_fields = []
            if not runner.horse_name:
                missing_fields.append("horse_name")
            if runner.saddlecloth is None:
                missing_fields.append("saddlecloth")
            if not runner.jockey:
                missing_fields.append("jockey")
            if not runner.trainer:
                missing_fields.append("trainer")
            if missing_fields:
                logger.warning(
                    f"Runner data incomplete in {race.id}: "
                    f"sc={runner.saddlecloth} name={runner.horse_name!r} "
                    f"missing=[{', '.join(missing_fields)}]"
                )

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
                    runner_data["place_odds"] = runner.place_odds  # Fixed place odds

                    # Parse flucs for true opening price and market movement
                    flucs_data = self._parse_flucs(runner.odds_flucs)
                    if flucs_data:
                        runner_data["opening_odds"] = flucs_data.get("opening_price")
                        runner_data["market_movement"] = {
                            "direction": flucs_data.get("direction"),
                            "summary": flucs_data.get("summary"),
                            "pct_change": flucs_data.get("pct_change"),
                            "from": flucs_data.get("opening_price"),
                            "to": flucs_data.get("latest_price"),
                        }
                        runner_data["odds_movement"] = flucs_data.get("direction", "stable")
                    else:
                        runner_data["opening_odds"] = runner.opening_odds
                        if runner.current_odds and runner.opening_odds:
                            runner_data["odds_movement"] = self._calculate_odds_movement(
                                runner.opening_odds, runner.current_odds
                            )

                    # Multi-provider odds (keep for reference)
                    runner_data["odds_tab"] = runner.odds_tab
                    runner_data["odds_sportsbet"] = runner.odds_sportsbet
                    runner_data["odds_bet365"] = runner.odds_bet365
                    runner_data["odds_ladbrokes"] = runner.odds_ladbrokes
                    runner_data["odds_betfair"] = runner.odds_betfair

                if include_form:
                    runner_data["form"] = runner.form
                    runner_data["last_five"] = runner.last_five
                    runner_data["career_record"] = runner.career_record
                    runner_data["form_history"] = runner.form_history
                    runner_data["comments"] = runner.comments
                    runner_data["comment_long"] = runner.comment_long
                    runner_data["comment_short"] = runner.comment_short
                    runner_data["stewards_comment"] = runner.stewards_comment

                    # Parse form history for enriched analysis
                    fh_parsed = None
                    if runner.form_history:
                        try:
                            fh_parsed = json.loads(runner.form_history) if isinstance(runner.form_history, str) else runner.form_history
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Stewards excuse parsing (excuses for poor finishes)
                    if fh_parsed:
                        try:
                            from punty.context.stewards import extract_form_excuses
                            excuses = extract_form_excuses(fh_parsed)
                            if excuses:
                                runner_data["form_excuses"] = excuses
                        except Exception:
                            pass

                    # Standard time ratings (FAST/STANDARD/SLOW vs venue benchmarks)
                    if fh_parsed and self._standard_times:
                        try:
                            from punty.context.time_ratings import rate_form_times
                            ratings = rate_form_times(fh_parsed, self._standard_times)
                            if ratings:
                                runner_data["time_ratings"] = ratings
                        except Exception:
                            pass

                    # Weight-specific form analysis
                    if fh_parsed and runner.weight:
                        try:
                            from punty.context.weight_analysis import analyse_weight_form
                            weight_data = analyse_weight_form(fh_parsed, float(runner.weight))
                            if weight_data:
                                runner_data["weight_analysis"] = weight_data
                        except Exception:
                            pass

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

                    # PF strike rates (global career stats)
                    jockey_name = (runner.jockey or "").strip().lower()
                    trainer_name = (runner.trainer or "").strip().lower()
                    if jockey_name and jockey_name in self._jockey_strike_rates:
                        runner_data["jockey_strike_rate"] = self._jockey_strike_rates[jockey_name]
                    if trainer_name and trainer_name in self._trainer_strike_rates:
                        runner_data["trainer_strike_rate"] = self._trainer_strike_rates[trainer_name]

                    # Gear
                    runner_data["gear"] = runner.gear
                    runner_data["gear_changes"] = runner.gear_changes

                if include_speed_maps:
                    runner_data["speed_map_position"] = runner.speed_map_position
                    runner_data["speed_value"] = runner.speed_value
                    # Pace analysis insights
                    runner_data["pf_speed_rank"] = runner.pf_speed_rank  # 1-25, lower = faster early speed
                    runner_data["pf_settle"] = runner.pf_settle  # Historical avg settling position
                    runner_data["pf_map_factor"] = runner.pf_map_factor  # >1.0 = pace advantage
                    runner_data["pf_jockey_factor"] = runner.pf_jockey_factor  # Jockey effectiveness
                    # PF AI predictions
                    runner_data["pf_ai_score"] = runner.pf_ai_score  # 0-100 confidence
                    runner_data["pf_ai_price"] = runner.pf_ai_price  # AI-estimated odds
                    runner_data["pf_ai_rank"] = runner.pf_ai_rank  # AI rank in race
                    runner_data["pf_assessed_price"] = runner.pf_assessed_price  # Fundamental value

            race_context["runners"].append(runner_data)

            if not runner.scratched:
                active_runners.append(runner)

        race_context["analysis"] = self._analyze_race(active_runners)

        # Calculate probabilities for all active runners
        race_context["probabilities"] = self._calculate_probabilities(
            active_runners, race, race_context,
        )

        # Pre-calculate deterministic selections (bet types, stakes, Punty's Pick)
        try:
            from punty.context.pre_selections import calculate_pre_selections
            from punty.probability import _context_venue_type
            from punty.venues import guess_state
            _venue = race.meeting.venue if race.meeting else ""
            _state = guess_state(_venue)
            _vtype = _context_venue_type(_venue, _state)

            pre_sel = calculate_pre_selections(
                race_context,
                selection_thresholds=getattr(self, "_sel_thresholds", None),
                place_context_multipliers=race_context.get("probabilities", {}).get("context_multipliers_place"),
                venue_type=_vtype,
                meeting_hit_count=getattr(self, "_meeting_hit_count", None),
                meeting_race_count=getattr(self, "_meeting_race_count", None),
            )
            race_context["pre_selections"] = pre_sel
        except Exception as e:
            logger.debug(f"Pre-selection calculation failed R{race.race_number}: {e}")

        return race_context

    def _calculate_probabilities(
        self, active_runners: list, race, race_context: dict,
    ) -> dict[str, Any]:
        """Calculate and inject probabilities for all active runners."""
        from punty.probability import (
            calculate_race_probabilities,
            calculate_exotic_combinations,
        )

        try:
            meeting_ctx = {"track_condition": race.track_condition}
            if hasattr(race, "meeting") and race.meeting:
                meeting_ctx["track_condition"] = (
                    race.meeting.track_condition or race.track_condition
                )
                meeting_ctx["venue"] = race.meeting.venue

            probs = calculate_race_probabilities(
                active_runners, race, meeting_ctx,
                weights=getattr(self, "_probability_weights", None),
                dl_patterns=getattr(self, "_dl_patterns", None),
            )

            # Inject probability data into each runner's context dict
            prob_summary = {}
            exotic_runners_data = []  # For exotic combination calculations

            for runner_data in race_context["runners"]:
                if runner_data.get("scratched"):
                    continue
                # Match by runner id
                rid = None
                for r in active_runners:
                    if r.horse_name == runner_data.get("horse_name"):
                        rid = r.id
                        break
                if rid and rid in probs:
                    rp = probs[rid]
                    runner_data["punty_win_probability"] = f"{rp.win_probability * 100:.1f}%"
                    runner_data["punty_place_probability"] = f"{rp.place_probability * 100:.1f}%"
                    runner_data["punty_value_rating"] = round(rp.value_rating, 2)
                    runner_data["punty_place_value_rating"] = round(rp.place_value_rating, 2)
                    runner_data["punty_recommended_stake"] = rp.recommended_stake
                    runner_data["punty_market_implied"] = f"{rp.market_implied * 100:.1f}%"
                    # Store raw values for generator rendering
                    runner_data["_win_prob_raw"] = rp.win_probability
                    runner_data["_place_prob_raw"] = rp.place_probability
                    runner_data["_edge_raw"] = rp.edge
                    if rp.matched_patterns:
                        runner_data["_matched_patterns"] = rp.matched_patterns

                    prob_summary[runner_data["horse_name"]] = {
                        "win_prob": rp.win_probability,
                        "place_prob": rp.place_probability,
                        "value_rating": rp.value_rating,
                        "place_value_rating": rp.place_value_rating,
                        "edge": rp.edge,
                        "recommended_stake": rp.recommended_stake,
                        "saddlecloth": runner_data.get("saddlecloth"),
                    }

                    # Build data for exotic calculator
                    exotic_runners_data.append({
                        "saddlecloth": runner_data.get("saddlecloth", 0),
                        "horse_name": runner_data.get("horse_name", ""),
                        "win_prob": rp.win_probability,
                        "market_implied": rp.market_implied,
                        "value_rating": rp.value_rating,
                    })

            # Build sorted probability ranking and value plays for AI
            ranked = sorted(
                prob_summary.items(), key=lambda x: x[1]["win_prob"], reverse=True,
            )
            value_plays = [
                {"horse": name, "value": data["value_rating"], "edge": round(data["edge"] * 100, 1)}
                for name, data in ranked
                if data["value_rating"] > 0.90
            ]

            # Calculate exotic combinations
            exotic_combos = []
            if len(exotic_runners_data) >= 2:
                try:
                    combos = calculate_exotic_combinations(exotic_runners_data)
                    exotic_combos = [
                        {
                            "type": c.exotic_type,
                            "runners": c.runners,
                            "runner_names": c.runner_names,
                            "probability": f"{c.estimated_probability * 100:.1f}%",
                            "value": c.value_ratio,
                            "combos": c.num_combos,
                            "format": c.format,
                        }
                        for c in combos
                    ]
                except Exception as e:
                    logger.debug(f"Exotic combination calculation failed: {e}")

            # Get context multipliers for this race (for AI prompt)
            context_mults = {}
            context_mults_place = {}
            try:
                from punty.probability import _get_context_multipliers
                meeting_obj = race.meeting if hasattr(race, "meeting") else None
                if meeting_obj:
                    context_mults = _get_context_multipliers(race, meeting_obj, "win")
                    context_mults_place = _get_context_multipliers(race, meeting_obj, "place")
            except Exception:
                pass

            return {
                "probability_ranked": [
                    {"horse": name, "win_prob": f"{data['win_prob'] * 100:.1f}%",
                     "saddlecloth": data.get("saddlecloth")}
                    for name, data in ranked
                ],
                "value_plays": value_plays,
                "exotic_combinations": exotic_combos,
                "context_multipliers": context_mults,
                "context_multipliers_place": context_mults_place,
            }
        except Exception as e:
            logger.warning(f"Probability calculation failed: {e}")
            return {}

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

    def _parse_flucs(self, flucs_json: Optional[str]) -> dict[str, Any]:
        """Parse odds fluctuation data and calculate market movement."""
        if not flucs_json:
            return {}

        try:
            flucs = json.loads(flucs_json)
            if not flucs or not isinstance(flucs, list):
                return {}

            # Parse odds values (remove $ sign if present)
            def parse_odds(val):
                if isinstance(val, str):
                    return float(val.replace("$", "").strip())
                return float(val) if val else None

            # Get opening (first) and latest (last) prices
            opening_entry = flucs[0]
            latest_entry = flucs[-1]

            opening_price = parse_odds(opening_entry.get("odds"))
            latest_price = parse_odds(latest_entry.get("odds"))

            if not opening_price or not latest_price:
                return {}

            # Calculate movement
            price_change = latest_price - opening_price
            pct_change = (price_change / opening_price) * 100

            # Determine direction and strength
            if pct_change <= -20:
                direction = "heavy_support"
                summary = f"Heavily backed ${opening_price:.2f} → ${latest_price:.2f}"
            elif pct_change <= -10:
                direction = "firming"
                summary = f"Firming ${opening_price:.2f} → ${latest_price:.2f}"
            elif pct_change >= 30:
                direction = "big_drift"
                summary = f"Big drift ${opening_price:.2f} → ${latest_price:.2f}"
            elif pct_change >= 15:
                direction = "drifting"
                summary = f"Drifting ${opening_price:.2f} → ${latest_price:.2f}"
            else:
                direction = "stable"
                summary = f"Stable around ${latest_price:.2f}"

            # Format timestamps
            opening_time = None
            if opening_entry.get("time"):
                try:
                    opening_time = datetime.fromtimestamp(opening_entry["time"], MELB).strftime("%H:%M")
                except (ValueError, OSError):
                    pass

            return {
                "opening_price": opening_price,
                "latest_price": latest_price,
                "price_change": round(price_change, 2),
                "pct_change": round(pct_change, 1),
                "direction": direction,
                "summary": summary,
                "num_movements": len(flucs),
                "opening_time": opening_time,
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse flucs: {e}")
            return {}

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

        # Map factor analysis
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

        # Top early speed runners by speed rank
        runners_with_speed = [r for r in runners if r.pf_speed_rank]
        if runners_with_speed:
            sorted_by_speed = sorted(runners_with_speed, key=lambda r: r.pf_speed_rank)[:5]
            analysis["early_speed_ranks"] = [
                {"horse": r.horse_name, "speed_rank": r.pf_speed_rank}
                for r in sorted_by_speed
            ]

        # Market movers using flucs data
        for runner in runners:
            flucs_data = self._parse_flucs(runner.odds_flucs)
            if flucs_data and flucs_data.get("direction"):
                direction = flucs_data["direction"]
                if direction in ["heavy_support", "firming"]:
                    analysis["market_movers"].append({
                        "horse": runner.horse_name,
                        "direction": "in",
                        "movement": direction,
                        "summary": flucs_data.get("summary"),
                        "from": flucs_data.get("opening_price"),
                        "to": flucs_data.get("latest_price"),
                        "pct_change": flucs_data.get("pct_change"),
                    })
                elif direction in ["drifting", "big_drift"]:
                    analysis["market_movers"].append({
                        "horse": runner.horse_name,
                        "direction": "out",
                        "movement": direction,
                        "summary": flucs_data.get("summary"),
                        "from": flucs_data.get("opening_price"),
                        "to": flucs_data.get("latest_price"),
                        "pct_change": flucs_data.get("pct_change"),
                    })
            # Fallback to simple comparison if no flucs
            elif runner.current_odds and runner.opening_odds:
                movement = self._calculate_odds_movement(
                    runner.opening_odds, runner.current_odds
                )
                if movement in ["heavy_support", "firming"]:
                    analysis["market_movers"].append({
                        "horse": runner.horse_name,
                        "direction": "in",
                        "movement": movement,
                        "from": runner.opening_odds,
                        "to": runner.current_odds,
                    })
                elif movement == "drifting":
                    analysis["market_movers"].append({
                        "horse": runner.horse_name,
                        "direction": "out",
                        "movement": movement,
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
            "weather_humidity": race.meeting.weather_humidity,
            "rail_bias_comment": race.meeting.rail_bias_comment,
            "rainfall": race.meeting.rainfall,
            "irrigation": race.meeting.irrigation,
            "going_stick": race.meeting.going_stick,
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
