"""Scraping orchestrator — coordinates scrapers and stores results."""

import json as _json
import logging
from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.meeting import Meeting, Race, Runner

logger = logging.getLogger(__name__)

# All Runner fields that come from the scraper
RUNNER_FIELDS = [
    "saddlecloth", "weight", "jockey", "trainer", "form", "current_odds", "opening_odds", "place_odds",
    "scratched", "comments", "horse_age", "horse_sex", "horse_colour",
    "sire", "dam", "dam_sire", "career_prize_money", "last_five",
    "days_since_last_run", "handicap_rating", "speed_value",
    "track_dist_stats", "track_stats", "distance_stats",
    "first_up_stats", "second_up_stats", "good_track_stats",
    "soft_track_stats", "heavy_track_stats", "jockey_stats", "trainer_stats", "class_stats",
    "gear", "gear_changes", "stewards_comment", "comment_long", "comment_short",
    "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes",
    "odds_betfair", "odds_flucs", "trainer_location", "form_history",
    # Punting Form insights
    "pf_speed_rank", "pf_settle", "pf_map_factor", "pf_jockey_factor",
]

# Race fields from scraper
RACE_FIELDS = [
    "name", "distance", "class_", "prize_money", "start_time",
    "track_condition", "race_type", "age_restriction", "weight_type", "field_size",
]

# Meeting fields from scraper
MEETING_FIELDS = [
    "track_condition", "weather", "rail_position",
    "penetrometer", "weather_condition", "weather_temp",
    "weather_wind_speed", "weather_wind_dir", "rail_bias_comment",
]


async def scrape_calendar(db: AsyncSession) -> list[dict]:
    """Scrape today's calendar and populate meetings in DB."""
    from punty.scrapers.calendar import scrape_calendar as _scrape

    today = melb_today()
    raw_meetings = await _scrape(today)
    results = []

    for m in raw_meetings:
        venue = m["venue"]
        venue_slug = venue.lower().replace(" ", "-")
        meeting_id = f"{venue_slug}-{today.isoformat()}"

        existing = await db.get(Meeting, meeting_id)
        if existing:
            if not existing.source:
                existing.source = "racing.com/calendar"
            # Update meeting_type if calendar detected it as trial/jumpout
            if m.get("meeting_type") in ("trial", "jumpout"):
                existing.meeting_type = m["meeting_type"]
            results.append(existing.to_dict())
            continue

        # Use meeting_type from calendar data if available, otherwise classify by venue name
        meeting_type = m.get("meeting_type") or _classify_meeting_type(venue)
        meeting = Meeting(
            id=meeting_id,
            venue=venue,
            date=today,
            selected=False,
            source="racing.com/calendar",
            meeting_type=meeting_type,
        )
        db.add(meeting)
        results.append({
            "id": meeting_id,
            "venue": venue,
            "date": today.isoformat(),
            "state": m.get("state", ""),
            "num_races": m.get("num_races", 0),
            "selected": False,
            "source": "racing.com/calendar",
            "meeting_type": meeting_type,
        })

    await db.commit()
    logger.info(f"Calendar scrape complete: {len(results)} meetings")
    return results


async def scrape_meeting_full(meeting_id: str, db: AsyncSession) -> dict:
    """Run all scrapers for a selected meeting and merge data into DB."""
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    venue = meeting.venue
    race_date = meeting.date
    errors = []

    # Primary form data — racing.com GraphQL
    try:
        from punty.scrapers.racing_com import RacingComScraper
        scraper = RacingComScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _upsert_meeting_data(db, meeting, data)
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racing.com scrape failed: {e}")
        errors.append(f"racing.com: {e}")

    # Track conditions (supplementary)
    try:
        from punty.scrapers.track_conditions import scrape_track_conditions
        state = _guess_state(venue)
        if state:
            conditions = await scrape_track_conditions(state)
            for cond in conditions:
                if cond["venue"] and cond["venue"].lower() in venue.lower():
                    meeting.track_condition = cond.get("condition") or meeting.track_condition
                    meeting.rail_position = cond.get("rail") or meeting.rail_position
                    meeting.weather = cond.get("weather") or meeting.weather
                    break
    except Exception as e:
        logger.error(f"track conditions scrape failed: {e}")
        errors.append(f"track_conditions: {e}")

    # If no races were found and not already classified, mark as trial/jumpout
    if not meeting.meeting_type or meeting.meeting_type == "race":
        race_count = await db.execute(
            select(Race).where(Race.meeting_id == meeting_id).limit(1)
        )
        if not race_count.scalar_one_or_none():
            meeting.meeting_type = _classify_meeting_type(venue)
            if meeting.meeting_type == "race":
                # No races found but venue name didn't match trial patterns —
                # still likely a trial/jumpout if racing.com has no form data
                meeting.meeting_type = "trial"
                logger.info(f"No races found for {venue} — classified as trial")

    await db.commit()
    return {"meeting_id": meeting_id, "errors": errors}


async def scrape_meeting_full_stream(meeting_id: str, db: AsyncSession) -> AsyncGenerator[dict, None]:
    """Run all scrapers for a meeting, yielding progress events."""
    logger.info(f"scrape_meeting_full_stream called for meeting_id={meeting_id}")
    # Use explicit select instead of db.get() to avoid session state issues
    from sqlalchemy import select
    try:
        result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
        meeting = result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"DB query failed for {meeting_id}: {e}")
        yield {"step": 0, "total": 1, "label": f"DB error: {e}", "status": "error"}
        return

    if not meeting:
        logger.error(f"Meeting not found in DB: {meeting_id}")
        yield {"step": 0, "total": 1, "label": f"Meeting not found: {meeting_id}", "status": "error"}
        return

    logger.info(f"Found meeting: {meeting.venue} on {meeting.date}")
    venue = meeting.venue
    race_date = meeting.date
    errors = []
    total_steps = 4

    # Step 1: racing.com GraphQL data (form + speed maps + odds in one pass)
    yield {"step": 0, "total": total_steps, "label": "Scraping racing.com GraphQL data...", "status": "running"}
    try:
        from punty.scrapers.racing_com import RacingComScraper
        scraper = RacingComScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            race_count = len(data.get("races", []))
            runner_count = len(data.get("runners", []))
            await _upsert_meeting_data(db, meeting, data)
            yield {"step": 1, "total": total_steps,
                   "label": f"GraphQL complete — {race_count} races, {runner_count} runners", "status": "done"}
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"racing.com scrape failed: {e}")
        errors.append(f"racing.com: {e}")
        yield {"step": 1, "total": total_steps, "label": f"racing.com failed: {e}", "status": "error"}

    # Step 2: Track conditions (supplementary)
    yield {"step": 1, "total": total_steps, "label": "Scraping track conditions...", "status": "running"}
    try:
        from punty.scrapers.track_conditions import scrape_track_conditions
        state = _guess_state(venue)
        if state:
            conditions = await scrape_track_conditions(state)
            found = False
            for cond in conditions:
                if cond["venue"] and cond["venue"].lower() in venue.lower():
                    meeting.track_condition = cond.get("condition") or meeting.track_condition
                    meeting.rail_position = cond.get("rail") or meeting.rail_position
                    meeting.weather = cond.get("weather") or meeting.weather
                    found = True
                    break
            if found:
                yield {"step": 2, "total": total_steps, "label": f"Track conditions: {meeting.track_condition or 'N/A'}", "status": "done"}
            else:
                # Supplementary source didn't have venue, but show GraphQL data if available
                if meeting.track_condition:
                    yield {"step": 2, "total": total_steps, "label": f"Track conditions: {meeting.track_condition} (from GraphQL)", "status": "done"}
                else:
                    yield {"step": 2, "total": total_steps, "label": "Track conditions: not available", "status": "done"}
        else:
            # Unknown state, but show GraphQL data if available
            if meeting.track_condition:
                yield {"step": 2, "total": total_steps, "label": f"Track conditions: {meeting.track_condition} (from GraphQL)", "status": "done"}
            else:
                yield {"step": 2, "total": total_steps, "label": "Track conditions: unknown state", "status": "done"}
    except Exception as e:
        logger.error(f"track conditions scrape failed: {e}")
        errors.append(f"track_conditions: {e}")
        yield {"step": 2, "total": total_steps, "label": f"Track conditions failed: {e}", "status": "error"}

    # Step 3: TAB odds (supplementary — GraphQL already has multi-provider odds)
    yield {"step": 2, "total": total_steps, "label": "Scraping TAB odds...", "status": "running"}
    try:
        from punty.scrapers.tab import TabScraper
        scraper = TabScraper()
        try:
            data = await scraper.scrape_meeting(venue, race_date)
            await _merge_odds(db, meeting_id, data.get("runners_odds", []))
        finally:
            await scraper.close()
        yield {"step": 3, "total": total_steps, "label": "TAB odds complete", "status": "done"}
    except Exception as e:
        logger.error(f"TAB scrape failed: {e}")
        errors.append(f"tab: {e}")
        yield {"step": 3, "total": total_steps, "label": f"TAB odds failed: {e}", "status": "error"}

    # Step 4: Trainer stats from Racing Australia
    yield {"step": 3, "total": total_steps, "label": "Fetching trainer stats...", "status": "running"}
    try:
        result = await populate_trainer_stats(db, meeting_id)
        matched = result.get("matched", 0)
        total = result.get("total", 0)
        yield {"step": 4, "total": total_steps, "label": f"Trainer stats: {matched}/{total} matched", "status": "done"}
    except Exception as e:
        logger.error(f"Trainer stats failed: {e}")
        errors.append(f"trainer_stats: {e}")
        yield {"step": 4, "total": total_steps, "label": f"Trainer stats failed: {e}", "status": "error"}

    await db.commit()
    error_count = len(errors)
    # Use "meeting_done" instead of "complete" to avoid bulk scrape JS thinking entire operation is done
    if error_count:
        yield {"step": total_steps, "total": total_steps, "label": f"Complete with {error_count} error(s)", "status": "meeting_done", "errors": errors}
    else:
        yield {"step": total_steps, "total": total_steps, "label": "All scrapers complete!", "status": "meeting_done", "errors": []}


async def scrape_speed_maps_stream(meeting_id: str, db: AsyncSession) -> AsyncGenerator[dict, None]:
    """Scrape speed maps for all races in a meeting, yielding progress events.

    Uses Punting Form as PRIMARY source (has speed rank, settle, map factor, jockey factor).
    Falls back to racing.com if Punting Form fails or has no data.
    """
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        yield {"step": 0, "total": 1, "label": "Meeting not found", "status": "error"}
        return

    result = await db.execute(
        select(Race).where(Race.meeting_id == meeting_id)
    )
    races = result.scalars().all()
    race_count = len(races)

    if race_count == 0:
        yield {"step": 0, "total": 1, "label": "No races found — scrape form data first", "status": "error"}
        return

    pos_map = {
        "leader": "leader",
        "on pace": "on_pace",
        "on_pace": "on_pace",
        "midfield": "midfield",
        "backmarker": "backmarker",
        "off pace": "backmarker",
        "off_pace": "backmarker",
    }

    # Track how many positions we found
    total_positions_found = 0

    # Helper to update runners with positions and PF insights
    async def update_runner_positions(event: dict, include_pf_insights: bool = False) -> int:
        """Update runners with speed map positions and optionally PF insights. Returns count of positions set."""
        count = 0
        if event.get("positions"):
            race_num = event["race_number"]
            race_id = f"{meeting_id}-r{race_num}"
            for pos in event["positions"]:
                horse_name = pos.get("horse_name", "")
                raw_pos = pos.get("position", "").lower()
                norm_pos = pos_map.get(raw_pos)
                if not norm_pos:
                    continue
                runner_result = await db.execute(
                    select(Runner).where(
                        Runner.race_id == race_id,
                        Runner.horse_name == horse_name,
                    )
                )
                runner = runner_result.scalar_one_or_none()
                if runner:
                    runner.speed_map_position = norm_pos
                    count += 1

                    # Store Punting Form insights if available
                    if include_pf_insights:
                        if pos.get("pf_speed_rank"):
                            try:
                                runner.pf_speed_rank = int(pos["pf_speed_rank"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_settle"):
                            try:
                                runner.pf_settle = float(pos["pf_settle"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_map_factor"):
                            try:
                                runner.pf_map_factor = float(pos["pf_map_factor"])
                            except (ValueError, TypeError):
                                pass
                        if pos.get("pf_jockey_factor"):
                            try:
                                runner.pf_jockey_factor = float(pos["pf_jockey_factor"])
                            except (ValueError, TypeError):
                                pass
        return count

    # PRIMARY: Try Punting Form first (has richer data)
    pf_failed = False
    try:
        from punty.scrapers.punting_form import PuntingFormScraper
        pf_scraper = await PuntingFormScraper.from_settings(db)

        async for event in pf_scraper.scrape_speed_maps(meeting.venue, meeting.date, race_count):
            positions_set = await update_runner_positions(event, include_pf_insights=True)
            total_positions_found += positions_set
            yield {k: v for k, v in event.items() if k != "positions"}

    except Exception as e:
        logger.warning(f"Punting Form speed map scrape failed: {e}")
        pf_failed = True
        yield {"step": 0, "total": race_count + 1, "label": f"Punting Form unavailable: {e}", "status": "running"}

    # FALLBACK: Use racing.com if Punting Form failed or found nothing
    if pf_failed or total_positions_found == 0:
        if not pf_failed:
            logger.info(f"No Punting Form data for {meeting.venue}, trying racing.com fallback...")
            yield {"step": 0, "total": race_count + 1, "label": "Trying racing.com fallback...", "status": "running"}

        try:
            from punty.scrapers.racing_com import RacingComScraper
            scraper = RacingComScraper()
            try:
                async for event in scraper.scrape_speed_maps(meeting.venue, meeting.date, race_count):
                    positions_set = await update_runner_positions(event, include_pf_insights=False)
                    total_positions_found += positions_set
                    yield {k: v for k, v in event.items() if k != "positions"}
            finally:
                await scraper.close()
        except Exception as e:
            logger.error(f"Racing.com speed map scrape failed: {e}")
            yield {"step": 0, "total": 1, "label": f"Racing.com fallback failed: {e}", "status": "error"}

    # Calculate completeness - count active (non-scratched) runners vs those with speed maps
    total_active_runners = 0
    runners_with_speedmap = 0
    for race in races:
        runner_result = await db.execute(
            select(Runner).where(Runner.race_id == race.id, Runner.scratched == False)
        )
        race_runners = runner_result.scalars().all()
        total_active_runners += len(race_runners)
        runners_with_speedmap += sum(1 for r in race_runners if r.speed_map_position)

    # Consider complete if at least 50% of runners have speed map data
    # (some horses genuinely don't have sectional history)
    completeness_ratio = runners_with_speedmap / total_active_runners if total_active_runners > 0 else 0
    is_complete = completeness_ratio >= 0.5

    # Update meeting status
    meeting.speed_map_complete = is_complete

    if total_positions_found == 0:
        logger.warning(f"No speed map data found for {meeting.venue} from any source")
        meeting.speed_map_complete = False
        # Unselect meetings with no data
        if meeting.selected:
            meeting.selected = False
            logger.info(f"Auto-unselected {meeting.venue} due to missing speed map data")
            yield {
                "step": race_count + 1,
                "total": race_count + 1,
                "label": f"WARNING: No speed map data available - meeting unselected",
                "status": "warning",
                "incomplete": True,
            }
    elif not is_complete:
        logger.warning(f"Incomplete speed map data for {meeting.venue}: {runners_with_speedmap}/{total_active_runners} ({completeness_ratio:.0%})")
        # Unselect meetings with very incomplete data (less than 30%)
        if completeness_ratio < 0.3 and meeting.selected:
            meeting.selected = False
            logger.info(f"Auto-unselected {meeting.venue} due to very incomplete speed map data ({completeness_ratio:.0%})")
            yield {
                "step": race_count + 1,
                "total": race_count + 1,
                "label": f"WARNING: Only {completeness_ratio:.0%} speed map coverage - meeting unselected",
                "status": "warning",
                "incomplete": True,
            }
    else:
        logger.info(f"Set {total_positions_found} speed map positions for {meeting.venue} ({completeness_ratio:.0%} coverage)")

    await db.commit()


async def refresh_odds(meeting_id: str, db: AsyncSession) -> dict:
    """Quick refresh of odds and scratchings for a meeting."""
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise ValueError(f"Meeting not found: {meeting_id}")

    try:
        from punty.scrapers.tab import TabScraper
        scraper = TabScraper()
        try:
            data = await scraper.scrape_meeting(meeting.venue, meeting.date)
            await _merge_odds(db, meeting_id, data.get("runners_odds", []))
        finally:
            await scraper.close()
        await db.commit()
        return {"meeting_id": meeting_id, "status": "ok"}
    except Exception as e:
        logger.error(f"Odds refresh failed: {e}")
        return {"meeting_id": meeting_id, "status": "error", "error": str(e)}


# --- Internal helpers ---

async def _upsert_meeting_data(db: AsyncSession, meeting: Meeting, data: dict) -> None:
    """Upsert meeting, races, and runners from scraper output."""
    m = data["meeting"]

    # Update all meeting fields
    for field in MEETING_FIELDS:
        val = m.get(field)
        if val is not None:
            setattr(meeting, field, val)

    for r in data.get("races", []):
        existing_race = await db.get(Race, r["id"])
        if existing_race:
            for field in RACE_FIELDS:
                val = r.get(field)
                if val is not None:
                    setattr(existing_race, field, val)
        else:
            race_kwargs = {
                "id": r["id"],
                "meeting_id": r["meeting_id"],
                "race_number": r["race_number"],
                "name": r.get("name", f"Race {r['race_number']}"),
                "distance": r.get("distance", 1200),
                "class_": r.get("class_"),
                "prize_money": r.get("prize_money"),
                "start_time": r.get("start_time"),
                "status": r.get("status", "scheduled"),
                "track_condition": r.get("track_condition"),
                "race_type": r.get("race_type"),
                "age_restriction": r.get("age_restriction"),
                "weight_type": r.get("weight_type"),
                "field_size": r.get("field_size"),
            }
            db.add(Race(**race_kwargs))

    for runner in data.get("runners", []):
        existing_runner = await db.get(Runner, runner["id"])
        if existing_runner:
            for field in RUNNER_FIELDS:
                val = runner.get(field)
                if val is not None:
                    setattr(existing_runner, field, val)
            if runner.get("career_record"):
                existing_runner.career_record = runner["career_record"]
            if runner.get("speed_map_position"):
                existing_runner.speed_map_position = runner["speed_map_position"]
        else:
            runner_kwargs = {
                "id": runner["id"],
                "race_id": runner["race_id"],
                "horse_name": runner["horse_name"],
                "barrier": runner.get("barrier"),
                "career_record": runner.get("career_record"),
                "speed_map_position": runner.get("speed_map_position"),
            }
            for field in RUNNER_FIELDS:
                runner_kwargs[field] = runner.get(field)
            db.add(Runner(**runner_kwargs))

    await db.flush()


async def _merge_odds(db: AsyncSession, meeting_id: str, odds_list: list[dict]) -> None:
    """Merge odds data into existing runners."""
    for odds in odds_list:
        race_num = odds.get("race_number")
        horse_name = odds.get("horse_name")
        if not (race_num and horse_name):
            continue

        race_id = f"{meeting_id}-r{race_num}"
        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.horse_name == horse_name,
            )
        )
        runner = result.scalar_one_or_none()
        if runner:
            if odds.get("current_odds") is not None:
                runner.current_odds = odds["current_odds"]
            if odds.get("opening_odds") is not None and not runner.opening_odds:
                runner.opening_odds = odds["opening_odds"]
            if odds.get("place_odds") is not None:
                runner.place_odds = odds["place_odds"]
            if odds.get("scratched"):
                runner.scratched = True
                runner.scratching_reason = odds.get("scratching_reason") or runner.scratching_reason


def _validate_result(result: dict, race_id: str) -> dict | None:
    """Validate and sanitize a single result entry. Returns None if invalid."""
    # Validate position
    position = result.get("position")
    if position is not None:
        try:
            position = int(position)
            if position < 1 or position > 30:
                logger.warning(f"{race_id}: Invalid position {position}, skipping")
                return None
        except (ValueError, TypeError):
            logger.warning(f"{race_id}: Non-integer position '{position}', skipping")
            return None
        result["position"] = position

    # Validate dividends (must be positive if present)
    for div_field in ("win_dividend", "place_dividend"):
        val = result.get(div_field)
        if val is not None:
            try:
                val = float(val)
                if val < 0:
                    logger.warning(f"{race_id}: Negative {div_field} {val}, setting to None")
                    result[div_field] = None
                else:
                    result[div_field] = val
            except (ValueError, TypeError):
                logger.warning(f"{race_id}: Invalid {div_field} '{val}', setting to None")
                result[div_field] = None

    # Validate saddlecloth
    saddlecloth = result.get("saddlecloth")
    if saddlecloth is not None:
        try:
            saddlecloth = int(saddlecloth)
            if saddlecloth < 1 or saddlecloth > 30:
                result["saddlecloth"] = None
            else:
                result["saddlecloth"] = saddlecloth
        except (ValueError, TypeError):
            result["saddlecloth"] = None

    return result


async def upsert_race_results(db: AsyncSession, meeting_id: str, race_number: int, results_data: dict) -> None:
    """Update Runner result fields and Race status from scraped results."""
    race_id = f"{meeting_id}-r{race_number}"
    race = await db.get(Race, race_id)
    if not race:
        logger.warning(f"Race not found for results: {race_id}")
        return

    # Validate: check for duplicate winners
    positions = [r.get("position") for r in results_data.get("results", []) if r.get("position") is not None]
    if positions.count(1) > 1:
        logger.warning(f"{race_id}: Multiple winners detected ({positions.count(1)}), data may be corrupt")

    # Update runner result fields
    matched = 0
    for result in results_data.get("results", []):
        # Validate result before processing
        result = _validate_result(result, race_id)
        if result is None:
            continue
        horse_name = result.get("horse_name", "")
        saddlecloth = result.get("saddlecloth")

        runner = None
        # Try horse name first
        if horse_name:
            runner_result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.horse_name == horse_name,
                ).limit(1)
            )
            runner = runner_result.scalar_one_or_none()

        # Fallback to saddlecloth
        if not runner and saddlecloth is not None:
            runner_result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.saddlecloth == int(saddlecloth),
                ).limit(1)
            )
            runner = runner_result.scalar_one_or_none()

        if not runner:
            continue

        matched += 1
        if result.get("position") is not None:
            runner.finish_position = result["position"]
        if result.get("margin") is not None:
            runner.result_margin = result["margin"]
        if result.get("starting_price") is not None:
            runner.starting_price = result["starting_price"]
        if result.get("win_dividend") is not None:
            runner.win_dividend = result["win_dividend"]
        elif result.get("starting_price") and result.get("position") in (1, 2, 3):
            # Use starting price as fallback when dividends aren't available
            try:
                sp_val = float(str(result["starting_price"]).replace("$", "").replace(",", ""))
                if result["position"] == 1 and not runner.win_dividend:
                    runner.win_dividend = sp_val
            except (ValueError, TypeError):
                pass
        if result.get("place_dividend") is not None:
            runner.place_dividend = result["place_dividend"]
        if result.get("sectional_400"):
            runner.sectional_400 = result["sectional_400"]
        if result.get("sectional_800"):
            runner.sectional_800 = result["sectional_800"]

    scraped_count = len(results_data.get("results", []))
    if scraped_count > 0 and matched == 0:
        logger.warning(f"Results for {race_id}: 0/{scraped_count} runners matched — wrong race data? Skipping status update.")
    else:
        # Only update race-level fields if runners actually matched
        race.results_status = "Paying"
        if results_data.get("winning_time"):
            race.winning_time = results_data["winning_time"]
        if results_data.get("exotics"):
            race.exotic_results = _json.dumps(results_data["exotics"])

    await db.flush()
    logger.info(f"Upserted results for {race_id}: {matched}/{scraped_count} runners matched")


def _classify_meeting_type(venue: str) -> str:
    """Classify whether a meeting is a race, trial, or jump out based on venue name."""
    v = venue.lower()
    # Jump out venues typically have a prefix like "Southside", "Inside", "Course Proper"
    # or explicitly say "jump" / "jumpout" / "barrier trial"
    jumpout_keywords = ("jump out", "jumpout", "jump-out")
    trial_keywords = ("trial", "barrier trial")
    # Common jump out venue prefixes used by Racing Victoria, NSW, etc.
    jumpout_prefixes = ("southside", "inside", "course proper", "lakeside")

    for kw in jumpout_keywords:
        if kw in v:
            return "jumpout"
    for kw in trial_keywords:
        if kw in v:
            return "trial"
    for prefix in jumpout_prefixes:
        if v.startswith(prefix):
            return "jumpout"
    return "race"


# Complete Australian racetracks by state (from racingaustralia.horse)
_STATE_TRACKS = {
    "NSW": [
        "adaminaby", "albury", "ardlethan", "armidale", "ballina", "bathurst",
        "beaumont newcastle", "bingara", "binnaway", "boorowa", "bourke", "bowraville",
        "braidwood", "brewarrina", "broken hill", "canberra", "canterbury", "canterbury park",
        "carinda", "carrathool", "casino", "cessnock", "cobar", "coffs harbour",
        "collarenebri", "come-by-chance", "condobolin", "coolabah", "cooma", "coonabarabran",
        "coonamble", "cootamundra", "corowa", "cowra", "crookwell", "deepwater", "deniliquin",
        "enngonia", "fernhill", "forbes", "geurie", "gilgandra", "glen innes", "gosford",
        "goulburn", "grafton", "grenfell", "griffith", "gulargambone", "gulgong", "gundagai",
        "gunnedah", "harden", "hawkesbury", "hay", "hillston", "holbrook", "jerilderie",
        "kembla grange", "kempsey", "kensington", "lakelands", "leeton", "lightning ridge",
        "lismore", "lockhart", "louth", "mallawa", "mendooran", "merriwa", "moama", "moree",
        "moruya", "moulamein", "mudgee", "mungery", "mungindi", "murwillumbah", "muswellbrook",
        "narrabri", "narrandera", "narromine", "newcastle", "nowra", "nyngan", "orange",
        "parkes", "pooncarie", "quambone", "queanbeyan", "quirindi", "randwick", "rosehill",
        "rosehill gardens", "royal randwick", "sapphire coast", "scone", "tabulam", "talmoi",
        "tamworth", "taree", "tocumwal", "tomingley", "tottenham", "trangie", "trundle",
        "tullamore", "tullibigeal", "tumbarumba", "tumut", "tuncurry", "wagga", "wagga riverside",
        "walcha", "walgett", "wallabadah", "wamboyne", "warialda", "warren", "warwick farm",
        "wauchope", "wean", "wellington", "wentworth", "wyong", "yass", "young",
    ],
    "VIC": [
        "alexandra", "ararat", "avoca", "bairnsdale", "ballan", "ballarat", "balnarring",
        "benalla", "bendigo", "burrumbeet", "caulfield", "colac", "coleraine", "cranbourne",
        "donald", "drouin", "dunkeld", "echuca", "edenhope", "flemington", "geelong",
        "great western", "gunbower", "hanging rock", "healesville", "hinnomunjie", "horsham",
        "kerang", "kilmore", "kyneton", "manangatang", "mansfield", "merton", "mildura",
        "moe", "moonee valley", "the valley", "mornington", "mortlake", "murtoa", "nhill",
        "oak park", "pakenham", "penshurst", "sale", "sandown", "seymour", "st arnaud",
        "stawell", "stony creek", "swan hill", "swifts creek", "tatura", "towong", "traralgon",
        "warracknabeal", "warrnambool", "werribee", "werribee park", "wodonga", "wycheproof",
        "yarra glen", "yarra valley", "yea",
    ],
    "QLD": [
        "almaden", "alpha", "aramac", "augathella", "beaudesert", "bedourie", "bell",
        "betoota", "birdsville", "blackall", "bluff", "boulia", "bowen", "bundaberg",
        "burketown", "burrandowan", "cairns", "calliope", "camooweal", "capella",
        "charleville", "charters towers", "chillagoe", "chinchilla", "clifton", "cloncurry",
        "coen", "cooktown", "corfield", "cunnamulla", "dalby", "deagon", "dingo", "doomben",
        "duaringa", "eagle farm", "eidsvold", "einasleigh", "emerald", "eromanga", "esk",
        "ewan", "flinton", "gatton", "gayndah", "georgetown", "gladstone", "gold coast",
        "goondiwindi", "gordonvale", "gregory downs", "gympie", "hebel", "home hill",
        "hughenden", "ilfracombe", "ingham", "injune", "innisfail", "ipswich", "isisford",
        "jandowae", "jericho", "julia creek", "jundah", "kilcoy", "kumbia", "laura",
        "longreach", "mackay", "mareeba", "maxwelton", "mckinlay", "middlemount", "miles",
        "mingela", "mitchell", "monto", "moranbah", "morven", "mount garnet", "mount isa",
        "mount perry", "muttaburra", "nanango", "noccundra", "normanton", "oakey", "oakley",
        "prairie", "quamby", "quilpie", "richmond", "ridgelands", "rockhampton", "roma",
        "springsure", "stamford", "stanthorpe", "st george", "stonehenge", "sunshine coast",
        "surat", "tambo", "tara", "taroom", "thangool", "theodore", "toowoomba", "townsville",
        "tower hill", "twin hills", "wandoan", "warra", "warwick", "wilpeena", "windorah",
        "winton", "wondai", "wyandra",
    ],
    "SA": [
        "balaklava", "bordertown", "ceduna", "cheltenham park", "clare", "gawler", "hawker",
        "jamestown", "kingscote", "kimba", "lock", "mindarie-halidon", "morphettville",
        "morphettville parks", "mount gambier", "murray bridge", "naracoorte", "oakbank",
        "penola", "penong", "port augusta", "port lincoln", "port pirie", "quorn",
        "roxby downs", "strathalbyn", "streaky bay", "tumby bay", "victoria park",
    ],
    "WA": [
        "albany", "ascot", "ashburton", "belmont", "beverley", "broome", "bunbury",
        "carnarvon", "collie", "derby", "dongara", "esperance", "exmouth", "fitzroy",
        "geraldton", "junction", "kalgoorlie", "kimberley", "kojonup", "kununurra", "landor",
        "lark hill", "laverton", "leinster", "leonora", "meekatharra", "mingenew", "moora",
        "mount barker", "mount magnet", "narrogin", "newman", "norseman", "northam", "perth",
        "pingrup", "pinjarra", "pinjarra park", "port hedland", "roebourne", "toodyay",
        "wiluna", "wyndham", "yalgoo", "york",
    ],
    "TAS": [
        "deloraine", "devonport", "hobart", "king island", "launceston", "longford", "spreyton",
    ],
    "NT": [
        "adelaide river", "alice springs", "barrow creek", "darwin", "katherine", "larrimah",
        "mataranka", "pine creek", "pioneer park", "renner", "tennant creek", "timber creek",
    ],
    "ACT": ["canberra", "canberra acton"],
}

# Build reverse lookup: venue -> state
_VENUE_TO_STATE = {}
for state, tracks in _STATE_TRACKS.items():
    for track in tracks:
        _VENUE_TO_STATE[track] = state


def _guess_state(venue: str) -> str:
    """Guess state from venue name using complete Australian track database."""
    v = venue.lower()

    # Direct match
    if v in _VENUE_TO_STATE:
        return _VENUE_TO_STATE[v]

    # Try partial match (venue name contains track name or vice versa)
    for track, state in _VENUE_TO_STATE.items():
        if track in v or v in track:
            return state

    # Default to VIC if unknown
    return "VIC"


# ============ TRAINER STATS ============
# Cache trainer premiership data (refreshed once per session/day)
_trainer_premiership_cache: list[dict] = []
_trainer_cache_loaded: bool = False


async def fetch_trainer_premiership(force_refresh: bool = False) -> list[dict]:
    """Fetch trainer ranking data, using cache if available.

    Tries TRC (Thoroughbred Racing) global rankings first, falls back to Racing Australia.
    """
    global _trainer_premiership_cache, _trainer_cache_loaded

    if _trainer_cache_loaded and not force_refresh:
        return _trainer_premiership_cache

    # Try TRC first (better data with global rankings, group wins, etc.)
    try:
        from punty.scrapers.racing_australia import TRCTrainerScraper

        scraper = TRCTrainerScraper()
        try:
            trainers = await scraper.scrape_trainer_rankings(country="AUS", pages=2)
            if trainers:
                _trainer_premiership_cache = trainers
                _trainer_cache_loaded = True
                logger.info(f"Loaded {len(trainers)} trainers from TRC global rankings")
                return trainers
        finally:
            await scraper.close()
    except Exception as e:
        logger.warning(f"TRC trainer scrape failed, trying Racing Australia: {e}")

    # Fallback to Racing Australia premiership
    try:
        from punty.scrapers.racing_australia import RacingAustraliaScraper

        scraper = RacingAustraliaScraper()
        try:
            trainers = await scraper.scrape_trainer_premiership(season="2025")
            _trainer_premiership_cache = trainers
            _trainer_cache_loaded = True
            logger.info(f"Loaded {len(trainers)} trainers from Racing Australia premiership")
            return trainers
        finally:
            await scraper.close()
    except Exception as e:
        logger.error(f"Failed to fetch trainer data from both sources: {e}")
        return _trainer_premiership_cache  # Return stale cache if available


async def populate_trainer_stats(db: AsyncSession, meeting_id: str) -> dict:
    """Populate trainer_stats for all runners in a meeting.

    Fetches trainer premiership data and matches to each runner's trainer.
    Returns dict with count of matches.
    """
    from punty.scrapers.racing_australia import match_trainer_name, format_trainer_stats

    # Get trainer premiership data
    trainers = await fetch_trainer_premiership()
    if not trainers:
        return {"matched": 0, "total": 0, "error": "No trainer data available"}

    # Get all runners for this meeting
    result = await db.execute(
        select(Runner)
        .join(Race)
        .where(Race.meeting_id == meeting_id)
    )
    runners = result.scalars().all()

    matched = 0
    for runner in runners:
        if not runner.trainer:
            continue

        # Try to match trainer
        trainer_data = match_trainer_name(runner.trainer, trainers)
        if trainer_data:
            runner.trainer_stats = format_trainer_stats(trainer_data)
            matched += 1

    await db.commit()
    logger.info(f"Trainer stats: matched {matched}/{len(runners)} runners for {meeting_id}")
    return {"matched": matched, "total": len(runners)}
