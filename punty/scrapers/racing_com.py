"""Racing.com scraper — intercepts GraphQL API responses for structured data.

Uses Playwright to load racing.com pages and captures all GraphQL responses
from graphql.rmdprod.racing.com, extracting 70+ fields per runner.

URL pattern: racing.com/form/{date}/{venue-slug}/race/{n}
"""

import asyncio
import json as _json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, AsyncGenerator, Optional

from punty.config import melb_now
from punty.scrapers.base import BaseScraper, ScraperError
from punty.scrapers.playwright_base import new_page

logger = logging.getLogger(__name__)

# Map racing.com odds provider codes to our field names
PROVIDER_MAP = {
    "Q": "odds_tab",       # TAB/SuperTAB
    "SB2": "odds_sportsbet",
    "B3": "odds_bet365",
    "LD": "odds_ladbrokes",
    "BF": "odds_betfair",
}


class RacingComScraper(BaseScraper):
    """Scraper for racing.com — primary source for Australian racing data."""

    BASE_URL = "https://www.racing.com"

    # Common sponsor prefixes to strip from venue names
    SPONSOR_PREFIXES = [
        "ladbrokes", "tab", "bet365", "sportsbet", "neds", "pointsbet",
        "unibet", "betfair", "palmerbet", "bluebet", "topsport",
    ]

    # Venue name aliases - map calendar names to racing.com URL slugs
    VENUE_ALIASES = {
        "sandown lakeside": "sandown",
        "sandown-lakeside": "sandown",
        "thomas farms rc murray bridge": "murray-bridge",
        "thomas-farms-rc-murray-bridge": "murray-bridge",
    }

    def _venue_slug(self, venue: str) -> str:
        """Convert venue name to URL slug, stripping sponsor prefixes."""
        slug = venue.lower().strip()
        # Strip sponsor prefixes (e.g., "Ladbrokes Geelong" -> "geelong")
        for prefix in self.SPONSOR_PREFIXES:
            if slug.startswith(prefix + " "):
                slug = slug[len(prefix) + 1:]
                break
            if slug.startswith(prefix + "-"):
                slug = slug[len(prefix) + 1:]
                break
        # Check for venue aliases
        if slug in self.VENUE_ALIASES:
            return self.VENUE_ALIASES[slug]
        slug_dashed = slug.replace(" ", "-")
        if slug_dashed in self.VENUE_ALIASES:
            return self.VENUE_ALIASES[slug_dashed]
        return slug_dashed

    def _build_meeting_url(self, venue: str, race_date: date) -> str:
        slug = self._venue_slug(venue)
        return f"{self.BASE_URL}/form/{race_date.isoformat()}/{slug}"

    def _build_race_url(self, venue: str, race_date: date, race_num: int) -> str:
        slug = self._venue_slug(venue)
        return f"{self.BASE_URL}/form/{race_date.isoformat()}/{slug}/race/{race_num}"

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape full meeting data by intercepting GraphQL API responses.

        Opens one Playwright session, navigates to each race page, and captures
        getMeeting_CD, getRaceEntriesForField_CD, GetBettingData_CD, and
        getRaceNumberList_CD GraphQL responses.
        """
        meeting_url = self._build_meeting_url(venue, race_date)
        logger.info(f"Scraping racing.com meeting (GraphQL): {meeting_url}")

        meeting_id = self.generate_meeting_id(venue, race_date)

        # Accumulators for captured GraphQL data
        meeting_gql: dict = {}
        race_list_gql: list = []
        race_entries_by_num: dict[int, list] = {}
        betting_by_num: dict[int, list] = {}
        form_history_by_horse: dict[str, list] = {}  # horseCode -> list of past starts

        async with new_page() as page:
            # --- GraphQL response interceptor ---
            async def capture_graphql(response):
                if "graphql.rmdprod.racing.com" not in response.url:
                    return
                try:
                    body = await response.text()
                    data = _json.loads(body)
                except Exception:
                    return

                url = response.url

                if "getMeeting_CD" in url:
                    gm = data.get("data", {}).get("getMeeting")
                    if gm:
                        meeting_gql.update(gm)

                elif "getRaceNumberList_CD" in url:
                    races = data.get("data", {}).get("getNoCacheRacesForMeet", [])
                    if races:
                        race_list_gql.clear()
                        race_list_gql.extend(races)

                elif "getRaceEntriesForField_CD" in url or "getRaceEntries" in url:
                    form = data.get("data", {}).get("getRaceForm", {})
                    if not form:
                        # Try alternative data paths
                        form = data.get("data", {}).get("getRaceEntries", {})
                    entries = form.get("formRaceEntries", [])
                    # Determine race number from the entries or form
                    race_num = None
                    if entries:
                        race_num = entries[0].get("raceNumber")
                    if race_num and entries:
                        race_entries_by_num[race_num] = entries
                        logger.debug(f"Captured {len(entries)} entries for race {race_num}")

                if "getRaceEntryItemByHorsePaged_CD" in url:
                    # Note: API returns key with capital G "GetRaceEntryItemByHorsePaged"
                    items = data.get("data", {}).get("GetRaceEntryItemByHorsePaged", [])
                    if isinstance(items, list) and items:
                        # Identify horse by horseName from first item
                        horse_name = items[0].get("horseName", "")
                        if horse_name:
                            existing = form_history_by_horse.get(horse_name, [])
                            for item in items:
                                race_info = item.get("race", {}) or {}
                                # Try multiple field names for class
                                race_class = (
                                    race_info.get("rdcClass")
                                    or race_info.get("raceClass")
                                    or race_info.get("className")
                                    or race_info.get("class")
                                    or race_info.get("raceClassName")
                                    or race_info.get("classCode")
                                )
                                # Also try to get prize money which often indicates class
                                prize = race_info.get("prizeMoney") or race_info.get("prize") or race_info.get("totalPrizeMoney")
                                start = {
                                    "date": race_info.get("date"),
                                    "venue": race_info.get("venueAbbr") or race_info.get("location"),
                                    "distance": race_info.get("distance"),
                                    "class": race_class,
                                    "prize": prize,
                                    "track": ((race_info.get("trackCondition") or "") + " " + (race_info.get("trackRating") or "")).strip(),
                                    "field": race_info.get("runnersCount"),
                                    "pos": item.get("finish") or item.get("finishAbv"),
                                    "margin": item.get("margin"),
                                    "weight": item.get("weightCarried"),
                                    "jockey": item.get("jockeyName"),
                                    "barrier": item.get("barrierNumber"),
                                    "sp": item.get("startingPrice"),
                                    "time": race_info.get("raceTime"),
                                    "settled": item.get("positionAtSettledAbv"),
                                    "at800": item.get("positionAt800Abv"),
                                    "at400": item.get("positionAt400Abv"),
                                    "comment": item.get("comment"),
                                }
                                # Top 4 finishers
                                top4_raw = item.get("raceEntriesByMaxFinish", [])
                                if top4_raw:
                                    start["top4"] = [
                                        {"name": t.get("horseName"), "pos": t.get("finish")}
                                        for t in top4_raw[:4]
                                    ]
                                existing.append(start)
                            form_history_by_horse[horse_name] = existing
                            logger.debug(f"Captured {len(items)} form items for {horse_name}")

                # Always scan for formRaceEntries as fallback
                self._try_extract_entries(data, race_entries_by_num)

                if "GetBettingData_CD" in url:
                    bd = data.get("data", {}).get("GetBettingData", {})
                    entries = bd.get("formRaceEntries", [])
                    # We need to figure out race number — stored in captured context
                    if entries:
                        # Store temporarily keyed by first entry id; will merge later
                        betting_by_num.setdefault("_pending", []).append(entries)

            page.on("response", capture_graphql)

            # Track which queries we capture for debugging
            captured_queries = set()

            original_capture = capture_graphql
            async def capture_graphql_debug(response):
                if "graphql.rmdprod.racing.com" in response.url:
                    # Log which query types we see
                    for qname in ["getMeeting_CD", "getRaceNumberList_CD",
                                  "getRaceEntriesForField_CD", "GetBettingData_CD",
                                  "getRaceEntries", "getRaceForm",
                                  "getRaceEntryItemByHorsePaged_CD"]:
                        if qname in response.url:
                            captured_queries.add(qname)
                            break
                    else:
                        # Log unknown queries
                        captured_queries.add(response.url.split("/")[-1][:60])
                await original_capture(response)

            page.remove_listener("response", capture_graphql)
            page.on("response", capture_graphql_debug)

            try:
                # 1. Load meeting page to get meeting info and race list
                resp = await page.goto(meeting_url, wait_until="load")
                if resp and resp.status == 404:
                    raise ScraperError(f"HTTP 404: {meeting_url}")
                await page.wait_for_timeout(3000)  # Reduced from 5000

                # Dismiss cookie banner
                try:
                    btn = page.locator("button:has-text('Decline')").first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                except Exception:
                    pass

                # Determine race count from GraphQL race list or DOM
                race_count = len(race_list_gql)
                if race_count == 0:
                    # Fallback: count from DOM links
                    race_count = await page.evaluate("""() => {
                        var links = document.querySelectorAll('a[href*="/race/"]');
                        var maxNum = 0;
                        for (var i = 0; i < links.length; i++) {
                            var m = links[i].href.match(/\\/race\\/(\\d+)$/);
                            if (m) { var n = parseInt(m[1]); if (n > maxNum) maxNum = n; }
                        }
                        return maxNum;
                    }""")

                if race_count == 0:
                    logger.warning(f"No races found on {meeting_url}")
                    return {"meeting": self._build_meeting_dict(meeting_id, venue, race_date, meeting_gql),
                            "races": [], "runners": []}

                logger.info(f"Found {race_count} races for {venue}")

                # 2. Navigate to each race page to trigger GraphQL queries
                for race_num in range(1, race_count + 1):
                    # Use #/full-form hash to go directly to Full Form view
                    # This triggers lazy loading of extended form history on scroll
                    race_url = f"{self._build_race_url(venue, race_date, race_num)}#/full-form"
                    logger.info(f"Scraping race {race_num}/{race_count}: {race_url}")

                    await page.goto(race_url, wait_until="networkidle")
                    await page.wait_for_timeout(2500)

                    # Dismiss cookie banner on first race
                    if race_num == 1:
                        try:
                            btn = page.locator("button:has-text('Decline')").first
                            if await btn.is_visible(timeout=1500):
                                await btn.click()
                        except Exception:
                            pass

                    # Scroll down to trigger lazy loading of getRaceEntryItemByHorsePaged_CD
                    # The extended form history loads as each horse row scrolls into view
                    # Slower scroll with longer waits to ensure GraphQL requests complete
                    for _ in range(20):
                        await page.evaluate("window.scrollBy(0, 400)")
                        await page.wait_for_timeout(600)

                    # Scroll back up and down again to catch any missed horses
                    await page.evaluate("window.scrollTo(0, 0)")
                    await page.wait_for_timeout(500)
                    for _ in range(20):
                        await page.evaluate("window.scrollBy(0, 400)")
                        await page.wait_for_timeout(500)

                    # Check if we got entries for this race
                    if race_num not in race_entries_by_num:
                        logger.warning(f"Race {race_num}: no entries captured after page load")
            finally:
                # Always remove listener to prevent memory leaks
                try:
                    page.remove_listener("response", capture_graphql_debug)
                except Exception:
                    pass

        logger.info(f"Captured GraphQL queries: {captured_queries}")
        logger.info(f"Entries captured for races: {list(race_entries_by_num.keys())}")
        logger.info(f"Total entries per race: { {k: len(v) for k, v in race_entries_by_num.items()} }")
        logger.info(f"Form history captured for {len(form_history_by_horse)} horses")

        # --- Build output from captured GraphQL data ---
        meeting_dict = self._build_meeting_dict(meeting_id, venue, race_date, meeting_gql)

        # Build race-level lookup from getRaceNumberList
        race_info_by_num: dict[int, dict] = {}
        for r in race_list_gql:
            rn = r.get("raceNumber")
            if rn:
                race_info_by_num[rn] = r

        # Build betting lookup by horse name for merging
        betting_by_horse: dict[str, list] = {}
        for entries_list in betting_by_num.get("_pending", []):
            for entry in entries_list:
                name = entry.get("horseName", "")
                if name:
                    betting_by_horse[name] = entry.get("oddsByProvider", [])

        races = []
        runners = []

        for race_num in range(1, race_count + 1):
            race_id = self.generate_race_id(meeting_id, race_num)
            rl = race_info_by_num.get(race_num, {})

            # Parse distance
            dist_str = rl.get("distance", "")
            distance = self.parse_distance(dist_str) or 1200

            # Parse prize money
            prize_money = None
            total_pm = rl.get("totalPrizeMoney")
            if total_pm:
                try:
                    prize_money = int(float(total_pm))
                except (ValueError, TypeError):
                    pass

            # Parse start time
            start_time = self._parse_iso_time(rl.get("time"))

            # Parse condition string for weight type / age restriction
            condition_str = rl.get("condition", "")
            weight_type = self._extract_weight_type(condition_str)
            age_restriction = self._extract_age_restriction(condition_str)

            race_data = {
                "id": race_id,
                "meeting_id": meeting_id,
                "race_number": race_num,
                "name": rl.get("name", f"Race {race_num}"),
                "distance": distance,
                "class_": rl.get("rdcClass") or rl.get("nameForm"),
                "prize_money": prize_money,
                "start_time": start_time,
                "status": "scheduled",
                "track_condition": rl.get("trackCondition"),
                "race_type": "Thoroughbred",
                "age_restriction": age_restriction,
                "weight_type": weight_type,
                "field_size": None,  # set after counting runners
            }

            # Process entries from getRaceEntriesForField_CD
            entries = race_entries_by_num.get(race_num, [])
            race_runners = []
            for entry in entries:
                runner = self._parse_graphql_entry(entry, race_id, betting_by_horse)
                if runner:
                    # Attach form history if captured
                    horse_name = runner["horse_name"]
                    fh = form_history_by_horse.get(horse_name)
                    if fh:
                        runner["form_history"] = _json.dumps(fh)
                    race_runners.append(runner)

            # Set field size (non-scratched)
            race_data["field_size"] = sum(1 for r in race_runners if not r.get("scratched"))

            races.append(race_data)
            runners.extend(race_runners)

        return {"meeting": meeting_dict, "races": races, "runners": runners}

    def _try_extract_entries(self, data: dict, race_entries_by_num: dict) -> None:
        """Scan a GraphQL response for formRaceEntries, regardless of query name."""
        def _find_entries(obj, depth=0):
            if depth > 5:
                return
            if isinstance(obj, dict):
                # Look for formRaceEntries or raceEntries
                for key in ["formRaceEntries", "raceEntries", "entries"]:
                    entries = obj.get(key)
                    if isinstance(entries, list) and len(entries) > 0:
                        # Check if entries have horseName (to confirm they're runner data)
                        if entries[0].get("horseName") or entries[0].get("raceEntryNumber"):
                            race_num = entries[0].get("raceNumber")
                            if race_num and race_num not in race_entries_by_num:
                                race_entries_by_num[race_num] = entries
                                logger.info(f"Captured {len(entries)} entries for race {race_num} via '{key}'")
                            return
                for v in obj.values():
                    _find_entries(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _find_entries(item, depth + 1)
        _find_entries(data)

    def _build_meeting_dict(self, meeting_id: str, venue: str, race_date: date, gql: dict) -> dict:
        """Build meeting dict from getMeeting_CD GraphQL data."""
        # Parse penetrometer
        penetrometer = None
        pen_str = gql.get("penetrometer")
        if pen_str:
            try:
                penetrometer = float(pen_str)
            except (ValueError, TypeError):
                pass

        # Parse weather temp
        weather_temp = None
        temp_str = gql.get("weatherAirTemp")
        if temp_str:
            try:
                weather_temp = int(float(temp_str))
            except (ValueError, TypeError):
                pass

        # Parse wind speed
        weather_wind_speed = None
        wind_str = gql.get("weatherWindSpeed")
        if wind_str:
            try:
                weather_wind_speed = int(float(wind_str))
            except (ValueError, TypeError):
                pass

        # Rail bias comment from previous rail positions
        rail_bias_comment = None
        prev_positions = gql.get("previousRailPositions", [])
        if prev_positions:
            first = prev_positions[0]
            comments = first.get("comments")
            if comments:
                rail_bias_comment = comments
            # Also check tips for short/long comments
            meet_data = first.get("meet", {})
            tips = meet_data.get("meetTips", [])
            if tips:
                short = tips[0].get("shortComment")
                long_c = tips[0].get("longComment")
                if short or long_c:
                    rail_bias_comment = long_c or short

        # Track condition with rating
        tc = gql.get("trackCondition", "")
        tr = gql.get("trackRating", "")
        track_condition = f"{tc} {tr}".strip() if tc else None

        return {
            "id": meeting_id,
            "venue": venue.title(),
            "date": race_date,
            "track_condition": track_condition,
            "weather": gql.get("weather") or gql.get("weatherText"),
            "rail_position": gql.get("railPosition"),
            "penetrometer": penetrometer,
            "weather_condition": gql.get("weather"),
            "weather_temp": weather_temp,
            "weather_wind_speed": weather_wind_speed,
            "weather_wind_dir": gql.get("weatherWindDirection"),
            "rail_bias_comment": rail_bias_comment,
            "meet_code": gql.get("id"),  # Internal racing.com meeting ID for CSV downloads
        }

    def _parse_graphql_entry(self, entry: dict, race_id: str, betting_by_horse: dict) -> Optional[dict]:
        """Parse a single formRaceEntry from getRaceEntriesForField_CD."""
        horse_name = entry.get("horseName", "").strip()
        if not horse_name:
            return None

        barrier = entry.get("barrierNumber")
        tab_number = entry.get("raceEntryNumber")
        runner_id = self.generate_runner_id(race_id, barrier or tab_number or 0, horse_name)

        # Weight
        weight = self.parse_weight(entry.get("weight"))

        # Speed value and map position
        sv = None
        sv_str = entry.get("speedValue")
        if sv_str:
            try:
                sv = int(sv_str)
            except (ValueError, TypeError):
                pass

        # Horse details from nested horse object
        horse = entry.get("horse", {}) or {}
        age_str = horse.get("age", "")  # e.g. "4YO"
        horse_age = None
        if age_str:
            m = re.match(r"(\d+)", age_str)
            if m:
                horse_age = int(m.group(1))

        # Last five — comes as JSON string like '["4","3","1"]'
        last_five_raw = horse.get("lastFive", "")
        last_five = None
        if last_five_raw:
            try:
                arr = _json.loads(last_five_raw)
                last_five = "".join(str(x) for x in arr)
            except (ValueError, TypeError):
                last_five = str(last_five_raw)

        # Days since last run
        days_since = None
        last_race_str = entry.get("lastRaceDate") or horse.get("lastRaceDate")
        if last_race_str:
            try:
                last_dt = datetime.fromisoformat(last_race_str.replace("Z", ""))
                # Use Melbourne time for "today" comparison
                days_since = (melb_now().replace(tzinfo=None) - last_dt).days
            except Exception:
                pass

        # Career prize money
        career_pm = None
        pm_str = horse.get("careerPrizeMoney", "")
        if pm_str:
            career_pm = self.parse_prize_money(pm_str)

        # Handicap rating
        hr = entry.get("handicapRating")
        handicap_rating = None
        if hr is not None:
            try:
                handicap_rating = float(hr)
            except (ValueError, TypeError):
                pass

        # Career record from horse stats
        career_record = None
        last_ten = horse.get("lastTenStats")
        if last_ten:
            career_record = last_ten

        # Form from last five
        form = last_five

        # Odds — from entry.odds array
        odds_dict = self._parse_odds_array(entry.get("odds", []))

        # Merge betting data if available
        betting_odds = betting_by_horse.get(horse_name, [])
        if betting_odds:
            for bp in betting_odds:
                code = bp.get("providerCode", "")
                field = PROVIDER_MAP.get(code)
                if field and field not in odds_dict:
                    odds_dict[field] = self.parse_odds(bp.get("oddsWin"))
                # Capture flucs from first provider that has them
                if bp.get("flucsWin") and "odds_flucs" not in odds_dict:
                    flucs = [{"time": f.get("updateTime"), "odds": f.get("amount")}
                             for f in bp["flucsWin"][:20]]
                    odds_dict["odds_flucs"] = _json.dumps(flucs)

        # Primary odds (use TAB as current_odds)
        current_odds = odds_dict.get("odds_tab")
        if not current_odds:
            # Fallback to first available
            for k in ["odds_sportsbet", "odds_bet365", "odds_ladbrokes"]:
                if odds_dict.get(k):
                    current_odds = odds_dict[k]
                    break

        # Sire/dam — from nested horse.sireHorseName etc.
        sire = horse.get("sireHorseName")
        dam = horse.get("damHorseName")

        return {
            "id": runner_id,
            "race_id": race_id,
            "horse_name": horse_name,
            "saddlecloth": tab_number,
            "barrier": barrier,
            "weight": weight,
            "jockey": entry.get("jockeyName"),
            "trainer": entry.get("trainerName"),
            "form": form,
            "career_record": career_record,
            "speed_map_position": self._speed_value_to_position(sv),
            "current_odds": current_odds,
            "opening_odds": current_odds,  # Will be updated by TAB scraper
            "scratched": bool(entry.get("scratched")),
            "comments": entry.get("comment"),
            # New fields
            "horse_age": horse_age,
            "horse_sex": horse.get("sex"),
            "horse_colour": horse.get("colour"),
            "sire": sire,
            "dam": dam,
            "dam_sire": None,  # Not directly available in this query
            "career_prize_money": career_pm,
            "last_five": last_five,
            "days_since_last_run": days_since,
            "handicap_rating": handicap_rating,
            "speed_value": sv,
            "track_dist_stats": entry.get("trackDistanceStats"),
            "track_stats": entry.get("trackStats"),
            "distance_stats": entry.get("distanceStats"),
            "first_up_stats": horse.get("firstUpStats"),
            "second_up_stats": horse.get("secondUpStats"),
            "good_track_stats": horse.get("goodStats"),
            "soft_track_stats": horse.get("softStats"),
            "heavy_track_stats": horse.get("heavyStats"),
            "jockey_stats": entry.get("jockeyStats"),
            "class_stats": entry.get("atThisClassStats"),
            "gear": entry.get("lastGear"),
            "gear_changes": entry.get("gearChanges"),
            "stewards_comment": entry.get("commentStewards"),
            "comment_long": entry.get("comment"),
            "comment_short": entry.get("commentShort"),
            "odds_tab": odds_dict.get("odds_tab"),
            "odds_sportsbet": odds_dict.get("odds_sportsbet"),
            "odds_bet365": odds_dict.get("odds_bet365"),
            "odds_ladbrokes": odds_dict.get("odds_ladbrokes"),
            "odds_betfair": odds_dict.get("odds_betfair"),
            "odds_flucs": odds_dict.get("odds_flucs"),
            "trainer_location": None,  # Not in this query
        }

    def _parse_odds_array(self, odds_list: list) -> dict:
        """Parse the odds array from a GraphQL entry into our field names."""
        result = {}
        for o in odds_list:
            code = o.get("providerCode", "")
            field = PROVIDER_MAP.get(code)
            if field:
                val = self.parse_odds(o.get("oddsWin"))
                if val:
                    result[field] = val
            # Capture flucs from any provider that has them
            flucs = o.get("flucsWin")
            if flucs and "odds_flucs" not in result:
                flucs_data = [{"time": f.get("updateTime"), "odds": f.get("amount")}
                              for f in flucs[:20]]
                result["odds_flucs"] = _json.dumps(flucs_data)
        return result

    @staticmethod
    def _extract_weight_type(condition: str) -> Optional[str]:
        """Extract weight type from race condition string."""
        if not condition:
            return None
        low = condition.lower()
        if "set weights" in low:
            return "Set Weights"
        if "handicap" in low:
            return "Handicap"
        if "weight for age" in low:
            return "Weight For Age"
        if "quality" in low:
            return "Quality"
        return None

    @staticmethod
    def _extract_age_restriction(condition: str) -> Optional[str]:
        """Extract age restriction from race condition string."""
        if not condition:
            return None
        m = re.search(r"(Two|Three|Four|Five|Six)[\s-]Year[s]?[\s-]Old[s]?", condition, re.IGNORECASE)
        if m:
            age_words = {"two": "2yo", "three": "3yo", "four": "4yo", "five": "5yo", "six": "6yo"}
            word = m.group(1).lower()
            return age_words.get(word, m.group(0))
        m = re.search(r"(\d)yo\+?", condition, re.IGNORECASE)
        if m:
            return m.group(0)
        return None

    def _parse_iso_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO 8601 time string from GraphQL and convert to Melbourne local time.

        Racing.com returns UTC times for all venues (including interstate).
        We store as Melbourne local time (AEDT/AEST) since that's the display timezone.
        """
        if not time_str:
            return None
        try:
            from punty.config import MELB_TZ
            # Handle "2026-01-30T07:15:00.000Z" or "2026-01-30T07:15:00.0000000Z"
            cleaned = time_str.replace("Z", "+00:00")
            utc_dt = datetime.fromisoformat(cleaned)
            # Convert to Melbourne time and strip tzinfo for naive storage
            melb_dt = utc_dt.astimezone(MELB_TZ).replace(tzinfo=None)
            return melb_dt
        except Exception:
            return None

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse time string like '6:15pm' to datetime."""
        try:
            now = melb_now()
            for fmt in ["%I:%M%p", "%I:%M %p", "%H:%M"]:
                try:
                    parsed = datetime.strptime(time_str.strip().upper(), fmt.upper())
                    return parsed.replace(year=now.year,
                                         month=now.month,
                                         day=now.day)
                except ValueError:
                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _speed_value_to_position(speed_value) -> str | None:
        """Convert racing.com speedValue (1-12) to position label."""
        if speed_value is None:
            return None
        try:
            sv = int(speed_value)
        except (ValueError, TypeError):
            return None
        if sv >= 10:
            return "leader"
        if sv >= 7:
            return "on_pace"
        if sv >= 4:
            return "midfield"
        return "backmarker"

    async def scrape_speed_maps(self, venue: str, race_date: date, race_count: int) -> AsyncGenerator[dict, None]:
        """Scrape speed maps for all races by intercepting the GraphQL API.

        Racing.com loads speed map data via graphql.rmdprod.racing.com with a
        `speedValue` field per entrant (1=backmarker, 12=leader). We load each
        race page and intercept the GraphQL response containing formRaceEntries.
        """
        total = race_count + 1  # +1 for completion event

        async with new_page() as page:
            for race_num in range(1, race_count + 1):
                yield {"step": race_num - 1, "total": total,
                       "label": f"Fetching speed map for Race {race_num}...", "status": "running"}

                race_url = self._build_race_url(venue, race_date, race_num)
                positions = []

                try:
                    captured_entries = []

                    async def capture_graphql(response):
                        if "graphql.rmdprod.racing.com" not in response.url:
                            return
                        if "getRaceEntriesForField" not in response.url:
                            return
                        try:
                            body = await response.text()
                            data = _json.loads(body)
                            entries = data.get("data", {}).get("getRaceForm", {}).get("formRaceEntries", [])
                            if entries:
                                captured_entries.extend(entries)
                        except Exception:
                            pass

                    page.on("response", capture_graphql)

                    await page.goto(race_url, wait_until="load")
                    await page.wait_for_timeout(3000)  # Reduced from 5000

                    page.remove_listener("response", capture_graphql)

                    for entry in captured_entries:
                        if entry.get("scratched"):
                            continue
                        sv = entry.get("speedValue")
                        pos = self._speed_value_to_position(sv)
                        if pos:
                            positions.append({
                                "tab_number": entry.get("raceEntryNumber"),
                                "horse_name": entry.get("horseName", ""),
                                "position": pos,
                            })

                    count = len(positions)
                    if count > 0:
                        yield {"step": race_num, "total": total,
                               "label": f"Race {race_num}: {count} positions found", "status": "done",
                               "race_number": race_num, "positions": positions}
                    else:
                        yield {"step": race_num, "total": total,
                               "label": f"Race {race_num}: no speed map data found", "status": "done",
                               "race_number": race_num, "positions": []}

                except Exception as e:
                    logger.error(f"Error scraping speed map for race {race_num}: {e}")
                    try:
                        page.remove_listener("response", capture_graphql)
                    except Exception:
                        pass
                    yield {"step": race_num, "total": total,
                           "label": f"Race {race_num}: error - {e}", "status": "error",
                           "race_number": race_num, "positions": []}

        yield {"step": total, "total": total, "label": "Speed maps complete", "status": "complete"}

    async def check_race_statuses(self, venue: str, race_date: date) -> dict[int, str]:
        """Lightweight poll — single page load to get race statuses.

        Returns {1: "Open", 2: "Paying", ...}
        Status values: "Open", "Closed", "Interim", "Paying", "Abandoned"
        """
        meeting_url = self._build_meeting_url(venue, race_date)
        logger.info(f"Checking race statuses: {meeting_url}")

        statuses: dict[int, str] = {}

        async with new_page() as page:
            captured_races = []

            async def capture_graphql(response):
                if "graphql.rmdprod.racing.com" not in response.url:
                    return
                if "getRaceNumberList_CD" not in response.url:
                    return
                try:
                    body = await response.text()
                    data = _json.loads(body)
                    races = data.get("data", {}).get("getNoCacheRacesForMeet", [])
                    if races:
                        captured_races.extend(races)
                except Exception:
                    pass

            page.on("response", capture_graphql)
            await page.goto(meeting_url, wait_until="load")
            await page.wait_for_timeout(4000)
            page.remove_listener("response", capture_graphql)

        for race in captured_races:
            race_num = race.get("raceNumber")
            status = race.get("raceStatus") or race.get("status") or "Open"
            if race_num:
                statuses[race_num] = status

        logger.info(f"Race statuses for {venue}: {statuses}")
        return statuses

    async def scrape_race_result(self, venue: str, race_date: date, race_number: int) -> dict:
        """Scrape results for a single completed race.

        Returns: {race_number, winning_time, results: [{horse_name, saddlecloth,
        position, margin, starting_price, sectional_400, sectional_800}],
        exotics: {exacta, trifecta, quinella, first4}}
        """
        race_url = self._build_race_url(venue, race_date, race_number)
        logger.info(f"Scraping race result: {race_url}")

        entries_data: list = []
        betting_data: dict = {}

        async with new_page() as page:
            async def capture_graphql(response):
                if "graphql.rmdprod.racing.com" not in response.url:
                    return
                try:
                    body = await response.text()
                    data = _json.loads(body)
                except Exception:
                    return

                url = response.url

                if "getRaceEntriesForField" in url or "getRaceEntries" in url or "getRaceResults" in url:
                    form = data.get("data", {}).get("getRaceForm", {})
                    if not form:
                        form = data.get("data", {}).get("getRaceEntries", {})

                    # getRaceResults returns entries as a list directly under getRaceForm
                    if isinstance(form, list):
                        entries = form
                    else:
                        entries = form.get("formRaceEntries", []) if isinstance(form, dict) else []

                    if entries:
                        # Prefer entries that have position data (results)
                        has_positions = any(e.get("position") is not None for e in entries)
                        existing_has_positions = any(e.get("position") is not None for e in entries_data)
                        if not entries_data or (has_positions and not existing_has_positions):
                            entries_data.clear()
                            entries_data.extend(entries)

                # GetBettingData can be in its own response or bundled with getRaceResults
                bd = data.get("data", {}).get("GetBettingData", {})
                if bd:
                    betting_data.update(bd)

                # NOTE: removed greedy _try_extract_result_entries fallback
                # — it could grab entries from a different race's GraphQL response

            page.on("response", capture_graphql)
            await page.goto(race_url, wait_until="load")
            await page.wait_for_timeout(5000)

            # Try clicking Results tab
            try:
                for tab_text in ["Results", "Full Form", "Form"]:
                    btn = page.locator(f"button:has-text('{tab_text}'), a:has-text('{tab_text}')").first
                    if await btn.is_visible(timeout=1500):
                        await btn.click()
                        await page.wait_for_timeout(3000)
                        break
            except Exception:
                pass

            page.remove_listener("response", capture_graphql)

        # Parse results from entries
        results = []
        winning_time = None
        for entry in entries_data:
            horse_name = entry.get("horseName", "").strip()
            if not horse_name:
                continue

            position = entry.get("position")
            if position is not None:
                try:
                    position = int(position)
                except (ValueError, TypeError):
                    position = None

            margin = entry.get("margin")
            if margin is not None:
                margin = str(margin)

            if not winning_time:
                wt = entry.get("winningTime")
                if wt:
                    winning_time = str(wt)

            sp = entry.get("startingPrice")
            if sp is not None:
                sp = str(sp)

            # Extract win/place dividends from odds array (tote providers)
            win_div = self._parse_float(entry.get("dividendWin")) or self._parse_float(entry.get("winDividend"))
            place_div = self._parse_float(entry.get("dividendPlace")) or self._parse_float(entry.get("placeDividend"))
            if not win_div or not place_div:
                for odds_item in (entry.get("odds") or []):
                    provider = (odds_item.get("providerCode") or "").upper()
                    # Prefer tote providers: N (NSW), V (VIC), Q (QLD), BTOTE
                    if provider in ("N", "V", "Q", "BTOTE"):
                        if not win_div:
                            win_div = self._parse_dollar(odds_item.get("oddsWin"))
                        if not place_div:
                            place_div = self._parse_dollar(odds_item.get("oddsPlace"))
                        if win_div and place_div:
                            break

            results.append({
                "horse_name": horse_name,
                "saddlecloth": entry.get("raceEntryNumber"),
                "position": position,
                "margin": margin,
                "starting_price": sp,
                "sectional_400": str(entry.get("positionAt400", "")) or None,
                "sectional_800": str(entry.get("positionAt800", "")) or None,
                "win_dividend": win_div,
                "place_dividend": place_div,
            })

        # Parse exotics from betting data
        exotics = {}
        for exotic in (betting_data.get("exotics") or []):
            exotic_type = (exotic.get("poolType") or exotic.get("poolStatusCode") or exotic.get("type") or "").lower()
            dividend = exotic.get("dividend") or exotic.get("amount")
            if exotic_type and dividend:
                # Clean "$" from amount strings
                div_str = str(dividend).replace("$", "").replace(",", "").strip()
                exotics[exotic_type] = div_str

        # Sort by position
        results.sort(key=lambda r: r.get("position") or 999)

        return {
            "race_number": race_number,
            "winning_time": winning_time,
            "results": results,
            "exotics": exotics,
        }

    def _try_extract_result_entries(self, data: dict, entries_data: list) -> None:
        """Scan GraphQL response for result entries with position data."""
        def _find(obj, depth=0):
            if depth > 5:
                return
            if isinstance(obj, dict):
                for key in ["formRaceEntries", "raceEntries"]:
                    entries = obj.get(key)
                    if isinstance(entries, list) and len(entries) > 0:
                        if entries[0].get("position") is not None:
                            if not entries_data or entries_data[0].get("position") is None:
                                entries_data.clear()
                                entries_data.extend(entries)
                            return
                for v in obj.values():
                    _find(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _find(item, depth + 1)
        _find(data)

    async def scrape_sectional_times(
        self, venue: str, race_date: date, race_number: int, meet_code: Optional[str] = None
    ) -> Optional[dict]:
        """Scrape post-race sectional times data.

        This data is typically available 10-15 minutes after a race finishes.
        It shows actual running positions and times at each checkpoint.

        Args:
            venue: Venue name
            race_date: Date of the meeting
            race_number: Race number
            meet_code: Optional racing.com internal meeting ID for CSV fallback

        Returns: {
            race_number, has_sectionals, meet_code,
            horses: [{
                saddlecloth, horse_name, final_position,
                sectional_times: [{distance, position, time, avg_speed}],
                split_times: [{distance, position, time, avg_speed}],
                comment, race_time, beaten_margin, time_var_to_winner,
                six_hundred_time, two_hundred_time, distance_run,
                early_speed, mid_speed, late_speed, peak_speed, avg_speed
            }]
        }
        """
        race_url = self._build_race_url(venue, race_date, race_number)
        logger.info(f"Scraping sectional times: {race_url}")

        sectional_data = {}
        captured_meet_code = None

        async with new_page() as page:
            async def capture_sectionals(response):
                nonlocal captured_meet_code
                if "graphql.rmdprod.racing.com" not in response.url:
                    return
                try:
                    body = await response.text()
                    data = _json.loads(body)
                    # Look for sectionaltimes_callback
                    self._try_extract_sectional_times(data, sectional_data)
                    if sectional_data.get("horses"):
                        logger.debug(f"Captured sectional data for {len(sectional_data['horses'])} horses")
                    # Try to extract meet_code from getMeeting response
                    if "getMeeting" in body and not captured_meet_code:
                        meeting_data = data.get("data", {}).get("getMeeting", {})
                        if meeting_data and meeting_data.get("id"):
                            captured_meet_code = meeting_data["id"]
                except Exception as e:
                    logger.debug(f"Error parsing graphql response: {e}")

            page.on("response", capture_sectionals)
            try:
                # Load the race page first
                await page.goto(race_url, wait_until="load", timeout=30000)
                await page.wait_for_timeout(2000)

                # Click on the Speed Data tab to trigger sectional times GraphQL request
                try:
                    speed_tab = page.locator('text="Speed Data"').first
                    if await speed_tab.is_visible(timeout=3000):
                        await speed_tab.click()
                        await page.wait_for_timeout(4000)  # Wait for sectional data
                except Exception as e:
                    logger.debug(f"Could not click Speed Data tab: {e}")
            except Exception as e:
                logger.warning(f"Error loading sectional times page: {e}")
            finally:
                page.remove_listener("response", capture_sectionals)

        # If GraphQL scraping failed, try CSV fallback
        effective_meet_code = meet_code or captured_meet_code
        if not sectional_data.get("horses") and effective_meet_code:
            logger.info(f"Trying CSV fallback for {venue} R{race_number} (meet_code={effective_meet_code})")
            csv_data = await self._fetch_sectional_csv(effective_meet_code, race_number)
            if csv_data:
                sectional_data = csv_data
                sectional_data["source"] = "csv"

        if not sectional_data.get("horses"):
            logger.info(f"No sectional data found for {venue} R{race_number}")
            return None

        sectional_data["race_number"] = race_number
        if captured_meet_code:
            sectional_data["meet_code"] = captured_meet_code
        return sectional_data

    async def _fetch_sectional_csv(self, meet_code: str, race_number: int) -> Optional[dict]:
        """Fetch and parse sectional times from CSV download.

        CSV URL pattern: https://d3qmfyv6ad9vwv.cloudfront.net/{meet_code}_{race_number:02d}.csv
        """
        csv_url = f"https://d3qmfyv6ad9vwv.cloudfront.net/{meet_code}_{race_number:02d}.csv"
        logger.debug(f"Fetching sectional CSV: {csv_url}")

        try:
            response = await self.client.get(csv_url, timeout=10.0)
            if response.status_code != 200:
                logger.debug(f"CSV not available: {response.status_code}")
                return None

            csv_text = response.text
            return self._parse_sectional_csv(csv_text, race_number)
        except Exception as e:
            logger.debug(f"Failed to fetch sectional CSV: {e}")
            return None

    def _parse_sectional_csv(self, csv_text: str, race_number: int) -> Optional[dict]:
        """Parse sectional times CSV into our standard format.

        CSV format (semicolon-delimited):
        Row 1: date;venue-info;race_name;winning_time;track_info
        Row 2+: horse_name;saddlecloth;dist1;speed1;time1;dist2;speed2;time2;...
        """
        lines = csv_text.strip().split("\n")
        if len(lines) < 2:
            return None

        horses = []
        for i, line in enumerate(lines[1:], start=1):  # Skip header
            parts = line.split(";")
            if len(parts) < 5:
                continue

            horse_name = parts[0]
            try:
                saddlecloth = int(parts[1])
            except (ValueError, IndexError):
                continue

            # Parse distance/speed/time triplets (100m intervals)
            sectional_times = []
            split_times = []
            j = 2
            prev_time_seconds = 0.0
            while j + 2 < len(parts):
                try:
                    distance = int(parts[j])
                    speed = float(parts[j + 1])
                    time_str = parts[j + 2]  # Format: 00:00:08.640

                    # Parse time to seconds
                    time_parts = time_str.split(":")
                    if len(time_parts) == 3:
                        time_seconds = (
                            int(time_parts[0]) * 3600 +
                            int(time_parts[1]) * 60 +
                            float(time_parts[2])
                        )
                    else:
                        time_seconds = 0.0

                    # Cumulative time for sectionals
                    cumulative_time = prev_time_seconds + time_seconds if prev_time_seconds else time_seconds

                    sectional_times.append({
                        "distance": f"{distance}m",
                        "position": i,  # Use row order as rough position
                        "time": f"{cumulative_time:.2f}",
                        "avg_speed": speed,
                    })

                    # Split time for this segment
                    split_times.append({
                        "distance": f"{distance}m",
                        "position": i,
                        "time": f"{time_seconds:.2f}",
                        "avg_speed": speed,
                    })

                    prev_time_seconds = cumulative_time
                    j += 3
                except (ValueError, IndexError):
                    break

            if sectional_times:
                # Calculate early/mid/late speeds from the data
                speeds = [s["avg_speed"] for s in sectional_times if s.get("avg_speed")]
                third = len(speeds) // 3 if speeds else 0

                horse_data = {
                    "saddlecloth": saddlecloth,
                    "horse_name": horse_name,
                    "final_position": i,  # Approximate from row order
                    "race_time": f"{prev_time_seconds:.2f}" if prev_time_seconds else None,
                    "sectional_times": sectional_times,
                    "split_times": split_times,
                    # Calculate overview speeds
                    "early_speed": round(sum(speeds[:third]) / third, 2) if third > 0 else None,
                    "mid_speed": round(sum(speeds[third:2*third]) / third, 2) if third > 0 else None,
                    "late_speed": round(sum(speeds[2*third:]) / (len(speeds) - 2*third), 2) if len(speeds) > 2*third else None,
                    "avg_speed": round(sum(speeds) / len(speeds), 2) if speeds else None,
                }
                horses.append(horse_data)

        if horses:
            return {"horses": horses, "has_sectionals": True}
        return None

    def _try_extract_sectional_times(self, data: dict, result: dict) -> None:
        """Extract sectional times from GraphQL response."""
        def _find(obj, depth=0):
            if depth > 6:
                return
            if isinstance(obj, dict):
                # Check for sectionaltimes_callback data
                if "sectionaltimes_callback" in obj:
                    callback = obj["sectionaltimes_callback"]
                    if callback and "Horses" in callback:
                        horses = []
                        for h in callback["Horses"]:
                            horse_data = {
                                "saddlecloth": h.get("SaddleNumber"),
                                "horse_name": h.get("FullName"),
                                "final_position": h.get("FinalPosition"),
                                "final_position_abbr": h.get("FinalPositionAbbreviation"),
                                "barrier": h.get("BarrierNumber"),
                                "jockey": h.get("Jockey"),
                                "trainer": h.get("Trainer"),
                                "comment": h.get("Comment"),
                                "race_time": h.get("RaceTime"),
                                "beaten_margin": h.get("BeatenMargin"),
                                "time_var_to_winner": h.get("TimeVarToWinner"),
                                "distance_run": h.get("DistanceRun"),
                                "six_hundred_time": h.get("SixHundredMetresTime"),
                                "two_hundred_time": h.get("TwoHundredMetresTime"),
                                # Overview/Speed report data
                                "early_speed": h.get("Early"),
                                "mid_speed": h.get("Mid"),
                                "late_speed": h.get("Late"),
                                "peak_speed": h.get("OverallPeakSpeed"),
                                "peak_speed_location": h.get("PeakSpeedLocation"),
                                "avg_speed": h.get("OverallAvgSpeed"),
                                "distance_from_rail": h.get("DistanceFromRail"),
                                "sectional_times": [
                                    {
                                        "distance": s.get("Distance"),
                                        "position": s.get("Position"),
                                        "time": s.get("Time"),
                                        "avg_speed": s.get("AvgSpeed"),
                                    }
                                    for s in (h.get("SectionalTimes") or [])
                                ],
                                "split_times": [
                                    {
                                        "distance": s.get("Distance"),
                                        "position": s.get("Position"),
                                        "time": s.get("Time"),
                                        "avg_speed": s.get("AvgSpeed"),
                                    }
                                    for s in (h.get("SplitTimes") or [])
                                ],
                            }
                            horses.append(horse_data)

                        if horses:
                            result["horses"] = horses
                            result["has_sectionals"] = True
                        return

                # Check for hasSectionals flag in race data
                if obj.get("hasSectionals") is True:
                    result["has_sectionals"] = True

                for v in obj.values():
                    _find(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _find(item, depth + 1)

        _find(data)

    @staticmethod
    def _parse_float(val) -> Optional[float]:
        """Parse a value to float, returning None on failure."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_dollar(val) -> Optional[float]:
        """Parse a dollar string like '$3.60' to float."""
        if not val:
            return None
        try:
            return float(str(val).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Scrape all race results for a venue."""
        statuses = await self.check_race_statuses(venue, race_date)
        results = []
        for race_num, status in sorted(statuses.items()):
            if status in ("Paying", "Closed", "Interim"):
                result = await self.scrape_race_result(venue, race_date, race_num)
                results.append(result)
        return results
