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

    def _venue_slug(self, venue: str, strip_sponsor: bool = True) -> str:
        """Convert venue name to URL slug, optionally stripping sponsor prefixes."""
        from punty.venues import normalize_venue, venue_slug as _vs
        if strip_sponsor:
            return _vs(venue)
        # No sponsor stripping — just lowercase + dash
        return venue.lower().strip().replace(" ", "-")

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
        tips_by_race: dict[int, list] = {}  # race_number -> list of tipster picks

        async with new_page() as page:
            # --- GraphQL response interceptor ---
            async def capture_graphql(response):
                if "graphql.rmdprod.racing.com" not in response.url:
                    return
                try:
                    body = await response.text()
                    data = _json.loads(body)
                except Exception:
                    # Protocol errors ("No resource with given identifier") are
                    # expected when navigating away before response body is read.
                    # Suppress — these are not actionable warnings.
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

                # Capture tips/expert selections from any tips-related query
                self._try_extract_tips(data, url, tips_by_race)

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
                        # Log unknown queries with data keys (once per type)
                        url_tail = response.url.split("/")[-1][:80]
                        if url_tail not in captured_queries:
                            captured_queries.add(url_tail)
                            try:
                                body = await response.text()
                                d = _json.loads(body)
                                keys = list((d.get("data") or {}).keys())
                                if keys:
                                    logger.info(f"Unknown GraphQL query '{url_tail}': data keys={keys}")
                            except Exception:
                                pass
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
                    # Retry with full sponsor-prefixed slug (e.g., "ladbrokes-cannon-park")
                    full_slug = self._venue_slug(venue, strip_sponsor=False)
                    stripped_slug = self._venue_slug(venue, strip_sponsor=True)
                    if full_slug != stripped_slug:
                        alt_url = f"{self.BASE_URL}/form/{race_date.isoformat()}/{full_slug}"
                        logger.info(f"No races with stripped slug, retrying with full slug: {alt_url}")
                        resp = await page.goto(alt_url, wait_until="load")
                        if resp and resp.status != 404:
                            await page.wait_for_timeout(3000)
                            race_count = len(race_list_gql)
                            if race_count == 0:
                                race_count = await page.evaluate("""() => {
                                    var links = document.querySelectorAll('a[href*="/race/"]');
                                    var maxNum = 0;
                                    for (var i = 0; i < links.length; i++) {
                                        var m = links[i].href.match(/\\/race\\/(\\d+)$/);
                                        if (m) { var n = parseInt(m[1]); if (n > maxNum) maxNum = n; }
                                    }
                                    return maxNum;
                                }""")
                            if race_count > 0:
                                # Update meeting_url for subsequent navigation
                                meeting_url = alt_url
                                logger.info(f"Found {race_count} races with full slug {full_slug}")

                if race_count == 0:
                    logger.warning(f"No races found on {meeting_url}")
                    return {"meeting": self._build_meeting_dict(meeting_id, venue, race_date, meeting_gql),
                            "races": [], "runners": []}

                logger.info(f"Found {race_count} races for {venue}")

                # 2. Load "All" view to capture all race entries + odds in one page
                all_url = f"{meeting_url}#/"
                logger.info(f"Loading All view: {all_url}")
                await page.goto(all_url, wait_until="domcontentloaded")

                # Wait for entries to start appearing
                try:
                    await page.wait_for_selector("[class*='runner'], [class*='entry'], [class*='field']", timeout=10000)
                except Exception:
                    pass
                await page.wait_for_timeout(3000)

                # Dismiss cookie banner
                try:
                    btn = page.locator("button:has-text('Decline')").first
                    if await btn.is_visible(timeout=1500):
                        await btn.click()
                except Exception:
                    pass

                # Scroll through the All view to trigger lazy loading of all race entries
                page_height = await page.evaluate("document.body.scrollHeight")
                scroll_step = 400
                scroll_steps = max(30, page_height // scroll_step)
                for _ in range(scroll_steps):
                    await page.evaluate(f"window.scrollBy(0, {scroll_step})")
                    await page.wait_for_timeout(600)

                # Second pass — scroll back up and down to catch any missed lazy loads
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(500)
                for _ in range(scroll_steps):
                    await page.evaluate(f"window.scrollBy(0, {scroll_step})")
                    await page.wait_for_timeout(500)

                # Check which races we captured — do targeted per-race loads for any missing
                missing_races = [n for n in range(1, race_count + 1) if n not in race_entries_by_num]
                if missing_races:
                    logger.info(f"All view missed {len(missing_races)} races: {missing_races}. Loading individually...")
                    for race_num in missing_races:
                        try:
                            race_url = f"{meeting_url}/race/{race_num}"
                            await page.goto(race_url, wait_until="domcontentloaded")
                            await page.wait_for_timeout(4000)
                            # Scroll to trigger entry loading
                            for _ in range(10):
                                await page.evaluate("window.scrollBy(0, 400)")
                                await page.wait_for_timeout(500)
                        except Exception as e:
                            logger.error(f"Race {race_num} fallback load failed: {e}")

                captured_count = len(race_entries_by_num)
                logger.info(f"Captured entries for {captured_count}/{race_count} races from All view")

                # 3. Quick pass through each race page to trigger odds queries
                # The All view only fires getRaceForm (no odds). Per-race pages
                # fire getRaceEntriesForField_CD which includes odds per entry.
                logger.info(f"Loading per-race pages for odds ({race_count} races)...")
                for race_num in range(1, race_count + 1):
                    try:
                        race_url = f"{meeting_url}/race/{race_num}"
                        await page.goto(race_url, wait_until="domcontentloaded")
                        await page.wait_for_timeout(2000)
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "closed" in err_msg or "disposed" in err_msg:
                            logger.warning(f"Page closed during odds fetch at R{race_num}, stopping")
                            break
                        logger.warning(f"Race {race_num} odds fetch failed: {e}")

                # Load expert-tips page to capture all race tips in one go
                try:
                    tips_url = f"{meeting_url}#/expert-tips"
                    logger.info(f"Loading expert tips page: {tips_url}")
                    await page.goto(tips_url, wait_until="domcontentloaded")
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                    await page.wait_for_timeout(1000)
                    race_level_tips = [k for k in tips_by_race if k != 0]
                    logger.info(f"Expert tips page loaded, race-level tips: {len(race_level_tips)} races")
                except Exception as e:
                    logger.warning(f"Expert tips page failed: {e}")

                # Fallback: if no race-level tips from overview page, try per-race tips tabs
                race_level_tips = [k for k in tips_by_race if k != 0]
                if not race_level_tips:
                    logger.info("No race-level tips from overview — falling back to per-race tips tabs")
                    for race_num in range(1, race_count + 1):
                        try:
                            tips_url = f"{meeting_url}/race/{race_num}#/tips"
                            await page.goto(tips_url, wait_until="domcontentloaded")
                            await page.wait_for_timeout(3000)
                        except Exception as e:
                            err_msg = str(e).lower()
                            if "closed" in err_msg or "disposed" in err_msg:
                                logger.warning(f"Page closed during tips fallback at R{race_num}, stopping")
                                break
                            logger.warning(f"Race {race_num} tips fallback failed: {e}")
                    race_level_tips = [k for k in tips_by_race if k != 0]
                    logger.info(f"Per-race tips fallback captured tips for {len(race_level_tips)} races")
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

        # Distribute unmatched meeting-level tips (key 0) to races by horse name lookup
        unmatched_tips = tips_by_race.pop(0, [])
        if unmatched_tips:
            # Build horse->race_num lookup from captured entries
            horse_to_race: dict[str, int] = {}
            for rn, entries in race_entries_by_num.items():
                for entry in entries:
                    hname = entry.get("horseName", "").strip()
                    if hname:
                        horse_to_race[hname.lower()] = rn
            distributed = 0
            for tip in unmatched_tips:
                horse = (tip.get("horse") or "").strip().lower()
                if horse and horse in horse_to_race:
                    race_num = horse_to_race[horse]
                    tips_by_race.setdefault(race_num, []).append(tip)
                    distributed += 1
                    logger.info(f"Distributed meeting tip '{tip.get('pick_type')}' {tip.get('horse')} → R{race_num}")
            if distributed:
                logger.info(f"Distributed {distributed}/{len(unmatched_tips)} meeting-level tips to races")

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

            # Attach expert tips if captured
            race_tips = tips_by_race.get(race_num, [])
            if race_tips:
                race_data["expert_tips"] = race_tips

            races.append(race_data)
            runners.extend(race_runners)

        if tips_by_race:
            race_nums = list(tips_by_race.keys())
            logger.info(f"Expert tips captured for {len(race_nums)} races: {race_nums}")

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

    def _try_extract_tips(self, data: dict, url: str, tips_by_race: dict) -> None:
        """Extract expert tips from racing.com GraphQL responses.

        Three known sources:
        1. getRaceForm.raceTips[] — per-race tipster picks (race number from
           sibling formRaceEntries)
        2. Meet.meetTips[] — meeting-level bestBet/bestRoughie/bestValue with
           comments (from GetRaceEntryBadges_CD)
        3. GetBetEasyMeetTipByMeetCode[] — per-race tipster selections from
           the expert-tips/tips pages (from GetTipsters query)
        """
        payload = data.get("data", {})
        if not payload:
            return

        # --- 1. Race-level tips from getRaceForm.raceTips ---
        race_form = payload.get("getRaceForm", {})
        if isinstance(race_form, dict):
            race_tips = race_form.get("raceTips", [])
            if isinstance(race_tips, list) and race_tips:
                # Get race number from sibling formRaceEntries
                entries = race_form.get("formRaceEntries", [])
                race_num = None
                if entries and isinstance(entries, list):
                    race_num = entries[0].get("raceNumber")

                for tip_obj in race_tips:
                    if not isinstance(tip_obj, dict):
                        continue
                    tipster_info = tip_obj.get("tipster", {})
                    if isinstance(tipster_info, dict):
                        tipster_name = tipster_info.get("tipsterName", "Unknown")
                    else:
                        tipster_name = str(tipster_info) if tipster_info else "Unknown"

                    # Selections are nested arrays
                    selections = tip_obj.get("selections", [])
                    if isinstance(selections, list) and selections and race_num:
                        parsed = []
                        for i, sel in enumerate(selections):
                            if isinstance(sel, dict):
                                parsed.append({
                                    "tipster": tipster_name,
                                    "horse": sel.get("horseName") or sel.get("name"),
                                    "number": sel.get("raceEntryNumber") or sel.get("tabNumber"),
                                    "rank": i + 1,
                                    "comment": sel.get("comment"),
                                })
                        if parsed:
                            tips_by_race.setdefault(race_num, []).extend(parsed)
                            logger.info(
                                f"Race {race_num}: captured {len(parsed)} tips from {tipster_name}"
                            )

        # --- 2. Meeting-level tips from Meet.meetTips ---
        meet = payload.get("Meet", {})
        if isinstance(meet, dict):
            meet_tips = meet.get("meetTips", [])
            if isinstance(meet_tips, list) and meet_tips:
                for tip_obj in meet_tips:
                    if not isinstance(tip_obj, dict):
                        continue
                    tipster_info = tip_obj.get("tipster", {})
                    if isinstance(tipster_info, dict):
                        tipster_name = tipster_info.get("tipsterName", "Unknown")
                    else:
                        continue

                    # Meeting-level picks: bestBet, bestRoughie, bestValue, getOn
                    for pick_type in ["bestBet", "bestRoughie", "bestValue", "getOn"]:
                        pick = tip_obj.get(pick_type)
                        if not isinstance(pick, dict):
                            continue
                        comment = pick.get("comment")
                        horse_name = pick.get("horseName") or pick.get("name")
                        race_num = pick.get("raceNumber")
                        entry_code = pick.get("raceEntryItemCode")
                        if not (comment or horse_name):
                            continue
                        tip_data = {
                            "tipster": tipster_name,
                            "pick_type": pick_type,
                            "comment": comment,
                            "horse": horse_name,
                            "number": pick.get("raceEntryNumber") or pick.get("tabNumber"),
                            "rank": pick_type,  # bestBet/bestRoughie/bestValue as rank
                        }
                        if race_num:
                            # Attach directly to the race
                            tips_by_race.setdefault(race_num, []).append(tip_data)
                            logger.info(f"Meeting tip {pick_type} from {tipster_name}: {horse_name} (R{race_num})")
                        else:
                            # Store at meeting level if no race number
                            tip_data["entry_code"] = entry_code
                            tips_by_race.setdefault(0, []).append(tip_data)

                    # Log that we captured meeting tips
                    has_content = any(
                        isinstance(tip_obj.get(pt), dict) and tip_obj.get(pt, {}).get("comment")
                        for pt in ["bestBet", "bestRoughie", "bestValue", "getOn"]
                    )
                    if has_content:
                        logger.info(f"Meeting tips from {tipster_name}: bestBet/roughie/value captured")

        # --- 3. Per-race tipster selections from GetBetEasyMeetTipByMeetCode ---
        # Structure: [{tipster: {tipsterName}, bestBet: {horseName, raceNumber, ...},
        #              bestValue: {...}, nextBestTips: [{horseName, raceNumber}, ...]}]
        beteasy = payload.get("GetBetEasyMeetTipByMeetCode")
        if isinstance(beteasy, list) and beteasy:
            logger.info(f"BetEasy raw: {len(beteasy)} tip objects")
            for tip_obj in beteasy:
                if not isinstance(tip_obj, dict):
                    logger.info(f"  BetEasy obj not a dict: {type(tip_obj).__name__}")
                    continue
                # Debug: dump top-level keys and pick field types
                logger.info(f"  BetEasy obj keys={list(tip_obj.keys())}")
                for fld in ["bestBet", "bestValue", "nextBestTips"]:
                    val = tip_obj.get(fld)
                    if val is None:
                        logger.info(f"    {fld}: None")
                    elif isinstance(val, dict):
                        logger.info(f"    {fld}: dict keys={list(val.keys())}")
                    elif isinstance(val, list):
                        first_info = ""
                        if val and isinstance(val[0], dict):
                            first_info = f" first_keys={list(val[0].keys())}"
                        logger.info(f"    {fld}: list[{len(val)}]{first_info}")
                    else:
                        logger.info(f"    {fld}: {type(val).__name__}={str(val)[:80]}")

                tipster_info = tip_obj.get("tipster", {}) or {}
                tipster_name = (
                    tipster_info.get("tipsterName")
                    if isinstance(tipster_info, dict)
                    else tip_obj.get("tipsterName") or "Unknown"
                )

                parsed_count = 0

                def _extract_horse(pick: dict) -> str | None:
                    """Get horse name from a BetEasy pick dict.

                    Horse name can be directly on the pick or nested
                    inside raceEntryItem sub-dict.
                    """
                    horse = pick.get("horseName") or pick.get("name")
                    if not horse:
                        entry_item = pick.get("raceEntryItem")
                        if isinstance(entry_item, dict):
                            horse = entry_item.get("horseName") or entry_item.get("name")
                    return horse

                # Parse single-pick fields: bestBet, bestValue, bestRoughie, getOn
                for pick_type in ["bestBet", "bestValue", "bestRoughie", "getOn"]:
                    pick = tip_obj.get(pick_type)
                    if not isinstance(pick, dict):
                        continue
                    horse = _extract_horse(pick)
                    race_num = pick.get("raceNumber")
                    comment = pick.get("comment") or tip_obj.get("tipComment")
                    if horse:
                        tip_data = {
                            "tipster": tipster_name,
                            "pick_type": pick_type,
                            "horse": horse,
                            "number": pick.get("raceEntryNumber") or pick.get("tabNumber"),
                            "rank": pick_type,
                            "comment": comment,
                        }
                        if race_num:
                            tips_by_race.setdefault(race_num, []).append(tip_data)
                        else:
                            tip_data["entry_code"] = pick.get("raceEntryItemCode")
                            tips_by_race.setdefault(0, []).append(tip_data)
                        parsed_count += 1

                # Parse nextBestTips (can be a list of picks OR a single dict)
                next_best = tip_obj.get("nextBestTips")
                if isinstance(next_best, dict):
                    next_best = [next_best]
                if isinstance(next_best, list):
                    for i, pick in enumerate(next_best):
                        if not isinstance(pick, dict):
                            continue
                        horse = _extract_horse(pick)
                        race_num = pick.get("raceNumber")
                        if horse:
                            tip_data = {
                                "tipster": tipster_name,
                                "pick_type": "nextBest",
                                "horse": horse,
                                "number": pick.get("raceEntryNumber") or pick.get("tabNumber"),
                                "rank": f"nextBest_{i+1}",
                                "comment": pick.get("comment"),
                            }
                            if race_num:
                                tips_by_race.setdefault(race_num, []).append(tip_data)
                            else:
                                tip_data["entry_code"] = pick.get("raceEntryItemCode")
                                tips_by_race.setdefault(0, []).append(tip_data)
                            parsed_count += 1

                if parsed_count:
                    logger.info(f"BetEasy tips: {parsed_count} selections from {tipster_name}")

                # Also capture tipComment/shortComment/longComment as meeting-level context
                tip_comment = tip_obj.get("tipComment") or tip_obj.get("longComment") or tip_obj.get("shortComment")
                if tip_comment and not parsed_count:
                    logger.info(f"BetEasy comment from {tipster_name}: {tip_comment[:80]}")

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
                # Extract place odds from TAB provider
                if code == "Q" and "place_odds" not in odds_dict:
                    place_val = self.parse_odds(bp.get("oddsPlace"))
                    if place_val:
                        odds_dict["place_odds"] = place_val
                # Capture flucs from first provider that has them
                if bp.get("flucsWin") and "odds_flucs" not in odds_dict:
                    flucs = [{"time": f.get("updateTime"), "odds": f.get("amount")}
                             for f in bp["flucsWin"][:20]]
                    odds_dict["odds_flucs"] = _json.dumps(flucs)

        # Primary odds — cross-validate TAB against other providers
        current_odds = self._resolve_current_odds(odds_dict, horse_name)

        # If TAB was flagged unreliable, discard its place odds too
        if "odds_tab_raw" in odds_dict:
            odds_dict.pop("place_odds", None)

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
            "place_odds": odds_dict.get("place_odds"),
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
                # Extract place odds from TAB provider
                if code == "Q" and "place_odds" not in result:
                    place_val = self.parse_odds(o.get("oddsPlace"))
                    if place_val:
                        result["place_odds"] = place_val
            # Capture flucs from any provider that has them
            flucs = o.get("flucsWin")
            if flucs and "odds_flucs" not in result:
                flucs_data = [{"time": f.get("updateTime"), "odds": f.get("amount")}
                              for f in flucs[:20]]
                result["odds_flucs"] = _json.dumps(flucs_data)
        return result

    # TAB maximum payout odds ceiling — any odds above this are garbage data
    MAX_VALID_ODDS = 501.0

    def _resolve_current_odds(self, odds_dict: dict, horse_name: str) -> Optional[float]:
        """Pick the best current_odds using Sportsbet as primary provider.

        Sportsbet has 24% avg error vs TAB's 334% from racing.com GraphQL.
        TAB odds from racing.com are unreliable across many venues (not just WA).
        Rejects any odds above MAX_VALID_ODDS ($501) as garbage data.
        """
        sb = odds_dict.get("odds_sportsbet")
        tab = odds_dict.get("odds_tab")

        # Reject garbage odds above TAB ceiling
        if sb and isinstance(sb, (int, float)) and sb > self.MAX_VALID_ODDS:
            logger.warning(f"Rejecting garbage SB odds for {horse_name}: ${sb:.2f}")
            sb = None
        if tab and isinstance(tab, (int, float)) and tab > self.MAX_VALID_ODDS:
            logger.warning(f"Rejecting garbage TAB odds for {horse_name}: ${tab:.2f}")
            tab = None

        # Primary: Sportsbet
        if sb and isinstance(sb, (int, float)) and sb > 1.0:
            # Cross-validate TAB against SB and fix if wildly off
            if tab and isinstance(tab, (int, float)) and tab > 1.0:
                ratio = tab / sb
                if ratio > 3.0 or ratio < 0.33:
                    logger.warning(
                        f"Odds mismatch for {horse_name}: TAB=${tab:.2f} vs "
                        f"SB=${sb:.2f} (ratio={ratio:.2f}). Using SB."
                    )
                    odds_dict["odds_tab_raw"] = tab
                    odds_dict["odds_tab"] = sb
            return sb

        # Fallback: TAB (only when SB unavailable)
        if tab and isinstance(tab, (int, float)) and tab > 1.0:
            return tab

        # Last resort: any other provider (also capped)
        for k in ["odds_bet365", "odds_ladbrokes", "odds_betfair"]:
            v = odds_dict.get(k)
            if v and isinstance(v, (int, float)) and 1.0 < v <= self.MAX_VALID_ODDS:
                return v

        return None

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

    async def check_race_statuses(self, venue: str, race_date: date) -> dict:
        """Lightweight poll — single page load to get race statuses + track condition.

        Returns {"statuses": {1: "Open", 2: "Paying", ...}, "track_condition": str | None}
        Status values: "Open", "Closed", "Interim", "Paying", "Abandoned"
        """
        meeting_url = self._build_meeting_url(venue, race_date)
        logger.info(f"Checking race statuses: {meeting_url}")

        captured_races = []
        captured_meeting = {}

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
                if "getRaceNumberList_CD" in url:
                    races = data.get("data", {}).get("getNoCacheRacesForMeet", [])
                    if races:
                        captured_races.extend(races)
                if "getMeeting_CD" in url:
                    gm = data.get("data", {}).get("getMeeting", {})
                    if gm:
                        captured_meeting.update(gm)

            page.on("response", capture_graphql)
            await page.goto(meeting_url, wait_until="load")
            await page.wait_for_timeout(4000)

            # Retry with full sponsor-prefixed slug if no data captured
            if not captured_races:
                full_slug = self._venue_slug(venue, strip_sponsor=False)
                stripped_slug = self._venue_slug(venue, strip_sponsor=True)
                if full_slug != stripped_slug:
                    alt_url = f"{self.BASE_URL}/form/{race_date.isoformat()}/{full_slug}"
                    logger.info(f"No statuses with stripped slug, retrying: {alt_url}")
                    await page.goto(alt_url, wait_until="load")
                    await page.wait_for_timeout(4000)

            page.remove_listener("response", capture_graphql)

        statuses: dict[int, str] = {}
        for race in captured_races:
            race_num = race.get("raceNumber")
            status = race.get("raceStatus") or race.get("status") or "Open"
            if race_num:
                statuses[race_num] = status

        track_condition = None
        if captured_meeting:
            tc = captured_meeting.get("trackCondition", "")
            tr = captured_meeting.get("trackRating", "")
            # Combine condition + rating for consistency with initial scrape format
            track_condition = f"{tc} {tr}".strip() if tc else None
            if not track_condition:
                rail_tc = captured_meeting.get("railAndTrackCondition")
                track_condition = rail_tc

        logger.info(f"Race statuses for {venue}: {statuses}" + (f" | Track: {track_condition}" if track_condition else ""))
        return {"statuses": statuses, "track_condition": track_condition}

    async def check_race_fields(
        self, venue: str, race_date: date, race_numbers: list[int]
    ) -> dict:
        """Lightweight poll — load race field pages to capture jockey/gear/scratching/odds changes.

        Returns:
            {
                "meeting": {"track_condition": str | None},
                "races": {
                    race_num: [
                        {
                            "horse_name": str,
                            "saddlecloth": int,
                            "scratched": bool,
                            "jockey": str | None,
                            "gear": str | None,
                            "gear_changes": str | None,
                            "odds": dict | None,  # {odds_tab, odds_sportsbet, ...}
                        }
                    ]
                }
            }
        """
        result: dict = {"meeting": {}, "races": {}}
        if not race_numbers:
            return result

        async with new_page() as page:
            meeting_tc = None

            # Shared accumulators — keyed by race number
            entries_by_race: dict[int, list] = {}
            betting_by_race: dict[int, list] = {}
            captured_meeting: dict = {}
            current_race: list[int] = [race_numbers[0]]  # mutable ref for closure

            async def capture_graphql(response):
                if "graphql.rmdprod.racing.com" not in response.url:
                    return
                try:
                    body = await response.text()
                    data = _json.loads(body)
                except Exception:
                    return

                url = response.url
                rn = current_race[0]

                if "getMeeting_CD" in url:
                    gm = data.get("data", {}).get("getMeeting", {})
                    if gm:
                        captured_meeting.update(gm)

                if "getRaceEntriesForField" in url or "getRaceEntries" in url:
                    form = data.get("data", {}).get("getRaceForm", {})
                    entries = form.get("formRaceEntries", []) if isinstance(form, dict) else []
                    if entries:
                        # Determine race number from entries if possible
                        entry_rn = entries[0].get("raceNumber") if entries else None
                        key = entry_rn or rn
                        entries_by_race.setdefault(key, []).extend(entries)

                if "GetBettingData" in url:
                    bd = data.get("data", {}).get("GetBettingData", {})
                    entries = bd.get("formRaceEntries", []) if isinstance(bd, dict) else []
                    if entries:
                        # Determine race number from entries
                        entry_rn = entries[0].get("raceNumber") if entries else None
                        key = entry_rn or rn
                        betting_by_race.setdefault(key, []).extend(entries)

            page.on("response", capture_graphql)

            try:
                for race_num in race_numbers:
                    current_race[0] = race_num
                    try:
                        race_url = self._build_race_url(venue, race_date, race_num)
                        # Navigate to blank first to force full SPA reload
                        if race_num > race_numbers[0]:
                            await page.goto("about:blank", wait_until="domcontentloaded")
                            await page.wait_for_timeout(300)
                        await page.goto(race_url, wait_until="domcontentloaded")
                        await page.wait_for_timeout(4000)

                        # Click odds element to trigger GetBettingData_CD (lazy-loaded)
                        try:
                            odds_el = page.locator("[class*=odds]").first
                            if await odds_el.is_visible(timeout=2000):
                                await odds_el.click()
                                await page.wait_for_timeout(3000)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug(f"Failed to load R{race_num} fields: {e}")
                        continue
            finally:
                page.remove_listener("response", capture_graphql)

            # Parse meeting track condition
            if captured_meeting:
                tc = captured_meeting.get("trackCondition", "")
                tr = captured_meeting.get("trackRating", "")
                meeting_tc = f"{tc} {tr}".strip() if tc else None
                if not meeting_tc:
                    rail_tc = captured_meeting.get("railAndTrackCondition")
                    meeting_tc = rail_tc

            # Build results per race
            for race_num in race_numbers:
                entries = entries_by_race.get(race_num, [])
                betting = betting_by_race.get(race_num, [])

                # Build odds lookup from betting data
                odds_by_horse: dict[str, dict] = {}
                for entry in betting:
                    name = (entry.get("horseName") or "").strip()
                    odds_arr = entry.get("oddsByProvider", [])
                    if name and odds_arr:
                        odds_by_horse[name] = self._parse_odds_array(odds_arr)

                runners = []
                for entry in entries:
                    horse_name = (entry.get("horseName") or "").strip()
                    if not horse_name:
                        continue
                    runners.append({
                        "horse_name": horse_name,
                        "saddlecloth": entry.get("raceEntryNumber"),
                        "scratched": bool(entry.get("scratched")),
                        "jockey": entry.get("jockeyName"),
                        "gear": entry.get("lastGear"),
                        "gear_changes": entry.get("gearChanges"),
                        "odds": odds_by_horse.get(horse_name),
                    })

                odds_count = sum(1 for r in runners if r.get("odds"))
                result["races"][race_num] = runners
                logger.debug(f"Field check {venue} R{race_num}: {len(runners)} runners, {odds_count} with odds")

            result["meeting"]["track_condition"] = meeting_tc

        return result

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

            # Retry with full sponsor-prefixed slug if no entries captured
            if not entries_data:
                full_slug = self._venue_slug(venue, strip_sponsor=False)
                stripped_slug = self._venue_slug(venue, strip_sponsor=True)
                if full_slug != stripped_slug:
                    alt_url = f"{self.BASE_URL}/form/{race_date.isoformat()}/{full_slug}/race/{race_number}"
                    logger.info(f"No result entries with stripped slug, retrying: {alt_url}")
                    await page.goto(alt_url, wait_until="load")
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

            # Extract win/place dividends — check multiple field names
            # Use 'is None' not truthiness: 0.0 is falsy but means "field present, no dividend"
            win_div = self._parse_float(entry.get("dividendWin"))
            if win_div is None:
                win_div = self._parse_float(entry.get("winDividend"))
            place_div = self._parse_float(entry.get("dividendPlace"))
            if place_div is None:
                place_div = self._parse_float(entry.get("placeDividend"))

            # Fallback: scan odds array for tote provider dividends
            if win_div is None or place_div is None:
                for odds_item in (entry.get("odds") or []):
                    provider = (odds_item.get("providerCode") or "").upper()
                    # Tote providers: N (NSW), V (VIC), Q (QLD), BTOTE, NSW, VIC, QLD
                    if provider in ("N", "V", "Q", "BTOTE", "NSW", "VIC", "QLD"):
                        if win_div is None:
                            win_div = self._parse_dollar(odds_item.get("oddsWin"))
                        if place_div is None:
                            place_div = self._parse_dollar(odds_item.get("oddsPlace"))
                        if win_div is not None and place_div is not None:
                            break

            # Filter out zero dividends (0.0 = not yet finalized, not a real payout)
            if win_div is not None and win_div <= 0:
                win_div = None
            if place_div is not None and place_div <= 0:
                place_div = None

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
        status_data = await self.check_race_statuses(venue, race_date)
        statuses = status_data["statuses"]
        results = []
        for race_num, status in sorted(statuses.items()):
            if status in ("Paying", "Closed", "Interim"):
                result = await self.scrape_race_result(venue, race_date, race_num)
                results.append(result)
        return results
