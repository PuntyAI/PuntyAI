"""TAB Playwright scraper for international venue odds (HK, etc.).

Uses Playwright to load TAB race pages and intercepts API responses
from api.beta.tab.com.au, extracting structured JSON odds data.
This bypasses Akamai bot protection that blocks direct httpx/curl requests.

Also includes HKJC sectional times scraper for Hong Kong races.
"""

import asyncio
import json
import logging
import re
from datetime import date

from punty.venues import get_tab_mnemonic, normalize_venue

logger = logging.getLogger(__name__)

MAX_VALID_ODDS = 501.0  # TAB maximum payout odds ceiling


class TabPlaywrightScraper:
    """Scrape TAB fixed odds for international venues via Playwright API interception."""

    BASE_URL = "https://www.tab.com.au"
    API_HOST = "api.beta.tab.com.au"

    async def scrape_odds_for_meeting(
        self,
        venue: str,
        race_date: date,
        meeting_id: str,
        race_count: int,
    ) -> list[dict]:
        """Scrape TAB fixed odds for all races at an international venue.

        Returns list of dicts: {race_number, horse_name, saddlecloth,
        current_odds, opening_odds, place_odds, scratched}
        """
        tab_info = get_tab_mnemonic(venue)
        if not tab_info:
            logger.info(f"No TAB mnemonic for {venue} — skipping TAB Playwright")
            return []

        mnemonic, jurisdiction, url_slug = tab_info
        date_str = race_date.strftime("%Y-%m-%d")

        # Accumulate odds from intercepted API responses
        captured_odds: list[dict] = []
        capture_done = asyncio.Event()

        async def capture_tab_api(response):
            """Intercept TAB API responses containing runner/odds data."""
            url = response.url

            # Log all API-like responses for diagnostics
            if ("api" in url or "tab" in url) and response.status == 200:
                content_type = response.headers.get("content-type", "")
                if "json" in content_type or "javascript" in content_type:
                    if self.API_HOST not in url and "tab.com.au" in url:
                        logger.debug(f"TAB non-API response: {url[:200]}")

            # Match TAB API responses — try both known API hosts
            is_tab_api = self.API_HOST in url or "webapi.tab.com.au" in url
            if not is_tab_api:
                return

            # Accept any tab-info-service or racing-related endpoint
            if "tab-info-service" not in url and "racing" not in url and "meetings" not in url:
                logger.debug(f"TAB API non-racing response: {url[:200]}")
                return

            try:
                body = await response.text()
                data = json.loads(body)
            except Exception:
                return

            # TAB API can return meeting-level or race-level data
            # Meeting-level: has "races" array directly
            # Race-level: has "runners" array directly
            races = []

            if "races" in data:
                races = data["races"]
            elif isinstance(data, dict) and "runners" in data:
                races = [data]

            for race in races:
                race_num = race.get("raceNumber")
                if not race_num:
                    continue

                for runner in race.get("runners", []):
                    fixed_odds = runner.get("fixedOdds", {})
                    if not fixed_odds:
                        continue

                    win_odds = fixed_odds.get("returnWin")
                    place_odds = fixed_odds.get("returnPlace")
                    opening_odds = fixed_odds.get("openingPrice")

                    # Apply MAX_VALID_ODDS guard
                    if win_odds and win_odds > MAX_VALID_ODDS:
                        win_odds = None
                    if place_odds and place_odds > MAX_VALID_ODDS:
                        place_odds = None
                    if opening_odds and opening_odds > MAX_VALID_ODDS:
                        opening_odds = None

                    runner_name = runner.get("runnerName", "")
                    runner_number = runner.get("runnerNumber")
                    scratched = runner.get("fixedOdds", {}).get("isFavScratchings") or runner.get("scratched", False)

                    if not runner_name:
                        continue

                    captured_odds.append({
                        "race_number": race_num,
                        "horse_name": runner_name,
                        "saddlecloth": runner_number,
                        "current_odds": win_odds,
                        "opening_odds": opening_odds,
                        "place_odds": place_odds,
                        "scratched": bool(scratched),
                    })

            if races:
                logger.info(
                    f"TAB API intercepted: {len(races)} races, "
                    f"{sum(len(r.get('runners', [])) for r in races)} runners"
                )
                capture_done.set()

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=30000) as page:
            page.on("response", capture_tab_api)

            # Navigate to first race — the SPA typically loads the full meeting data
            url = f"{self.BASE_URL}/racing/{date_str}/{url_slug}/{mnemonic}/R/1"
            logger.info(f"TAB Playwright: navigating to {url}")

            try:
                await page.goto(url, wait_until="domcontentloaded")
                # Give SPA time to hydrate and fire API calls
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"TAB Playwright navigation failed: {e}")
                return []

            # Wait for API response to be captured
            try:
                await asyncio.wait_for(capture_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                # Log page title and snippet for diagnostics
                try:
                    title = await page.title()
                    snippet = await page.evaluate("() => document.body?.innerText?.substring(0, 300) || ''")
                    logger.warning(
                        f"TAB API response not captured within 15s. "
                        f"Page title: {title!r}, snippet: {snippet[:200]!r}"
                    )
                except Exception:
                    logger.warning("TAB API response not captured within 15s, trying race-by-race")

            # If we didn't get data from the initial load, try each race
            if not captured_odds and race_count > 0:
                for race_num in range(1, race_count + 1):
                    race_url = f"{self.BASE_URL}/racing/{date_str}/{url_slug}/{mnemonic}/R/{race_num}"
                    logger.info(f"TAB Playwright: navigating to R{race_num}")
                    try:
                        await page.goto(race_url, wait_until="domcontentloaded")
                        await asyncio.sleep(5)  # Allow SPA to fire API calls
                    except Exception as e:
                        logger.warning(f"TAB Playwright R{race_num} failed: {e}")
                        continue

            # Final wait if still collecting
            if not captured_odds:
                await asyncio.sleep(5)

        logger.info(f"TAB Playwright: captured {len(captured_odds)} runner odds for {venue}")
        return captured_odds

    async def scrape_race_statuses(
        self,
        venue: str,
        race_date: date,
    ) -> dict:
        """Check race statuses for an international venue via TAB.

        Returns dict matching RacingComScraper.check_race_statuses format:
        {"statuses": {race_num: status_str}, "track_condition": str|None}
        """
        tab_info = get_tab_mnemonic(venue)
        if not tab_info:
            return {"statuses": {}, "track_condition": None}

        mnemonic, jurisdiction, url_slug = tab_info
        date_str = race_date.strftime("%Y-%m-%d")

        statuses: dict[int, str] = {}
        track_condition = None
        capture_done = asyncio.Event()

        async def capture_tab_api(response):
            nonlocal track_condition
            url = response.url
            is_tab_api = self.API_HOST in url or "webapi.tab.com.au" in url
            if not is_tab_api:
                return
            if "tab-info-service" not in url and "racing" not in url and "meetings" not in url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
            except Exception:
                return

            # Extract race statuses from meeting data
            races = data.get("races", [])
            if not races:
                return

            track_condition = data.get("weatherCondition") or data.get("trackCondition")

            for race in races:
                race_num = race.get("raceNumber")
                race_status = race.get("raceStatus", "")
                if race_num:
                    # Map TAB statuses to racing.com format
                    status_map = {
                        "Open": "Open",
                        "Closed": "Closed",
                        "Interim": "Paying",
                        "Final": "Closed",
                        "Paying": "Paying",
                        "Abandoned": "Abandoned",
                        "Resulted": "Paying",
                    }
                    statuses[race_num] = status_map.get(race_status, race_status)

            if statuses:
                capture_done.set()

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=20000) as page:
            page.on("response", capture_tab_api)

            url = f"{self.BASE_URL}/racing/{date_str}/{url_slug}/{mnemonic}/R/1"
            try:
                await page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                logger.error(f"TAB status check navigation failed: {e}")
                return {"statuses": {}, "track_condition": None}

            try:
                await asyncio.wait_for(capture_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning(f"TAB status check timed out for {venue}")

        logger.info(f"TAB statuses for {venue}: {statuses}")
        return {"statuses": statuses, "track_condition": track_condition}

    async def scrape_race_result(
        self,
        venue: str,
        race_date: date,
        race_number: int,
    ) -> dict:
        """Scrape race results (finish positions, dividends) for a specific race.

        Returns dict matching orchestrator.upsert_race_results format:
        {"results": [{horse_name, saddlecloth, finish_position, win_dividend, place_dividend}]}
        """
        tab_info = get_tab_mnemonic(venue)
        if not tab_info:
            return {"results": []}

        mnemonic, jurisdiction, url_slug = tab_info
        date_str = race_date.strftime("%Y-%m-%d")

        results: list[dict] = []
        capture_done = asyncio.Event()

        async def capture_tab_api(response):
            url = response.url
            is_tab_api = self.API_HOST in url or "webapi.tab.com.au" in url
            if not is_tab_api:
                return
            if "tab-info-service" not in url and "racing" not in url and "meetings" not in url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
            except Exception:
                return

            # Find the specific race in the response
            races = data.get("races", [])
            if not races and "runners" in data:
                races = [data]

            for race in races:
                if race.get("raceNumber") != race_number:
                    continue

                for runner in race.get("runners", []):
                    result_place = runner.get("finishingPosition")
                    if result_place is None:
                        continue

                    # Extract dividends from results
                    fixed_odds = runner.get("fixedOdds", {})
                    win_deductions = runner.get("parimutuel", {})

                    results.append({
                        "horse_name": runner.get("runnerName", ""),
                        "saddlecloth": runner.get("runnerNumber"),
                        "position": result_place,
                        "win_dividend": win_deductions.get("returnWin") or fixed_odds.get("returnWin"),
                        "place_dividend": win_deductions.get("returnPlace") or fixed_odds.get("returnPlace"),
                        "margin": runner.get("resultedMargin"),
                    })

                if results:
                    capture_done.set()

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=20000) as page:
            page.on("response", capture_tab_api)

            # Navigate to the specific race results
            url = f"{self.BASE_URL}/racing/{date_str}/{url_slug}/{mnemonic}/R/{race_number}"
            try:
                await page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                logger.error(f"TAB results navigation failed: {e}")
                return {"results": []}

            try:
                await asyncio.wait_for(capture_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning(f"TAB results timed out for {venue} R{race_number}")

        logger.info(f"TAB results for {venue} R{race_number}: {len(results)} runners")
        return {"results": results}


class HKJCSectionalScraper:
    """Scrape HKJC sectional times for Hong Kong races via Playwright.

    HKJC provides sectional times at:
    https://racing.hkjc.com/en-us/local/information/displaysectionaltime?racedate=DD/MM/YYYY&RaceNo=N
    """

    BASE_URL = "https://racing.hkjc.com/en-us/local/information/displaysectionaltime"

    async def scrape_sectional_times(
        self,
        race_date: date,
        race_number: int,
    ) -> dict | None:
        """Scrape HKJC sectional times for a specific race.

        Returns dict matching racing.com sectional format:
        {
            race_number, has_sectionals,
            horses: [{saddlecloth, horse_name, final_position,
                      sectional_times: [{distance, position, time}]}]
        }
        """
        date_str = race_date.strftime("%d/%m/%Y")
        url = f"{self.BASE_URL}?racedate={date_str}&RaceNo={race_number}"
        logger.info(f"HKJC sectionals: navigating to {url}")

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=20000) as page:
            try:
                await page.goto(url, wait_until="load")
                await page.wait_for_timeout(3000)  # Let JS render the table
            except Exception as e:
                logger.warning(f"HKJC sectionals navigation failed: {e}")
                return None

            # Extract sectional data from the rendered table
            # HKJC table structure:
            # Col 0: Finishing Order | Col 1: Horse No. | Col 2: Horse Name
            # Col 3..N-1: Sectional times (1st Sec, 2nd Sec, ...) | Col N: Total Time
            try:
                data = await page.evaluate("""() => {
                    // Find the sectional times table (try multiple selectors)
                    const table = document.querySelector('table.Race') ||
                                  document.querySelector('table.sectionaltime') ||
                                  document.querySelector('table.table_bd');
                    if (!table) return null;

                    const thead = table.querySelector('thead');
                    const tbody = table.querySelector('tbody');
                    if (!tbody) return null;

                    // Extract section headers (e.g., "1st Sec.", "2nd Sec.", ...)
                    const headers = [];
                    if (thead) {
                        const headerCells = thead.querySelectorAll('th, td');
                        headerCells.forEach(c => headers.push(c.textContent.trim()));
                    }

                    const rows = tbody.querySelectorAll('tr');
                    if (!rows || rows.length === 0) return null;

                    const horses = [];
                    for (let i = 0; i < rows.length; i++) {
                        const cells = rows[i].querySelectorAll('td');
                        if (cells.length < 4) continue;

                        // HKJC columns: [0]=Position, [1]=Horse No, [2]=Horse Name, [3..N-1]=Sections, [N]=Time
                        const pos = parseInt(cells[0]?.textContent?.trim()) || null;
                        const saddlecloth = parseInt(cells[1]?.textContent?.trim()) || null;
                        // Horse name may include ID in parens, e.g. "COME FAST FAY FAY (K121)"
                        let horseName = cells[2]?.textContent?.trim() || '';
                        // Strip HKJC horse ID suffix like "(K121)"
                        horseName = horseName.replace(/\\s*\\([A-Z]\\d{3}\\)\\s*$/, '').trim();

                        const horse = {
                            saddlecloth: saddlecloth,
                            horse_name: horseName,
                            final_position: pos,
                            sectional_times: [],
                            race_time: null
                        };

                        // Sectional columns are between horse name and total time
                        // Last column is total time
                        const lastIdx = cells.length - 1;
                        horse.race_time = cells[lastIdx]?.textContent?.trim() || null;

                        for (let j = 3; j < lastIdx; j++) {
                            const time = cells[j]?.textContent?.trim();
                            if (time && time !== '-') {
                                const label = (j < headers.length) ? headers[j] : ('Sec ' + (j - 2));
                                horse.sectional_times.push({
                                    section: label,
                                    time: parseFloat(time) || time,
                                });
                            }
                        }

                        if (horse.horse_name) {
                            horses.push(horse);
                        }
                    }
                    return horses;
                }""")
            except Exception as e:
                logger.warning(f"HKJC sectional extraction failed: {e}")
                return None

        if not data:
            logger.debug(f"No HKJC sectional data found for R{race_number}")
            return None

        logger.info(f"HKJC sectionals for R{race_number}: {len(data)} horses")
        return {
            "race_number": race_number,
            "has_sectionals": True,
            "source": "hkjc",
            "horses": data,
        }


class HKJCTrackInfoScraper:
    """Scrape HKJC wind tracker / track conditions for HK venues via Playwright.

    HKJC provides live wind and weather data at:
    https://racing.hkjc.com/en-us/local/info/windtracker
    """

    WIND_TRACKER_URL = "https://racing.hkjc.com/en-us/local/info/windtracker"

    async def scrape_track_info(self, race_date: date) -> dict | None:
        """Scrape HKJC wind tracker for weather and track conditions.

        Returns dict with fields matching Meeting model:
        {
            weather_wind_speed: int | None,  # km/h
            weather_wind_dir: str | None,    # e.g. "NE", "SW"
            weather_condition: str | None,   # e.g. "Fine", "Cloudy"
            weather_temp: int | None,        # °C
            weather_humidity: int | None,    # %
            track_condition: str | None,     # e.g. "Good", "Good to Yielding"
        }
        """
        logger.info(f"HKJC wind tracker: navigating to {self.WIND_TRACKER_URL}")

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=20000) as page:
            try:
                await page.goto(self.WIND_TRACKER_URL, wait_until="load")
                await page.wait_for_timeout(4000)  # Next.js app needs time to hydrate
            except Exception as e:
                logger.warning(f"HKJC wind tracker navigation failed: {e}")
                return None

            # Extract weather data from the HKJC wind tracker page.
            # Page layout after React render:
            #   "Going GOOD", "Penetrometer Reading\n2.72"
            #   "Temperature\n22.3°C", "Relative Humidity\n84%"
            #   Wind readings: "ENE\n4 km/h\n5 km/h" per measurement point
            #   Course config: '"A+3" COURSE'
            try:
                data = await page.evaluate("""() => {
                    const result = {
                        weather_wind_speed: null,
                        weather_wind_dir: null,
                        weather_condition: null,
                        weather_temp: null,
                        weather_humidity: null,
                        track_condition: null,
                        penetrometer: null,
                        rainfall: null,
                        soil_moisture: null,
                        course_config: null,
                        raw_text: null
                    };

                    const body = document.body;
                    if (!body) return result;

                    const allText = body.innerText || '';
                    result.raw_text = allText.substring(0, 3000);

                    // Temperature: "Temperature 22.3°C"
                    const tempMatch = allText.match(/Temperature\\s*\\n?\\s*(\\d+\\.?\\d*)\\s*°\\s*C/i);
                    if (tempMatch) result.weather_temp = parseFloat(tempMatch[1]);

                    // Humidity: "Relative Humidity 84%"
                    const humidMatch = allText.match(/(?:Relative\\s+)?Humidity\\s*\\n?\\s*(\\d{1,3})\\s*%/i);
                    if (humidMatch) result.weather_humidity = parseInt(humidMatch[1]);

                    // Going: "Going GOOD" or "Going GOOD TO YIELDING"
                    const goingMatch = allText.match(/Going\\s+(Good to Firm|Good to Yielding|Good|Firm|Yielding to Soft|Yielding|Soft|Heavy|Wet Fast|Fast)/i);
                    if (goingMatch) result.track_condition = goingMatch[1];

                    // Penetrometer: "Penetrometer Reading 2.72"
                    const penMatch = allText.match(/Penetrometer\\s+Reading\\s*\\n?\\s*(\\d+\\.?\\d*)/i);
                    if (penMatch) result.penetrometer = parseFloat(penMatch[1]);

                    // Rainfall: "Total 0mm"
                    const rainMatch = allText.match(/Rainfall[\\s\\S]*?Total\\s*\\n?\\s*(\\d+\\.?\\d*)\\s*mm/i);
                    if (rainMatch) result.rainfall = parseFloat(rainMatch[1]);

                    // Soil moisture: "Soil Moisture 16.8%"
                    const soilMatch = allText.match(/Soil\\s+Moisture\\s*\\n?\\s*(\\d+\\.?\\d*)\\s*%/i);
                    if (soilMatch) result.soil_moisture = parseFloat(soilMatch[1]);

                    // Course config: '"A+3" COURSE'
                    const courseMatch = allText.match(/[\\u201c"']([A-C]\\+?\\d?)[\\u201d"']\\s*COURSE/i);
                    if (courseMatch) result.course_config = courseMatch[1];

                    // Wind: actual measurement readings after "Wind Gust" label
                    // Format: "ENE\\n4 km/h\\n5 km/h" per sensor point
                    const gustIdx = allText.indexOf('Wind Gust');
                    if (gustIdx >= 0) {
                        const windSection = allText.substring(gustIdx);
                        const dirs = [];
                        const speeds = [];
                        const wp = /(N|NE|NNE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\\s*\\n\\s*(\\d+)\\s*km\\/h/gi;
                        let wm;
                        while ((wm = wp.exec(windSection)) !== null) {
                            dirs.push(wm[1].toUpperCase());
                            speeds.push(parseInt(wm[2]));
                        }
                        if (speeds.length > 0) {
                            result.weather_wind_speed = Math.round(
                                speeds.reduce((a, b) => a + b, 0) / speeds.length
                            );
                            const dc = {};
                            dirs.forEach(d => { dc[d] = (dc[d] || 0) + 1; });
                            result.weather_wind_dir = Object.entries(dc)
                                .sort((a, b) => b[1] - a[1])[0][0];
                        }
                    }

                    return result;
                }""")
            except Exception as e:
                logger.warning(f"HKJC wind tracker extraction failed: {e}")
                return None

        if not data:
            return None

        # Log raw text for debugging (first run — helps refine selectors)
        raw = data.pop("raw_text", None)
        if raw:
            logger.info(f"HKJC wind tracker raw text (first 500 chars): {raw[:500]}")

        # Check if we got any useful data
        has_data = any(v is not None for v in data.values())
        if not has_data:
            logger.warning("HKJC wind tracker: no weather data extracted from page")
            return None

        logger.info(
            f"HKJC wind tracker: wind={data.get('weather_wind_speed')}km/h "
            f"{data.get('weather_wind_dir')}, temp={data.get('weather_temp')}°C, "
            f"humidity={data.get('weather_humidity')}%, "
            f"track={data.get('track_condition')}, pen={data.get('penetrometer')}, "
            f"rain={data.get('rainfall')}mm, soil={data.get('soil_moisture')}%, "
            f"course={data.get('course_config')}"
        )
        return data


class HKJCOddsScraper:
    """Scrape HKJC win/place odds directly from bet.hkjc.com via Playwright.

    Fallback when TAB is blocked by Akamai. HKJC doesn't have aggressive
    bot protection and serves odds on their betting SPA.
    """

    # HKJC venue codes
    VENUE_CODES = {"sha tin": "ST", "happy valley": "HV"}

    async def scrape_odds_for_meeting(
        self,
        venue: str,
        race_date: date,
        race_count: int,
    ) -> list[dict]:
        """Scrape HKJC win/place odds for all races at a HK venue.

        Returns list of dicts matching TabPlaywrightScraper format:
        {race_number, horse_name, saddlecloth, current_odds, place_odds, scratched}
        """
        v = normalize_venue(venue)
        venue_code = self.VENUE_CODES.get(v)
        if not venue_code:
            logger.info(f"No HKJC venue code for {venue}")
            return []

        date_str = race_date.strftime("%Y-%m-%d")
        all_odds: list[dict] = []

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=30000) as page:
            for race_num in range(1, race_count + 1):
                url = f"https://bet.hkjc.com/en/racing/wp/{date_str}/{venue_code}/{race_num}"
                logger.info(f"HKJC odds: navigating to R{race_num} — {url}")

                try:
                    await page.goto(url, wait_until="domcontentloaded")
                    await asyncio.sleep(4)  # SPA needs time to load odds
                except Exception as e:
                    logger.warning(f"HKJC odds R{race_num} navigation failed: {e}")
                    continue

                # Extract odds from rendered page
                try:
                    race_odds = await page.evaluate("""(raceNum) => {
                        const runners = [];

                        // Try multiple selectors for the odds table
                        // HKJC shows runner rows with horse number, name, win odds, place odds
                        const rows = document.querySelectorAll(
                            'tr[data-runner], .runner-row, [class*="runner"], [class*="odds-row"], table tbody tr'
                        );

                        for (const row of rows) {
                            const cells = row.querySelectorAll('td, [class*="cell"]');
                            if (cells.length < 3) continue;

                            // Try to extract structured odds data
                            const text = row.textContent || '';

                            // Look for patterns like: "1  HORSE NAME  3.5  1.4"
                            // or extract from specific data attributes
                            const numEl = row.querySelector('[class*="number"], [class*="no"], td:first-child');
                            const nameEl = row.querySelector('[class*="name"], [class*="horse"]');

                            let saddlecloth = null;
                            let horseName = '';
                            let winOdds = null;
                            let placeOdds = null;
                            let scratched = false;

                            if (numEl) {
                                saddlecloth = parseInt(numEl.textContent.trim()) || null;
                            }
                            if (nameEl) {
                                horseName = nameEl.textContent.trim();
                            }

                            // Look for odds values (decimal numbers in cells)
                            const oddsValues = [];
                            for (const cell of cells) {
                                const val = parseFloat(cell.textContent.trim());
                                if (!isNaN(val) && val > 1.0 && val < 999) {
                                    oddsValues.push(val);
                                }
                            }

                            if (oddsValues.length >= 1) {
                                winOdds = oddsValues[0];
                            }
                            if (oddsValues.length >= 2) {
                                placeOdds = oddsValues[1];
                            }

                            // Check for scratched indicators
                            if (text.includes('SCR') || text.includes('Scratched') ||
                                row.classList.contains('scratched') ||
                                row.querySelector('[class*="scratch"]')) {
                                scratched = true;
                            }

                            if ((saddlecloth || horseName) && (winOdds || scratched)) {
                                runners.push({
                                    race_number: raceNum,
                                    horse_name: horseName,
                                    saddlecloth: saddlecloth,
                                    current_odds: winOdds,
                                    place_odds: placeOdds,
                                    scratched: scratched
                                });
                            }
                        }

                        // If structured extraction failed, try to get raw text and log it
                        if (runners.length === 0) {
                            const body = document.body?.innerText?.substring(0, 1000) || '';
                            return {runners: [], debug_text: body};
                        }

                        return {runners: runners, debug_text: null};
                    }""", race_num)
                except Exception as e:
                    logger.warning(f"HKJC odds extraction failed for R{race_num}: {e}")
                    continue

                if race_odds and race_odds.get("runners"):
                    all_odds.extend(race_odds["runners"])
                    logger.info(f"HKJC odds R{race_num}: {len(race_odds['runners'])} runners")
                elif race_odds and race_odds.get("debug_text"):
                    logger.info(f"HKJC odds R{race_num}: no runners extracted. Page text: {race_odds['debug_text'][:300]}")

        logger.info(f"HKJC odds total: {len(all_odds)} runners for {venue}")
        return all_odds


# ============ HKJC RESULTS SCRAPER ============


class HKJCResultsScraper:
    """Scrape HKJC race results and statuses via httpx (no Playwright needed).

    HKJC results pages are server-rendered HTML. Much faster and more
    reliable than TAB Playwright which always times out for HK.

    Results URL:
    https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx?RaceDate=YYYY/MM/DD&Racecourse=ST&RaceNo=N
    """

    RESULTS_URL = "https://racing.hkjc.com/racing/information/english/Racing/LocalResults.aspx"

    # HKJC venue codes
    VENUE_CODES = {"sha tin": "ST", "happy valley": "HV"}

    def _get_venue_code(self, venue: str) -> str | None:
        from punty.venues import normalize_venue
        v = normalize_venue(venue)
        return self.VENUE_CODES.get(v)

    async def scrape_race_statuses(
        self,
        venue: str,
        race_date: date,
        race_count: int,
    ) -> dict:
        """Check which races have results (= Paying/Closed status).

        Returns dict matching monitor format:
        {"statuses": {race_num: status_str, ...}, "track_condition": str|None}
        """
        import httpx

        venue_code = self._get_venue_code(venue)
        if not venue_code:
            logger.info(f"No HKJC venue code for {venue}")
            return {"statuses": {}, "track_condition": None}

        date_str = race_date.strftime("%Y/%m/%d")
        statuses: dict[int, str] = {}
        track_condition = None

        # Check each race — httpx is fast, no browser overhead
        async with httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            follow_redirects=True,
        ) as client:
            for race_num in range(1, race_count + 1):
                try:
                    resp = await client.get(
                        self.RESULTS_URL,
                        params={
                            "RaceDate": date_str,
                            "Racecourse": venue_code,
                            "RaceNo": str(race_num),
                        },
                    )
                    resp.raise_for_status()
                    html = resp.text

                    # If the page has a results tbody, race is done
                    if 'class="f_fs12"' in html and '>WIN</td>' in html:
                        statuses[race_num] = "Paying"
                    elif 'class="f_fs12"' in html:
                        statuses[race_num] = "Interim"
                    else:
                        statuses[race_num] = "Open"

                    # Extract track condition from first race page
                    # HKJC format: <td>Going :</td><td colspan="14">GOOD</td>
                    if race_num == 1 and track_condition is None:
                        tc_match = re.search(
                            r'Going\s*:\s*</td>\s*<td[^>]*>\s*([A-Z][A-Z\s]+)',
                            html,
                        )
                        if tc_match:
                            track_condition = tc_match.group(1).strip().title()

                except Exception as e:
                    logger.warning(f"HKJC status check failed for R{race_num}: {e}")
                    statuses[race_num] = "Open"  # Assume open on error

        logger.info(f"HKJC statuses for {venue}: {statuses} | Track: {track_condition}")
        return {"statuses": statuses, "track_condition": track_condition}

    async def scrape_race_result(
        self,
        venue: str,
        race_date: date,
        race_number: int,
    ) -> dict:
        """Scrape full race results from HKJC for a specific race.

        Returns dict matching orchestrator.upsert_race_results format:
        {"results": [{horse_name, saddlecloth, finish_position, win_dividend, place_dividend, result_margin}]}
        """
        import httpx

        venue_code = self._get_venue_code(venue)
        if not venue_code:
            return {"results": []}

        date_str = race_date.strftime("%Y/%m/%d")

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    self.RESULTS_URL,
                    params={
                        "RaceDate": date_str,
                        "Racecourse": venue_code,
                        "RaceNo": str(race_number),
                    },
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            logger.warning(f"HKJC results fetch failed for R{race_number}: {e}")
            return {"results": []}

        return self._parse_results_html(html, race_number)

    def _parse_results_html(self, html: str, race_number: int) -> dict:
        """Parse HKJC results HTML into structured results list.

        HKJC results table structure:
        - Runner rows inside <tbody class="f_fs12">
        - Columns: Pla | Horse No | Horse | Jockey | Trainer | Act.Wt | Horse Wt | Dr | LBW | Running Pos | Time | Win Odds
        - Horse name includes ID suffix like "(L126)" to strip
        - Dividends per HK$10 unit
        """
        results: list[dict] = []

        def clean(s):
            return re.sub(r'<[^>]+>', '', s).strip()

        # Extract runner rows from <tbody class="f_fs12">
        tbody_match = re.search(
            r'<tbody\s+class="f_fs12">(.*?)</tbody>',
            html,
            re.DOTALL,
        )
        if tbody_match:
            tbody_html = tbody_match.group(1)
            row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
            cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)

            for row_html in row_pattern.findall(tbody_html):
                cells = cell_pattern.findall(row_html)
                if len(cells) < 5:
                    continue

                place_str = clean(cells[0])
                saddlecloth_str = clean(cells[1])
                horse_name = clean(cells[2])

                # Parse finish position (handle "1", "2", "DH", "DNF", "WV" etc.)
                finish_pos = None
                if place_str.isdigit():
                    finish_pos = int(place_str)
                elif re.match(r'DH\s*\d+', place_str, re.IGNORECASE):
                    dh_match = re.search(r'\d+', place_str)
                    if dh_match:
                        finish_pos = int(dh_match.group())

                saddlecloth = None
                if saddlecloth_str.strip().isdigit():
                    saddlecloth = int(saddlecloth_str.strip())

                # Strip HKJC horse ID suffix like "(L126)" and trailing whitespace
                horse_name = re.sub(r'\s*\([A-Z]\d{3}\)\s*$', '', horse_name).strip()
                # Strip HTML entities and non-breaking spaces, convert to title case
                horse_name = horse_name.replace('&nbsp;', ' ').replace('\xa0', ' ').strip()
                # HKJC returns ALL CAPS — convert to title case to match DB format
                horse_name = horse_name.title()

                if not horse_name or finish_pos is None:
                    continue

                # LBW (Length Behind Winner) is column 8
                margin = None
                if len(cells) > 8:
                    lbw = clean(cells[8])
                    if lbw and lbw != '-':
                        margin = lbw

                # Win odds column is last
                win_odds = None
                if len(cells) > 10:
                    odds_str = clean(cells[-1])
                    try:
                        win_odds = float(odds_str)
                    except (ValueError, TypeError):
                        pass

                results.append({
                    "horse_name": horse_name,
                    "saddlecloth": saddlecloth,
                    "position": finish_pos,
                    "win_dividend": None,  # Filled from dividends section
                    "place_dividend": None,
                    "margin": margin,
                    "win_odds": win_odds,
                })

        # Extract dividends — structure: <td>WIN</td><td>5</td><td>30.00</td>
        # Win dividend
        win_match = re.search(
            r'>WIN</td>\s*<td[^>]*>\s*(\d+)\s*</td>\s*<td[^>]*>\s*([\d,.]+)',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if win_match and results:
            win_saddlecloth = int(win_match.group(1))
            win_div = float(win_match.group(2).replace(',', ''))
            # HKJC dividends are per HK$10 unit, convert to per-$1
            win_div_per_unit = win_div / 10.0
            for r in results:
                if r["saddlecloth"] == win_saddlecloth:
                    r["win_dividend"] = win_div_per_unit
                    break

        # Place dividends — multiple PLACE rows, or continuation rows without label
        # Pattern: PLACE | saddlecloth | dividend  or  (empty) | saddlecloth | dividend
        place_section = re.search(r'>PLACE</td>(.*?)(?:>QUINELLA|>FORECAST|</table>)', html, re.DOTALL | re.IGNORECASE)
        if place_section and results:
            place_html = ">PLACE</td>" + place_section.group(1)
            pl_rows = re.findall(
                r'<td[^>]*>\s*(\d+)\s*</td>\s*<td[^>]*>\s*([\d,.]+)',
                place_html,
            )
            for pl_sc_str, pl_div_str in pl_rows:
                pl_sc = int(pl_sc_str)
                pl_div = float(pl_div_str.replace(',', '')) / 10.0
                for r in results:
                    if r["saddlecloth"] == pl_sc and r["place_dividend"] is None:
                        r["place_dividend"] = pl_div
                        break

        logger.info(f"HKJC results for R{race_number}: {len(results)} runners")
        return {"results": results}


# ============ HKJC JOCKEY / TRAINER RANKINGS ============

# Module-level cache: refreshed once per day.
_hkjc_ranking_cache: dict[str, list[dict]] = {}


class HKJCRankingScraper:
    """Scrape HKJC jockey and trainer season rankings with track breakdowns.

    Uses Playwright because HKJC is a React SPA — the ranking table body
    is empty in the static HTML and only populated by client-side JS.

    Data source:
      - https://racing.hkjc.com/en-us/local/info/jockey-ranking
      - https://racing.hkjc.com/en-us/local/info/trainer-ranking

    Each page supports racecourse filters: ALL, ST (Sha Tin Turf),
    STAWT (Sha Tin AWT), HV (Happy Valley Turf).

    Returns data keyed by name with overall + per-track stats.
    """

    JOCKEY_URL = "https://racing.hkjc.com/en-us/local/info/jockey-ranking"
    TRAINER_URL = "https://racing.hkjc.com/en-us/local/info/trainer-ranking"

    # <select id="ddlVenueTrackType"> option values → output dict keys
    TRACKS = {
        "ALL": "overall",
        "STT": "sha_tin_turf",
        "STA": "sha_tin_awt",
        "HVT": "happy_valley",
    }

    async def scrape_rankings(
        self,
        role: str = "jockey",
        tracks: list[str] | None = None,
    ) -> list[dict]:
        """Scrape jockey or trainer rankings across one or more tracks.

        Args:
            role: "jockey" or "trainer"
            tracks: List of track codes (ALL, ST, STAWT, HV).
                    Defaults to all tracks.

        Returns list of dicts, one per person:
            {
                "name": "Z Purton",
                "overall":       {"win": 80, "second": 50, "third": 40, ...},
                "sha_tin_turf":  {...},
                "sha_tin_awt":   {...},
                "happy_valley":  {...},
            }
        """
        base_url = self.JOCKEY_URL if role == "jockey" else self.TRAINER_URL
        track_list = tracks or list(self.TRACKS.keys())
        rides_label = "rides" if role == "jockey" else "runs"

        merged: dict[str, dict] = {}

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=25000) as page:
            # Load the ALL page first — track filters are JS tabs, not URL params
            url = f"{base_url}?season=Current&view=Numbers&racecourse=ALL"
            logger.info(f"HKJC {role} ranking: loading {url}")

            try:
                await page.goto(url, wait_until="domcontentloaded")
                await asyncio.sleep(5)  # React needs time to hydrate + render table
            except Exception as e:
                logger.warning(f"HKJC {role} ranking page load failed: {e}")
                return []

            for track_code in track_list:
                track_key = self.TRACKS.get(track_code, track_code.lower())

                # Change the racecourse dropdown (ALL is the default loaded state)
                if track_code != "ALL":
                    try:
                        await page.select_option("#ddlVenueTrackType", track_code)
                        await asyncio.sleep(2)  # Wait for table to re-render
                    except Exception as e:
                        logger.warning(f"HKJC {role} ranking filter change failed ({track_code}): {e}")
                        continue

                # Extract table rows — find the table with most data rows
                try:
                    rows = await page.evaluate("""(ridesLabel) => {
                        const tables = document.querySelectorAll('table');
                        let target = null;
                        let maxRows = 0;
                        for (const t of tables) {
                            const tbodyRows = t.querySelectorAll('tbody tr').length;
                            if (tbodyRows > maxRows) {
                                maxRows = tbodyRows;
                                target = t;
                            }
                        }
                        if (!target || maxRows < 2) return [];

                        const result = [];
                        const trs = target.querySelectorAll('tbody tr');
                        for (const tr of trs) {
                            const cells = tr.querySelectorAll('td');
                            if (cells.length < 7) continue;

                            const name = (cells[0].querySelector('a') || cells[0]).textContent.trim();
                            if (!name || name.length < 2) continue;
                            // Skip non-data rows (headers, separators, etc.)
                            if (name === '---' || name === 'Others' || name.toLowerCase().includes('ranking'))
                                continue;

                            const nums = [];
                            for (let i = 1; i < 7; i++) {
                                const txt = cells[i].textContent.trim().replace(/,/g, '');
                                nums.push(parseInt(txt) || 0);
                            }
                            const [win, second, third, fourth, fifth, total] = nums;
                            // Skip rows where total is 0 (invalid data)
                            if (total === 0) continue;

                            const sr = Math.round(win / total * 1000) / 10;
                            const placeSr = Math.round((win + second + third) / total * 1000) / 10;

                            const stats = {
                                win, second, third, fourth, fifth, sr, place_sr: placeSr,
                            };
                            stats[ridesLabel] = total;
                            result.push({name, stats});
                        }
                        return result;
                    }""", rides_label)
                except Exception as e:
                    logger.warning(f"HKJC {role} ranking extraction failed ({track_code}): {e}")
                    continue

                for row in rows:
                    name = row["name"]
                    if name not in merged:
                        merged[name] = {"name": name}
                    merged[name][track_key] = row["stats"]

                logger.info(f"HKJC {role} ranking {track_code}: {len(rows)} entries")

        result = list(merged.values())
        logger.info(f"HKJC {role} rankings: {len(result)} entries across {len(track_list)} tracks")
        return result


async def fetch_hkjc_rankings(venue: str) -> dict[str, list[dict]]:
    """Fetch and cache HKJC jockey + trainer rankings for context builder.

    Args:
        venue: Normalized venue name (e.g. "sha tin", "happy valley")

    Returns dict with keys "jockeys" and "trainers", each a list of
    ranking dicts with per-track breakdowns.
    """
    global _hkjc_ranking_cache

    if "jockeys" in _hkjc_ranking_cache and "trainers" in _hkjc_ranking_cache:
        return _hkjc_ranking_cache

    scraper = HKJCRankingScraper()
    try:
        jockeys = await scraper.scrape_rankings(role="jockey")
        trainers = await scraper.scrape_rankings(role="trainer")
        _hkjc_ranking_cache["jockeys"] = jockeys
        _hkjc_ranking_cache["trainers"] = trainers
        logger.info(
            f"HKJC rankings cached: {len(jockeys)} jockeys, {len(trainers)} trainers"
        )
    except Exception as e:
        logger.error(f"HKJC ranking scrape failed: {e}")
        _hkjc_ranking_cache.setdefault("jockeys", [])
        _hkjc_ranking_cache.setdefault("trainers", [])

    return _hkjc_ranking_cache


def format_hkjc_ranking(person: dict, role: str, venue: str) -> str | None:
    """Format a single jockey/trainer's HKJC ranking for AI context.

    Returns a compact string like:
      "Z Purton — Season: 80W/300R (26.7% SR), Sha Tin: 50W/180R (27.8%), HV: 30W/120R (25.0%)"
    """
    v = venue.lower().strip()
    # Pick the venue-specific key
    venue_key = "happy_valley" if "happy" in v else "sha_tin_turf"

    overall = person.get("overall")
    if not overall:
        return None

    rides_key = "rides" if role == "jockey" else "runs"
    parts = [f"Season: {overall['win']}W/{overall.get(rides_key, 0)}R ({overall['sr']}% SR)"]

    track = person.get(venue_key)
    if track:
        label = "HV" if "happy" in v else "ST"
        parts.append(f"{label}: {track['win']}W/{track.get(rides_key, 0)}R ({track['sr']}%)")

    # Add AWT stats if at Sha Tin
    awt = person.get("sha_tin_awt")
    if awt and "happy" not in v:
        parts.append(f"AWT: {awt['win']}W/{awt.get(rides_key, 0)}R ({awt['sr']}%)")

    return " | ".join(parts)
