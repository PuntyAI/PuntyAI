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
            if self.API_HOST not in response.url:
                return
            if "tab-info-service" not in response.url:
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
            except Exception as e:
                logger.error(f"TAB Playwright navigation failed: {e}")
                return []

            # Wait for API response to be captured
            try:
                await asyncio.wait_for(capture_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning("TAB API response not captured within 15s, trying race-by-race")

            # If we didn't get data from the initial load, try each race
            if not captured_odds and race_count > 0:
                for race_num in range(1, race_count + 1):
                    race_url = f"{self.BASE_URL}/racing/{date_str}/{url_slug}/{mnemonic}/R/{race_num}"
                    logger.info(f"TAB Playwright: navigating to R{race_num}")
                    try:
                        await page.goto(race_url, wait_until="domcontentloaded")
                        await asyncio.sleep(3)  # Allow API call to complete
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
            if self.API_HOST not in response.url:
                return
            if "tab-info-service" not in response.url:
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
            if self.API_HOST not in response.url:
                return
            if "tab-info-service" not in response.url:
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
                        "finish_position": result_place,
                        "win_dividend": win_deductions.get("returnWin") or fixed_odds.get("returnWin"),
                        "place_dividend": win_deductions.get("returnPlace") or fixed_odds.get("returnPlace"),
                        "result_margin": runner.get("resultedMargin"),
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
