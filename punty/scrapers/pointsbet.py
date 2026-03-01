"""PointsBet Playwright scraper for odds.

Uses Playwright to load PointsBet racing pages and intercepts JSON API
responses containing runner/odds data.  Follows the same pattern as
TabPlaywrightScraper.

URL pattern: https://pointsbet.com.au/racing/Thoroughbred/{country}/{venue_slug}
"""

import asyncio
import json
import logging
import re
from datetime import date

from punty.venues import get_pointsbet_slug, normalize_venue

logger = logging.getLogger(__name__)

MAX_VALID_ODDS = 501.0  # Maximum payout odds ceiling


class PointsBetScraper:
    """Scrape PointsBet fixed odds via Playwright API interception."""

    BASE_URL = "https://pointsbet.com.au"

    async def scrape_odds_for_meeting(
        self,
        venue: str,
        race_date: date,
        meeting_id: str,
        race_count: int,
    ) -> list[dict]:
        """Scrape PointsBet odds for all races at a venue.

        Returns list of dicts: {race_number, horse_name, saddlecloth,
        current_odds, opening_odds, place_odds, scratched}
        """
        slug_info = get_pointsbet_slug(venue)
        if not slug_info:
            logger.info(f"No PointsBet slug for {venue} — skipping")
            return []

        country, venue_slug = slug_info

        # Accumulate odds from intercepted API responses
        captured_odds: list[dict] = []
        seen_keys: set[str] = set()  # Deduplicate across multiple API responses
        capture_done = asyncio.Event()

        async def capture_api(response):
            """Intercept PointsBet API responses containing runner/odds data."""
            url = response.url

            # Only process JSON responses from PointsBet domains
            if response.status != 200:
                return
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type:
                return

            # Log all JSON responses for discovery (helps narrow filter later)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.hostname or ""
                path = parsed.path or ""
            except Exception:
                host, path = "", ""

            if "pointsbet" in host or "pointsbet" in url:
                logger.debug(f"PointsBet API response: {host}{path[:150]}")

            try:
                body = await response.text()
                data = json.loads(body)
            except Exception:
                return

            # Try to extract runner/odds data from various possible structures
            runners_found = _extract_runners(data, captured_odds, seen_keys)
            if runners_found:
                logger.info(
                    f"PointsBet API intercepted: {runners_found} runners "
                    f"(total: {len(captured_odds)})"
                )
                capture_done.set()

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=30000) as page:
            page.on("response", capture_api)

            # Navigate to the venue's racing page
            url = f"{self.BASE_URL}/racing/Thoroughbred/{country}/{venue_slug}"
            logger.info(f"PointsBet: navigating to {url}")

            try:
                await page.goto(url, wait_until="domcontentloaded")
                # Give SPA time to hydrate and fire API calls
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"PointsBet navigation failed: {e}")
                return []

            # Wait for API response to be captured
            try:
                await asyncio.wait_for(capture_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                try:
                    title = await page.title()
                    snippet = await page.evaluate(
                        "() => document.body?.innerText?.substring(0, 300) || ''"
                    )
                    logger.warning(
                        f"PointsBet API not captured within 15s. "
                        f"Title: {title!r}, snippet: {snippet[:200]!r}"
                    )
                except Exception:
                    logger.warning("PointsBet API not captured within 15s")

            # DOM fallback: if API interception got nothing, try extracting from page
            if not captured_odds:
                logger.info("PointsBet: trying DOM fallback extraction")
                dom_odds = await _extract_from_dom(page, race_count)
                captured_odds.extend(dom_odds)

        if captured_odds:
            logger.info(
                f"PointsBet: {len(captured_odds)} runner odds captured for {venue}"
            )
        else:
            logger.warning(f"PointsBet: no odds captured for {venue}")

        return captured_odds

    async def scrape_results_for_race(
        self,
        venue: str,
        race_date: date,
        race_number: int,
    ) -> dict:
        """Scrape PointsBet race results (positions + exotic dividends).

        Uses Playwright API interception to capture JSON responses from the
        PointsBet results page.  Returns standardised format:
            {"results": [...], "exotics": {"exacta": 78.5, ...}}
        """
        slug_info = get_pointsbet_slug(venue)
        if not slug_info:
            logger.info(f"No PointsBet slug for {venue} — skipping results")
            return {"results": []}

        country, venue_slug = slug_info

        captured_results: list[dict] = []
        captured_exotics: dict[str, float] = {}
        capture_done = asyncio.Event()

        async def capture_api(response):
            url = response.url
            if response.status != 200:
                return
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type:
                return

            try:
                body = await response.text()
                data = json.loads(body)
            except Exception:
                return

            # Extract results from intercepted JSON
            found = _extract_results(
                data, race_number, captured_results, captured_exotics
            )
            if found:
                logger.info(
                    f"PointsBet results API intercepted: {found} runners "
                    f"for R{race_number}"
                )
                capture_done.set()

        from punty.scrapers.playwright_base import new_page

        async with new_page(timeout=30000) as page:
            page.on("response", capture_api)

            # PointsBet results URL pattern
            url = (
                f"{self.BASE_URL}/racing/Thoroughbred/{country}/{venue_slug}"
                f"?race={race_number}"
            )
            logger.info(f"PointsBet results: navigating to {url}")

            try:
                await page.goto(url, wait_until="domcontentloaded")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"PointsBet results navigation failed: {e}")
                return {"results": []}

            try:
                await asyncio.wait_for(capture_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning(
                    f"PointsBet results API not captured within 15s "
                    f"for {venue} R{race_number}"
                )

        result_data: dict = {"results": captured_results}
        if captured_exotics:
            result_data["exotics"] = captured_exotics

        logger.info(
            f"PointsBet results for {venue} R{race_number}: "
            f"{len(captured_results)} runners, {len(captured_exotics)} exotics"
        )
        return result_data


def _extract_results(
    data: dict | list,
    target_race: int,
    captured_results: list[dict],
    captured_exotics: dict[str, float],
) -> int:
    """Extract race results and exotic dividends from a PointsBet API response.

    Looks for finish positions, dividends, and exotic payout data.
    Returns number of result runners found.
    """
    if not isinstance(data, dict):
        return 0

    count = 0

    # Look for race-level objects containing results
    for key in ("races", "events", "results", "raceResults"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for race_obj in items:
            if not isinstance(race_obj, dict):
                continue
            race_num = (
                race_obj.get("raceNumber")
                or race_obj.get("race_number")
                or race_obj.get("number")
            )
            if race_num and int(race_num) != target_race:
                continue

            # Extract runners with positions
            for runner_key in ("runners", "results", "competitors",
                               "entries", "selections"):
                runners = race_obj.get(runner_key, [])
                if not isinstance(runners, list):
                    continue
                for runner in runners:
                    if not isinstance(runner, dict):
                        continue
                    parsed = _parse_result_runner(runner)
                    if parsed:
                        captured_results.append(parsed)
                        count += 1

            # Extract exotic dividends
            _extract_exotic_dividends(race_obj, captured_exotics)

    # Also check top-level structure (single race response)
    if not count:
        for runner_key in ("runners", "results", "competitors", "entries"):
            runners = data.get(runner_key, [])
            if not isinstance(runners, list):
                continue
            for runner in runners:
                if not isinstance(runner, dict):
                    continue
                parsed = _parse_result_runner(runner)
                if parsed:
                    captured_results.append(parsed)
                    count += 1
        if count:
            _extract_exotic_dividends(data, captured_exotics)

    return count


def _parse_result_runner(runner: dict) -> dict | None:
    """Parse a single runner dict into our result format."""
    name = (
        runner.get("name")
        or runner.get("runnerName")
        or runner.get("runner_name")
        or runner.get("horseName")
        or ""
    )
    if not name or len(name) < 2:
        return None

    position = (
        runner.get("finishPosition")
        or runner.get("finish_position")
        or runner.get("position")
        or runner.get("place")
    )
    if not position:
        return None
    try:
        position = int(position)
    except (ValueError, TypeError):
        return None

    number = (
        runner.get("number")
        or runner.get("runnerNumber")
        or runner.get("saddleCloth")
        or runner.get("tabNo")
    )
    try:
        number = int(number) if number else None
    except (ValueError, TypeError):
        number = None

    win_div = runner.get("winDividend") or runner.get("win_dividend")
    place_div = runner.get("placeDividend") or runner.get("place_dividend")

    try:
        win_div = float(win_div) if win_div else None
    except (ValueError, TypeError):
        win_div = None
    try:
        place_div = float(place_div) if place_div else None
    except (ValueError, TypeError):
        place_div = None

    return {
        "horse_name": name.strip().title(),
        "saddlecloth": number,
        "position": position,
        "win_dividend": win_div,
        "place_dividend": place_div,
        "margin": runner.get("margin"),
    }


def _extract_exotic_dividends(obj: dict, exotics: dict[str, float]) -> None:
    """Extract exotic dividend data from a race object."""
    # PointsBet exotic type names → our canonical types
    _PB_EXOTIC_MAP = {
        "exacta": "exacta",
        "quinella": "quinella",
        "trifecta": "trifecta",
        "first4": "first4",
        "first_4": "first4",
        "firstfour": "first4",
        "first four": "first4",
        "quadrella": "quaddie",
        "quaddie": "quaddie",
        "duo": "quinella",
    }

    for key in ("exotics", "exoticResults", "exotic_results", "dividends",
                "pools", "exoticDividends"):
        items = obj.get(key)
        if isinstance(items, dict):
            # Direct mapping: {"exacta": 45.20, "trifecta": 120.50}
            for etype, div in items.items():
                canonical = _PB_EXOTIC_MAP.get(etype.lower(), etype.lower())
                try:
                    exotics[canonical] = round(float(div), 2)
                except (ValueError, TypeError):
                    pass
        elif isinstance(items, list):
            # Array of objects: [{"type": "exacta", "dividend": 45.20}, ...]
            for item in items:
                if not isinstance(item, dict):
                    continue
                etype = (
                    item.get("type")
                    or item.get("poolType")
                    or item.get("pool_type")
                    or item.get("name")
                    or ""
                )
                div = (
                    item.get("dividend")
                    or item.get("amount")
                    or item.get("payout")
                    or item.get("returnAmount")
                )
                if etype and div:
                    canonical = _PB_EXOTIC_MAP.get(
                        etype.lower(), etype.lower()
                    )
                    try:
                        exotics[canonical] = round(float(div), 2)
                    except (ValueError, TypeError):
                        pass


def _extract_runners(
    data: dict | list,
    captured_odds: list[dict],
    seen_keys: set[str],
) -> int:
    """Extract runner odds from a PointsBet API response.

    Tries multiple JSON structures since we don't know the exact API schema.
    Returns the number of new runners found.
    """
    count = 0

    # Strategy 1: Look for arrays of objects with runner-like fields
    candidates = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        # Search for nested arrays that might contain runners
        for key in ("runners", "entries", "competitors", "selections",
                     "outcomes", "markets", "events", "races"):
            if key in data and isinstance(data[key], list):
                candidates = data[key]
                break

        # Check for nested race structure: races[].runners[]
        if not candidates:
            for key in ("races", "events", "meetings"):
                if key in data and isinstance(data[key], list):
                    for race_obj in data[key]:
                        if isinstance(race_obj, dict):
                            count += _extract_from_race_obj(
                                race_obj, captured_odds, seen_keys
                            )
                    if count:
                        return count

    for item in candidates:
        if not isinstance(item, dict):
            continue
        result = _parse_runner_dict(item, captured_odds, seen_keys)
        if result:
            count += 1

    return count


def _extract_from_race_obj(
    race_obj: dict,
    captured_odds: list[dict],
    seen_keys: set[str],
) -> int:
    """Extract runners from a race-level object."""
    count = 0
    race_num = (
        race_obj.get("raceNumber")
        or race_obj.get("race_number")
        or race_obj.get("number")
    )

    for runner_key in ("runners", "entries", "competitors", "selections", "outcomes"):
        runners = race_obj.get(runner_key, [])
        if not isinstance(runners, list):
            continue
        for runner in runners:
            if not isinstance(runner, dict):
                continue
            result = _parse_runner_dict(
                runner, captured_odds, seen_keys, default_race_num=race_num
            )
            if result:
                count += 1

    return count


def _parse_runner_dict(
    runner: dict,
    captured_odds: list[dict],
    seen_keys: set[str],
    default_race_num: int | None = None,
) -> bool:
    """Try to parse a single runner dict into our standard format.

    Returns True if a runner was added.
    """
    # Try various field names for horse name
    name = (
        runner.get("name")
        or runner.get("runnerName")
        or runner.get("runner_name")
        or runner.get("horseName")
        or runner.get("horse_name")
        or runner.get("competitorName")
        or ""
    )
    if not name or len(name) < 2:
        return False

    # Try various field names for saddlecloth/number
    number = (
        runner.get("number")
        or runner.get("runnerNumber")
        or runner.get("runner_number")
        or runner.get("saddleCloth")
        or runner.get("tabNo")
        or runner.get("barrier")
    )

    # Try various field names for race number
    race_num = (
        runner.get("raceNumber")
        or runner.get("race_number")
        or default_race_num
    )

    if not race_num or not number:
        return False

    # Try various field names for odds
    win_odds = (
        runner.get("fixedOdds", {}).get("returnWin")
        if isinstance(runner.get("fixedOdds"), dict) else None
    ) or runner.get("winPrice") or runner.get("win_price") or runner.get("price")

    place_odds = (
        runner.get("fixedOdds", {}).get("returnPlace")
        if isinstance(runner.get("fixedOdds"), dict) else None
    ) or runner.get("placePrice") or runner.get("place_price")

    opening_odds = (
        runner.get("fixedOdds", {}).get("openingPrice")
        if isinstance(runner.get("fixedOdds"), dict) else None
    ) or runner.get("openingPrice") or runner.get("opening_price")

    # Apply MAX_VALID_ODDS guard
    if win_odds and (not isinstance(win_odds, (int, float)) or win_odds > MAX_VALID_ODDS):
        win_odds = None
    if place_odds and (not isinstance(place_odds, (int, float)) or place_odds > MAX_VALID_ODDS):
        place_odds = None
    if opening_odds and (not isinstance(opening_odds, (int, float)) or opening_odds > MAX_VALID_ODDS):
        opening_odds = None

    # Need at least win odds to be useful
    if not win_odds:
        return False

    scratched = (
        runner.get("scratched", False)
        or runner.get("isScratched", False)
        or (runner.get("status", "").lower() in ("scratched", "late_scratching"))
    )

    # Deduplicate
    key = f"{race_num}-{number}"
    if key in seen_keys:
        return False
    seen_keys.add(key)

    captured_odds.append({
        "race_number": int(race_num),
        "horse_name": name.strip(),
        "saddlecloth": int(number),
        "current_odds": float(win_odds) if win_odds else None,
        "opening_odds": float(opening_odds) if opening_odds else None,
        "place_odds": float(place_odds) if place_odds else None,
        "scratched": bool(scratched),
    })
    return True


async def _extract_from_dom(page, race_count: int) -> list[dict]:
    """Fallback: extract odds from the rendered PointsBet DOM.

    This is less reliable than API interception but works if the API
    endpoints change or are blocked.
    """
    odds_data = []

    try:
        # Try to find runner rows in the page
        # PointsBet typically renders tables/cards with runner name + odds
        runners_js = (
            "() => {"
            "  const results = [];"
            "  const rows = document.querySelectorAll("
            "    '[data-testid*=\"runner\"], [class*=\"runner\"], [class*=\"Runner\"], '"
            "    + 'tr[class*=\"race\"], [class*=\"entrant\"], [class*=\"Entrant\"]'"
            "  );"
            "  for (const row of rows) {"
            "    const text = row.innerText || '';"
            "    const match = text.match(/(\\d{1,2})\\.?\\s+([A-Z][A-Za-z' ]+?)\\s+.*?\\$?(\\d+\\.\\d{2})/);"
            "    if (match) {"
            "      results.push({number: parseInt(match[1]), name: match[2].trim(), odds: parseFloat(match[3])});"
            "    }"
            "  }"
            "  return results;"
            "}"
        )
        dom_runners = await page.evaluate(runners_js)

        for runner in dom_runners:
            if runner["odds"] > MAX_VALID_ODDS:
                continue
            odds_data.append({
                "race_number": 1,  # DOM fallback can't reliably determine race number
                "horse_name": runner["name"],
                "saddlecloth": runner["number"],
                "current_odds": runner["odds"],
                "opening_odds": None,
                "place_odds": None,
                "scratched": False,
            })

        if odds_data:
            logger.info(f"PointsBet DOM fallback: extracted {len(odds_data)} runners")

    except Exception as e:
        logger.warning(f"PointsBet DOM fallback failed: {e}")

    return odds_data
