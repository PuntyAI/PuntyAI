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
