"""Track conditions scraper with multiple source fallbacks."""

import logging
import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from punty.scrapers.playwright_base import new_page, wait_and_get_content

logger = logging.getLogger(__name__)

# State codes to URL parameter for racingaustralia.horse
STATE_CODES = {
    "VIC": "VIC",
    "NSW": "NSW",
    "QLD": "QLD",
    "SA": "SA",
    "WA": "WA",
    "TAS": "TAS",
    "ACT": "ACT",
    "NT": "NT",
}

# Venue to state mapping for sportsbetform fallback
VENUE_TO_STATE = {
    # VIC
    "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC", "sandown": "VIC",
    "cranbourne": "VIC", "pakenham": "VIC", "mornington": "VIC", "geelong": "VIC",
    "ballarat": "VIC", "bendigo": "VIC", "sale": "VIC", "yarra valley": "VIC",
    "colac": "VIC", "warrnambool": "VIC", "stawell": "VIC", "hamilton": "VIC",
    # NSW
    "randwick": "NSW", "royal randwick": "NSW", "rosehill": "NSW", "canterbury": "NSW",
    "warwick farm": "NSW", "newcastle": "NSW", "kembla grange": "NSW", "gosford": "NSW",
    "wyong": "NSW", "hawkesbury": "NSW", "scone": "NSW", "tamworth": "NSW",
    "muswellbrook": "NSW", "dubbo": "NSW", "goulburn": "NSW", "albury": "NSW",
    "wagga": "NSW", "canberra": "NSW", "queanbeyan": "NSW",
    # QLD
    "eagle farm": "QLD", "doomben": "QLD", "gold coast": "QLD", "sunshine coast": "QLD",
    "ipswich": "QLD", "toowoomba": "QLD", "rockhampton": "QLD", "mackay": "QLD",
    "cairns": "QLD", "townsville": "QLD", "beaudesert": "QLD",
    # SA
    "morphettville": "SA", "murray bridge": "SA", "gawler": "SA", "strathalbyn": "SA",
    "mount gambier": "SA", "port lincoln": "SA", "balaklava": "SA",
    # WA
    "ascot": "WA", "belmont": "WA", "pinjarra": "WA", "bunbury": "WA",
    "albany": "WA", "geraldton": "WA", "kalgoorlie": "WA", "northam": "WA",
    # TAS
    "hobart": "TAS", "launceston": "TAS", "devonport": "TAS",
}


async def scrape_track_conditions(state: str = "VIC") -> list[dict[str, Any]]:
    """Scrape track conditions from racingaustralia.horse for a given state.

    Returns list of dicts with:
        - venue: str
        - condition: str (e.g. "Good 4", "Soft 5")
        - rail: str | None
        - weather: str | None
        - penetrometer: str | None
    """
    state_param = STATE_CODES.get(state.upper(), "V")
    url = (
        f"https://www.racingaustralia.horse/InteractiveForm/"
        f"TrackCondition.aspx?State={state_param}"
    )

    logger.info(f"Scraping track conditions for {state}: {url}")
    conditions: list[dict[str, Any]] = []

    async with new_page() as page:
        html = await wait_and_get_content(
            page, url, wait_selector="table, .track-condition, .conditions"
        )
        soup = BeautifulSoup(html, "lxml")

        # Typically rendered as a table
        tables = soup.select("table")
        for table in tables:
            rows = table.select("tr")
            headers = [th.get_text(strip=True).lower() for th in rows[0].select("th, td")] if rows else []

            # Check for track conditions table - "meeting details" or "track condition" in headers
            if not any(kw in " ".join(headers) for kw in ["meeting", "track", "condition", "course", "venue"]):
                continue

            logger.debug(f"Found track conditions table with headers: {headers}")

            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.select("td")]
                if len(cells) < 2:
                    continue

                entry = _map_cells(headers, cells)
                if entry.get("venue"):
                    conditions.append(entry)
                    logger.debug(f"Parsed track condition: {entry}")

        # Fallback: look for non-table layout
        if not conditions:
            items = soup.select(".track-item, .venue-condition, .condition-row")
            for item in items:
                venue_tag = item.select_one(".venue, .venue-name, h3, h4")
                cond_tag = item.select_one(".condition, .track-rating, .rating")
                if venue_tag:
                    conditions.append({
                        "venue": venue_tag.get_text(strip=True),
                        "condition": cond_tag.get_text(strip=True) if cond_tag else None,
                        "rail": None,
                        "weather": None,
                        "penetrometer": None,
                    })

    logger.info(f"Found {len(conditions)} track conditions for {state}")
    return conditions


def _map_cells(headers: list[str], cells: list[str]) -> dict[str, Any]:
    """Map table cells to a structured dict using header names."""
    data: dict[str, Any] = {
        "venue": None,
        "condition": None,
        "rail": None,
        "weather": None,
        "penetrometer": None,
    }

    for i, header in enumerate(headers):
        if i >= len(cells):
            break
        val = cells[i].strip() or None

        if any(kw in header for kw in ["venue", "course", "track name", "meeting"]):
            # Handle "Sat 07-Feb Royal Randwick" format - extract venue name
            if val and re.match(r"^\w{3}\s+\d{1,2}-\w{3}\s+", val):
                # Strip date prefix like "Sat 07-Feb "
                val = re.sub(r"^\w{3}\s+\d{1,2}-\w{3}\s+", "", val)
            data["venue"] = val
        elif any(kw in header for kw in ["condition", "rating", "going"]):
            data["condition"] = val
        elif "rail" in header:
            data["rail"] = val
        elif "weather" in header:
            data["weather"] = val
        elif any(kw in header for kw in ["penetrometer", "peno"]):
            data["penetrometer"] = val

    return data


async def scrape_sportsbetform_conditions() -> list[dict[str, Any]]:
    """Scrape track conditions from sportsbetform.com.au (all states).

    This is a backup source when racingaustralia.horse fails.
    Returns list of dicts with venue, condition, rail, weather, penetrometer.
    """
    url = "https://www.sportsbetform.com.au/track-conditions/"
    logger.info(f"Scraping sportsbetform track conditions: {url}")
    conditions: list[dict[str, Any]] = []

    try:
        async with new_page(timeout=45000) as page:
            html = await wait_and_get_content(
                page, url, wait_selector="table, .track-table, .conditions-table"
            )
            soup = BeautifulSoup(html, "lxml")

            # Look for tables with track condition data
            tables = soup.select("table")
            for table in tables:
                rows = table.select("tr")
                if not rows:
                    continue

                # Try to identify headers
                first_row = rows[0]
                headers = [th.get_text(strip=True).lower() for th in first_row.select("th, td")]

                # Check if this looks like a track conditions table
                if not any(kw in " ".join(headers) for kw in ["venue", "track", "course", "condition", "rating"]):
                    continue

                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.select("td")]
                    if len(cells) < 2:
                        continue

                    entry = _map_cells(headers, cells)
                    if entry.get("venue"):
                        # Try to determine state from venue name
                        venue_lower = entry["venue"].lower()
                        for venue_key, state in VENUE_TO_STATE.items():
                            if venue_key in venue_lower:
                                entry["state"] = state
                                break
                        conditions.append(entry)

            # Alternative: look for div-based layouts
            if not conditions:
                track_items = soup.select(".track-condition, .venue-row, [class*='condition']")
                for item in track_items:
                    venue = item.select_one(".venue, .track-name, h3, h4, td:first-child")
                    condition = item.select_one(".condition, .rating, .track-rating")
                    weather = item.select_one(".weather")
                    rail = item.select_one(".rail")

                    if venue:
                        entry = {
                            "venue": venue.get_text(strip=True),
                            "condition": condition.get_text(strip=True) if condition else None,
                            "rail": rail.get_text(strip=True) if rail else None,
                            "weather": weather.get_text(strip=True) if weather else None,
                            "penetrometer": None,
                        }
                        conditions.append(entry)

        logger.info(f"Sportsbetform: found {len(conditions)} track conditions")
    except Exception as e:
        logger.error(f"Sportsbetform scrape failed: {e}")

    return conditions


async def get_track_condition_for_venue(venue: str, state: str = None) -> Optional[dict[str, Any]]:
    """Get track condition for a specific venue, trying multiple sources.

    This is a convenience function that tries:
    1. racingaustralia.horse for the state
    2. sportsbetform.com.au as fallback

    Returns dict with condition info or None if not found.
    """
    venue_lower = venue.lower()

    # Determine state if not provided
    if not state:
        for venue_key, mapped_state in VENUE_TO_STATE.items():
            if venue_key in venue_lower:
                state = mapped_state
                break

    if not state:
        state = "VIC"  # Default

    # Try primary source
    conditions = await scrape_track_conditions(state)
    for cond in conditions:
        if cond.get("venue") and cond["venue"].lower() in venue_lower:
            return cond
        if cond.get("venue") and venue_lower in cond["venue"].lower():
            return cond

    # Try sportsbetform fallback
    logger.info(f"Primary source found no condition for {venue}, trying sportsbetform...")
    fallback = await scrape_sportsbetform_conditions()
    for cond in fallback:
        if cond.get("venue") and cond["venue"].lower() in venue_lower:
            return cond
        if cond.get("venue") and venue_lower in cond["venue"].lower():
            return cond

    logger.warning(f"No track condition found for {venue} in any source")
    return None
