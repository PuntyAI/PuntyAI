"""racingaustralia.horse scraper for track conditions."""

import logging
import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from punty.scrapers.playwright_base import new_page, wait_and_get_content

logger = logging.getLogger(__name__)

# State codes to URL parameter
STATE_CODES = {
    "VIC": "V",
    "NSW": "N",
    "QLD": "Q",
    "SA": "S",
    "WA": "W",
    "TAS": "T",
    "ACT": "A",
    "NT": "D",
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

            if not any(kw in " ".join(headers) for kw in ["venue", "track", "condition", "course"]):
                continue

            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.select("td")]
                if len(cells) < 2:
                    continue

                entry = _map_cells(headers, cells)
                if entry.get("venue"):
                    conditions.append(entry)

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

        if any(kw in header for kw in ["venue", "course", "track name"]):
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
