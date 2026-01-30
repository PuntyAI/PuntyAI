"""Scrape racing.com/calendar for today's meetings list."""

import logging
import re
from datetime import date
from typing import Any

from bs4 import BeautifulSoup

from punty.scrapers.playwright_base import new_page, wait_and_get_content

logger = logging.getLogger(__name__)


async def scrape_calendar(race_date: date | None = None) -> list[dict[str, Any]]:
    """Scrape racing.com calendar page for today's meetings.

    Returns a list of dicts, each with:
        - venue: str
        - state: str (e.g. "VIC", "NSW")
        - num_races: int
        - race_type: str (e.g. "Thoroughbred")
        - date: date
    """
    race_date = race_date or date.today()
    date_str = race_date.strftime("%Y-%m-%d")
    url = f"https://www.racing.com/calendar/{date_str}"

    logger.info(f"Scraping calendar for {race_date}: {url}")

    meetings: list[dict[str, Any]] = []

    async with new_page() as page:
        html = await wait_and_get_content(
            page, url, wait_selector=".meeting-list, .calendar-day, .race-meeting, table"
        )
        soup = BeautifulSoup(html, "lxml")

        # Try several selectors that racing.com calendar pages may use
        meeting_elements = soup.select(
            ".meeting-item, .meeting-row, .race-meeting, "
            "[data-meeting], tr.meeting, .calendar-meeting"
        )

        if not meeting_elements:
            # Fallback: look for links that contain venue info
            meeting_elements = soup.select("a[href*='/races/']")

        for elem in meeting_elements:
            venue = _extract_venue(elem)
            if not venue:
                continue

            state = _extract_state(elem) or _guess_state(venue)
            num_races = _extract_num_races(elem)
            race_type = _extract_race_type(elem)

            meetings.append({
                "venue": venue,
                "state": state,
                "num_races": num_races,
                "race_type": race_type,
                "date": race_date,
            })

        # Deduplicate by venue
        seen = set()
        unique = []
        for m in meetings:
            key = m["venue"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(m)
        meetings = unique

    logger.info(f"Found {len(meetings)} meetings for {race_date}")
    return meetings


def _extract_venue(elem) -> str | None:
    """Extract venue name from a meeting element."""
    # Try dedicated selectors
    for sel in [".venue-name", ".meeting-name", ".venue", "h3", "h4", ".name"]:
        tag = elem.select_one(sel)
        if tag:
            text = tag.get_text(strip=True)
            if text:
                return _clean_venue(text)

    # Try element text or href
    text = elem.get_text(strip=True)
    if text and len(text) < 60:
        return _clean_venue(text)

    href = elem.get("href", "")
    match = re.search(r"/races/([a-z\-]+)/", href)
    if match:
        return match.group(1).replace("-", " ").title()

    return None


def _clean_venue(text: str) -> str:
    """Clean venue name."""
    # Remove trailing info like "(VIC)" or race counts
    text = re.sub(r"\s*\(.*?\)\s*", " ", text)
    text = re.sub(r"\s*-\s*\d+\s*races?\s*$", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip().title()


def _extract_state(elem) -> str | None:
    """Extract state abbreviation."""
    for sel in [".state", ".location", ".region"]:
        tag = elem.select_one(sel)
        if tag:
            text = tag.get_text(strip=True).upper()
            if text in ("VIC", "NSW", "QLD", "SA", "WA", "TAS", "ACT", "NT"):
                return text

    text = elem.get_text()
    match = re.search(r"\b(VIC|NSW|QLD|SA|WA|TAS|ACT|NT)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def _guess_state(venue: str) -> str:
    """Guess state from known venue names."""
    mapping = {
        "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC",
        "sandown": "VIC", "cranbourne": "VIC", "pakenham": "VIC",
        "randwick": "NSW", "rosehill": "NSW", "warwick farm": "NSW",
        "canterbury": "NSW", "newcastle": "NSW", "kembla grange": "NSW",
        "doomben": "QLD", "eagle farm": "QLD", "gold coast": "QLD",
        "sunshine coast": "QLD",
        "morphettville": "SA", "murray bridge": "SA",
        "ascot": "WA", "belmont": "WA",
        "elwick": "TAS", "launceston": "TAS",
    }
    return mapping.get(venue.lower(), "")


def _extract_num_races(elem) -> int:
    """Extract number of races from element."""
    for sel in [".num-races", ".race-count", ".races"]:
        tag = elem.select_one(sel)
        if tag:
            match = re.search(r"(\d+)", tag.get_text())
            if match:
                return int(match.group(1))

    text = elem.get_text()
    match = re.search(r"(\d+)\s*race", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return 0


def _extract_race_type(elem) -> str:
    """Extract race type (Thoroughbred, Harness, Greyhound)."""
    text = elem.get_text().lower()
    if "harness" in text:
        return "Harness"
    if "greyhound" in text or "dogs" in text:
        return "Greyhound"
    return "Thoroughbred"
