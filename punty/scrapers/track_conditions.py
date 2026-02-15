"""Racing Australia track conditions scraper.

Scrapes official track conditions from racingaustralia.horse.
This is the authoritative source for Australian track conditions.
Uses httpx (static HTML page, no JS rendering needed).
"""

import asyncio
import logging
import re
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ALL_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]

_BASE_URL = (
    "https://www.racingaustralia.horse/InteractiveForm/"
    "TrackCondition.aspx?State={state}"
)

# Sponsor prefixes to strip when matching venue names
_SPONSOR_PREFIXES = (
    "sportsbet ", "ladbrokes ", "bet365 ", "picklebet park ",
    "southside ", "tab ", "aquis ", "neds ", "aquis park ",
    "beaumont ", "picklebet ",
)

# Venue name → state mapping (reuses pattern from deep_learning/importer.py)
VENUE_TO_STATE: dict[str, str] = {
    # NSW
    "randwick": "NSW", "rosehill": "NSW", "warwick farm": "NSW",
    "canterbury": "NSW", "newcastle": "NSW", "kembla grange": "NSW",
    "wyong": "NSW", "gosford": "NSW", "hawkesbury": "NSW",
    "scone": "NSW", "tamworth": "NSW", "mudgee": "NSW",
    "dubbo": "NSW", "bathurst": "NSW", "orange": "NSW",
    "wagga": "NSW", "albury": "NSW", "canberra": "ACT",
    "queanbeyan": "NSW", "moruya": "NSW", "nowra": "NSW",
    "port macquarie": "NSW", "coffs harbour": "NSW", "grafton": "NSW",
    "ballina": "NSW", "lismore": "NSW", "taree": "NSW",
    "muswellbrook": "NSW", "cessnock": "NSW", "glen innes": "NSW",
    "gilgandra": "NSW", "inverell": "NSW", "gundagai": "NSW",
    "quirindi": "NSW", "coonamble": "NSW", "wellington": "NSW",
    "cowra": "NSW", "young": "NSW", "sapphire coast": "NSW",
    "moree": "NSW", "narromine": "NSW", "armidale": "NSW",
    "goulburn": "NSW", "broken hill": "NSW", "tuncurry": "NSW",
    "corowa": "NSW", "casino": "NSW", "condobolin": "NSW",
    "canterbury park": "NSW", "rosehill gardens": "NSW",
    # VIC
    "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC",
    "sandown": "VIC", "cranbourne": "VIC", "pakenham": "VIC",
    "mornington": "VIC", "geelong": "VIC", "ballarat": "VIC",
    "bendigo": "VIC", "sale": "VIC", "wangaratta": "VIC",
    "wodonga": "VIC", "hamilton": "VIC", "stawell": "VIC",
    "werribee": "VIC", "kilmore": "VIC", "yarra glen": "VIC",
    "stony creek": "VIC", "healesville": "VIC", "woolamai": "VIC",
    "seymour": "VIC", "donald": "VIC", "echuca": "VIC",
    "terang": "VIC", "warrnambool": "VIC", "merton": "VIC",
    "kyneton": "VIC", "colac": "VIC", "swan hill": "VIC",
    "mildura": "VIC", "tatura": "VIC", "ararat": "VIC",
    "avoca": "VIC", "caulfield heath": "VIC", "moe": "VIC",
    "dederang": "VIC", "yarra valley": "VIC",
    # QLD
    "eagle farm": "QLD", "doomben": "QLD", "gold coast": "QLD",
    "sunshine coast": "QLD", "ipswich": "QLD", "toowoomba": "QLD",
    "rockhampton": "QLD", "mackay": "QLD", "townsville": "QLD",
    "cairns": "QLD", "atherton": "QLD", "roma": "QLD",
    "dalby": "QLD", "beaudesert": "QLD", "kilcoy": "QLD",
    "bundaberg": "QLD", "gladstone": "QLD", "emerald": "QLD",
    "cannon park": "QLD", "innisfail": "QLD", "goondiwindi": "QLD",
    "mount isa": "QLD", "warwick": "QLD",
    # SA
    "morphettville": "SA", "murray bridge": "SA", "gawler": "SA",
    "oakbank": "SA", "naracoorte": "SA", "port augusta": "SA",
    "balaklava": "SA", "clare": "SA", "strathalbyn": "SA",
    "mount gambier": "SA", "port lincoln": "SA", "kingscote": "SA",
    # WA
    "ascot": "WA", "belmont": "WA", "pinjarra": "WA",
    "bunbury": "WA", "northam": "WA", "geraldton": "WA",
    "kalgoorlie": "WA", "albany": "WA", "esperance": "WA",
    "narrogin": "WA", "york": "WA", "lark hill": "WA",
    "carnarvon": "WA", "broome": "WA",
    # TAS
    "hobart": "TAS", "launceston": "TAS", "devonport": "TAS",
    "longford": "TAS", "king island": "TAS",
    # NT
    "fannie bay": "NT", "alice springs": "NT", "darwin": "NT",
    "pioneer park": "NT",
}


def _strip_sponsor(venue: str) -> str:
    """Strip sponsor prefix from venue name for matching."""
    v = venue.lower().strip()
    for prefix in _SPONSOR_PREFIXES:
        if v.startswith(prefix):
            v = v[len(prefix):]
            break
    # Also strip trailing "park" if it's a suffix like "Cannon Park"
    return v


def _resolve_state(venue: str) -> str | None:
    """Resolve a venue name to its state code."""
    v = _strip_sponsor(venue)
    # Direct match
    if v in VENUE_TO_STATE:
        return VENUE_TO_STATE[v]
    # Partial match — check if any known venue is contained in the input
    for key, state in VENUE_TO_STATE.items():
        if key in v or v in key:
            return state
    return None


def _match_venue(ra_venue: str, meeting_venue: str) -> bool:
    """Check if a Racing Australia venue name matches a meeting venue."""
    ra = _strip_sponsor(ra_venue)
    mv = _strip_sponsor(meeting_venue)
    if ra == mv:
        return True
    # One contains the other
    if ra in mv or mv in ra:
        return True
    # Handle "Cannon Park" in RA matching "Ladbrokes Cannon Park" in DB
    # by checking if the core words overlap
    ra_words = set(ra.split())
    mv_words = set(mv.split())
    if ra_words and mv_words and ra_words == mv_words:
        return True
    return False


async def scrape_track_conditions(state: str) -> list[dict[str, Any]]:
    """Scrape track conditions from racingaustralia.horse for a given state.

    Returns list of dicts with:
        - venue: str (sponsor-stripped)
        - condition: str | None (e.g. "Good 4", "Soft 5")
        - rail: str | None
        - weather: str | None
        - penetrometer: float | None
        - rainfall: str | None (raw string e.g. "2mm/24hrs, 19mm/7d")
        - irrigation: str | None (raw string)
    """
    url = _BASE_URL.format(state=state.upper())
    logger.info(f"Scraping RA track conditions for {state}: {url}")
    conditions: list[dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch RA conditions for {state}: {e}")
        return conditions

    soup = BeautifulSoup(html, "html.parser")

    for table in soup.select("table"):
        rows = table.select("tr")
        if not rows:
            continue
        headers = [
            th.get_text(strip=True).lower()
            for th in rows[0].select("th, td")
        ]
        # Identify the track conditions table
        header_str = " ".join(headers)
        if "meeting" not in header_str and "track" not in header_str:
            continue

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) < 3:
                continue
            entry = _map_cells(headers, cells)
            if entry.get("venue"):
                conditions.append(entry)

    logger.info(f"RA {state}: found {len(conditions)} track conditions")
    return conditions


def _map_cells(headers: list[str], cells: list[str]) -> dict[str, Any]:
    """Map table cells to a structured dict using header names."""
    data: dict[str, Any] = {
        "venue": None,
        "condition": None,
        "rail": None,
        "weather": None,
        "penetrometer": None,
        "rainfall": None,
        "irrigation": None,
    }

    for i, header in enumerate(headers):
        if i >= len(cells):
            break
        val = cells[i].strip() or None

        if any(kw in header for kw in ["meeting", "venue", "course"]):
            # Strip date prefix "Mon 16-Feb " from "Mon 16-Feb Beaumont Newcastle"
            if val and re.match(r"^\w{3}\s+\d{1,2}-\w{3}\s+", val):
                val = re.sub(r"^\w{3}\s+\d{1,2}-\w{3}\s+", "", val)
            data["venue"] = val
        elif "condition" in header or "rating" in header or "going" in header:
            if val and val.upper() != "N/A":
                data["condition"] = val
        elif "rail" in header:
            data["rail"] = val
        elif "weather" in header:
            data["weather"] = val
        elif "penetrometer" in header or "peno" in header:
            if val:
                try:
                    data["penetrometer"] = float(val)
                except ValueError:
                    pass
        elif "rainfall" in header:
            data["rainfall"] = val
        elif "irrigation" in header:
            data["irrigation"] = val

    return data


async def scrape_all_track_conditions() -> list[dict[str, Any]]:
    """Scrape track conditions for all Australian states in parallel."""
    tasks = [scrape_track_conditions(state) for state in ALL_STATES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"RA scrape failed for {ALL_STATES[i]}: {result}")
        else:
            combined.extend(result)
    return combined


async def get_conditions_for_meeting(venue: str) -> Optional[dict[str, Any]]:
    """Get RA track condition for a specific meeting venue.

    Resolves venue → state, scrapes that state, and fuzzy-matches the venue.
    Returns dict with condition info or None if not found.
    """
    state = _resolve_state(venue)
    if not state:
        logger.warning(f"Cannot resolve state for venue: {venue}")
        return None

    conditions = await scrape_track_conditions(state)
    for cond in conditions:
        ra_venue = cond.get("venue", "")
        if _match_venue(ra_venue, venue):
            logger.info(f"RA match for '{venue}': {cond}")
            return cond

    logger.info(f"No RA track condition match for '{venue}' in {state}")
    return None
