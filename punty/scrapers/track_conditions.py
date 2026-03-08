"""Track conditions scraper (AU + NZ).

AU: Scrapes official track conditions from racingaustralia.horse.
NZ: Scrapes track conditions from loveracing.nz.
Uses httpx (static HTML pages, no JS rendering needed).
"""

import asyncio
import logging
import re
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ALL_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]

# NZ racing club → venue/track aliases (loveracing.nz uses club names, not venue names)
_NZ_CLUB_ALIASES: dict[str, list[str]] = {
    "wyndham rc": ["riverton"],
    "south waikato rc": ["matamata"],
    "manawatu rc": ["awapuni", "trentham"],
    "auckland thoroughbred racing": ["ellerslie"],
    "canterbury jockey club": ["riccarton"],
    "otago racing club": ["wingatui"],
    "waikato racing club": ["te rapa"],
    "hawke's bay racing": ["hastings"],
    "taranaki thoroughbred racing": ["new plymouth"],
    "racing tauranga": ["tauranga"],
    "whanganui racing club": ["wanganui"],
    "gore racing club": ["gore"],
}

_BASE_URL = (
    "https://www.racingaustralia.horse/InteractiveForm/"
    "TrackCondition.aspx?State={state}"
)

from punty.venues import SPONSOR_PREFIXES as _SPONSOR_PREFIXES_LIST

# Sponsor prefixes to strip when matching venue names (tuple with trailing space)
_SPONSOR_PREFIXES = tuple(p + " " for p in _SPONSOR_PREFIXES_LIST)

from punty.venues import get_all_venues

# Venue name → state mapping (from centralised venues registry)
VENUE_TO_STATE: dict[str, str] = get_all_venues()


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
    from punty.venues import normalize_venue
    ra = _strip_sponsor(ra_venue)
    mv = _strip_sponsor(meeting_venue)
    if ra == mv:
        return True
    # Check via venue alias normalization (e.g., "yarra valley" → "yarra glen")
    if normalize_venue(ra) == normalize_venue(mv):
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
    """Get track condition for a specific meeting venue.

    AU venues: resolves venue → state, scrapes racingaustralia.horse.
    NZ venues: scrapes loveracing.nz meeting overview pages.
    Returns dict with condition info or None if not found.
    """
    state = _resolve_state(venue)
    if not state:
        logger.warning(f"Cannot resolve state for venue: {venue}")
        return None

    # NZ venues — use loveracing.nz
    if state == "NZ":
        return await _get_nz_conditions(venue)

    conditions = await scrape_track_conditions(state)
    for cond in conditions:
        ra_venue = cond.get("venue", "")
        if _match_venue(ra_venue, venue):
            logger.info(f"RA match for '{venue}': {cond}")
            return cond

    logger.info(f"No RA track condition match for '{venue}' in {state}")
    return None


# ---------------------------------------------------------------------------
# New Zealand track conditions (loveracing.nz)
# ---------------------------------------------------------------------------

_NZ_RACEINFO_URL = "https://loveracing.nz/RaceInfo.aspx"
_NZ_MEETING_URL = "https://loveracing.nz/RaceInfo/{meeting_id}/Meeting-Overview.aspx"


async def _get_nz_meeting_ids() -> list[dict[str, Any]]:
    """Scrape loveracing.nz/RaceInfo.aspx to find meeting IDs and venue names.

    Returns list of dicts: {"meeting_id": str, "venue": str, "club": str}
    """
    meetings: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(_NZ_RACEINFO_URL, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch loveracing.nz RaceInfo: {e}")
        return meetings

    soup = BeautifulSoup(html, "html.parser")

    # Find all meeting links: /RaceInfo/{id}/Meeting-Overview.aspx
    for link in soup.find_all("a", href=re.compile(r"/RaceInfo/\d+/Meeting-Overview")):
        href = link.get("href", "")
        m = re.search(r"/RaceInfo/(\d+)/Meeting-Overview", href)
        if not m:
            continue
        meeting_id = m.group(1)
        club = link.get_text(strip=True)

        # Venue name is often in a sibling or parent cell — look for it
        # in the surrounding text (table row or list item)
        parent = link.find_parent(["tr", "li", "div"])
        venue = ""
        if parent:
            text = parent.get_text(" ", strip=True)
            # Common patterns: "Sat 7 March | Venue Name" or "Venue: Name"
            # The venue is usually the last meaningful text after the date
            for part in text.split("|"):
                part = part.strip()
                # Skip parts that are dates or the club name
                if re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s", part):
                    continue
                if part == club:
                    continue
                if part:
                    venue = part

        meetings.append({
            "meeting_id": meeting_id,
            "club": club,
            "venue": venue or club,
        })

    logger.info(f"loveracing.nz: found {len(meetings)} NZ meetings")
    return meetings


def _parse_nz_going(text: str) -> Optional[str]:
    """Parse NZ going string like 'Soft5 6.30 am 07/03/26' → 'Soft 5'."""
    if not text:
        return None
    # Match patterns: "Soft5", "Good3", "Heavy10", "Dead4"
    m = re.match(r"(Good|Soft|Heavy|Dead|Slow|Synthetic|Firm|Wet Fast)\s*(\d+)?", text, re.IGNORECASE)
    if m:
        condition = m.group(1).capitalize()
        rating = m.group(2) or ""
        return f"{condition} {rating}".strip()
    return text.split()[0] if text.strip() else None


async def _scrape_nz_meeting_conditions(meeting_id: str) -> Optional[dict[str, Any]]:
    """Scrape track conditions from a single loveracing.nz meeting page."""
    url = _NZ_MEETING_URL.format(meeting_id=meeting_id)
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.error(f"Failed to fetch NZ meeting {meeting_id}: {e}")
        return None

    soup = BeautifulSoup(html, "html.parser")
    data: dict[str, Any] = {
        "venue": None,
        "condition": None,
        "rail": None,
        "weather": None,
        "penetrometer": None,
        "rainfall": None,
        "irrigation": None,
    }

    # Extract venue name from page title/heading
    title = soup.find("h1") or soup.find("h2")
    if title:
        data["venue"] = title.get_text(strip=True)

    # Parse condition sections: <h4>Going</h4>, <h4>Weather</h4>, <h4>Rail</h4>
    for h4 in soup.find_all(["h4", "h3"]):
        label = h4.get_text(strip=True).lower()
        # Get the next <em> or <i> sibling for the value
        em = h4.find_next(["em", "i", "span"])
        if not em:
            continue
        val = em.get_text(strip=True)

        if "going" in label:
            data["condition"] = _parse_nz_going(val)
        elif "weather" in label:
            data["weather"] = val
        elif "rail" in label:
            # "Out 3m | No Rain Last 24 Hours | 2mm Rain Last 7 Days | 3mm Irrigation"
            parts = [p.strip() for p in val.split("|")]
            if parts:
                data["rail"] = parts[0]
            for part in parts[1:]:
                pl = part.lower()
                if "rain" in pl and "24" in pl:
                    data["rainfall"] = part
                elif "rain" in pl and "7" in pl:
                    data["rainfall"] = (data["rainfall"] + "; " + part) if data["rainfall"] else part
                elif "irrigation" in pl:
                    data["irrigation"] = part

    # Also try img alt text for going if <em> parsing missed it
    if not data["condition"]:
        for img in soup.find_all("img", src=re.compile(r"icon-going")):
            alt = img.get("alt", "") or img.get("title", "")
            if alt:
                data["condition"] = _parse_nz_going(alt)
                break

    return data if data["condition"] else None


async def _get_nz_conditions(venue: str) -> Optional[dict[str, Any]]:
    """Get track conditions for an NZ venue from loveracing.nz."""
    meetings = await _get_nz_meeting_ids()
    if not meetings:
        logger.warning("No NZ meetings found on loveracing.nz")
        return None

    # Try to match venue to a meeting
    venue_lower = venue.lower().strip()
    for meet in meetings:
        meet_venue = meet["venue"].lower().strip()
        meet_club = meet["club"].lower().strip()
        # Direct match
        matched = (venue_lower in meet_venue or meet_venue in venue_lower
                   or venue_lower in meet_club or meet_club in venue_lower
                   or _match_venue(meet_venue, venue))
        # Club alias match (e.g. "wyndham rc" → ["riverton"])
        if not matched:
            for club_key, aliases in _NZ_CLUB_ALIASES.items():
                if club_key in meet_club and venue_lower in aliases:
                    matched = True
                    break
        if matched:
            logger.info(f"NZ venue match: '{venue}' -> meeting {meet['meeting_id']} ({meet['venue']})")
            cond = await _scrape_nz_meeting_conditions(meet["meeting_id"])
            if cond:
                cond["venue"] = venue
                logger.info(f"NZ conditions for '{venue}': {cond}")
                return cond

    logger.info(f"No NZ meeting match for '{venue}' on loveracing.nz")
    return None
