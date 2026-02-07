"""TabTouch.com.au scraper for exotic dividends (quaddie, trifecta, etc).

TabTouch serves server-rendered HTML (no JS required), making it reliable
for scraping from environments where TAB.com.au is blocked by Akamai.
"""

import logging
import re
from datetime import date
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.tabtouch.com.au"


async def scrape_meeting_exotics(
    venue_code: str, race_date: date
) -> dict[int, dict[str, str]]:
    """Scrape exotic dividends for all races in a meeting.

    Args:
        venue_code: TabTouch venue code (e.g. 'srx' for Newcastle)
        race_date: Date of the meeting

    Returns:
        {race_number: {exotic_type: dividend_str, ...}, ...}
        e.g. {4: {"quaddie": "24.50", "trifecta": "72.80"}, ...}
    """
    url = f"{BASE_URL}/racing/{race_date.isoformat()}/{venue_code}"
    logger.info(f"Scraping TabTouch exotics: {url}")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        if resp.status_code != 200:
            logger.warning(f"TabTouch returned {resp.status_code} for {url}")
            return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    results: dict[int, dict[str, str]] = {}

    # Each race is in a large <tr> that contains all results.
    # We parse by finding all exotic rows: <strong>ExoticType</strong> (legs) ... dividend
    # The pattern in td cells is: [pool_name] [combo] [dividend]
    all_trs = soup.find_all("tr")

    for tr in all_trs:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # Extract text from all tds
        td_texts = []
        for td in tds:
            text = td.get_text(separator=" ", strip=True)
            td_texts.append(text)

        # Find the race number (second td should be a small integer for race#)
        race_num = None
        if len(td_texts) > 1:
            try:
                num = int(td_texts[1])
                if 1 <= num <= 12:
                    race_num = num
            except (ValueError, IndexError):
                pass

        if race_num is None:
            # Check if this is a standalone quaddie row (e.g. "Quaddie (4,5,6,7) 3-1-9-3 419.40")
            full_text = " ".join(td_texts)
            m = re.search(
                r"(Quaddie|Early Quaddie)\s*\((\d+(?:,\d+)+)\)\s+([\d-]+)\s+([\d,.]+)",
                full_text,
                re.IGNORECASE,
            )
            if m:
                legs_str = m.group(2)  # e.g. "4,5,6,7"
                last_leg = int(legs_str.split(",")[-1])
                dividend = m.group(4).replace(",", "")
                key = "quaddie" if "early" not in m.group(1).lower() else "early quaddie"
                results.setdefault(last_leg, {})[key] = dividend
            continue

        # Parse exotic pools from this race's td cells
        race_exotics: dict[str, str] = {}
        i = 0
        while i < len(td_texts):
            text = td_texts[i]
            # Match exotic pool names
            m = re.match(
                r"(Quinella|Exacta|Trifecta|First 4|Double|Quaddie|Early Quaddie)"
                r"(?:\s*\([\d,]+\))?\s*$",
                text,
                re.IGNORECASE,
            )
            if m and i + 2 < len(td_texts):
                pool_name = m.group(1).lower().replace(" ", "")
                # combo is next, dividend after
                combo = td_texts[i + 1].strip()
                div_text = td_texts[i + 2].strip().replace(",", "")
                try:
                    float(div_text)
                    # For quaddie, determine which one based on legs
                    if "quaddie" in pool_name:
                        legs_m = re.search(r"\((\d+(?:,\d+)+)\)", text)
                        if legs_m:
                            legs = legs_m.group(1)
                            last_leg = int(legs.split(",")[-1])
                            # Store on the last leg's race
                            if last_leg != race_num:
                                results.setdefault(last_leg, {})[pool_name] = div_text
                                i += 3
                                continue
                    race_exotics[pool_name] = div_text
                except ValueError:
                    pass
                i += 3
                continue
            i += 1

        if race_exotics:
            results.setdefault(race_num, {}).update(race_exotics)

    logger.info(f"TabTouch exotics for {venue_code}: {len(results)} races with data")
    return results


async def find_venue_code(venue_name: str, race_date: date) -> Optional[str]:
    """Look up TabTouch venue code from venue name by scraping the day's calendar.

    Returns venue code (e.g. 'srx') or None if not found.
    """
    url = f"{BASE_URL}/racing/{race_date.isoformat()}"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        if resp.status_code != 200:
            return None

    # Find meeting links: href="/racing/2026-02-02/srx" class="meeting">NEWCASTLE
    venue_lower = venue_name.lower()
    # Strip common prefixes like "Beaumont " from "Beaumont Newcastle"
    venue_words = venue_lower.split()

    for m in re.finditer(
        r'href="/racing/[^/]+/([^"]+)"\s+class="meeting">([^<]+)',
        resp.text,
    ):
        code = m.group(1)
        tt_venue = m.group(2).strip().lower()
        # Match if any word in our venue name matches the tabtouch venue
        if tt_venue == venue_lower or any(w in tt_venue for w in venue_words if len(w) > 3):
            return code

    return None
