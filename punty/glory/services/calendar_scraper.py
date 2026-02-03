"""Racing Australia Group 1 calendar scraper for Group One Glory."""

import logging
import re
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

from punty.scrapers.playwright_base import new_page

logger = logging.getLogger(__name__)

# Melbourne and Sydney venues for filtering
MELBOURNE_VENUES = {
    "flemington",
    "caulfield",
    "moonee valley",
    "sandown",
    "cranbourne",
}

SYDNEY_VENUES = {
    "randwick",
    "rosehill",
    "royal randwick",
    "warwick farm",
    "canterbury",
}

VIC_NSW_VENUES = MELBOURNE_VENUES | SYDNEY_VENUES


class RacingAustraliaCalendarScraper:
    """Scrape Group 1 race calendar from Racing Australia."""

    BASE_URL = "https://racingaustralia.horse/FreeFields/GroupAndListedRaces.aspx"

    # Months to scrape (calendar typically shows Feb-May for Autumn, Aug-Nov for Spring)
    MONTHS = [
        "February", "March", "April", "May",
        "August", "September", "October", "November", "December",
    ]

    async def scrape_group1_calendar(self, year: int) -> list[dict]:
        """Scrape all Group 1 races for Melbourne and Sydney.

        Args:
            year: The year to scrape

        Returns:
            List of race dictionaries with:
            - race_name: str
            - venue: str
            - race_date: datetime
            - distance: int (meters)
            - prize_money: Optional[int]
            - external_id: Optional[str]
        """
        all_races = []

        async with new_page(timeout=60000) as page:
            for month in self.MONTHS:
                try:
                    url = f"{self.BASE_URL}?Month={month}"
                    logger.info(f"Loading Racing Australia calendar for {month}: {url}")
                    await page.goto(url)
                    await page.wait_for_timeout(2000)

                    # Try to dismiss any popups
                    try:
                        popup = page.locator("button:has-text('Accept')")
                        if await popup.is_visible(timeout=1000):
                            await popup.click()
                    except Exception:
                        pass

                    # Get the page content and parse
                    html = await page.content()
                    month_races = self._parse_calendar_page(html, year)
                    logger.info(f"Found {len(month_races)} Group 1 races in {month}")
                    all_races.extend(month_races)

                except Exception as e:
                    logger.warning(f"Failed to scrape {month}: {e}")
                    continue

        # Deduplicate races by name and date
        seen = set()
        unique_races = []
        for race in all_races:
            key = (race["race_name"], race["race_date"].date())
            if key not in seen:
                seen.add(key)
                unique_races.append(race)

        logger.info(f"Found {len(unique_races)} unique Group 1 races for Melbourne/Sydney")
        return unique_races

    def _parse_calendar_page(self, html: str, year: int) -> list[dict]:
        """Parse the calendar page HTML and extract Group 1 races."""
        soup = BeautifulSoup(html, "html.parser")
        races = []

        # Find all tables on the page
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 4:
                    continue

                # Get cell text
                cell_texts = [c.get_text(strip=True) for c in cells]

                # Look for "Group 1" in any cell (classification column)
                is_group1 = False
                for text in cell_texts:
                    if "group 1" in text.lower():
                        is_group1 = True
                        break

                if not is_group1:
                    continue

                # Try to parse this row as a race
                race = self._parse_race_row(cell_texts, year)
                if race:
                    races.append(race)

        return races

    def _parse_race_row(self, cells: list[str], year: int) -> Optional[dict]:
        """Parse a table row into a race dictionary.

        Expected format from Racing Australia:
        - Cell 0: Date and Venue (e.g., "7-Feb-2026 Caulfield")
        - Cell 1: State (e.g., "VIC", "NSW")
        - Cell 2: Race Name with distance (e.g., "Black Caviar Lightning 1000M")
        - Cell 3: Classification (e.g., "Group 1")
        - Cell 4: Prizemoney
        """
        try:
            # First cell should have date and venue
            date_venue = cells[0] if cells else ""

            # Parse date - format is "7-Feb-2026" or "14-Feb-2026 Flemington"
            date_match = re.search(
                r"(\d{1,2})-([A-Za-z]{3})-(\d{4})",
                date_venue
            )

            if not date_match:
                return None

            day = date_match.group(1)
            month = date_match.group(2)
            parsed_year = date_match.group(3)

            # Parse the date
            try:
                race_date = datetime.strptime(f"{day}-{month}-{parsed_year}", "%d-%b-%Y")
            except ValueError:
                return None

            # Extract venue from the date/venue cell
            venue = None
            date_venue_lower = date_venue.lower()
            for v in VIC_NSW_VENUES:
                if v in date_venue_lower:
                    venue = v.title()
                    # Fix capitalization for multi-word venues
                    if venue == "Moonee Valley":
                        venue = "Moonee Valley"
                    elif venue == "Royal Randwick":
                        venue = "Royal Randwick"
                    break

            # Also check state column to filter
            state = cells[1].strip().upper() if len(cells) > 1 else ""
            if state not in ("VIC", "NSW"):
                return None

            if not venue:
                return None

            # Race name is typically in cell 2 or 3, with distance
            race_name = None
            distance = 0

            for cell in cells[2:5]:  # Check cells 2, 3, 4 for race name
                # Look for race name with distance pattern
                # Format: "Black Caviar Lightning 1000M" or "Sportsbet Blue Diamond Stakes 1200M"
                name_match = re.search(r"^(.+?)\s+(\d{3,4})\s*[Mm]?\s*$", cell.strip())
                if name_match:
                    race_name = name_match.group(1).strip()
                    distance = int(name_match.group(2))
                    break

                # Also try just extracting distance if name doesn't have it
                if not distance:
                    dist_match = re.search(r"(\d{3,4})\s*[Mm]", cell)
                    if dist_match:
                        d = int(dist_match.group(1))
                        if 800 <= d <= 4000:
                            distance = d

                # Try to find race name without distance
                if not race_name and len(cell) > 10:
                    # Skip cells that are just dates, states, or classifications
                    if not re.match(r"^\d+-[A-Za-z]{3}-\d{4}", cell) and \
                       cell.upper() not in ("VIC", "NSW", "QLD", "SA", "WA", "TAS") and \
                       "group" not in cell.lower() and \
                       "listed" not in cell.lower():
                        race_name = cell.strip()

            if not race_name:
                return None

            # Clean up race name - remove distance if still there
            race_name = re.sub(r"\s+\d{3,4}\s*[Mm]?\s*$", "", race_name).strip()

            # Remove sponsor prefixes if too long (keep the main race name)
            # e.g., "Sportsbet Blue Diamond Stakes" -> "Blue Diamond Stakes"
            if len(race_name) > 40:
                # Try to extract the core race name
                words = race_name.split()
                if len(words) > 3:
                    # Look for common race name patterns
                    for pattern in ["Stakes", "Cup", "Plate", "Handicap", "Classic", "Guineas", "Oaks", "Derby"]:
                        for i, word in enumerate(words):
                            if pattern.lower() in word.lower():
                                # Keep from this word back to a reasonable starting point
                                start = max(0, i - 3)
                                race_name = " ".join(words[start:i+1])
                                break

            # Default distance if not found
            if not distance:
                distance = 2000

            # Extract prize money
            prize_money = None
            for cell in cells:
                prize_match = re.search(r"\$[\d,]+", cell)
                if prize_match:
                    prize_str = prize_match.group(0).replace("$", "").replace(",", "")
                    try:
                        prize_money = int(float(prize_str) * 100)  # Convert to cents
                    except ValueError:
                        pass
                    break

            return {
                "race_name": race_name,
                "venue": venue,
                "race_date": race_date,
                "distance": distance,
                "prize_money": prize_money,
                "external_id": f"ra_{race_name.lower().replace(' ', '_')}_{race_date.strftime('%Y%m%d')}",
            }

        except Exception as e:
            logger.debug(f"Failed to parse race row: {e}")
            return None


# Hardcoded calendar for well-known races (fallback if scraping fails)
MELBOURNE_GROUP1_RACES = [
    # Autumn Carnival (Feb-April)
    {"race_name": "Black Caviar Lightning", "venue": "Flemington", "month": 2, "day": 14, "distance": 1000},
    {"race_name": "Oakleigh Plate", "venue": "Caulfield", "month": 2, "day": 21, "distance": 1100},
    {"race_name": "Blue Diamond Stakes", "venue": "Caulfield", "month": 2, "day": 21, "distance": 1200},
    {"race_name": "Futurity Stakes", "venue": "Caulfield", "month": 2, "day": 21, "distance": 1400},
    {"race_name": "Australian Guineas", "venue": "Flemington", "month": 2, "day": 28, "distance": 1600},
    {"race_name": "Newmarket Handicap", "venue": "Flemington", "month": 3, "day": 7, "distance": 1200},
    {"race_name": "Australian Cup", "venue": "Flemington", "month": 3, "day": 7, "distance": 2000},
    {"race_name": "William Reid Stakes", "venue": "Moonee Valley", "month": 3, "day": 21, "distance": 1200},

    # Spring Carnival (Sep-Nov)
    {"race_name": "Memsie Stakes", "venue": "Caulfield", "month": 9, "day": 5, "distance": 1400},
    {"race_name": "Makybe Diva Stakes", "venue": "Flemington", "month": 9, "day": 12, "distance": 1600},
    {"race_name": "Sir Rupert Clarke Stakes", "venue": "Caulfield", "month": 9, "day": 19, "distance": 1400},
    {"race_name": "Underwood Stakes", "venue": "Sandown", "month": 9, "day": 26, "distance": 1800},
    {"race_name": "Turnbull Stakes", "venue": "Flemington", "month": 10, "day": 3, "distance": 2000},
    {"race_name": "Caulfield Guineas", "venue": "Caulfield", "month": 10, "day": 10, "distance": 1600},
    {"race_name": "Caulfield Stakes", "venue": "Caulfield", "month": 10, "day": 10, "distance": 2000},
    {"race_name": "Thousand Guineas", "venue": "Caulfield", "month": 10, "day": 10, "distance": 1600},
    {"race_name": "Toorak Handicap", "venue": "Caulfield", "month": 10, "day": 10, "distance": 1600},
    {"race_name": "Caulfield Cup", "venue": "Caulfield", "month": 10, "day": 17, "distance": 2400},
    {"race_name": "Manikato Stakes", "venue": "Moonee Valley", "month": 10, "day": 23, "distance": 1200},
    {"race_name": "Cox Plate", "venue": "Moonee Valley", "month": 10, "day": 24, "distance": 2040},
    {"race_name": "Victoria Derby", "venue": "Flemington", "month": 11, "day": 1, "distance": 2500},
    {"race_name": "Coolmore Stud Stakes", "venue": "Flemington", "month": 11, "day": 1, "distance": 1200},
    {"race_name": "Melbourne Cup", "venue": "Flemington", "month": 11, "day": 2, "distance": 3200},
    {"race_name": "Kennedy Oaks", "venue": "Flemington", "month": 11, "day": 4, "distance": 2500},
    {"race_name": "Mackinnon Stakes", "venue": "Flemington", "month": 11, "day": 5, "distance": 2000},
    {"race_name": "Empire Rose Stakes", "venue": "Flemington", "month": 11, "day": 5, "distance": 1600},
]

SYDNEY_GROUP1_RACES = [
    # Autumn Carnival (Feb-April)
    {"race_name": "Chipping Norton Stakes", "venue": "Randwick", "month": 2, "day": 28, "distance": 1600},
    {"race_name": "Surround Stakes", "venue": "Randwick", "month": 2, "day": 28, "distance": 1400},
    {"race_name": "Canterbury Stakes", "venue": "Randwick", "month": 3, "day": 7, "distance": 1300},
    {"race_name": "Randwick Guineas", "venue": "Randwick", "month": 3, "day": 7, "distance": 1600},
    {"race_name": "Golden Slipper", "venue": "Rosehill", "month": 3, "day": 21, "distance": 1200},
    {"race_name": "George Ryder Stakes", "venue": "Rosehill", "month": 3, "day": 21, "distance": 1500},
    {"race_name": "Ranvet Stakes", "venue": "Rosehill", "month": 3, "day": 21, "distance": 2000},
    {"race_name": "Rosehill Guineas", "venue": "Rosehill", "month": 3, "day": 21, "distance": 2000},
    {"race_name": "The Galaxy", "venue": "Rosehill", "month": 3, "day": 21, "distance": 1100},
    {"race_name": "Vinery Stud Stakes", "venue": "Rosehill", "month": 4, "day": 4, "distance": 2000},
    {"race_name": "Doncaster Mile", "venue": "Randwick", "month": 4, "day": 4, "distance": 1600},
    {"race_name": "T J Smith Stakes", "venue": "Randwick", "month": 4, "day": 4, "distance": 1200},
    {"race_name": "ATC Australian Derby", "venue": "Randwick", "month": 4, "day": 4, "distance": 2400},
    {"race_name": "ATC Oaks", "venue": "Randwick", "month": 4, "day": 11, "distance": 2400},
    {"race_name": "Queen Elizabeth Stakes", "venue": "Randwick", "month": 4, "day": 11, "distance": 2000},
    {"race_name": "Sydney Cup", "venue": "Randwick", "month": 4, "day": 11, "distance": 3200},
    {"race_name": "Queen of the Turf Stakes", "venue": "Randwick", "month": 4, "day": 18, "distance": 1600},
    {"race_name": "Champagne Stakes", "venue": "Randwick", "month": 4, "day": 18, "distance": 1600},
    {"race_name": "All Aged Stakes", "venue": "Randwick", "month": 4, "day": 18, "distance": 1400},

    # Spring Carnival (Aug-Oct)
    {"race_name": "Winx Stakes", "venue": "Randwick", "month": 8, "day": 21, "distance": 1400},
    {"race_name": "George Main Stakes", "venue": "Randwick", "month": 9, "day": 18, "distance": 1600},
    {"race_name": "Flight Stakes", "venue": "Randwick", "month": 10, "day": 2, "distance": 1600},
    {"race_name": "Epsom Handicap", "venue": "Randwick", "month": 10, "day": 2, "distance": 1600},
    {"race_name": "The Metropolitan", "venue": "Randwick", "month": 10, "day": 2, "distance": 2400},
    {"race_name": "Spring Champion Stakes", "venue": "Randwick", "month": 10, "day": 9, "distance": 2000},
    {"race_name": "The Everest", "venue": "Randwick", "month": 10, "day": 16, "distance": 1200},
]


def get_hardcoded_races(year: int) -> list[dict]:
    """Get hardcoded Group 1 races as fallback."""
    races = []

    for race_template in MELBOURNE_GROUP1_RACES + SYDNEY_GROUP1_RACES:
        # Create a race for the specified year
        day = race_template.get("day", 15)
        race_date = datetime(year, race_template["month"], day)

        races.append({
            "race_name": race_template["race_name"],
            "venue": race_template["venue"],
            "race_date": race_date,
            "distance": race_template["distance"],
            "prize_money": None,
            "external_id": f"hc_{race_template['race_name'].lower().replace(' ', '_')}",
        })

    return races
