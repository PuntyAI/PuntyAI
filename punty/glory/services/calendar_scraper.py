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

    BASE_URL = "http://racingaustralia.horse/FreeFields/GroupAndListedRaces.aspx"

    # Month tabs in the Racing Australia calendar
    MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
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
        races = []

        async with new_page(timeout=60000) as page:
            logger.info(f"Loading Racing Australia calendar: {self.BASE_URL}")
            await page.goto(self.BASE_URL)

            # Wait for the page to load
            await page.wait_for_timeout(3000)

            # Try to dismiss any popups
            try:
                popup = page.locator("button:has-text('Accept')")
                if await popup.is_visible(timeout=2000):
                    await popup.click()
            except Exception:
                pass

            # Get the page content and parse
            html = await page.content()
            races = self._parse_calendar_page(html, year)

            # The Racing Australia site may have tabs for different months
            # Try clicking through each month tab if they exist
            for month in self.MONTHS:
                try:
                    # Look for month tab/link
                    month_link = page.locator(f"a:has-text('{month}'), button:has-text('{month}')")
                    if await month_link.count() > 0:
                        await month_link.first.click()
                        await page.wait_for_timeout(2000)
                        html = await page.content()
                        month_races = self._parse_calendar_page(html, year)
                        races.extend(month_races)
                except Exception as e:
                    logger.debug(f"No tab for {month}: {e}")

        # Deduplicate races by name and date
        seen = set()
        unique_races = []
        for race in races:
            key = (race["race_name"], race["race_date"].date())
            if key not in seen:
                seen.add(key)
                unique_races.append(race)

        logger.info(f"Found {len(unique_races)} Group 1 races for Melbourne/Sydney")
        return unique_races

    def _parse_calendar_page(self, html: str, year: int) -> list[dict]:
        """Parse the calendar page HTML and extract Group 1 races."""
        soup = BeautifulSoup(html, "html.parser")
        races = []

        # Find the table with race listings
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:
                    continue

                # Get cell text
                cell_texts = [c.get_text(strip=True) for c in cells]
                row_text = " ".join(cell_texts).lower()

                # Check if this is a Group 1 race
                if "[group 1]" not in row_text and "group 1" not in row_text:
                    continue

                # Check if it's a Melbourne or Sydney venue
                venue = self._extract_venue(row_text)
                if not venue:
                    continue

                # Extract race details
                race = self._parse_race_row(cell_texts, year, venue)
                if race:
                    races.append(race)

        return races

    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract and validate venue from text."""
        text_lower = text.lower()

        for venue in VIC_NSW_VENUES:
            if venue in text_lower:
                return venue.title()

        return None

    def _parse_race_row(
        self, cells: list[str], year: int, venue: str
    ) -> Optional[dict]:
        """Parse a table row into a race dictionary."""
        try:
            # Try to find date, race name, distance, prize
            date_str = None
            race_name = None
            distance = 0
            prize_money = None

            for cell in cells:
                # Check for date pattern (e.g., "25 Feb", "1 November")
                date_match = re.search(
                    r"(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
                    cell,
                    re.IGNORECASE,
                )
                if date_match and not date_str:
                    day = date_match.group(1)
                    month = date_match.group(2)
                    date_str = f"{day} {month} {year}"

                # Check for race name (contains Group 1 or common race words)
                if (
                    "group 1" in cell.lower()
                    or "cup" in cell.lower()
                    or "stakes" in cell.lower()
                    or "classic" in cell.lower()
                    or "guineas" in cell.lower()
                    or "oaks" in cell.lower()
                    or "derby" in cell.lower()
                ):
                    # Clean up the race name
                    name = re.sub(r"\[GROUP \d\]", "", cell, flags=re.IGNORECASE).strip()
                    name = re.sub(r"\s+", " ", name)
                    if len(name) > 3 and not race_name:
                        race_name = name

                # Check for distance (e.g., "1600m", "2000")
                dist_match = re.search(r"(\d{3,4})\s*m?", cell)
                if dist_match and not distance:
                    d = int(dist_match.group(1))
                    if 800 <= d <= 4000:  # Valid race distance
                        distance = d

                # Check for prize money (e.g., "$5,000,000", "$2M")
                prize_match = re.search(r"\$[\d,]+(?:\.\d+)?[MmKk]?", cell)
                if prize_match and not prize_money:
                    prize_str = prize_match.group(0)
                    prize_money = self._parse_prize(prize_str)

            # Must have at minimum a date and race name
            if not date_str or not race_name:
                return None

            # Parse the date
            try:
                race_date = datetime.strptime(date_str, "%d %b %Y")
            except ValueError:
                try:
                    race_date = datetime.strptime(date_str, "%d %B %Y")
                except ValueError:
                    return None

            # Default distance if not found
            if not distance:
                distance = 2000  # Default for Group 1

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

    def _parse_prize(self, prize_str: str) -> Optional[int]:
        """Parse a prize string to cents."""
        try:
            # Remove $ and commas
            cleaned = prize_str.replace("$", "").replace(",", "").strip()

            # Handle M/K suffixes
            multiplier = 1
            if cleaned.lower().endswith("m"):
                multiplier = 1_000_000
                cleaned = cleaned[:-1]
            elif cleaned.lower().endswith("k"):
                multiplier = 1_000
                cleaned = cleaned[:-1]

            amount = float(cleaned)
            return int(amount * multiplier * 100)  # Convert to cents

        except Exception:
            return None


# Alternative: Hardcoded calendar for well-known races
# This can be used as a fallback if scraping fails

MELBOURNE_GROUP1_RACES = [
    # Autumn Carnival (Feb-April)
    {"race_name": "Black Caviar Lightning", "venue": "Flemington", "month": 2, "distance": 1000},
    {"race_name": "Oakleigh Plate", "venue": "Caulfield", "month": 2, "distance": 1100},
    {"race_name": "Blue Diamond Stakes", "venue": "Caulfield", "month": 2, "distance": 1200},
    {"race_name": "Futurity Stakes", "venue": "Caulfield", "month": 2, "distance": 1400},
    {"race_name": "Australian Guineas", "venue": "Flemington", "month": 3, "distance": 1600},
    {"race_name": "Newmarket Handicap", "venue": "Flemington", "month": 3, "distance": 1200},
    {"race_name": "Australian Cup", "venue": "Flemington", "month": 3, "distance": 2000},
    {"race_name": "William Reid Stakes", "venue": "Moonee Valley", "month": 3, "distance": 1200},
    {"race_name": "Lexus Newmarket Stakes", "venue": "Flemington", "month": 3, "distance": 1200},

    # Spring Carnival (Sep-Nov)
    {"race_name": "Memsie Stakes", "venue": "Caulfield", "month": 9, "distance": 1400},
    {"race_name": "Makybe Diva Stakes", "venue": "Flemington", "month": 9, "distance": 1600},
    {"race_name": "Sir Rupert Clarke Stakes", "venue": "Caulfield", "month": 9, "distance": 1400},
    {"race_name": "Underwood Stakes", "venue": "Sandown", "month": 9, "distance": 1800},
    {"race_name": "Turnbull Stakes", "venue": "Flemington", "month": 10, "distance": 2000},
    {"race_name": "Caulfield Guineas", "venue": "Caulfield", "month": 10, "distance": 1600},
    {"race_name": "Caulfield Stakes", "venue": "Caulfield", "month": 10, "distance": 2000},
    {"race_name": "Thousand Guineas", "venue": "Caulfield", "month": 10, "distance": 1600},
    {"race_name": "Toorak Handicap", "venue": "Caulfield", "month": 10, "distance": 1600},
    {"race_name": "Caulfield Cup", "venue": "Caulfield", "month": 10, "distance": 2400},
    {"race_name": "Manikato Stakes", "venue": "Moonee Valley", "month": 10, "distance": 1200},
    {"race_name": "Cox Plate", "venue": "Moonee Valley", "month": 10, "distance": 2040},
    {"race_name": "Victoria Derby", "venue": "Flemington", "month": 11, "distance": 2500},
    {"race_name": "Coolmore Stud Stakes", "venue": "Flemington", "month": 11, "distance": 1200},
    {"race_name": "Melbourne Cup", "venue": "Flemington", "month": 11, "distance": 3200},
    {"race_name": "Kennedy Oaks", "venue": "Flemington", "month": 11, "distance": 2500},
    {"race_name": "Mackinnon Stakes", "venue": "Flemington", "month": 11, "distance": 2000},
    {"race_name": "Empire Rose Stakes", "venue": "Flemington", "month": 11, "distance": 1600},
]

SYDNEY_GROUP1_RACES = [
    # Autumn Carnival (Feb-April)
    {"race_name": "Chipping Norton Stakes", "venue": "Randwick", "month": 2, "distance": 1600},
    {"race_name": "Canterbury Stakes", "venue": "Randwick", "month": 3, "distance": 1300},
    {"race_name": "Surround Stakes", "venue": "Randwick", "month": 2, "distance": 1400},
    {"race_name": "Randwick Guineas", "venue": "Randwick", "month": 3, "distance": 1600},
    {"race_name": "Golden Slipper", "venue": "Rosehill", "month": 3, "distance": 1200},
    {"race_name": "George Ryder Stakes", "venue": "Rosehill", "month": 3, "distance": 1500},
    {"race_name": "Ranvet Stakes", "venue": "Rosehill", "month": 3, "distance": 2000},
    {"race_name": "Rosehill Guineas", "venue": "Rosehill", "month": 3, "distance": 2000},
    {"race_name": "The Galaxy", "venue": "Rosehill", "month": 3, "distance": 1100},
    {"race_name": "Vinery Stud Stakes", "venue": "Rosehill", "month": 4, "distance": 2000},
    {"race_name": "Doncaster Mile", "venue": "Randwick", "month": 4, "distance": 1600},
    {"race_name": "T J Smith Stakes", "venue": "Randwick", "month": 4, "distance": 1200},
    {"race_name": "ATC Australian Derby", "venue": "Randwick", "month": 4, "distance": 2400},
    {"race_name": "ATC Oaks", "venue": "Randwick", "month": 4, "distance": 2400},
    {"race_name": "Queen Elizabeth Stakes", "venue": "Randwick", "month": 4, "distance": 2000},
    {"race_name": "Sydney Cup", "venue": "Randwick", "month": 4, "distance": 3200},
    {"race_name": "Queen of the Turf Stakes", "venue": "Randwick", "month": 4, "distance": 1600},
    {"race_name": "Champagne Stakes", "venue": "Randwick", "month": 4, "distance": 1600},
    {"race_name": "All Aged Stakes", "venue": "Randwick", "month": 4, "distance": 1400},

    # Spring Carnival (Aug-Oct)
    {"race_name": "Winx Stakes", "venue": "Randwick", "month": 8, "distance": 1400},
    {"race_name": "Chipping Norton Stakes", "venue": "Randwick", "month": 9, "distance": 1600},
    {"race_name": "George Main Stakes", "venue": "Randwick", "month": 9, "distance": 1600},
    {"race_name": "Flight Stakes", "venue": "Randwick", "month": 10, "distance": 1600},
    {"race_name": "Epsom Handicap", "venue": "Randwick", "month": 10, "distance": 1600},
    {"race_name": "The Metropolitan", "venue": "Randwick", "month": 10, "distance": 2400},
    {"race_name": "Spring Champion Stakes", "venue": "Randwick", "month": 10, "distance": 2000},
]


def get_hardcoded_races(year: int) -> list[dict]:
    """Get hardcoded Group 1 races as fallback."""
    races = []

    for race_template in MELBOURNE_GROUP1_RACES + SYDNEY_GROUP1_RACES:
        # Create a race for the specified year
        # Note: Actual dates vary year to year, this is approximate
        race_date = datetime(year, race_template["month"], 15)  # Mid-month approximation

        races.append({
            "race_name": race_template["race_name"],
            "venue": race_template["venue"],
            "race_date": race_date,
            "distance": race_template["distance"],
            "prize_money": None,
            "external_id": f"hc_{race_template['race_name'].lower().replace(' ', '_')}",
        })

    return races
