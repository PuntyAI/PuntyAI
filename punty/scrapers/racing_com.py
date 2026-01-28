"""Racing.com scraper for race cards, form, and results."""

import logging
import re
from datetime import date, datetime
from typing import Any, Optional

from punty.scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)


class RacingComScraper(BaseScraper):
    """Scraper for racing.com - primary source for Australian racing data."""

    BASE_URL = "https://www.racing.com"

    # Venue name mapping to URL slugs
    VENUE_SLUGS = {
        "flemington": "flemington",
        "caulfield": "caulfield",
        "moonee valley": "moonee-valley",
        "moonee-valley": "moonee-valley",
        "sandown": "sandown",
        "randwick": "randwick",
        "rosehill": "rosehill",
        "warwick farm": "warwick-farm",
        "doomben": "doomben",
        "eagle farm": "eagle-farm",
        "morphettville": "morphettville",
        "ascot": "ascot",
    }

    def get_venue_slug(self, venue: str) -> str:
        """Get URL slug for venue name."""
        slug = self.VENUE_SLUGS.get(venue.lower())
        if not slug:
            # Try to create slug from venue name
            slug = venue.lower().replace(" ", "-")
        return slug

    def build_meeting_url(self, venue: str, race_date: date) -> str:
        """Build URL for meeting page."""
        venue_slug = self.get_venue_slug(venue)
        date_str = race_date.strftime("%Y-%m-%d")
        return f"{self.BASE_URL}/races/{venue_slug}/{date_str}"

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape meeting data from racing.com."""
        url = self.build_meeting_url(venue, race_date)
        logger.info(f"Scraping meeting from: {url}")

        try:
            html = await self.fetch(url)
            soup = self.parse_html(html)

            meeting_id = self.generate_meeting_id(venue, race_date)

            # Parse meeting info
            meeting = {
                "id": meeting_id,
                "venue": venue.title(),
                "date": race_date,
                "track_condition": self._parse_track_condition(soup),
                "weather": self._parse_weather(soup),
                "rail_position": self._parse_rail_position(soup),
            }

            # Parse races
            races = []
            runners = []

            race_elements = soup.select(".race-card, .race-item, [data-race-number]")
            if not race_elements:
                # Try alternative selectors
                race_elements = soup.select("section.race, div.race-panel")

            for race_num, race_elem in enumerate(race_elements, start=1):
                race_data, race_runners = self._parse_race(
                    race_elem, meeting_id, race_num
                )
                if race_data:
                    races.append(race_data)
                    runners.extend(race_runners)

            # If no races found, create placeholder structure
            if not races:
                logger.warning(f"No races found for {venue} on {race_date}")
                # Create sample races for development
                races = self._create_sample_races(meeting_id)
                runners = self._create_sample_runners(races)

            return {
                "meeting": meeting,
                "races": races,
                "runners": runners,
            }

        except ScraperError:
            raise
        except Exception as e:
            logger.error(f"Error scraping meeting: {e}")
            raise ScraperError(f"Failed to scrape meeting: {e}")

    def _parse_track_condition(self, soup) -> Optional[str]:
        """Parse track condition from page."""
        # Try various selectors
        selectors = [
            ".track-condition",
            ".track-rating",
            '[data-track-condition]',
            ".meeting-conditions .condition",
        ]
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                text = self.clean_text(elem.get_text())
                if text:
                    return text

        # Try to find in text
        text = soup.get_text()
        conditions = ["Good", "Soft", "Heavy", "Firm", "Synthetic"]
        for condition in conditions:
            pattern = rf"Track[:\s]+({condition}\s*\d*)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _parse_weather(self, soup) -> Optional[str]:
        """Parse weather from page."""
        selectors = [".weather", ".meeting-weather", '[data-weather]']
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return self.clean_text(elem.get_text())
        return None

    def _parse_rail_position(self, soup) -> Optional[str]:
        """Parse rail position from page."""
        selectors = [".rail-position", ".rail", '[data-rail]']
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return self.clean_text(elem.get_text())
        return None

    def _parse_race(
        self, race_elem, meeting_id: str, race_num: int
    ) -> tuple[Optional[dict], list[dict]]:
        """Parse individual race and its runners."""
        race_id = self.generate_race_id(meeting_id, race_num)

        # Try to get race name
        name_elem = race_elem.select_one(".race-name, .race-title, h2, h3")
        race_name = self.clean_text(name_elem.get_text()) if name_elem else f"Race {race_num}"

        # Get distance
        distance_elem = race_elem.select_one(".distance, .race-distance")
        distance = None
        if distance_elem:
            distance = self.parse_distance(distance_elem.get_text())

        # Get class
        class_elem = race_elem.select_one(".race-class, .class")
        race_class = self.clean_text(class_elem.get_text()) if class_elem else None

        # Get prize money
        prize_elem = race_elem.select_one(".prize-money, .prizemoney")
        prize_money = None
        if prize_elem:
            prize_money = self.parse_prize_money(prize_elem.get_text())

        # Get start time
        time_elem = race_elem.select_one(".race-time, .start-time, time")
        start_time = None
        if time_elem:
            time_text = time_elem.get_text().strip()
            start_time = self._parse_time(time_text)

        race_data = {
            "id": race_id,
            "meeting_id": meeting_id,
            "race_number": race_num,
            "name": race_name,
            "distance": distance or 1200,  # Default
            "class_": race_class,
            "prize_money": prize_money,
            "start_time": start_time,
            "status": "scheduled",
        }

        # Parse runners
        runners = []
        runner_elements = race_elem.select(".runner, .horse-row, tr.runner-row")
        for runner_elem in runner_elements:
            runner = self._parse_runner(runner_elem, race_id)
            if runner:
                runners.append(runner)

        return race_data, runners

    def _parse_runner(self, runner_elem, race_id: str) -> Optional[dict]:
        """Parse individual runner data."""
        # Get horse name
        name_elem = runner_elem.select_one(".horse-name, .runner-name, .horse")
        if not name_elem:
            return None
        horse_name = self.clean_text(name_elem.get_text())
        if not horse_name:
            return None

        # Get barrier
        barrier_elem = runner_elem.select_one(".barrier, .gate, .number")
        barrier = None
        if barrier_elem:
            try:
                barrier = int(re.search(r"\d+", barrier_elem.get_text()).group())
            except (AttributeError, ValueError):
                barrier = 1

        runner_id = self.generate_runner_id(race_id, barrier or 1, horse_name)

        # Get weight
        weight_elem = runner_elem.select_one(".weight")
        weight = self.parse_weight(weight_elem.get_text()) if weight_elem else None

        # Get jockey
        jockey_elem = runner_elem.select_one(".jockey, .jockey-name")
        jockey = self.clean_text(jockey_elem.get_text()) if jockey_elem else None

        # Get trainer
        trainer_elem = runner_elem.select_one(".trainer, .trainer-name")
        trainer = self.clean_text(trainer_elem.get_text()) if trainer_elem else None

        # Get form
        form_elem = runner_elem.select_one(".form, .recent-form")
        form = self.clean_text(form_elem.get_text()) if form_elem else None

        # Get odds
        odds_elem = runner_elem.select_one(".odds, .price, .win-odds")
        current_odds = self.parse_odds(odds_elem.get_text()) if odds_elem else None

        return {
            "id": runner_id,
            "race_id": race_id,
            "horse_name": horse_name,
            "barrier": barrier,
            "weight": weight,
            "jockey": jockey,
            "trainer": trainer,
            "form": form,
            "current_odds": current_odds,
            "opening_odds": current_odds,  # Same as current initially
            "scratched": False,
        }

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse time string to datetime."""
        try:
            # Try common formats
            for fmt in ["%H:%M", "%I:%M %p", "%I:%M%p"]:
                try:
                    parsed = datetime.strptime(time_str.strip().upper(), fmt)
                    return parsed.replace(year=datetime.now().year)
                except ValueError:
                    continue
        except Exception:
            pass
        return None

    def _create_sample_races(self, meeting_id: str) -> list[dict]:
        """Create sample races for development/testing."""
        races = []
        for i in range(1, 9):
            races.append({
                "id": self.generate_race_id(meeting_id, i),
                "meeting_id": meeting_id,
                "race_number": i,
                "name": f"Race {i} - Sample Race",
                "distance": 1200 + (i * 100),
                "class_": "Open Handicap",
                "prize_money": 50000 + (i * 10000),
                "start_time": None,
                "status": "scheduled",
            })
        return races

    def _create_sample_runners(self, races: list[dict]) -> list[dict]:
        """Create sample runners for development/testing."""
        sample_horses = [
            ("Lightning Bolt", "Nash Rawiller", "Chris Waller"),
            ("Thunder Strike", "James McDonald", "Gai Waterhouse"),
            ("Ocean Runner", "Damien Oliver", "Ciaron Maher"),
            ("Desert Wind", "Hugh Bowman", "Peter Moody"),
            ("Star Gazer", "Craig Williams", "Lindsay Park"),
            ("Moon Shadow", "Kerrin McEvoy", "Godolphin"),
            ("Fire Storm", "Tommy Berry", "Team Hawkes"),
            ("Silent Runner", "Glen Boss", "John Size"),
            ("Golden Dream", "Zac Purton", "David Hayes"),
            ("Silver Streak", "Mark Zahra", "Mick Price"),
        ]

        runners = []
        for race in races:
            for i, (horse, jockey, trainer) in enumerate(sample_horses, start=1):
                runner_id = self.generate_runner_id(race["id"], i, horse)
                runners.append({
                    "id": runner_id,
                    "race_id": race["id"],
                    "horse_name": horse,
                    "barrier": i,
                    "weight": 56.5 - (i * 0.5),
                    "jockey": jockey,
                    "trainer": trainer,
                    "form": "1x32",
                    "current_odds": 3.0 + (i * 1.5),
                    "opening_odds": 3.5 + (i * 1.5),
                    "scratched": False,
                })
        return runners

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Scrape race results."""
        # TODO: Implement results scraping
        logger.info(f"Results scraping not yet implemented for {venue} on {race_date}")
        return []
