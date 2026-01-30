"""racenet.com.au scraper for form guides."""

import logging
import re
from datetime import date
from typing import Any, Optional

from bs4 import BeautifulSoup

from punty.scrapers.base import BaseScraper, ScraperError
from punty.scrapers.playwright_base import new_page, wait_and_get_content

logger = logging.getLogger(__name__)


class RacenetScraper(BaseScraper):
    """Scraper for racenet.com.au — form guide data."""

    BASE_URL = "https://www.racenet.com.au"

    def _build_url(self, venue: str, race_date: date) -> str:
        venue_slug = venue.lower().replace(" ", "-")
        date_str = race_date.strftime("%Y-%m-%d")
        return f"{self.BASE_URL}/form-guide/horse-racing/{venue_slug}-{date_str}/all-races"

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape form guide from racenet.com.au using Playwright."""
        url = self._build_url(venue, race_date)
        logger.info(f"Scraping Racenet form guide: {url}")

        try:
            async with new_page() as page:
                html = await wait_and_get_content(
                    page, url, wait_selector=".race-card, .form-guide, table, .runners"
                )

            soup = BeautifulSoup(html, "lxml")
            meeting_id = self.generate_meeting_id(venue, race_date)

            meeting = {
                "id": meeting_id,
                "venue": venue.title(),
                "date": race_date,
                "track_condition": self._find_text(soup, "track"),
                "weather": self._find_text(soup, "weather"),
                "rail_position": self._find_text(soup, "rail"),
            }

            races = []
            runners = []

            race_sections = soup.select(
                ".race-card, .race-section, [data-race], section.race"
            )

            for idx, section in enumerate(race_sections, start=1):
                race_num = self._race_num(section) or idx
                race_id = self.generate_race_id(meeting_id, race_num)

                name_tag = section.select_one(".race-name, .race-title, h2, h3")
                name = self.clean_text(name_tag.get_text()) if name_tag else f"Race {race_num}"

                dist_tag = section.select_one(".distance, .race-distance")
                distance = self.parse_distance(dist_tag.get_text()) if dist_tag else None

                class_tag = section.select_one(".race-class, .class")
                prize_tag = section.select_one(".prize-money, .prizemoney")

                races.append({
                    "id": race_id,
                    "meeting_id": meeting_id,
                    "race_number": race_num,
                    "name": name,
                    "distance": distance or 1200,
                    "class_": self.clean_text(class_tag.get_text()) if class_tag else None,
                    "prize_money": self.parse_prize_money(prize_tag.get_text()) if prize_tag else None,
                    "start_time": None,
                    "status": "scheduled",
                })

                for row in section.select(".runner, .runner-row, tr.runner, .horse-row"):
                    runner = self._parse_runner(row, race_id)
                    if runner:
                        runners.append(runner)

            return {"meeting": meeting, "races": races, "runners": runners}

        except ScraperError:
            raise
        except Exception as e:
            logger.error(f"Error scraping Racenet: {e}")
            raise ScraperError(f"Failed to scrape Racenet: {e}")

    def _parse_runner(self, row, race_id: str) -> Optional[dict]:
        name_tag = row.select_one(".horse-name, .runner-name, .horse")
        if not name_tag:
            return None
        horse_name = self.clean_text(name_tag.get_text())
        if not horse_name:
            return None

        barrier = None
        b_tag = row.select_one(".barrier, .gate, .number")
        if b_tag:
            m = re.search(r"\d+", b_tag.get_text())
            if m:
                barrier = int(m.group())

        runner_id = self.generate_runner_id(race_id, barrier or 1, horse_name)

        jockey_tag = row.select_one(".jockey, .jockey-name")
        trainer_tag = row.select_one(".trainer, .trainer-name")
        form_tag = row.select_one(".form, .recent-form")
        weight_tag = row.select_one(".weight")
        odds_tag = row.select_one(".odds, .price")

        return {
            "id": runner_id,
            "race_id": race_id,
            "horse_name": horse_name,
            "barrier": barrier,
            "weight": self.parse_weight(weight_tag.get_text()) if weight_tag else None,
            "jockey": self.clean_text(jockey_tag.get_text()) if jockey_tag else None,
            "trainer": self.clean_text(trainer_tag.get_text()) if trainer_tag else None,
            "form": self.clean_text(form_tag.get_text()) if form_tag else None,
            "current_odds": self.parse_odds(odds_tag.get_text()) if odds_tag else None,
            "opening_odds": None,
            "scratched": "scratched" in " ".join(row.get("class", [])).lower(),
        }

    def _find_text(self, soup, keyword: str) -> Optional[str]:
        for sel in [f".{keyword}", f".meeting-{keyword}", f"[data-{keyword}]"]:
            tag = soup.select_one(sel)
            if tag:
                return self.clean_text(tag.get_text())
        return None

    def _race_num(self, section) -> Optional[int]:
        val = section.get("data-race") or section.get("data-race-number")
        if val:
            try:
                return int(val)
            except ValueError:
                pass
        heading = section.select_one("h2, h3, h4, .race-title")
        if heading:
            m = re.search(r"Race\s*(\d+)", heading.get_text(), re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        return []
