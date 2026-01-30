"""racingandsports.com.au scraper for form guide, stats, speed maps, tips."""

import logging
import re
from datetime import date
from typing import Any, Optional

from bs4 import BeautifulSoup

from punty.scrapers.base import BaseScraper, ScraperError
from punty.scrapers.playwright_base import new_page, wait_and_get_content

logger = logging.getLogger(__name__)


class RacingSportsScraper(BaseScraper):
    """Scraper for racingandsports.com.au — form guide, stats, speed maps, tips."""

    BASE_URL = "https://www.racingandsports.com.au"

    def _build_form_url(self, venue: str, race_date: date) -> str:
        venue_slug = venue.lower().replace(" ", "-")
        date_str = race_date.strftime("%Y-%m-%d")
        return f"{self.BASE_URL}/form-guide/thoroughbred/australia/{venue_slug}/{date_str}"

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape full form guide for a meeting using Playwright."""
        url = self._build_form_url(venue, race_date)
        logger.info(f"Scraping Racing & Sports form guide: {url}")

        try:
            async with new_page() as page:
                html = await wait_and_get_content(
                    page, url, wait_selector=".form-guide, .race-card, table, .runners"
                )

            soup = BeautifulSoup(html, "lxml")
            meeting_id = self.generate_meeting_id(venue, race_date)

            meeting = {
                "id": meeting_id,
                "venue": venue.title(),
                "date": race_date,
                "track_condition": self._parse_field(soup, "track"),
                "weather": self._parse_field(soup, "weather"),
                "rail_position": self._parse_field(soup, "rail"),
            }

            races = []
            runners = []

            race_sections = soup.select(
                ".race-card, .race-section, [data-race-number], section.race"
            )

            for idx, section in enumerate(race_sections, start=1):
                race_num = self._get_race_num(section) or idx
                race_id = self.generate_race_id(meeting_id, race_num)

                race_data = self._parse_race_header(section, race_id, meeting_id, race_num)
                races.append(race_data)

                for runner_data in self._parse_runners(section, race_id):
                    runners.append(runner_data)

            return {"meeting": meeting, "races": races, "runners": runners}

        except ScraperError:
            raise
        except Exception as e:
            logger.error(f"Error scraping Racing & Sports: {e}")
            raise ScraperError(f"Failed to scrape Racing & Sports: {e}")

    def _parse_field(self, soup, keyword: str) -> Optional[str]:
        """Generic field parser — looks for keyword in common selectors."""
        for sel in [f".{keyword}", f"[data-{keyword}]", f".meeting-{keyword}"]:
            tag = soup.select_one(sel)
            if tag:
                return self.clean_text(tag.get_text())
        text = soup.get_text()
        pattern = rf"(?i){keyword}[:\s]+([^\n,]+)"
        m = re.search(pattern, text)
        return m.group(1).strip() if m else None

    def _get_race_num(self, section) -> Optional[int]:
        num = section.get("data-race-number") or section.get("data-race")
        if num:
            try:
                return int(num)
            except ValueError:
                pass
        heading = section.select_one("h2, h3, h4, .race-title, .race-name")
        if heading:
            m = re.search(r"Race\s*(\d+)", heading.get_text(), re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _parse_race_header(self, section, race_id: str, meeting_id: str, race_num: int) -> dict:
        name_tag = section.select_one(".race-name, .race-title, h2, h3")
        name = self.clean_text(name_tag.get_text()) if name_tag else f"Race {race_num}"

        dist_tag = section.select_one(".distance, .race-distance")
        distance = self.parse_distance(dist_tag.get_text()) if dist_tag else None

        class_tag = section.select_one(".race-class, .class")
        class_ = self.clean_text(class_tag.get_text()) if class_tag else None

        prize_tag = section.select_one(".prize-money, .prizemoney")
        prize = self.parse_prize_money(prize_tag.get_text()) if prize_tag else None

        return {
            "id": race_id,
            "meeting_id": meeting_id,
            "race_number": race_num,
            "name": name,
            "distance": distance or 1200,
            "class_": class_,
            "prize_money": prize,
            "start_time": None,
            "status": "scheduled",
        }

    def _parse_runners(self, section, race_id: str) -> list[dict]:
        runners = []
        rows = section.select(".runner, .runner-row, tr.runner, .horse-row")
        for row in rows:
            name_tag = row.select_one(".horse-name, .runner-name, .horse")
            if not name_tag:
                continue
            horse_name = self.clean_text(name_tag.get_text())
            if not horse_name:
                continue

            barrier = self._int_from(row, ".barrier, .gate, .number")
            runner_id = self.generate_runner_id(race_id, barrier or 1, horse_name)

            jockey_tag = row.select_one(".jockey, .jockey-name")
            trainer_tag = row.select_one(".trainer, .trainer-name")
            form_tag = row.select_one(".form, .recent-form")
            weight_tag = row.select_one(".weight")
            odds_tag = row.select_one(".odds, .price")
            comment_tag = row.select_one(".comment, .comments, .analysis")

            runners.append({
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
                "comments": self.clean_text(comment_tag.get_text()) if comment_tag else None,
            })
        return runners

    def _int_from(self, elem, selectors: str) -> Optional[int]:
        for sel in selectors.split(","):
            tag = elem.select_one(sel.strip())
            if tag:
                m = re.search(r"\d+", tag.get_text())
                if m:
                    return int(m.group())
        return None

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Results scraping — not primary source, returns empty."""
        return []
