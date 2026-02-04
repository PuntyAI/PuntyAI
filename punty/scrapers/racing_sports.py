"""racingandsports.com.au scraper for form guide, stats, speed maps, tips."""

import logging
import re
from datetime import date
from typing import Any, AsyncGenerator, Optional

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

    def _build_speed_map_url(self, venue: str, race_date: date, race_num: int) -> str:
        """Build URL for a specific race's speed map page."""
        venue_slug = venue.lower().replace(" ", "-")
        date_str = race_date.strftime("%Y-%m-%d")
        return f"{self.BASE_URL}/form-guide/thoroughbred/australia/{venue_slug}/{date_str}/R{race_num}/speedmap"

    # Position keywords to look for in page text/classes
    POSITION_KEYWORDS = {
        "leader": "leader",
        "leaders": "leader",
        "lead": "leader",
        "on pace": "on_pace",
        "on the pace": "on_pace",
        "on-pace": "on_pace",
        "pace": "on_pace",
        "stalker": "on_pace",
        "stalking": "on_pace",
        "prominent": "on_pace",
        "midfield": "midfield",
        "mid-field": "midfield",
        "middle": "midfield",
        "off pace": "backmarker",
        "off-pace": "backmarker",
        "back": "backmarker",
        "backmarker": "backmarker",
        "backmarkers": "backmarker",
        "rear": "backmarker",
        "tail": "backmarker",
    }

    def _normalize_position(self, text: Optional[str]) -> Optional[str]:
        """Normalize position text to standard values."""
        if not text:
            return None
        text_lower = text.lower().strip()
        # Direct match
        if text_lower in self.POSITION_KEYWORDS:
            return self.POSITION_KEYWORDS[text_lower]
        # Partial match
        for keyword, position in self.POSITION_KEYWORDS.items():
            if keyword in text_lower:
                return position
        return None

    async def scrape_speed_maps(
        self, venue: str, race_date: date, race_count: int
    ) -> AsyncGenerator[dict, None]:
        """Scrape speed maps for all races, yielding progress events.

        Racing and Sports provides speed maps at URLs like:
        /form-guide/thoroughbred/australia/{venue}/{date}/R{num}/speedmap

        The page contains a visual speed map with horses grouped by expected
        settling position (Leaders, On Pace, Midfield, Backmarkers).
        """
        total = race_count + 1  # +1 for completion event
        venue_slug = venue.lower().replace(" ", "-")

        async with new_page() as page:
            for race_num in range(1, race_count + 1):
                yield {
                    "step": race_num - 1,
                    "total": total,
                    "label": f"Fetching speed map for Race {race_num}...",
                    "status": "running",
                }

                speed_map_url = self._build_speed_map_url(venue, race_date, race_num)
                positions = []

                try:
                    # Navigate to speed map page
                    await page.goto(speed_map_url, wait_until="domcontentloaded")
                    await page.wait_for_timeout(2000)  # Allow JS to render

                    # Get page content
                    html = await page.content()
                    soup = BeautifulSoup(html, "lxml")

                    # Strategy 1: Look for speed map sections with position headers
                    # Racing and Sports typically groups horses under position headings
                    for section_class in [".speedmap", ".speed-map", ".pace-map",
                                          ".settling-positions", "[data-speedmap]"]:
                        container = soup.select_one(section_class)
                        if container:
                            positions = self._parse_speed_map_container(container)
                            if positions:
                                break

                    # Strategy 2: Look for tables with position columns
                    if not positions:
                        tables = soup.select("table")
                        for table in tables:
                            positions = self._parse_speed_map_table(table)
                            if positions:
                                break

                    # Strategy 3: Look for grouped runners by position keywords
                    if not positions:
                        positions = self._parse_speed_map_from_groups(soup)

                    # Strategy 4: Look for any runner elements with position data
                    if not positions:
                        positions = self._parse_speed_map_from_runners(soup)

                    count = len(positions)
                    if count > 0:
                        yield {
                            "step": race_num,
                            "total": total,
                            "label": f"Race {race_num}: {count} positions found",
                            "status": "done",
                            "race_number": race_num,
                            "positions": positions,
                        }
                    else:
                        yield {
                            "step": race_num,
                            "total": total,
                            "label": f"Race {race_num}: no speed map data found",
                            "status": "done",
                            "race_number": race_num,
                            "positions": [],
                        }

                except Exception as e:
                    logger.error(f"Error scraping speed map for race {race_num}: {e}")
                    yield {
                        "step": race_num,
                        "total": total,
                        "label": f"Race {race_num}: error - {e}",
                        "status": "error",
                        "race_number": race_num,
                        "positions": [],
                    }

        yield {
            "step": total,
            "total": total,
            "label": "Speed maps complete",
            "status": "complete",
        }

    def _parse_speed_map_container(self, container) -> list[dict]:
        """Parse a speed map container element."""
        positions = []

        # Look for position group sections
        for pos_name, norm_pos in [
            ("leader", "leader"),
            ("on pace", "on_pace"),
            ("on-pace", "on_pace"),
            ("midfield", "midfield"),
            ("backmarker", "backmarker"),
            ("back", "backmarker"),
        ]:
            # Find sections containing position keywords
            for elem in container.find_all(string=re.compile(pos_name, re.IGNORECASE)):
                parent = elem.find_parent(["div", "section", "li", "td", "tr"])
                if parent:
                    # Find horse names near this element
                    horses = parent.select(".horse-name, .runner-name, .horse, a[href*='horse']")
                    for horse in horses:
                        horse_name = self.clean_text(horse.get_text())
                        if horse_name and len(horse_name) > 1:
                            positions.append({
                                "horse_name": horse_name,
                                "position": norm_pos,
                            })

        return positions

    def _parse_speed_map_table(self, table) -> list[dict]:
        """Parse speed map from a table format."""
        positions = []
        headers = [th.get_text().lower().strip() for th in table.select("th")]

        # Find position column index
        pos_col = None
        name_col = None
        for i, h in enumerate(headers):
            if any(kw in h for kw in ["position", "settling", "pace", "map"]):
                pos_col = i
            if any(kw in h for kw in ["horse", "runner", "name"]):
                name_col = i

        if pos_col is None or name_col is None:
            return []

        for row in table.select("tr"):
            cells = row.select("td")
            if len(cells) > max(pos_col, name_col):
                horse_name = self.clean_text(cells[name_col].get_text())
                pos_text = cells[pos_col].get_text()
                position = self._normalize_position(pos_text)

                if horse_name and position:
                    positions.append({
                        "horse_name": horse_name,
                        "position": position,
                    })

        return positions

    def _parse_speed_map_from_groups(self, soup) -> list[dict]:
        """Parse speed map by finding position group headings."""
        positions = []

        # Common heading patterns for position groups
        position_headings = [
            (r"leader", "leader"),
            (r"on[\s-]?pace", "on_pace"),
            (r"stalker", "on_pace"),
            (r"midfield", "midfield"),
            (r"mid[\s-]?field", "midfield"),
            (r"backmarker", "backmarker"),
            (r"back[\s-]?marker", "backmarker"),
        ]

        for pattern, norm_pos in position_headings:
            # Find headings matching the pattern
            for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"]):
                if re.search(pattern, heading.get_text(), re.IGNORECASE):
                    # Get the next sibling elements until another heading
                    sibling = heading.find_next_sibling()
                    while sibling and sibling.name not in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        # Look for horse names in this section
                        text = self.clean_text(sibling.get_text())
                        if text and len(text) > 2 and len(text) < 50:
                            # Could be a horse name
                            # Check it's not another position keyword
                            if not self._normalize_position(text):
                                positions.append({
                                    "horse_name": text,
                                    "position": norm_pos,
                                })
                        sibling = sibling.find_next_sibling()

        return positions

    def _parse_speed_map_from_runners(self, soup) -> list[dict]:
        """Parse speed map from individual runner elements with position data."""
        positions = []

        # Look for runner elements with position indicators
        runner_selectors = [
            ".runner", ".horse", ".runner-row", ".horse-row",
            "[data-runner]", "[data-horse]", "tr"
        ]

        for selector in runner_selectors:
            for elem in soup.select(selector):
                # Get horse name
                name_elem = elem.select_one(".horse-name, .runner-name, .name, a")
                if not name_elem:
                    continue
                horse_name = self.clean_text(name_elem.get_text())
                if not horse_name or len(horse_name) < 2:
                    continue

                # Look for position in various places
                position = None

                # Check data attributes
                for attr in ["data-position", "data-pace", "data-settling"]:
                    if elem.get(attr):
                        position = self._normalize_position(elem.get(attr))
                        if position:
                            break

                # Check class names
                if not position:
                    classes = " ".join(elem.get("class", []))
                    position = self._normalize_position(classes)

                # Check child elements
                if not position:
                    pos_elem = elem.select_one(".position, .pace, .settling, .map-position")
                    if pos_elem:
                        position = self._normalize_position(pos_elem.get_text())

                if horse_name and position:
                    positions.append({
                        "horse_name": horse_name,
                        "position": position,
                    })

        return positions
