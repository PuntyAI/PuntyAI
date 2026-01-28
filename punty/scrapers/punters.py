"""Punters.com.au scraper for speed maps and tips."""

import logging
import re
from datetime import date
from typing import Any, Optional

from punty.scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)


class PuntersScraper(BaseScraper):
    """Scraper for Punters.com.au - speed maps and tips consensus."""

    BASE_URL = "https://www.punters.com.au"

    # Speed map position mapping
    POSITION_MAPPING = {
        "lead": "leader",
        "leader": "leader",
        "on pace": "on_pace",
        "on the pace": "on_pace",
        "stalking": "on_pace",
        "midfield": "midfield",
        "mid-field": "midfield",
        "back": "backmarker",
        "backmarker": "backmarker",
        "back marker": "backmarker",
        "settle back": "backmarker",
    }

    def normalize_position(self, position: Optional[str]) -> Optional[str]:
        """Normalize speed map position to standard values."""
        if not position:
            return None
        position_lower = position.lower().strip()
        return self.POSITION_MAPPING.get(position_lower, position_lower)

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape speed maps and tips from punters.com.au."""
        logger.info(f"Scraping Punters.com.au for {venue} on {race_date}")

        try:
            # Build URL
            venue_slug = venue.lower().replace(" ", "-")
            date_str = race_date.strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/form-guide/{venue_slug}/{date_str}"

            html = await self.fetch(url)
            soup = self.parse_html(html)

            # Parse speed maps
            speed_maps = self._parse_speed_maps(soup)

            # Parse tips consensus
            tips = self._parse_tips(soup)

            # Parse comments/form analysis
            comments = self._parse_comments(soup)

            return {
                "speed_maps": speed_maps,
                "tips": tips,
                "comments": comments,
            }

        except ScraperError:
            raise
        except Exception as e:
            logger.error(f"Error scraping Punters: {e}")
            raise ScraperError(f"Failed to scrape Punters: {e}")

    def _parse_speed_maps(self, soup) -> dict[int, list[dict]]:
        """Parse speed maps for all races.

        Returns dict mapping race_number to list of runner positions.
        """
        speed_maps = {}

        # Try to find speed map sections
        speed_map_sections = soup.select(".speed-map, .speedmap, [data-speedmap]")

        if not speed_map_sections:
            # Try alternative approach - look for pace indicators
            speed_map_sections = soup.select(".pace-indicators, .settling-positions")

        for section in speed_map_sections:
            # Get race number from parent or data attribute
            race_num = self._get_race_number(section)
            if not race_num:
                continue

            positions = []

            # Parse each runner's position
            runner_positions = section.select(".runner-position, .horse-pace, tr")
            for pos_elem in runner_positions:
                horse_elem = pos_elem.select_one(".horse-name, .runner-name")
                position_elem = pos_elem.select_one(".position, .pace, .settling")

                if not horse_elem:
                    continue

                horse_name = self.clean_text(horse_elem.get_text())
                raw_position = self.clean_text(position_elem.get_text()) if position_elem else None
                position = self.normalize_position(raw_position)

                positions.append({
                    "horse_name": horse_name,
                    "speed_map_position": position,
                    "raw_position": raw_position,
                })

            if positions:
                speed_maps[race_num] = positions

        return speed_maps

    def _parse_tips(self, soup) -> dict[int, list[dict]]:
        """Parse tips consensus for all races.

        Returns dict mapping race_number to list of tipped horses.
        """
        tips = {}

        tips_sections = soup.select(".tips-consensus, .tips, [data-tips]")

        for section in tips_sections:
            race_num = self._get_race_number(section)
            if not race_num:
                continue

            race_tips = []

            tip_rows = section.select(".tip-row, .tipped-horse, tr")
            for row in tip_rows:
                horse_elem = row.select_one(".horse-name, .runner-name")
                tips_count_elem = row.select_one(".tips-count, .tip-count")
                rank_elem = row.select_one(".tip-rank, .rank")

                if not horse_elem:
                    continue

                horse_name = self.clean_text(horse_elem.get_text())

                tips_count = 0
                if tips_count_elem:
                    try:
                        tips_count = int(re.search(r"\d+", tips_count_elem.get_text()).group())
                    except (AttributeError, ValueError):
                        pass

                rank = None
                if rank_elem:
                    try:
                        rank = int(re.search(r"\d+", rank_elem.get_text()).group())
                    except (AttributeError, ValueError):
                        pass

                race_tips.append({
                    "horse_name": horse_name,
                    "tips_count": tips_count,
                    "rank": rank,
                })

            if race_tips:
                tips[race_num] = sorted(race_tips, key=lambda x: x.get("tips_count", 0), reverse=True)

        return tips

    def _parse_comments(self, soup) -> dict[int, dict[str, str]]:
        """Parse form comments for runners.

        Returns dict mapping race_number to dict of horse_name -> comment.
        """
        comments = {}

        comment_sections = soup.select(".form-comments, .runner-comments, .analysis")

        for section in comment_sections:
            race_num = self._get_race_number(section)
            if not race_num:
                continue

            race_comments = {}

            comment_rows = section.select(".comment-row, .runner-comment, .horse-analysis")
            for row in comment_rows:
                horse_elem = row.select_one(".horse-name, .runner-name")
                comment_elem = row.select_one(".comment, .analysis-text, p")

                if not (horse_elem and comment_elem):
                    continue

                horse_name = self.clean_text(horse_elem.get_text())
                comment = self.clean_text(comment_elem.get_text())

                if horse_name and comment:
                    race_comments[horse_name] = comment

            if race_comments:
                comments[race_num] = race_comments

        return comments

    def _get_race_number(self, element) -> Optional[int]:
        """Extract race number from element."""
        # Try data attribute
        race_num = element.get("data-race") or element.get("data-race-number")
        if race_num:
            try:
                return int(race_num)
            except ValueError:
                pass

        # Try to find in parent
        parent = element.find_parent(attrs={"data-race": True})
        if parent:
            try:
                return int(parent["data-race"])
            except (KeyError, ValueError):
                pass

        # Try to find race number in text
        heading = element.find_previous(["h2", "h3", "h4"])
        if heading:
            match = re.search(r"Race\s*(\d+)", heading.get_text(), re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    async def scrape_speed_map(self, venue: str, race_date: date, race_number: int) -> list[dict]:
        """Scrape speed map for a specific race."""
        logger.info(f"Scraping speed map for R{race_number} at {venue}")

        try:
            venue_slug = venue.lower().replace(" ", "-")
            date_str = race_date.strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/form-guide/{venue_slug}/{date_str}/race-{race_number}/speed-map"

            html = await self.fetch(url)
            soup = self.parse_html(html)

            positions = []

            # Try various selectors for speed map visualization
            map_container = soup.select_one(".speed-map-visual, .speedmap-container, .pace-map")

            if map_container:
                # Parse the visual speed map
                runners = map_container.select(".runner, .horse, [data-horse]")

                for runner in runners:
                    horse_name = self.clean_text(runner.get_text())
                    if not horse_name:
                        horse_elem = runner.select_one(".horse-name")
                        horse_name = self.clean_text(horse_elem.get_text()) if horse_elem else None

                    # Position might be indicated by CSS class or position
                    position = None
                    for cls in runner.get("class", []):
                        cls_lower = cls.lower()
                        if any(pos in cls_lower for pos in ["lead", "pace", "mid", "back"]):
                            position = self.normalize_position(cls)
                            break

                    # Or from data attribute
                    if not position:
                        position = self.normalize_position(runner.get("data-position"))

                    if horse_name:
                        positions.append({
                            "horse_name": horse_name,
                            "speed_map_position": position,
                        })

            # Fallback to table format
            if not positions:
                table = soup.select_one(".speed-map-table, table")
                if table:
                    for row in table.select("tr"):
                        cells = row.select("td")
                        if len(cells) >= 2:
                            horse_name = self.clean_text(cells[0].get_text())
                            position = self.normalize_position(cells[1].get_text())
                            if horse_name:
                                positions.append({
                                    "horse_name": horse_name,
                                    "speed_map_position": position,
                                })

            return positions

        except Exception as e:
            logger.error(f"Error scraping speed map: {e}")
            return []

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Scrape results from Punters.com.au (not typically used for results)."""
        return []
