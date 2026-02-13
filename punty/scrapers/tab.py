"""TAB.com.au scraper for odds and market data."""

import logging
import re
from datetime import date
from typing import Any, Optional

from punty.scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)


class TabScraper(BaseScraper):
    """Scraper for TAB.com.au - odds and market data."""

    BASE_URL = "https://www.tab.com.au"
    API_URL = "https://api.tab.com.au/v1"

    SPONSOR_PREFIXES = [
        "ladbrokes", "tab", "bet365", "sportsbet", "neds", "pointsbet",
        "unibet", "betfair", "palmerbet", "bluebet", "topsport", "aquis",
        "picklebet",
    ]

    VENUE_ALIASES = {
        "park kilmore": "kilmore",
        "park-kilmore": "kilmore",
        "sandown lakeside": "sandown",
        "sandown-lakeside": "sandown",
        "thomas farms rc murray bridge": "murray-bridge",
        "thomas-farms-rc-murray-bridge": "murray-bridge",
    }

    def _venue_slug(self, venue: str) -> str:
        """Convert venue name to TAB URL slug, stripping sponsor prefixes."""
        slug = venue.lower().strip()
        for prefix in self.SPONSOR_PREFIXES:
            if slug.startswith(prefix + " "):
                slug = slug[len(prefix) + 1:]
                break
            if slug.startswith(prefix + "-"):
                slug = slug[len(prefix) + 1:]
                break
        if slug in self.VENUE_ALIASES:
            return self.VENUE_ALIASES[slug]
        slug_dashed = slug.replace(" ", "-")
        if slug_dashed in self.VENUE_ALIASES:
            return self.VENUE_ALIASES[slug_dashed]
        return slug_dashed

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape odds data from TAB for a meeting.

        Note: This supplements data from racing.com with live odds.
        """
        logger.info(f"Scraping TAB odds for {venue} on {race_date}")

        try:
            # Build API URL
            date_str = race_date.strftime("%Y-%m-%d")
            api_url = f"{self.API_URL}/racing/{date_str}/meetings"

            # Try API first (may not be publicly accessible)
            try:
                html = await self.fetch(api_url)
                return self._parse_api_response(html, venue, race_date)
            except ScraperError:
                logger.debug("API not accessible, trying website")

            # Fall back to website scraping
            venue_slug = self._venue_slug(venue)
            web_url = f"{self.BASE_URL}/racing/meetings/{venue_slug}/{date_str}"

            html = await self.fetch(web_url)
            return self._parse_web_page(html, venue, race_date)

        except ScraperError:
            raise
        except Exception as e:
            logger.error(f"Error scraping TAB: {e}")
            raise ScraperError(f"Failed to scrape TAB: {e}")

    def _parse_api_response(self, response: str, venue: str, race_date: date) -> dict[str, Any]:
        """Parse TAB API response."""
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            raise ScraperError("Invalid API response")

        # Extract relevant meeting — match on raw name or cleaned slug
        meeting = None
        venue_lower = venue.lower()
        venue_slug = self._venue_slug(venue)
        for m in data.get("meetings", []):
            api_venue = m.get("venueName", "").lower()
            if api_venue == venue_lower or api_venue == venue_slug:
                meeting = m
                break
            # Partial match: "Kilmore" in "bet365 Park Kilmore"
            if api_venue in venue_lower or venue_slug in api_venue:
                meeting = m
                break

        if not meeting:
            raise ScraperError(f"Meeting not found: {venue}")

        # Map to our format - only odds data
        runners_odds = []
        for race in meeting.get("races", []):
            for runner in race.get("runners", []):
                fixed_odds = runner.get("fixedOdds", {})
                runners_odds.append({
                    "race_number": race.get("raceNumber"),
                    "horse_name": runner.get("runnerName"),
                    "barrier": runner.get("barrierNumber"),
                    "current_odds": fixed_odds.get("returnWin"),
                    "opening_odds": fixed_odds.get("openingPrice"),
                    "place_odds": fixed_odds.get("returnPlace"),
                    "scratched": runner.get("scratched", False),
                    "scratching_reason": runner.get("scratchingReason"),
                })

        return {
            "meeting": {
                "venue": venue,
                "date": race_date,
                "track_condition": meeting.get("trackCondition"),
                "weather": meeting.get("weather"),
            },
            "runners_odds": runners_odds,
        }

    def _parse_web_page(self, html: str, venue: str, race_date: date) -> dict[str, Any]:
        """Parse TAB website HTML."""
        soup = self.parse_html(html)

        # Track condition
        track_elem = soup.select_one(".track-condition, .meeting-track")
        track_condition = self.clean_text(track_elem.get_text()) if track_elem else None

        # Weather
        weather_elem = soup.select_one(".weather, .meeting-weather")
        weather = self.clean_text(weather_elem.get_text()) if weather_elem else None

        # Parse odds for each runner
        runners_odds = []
        race_sections = soup.select(".race-card, .race-container, [data-race]")

        for race_num, race_section in enumerate(race_sections, start=1):
            runner_rows = race_section.select(".runner-row, .runner, tr")

            for row in runner_rows:
                horse_name_elem = row.select_one(".runner-name, .horse-name")
                if not horse_name_elem:
                    continue

                horse_name = self.clean_text(horse_name_elem.get_text())

                # Get barrier
                barrier_elem = row.select_one(".barrier, .saddle")
                barrier = None
                if barrier_elem:
                    try:
                        barrier = int(re.search(r"\d+", barrier_elem.get_text()).group())
                    except (AttributeError, ValueError):
                        pass

                # Get odds
                win_odds_elem = row.select_one(".win-odds, .odds-win, .price")
                current_odds = self.parse_odds(win_odds_elem.get_text()) if win_odds_elem else None

                # Check if scratched
                scratched = "scratched" in row.get("class", []) or row.select_one(".scratched")

                runners_odds.append({
                    "race_number": race_num,
                    "horse_name": horse_name,
                    "barrier": barrier,
                    "current_odds": current_odds,
                    "opening_odds": current_odds,  # Opening odds usually require historical data
                    "scratched": bool(scratched),
                })

        return {
            "meeting": {
                "venue": venue,
                "date": race_date,
                "track_condition": track_condition,
                "weather": weather,
            },
            "runners_odds": runners_odds,
        }

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Scrape race results with dividends from TAB."""
        logger.info(f"Scraping TAB results for {venue} on {race_date}")

        results = []

        try:
            venue_slug = self._venue_slug(venue)
            date_str = race_date.strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/racing/results/{venue_slug}/{date_str}"

            html = await self.fetch(url)
            soup = self.parse_html(html)

            result_sections = soup.select(".race-result, .result-card, [data-result]")

            for race_num, section in enumerate(result_sections, start=1):
                # Parse placings
                placings = section.select(".placing, .result-row")

                for placing in placings[:4]:  # Top 4 placings
                    position_elem = placing.select_one(".position, .placing-number")
                    horse_elem = placing.select_one(".horse-name, .runner-name")
                    dividend_elem = placing.select_one(".dividend, .payout")

                    if not (position_elem and horse_elem):
                        continue

                    position = None
                    try:
                        position = int(re.search(r"\d+", position_elem.get_text()).group())
                    except (AttributeError, ValueError):
                        continue

                    results.append({
                        "race_number": race_num,
                        "position": position,
                        "horse_name": self.clean_text(horse_elem.get_text()),
                        "dividend_win": self.parse_odds(dividend_elem.get_text()) if dividend_elem and position == 1 else None,
                        "dividend_place": self.parse_odds(dividend_elem.get_text()) if dividend_elem and position <= 3 else None,
                    })

        except ScraperError:
            raise
        except Exception as e:
            logger.error(f"Error scraping TAB results: {e}")

        return results

    async def scrape_live_odds(self, venue: str, race_date: date, race_number: int) -> list[dict]:
        """Scrape live odds for a specific race."""
        logger.info(f"Scraping live odds for R{race_number} at {venue}")

        try:
            venue_slug = self._venue_slug(venue)
            date_str = race_date.strftime("%Y-%m-%d")
            url = f"{self.BASE_URL}/racing/{venue_slug}/{date_str}/race-{race_number}"

            html = await self.fetch(url)
            soup = self.parse_html(html)

            odds_data = []
            runner_rows = soup.select(".runner-row, .runner")

            for row in runner_rows:
                horse_elem = row.select_one(".runner-name, .horse-name")
                if not horse_elem:
                    continue

                odds_elem = row.select_one(".win-odds, .price")
                flucs_elem = row.select_one(".flucs, .fluctuations")

                horse_name = self.clean_text(horse_elem.get_text())
                current_odds = self.parse_odds(odds_elem.get_text()) if odds_elem else None

                # Parse fluctuations if available
                opening_odds = None
                if flucs_elem:
                    flucs_text = flucs_elem.get_text()
                    # Often formatted as "3.50 > 2.80" or similar
                    match = re.search(r"([\d.]+)\s*[>→]\s*([\d.]+)", flucs_text)
                    if match:
                        opening_odds = float(match.group(1))

                odds_data.append({
                    "horse_name": horse_name,
                    "current_odds": current_odds,
                    "opening_odds": opening_odds or current_odds,
                })

            return odds_data

        except Exception as e:
            logger.error(f"Error scraping live odds: {e}")
            return []
