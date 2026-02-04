"""Punting Form scraper for speed maps and AI insights.

Requires credentials stored in app_settings:
- punting_form_email
- punting_form_password
- punting_form_totp_secret
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import pyotp
from bs4 import BeautifulSoup

from punty.scrapers.base import BaseScraper, ScraperError
from punty.scrapers.playwright_base import new_page

logger = logging.getLogger(__name__)

# Cookie cache path
COOKIE_CACHE_PATH = Path("/tmp/punting_form_cookies.json")


class PuntingFormScraper(BaseScraper):
    """Scraper for Punting Form - speed maps, AI insights, and ratings."""

    BASE_URL = "https://www.puntingform.com.au"

    # Venue name mapping: racing.com name -> Punting Form slug
    VENUE_MAP = {
        "sportsbet sandown lakeside": "sandown_hillside",
        "sandown lakeside": "sandown_hillside",
        "sandown": "sandown_hillside",
        "thomas farms rc murray bridge": "murray-bridge-gh",
        "murray bridge": "murray-bridge-gh",
        "ladbrokes geelong": "geelong",
        "bet365 geelong": "geelong",
    }

    # Position column mapping to standard values
    POSITION_MAP = {
        "leader": "leader",
        "onpace": "on_pace",
        "on pace": "on_pace",
        "onpace-midfield": "on_pace",
        "midfield": "midfield",
        "midfield-backmarker": "midfield",
        "backmarker": "backmarker",
        "nodata": None,
    }

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):
        super().__init__()
        self.email = email
        self.password = password
        self.totp_secret = totp_secret
        self._cookies: Optional[list[dict]] = None

    @classmethod
    async def from_settings(cls, db) -> "PuntingFormScraper":
        """Create scraper with credentials from app_settings."""
        from punty.models.settings import get_api_key

        email = await get_api_key(db, "punting_form_email")
        password = await get_api_key(db, "punting_form_password")
        totp_secret = await get_api_key(db, "punting_form_totp_secret")

        if not all([email, password, totp_secret]):
            raise ScraperError(
                "Punting Form credentials not configured. "
                "Set punting_form_email, punting_form_password, and punting_form_totp_secret in app_settings."
            )

        return cls(email=email, password=password, totp_secret=totp_secret)

    def _load_cached_cookies(self) -> Optional[list[dict]]:
        """Load cookies from cache file if they exist."""
        if COOKIE_CACHE_PATH.exists():
            try:
                with open(COOKIE_CACHE_PATH, "r") as f:
                    cookies = json.load(f)
                    logger.info(f"Loaded {len(cookies)} cached cookies")
                    return cookies
            except Exception as e:
                logger.warning(f"Failed to load cached cookies: {e}")
        return None

    def _save_cookies(self, cookies: list[dict]) -> None:
        """Save cookies to cache file."""
        try:
            with open(COOKIE_CACHE_PATH, "w") as f:
                json.dump(cookies, f)
            logger.info(f"Saved {len(cookies)} cookies to cache")
        except Exception as e:
            logger.warning(f"Failed to save cookies: {e}")

    async def _login(self, page) -> bool:
        """Perform login with TOTP authentication."""
        if not all([self.email, self.password, self.totp_secret]):
            logger.error("Missing credentials for Punting Form login")
            return False

        try:
            logger.info("Logging into Punting Form...")

            # Step 1: Navigate to login page
            await page.goto(f"{self.BASE_URL}/member/login", wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            # Step 2: Enter email and click NEXT
            await page.fill("input#Email", self.email)
            await page.click('button:has-text("NEXT")')
            await page.wait_for_timeout(3000)

            # Step 3: Enter password
            await page.fill("input#Password", self.password)

            # Step 4: Generate and enter TOTP code
            # Remove spaces from secret if present
            clean_secret = self.totp_secret.replace(" ", "")
            totp = pyotp.TOTP(clean_secret)
            code = totp.now()
            logger.info(f"Generated TOTP code: {code[:2]}****")

            # Fill each digit in separate input fields
            for i, digit in enumerate(code):
                await page.fill(f"input#otp{i+1}", digit)

            # Step 5: Click login
            await page.click("button#btn-login")
            await page.wait_for_timeout(5000)

            # Check if login succeeded
            current_url = page.url
            if "form-guide" in current_url or "dashboard" in current_url:
                logger.info("Punting Form login successful")
                # Save cookies
                cookies = await page.context.cookies()
                self._cookies = cookies
                self._save_cookies(cookies)
                return True
            else:
                logger.error(f"Login may have failed. Current URL: {current_url}")
                return False

        except Exception as e:
            logger.error(f"Punting Form login error: {e}")
            return False

    async def _ensure_authenticated(self, page) -> bool:
        """Ensure we have a valid session, logging in if needed."""
        # Try cached cookies first
        if not self._cookies:
            self._cookies = self._load_cached_cookies()

        if self._cookies:
            await page.context.add_cookies(self._cookies)
            # Verify session is still valid
            await page.goto(f"{self.BASE_URL}/form-guide", wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            html = await page.content()
            if "member/login" not in page.url and "logout" in html.lower():
                logger.info("Using cached session")
                return True
            else:
                logger.info("Cached session expired, logging in again")

        # Need to login
        return await self._login(page)

    def _build_race_url(self, venue: str, race_date: date, race_num: int) -> str:
        """Build URL for a specific race."""
        venue_lower = venue.lower()
        # Check venue mapping first
        venue_slug = self.VENUE_MAP.get(venue_lower)
        if not venue_slug:
            # Default: convert to slug format
            venue_slug = venue_lower.replace(" ", "-")
        date_str = race_date.strftime("%d-%m-%Y")
        return f"{self.BASE_URL}/form-guide/race/{venue_slug}-{date_str}-{race_num}-form"

    def _normalize_position(self, col: str) -> Optional[str]:
        """Normalize position column to standard values."""
        col_lower = col.lower().replace(" ", "").replace("/", "-")
        return self.POSITION_MAP.get(col_lower)

    def _parse_speed_map(self, soup: BeautifulSoup) -> list[dict]:
        """Parse speed map data from page."""
        positions = []

        runners = soup.select(".speed-map-grid .runner")
        for runner in runners:
            try:
                name_elem = runner.select_one(".runner-name .name")
                no_elem = runner.select_one(".runner-name .no")

                if not name_elem:
                    continue

                horse_name = name_elem.get_text().strip()
                saddlecloth = int(no_elem.get_text().strip()) if no_elem else None

                # Get position from parent cell
                parent_cell = runner.find_parent("li", class_="speed-map-cell")
                col = parent_cell.get("data-col", "") if parent_cell else ""
                position = self._normalize_position(col)

                # Get insights
                insights = {}
                for insight in runner.select(".col-item"):
                    label_elem = insight.select_one(".insights-label")
                    value_elem = insight.select_one(".insights-value")
                    if label_elem and value_elem:
                        label = label_elem.get_text().strip().lower()
                        value = value_elem.get_text().strip()
                        # Clean up value (remove icons)
                        value = "".join(c for c in value if c.isdigit() or c == ".")
                        insights[label] = value

                runner_data = {
                    "horse_name": horse_name,
                    "saddlecloth": saddlecloth,
                    "position": position,
                    "pf_speed_rank": insights.get("speed"),  # Early speed rank
                    "pf_settle": insights.get("settle"),  # Avg settle position
                    "pf_map_factor": insights.get("map"),  # Map advantage
                    "pf_jockey_factor": insights.get("j"),  # Jockey factor
                }

                if position:  # Only include runners with valid positions
                    positions.append(runner_data)

            except Exception as e:
                logger.warning(f"Error parsing runner in speed map: {e}")
                continue

        return positions

    async def scrape_speed_maps(
        self, venue: str, race_date: date, race_count: int
    ) -> AsyncGenerator[dict, None]:
        """Scrape speed maps for all races in a meeting.

        Yields progress events compatible with the orchestrator.
        """
        total = race_count + 1  # +1 for completion

        async with new_page() as page:
            # Authenticate
            if not await self._ensure_authenticated(page):
                yield {
                    "step": 0,
                    "total": total,
                    "label": "Punting Form authentication failed",
                    "status": "error",
                }
                return

            for race_num in range(1, race_count + 1):
                yield {
                    "step": race_num - 1,
                    "total": total,
                    "label": f"[PF] Fetching speed map for Race {race_num}...",
                    "status": "running",
                }

                race_url = self._build_race_url(venue, race_date, race_num)

                try:
                    await page.goto(race_url, wait_until="domcontentloaded")
                    await page.wait_for_timeout(2000)

                    # Check for 404
                    if "404" in await page.title():
                        yield {
                            "step": race_num,
                            "total": total,
                            "label": f"[PF] Race {race_num}: page not found",
                            "status": "done",
                            "race_number": race_num,
                            "positions": [],
                        }
                        continue

                    html = await page.content()
                    soup = BeautifulSoup(html, "lxml")

                    positions = self._parse_speed_map(soup)

                    if positions:
                        yield {
                            "step": race_num,
                            "total": total,
                            "label": f"[PF] Race {race_num}: {len(positions)} positions found",
                            "status": "done",
                            "race_number": race_num,
                            "positions": positions,
                        }
                    else:
                        yield {
                            "step": race_num,
                            "total": total,
                            "label": f"[PF] Race {race_num}: no speed map data",
                            "status": "done",
                            "race_number": race_num,
                            "positions": [],
                        }

                except Exception as e:
                    logger.error(f"Error scraping Punting Form race {race_num}: {e}")
                    yield {
                        "step": race_num,
                        "total": total,
                        "label": f"[PF] Race {race_num}: error - {e}",
                        "status": "error",
                        "race_number": race_num,
                        "positions": [],
                    }

        yield {
            "step": total,
            "total": total,
            "label": "[PF] Speed maps complete",
            "status": "complete",
        }

    async def scrape_race_insights(
        self, venue: str, race_date: date, race_num: int
    ) -> dict[str, Any]:
        """Scrape full insights for a single race.

        Returns detailed data including speed map, AI ratings, and form insights.
        """
        async with new_page() as page:
            if not await self._ensure_authenticated(page):
                raise ScraperError("Punting Form authentication failed")

            race_url = self._build_race_url(venue, race_date, race_num)
            await page.goto(race_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            return {
                "speed_map": self._parse_speed_map(soup),
                "venue": venue,
                "date": race_date.isoformat(),
                "race_number": race_num,
            }

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Scrape meeting data - not primary use case for Punting Form."""
        # Punting Form is primarily used for speed maps, not full meeting data
        return {"meeting": {}, "races": [], "runners": []}

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Scrape results - not implemented for Punting Form."""
        # Punting Form is not used for results
        return []
