"""Scraper for Racing Australia and TRC trainer rankings."""

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TRAINER_PREMIERSHIP_URL = "https://www.racingaustralia.horse/FreeServices/Premierships.aspx"
TRC_TRAINER_RANKINGS_URL = "https://www.thoroughbredracing.com/rankings/category/trainer/"


class RacingAustraliaScraper:
    """Scraper for Racing Australia statistics."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def scrape_trainer_premiership(
        self,
        season: str = "2025",
        category: str = "All",  # All, Metro, Provincial, Country, Picnic
    ) -> list[dict]:
        """Scrape trainer premiership table.

        Returns list of dicts with trainer stats:
        - name: Trainer name
        - wins: First place finishes
        - seconds: Second place finishes
        - thirds: Third place finishes
        - fourths: Fourth place finishes
        - fifths: Fifth place finishes
        - prize_money: Total prize money
        - strike_rate: Win percentage
        - starts: Total starts
        """
        # Map category to URL parameter
        category_map = {
            "All": "All",
            "Metro": "Metro",
            "Provincial": "Provincial",
            "Country": "Country",
            "Picnic": "Picnic",
        }
        cat_param = category_map.get(category, "All")

        url = f"{TRAINER_PREMIERSHIP_URL}?State=Undefined&Season={season}&Table=Trainer"
        logger.info(f"Scraping trainer premiership: {url}")

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            logger.error(f"Failed to fetch trainer premiership: {e}")
            return []

        soup = BeautifulSoup(html, "lxml")
        trainers = []

        # Find the premiership table
        table = soup.find("table", class_="premiership-table")
        if not table:
            # Try alternate table finding
            table = soup.find("table", id=lambda x: x and "Trainer" in str(x))
        if not table:
            # Find any table with trainer data
            tables = soup.find_all("table")
            for t in tables:
                if t.find("th", string=re.compile(r"Trainer|Strike Rate", re.I)):
                    table = t
                    break

        if not table:
            logger.warning("Could not find trainer premiership table")
            return []

        # Parse table rows
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 8:
                continue

            # Skip header rows
            if row.find("th"):
                continue

            try:
                # Extract trainer name (first cell, may have link)
                name_cell = cells[0]
                name_link = name_cell.find("a")
                name = name_link.get_text(strip=True) if name_link else name_cell.get_text(strip=True)

                if not name or name.lower() in ["trainer", "name", ""]:
                    continue

                # Extract stats - column order: Name, 1st, 2nd, 3rd, 4th, 5th, Prize$, SR%, Starts
                trainer_data = {
                    "name": self._normalize_trainer_name(name),
                    "name_original": name,
                    "wins": self._parse_int(cells[1].get_text(strip=True)),
                    "seconds": self._parse_int(cells[2].get_text(strip=True)),
                    "thirds": self._parse_int(cells[3].get_text(strip=True)),
                    "fourths": self._parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                    "fifths": self._parse_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0,
                    "prize_money": self._parse_prize(cells[6].get_text(strip=True)) if len(cells) > 6 else 0,
                    "strike_rate": self._parse_percentage(cells[7].get_text(strip=True)) if len(cells) > 7 else 0.0,
                    "starts": self._parse_int(cells[8].get_text(strip=True)) if len(cells) > 8 else 0,
                    "category": category,
                    "season": season,
                }

                if trainer_data["wins"] is not None and trainer_data["wins"] > 0:
                    trainers.append(trainer_data)

            except Exception as e:
                logger.debug(f"Error parsing trainer row: {e}")
                continue

        logger.info(f"Scraped {len(trainers)} trainers from premiership table")
        return trainers

    def _normalize_trainer_name(self, name: str) -> str:
        """Normalize trainer name for matching.

        Handles variations like:
        - "C Waller" vs "Chris Waller"
        - "G Waterhouse & A Bott" vs "Gai Waterhouse & Adrian Bott"
        """
        if not name:
            return ""

        # Uppercase for consistent matching
        normalized = name.upper().strip()

        # Remove common prefixes/suffixes
        normalized = re.sub(r"\s*\(.*?\)\s*", "", normalized)  # Remove parentheticals
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize spaces

        return normalized

    def _parse_int(self, value: str) -> Optional[int]:
        """Parse integer from string."""
        if not value:
            return None
        try:
            cleaned = re.sub(r"[^\d]", "", value)
            return int(cleaned) if cleaned else None
        except ValueError:
            return None

    def _parse_prize(self, value: str) -> Optional[int]:
        """Parse prize money from string like '$1,234,567'."""
        if not value:
            return None
        try:
            cleaned = re.sub(r"[^\d]", "", value)
            return int(cleaned) if cleaned else None
        except ValueError:
            return None

    def _parse_percentage(self, value: str) -> Optional[float]:
        """Parse percentage from string like '23.5%'."""
        if not value:
            return None
        try:
            cleaned = value.replace("%", "").strip()
            return float(cleaned)
        except ValueError:
            return None


class TRCTrainerScraper:
    """Scraper for TRC (Thoroughbred Racing Commentary) global trainer rankings."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def scrape_trainer_rankings(
        self,
        country: str = "AUS",
        pages: int = 2,  # Scrape first 2 pages (100 trainers at 50/page)
    ) -> list[dict]:
        """Scrape TRC global trainer rankings.

        Returns list of dicts with trainer stats:
        - name: Trainer name
        - name_normalized: Uppercase normalized name for matching
        - global_rank: Global ranking position
        - runners: Number of horses trained
        - runs: Total race starts
        - strike_rate: Win percentage
        - rating: TRC performance rating
        - g1_wins: Group 1 wins
        - g2_wins: Group 2 wins
        - g3_wins: Group 3 wins
        - points: Ranking points
        """
        trainers = []

        for page in range(1, pages + 1):
            url = f"{TRC_TRAINER_RANKINGS_URL}?page={page}&countries={country}"
            logger.info(f"Scraping TRC trainer rankings page {page}: {url}")

            try:
                response = await self.client.get(url)
                response.raise_for_status()
                html = response.text
            except Exception as e:
                logger.error(f"Failed to fetch TRC trainer rankings page {page}: {e}")
                continue

            soup = BeautifulSoup(html, "lxml")

            # Find trainer rows - they're in a table with ranking data
            table = soup.find("table", class_=lambda x: x and "ranking" in str(x).lower())
            if not table:
                # Try finding by structure
                table = soup.find("table")

            if not table:
                logger.warning(f"Could not find rankings table on page {page}")
                continue

            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                # TRC table has 14 columns:
                # 0: Empty, 1: Rank, 2: Last week, 3: Name, 4: Country,
                # 5: Runners, 6: Runs, 7: SR, 8: Rating, 9: G1, 10: G2, 11: G3, 12: Points, 13: Change
                if len(cells) < 13:
                    continue

                try:
                    # Parse rank from column 1 (not 0 which is empty)
                    rank = self._parse_int(cells[1].get_text(strip=True))
                    if rank is None:
                        continue

                    # Name is in column 3
                    name = cells[3].get_text(strip=True)
                    if not name or len(name) < 3:
                        continue

                    # Parse stats from fixed columns
                    trainer_data = {
                        "name": name,
                        "name_normalized": self._normalize_trainer_name(name),
                        "global_rank": rank,
                        "runners": self._parse_int(cells[5].get_text(strip=True)) or 0,
                        "runs": self._parse_int(cells[6].get_text(strip=True)) or 0,
                        "strike_rate": self._parse_float(cells[7].get_text(strip=True)) or 0.0,
                        "rating": self._parse_float(cells[8].get_text(strip=True)) or 0.0,
                        "g1_wins": self._parse_int(cells[9].get_text(strip=True)) or 0,
                        "g2_wins": self._parse_int(cells[10].get_text(strip=True)) or 0,
                        "g3_wins": self._parse_int(cells[11].get_text(strip=True)) or 0,
                        "points": self._parse_int(cells[12].get_text(strip=True)) or 0,
                        "source": "TRC",
                    }

                    trainers.append(trainer_data)

                except Exception as e:
                    logger.debug(f"Error parsing TRC trainer row: {e}")
                    continue

        logger.info(f"Scraped {len(trainers)} trainers from TRC rankings")
        return trainers

    def _normalize_trainer_name(self, name: str) -> str:
        """Normalize trainer name for matching."""
        if not name:
            return ""
        normalized = name.upper().strip()
        normalized = re.sub(r"\s*\(.*?\)\s*", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _parse_int(self, value: str) -> Optional[int]:
        """Parse integer from string."""
        if not value:
            return None
        try:
            cleaned = re.sub(r"[^\d]", "", value)
            return int(cleaned) if cleaned else None
        except ValueError:
            return None

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float from string."""
        if not value:
            return None
        try:
            cleaned = value.replace(",", "").replace("%", "").strip()
            return float(cleaned) if cleaned else None
        except ValueError:
            return None


def match_trainer_name(runner_trainer: str, premiership_trainers: list[dict]) -> Optional[dict]:
    """Match a runner's trainer name to premiership/rankings data.

    Handles common variations in trainer name formatting.
    Works with both Racing Australia and TRC data formats.
    """
    if not runner_trainer:
        return None

    # Normalize the runner trainer name
    runner_normalized = runner_trainer.upper().strip()
    runner_normalized = re.sub(r"\s+", " ", runner_normalized)

    # Try exact match first
    for trainer in premiership_trainers:
        # Handle both TRC (name_normalized) and Racing Australia (name) formats
        trainer_name = trainer.get("name_normalized") or trainer.get("name", "")
        trainer_name_upper = trainer_name.upper() if trainer_name else ""

        if trainer_name_upper == runner_normalized:
            return trainer
        # Also try original name field
        original_name = trainer.get("name_original") or trainer.get("name", "")
        if original_name.upper() == runner_normalized:
            return trainer

    # Try partial matching for partnership trainers (e.g., "Waterhouse" matches "G Waterhouse & A Bott")
    runner_parts = runner_normalized.split()
    for trainer in premiership_trainers:
        trainer_name = trainer.get("name_normalized") or trainer.get("name", "")
        trainer_parts = trainer_name.upper().split()
        # Check if last name matches
        if runner_parts and trainer_parts:
            runner_surname = runner_parts[-1]
            if any(runner_surname == part for part in trainer_parts):
                return trainer

    # Try fuzzy matching - check if runner name is contained in trainer name or vice versa
    for trainer in premiership_trainers:
        trainer_name = trainer.get("name_normalized") or trainer.get("name", "")
        trainer_name_upper = trainer_name.upper() if trainer_name else ""
        if runner_normalized in trainer_name_upper or trainer_name_upper in runner_normalized:
            return trainer

    return None


def format_trainer_stats(trainer_data: dict) -> str:
    """Format trainer stats for AI context."""
    if not trainer_data:
        return ""

    # Check if it's TRC data (has global_rank) or Racing Australia data (has wins)
    if trainer_data.get("source") == "TRC" or "global_rank" in trainer_data:
        # TRC format: ranking, strike rate, rating, group wins
        parts = [f"Ranked #{trainer_data.get('global_rank', '?')} (AUS)"]
        if trainer_data.get("strike_rate"):
            parts.append(f"{trainer_data['strike_rate']}% SR")
        if trainer_data.get("rating"):
            parts.append(f"Rating {trainer_data['rating']}")
        g1 = trainer_data.get("g1_wins", 0)
        g2 = trainer_data.get("g2_wins", 0)
        g3 = trainer_data.get("g3_wins", 0)
        if g1 or g2 or g3:
            parts.append(f"G1/G2/G3: {g1}/{g2}/{g3}")
        return ", ".join(parts)
    else:
        # Racing Australia format
        return (
            f"{trainer_data.get('wins', 0)}-{trainer_data.get('seconds', 0)}-{trainer_data.get('thirds', 0)} "
            f"from {trainer_data.get('starts', 0)} ({trainer_data.get('strike_rate', 0)}% SR), "
            f"${trainer_data.get('prize_money', 0):,} prize money"
        )
