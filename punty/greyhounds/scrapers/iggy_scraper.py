"""Iggy-Joey greyhound predictions scraper.

Data source: Betfair Data Scientists Iggy-Joey model
  - GitHub: https://betfair-datascientists.github.io/
  - Daily predictions CSV with rated prices and early speed ratings

The Iggy model provides:
  - iggy_rated_price: model's estimated true probability (as decimal odds)
  - iggy_early_speed: predicted early speed rating

Data files are published as CSVs:
  - Historical yearly ZIPs: ANZ_Greyhounds_{YYYY}.zip
  - Current month CSVs: ANZ_Greyhounds_{YYYY}_{MM}.csv
  - Model V2 predictions: Iggy_Model_V2_Results_{YYYY}.csv

Download base URL: https://betfair-datascientists.github.io/data/assets/

Betfair market CSV columns (ANZ_Greyhounds_*.csv):
  LOCAL_MEETING_DATE, SCHEDULED_RACE_TIME, ACTUAL_OFF_TIME, TRACK, STATE_CODE,
  RACE_NO, WIN_MARKET_ID, WIN_MARKET_NAME, PLACE_MARKET_ID, RACING_TYPE,
  DISTANCE, RACE_TYPE, SELECTION_ID, TAB_NUMBER, SELECTION_NAME,
  WIN_RESULT, WIN_BSP, PLACE_RESULT, PLACE_BSP, WIN_BSP_VOLUME,
  WIN_PREPLAY_MAX_PRICE_TAKEN, WIN_PREPLAY_MIN_PRICE_TAKEN,
  WIN_PREPLAY_LAST_PRICE_TAKEN, WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN,
  WIN_PREPLAY_VOLUME, WIN_INPLAY_MAX_PRICE_TAKEN, WIN_INPLAY_MIN_PRICE_TAKEN,
  WIN_LAST_PRICE_TAKEN, WIN_INPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN,
  WIN_INPLAY_VOLUME, PLACE_BSP_VOLUME, PLACE_MAX_PRICE_TAKEN,
  PLACE_MIN_PRICE_TAKEN, PLACE_LAST_PRICE_TAKEN,
  PLACE_WEIGHTED_AVERAGE_PRICE_TAKEN, PLACE_PREPLAY_VOLUME,
  BEST_AVAIL_BACK_AT_SCHEDULED_OFF, BEST_AVAIL_LAY_AT_SCHEDULED_OFF,
  BACK_MARKET_PERCENTAGE_AT_SCHEDULED_OFF, LAY_MARKET_PERCENTAGE_AT_SCHEDULED_OFF

Iggy Model V2 CSV columns (Iggy_Model_V2_Results_*.csv):
  Date, Track, Race Name, Race, MarketId, SelectionId,
  Rug, Dog, Rated Price, Early Speed,
  WIN_RESULT, WIN_BSP, PLACE_RESULT, PLACE_BSP, WIN_VALUE

Local data cache: scripts/_greyhound_data/
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

ASSETS_BASE = "https://betfair-datascientists.github.io/data/assets"
_LOCAL_CACHE = Path("scripts/_greyhound_data")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PuntyAI/1.0)",
    "Accept": "*/*",
}


class IggyScraperError(Exception):
    """Raised when Iggy data fetch fails."""
    pass


class IggyScraper:
    """Fetch and parse Iggy-Joey greyhound model predictions.

    Usage:
        scraper = IggyScraper()
        predictions = await scraper.fetch_predictions_for_date(date(2026, 3, 27))
        # Or load from local cache:
        predictions = scraper.load_cached_predictions(date(2026, 3, 27))
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or _LOCAL_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_monthly_csv(self, year: int, month: int) -> Path:
        """Download a monthly CSV from the Iggy data source.

        Saves to local cache and returns the file path.

        TODO: Implement download logic.
        """
        filename = f"ANZ_Greyhounds_{year}_{month:02d}.csv"
        url = f"{ASSETS_BASE}/{filename}"
        dest = self.cache_dir / filename

        if dest.exists():
            logger.info(f"Iggy: using cached {filename}")
            return dest

        logger.info(f"Iggy: downloading {url}")
        async with httpx.AsyncClient(headers=_HEADERS, timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"Iggy: saved {filename} ({len(resp.content)} bytes)")

        return dest

    async def fetch_yearly_zip(self, year: int) -> Path:
        """Download a yearly ZIP and extract CSVs to local cache.

        TODO: Implement download + extraction.
        """
        filename = f"ANZ_Greyhounds_{year}.zip"
        url = f"{ASSETS_BASE}/{filename}"
        zip_dest = self.cache_dir / filename

        if zip_dest.exists():
            logger.info(f"Iggy: using cached {filename}")
            return zip_dest

        logger.info(f"Iggy: downloading {url}")
        async with httpx.AsyncClient(headers=_HEADERS, timeout=120.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            zip_dest.write_bytes(resp.content)
            logger.info(f"Iggy: saved {filename} ({len(resp.content)} bytes)")

        # Extract CSVs
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(self.cache_dir)
            logger.info(f"Iggy: extracted {len(zf.namelist())} files from {filename}")

        return zip_dest

    async def fetch_iggy_v2_predictions(self, year: int, month: Optional[int] = None) -> Path:
        """Download Iggy Model V2 predictions CSV.

        V2 is the current model (from Aug 2024 onwards).
        """
        if month:
            filename = f"Iggy_Model_V2_Results_{year}_{month:02d}.csv"
        else:
            filename = f"Iggy_Model_V2_Results_{year}.csv"

        url = f"{ASSETS_BASE}/{filename}"
        dest = self.cache_dir / filename

        if dest.exists():
            logger.info(f"Iggy V2: using cached {filename}")
            return dest

        logger.info(f"Iggy V2: downloading {url}")
        async with httpx.AsyncClient(headers=_HEADERS, timeout=60.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"Iggy V2: saved {filename} ({len(resp.content)} bytes)")

        return dest

    def load_predictions_for_date(
        self, race_date: date
    ) -> list[dict[str, Any]]:
        """Load Iggy predictions for a specific date from local cache.

        Searches cached CSVs for rows matching the given date.
        Returns list of dicts per runner:
            {venue, race_number, box_number, dog_name,
             iggy_rated_price, iggy_early_speed}

        TODO: Parse actual CSV columns once format is confirmed from downloaded data.
        """
        # Try monthly CSV first, then yearly
        year = race_date.year
        month = race_date.month
        monthly = self.cache_dir / f"ANZ_Greyhounds_{year}_{month:02d}.csv"
        if monthly.exists():
            return self._parse_csv_for_date(monthly, race_date)

        # Try yearly extracted CSVs
        # Yearly ZIPs may extract to a single combined CSV or multiple monthly ones
        yearly_csv = self.cache_dir / f"ANZ_Greyhounds_{year}.csv"
        if yearly_csv.exists():
            return self._parse_csv_for_date(yearly_csv, race_date)

        logger.warning(f"Iggy: no cached data for {race_date}")
        return []

    def _parse_csv_for_date(
        self, csv_path: Path, race_date: date
    ) -> list[dict[str, Any]]:
        """Parse a CSV file and extract rows matching a specific date.

        Handles two CSV formats:
          1. Betfair market data (ANZ_Greyhounds_*.csv):
             Columns: LOCAL_MEETING_DATE, TRACK, RACE_NO, TAB_NUMBER,
                      SELECTION_NAME, WIN_BSP, PLACE_BSP, ...
          2. Iggy V2 predictions (Iggy_Model_V2_Results_*.csv):
             Columns: Date, Track, Race, Rug, Dog, Rated Price, Early Speed, ...
        """
        results = []
        date_str = race_date.isoformat()

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Detect format by column names
                    if "Rated Price" in row:
                        # Iggy V2 format
                        row_date = row.get("Date", "")
                        if date_str in row_date:
                            results.append({
                                "venue": row.get("Track", ""),
                                "race_number": int(row.get("Race", 0)),
                                "box_number": int(row.get("Rug", 0)),
                                "dog_name": row.get("Dog", ""),
                                "iggy_rated_price": _safe_float(row.get("Rated Price")),
                                "iggy_early_speed": _safe_float(row.get("Early Speed")),
                                "win_result": _safe_float(row.get("WIN_RESULT")),
                                "win_bsp": _safe_float(row.get("WIN_BSP")),
                                "place_result": _safe_float(row.get("PLACE_RESULT")),
                                "place_bsp": _safe_float(row.get("PLACE_BSP")),
                            })
                    elif "LOCAL_MEETING_DATE" in row:
                        # Betfair market data format
                        row_date = row.get("LOCAL_MEETING_DATE", "")
                        if date_str in row_date:
                            results.append({
                                "venue": row.get("TRACK", ""),
                                "race_number": int(row.get("RACE_NO", 0)),
                                "box_number": int(row.get("TAB_NUMBER", 0)),
                                "dog_name": row.get("SELECTION_NAME", ""),
                                "state": row.get("STATE_CODE", ""),
                                "distance": int(row.get("DISTANCE", 0) or 0),
                                "grade": row.get("RACE_TYPE", ""),
                                "win_bsp": _safe_float(row.get("WIN_BSP")),
                                "place_bsp": _safe_float(row.get("PLACE_BSP")),
                                "win_result": row.get("WIN_RESULT", ""),
                                "place_result": row.get("PLACE_RESULT", ""),
                                "selection_id": row.get("SELECTION_ID", ""),
                                "market_id": row.get("WIN_MARKET_ID", ""),
                            })
        except Exception as e:
            logger.error(f"Iggy: error parsing {csv_path}: {e}")

        logger.info(f"Iggy: found {len(results)} predictions for {race_date} in {csv_path.name}")
        return results


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
