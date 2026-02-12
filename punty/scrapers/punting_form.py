"""Punting Form API scraper for speed maps, ratings, and form data.

Requires API key stored in app_settings as 'punting_form_api_key'.
API docs: https://docs.puntingform.com.au/
"""

import logging
from datetime import date
from typing import Any, AsyncGenerator, Optional

import httpx

from punty.scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

BASE_URL = "https://api.puntingform.com.au/v2"

# Map PF runStyle abbreviations to our speed_map_position values
RUN_STYLE_MAP = {
    "l": "leader",
    "op": "on_pace",
    "op/mf": "on_pace",
    "mf": "midfield",
    "mf/bm": "midfield",
    "bm": "backmarker",
}


def _settle_to_position(settle: int, field_size: int) -> Optional[str]:
    """Convert PF predicted settle position to speed_map_position.

    Settle is the predicted position in the field (1 = leads).
    We scale relative to field size to handle small vs large fields.
    """
    if settle <= 0 or settle >= 25:
        return None
    if field_size <= 0:
        field_size = 12  # default assumption

    # Settle=1 always means leads
    if settle == 1:
        return "leader"

    # Scale relative to field size for remaining positions
    ratio = settle / field_size
    if ratio <= 0.35:
        return "on_pace"
    elif ratio <= 0.65:
        return "midfield"
    else:
        return "backmarker"


class PuntingFormScraper(BaseScraper):
    """Scraper for Punting Form API — speed maps, ratings, and form data."""

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def from_settings(cls, db) -> "PuntingFormScraper":
        """Create scraper with API key from app_settings."""
        from punty.models.settings import get_api_key

        api_key = await get_api_key(db, "punting_form_api_key")
        if not api_key:
            raise ScraperError(
                "Punting Form API key not configured. "
                "Set punting_form_api_key in Settings → Punting Form."
            )
        return cls(api_key=api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"accept": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _api_get(self, path: str, params: dict) -> dict:
        """Make authenticated GET request to PF API."""
        params["apiKey"] = self.api_key
        client = await self._get_client()
        url = f"{BASE_URL}{path}"
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get("statusCode") != 200:
            error = data.get("error") or data.get("errors") or "Unknown API error"
            raise ScraperError(f"PF API error: {error}")

        return data.get("payLoad", [])

    # ---- Meeting resolution ----

    async def get_meetings(self, race_date: date) -> list[dict]:
        """Get list of meetings for a date. Returns PF meeting objects."""
        date_str = race_date.strftime("%Y-%m-%d")
        return await self._api_get("/form/meetingslist", {"meetingDate": date_str})

    async def resolve_meeting_id(self, venue: str, race_date: date) -> Optional[int]:
        """Resolve our venue name to a PF integer meetingId."""
        meetings = await self.get_meetings(race_date)
        venue_lower = venue.lower().strip()

        # Strip common sponsor prefixes
        sponsor_prefixes = [
            "sportsbet ", "ladbrokes ", "bet365 ", "aquis ", "pointsbet ",
            "picklebet park ", "southside ", "tab ",
        ]
        clean_venue = venue_lower
        for prefix in sponsor_prefixes:
            if clean_venue.startswith(prefix):
                clean_venue = clean_venue[len(prefix):]
                break

        for m in meetings:
            track_name = m.get("track", {}).get("name", "").lower().strip()
            if track_name == clean_venue or track_name == venue_lower:
                return int(m["meetingId"])
            # Partial match: "Pakenham" matches "Sportsbet Pakenham"
            if track_name in clean_venue or clean_venue in track_name:
                return int(m["meetingId"])

        logger.warning(f"PF: Could not resolve meetingId for venue={venue!r} on {race_date}")
        available = [m.get("track", {}).get("name", "?") for m in meetings]
        logger.info(f"PF: Available meetings: {available}")
        return None

    # ---- Speed Maps ----

    async def get_speed_maps(self, meeting_id: int, race_no: int = 0) -> list[dict]:
        """Fetch speed maps for a meeting (race_no=0 for all races)."""
        return await self._api_get("/User/Speedmaps", {
            "meetingId": meeting_id,
            "raceNo": race_no,
        })

    # ---- Ratings ----

    async def get_ratings(self, meeting_id: int) -> list[dict]:
        """Fetch PF ratings for a meeting."""
        return await self._api_get("/Ratings/MeetingRatings", {
            "meetingId": meeting_id,
        })

    # ---- Conditions ----

    async def get_conditions(self, jurisdiction: int = 0) -> list:
        """Fetch track conditions and weather for upcoming meetings.

        jurisdiction: 0=all, 1=?, 2=?
        """
        return await self._api_get("/Updates/Conditions", {
            "jurisdiction": jurisdiction,
        })

    # ---- Scratchings ----

    async def get_scratchings(self, jurisdiction: int = 0) -> list:
        """Fetch upcoming scratchings with timestamps and deductions."""
        return await self._api_get("/Updates/Scratchings", {
            "jurisdiction": jurisdiction,
        })

    # ---- Combined speed map + ratings scrape ----

    def _parse_speed_map_items(self, race_items: list[dict], ratings_by_tab: dict, field_size: int) -> list[dict]:
        """Parse speed map items for a single race, enriched with ratings data."""
        positions = []

        for item in race_items:
            runner_name = item.get("runnerName", "")
            tab_no = item.get("tabNo")
            speed = item.get("speed", 0)
            settle = item.get("settle", 25)
            map_a2e = item.get("mapA2E", 0)
            jockey_a2e = item.get("jockeyA2E", 0)

            # Determine speed_map_position from ratings runStyle (more reliable)
            # then fall back to settle position from speed maps
            position = None
            rating = ratings_by_tab.get(tab_no, {})
            run_style = rating.get("runStyle", "").strip().lower()
            if run_style and run_style != "no data":
                position = RUN_STYLE_MAP.get(run_style)

            if not position:
                position = _settle_to_position(settle, field_size)

            runner_data = {
                "horse_name": runner_name,
                "saddlecloth": tab_no,
                "position": position,
                # PF insight fields (map to existing Runner columns)
                "pf_speed_rank": speed if speed > 0 and speed < 25 else None,
                "pf_settle": float(settle) if settle > 0 and settle < 25 else None,
                "pf_map_factor": float(map_a2e) if map_a2e else None,
                "pf_jockey_factor": float(jockey_a2e) if jockey_a2e else None,
                # Extra PF data (stored as JSON or used in context)
                "pf_ai_score": item.get("pfaiScore"),
                "pf_ai_price": item.get("pfaiPrice"),
                "pf_ai_rank": item.get("pfaiRank"),
                "pf_assessed_price": item.get("assessedPrice"),
            }

            if position:  # Only include runners with valid positions
                positions.append(runner_data)

        return positions

    async def scrape_speed_maps(
        self, venue: str, race_date: date, race_count: int
    ) -> AsyncGenerator[dict, None]:
        """Scrape speed maps for all races in a meeting.

        Yields progress events compatible with the orchestrator.
        """
        total = race_count + 1  # +1 for completion

        # Step 1: Resolve PF meetingId from venue name
        yield {
            "step": 0,
            "total": total,
            "label": "[PF] Resolving meeting ID...",
            "status": "running",
        }

        try:
            meeting_id = await self.resolve_meeting_id(venue, race_date)
        except Exception as e:
            logger.error(f"PF API error resolving meeting: {e}")
            yield {
                "step": 0,
                "total": total,
                "label": f"[PF] API error: {e}",
                "status": "error",
            }
            return

        if not meeting_id:
            yield {
                "step": 0,
                "total": total,
                "label": f"[PF] Meeting not found for {venue}",
                "status": "error",
            }
            return

        logger.info(f"PF: Resolved {venue} → meetingId={meeting_id}")

        # Step 2: Fetch speed maps + ratings in parallel
        try:
            speed_map_data = await self.get_speed_maps(meeting_id, race_no=0)
        except Exception as e:
            logger.error(f"PF speed maps API failed: {e}")
            yield {
                "step": 0,
                "total": total,
                "label": f"[PF] Speed maps API failed: {e}",
                "status": "error",
            }
            return

        # Build ratings lookup: {race_no: {tab_no: rating_data}}
        ratings_by_race: dict[int, dict[int, dict]] = {}
        try:
            ratings_data = await self.get_ratings(meeting_id)
            for r in ratings_data:
                race_no = r.get("raceNo", 0)
                tab_no = r.get("tabNo", 0)
                if race_no not in ratings_by_race:
                    ratings_by_race[race_no] = {}
                ratings_by_race[race_no][tab_no] = r
        except Exception as e:
            logger.warning(f"PF ratings API failed (non-fatal): {e}")

        # Step 3: Process each race
        for race_entry in speed_map_data:
            race_no = race_entry.get("raceNo", 0)
            items = race_entry.get("items", [])
            field_size = len(items)
            ratings_for_race = ratings_by_race.get(race_no, {})

            positions = self._parse_speed_map_items(items, ratings_for_race, field_size)

            if positions:
                yield {
                    "step": race_no,
                    "total": total,
                    "label": f"[PF] Race {race_no}: {len(positions)} positions found",
                    "status": "done",
                    "race_number": race_no,
                    "positions": positions,
                }
            else:
                yield {
                    "step": race_no,
                    "total": total,
                    "label": f"[PF] Race {race_no}: no speed map data",
                    "status": "done",
                    "race_number": race_no,
                    "positions": [],
                }

        # Fill in any missing races (PF might not have data for every race)
        returned_races = {r.get("raceNo") for r in speed_map_data}
        for race_num in range(1, race_count + 1):
            if race_num not in returned_races:
                yield {
                    "step": race_num,
                    "total": total,
                    "label": f"[PF] Race {race_num}: not in API response",
                    "status": "done",
                    "race_number": race_num,
                    "positions": [],
                }

        await self.close()

        yield {
            "step": total,
            "total": total,
            "label": "[PF] Speed maps complete",
            "status": "complete",
        }

    # ---- Legacy interface (kept for compatibility) ----

    async def scrape_race_insights(
        self, venue: str, race_date: date, race_num: int
    ) -> dict[str, Any]:
        """Fetch insights for a single race via API."""
        meeting_id = await self.resolve_meeting_id(venue, race_date)
        if not meeting_id:
            return {"speed_map": [], "venue": venue, "date": race_date.isoformat(), "race_number": race_num}

        speed_maps = await self.get_speed_maps(meeting_id, race_no=race_num)
        ratings_by_tab = {}
        try:
            ratings = await self.get_ratings(meeting_id)
            for r in ratings:
                if r.get("raceNo") == race_num:
                    ratings_by_tab[r.get("tabNo", 0)] = r
        except Exception:
            pass

        items = speed_maps[0].get("items", []) if speed_maps else []
        positions = self._parse_speed_map_items(items, ratings_by_tab, len(items))

        await self.close()
        return {
            "speed_map": positions,
            "venue": venue,
            "date": race_date.isoformat(),
            "race_number": race_num,
        }

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Not primary use case for PF — returns empty data."""
        return {"meeting": {}, "races": [], "runners": []}

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Not implemented for PF."""
        return []
