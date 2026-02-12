"""Primary form data API scraper for speed maps, ratings, form data, conditions.

Primary data source for PuntyAI. Provides runner fields, form history,
speed maps, ratings, track conditions, weather, and scratchings.

Requires API key stored in app_settings as 'punting_form_api_key'.
"""

import json as _json
import logging
from datetime import date, datetime
from typing import Any, AsyncGenerator, Optional

import httpx

from punty.scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

BASE_URL = "https://api.puntingform.com.au/v2"

# Map runStyle abbreviations to our speed_map_position values
RUN_STYLE_MAP = {
    "l": "leader",
    "op": "on_pace",
    "op/mf": "on_pace",
    "mf": "midfield",
    "mf/bm": "midfield",
    "bm": "backmarker",
}

# Track condition number to label
_CONDITION_LABELS = {
    1: "Firm 1", 2: "Firm 2",
    3: "Good 3", 4: "Good 4",
    5: "Soft 5", 6: "Soft 6", 7: "Soft 7",
    8: "Heavy 8", 9: "Heavy 9", 10: "Heavy 10",
}

# Module-level cache: meetings list keyed by date string.
# Survives across scraper instances within the same process.
_meetings_cache: dict[str, list[dict]] = {}


def clear_meetings_cache():
    """Clear stale meeting cache. Called at midnight by scheduler."""
    _meetings_cache.clear()


def _settle_to_position(settle: int, field_size: int) -> Optional[str]:
    """Convert predicted settle position to speed_map_position.

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


def _record_to_json(record: dict | None) -> str | None:
    """Convert record dict {starts, firsts, seconds, thirds} to JSON string."""
    if not record or record.get("starts", 0) == 0:
        return None
    return _json.dumps({
        "starts": record.get("starts", 0),
        "wins": record.get("firsts", 0),
        "seconds": record.get("seconds", 0),
        "thirds": record.get("thirds", 0),
    })


def _a2e_to_json(career: dict | None, last100: dict | None,
                 combo_career: dict | None = None,
                 combo_last100: dict | None = None) -> str | None:
    """Convert A2E stat dicts to JSON string for jockey_stats/trainer_stats."""
    data = {}

    def _extract(src: dict | None, key: str):
        if not src or not src.get("runners"):
            return
        data[key] = {
            "a2e": src.get("a2E"),
            "pot": src.get("poT"),
            "strike_rate": src.get("strikeRate"),
            "wins": src.get("wins"),
            "runners": src.get("runners"),
        }

    _extract(career, "career")
    _extract(last100, "last100")
    _extract(combo_career, "combo_career")
    _extract(combo_last100, "combo_last100")
    return _json.dumps(data) if data else None


def _parse_pf_start_time(time_str: str | None) -> datetime | None:
    """Parse startTimeUTC like '2/12/2026 7:45:00 AM' to datetime."""
    if not time_str:
        return None
    try:
        # US-style: M/D/YYYY H:MM:SS AM/PM
        return datetime.strptime(time_str.strip(), "%m/%d/%Y %I:%M:%S %p")
    except (ValueError, TypeError):
        try:
            # Try ISO format as fallback
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


def _parse_last10(last10: str | None) -> tuple[str, str]:
    """Parse last10 like '40x42' → (form='4042', last_five='4042').

    'x' marks spell breaks, spaces are padding. Strip both.
    Returns (form_string, last_five_string).
    """
    if not last10:
        return ("", "")
    # Strip spaces and 'x' separators
    cleaned = last10.strip().replace("x", "").replace(" ", "")
    # '0' = unplaced (10th+), keep as-is
    last_five = cleaned[:5]
    return (cleaned, last_five)


def _pf_runner_to_dict(pf_runner: dict, race_id: str, meeting_id: str) -> dict:
    """Convert /form/fields runner to our Runner dict format."""
    jockey = pf_runner.get("jockey") or {}
    trainer = pf_runner.get("trainer") or {}

    horse_name = (pf_runner.get("name") or "").strip()
    barrier = pf_runner.get("barrier") or 0
    tab_no = pf_runner.get("tabNo")

    # Use saddlecloth (tabNo) in ID — stable across re-scrapes unlike barrier
    horse_slug = horse_name.lower().replace(" ", "-").replace("'", "")[:20]
    runner_id = f"{race_id}-{tab_no or barrier}-{horse_slug}"

    # Form
    form_full, last_five = _parse_last10(pf_runner.get("last10"))

    # Career record string
    starts = pf_runner.get("careerStarts", 0)
    wins = pf_runner.get("careerWins", 0)
    seconds = pf_runner.get("careerSeconds", 0)
    thirds = pf_runner.get("careerThirds", 0)
    career_record = f"{starts}: {wins}-{seconds}-{thirds}"

    # Jockey stats (A2E) — includes combo stats for trainer+jockey combo
    jockey_stats = _a2e_to_json(
        pf_runner.get("jockeyA2E_Career"),
        pf_runner.get("jockeyA2E_Last100"),
        pf_runner.get("trainerJockeyA2E_Career"),
        pf_runner.get("trainerJockeyA2E_Last100"),
    )

    # Trainer stats (A2E)
    trainer_stats = _a2e_to_json(
        pf_runner.get("trainerA2E_Career"),
        pf_runner.get("trainerA2E_Last100"),
    )

    return {
        "id": runner_id,
        "race_id": race_id,
        "horse_name": horse_name,
        "saddlecloth": tab_no,
        "barrier": barrier,
        "weight": pf_runner.get("weightTotal"),
        "jockey": (jockey.get("fullName") or "").strip(),
        "trainer": (trainer.get("fullName") or "").strip(),
        "trainer_location": (trainer.get("location") or "").strip() or None,
        "form": form_full,
        "last_five": last_five,
        "career_record": career_record,
        "horse_age": pf_runner.get("age"),
        "horse_sex": pf_runner.get("sex"),
        "horse_colour": pf_runner.get("colour"),
        "sire": pf_runner.get("sire"),
        "dam": pf_runner.get("dam"),
        "dam_sire": pf_runner.get("sireofDam"),
        "career_prize_money": pf_runner.get("prizeMoney"),
        "handicap_rating": pf_runner.get("handicap") if pf_runner.get("handicap") else None,
        "gear_changes": (pf_runner.get("gearChanges") or "").strip() or None,
        "scratched": pf_runner.get("emergencyIndicator", False) is True and pf_runner.get("tabNo") is None,
        # Stats (structured records → JSON)
        "track_stats": _record_to_json(pf_runner.get("trackRecord")),
        "distance_stats": _record_to_json(pf_runner.get("distanceRecord")),
        "track_dist_stats": _record_to_json(pf_runner.get("trackDistRecord")),
        "first_up_stats": _record_to_json(pf_runner.get("firstUpRecord")),
        "second_up_stats": _record_to_json(pf_runner.get("secondUpRecord")),
        "good_track_stats": _record_to_json(pf_runner.get("goodRecord")),
        "soft_track_stats": _record_to_json(pf_runner.get("softRecord")),
        "heavy_track_stats": _record_to_json(pf_runner.get("heavyRecord")),
        "jockey_stats": jockey_stats,
        "trainer_stats": trainer_stats,
        # Fields not in primary API (filled by racing.com supplement)
        "current_odds": None,
        "opening_odds": None,
        "place_odds": None,
    }


def _pf_form_entry_to_history(entry: dict) -> dict:
    """Convert a single form entry to our form_history JSON format."""
    track = entry.get("track") or {}
    jockey = entry.get("jockey") or {}

    # Parse flucs string like "opening,1.70;mid,1.75;starting,1.65;"
    flucs_str = entry.get("flucs", "")
    flucs = {}
    if flucs_str:
        for part in flucs_str.strip(";").split(";"):
            if "," in part:
                k, v = part.split(",", 1)
                try:
                    flucs[k.strip()] = float(v.strip())
                except (ValueError, TypeError):
                    pass

    # Parse in-run positions like "finish,1;settling_down,1;m800,1;m400,1;"
    in_run = entry.get("inRun", "")
    at800 = None
    at400 = None
    settled = None
    if in_run:
        for part in in_run.strip(";").split(";"):
            if "," in part:
                k, v = part.split(",", 1)
                k = k.strip()
                if k == "m800":
                    at800 = v.strip()
                elif k == "m400":
                    at400 = v.strip()
                elif k == "settling_down":
                    settled = v.strip()

    # Top 4 finishers
    top4_raw = entry.get("top4Finishers") or []
    top4 = [{"name": t.get("runnerName"), "pos": t.get("position")} for t in top4_raw[:4]]

    # Meeting date
    meet_date = entry.get("meetingDate", "")
    if "T" in meet_date:
        meet_date = meet_date.split("T")[0]

    return {
        "date": meet_date,
        "venue": track.get("name"),
        "distance": entry.get("distance"),
        "class": entry.get("raceClass"),
        "prize": entry.get("prizeMoney"),
        "track": entry.get("trackCondition"),
        "field_size": entry.get("starters"),
        "position": entry.get("position"),
        "margin": entry.get("margin"),
        "weight": entry.get("weight") or entry.get("weightTotal"),
        "jockey": jockey.get("fullName"),
        "barrier": entry.get("barrier"),
        "sp": entry.get("priceSP"),
        "sp_tab": entry.get("priceTAB"),
        "sp_bf": entry.get("priceBF"),
        "flucs": flucs if flucs else None,
        "time": entry.get("officialRaceTime"),
        "settled": settled,
        "at800": at800,
        "at400": at400,
        "comment": entry.get("stewardsReport"),
        "top4": top4 if top4 else None,
        "is_trial": entry.get("isBarrierTrial", False),
    }


class PuntingFormScraper(BaseScraper):
    """Primary form data API scraper."""

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        # Session-level caches (one API call covers all venues)
        self._conditions_cache: list | None = None
        self._scratchings_cache: list | None = None

    @classmethod
    async def from_settings(cls, db) -> "PuntingFormScraper":
        """Create scraper with API key from app_settings."""
        from punty.models.settings import get_api_key

        api_key = await get_api_key(db, "punting_form_api_key")
        if not api_key:
            raise ScraperError(
                "Form data API key not configured. "
                "Set punting_form_api_key in Settings → Form Data."
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
        """Make authenticated GET request."""
        params["apiKey"] = self.api_key
        client = await self._get_client()
        url = f"{BASE_URL}{path}"
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get("statusCode") != 200:
            error = data.get("error") or data.get("errors") or "Unknown API error"
            raise ScraperError(f"API error: {error}")

        return data.get("payLoad", [])

    # ---- Meeting resolution ----

    async def get_meetings(self, race_date: date) -> list[dict]:
        """Get list of meetings for a date. Uses module-level cache."""
        date_str = race_date.strftime("%Y-%m-%d")
        if date_str in _meetings_cache:
            return _meetings_cache[date_str]
        result = await self._api_get("/form/meetingslist", {"meetingDate": date_str})
        _meetings_cache[date_str] = result
        return result

    async def resolve_meeting_id(self, venue: str, race_date: date) -> Optional[int]:
        """Resolve our venue name to an API integer meetingId."""
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

        logger.warning(f"Could not resolve meetingId for venue={venue!r} on {race_date}")
        available = [m.get("track", {}).get("name", "?") for m in meetings]
        logger.info(f"Available meetings: {available}")
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
        """Fetch ratings for a meeting."""
        return await self._api_get("/Ratings/MeetingRatings", {
            "meetingId": meeting_id,
        })

    # ---- Fields (runner base data) ----

    async def get_fields(self, meeting_id: int, race_number: int = 0) -> dict:
        """Fetch race fields with full runner data.

        race_number=0 returns all races. Returns dict with track, races, runners etc.
        """
        params = {"meetingId": meeting_id}
        if race_number > 0:
            params["raceNumber"] = race_number
        return await self._api_get("/form/fields", params)

    # ---- Form history ----

    async def get_form(self, meeting_id: int, race_number: int = 0, runs: int = 10) -> list:
        """Fetch historical form data (up to 10 past starts per runner)."""
        params = {"meetingId": meeting_id, "runs": runs}
        if race_number > 0:
            params["raceNumber"] = race_number
        return await self._api_get("/form/form", params)

    # ---- Conditions (cached) ----

    async def get_conditions(self, jurisdiction: int = 0) -> list:
        """Fetch track conditions and weather for all upcoming meetings.

        Results are cached for the session (one call covers all venues).
        """
        if self._conditions_cache is None:
            self._conditions_cache = await self._api_get("/Updates/Conditions", {
                "jurisdiction": jurisdiction,
            })
        return self._conditions_cache

    async def get_conditions_for_venue(self, venue: str) -> dict | None:
        """Get conditions for a specific venue. Returns parsed dict or None."""
        conditions = await self.get_conditions()
        venue_lower = venue.lower().strip()

        # Strip sponsor prefixes for matching
        sponsor_prefixes = [
            "sportsbet ", "ladbrokes ", "bet365 ", "aquis ", "pointsbet ",
            "picklebet park ", "southside ", "tab ",
        ]
        clean_venue = venue_lower
        for prefix in sponsor_prefixes:
            if clean_venue.startswith(prefix):
                clean_venue = clean_venue[len(prefix):]
                break

        for cond in conditions:
            track = (cond.get("track") or "").lower().strip()
            if not track:
                continue
            if track == clean_venue or track == venue_lower:
                return self._parse_condition(cond)
            if track in clean_venue or clean_venue in track:
                return self._parse_condition(cond)
        return None

    def _parse_condition(self, cond: dict) -> dict:
        """Parse a conditions entry into our format."""
        tc_num = cond.get("trackConditionNumber", 0)
        tc_label = cond.get("trackCondition", "")
        # Build condition string like "Good 4" from number if label is missing
        condition = tc_label or _CONDITION_LABELS.get(tc_num) or None

        # Parse going stick from comment field (e.g. "Going Stick: 11.8")
        going_stick = None
        comment = cond.get("comment") or ""
        if "going stick" in comment.lower():
            import re
            m = re.search(r"going\s*stick[:\s]*([0-9.]+)", comment, re.IGNORECASE)
            if m:
                try:
                    going_stick = float(m.group(1))
                except (ValueError, TypeError):
                    pass

        # Parse numeric fields (API returns strings like "6.40", "Nil", etc.)
        def _to_float(val) -> float | None:
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def _to_int(val) -> int | None:
            if val is None:
                return None
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        # Rainfall: API returns "Nil" or "2.5" or descriptive strings
        raw_rainfall = cond.get("rainfall")
        rainfall = _to_float(raw_rainfall)

        # Irrigation: API returns descriptive strings like "6mm last 24hrs..."
        # Truthy if any irrigation string is present and not empty/Nil
        raw_irrigation = cond.get("irrigation")
        irrigation = bool(raw_irrigation and str(raw_irrigation).strip().lower() not in ("", "nil", "none", "no", "0"))

        return {
            "venue": cond.get("track"),
            "condition": condition,
            "rail": cond.get("rail"),
            "weather": cond.get("weather"),
            "penetrometer": _to_float(cond.get("penetrometer")),
            "wind_speed": _to_int(cond.get("wind")),
            "wind_direction": cond.get("windDirection"),
            "rainfall": rainfall,
            "irrigation": irrigation,
            "going_stick": going_stick,
            "abandoned": cond.get("abandonded", False),  # API has a typo: "abandonded"
        }

    # ---- Scratchings (cached) ----

    async def get_scratchings(self, jurisdiction: int = 0) -> list:
        """Fetch upcoming scratchings. Cached for session."""
        if self._scratchings_cache is None:
            self._scratchings_cache = await self._api_get("/Updates/Scratchings", {
                "jurisdiction": jurisdiction,
            })
        return self._scratchings_cache

    async def get_scratchings_for_meeting(self, pf_meeting_id: int) -> list[dict]:
        """Get scratchings for a specific meeting ID."""
        all_scratchings = await self.get_scratchings()
        return [
            {
                "race_number": s.get("raceNo"),
                "tab_no": s.get("tabNo"),
                "deduction": s.get("deduction"),
                "timestamp": s.get("timeStamp"),
            }
            for s in all_scratchings
            if s.get("meetingId") == pf_meeting_id or str(s.get("meetingId")) == str(pf_meeting_id)
        ]

    # ---- Full meeting scrape (primary data source) ----

    async def scrape_meeting_data(
        self, venue: str, race_date: date
    ) -> dict[str, Any]:
        """Scrape full meeting data from API.

        Returns dict in same format as RacingComScraper.scrape_meeting():
        {"meeting": {...}, "races": [...], "runners": [...]}
        """
        meeting_id_str = self.generate_meeting_id(venue, race_date)

        # Resolve PF meeting ID
        pf_meeting_id = await self.resolve_meeting_id(venue, race_date)
        if not pf_meeting_id:
            logger.warning(f"Meeting not found for {venue} on {race_date}")
            return {"meeting": {"id": meeting_id_str, "venue": venue, "date": race_date},
                    "races": [], "runners": []}

        # Fetch fields (all races)
        fields_data = await self.get_fields(pf_meeting_id)

        # fields_data is a dict with track, races, meetingId, railPosition, etc.
        track_info = fields_data.get("track") or {}

        meeting_dict = {
            "id": meeting_id_str,
            "venue": track_info.get("name") or venue,
            "date": race_date,
            "rail_position": fields_data.get("railPosition"),
            "meet_code": track_info.get("abbrev"),
        }

        races = []
        runners = []

        for pf_race in fields_data.get("races", []):
            race_num = pf_race.get("number", 0)
            race_id = f"{meeting_id_str}-r{race_num}"

            # Parse start time
            start_time = _parse_pf_start_time(pf_race.get("startTimeUTC"))

            # Parse prize money
            prize = pf_race.get("prizeMoney")
            try:
                prize = int(str(prize).replace(",", "")) if prize else None
            except (ValueError, TypeError):
                prize = None

            race_data = {
                "id": race_id,
                "meeting_id": meeting_id_str,
                "race_number": race_num,
                "name": pf_race.get("name", f"Race {race_num}"),
                "distance": pf_race.get("distance"),
                "class_": pf_race.get("raceClass"),
                "prize_money": prize,
                "start_time": start_time,
                "status": "scheduled",
                "race_type": "Thoroughbred",
                "age_restriction": pf_race.get("ageRestrictions"),
                "weight_type": pf_race.get("weightType"),
                "field_size": None,  # set after counting non-scratched runners
            }

            # Process runners
            race_runners = []
            pf_runners = pf_race.get("runners", [])
            for pf_runner in pf_runners:
                runner_dict = _pf_runner_to_dict(pf_runner, race_id, meeting_id_str)
                race_runners.append(runner_dict)

            # Set field size (non-emergency runners)
            race_data["field_size"] = sum(
                1 for r in race_runners if not r.get("scratched")
            )

            races.append(race_data)
            runners.extend(race_runners)

        logger.info(f"Primary: scraped {len(races)} races, {len(runners)} runners for {venue}")

        # Fetch form history and attach to runners
        try:
            form_data = await self.get_form(pf_meeting_id, runs=10)
            # form_data is a list of runners (when raceNumber=0, it's per the fields endpoint structure)
            # Actually /form/form returns same structure as /form/fields but with forms[] populated
            if isinstance(form_data, dict):
                # Dict response: iterate races
                for pf_race in form_data.get("races", []):
                    race_num = pf_race.get("number", 0)
                    for pf_runner in pf_race.get("runners", []):
                        forms = pf_runner.get("forms", [])
                        if forms:
                            name = (pf_runner.get("name") or "").strip()
                            history = [_pf_form_entry_to_history(f) for f in forms]
                            # Find matching runner and attach form_history
                            for r in runners:
                                if r["horse_name"] == name and r["race_id"].endswith(f"-r{race_num}"):
                                    r["form_history"] = _json.dumps(history)
                                    break
            elif isinstance(form_data, list):
                # List response: each item is a runner with forms[]
                for pf_runner in form_data:
                    forms = pf_runner.get("forms", [])
                    if forms:
                        name = (pf_runner.get("name") or "").strip()
                        history = [_pf_form_entry_to_history(f) for f in forms]
                        for r in runners:
                            if r["horse_name"] == name:
                                r["form_history"] = _json.dumps(history)
                                break
        except Exception as e:
            logger.warning(f"Form history failed (non-fatal): {e}")

        # Apply scratchings
        try:
            scratchings = await self.get_scratchings_for_meeting(pf_meeting_id)
            for s in scratchings:
                race_num = s.get("race_number")
                tab_no = s.get("tab_no")
                if race_num and tab_no:
                    race_id = f"{meeting_id_str}-r{race_num}"
                    for r in runners:
                        if r["race_id"] == race_id and r.get("saddlecloth") == tab_no:
                            r["scratched"] = True
                            break
        except Exception as e:
            logger.warning(f"Scratchings failed (non-fatal): {e}")

        return {"meeting": meeting_dict, "races": races, "runners": runners}

    async def scrape_meeting_fields_only(
        self, venue: str, race_date: date
    ) -> dict[str, Any]:
        """Lightweight scrape — fields only, no form history or scratchings.

        Used by the midnight calendar scrape where we only need race times
        and basic runner data for scheduling automation. Form history and
        scratchings are fetched later in the 5am morning scrape.

        Returns dict in same format as scrape_meeting_data().
        """
        meeting_id_str = self.generate_meeting_id(venue, race_date)

        pf_meeting_id = await self.resolve_meeting_id(venue, race_date)
        if not pf_meeting_id:
            logger.warning(f"Meeting not found for {venue} on {race_date}")
            return {"meeting": {"id": meeting_id_str, "venue": venue, "date": race_date},
                    "races": [], "runners": []}

        fields_data = await self.get_fields(pf_meeting_id)
        track_info = fields_data.get("track") or {}

        meeting_dict = {
            "id": meeting_id_str,
            "venue": track_info.get("name") or venue,
            "date": race_date,
            "rail_position": fields_data.get("railPosition"),
            "meet_code": track_info.get("abbrev"),
        }

        races = []
        runners = []

        for pf_race in fields_data.get("races", []):
            race_num = pf_race.get("number", 0)
            race_id = f"{meeting_id_str}-r{race_num}"

            start_time = _parse_pf_start_time(pf_race.get("startTimeUTC"))

            prize = pf_race.get("prizeMoney")
            try:
                prize = int(str(prize).replace(",", "")) if prize else None
            except (ValueError, TypeError):
                prize = None

            race_data = {
                "id": race_id,
                "meeting_id": meeting_id_str,
                "race_number": race_num,
                "name": pf_race.get("name", f"Race {race_num}"),
                "distance": pf_race.get("distance"),
                "class_": pf_race.get("raceClass"),
                "prize_money": prize,
                "start_time": start_time,
                "status": "scheduled",
                "race_type": "Thoroughbred",
                "age_restriction": pf_race.get("ageRestrictions"),
                "weight_type": pf_race.get("weightType"),
                "field_size": None,
            }

            race_runners = []
            for pf_runner in pf_race.get("runners", []):
                runner_dict = _pf_runner_to_dict(pf_runner, race_id, meeting_id_str)
                race_runners.append(runner_dict)

            race_data["field_size"] = sum(
                1 for r in race_runners if not r.get("scratched")
            )

            races.append(race_data)
            runners.extend(race_runners)

        logger.info(f"Fields-only: {len(races)} races, {len(runners)} runners for {venue}")

        return {"meeting": meeting_dict, "races": races, "runners": runners}

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
                # Pace insight fields (map to existing Runner columns)
                "pf_speed_rank": speed if speed > 0 and speed < 25 else None,
                "pf_settle": float(settle) if settle > 0 and settle < 25 else None,
                "pf_map_factor": float(map_a2e) if map_a2e else None,
                "pf_jockey_factor": float(jockey_a2e) if jockey_a2e else None,
                # Extra data (stored as JSON or used in context)
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

        # Step 1: Resolve meetingId from venue name
        yield {
            "step": 0,
            "total": total,
            "label": "Resolving meeting...",
            "status": "running",
        }

        try:
            meeting_id = await self.resolve_meeting_id(venue, race_date)
        except Exception as e:
            logger.error(f"API error resolving meeting: {e}")
            yield {
                "step": 0,
                "total": total,
                "label": f"API error: {e}",
                "status": "error",
            }
            return

        if not meeting_id:
            yield {
                "step": 0,
                "total": total,
                "label": f"Meeting not found for {venue}",
                "status": "error",
            }
            return

        logger.info(f"Resolved {venue} → meetingId={meeting_id}")

        # Step 2: Fetch speed maps + ratings in parallel
        try:
            speed_map_data = await self.get_speed_maps(meeting_id, race_no=0)
        except Exception as e:
            logger.error(f"Speed maps API failed: {e}")
            yield {
                "step": 0,
                "total": total,
                "label": f"Speed maps API failed: {e}",
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
            logger.warning(f"Ratings API failed (non-fatal): {e}")

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
                    "label": f"Race {race_no}: {len(positions)} positions found",
                    "status": "done",
                    "race_number": race_no,
                    "positions": positions,
                }
            else:
                yield {
                    "step": race_no,
                    "total": total,
                    "label": f"Race {race_no}: no speed map data",
                    "status": "done",
                    "race_number": race_no,
                    "positions": [],
                }

        # Fill in any missing races (API might not have data for every race)
        returned_races = {r.get("raceNo") for r in speed_map_data}
        for race_num in range(1, race_count + 1):
            if race_num not in returned_races:
                yield {
                    "step": race_num,
                    "total": total,
                    "label": f"Race {race_num}: not in API response",
                    "status": "done",
                    "race_number": race_num,
                    "positions": [],
                }

        await self.close()

        yield {
            "step": total,
            "total": total,
            "label": "Speed maps complete",
            "status": "complete",
        }

    # ---- Legacy interface (kept for compatibility) ----

    async def scrape_meeting(self, venue: str, race_date: date) -> dict[str, Any]:
        """Full meeting scrape via primary API."""
        return await self.scrape_meeting_data(venue, race_date)

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

    async def scrape_results(self, venue: str, race_date: date) -> list[dict[str, Any]]:
        """Not implemented (dividends come from TAB)."""
        return []
