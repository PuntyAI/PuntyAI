"""PointsBet odds scraper via public JSON API.

Uses direct httpx HTTP calls to the PointsBet racing API — no browser needed.

API endpoints:
  - Meetings list: GET api.au.pointsbet.com/api/racing/v4/meetings?startDate=...&endDate=...
  - Race detail:   GET api.au.pointsbet.com/api/v2/racing/races/{eventId}
"""

import logging
from datetime import date, datetime, timezone

import httpx

from punty.venues import normalize_venue

logger = logging.getLogger(__name__)

API_BASE = "https://api.au.pointsbet.com"
MAX_VALID_ODDS = 501.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://pointsbet.com.au",
    "Referer": "https://pointsbet.com.au/",
}


class PointsBetScraper:
    """Scrape PointsBet fixed odds via public JSON API."""

    async def scrape_odds_for_meeting(
        self,
        venue: str,
        race_date: date,
        meeting_id: str,
        race_count: int,
    ) -> list[dict]:
        """Fetch PointsBet odds for all races at a venue.

        Returns list of dicts: {race_number, horse_name, saddlecloth,
        current_odds, opening_odds, place_odds, scratched}
        """
        venue_norm = normalize_venue(venue)

        # Step 1: Find race IDs for this venue from meetings endpoint
        race_ids = await self._find_race_ids(venue_norm, race_date)
        if not race_ids:
            logger.warning(f"PointsBet: no races found for {venue} on {race_date}")
            return []

        logger.info(f"PointsBet: found {len(race_ids)} races for {venue}")

        # Step 2: Fetch odds for each race
        all_odds: list[dict] = []
        async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
            for race_num, event_id in race_ids:
                try:
                    odds = await self._fetch_race_odds(client, event_id, race_num)
                    all_odds.extend(odds)
                except Exception as e:
                    logger.warning(f"PointsBet: failed R{race_num} ({event_id}): {e}")

        if all_odds:
            logger.info(f"PointsBet: {len(all_odds)} runner odds for {venue}")
        else:
            logger.warning(f"PointsBet: no odds captured for {venue}")

        return all_odds

    async def _find_race_ids(
        self, venue_norm: str, race_date: date
    ) -> list[tuple[int, str]]:
        """Find PB race event IDs for a venue on a given date.

        Returns list of (race_number, event_id) tuples sorted by race number.
        """
        # PB meetings endpoint uses UTC date range
        dt = datetime(race_date.year, race_date.month, race_date.day, tzinfo=timezone.utc)
        date_str = dt.strftime("%Y-%m-%dT01:00:00.000Z")
        url = f"{API_BASE}/api/racing/v4/meetings"
        params = {"startDate": date_str, "endDate": date_str}

        try:
            async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"PointsBet meetings API failed: {e}")
            return []

        # Response is [{groupLabel: "date", meetings: [...]}, ...]
        all_meetings: list[dict] = []
        if isinstance(data, list):
            for group in data:
                if isinstance(group, dict) and "meetings" in group:
                    all_meetings.extend(group.get("meetings", []))
                elif isinstance(group, dict) and "venue" in group:
                    # Flat list of meetings (alternative format)
                    all_meetings.append(group)

        # Find meeting matching our venue
        race_ids: list[tuple[int, str]] = []
        for meeting in all_meetings:
            if not isinstance(meeting, dict):
                continue
            # Only thoroughbred racing (1 = thoroughbred)
            if meeting.get("racingType") not in ("Thoroughbred", 1):
                continue
            pb_venue = normalize_venue(meeting.get("venue") or "")
            if pb_venue != venue_norm and venue_norm not in pb_venue and pb_venue not in venue_norm:
                continue

            # Found our venue — extract race IDs
            races = meeting.get("races", [])
            for race in races:
                if not isinstance(race, dict):
                    continue
                event_id = str(race.get("eventId") or race.get("raceId") or "")
                race_num = race.get("raceNumber") or race.get("number")
                if not event_id or not race_num:
                    continue
                # Skip suspended races
                trading = race.get("tradingStatus")
                if trading in ("Suspended", 3):
                    continue
                race_ids.append((int(race_num), event_id))

            if race_ids:
                break  # Found our venue

        return sorted(race_ids, key=lambda x: x[0])

    async def scrape_race_statuses(
        self, venue: str, race_date: date
    ) -> dict:
        """Get race statuses from PointsBet meetings API.

        Maps PB resultStatus to standard status strings:
          resultStatus 2 → "Paying"
          tradingStatus 3 + resultStatus 0 → "Open" (in progress)
          tradingStatus 1 + resultStatus 0 → "Open" (not started)

        Returns: {"statuses": {1: "Paying", 2: "Open", ...}}
        """
        venue_norm = normalize_venue(venue)
        dt = datetime(race_date.year, race_date.month, race_date.day, tzinfo=timezone.utc)
        date_str = dt.strftime("%Y-%m-%dT01:00:00.000Z")
        url = f"{API_BASE}/api/racing/v4/meetings"
        params = {"startDate": date_str, "endDate": date_str}

        try:
            async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"PointsBet status API failed: {e}")
            return {"statuses": {}}

        all_meetings: list[dict] = []
        if isinstance(data, list):
            for group in data:
                if isinstance(group, dict) and "meetings" in group:
                    all_meetings.extend(group.get("meetings", []))
                elif isinstance(group, dict) and "venue" in group:
                    all_meetings.append(group)

        statuses: dict[int, str] = {}
        for meeting in all_meetings:
            if not isinstance(meeting, dict):
                continue
            if meeting.get("racingType") not in ("Thoroughbred", 1):
                continue
            pb_venue = normalize_venue(meeting.get("venue") or "")
            if pb_venue != venue_norm and venue_norm not in pb_venue and pb_venue not in venue_norm:
                continue

            for race in meeting.get("races", []):
                if not isinstance(race, dict):
                    continue
                race_num = race.get("raceNumber") or race.get("number")
                if not race_num:
                    continue
                result_status = race.get("resultStatus", 0)
                if result_status == 2:
                    statuses[int(race_num)] = "Paying"
                else:
                    statuses[int(race_num)] = "Open"
            if statuses:
                break

        return {"statuses": statuses}

    async def _fetch_race_odds(
        self, client: httpx.AsyncClient, event_id: str, race_num: int
    ) -> list[dict]:
        """Fetch runner odds for a single race via v2 API."""
        url = f"{API_BASE}/api/v2/racing/races/{event_id}"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        outcomes = data.get("outcomes", [])
        if not outcomes:
            return []

        odds_list: list[dict] = []
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            parsed = _parse_v2_outcome(outcome, race_num)
            if parsed:
                odds_list.append(parsed)

        return odds_list

    async def scrape_results_for_race(
        self,
        venue: str,
        race_date: date,
        race_number: int,
    ) -> dict:
        """Fetch PointsBet race results (positions + exotic dividends).

        Returns: {"results": [...], "exotics": {"exacta": 78.5, ...}}
        """
        venue_norm = normalize_venue(venue)
        race_ids = await self._find_race_ids(venue_norm, race_date)

        # Find the event ID for the target race number
        event_id = None
        for rn, eid in race_ids:
            if rn == race_number:
                event_id = eid
                break

        if not event_id:
            logger.warning(f"PointsBet: no event ID for {venue} R{race_number}")
            return {"results": []}

        try:
            async with httpx.AsyncClient(headers=_HEADERS, timeout=15.0) as client:
                url = f"{API_BASE}/api/v2/racing/races/{event_id}"
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"PointsBet results fetch failed: {e}")
            return {"results": []}

        results_data = _parse_v2_results(data)
        logger.info(
            f"PointsBet results for {venue} R{race_number}: "
            f"{len(results_data.get('results', []))} runners, "
            f"{len(results_data.get('exotics', {}))} exotics"
        )
        return results_data


def _parse_v2_outcome(outcome: dict, race_num: int) -> dict | None:
    """Parse a v2 API outcome into our standard odds format."""
    name = outcome.get("outcomeName", "")
    if not name or len(name) < 2:
        return None

    # outcomeId is 1-indexed tab number, but barrierBox is the actual saddlecloth
    saddlecloth = outcome.get("outcomeId")
    if not saddlecloth:
        return None

    if outcome.get("scratched", False):
        return None

    # Extract fixed prices
    win_odds = None
    place_odds = None
    opening_odds = None
    for fp in outcome.get("fixedPrices", []):
        if not isinstance(fp, dict):
            continue
        mtype = fp.get("marketTypeCode", "")
        price = fp.get("price")
        if mtype == "WIN" and price:
            win_odds = float(price)
            open_p = fp.get("openPrice")
            if open_p:
                opening_odds = float(open_p)
        elif mtype == "PLC" and price:
            place_odds = float(price)

    # Apply odds ceiling
    if win_odds and win_odds > MAX_VALID_ODDS:
        win_odds = None
    if place_odds and place_odds > MAX_VALID_ODDS:
        place_odds = None
    if opening_odds and opening_odds > MAX_VALID_ODDS:
        opening_odds = None

    if not win_odds:
        return None

    return {
        "race_number": race_num,
        "horse_name": name.strip(),
        "saddlecloth": int(saddlecloth),
        "current_odds": win_odds,
        "opening_odds": opening_odds,
        "place_odds": place_odds,
        "scratched": False,
    }


def _parse_v2_results(data: dict) -> dict:
    """Parse v2 race response for results (post-race).

    PB v2 API structure:
      results.winners[]: {finalPlacing, outcome: {outcomeId, outcomeName, ...}, straightDividends: [...]}
      results.exoticDividends[]: {exoticType, dividends: [{dividendTypeCode, dividend}]}
    """
    results: list[dict] = []
    exotics: dict[str, float] = {}

    results_obj = data.get("results", {})
    if not isinstance(results_obj, dict):
        return {"results": []}

    winners = results_obj.get("winners", [])
    if not winners:
        return {"results": []}

    for winner in winners:
        if not isinstance(winner, dict):
            continue
        position = winner.get("finalPlacing")
        outcome = winner.get("outcome", {})
        if not isinstance(outcome, dict) or not position:
            continue

        oid = outcome.get("outcomeId")
        name = outcome.get("outcomeName", "").strip().title()
        if not oid:
            continue

        # Extract dividends from straightDividends on the winner entry
        win_div = None
        place_div = None
        for sd in winner.get("straightDividends", []):
            if not isinstance(sd, dict):
                continue
            mtype = sd.get("marketType", "")
            div_code = sd.get("dividendTypeCode", "")
            div_val = sd.get("dividend")
            if not div_val:
                continue
            # Prefer BT+SP or VIC dividends (most common for AU racing)
            if mtype == "WIN" and div_code in ("BT+SP", "VIC", "NSW", "MIDDIV"):
                if win_div is None or div_code == "BT+SP":
                    win_div = float(div_val)
            elif mtype == "PLC" and div_code in ("MIDDIV", "BT+SP", "VIC", "NSW"):
                if place_div is None or div_code == "MIDDIV":
                    place_div = float(div_val)

        results.append({
            "horse_name": name,
            "saddlecloth": int(oid),
            "position": int(position),
            "win_dividend": win_div,
            "place_dividend": place_div,
            "margin": None,
        })

    # Extract exotic dividends
    # PB exoticType can be string ("Exacta") or int (enum). Map both.
    _PB_EXOTIC_MAP = {
        "exacta": "exacta", "quinella": "quinella",
        "trifecta": "trifecta", "first4": "first4",
        "firstfour": "first4", "first 4": "first4",
        "quadrella": "quaddie",
    }
    _PB_EXOTIC_INT_MAP = {1: "quinella", 2: "exacta", 3: "trifecta", 4: "first4", 5: "quaddie"}
    for item in results_obj.get("exoticDividends", []):
        if not isinstance(item, dict):
            continue
        raw_type = item.get("exoticType") or item.get("type") or item.get("name") or ""
        if isinstance(raw_type, int):
            canonical = _PB_EXOTIC_INT_MAP.get(raw_type, "")
        else:
            canonical = _PB_EXOTIC_MAP.get(str(raw_type).lower().strip(), str(raw_type).lower().strip())
        if not canonical:
            continue
        # Pick best dividend from jurisdictions (prefer VIC/BT+SP)
        best_div = None
        for d in item.get("dividends", []):
            if isinstance(d, dict):
                dv = d.get("dividend")
                dc = d.get("dividendTypeCode", "")
                if dv:
                    if best_div is None or dc in ("BT+SP", "VIC"):
                        best_div = float(dv)
        # Fallback: dividend directly on item
        if best_div is None:
            dv = item.get("dividend") or item.get("amount")
            if dv:
                best_div = float(dv)
        if canonical and best_div:
            exotics[canonical] = round(best_div, 2)

    result_data: dict = {"results": results}
    if exotics:
        result_data["exotics"] = exotics
    return result_data
