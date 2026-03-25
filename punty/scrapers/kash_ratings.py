"""Fetch KASH model ratings from Betfair Data Scientists API.

KASH provides daily rated prices + speed data for all ANZ thoroughbred runners.
Used as a benchmark/feature: where our LGBM agrees with KASH = less edge,
where we diverge = potential edge or model error.

API: https://betfair-data-supplier-prod.herokuapp.com/api/widgets/kash-ratings-model/datasets
"""

import logging
from datetime import date

import httpx

from punty.config import melb_today
from punty.models.database import async_session
from punty.models.meeting import Runner
from sqlalchemy import select

logger = logging.getLogger(__name__)

KASH_API_URL = (
    "https://betfair-data-supplier-prod.herokuapp.com/api/widgets/"
    "kash-ratings-model/datasets"
)


async def fetch_kash_ratings(race_date: date | None = None) -> list[dict]:
    """Fetch KASH ratings for a given date. Returns list of runner dicts."""
    if race_date is None:
        race_date = melb_today()

    params = {
        "date": race_date.strftime("%Y-%m-%d"),
        "presenter": "RatingsPresenter",
        "json": "true",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(KASH_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"KASH ratings fetch failed for {race_date}: {e}")
        return []

    # Parse the nested JSON structure
    runners = []
    meetings = data if isinstance(data, list) else data.get("meetings", [])
    for meeting in meetings:
        venue = meeting.get("name", "")
        for race in meeting.get("races", []):
            race_num = race.get("number")
            race_speed = race.get("race_speed", "")
            for runner in race.get("runners", []):
                # Name format: "4. Cosmic Gem" — extract saddlecloth + name
                raw_name = runner.get("name", "")
                parts = raw_name.split(". ", 1)
                saddlecloth = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else None
                horse_name = parts[1] if len(parts) == 2 else raw_name

                # Parse numeric fields safely — KASH sometimes returns "Unknown"
                def _safe_float(val):
                    try:
                        return float(val) if val is not None else None
                    except (ValueError, TypeError):
                        return None

                runners.append({
                    "venue": venue,
                    "race_number": race_num,
                    "race_speed": race_speed,
                    "saddlecloth": saddlecloth,
                    "horse_name": horse_name,
                    "rated_price": _safe_float(runner.get("ratedPrice")),
                    "speed_cat": runner.get("speedcat", "") or "",
                    "early_speed": _safe_float(runner.get("early_speed")),
                    "late_speed": _safe_float(runner.get("late_speed")),
                    "bf_selection_id": runner.get("bfExchangeSelectionId"),
                    "bf_market_id": runner.get("bfExchangeMarketId"),
                })

    logger.info(f"KASH ratings: {len(runners)} runners for {race_date} across {len(meetings)} meetings")
    return runners


async def apply_kash_ratings(race_date: date | None = None) -> int:
    """Fetch KASH ratings and store rated_price on Runner records.

    Matches by venue + race_number + saddlecloth. Stores KASH rated price
    in the runner's kash_rated_price field (if the column exists), or logs
    a summary for analysis.

    Returns count of runners matched.
    """
    ratings = await fetch_kash_ratings(race_date)
    if not ratings:
        return 0

    matched = 0
    async with async_session() as db:
        for r in ratings:
            if not r["saddlecloth"] or not r["race_number"]:
                continue

            # Find matching runner by venue pattern + race number + saddlecloth
            # Our race_id format: "venue-YYYY-MM-DD-rN"
            venue_lower = (r["venue"] or "").lower().replace(" ", "-")
            date_str = (race_date or melb_today()).strftime("%Y-%m-%d")
            race_id = f"{venue_lower}-{date_str}-r{r['race_number']}"

            result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.saddlecloth == r["saddlecloth"],
                )
            )
            runner = result.scalar_one_or_none()
            if not runner:
                continue

            # Store KASH data on runner if fields exist
            if hasattr(Runner, "kash_rated_price"):
                runner.kash_rated_price = r["rated_price"]
            if hasattr(Runner, "kash_speed_cat"):
                runner.kash_speed_cat = r["speed_cat"]
            if hasattr(Runner, "kash_early_speed"):
                runner.kash_early_speed = r["early_speed"]
            if hasattr(Runner, "kash_late_speed"):
                runner.kash_late_speed = r["late_speed"]
            matched += 1

        if matched:
            await db.commit()

    logger.info(f"KASH ratings applied: {matched}/{len(ratings)} runners matched")
    return matched
