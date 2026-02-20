"""Calendar scraper — discovers today's meetings via PuntingForm API.

Primary: PuntingForm get_meetings() API (fast, accurate venue names).
Fallback: racing.com Playwright scrape (if PF fails).
"""

import logging
from datetime import date
from typing import Any

from punty.config import melb_today
from punty.venues import normalize_venue, guess_state

logger = logging.getLogger(__name__)


async def scrape_calendar(race_date: date | None = None) -> list[dict[str, Any]]:
    """Get meetings for a date using PuntingForm API, with racing.com fallback.

    Returns a list of dicts with: venue, state, num_races, race_type, status, date, meeting_type.
    """
    race_date = race_date or melb_today()

    # Primary: PuntingForm API
    try:
        meetings = await _scrape_calendar_pf(race_date)
        if meetings:
            # Validate any unknown venues via AI
            from punty.venues_validator import validate_calendar_venues
            meetings = await validate_calendar_venues(meetings)
            logger.info(f"PF calendar: {len(meetings)} meetings for {race_date}")
            return meetings
        logger.warning(f"PF calendar returned 0 meetings for {race_date}, trying racing.com")
    except Exception as e:
        logger.warning(f"PF calendar failed: {e}, falling back to racing.com")

    # Fallback: racing.com Playwright
    try:
        meetings = await _scrape_calendar_racing_com(race_date)
        logger.info(f"Racing.com calendar fallback: {len(meetings)} meetings for {race_date}")
        return meetings
    except Exception as e:
        logger.error(f"Racing.com calendar also failed: {e}")
        return []


async def _scrape_calendar_pf(race_date: date) -> list[dict[str, Any]]:
    """Scrape calendar via PuntingForm get_meetings() API."""
    from punty.scrapers.punting_form import PuntingFormScraper

    from punty.models.database import async_session

    async with async_session() as db:
        scraper = await PuntingFormScraper.from_settings(db)
    try:
        pf_meetings = await scraper.get_meetings(race_date)
    finally:
        await scraper.close()

    meetings: list[dict[str, Any]] = []
    seen: set[str] = set()

    for m in pf_meetings:
        track = m.get("track", {})
        venue_raw = track.get("name", "")
        if not venue_raw:
            continue

        # Skip barrier trials
        if m.get("isBarrierTrial"):
            logger.debug(f"Skipping barrier trial: {venue_raw}")
            continue

        venue = normalize_venue(venue_raw)
        if not venue:
            continue

        # Deduplicate
        key = venue.lower()
        if key in seen:
            continue
        seen.add(key)

        state = track.get("state", "") or guess_state(venue)

        meetings.append({
            "venue": venue_raw,  # Keep original case for display
            "state": state,
            "num_races": 0,
            "race_type": "Thoroughbred",
            "status": "",
            "date": race_date,
            "meeting_type": "race",
        })

    return meetings


async def _scrape_calendar_racing_com(race_date: date) -> list[dict[str, Any]]:
    """Fallback: scrape racing.com/calendar via Playwright."""
    from punty.scrapers.playwright_base import new_page

    CALENDAR_URL = "https://www.racing.com/calendar"
    logger.info(f"Scraping racing.com calendar for {race_date}")

    meetings: list[dict[str, Any]] = []

    async with new_page() as page:
        await page.goto(CALENDAR_URL, wait_until="load")

        try:
            await page.locator(".calendar__grid-item-container").first.wait_for(timeout=15000)
        except Exception:
            logger.warning("Could not find calendar grid on page")
            return meetings

        # Dismiss cookie banner
        for sel in ["button:has-text('Accept')", "button:has-text('Decline')"]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=1500):
                    await btn.click()
            except Exception:
                pass

        containers = page.locator(".calendar__grid-item-container")
        container_count = await containers.count()
        target_day = race_date.day
        target_container = None
        target_container_index = 0

        for ci in range(container_count):
            c = containers.nth(ci)
            day_elem = c.locator(".calendar__grid-item-day").first
            try:
                day_text = (await day_elem.text_content(timeout=2000) or "").strip()
                day_num = None
                for part in day_text.split():
                    if part.isdigit():
                        day_num = int(part)
                        break
                if day_num is not None and day_num == target_day:
                    target_container = c
                    target_container_index = ci
                    break
            except Exception:
                continue

        # Fallback to --today CSS class
        if target_container is None and race_date == melb_today():
            for ci in range(container_count):
                c = containers.nth(ci)
                if await c.locator(".calendar__grid-item-day--today").count() > 0:
                    target_container = c
                    target_container_index = ci
                    break

        if target_container is None:
            logger.warning(f"Could not locate column for {race_date}")
            return meetings

        # Click "Show more" if present
        show_more = target_container.locator(".calendar__grid-item-list-show-more")
        try:
            if await show_more.is_visible(timeout=2000):
                await show_more.click()
                await page.wait_for_timeout(1000)
        except Exception:
            pass

        meetings_data = await page.evaluate("""(containerIndex) => {
            const containers = document.querySelectorAll('.calendar__grid-item-container');
            const container = containers[containerIndex];
            if (!container) return [];
            const results = [];
            const btns = container.querySelectorAll('.calendar__grid-item-btn');
            btns.forEach(btn => {
                const span = btn.querySelector('span');
                const venue = span?.textContent?.trim() || '';
                if (!venue) return;
                const right = btn.querySelector('.calendar__grid-item-right');
                let state = '';
                if (right) {
                    state = (right.textContent || '').replace(/[^a-zA-Z]/g, '').toUpperCase();
                }
                const abbr = btn.querySelector('abbr');
                const status = abbr?.getAttribute('title') || '';
                results.push({ venue, state, status });
            });
            return results;
        }""", target_container_index)

        for m in meetings_data:
            venue = m.get("venue", "")
            if not venue:
                continue
            meetings.append({
                "venue": venue,
                "state": m.get("state", ""),
                "num_races": 0,
                "race_type": "Thoroughbred",
                "status": m.get("status", ""),
                "date": race_date,
                "meeting_type": "race",
            })

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for m in meetings:
        key = m["venue"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique
