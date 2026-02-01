"""Scrape racing.com/calendar for today's meetings list."""

import logging
from datetime import date

from punty.config import melb_today
from typing import Any

from punty.scrapers.playwright_base import new_page

logger = logging.getLogger(__name__)

CALENDAR_URL = "https://www.racing.com/calendar"


async def scrape_calendar(race_date: date | None = None) -> list[dict[str, Any]]:
    """Scrape racing.com calendar page for today's meetings.

    Uses Playwright to render the JS-heavy Next.js calendar page, locates
    today's column, clicks "Show more" if present, then extracts meeting
    data using Playwright locators (avoids page.content() / BS4 issues
    with React-hydrated classes).

    Returns a list of dicts with: venue, state, num_races, race_type, status, date.
    """
    race_date = race_date or melb_today()
    logger.info(f"Scraping calendar for {race_date}: {CALENDAR_URL}")

    meetings: list[dict[str, Any]] = []

    async with new_page() as page:
        await page.goto(CALENDAR_URL, wait_until="load")

        # Wait for the calendar grid to render
        today_locator = page.locator(".calendar__grid-item-day--today")
        try:
            await today_locator.wait_for(timeout=30000)
        except Exception:
            logger.warning("Could not find today's cell on calendar page")
            return meetings

        # Dismiss cookie banner
        for sel in [
            "button:has-text('Accept')",
            "button:has-text('Decline')",
        ]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=1500):
                    await btn.click()
            except Exception:
                pass

        # Find the container that holds today's cell by checking each
        # grid-item-container for the --today element inside it
        containers = page.locator(".calendar__grid-item-container")
        container_count = await containers.count()
        today_container = None
        for ci in range(container_count):
            c = containers.nth(ci)
            today_inside = c.locator(".calendar__grid-item-day--today")
            if await today_inside.count() > 0:
                today_container = c
                break

        if today_container is None:
            logger.warning("Could not locate today's container among grid items")
            return meetings

        # Click "Show more" if present
        show_more = today_container.locator(".calendar__grid-item-list-show-more")
        try:
            if await show_more.is_visible(timeout=2000):
                await show_more.click()
                await page.wait_for_timeout(1000)
        except Exception:
            pass

        # Extract meetings using Playwright locators
        btns = today_container.locator(".calendar__grid-item-btn")
        count = await btns.count()
        logger.info(f"Found {count} meeting buttons in today's column")

        for i in range(count):
            btn = btns.nth(i)
            try:
                span = btn.locator("span").first
                venue = (await span.text_content() or "").strip()
                if not venue:
                    continue

                right = btn.locator(".calendar__grid-item-right").first
                state = ""
                try:
                    raw_state = (await right.text_content() or "").strip()
                    state = "".join(c for c in raw_state if c.isalpha()).upper()
                except Exception:
                    pass

                abbr = btn.locator("abbr").first
                status = ""
                try:
                    status = await abbr.get_attribute("title") or ""
                except Exception:
                    pass

                meetings.append({
                    "venue": venue,
                    "state": state,
                    "num_races": 0,
                    "race_type": "Thoroughbred",
                    "status": status,
                    "date": race_date,
                })
            except Exception as e:
                logger.debug(f"Error extracting meeting button {i}: {e}")

    # Deduplicate by venue
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for m in meetings:
        key = m["venue"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(m)
    meetings = unique

    logger.info(f"Found {len(meetings)} meetings for {race_date}")
    return meetings
