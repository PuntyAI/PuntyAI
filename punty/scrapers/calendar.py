"""Scrape racing.com/calendar for meetings on a specific date."""

import logging
from datetime import date, timedelta

from punty.config import melb_today
from typing import Any

from punty.scrapers.playwright_base import new_page

logger = logging.getLogger(__name__)

CALENDAR_URL = "https://www.racing.com/calendar"




async def scrape_calendar(race_date: date | None = None) -> list[dict[str, Any]]:
    """Scrape racing.com calendar page for meetings on a specific date.

    Uses Playwright to render the JS-heavy Next.js calendar page, finds the
    column matching the requested date by checking the displayed day number
    (not relying on CSS "today" class which uses racing.com's server timezone),
    then extracts meeting data.

    Returns a list of dicts with: venue, state, num_races, race_type, status, date.
    """
    race_date = race_date or melb_today()
    logger.info(f"Scraping calendar for {race_date}: {CALENDAR_URL}")

    meetings: list[dict[str, Any]] = []

    async with new_page() as page:
        await page.goto(CALENDAR_URL, wait_until="load")

        # Wait for the calendar grid to render
        try:
            await page.locator(".calendar__grid-item-container").first.wait_for(timeout=15000)
        except Exception:
            logger.warning("Could not find calendar grid on page")
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

        # Find the container matching our target date
        # The calendar shows a week, with day numbers in .calendar__grid-item-day
        containers = page.locator(".calendar__grid-item-container")
        container_count = await containers.count()

        target_day = race_date.day
        target_container = None
        target_container_index = 0
        found_date_info = None

        logger.debug(f"Looking for day {target_day} in {container_count} containers")

        for ci in range(container_count):
            c = containers.nth(ci)
            # Each container has a day element with the date number
            day_elem = c.locator(".calendar__grid-item-day").first
            try:
                day_text = (await day_elem.text_content() or "").strip()
                # Day text is usually just the number (e.g., "3" for Feb 3)
                # Sometimes it might have additional text
                day_num = None
                for part in day_text.split():
                    if part.isdigit():
                        day_num = int(part)
                        break

                if day_num is not None and day_num == target_day:
                    # Verify this is the right month by checking we're in the right range
                    # The calendar typically shows ~7 days, so if our target is within
                    # a reasonable range of today, we're good
                    target_container = c
                    target_container_index = ci
                    found_date_info = f"day {day_num}"
                    logger.info(f"Found target date column: {found_date_info}")
                    break
            except Exception as e:
                logger.debug(f"Error reading day from container {ci}: {e}")
                continue

        # If we couldn't find by day number, try the --today class as fallback
        # but only if the requested date IS actually today
        if target_container is None and race_date == melb_today():
            logger.info("Falling back to CSS --today class for today's date")
            for ci in range(container_count):
                c = containers.nth(ci)
                today_inside = c.locator(".calendar__grid-item-day--today")
                if await today_inside.count() > 0:
                    target_container = c
                    target_container_index = ci
                    found_date_info = "CSS --today class"
                    break

        if target_container is None:
            logger.warning(f"Could not locate column for {race_date} (day {target_day}) among {container_count} grid items")
            # Log what days we did find for debugging
            try:
                for ci in range(min(container_count, 7)):
                    c = containers.nth(ci)
                    day_elem = c.locator(".calendar__grid-item-day").first
                    day_text = (await day_elem.text_content() or "").strip()
                    logger.debug(f"  Container {ci}: day text = '{day_text}'")
            except Exception:
                pass
            return meetings

        # Click "Show more" if present
        show_more = target_container.locator(".calendar__grid-item-list-show-more")
        try:
            if await show_more.is_visible(timeout=2000):
                await show_more.click()
                await page.wait_for_timeout(1000)
        except Exception:
            pass

        # Extract meetings, detecting races vs trials/jumpouts
        # The abbr element contains a code: F = Full form (race), W = Working (trial/jumpout)
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
                // The abbr text content indicates type: F = race, W = trial/jumpout
                const abbrCode = abbr?.textContent?.trim().toUpperCase() || 'F';
                const isTrialOrJumpout = abbrCode === 'W';

                results.push({
                    venue: venue,
                    state: state,
                    status: status,
                    abbrCode: abbrCode,
                    isTrialOrJumpout: isTrialOrJumpout
                });
            });

            return results;
        }""", target_container_index)

        # Log what we found
        races = [m for m in meetings_data if not m.get("isTrialOrJumpout")]
        trials = [m for m in meetings_data if m.get("isTrialOrJumpout")]
        logger.info(f"Found {len(meetings_data)} meetings: {len(races)} races, {len(trials)} trials")
        for m in trials:
            logger.info(f"  Trial detected: {m.get('venue')} (code={m.get('abbrCode')})")

        for m in meetings_data:
            venue = m.get("venue", "")
            if not venue:
                continue

            meeting_type = "trial" if m.get("isTrialOrJumpout") else "race"

            meetings.append({
                "venue": venue,
                "state": m.get("state", ""),
                "num_races": 0,
                "race_type": "Thoroughbred",
                "status": m.get("status", ""),
                "date": race_date,
                "meeting_type": meeting_type,
            })

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
