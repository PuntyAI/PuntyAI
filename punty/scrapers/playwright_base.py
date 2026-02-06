"""Shared Playwright browser management for JS-heavy scraping."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

# Module-level singleton for browser reuse
_browser: Optional[Browser] = None
_playwright = None

# Lock to prevent concurrent scrapes (resource-intensive)
_scrape_lock = asyncio.Lock()
_current_scrape: Optional[str] = None  # Track what's being scraped


async def get_browser() -> Browser:
    """Get or launch the shared Chromium browser instance."""
    global _browser, _playwright
    if _browser is None or not _browser.is_connected():
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        logger.info("Playwright browser launched")
    return _browser


async def close_browser() -> None:
    """Close the shared browser instance."""
    global _browser, _playwright
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None
    logger.info("Playwright browser closed")


def is_scrape_in_progress() -> tuple[bool, Optional[str]]:
    """Check if a scrape is currently in progress."""
    return _scrape_lock.locked(), _current_scrape


@asynccontextmanager
async def scrape_lock(meeting_name: str):
    """Acquire exclusive lock for scraping. Prevents concurrent Playwright operations.

    Raises RuntimeError if another scrape is already in progress.
    """
    global _current_scrape

    if _scrape_lock.locked():
        raise RuntimeError(f"Another scrape is in progress: {_current_scrape}. Please wait for it to complete.")

    async with _scrape_lock:
        _current_scrape = meeting_name
        logger.info(f"Acquired scrape lock for {meeting_name}")
        try:
            yield
        finally:
            logger.info(f"Released scrape lock for {meeting_name}")
            _current_scrape = None


@asynccontextmanager
async def new_page(timeout: float = 30000):
    """Async context manager that yields a new browser page.

    Handles page lifecycle: creates page, dismisses common popups,
    and closes page on exit.
    """
    browser = await get_browser()
    context: BrowserContext = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 900},
        locale="en-AU",
    )
    page: Page = await context.new_page()
    page.set_default_timeout(timeout)

    try:
        yield page
    finally:
        await context.close()


async def wait_and_get_content(page: Page, url: str, wait_selector: Optional[str] = None) -> str:
    """Navigate to URL, optionally wait for a selector, dismiss popups, return page HTML."""
    await page.goto(url, wait_until="domcontentloaded")

    # Dismiss common cookie/popup overlays
    for selector in [
        "button:has-text('Accept')",
        "button:has-text('Got it')",
        "button:has-text('OK')",
        "[id*='cookie'] button",
        "[class*='cookie'] button",
    ]:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
        except Exception:
            pass

    if wait_selector:
        try:
            await page.wait_for_selector(wait_selector, timeout=15000)
        except Exception:
            logger.warning(f"Timed out waiting for {wait_selector} on {url}")

    return await page.content()


async def screenshot_debug(page: Page, name: str = "debug") -> None:
    """Save a debug screenshot (useful during development)."""
    from pathlib import Path

    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{name}.png"
    await page.screenshot(path=str(path))
    logger.debug(f"Debug screenshot saved: {path}")
