"""Capture ALL network requests (not just GraphQL) when Full Form is loaded."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    api_calls = []

    async with new_page() as page:
        async def capture_all(response):
            url = response.url
            # Skip static assets
            if any(x in url for x in ['.css', '.js', '.png', '.svg', '.woff', '.gif', 'favicon',
                                       'google', 'facebook', 'analytics', 'doubleclick', 'amazon',
                                       'brightcove', 'bam-cell']):
                return
            ct = response.headers.get("content-type", "")
            if "json" in ct or "graphql" in url:
                try:
                    body = await response.text()
                    api_calls.append((url[:120], body[:2000]))
                except Exception:
                    api_calls.append((url[:120], "ERROR reading body"))

        page.on("response", capture_all)
        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/3",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        # Note how many calls before Full Form click
        before = len(api_calls)

        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(6000)
        except Exception:
            pass

        # Scroll down to trigger lazy loading
        for _ in range(15):
            await page.evaluate("window.scrollBy(0, 600)")
            await page.wait_for_timeout(300)
        await page.wait_for_timeout(3000)

        page.remove_listener("response", capture_all)

    print(f"Total API calls: {len(api_calls)}, before Full Form: {before}")
    print("\n=== Calls AFTER Full Form click ===")
    for url, body in api_calls[before:]:
        # Check if body contains form-like data
        has_form = any(k in body for k in ["raceDate", "margin", "settledPosition", "startingPrice",
                                            "pastStart", "raceHistory", "formHistory"])
        print(f"\n{'*** ' if has_form else ''}URL: {url}")
        if has_form:
            print(f"  Body: {body[:1000]}")
        elif "graphql" in url:
            # Show first 200 chars for graphql
            print(f"  Body: {body[:200]}")

asyncio.run(main())
