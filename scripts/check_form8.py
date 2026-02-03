"""Get the full innerHTML of the first horse's form item to find past starts."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    async with new_page() as page:
        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/3",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(5000)
        except Exception:
            pass

        # Get ALL text content from the full form, much longer
        text = await page.evaluate("""() => {
            const item = document.getElementById('id-full-form-item-15453518');
            if (!item) return 'NOT FOUND';
            return item.textContent;
        }""")
        print(f"Total text length: {len(text)}")
        print(text[:5000])

asyncio.run(main())
