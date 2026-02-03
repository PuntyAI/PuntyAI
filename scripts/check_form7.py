"""Extract full form text for one horse from the rendered DOM."""
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

        # Get first horse's full form text
        text = await page.evaluate("""() => {
            const item = document.getElementById('id-full-form-item-15453518');
            if (!item) return 'NOT FOUND';
            return item.textContent.substring(0, 3000);
        }""")
        print("=== Horse 1 Full Form Text ===")
        print(text)

        # Also try getting the __NEXT_DATA__ which Next.js apps embed
        next_data = await page.evaluate("""() => {
            const el = document.getElementById('__NEXT_DATA__');
            if (!el) return null;
            try {
                const data = JSON.parse(el.textContent);
                // Find the entries with form history
                const props = data.props?.pageProps;
                if (props) return Object.keys(props);
                return Object.keys(data);
            } catch(e) {
                return 'parse error: ' + e.message;
            }
        }""")
        print(f"\n__NEXT_DATA__ keys: {next_data}")

asyncio.run(main())
