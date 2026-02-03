"""Scroll to horse and capture any lazy-loaded GraphQL for past starts."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    post_scroll = []

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                post_scroll.append((response.url, data))
            except Exception:
                pass

        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/3#id-full-form-item-15453518",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(3000)
        except Exception:
            pass

        # Clear previous captures, only interested in new ones
        post_scroll.clear()
        page.on("response", capture)

        # Scroll to horse 1 and interact
        await page.evaluate("""() => {
            const el = document.getElementById('id-full-form-item-15453518');
            if (el) el.scrollIntoView();
        }""")
        await page.wait_for_timeout(3000)

        # Try clicking on the horse name/number to expand
        try:
            horse_link = page.locator("#id-full-form-item-15453518 strong").first
            if await horse_link.is_visible(timeout=1000):
                await horse_link.click()
                await page.wait_for_timeout(3000)
        except Exception:
            pass

        # Scroll down through the page to trigger any lazy loading
        for _ in range(10):
            await page.evaluate("window.scrollBy(0, 500)")
            await page.wait_for_timeout(500)

        await page.wait_for_timeout(3000)

        # Check the full page text for any past race data patterns
        has_past = await page.evaluate("""() => {
            const text = document.body.textContent;
            // Look for patterns like "15 Jan 2026" or "Seymour 1300m"
            const datePattern = /\\d{1,2}\\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{4}/;
            const matches = text.match(new RegExp(datePattern.source, 'g'));
            return {
                dateMatches: matches ? matches.slice(0, 10) : [],
                hasMarginText: text.includes('margin'),
                hasSP: text.includes('SP:'),
                bodyLen: text.length
            };
        }""")
        print(f"Page analysis: {json.dumps(has_past, indent=2)}")

        page.remove_listener("response", capture)

    print(f"\nNew GraphQL responses after scroll: {len(post_scroll)}")
    for url, data in post_scroll:
        qname = "getHorseProfile" if "getHorseProfile" in url else url.split("query=")[-1][:50] if "query=" in url else url[-50:]
        d = data.get("data", {})
        for key, val in d.items():
            if isinstance(val, dict):
                ks = list(val.keys())
                print(f"  {qname} -> {key}: {ks[:15]}")
                for k2, v2 in val.items():
                    if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                        print(f"    {k2} ({len(v2)} items): keys={list(v2[0].keys())}")
                        print(f"      first: {json.dumps(v2[0])[:500]}")

asyncio.run(main())
