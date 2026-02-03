"""Final attempt - get full body around 'margin' text and inspect post-scroll responses."""
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
                post_scroll.append((response.url, body[:5000]))
            except Exception:
                pass

        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/3",
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

        post_scroll.clear()
        page.on("response", capture)

        # Scroll all the way down
        for _ in range(20):
            await page.evaluate("window.scrollBy(0, 800)")
            await page.wait_for_timeout(300)
        await page.wait_for_timeout(4000)

        page.remove_listener("response", capture)

        # Extract text around "margin" keyword
        result = await page.evaluate("""() => {
            const text = document.body.innerText;
            const idx = text.indexOf('Margin');
            if (idx === -1) return {found: false};
            return {found: true, context: text.substring(Math.max(0, idx-200), idx+1000)};
        }""")
        if result.get("found"):
            print("=== Text around 'Margin' ===")
            print(result["context"][:2000])

        # Also check what full form item looks like after scrolling
        form_text = await page.evaluate("""() => {
            const item = document.getElementById('id-full-form-item-15453518');
            if (!item) return 'NOT FOUND';
            return item.innerText;
        }""")
        print(f"\n=== Horse 1 innerText after scroll ({len(form_text)} chars) ===")
        print(form_text[:3000])

    print(f"\nPost-scroll responses: {len(post_scroll)}")
    for url, body in post_scroll:
        qname = url.split("query=")[-1][:60] if "query=" in url else url[-60:]
        print(f"\n  Query: {qname}")
        print(f"  Body: {body[:800]}")

asyncio.run(main())
