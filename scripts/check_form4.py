"""Check if full form history is in a separate GraphQL call when expanding a horse."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    pre_click = []
    post_click = []

    async with new_page() as page:
        phase = ["pre"]

        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                target = pre_click if phase[0] == "pre" else post_click
                target.append((response.url[-80:], data))
            except Exception:
                pass

        page.on("response", capture)
        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/1",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        # Click Full Form tab
        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(4000)
                print("Clicked Full Form tab")
        except Exception as e:
            print("Click error:", e)

        phase[0] = "post"

        # Now click on a specific horse's expand button to see full form
        # The URL fragment #id-full-form-item-15453692 suggests a clickable element
        # Try clicking the second horse (which has race history, not a debutant)
        try:
            # Look for expandable form items
            items = page.locator("[class*='form-item'], [class*='fullForm'], [data-testid*='form'], details, [class*='expand']")
            count = await items.count()
            print(f"Found {count} expandable items")

            # Try clicking horse name links or expand buttons
            expand_btns = page.locator("button[class*='expand'], button[class*='toggle'], [class*='accordion']")
            ec = await expand_btns.count()
            print(f"Found {ec} expand buttons")

            # Try a different approach - look for the horse entry rows
            horse_rows = page.locator("[class*='runner'], [class*='horse'], [class*='entry']")
            hc = await horse_rows.count()
            print(f"Found {hc} horse/runner/entry elements")

            # Click on a non-debutant horse to expand form
            # Horse ID 15453692 from URL - try clicking related element
            horse_el = page.locator("#id-full-form-item-15453692, [id*='15453692'], [data-id='15453692']")
            hec = await horse_el.count()
            print(f"Horse 15453692 elements: {hec}")

            # Try just getting all visible text about past starts
            form_text = await page.evaluate("""() => {
                const els = document.querySelectorAll('[class*="form"], [class*="history"], [class*="past"]');
                return Array.from(els).map(e => e.className + ': ' + e.textContent.substring(0, 200)).slice(0, 10);
            }""")
            for t in form_text:
                print(f"  {t[:200]}")

        except Exception as e:
            print(f"Expand error: {e}")

        await page.wait_for_timeout(3000)
        page.remove_listener("response", capture)

    print(f"\nPost-click responses: {len(post_click)}")
    for url, data in post_click:
        qname = url.split("query=")[-1][:50] if "query=" in url else url[:50]
        print(f"  {qname}")
        d = data.get("data", {})
        for key, val in d.items():
            if isinstance(val, dict):
                print(f"    {key}: keys={list(val.keys())[:15]}")
                # Look for any list that could be past starts
                for k2, v2 in val.items():
                    if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                        print(f"      {k2} ({len(v2)}): keys={list(v2[0].keys())}")
                        print(f"        first: {json.dumps(v2[0])[:400]}")

asyncio.run(main())
