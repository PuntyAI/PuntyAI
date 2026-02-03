"""Dump full structure of getRaceEntryItemByHorsePaged for one horse."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    form_entries = {}  # horse_code -> list of past starts

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            if "getRaceEntryItemByHorsePaged" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                items = data.get("data", {}).get("GetRaceEntryItemByHorsePaged", [])
                if items:
                    horse_name = items[0].get("horseName", "unknown")
                    form_entries.setdefault(horse_name, []).extend(items)
            except Exception:
                pass

        page.on("response", capture)
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

        # Scroll to trigger lazy loading
        for _ in range(25):
            await page.evaluate("window.scrollBy(0, 600)")
            await page.wait_for_timeout(400)
        await page.wait_for_timeout(3000)

        page.remove_listener("response", capture)

    # Dump one horse's full form history
    for horse_name, items in form_entries.items():
        if len(items) >= 3:
            print(f"=== {horse_name} ({len(items)} starts) ===")
            # Full dump of first item
            print(json.dumps(items[0], indent=2))
            print(f"\n--- All starts summary ---")
            for item in items:
                race = item.get("race", {})
                print(f"  {race.get('date')} {race.get('location')} {race.get('distance')} "
                      f"{item.get('finishAbv')} class={item.get('rdcClass')}")
            break

    print(f"\nTotal horses with form: {len(form_entries)}")
    for name, items in form_entries.items():
        print(f"  {name}: {len(items)} starts")

asyncio.run(main())
