"""Dump ALL fields from a non-debutant entry to find form history."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    entries_data = []

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            if "getRaceEntriesForField" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                form = data.get("data", {}).get("getRaceForm", {})
                entries = form.get("formRaceEntries", []) if isinstance(form, dict) else []
                if entries:
                    entries_data.clear()
                    entries_data.extend(entries)
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
                await page.wait_for_timeout(5000)
        except Exception:
            pass

        page.remove_listener("response", capture)

    # Find Veins Within Rock (has form) and dump EVERYTHING
    for e in entries_data:
        if "Veins" in e.get("horseName", ""):
            # Full JSON dump
            full = json.dumps(e, indent=2)
            print(f"=== {e['horseName']} - FULL ENTRY ({len(full)} chars) ===")
            print(full[:8000])
            break
    else:
        # Just dump first entry with lastFive
        for e in entries_data:
            horse = e.get("horse", {}) or {}
            if horse.get("lastFive"):
                full = json.dumps(e, indent=2)
                print(f"=== {e['horseName']} - FULL ENTRY ({len(full)} chars) ===")
                print(full[:8000])
                break

asyncio.run(main())
