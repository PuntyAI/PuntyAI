"""Dump all horses from getRaceEntriesForField - find ones with form data."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    entries_data = []
    horse_profiles = {}

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)

                if "getRaceEntriesForField" in response.url:
                    form = data.get("data", {}).get("getRaceForm", {})
                    entries = form.get("formRaceEntries", []) if isinstance(form, dict) else []
                    if entries:
                        entries_data.clear()
                        entries_data.extend(entries)

                if "getHorseProfile" in response.url:
                    hp = data.get("data", {}).get("getHorseProfile", {})
                    if hp and hp.get("id"):
                        horse_profiles[hp["id"]] = hp
            except Exception:
                pass

        page.on("response", capture)
        # Use race 3 instead - more likely to have experienced horses
        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/3",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(6000)
        except Exception:
            pass

        page.remove_listener("response", capture)

    # Find horses with actual form
    for e in entries_data[:3]:
        name = e.get("horseName", "")
        horse = e.get("horse", {}) or {}
        horse_id = horse.get("id")
        lf = horse.get("lastFive")
        last_pro = horse.get("lastProfessionalRaceEntryItem")
        print(f"\n=== {name} (id={horse_id}) ===")
        print(f"  lastFive: {lf}")
        print(f"  lastTenStats: {horse.get('lastTenStats')}")
        print(f"  firstUpStats: {horse.get('firstUpStats')}")
        print(f"  lastProfessionalRaceEntryItem: {json.dumps(last_pro)[:500] if last_pro else None}")

        # Check horse profile
        if horse_id and horse_id in horse_profiles:
            hp = horse_profiles[horse_id]
            print(f"  Profile keys: {list(hp.keys())}")
            print(f"  Profile full: {json.dumps(hp)[:1000]}")

    # Also dump what the "Full Form" section actually shows via DOM
    print("\n\n--- Done ---")
    print(f"Total entries: {len(entries_data)}, profiles: {len(horse_profiles)}")

asyncio.run(main())
