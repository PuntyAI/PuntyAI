"""Dump full horse object and lastProfessionalRaceEntryItem from getRaceEntriesForField."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    entries_data = []
    all_responses = []

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                qname = response.url.split("query=")[1][:60] if "query=" in response.url else response.url[-60:]
                all_responses.append((qname, data))

                # Capture entries from getRaceEntriesForField
                if "getRaceEntriesForField" in response.url:
                    form = data.get("data", {}).get("getRaceForm", {})
                    entries = form.get("formRaceEntries", []) if isinstance(form, dict) else []
                    if entries:
                        entries_data.clear()
                        entries_data.extend(entries)
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
                await page.wait_for_timeout(6000)
                print("Clicked Full Form")
        except Exception as e:
            print("Click error:", e)

        page.remove_listener("response", capture)

    # Dump the first horse's complete data
    if entries_data:
        e = entries_data[0]
        horse = e.get("horse", {}) or {}
        print(f"\n=== Horse: {e.get('horseName')} ===")

        # lastProfessionalRaceEntryItem
        lp = horse.get("lastProfessionalRaceEntryItem")
        if lp:
            print(f"\nlastProfessionalRaceEntryItem: {json.dumps(lp)[:1000]}")

        # Print full horse object
        print(f"\nFull horse object:")
        print(json.dumps(horse, indent=2)[:3000])

    # Check for any response that arrived AFTER Full Form click that contains form history
    print(f"\n\nTotal responses captured: {len(all_responses)}")
    for qname, data in all_responses:
        d = data.get("data", {})
        for key, val in d.items():
            if key == "getHorseProfile":
                continue
            if isinstance(val, (dict, list)):
                s = json.dumps(val)
                if "raceDate" in s or "trackName" in s or "pastStart" in s:
                    print(f"\nFOUND form-like data in {qname[:40]}.{key}")
                    print(s[:1000])

asyncio.run(main())
