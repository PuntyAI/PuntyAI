"""Inspect GraphQL full form data structure from racing.com."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    captured = []

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                qname = response.url.split("/")[-1][:80]
                captured.append((qname, data))
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
        except Exception as e:
            print("Click error:", e)

        page.remove_listener("response", capture)

    for qname, data in captured:
        print(f"\n=== {qname} ===")
        d = data.get("data", {})
        for key, val in d.items():
            if isinstance(val, dict):
                entries = val.get("formRaceEntries", [])
                if entries:
                    e = entries[0]
                    ek = list(e.keys())
                    print(f"  Entry keys ({len(entries)} entries): {ek}")
                    horse = e.get("horse", {}) or {}
                    if horse:
                        print(f"  Horse keys: {list(horse.keys())}")
                    # Check for nested list/dict values (form history)
                    for k, v in e.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            print(f"  e.{k} ({len(v)} items): keys={list(v[0].keys())}")
                            print(f"    first: {json.dumps(v[0])[:500]}")
                    for k, v in (horse or {}).items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            print(f"  horse.{k} ({len(v)} items): keys={list(v[0].keys())}")
                            print(f"    first: {json.dumps(v[0])[:500]}")

asyncio.run(main())
