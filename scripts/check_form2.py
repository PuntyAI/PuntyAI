"""Inspect getHorseProfile GraphQL response."""
import asyncio
import json
import sys
sys.path.insert(0, "/opt/puntyai")

async def main():
    from punty.scrapers.playwright_base import new_page

    profiles = []

    async with new_page() as page:
        async def capture(response):
            if "graphql.rmdprod.racing.com" not in response.url:
                return
            if "getHorseProfile" not in response.url:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                profiles.append(data)
            except Exception:
                pass

        page.on("response", capture)
        await page.goto(
            "https://www.racing.com/form/2026-02-03/bet365-seymour/race/1",
            wait_until="load",
        )
        await page.wait_for_timeout(5000)

        try:
            btn = page.locator("text=Full Form").first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                await page.wait_for_timeout(6000)
        except Exception as e:
            print("Click error:", e)

        page.remove_listener("response", capture)

    if not profiles:
        print("No horse profiles captured!")
        return

    # Inspect first profile
    p = profiles[0]
    hp = p.get("data", {}).get("getHorseProfile", {})
    if not hp:
        print("No getHorseProfile data")
        print("Keys:", list(p.get("data", {}).keys()))
        return

    print("getHorseProfile top keys:", list(hp.keys()))
    for k, v in hp.items():
        if isinstance(v, list) and v:
            if isinstance(v[0], dict):
                print(f"\n  {k} ({len(v)} items): keys={list(v[0].keys())}")
                print(f"    first: {json.dumps(v[0])[:600]}")
            else:
                print(f"\n  {k} ({len(v)} items): first={v[0]}")
        elif isinstance(v, dict):
            print(f"\n  {k}: keys={list(v.keys())}")
        else:
            val_str = str(v)[:100]
            print(f"\n  {k}: {val_str}")

asyncio.run(main())
