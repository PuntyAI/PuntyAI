"""Download Betfair historical data files for ANZ Thoroughbreds, BSP prices, and KASH model results."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent / "betfair_data"

ANZ_DIR = BASE_DIR / "anz_thoroughbreds"
BSP_DIR = BASE_DIR / "bsp_prices"
KASH_DIR = BASE_DIR / "kash_results"

ANZ_URL = "https://betfair-datascientists.github.io/data/assets/ANZ_Thoroughbreds_{year}_{month:02d}.csv"
BSP_WIN_URL = "https://promo.betfair.com/betfairsp/prices/dwbfpricesauswin{day:02d}{month:02d}{year}.csv"
BSP_PLACE_URL = "https://promo.betfair.com/betfairsp/prices/dwbfpricesausplace{day:02d}{month:02d}{year}.csv"
KASH_URL = "https://betfair-datascientists.github.io/data/assets/Kash_Model_Results_{year}.csv"


async def download_file(client: httpx.AsyncClient, url: str, dest: Path) -> bool:
    """Download a single file. Returns True if downloaded, False if skipped/failed."""
    if dest.exists():
        print(f"  SKIP (exists): {dest.name}")
        return False

    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            print(f"  404 (not found): {dest.name}")
            return False
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  OK ({size_kb:.1f} KB): {dest.name}")
        return True
    except httpx.HTTPStatusError as e:
        print(f"  ERROR ({e.response.status_code}): {dest.name}")
        return False
    except httpx.RequestError as e:
        print(f"  ERROR ({type(e).__name__}): {dest.name} - {e}")
        return False


async def download_anz_thoroughbreds(client: httpx.AsyncClient) -> int:
    """Download ANZ Thoroughbred CSVs for 2025 (all months) and 2026 (Jan-Mar)."""
    print("\n=== ANZ Thoroughbreds ===")
    ANZ_DIR.mkdir(parents=True, exist_ok=True)

    months = [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 4)]
    count = 0
    for year, month in months:
        url = ANZ_URL.format(year=year, month=month)
        dest = ANZ_DIR / f"ANZ_Thoroughbreds_{year}_{month:02d}.csv"
        if await download_file(client, url, dest):
            count += 1
    return count


async def download_bsp_prices(client: httpx.AsyncClient) -> int:
    """Download BSP win and place price files for the last 14 days."""
    print("\n=== BSP Prices (last 14 days) ===")
    BSP_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().date()
    count = 0
    for days_ago in range(14, 0, -1):
        d = today - timedelta(days=days_ago)
        for label, url_tmpl in [("win", BSP_WIN_URL), ("place", BSP_PLACE_URL)]:
            url = url_tmpl.format(day=d.day, month=d.month, year=d.year)
            fname = f"dwbfpricesaus{label}{d.day:02d}{d.month:02d}{d.year}.csv"
            dest = BSP_DIR / fname
            if await download_file(client, url, dest):
                count += 1
    return count


async def download_kash_results(client: httpx.AsyncClient) -> int:
    """Download KASH model results for 2025 and 2026."""
    print("\n=== KASH Model Results ===")
    KASH_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for year in (2025, 2026):
        url = KASH_URL.format(year=year)
        dest = KASH_DIR / f"Kash_Model_Results_{year}.csv"
        if await download_file(client, url, dest):
            count += 1
    return count


async def main():
    print(f"Betfair data download — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Output directory: {BASE_DIR}")

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        anz = await download_anz_thoroughbreds(client)
        bsp = await download_bsp_prices(client)
        kash = await download_kash_results(client)

    total = anz + bsp + kash
    print(f"\nDone. Downloaded {total} new file(s).")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
