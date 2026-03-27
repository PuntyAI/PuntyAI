"""Download Betfair historical greyhound data and Iggy-Joey model predictions.

Downloads to scripts/_greyhound_data/:
  - ANZ_Greyhounds_{YYYY}.zip — yearly Betfair market data (2020-2025)
  - ANZ_Greyhounds_{YYYY}_{MM}.csv — monthly Betfair data (2026+)
  - Iggy_Model_V2_Results_{YYYY}.csv — Iggy-Joey V2 model predictions

Source: https://betfair-datascientists.github.io/data/dataListing/

Usage:
    python scripts/download_greyhound_data.py
    python scripts/download_greyhound_data.py --years 2025 2026
    python scripts/download_greyhound_data.py --iggy-only
"""

import argparse
import sys
import zipfile
from pathlib import Path

import httpx

ASSETS_BASE = "https://betfair-datascientists.github.io/data/assets"
DEST_DIR = Path(__file__).parent / "_greyhound_data"


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file if it doesn't already exist."""
    if dest.exists() and not force:
        print(f"  [SKIP] {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return True

    print(f"  [GET]  {url}")
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                print(f"  [404]  Not found: {url}")
                return False
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  [OK]   {dest.name} ({len(resp.content):,} bytes)")
            return True
    except Exception as e:
        print(f"  [ERR]  {e}")
        return False


def extract_zip(zip_path: Path) -> None:
    """Extract a zip file into the same directory."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        zf.extractall(zip_path.parent)
        print(f"  [UNZIP] Extracted {len(names)} files from {zip_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Download Betfair greyhound data")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025],
                        help="Years to download (default: 2024 2025)")
    parser.add_argument("--iggy-only", action="store_true",
                        help="Only download Iggy model predictions")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    parser.add_argument("--extract", action="store_true", default=True,
                        help="Extract ZIP files after download")
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Download directory: {DEST_DIR}")

    if not args.iggy_only:
        print("\n--- Betfair Greyhound Market Data ---")
        for year in args.years:
            if year >= 2026:
                # 2026+ uses monthly CSVs (no yearly zip yet)
                for month in range(1, 13):
                    filename = f"ANZ_Greyhounds_{year}_{month:02d}.csv"
                    download_file(f"{ASSETS_BASE}/{filename}", DEST_DIR / filename, args.force)
            else:
                # 2020-2025 use yearly ZIPs
                filename = f"ANZ_Greyhounds_{year}.zip"
                dest = DEST_DIR / filename
                if download_file(f"{ASSETS_BASE}/{filename}", dest, args.force):
                    if args.extract and dest.exists():
                        extract_zip(dest)

    print("\n--- Iggy-Joey V2 Model Predictions ---")
    # V2 predictions (current model, from Aug 2024)
    for year in args.years:
        if year <= 2024:
            # 2024 has split file: Aug-Dec only for V2
            filename = f"Iggy_Model_V2_Results_{year}_08-12.csv"
            download_file(f"{ASSETS_BASE}/{filename}", DEST_DIR / filename, args.force)
        elif year >= 2026:
            # 2026+ monthly
            for month in range(1, 13):
                filename = f"Iggy_Model_V2_Results_{year}_{month:02d}.csv"
                download_file(f"{ASSETS_BASE}/{filename}", DEST_DIR / filename, args.force)
        else:
            # 2025 full year
            filename = f"Iggy_Model_V2_Results_{year}.csv"
            download_file(f"{ASSETS_BASE}/{filename}", DEST_DIR / filename, args.force)

    print("\nDone!")
    print(f"\nFiles in {DEST_DIR}:")
    for f in sorted(DEST_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name:50s} {f.stat().st_size:>12,} bytes")


if __name__ == "__main__":
    main()
