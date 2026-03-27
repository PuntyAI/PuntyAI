#!/usr/bin/env python3
"""Upload v2 optimised context weights to the DB AppSettings table.

Stores the hierarchical weight structure (global, distance, dc, dcc) under
the 'calibrated_context_weights' key, which the calibration engine reads
at startup via load_calibration_cache().

Usage:
    python scripts/upload_v2_weights.py [--json-path PATH]

Default JSON path: scripts/_backtest_data/optimised_weights_v2.json
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from punty.models.database import async_session_factory, init_db
from punty.models.settings import AppSettings
from punty.calibration_engine import CALIBRATION_KEY


DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_backtest_data",
    "optimised_weights_v2.json",
)


async def upload_weights(json_path: str) -> None:
    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Validate structure
    required = ["global_weights", "distance_weights", "dc_weights", "dcc_weights"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"ERROR: Missing keys in JSON: {missing}")
        sys.exit(1)

    # Strip non-weight metadata before storing
    store_data = {
        "global_weights": data["global_weights"],
        "distance_weights": data["distance_weights"],
        "dc_weights": data["dc_weights"],
        "dcc_weights": data["dcc_weights"],
    }

    n_dcc = len(store_data["dcc_weights"])
    n_dc = len(store_data["dc_weights"])
    n_dist = len(store_data["distance_weights"])
    total = n_dcc + n_dc + n_dist + 1  # +1 for global

    print(f"v2 weights: {n_dcc} dcc, {n_dc} dc, {n_dist} distance, 1 global ({total} total cells)")
    print(f"Factors per cell: {list(store_data['global_weights'].keys())}")

    await init_db()
    async with async_session_factory() as db:
        result = await db.execute(
            select(AppSettings).where(AppSettings.key == CALIBRATION_KEY)
        )
        setting = result.scalar_one_or_none()

        json_str = json.dumps(store_data)
        if setting:
            setting.value = json_str
            print(f"Updated existing '{CALIBRATION_KEY}' ({len(json_str)} bytes)")
        else:
            db.add(AppSettings(key=CALIBRATION_KEY, value=json_str))
            print(f"Created new '{CALIBRATION_KEY}' ({len(json_str)} bytes)")

        await db.commit()
        print("OK — weights stored in DB. Restart app to reload cache.")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    asyncio.run(upload_weights(json_path))
