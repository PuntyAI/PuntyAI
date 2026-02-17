"""Analytics API — serves backtest results to the dashboard."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

BACKTEST_RESULTS_PATH = Path("data/backtest_results.json")


def _load_results() -> dict:
    """Load backtest results from JSON file."""
    if not BACKTEST_RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="No backtest results found. Run scripts/run_backtest.py first.")
    with open(BACKTEST_RESULTS_PATH) as f:
        return json.load(f)


@router.get("/backtest")
async def get_backtest_summary():
    """Full backtest summary."""
    return _load_results()


@router.get("/calibration")
async def get_calibration():
    """Calibration chart data (predicted vs actual win/place rates)."""
    data = _load_results()
    return data.get("calibration", {})


@router.get("/breakdown/{dimension}")
async def get_breakdown(dimension: str):
    """Get breakdown by dimension (venue, distance, odds_bands, value_bins)."""
    data = _load_results()
    key_map = {
        "venue": "venue_breakdown",
        "distance": "distance_breakdown",
        "odds": "odds_bands",
        "value": "value_bins",
        "factors": "factor_sensitivity",
    }
    key = key_map.get(dimension)
    if not key or key not in data:
        raise HTTPException(status_code=404, detail=f"Unknown dimension: {dimension}")
    return data[key]
