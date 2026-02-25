"""Analytics API — serves backtest results and DuckDB analytics to the dashboard."""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

BACKTEST_RESULTS_PATH = Path("data/backtest_results.json")


# ---------------------------------------------------------------------------
# Legacy JSON-file endpoints (fallback when DuckDB not available)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DuckDB-backed endpoints (interactive, filtered)
# ---------------------------------------------------------------------------

def _get_filters(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    venue: Optional[str] = None,
    state: Optional[str] = None,
    odds_min: Optional[float] = None,
    odds_max: Optional[float] = None,
    distance_category: Optional[str] = None,
    track_condition: Optional[str] = None,
    speed_map_position: Optional[str] = None,
) -> dict:
    """Build a filter dict from query params, excluding None values."""
    return {k: v for k, v in {
        "date_from": date_from,
        "date_to": date_to,
        "venue": venue,
        "state": state,
        "odds_min": odds_min,
        "odds_max": odds_max,
        "distance_category": distance_category,
        "track_condition": track_condition,
        "speed_map_position": speed_map_position,
    }.items() if v is not None}


def _check_duckdb():
    """Check DuckDB is available, raise 503 if not."""
    from punty.analytics.engine import is_available
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail="Analytics database not built. Run: python scripts/build_analytics.py",
        )


@router.get("/filters")
async def get_filter_options():
    """Dropdown options for filter bar."""
    _check_duckdb()
    from punty.analytics import engine, queries

    sql, params = queries.filter_options()
    rows = await engine.query(sql, params)

    # Group by category
    result: dict[str, list] = {}
    for row in rows:
        cat = row["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append({"value": row["value"], "count": row["count"]})

    return result


@router.get("/summary")
async def get_summary(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Filtered summary stats."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.summary(filters)
    return await engine.query_one(sql, params)


@router.get("/calibration")
async def get_calibration(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Filtered calibration data."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.calibration(filters)
    return await engine.query(sql, params)


@router.get("/odds-bands")
async def get_odds_bands(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Odds band performance."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.odds_band_performance(filters)
    return await engine.query(sql, params)


@router.get("/speed-maps")
async def get_speed_maps(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Speed map position analysis."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.speed_map_analysis(filters)
    return await engine.query(sql, params)


@router.get("/speed-map-heatmap")
async def get_speed_map_heatmap(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Speed map settle vs finish heatmap."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.speed_map_heatmap(filters)
    return await engine.query(sql, params)


@router.get("/venues")
async def get_venues(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Venue performance."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.venue_performance(filters)
    return await engine.query(sql, params)


@router.get("/distance")
async def get_distance(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Distance category performance."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.distance_performance(filters)
    return await engine.query(sql, params)


@router.get("/bet-types")
async def get_bet_types(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Production bet type P&L."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.bet_type_performance(filters)
    return await engine.query(sql, params)


@router.get("/factors")
async def get_factors(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Factor importance (winners vs losers)."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.factor_importance(filters)
    return await engine.query(sql, params)


@router.get("/a2e")
async def get_a2e(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Jockey/trainer A2E combos."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.jockey_trainer_a2e(filters)
    return await engine.query(sql, params)


@router.get("/time-series")
async def get_time_series(
    group_by: str = Query("month", regex="^(month|week)$"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    venue: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    odds_min: Optional[float] = Query(None),
    odds_max: Optional[float] = Query(None),
    distance_category: Optional[str] = Query(None),
    track_condition: Optional[str] = Query(None),
    speed_map_position: Optional[str] = Query(None),
):
    """Time series P&L trend from production picks."""
    _check_duckdb()
    from punty.analytics import engine, queries

    filters = _get_filters(date_from, date_to, venue, state, odds_min, odds_max,
                           distance_category, track_condition, speed_map_position)
    sql, params = queries.time_series(filters, group_by)
    return await engine.query(sql, params)
