"""Context-Aware Auto-Calibration Engine.

Learns optimal probability factor weights per context cell from settled results.
Context cells: (distance_bucket × condition_bucket × class_bucket × venue_type)

Weekly auto-recalibration from settled picks replaces hardcoded weight tables
with data-driven weights. Cells with fewer than MIN_CELL_SAMPLES fall back to
the distance-level defaults.

Usage:
    # Build/update calibration table (weekly job):
    await calibrate_weights(db)

    # Get optimal weights for a specific race context:
    weights = get_calibrated_weights(distance=1400, condition="Good 4",
                                     race_class="BM68", venue="Caulfield")
"""

import json
import logging
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick
from punty.models.settings import AppSettings
from punty.probability import (
    DEFAULT_WEIGHTS,
    DISTANCE_WEIGHT_OVERRIDES,
)

logger = logging.getLogger(__name__)

MIN_CELL_SAMPLES = 30  # Minimum settled picks per cell to trust calibrated weights
CALIBRATION_WINDOW_DAYS = 90  # Use last 3 months of data
CALIBRATION_KEY = "calibrated_context_weights"

# Context buckets
def _distance_bucket(d: int) -> str:
    if not d:
        return "middle"
    if d <= 1100:
        return "sprint"
    if d <= 1399:
        return "short"
    if d <= 1799:
        return "middle"
    if d <= 2199:
        return "classic"
    return "staying"


def _condition_bucket(c: str) -> str:
    if not c:
        return "good"
    c = c.upper()
    if c.startswith("H"):
        return "heavy"
    if c.startswith("S"):
        return "soft"
    return "good"


def _class_bucket(cls: str) -> str:
    if not cls:
        return "other"
    cls = cls.lower()
    if "maiden" in cls:
        return "maiden"
    if "class 1" in cls or "restricted" in cls:
        return "restricted"
    if "benchmark" in cls or "bm" in cls:
        return "benchmark"
    if "handicap" in cls:
        return "handicap"
    if "open" in cls or "group" in cls or "listed" in cls:
        return "open"
    return "other"


METRO_VENUES = {
    "flemington", "caulfield", "moonee valley", "sandown", "randwick",
    "rosehill", "canterbury", "warwick farm", "eagle farm", "doomben",
    "morphettville", "ascot", "belmont", "hobart",
}

def _venue_type(venue: str) -> str:
    """Metro vs provincial vs country."""
    if not venue:
        return "country"
    v = venue.lower().strip()
    if v in METRO_VENUES:
        return "metro"
    return "country"


def get_context_key(distance: int, condition: str, race_class: str, venue: str) -> str:
    """Build context key for weight lookup."""
    return f"{_distance_bucket(distance)}|{_condition_bucket(condition)}|{_class_bucket(race_class)}|{_venue_type(venue)}"


def get_calibrated_weights(
    distance: int = 1400,
    condition: str = "Good 4",
    race_class: str = "",
    venue: str = "",
    _cache: dict | None = None,
) -> dict[str, float]:
    """Get optimal weights for a specific racing context.

    Tries context cell first, falls back to distance-level defaults.
    Thread-safe (read-only after calibration).
    """
    # Try cached calibrated weights
    if _cache is None:
        _cache = _CALIBRATION_CACHE

    key = get_context_key(distance, condition, race_class, venue)
    if key in _cache:
        return _cache[key]

    # Fallback: try just distance × condition
    dist_cond_key = f"{_distance_bucket(distance)}|{_condition_bucket(condition)}|*|*"
    if dist_cond_key in _cache:
        return _cache[dist_cond_key]

    # Fallback: distance-level defaults
    dist_bucket = _distance_bucket(distance)
    return DISTANCE_WEIGHT_OVERRIDES.get(dist_bucket, DEFAULT_WEIGHTS)


# In-memory cache (loaded at startup, refreshed weekly)
_CALIBRATION_CACHE: dict[str, dict[str, float]] = {}


async def load_calibration_cache(db: AsyncSession) -> int:
    """Load calibrated weights from DB into memory cache."""
    global _CALIBRATION_CACHE

    result = await db.execute(
        select(AppSettings).where(AppSettings.key == CALIBRATION_KEY)
    )
    setting = result.scalar_one_or_none()
    if not setting or not setting.value:
        return 0

    try:
        data = json.loads(setting.value)
        _CALIBRATION_CACHE = data
        logger.info(f"Loaded {len(data)} calibrated weight cells")
        return len(data)
    except (json.JSONDecodeError, TypeError):
        return 0


async def calibrate_weights(db: AsyncSession) -> dict:
    """Build optimal weights per context cell from settled results.

    Analyses last 90 days of settled picks, groups by context cell,
    and stores the factor scores that produced the best outcomes.

    This is a simplified calibration — it doesn't grid-search weights,
    it measures which factor scores correlate most with winners in each cell
    and adjusts weights accordingly.
    """
    global _CALIBRATION_CACHE

    cutoff = melb_today() - timedelta(days=CALIBRATION_WINDOW_DAYS)

    # Load settled R1 picks with factor details
    result = await db.execute(
        select(Pick, Race, Meeting).join(
            Meeting, Meeting.id == Pick.meeting_id
        ).join(
            Race, (Race.meeting_id == Pick.meeting_id) & (Race.race_number == Pick.race_number)
        ).where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.tip_rank == 1,
            Meeting.date >= str(cutoff),
            Pick.factors_json.isnot(None),
        )
    )
    rows = result.all()

    if not rows:
        logger.warning("No settled picks with factor data for calibration")
        return {"status": "no_data", "cells": 0}

    # Group by context cell
    cells = defaultdict(lambda: {"winners": [], "losers": []})
    for pick, race, meeting in rows:
        try:
            factors = json.loads(pick.factors_json) if isinstance(pick.factors_json, str) else pick.factors_json
        except (json.JSONDecodeError, TypeError):
            continue
        if not factors:
            continue

        distance = race.distance or 1400
        condition = meeting.track_condition or "Good 4"
        race_class = race.class_ or ""
        venue = meeting.venue or ""

        key = get_context_key(distance, condition, race_class, venue)

        fp = None
        # Try to get finish position from runner
        runner_result = await db.execute(
            select(Runner.finish_position).where(
                Runner.race_id == race.id,
                Runner.saddlecloth == pick.saddlecloth,
            )
        )
        fp_row = runner_result.scalar_one_or_none()

        if fp_row and fp_row <= 3:
            cells[key]["winners"].append(factors)
        else:
            cells[key]["losers"].append(factors)

    # For each cell with enough data, compute weight adjustments
    calibrated = {}
    for key, data in cells.items():
        total = len(data["winners"]) + len(data["losers"])
        if total < MIN_CELL_SAMPLES:
            continue

        # Simple approach: for each factor, compare avg score of winners vs losers
        # Higher avg in winners = this factor matters more in this context
        factor_names = list(DEFAULT_WEIGHTS.keys())
        adjustments = {}

        for factor in factor_names:
            win_scores = [f.get(factor, 0.5) for f in data["winners"] if factor in f]
            lose_scores = [f.get(factor, 0.5) for f in data["losers"] if factor in f]

            if not win_scores or not lose_scores:
                continue

            win_avg = sum(win_scores) / len(win_scores)
            lose_avg = sum(lose_scores) / len(lose_scores)
            spread = win_avg - lose_avg  # Positive = factor helps identify winners

            # Scale: 0.10 spread = +50% weight boost, -0.10 = -50%
            if abs(spread) > 0.02:  # Only adjust if meaningful
                adjustments[factor] = spread

        if not adjustments:
            continue

        # Start from distance defaults, apply adjustments
        dist_bucket = key.split("|")[0]
        base_weights = dict(DISTANCE_WEIGHT_OVERRIDES.get(dist_bucket, DEFAULT_WEIGHTS))

        for factor, spread in adjustments.items():
            if factor in base_weights:
                # Spread of 0.10 = +50% weight, capped at ±100%
                multiplier = 1.0 + min(1.0, max(-0.5, spread * 5))
                base_weights[factor] *= multiplier

        # Normalize to sum to 1.0
        total_w = sum(base_weights.values())
        if total_w > 0:
            calibrated[key] = {k: round(v / total_w, 4) for k, v in base_weights.items()}

    # Store in DB
    _CALIBRATION_CACHE = calibrated
    setting_result = await db.execute(
        select(AppSettings).where(AppSettings.key == CALIBRATION_KEY)
    )
    setting = setting_result.scalar_one_or_none()
    if setting:
        setting.value = json.dumps(calibrated)
    else:
        db.add(AppSettings(key=CALIBRATION_KEY, value=json.dumps(calibrated)))
    await db.commit()

    logger.info(f"Calibrated {len(calibrated)} context cells from {len(rows)} picks")
    return {"status": "ok", "cells": len(calibrated), "picks_analysed": len(rows)}
