"""API endpoints for settings management."""

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from punty.models.database import get_db

router = APIRouter()


class WeightsUpdate(BaseModel):
    """Update analysis weights."""

    weights: dict[str, str]


class SettingUpdate(BaseModel):
    """Update a single setting."""

    value: str


@router.get("/weights")
async def get_analysis_weights(db: AsyncSession = Depends(get_db)):
    """Get current analysis framework weights."""
    from punty.models.settings import AnalysisWeights

    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    weights = result.scalar_one_or_none()

    if not weights:
        # Create default weights
        weights = AnalysisWeights(id="default", name="Default Weights")
        weights.weights = AnalysisWeights.DEFAULT_WEIGHTS
        db.add(weights)
        await db.commit()

    return weights.to_dict()


@router.put("/weights")
async def update_analysis_weights(update: WeightsUpdate, db: AsyncSession = Depends(get_db)):
    """Update analysis framework weights."""
    from punty.models.settings import AnalysisWeights

    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    weights = result.scalar_one_or_none()

    if not weights:
        weights = AnalysisWeights(id="default", name="Default Weights")
        db.add(weights)

    # Validate weight values
    valid_options = AnalysisWeights.WEIGHT_OPTIONS
    for key, value in update.weights.items():
        if value not in valid_options:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid weight value '{value}' for '{key}'. Must be one of: {valid_options}",
            )

    weights.weights = update.weights
    await db.commit()

    return weights.to_dict()


@router.post("/weights/reset")
async def reset_analysis_weights(db: AsyncSession = Depends(get_db)):
    """Reset analysis weights to defaults."""
    from punty.models.settings import AnalysisWeights

    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    weights = result.scalar_one_or_none()

    if weights:
        weights.weights = AnalysisWeights.DEFAULT_WEIGHTS
        await db.commit()
        return weights.to_dict()
    else:
        weights = AnalysisWeights(id="default", name="Default Weights")
        weights.weights = AnalysisWeights.DEFAULT_WEIGHTS
        db.add(weights)
        await db.commit()
        return weights.to_dict()


@router.get("/")
async def get_all_settings(db: AsyncSession = Depends(get_db)):
    """Get all application settings."""
    from punty.models.settings import AppSettings

    result = await db.execute(select(AppSettings))
    settings = result.scalars().all()

    # Build dict with defaults for missing settings
    settings_dict = {s.key: s.to_dict() for s in settings}

    # Add defaults for any missing
    for key, default in AppSettings.DEFAULTS.items():
        if key not in settings_dict:
            settings_dict[key] = {
                "key": key,
                "value": default["value"],
                "description": default["description"],
                "updated_at": None,
            }

    return settings_dict


@router.get("/{key}")
async def get_setting(key: str, db: AsyncSession = Depends(get_db)):
    """Get a specific setting."""
    from punty.models.settings import AppSettings

    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()

    if setting:
        return setting.to_dict()

    # Check defaults
    if key in AppSettings.DEFAULTS:
        return {
            "key": key,
            "value": AppSettings.DEFAULTS[key]["value"],
            "description": AppSettings.DEFAULTS[key]["description"],
            "updated_at": None,
        }

    raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")


@router.put("/{key}")
async def update_setting(key: str, update: SettingUpdate, db: AsyncSession = Depends(get_db)):
    """Update a specific setting."""
    from punty.models.settings import AppSettings

    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()

    if setting:
        setting.value = update.value
    else:
        # Create new setting
        description = AppSettings.DEFAULTS.get(key, {}).get("description", "")
        setting = AppSettings(key=key, value=update.value, description=description)
        db.add(setting)

    await db.commit()
    return setting.to_dict()


@router.post("/initialize")
async def initialize_settings(db: AsyncSession = Depends(get_db)):
    """Initialize all default settings."""
    from punty.models.settings import AppSettings, AnalysisWeights

    # Initialize app settings
    for key, default in AppSettings.DEFAULTS.items():
        result = await db.execute(select(AppSettings).where(AppSettings.key == key))
        if not result.scalar_one_or_none():
            setting = AppSettings(
                key=key, value=default["value"], description=default["description"]
            )
            db.add(setting)

    # Initialize analysis weights
    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    if not result.scalar_one_or_none():
        weights = AnalysisWeights(id="default", name="Default Weights")
        weights.weights = AnalysisWeights.DEFAULT_WEIGHTS
        db.add(weights)

    await db.commit()
    return {"status": "initialized"}
