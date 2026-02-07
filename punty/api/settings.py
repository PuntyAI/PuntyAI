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


class ApiKeysUpdate(BaseModel):
    """Update multiple API keys for a provider."""

    keys: dict[str, str]


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


PROVIDER_KEYS = {
    "openai": ["openai_api_key"],
    "twitter": ["twitter_api_key", "twitter_api_secret", "twitter_access_token", "twitter_access_secret"],
    "whatsapp": ["whatsapp_api_token", "whatsapp_phone_number_id"],
    "smtp": ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "smtp_from"],
    "resend": ["resend_api_key", "email_from", "notification_email"],
}


@router.put("/api-keys/{provider}")
async def update_api_keys(provider: str, update: ApiKeysUpdate, db: AsyncSession = Depends(get_db)):
    """Update API keys for a provider (openai, twitter, whatsapp)."""
    from punty.models.settings import AppSettings

    allowed = PROVIDER_KEYS.get(provider)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")

    saved = {}
    for key, value in update.keys.items():
        if key not in allowed:
            continue
        result = await db.execute(select(AppSettings).where(AppSettings.key == key))
        setting = result.scalar_one_or_none()
        if setting:
            setting.value = value
        else:
            desc = AppSettings.DEFAULTS.get(key, {}).get("description", "")
            setting = AppSettings(key=key, value=value, description=desc)
            db.add(setting)
        saved[key] = True

    await db.commit()
    return {"provider": provider, "saved": list(saved.keys())}


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


class TestEmailRequest(BaseModel):
    """Test email request."""
    to_email: str


class PersonalityUpdate(BaseModel):
    """Update personality prompt."""
    content: str


@router.get("/personality")
async def get_personality():
    """Get the personality prompt."""
    from pathlib import Path
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "personality.md"
    if prompt_path.exists():
        return {"content": prompt_path.read_text(encoding="utf-8")}
    return {"content": ""}


@router.put("/personality")
async def save_personality(update: PersonalityUpdate):
    """Save the personality prompt."""
    from pathlib import Path
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "personality.md"
    prompt_path.write_text(update.content, encoding="utf-8")
    return {"status": "saved", "length": len(update.content)}


@router.post("/test-email")
async def test_email(request: TestEmailRequest):
    """Send a test email to verify SMTP settings."""
    from punty.delivery.email import send_email

    result = await send_email(
        to_email=request.to_email,
        subject="PuntyAI Test Email",
        body_html="""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h1 style="color: #e91e63;">PuntyAI Email Test</h1>
            <p>If you're reading this, your SMTP settings are working correctly!</p>
            <p style="color: #666;">This is a test email from PuntyAI.</p>
        </body>
        </html>
        """,
        body_text="PuntyAI Email Test\n\nIf you're reading this, your SMTP settings are working correctly!",
    )

    return result


# --- Race Learnings / Assessments ---

@router.get("/learnings")
async def get_learnings(
    db: AsyncSession = Depends(get_db),
    meeting_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Get race assessments/learnings with optional meeting filter."""
    from punty.memory.models import RaceAssessment
    from sqlalchemy import desc
    import json

    query = select(RaceAssessment).order_by(desc(RaceAssessment.created_at))

    if meeting_id:
        query = query.where(RaceAssessment.meeting_id == meeting_id)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    assessments = result.scalars().all()

    # Also get total count
    from sqlalchemy import func
    count_query = select(func.count(RaceAssessment.id))
    if meeting_id:
        count_query = count_query.where(RaceAssessment.meeting_id == meeting_id)
    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return {
        "assessments": [
            {
                "id": a.id,
                "race_id": a.race_id,
                "meeting_id": a.meeting_id,
                "race_number": a.race_number,
                "track": a.track,
                "distance": a.distance,
                "race_class": a.race_class,
                "going": a.going,
                "key_learnings": a.key_learnings,
                "top_pick_hit": a.top_pick_hit,
                "any_pick_hit": a.any_pick_hit,
                "total_pnl": a.total_pnl,
                "assessment": json.loads(a.assessment_json) if a.assessment_json else {},
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in assessments
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/learnings/meetings")
async def get_learnings_meetings(db: AsyncSession = Depends(get_db)):
    """Get list of meetings that have learnings."""
    from punty.memory.models import RaceAssessment
    from sqlalchemy import func, desc

    result = await db.execute(
        select(
            RaceAssessment.meeting_id,
            RaceAssessment.track,
            func.count(RaceAssessment.id).label("count"),
            func.sum(RaceAssessment.total_pnl).label("total_pnl"),
            func.max(RaceAssessment.created_at).label("latest"),
        )
        .group_by(RaceAssessment.meeting_id, RaceAssessment.track)
        .order_by(desc("latest"))
    )
    rows = result.all()

    return [
        {
            "meeting_id": row.meeting_id,
            "track": row.track,
            "assessment_count": row.count,
            "total_pnl": round(row.total_pnl or 0, 2),
            "latest": row.latest.isoformat() if row.latest else None,
        }
        for row in rows
    ]


@router.get("/learnings/{assessment_id}")
async def get_learning(assessment_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific learning/assessment by ID."""
    from punty.memory.models import RaceAssessment
    import json

    result = await db.execute(
        select(RaceAssessment).where(RaceAssessment.id == assessment_id)
    )
    a = result.scalar_one_or_none()

    if not a:
        raise HTTPException(status_code=404, detail="Assessment not found")

    return {
        "id": a.id,
        "race_id": a.race_id,
        "meeting_id": a.meeting_id,
        "race_number": a.race_number,
        "track": a.track,
        "distance": a.distance,
        "race_class": a.race_class,
        "going": a.going,
        "rail_position": a.rail_position,
        "key_learnings": a.key_learnings,
        "top_pick_hit": a.top_pick_hit,
        "any_pick_hit": a.any_pick_hit,
        "total_pnl": a.total_pnl,
        "assessment": json.loads(a.assessment_json) if a.assessment_json else {},
        "has_embedding": a.embedding_json is not None,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


@router.delete("/learnings/{assessment_id}")
async def delete_learning(assessment_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a learning/assessment if it's incorrect."""
    from punty.memory.models import RaceAssessment

    result = await db.execute(
        select(RaceAssessment).where(RaceAssessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    race_id = assessment.race_id
    await db.delete(assessment)
    await db.commit()

    return {"status": "deleted", "race_id": race_id}


@router.get("/learnings/stats/summary")
async def get_learnings_stats(db: AsyncSession = Depends(get_db)):
    """Get overall learnings statistics."""
    from punty.memory.models import RaceAssessment
    from sqlalchemy import func

    # Total assessments
    total_result = await db.execute(select(func.count(RaceAssessment.id)))
    total = total_result.scalar_one()

    if total == 0:
        return {
            "total_assessments": 0,
            "top_pick_hit_rate": 0,
            "any_pick_hit_rate": 0,
            "avg_pnl": 0,
            "total_pnl": 0,
        }

    # Top pick hit rate
    top_hit_result = await db.execute(
        select(func.count(RaceAssessment.id)).where(RaceAssessment.top_pick_hit == True)
    )
    top_hits = top_hit_result.scalar_one()

    # Any pick hit rate
    any_hit_result = await db.execute(
        select(func.count(RaceAssessment.id)).where(RaceAssessment.any_pick_hit == True)
    )
    any_hits = any_hit_result.scalar_one()

    # PNL stats
    pnl_result = await db.execute(
        select(
            func.avg(RaceAssessment.total_pnl),
            func.sum(RaceAssessment.total_pnl),
        )
    )
    pnl_row = pnl_result.one()
    avg_pnl = pnl_row[0] or 0
    total_pnl = pnl_row[1] or 0

    return {
        "total_assessments": total,
        "top_pick_hit_rate": round(top_hits / total * 100, 1) if total > 0 else 0,
        "any_pick_hit_rate": round(any_hits / total * 100, 1) if total > 0 else 0,
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
    }
