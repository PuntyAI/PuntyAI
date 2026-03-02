"""API endpoints for settings management."""

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, delete, func

from punty.config import melb_now_naive
from punty.models.database import get_db

router = APIRouter()


def _get_user_email(request: Request) -> str:
    """Extract user email from session, or 'system' if unavailable."""
    try:
        user = request.session.get("user", {})
        return user.get("email", "system")
    except Exception:
        return "system"


_SENSITIVE_KEYS = {
    "anthropic_api_key", "openai_api_key", "twitter_api_secret",
    "twitter_access_token", "twitter_access_secret", "smtp_password",
    "facebook_page_access_token", "facebook_app_secret",
    "telegram_bot_token", "resend_api_key", "punting_form_api_key",
    "willyweather_api_key",
}


async def _record_audit(db: AsyncSession, key: str, old_value: str | None, new_value: str | None, changed_by: str, action: str = "updated"):
    """Record a settings change in the audit log."""
    from punty.models.settings import SettingsAudit

    # Mask sensitive values — never store secrets in audit log
    if key in _SENSITIVE_KEYS:
        old_value = "***REDACTED***" if old_value else None
        new_value = "***CHANGED***" if new_value else None

    entry = SettingsAudit(
        key=key,
        old_value=old_value,
        new_value=new_value,
        changed_by=changed_by,
        action=action,
        changed_at=melb_now_naive(),
    )
    db.add(entry)


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
async def update_analysis_weights(update: WeightsUpdate, request: Request, db: AsyncSession = Depends(get_db)):
    """Update analysis framework weights."""
    from punty.models.settings import AnalysisWeights
    import json

    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    weights = result.scalar_one_or_none()

    old_weights = json.dumps(weights.weights) if weights else None

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
    await _record_audit(db, "analysis_weights", old_weights, json.dumps(update.weights), _get_user_email(request))
    await db.commit()

    return weights.to_dict()


@router.post("/weights/reset")
async def reset_analysis_weights(request: Request, db: AsyncSession = Depends(get_db)):
    """Reset analysis weights to defaults."""
    from punty.models.settings import AnalysisWeights
    import json

    result = await db.execute(select(AnalysisWeights).where(AnalysisWeights.id == "default"))
    weights = result.scalar_one_or_none()

    old_weights = json.dumps(weights.weights) if weights else None

    if weights:
        weights.weights = AnalysisWeights.DEFAULT_WEIGHTS
    else:
        weights = AnalysisWeights(id="default", name="Default Weights")
        weights.weights = AnalysisWeights.DEFAULT_WEIGHTS
        db.add(weights)

    await _record_audit(db, "analysis_weights", old_weights, "RESET TO DEFAULTS", _get_user_email(request))
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


# --- Personality prompt (must come BEFORE /{key} catch-all) ---

class PersonalityUpdate(BaseModel):
    """Update personality prompt."""
    content: str


@router.get("/personality")
async def get_personality(db: AsyncSession = Depends(get_db)):
    """Get the personality prompt (from DB, falling back to file)."""
    from punty.models.settings import AppSettings

    result = await db.execute(select(AppSettings).where(AppSettings.key == "personality_prompt"))
    setting = result.scalar_one_or_none()
    if setting and setting.value:
        return {"content": setting.value}

    # Fall back to file for migration
    from pathlib import Path
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "personality.md"
    if prompt_path.exists():
        return {"content": prompt_path.read_text(encoding="utf-8")}
    return {"content": ""}


@router.put("/personality")
async def save_personality(update: PersonalityUpdate, request: Request, db: AsyncSession = Depends(get_db)):
    """Save the personality prompt to DB (survives deploys)."""
    from punty.models.settings import AppSettings
    from punty.ai.generator import _personality_cache

    # Save to DB
    result = await db.execute(select(AppSettings).where(AppSettings.key == "personality_prompt"))
    setting = result.scalar_one_or_none()
    old_value = setting.value if setting else None
    if setting:
        setting.value = update.content
    else:
        db.add(AppSettings(key="personality_prompt", value=update.content))

    action = "updated" if old_value else "created"
    await _record_audit(db, "personality_prompt", f"({len(old_value)} chars)" if old_value else None, f"({len(update.content)} chars)", _get_user_email(request), action)
    await db.commit()

    # Update in-memory cache
    _personality_cache.set(update.content)

    return {"status": "saved", "length": len(update.content)}


# --- Audit Log (must come BEFORE /{key} catch-all) ---

@router.get("/audit-log")
async def get_audit_log(
    limit: int = 50,
    offset: int = 0,
    key: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get settings change audit log."""
    from punty.models.settings import SettingsAudit
    from sqlalchemy import func

    query = select(SettingsAudit).order_by(desc(SettingsAudit.changed_at))
    count_query = select(func.count(SettingsAudit.id))

    if key:
        query = query.where(SettingsAudit.key == key)
        count_query = count_query.where(SettingsAudit.key == key)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    entries = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return {
        "entries": [e.to_dict() for e in entries],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# --- Probability Model Weights (must come BEFORE /{key} catch-all) ---

class ProbabilityWeightsUpdate(BaseModel):
    """Update probability model weights."""
    weights: dict[str, float]  # factor_key → percentage (0-100)


@router.get("/probability-weights")
async def get_probability_weights(db: AsyncSession = Depends(get_db)):
    """Get current probability model weights and factor metadata."""
    from punty.models.settings import AppSettings
    from punty.probability import FACTOR_REGISTRY, DEFAULT_WEIGHTS
    import json

    result = await db.execute(select(AppSettings).where(AppSettings.key == "probability_weights"))
    setting = result.scalar_one_or_none()

    if setting and setting.value:
        stored = json.loads(setting.value)
    else:
        # Convert decimals to percentages for display
        stored = {k: round(v * 100) for k, v in DEFAULT_WEIGHTS.items()}

    # Build factor metadata with categories
    factors = []
    for key, meta in FACTOR_REGISTRY.items():
        factors.append({
            "key": key,
            "label": meta["label"],
            "category": meta["category"],
            "description": meta["description"],
            "default": round(DEFAULT_WEIGHTS.get(key, 0) * 100),
            "weight": stored.get(key, round(DEFAULT_WEIGHTS.get(key, 0) * 100)),
        })

    return {
        "weights": stored,
        "factors": factors,
    }


@router.put("/probability-weights")
async def update_probability_weights(
    update: ProbabilityWeightsUpdate, request: Request, db: AsyncSession = Depends(get_db),
):
    """Update probability model weights. Values are percentages (0-100) and must sum to ~100."""
    from punty.models.settings import AppSettings
    from punty.probability import FACTOR_REGISTRY
    import json

    # Validate all keys exist in registry
    for key in update.weights:
        if key not in FACTOR_REGISTRY:
            raise HTTPException(status_code=400, detail=f"Unknown factor '{key}'")

    # Validate all values are non-negative
    for key, val in update.weights.items():
        if val < 0:
            raise HTTPException(status_code=400, detail=f"Weight for '{key}' cannot be negative")

    # Validate sum is close to 100
    total = sum(update.weights.values())
    if abs(total - 100) > 2:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 100% (currently {total:.1f}%)",
        )

    result = await db.execute(select(AppSettings).where(AppSettings.key == "probability_weights"))
    setting = result.scalar_one_or_none()
    old_value = setting.value if setting else None

    weights_json = json.dumps(update.weights)
    if setting:
        setting.value = weights_json
    else:
        setting = AppSettings(
            key="probability_weights",
            value=weights_json,
            description="Probability model factor weights (percentages)",
        )
        db.add(setting)

    action = "updated" if old_value is not None else "created"
    await _record_audit(db, "probability_weights", old_value, weights_json, _get_user_email(request), action)
    await db.commit()

    return {"status": "saved", "total": total}


@router.post("/probability-weights/reset")
async def reset_probability_weights(request: Request, db: AsyncSession = Depends(get_db)):
    """Reset probability weights to defaults."""
    from punty.models.settings import AppSettings
    from punty.probability import DEFAULT_WEIGHTS
    import json

    result = await db.execute(select(AppSettings).where(AppSettings.key == "probability_weights"))
    setting = result.scalar_one_or_none()
    old_value = setting.value if setting else None

    if setting:
        await db.delete(setting)

    await _record_audit(db, "probability_weights", old_value, "RESET TO DEFAULTS", _get_user_email(request))
    await db.commit()

    return {
        "status": "reset",
        "weights": {k: round(v * 100) for k, v in DEFAULT_WEIGHTS.items()},
    }


# --- Generic setting by key (catch-all, must come AFTER specific routes) ---

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
async def update_setting(key: str, update: SettingUpdate, request: Request, db: AsyncSession = Depends(get_db)):
    """Update a specific setting."""
    from punty.models.settings import AppSettings

    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()

    old_value = setting.value if setting else None
    if setting:
        setting.value = update.value
    else:
        # Create new setting
        description = AppSettings.DEFAULTS.get(key, {}).get("description", "")
        setting = AppSettings(key=key, value=update.value, description=description)
        db.add(setting)

    action = "updated" if old_value is not None else "created"
    await _record_audit(db, key, old_value, update.value, _get_user_email(request), action)
    await db.commit()
    return setting.to_dict()


PROVIDER_KEYS = {
    "openai": ["openai_api_key"],
    "twitter": ["twitter_api_key", "twitter_api_secret", "twitter_access_token", "twitter_access_secret"],
    "smtp": ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "smtp_from"],
    "resend": ["resend_api_key", "email_from", "notification_email"],
    "facebook": ["facebook_page_id", "facebook_page_access_token", "facebook_app_id", "facebook_app_secret"],
    "telegram": ["telegram_bot_token", "telegram_owner_id"],
    "anthropic": ["anthropic_api_key"],
    "punting_form": ["punting_form_api_key"],
    "willyweather": ["willyweather_api_key"],
    "betfair": [
        "betfair_auto_bet_enabled", "betfair_balance", "betfair_initial_balance",
        "betfair_stake", "betfair_stake_mode", "betfair_min_odds", "betfair_commission_rate", "betfair_max_daily_loss",
    ],
}


@router.put("/api-keys/{provider}")
async def update_api_keys(provider: str, update: ApiKeysUpdate, request: Request, db: AsyncSession = Depends(get_db)):
    """Update API keys for a provider (openai, twitter, smtp, resend)."""
    from punty.models.settings import AppSettings

    allowed = PROVIDER_KEYS.get(provider)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")

    user_email = _get_user_email(request)
    saved = {}
    for key, value in update.keys.items():
        if key not in allowed:
            continue
        result = await db.execute(select(AppSettings).where(AppSettings.key == key))
        setting = result.scalar_one_or_none()
        old_value = setting.value if setting else None
        if setting:
            setting.value = value
        else:
            description = AppSettings.DEFAULTS.get(key, {}).get("description", "")
            setting = AppSettings(key=key, value=value, description=description)
            db.add(setting)
        action = "updated" if old_value is not None else "created"
        await _record_audit(db, key, old_value, value, user_email, action)
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


@router.post("/cleanup")
async def cleanup_database(request: Request, db: AsyncSession = Depends(get_db)):
    """Delete superseded/rejected content and other accumulated stale data."""
    from punty.models.content import Content, ContextSnapshot
    from punty.models.settings import SettingsAudit
    from datetime import timedelta
    import asyncio

    user_email = _get_user_email(request)
    now = melb_now_naive()

    # Retry logic for SQLite lock contention with results monitor
    max_retries = 3
    for attempt in range(max_retries):
        try:
            deleted = {}

            from punty.models.pick import Pick

            # 1. Superseded content — delete associated picks first
            sup_ids = (await db.execute(
                select(Content.id).where(Content.status == "superseded")
            )).scalars().all()
            if sup_ids:
                await db.execute(delete(Pick).where(Pick.content_id.in_(sup_ids)))
                result = await db.execute(delete(Content).where(Content.id.in_(sup_ids)))
                deleted["superseded_content"] = result.rowcount
            else:
                deleted["superseded_content"] = 0

            # 2. Rejected content — delete associated picks first
            rej_ids = (await db.execute(
                select(Content.id).where(Content.status == "rejected")
            )).scalars().all()
            if rej_ids:
                await db.execute(delete(Pick).where(Pick.content_id.in_(rej_ids)))
                result = await db.execute(delete(Content).where(Content.id.in_(rej_ids)))
                deleted["rejected_content"] = result.rowcount
            else:
                deleted["rejected_content"] = 0

            # 3. Draft content older than 7 days — delete associated picks first
            cutoff = now - timedelta(days=7)
            draft_ids = (await db.execute(
                select(Content.id).where(Content.status == "draft", Content.created_at < cutoff)
            )).scalars().all()
            if draft_ids:
                await db.execute(delete(Pick).where(Pick.content_id.in_(draft_ids)))
                result = await db.execute(delete(Content).where(Content.id.in_(draft_ids)))
                deleted["old_drafts"] = result.rowcount
            else:
                deleted["old_drafts"] = 0

            # 4. Context snapshots older than 14 days (keep recent for change detection)
            snap_cutoff = now - timedelta(days=14)
            result = await db.execute(
                delete(ContextSnapshot).where(ContextSnapshot.created_at < snap_cutoff)
            )
            deleted["old_snapshots"] = result.rowcount

            # 5. Audit log entries older than 90 days
            audit_cutoff = now - timedelta(days=90)
            result = await db.execute(
                delete(SettingsAudit).where(SettingsAudit.changed_at < audit_cutoff)
            )
            deleted["old_audit_logs"] = result.rowcount

            total = sum(deleted.values())

            await _record_audit(
                db, "database_cleanup",
                None, str(deleted),
                user_email, action="cleanup",
            )
            await db.commit()

            return {
                "status": "success",
                "message": f"Cleaned up {total} records",
                "total": total,
                "breakdown": deleted,
            }
        except Exception as e:
            await db.rollback()
            if "database is locked" in str(e) and attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise HTTPException(
                status_code=503,
                detail=f"Database busy, please try again in a moment ({e})",
            )
