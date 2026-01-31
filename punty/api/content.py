"""API endpoints for content management."""

import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.database import get_db

router = APIRouter()


class GenerateRequest(BaseModel):
    """Request to generate content."""

    meeting_id: str
    content_type: str  # early_mail, race_preview, results, update_alert
    race_id: Optional[str] = None


class ReviewAction(BaseModel):
    """Action to take on content review."""

    action: str  # approve, reject, regenerate, ai_fix
    notes: Optional[str] = None
    issue_type: Optional[str] = None  # For ai_fix: tone_wrong, factually_incorrect, etc.


@router.get("/")
async def list_content(
    meeting_id: Optional[str] = None,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List generated content with optional filters."""
    from punty.models.content import Content
    from sqlalchemy import select

    query = select(Content).order_by(Content.created_at.desc())

    if meeting_id:
        query = query.where(Content.meeting_id == meeting_id)
    if status:
        query = query.where(Content.status == status)
    if content_type:
        query = query.where(Content.content_type == content_type)

    result = await db.execute(query)
    items = result.scalars().all()
    return [c.to_dict() for c in items]


@router.get("/review-queue")
async def get_review_queue(db: AsyncSession = Depends(get_db)):
    """Get content pending review."""
    from punty.models.content import Content, ContentStatus
    from sqlalchemy import select

    result = await db.execute(
        select(Content)
        .where(Content.status == ContentStatus.PENDING_REVIEW)
        .order_by(Content.created_at.asc())
    )
    items = result.scalars().all()
    return [c.to_dict() for c in items]


@router.get("/generate-stream")
async def generate_content_stream(
    meeting_id: str,
    content_type: str,
    db: AsyncSession = Depends(get_db),
):
    """SSE stream of content generation progress."""
    from punty.ai.generator import ContentGenerator

    generator = ContentGenerator(db)

    async def event_generator():
        if content_type == "early_mail":
            async for event in generator.generate_early_mail_stream(meeting_id):
                yield f"data: {json.dumps(event)}\n\n"
        else:
            yield f"data: {json.dumps({'step': 1, 'total': 1, 'label': 'Unsupported content type for streaming', 'status': 'error'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{content_id}")
async def get_content(content_id: str, db: AsyncSession = Depends(get_db)):
    """Get specific content item."""
    from punty.models.content import Content
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    return content.to_dict()


@router.post("/generate")
async def generate_content(request: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Generate new content for a meeting."""
    from punty.ai.generator import ContentGenerator

    generator = ContentGenerator(db)

    try:
        if request.content_type == "early_mail":
            result = await generator.generate_early_mail(request.meeting_id)
        elif request.content_type == "race_preview":
            if not request.race_id:
                raise HTTPException(status_code=400, detail="race_id required for race_preview")
            # Extract race number from race_id
            from sqlalchemy import select
            from punty.models.meeting import Race
            race_result = await db.execute(select(Race).where(Race.id == request.race_id))
            race = race_result.scalar_one_or_none()
            if not race:
                raise HTTPException(status_code=404, detail="Race not found")
            result = await generator.generate_race_preview(request.meeting_id, race.race_number)
        elif request.content_type == "results":
            if not request.race_id:
                raise HTTPException(status_code=400, detail="race_id required for results")
            from sqlalchemy import select
            from punty.models.meeting import Race
            race_result = await db.execute(select(Race).where(Race.id == request.race_id))
            race = race_result.scalar_one_or_none()
            if not race:
                raise HTTPException(status_code=404, detail="Race not found")
            result = await generator.generate_results(request.meeting_id, race.race_number)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown content type: {request.content_type}")

        return {
            "status": "success",
            "content_id": result.get("content_id"),
            "content_type": result.get("content_type"),
            "raw_content": result.get("raw_content"),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{content_id}/review")
async def review_content(
    content_id: str, action: ReviewAction, db: AsyncSession = Depends(get_db)
):
    """Take action on content (approve, reject, regenerate, ai_fix)."""
    from punty.models.content import Content, ContentStatus
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    if action.action == "approve":
        content.status = ContentStatus.APPROVED
        content.review_notes = action.notes
    elif action.action == "reject":
        content.status = ContentStatus.REJECTED
        content.review_notes = action.notes
    elif action.action == "regenerate":
        # Will trigger regeneration
        content.status = ContentStatus.REGENERATING
        content.review_notes = action.notes
    elif action.action == "ai_fix":
        # Will trigger AI fix
        content.status = ContentStatus.REGENERATING
        content.review_notes = f"AI Fix requested: {action.issue_type} - {action.notes}"

    await db.commit()
    return content.to_dict()


@router.put("/{content_id}")
async def update_content(
    content_id: str, raw_content: str, db: AsyncSession = Depends(get_db)
):
    """Update content manually."""
    from punty.models.content import Content
    from punty.formatters.whatsapp import format_whatsapp
    from punty.formatters.twitter import format_twitter
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    content.raw_content = raw_content
    # Re-format for platforms
    content.whatsapp_formatted = format_whatsapp(raw_content, content.content_type)
    content.twitter_formatted = format_twitter(raw_content, content.content_type)

    await db.commit()
    return content.to_dict()


@router.post("/{content_id}/format")
async def format_content(content_id: str, db: AsyncSession = Depends(get_db)):
    """Re-format content for all platforms."""
    from punty.models.content import Content
    from punty.formatters.whatsapp import format_whatsapp
    from punty.formatters.twitter import format_twitter
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    # Re-format for platforms
    content.whatsapp_formatted = format_whatsapp(content.raw_content, content.content_type)
    content.twitter_formatted = format_twitter(content.raw_content, content.content_type)

    await db.commit()
    return content.to_dict()


class WidenRequest(BaseModel):
    """Request to widen selections."""

    feedback: Optional[str] = None  # Optional user feedback


@router.post("/{content_id}/widen")
async def widen_selections(
    content_id: str,
    request: Optional[WidenRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """Request AI to widen selections when too many favorites detected.

    This tells the AI that it has selected too many market favorites
    and asks it to reconsider with more value-oriented picks.
    """
    from punty.ai.generator import ContentGenerator

    generator = ContentGenerator(db)

    try:
        feedback = request.feedback if request else None
        result = await generator.request_widen_selections(content_id, feedback)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
