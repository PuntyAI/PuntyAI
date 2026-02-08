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

    action: str  # approve, reject, regenerate, ai_fix, unapprove
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


@router.post("/reject-all-pending")
async def reject_all_pending(db: AsyncSession = Depends(get_db)):
    """Reject all content in pending_review status."""
    from punty.models.content import Content, ContentStatus
    from sqlalchemy import select, update

    result = await db.execute(
        select(Content).where(Content.status == ContentStatus.PENDING_REVIEW)
    )
    items = result.scalars().all()
    count = len(items)
    for item in items:
        item.status = ContentStatus.REJECTED.value
        item.review_notes = "Bulk rejected from review queue"
    await db.commit()
    return {"rejected": count}


@router.get("/review-count")
async def get_review_count(db: AsyncSession = Depends(get_db)):
    """Get count of pending reviews as plain text for badge."""
    from punty.models.content import Content, ContentStatus
    from sqlalchemy import select, func
    from fastapi.responses import PlainTextResponse

    result = await db.execute(
        select(func.count(Content.id))
        .where(Content.status == ContentStatus.PENDING_REVIEW)
    )
    count = result.scalar() or 0
    return PlainTextResponse(str(count) if count > 0 else "")


@router.get("/generate-stream")
async def generate_content_stream(
    meeting_id: str,
    content_type: str,
    db: AsyncSession = Depends(get_db),
):
    """SSE stream of content generation progress."""
    from punty.ai.generator import ContentGenerator
    from punty.models.settings import AppSettings
    from sqlalchemy import select

    # Check if content type is enabled
    setting_key = f"enable_{content_type}"
    result = await db.execute(select(AppSettings).where(AppSettings.key == setting_key))
    setting = result.scalar_one_or_none()

    # Default: early_mail and meeting_wrapup are on, race_previews is off
    defaults = {"enable_early_mail": "true", "enable_meeting_wrapup": "true", "enable_race_previews": "false"}
    is_enabled = (setting.value if setting else defaults.get(setting_key, "true")) == "true"

    if not is_enabled:
        label = f"{content_type.replace('_', ' ').title()} is disabled in Settings"
        async def disabled_generator():
            yield f"data: {json.dumps({'step': 1, 'total': 1, 'label': label, 'status': 'error'})}\n\n"
        return StreamingResponse(disabled_generator(), media_type="text/event-stream")

    generator = ContentGenerator(db)

    async def event_generator():
        if content_type == "early_mail":
            async for event in generator.generate_early_mail_stream(meeting_id):
                yield f"data: {json.dumps(event)}\n\n"
        elif content_type == "meeting_wrapup":
            async for event in generator.generate_meeting_wrapup_stream(meeting_id):
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
    from punty.models.settings import AppSettings
    from sqlalchemy import select

    generator = ContentGenerator(db)

    try:
        if request.content_type == "early_mail":
            # Check if early mail is enabled
            setting_result = await db.execute(
                select(AppSettings).where(AppSettings.key == "enable_early_mail")
            )
            setting = setting_result.scalar_one_or_none()
            if setting and setting.value != "true":
                raise HTTPException(status_code=400, detail="Early mail is disabled. Enable in Settings.")
            result = await generator.generate_early_mail(request.meeting_id)
        elif request.content_type == "race_preview":
            # Check if race previews are enabled
            from punty.models.settings import AppSettings
            setting_result = await db.execute(
                select(AppSettings).where(AppSettings.key == "enable_race_previews")
            )
            setting = setting_result.scalar_one_or_none()
            if not setting or setting.value != "true":
                raise HTTPException(status_code=400, detail="Race previews are disabled. Enable in Settings.")

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

        # Store picks from early mail on approval
        if content.content_type == "early_mail" and content.raw_content:
            # Supersede any previously approved early_mail for this meeting
            from sqlalchemy import delete as sa_delete
            from punty.models.pick import Pick
            old_result = await db.execute(
                select(Content).where(
                    Content.meeting_id == content.meeting_id,
                    Content.content_type == "early_mail",
                    Content.status.in_(["approved", "sent"]),
                    Content.id != content.id,
                )
            )
            for old in old_result.scalars().all():
                old.status = ContentStatus.SUPERSEDED.value
                await db.execute(sa_delete(Pick).where(Pick.content_id == old.id))

            try:
                from punty.results.picks import store_picks_from_content, store_picks_as_memories
                await store_picks_from_content(db, content.id, content.meeting_id, content.raw_content)
                # Store picks as memories for learning system
                await store_picks_as_memories(db, content.meeting_id, content.id)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to store picks/memories: {e}")
    elif action.action == "unapprove":
        content.status = ContentStatus.PENDING_REVIEW.value
        content.review_notes = action.notes or "Unapproved by user"
        from sqlalchemy import delete as sa_delete
        from punty.models.pick import Pick
        await db.execute(sa_delete(Pick).where(Pick.content_id == content.id))
    elif action.action == "reject":
        content.status = ContentStatus.REJECTED
        content.review_notes = action.notes
    elif action.action == "regenerate":
        # Trigger actual regeneration
        content.status = ContentStatus.REGENERATING
        content.review_notes = action.notes
        await db.commit()

        # Actually regenerate
        from punty.ai.reviewer import ContentReviewer
        reviewer = ContentReviewer(db)
        try:
            result = await reviewer.regenerate_content(content_id, action.notes)
            # Refresh content after regeneration
            await db.refresh(content)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Regeneration failed: {e}")
            content.status = ContentStatus.PENDING_REVIEW
            content.review_notes = f"Regeneration failed: {e}"
            await db.commit()
        return content.to_dict()

    elif action.action == "ai_fix":
        # Trigger actual AI fix
        content.status = ContentStatus.REGENERATING
        content.review_notes = f"AI Fix requested: {action.issue_type} - {action.notes}"
        await db.commit()

        # Actually fix with AI
        from punty.ai.reviewer import ContentReviewer
        reviewer = ContentReviewer(db)
        try:
            result = await reviewer.fix_content(content_id, action.issue_type, action.notes)
            # Refresh content after fix
            await db.refresh(content)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"AI Fix failed: {e}")
            content.status = ContentStatus.PENDING_REVIEW
            content.review_notes = f"AI Fix failed: {e}"
            await db.commit()
        return content.to_dict()

    await db.commit()
    return content.to_dict()


@router.put("/{content_id}")
async def update_content(
    content_id: str, raw_content: str, db: AsyncSession = Depends(get_db)
):
    """Update content manually."""
    from punty.models.content import Content
    from punty.formatters.twitter import format_twitter
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    content.raw_content = raw_content
    content.twitter_formatted = format_twitter(raw_content, content.content_type)

    await db.commit()
    return content.to_dict()


@router.post("/{content_id}/format")
async def format_content(content_id: str, db: AsyncSession = Depends(get_db)):
    """Re-format content for Twitter."""
    from punty.models.content import Content
    from punty.formatters.twitter import format_twitter
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    content.twitter_formatted = format_twitter(content.raw_content, content.content_type)

    await db.commit()
    return content.to_dict()


class PickUpdate(BaseModel):
    """Update a single pick's fields."""
    horse_name: Optional[str] = None
    saddlecloth: Optional[int] = None
    odds_at_tip: Optional[float] = None
    bet_type: Optional[str] = None
    bet_stake: Optional[float] = None


class BulkPickUpdate(BaseModel):
    """Bulk update picks for a content item."""
    picks: dict[str, PickUpdate]  # pick_id -> fields to update


@router.get("/{content_id}/picks")
async def get_content_picks(content_id: str, db: AsyncSession = Depends(get_db)):
    """Get all picks for a content item."""
    from punty.models.pick import Pick
    from sqlalchemy import select

    result = await db.execute(
        select(Pick).where(Pick.content_id == content_id).order_by(Pick.race_number, Pick.tip_rank)
    )
    picks = result.scalars().all()
    return [p.to_dict() for p in picks]


@router.put("/{content_id}/picks")
async def update_content_picks(content_id: str, update: BulkPickUpdate, db: AsyncSession = Depends(get_db)):
    """Bulk update picks for a content item."""
    from punty.models.pick import Pick
    from sqlalchemy import select

    # Verify content exists
    from punty.models.content import Content
    content_result = await db.execute(select(Content).where(Content.id == content_id))
    if not content_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Content not found")

    updated = []
    for pick_id, fields in update.picks.items():
        result = await db.execute(
            select(Pick).where(Pick.id == pick_id, Pick.content_id == content_id)
        )
        pick = result.scalar_one_or_none()
        if not pick:
            continue

        if fields.horse_name is not None:
            pick.horse_name = fields.horse_name
        if fields.saddlecloth is not None:
            pick.saddlecloth = fields.saddlecloth
        if fields.odds_at_tip is not None:
            pick.odds_at_tip = fields.odds_at_tip
        if fields.bet_type is not None:
            pick.bet_type = fields.bet_type
        if fields.bet_stake is not None:
            pick.bet_stake = fields.bet_stake
        updated.append(pick_id)

    await db.commit()
    return {"updated": updated, "count": len(updated)}


@router.post("/{content_id}/reparse-picks")
async def reparse_content_picks(content_id: str, db: AsyncSession = Depends(get_db)):
    """Re-parse picks from the content's raw text. Deletes existing picks and re-extracts."""
    from punty.models.content import Content
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    if not content.raw_content:
        raise HTTPException(status_code=400, detail="Content has no raw text to parse")

    from punty.results.picks import store_picks_from_content
    count = await store_picks_from_content(db, content.id, content.meeting_id, content.raw_content)
    await db.commit()
    return {"status": "reparsed", "pick_count": count}


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
