"""API endpoints for content delivery."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.database import get_db

router = APIRouter()


class SendRequest(BaseModel):
    """Request to send content to a platform."""

    content_id: str
    platform: str  # whatsapp, twitter
    schedule_at: Optional[str] = None  # ISO datetime, or None for immediate


class TweetRequest(BaseModel):
    """Request to post a standalone tweet."""

    text: str


@router.get("/twitter/status")
async def twitter_status(db: AsyncSession = Depends(get_db)):
    """Check Twitter API configuration status."""
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    if not await twitter.is_configured():
        return {
            "configured": False,
            "message": "Twitter API not configured. Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET in .env",
        }

    # Try to get user info
    user_info = await twitter.get_me()
    return {
        "configured": True,
        "user": user_info,
    }


@router.post("/twitter/thread/{content_id}")
async def send_twitter_thread(content_id: str, db: AsyncSession = Depends(get_db)):
    """Post content as a Twitter thread."""
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    try:
        result = await twitter.send_thread(content_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/twitter/post/{content_id}")
async def send_twitter_long_post(content_id: str, db: AsyncSession = Depends(get_db)):
    """Post content as a single long-form post (for verified/premium accounts).

    Verified accounts can post up to 25,000 characters in a single post.
    """
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    try:
        result = await twitter.send_long_post(content_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twitter/thread/{content_id}/preview")
async def preview_twitter_thread(content_id: str, db: AsyncSession = Depends(get_db)):
    """Preview how content would be split into a Twitter thread."""
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    try:
        result = await twitter.preview_thread(content_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/twitter/tweet")
async def post_tweet(request: TweetRequest, db: AsyncSession = Depends(get_db)):
    """Post a standalone tweet."""
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    try:
        result = await twitter.post_tweet(request.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/twitter/tweet/{tweet_id}")
async def delete_tweet(tweet_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a tweet."""
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    try:
        result = await twitter.delete_tweet(tweet_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/send")
async def send_content(request: SendRequest, db: AsyncSession = Depends(get_db)):
    """Send content to a platform."""
    from punty.models.content import Content, ContentStatus
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == request.content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    if content.status != ContentStatus.APPROVED.value:
        raise HTTPException(
            status_code=400, detail="Content must be approved before sending"
        )

    if request.schedule_at:
        content.status = ContentStatus.SCHEDULED.value
        content.scheduled_send_at = request.schedule_at
        await db.commit()
        return {"status": "scheduled", "send_at": request.schedule_at}

    # Send immediately based on platform
    if request.platform == "twitter":
        from punty.delivery.twitter import TwitterDelivery

        twitter = TwitterDelivery(db)
        try:
            # Use thread for Early Mail (too long for single tweet)
            result = await twitter.send_thread(request.content_id)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    elif request.platform == "whatsapp":
        # WhatsApp delivery not yet implemented
        raise HTTPException(
            status_code=501, detail="WhatsApp delivery not yet implemented"
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {request.platform}")


@router.get("/preview/{content_id}")
async def preview_content(
    content_id: str, platform: str, db: AsyncSession = Depends(get_db)
):
    """Preview formatted content for a platform."""
    from punty.models.content import Content
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    if platform == "whatsapp":
        formatted = content.whatsapp_formatted or content.raw_content
        return {
            "content_id": content_id,
            "platform": platform,
            "formatted": formatted,
            "character_count": len(formatted) if formatted else 0,
        }
    elif platform == "twitter":
        # For Twitter, show thread preview
        from punty.delivery.twitter import TwitterDelivery

        twitter = TwitterDelivery(db)
        try:
            thread_preview = await twitter.preview_thread(content_id)
            return {
                "content_id": content_id,
                "platform": platform,
                "tweet_count": thread_preview["tweet_count"],
                "tweets": thread_preview["tweets"],
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")
