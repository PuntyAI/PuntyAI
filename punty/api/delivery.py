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
    platform: str  # twitter
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


@router.get("/facebook/status")
async def facebook_status(db: AsyncSession = Depends(get_db)):
    """Check Facebook Page configuration status."""
    from punty.delivery.facebook import FacebookDelivery

    fb = FacebookDelivery(db)

    if not await fb.is_configured():
        return {
            "configured": False,
            "message": "Facebook not configured. Set Page ID and Access Token in Settings.",
        }

    page_info = await fb.get_page_info()
    return {"configured": True, "page": page_info}


@router.post("/facebook/post/{content_id}")
async def send_facebook_post(content_id: str, db: AsyncSession = Depends(get_db)):
    """Post content to Facebook Page."""
    from punty.delivery.facebook import FacebookDelivery

    fb = FacebookDelivery(db)

    try:
        result = await fb.send(content_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/facebook/post/{post_id}")
async def delete_facebook_post(post_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a Facebook post."""
    from punty.delivery.facebook import FacebookDelivery

    fb = FacebookDelivery(db)

    try:
        result = await fb.delete_post(post_id)
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
    if request.platform == "socials":
        # Post to both Twitter and Facebook
        results = {}
        from punty.delivery.twitter import TwitterDelivery
        from punty.delivery.facebook import FacebookDelivery

        twitter = TwitterDelivery(db)
        try:
            results["twitter"] = await twitter.send_long_post(request.content_id)
        except Exception as e:
            results["twitter"] = {"status": "error", "message": str(e)}

        fb = FacebookDelivery(db)
        try:
            results["facebook"] = await fb.send(request.content_id)
        except Exception as e:
            results["facebook"] = {"status": "error", "message": str(e)}

        # Check if either succeeded
        tw_ok = results["twitter"].get("status") != "error"
        fb_ok = results["facebook"].get("status") != "error"
        if not tw_ok and not fb_ok:
            raise HTTPException(
                status_code=500,
                detail=f"Both failed — Twitter: {results['twitter'].get('message')}, Facebook: {results['facebook'].get('message')}",
            )
        return {"status": "posted", "results": results}

    elif request.platform == "twitter":
        from punty.delivery.twitter import TwitterDelivery

        twitter = TwitterDelivery(db)
        try:
            result = await twitter.send_long_post(request.content_id)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    elif request.platform == "facebook":
        from punty.delivery.facebook import FacebookDelivery

        fb = FacebookDelivery(db)
        try:
            result = await fb.send(request.content_id)
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

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

    if platform == "twitter":
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
