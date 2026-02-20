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


class FacebookTokenExchange(BaseModel):
    """Request to exchange a short-lived token for a permanent page token."""

    short_lived_token: str
    app_id: str
    app_secret: str


@router.post("/facebook/exchange-token")
async def exchange_facebook_token(
    request: FacebookTokenExchange, db: AsyncSession = Depends(get_db)
):
    """Exchange a short-lived user token for a permanent Page Access Token.

    Flow:
    1. Exchange short-lived user token for long-lived user token
    2. Use long-lived user token to get permanent page access token
    3. Store the permanent page token in settings
    """
    import httpx
    from punty.models.settings import AppSettings
    from sqlalchemy import select

    base = "https://graph.facebook.com/v21.0"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Exchange short-lived for long-lived user token
        resp = await client.get(
            f"{base}/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": request.app_id,
                "client_secret": request.app_secret,
                "fb_exchange_token": request.short_lived_token,
            },
        )

        if resp.status_code != 200:
            error = resp.json().get("error", {}).get("message", resp.text)
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {error}")

        long_lived_token = resp.json().get("access_token")
        if not long_lived_token:
            raise HTTPException(status_code=400, detail="No access_token in exchange response")

        # Step 2: Get page access token using long-lived user token
        # First get the page ID from settings
        result = await db.execute(
            select(AppSettings).where(AppSettings.key == "facebook_page_id")
        )
        page_id_setting = result.scalar_one_or_none()
        if not page_id_setting or not page_id_setting.value:
            raise HTTPException(status_code=400, detail="Set facebook_page_id in Settings first")

        page_id = page_id_setting.value

        resp2 = await client.get(
            f"{base}/{page_id}",
            params={
                "fields": "name,access_token",
                "access_token": long_lived_token,
            },
        )

        if resp2.status_code != 200:
            error = resp2.json().get("error", {}).get("message", resp2.text)
            raise HTTPException(status_code=400, detail=f"Page token fetch failed: {error}")

        page_data = resp2.json()
        page_token = page_data.get("access_token")
        page_name = page_data.get("name", "Unknown")

        if not page_token:
            raise HTTPException(status_code=400, detail="No page access_token returned")

        # Step 3: Verify the token is permanent by debugging it
        resp3 = await client.get(
            f"{base}/debug_token",
            params={
                "input_token": page_token,
                "access_token": f"{request.app_id}|{request.app_secret}",
            },
        )
        debug_data = resp3.json().get("data", {})
        expires_at = debug_data.get("expires_at", -1)
        is_permanent = expires_at == 0

        # Step 4: Store the permanent page token
        result = await db.execute(
            select(AppSettings).where(AppSettings.key == "facebook_page_access_token")
        )
        token_setting = result.scalar_one_or_none()
        if token_setting:
            token_setting.value = page_token
        else:
            db.add(AppSettings(key="facebook_page_access_token", value=page_token))

        # Also store app credentials for future token debugging
        for key, value in [("facebook_app_id", request.app_id), ("facebook_app_secret", request.app_secret)]:
            result = await db.execute(select(AppSettings).where(AppSettings.key == key))
            setting = result.scalar_one_or_none()
            if setting:
                setting.value = value
            else:
                db.add(AppSettings(key=key, value=value))

        await db.commit()

        return {
            "status": "ok",
            "page_name": page_name,
            "page_id": page_id,
            "is_permanent": is_permanent,
            "expires_at": expires_at,
            "message": f"Permanent page token stored for {page_name}" if is_permanent else f"Token stored but expires_at={expires_at} (may not be permanent)",
        }


@router.post("/send")
async def send_content(request: SendRequest, db: AsyncSession = Depends(get_db)):
    """Send content to a platform."""
    from punty.models.content import Content, ContentStatus
    from sqlalchemy import select

    result = await db.execute(select(Content).where(Content.id == request.content_id))
    content = result.scalar_one_or_none()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    if content.status not in (ContentStatus.APPROVED.value, ContentStatus.SENT.value):
        raise HTTPException(
            status_code=400, detail="Content must be approved (or already sent) before sending"
        )

    if request.schedule_at:
        content.status = ContentStatus.SCHEDULED.value
        content.scheduled_send_at = request.schedule_at
        await db.commit()
        return {"status": "scheduled", "send_at": request.schedule_at}

    # Send immediately based on platform
    if request.platform == "socials":
        # Post to both Twitter and Facebook
        # Refresh content to ensure we have the latest status before each send
        results = {}
        from punty.delivery.twitter import TwitterDelivery
        from punty.delivery.facebook import FacebookDelivery

        # Save the approved status so a Twitter failure doesn't block Facebook
        saved_status = content.status

        twitter = TwitterDelivery(db)
        try:
            results["twitter"] = await twitter.send_long_post(request.content_id)
        except Exception as e:
            results["twitter"] = {"status": "error", "message": str(e)}
            # Restore approved status so Facebook can still send
            content.status = saved_status
            await db.flush()

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
