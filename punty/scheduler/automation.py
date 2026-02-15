"""Automated content approval and posting for scheduled jobs."""

import logging
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.content import Content, ContentStatus
from punty.models.meeting import Meeting, Race
from punty.models.pick import Pick

logger = logging.getLogger(__name__)


async def _delete_social_posts(content: Content, db: AsyncSession) -> None:
    """Delete Twitter and Facebook posts for superseded/unapproved content."""
    if content.twitter_id:
        try:
            from punty.delivery.twitter import TwitterDelivery
            twitter = TwitterDelivery(db)
            await twitter.delete_tweet(content.twitter_id)
            logger.info(f"Deleted tweet {content.twitter_id} for superseded content {content.id}")
            content.twitter_id = None
            content.sent_to_twitter = False
        except Exception as e:
            logger.warning(f"Could not delete tweet {content.twitter_id}: {e}")

    if content.facebook_id:
        try:
            from punty.delivery.facebook import FacebookDelivery
            fb = FacebookDelivery(db)
            await fb.delete_post(content.facebook_id)
            logger.info(f"Deleted Facebook post {content.facebook_id} for superseded content {content.id}")
            content.facebook_id = None
            content.sent_to_facebook = False
        except Exception as e:
            logger.warning(f"Could not delete Facebook post {content.facebook_id}: {e}")


async def validate_meeting_readiness(meeting_id: str, db: AsyncSession) -> tuple[bool, list[str]]:
    """Validate that a meeting has enough data for generation and publishing.

    Checks:
    - Meeting has races (not empty like Hobart)
    - Runners have odds from TAB (not blank like Kilcoy)
    - Speed maps are populated (at least partially)
    - Minimum active (non-scratched) runners per race

    Returns: (is_ready, list_of_issues)
    """
    from punty.models.meeting import Runner

    issues = []

    # Check meeting exists
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        return False, [f"Meeting not found: {meeting_id}"]

    # Check races exist
    result = await db.execute(select(Race).where(Race.meeting_id == meeting_id))
    races = result.scalars().all()

    if not races:
        return False, [f"No races found for {meeting.venue}"]

    # Check runners have odds — load all runners and count in Python
    # (SQLite doesn't handle complex aggregates well)
    total_runners = 0
    runners_with_odds = 0
    runners_with_speedmap = 0
    active_runners = 0

    for race in races:
        result = await db.execute(select(Runner).where(Runner.race_id == race.id))
        runners = result.scalars().all()
        for r in runners:
            total_runners += 1
            is_active = not r.scratched
            if is_active:
                active_runners += 1
                if r.current_odds is not None:
                    runners_with_odds += 1
                if r.speed_map_position and r.speed_map_position.strip():
                    runners_with_speedmap += 1

    if active_runners == 0:
        return False, [f"No active runners for {meeting.venue}"]

    # Odds check: at least 50% of active runners should have odds
    odds_pct = (runners_with_odds / active_runners * 100) if active_runners > 0 else 0
    if odds_pct < 50:
        issues.append(
            f"Only {runners_with_odds}/{active_runners} runners ({odds_pct:.0f}%) have odds — "
            f"TAB may not cover this venue"
        )

    # Speed map check: at least 30% coverage (some venues don't have maps)
    speedmap_pct = (runners_with_speedmap / active_runners * 100) if active_runners > 0 else 0
    if speedmap_pct < 30:
        issues.append(
            f"Only {runners_with_speedmap}/{active_runners} runners ({speedmap_pct:.0f}%) have speed maps"
        )

    # Minimum races check (at least 4 for a viable meeting)
    if len(races) < 4:
        issues.append(f"Only {len(races)} races — minimum 4 for a viable meeting")

    is_ready = len(issues) == 0
    if not is_ready:
        logger.warning(
            f"Meeting {meeting.venue} ({meeting_id}) failed readiness check: {issues}"
        )
    else:
        logger.info(
            f"Meeting {meeting.venue} passed readiness: {len(races)} races, "
            f"{active_runners} active runners, {odds_pct:.0f}% odds, {speedmap_pct:.0f}% speed maps"
        )

    return is_ready, issues


async def validate_early_mail(content: Content, db: AsyncSession) -> tuple[bool, list[str]]:
    """Validate early mail content before auto-approval.

    Checks:
    - Has required sections (Big 3, Race-by-Race, Sequences)
    - All races covered
    - No placeholder text
    - Reasonable length
    - Odds are present

    Returns: (is_valid, list_of_issues)
    """
    issues = []
    raw = content.raw_content or ""

    # Check minimum length
    if len(raw) < 2000:
        issues.append(f"Content too short ({len(raw)} chars, need 2000+)")

    # Check for placeholder text
    placeholders = re.findall(r'\{[A-Z_]+\}', raw)
    if placeholders:
        issues.append(f"Contains placeholders: {placeholders[:5]}")

    # Check for Big 3 section
    if not re.search(r"(?i)(big\s*3|punty'?s?\s*big\s*3)", raw):
        issues.append("Missing Big 3 section")

    # Check for Sequence Lanes section
    if not re.search(r"(?i)(sequence|quaddie|big\s*6)", raw):
        issues.append("Missing Sequence Lanes section")

    # Get race count for this meeting
    result = await db.execute(
        select(Race).where(Race.meeting_id == content.meeting_id)
    )
    races = result.scalars().all()
    race_count = len(races)

    if race_count > 0:
        # Check that each race is covered
        for race in races:
            race_pattern = rf"(?i)race\s*{race.race_number}\b"
            if not re.search(race_pattern, raw):
                issues.append(f"Race {race.race_number} not covered")

    # Check for odds (should have multiple $X.XX patterns)
    odds_matches = re.findall(r'\$\d+\.\d{2}', raw)
    if len(odds_matches) < 10:
        issues.append(f"Too few odds found ({len(odds_matches)}, need 10+)")

    # Check for saddlecloth numbers (No.X pattern)
    saddlecloth_matches = re.findall(r'No\.?\s*\d+', raw)
    if len(saddlecloth_matches) < 5:
        issues.append(f"Too few saddlecloth numbers ({len(saddlecloth_matches)})")

    # Probability-based validation (picks vs race data)
    try:
        from punty.validation.content_validator import validate_early_mail_probability
        prob_result = await validate_early_mail_probability(raw, content.meeting_id, db)
        for issue in prob_result.errors:
            issues.append(f"R{issue.race_number} [{issue.category}]: {issue.message}")
        # Log warnings but don't block approval
        for issue in prob_result.warnings:
            logger.info(f"Validation warning R{issue.race_number}: {issue.message}")
    except Exception as e:
        logger.debug(f"Probability validation skipped: {e}")

    is_valid = len(issues) == 0
    return is_valid, issues


async def validate_wrapup(content: Content, db: AsyncSession) -> tuple[bool, list[str]]:
    """Validate wrap-up content before auto-approval.

    Checks:
    - References correct venue
    - Has results/ledger section
    - Has Quick Hits section
    - No placeholder text
    - Reasonable length

    Returns: (is_valid, list_of_issues)
    """
    issues = []
    raw = content.raw_content or ""

    # Check minimum length
    if len(raw) < 1000:
        issues.append(f"Content too short ({len(raw)} chars, need 1000+)")

    # Check for placeholder text
    placeholders = re.findall(r'\{[A-Z_]+\}', raw)
    if placeholders:
        issues.append(f"Contains placeholders: {placeholders[:5]}")

    # Get venue name
    result = await db.execute(
        select(Meeting).where(Meeting.id == content.meeting_id)
    )
    meeting = result.scalar_one_or_none()
    if meeting:
        # Check venue is mentioned
        venue_pattern = re.escape(meeting.venue)
        if not re.search(venue_pattern, raw, re.IGNORECASE):
            issues.append(f"Venue '{meeting.venue}' not mentioned")

    # Check for Punty Ledger section
    if not re.search(r"(?i)(punty\s*ledger|ledger)", raw):
        issues.append("Missing Punty Ledger section")

    # Check for Quick Hits section
    if not re.search(r"(?i)(quick\s*hits|race.by.race)", raw):
        issues.append("Missing Quick Hits section")

    # Check for P&L figures (should have +$X or -$X patterns)
    pnl_matches = re.findall(r'[+-]\$\d+', raw)
    if len(pnl_matches) < 3:
        issues.append(f"Too few P&L figures ({len(pnl_matches)})")

    is_valid = len(issues) == 0
    return is_valid, issues


async def validate_weekly_blog(content: Content, db: AsyncSession) -> tuple[bool, list[str]]:
    """Validate weekly blog content before auto-approval.

    Checks:
    - Min 3000 chars
    - Required sections present
    - Contains Gamble Responsibly footer
    - No placeholder text

    Returns: (is_valid, list_of_issues)
    """
    issues = []
    raw = content.raw_content or ""

    if len(raw) < 1500:
        issues.append(f"Blog too short ({len(raw)} chars, need 1500+)")

    # Check for placeholder text
    placeholders = re.findall(r'\{[A-Z_]+\}', raw)
    if placeholders:
        issues.append(f"Contains placeholders: {placeholders[:5]}")

    # Required sections
    required = [
        (r"(?i)punty\s*awards", "PUNTY AWARDS"),
        (r"(?i)(crystal\s*ball|upcoming\s*group)", "CRYSTAL BALL"),
        (r"(?i)pattern\s*spotlight", "PATTERN SPOTLIGHT"),
        (r"(?i)(the\s*ledger|weekly\s*p&l|weekly\s*ledger)", "THE LEDGER"),
    ]
    for pattern, name in required:
        if not re.search(pattern, raw):
            issues.append(f"Missing section: {name}")

    # Must contain Gamble Responsibly
    if not re.search(r"(?i)gamble\s*responsibly", raw):
        issues.append("Missing 'Gamble Responsibly' footer")

    is_valid = len(issues) == 0
    return is_valid, issues


async def auto_approve_content(content_id: str, db: AsyncSession) -> dict:
    """Auto-approve content after validation.

    This mimics the manual approval flow:
    1. Set status to APPROVED
    2. Store picks from content (for early_mail)
    3. Store picks as memories

    Returns: dict with approval status and any issues
    """
    from punty.results.picks import store_picks_from_content, store_picks_as_memories
    from punty.models.pick import Pick
    from sqlalchemy import delete as sa_delete

    result = await db.execute(select(Content).where(Content.id == content_id))
    content = result.scalar_one_or_none()

    if not content:
        return {"status": "error", "message": f"Content not found: {content_id}"}

    # Validate based on content type
    if content.content_type == "early_mail":
        is_valid, issues = await validate_early_mail(content, db)
    elif content.content_type == "meeting_wrapup":
        is_valid, issues = await validate_wrapup(content, db)
    elif content.content_type == "weekly_blog":
        is_valid, issues = await validate_weekly_blog(content, db)
    else:
        is_valid, issues = True, []

    if not is_valid:
        logger.warning(f"Auto-approval validation failed for {content_id}: {issues}")
        return {
            "status": "validation_failed",
            "content_id": content_id,
            "issues": issues,
        }

    # Approve content
    content.status = ContentStatus.APPROVED
    content.review_notes = "Auto-approved by scheduler"

    # Store picks for early mail
    if content.content_type == "early_mail" and content.raw_content:
        # Supersede any previously approved early_mail for this meeting
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
            # Delete old social media posts
            await _delete_social_posts(old, db)

        try:
            await store_picks_from_content(db, content.id, content.meeting_id, content.raw_content)
            await store_picks_as_memories(db, content.meeting_id, content.id)
        except Exception as e:
            logger.error(f"Failed to store picks for {content_id}: {e}")

    await db.commit()

    logger.info(f"Auto-approved content: {content_id}")
    return {
        "status": "approved",
        "content_id": content_id,
    }


async def auto_post_to_twitter(content_id: str, db: AsyncSession) -> dict:
    """Post approved content to Twitter.

    Returns: dict with post status
    """
    from punty.delivery.twitter import TwitterDelivery

    twitter = TwitterDelivery(db)

    if not await twitter.is_configured():
        logger.warning("Twitter not configured, skipping auto-post")
        return {"status": "skipped", "reason": "Twitter not configured"}

    try:
        # Use long-form post for early mail (usually > 280 chars)
        result = await twitter.send_long_post(content_id)
        logger.info(f"Auto-posted to Twitter: {content_id} -> {result.get('tweet_id')}")
        return result
    except Exception as e:
        logger.error(f"Twitter auto-post failed for {content_id}: {e}")
        return {"status": "error", "message": str(e)}


async def auto_post_to_facebook(content_id: str, db: AsyncSession) -> dict:
    """Post approved content to Facebook Page.

    Returns: dict with post status
    """
    from punty.delivery.facebook import FacebookDelivery

    fb = FacebookDelivery(db)

    if not await fb.is_configured():
        logger.warning("Facebook not configured, skipping auto-post")
        return {"status": "skipped", "reason": "Facebook not configured"}

    try:
        result = await fb.send(content_id)
        logger.info(f"Auto-posted to Facebook: {content_id} -> {result.get('post_id')}")
        return result
    except Exception as e:
        logger.error(f"Facebook auto-post failed for {content_id}: {e}")
        return {"status": "error", "message": str(e)}


async def auto_approve_and_post(content_id: str, db: AsyncSession) -> dict:
    """Full automation: validate, approve, and post to Twitter and Facebook.

    Returns: combined result dict
    """
    from punty.scheduler.activity_log import log_system

    # Step 1: Auto-approve
    approve_result = await auto_approve_content(content_id, db)

    if approve_result["status"] != "approved":
        log_system(f"Auto-approval failed: {approve_result.get('issues', [])}", status="warning")
        return approve_result

    # Step 2: Post to Twitter
    twitter_result = await auto_post_to_twitter(content_id, db)

    # Step 3: Post to Facebook
    facebook_result = await auto_post_to_facebook(content_id, db)

    # Check for delivery failures and alert
    failures = []
    if twitter_result.get("status") == "error":
        failures.append(f"Twitter: {twitter_result.get('message', 'unknown error')}")
    if facebook_result.get("status") == "error":
        failures.append(f"Facebook: {facebook_result.get('message', 'unknown error')}")

    if failures:
        failure_msg = "; ".join(failures)
        log_system(f"Auto-approved {content_id} but delivery failed — {failure_msg}", status="warning")
        await _send_delivery_failure_alert(db, content_id, failures)
    else:
        log_system(f"Auto-approved and posted: {content_id}", status="success")

    return {
        "status": "complete",
        "content_id": content_id,
        "approval": approve_result,
        "twitter": twitter_result,
        "facebook": facebook_result,
    }


async def _send_delivery_failure_alert(db: AsyncSession, content_id: str, failures: list[str]):
    """Send a Telegram alert when auto-delivery fails."""
    try:
        from punty.telegram.bot import telegram_bot
        if telegram_bot and telegram_bot._running:
            msg = f"⚠️ Delivery failed for {content_id[:8]}...\n"
            for f in failures:
                msg += f"• {f}\n"
            await telegram_bot.send_alert(msg)
    except Exception as e:
        logger.error(f"Failed to send delivery alert: {e}")


async def check_all_settled(meeting_id: str, db: AsyncSession) -> tuple[bool, int, int]:
    """Check if all picks for a meeting are settled.

    Returns: (all_settled, settled_count, total_count)
    """
    result = await db.execute(
        select(Pick).where(Pick.meeting_id == meeting_id)
    )
    picks = result.scalars().all()

    if not picks:
        return True, 0, 0

    total = len(picks)
    settled = sum(1 for p in picks if p.settled)

    return settled == total, settled, total
