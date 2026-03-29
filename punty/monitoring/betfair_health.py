"""Betfair pipeline health monitoring.

1. Startup smoke test — verify JIT can rank runners
2. Heartbeat — alert if zero bets placed by afternoon
"""

import logging

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today, melb_now
from punty.models.database import async_session

logger = logging.getLogger(__name__)


async def jit_smoke_test(db: AsyncSession) -> dict:
    """Run at startup: verify the JIT pipeline can rank runners.

    Finds one race from today, runs probability, checks probs.get(runner.id) works.
    Returns {"ok": True/False, "detail": str}.
    """
    from punty.models.meeting import Meeting, Race, Runner
    from punty.probability import calculate_race_probabilities

    # Find a race from today with runners
    result = await db.execute(
        select(Race, Meeting).join(Meeting, Meeting.id == Race.meeting_id).where(
            Meeting.date == melb_today(),
            Meeting.selected == True,
        ).limit(1)
    )
    row = result.first()
    if not row:
        return {"ok": True, "detail": "No races today — smoke test skipped"}

    race, meeting = row
    runner_result = await db.execute(
        select(Runner).where(Runner.race_id == race.id, Runner.scratched != True).limit(10)
    )
    runners = runner_result.scalars().all()
    if len(runners) < 2:
        return {"ok": True, "detail": "Not enough runners for smoke test"}

    # Run probability engine
    try:
        probs = calculate_race_probabilities(runners, race, meeting)
    except Exception as e:
        return {"ok": False, "detail": f"Probability engine crashed: {e}"}

    if not probs:
        return {"ok": False, "detail": "Probability engine returned empty results"}

    # THE KEY CHECK: verify probs are keyed by runner.id
    matched = 0
    for runner in runners:
        prob = probs.get(runner.id) or probs.get(runner.horse_name)
        if prob and prob.win_probability > 0:
            matched += 1

    if matched == 0:
        # This is exactly the bug that cost us a full day
        sample_keys = list(probs.keys())[:3]
        sample_ids = [r.id for r in runners[:3]]
        return {
            "ok": False,
            "detail": (
                f"CRITICAL: probs.get(runner.id) returned None for ALL {len(runners)} runners. "
                f"Prob keys sample: {sample_keys}. "
                f"Runner ID sample: {sample_ids}. "
                f"JIT will skip every race!"
            ),
        }

    return {
        "ok": True,
        "detail": f"Smoke test passed: {matched}/{len(runners)} runners matched ({meeting.venue} R{race.race_number})",
    }


async def betfair_heartbeat():
    """Check if Betfair bets are being placed. Alert if not.

    Scheduled at 14:00 and 17:00 AEDT.
    """
    from punty.models.betfair_bet import BetfairBet
    from punty.models.meeting import Meeting, Race

    now = melb_now().replace(tzinfo=None)
    today = melb_today()

    async with async_session() as db:
        # Count races that have started
        started_result = await db.execute(
            select(func.count(Race.id)).join(Meeting, Meeting.id == Race.meeting_id).where(
                Meeting.date == today,
                Meeting.selected == True,
                Race.start_time < now,
            )
        )
        races_started = started_result.scalar() or 0

        # Get all betfair bets today
        bets_result = await db.execute(
            select(BetfairBet).where(
                BetfairBet.meeting_id.like(f"%-{today}"),
            )
        )
        all_bets = bets_result.scalars().all()

        placed = [b for b in all_bets if b.status in ("placed", "matched", "settled", "won", "lost")]
        skipped = [b for b in all_bets if b.status == "skipped"]
        evaluated = len(all_bets)

        # Only alert if enough races have run and zero bets placed
        if races_started < 5:
            logger.info(f"Betfair heartbeat: only {races_started} races started, too early to check")
            return

        if len(placed) > 0:
            logger.info(
                f"Betfair heartbeat OK: {len(placed)} bets placed, "
                f"{len(skipped)} skipped, {races_started} races started"
            )
            return

        # ALERT: Zero bets placed
        msg = f"BETFAIR HEARTBEAT FAIL\n"
        msg += f"Races started: {races_started}\n"
        msg += f"Bets placed: 0\n"
        msg += f"Evaluated: {evaluated}\n"
        msg += f"Skipped: {len(skipped)}\n"

        # Check if all skips have same error (like the prob key bug)
        if skipped:
            errors = {}
            for b in skipped:
                err = (b.error_message or "unknown")[:80]
                errors[err] = errors.get(err, 0) + 1
            top_error = max(errors.items(), key=lambda x: x[1])
            msg += f"Top skip reason ({top_error[1]}x): {top_error[0]}\n"

            if len(errors) == 1:
                msg += "ALL skips have same error — likely a code bug!\n"

        msg += "Check JIT pipeline immediately!"

        logger.error(f"Betfair heartbeat FAIL: {msg}")

        # Send Telegram alert
        try:
            from punty.telegram.bot import telegram_bot
            if telegram_bot and hasattr(telegram_bot, '_running') and telegram_bot._running:
                await telegram_bot.send_alert(f"⚠️ {msg}")
        except Exception as e:
            logger.warning(f"Failed to send heartbeat alert: {e}")
