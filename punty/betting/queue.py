"""Bet queue business logic — populate, execute, and settle Betfair bets."""

import logging
import math
from datetime import timedelta
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive, melb_now
from punty.models.betfair_bet import BetfairBet
from punty.models.pick import Pick
from punty.models.meeting import Meeting, Race, Runner
from punty.models.settings import AppSettings

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_INITIAL_BALANCE = 50.0
DEFAULT_BASE_STAKE = 2.0
DEFAULT_MIN_ODDS = 1.10
DEFAULT_COMMISSION_RATE = 0.05  # 5% Betfair commission
DEFAULT_MAX_DAILY_LOSS = -20.0


async def _get_setting(db: AsyncSession, key: str, default: str = "") -> str:
    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()
    return setting.value if setting and setting.value else default


async def get_balance(db: AsyncSession) -> float:
    val = await _get_setting(db, "betfair_balance", str(DEFAULT_INITIAL_BALANCE))
    return float(val)


async def set_balance(db: AsyncSession, balance: float) -> None:
    result = await db.execute(select(AppSettings).where(AppSettings.key == "betfair_balance"))
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = str(round(balance, 2))
    else:
        db.add(AppSettings(key="betfair_balance", value=str(round(balance, 2)),
                           description="Betfair auto-bet balance tracker"))
    await db.commit()


def calculate_stake(balance: float, initial_balance: float = DEFAULT_INITIAL_BALANCE,
                    base_stake: float = DEFAULT_BASE_STAKE) -> float:
    """Calculate current stake based on balance growth.

    Doubles stake every time balance doubles from initial:
    - $50-99: $2
    - $100-199: $4
    - $200-399: $8
    - etc.
    """
    if balance <= 0 or initial_balance <= 0:
        return base_stake
    ratio = balance / initial_balance
    if ratio < 1:
        return base_stake  # Don't reduce below base when in drawdown
    doublings = int(math.log2(ratio))
    return base_stake * (2 ** doublings)


async def populate_bet_queue(
    db: AsyncSession,
    meeting_id: str,
    content_id: str,
) -> int:
    """Create BetfairBet entries for rank 1 selection picks from approved content.

    Called after content approval. Creates one bet per race for the top-ranked pick.
    Returns count of bets queued.
    """
    auto_enabled = await _get_setting(db, "betfair_auto_bet_enabled", "true")
    if auto_enabled.lower() != "true":
        return 0

    # Get rank 1 selection picks for this content (non-tracked-only)
    result = await db.execute(
        select(Pick).where(
            Pick.content_id == content_id,
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "selection",
            Pick.tip_rank == 1,
            Pick.tracked_only != True,
        )
    )
    rank1_picks = result.scalars().all()
    if not rank1_picks:
        return 0

    # Load races to get start times
    race_result = await db.execute(
        select(Race).where(Race.meeting_id == meeting_id)
    )
    races_by_num = {r.race_number: r for r in race_result.scalars().all()}

    # Get current balance for stake calculation
    balance = await get_balance(db)
    initial_balance = float(await _get_setting(db, "betfair_initial_balance", str(DEFAULT_INITIAL_BALANCE)))
    base_stake = float(await _get_setting(db, "betfair_stake", str(DEFAULT_BASE_STAKE)))
    stake = calculate_stake(balance, initial_balance, base_stake)

    queued = 0
    for pick in rank1_picks:
        race = races_by_num.get(pick.race_number)
        if not race or not race.start_time:
            logger.debug(f"Skipping bet for {meeting_id} R{pick.race_number}: no start time")
            continue

        # Check if already queued (unique constraint)
        existing = await db.execute(
            select(BetfairBet).where(
                BetfairBet.meeting_id == meeting_id,
                BetfairBet.race_number == pick.race_number,
            )
        )
        if existing.scalar_one_or_none():
            continue

        # Check runner not scratched
        runner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == race.id,
                Runner.saddlecloth == pick.saddlecloth,
            )
        )
        runner = runner_result.scalar_one_or_none()
        if runner and runner.scratched:
            logger.info(f"Skipping bet for {pick.horse_name} R{pick.race_number}: scratched")
            continue

        bet_id = f"bf-{meeting_id}-r{pick.race_number}"
        scheduled_at = race.start_time - timedelta(minutes=5)

        bet = BetfairBet(
            id=bet_id,
            pick_id=pick.id,
            meeting_id=meeting_id,
            race_number=pick.race_number,
            horse_name=pick.horse_name or "Unknown",
            saddlecloth=pick.saddlecloth,
            stake=stake,
            requested_odds=pick.place_odds_at_tip,
            scheduled_at=scheduled_at,
        )
        db.add(bet)
        queued += 1

    if queued:
        await db.commit()
        logger.info(f"Betfair queue: {queued} bets queued for {meeting_id} (stake=${stake:.2f})")
    return queued


async def execute_due_bets(db: AsyncSession) -> int:
    """Find and execute bets whose scheduled_at has arrived.

    Called every 30s by the scheduler. Processes bets where:
    - status = 'queued'
    - enabled = True
    - scheduled_at <= now
    - scheduled_at > now - 10min (don't bet on races that started 5+ min ago)
    """
    from punty.betting.betfair_client import resolve_place_market, get_place_odds, place_bet

    now = melb_now_naive()
    cutoff = now - timedelta(minutes=10)

    result = await db.execute(
        select(BetfairBet).where(
            BetfairBet.status == "queued",
            BetfairBet.enabled == True,
            BetfairBet.scheduled_at <= now,
            BetfairBet.scheduled_at > cutoff,
        )
    )
    due_bets = result.scalars().all()
    if not due_bets:
        return 0

    # Safety: check daily P&L
    max_daily_loss = float(await _get_setting(db, "betfair_max_daily_loss", str(DEFAULT_MAX_DAILY_LOSS)))
    today_pnl = await _get_today_pnl(db)
    if today_pnl <= max_daily_loss:
        logger.warning(f"Betfair: daily P&L ${today_pnl:.2f} hit limit ${max_daily_loss:.2f} — pausing bets")
        return 0

    # Safety: check balance
    balance = await get_balance(db)
    min_odds = float(await _get_setting(db, "betfair_min_odds", str(DEFAULT_MIN_ODDS)))

    placed = 0
    for bet in due_bets:
        if balance < bet.stake:
            bet.status = "cancelled"
            bet.error_message = f"Insufficient balance (${balance:.2f} < ${bet.stake:.2f})"
            logger.warning(f"Betfair: skipping {bet.id} — insufficient balance")
            continue

        # Check horse not scratched since queuing
        runner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == f"{bet.meeting_id}-r{bet.race_number}",
                Runner.saddlecloth == bet.saddlecloth,
            )
        )
        runner = runner_result.scalar_one_or_none()
        if runner and runner.scratched:
            bet.status = "cancelled"
            bet.error_message = "Horse scratched"
            logger.info(f"Betfair: {bet.id} cancelled — {bet.horse_name} scratched")
            continue

        # Resolve market
        meeting_result = await db.execute(select(Meeting).where(Meeting.id == bet.meeting_id))
        meeting = meeting_result.scalar_one_or_none()
        if not meeting:
            bet.status = "failed"
            bet.error_message = "Meeting not found"
            continue

        bet.status = "placing"
        await db.commit()

        market = await resolve_place_market(
            db, meeting.venue, meeting.date, bet.meeting_id, bet.race_number
        )
        if not market:
            bet.status = "failed"
            bet.error_message = "Could not resolve Betfair place market"
            await db.commit()
            continue

        bet.market_id = market["market_id"]

        # Match horse by name
        selection_id = None
        from punty.scrapers.betfair import _normalize_name
        target_name = _normalize_name(bet.horse_name)
        for runner_info in market["runners"]:
            if _normalize_name(runner_info["horse_name"]) == target_name:
                selection_id = runner_info["selection_id"]
                break

        if not selection_id:
            bet.status = "failed"
            bet.error_message = f"Horse '{bet.horse_name}' not found in Betfair market"
            await db.commit()
            continue

        bet.selection_id = selection_id

        # Get current odds
        current_odds = await get_place_odds(db, market["market_id"], selection_id)
        if not current_odds or current_odds < min_odds:
            bet.status = "cancelled"
            bet.error_message = f"Odds too low: ${current_odds}" if current_odds else "No odds available"
            await db.commit()
            continue

        bet.requested_odds = current_odds

        # Place the bet
        result = await place_bet(db, market["market_id"], selection_id, bet.stake, current_odds)

        if result.get("status") == "SUCCESS" or result.get("bet_id"):
            bet.status = "placed"
            bet.bet_id = result.get("bet_id")
            bet.size_matched = result.get("size_matched", 0)
            bet.average_price_matched = result.get("average_price_matched", 0)
            bet.matched_odds = result.get("average_price_matched", current_odds)
            bet.placed_at = melb_now_naive()
            placed += 1
            # Deduct stake from tracked balance
            balance -= bet.stake
            await set_balance(db, balance)
            logger.info(
                f"Betfair: placed {bet.id} — {bet.horse_name} ${bet.stake} "
                f"@ {bet.matched_odds} (balance: ${balance:.2f})"
            )
        else:
            bet.status = "failed"
            bet.error_message = result.get("error", "Unknown error")
            logger.error(f"Betfair: failed {bet.id} — {bet.error_message}")

    await db.commit()
    return placed


async def settle_betfair_bets(
    db: AsyncSession,
    meeting_id: str,
    race_number: int,
) -> int:
    """Settle Betfair bets after race results are in.

    Hit = finish position 1-3 (place). P&L accounts for 5% Betfair commission.
    """
    result = await db.execute(
        select(BetfairBet).where(
            BetfairBet.meeting_id == meeting_id,
            BetfairBet.race_number == race_number,
            BetfairBet.status == "placed",
            BetfairBet.settled == False,
        )
    )
    bet = result.scalar_one_or_none()
    if not bet:
        return 0

    # Find the runner's finish position
    race_id = f"{meeting_id}-r{race_number}"
    runner_result = await db.execute(
        select(Runner).where(
            Runner.race_id == race_id,
            Runner.saddlecloth == bet.saddlecloth,
        )
    )
    runner = runner_result.scalar_one_or_none()
    if not runner or runner.finish_position is None:
        return 0  # Results not in yet

    commission_rate = float(await _get_setting(db, "betfair_commission_rate", str(DEFAULT_COMMISSION_RATE)))
    odds = bet.matched_odds or bet.requested_odds or 0

    now = melb_now_naive()
    if runner.finish_position <= 3:
        # Place hit — profit after commission
        gross_profit = (odds - 1) * bet.stake
        commission = gross_profit * commission_rate
        bet.pnl = round(gross_profit - commission, 2)
        bet.hit = True
    else:
        # Miss — lose stake (already deducted)
        bet.pnl = round(-bet.stake, 2)
        bet.hit = False

    bet.settled = True
    bet.settled_at = now
    bet.status = "settled"

    # Update balance: stake was deducted at placement, now add back stake + net profit if hit
    balance = await get_balance(db)
    if bet.hit:
        balance += bet.stake + bet.pnl  # Return stake + net profit
    # If miss, stake was already deducted — nothing to do
    await set_balance(db, balance)

    await db.commit()
    logger.info(
        f"Betfair settled: {bet.id} — {'HIT' if bet.hit else 'MISS'} "
        f"pnl=${bet.pnl:+.2f} (balance: ${balance:.2f})"
    )
    return 1


async def _get_today_pnl(db: AsyncSession) -> float:
    """Get today's total P&L from settled Betfair bets."""
    from punty.config import melb_today
    today = melb_today()
    result = await db.execute(
        select(func.coalesce(func.sum(BetfairBet.pnl), 0.0)).where(
            BetfairBet.settled == True,
            BetfairBet.meeting_id.like(f"%-{today.isoformat()}%"),
        )
    )
    return float(result.scalar())


async def get_queue_summary(db: AsyncSession) -> dict:
    """Get summary stats for the Betfair betting queue."""
    from punty.config import melb_today
    today = melb_today()

    # Today's bets
    result = await db.execute(
        select(BetfairBet).where(
            BetfairBet.meeting_id.like(f"%-{today.isoformat()}%"),
        ).order_by(BetfairBet.scheduled_at)
    )
    today_bets = result.scalars().all()

    # All-time settled stats
    all_result = await db.execute(
        select(
            func.count(BetfairBet.id),
            func.coalesce(func.sum(BetfairBet.pnl), 0.0),
            func.sum(func.cast(BetfairBet.hit == True, Integer)),
        ).where(BetfairBet.settled == True).select_from(BetfairBet)
    )
    row = all_result.one()
    total_bets = row[0] or 0
    total_pnl = float(row[1] or 0)
    total_hits = int(row[2] or 0)

    balance = await get_balance(db)
    initial_balance = float(await _get_setting(db, "betfair_initial_balance", str(DEFAULT_INITIAL_BALANCE)))
    stake = calculate_stake(balance, initial_balance)

    return {
        "balance": balance,
        "initial_balance": initial_balance,
        "current_stake": stake,
        "today_bets": [b.to_dict() for b in today_bets],
        "today_pnl": sum(b.pnl or 0 for b in today_bets if b.settled),
        "today_placed": sum(1 for b in today_bets if b.status in ("placed", "settled")),
        "today_queued": sum(1 for b in today_bets if b.status == "queued"),
        "total_bets": total_bets,
        "total_pnl": total_pnl,
        "total_hits": total_hits,
        "strike_rate": round(total_hits / total_bets * 100, 1) if total_bets else 0,
        "roi": round(total_pnl / (total_bets * 2) * 100, 1) if total_bets else 0,  # Approximate
    }
