"""Just-In-Time Betfair betting — evaluate and bet 5 minutes before each race.

Replaces the pre-queue → refresh → execute pipeline with a single function
that runs fresh probability with all available data (KASH + live odds)
and places a bet if gates pass.

Called by BetfairBetScheduler._tick() when a race enters the T-5min window.
"""

import logging
from datetime import timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.models.betfair_bet import BetfairBet
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick
from punty.models.settings import AppSettings

logger = logging.getLogger(__name__)

# Gates
PP_FLOOR = 0.59
WIN_GAP_THRESHOLD = 0.15
MIN_PLACE_ODDS = 1.30
MIN_WIN_ODDS = 1.50
MAX_EDGE_CAP = 0.10


async def _get_setting(db, key: str, default: str = "") -> str:
    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    s = result.scalar_one_or_none()
    return s.value if s and s.value else default


async def evaluate_and_bet_race(
    db: AsyncSession,
    meeting_id: str,
    race_number: int,
) -> dict:
    """JIT race evaluation: fresh probability -> gates -> Kelly -> place bet.

    Called by the scheduler at T-5min for each upcoming race.
    Creates a BetfairBet row regardless of outcome (for audit trail).

    Returns dict with action, reason, and bet details.
    """
    now = melb_now_naive()
    result = {"action": "skipped", "reason": "", "horse_name": None,
              "bet_type": None, "stake": 0, "meeting_id": meeting_id,
              "race_number": race_number}

    # ── Pre-conditions ──
    auto_enabled = await _get_setting(db, "betfair_auto_bet_enabled", "true")
    if auto_enabled.lower() != "true":
        result["reason"] = "Auto-bet disabled"
        return result

    # Load meeting
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or not meeting.selected:
        result["reason"] = "Meeting not found or not selected"
        return result

    # Load race
    race_id = f"{meeting_id}-r{race_number}"
    race_result = await db.execute(select(Race).where(Race.id == race_id))
    race = race_result.scalar_one_or_none()
    if not race or not race.start_time:
        result["reason"] = "Race not found or no start time"
        return result

    # Check not already evaluated
    existing = await db.execute(
        select(BetfairBet).where(
            BetfairBet.meeting_id == meeting_id,
            BetfairBet.race_number == race_number,
        )
    )
    if existing.scalar_one_or_none():
        result["reason"] = "Already evaluated"
        return result

    # Load active runners
    runner_result = await db.execute(
        select(Runner).where(
            Runner.race_id == race_id,
            Runner.scratched != True,
        )
    )
    runners = runner_result.scalars().all()
    field_size = len(runners)

    # NTD check
    if field_size < 8:
        result["reason"] = f"NTD ({field_size} runners)"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now)
        return result

    # Daily P&L check
    from punty.betting.queue import get_balance, _get_today_pnl, DEFAULT_MAX_DAILY_LOSS
    max_loss = float(await _get_setting(db, "betfair_max_daily_loss", str(DEFAULT_MAX_DAILY_LOSS)))
    today_pnl = await _get_today_pnl(db)
    if today_pnl <= max_loss:
        result["reason"] = f"Daily loss limit hit (${today_pnl:.2f})"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now)
        return result

    balance = await get_balance(db)
    if balance < 2.0:
        result["reason"] = f"Insufficient balance (${balance:.2f})"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now)
        return result

    # ── Run FRESH probability ──
    from punty.probability import calculate_race_probabilities
    try:
        probs = calculate_race_probabilities(
            runners, race, meeting,
        )
    except Exception as e:
        result["reason"] = f"Probability engine error: {e}"
        result["action"] = "error"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now)
        return result

    if not probs:
        result["reason"] = "No probabilities computed"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now)
        return result

    # ── Rank runners by PP, select R1 ──
    ranked = []
    for runner in runners:
        key = runner.horse_name
        prob = probs.get(key)
        if not prob:
            continue
        wp = prob.win_probability
        pp = prob.place_probability
        if runner.horse_age and runner.horse_age >= 6:
            continue  # Age gate
        ranked.append({
            "runner": runner,
            "wp": wp,
            "pp": pp,
            "odds": runner.current_odds or 0,
            "kash_rp": runner.kash_rated_price,
        })

    ranked.sort(key=lambda x: x["pp"], reverse=True)

    if not ranked:
        result["reason"] = "No eligible runners after age filter"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now)
        return result

    r1 = ranked[0]
    r2 = ranked[1] if len(ranked) > 1 else None
    runner = r1["runner"]
    wp = r1["wp"]
    pp = r1["pp"]
    odds = r1["odds"]

    result["horse_name"] = runner.horse_name

    # ── PP floor gate ──
    if pp < PP_FLOOR:
        result["reason"] = f"PP {pp:.0%} < {PP_FLOOR:.0%}"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                  horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                  jit_wp=wp, jit_pp=pp)
        return result

    # ── Bet type routing (WP gap) ──
    r2_wp = r2["wp"] if r2 else 0
    wp_gap = wp - r2_wp

    if wp_gap >= WIN_GAP_THRESHOLD and odds > 1.0:
        bet_type = "win"
        min_odds = MIN_WIN_ODDS
    else:
        bet_type = "place"
        min_odds = MIN_PLACE_ODDS

    result["bet_type"] = bet_type

    # ── 4-Model Sense Check (Us vs PF vs Market vs KASH) ──
    from punty.sense_check import sense_check_race
    sense = sense_check_race(runner.saddlecloth, runners)
    consensus_mult = sense["kelly_mult"]

    if sense["action"] == "skip":
        result["reason"] = f"Sense check SKIP: {sense['detail']}"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                  horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                  jit_wp=wp, jit_pp=pp)
        logger.info(f"JIT SKIP (outlier): {runner.horse_name} {meeting.venue} R{race_number} — {sense['detail']}")
        return result

    logger.info(
        f"JIT Sense: {runner.horse_name} R{race_number} {sense['consensus']} "
        f"({sense['detail']}) kelly_mult={consensus_mult}"
    )

    # ── Resolve Betfair market + get live odds ──
    from punty.betting.betfair_client import resolve_place_market, resolve_win_market, get_place_odds

    if bet_type == "win":
        market = await resolve_win_market(db, meeting.venue, meeting.date, meeting_id, race_number)
    else:
        market = await resolve_place_market(db, meeting.venue, meeting.date, meeting_id, race_number)
        if not market:
            market = await resolve_win_market(db, meeting.venue, meeting.date, meeting_id, race_number)
            if market:
                bet_type = "win"
                min_odds = MIN_WIN_ODDS
                result["bet_type"] = "win"

    if not market:
        result["reason"] = "No Betfair market available"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                  horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                  jit_wp=wp, jit_pp=pp)
        return result

    # Match horse in market
    from punty.scrapers.betfair import _normalize_name
    selection_id = None
    target = _normalize_name(runner.horse_name)
    for ri in market["runners"]:
        if _normalize_name(ri["horse_name"]) == target:
            selection_id = ri["selection_id"]
            break

    if not selection_id:
        result["reason"] = f"Horse '{runner.horse_name}' not found in Betfair market"
        await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                  horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                  jit_wp=wp, jit_pp=pp)
        return result

    # Get live exchange odds for Kelly
    live_odds = await get_place_odds(db, market["market_id"], selection_id)
    if not live_odds or live_odds <= 1.0:
        from punty.betting.queue import _estimate_place_odds
        live_odds = _estimate_place_odds(odds) if bet_type == "place" else (odds if odds > 1 else 2.0)

    # ── Kelly stake ──
    from punty.betting.queue import calculate_kelly_stake, get_balance as _get_bal
    kelly_prob = wp if bet_type == "win" else pp
    from punty.betting.queue import DEFAULT_MAX_KELLY_FRACTION, DEFAULT_MIN_KELLY_STAKE, DEFAULT_KELLY_HALF
    max_frac = float(await _get_setting(db, "betfair_max_kelly_fraction", str(DEFAULT_MAX_KELLY_FRACTION)))

    stake = calculate_kelly_stake(
        balance=balance,
        place_probability=kelly_prob,
        odds=live_odds,
        max_fraction=max_frac,
    )
    stake *= consensus_mult  # 4-model sense check: HIGH=1.0, MEDIUM=0.85

    if stake <= 0:
        stake = round(balance * 0.005, 2)
        if stake < 2.0:
            stake = 2.0

    stake = round(stake, 2)
    result["stake"] = stake

    # ── Place bet ──
    bet_id = f"bf-{meeting_id}-r{race_number}"
    bet = BetfairBet(
        id=bet_id,
        meeting_id=meeting_id,
        race_number=race_number,
        horse_name=runner.horse_name or "Unknown",
        saddlecloth=runner.saddlecloth,
        stake=stake,
        requested_odds=round(min_odds, 2),
        bet_type=bet_type,
        market_id=market["market_id"],
        selection_id=selection_id,
        status="placing",
        enabled=True,
        scheduled_at=race.start_time - timedelta(minutes=3),
        jit_win_probability=round(wp, 4),
        jit_place_probability=round(pp, 4),
        jit_evaluated_at=now,
    )
    db.add(bet)
    await db.commit()

    from punty.betting.betfair_client import place_bet
    place_result = await place_bet(db, market["market_id"], selection_id, stake, min_odds, use_bsp=True)

    if place_result.get("status") == "SUCCESS" or place_result.get("bet_id"):
        bet.status = "placed"
        bet.bet_id = place_result.get("bet_id")
        bet.size_matched = place_result.get("size_matched", 0)
        bet.average_price_matched = place_result.get("average_price_matched", 0)
        bet.matched_odds = place_result.get("average_price_matched") or live_odds
        bet.placed_at = now
        balance -= stake
        from punty.betting.queue import set_balance
        await set_balance(db, balance)
        await db.commit()

        result["action"] = "bet_placed"
        result["reason"] = (
            f"{bet_type.upper()} gap={wp_gap:.0%} PP={pp:.0%} WP={wp:.0%} "
            f"odds=${odds:.2f} live=${live_odds:.2f} stake=${stake:.2f} "
            f"kash={kash_multiplier}"
        )
        logger.info(f"JIT BET: {runner.horse_name} {meeting.venue} R{race_number} — {result['reason']}")
    else:
        error = place_result.get("error", "Unknown")
        bet.status = "failed"
        bet.error_message = str(error)[:200]
        await db.commit()

        result["action"] = "error"
        result["reason"] = f"Bet placement failed: {error}"
        logger.warning(f"JIT FAIL: {runner.horse_name} {meeting.venue} R{race_number} — {error}")

    return result


async def _create_skip_record(
    db, meeting_id, race_number, reason, now,
    horse_name=None, saddlecloth=None, jit_wp=None, jit_pp=None,
):
    """Create a BetfairBet row with status='skipped' for audit trail."""
    bet_id = f"bf-{meeting_id}-r{race_number}"
    # Check it doesn't already exist
    existing = await db.execute(select(BetfairBet).where(BetfairBet.id == bet_id))
    if existing.scalar_one_or_none():
        return

    bet = BetfairBet(
        id=bet_id,
        meeting_id=meeting_id,
        race_number=race_number,
        horse_name=horse_name or "—",
        saddlecloth=saddlecloth or 0,
        stake=0,
        status="skipped",
        error_message=reason[:200] if reason else None,
        enabled=False,
        jit_win_probability=round(jit_wp, 4) if jit_wp else None,
        jit_place_probability=round(jit_pp, 4) if jit_pp else None,
        jit_evaluated_at=now,
    )
    db.add(bet)
    await db.commit()
