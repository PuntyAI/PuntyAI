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
PP_FLOOR = 0.55  # v4: lowered from 0.59 — v4 model produces honest (lower) PPs
WIN_GAP_THRESHOLD = 0.10  # Default — overridden by _win_gap_for_race() context
MIN_PLACE_ODDS = 1.20  # Lowered from 1.30 — 86% of voided BSP bets would have placed
MIN_WIN_ODDS = 1.50
MAX_EDGE_CAP = 0.10

# ── Venue/Condition Kelly Modifiers ──
# Based on historical Betfair place SR analysis
STRONG_VENUES = {
    "gold coast", "kembla grange", "scone", "grafton", "sale",
    "matamata", "port macquarie", "kyneton",
}  # 70%+ place SR → boost 1.2x

WEAK_VENUES = {
    "gosford", "sapphire coast", "sunshine coast", "cairns",
    "toowoomba", "pakenham", "happy valley", "sha tin", "hawera",
}  # sub-45% place SR → reduce 0.7x

# Track condition multipliers (applied after Kelly + consensus)
CONDITION_MULTIPLIERS = {
    "Good 3": 1.0, "Good 4": 1.0,
    "Soft 5": 1.0,
    "Soft 6": 0.9, "Soft 7": 0.9,
    "Heavy 8": 0.75,
    "Heavy 9": 0.6, "Heavy 10": 0.6,
}


def _win_gap_for_race(race, meeting) -> float:
    """Context-aware WP gap threshold for win bet routing.

    Weaker races (maidens, low class, country) need a smaller gap to
    justify a win bet because field quality is lower and our model's
    edge is more pronounced. Stronger races (open, group, metro) need
    a bigger gap because competition is tighter.
    """
    threshold = WIN_GAP_THRESHOLD  # 0.10 base

    # Class adjustment
    race_class = (race.class_ or "").lower()
    if "maiden" in race_class:
        threshold = 0.08
    elif any(x in race_class for x in ("class 1", "class 2", "class 3", "bm56", "bm58", "bm60", "bm62", "benchmark 58", "benchmark 60", "benchmark 62")):
        threshold = 0.09
    elif any(x in race_class for x in ("group", "listed", "open", "stakes")):
        threshold = 0.12

    # Small fields — natural gaps are bigger, lower the bar
    field = race.field_size or 0
    if 0 < field <= 8:
        threshold -= 0.02

    # Country venues — weaker competition, gaps more reliable
    from punty.venues import guess_state
    venue_lower = (meeting.venue or "").lower()
    if venue_lower not in {"flemington", "randwick", "rosehill", "caulfield", "moonee valley",
                           "eagle farm", "doomben", "morphettville", "ascot", "ellerslie"}:
        threshold -= 0.01

    return max(threshold, 0.06)  # floor at 6%


def _venue_condition_modifier(venue: str, track_condition: str) -> float:
    """Return a combined venue + condition multiplier for Kelly staking."""
    venue_lower = (venue or "").lower()
    # Venue modifier
    if venue_lower in STRONG_VENUES:
        venue_mult = 1.2
    elif venue_lower in WEAK_VENUES:
        venue_mult = 0.7
    else:
        venue_mult = 1.0
    # Condition modifier — match "Good 4", "Heavy 8" etc
    cond_mult = CONDITION_MULTIPLIERS.get(track_condition or "", 1.0)
    return round(venue_mult * cond_mult, 3)


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

    # Skip venues not on Betfair (Hong Kong)
    from punty.venues import guess_state
    state = guess_state(meeting.venue or "")
    if state == "HK":
        result["reason"] = "HK not on Betfair"
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
        # Probs are keyed by runner.id, not horse_name
        prob = probs.get(runner.id) or probs.get(runner.horse_name)
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

    # ── Bet type routing (contextual WP gap) ──
    r2_wp = r2["wp"] if r2 else 0
    wp_gap = wp - r2_wp
    gap_threshold = _win_gap_for_race(race, meeting)

    if wp_gap >= gap_threshold and odds > 1.0:
        bet_type = "win"
        min_odds = MIN_WIN_ODDS
        logger.info(
            f"JIT WIN route: {runner.horse_name} gap={wp_gap:.0%} >= {gap_threshold:.0%} "
            f"({race.class_ or 'unknown'}, {race.field_size or '?'} runners)"
        )
    else:
        bet_type = "place"
        min_odds = MIN_PLACE_ODDS

    result["bet_type"] = bet_type

    # ── 4-Model Sense Check (Us vs PF vs Market vs KASH) ──
    from punty.sense_check import sense_check_race, find_consensus_pick
    sense = sense_check_race(runner.saddlecloth, runners)
    consensus_mult = sense["kelly_mult"]

    if sense["action"] == "skip":
        # Before giving up, check if any of our R1/R2/R3 picks matches
        # the consensus top pick of ALL 3 external models
        picks_result = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting_id,
                Pick.race_number == race_number,
                Pick.pick_type == "selection",
                Pick.tip_rank.in_([1, 2, 3]),
            ).order_by(Pick.tip_rank)
        )
        our_picks = picks_result.scalars().all()
        consensus = find_consensus_pick(our_picks, runners)

        if consensus:
            # Consensus override — switch to the agreed horse
            override_sc = consensus["saddlecloth"]
            override_runner = next((r for r in runners if r.saddlecloth == override_sc), None)
            if override_runner:
                # Re-derive WP/PP for the consensus horse
                prob = probs.get(override_runner.horse_name)
                if prob:
                    runner = override_runner
                    wp = prob.win_probability
                    pp = prob.place_probability
                    odds = runner.current_odds or 0
                    consensus_mult = consensus["kelly_mult"]
                    result["horse_name"] = runner.horse_name

                    # Re-check PP floor for the consensus horse
                    if pp < PP_FLOOR:
                        result["reason"] = f"Consensus R{consensus['tip_rank']} {runner.horse_name} PP {pp:.0%} < {PP_FLOOR:.0%}"
                        await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                                  horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                                  jit_wp=wp, jit_pp=pp)
                        return result

                    # Re-route bet type
                    r2_wp = ranked[1]["wp"] if len(ranked) > 1 else 0
                    wp_gap = wp - r2_wp
                    if wp_gap >= WIN_GAP_THRESHOLD and odds > 1.0:
                        bet_type = "win"
                        min_odds = MIN_WIN_ODDS
                    else:
                        bet_type = "place"
                        min_odds = MIN_PLACE_ODDS
                    result["bet_type"] = bet_type

                    logger.info(
                        f"JIT CONSENSUS OVERRIDE: R{consensus['tip_rank']} {runner.horse_name} "
                        f"{meeting.venue} R{race_number} — {consensus['detail']} PP={pp:.0%}"
                    )
                    # Fall through to market resolution and betting
                else:
                    result["reason"] = f"Sense check SKIP + consensus horse no probs: {sense['detail']}"
                    await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                              horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                              jit_wp=wp, jit_pp=pp)
                    return result
            else:
                result["reason"] = f"Sense check SKIP: {sense['detail']}"
                await _create_skip_record(db, meeting_id, race_number, result["reason"], now,
                                          horse_name=runner.horse_name, saddlecloth=runner.saddlecloth,
                                          jit_wp=wp, jit_pp=pp)
                return result
        else:
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

    # ── Resolve Betfair market ──
    from punty.betting.betfair_client import resolve_place_market, resolve_win_market, get_place_odds
    from punty.betting.flumine_client import flumine_manager

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

    # ── Get live odds from Flumine stream ──
    # Flumine tracks the price stream and detects peaks (price drifts out
    # then starts shortening). We bet at the peak for better odds than BSP.
    flumine_price = None
    flumine_is_peak = False
    if flumine_manager.is_available():
        best = flumine_manager.get_best_price(market["market_id"], selection_id)
        if best and best["price"] > 1.0:
            flumine_price = best["price"]
            flumine_is_peak = best["is_peak"]
            if flumine_is_peak:
                logger.info(
                    f"JIT PEAK PRICE: {runner.horse_name} peak=${best['peak_price']:.2f} "
                    f"now=${best['current_price']:.2f} ({best['ticks_since_peak']} ticks ago)"
                )
            else:
                logger.info(
                    f"JIT Flumine price: {runner.horse_name} ${best['current_price']:.2f} "
                    f"(peak=${best['peak_price']:.2f})"
                )

    live_odds = flumine_price if flumine_price and flumine_price > 1.0 else None
    if not live_odds:
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

    # Venue + track condition modifier
    vc_mult = _venue_condition_modifier(meeting.venue, meeting.track_condition)
    stake *= vc_mult
    if vc_mult != 1.0:
        logger.info(
            f"JIT venue/condition modifier: {meeting.venue} ({meeting.track_condition}) "
            f"mult={vc_mult} (venue={'strong' if (meeting.venue or '').lower() in STRONG_VENUES else 'weak' if (meeting.venue or '').lower() in WEAK_VENUES else 'neutral'}, "
            f"cond={CONDITION_MULTIPLIERS.get(meeting.track_condition or '', 1.0)})"
        )

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

    if flumine_price and flumine_price >= min_odds:
        # LIMIT order at the Flumine price (peak or current).
        # Peak = price drifted out then started shortening, we lock in the high.
        # Current = no peak detected yet, take what's available.
        # Both better than blind BSP because we know the exact price.
        order_price = flumine_price
        place_result = await place_bet(
            db, market["market_id"], selection_id, stake,
            price=order_price, use_bsp=False,
        )
        tag = "PEAK" if flumine_is_peak else "LIVE"
        logger.info(
            f"JIT {tag} LIMIT ${order_price:.2f}: {place_result.get('status')} "
            f"matched={place_result.get('size_matched', 0)}"
        )
    else:
        # No streaming price or below minimum — BSP fallback
        place_result = await place_bet(
            db, market["market_id"], selection_id, stake,
            price=min_odds, use_bsp=True,
        )
        logger.info(f"JIT BSP fallback (min ${min_odds:.2f}): {place_result.get('status')}")

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
            f"consensus={consensus_mult} vc={vc_mult}"
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
