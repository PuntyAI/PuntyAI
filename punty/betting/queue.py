"""Bet queue business logic — populate, execute, and settle Betfair bets."""

import logging
import math
from datetime import timedelta
from typing import Optional

from sqlalchemy import select, func, and_, Integer
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
DEFAULT_MIN_ODDS = 1.30  # Minimum place odds floor — below this, one miss wipes too many wins
DEFAULT_COMMISSION_RATE = 0.00  # Betfair commission (0% with discount rate)
DEFAULT_MAX_DAILY_LOSS = -20.0
DEFAULT_MIN_PLACE_PROB = 0.45  # 45% minimum place probability (tissue engine calibration)
DEFAULT_EDGE_MULTIPLIER = 1.10  # 10% edge over implied probability required
DEFAULT_DEAD_ZONE_LOW = 1.60  # Dead zone lower bound (skip bets in this range)
DEFAULT_DEAD_ZONE_HIGH = 2.00  # Dead zone upper bound
DEFAULT_MAX_KELLY_FRACTION = 0.06  # Cap Kelly fraction at 6% (half-Kelly conservative)
DEFAULT_KELLY_HALF = True  # True half-Kelly: halve the fraction for 75% less variance
DEFAULT_MIN_KELLY_STAKE = 5.00  # Betfair minimum bet size (AUD)
DEFAULT_MIN_CALIBRATED_PP = 0.50  # Only bet when calibrated PP >= 50% (volume for compound growth)
DEFAULT_MAX_PLACE_ODDS = 6.0  # Maximum place odds for queue eligibility
DEFAULT_MIN_RUNNERS = 8  # Minimum runners for 3 place dividends (NTD below this)
DEFAULT_NTD_HIGH_PP = 0.70  # Allow 5-7 runners if PP >= this threshold
MAIDEN_PREFIXES = ("maiden",)  # Case-insensitive startswith check
DEFAULT_MAIDEN_MAX_PLACE_ODDS = 1.75  # Tighter ceiling for maiden races (≈$3 win)


async def _count_active_runners(db: AsyncSession, race_id: str) -> int:
    """Count non-scratched runners in a race."""
    result = await db.execute(
        select(func.count(Runner.id)).where(
            Runner.race_id == race_id,
            Runner.scratched != True,
        )
    )
    return result.scalar() or 0


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
    """Calculate current stake based on balance growth (legacy doubling formula).

    Used only as default/fallback when Kelly parameters aren't available.
    Doubles stake every time balance doubles from initial.
    """
    if balance <= 0 or initial_balance <= 0:
        return base_stake
    ratio = balance / initial_balance
    if ratio < 1:
        return base_stake
    doublings = int(math.log2(ratio))
    return base_stake * (2 ** doublings)


def calculate_kelly_stake(
    balance: float,
    place_probability: float,
    odds: float,
    max_fraction: float = DEFAULT_MAX_KELLY_FRACTION,
    min_stake: float = DEFAULT_MIN_KELLY_STAKE,
    half_kelly: bool = DEFAULT_KELLY_HALF,
) -> float:
    """Kelly-proportional staking: bet more when edge is larger.

    Uses half-Kelly by default: 75% less variance, only 25% less expected growth.
    Losing hurts more than winning — conservative sizing protects compound growth.

    Kelly fraction = edge / (odds - 1), halved, capped at max_fraction.
    Stake = kelly_fraction * balance, floored at min_stake.
    """
    if odds <= 1 or balance <= 0 or place_probability <= 0:
        return 0  # No valid bet
    implied_prob = 1.0 / odds
    edge = place_probability - implied_prob
    if edge <= 0:
        return 0  # Negative edge — don't bet
    kelly = edge / (odds - 1)
    if half_kelly:
        kelly *= 0.5  # Half-Kelly: dramatically reduces drawdowns
    kelly = min(kelly, max_fraction)
    stake = kelly * balance
    if stake < min_stake:
        return min_stake  # Round up to Betfair minimum if edge exists
    return stake


async def get_current_stake(db: AsyncSession, place_probability: float = 0,
                           odds: float = 0) -> float:
    """Get current stake based on mode setting.

    Modes:
    - 'kelly': Kelly-proportional staking based on PP and odds (preferred)
    - 'auto': Legacy doubling formula (fallback)
    - numeric: Fixed stake amount
    """
    mode = await _get_setting(db, "betfair_stake_mode", "kelly")
    balance = await get_balance(db)

    if mode.lower() == "kelly":
        if place_probability > 0 and odds > 1:
            max_frac = float(await _get_setting(db, "betfair_max_kelly_fraction",
                                                 str(DEFAULT_MAX_KELLY_FRACTION)))
            min_pp = float(await _get_setting(db, "betfair_min_calibrated_pp",
                                               str(DEFAULT_MIN_CALIBRATED_PP)))
            from punty.betting.calibration import calibrated_kelly_stake
            return await calibrated_kelly_stake(
                db, balance, place_probability, odds,
                max_fraction=max_frac, min_calibrated_pp=min_pp)
        # Fall through to auto if PP/odds not available
        mode = "auto"

    if mode.lower() == "auto":
        initial_balance = float(await _get_setting(db, "betfair_initial_balance", str(DEFAULT_INITIAL_BALANCE)))
        base_stake = float(await _get_setting(db, "betfair_stake", str(DEFAULT_BASE_STAKE)))
        return calculate_stake(balance, initial_balance, base_stake)
    else:
        try:
            return max(0.50, float(mode))
        except ValueError:
            return DEFAULT_BASE_STAKE


async def populate_bet_queue(
    db: AsyncSession,
    meeting_id: str,
    content_id: str,
) -> int:
    """Create BetfairBet entries for the highest place-probability pick per race.

    Called after content approval. Selects the pick with the best place_probability
    from each race's selections — pure probability, no value weighting.
    Returns count of bets queued.
    """
    auto_enabled = await _get_setting(db, "betfair_auto_bet_enabled", "true")
    if auto_enabled.lower() != "true":
        return 0

    # Get best pick per race by PP — for place bets, PP matters more than tip rank
    # Include tracked_only picks — short-priced favs marked tracked_only by the
    # standard system are exactly the highest-PP picks Betfair should bet on
    result = await db.execute(
        select(Pick).where(
            Pick.content_id == content_id,
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "selection",
            Pick.tip_rank.in_([1, 2, 3, 4]),
        ).order_by(Pick.race_number)
    )
    all_picks = result.scalars().all()
    # Select highest PP pick per race
    best_by_race: dict[int, Pick] = {}
    for pick in all_picks:
        pp = pick.place_probability or 0
        rn = pick.race_number
        if rn not in best_by_race or pp > (best_by_race[rn].place_probability or 0):
            best_by_race[rn] = pick
    rank1_picks = list(best_by_race.values())
    if not rank1_picks:
        return 0

    # Load races to get start times
    race_result = await db.execute(
        select(Race).where(Race.meeting_id == meeting_id)
    )
    races_by_num = {r.race_number: r for r in race_result.scalars().all()}

    min_place_prob = float(await _get_setting(db, "betfair_min_place_prob", str(DEFAULT_MIN_PLACE_PROB)))
    max_place_odds = float(await _get_setting(db, "betfair_max_place_odds", str(DEFAULT_MAX_PLACE_ODDS)))
    min_odds = float(await _get_setting(db, "betfair_min_odds", str(DEFAULT_MIN_ODDS)))

    queued = 0
    for pick in rank1_picks:
        race = races_by_num.get(pick.race_number)
        if not race or not race.start_time:
            logger.debug(f"Skipping bet for {meeting_id} R{pick.race_number}: no start time")
            continue

        # Filter by odds floor — ultra-short odds bleed value
        if pick.place_odds_at_tip and pick.place_odds_at_tip < min_odds:
            logger.info(
                f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                f"place_odds ${pick.place_odds_at_tip:.2f} < ${min_odds:.2f} floor"
            )
            continue

        # Filter by odds ceiling — reject longshots
        if pick.place_odds_at_tip and pick.place_odds_at_tip > max_place_odds:
            logger.info(
                f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                f"place_odds ${pick.place_odds_at_tip:.2f} > ${max_place_odds:.2f} ceiling"
            )
            continue

        # Maiden races — allow only short-priced favourites (place odds <= $1.75)
        if race:
            race_class = (race.class_ or "").lower().rstrip(";").strip()
            if race_class.startswith(MAIDEN_PREFIXES):
                maiden_ceiling = DEFAULT_MAIDEN_MAX_PLACE_ODDS
                place_odds = pick.place_odds_at_tip or 999
                if place_odds > maiden_ceiling:
                    logger.info(
                        f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                        f"maiden race, place odds ${place_odds:.2f} > ${maiden_ceiling:.2f} ceiling"
                    )
                    continue

        # NTD filter — need 8+ runners for 3 place dividends
        # Allow 5-7 runners if place_probability is very high (>= 70%)
        race_id = f"{meeting_id}-r{race.race_number}"
        runner_count = await _count_active_runners(db, race_id)
        if runner_count < DEFAULT_MIN_RUNNERS:
            pp = pick.place_probability or 0
            if runner_count < 5:
                logger.info(
                    f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                    f"NTD — {runner_count} runners (too few for place betting)"
                )
                continue
            if pp < DEFAULT_NTD_HIGH_PP:
                logger.info(
                    f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                    f"NTD — {runner_count} runners, PP {pp:.0%} < {DEFAULT_NTD_HIGH_PP:.0%} threshold"
                )
                continue
            logger.info(
                f"NTD override for {pick.horse_name} R{pick.race_number}: "
                f"{runner_count} runners but PP {pp:.0%} >= {DEFAULT_NTD_HIGH_PP:.0%}"
            )

        # Filter by place probability — only bet on high-confidence selections
        if pick.place_probability is not None and pick.place_probability < min_place_prob:
            logger.info(
                f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                f"place_prob {pick.place_probability:.1%} < {min_place_prob:.0%} threshold"
            )
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

        # Kelly stake per-bet based on this pick's PP and odds
        pp = pick.place_probability or 0
        odds = pick.place_odds_at_tip or 0
        stake = await get_current_stake(db, place_probability=pp, odds=odds)
        if stake <= 0:
            logger.info(
                f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                f"Kelly says no edge (PP={pp:.0%}, odds=${odds:.2f})"
            )
            continue

        bet_id = f"bf-{meeting_id}-r{pick.race_number}"
        scheduled_at = race.start_time - timedelta(minutes=10)

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


SWAP_THRESHOLD = 0.03  # 3 percentage-point gap required to swap picks


async def cycle_bet_selection(db: AsyncSession, bet_id: str) -> dict:
    """Manually cycle a queued bet to the next-best pick for that race.

    Each call moves to the next pick by place_probability, skipping
    scratched runners and the current selection. Wraps around to the
    best pick after exhausting all candidates.

    Returns: {swapped: bool, horse_name, saddlecloth, place_probability, message}
    """
    result = await db.execute(select(BetfairBet).where(BetfairBet.id == bet_id))
    bet = result.scalar_one_or_none()
    if not bet:
        return {"swapped": False, "message": "Bet not found"}
    if bet.status != "queued":
        return {"swapped": False, "message": f"Cannot cycle bet in '{bet.status}' status"}

    race_id = f"{bet.meeting_id}-r{bet.race_number}"

    max_place_odds = float(await _get_setting(db, "betfair_max_place_odds", str(DEFAULT_MAX_PLACE_ODDS)))

    # Load all selection picks for this race, sorted by tip_rank (rank 1 first)
    # Include tracked_only picks — manual cycle is an explicit override
    pick_result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == bet.meeting_id,
            Pick.race_number == bet.race_number,
            Pick.pick_type == "selection",
        ).order_by(Pick.tip_rank)
    )
    candidates = pick_result.scalars().all()

    # Filter out scratched runners and longshots
    valid = []
    for pick in candidates:
        runner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.saddlecloth == pick.saddlecloth,
            )
        )
        runner = runner_result.scalar_one_or_none()
        if runner and runner.scratched:
            continue
        if pick.place_odds_at_tip and pick.place_odds_at_tip > max_place_odds:
            continue
        valid.append(pick)

    if not valid:
        return {"swapped": False, "message": "No valid candidates (all scratched or filtered)"}

    # Find current pick's position and select the next one
    current_idx = None
    for i, pick in enumerate(valid):
        if pick.id == bet.pick_id:
            current_idx = i
            break

    if current_idx is not None:
        next_idx = (current_idx + 1) % len(valid)
    else:
        next_idx = 0  # Current pick not found — start from best

    next_pick = valid[next_idx]
    if next_pick.id == bet.pick_id:
        return {
            "swapped": False,
            "message": "Only one valid candidate available",
            "horse_name": bet.horse_name,
            "saddlecloth": bet.saddlecloth,
            "place_probability": next_pick.place_probability,
        }

    old_name = bet.horse_name
    bet.pick_id = next_pick.id
    bet.horse_name = next_pick.horse_name or "Unknown"
    bet.saddlecloth = next_pick.saddlecloth
    bet.requested_odds = next_pick.place_odds_at_tip

    bet.bet_type = "place"  # Always place — no win bets

    await db.commit()
    rank = next_idx + 1
    logger.info(
        f"Manual cycle {bet.id}: {old_name} → {bet.horse_name} "
        f"(#{rank}/{len(valid)}, {next_pick.place_probability:.0%}pp)"
    )
    return {
        "swapped": True,
        "horse_name": bet.horse_name,
        "saddlecloth": bet.saddlecloth,
        "place_probability": next_pick.place_probability,
        "bet_type": bet.bet_type,
        "rank": rank,
        "total": len(valid),
        "message": f"Cycled to #{rank}: {bet.horse_name}",
    }


async def refresh_bet_selections(db: AsyncSession) -> int:
    """Re-evaluate queued bets: swap scratched horses, upgrade to higher place_prob,
    and flip odds-on horses to win bets.

    Called every 30s by the scheduler, before execute_due_bets.
    Only touches queued + enabled bets — never placed/settled/cancelled.

    Returns count of swaps + cancellations performed.
    """
    result = await db.execute(
        select(BetfairBet).where(
            BetfairBet.status == "queued",
            BetfairBet.enabled == True,
        )
    )
    queued_bets = result.scalars().all()
    if not queued_bets:
        return 0

    min_place_prob = float(await _get_setting(db, "betfair_min_place_prob", str(DEFAULT_MIN_PLACE_PROB)))
    max_place_odds = float(await _get_setting(db, "betfair_max_place_odds", str(DEFAULT_MAX_PLACE_ODDS)))
    changes = 0

    for bet in queued_bets:
        race_id = f"{bet.meeting_id}-r{bet.race_number}"

        # NTD check — cancel if scratchings dropped field below threshold
        # Allow 5-7 runners if the pick's PP >= 70%
        runner_count = await _count_active_runners(db, race_id)
        if runner_count < DEFAULT_MIN_RUNNERS:
            # Look up PP from current active pick (handles superseded content)
            pick_pp_result = await db.execute(select(Pick).where(Pick.id == bet.pick_id))
            current_pick = pick_pp_result.scalar_one_or_none()
            pp = (current_pick.place_probability if current_pick else None) or 0
            # If PP is 0, try finding by horse name in active content
            if pp == 0:
                active_pick_result = await db.execute(
                    select(Pick).where(
                        Pick.meeting_id == bet.meeting_id,
                        Pick.race_number == bet.race_number,
                        Pick.horse_name == bet.horse_name,
                        Pick.pick_type == "selection",
                        Pick.place_probability > 0,
                    ).order_by(Pick.place_probability.desc()).limit(1)
                )
                active_pick = active_pick_result.scalar_one_or_none()
                if active_pick:
                    pp = active_pick.place_probability or 0
                    bet.pick_id = active_pick.id  # Re-link to active pick
            if runner_count < 5 or pp < DEFAULT_NTD_HIGH_PP:
                bet.status = "cancelled"
                reason = f"NTD — {runner_count} runners" + (f", PP {pp:.0%} < {DEFAULT_NTD_HIGH_PP:.0%}" if runner_count >= 5 else "")
                bet.error_message = reason
                changes += 1
                logger.info(f"Cancelled {bet.id}: {reason}")
                continue

        # Check maiden race — cancel if odds drifted above maiden ceiling
        race_result = await db.execute(
            select(Race).where(Race.id == race_id)
        )
        race = race_result.scalar_one_or_none()
        race_class_raw = (race.class_ if race else None) or ""
        if race_class_raw.lower().rstrip(";").strip().startswith(MAIDEN_PREFIXES):
            place_odds = bet.requested_odds or 999
            if place_odds > DEFAULT_MAIDEN_MAX_PLACE_ODDS:
                bet.status = "cancelled"
                bet.error_message = f"Maiden race, odds ${place_odds:.2f} > ${DEFAULT_MAIDEN_MAX_PLACE_ODDS:.2f}"
                changes += 1
                logger.info(f"Cancelled {bet.id}: maiden odds ${place_odds:.2f} above ceiling")
                continue

        # Load all selection picks for this race — pick by best PP for place bets
        # Include tracked_only — short-priced favs are our best Betfair candidates
        pick_result = await db.execute(
            select(Pick).where(
                Pick.meeting_id == bet.meeting_id,
                Pick.race_number == bet.race_number,
                Pick.pick_type == "selection",
                Pick.tip_rank.in_([1, 2, 3, 4]),
                Pick.place_probability > 0,
            )
        )
        candidates = pick_result.scalars().all()
        if not candidates:
            continue

        # Check if current horse is scratched
        current_runner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.saddlecloth == bet.saddlecloth,
            )
        )
        current_runner = current_runner_result.scalar_one_or_none()
        current_scratched = current_runner.scratched if current_runner else False

        # Find best non-scratched candidate above min_place_prob and below odds ceiling
        best_pick = None
        best_pp = 0.0
        for pick in candidates:
            runner_result = await db.execute(
                select(Runner).where(
                    Runner.race_id == race_id,
                    Runner.saddlecloth == pick.saddlecloth,
                )
            )
            runner = runner_result.scalar_one_or_none()
            if runner and runner.scratched:
                continue
            if pick.place_odds_at_tip and pick.place_odds_at_tip > max_place_odds:
                continue
            # VR ceiling — Data: VR 1.5+ unprofitable
            vr = getattr(pick, "value_rating", None)
            if isinstance(vr, (int, float)) and vr >= 1.5:
                continue
            pp = pick.place_probability or 0
            if pp >= min_place_prob and pp > best_pp:
                best_pp = pp
                best_pick = pick

        # Check if current bet's pick is still a valid candidate
        current_is_valid = any(p.id == bet.pick_id for p in candidates)

        if current_scratched or not current_is_valid:
            # Must swap: horse scratched or pick no longer valid
            reason = "scratched" if current_scratched else "pick removed"
            if best_pick and best_pick.id != bet.pick_id:
                old_name = bet.horse_name
                bet.pick_id = best_pick.id
                bet.horse_name = best_pick.horse_name or "Unknown"
                bet.saddlecloth = best_pick.saddlecloth
                bet.requested_odds = best_pick.place_odds_at_tip
                changes += 1
                logger.info(
                    f"Swapped {bet.id}: {old_name} ({reason}) → "
                    f"{bet.horse_name} ({best_pp:.0%}pp)"
                )
            else:
                # No viable replacement — cancel
                bet.status = "cancelled"
                bet.error_message = f"No eligible pick ({reason})"
                changes += 1
                logger.info(f"Cancelled {bet.id}: {bet.horse_name} {reason}, no replacement")
        elif best_pick and best_pick.id != bet.pick_id:
            # Current horse is rank 1, not scratched — only swap if replacement is significantly better
            current_pp = 0.0
            for pick in candidates:
                if pick.id == bet.pick_id:
                    current_pp = pick.place_probability or 0
                    break
            if best_pp - current_pp >= SWAP_THRESHOLD:
                old_name = bet.horse_name
                bet.pick_id = best_pick.id
                bet.horse_name = best_pick.horse_name or "Unknown"
                bet.saddlecloth = best_pick.saddlecloth
                bet.requested_odds = best_pick.place_odds_at_tip
                changes += 1
                logger.info(
                    f"Swapped {bet.id}: {old_name} ({current_pp:.0%}pp) → "
                    f"{bet.horse_name} ({best_pp:.0%}pp)"
                )

    # --- Odds refresh: update requested_odds from live Runner place_odds ---
    # Picks store odds at approval time which can be stale tote data.
    # Live Runner.place_odds gets updated by Betfair/PointsBet scrapers.
    for bet in queued_bets:
        if bet.status != "queued":
            continue  # skip any we just cancelled above
        race_id = f"{bet.meeting_id}-r{bet.race_number}"
        runner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.saddlecloth == bet.saddlecloth,
            )
        )
        runner = runner_result.scalar_one_or_none()
        if not runner or not runner.place_odds or runner.place_odds <= 1.0:
            continue
        old_odds = bet.requested_odds or 0
        if abs(runner.place_odds - old_odds) > 0.05:
            bet.requested_odds = round(runner.place_odds, 2)
            # Re-check odds floor/ceiling after update
            min_odds = float(await _get_setting(db, "betfair_min_odds", str(DEFAULT_MIN_ODDS)))
            if bet.requested_odds < min_odds:
                bet.status = "cancelled"
                bet.error_message = f"Live odds ${bet.requested_odds:.2f} below ${min_odds:.2f} floor"
                changes += 1
                logger.info(f"Cancelled {bet.id}: odds dropped to ${bet.requested_odds:.2f}")
                continue
            if bet.requested_odds > max_place_odds:
                bet.status = "cancelled"
                bet.error_message = f"Live odds ${bet.requested_odds:.2f} above ${max_place_odds:.2f} ceiling"
                changes += 1
                logger.info(f"Cancelled {bet.id}: odds drifted to ${bet.requested_odds:.2f}")
                continue
            # Recalculate Kelly stake with updated odds
            pp_result = await db.execute(select(Pick).where(Pick.id == bet.pick_id))
            pick = pp_result.scalar_one_or_none()
            if pick and pick.place_probability:
                new_stake = await get_current_stake(
                    db, place_probability=pick.place_probability,
                    odds=bet.requested_odds,
                )
                if new_stake <= 0:
                    bet.status = "cancelled"
                    bet.error_message = f"No Kelly edge at live odds ${bet.requested_odds:.2f}"
                    changes += 1
                    logger.info(f"Cancelled {bet.id}: no edge at ${bet.requested_odds:.2f}")
                    continue
                if abs(new_stake - bet.stake) > 0.50:
                    bet.stake = round(new_stake, 2)
            if old_odds > 0:
                logger.info(
                    f"Odds update {bet.id}: {bet.horse_name} "
                    f"${old_odds:.2f} -> ${bet.requested_odds:.2f}"
                )
                changes += 1

    if changes:
        await db.commit()
    return changes


async def execute_due_bets(db: AsyncSession) -> int:
    """Find and execute bets whose scheduled_at has arrived.

    Called every 30s by the scheduler. Processes bets where:
    - status = 'queued'
    - enabled = True
    - scheduled_at <= now
    - scheduled_at > now - 15min (don't bet on races that started long ago)
    """
    from punty.betting.betfair_client import (
        resolve_place_market, get_place_odds, place_bet,
    )

    now = melb_now_naive()
    cutoff = now - timedelta(minutes=15)

    # Auto-cancel bets that missed their window (scheduled_at + 10min < now)
    expired_result = await db.execute(
        select(BetfairBet).where(
            BetfairBet.status == "queued",
            BetfairBet.scheduled_at <= cutoff,
        )
    )
    expired = expired_result.scalars().all()
    for bet in expired:
        bet.status = "cancelled"
        bet.error_message = "Missed betting window"
        logger.info(f"Betfair: auto-cancelled {bet.id} — missed window")
    if expired:
        await db.commit()

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
    min_place_prob = float(await _get_setting(db, "betfair_min_place_prob", str(DEFAULT_MIN_PLACE_PROB)))
    dead_zone_low = float(await _get_setting(db, "betfair_dead_zone_low", str(DEFAULT_DEAD_ZONE_LOW)))
    dead_zone_high = float(await _get_setting(db, "betfair_dead_zone_high", str(DEFAULT_DEAD_ZONE_HIGH)))

    placed = 0
    for bet in due_bets:
        if balance < DEFAULT_MIN_KELLY_STAKE:
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

        # Load pick to get place_probability for edge gate + Kelly staking
        pick_result = await db.execute(select(Pick).where(Pick.id == bet.pick_id))
        pick = pick_result.scalar_one_or_none()
        place_prob = (pick.place_probability if pick else None) or 0

        # Resolve meeting
        meeting_result = await db.execute(select(Meeting).where(Meeting.id == bet.meeting_id))
        meeting = meeting_result.scalar_one_or_none()
        if not meeting:
            bet.status = "failed"
            bet.error_message = "Meeting not found"
            continue

        bet.status = "placing"
        await db.commit()

        # Prefer place market; fall back to win if unavailable
        bet.bet_type = "place"
        market = await resolve_place_market(
            db, meeting.venue, meeting.date, bet.meeting_id, bet.race_number
        )
        if not market:
            # Betfair often lacks Place markets for smaller venues — try Win
            from punty.betting.betfair_client import resolve_win_market
            market = await resolve_win_market(
                db, meeting.venue, meeting.date, bet.meeting_id, bet.race_number
            )
            if market:
                bet.bet_type = "win"
                logger.info(
                    "No Place market for %s R%s — falling back to Win",
                    meeting.venue, bet.race_number,
                )
            else:
                bet.status = "failed"
                bet.error_message = "No Betfair Place or Win market available"
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

        # Get live Betfair place odds (for Kelly sizing and edge check)
        current_odds = await get_place_odds(db, market["market_id"], selection_id)
        if not current_odds:
            # BSP doesn't need live odds, but we need them for Kelly sizing
            # Use requested_odds as fallback for stake calculation
            current_odds = bet.requested_odds or 0

        # Min odds floor — ultra-short odds bleed value
        if current_odds and current_odds < min_odds:
            bet.status = "cancelled"
            bet.error_message = f"Odds ${current_odds:.2f} below ${min_odds:.2f} floor"
            logger.info(f"Betfair: {bet.id} cancelled — odds ${current_odds:.2f} < ${min_odds:.2f} floor")
            await db.commit()
            continue

        # Edge gate: PP floor guarantees high strike rate
        if place_prob > 0 and place_prob < min_place_prob:
            bet.status = "cancelled"
            bet.error_message = f"PP {place_prob:.1%} below {min_place_prob:.0%} floor"
            logger.info(f"Betfair: {bet.id} cancelled — PP {place_prob:.1%} < {min_place_prob:.0%}")
            await db.commit()
            continue

        # Kelly stake based on live odds and PP
        stake = await get_current_stake(db, place_probability=place_prob,
                                         odds=current_odds if current_odds > 1 else 0)
        if stake <= 0:
            bet.status = "cancelled"
            bet.error_message = f"Kelly: no edge (PP={place_prob:.0%}, odds=${current_odds:.2f})"
            logger.info(f"Betfair: {bet.id} cancelled — Kelly says no edge")
            await db.commit()
            continue

        bet.stake = round(stake, 2)
        # BSP: use min_odds as the floor price (minimum acceptable BSP)
        bet.requested_odds = min_odds

        if balance < bet.stake:
            bet.status = "cancelled"
            bet.error_message = f"Insufficient balance (${balance:.2f} < ${bet.stake:.2f})"
            logger.warning(f"Betfair: skipping {bet.id} — insufficient balance")
            await db.commit()
            continue

        # Place BSP bet — guaranteed fill at market-clearing price
        result = await place_bet(db, market["market_id"], selection_id,
                                  bet.stake, min_odds, use_bsp=True)

        if result.get("status") == "SUCCESS" or result.get("bet_id"):
            bet.status = "placed"
            bet.bet_id = result.get("bet_id")
            bet.size_matched = result.get("size_matched", 0)
            bet.average_price_matched = result.get("average_price_matched", 0)
            bet.matched_odds = result.get("average_price_matched") or current_odds
            bet.placed_at = melb_now_naive()
            placed += 1
            # BSP: stake is the liability (what we risk), deduct now
            balance -= bet.stake
            await set_balance(db, balance)
            edge_pct = (place_prob - 1.0 / current_odds) * 100 if current_odds > 1 else 0
            logger.info(
                f"Betfair BSP: placed {bet.id} — {bet.horse_name} ${bet.stake:.2f} "
                f"@ BSP (live ${current_odds:.2f}) "
                f"PP={place_prob:.0%} edge={edge_pct:+.1f}% (balance: ${balance:.2f})"
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

    Hit condition depends on bet_type:
    - place: finish position 1-3
    - win: finish position 1 only
    P&L accounts for 5% Betfair commission.
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
    if not runner:
        return 0  # Runner not found

    # Check if race has results via results_status
    from punty.models.meeting import Race
    race_result = await db.execute(
        select(Race.results_status).where(Race.id == race_id)
    )
    race_status = race_result.scalar_one_or_none()

    # Determine finish position — use actual position, or infer from other runners
    # NOTE: win_dividend/place_dividend are tote odds stored on ALL runners,
    # NOT result indicators — cannot be used to infer finish position.
    is_win_bet = getattr(bet, "bet_type", "place") == "win"
    finish_pos = runner.finish_position
    if finish_pos is None:
        if race_status in ("Paying", "Closed"):
            # Check if the top positions are filled by OTHER runners
            needed_positions = [1] if is_win_bet else [1, 2, 3]
            filled_result = await db.execute(
                select(func.count(Runner.id)).where(
                    Runner.race_id == race_id,
                    Runner.finish_position.in_(needed_positions),
                    Runner.saddlecloth != bet.saddlecloth,
                )
            )
            filled_count = filled_result.scalar()
            if filled_count >= len(needed_positions):
                # All required positions taken by other runners — our horse missed
                finish_pos = 99
            elif race_status == "Closed":
                # Race closed but positions incomplete — wait for data backfill
                # unless at least 1st place is filled (then we know enough)
                first_result = await db.execute(
                    select(func.count(Runner.id)).where(
                        Runner.race_id == race_id,
                        Runner.finish_position == 1,
                    )
                )
                if first_result.scalar() > 0:
                    finish_pos = 99  # 1st is known, our horse isn't in it
                else:
                    return 0  # No finish data at all yet
            else:
                return 0  # Paying but positions not filled yet — wait
        else:
            return 0  # Results not in yet

    # Fetch actual BSP from Betfair before settling — BSP orders don't have
    # matched odds at placement time, only after the race
    if bet.bet_id and not bet.bet_id.startswith("mock-"):
        from punty.betting.betfair_client import get_bet_result
        bf_result = await get_bet_result(db, bet.bet_id)
        if bf_result and bf_result.get("price_matched"):
            bsp = bf_result["price_matched"]
            logger.info(
                f"BSP update for {bet.id}: {bet.matched_odds or bet.requested_odds:.2f} -> {bsp:.2f}"
            )
            bet.matched_odds = bsp

    commission_rate = float(await _get_setting(db, "betfair_commission_rate", str(DEFAULT_COMMISSION_RATE)))
    odds = bet.matched_odds or bet.requested_odds or 0

    now = melb_now_naive()
    hit_threshold = 1 if is_win_bet else 3  # win = 1st only, place = top 3

    if finish_pos <= hit_threshold:
        gross_profit = (odds - 1) * bet.stake
        commission = gross_profit * commission_rate
        bet.pnl = round(gross_profit - commission, 2)
        bet.hit = True
    else:
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

    # Invalidate calibration cache so next bet uses updated data
    from punty.betting.calibration import invalidate_cache
    invalidate_cache()

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
    stake = await get_current_stake(db, place_probability=0.75, odds=1.50)  # Representative
    stake_mode = await _get_setting(db, "betfair_stake_mode", "kelly")

    # Enrich bets with runner counts and place probability from linked pick
    enriched_bets = []
    for b in today_bets:
        d = b.to_dict()
        race_id = f"{b.meeting_id}-r{b.race_number}"
        d["runners"] = await _count_active_runners(db, race_id)
        # Attach place_probability from the linked pick
        if b.pick_id:
            pick_result = await db.execute(select(Pick).where(Pick.id == b.pick_id))
            pick = pick_result.scalar_one_or_none()
            d["place_probability"] = pick.place_probability if pick else None
        else:
            d["place_probability"] = None
        enriched_bets.append(d)

    return {
        "balance": balance,
        "initial_balance": initial_balance,
        "current_stake": stake,
        "today_bets": enriched_bets,
        "today_pnl": sum(b.pnl or 0 for b in today_bets if b.settled),
        "today_placed": sum(1 for b in today_bets if b.status in ("placed", "settled")),
        "today_queued": sum(1 for b in today_bets if b.status == "queued"),
        "total_bets": total_bets,
        "total_pnl": total_pnl,
        "total_hits": total_hits,
        "strike_rate": round(total_hits / total_bets * 100, 1) if total_bets else 0,
        "roi": round(total_pnl / (total_bets * 2) * 100, 1) if total_bets else 0,  # Approximate
        "stake_mode": stake_mode,
    }
