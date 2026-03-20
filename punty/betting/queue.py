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

from punty.betting.meta_model import meta_model_available, should_bet as meta_should_bet, extract_meta_features

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_INITIAL_BALANCE = 50.0
DEFAULT_BASE_STAKE = 2.0
DEFAULT_MIN_ODDS = 1.01  # Betfair exchange minimum (effectively no floor)
DEFAULT_COMMISSION_RATE = 0.00  # Betfair commission (0% with discount rate)
DEFAULT_MAX_DAILY_LOSS = -20.0
DEFAULT_MIN_PLACE_PROB = 0.40  # Absolute floor — field-size tiers set the real PP thresholds
DEFAULT_EDGE_MULTIPLIER = 1.10  # 10% edge over implied probability required
DEFAULT_DEAD_ZONE_LOW = 1.60  # Dead zone lower bound (skip bets in this range)
DEFAULT_DEAD_ZONE_HIGH = 2.00  # Dead zone upper bound
DEFAULT_MAX_KELLY_FRACTION = 0.06  # Cap Kelly fraction at 6% (half-Kelly conservative)
DEFAULT_KELLY_HALF = True  # True half-Kelly: halve the fraction for 75% less variance
DEFAULT_MIN_KELLY_STAKE = 5.00  # Betfair minimum bet size (AUD)
DEFAULT_MIN_CALIBRATED_PP = 0.50  # Only bet when calibrated PP >= 50% (volume for compound growth)
DEFAULT_MAX_PLACE_ODDS = 999.0  # No ceiling — let PP ranking decide
MAIDEN_PREFIXES = ("maiden",)  # Case-insensitive startswith check
DEFAULT_MAIDEN_MAX_PLACE_ODDS = 999.0  # No maiden ceiling — best 4 per meet by PP
# Select from all 4 ranked picks per race, then take best 4 per meeting by PP.
MAX_BETFAIR_RANK = 4
# Best 4 bets per meeting — pure PP ranking, no per-race or price gates.
MAX_BETS_PER_MEETING = 4


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

    Kelly fraction = (p * odds - 1) / (odds - 1), halved, capped at max_fraction.
    Stake = kelly_fraction * balance, floored at min_stake.
    """
    if odds <= 1 or balance <= 0 or place_probability <= 0:
        return 0  # No valid bet
    b = odds - 1
    kelly = (b * place_probability - (1 - place_probability)) / b
    if kelly <= 0:
        return 0  # Negative edge — don't bet
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


async def _meta_model_decision(
    db: AsyncSession,
    pick, runner, race, meeting, wp: float,
    race_picks: list, runner_count: int,
) -> tuple[bool, float, str] | None:
    """Run the meta-model to decide if this pick should be queued.

    Returns (should_bet, probability, reason) or None if model unavailable.
    Extracts meta-features from the pick/runner/race/meeting context.
    """
    if not meta_model_available():
        return None

    try:
        from punty.ml.features import (
            _distance_bucket, _track_cond_bucket, _class_bucket,
            _venue_type_code, _score_last_five, _form_trend,
            _parse_stats, _safe_float,
        )

        # WP margin — gap between this pick's WP and next best in race
        all_wps = sorted(
            [p.win_probability or 0 for p in race_picks if p.win_probability],
            reverse=True,
        )
        if len(all_wps) >= 2:
            wp_margin = all_wps[0] - all_wps[1]
        else:
            wp_margin = 0.0

        # Odds
        odds = pick.odds_at_tip or (pick.place_odds_at_tip * 3 if pick.place_odds_at_tip else 0) or 0

        # Buckets from race/meeting
        distance = race.distance if race else 1400
        dist_bucket = _distance_bucket(distance)
        cls_bucket = _class_bucket(race.class_ if race else "")
        tc = meeting.track_condition if meeting else ""
        tc_bucket = _track_cond_bucket(tc)
        venue_type = _venue_type_code(meeting.venue if meeting else "")

        # Runner-level features
        barrier_rel = float("nan")
        age = float("nan")
        days_since = float("nan")
        form_score = float("nan")
        form_trend_val = float("nan")
        speed_map_pos = float("nan")
        weight_diff = float("nan")
        career_win_pct = float("nan")
        career_place_pct = float("nan")

        if runner:
            barrier = runner.barrier or 0
            barrier_rel = (barrier - 1) / (runner_count - 1) if barrier and runner_count > 1 else float("nan")
            age = float(runner.horse_age) if runner.horse_age else float("nan")
            days_since = float(runner.days_since_last_run) if runner.days_since_last_run else float("nan")

            last_five = runner.last_five or ""
            form_score = _safe_float(_score_last_five(last_five))
            form_trend_val = _safe_float(_form_trend(last_five))

            smp = (runner.speed_map_position or "").lower()
            speed_map_pos = {"leader": 1.0, "on_pace": 2.0, "midfield": 3.0, "backmarker": 4.0}.get(smp, float("nan"))

            weight = runner.weight or 0
            weight_diff = float("nan")  # Would need field avg — use NaN (LightGBM handles it)

            career = _parse_stats(runner.career_record)
            if career and career[0] > 0:
                career_win_pct = career[1] / career[0]
                career_place_pct = (career[1] + career[2] + career[3]) / career[0]

        # Value rating = WP / market implied
        market_implied = 1.0 / odds if odds and odds > 1 else 0.0
        value_rating = wp / market_implied if market_implied > 0 else float("nan")

        features = extract_meta_features(
            wp=wp,
            wp_margin=wp_margin,
            odds=odds,
            field_size=runner_count,
            distance_bucket=dist_bucket,
            class_bucket=cls_bucket,
            track_cond_bucket=tc_bucket,
            venue_type=venue_type,
            barrier_relative=barrier_rel,
            age=age,
            days_since=days_since,
            form_score=form_score,
            form_trend=form_trend_val,
            value_rating=value_rating,
            speed_map_pos=speed_map_pos,
            weight_diff=weight_diff,
            career_win_pct=career_win_pct,
            career_place_pct=career_place_pct,
        )

        return meta_should_bet(features, threshold=0.65, wp=wp)

    except Exception as e:
        logger.warning("Meta-model decision failed: %s — falling back to WP", e)
        return None


async def populate_bet_queue(
    db: AsyncSession,
    meeting_id: str,
    content_id: str,
) -> int:
    """Create BetfairBet entries for the top 4 place-probability picks per meeting.

    Called after content approval. Collects the best PP pick from each race,
    ranks them across the whole meeting, and queues the top 4.
    No price gates — pure probability ranking decides.
    Returns count of bets queued.
    """
    auto_enabled = await _get_setting(db, "betfair_auto_bet_enabled", "true")
    if auto_enabled.lower() != "true":
        return 0

    # Load meeting for track condition context
    meeting_result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = meeting_result.scalar_one_or_none()
    meeting_tc = meeting.track_condition if meeting else ""

    # Get best pick per race by PP — consider all 4 ranked picks.
    result = await db.execute(
        select(Pick).where(
            Pick.content_id == content_id,
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "selection",
            Pick.tip_rank.in_(list(range(1, MAX_BETFAIR_RANK + 1))),
        ).order_by(Pick.race_number)
    )
    all_picks = result.scalars().all()
    # Select highest WIN PROBABILITY pick per race.
    # WP is context-aware (v7 LambdaRank with 102 interaction features) and
    # WP >= 22% yields 75.4% place SR at 18.4% ROI on 31-day backtest.
    # Runners with highest WP reliably place — simpler and more consistent
    # than using the Harville-derived PP which has normalisation artifacts.
    best_by_race: dict[int, Pick] = {}
    for pick in all_picks:
        wp = pick.win_probability or 0
        rn = pick.race_number
        if rn not in best_by_race or wp > (best_by_race[rn].win_probability or 0):
            best_by_race[rn] = pick
    race_picks = list(best_by_race.values())
    if not race_picks:
        return 0

    # Load races to get start times
    race_result = await db.execute(
        select(Race).where(Race.meeting_id == meeting_id)
    )
    races_by_num = {r.race_number: r for r in race_result.scalars().all()}

    # Count how many are already queued for this meeting
    existing_count_result = await db.execute(
        select(func.count(BetfairBet.id)).where(
            BetfairBet.meeting_id == meeting_id,
        )
    )
    existing_count = existing_count_result.scalar() or 0
    slots_remaining = max(0, MAX_BETS_PER_MEETING - existing_count)
    if slots_remaining == 0:
        return 0

    # Filter only: scratched runners and NTD (<5 runners = no place market)
    eligible = []
    for pick in race_picks:
        race = races_by_num.get(pick.race_number)
        if not race or not race.start_time:
            logger.debug(f"Skipping bet for {meeting_id} R{pick.race_number}: no start time")
            continue

        # NTD hard kill — fewer than 5 runners often means only 1 paid place
        race_id = f"{meeting_id}-r{race.race_number}"
        runner_count = await _count_active_runners(db, race_id)
        if runner_count < 5:
            logger.info(
                f"Skipping bet for {pick.horse_name} R{pick.race_number}: "
                f"NTD — {runner_count} runners (too few for place betting)"
            )
            continue

        # Check if already queued for this race (unique constraint)
        existing = await db.execute(
            select(BetfairBet).where(
                BetfairBet.meeting_id == meeting_id,
                BetfairBet.race_number == pick.race_number,
            )
        )
        if existing.scalar_one_or_none():
            continue

        # Check runner not scratched + load runner data for context filters
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

        # ── Context-aware queue filters ──
        # Hard blocks only for data dead zones where LGBM can't help.
        # For everything else, trust the LGBM's PP — the v7 model with
        # 102 context interaction features already penalises wide barriers,
        # backmarkers, long spells etc. via learned patterns.

        # 6yo+: 0% SR on 54 Betfair bets — genuine dead zone
        if runner and runner.horse_age and runner.horse_age >= 6:
            logger.info(f"Betfair queue BLOCKED: {pick.horse_name} R{pick.race_number} — age {runner.horse_age}yo")
            continue

        # ── Meta-model or WP floor ──
        # The meta-model is a learned selector trained on "does LGBM rank 1
        # actually place?" — it replaces the flat WP >= 22% threshold when
        # available. Falls back to WP >= 22% if model not loaded.
        wp = pick.win_probability or 0

        # Try meta-model first
        meta_decision = await _meta_model_decision(
            db, pick, runner, race, meeting, wp, race_picks, runner_count,
        )
        if meta_decision is not None:
            should, prob, reason = meta_decision
            if not should:
                logger.info(
                    f"Betfair queue SKIP: {pick.horse_name} R{pick.race_number} "
                    f"— {reason}"
                )
                continue
            logger.info(
                f"Betfair queue PASS: {pick.horse_name} R{pick.race_number} "
                f"— {reason}"
            )
        else:
            # Fallback: flat WP threshold
            if wp < 0.22:
                logger.info(
                    f"Betfair queue SKIP: {pick.horse_name} R{pick.race_number} "
                    f"— WP {wp:.0%} < 22%"
                )
                continue

        eligible.append((pick, race))

    if not eligible:
        return 0

    # Rank by WIN PROBABILITY across the entire meeting — take the top N
    eligible.sort(key=lambda x: x[0].win_probability or 0, reverse=True)
    top_picks = eligible[:slots_remaining]

    queued = 0
    for pick, race in top_picks:
        # Kelly stake — use $5 minimum even if Kelly says no edge
        pp = pick.place_probability or 0
        odds = pick.place_odds_at_tip or 0
        stake = await get_current_stake(db, place_probability=pp, odds=odds)
        if stake <= 0:
            stake = DEFAULT_MIN_KELLY_STAKE  # Force minimum stake — we trust PP ranking

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
        logger.info(
            f"Betfair queue: {pick.horse_name} R{pick.race_number} "
            f"PP={pp:.0%} odds=${odds:.2f} stake=${stake:.2f}"
        )

    if queued:
        await db.commit()
        logger.info(f"Betfair queue: {queued}/{MAX_BETS_PER_MEETING} bets queued for {meeting_id}")
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

    # Load all selection picks for this race — all ranks eligible
    pick_result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == bet.meeting_id,
            Pick.race_number == bet.race_number,
            Pick.pick_type == "selection",
            Pick.tip_rank.in_(list(range(1, MAX_BETFAIR_RANK + 1))),
        ).order_by(Pick.tip_rank)
    )
    candidates = pick_result.scalars().all()

    # Filter out scratched runners only — no price gates
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

    changes = 0

    for bet in queued_bets:
        race_id = f"{bet.meeting_id}-r{bet.race_number}"

        # NTD hard kill — fewer than 5 runners often means only 1 paid place
        runner_count = await _count_active_runners(db, race_id)
        if runner_count < 5:
            bet.status = "cancelled"
            bet.error_message = f"NTD — {runner_count} runners (too few for place betting)"
            changes += 1
            logger.info(f"Cancelled {bet.id}: NTD {runner_count} runners")
            continue

        # Load all selection picks for this race — all ranks eligible
        pick_result = await db.execute(
            select(Pick).where(
                Pick.meeting_id == bet.meeting_id,
                Pick.race_number == bet.race_number,
                Pick.pick_type == "selection",
                Pick.tip_rank.in_(list(range(1, MAX_BETFAIR_RANK + 1))),
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

        # Find best non-scratched candidate — pure PP ranking, no price gates
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
            pp = pick.place_probability or 0
            if pp > best_pp:
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
            # Recalculate Kelly stake with updated odds (no cancellation gates)
            pp_result = await db.execute(select(Pick).where(Pick.id == bet.pick_id))
            pick = pp_result.scalar_one_or_none()
            if pick and pick.place_probability:
                new_stake = await get_current_stake(
                    db, place_probability=pick.place_probability,
                    odds=bet.requested_odds,
                )
                if new_stake <= 0:
                    new_stake = DEFAULT_MIN_KELLY_STAKE  # Force min stake — trust PP
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

        # Kelly stake based on live odds and PP — force minimum if no edge
        stake = await get_current_stake(db, place_probability=place_prob,
                                         odds=current_odds if current_odds > 1 else 0)
        if stake <= 0:
            stake = DEFAULT_MIN_KELLY_STAKE  # Force min stake — trust PP ranking
            logger.info(
                f"Betfair: {bet.id} no Kelly edge (PP={place_prob:.0%}, odds=${current_odds:.2f}) "
                f"— using min stake ${stake:.2f}"
            )

        bet.stake = round(stake, 2)
        # BSP: use $1.01 as the floor price (Betfair exchange minimum)
        bet.requested_odds = 1.01

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
            bet.size_matched = bf_result.get("size_matched", bet.stake)

    commission_rate = float(await _get_setting(db, "betfair_commission_rate", str(DEFAULT_COMMISSION_RATE)))
    odds = bet.matched_odds or bet.requested_odds or 0

    now = melb_now_naive()
    # Betfair place cutoff: 8+ runners = top 3, 5-7 = top 2, ≤4 = win only
    if is_win_bet:
        hit_threshold = 1
    else:
        race_id_q = await db.execute(
            select(Race.id).where(Race.meeting_id == bet.meeting_id, Race.race_number == bet.race_number)
        )
        race_id_val = race_id_q.scalar()
        runner_count = await _count_active_runners(db, race_id_val) if race_id_val else 8
        hit_threshold = 3 if runner_count >= 8 else 2

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
