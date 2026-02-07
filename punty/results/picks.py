"""Pick storage, settlement, and performance summary queries."""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import select, delete, func, case, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.models.pick import Pick
from punty.models.meeting import Meeting, Race, Runner
from punty.models.content import Content
from punty.results.parser import parse_early_mail

logger = logging.getLogger(__name__)


async def store_picks_from_content(
    db: AsyncSession, content_id: str, meeting_id: str, raw_content: str
) -> int:
    """Parse early mail and store Pick rows. Idempotent — deletes existing picks first."""
    # Delete existing picks for this content
    await db.execute(delete(Pick).where(Pick.content_id == content_id))

    pick_dicts = parse_early_mail(raw_content, content_id, meeting_id)
    if not pick_dicts:
        logger.warning(f"No picks parsed from content {content_id}")
        return 0

    for pd in pick_dicts:
        pick = Pick(**pd)
        db.add(pick)

    await db.flush()
    logger.info(f"Stored {len(pick_dicts)} picks for content {content_id}")
    return len(pick_dicts)


async def settle_picks_for_race(
    db: AsyncSession, meeting_id: str, race_number: int
) -> int:
    """Settle unsettled picks that involve this race. Returns count settled.

    Uses try/except to ensure partial settlement doesn't corrupt data.
    On error, changes are rolled back and exception is re-raised.
    """
    try:
        return await _settle_picks_for_race_impl(db, meeting_id, race_number)
    except Exception as e:
        logger.error(f"Settlement failed for {meeting_id} R{race_number}: {e}")
        await db.rollback()
        raise


async def _settle_picks_for_race_impl(
    db: AsyncSession, meeting_id: str, race_number: int
) -> int:
    """Internal implementation of pick settlement."""
    now = melb_now_naive()
    settled_count = 0

    # Load race + runners for this race
    race_id = f"{meeting_id}-r{race_number}"
    result = await db.execute(select(Runner).where(Runner.race_id == race_id))
    runners = result.scalars().all()
    if not runners:
        return 0

    runners_by_saddlecloth = {r.saddlecloth: r for r in runners if r.saddlecloth}
    runners_by_name = {r.horse_name.upper(): r for r in runners}

    # Load race for exotic results
    race_result = await db.execute(select(Race).where(Race.id == race_id))
    race = race_result.scalar_one_or_none()

    # --- Selections ---
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type == "selection",
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        runner = None
        if pick.saddlecloth and pick.saddlecloth in runners_by_saddlecloth:
            runner = runners_by_saddlecloth[pick.saddlecloth]
        elif pick.horse_name:
            runner = runners_by_name.get(pick.horse_name.upper())

        # Settle if runner has finish position, OR if race is paying/closed (unplaced = loss)
        race_final = race and race.results_status in ("Paying", "Closed")
        has_result = runner and (runner.finish_position is not None or race_final)

        if has_result:
            bet_type = (pick.bet_type or "win").lower().replace(" ", "_")

            # "Exotics only" picks have no straight bet — settle with $0 P&L
            if bet_type == "exotics_only":
                pick.hit = runner.finish_position == 1
                pick.pnl = 0.0
                pick.settled = True
                pick.settled_at = now
                settled_count += 1
                continue

            stake = pick.bet_stake or 1.0
            won = runner.finish_position == 1
            placed = runner.finish_position is not None and runner.finish_position <= 3

            # Use fixed odds from tip time, fall back to tote dividends for backwards compatibility
            win_odds = pick.odds_at_tip or runner.win_dividend
            place_odds = pick.place_odds_at_tip or runner.place_dividend

            if bet_type in ("win", "saver_win"):
                pick.hit = won
                if won and win_odds:
                    pick.pnl = round(win_odds * stake - stake, 2)
                else:
                    pick.pnl = round(-stake, 2)
            elif bet_type == "place":
                pick.hit = placed
                if placed and place_odds:
                    pick.pnl = round(place_odds * stake - stake, 2)
                else:
                    pick.pnl = round(-stake, 2)
            elif bet_type == "each_way":
                half = stake / 2
                if won and win_odds and place_odds:
                    pick.pnl = round(win_odds * half + place_odds * half - stake, 2)
                    pick.hit = True
                elif placed and place_odds:
                    pick.pnl = round(place_odds * half - stake, 2)
                    pick.hit = True
                else:
                    pick.pnl = round(-stake, 2)
                    pick.hit = False
            else:
                # Fallback: treat as win
                pick.hit = won
                if won and win_odds:
                    pick.pnl = round(win_odds * stake - stake, 2)
                else:
                    pick.pnl = round(-stake, 2)

            pick.settled = True
            pick.settled_at = now
            settled_count += 1

    # --- Big3 individual horses ---
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type == "big3",
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        runner = None
        if pick.saddlecloth and pick.saddlecloth in runners_by_saddlecloth:
            runner = runners_by_saddlecloth[pick.saddlecloth]
        elif pick.horse_name:
            runner = runners_by_name.get(pick.horse_name.upper())

        # Settle if runner has finish position, OR if race is paying/closed (unplaced = loss)
        race_final = race and race.results_status in ("Paying", "Closed")
        has_result = runner and (runner.finish_position is not None or race_final)

        if has_result:
            pick.hit = runner.finish_position == 1
            pick.pnl = 0.0  # P&L tracked on the multi row
            pick.settled = True
            pick.settled_at = now
            settled_count += 1

    # --- Big3 multi — only settle when all 3 individual big3 picks are settled AND races complete ---
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "big3_multi",
            Pick.settled == False,
        )
    )
    for multi_pick in result.scalars().all():
        # Find all big3 individual picks for same content
        b3_result = await db.execute(
            select(Pick).where(
                Pick.content_id == multi_pick.content_id,
                Pick.pick_type == "big3",
            )
        )
        big3_picks = b3_result.scalars().all()
        if not big3_picks:
            continue

        all_settled = all(p.settled for p in big3_picks)
        if not all_settled:
            continue

        # CRITICAL: Verify all Big3 races have actually completed (not just picks settled)
        # This prevents race condition when multiple races complete simultaneously
        all_races_complete = True
        for b3_pick in big3_picks:
            if b3_pick.race_number:
                b3_race_id = f"{meeting_id}-r{b3_pick.race_number}"
                b3_race_result = await db.execute(select(Race).where(Race.id == b3_race_id))
                b3_race = b3_race_result.scalar_one_or_none()
                if not b3_race or b3_race.results_status not in ("Paying", "Closed"):
                    all_races_complete = False
                    break

        if not all_races_complete:
            continue

        all_won = all(p.hit for p in big3_picks)
        stake = multi_pick.exotic_stake or 10.0
        if all_won and multi_pick.multi_odds:
            multi_pick.pnl = round(multi_pick.multi_odds * stake - stake, 2)
        else:
            multi_pick.pnl = round(-stake, 2)
        multi_pick.hit = all_won
        multi_pick.settled = True
        multi_pick.settled_at = now
        settled_count += 1

    # --- Exotics ---
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type == "exotic",
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        try:
            exotic_runners = json.loads(pick.exotic_runners) if pick.exotic_runners else []
            stake = pick.exotic_stake or 1.0
            exotic_type = (pick.exotic_type or "").lower()

            # Get finish order
            finish_order = sorted(
                [r for r in runners if r.finish_position and r.finish_position <= 4],
                key=lambda r: r.finish_position,
            )
            top_saddlecloths = [r.saddlecloth for r in finish_order]

            hit = False
            dividend = 0.0

            # Parse exotic dividends from race
            exotic_divs = {}
            if race and race.exotic_results:
                try:
                    exotic_divs = json.loads(race.exotic_results)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Normalise saddlecloths to ints for consistent comparison
            exotic_runners_int = [int(x) for x in exotic_runners if str(x).isdigit()]
            top_sc_int = [int(x) for x in top_saddlecloths if x is not None]

            is_boxed = "box" in exotic_type or "standout" in exotic_type
            if "trifecta" in exotic_type:
                if len(top_sc_int) >= 3 and len(exotic_runners_int) >= 3:
                    if is_boxed:
                        # Boxed/standout: our runners in top 3 in any order
                        hit = set(top_sc_int[:3]).issubset(set(exotic_runners_int))
                    elif len(exotic_runners_int) == 3:
                        # Straight trifecta: exact order match
                        hit = list(top_sc_int[:3]) == list(exotic_runners_int[:3])
                    else:
                        # Flexi trifecta: first must win, rest must include 2nd and 3rd
                        # e.g., "1 to win, 2, 4 for second, 3, 5 for third" = [1, 2, 4, 3, 5]
                        first_ok = top_sc_int[0] == exotic_runners_int[0]
                        rest_ok = set(top_sc_int[1:3]).issubset(set(exotic_runners_int[1:]))
                        hit = first_ok and rest_ok
                if hit:
                    dividend = _find_dividend(exotic_divs, "trifecta")
            elif "exacta" in exotic_type:
                if len(top_sc_int) >= 2 and len(exotic_runners_int) >= 2:
                    if is_boxed:
                        # Boxed: our runners in top 2 in any order
                        hit = set(top_sc_int[:2]).issubset(set(exotic_runners_int))
                    elif len(exotic_runners_int) == 2:
                        # Straight exacta: exact order match
                        hit = list(top_sc_int[:2]) == list(exotic_runners_int[:2])
                    else:
                        # Standout/flexi: first runner must win, any of rest for second
                        # e.g., "1 to win, 2, 4 for second" = [1, 2, 4]
                        first_ok = top_sc_int[0] == exotic_runners_int[0]
                        second_ok = top_sc_int[1] in exotic_runners_int[1:]
                        hit = first_ok and second_ok
                if hit:
                    dividend = _find_dividend(exotic_divs, "exacta")
            elif "quinella" in exotic_type:
                if len(top_sc_int) >= 2 and len(exotic_runners_int) >= 2:
                    hit = set(top_sc_int[:2]).issubset(set(exotic_runners_int))
                if hit:
                    dividend = _find_dividend(exotic_divs, "quinella")
            elif "first" in exotic_type and ("four" in exotic_type or "4" in exotic_type):
                if len(top_sc_int) >= 4 and len(exotic_runners_int) >= 4:
                    if is_boxed:
                        hit = set(top_sc_int[:4]).issubset(set(exotic_runners_int))
                    elif len(exotic_runners_int) == 4:
                        # Straight first 4: exact order match
                        hit = list(top_sc_int[:4]) == list(exotic_runners_int[:4])
                    else:
                        # Flexi first 4: first must win, rest must include 2nd, 3rd, 4th
                        first_ok = top_sc_int[0] == exotic_runners_int[0]
                        rest_ok = set(top_sc_int[1:4]).issubset(set(exotic_runners_int[1:]))
                        hit = first_ok and rest_ok
                if hit:
                    dividend = _find_dividend(exotic_divs, "first4")

            # Stake is the total outlay for this exotic bet
            cost = stake

            if hit and dividend > 0:
                pick.pnl = round(dividend * stake - stake, 2)
            else:
                pick.pnl = round(-cost, 2)
            pick.hit = hit
            pick.settled = True
            pick.settled_at = now
            settled_count += 1
        except Exception as e:
            logger.error(f"Failed to settle exotic pick {pick.id}: {e}")
            # Continue with other exotics rather than crashing entire settlement
            continue

    # --- Sequences ---
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "sequence",
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        if not pick.sequence_legs or not pick.sequence_start_race:
            continue

        legs = json.loads(pick.sequence_legs)
        start = pick.sequence_start_race
        num_legs = len(legs)

        # Check if this race is part of this sequence
        if race_number < start or race_number >= start + num_legs:
            continue

        # Check if ALL legs have results
        all_resolved = True
        all_hit = True
        for leg_idx, leg_saddlecloths in enumerate(legs):
            leg_race_num = start + leg_idx
            leg_race_id = f"{meeting_id}-r{leg_race_num}"
            leg_race_result = await db.execute(
                select(Race).where(Race.id == leg_race_id)
            )
            leg_race = leg_race_result.scalar_one_or_none()
            if not leg_race or leg_race.results_status not in ("Paying", "Closed"):
                all_resolved = False
                break

            # Find winner of this leg
            winner_result = await db.execute(
                select(Runner).where(
                    Runner.race_id == leg_race_id,
                    Runner.finish_position == 1,
                )
            )
            winner = winner_result.scalar_one_or_none()
            if not winner or winner.saddlecloth not in leg_saddlecloths:
                all_hit = False

        if not all_resolved:
            continue

        pick.hit = all_hit

        # Look up sequence dividend from last leg's exotic_results
        seq_pnl = 0.0
        if all_hit:
            last_leg_race_id = f"{meeting_id}-r{start + num_legs - 1}"
            last_leg_result = await db.execute(
                select(Race).where(Race.id == last_leg_race_id)
            )
            last_leg_race = last_leg_result.scalar_one_or_none()
            if last_leg_race and last_leg_race.exotic_results:
                try:
                    exotic_divs = json.loads(last_leg_race.exotic_results)
                except (json.JSONDecodeError, TypeError):
                    exotic_divs = {}
                seq_type = (pick.sequence_type or "").lower()
                dividend = 0.0
                for key in ("quaddie", "quadrella", "big6", "big 6"):
                    if key in seq_type or seq_type == "":
                        dividend = _find_dividend(exotic_divs, key)
                        if dividend > 0:
                            break
                if dividend > 0:
                    # exotic_stake stores total outlay directly
                    # Calculate flexi return: dividend × (stake / num_combos)
                    legs_data = json.loads(pick.sequence_legs) if pick.sequence_legs else []
                    num_combos = 1
                    for leg in legs_data:
                        num_combos *= len(leg)
                    total_stake = pick.exotic_stake or 1.0
                    flexi_pct = total_stake / num_combos if num_combos > 0 else 1.0
                    return_amount = dividend * flexi_pct
                    seq_pnl = round(return_amount - total_stake, 2)
                else:
                    # Hit but no dividend found — treat as loss of stake
                    total_stake = pick.exotic_stake or 1.0
                    seq_pnl = round(-total_stake, 2)
        else:
            # Lost — exotic_stake stores total outlay directly
            total_stake = pick.exotic_stake or 1.0
            seq_pnl = round(-total_stake, 2)

        pick.pnl = seq_pnl
        pick.settled = True
        pick.settled_at = now
        settled_count += 1

    await db.flush()
    logger.info(f"Settled {settled_count} picks for {meeting_id} R{race_number}")

    # Update memory outcomes for the learning system
    if settled_count > 0:
        try:
            await update_memory_outcomes(db, meeting_id, race_number)
        except Exception as e:
            logger.warning(f"Failed to update memory outcomes: {e}")

        # Generate post-race assessment for RAG learning
        try:
            race_id = f"{meeting_id}-r{race_number}"
            from punty.memory.assessment import generate_race_assessment
            await generate_race_assessment(db, race_id)
        except Exception as e:
            logger.warning(f"Failed to generate race assessment: {e}")

    return settled_count


def _find_dividend(exotic_divs: dict, exotic_key: str) -> float:
    """Search exotic results dict for a dividend matching the key."""
    for key, val in exotic_divs.items():
        if exotic_key in key.lower():
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict) and "dividend" in val:
                return float(val["dividend"])
            if isinstance(val, str):
                try:
                    return float(val.replace("$", "").replace(",", ""))
                except ValueError:
                    pass
    return 0.0


async def get_performance_summary(db: AsyncSession, target_date: date) -> dict:
    """Get P&L summary for a given date, grouped by pick_type."""
    result = await db.execute(
        select(
            Pick.pick_type,
            func.count(Pick.id).label("count"),
            func.sum(Pick.pnl).label("total_pnl"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("winners"),
            func.sum(Pick.exotic_stake).label("total_staked_exotic"),
            func.sum(Pick.bet_stake).label("total_staked_bet"),
        )
        .join(Content, Pick.content_id == Content.id)
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Meeting.date == target_date,
            Pick.settled == True,
        )
        .group_by(Pick.pick_type)
    )
    rows = result.all()

    # For sequences, we need to calculate actual stake (combos × unit_price)
    # Query all sequence picks to get their legs for proper stake calculation
    seq_result = await db.execute(
        select(Pick)
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Meeting.date == target_date,
            Pick.settled == True,
            Pick.pick_type == "sequence",
        )
    )
    sequence_picks = seq_result.scalars().all()
    sequence_total_stake = 0.0
    for seq_pick in sequence_picks:
        if seq_pick.sequence_legs and seq_pick.exotic_stake:
            try:
                legs = json.loads(seq_pick.sequence_legs)
                combos = 1
                for leg in legs:
                    combos *= len(leg)
                sequence_total_stake += combos * seq_pick.exotic_stake
            except (json.JSONDecodeError, TypeError):
                sequence_total_stake += seq_pick.exotic_stake or 0

    by_product = {}
    total_bets = 0
    total_winners = 0
    total_pnl = 0.0
    total_staked = 0.0

    for row in rows:
        pick_type = row.pick_type
        count = row.count or 0
        pnl = float(row.total_pnl or 0)
        winners = int(row.winners or 0)

        # Estimate stake: selections use bet_stake sum, exotics/big3_multi use exotic_stake
        if pick_type == "selection":
            staked = float(row.total_staked_bet or count)  # sum of bet_stake, fallback to count
        elif pick_type == "big3":
            staked = 0.0  # P&L tracked on multi row
        elif pick_type == "big3_multi":
            staked = float(row.total_staked_exotic or 10.0)
        elif pick_type == "sequence":
            # Use properly calculated stake (combos × unit_price)
            staked = sequence_total_stake
        else:
            staked = float(row.total_staked_exotic or count)

        strike_rate = (winners / count * 100) if count > 0 else 0.0

        by_product[pick_type] = {
            "bets": count,
            "winners": winners,
            "strike_rate": round(strike_rate, 1),
            "staked": round(staked, 2),
            "pnl": round(pnl, 2),
        }
        total_bets += count
        total_winners += winners
        total_pnl += pnl
        total_staked += staked

    overall_strike = (total_winners / total_bets * 100) if total_bets > 0 else 0.0

    return {
        "date": target_date.isoformat(),
        "total_bets": total_bets,
        "total_winners": total_winners,
        "total_strike_rate": round(overall_strike, 1),
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_staked + total_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "by_product": by_product,
    }


async def get_performance_history(
    db: AsyncSession, start_date: date, end_date: date
) -> list[dict]:
    """Get daily P&L summaries for a date range."""
    result = await db.execute(
        select(
            Meeting.date,
            func.count(Pick.id).label("count"),
            func.sum(Pick.pnl).label("total_pnl"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("winners"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Meeting.date >= start_date,
            Meeting.date <= end_date,
            Pick.settled == True,
            Pick.pick_type != "big3",  # P&L tracked on multi row
        )
        .group_by(Meeting.date)
        .order_by(Meeting.date)
    )
    rows = result.all()

    days = []
    for row in rows:
        bets = row.count or 0
        pnl = float(row.total_pnl or 0)
        winners = int(row.winners or 0)
        strike = (winners / bets * 100) if bets > 0 else 0.0
        days.append({
            "date": row.date.isoformat() if hasattr(row.date, 'isoformat') else str(row.date),
            "bets": bets,
            "winners": winners,
            "strike_rate": round(strike, 1),
            "pnl": round(pnl, 2),
        })

    return days


async def get_cumulative_pnl(db: AsyncSession) -> list[dict]:
    """Get all-time P&L with adaptive time scaling based on race start times.

    - Today: hourly granularity (by race time)
    - Last 7 days: daily granularity
    - Last 60 days: daily granularity
    - Older: weekly granularity

    Sequences and big3_multi are attributed to their concluding race time.
    Chart starts at zero.
    """
    from punty.config import melb_today
    import json as _json

    today = melb_today()
    week_ago = today - timedelta(days=7)
    two_months_ago = today - timedelta(days=60)

    # Get all settled picks with their race times
    # For selections/exotics: use their race_number
    # For sequences: use sequence_start_race + len(legs) - 1
    # For big3_multi: need to find the max race among the 3 picks

    # Exclude losing sequences from all-time P&L per user request
    # Only include sequences if they hit (won)
    result = await db.execute(
        select(Pick, Race.start_time, Meeting.date)
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .outerjoin(Race, (Race.meeting_id == Pick.meeting_id) & (Race.race_number == Pick.race_number))
        .where(
            Pick.settled == True,
            Pick.pick_type != "big3",  # big3 individual rows don't have P&L, only big3_multi does
            # Exclude losing sequences - only include if not a sequence OR if sequence that hit
            or_(
                Pick.pick_type != "sequence",
                and_(Pick.pick_type == "sequence", Pick.hit == True)
            ),
        )
    )
    rows = result.all()

    # For sequences and big3_multi, we need to look up the concluding race time
    # Build a cache of race start times by meeting_id and race_number
    race_times_cache = {}
    all_meeting_ids = set(r[0].meeting_id for r in rows)

    if all_meeting_ids:
        race_result = await db.execute(
            select(Race.meeting_id, Race.race_number, Race.start_time)
            .where(Race.meeting_id.in_(all_meeting_ids))
        )
        for race_row in race_result.all():
            key = (race_row.meeting_id, race_row.race_number)
            race_times_cache[key] = race_row.start_time

    # Process each pick and determine its effective race time
    pick_events = []
    for pick, race_start_time, meeting_date in rows:
        effective_time = race_start_time
        effective_race_num = pick.race_number

        if pick.pick_type == "sequence" and pick.sequence_start_race and pick.sequence_legs:
            # Concluding race = start_race + num_legs - 1
            try:
                legs = _json.loads(pick.sequence_legs)
                concluding_race = pick.sequence_start_race + len(legs) - 1
                effective_race_num = concluding_race
                effective_time = race_times_cache.get((pick.meeting_id, concluding_race))
            except:
                pass

        elif pick.pick_type == "big3_multi":
            # Find the max race number among the big3 picks for this content
            big3_result = await db.execute(
                select(func.max(Pick.race_number))
                .where(
                    Pick.content_id == pick.content_id,
                    Pick.pick_type == "big3",
                )
            )
            max_race = big3_result.scalar()
            if max_race:
                effective_race_num = max_race
                effective_time = race_times_cache.get((pick.meeting_id, max_race))

        # Use meeting date + race time for sorting, fallback to just date
        if effective_time:
            sort_datetime = datetime.combine(meeting_date, effective_time.time()) if hasattr(effective_time, 'time') else effective_time
        else:
            # Fallback: use meeting date with estimated time based on race number
            estimated_hour = 12 + (effective_race_num or 1) - 1  # Race 1 at 12:00, Race 2 at 13:00, etc.
            sort_datetime = datetime.combine(meeting_date, datetime.min.time().replace(hour=min(estimated_hour, 18)))

        # Calculate actual staked amount
        # For sequences, exotic_stake is unit price - need to multiply by combos
        if pick.pick_type == "sequence" and pick.sequence_legs and pick.exotic_stake:
            try:
                legs = _json.loads(pick.sequence_legs)
                combos = 1
                for leg in legs:
                    combos *= len(leg)
                staked = combos * pick.exotic_stake
            except (json.JSONDecodeError, TypeError):
                staked = float(pick.exotic_stake or 0)
        else:
            staked = float(pick.bet_stake or 0) + float(pick.exotic_stake or 0)

        pick_events.append({
            "sort_datetime": sort_datetime,
            "meeting_date": meeting_date,
            "pnl": float(pick.pnl or 0),
            "staked": staked,
            "hit": pick.hit or False,
            "pick_type": pick.pick_type,
        })

    # Sort by race time
    pick_events.sort(key=lambda x: x["sort_datetime"])

    # Group picks into periods based on adaptive granularity
    periods_dict = {}  # key -> {bets, winners, pnl, staked}

    for event in pick_events:
        meeting_date = event["meeting_date"]
        sort_dt = event["sort_datetime"]

        if meeting_date == today:
            # Hourly for today
            period_key = sort_dt.strftime('%Y-%m-%d %H:00')
            period_type = "hour"
            label = sort_dt.strftime('%H:00')
        elif meeting_date >= week_ago:
            # Daily for last 7 days
            period_key = meeting_date.isoformat()
            period_type = "day"
            label = meeting_date.strftime('%a %d')
        elif meeting_date >= two_months_ago:
            # Daily for 8-60 days ago
            period_key = meeting_date.isoformat()
            period_type = "day"
            label = meeting_date.strftime('%d/%m')
        else:
            # Weekly for older
            week_num = meeting_date.isocalendar()[1]
            year = meeting_date.year
            period_key = f"{year}-W{week_num:02d}"
            period_type = "week"
            label = f"W{week_num:02d}"

        if period_key not in periods_dict:
            periods_dict[period_key] = {
                "sort_key": period_key,
                "label": label,
                "period_type": period_type,
                "date": meeting_date.isoformat() if isinstance(meeting_date, date) else str(meeting_date),
                "bets": 0,
                "winners": 0,
                "pnl": 0.0,
                "staked": 0.0,
                # Track by pick type
                "pnl_selection": 0.0,
                "pnl_exotic": 0.0,
                "pnl_sequence": 0.0,
                "pnl_big3_multi": 0.0,
            }

        periods_dict[period_key]["bets"] += 1
        periods_dict[period_key]["winners"] += 1 if event["hit"] else 0
        periods_dict[period_key]["pnl"] += event["pnl"]
        periods_dict[period_key]["staked"] += event["staked"]

        # Track P&L by pick type
        ptype = event["pick_type"]
        if ptype == "selection":
            periods_dict[period_key]["pnl_selection"] += event["pnl"]
        elif ptype == "exotic":
            periods_dict[period_key]["pnl_exotic"] += event["pnl"]
        elif ptype == "sequence":
            periods_dict[period_key]["pnl_sequence"] += event["pnl"]
        elif ptype == "big3_multi":
            periods_dict[period_key]["pnl_big3_multi"] += event["pnl"]

    # Sort periods chronologically
    sorted_periods = sorted(periods_dict.values(), key=lambda x: x["sort_key"])

    # Build result with cumulative values, starting at zero
    result_periods = []

    # Add starting zero point
    if sorted_periods:
        first_date = sorted_periods[0]["date"]
        result_periods.append({
            "date": first_date,
            "label": "Start",
            "period_type": "start",
            "bets": 0,
            "winners": 0,
            "pnl": 0,
            "staked": 0,
            "returned": 0,
            "cumulative_pnl": 0,
            "cumulative_staked": 0,
            "cumulative_returned": 0,
            "cumulative_selection": 0,
            "cumulative_exotic": 0,
            "cumulative_sequence": 0,
            "cumulative_big3_multi": 0,
        })

    cumulative = 0.0
    cumulative_staked = 0.0
    cumulative_returned = 0.0
    cumulative_selection = 0.0
    cumulative_exotic = 0.0
    cumulative_sequence = 0.0
    cumulative_big3_multi = 0.0

    for p in sorted_periods:
        pnl = p["pnl"]
        staked = p["staked"]
        returned = staked + pnl
        cumulative += pnl
        cumulative_staked += staked
        cumulative_returned += returned
        cumulative_selection += p.get("pnl_selection", 0)
        cumulative_exotic += p.get("pnl_exotic", 0)
        cumulative_sequence += p.get("pnl_sequence", 0)
        cumulative_big3_multi += p.get("pnl_big3_multi", 0)

        result_periods.append({
            "date": p["date"],
            "label": p["label"],
            "period_type": p["period_type"],
            "bets": p["bets"],
            "winners": p["winners"],
            "pnl": round(pnl, 2),
            "staked": round(staked, 2),
            "returned": round(returned, 2),
            "cumulative_pnl": round(cumulative, 2),
            "cumulative_staked": round(cumulative_staked, 2),
            "cumulative_returned": round(cumulative_returned, 2),
            "cumulative_selection": round(cumulative_selection, 2),
            "cumulative_exotic": round(cumulative_exotic, 2),
            "cumulative_sequence": round(cumulative_sequence, 2),
            "cumulative_big3_multi": round(cumulative_big3_multi, 2),
        })

    return result_periods


async def store_picks_as_memories(
    db: AsyncSession, meeting_id: str, content_id: str
) -> int:
    """Store selection picks as memories for pattern learning.

    Should be called after content is approved and picks are parsed.
    Only stores 'selection' picks (not exotics/sequences) as these
    represent our core predictions.
    """
    from punty.memory.models import RaceMemory
    from punty.memory.embeddings import EmbeddingService
    from punty.models.settings import get_api_key

    # Get all selection picks for this content
    result = await db.execute(
        select(Pick).where(
            Pick.content_id == content_id,
            Pick.pick_type == "selection",
        )
    )
    picks = result.scalars().all()
    if not picks:
        return 0

    # Get meeting and race data for context
    meeting_result = await db.execute(
        select(Meeting).where(Meeting.id == meeting_id)
    )
    meeting = meeting_result.scalar_one_or_none()
    if not meeting:
        return 0

    # Get API key from same session to avoid database locking
    api_key = await get_api_key(db, "openai_api_key")
    embedding_service = EmbeddingService(api_key=api_key)
    stored = 0

    for pick in picks:
        race_id = f"{meeting_id}-r{pick.race_number}"

        # Check if memory already exists
        existing = await db.execute(
            select(RaceMemory).where(
                RaceMemory.race_id == race_id,
                RaceMemory.saddlecloth == pick.saddlecloth,
            )
        )
        if existing.scalar_one_or_none():
            continue

        # Get race and runner data
        race_result = await db.execute(select(Race).where(Race.id == race_id))
        race = race_result.scalar_one_or_none()

        runner_result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.saddlecloth == pick.saddlecloth,
            )
        )
        runner = runner_result.scalar_one_or_none()

        if not race or not runner:
            continue

        # Build context and runner dictionaries
        race_context = {
            "track_condition": meeting.track_condition,
            "distance": race.distance,
            "class": race.class_,
            "rail_position": meeting.rail_position,
            "weather": meeting.weather,
        }

        runner_data = {
            "horse_name": runner.horse_name,
            "saddlecloth": runner.saddlecloth,
            "barrier": runner.barrier,
            "jockey": runner.jockey,
            "trainer": runner.trainer,
            "form": runner.form,
            "current_odds": runner.current_odds,
            "speed_map_position": runner.speed_map_position,
            "days_since_last_run": runner.days_since_last_run,
            "horse_age": runner.horse_age,
            "pf_map_factor": runner.pf_map_factor,
        }

        # Parse market movement from flucs if available
        if runner.odds_flucs:
            try:
                flucs = json.loads(runner.odds_flucs)
                if flucs and len(flucs) >= 2:
                    opening = flucs[-1].get("odds", 0)
                    current = flucs[0].get("odds", 0)
                    if opening and current:
                        pct_change = (current - opening) / opening * 100
                        if pct_change <= -20:
                            runner_data["odds_movement"] = "heavy_support"
                        elif pct_change <= -10:
                            runner_data["odds_movement"] = "firming"
                        elif pct_change >= 30:
                            runner_data["odds_movement"] = "big_drift"
                        elif pct_change >= 15:
                            runner_data["odds_movement"] = "drifting"
                        else:
                            runner_data["odds_movement"] = "stable"
            except (json.JSONDecodeError, TypeError):
                pass

        # Create memory
        memory = RaceMemory(
            meeting_id=meeting_id,
            race_number=pick.race_number,
            race_id=race_id,
            horse_name=pick.horse_name or runner.horse_name,
            saddlecloth=pick.saddlecloth,
            tip_rank=pick.tip_rank or 0,
            confidence=None,  # Could extract from content parsing
            odds_at_tip=pick.odds_at_tip,
            bet_type=pick.bet_type,
        )
        memory.context = race_context
        memory.runner = runner_data

        # Generate embedding (async)
        try:
            embedding = await embedding_service.embed_context(race_context, runner_data)
            if embedding:
                memory.embedding = embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {race_id}: {e}")

        db.add(memory)
        stored += 1

    await db.flush()
    logger.info(f"Stored {stored} memories for content {content_id}")
    return stored


async def update_memory_outcomes(
    db: AsyncSession, meeting_id: str, race_number: int
) -> int:
    """Update memories with race outcomes after settlement.

    Should be called after settle_picks_for_race.
    """
    from punty.memory.models import RaceMemory

    race_id = f"{meeting_id}-r{race_number}"
    now = melb_now_naive()

    # Get race results
    result = await db.execute(select(Runner).where(Runner.race_id == race_id))
    runners = result.scalars().all()
    if not runners:
        return 0

    runners_by_saddlecloth = {r.saddlecloth: r for r in runners if r.saddlecloth}

    # Get corresponding picks to get settlement info
    picks_result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type == "selection",
            Pick.settled == True,
        )
    )
    picks_by_sc = {p.saddlecloth: p for p in picks_result.scalars().all()}

    # Update memories
    mem_result = await db.execute(
        select(RaceMemory).where(
            RaceMemory.race_id == race_id,
            RaceMemory.settled_at.is_(None),
        )
    )
    updated = 0
    for memory in mem_result.scalars().all():
        runner = runners_by_saddlecloth.get(memory.saddlecloth)
        pick = picks_by_sc.get(memory.saddlecloth)

        if runner:
            memory.finish_position = runner.finish_position
            memory.sp_odds = runner.current_odds  # Use current as SP proxy

        if pick:
            memory.hit = pick.hit
            memory.pnl = pick.pnl

        memory.settled_at = now
        updated += 1

    await db.flush()
    if updated:
        logger.info(f"Updated {updated} memory outcomes for {race_id}")
    return updated


async def get_recent_wins(db: AsyncSession, limit: int = 20) -> list[dict]:
    """Get recent winning picks with celebration phrases.

    Returns list of dicts with win details and a Punty celebration phrase.
    """
    from punty.results.celebrations import get_celebration

    # Get recent settled wins, ordered by settled_at descending
    result = await db.execute(
        select(Pick, Meeting)
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Pick.hit == True,
            Pick.pnl > 0,  # Only profitable wins
        )
        .order_by(Pick.settled_at.desc())
        .limit(limit)
    )
    rows = result.all()

    wins = []
    for pick, meeting in rows:
        # Calculate stake and return
        stake = pick.bet_stake or pick.exotic_stake or 1.0
        returned = stake + (pick.pnl or 0)

        # Build display name
        if pick.pick_type == "selection":
            display_name = f"{pick.horse_name or 'Runner'} R{pick.race_number}"
        elif pick.pick_type == "exotic":
            display_name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"
        elif pick.pick_type == "sequence":
            display_name = f"{pick.sequence_type or 'Sequence'}"
        elif pick.pick_type == "big3_multi":
            display_name = "Big 3 Multi"
        else:
            display_name = f"Win R{pick.race_number}"

        wins.append({
            "id": pick.id,
            "venue": meeting.venue,
            "display_name": display_name,
            "pick_type": pick.pick_type,
            "stake": round(stake, 2),
            "returned": round(returned, 2),
            "pnl": round(pick.pnl, 2),
            "celebration": get_celebration(pick.pnl, pick.pick_type),
            "settled_at": pick.settled_at.isoformat() if pick.settled_at else None,
        })

    return wins


async def get_all_time_stats(db: AsyncSession) -> dict:
    """Get all-time win statistics for dashboard display."""
    today = melb_now_naive().date()

    # Today's winners
    today_result = await db.execute(
        select(func.count(Pick.id))
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Pick.hit == True,
            Meeting.date == today,
        )
    )
    today_winners = today_result.scalar() or 0

    # All-time winners
    total_result = await db.execute(
        select(func.count(Pick.id)).where(
            Pick.settled == True,
            Pick.hit == True,
        )
    )
    total_winners = total_result.scalar() or 0

    # All-time collected (returned from wins)
    collected_result = await db.execute(
        select(
            func.sum(Pick.bet_stake + Pick.pnl).filter(Pick.bet_stake.isnot(None)),
            func.sum(Pick.exotic_stake + Pick.pnl).filter(Pick.exotic_stake.isnot(None)),
        ).where(
            Pick.settled == True,
            Pick.hit == True,
        )
    )
    row = collected_result.one()
    collected = (row[0] or 0) + (row[1] or 0)

    return {
        "today_winners": today_winners,
        "total_winners": total_winners,
        "collected": round(collected, 0),
    }
