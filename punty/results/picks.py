"""Pick storage, settlement, and performance summary queries."""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import select, delete, func, case
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

        if runner and runner.finish_position is not None:
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

            if bet_type in ("win", "saver_win"):
                pick.hit = won
                if won and runner.win_dividend:
                    pick.pnl = round(runner.win_dividend * stake - stake, 2)
                else:
                    pick.pnl = round(-stake, 2)
            elif bet_type == "place":
                pick.hit = placed
                if placed and runner.place_dividend:
                    pick.pnl = round(runner.place_dividend * stake - stake, 2)
                else:
                    pick.pnl = round(-stake, 2)
            elif bet_type == "each_way":
                half = stake / 2
                if won and runner.win_dividend and runner.place_dividend:
                    pick.pnl = round(runner.win_dividend * half + runner.place_dividend * half - stake, 2)
                    pick.hit = True
                elif placed and runner.place_dividend:
                    pick.pnl = round(runner.place_dividend * half - stake, 2)
                    pick.hit = True
                else:
                    pick.pnl = round(-stake, 2)
                    pick.hit = False
            else:
                # Fallback: treat as win
                pick.hit = won
                if won and runner.win_dividend:
                    pick.pnl = round(runner.win_dividend * stake - stake, 2)
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

        if runner and runner.finish_position is not None:
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
                    else:
                        # Straight trifecta: exact order match
                        hit = list(top_sc_int[:3]) == list(exotic_runners_int[:3])
                if hit:
                    dividend = _find_dividend(exotic_divs, "trifecta")
            elif "exacta" in exotic_type:
                if len(top_sc_int) >= 2 and len(exotic_runners_int) >= 2:
                    if is_boxed:
                        hit = set(top_sc_int[:2]).issubset(set(exotic_runners_int))
                    else:
                        hit = list(top_sc_int[:2]) == list(exotic_runners_int[:2])
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
                    else:
                        hit = list(top_sc_int[:4]) == list(exotic_runners_int[:4])
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
                    # Estimate cost from combinations
                    legs = json.loads(pick.sequence_legs) if pick.sequence_legs else []
                    num_combos = 1
                    for leg in legs:
                        num_combos *= len(leg)
                    base_unit = pick.exotic_stake or 1.0
                    cost = num_combos * base_unit
                    seq_pnl = round(dividend - cost, 2)
                else:
                    # Hit but no dividend found — treat as loss of stake
                    legs_data = json.loads(pick.sequence_legs) if pick.sequence_legs else []
                    num_combos = 1
                    for leg in legs_data:
                        num_combos *= len(leg)
                    base_unit = pick.exotic_stake or 1.0
                    cost = num_combos * base_unit
                    seq_pnl = round(-cost, 2)
        else:
            # Lost — cost is combinations × base unit
            legs_data = json.loads(pick.sequence_legs) if pick.sequence_legs else []
            num_combos = 1
            for leg in legs_data:
                num_combos *= len(leg)
            base_unit = pick.exotic_stake or 1.0
            cost = num_combos * base_unit
            seq_pnl = round(-cost, 2)

        pick.pnl = seq_pnl
        pick.settled = True
        pick.settled_at = now
        settled_count += 1

    await db.flush()
    logger.info(f"Settled {settled_count} picks for {meeting_id} R{race_number}")
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
    """Get all-time daily P&L with running cumulative total for the chart."""
    result = await db.execute(
        select(
            Meeting.date,
            func.count(Pick.id).label("count"),
            func.sum(Pick.pnl).label("total_pnl"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("winners"),
            func.sum(Pick.bet_stake).label("total_bet_stake"),
            func.sum(Pick.exotic_stake).label("total_exotic_stake"),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Pick.settled == True,
            Pick.pick_type != "big3",
        )
        .group_by(Meeting.date)
        .order_by(Meeting.date)
    )
    rows = result.all()

    cumulative = 0.0
    cumulative_staked = 0.0
    cumulative_returned = 0.0
    days = []
    for row in rows:
        bets = row.count or 0
        pnl = float(row.total_pnl or 0)
        winners = int(row.winners or 0)
        staked = float(row.total_bet_stake or 0) + float(row.total_exotic_stake or 0)
        returned = staked + pnl
        cumulative += pnl
        cumulative_staked += staked
        cumulative_returned += returned
        days.append({
            "date": row.date.isoformat() if hasattr(row.date, 'isoformat') else str(row.date),
            "bets": bets,
            "winners": winners,
            "pnl": round(pnl, 2),
            "staked": round(staked, 2),
            "returned": round(returned, 2),
            "cumulative_pnl": round(cumulative, 2),
            "cumulative_staked": round(cumulative_staked, 2),
            "cumulative_returned": round(cumulative_returned, 2),
        })

    return days
