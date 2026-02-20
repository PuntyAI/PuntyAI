"""Pick storage, settlement, and performance summary queries."""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import select, delete, func, case, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
    from punty.probability import calculate_race_probabilities
    from punty.models.meeting import Meeting, Race, Runner

    # Delete existing picks for this content
    await db.execute(delete(Pick).where(Pick.content_id == content_id))

    # Also delete picks from superseded/rejected content for the same meeting
    # This catches orphan picks from manual approvals or bypassed workflows
    from punty.models.content import Content, ContentStatus
    stale_ids_result = await db.execute(
        select(Content.id).where(
            Content.meeting_id == meeting_id,
            Content.content_type == "early_mail",
            Content.status.in_([ContentStatus.SUPERSEDED.value, ContentStatus.REJECTED.value]),
        )
    )
    stale_ids = stale_ids_result.scalars().all()
    if stale_ids:
        await db.execute(delete(Pick).where(Pick.content_id.in_(stale_ids)))

    pick_dicts = parse_early_mail(raw_content, content_id, meeting_id)
    if not pick_dicts:
        logger.warning(f"No picks parsed from content {content_id} — flagging for review")
        # Flag content for manual review when parser finds zero picks
        from punty.models.content import Content
        content_result = await db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = content_result.scalar_one_or_none()
        if content and content.status not in ("sent", "delivery_failed"):
            content.status = "pending_review"
            await db.commit()
        return 0

    # Calculate probabilities for each race to attach to picks
    race_probs = {}
    try:
        result = await db.execute(
            select(Meeting).where(Meeting.id == meeting_id)
            .options(
                selectinload(Meeting.races).selectinload(Race.runners)
            )
        )
        meeting = result.scalar_one_or_none()
        if meeting:
            for race in meeting.races:
                active = [r for r in race.runners if not r.scratched]
                if active:
                    probs = calculate_race_probabilities(active, race, meeting)
                    # Map by saddlecloth for easy lookup
                    for runner in active:
                        if runner.id in probs:
                            rp = probs[runner.id]
                            race_probs[(race.race_number, runner.saddlecloth)] = rp
    except Exception as e:
        logger.warning(f"Could not calculate probabilities for picks: {e}")

    for pd in pick_dicts:
        # Attach calculated probability if not already parsed from text
        if pd.get("pick_type") == "selection" and pd.get("saddlecloth"):
            key = (pd.get("race_number"), pd.get("saddlecloth"))
            rp = race_probs.get(key)
            if rp:
                if not pd.get("win_probability"):
                    pd["win_probability"] = rp.win_probability
                if not pd.get("value_rating"):
                    pd["value_rating"] = rp.value_rating
                if not pd.get("recommended_stake"):
                    pd["recommended_stake"] = rp.recommended_stake
                pd.setdefault("place_probability", rp.place_probability)
                # Store factor breakdown for self-tuning analysis
                if rp.factors:
                    pd["factors_json"] = json.dumps(rp.factors)

        pick = Pick(**pd)
        db.add(pick)

    await db.flush()
    logger.info(f"Stored {len(pick_dicts)} picks for content {content_id}")
    return len(pick_dicts)


async def void_picks_for_meeting(db: AsyncSession, meeting_id: str) -> int:
    """Void all unsettled picks for an abandoned meeting. Returns count voided.

    Abandoned = no win, no loss. Sets pnl=0, hit=False, settled=True.
    """
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.settled == False,
        )
    )
    picks = result.scalars().all()
    now = melb_now_naive()
    count = 0
    for pick in picks:
        pick.settled = True
        pick.hit = False
        pick.pnl = 0.0
        pick.settled_at = now
        count += 1
    if count:
        await db.flush()
        logger.info(f"Voided {count} picks for abandoned meeting {meeting_id}")
    return count


async def void_picks_for_race(db: AsyncSession, meeting_id: str, race_number: int) -> int:
    """Void unsettled picks for a single abandoned race. Returns count voided.

    Abandoned = no win, no loss. Sets pnl=0, hit=False, settled=True.
    Only voids selection and exotic picks for this race (not sequences/big3 which span races).
    """
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.settled == False,
            Pick.pick_type.in_(["selection", "exotic"]),
        )
    )
    picks = result.scalars().all()
    now = melb_now_naive()
    count = 0
    for pick in picks:
        pick.settled = True
        pick.hit = False
        pick.pnl = 0.0
        pick.settled_at = now
        count += 1
    if count:
        await db.flush()
        logger.info(f"Voided {count} picks for abandoned race {meeting_id} R{race_number}")
    return count


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

    # Alert if all runners are scratched (race abandoned)
    active_runners = [r for r in runners if not r.scratched]
    if not active_runners:
        logger.warning(
            f"All runners scratched in {race_id} — race likely abandoned. "
            f"Voiding all unsettled picks for this race."
        )
        # Void all unsettled picks for this abandoned race
        unsettled = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting_id,
                Pick.race_number == race_number,
                Pick.settled == False,
            )
        )
        for pick in unsettled.scalars().all():
            pick.pnl = 0.0
            pick.hit = False
            pick.settled = True
            pick.settled_at = now
            settled_count += 1
        await db.commit()
        return settled_count

    runners_by_saddlecloth = {r.saddlecloth: r for r in runners if r.saddlecloth}
    runners_by_name = {r.horse_name.upper(): r for r in runners}

    # Load race for field_size and exotic results
    race_result = await db.execute(select(Race).where(Race.id == race_id))
    race = race_result.scalar_one_or_none()

    # Determine number of paying places based on field size (TAB NTD rules):
    # - 8+ starters at final scratching time → 3 places paid
    # - 5-7 starters → 2 places paid (No Third Dividend / NTD)
    # - ≤4 starters → no place betting
    # Key: if 8+ at final scratchings but late scratches reduce to 5-7, 3 places STILL apply.
    # race.field_size = original field count from scrape time (before late scratchings).
    # len(active_runners) = post-late-scratching count (runners with results).
    original_field = (race.field_size if race and race.field_size else None) or len(active_runners)
    post_scratch_field = len(active_runners)

    # Use the HIGHER of original field and post-scratch field for num_places.
    # If original field was 8+ but late scratches reduced to 5-7, TAB still pays 3 places.
    effective_field = max(original_field, post_scratch_field)
    if effective_field <= 4:
        num_places = 0  # No place betting
    elif effective_field <= 7:
        num_places = 2  # NTD — 1st and 2nd only
    else:
        num_places = 3  # 1st, 2nd, and 3rd

    field_size = post_scratch_field  # for logging

    # Check if results are actually populated (not just status flipped)
    race_final = race and race.results_status in ("Paying", "Closed")
    results_populated = any(r.finish_position is not None for r in runners)
    if race_final and not results_populated:
        logger.warning(
            f"Race {race_id} status is {race.results_status} but no runners "
            f"have finish positions yet — skipping settlement"
        )
        return 0

    # --- Dead heat detection ---
    # Count runners sharing the same finish position (dead heat = count > 1)
    from collections import Counter
    position_counts = Counter(
        r.finish_position for r in runners
        if r.finish_position is not None and r.finish_position > 0
    )
    dead_heat_divisors = {pos: count for pos, count in position_counts.items() if count > 1}
    if dead_heat_divisors:
        logger.info(f"Dead heat detected in {race_id}: positions {dead_heat_divisors}")

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

        # Settle if runner has finish position, OR if race is final with results populated
        has_result = runner and (runner.finish_position is not None or (race_final and results_populated))

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
            placed = runner.finish_position is not None and runner.finish_position <= num_places

            # No Third Dividend: if runner finished 3rd but has no place dividend,
            # TAB didn't pay a 3rd place (small field). Treat as NOT placed.
            if placed and not won and runner.finish_position == 3:
                actual_div = runner.place_dividend
                if not actual_div or actual_div <= 0:
                    logger.info(
                        f"No third dividend for {pick.horse_name} R{pick.race_number} "
                        f"(3rd in {field_size}-runner field). Treating as loss."
                    )
                    placed = False

            # Use fixed odds from tip time, fall back to tote dividends
            win_odds = pick.odds_at_tip or runner.win_dividend
            place_odds = pick.place_odds_at_tip or runner.place_dividend

            # Sanity: place odds should NEVER exceed win odds (impossible in real markets).
            # If they do, the place odds are likely garbage (e.g. win $2.70 / place $6.00).
            # Fall back to estimated place from win: (win - 1) / 3 + 1
            if win_odds and place_odds and place_odds > win_odds:
                estimated_place = round((win_odds - 1) / 3 + 1, 2)
                logger.warning(
                    f"Place odds > win odds for {pick.horse_name} R{pick.race_number}: "
                    f"win=${win_odds:.2f} place=${place_odds:.2f}. "
                    f"Using tote place={runner.place_dividend} or estimate=${estimated_place:.2f}"
                )
                # Prefer tote dividend if available, otherwise estimate from win odds
                if runner.place_dividend and runner.place_dividend > 0:
                    place_odds = runner.place_dividend
                else:
                    place_odds = estimated_place

            # Extreme ratio guard: if tip-time odds are wildly different from tote
            # (>10x), the tip-time odds are likely garbage (e.g. Kingscote/King Island).
            if win_odds and runner.win_dividend and runner.win_dividend > 0:
                ratio = win_odds / runner.win_dividend
                if ratio > 10 or ratio < 0.1:
                    logger.warning(
                        f"Odds sanity guard: {pick.horse_name} R{pick.race_number} "
                        f"win fixed={win_odds:.2f} vs tote={runner.win_dividend:.2f} "
                        f"(ratio={ratio:.1f}x). Using tote dividend."
                    )
                    win_odds = runner.win_dividend
            if place_odds and runner.place_dividend and runner.place_dividend > 0:
                ratio = place_odds / runner.place_dividend
                if ratio > 10 or ratio < 0.1:
                    logger.warning(
                        f"Odds sanity guard: {pick.horse_name} R{pick.race_number} "
                        f"place fixed={place_odds:.2f} vs tote={runner.place_dividend:.2f} "
                        f"(ratio={ratio:.1f}x). Using tote dividend."
                    )
                    place_odds = runner.place_dividend

            # Dead heat: divide dividends by number of runners sharing position
            dh = dead_heat_divisors.get(runner.finish_position, 1)
            if dh > 1:
                win_odds = win_odds / dh if win_odds else win_odds
                place_odds = place_odds / dh if place_odds else place_odds

            if bet_type in ("win", "saver_win"):
                pick.hit = won
                if won and win_odds:
                    pick.pnl = round(win_odds * stake - stake, 2)
                else:
                    pick.pnl = round(-stake, 2)
            elif bet_type == "place":
                if num_places == 0:
                    # No place betting in ≤4 runner fields — void (refund)
                    pick.hit = False
                    pick.pnl = 0.0
                    logger.info(
                        f"Voiding Place bet for {pick.horse_name} R{pick.race_number} "
                        f"— no place betting in {field_size}-runner field"
                    )
                elif placed and place_odds:
                    pick.hit = True
                    pick.pnl = round(place_odds * stake - stake, 2)
                else:
                    pick.hit = False
                    pick.pnl = round(-stake, 2)
            elif bet_type == "each_way":
                half = stake / 2
                if num_places == 0:
                    # No place betting — EW becomes Win only, place half refunded
                    if won and win_odds:
                        pick.pnl = round(win_odds * half - half, 2)  # win half pays, place half refunded ($0)
                        pick.hit = True
                    else:
                        pick.pnl = round(-half, 2)  # only lose win half, place half refunded
                        pick.hit = False
                    logger.info(
                        f"EW bet for {pick.horse_name} R{pick.race_number} "
                        f"— place half voided in {field_size}-runner field"
                    )
                elif won and win_odds:
                    # Win half always pays at win odds; place half pays if place_odds available
                    win_return = win_odds * half
                    place_return = place_odds * half if place_odds else half  # refund place half if no place odds
                    pick.pnl = round(win_return + place_return - stake, 2)
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

            # Settle Punty's Pick shadow result if bet type differs
            if pick.is_puntys_pick and pick.pp_bet_type and pick.pp_bet_type != bet_type:
                pp_bt = pick.pp_bet_type.lower().replace(" ", "_")
                pp_odds_val = pick.pp_odds or place_odds  # PP odds, fallback to place
                if pp_bt in ("win", "saver_win"):
                    pick.pp_hit = won
                    pick.pp_pnl = round(win_odds * stake - stake, 2) if won and win_odds else round(-stake, 2)
                elif pp_bt == "place":
                    pick.pp_hit = placed
                    pick.pp_pnl = round(pp_odds_val * stake - stake, 2) if placed and pp_odds_val else round(-stake, 2)
                elif pp_bt == "each_way":
                    half = stake / 2
                    if won and win_odds:
                        pick.pp_hit = True
                        win_return = win_odds * half
                        place_return = pp_odds_val * half if pp_odds_val else half
                        pick.pp_pnl = round(win_return + place_return - stake, 2)
                    elif placed and pp_odds_val:
                        pick.pp_hit = True
                        pick.pp_pnl = round(pp_odds_val * half - stake, 2)
                    else:
                        pick.pp_hit = False
                        pick.pp_pnl = round(-stake, 2)

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
    big3_multi_picks = result.scalars().all()

    # Batch-load all big3 individual picks and their races for multi settlement
    if big3_multi_picks:
        content_ids = [mp.content_id for mp in big3_multi_picks if mp.content_id]
        if content_ids:
            b3_all_result = await db.execute(
                select(Pick).where(
                    Pick.content_id.in_(content_ids),
                    Pick.pick_type == "big3",
                )
            )
            all_big3_picks = b3_all_result.scalars().all()
            big3_by_content = {}
            for p in all_big3_picks:
                big3_by_content.setdefault(p.content_id, []).append(p)

            # Batch-load all races referenced by big3 picks
            b3_race_nums = {p.race_number for p in all_big3_picks if p.race_number}
            b3_race_ids = [f"{meeting_id}-r{rn}" for rn in b3_race_nums]
            if b3_race_ids:
                b3_races_result = await db.execute(
                    select(Race).where(Race.id.in_(b3_race_ids))
                )
                b3_race_map = {r.id: r for r in b3_races_result.scalars().all()}
            else:
                b3_race_map = {}
        else:
            big3_by_content = {}
            b3_race_map = {}
    else:
        big3_by_content = {}
        b3_race_map = {}

    for multi_pick in big3_multi_picks:
        big3_picks = big3_by_content.get(multi_pick.content_id, [])
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
                b3_race = b3_race_map.get(b3_race_id)
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
    # Build set of scratched saddlecloths for exotic adjustment
    scratched_saddlecloths = {
        r.saddlecloth for r in runners
        if r.scratched and r.saddlecloth
    }

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

            top_sc_int = [int(x) for x in top_saddlecloths if x is not None]

            # Check if legs format [[1], [5, 8], [8, 9]] vs flat [1, 5, 8, 9]
            is_legs_format = exotic_runners and isinstance(exotic_runners[0], list)

            if is_legs_format:
                # Legs format: [[leg1], [leg2], ...]
                # Check hit: each position winner must be in corresponding leg
                # TAB rule: scratched runners in a leg become free passes (any runner fills)
                legs = exotic_runners
                required_positions = {"trifecta": 3, "exacta": 2, "quinella": 2, "first": 4}
                req_pos = 3  # default for trifecta
                for k, v in required_positions.items():
                    if k in exotic_type:
                        req_pos = v
                        break

                # Expand legs: if all runners in a leg are scratched, it's a free pass
                adjusted_legs = []
                for leg in legs:
                    leg_ints = [int(x) for x in leg]
                    non_scratched = [x for x in leg_ints if x not in scratched_saddlecloths]
                    if non_scratched:
                        adjusted_legs.append(non_scratched)
                    else:
                        # All selections in this leg scratched — any runner fills
                        adjusted_legs.append(top_sc_int[:req_pos + 1])
                legs = adjusted_legs

                if len(legs) >= req_pos and len(top_sc_int) >= req_pos:
                    if "quinella" in exotic_type:
                        a, b = top_sc_int[0], top_sc_int[1]
                        hit = (
                            (a in legs[0] and b in legs[1]) or
                            (b in legs[0] and a in legs[1])
                        )
                    else:
                        # Positional: check each position winner is in corresponding leg
                        hit = all(top_sc_int[i] in legs[i] for i in range(req_pos))

                if hit:
                    div_key = "first4" if "first" in exotic_type else exotic_type.split()[0]
                    dividend = _find_dividend(exotic_divs, div_key)

                # Combos = product of leg sizes
                combos = 1
                for leg in legs:
                    combos *= len(leg)

            else:
                # Flat format: [1, 5, 8, 9] - boxed bet
                exotic_runners_int = [int(x) for x in exotic_runners if str(x).isdigit()]
                original_count = len(exotic_runners_int)
                # Remove scratched runners from our selections
                exotic_runners_int = [x for x in exotic_runners_int if x not in scratched_saddlecloths]
                is_standout = "standout" in exotic_type
                is_boxed = "box" in exotic_type

                # Void (refund) if scratchings leave too few runners to form the bet,
                # or if the standout runner itself was scratched
                required = {"trifecta": 3, "exacta": 2, "quinella": 2, "first": 4}
                min_needed = next((v for k, v in required.items() if k in exotic_type), 3)
                standout_scratched = is_standout and original_count > len(exotic_runners_int) and (
                    exotic_runners_int == [] or exotic_runners_int[0] != int(exotic_runners[0])
                )
                if len(exotic_runners_int) < min_needed or standout_scratched:
                    # Void: refund stake
                    pick.pnl = 0.0
                    pick.hit = False
                    pick.settled = True
                    pick.settled_at = now
                    settled_count += 1
                    logger.info(f"Voided exotic {pick.id}: scratching left {len(exotic_runners_int)}/{min_needed} runners")
                    continue

                if "trifecta" in exotic_type:
                    if len(top_sc_int) >= 3 and len(exotic_runners_int) >= 3:
                        if is_standout:
                            # Standout: first runner must win, others fill 2nd/3rd
                            standout = exotic_runners_int[0]
                            others = set(exotic_runners_int[1:])
                            hit = (top_sc_int[0] == standout and
                                   set(top_sc_int[1:3]).issubset(others))
                        elif is_boxed or len(exotic_runners_int) > 3:
                            hit = set(top_sc_int[:3]).issubset(set(exotic_runners_int))
                        else:
                            hit = list(top_sc_int[:3]) == list(exotic_runners_int[:3])
                    if hit:
                        dividend = _find_dividend(exotic_divs, "trifecta")
                elif "exacta" in exotic_type:
                    if len(top_sc_int) >= 2 and len(exotic_runners_int) >= 2:
                        if is_standout:
                            # Standout: first runner must win, any other must be 2nd
                            standout = exotic_runners_int[0]
                            others = set(exotic_runners_int[1:])
                            hit = (top_sc_int[0] == standout and
                                   top_sc_int[1] in others)
                        elif is_boxed or len(exotic_runners_int) > 2:
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
                        if is_standout:
                            standout = exotic_runners_int[0]
                            others = set(exotic_runners_int[1:])
                            hit = (top_sc_int[0] == standout and
                                   set(top_sc_int[1:4]).issubset(others))
                        elif is_boxed or len(exotic_runners_int) > 4:
                            hit = set(top_sc_int[:4]).issubset(set(exotic_runners_int))
                        else:
                            hit = list(top_sc_int[:4]) == list(exotic_runners_int[:4])
                    if hit:
                        dividend = _find_dividend(exotic_divs, "first4")

                # Calculate combos for flexi betting
                n = len(set(exotic_runners_int))
                combos = 1
                if is_standout:
                    # Standout: 1 fixed runner × permutations of others
                    others_n = n - 1
                    if "trifecta" in exotic_type:
                        combos = others_n * (others_n - 1) if others_n >= 2 else 1
                    elif "exacta" in exotic_type:
                        combos = others_n if others_n >= 1 else 1
                    elif "first" in exotic_type:
                        combos = others_n * (others_n - 1) * (others_n - 2) if others_n >= 3 else 1
                else:
                    # Box formula
                    if "trifecta" in exotic_type:
                        combos = n * (n - 1) * (n - 2) if n >= 3 else 1
                    elif "exacta" in exotic_type:
                        combos = n * (n - 1) if n >= 2 else 1
                    elif "quinella" in exotic_type:
                        combos = n * (n - 1) // 2 if n >= 2 else 1
                    elif "first" in exotic_type:
                        combos = n * (n - 1) * (n - 2) * (n - 3) if n >= 4 else 1

            # Stake is the total outlay for this exotic bet
            cost = stake

            if hit and dividend > 0:
                # Flexi formula: return = dividend × (stake / combos)
                flexi_pct = stake / combos if combos > 0 else stake
                return_amount = dividend * flexi_pct
                pick.pnl = round(return_amount - stake, 2)
            elif hit and dividend == 0:
                # Hit but no dividend available — TAB rules: refund
                pick.pnl = 0.0
                logger.warning(f"Exotic pick {pick.id} hit but dividend=0 — treating as refund")
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
    sequence_picks = result.scalars().all()

    # Batch-load all races and winners needed for sequence settlement
    if sequence_picks:
        # Collect all race numbers that any sequence could reference
        all_seq_race_nums = set()
        for pick in sequence_picks:
            if pick.sequence_legs and pick.sequence_start_race:
                try:
                    legs = json.loads(pick.sequence_legs)
                    for i in range(len(legs)):
                        all_seq_race_nums.add(pick.sequence_start_race + i)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Load all referenced races in one query
        seq_race_ids = [f"{meeting_id}-r{rn}" for rn in all_seq_race_nums]
        if seq_race_ids:
            seq_races_result = await db.execute(
                select(Race).where(Race.id.in_(seq_race_ids))
            )
            seq_race_map = {r.id: r for r in seq_races_result.scalars().all()}

            # Load all winners for those races in one query
            seq_winners_result = await db.execute(
                select(Runner).where(
                    Runner.race_id.in_(seq_race_ids),
                    Runner.finish_position == 1,
                )
            )
            seq_winner_map = {r.race_id: r for r in seq_winners_result.scalars().all()}
        else:
            seq_race_map = {}
            seq_winner_map = {}
    else:
        seq_race_map = {}
        seq_winner_map = {}

    for pick in sequence_picks:
        if not pick.sequence_legs or not pick.sequence_start_race:
            continue

        legs = json.loads(pick.sequence_legs)
        start = pick.sequence_start_race
        num_legs = len(legs)

        # Check if this race is part of this sequence
        if race_number < start or race_number >= start + num_legs:
            continue

        # Check if ALL legs have results (using batch-loaded data)
        all_resolved = True
        all_hit = True
        for leg_idx, leg_saddlecloths in enumerate(legs):
            leg_race_num = start + leg_idx
            leg_race_id = f"{meeting_id}-r{leg_race_num}"
            leg_race = seq_race_map.get(leg_race_id)
            if not leg_race or leg_race.results_status not in ("Paying", "Closed"):
                all_resolved = False
                break

            # Find winner of this leg
            winner = seq_winner_map.get(leg_race_id)
            if not winner:
                all_hit = False
            elif winner.saddlecloth not in leg_saddlecloths:
                # Check if all our selections in this leg were scratched (free pass)
                # Load scratched runners for this leg's race
                leg_runners_result = await db.execute(
                    select(Runner).where(Runner.race_id == leg_race_id)
                )
                leg_runners = leg_runners_result.scalars().all()
                leg_scratched = {r.saddlecloth for r in leg_runners if r.scratched and r.saddlecloth}
                non_scratched_selections = [s for s in leg_saddlecloths if s not in leg_scratched]
                if not non_scratched_selections:
                    # All our selections scratched — free pass, leg counts as hit
                    pass
                else:
                    all_hit = False

        if not all_resolved:
            continue

        pick.hit = all_hit

        # Look up sequence dividend from last leg's exotic_results
        seq_pnl = 0.0
        if all_hit:
            last_leg_race_id = f"{meeting_id}-r{start + num_legs - 1}"
            last_leg_race = seq_race_map.get(last_leg_race_id)
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
                    # Hit but no dividend found — TAB rules: refund
                    total_stake = pick.exotic_stake or 1.0
                    seq_pnl = 0.0
                    logger.warning(f"Sequence pick {pick.id} hit but dividend=0 — treating as refund")
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

        # Refresh strategy pattern insights (aggregated bet-type performance)
        try:
            from punty.memory.strategy import populate_pattern_insights
            await populate_pattern_insights(db)
        except Exception as e:
            logger.warning(f"Failed to refresh strategy insights: {e}")

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

    # For sequences, exotic_stake already stores total outlay (parser.py:423)
    seq_result = await db.execute(
        select(func.sum(Pick.exotic_stake))
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(
            Meeting.date == target_date,
            Pick.settled == True,
            Pick.pick_type == "sequence",
        )
    )
    sequence_total_stake = float(seq_result.scalar() or 0)

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
            # exotic_stake already stores total outlay
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
        # exotic_stake already stores total outlay for sequences/exotics (parser.py:423)
        if pick.pick_type in ("sequence", "exotic", "big3_multi"):
            staked = float(pick.exotic_stake or 0)
        else:
            staked = float(pick.bet_stake or 0)

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
                    opening = flucs[0].get("odds", 0)
                    current = flucs[-1].get("odds", 0)
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
