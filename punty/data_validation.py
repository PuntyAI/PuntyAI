"""Data validation — systematic quality checks across the entire pipeline.

Runs after scraping, settlement, and as a daily sweep. Logs issues to
the /issues page for manual review.

Usage:
    # After scraping a race:
    await validate_race_data(db, meeting_id, race_number)

    # After settling a race:
    await validate_settlement(db, meeting_id, race_number)

    # Daily sweep (all today's races):
    await validate_today(db)
"""

import json
import logging
import math
from datetime import date
from typing import Optional

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.issues import log_issue
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick
from punty.models.content import Content

logger = logging.getLogger(__name__)


async def validate_race_data(
    db: AsyncSession, meeting_id: str, race_number: int
) -> int:
    """Validate scraped runner data for a single race. Returns issue count."""
    issues = 0
    race_id = f"{meeting_id}-r{race_number}"

    race_result = await db.execute(select(Race).where(Race.id == race_id))
    race = race_result.scalar_one_or_none()
    if not race:
        return 0

    runner_result = await db.execute(
        select(Runner).where(Runner.race_id == race_id, Runner.scratched != True)
    )
    runners = runner_result.scalars().all()

    meeting_result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = meeting_result.scalar_one_or_none()
    venue = meeting.venue if meeting else meeting_id

    # ── Field size sanity ──
    if len(runners) < 2:
        await log_issue(db, "data", "warning",
            f"Only {len(runners)} active runners",
            meeting_id=meeting_id, race_number=race_number,
            link=f"/meets/{meeting_id}")
        issues += 1
    elif len(runners) > 24:
        await log_issue(db, "data", "warning",
            f"Unusually large field: {len(runners)} runners",
            meeting_id=meeting_id, race_number=race_number,
            link=f"/meets/{meeting_id}")
        issues += 1

    for runner in runners:
        prefix = f"{runner.horse_name or f'SC{runner.saddlecloth}'} ({venue} R{race_number})"

        # ── Missing critical fields ──
        missing = []
        if not runner.horse_name:
            missing.append("horse_name")
        if not runner.saddlecloth:
            missing.append("saddlecloth")
        if not runner.jockey:
            missing.append("jockey")
        if not runner.trainer:
            missing.append("trainer")
        if missing:
            await log_issue(db, "data", "warning",
                f"Missing fields: {', '.join(missing)} — {prefix}",
                meeting_id=meeting_id, race_number=race_number,
                link=f"/meets/{meeting_id}")
            issues += 1

        # ── Invalid odds ──
        if runner.current_odds is not None:
            if runner.current_odds <= 0:
                await log_issue(db, "data", "error",
                    f"Invalid odds ${runner.current_odds} — {prefix}",
                    meeting_id=meeting_id, race_number=race_number,
                    link=f"/meets/{meeting_id}")
                issues += 1
            if runner.place_odds and runner.current_odds > 0:
                if runner.place_odds > runner.current_odds:
                    await log_issue(db, "data", "warning",
                        f"Place odds ${runner.place_odds:.2f} > win odds ${runner.current_odds:.2f} — {prefix}",
                        meeting_id=meeting_id, race_number=race_number,
                        link=f"/meets/{meeting_id}")
                    issues += 1

        # ── KASH divergence (if available) ──
        if runner.kash_rated_price and runner.current_odds and runner.current_odds > 0:
            ratio = max(runner.kash_rated_price, runner.current_odds) / min(runner.kash_rated_price, runner.current_odds)
            if ratio > 5:
                await log_issue(db, "data", "warning",
                    f"KASH ${runner.kash_rated_price:.2f} vs market ${runner.current_odds:.2f} ({ratio:.1f}x) — {prefix}",
                    meeting_id=meeting_id, race_number=race_number,
                    link=f"/meets/{meeting_id}")
                issues += 1

    # ── Duplicate saddlecloths ──
    saddlecloths = [r.saddlecloth for r in runners if r.saddlecloth]
    if len(saddlecloths) != len(set(saddlecloths)):
        dupes = [s for s in saddlecloths if saddlecloths.count(s) > 1]
        await log_issue(db, "data", "error",
            f"Duplicate saddlecloths: {set(dupes)}",
            meeting_id=meeting_id, race_number=race_number,
            link=f"/meets/{meeting_id}")
        issues += 1

    if issues:
        await db.commit()
    return issues


async def validate_probability(
    db: AsyncSession, meeting_id: str, race_number: int
) -> int:
    """Validate probability calculations for a race. Returns issue count."""
    issues = 0
    race_id = f"{meeting_id}-r{race_number}"

    meeting_result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = meeting_result.scalar_one_or_none()
    venue = meeting.venue if meeting else meeting_id

    # Get picks for this race
    pick_result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type == "selection",
        )
    )
    picks = pick_result.scalars().all()
    if not picks:
        return 0

    # ── Win probs should roughly sum to 1.0 ──
    wp_sum = sum(p.win_probability or 0 for p in picks if p.tip_rank and p.tip_rank <= 4)
    # Only check if we have at least the top 3
    top3 = [p for p in picks if p.tip_rank and p.tip_rank <= 3]
    if len(top3) >= 3:
        # Top 3 WP should be > 0.3 combined (they represent the favourites)
        top3_wp = sum(p.win_probability or 0 for p in top3)
        if top3_wp < 0.3:
            await log_issue(db, "probability", "warning",
                f"Top-3 WP sum only {top3_wp:.0%} — suspiciously low ({venue} R{race_number})",
                meeting_id=meeting_id, race_number=race_number,
                link=f"/meets/{meeting_id}")
            issues += 1

    for pick in picks:
        wp = pick.win_probability or 0
        pp = pick.place_probability or 0
        prefix = f"{pick.horse_name or f'SC{pick.saddlecloth}'} ({venue} R{race_number})"

        # ── Place prob must be >= win prob ──
        if pp < wp and pp > 0 and wp > 0:
            await log_issue(db, "probability", "error",
                f"PP {pp:.0%} < WP {wp:.0%} — impossible ({prefix})",
                meeting_id=meeting_id, race_number=race_number,
                pick_id=pick.id, link=f"/meets/{meeting_id}")
            issues += 1

        # ── Suspiciously high probabilities ──
        if wp > 0.80:
            await log_issue(db, "probability", "warning",
                f"WP {wp:.0%} suspiciously high ({prefix})",
                meeting_id=meeting_id, race_number=race_number,
                pick_id=pick.id, link=f"/meets/{meeting_id}")
            issues += 1

    # ── Same horse at multiple ranks ──
    horses_by_name = {}
    for pick in picks:
        name = (pick.horse_name or "").lower().strip()
        if name and name in horses_by_name:
            await log_issue(db, "data", "error",
                f"Duplicate pick: {pick.horse_name} at rank {pick.tip_rank} "
                f"and {horses_by_name[name]} ({venue} R{race_number})",
                meeting_id=meeting_id, race_number=race_number,
                link=f"/meets/{meeting_id}")
            issues += 1
        elif name:
            horses_by_name[name] = pick.tip_rank

    if issues:
        await db.commit()
    return issues


async def validate_settlement(
    db: AsyncSession, meeting_id: str, race_number: int
) -> int:
    """Validate settlement math for a settled race. Returns issue count."""
    issues = 0
    race_id = f"{meeting_id}-r{race_number}"

    meeting_result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = meeting_result.scalar_one_or_none()
    venue = meeting.venue if meeting else meeting_id

    # Get settled picks with runner data
    pick_result = await db.execute(
        select(Pick, Runner).outerjoin(
            Runner, and_(
                Runner.race_id == race_id,
                Runner.saddlecloth == Pick.saddlecloth,
            )
        ).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.settled == True,
        )
    )
    rows = pick_result.all()

    for pick, runner in rows:
        fp = runner.finish_position if runner else None
        prefix = f"{pick.horse_name or f'SC{pick.saddlecloth}'} ({venue} R{race_number})"

        # ── Settled but no finish position ──
        if fp is None and pick.pick_type == "selection":
            await log_issue(db, "settlement", "warning",
                f"Settled but no finish position — {prefix}",
                meeting_id=meeting_id, race_number=race_number,
                pick_id=pick.id, link=f"/meets/{meeting_id}")
            issues += 1

        if pick.pick_type == "selection" and pick.bet_stake and pick.bet_stake > 0:
            stake = pick.bet_stake
            pnl = pick.pnl or 0
            bt = (pick.bet_type or "").lower()

            # ── Win bet math ──
            if bt in ("win", "saver_win") and pick.hit and runner:
                win_div = runner.win_dividend
                if win_div:
                    expected = round(win_div * stake - stake, 2)
                    if abs(pnl - expected) > 1.0:
                        await log_issue(db, "settlement", "error",
                            f"Win PnL mismatch: got ${pnl:.2f}, expected ${expected:.2f} "
                            f"(div=${win_div} × stake=${stake}) — {prefix}",
                            meeting_id=meeting_id, race_number=race_number,
                            pick_id=pick.id, amount=abs(pnl - expected),
                            link=f"/meets/{meeting_id}")
                        issues += 1
                elif pick.hit:
                    await log_issue(db, "settlement", "error",
                        f"Win bet hit but no win_dividend on runner — {prefix}",
                        meeting_id=meeting_id, race_number=race_number,
                        pick_id=pick.id, amount=stake,
                        link=f"/meets/{meeting_id}")
                    issues += 1

            # ── Place bet math ──
            if bt == "place" and pick.hit and runner:
                place_div = runner.place_dividend
                if place_div:
                    expected = round(place_div * stake - stake, 2)
                    if abs(pnl - expected) > 1.0:
                        await log_issue(db, "settlement", "error",
                            f"Place PnL mismatch: got ${pnl:.2f}, expected ${expected:.2f} "
                            f"(div=${place_div} × stake=${stake}) — {prefix}",
                            meeting_id=meeting_id, race_number=race_number,
                            pick_id=pick.id, amount=abs(pnl - expected),
                            link=f"/meets/{meeting_id}")
                        issues += 1

            # ── Hit consistency ──
            if bt in ("win", "saver_win"):
                if pick.hit and fp != 1:
                    await log_issue(db, "settlement", "error",
                        f"Win bet marked hit but FP={fp} — {prefix}",
                        meeting_id=meeting_id, race_number=race_number,
                        pick_id=pick.id, link=f"/meets/{meeting_id}")
                    issues += 1
                if not pick.hit and fp == 1:
                    await log_issue(db, "settlement", "error",
                        f"Win bet NOT hit but FP=1 — {prefix}",
                        meeting_id=meeting_id, race_number=race_number,
                        pick_id=pick.id, amount=stake,
                        link=f"/meets/{meeting_id}")
                    issues += 1

            if bt == "place":
                if pick.hit and fp and fp > 3:
                    await log_issue(db, "settlement", "error",
                        f"Place bet marked hit but FP={fp} — {prefix}",
                        meeting_id=meeting_id, race_number=race_number,
                        pick_id=pick.id, link=f"/meets/{meeting_id}")
                    issues += 1
                if not pick.hit and fp and fp <= 3:
                    await log_issue(db, "settlement", "error",
                        f"Place bet NOT hit but FP={fp} — {prefix}",
                        meeting_id=meeting_id, race_number=race_number,
                        pick_id=pick.id, amount=stake,
                        link=f"/meets/{meeting_id}")
                    issues += 1

            # ── PnL impossibility ──
            if pick.hit and pnl < -stake:
                await log_issue(db, "settlement", "error",
                    f"Hit=True but PnL ${pnl:.2f} < -stake ${stake:.2f} — {prefix}",
                    meeting_id=meeting_id, race_number=race_number,
                    pick_id=pick.id, link=f"/meets/{meeting_id}")
                issues += 1

        # ── Exotic settlement checks ──
        if pick.pick_type == "exotic":
            stake = pick.exotic_stake or 0
            pnl = pick.pnl or 0

            if pick.hit and pnl == 0 and stake > 0:
                # Already logged by settlement code, but catch any missed
                await log_issue(db, "settlement", "error",
                    f"Exotic hit but $0 PnL — {pick.exotic_type} {prefix}",
                    meeting_id=meeting_id, race_number=race_number,
                    pick_id=pick.id, amount=stake,
                    link=f"/meets/{meeting_id}")
                issues += 1

            if pick.hit and pnl > 0:
                # Sanity: dividend shouldn't be >1000x stake
                if pnl > stake * 1000:
                    await log_issue(db, "settlement", "warning",
                        f"Exotic PnL ${pnl:.2f} is >{1000}x stake ${stake:.2f} — verify dividend ({prefix})",
                        meeting_id=meeting_id, race_number=race_number,
                        pick_id=pick.id, link=f"/meets/{meeting_id}")
                    issues += 1

    # ── Check race has exotic_results if it has settled exotic picks ──
    race_result = await db.execute(select(Race).where(Race.id == race_id))
    race = race_result.scalar_one_or_none()
    exotic_picks = [p for p, r in rows if p.pick_type == "exotic"]
    if exotic_picks and race and not race.exotic_results:
        await log_issue(db, "settlement", "warning",
            f"Race has {len(exotic_picks)} exotic picks but no exotic_results — dividends missing ({venue} R{race_number})",
            meeting_id=meeting_id, race_number=race_number,
            link=f"/meets/{meeting_id}")
        issues += 1

    if issues:
        await db.commit()
    return issues


async def validate_today(db: AsyncSession, target_date: date | None = None) -> dict:
    """Run all validations for today's (or specified date's) races.

    Returns summary dict with issue counts.
    """
    if target_date is None:
        target_date = melb_today()

    result = await db.execute(
        select(Meeting).where(Meeting.date == target_date, Meeting.selected == True)
    )
    meetings = result.scalars().all()

    summary = {"date": str(target_date), "meetings": 0, "races": 0, "issues": 0}

    for meeting in meetings:
        summary["meetings"] += 1
        race_result = await db.execute(
            select(Race.race_number).where(Race.meeting_id == meeting.id)
        )
        race_numbers = [r[0] for r in race_result.all()]

        for rn in race_numbers:
            summary["races"] += 1

            # Data validation
            data_issues = await validate_race_data(db, meeting.id, rn)
            summary["issues"] += data_issues

            # Probability validation (only if picks exist)
            prob_issues = await validate_probability(db, meeting.id, rn)
            summary["issues"] += prob_issues

            # Settlement validation (only if race is settled)
            race = await db.execute(
                select(Race.results_status).where(
                    Race.meeting_id == meeting.id, Race.race_number == rn
                )
            )
            status = race.scalar_one_or_none()
            if status in ("Paying", "Closed"):
                settle_issues = await validate_settlement(db, meeting.id, rn)
                summary["issues"] += settle_issues

    # ── Integration health checks ──
    integration_issues = await validate_integrations(db, target_date)
    summary["issues"] += integration_issues

    logger.info(
        f"Validation complete for {target_date}: "
        f"{summary['meetings']} meetings, {summary['races']} races, "
        f"{summary['issues']} issues found"
    )
    return summary


async def validate_integrations(db: AsyncSession, target_date: date | None = None) -> int:
    """Check health of all external integrations. Returns issue count."""
    if target_date is None:
        target_date = melb_today()
    issues = 0
    total_runners = 0

    # ── KASH ratings: should have data for today if it's after 10am ──
    from datetime import datetime
    now = datetime.now()
    if now.hour >= 11:  # Give KASH until 11am to load
        kash_result = await db.execute(
            select(func.count(Runner.id)).where(
                Runner.race_id.like(f"%-{target_date}-%"),
                Runner.kash_rated_price.isnot(None),
            )
        )
        kash_count = kash_result.scalar() or 0
        runner_result = await db.execute(
            select(func.count(Runner.id)).where(
                Runner.race_id.like(f"%-{target_date}-%"),
                Runner.scratched != True,
            )
        )
        total_runners = runner_result.scalar() or 0

        if total_runners > 0 and kash_count == 0:
            await log_issue(db, "integration", "error",
                f"KASH ratings: 0/{total_runners} runners have data — fetch may have failed",
                description="Check if 10am KASH fetch ran. Try manual: "
                            "python3 -c 'import asyncio; from punty.scrapers.kash_ratings import apply_kash_ratings; asyncio.run(apply_kash_ratings())'",
                link="/issues")
            issues += 1
        elif total_runners > 0 and kash_count < total_runners * 0.5:
            await log_issue(db, "integration", "warning",
                f"KASH ratings: only {kash_count}/{total_runners} runners matched ({kash_count/total_runners:.0%})",
                link="/issues")
            issues += 1

    # ── Betfair balance: DB vs API drift ──
    try:
        from punty.betting.queue import get_balance
        from punty.betting.betfair_client import get_account_balance
        db_balance = await get_balance(db)
        api_balance = await get_account_balance(db)
        if api_balance is not None and abs(db_balance - api_balance) > 5.0:
            await log_issue(db, "betfair", "error",
                f"Balance drift: DB=${db_balance:.2f} vs API=${api_balance:.2f} (${abs(db_balance - api_balance):.2f} gap)",
                description="Hourly sync may have failed. Balance affects Kelly stake sizing.",
                amount=abs(db_balance - api_balance),
                link="/betfair")
            issues += 1
    except Exception as e:
        await log_issue(db, "betfair", "warning",
            f"Betfair API unreachable: {str(e)[:100]}",
            link="/betfair")
        issues += 1

    # ── Exotic dividends: races with results but no exotic_results ──
    race_result = await db.execute(
        select(Race).join(Meeting, Meeting.id == Race.meeting_id).where(
            Meeting.date == target_date,
            Meeting.selected == True,
            Race.results_status.in_(["Paying", "Closed"]),
            or_(Race.exotic_results.is_(None), Race.exotic_results == "", Race.exotic_results == "{}"),
        )
    )
    missing_exotic_races = race_result.scalars().all()
    if missing_exotic_races:
        venues = set()
        for race in missing_exotic_races:
            venues.add(race.meeting_id.split("-2026")[0])
        await log_issue(db, "integration", "warning",
            f"{len(missing_exotic_races)} settled races missing exotic dividends",
            description=f"Venues: {', '.join(sorted(venues))}. "
                        f"PointsBet backfill may have failed. Check monitor logs.",
            link="/issues")
        issues += 1

    # ── Odds freshness: runners with no odds update ──
    stale_result = await db.execute(
        select(func.count(Runner.id)).where(
            Runner.race_id.like(f"%-{target_date}-%"),
            Runner.scratched != True,
            or_(Runner.current_odds.is_(None), Runner.current_odds <= 0),
        )
    )
    stale_count = stale_result.scalar() or 0
    if total_runners > 0 and stale_count > total_runners * 0.3:
        await log_issue(db, "integration", "warning",
            f"{stale_count}/{total_runners} runners have no odds ({stale_count/total_runners:.0%})",
            description="Odds scraping may have failed. Check PointsBet/Betfair/racing.com scrapers.",
            link="/issues")
        issues += 1

    # ── Unsettled races that should be settled ──
    # Races started >2 hours ago but still not settled
    from punty.config import melb_now_naive
    cutoff = melb_now_naive()
    from datetime import timedelta
    two_hours_ago = cutoff - timedelta(hours=2)

    stale_race_result = await db.execute(
        select(Race, Meeting).join(Meeting, Meeting.id == Race.meeting_id).where(
            Meeting.date == target_date,
            Meeting.selected == True,
            Race.start_time.isnot(None),
            Race.start_time < two_hours_ago,
            Race.results_status.notin_(["Paying", "Closed"]),
        )
    )
    stale_races = stale_race_result.all()
    if stale_races:
        for race, meeting in stale_races:
            await log_issue(db, "settlement", "warning",
                f"Race started >2h ago but not settled — {meeting.venue} R{race.race_number}",
                description=f"Start time: {race.start_time}, Status: {race.results_status}",
                meeting_id=meeting.id, race_number=race.race_number,
                link=f"/meets/{meeting.id}")
            issues += 1

    if issues:
        await db.commit()
    return issues
