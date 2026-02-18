"""E2E dry run: test the full pipeline on today's settled meetings.

For each meeting:
1. Verify data integrity (runners, odds, speed maps, form)
2. Re-run probability engine
3. Re-run pre-selections (bet types, stakes, exotics)
4. Re-run parser on approved content
5. Verify settlement math against actual results
6. Report any discrepancies
"""
import asyncio
import json
import sys
from collections import defaultdict
from sqlalchemy import select, func
from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner
from punty.models.content import Content
from punty.models.pick import Pick
from punty.config import melb_now

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results = {"pass": 0, "warn": 0, "fail": 0, "details": []}


def check(name, status, detail=""):
    results[status.lower()] += 1
    icon = {"PASS": "+", "WARN": "~", "FAIL": "!"}[status]
    msg = f"  [{icon}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results["details"].append({"name": name, "status": status, "detail": detail})


async def test_meeting(db, meeting, races, runners_by_race):
    venue = meeting.venue
    print(f"\n{'='*60}")
    print(f"MEETING: {venue} ({meeting.id})")
    print(f"{'='*60}")

    # --- 1. DATA INTEGRITY ---
    print("\n  --- Data Integrity ---")

    race_count = len(races)
    check("Race count", PASS if race_count >= 4 else WARN, f"{race_count} races")

    total_runners = 0
    active_runners = 0
    with_odds = 0
    with_speedmap = 0
    with_form = 0
    scratched = 0

    for race in races:
        runners = runners_by_race.get(race.id, [])
        for r in runners:
            total_runners += 1
            if r.scratched:
                scratched += 1
                continue
            active_runners += 1
            if r.current_odds and r.current_odds > 1.0:
                with_odds += 1
            if r.speed_map_position:
                with_speedmap += 1
            if r.form_history:
                with_form += 1

    odds_pct = (with_odds / active_runners * 100) if active_runners else 0
    sm_pct = (with_speedmap / active_runners * 100) if active_runners else 0
    form_pct = (with_form / active_runners * 100) if active_runners else 0

    check("Active runners", PASS if active_runners > 20 else WARN, f"{active_runners} active, {scratched} scratched")
    check("Odds coverage", PASS if odds_pct > 80 else (WARN if odds_pct > 50 else FAIL), f"{with_odds}/{active_runners} ({odds_pct:.0f}%)")
    check("Speed map coverage", PASS if sm_pct > 50 else (WARN if sm_pct > 30 else FAIL), f"{with_speedmap}/{active_runners} ({sm_pct:.0f}%)")
    check("Form history coverage", PASS if form_pct > 60 else WARN, f"{with_form}/{active_runners} ({form_pct:.0f}%)")

    # Check all races have results
    all_paying = all(r.results_status in ("Paying", "Closed", "Final") for r in races)
    check("All races paying", PASS if all_paying else FAIL, f"{sum(1 for r in races if r.results_status in ('Paying','Closed','Final'))}/{race_count}")

    # Check winners exist
    winner_count = 0
    for race in races:
        for r in runners_by_race.get(race.id, []):
            if r.finish_position == 1:
                winner_count += 1
    check("Winners recorded", PASS if winner_count == race_count else FAIL, f"{winner_count}/{race_count}")

    # Check dividends
    missing_win_div = 0
    missing_place_div = 0
    for race in races:
        for r in runners_by_race.get(race.id, []):
            if r.finish_position == 1 and not r.win_dividend:
                missing_win_div += 1
            if r.finish_position and 1 <= r.finish_position <= 3 and not r.place_dividend:
                missing_place_div += 1
    check("Win dividends", PASS if missing_win_div == 0 else WARN, f"{missing_win_div} winners missing win_dividend")
    check("Place dividends", PASS if missing_place_div == 0 else WARN, f"{missing_place_div} placed runners missing place_dividend")

    # --- 2. PROBABILITY ENGINE ---
    print("\n  --- Probability Engine ---")
    try:
        from punty.probability import calculate_race_probabilities
        prob_errors = 0
        prob_total = 0
        for race in races:
            runners = [r for r in runners_by_race.get(race.id, []) if not r.scratched]
            if not runners:
                continue
            try:
                probs = calculate_race_probabilities(
                    runners=runners,
                    race=race,
                    meeting=meeting,
                )
                prob_total += 1
                if not probs or len(probs) == 0:
                    prob_errors += 1
                else:
                    # Verify probabilities sum to ~1
                    total_prob = sum(p.win_probability for p in probs.values())
                    if abs(total_prob - 1.0) > 0.05:
                        check("Prob sum", WARN, f"R{race.race_number} sum={total_prob:.3f}")
            except Exception as e:
                prob_errors += 1
                check("Prob compute", FAIL, f"R{race.race_number}: {e}")

        check("Probability engine", PASS if prob_errors == 0 else FAIL, f"{prob_total - prob_errors}/{prob_total} races OK")
    except ImportError as e:
        check("Probability engine", WARN, f"Import error: {e}")
    except Exception as e:
        check("Probability engine", FAIL, f"Unexpected: {e}")

    # --- 3. PRE-SELECTIONS (bet types) ---
    print("\n  --- Pre-Selections ---")
    try:
        from punty.context.builder import ContextBuilder
        from punty.context.pre_selections import calculate_pre_selections
        builder = ContextBuilder(db)
        # Build full meeting context (same path as production)
        meeting_ctx = await builder.build_meeting_context(meeting.id)
        if not meeting_ctx or "races" not in meeting_ctx:
            check("Pre-selections", WARN, "No meeting context built")
        else:
            ps_pass = 0
            ps_fail = 0
            for race_ctx in meeting_ctx["races"][:3]:  # Test first 3 races
                try:
                    ps_result = calculate_pre_selections(race_ctx)
                    if ps_result and ps_result.picks:
                        ps_pass += 1
                        # Check bet types are valid
                        valid_bt = {"Win", "Place", "Each Way", "Saver Win", "win", "place", "each_way", "saver_win"}
                        for sel in ps_result.picks:
                            bt = sel.bet_type
                            if bt not in valid_bt:
                                check("Bet type valid", FAIL, f"R{race_ctx.get('race_number')}: invalid '{bt}'")
                        # Check stakes sum
                        total_stake = sum(s.stake for s in ps_result.picks)
                        if abs(total_stake - 20.0) > 1.0:
                            check("Stake sum", WARN, f"R{race_ctx.get('race_number')}: ${total_stake:.2f} (should be $20)")
                    else:
                        ps_fail += 1
                except Exception as e:
                    ps_fail += 1
                    check("Pre-sel race", FAIL, f"R{race_ctx.get('race_number')}: {e}")
            check("Pre-selections", PASS if ps_fail == 0 else WARN, f"{ps_pass}/{ps_pass+ps_fail} races OK")
    except Exception as e:
        check("Pre-selections", FAIL, f"{e}")

    # --- 4. PARSER ---
    print("\n  --- Parser ---")
    try:
        q = await db.execute(
            select(Content).where(
                Content.meeting_id == meeting.id,
                Content.content_type == "early_mail",
                Content.status.in_(["approved", "sent"]),
            ).order_by(Content.created_at.desc())
        )
        content = q.scalars().first()
        if content and content.raw_content:
            from punty.results.parser import parse_early_mail
            parsed = parse_early_mail(content.raw_content, str(content.id), meeting.id)
            parsed_sels = [p for p in parsed if p.get("pick_type") == "selection"]
            parsed_exotics = [p for p in parsed if p.get("pick_type") == "exotic"]
            parsed_seqs = [p for p in parsed if p.get("pick_type") == "sequence"]
            parsed_big3 = [p for p in parsed if p.get("pick_type") in ("big3", "big3_multi")]

            check("Parser selections", PASS if len(parsed_sels) >= race_count * 2 else WARN,
                  f"{len(parsed_sels)} selections parsed")
            check("Parser exotics", PASS if len(parsed_exotics) >= 1 else WARN,
                  f"{len(parsed_exotics)} exotics parsed")
            check("Parser sequences", PASS if len(parsed_seqs) >= 1 else WARN,
                  f"{len(parsed_seqs)} sequences parsed")
            check("Parser big3", PASS if len(parsed_big3) >= 3 else WARN,
                  f"{len(parsed_big3)} big3 entries parsed")

            # Cross-check: do parsed saddlecloths match valid runners?
            invalid_sc = 0
            for sel in parsed_sels:
                rn = sel.get("race_number")
                sc = sel.get("saddlecloth")
                if rn and sc:
                    race_id = f"{meeting.id}-r{rn}"
                    valid = any(
                        r.saddlecloth == sc and not r.scratched
                        for r in runners_by_race.get(race_id, [])
                    )
                    if not valid:
                        invalid_sc += 1
            check("Saddlecloth validity", PASS if invalid_sc == 0 else WARN,
                  f"{invalid_sc} parsed picks reference invalid/scratched runners")
        else:
            check("Parser", WARN, "No approved early mail content found")
    except Exception as e:
        check("Parser", FAIL, f"{e}")

    # --- 5. SETTLEMENT MATH ---
    print("\n  --- Settlement Math ---")
    try:
        q = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting.id,
                Pick.settled == True,
                Pick.pick_type == "selection",
            )
        )
        picks = q.scalars().all()

        math_errors = 0
        for pick in picks:
            rn = pick.race_number
            sc = pick.saddlecloth
            race_id = f"{meeting.id}-r{rn}"

            # Find the runner
            runner = None
            for r in runners_by_race.get(race_id, []):
                if r.saddlecloth == sc:
                    runner = r
                    break

            if not runner:
                continue

            expected_pnl = None
            stake = float(pick.bet_stake or 0)
            if stake <= 0:
                continue

            bt = (pick.bet_type or "").lower().replace(" ", "_")
            fp = runner.finish_position

            if bt in ("win", "saver_win"):
                if fp == 1 and runner.win_dividend:
                    expected_pnl = round(runner.win_dividend * stake - stake, 2)
                elif fp and fp > 0:
                    expected_pnl = -stake
            elif bt == "place":
                place_div = pick.place_odds_at_tip or runner.place_dividend
                if fp and 1 <= fp <= 3 and place_div:
                    expected_pnl = round(place_div * stake - stake, 2)
                elif fp and fp > 3:
                    expected_pnl = -stake
            elif bt == "each_way":
                half = stake / 2
                place_div = pick.place_odds_at_tip or runner.place_dividend
                if fp == 1 and runner.win_dividend and place_div:
                    expected_pnl = round((runner.win_dividend * half - half) + (place_div * half - half), 2)
                elif fp and 2 <= fp <= 3 and place_div:
                    expected_pnl = round(-half + (place_div * half - half), 2)
                elif fp and fp > 3:
                    expected_pnl = -stake

            if expected_pnl is not None:
                actual_pnl = float(pick.pnl or 0)
                diff = abs(actual_pnl - expected_pnl)
                if diff > 0.05:
                    math_errors += 1
                    if diff > 1.0:
                        check("Settlement math", WARN,
                              f"R{rn} {pick.horse_name} ({bt}): expected ${expected_pnl:+.2f}, got ${actual_pnl:+.2f} (diff ${diff:.2f})")

        check("Settlement math", PASS if math_errors == 0 else WARN,
              f"{len(picks)} picks checked, {math_errors} discrepancies")

        # Total P&L
        total_pnl = sum(float(p.pnl or 0) for p in picks)
        total_staked = sum(float(p.bet_stake or 0) for p in picks)
        roi = (total_pnl / total_staked * 100) if total_staked else 0
        check("Selection P&L", PASS, f"${total_pnl:+.2f} on ${total_staked:.2f} staked ({roi:+.1f}% ROI)")

        # Exotic P&L
        q2 = await db.execute(
            select(Pick).where(
                Pick.meeting_id == meeting.id,
                Pick.settled == True,
                Pick.pick_type.in_(["exotic", "sequence"]),
            )
        )
        exotic_picks = q2.scalars().all()
        exotic_pnl = sum(float(p.pnl or 0) for p in exotic_picks)
        exotic_staked = sum(float(p.exotic_stake or p.bet_stake or 0) for p in exotic_picks)
        exotic_hits = sum(1 for p in exotic_picks if p.hit)
        check("Exotic/Seq P&L", PASS,
              f"${exotic_pnl:+.2f} on ${exotic_staked:.2f} ({exotic_hits}/{len(exotic_picks)} hits)")

    except Exception as e:
        check("Settlement math", FAIL, f"{e}")

    # --- 6. CONTEXT BUILDER ---
    print("\n  --- Context Builder ---")
    try:
        from punty.context.builder import ContextBuilder
        builder = ContextBuilder(db)
        ctx = await builder.build_meeting_context(meeting.id)
        if ctx and "races" in ctx:
            race_count_ctx = len(ctx["races"])
            keys = list(ctx.keys())
            runner_count_ctx = sum(len(r.get("runners", [])) for r in ctx["races"])
            check("Context builder", PASS, f"{race_count_ctx} races, {runner_count_ctx} runners, keys: {keys}")
            # Verify each race has probabilities
            probs_ok = sum(1 for r in ctx["races"] if r.get("probabilities"))
            check("Context probs", PASS if probs_ok == race_count_ctx else WARN,
                  f"{probs_ok}/{race_count_ctx} races have probabilities")
        else:
            check("Context builder", WARN, "Returned None/empty")
    except Exception as e:
        check("Context builder", FAIL, f"{e}")


async def main():
    print(f"E2E Dry Run — {melb_now()}")
    print("Testing full pipeline on today's settled meetings")

    async with async_session() as db:
        q = await db.execute(
            select(Meeting).where(
                Meeting.date == "2026-02-18",
                Meeting.selected == True,
            )
        )
        meetings = q.scalars().all()

        for meeting in meetings:
            # Load races
            rq = await db.execute(
                select(Race).where(Race.meeting_id == meeting.id).order_by(Race.race_number)
            )
            races = rq.scalars().all()
            if not races:
                print(f"\n  SKIP {meeting.venue}: no races")
                continue

            # Load runners per race
            runners_by_race = {}
            for race in races:
                rrq = await db.execute(select(Runner).where(Runner.race_id == race.id))
                runners_by_race[race.id] = rrq.scalars().all()

            await test_meeting(db, meeting, races, runners_by_race)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  PASS: {results['pass']}")
    print(f"  WARN: {results['warn']}")
    print(f"  FAIL: {results['fail']}")

    fails = [d for d in results["details"] if d["status"] == "FAIL"]
    if fails:
        print(f"\n  FAILURES:")
        for f in fails:
            print(f"    ! {f['name']}: {f['detail']}")

    warns = [d for d in results["details"] if d["status"] == "WARN"]
    if warns:
        print(f"\n  WARNINGS:")
        for w in warns:
            print(f"    ~ {w['name']}: {w['detail']}")

    if results["fail"] > 0:
        print(f"\n  VERDICT: ISSUES FOUND — {results['fail']} failures need attention")
        sys.exit(1)
    else:
        print(f"\n  VERDICT: ALL CLEAR — pipeline is healthy")


asyncio.run(main())
