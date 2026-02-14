"""Compare OLD vs NEW probability model across ALL today's meetings.

Shows pick differences, DL pattern impact, accuracy metrics, and P&L simulation.
Runs both weight sets against every race with results and compares.

Usage (on server):
    cd /opt/puntyai && source venv/bin/activate
    python scripts/compare_all_today.py
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# V4 weights (calibrated but no sharpening, movement/pace still active)
OLD_WEIGHTS = {
    "market": 0.38, "form": 0.30, "jockey_trainer": 0.065,
    "deep_learning": 0.07, "weight_carried": 0.035, "horse_profile": 0.03,
    "movement": 0.03, "class_fitness": 0.03, "barrier": 0.02, "pace": 0.02,
}


async def main():
    from punty.models.database import async_session, init_db
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from punty.models.meeting import Meeting, Race, Runner
    from punty.models.pick import Pick
    from punty.probability import (
        calculate_race_probabilities, load_dl_patterns_for_probability,
        _get_calibrated_weights, DEFAULT_WEIGHTS,
    )

    await init_db()
    async with async_session() as db:
        # Load DL patterns
        dl_patterns = await load_dl_patterns_for_probability(db)

        # Find ALL today's meetings
        result = await db.execute(
            select(Meeting)
            .options(selectinload(Meeting.races).selectinload(Race.runners))
            .where(Meeting.date == "2026-02-14")
        )
        meetings = result.scalars().all()

        if not meetings:
            print("No meetings found for 2026-02-14")
            return

        print(f"\n{'='*90}")
        print(f"FULL DRY RUN COMPARISON — ALL MEETINGS 2026-02-14")
        print(f"{'='*90}")
        print(f"Found {len(meetings)} meetings today")

        # Get new weights
        new_w = DEFAULT_WEIGHTS
        print(f"\nOLD weights: mkt={OLD_WEIGHTS['market']*100:.0f}% form={OLD_WEIGHTS['form']*100:.0f}% "
              f"dl={OLD_WEIGHTS['deep_learning']*100:.0f}% mvmt={OLD_WEIGHTS['movement']*100:.0f}% "
              f"pace={OLD_WEIGHTS['pace']*100:.0f}%")
        print(f"NEW weights: mkt={new_w['market']*100:.0f}% form={new_w['form']*100:.0f}% "
              f"dl={new_w['deep_learning']*100:.0f}% mvmt={new_w['movement']*100:.0f}% "
              f"pace={new_w['pace']*100:.0f}%")
        print(f"NEW features: sharpening=1.45, DL patterns={len(dl_patterns)}, dead factors zeroed")

        # Load actual picks
        all_picks = {}
        for meeting in meetings:
            pick_result = await db.execute(
                select(Pick).where(Pick.meeting_id == meeting.id)
            )
            picks = pick_result.scalars().all()
            for p in picks:
                if p.race_number and p.pick_type == "selection":
                    key = (meeting.id, p.race_number)
                    all_picks.setdefault(key, []).append(p)

        # Aggregate stats
        total_races = 0
        total_with_results = 0
        old_stats = {"wins": 0, "places": 0, "pnl": 0.0, "bets": 0}
        new_stats = {"wins": 0, "places": 0, "pnl": 0.0, "bets": 0}
        no_dl_stats = {"wins": 0, "places": 0, "pnl": 0.0, "bets": 0}
        actual_stats = {"wins": 0, "places": 0, "pnl": 0.0, "bets": 0}

        # Track pick agreement/disagreement
        agree_count = 0
        disagree_count = 0
        old_right_new_wrong = 0
        new_right_old_wrong = 0
        dl_changed_pick = 0
        dl_helped = 0
        dl_hurt = 0

        # Calibration buckets
        cal_buckets = defaultdict(lambda: {"count": 0, "wins": 0})

        meetings_sorted = sorted(meetings, key=lambda m: m.venue or "")

        for meeting in meetings_sorted:
            races = sorted(meeting.races, key=lambda r: r.race_number)
            settled_races = [r for r in races if any(
                rn.finish_position for rn in r.runners if not rn.scratched
            )]

            if not settled_races:
                continue

            print(f"\n{'━'*90}")
            print(f"📍 {meeting.venue} — {meeting.track_condition or '?'} | "
                  f"{len(settled_races)}/{len(races)} races with results")
            print(f"{'━'*90}")

            for race in settled_races:
                runners = [r for r in race.runners if not r.scratched]
                if len(runners) < 3:
                    continue

                total_races += 1
                meeting_dict = {
                    "venue": meeting.venue,
                    "track_condition": meeting.track_condition or "",
                }
                race_dict = {
                    "id": race.id,
                    "distance": race.distance,
                    "race_number": race.race_number,
                    "class_": race.class_,
                }

                # OLD model (v4 weights, no sharpening)
                old_probs = calculate_race_probabilities(
                    runners, race_dict, meeting_dict,
                    weights=OLD_WEIGHTS, dl_patterns=dl_patterns,
                )
                # NEW model (v5 — new weights + sharpening + DL)
                new_probs = calculate_race_probabilities(
                    runners, race_dict, meeting_dict,
                    weights=None, dl_patterns=dl_patterns,
                )
                # NEW model WITHOUT DL (to isolate DL impact)
                no_dl_weights = dict(DEFAULT_WEIGHTS)
                no_dl_weights["deep_learning"] = 0.0
                # Redistribute DL weight to market+form
                dl_w = DEFAULT_WEIGHTS["deep_learning"]
                no_dl_weights["market"] += dl_w * 0.6
                no_dl_weights["form"] += dl_w * 0.4
                no_dl_probs = calculate_race_probabilities(
                    runners, race_dict, meeting_dict,
                    weights=no_dl_weights, dl_patterns=None,
                )

                runner_map = {str(r.id): r for r in runners}

                old_ranked = sorted(old_probs.items(), key=lambda x: -x[1].win_probability)
                new_ranked = sorted(new_probs.items(), key=lambda x: -x[1].win_probability)
                no_dl_ranked = sorted(no_dl_probs.items(), key=lambda x: -x[1].win_probability)

                # Find winner
                winner = None
                for r in runners:
                    if r.finish_position == 1:
                        winner = r

                if not winner:
                    continue

                total_with_results += 1

                # Top pick from each model
                old_top_r = runner_map.get(old_ranked[0][0]) if old_ranked else None
                new_top_r = runner_map.get(new_ranked[0][0]) if new_ranked else None
                no_dl_top_r = runner_map.get(no_dl_ranked[0][0]) if no_dl_ranked else None

                # Did they agree?
                old_top_id = old_ranked[0][0] if old_ranked else ""
                new_top_id = new_ranked[0][0] if new_ranked else ""
                no_dl_top_id = no_dl_ranked[0][0] if no_dl_ranked else ""

                if old_top_id == new_top_id:
                    agree_count += 1
                else:
                    disagree_count += 1

                # DL impact tracking
                if new_top_id != no_dl_top_id:
                    dl_changed_pick += 1
                    if new_top_r and new_top_r.finish_position == 1:
                        dl_helped += 1
                    elif no_dl_top_r and no_dl_top_r.finish_position == 1:
                        dl_hurt += 1

                # Simulate $10 win bets
                stake = 10.0
                for stats, top_r in [(old_stats, old_top_r), (new_stats, new_top_r), (no_dl_stats, no_dl_top_r)]:
                    stats["bets"] += 1
                    if top_r and top_r.finish_position == 1 and top_r.win_dividend:
                        stats["pnl"] += top_r.win_dividend * stake - stake
                        stats["wins"] += 1
                    elif top_r and top_r.finish_position and top_r.finish_position <= 3:
                        stats["pnl"] -= stake  # Lost the win bet
                        stats["places"] += 1
                    else:
                        stats["pnl"] -= stake

                # Track old vs new accuracy
                old_won = old_top_r and old_top_r.finish_position == 1
                new_won = new_top_r and new_top_r.finish_position == 1
                if old_won and not new_won:
                    old_right_new_wrong += 1
                elif new_won and not old_won:
                    new_right_old_wrong += 1

                # Calibration buckets - track all runners
                for rid, prob in new_probs.items():
                    r = runner_map.get(rid)
                    if not r or not r.finish_position:
                        continue
                    bucket = int(prob.win_probability * 100 / 5) * 5
                    bucket_key = f"{bucket}-{bucket+5}%"
                    cal_buckets[bucket_key]["count"] += 1
                    if r.finish_position == 1:
                        cal_buckets[bucket_key]["wins"] += 1

                # Actual picks tracking
                race_picks = all_picks.get((meeting.id, race.race_number), [])
                actual_top = None
                if race_picks:
                    actual_top = min(race_picks, key=lambda p: p.tip_rank or 99)
                    actual_stats["bets"] += 1
                    # Match actual pick to runner
                    actual_runner = next(
                        (r for r in runners if str(r.saddlecloth) == str(actual_top.saddlecloth)),
                        None
                    )
                    if actual_runner and actual_runner.finish_position == 1 and actual_runner.win_dividend:
                        actual_stats["pnl"] += actual_runner.win_dividend * stake - stake
                        actual_stats["wins"] += 1
                    elif actual_runner and actual_runner.finish_position and actual_runner.finish_position <= 3:
                        actual_stats["pnl"] -= stake
                        actual_stats["places"] += 1
                    else:
                        actual_stats["pnl"] -= stake

                # Display race result
                old_pos = old_top_r.finish_position if old_top_r else "?"
                new_pos = new_top_r.finish_position if new_top_r else "?"

                old_name = (old_top_r.horse_name[:14] if old_top_r else "?")
                new_name = (new_top_r.horse_name[:14] if new_top_r else "?")
                winner_name = winner.horse_name[:14]

                old_mark = "WIN" if old_pos == 1 else f"{old_pos}th" if old_pos else "?"
                new_mark = "WIN" if new_pos == 1 else f"{new_pos}th" if new_pos else "?"

                # DL factor score for top pick
                new_top_factors = new_probs[new_ranked[0][0]].factors if new_ranked else {}
                dl_score = new_top_factors.get("deep_learning", 0.5) if new_top_factors else 0.5
                dl_flag = f" DL={dl_score:.2f}" if abs(dl_score - 0.5) > 0.01 else ""

                diff_marker = " ←DIFF" if old_top_id != new_top_id else ""

                print(f"  R{race.race_number:>2} {race.distance:>4}m {(race.class_ or '')[:12]:12s} "
                      f"Winner: {winner_name:14s} ${winner.win_dividend or 0:.1f} | "
                      f"OLD:{old_name:14s}={old_mark:>4s} NEW:{new_name:14s}={new_mark:>4s}"
                      f"{dl_flag}{diff_marker}")

        # ── Summary ──
        print(f"\n{'='*90}")
        print(f"SUMMARY — {total_with_results} races with results across {len(meetings)} meetings")
        print(f"{'='*90}")

        print(f"\n  {'Model':<25s} {'Wins':>6s} {'SR':>7s} {'P&L':>10s} {'ROI':>8s}")
        print(f"  {'─'*56}")

        for label, s in [
            ("OLD (v4 calibrated)", old_stats),
            ("NEW (v5 sharpened+DL)", new_stats),
            ("NEW without DL", no_dl_stats),
            ("ACTUAL PICKS MADE", actual_stats),
        ]:
            if s["bets"] > 0:
                sr = s["wins"] / s["bets"] * 100
                roi = s["pnl"] / (s["bets"] * 10) * 100
                print(f"  {label:<25s} {s['wins']:>3d}/{s['bets']:<3d} {sr:>6.1f}% ${s['pnl']:>+8.2f} {roi:>+7.1f}%")

        print(f"\n  Pick Agreement: {agree_count} same, {disagree_count} different top pick")
        if disagree_count > 0:
            print(f"  When they disagreed: OLD right {old_right_new_wrong}x, NEW right {new_right_old_wrong}x")

        print(f"\n  DL Pattern Impact:")
        print(f"    Patterns loaded: {len(dl_patterns)}")
        print(f"    DL changed top pick: {dl_changed_pick}/{total_with_results} races")
        if dl_changed_pick > 0:
            print(f"    When DL changed pick: helped {dl_helped}x, hurt {dl_hurt}x, neutral {dl_changed_pick - dl_helped - dl_hurt}x")

        # Calibration table
        print(f"\n  Calibration by Probability Bucket (NEW model):")
        print(f"  {'Bucket':<10s} {'Count':>7s} {'Predicted':>10s} {'Actual':>8s} {'Error':>7s}")
        for bucket_key in sorted(cal_buckets.keys(), key=lambda x: int(x.split('-')[0])):
            b = cal_buckets[bucket_key]
            if b["count"] < 5:
                continue
            actual_wr = b["wins"] / b["count"] * 100
            predicted = (int(bucket_key.split('-')[0]) + 2.5)
            error = actual_wr - predicted
            print(f"  {bucket_key:<10s} {b['count']:>7d} {predicted:>9.1f}% {actual_wr:>7.1f}% {error:>+6.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
