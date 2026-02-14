"""Compare calibrated vs old probability model for today's Flemington.

Runs both weight sets against today's races and shows what each would have picked,
alongside actual results.

Usage (on server):
    cd /opt/puntyai && source venv/bin/activate
    python scripts/compare_flemington_today.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OLD_WEIGHTS = {
    "market": 0.22, "movement": 0.07, "form": 0.15, "class_fitness": 0.05,
    "pace": 0.11, "barrier": 0.09, "jockey_trainer": 0.11,
    "weight_carried": 0.05, "horse_profile": 0.05, "deep_learning": 0.10,
}


async def main():
    from punty.models.database import async_session, init_db
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from punty.models.meeting import Meeting, Race, Runner
    from punty.models.content import Content
    from punty.models.pick import Pick
    from punty.probability import (
        calculate_race_probabilities, load_dl_patterns_for_probability,
        _get_calibrated_weights, DEFAULT_WEIGHTS,
    )

    await init_db()
    async with async_session() as db:
        # Load DL patterns
        dl_patterns = await load_dl_patterns_for_probability(db)

        # Find today's Flemington meeting
        result = await db.execute(
            select(Meeting)
            .options(selectinload(Meeting.races).selectinload(Race.runners))
            .where(Meeting.date == "2026-02-14")
            .where(Meeting.venue.ilike("%flemington%"))
        )
        meeting = result.scalar_one_or_none()

        if not meeting:
            # Try broader search
            result = await db.execute(
                select(Meeting)
                .options(selectinload(Meeting.races).selectinload(Race.runners))
                .where(Meeting.date == "2026-02-14")
            )
            meetings = result.scalars().all()
            if not meetings:
                print("No meetings found for 2026-02-14")
                return
            print(f"Found {len(meetings)} meetings today:")
            for m in meetings:
                print(f"  {m.venue} ({m.id})")
            # Use first one or Flemington
            meeting = next((m for m in meetings if "flem" in (m.venue or "").lower()), meetings[0])

        print(f"\n{'='*80}")
        print(f"FLEMINGTON DRY RUN COMPARISON — {meeting.venue} {meeting.date}")
        print(f"{'='*80}")
        print(f"Track condition: {meeting.track_condition or 'Unknown'}")
        print(f"Weather: {meeting.weather_condition or 'Unknown'}")

        # Get calibrated weights
        cal_weights = _get_calibrated_weights()
        print(f"\nOLD weights: market={OLD_WEIGHTS['market']*100:.0f}% form={OLD_WEIGHTS['form']*100:.0f}% jt={OLD_WEIGHTS['jockey_trainer']*100:.0f}%")
        if cal_weights:
            print(f"NEW weights: market={cal_weights['market']*100:.0f}% form={cal_weights['form']*100:.0f}% jt={cal_weights['jockey_trainer']*100:.0f}%")
        else:
            print(f"NEW weights: market={DEFAULT_WEIGHTS['market']*100:.0f}% form={DEFAULT_WEIGHTS['form']*100:.0f}% jt={DEFAULT_WEIGHTS['jockey_trainer']*100:.0f}%")

        # Load actual picks for this meeting
        pick_result = await db.execute(
            select(Pick).where(Pick.meeting_id == meeting.id)
        )
        actual_picks = pick_result.scalars().all()
        picks_by_race = {}
        for p in actual_picks:
            if p.race_number and p.pick_type == "selection":
                picks_by_race.setdefault(p.race_number, []).append(p)

        races = sorted(meeting.races, key=lambda r: r.race_number)
        total_old_pnl = 0.0
        total_new_pnl = 0.0
        total_old_bets = 0
        total_new_bets = 0
        old_wins = 0
        new_wins = 0

        for race in races:
            runners = [r for r in race.runners if not r.scratched]
            if len(runners) < 3:
                continue

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

            # Run with OLD weights
            old_probs = calculate_race_probabilities(
                runners, race_dict, meeting_dict,
                weights=OLD_WEIGHTS, dl_patterns=dl_patterns,
            )

            # Run with NEW (calibrated) weights — pass None to use calibrated
            new_probs = calculate_race_probabilities(
                runners, race_dict, meeting_dict,
                weights=None, dl_patterns=dl_patterns,
            )

            # Sort by win probability
            old_ranked = sorted(old_probs.items(), key=lambda x: -x[1].win_probability)
            new_ranked = sorted(new_probs.items(), key=lambda x: -x[1].win_probability)

            # Find actual results
            winner = None
            placed = []
            runner_map = {str(r.id): r for r in runners}
            # Also map by saddlecloth for runner lookup
            sc_map = {str(r.saddlecloth): r for r in runners}

            for r in runners:
                if r.finish_position == 1:
                    winner = r
                if r.finish_position and r.finish_position <= 3:
                    placed.append(r)

            print(f"\n{'─'*80}")
            print(f"R{race.race_number} {race.name or ''} — {race.distance}m {race.class_ or ''} ({len(runners)} runners)")
            if winner:
                print(f"  RESULT: 1st {winner.horse_name} (#{winner.saddlecloth}) ${winner.win_dividend or '?'}")
                for p in sorted(placed, key=lambda x: x.finish_position):
                    if p.finish_position > 1:
                        print(f"          {p.finish_position}{'nd' if p.finish_position == 2 else 'rd'} {p.horse_name} (#{p.saddlecloth})")
            else:
                print(f"  RESULT: Not yet available")

            # Show top 4 picks from each model
            print(f"\n  {'OLD MODEL (hand-tuned)':^38s} │ {'NEW MODEL (calibrated)':^38s}")
            print(f"  {'─'*38} │ {'─'*38}")

            for i in range(min(4, len(old_ranked), len(new_ranked))):
                old_rid, old_p = old_ranked[i]
                new_rid, new_p = new_ranked[i]
                old_r = runner_map.get(old_rid)
                new_r = runner_map.get(new_rid)

                old_name = old_r.horse_name[:16] if old_r else old_rid[:16]
                new_name = new_r.horse_name[:16] if new_r else new_rid[:16]
                old_sc = f"#{old_r.saddlecloth}" if old_r else ""
                new_sc = f"#{new_r.saddlecloth}" if new_r else ""
                old_pos = old_r.finish_position if old_r else None
                new_pos = new_r.finish_position if new_r else None

                old_result = f" → {old_pos}{'st' if old_pos == 1 else 'nd' if old_pos == 2 else 'rd' if old_pos == 3 else 'th'}" if old_pos else ""
                new_result = f" → {new_pos}{'st' if new_pos == 1 else 'nd' if new_pos == 2 else 'rd' if new_pos == 3 else 'th'}" if new_pos else ""

                # Highlight winners
                old_mark = " ✓" if old_pos == 1 else ""
                new_mark = " ✓" if new_pos == 1 else ""

                print(f"  {i+1}. {old_name:16s} {old_sc:4s} {old_p.win_probability*100:5.1f}%{old_result}{old_mark:>3s} │ "
                      f"{i+1}. {new_name:16s} {new_sc:4s} {new_p.win_probability*100:5.1f}%{new_result}{new_mark:>3s}")

            # Simulate $10 win bet on top pick
            if winner:
                old_top_rid = old_ranked[0][0]
                new_top_rid = new_ranked[0][0]
                old_top_r = runner_map.get(old_top_rid)
                new_top_r = runner_map.get(new_top_rid)

                stake = 10.0
                total_old_bets += 1
                total_new_bets += 1

                if old_top_r and old_top_r.finish_position == 1 and old_top_r.win_dividend:
                    old_race_pnl = old_top_r.win_dividend * stake - stake
                    old_wins += 1
                else:
                    old_race_pnl = -stake

                if new_top_r and new_top_r.finish_position == 1 and new_top_r.win_dividend:
                    new_race_pnl = new_top_r.win_dividend * stake - stake
                    new_wins += 1
                else:
                    new_race_pnl = -stake

                total_old_pnl += old_race_pnl
                total_new_pnl += new_race_pnl

            # Show actual picks that were made
            race_picks = picks_by_race.get(race.race_number, [])
            if race_picks:
                pick_str = ", ".join(f"{p.horse_name} (#{p.saddlecloth})" for p in sorted(race_picks, key=lambda x: x.tip_rank or 99)[:3])
                print(f"\n  ACTUAL PICKS MADE: {pick_str}")

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY ($10 win on top pick per race)")
        print(f"{'='*80}")
        if total_old_bets > 0:
            print(f"OLD MODEL: {old_wins}/{total_old_bets} winners, P&L ${total_old_pnl:+.2f}, ROI {total_old_pnl/(total_old_bets*10)*100:+.1f}%")
            print(f"NEW MODEL: {new_wins}/{total_new_bets} winners, P&L ${total_new_pnl:+.2f}, ROI {total_new_pnl/(total_new_bets*10)*100:+.1f}%")
        else:
            print("No races with results yet")


if __name__ == "__main__":
    asyncio.run(main())
