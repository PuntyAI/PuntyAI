"""Run probability engine backtest against all imported Proform data.

Usage (local):
    python scripts/run_backtest.py

Reads data/backtest.db, runs calculate_race_probabilities() on every settled
race, compares predictions vs actual results, outputs data/backtest_results.json.
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Override DB path BEFORE importing punty modules
os.environ["PUNTY_DB_PATH"] = "data/backtest.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/backtest_results.json")


def safe_div(a, b):
    return a / b if b else 0


async def main():
    from sqlalchemy import select, text
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy.orm import selectinload
    from punty.models.meeting import Meeting, Race, Runner
    from punty.probability import calculate_race_probabilities

    # Create our own engine pointing to backtest.db
    bt_engine = create_async_engine(
        "sqlite+aiosqlite:///data/backtest.db",
        echo=False,
        connect_args={"timeout": 30},
    )
    async_session = async_sessionmaker(bt_engine, class_=AsyncSession, expire_on_commit=False)

    start_time = time.time()

    # Count total races first
    async with async_session() as db:
        result = await db.execute(text(
            "SELECT COUNT(*) FROM races r JOIN meetings m ON r.meeting_id = m.id "
            "WHERE r.results_status IS NOT NULL"
        ))
        total_races_expected = result.scalar()
        logger.info(f"Total settled races to process: {total_races_expected}")

    # Load all meetings
    async with async_session() as db:
        result = await db.execute(
            select(Meeting).order_by(Meeting.date)
        )
        meetings = result.scalars().all()

    logger.info(f"Loaded {len(meetings)} meetings")

    # -- Accumulators --
    total_races = 0
    total_runners = 0
    errors = 0

    # Calibration bins (predicted probability ranges vs actual win/place rates)
    cal_bins_win = defaultdict(lambda: {"count": 0, "wins": 0, "pred_sum": 0.0})
    cal_bins_place = defaultdict(lambda: {"count": 0, "placed": 0, "pred_sum": 0.0})

    # Value rating bins
    vr_bins = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0, "placed": 0, "place_pnl": 0.0})

    # Top-N accuracy
    top_n = {n: {"win_hits": 0, "place_hits": 0, "races": 0} for n in [1, 3, 5]}

    # Bet type simulation ($5 flat stake)
    STAKE = 5
    bet_sim = {
        "win_top1": {"bets": 0, "pnl": 0.0},
        "win_top3": {"bets": 0, "pnl": 0.0},
        "win_vr_1.2+": {"bets": 0, "pnl": 0.0},
        "place_top1": {"bets": 0, "pnl": 0.0},
        "place_top3": {"bets": 0, "pnl": 0.0},
        "place_pvr_1.0+": {"bets": 0, "pnl": 0.0},
    }

    # Venue breakdown
    venue_stats = defaultdict(lambda: {
        "races": 0, "top1_win": 0, "top3_win": 0,
        "win_pnl": 0.0, "place_pnl": 0.0, "win_bets": 0, "place_bets": 0,
    })

    # Distance breakdown
    dist_stats = defaultdict(lambda: {
        "races": 0, "top1_win": 0, "top3_win": 0,
        "win_pnl": 0.0, "place_pnl": 0.0, "win_bets": 0, "place_bets": 0,
    })

    # Odds band analysis
    odds_bands = defaultdict(lambda: {
        "count": 0, "wins": 0, "placed": 0,
        "win_pnl": 0.0, "place_pnl": 0.0,
    })

    # Factor sensitivity (track which factors correlate with winners)
    factor_winner_scores = defaultdict(lambda: {"total": 0.0, "count": 0})
    factor_loser_scores = defaultdict(lambda: {"total": 0.0, "count": 0})

    # Process meetings in batches
    batch_size = 20
    for mi in range(0, len(meetings), batch_size):
        batch = meetings[mi:mi + batch_size]
        meeting_ids = [m.id for m in batch]

        async with async_session() as db:
            result = await db.execute(
                select(Meeting)
                .where(Meeting.id.in_(meeting_ids))
                .options(selectinload(Meeting.races).selectinload(Race.runners))
            )
            loaded = {m.id: m for m in result.scalars().all()}

        for meeting in batch:
            m = loaded.get(meeting.id)
            if not m:
                continue

            venue = m.venue
            # Classify venue type
            venue_key = venue.lower().replace(" ", "_")

            for race in m.races:
                if not race.results_status:
                    continue

                active = [r for r in race.runners if not r.scratched and r.finish_position and r.finish_position > 0]
                if len(active) < 3:
                    continue

                # Must have at least one winner
                winners = [r for r in active if r.finish_position == 1]
                if not winners:
                    continue

                field_size = len(active)
                place_count = 2 if field_size <= 7 else 3
                placers = set(r.id for r in active if r.finish_position <= place_count)

                try:
                    probs = calculate_race_probabilities(active, race, m)
                except Exception as e:
                    errors += 1
                    continue

                if not probs:
                    errors += 1
                    continue

                total_races += 1
                total_runners += len(active)

                # Sort by win probability
                ranked = sorted(probs.items(), key=lambda x: -x[1].win_probability)

                # Distance category
                dist = race.distance or 0
                if dist < 1200:
                    dist_cat = "sprint (<1200m)"
                elif dist < 1600:
                    dist_cat = "short (1200-1600m)"
                elif dist < 2000:
                    dist_cat = "middle (1600-2000m)"
                else:
                    dist_cat = "staying (2000m+)"

                # -- Top-N accuracy --
                for n in [1, 3, 5]:
                    top_n[n]["races"] += 1
                    top_ids = set(rid for rid, _ in ranked[:n])
                    winner_id = winners[0].id
                    if winner_id in top_ids:
                        top_n[n]["win_hits"] += 1
                    top_n[n]["place_hits"] += len(top_ids & placers)

                # -- Per-runner analysis --
                for rank_idx, (rid, rp) in enumerate(ranked):
                    runner = next((r for r in active if r.id == rid), None)
                    if not runner:
                        continue

                    is_winner = runner.finish_position == 1
                    is_placed = rid in placers
                    odds = runner.current_odds or runner.opening_odds or 0
                    win_div = runner.win_dividend or 0
                    place_div = runner.place_dividend or 0

                    # -- Calibration bins (10% buckets) --
                    wp = rp.win_probability
                    pp = rp.place_probability
                    win_bin = f"{int(wp * 10) * 10}-{int(wp * 10) * 10 + 10}%"
                    place_bin = f"{int(pp * 10) * 10}-{int(pp * 10) * 10 + 10}%"

                    cal_bins_win[win_bin]["count"] += 1
                    cal_bins_win[win_bin]["pred_sum"] += wp
                    if is_winner:
                        cal_bins_win[win_bin]["wins"] += 1

                    cal_bins_place[place_bin]["count"] += 1
                    cal_bins_place[place_bin]["pred_sum"] += pp
                    if is_placed:
                        cal_bins_place[place_bin]["placed"] += 1

                    # -- Value rating bins --
                    vr = rp.value_rating
                    if vr < 0.5:
                        vr_key = "<0.5"
                    elif vr < 0.8:
                        vr_key = "0.5-0.8"
                    elif vr < 1.0:
                        vr_key = "0.8-1.0"
                    elif vr < 1.2:
                        vr_key = "1.0-1.2"
                    elif vr < 1.5:
                        vr_key = "1.2-1.5"
                    elif vr < 2.0:
                        vr_key = "1.5-2.0"
                    else:
                        vr_key = "2.0+"

                    vr_bins[vr_key]["count"] += 1
                    if is_winner:
                        vr_bins[vr_key]["wins"] += 1
                        vr_bins[vr_key]["pnl"] += (win_div - 1) * STAKE if win_div > 0 else 0
                    else:
                        vr_bins[vr_key]["pnl"] -= STAKE
                    if is_placed:
                        vr_bins[vr_key]["placed"] += 1
                        vr_bins[vr_key]["place_pnl"] += (place_div - 1) * STAKE if place_div > 0 else 0
                    else:
                        vr_bins[vr_key]["place_pnl"] -= STAKE

                    # -- Odds bands --
                    if odds > 0:
                        if odds < 2:
                            ob = "<$2"
                        elif odds < 4:
                            ob = "$2-4"
                        elif odds < 6:
                            ob = "$4-6"
                        elif odds < 10:
                            ob = "$6-10"
                        elif odds < 20:
                            ob = "$10-20"
                        elif odds < 50:
                            ob = "$20-50"
                        else:
                            ob = "$50+"

                        odds_bands[ob]["count"] += 1
                        if is_winner:
                            odds_bands[ob]["wins"] += 1
                            odds_bands[ob]["win_pnl"] += (win_div - 1) * STAKE if win_div > 0 else 0
                        else:
                            odds_bands[ob]["win_pnl"] -= STAKE
                        if is_placed:
                            odds_bands[ob]["placed"] += 1
                            odds_bands[ob]["place_pnl"] += (place_div - 1) * STAKE if place_div > 0 else 0
                        else:
                            odds_bands[ob]["place_pnl"] -= STAKE

                    # -- Bet simulations --
                    # Win on top 1
                    if rank_idx == 0:
                        bet_sim["win_top1"]["bets"] += 1
                        if is_winner and win_div > 0:
                            bet_sim["win_top1"]["pnl"] += (win_div - 1) * STAKE
                        else:
                            bet_sim["win_top1"]["pnl"] -= STAKE

                        bet_sim["place_top1"]["bets"] += 1
                        if is_placed and place_div > 0:
                            bet_sim["place_top1"]["pnl"] += (place_div - 1) * STAKE
                        else:
                            bet_sim["place_top1"]["pnl"] -= STAKE

                        # Venue stats
                        venue_stats[venue]["races"] += 1
                        if is_winner:
                            venue_stats[venue]["top1_win"] += 1
                        venue_stats[venue]["win_bets"] += 1
                        venue_stats[venue]["win_pnl"] += ((win_div - 1) * STAKE if is_winner and win_div > 0 else -STAKE)
                        venue_stats[venue]["place_bets"] += 1
                        venue_stats[venue]["place_pnl"] += ((place_div - 1) * STAKE if is_placed and place_div > 0 else -STAKE)

                        # Distance stats
                        dist_stats[dist_cat]["races"] += 1
                        if is_winner:
                            dist_stats[dist_cat]["top1_win"] += 1
                        dist_stats[dist_cat]["win_bets"] += 1
                        dist_stats[dist_cat]["win_pnl"] += ((win_div - 1) * STAKE if is_winner and win_div > 0 else -STAKE)
                        dist_stats[dist_cat]["place_bets"] += 1
                        dist_stats[dist_cat]["place_pnl"] += ((place_div - 1) * STAKE if is_placed and place_div > 0 else -STAKE)

                    # Win/place on top 3
                    if rank_idx < 3:
                        bet_sim["win_top3"]["bets"] += 1
                        if is_winner and win_div > 0:
                            bet_sim["win_top3"]["pnl"] += (win_div - 1) * STAKE
                        else:
                            bet_sim["win_top3"]["pnl"] -= STAKE

                        bet_sim["place_top3"]["bets"] += 1
                        if is_placed and place_div > 0:
                            bet_sim["place_top3"]["pnl"] += (place_div - 1) * STAKE
                        else:
                            bet_sim["place_top3"]["pnl"] -= STAKE

                        if is_winner:
                            venue_stats[venue]["top3_win"] += 1
                            dist_stats[dist_cat]["top3_win"] += 1

                    # Value-based bets
                    if vr >= 1.2:
                        bet_sim["win_vr_1.2+"]["bets"] += 1
                        if is_winner and win_div > 0:
                            bet_sim["win_vr_1.2+"]["pnl"] += (win_div - 1) * STAKE
                        else:
                            bet_sim["win_vr_1.2+"]["pnl"] -= STAKE

                    pvr = rp.place_value_rating
                    if pvr >= 1.0:
                        bet_sim["place_pvr_1.0+"]["bets"] += 1
                        if is_placed and place_div > 0:
                            bet_sim["place_pvr_1.0+"]["pnl"] += (place_div - 1) * STAKE
                        else:
                            bet_sim["place_pvr_1.0+"]["pnl"] -= STAKE

                    # -- Factor analysis --
                    for factor, score in rp.factors.items():
                        if is_winner:
                            factor_winner_scores[factor]["total"] += score
                            factor_winner_scores[factor]["count"] += 1
                        else:
                            factor_loser_scores[factor]["total"] += score
                            factor_loser_scores[factor]["count"] += 1

        # Progress logging
        processed = min(mi + batch_size, len(meetings))
        if processed % 100 == 0 or processed == len(meetings):
            elapsed = time.time() - start_time
            logger.info(f"Processed {processed}/{len(meetings)} meetings, "
                        f"{total_races} races, {elapsed:.0f}s elapsed")

    # -- Build results JSON --
    elapsed = time.time() - start_time
    logger.info(f"\nBacktest complete: {total_races} races, {total_runners} runners, "
                f"{errors} errors, {elapsed:.0f}s")

    results = {
        "meta": {
            "races": total_races,
            "runners": total_runners,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1),
            "date_range": "2025-01-01 to 2025-12-31",
        },
        "calibration": {
            "win": {k: {
                "count": v["count"],
                "wins": v["wins"],
                "actual_rate": round(safe_div(v["wins"], v["count"]) * 100, 2),
                "predicted_rate": round(safe_div(v["pred_sum"], v["count"]) * 100, 2),
            } for k, v in sorted(cal_bins_win.items())},
            "place": {k: {
                "count": v["count"],
                "placed": v["placed"],
                "actual_rate": round(safe_div(v["placed"], v["count"]) * 100, 2),
                "predicted_rate": round(safe_div(v["pred_sum"], v["count"]) * 100, 2),
            } for k, v in sorted(cal_bins_place.items())},
        },
        "value_bins": {k: {
            "count": v["count"],
            "wins": v["wins"],
            "win_rate": round(safe_div(v["wins"], v["count"]) * 100, 2),
            "win_roi": round(safe_div(v["pnl"], v["count"] * STAKE) * 100, 2),
            "placed": v["placed"],
            "place_rate": round(safe_div(v["placed"], v["count"]) * 100, 2),
            "place_roi": round(safe_div(v["place_pnl"], v["count"] * STAKE) * 100, 2),
        } for k, v in sorted(vr_bins.items())},
        "top_n_accuracy": {str(n): {
            "races": v["races"],
            "win_hits": v["win_hits"],
            "win_rate": round(safe_div(v["win_hits"], v["races"]) * 100, 2),
            "place_hits": v["place_hits"],
            "place_rate": round(safe_div(v["place_hits"], v["races"] * n) * 100, 2),
        } for n, v in top_n.items()},
        "bet_simulations": {k: {
            "bets": v["bets"],
            "pnl": round(v["pnl"], 2),
            "roi": round(safe_div(v["pnl"], v["bets"] * STAKE) * 100, 2),
        } for k, v in bet_sim.items()},
        "odds_bands": {k: {
            "count": v["count"],
            "wins": v["wins"],
            "win_rate": round(safe_div(v["wins"], v["count"]) * 100, 2),
            "win_roi": round(safe_div(v["win_pnl"], v["count"] * STAKE) * 100, 2),
            "placed": v["placed"],
            "place_rate": round(safe_div(v["placed"], v["count"]) * 100, 2),
            "place_roi": round(safe_div(v["place_pnl"], v["count"] * STAKE) * 100, 2),
        } for k, v in sorted(odds_bands.items())},
        "venue_breakdown": {k: {
            "races": v["races"],
            "top1_win_rate": round(safe_div(v["top1_win"], v["races"]) * 100, 2),
            "top3_win_rate": round(safe_div(v["top3_win"], v["races"]) * 100, 2),
            "win_roi": round(safe_div(v["win_pnl"], v["win_bets"] * STAKE) * 100, 2),
            "place_roi": round(safe_div(v["place_pnl"], v["place_bets"] * STAKE) * 100, 2),
        } for k, v in sorted(venue_stats.items(), key=lambda x: -x[1]["races"])},
        "distance_breakdown": {k: {
            "races": v["races"],
            "top1_win_rate": round(safe_div(v["top1_win"], v["races"]) * 100, 2),
            "top3_win_rate": round(safe_div(v["top3_win"], v["races"]) * 100, 2),
            "win_roi": round(safe_div(v["win_pnl"], v["win_bets"] * STAKE) * 100, 2),
            "place_roi": round(safe_div(v["place_pnl"], v["place_bets"] * STAKE) * 100, 2),
        } for k, v in dist_stats.items()},
        "factor_sensitivity": {},
    }

    # Factor sensitivity: avg score of winners vs losers
    all_factors = set(factor_winner_scores.keys()) | set(factor_loser_scores.keys())
    for f in sorted(all_factors):
        w = factor_winner_scores[f]
        l = factor_loser_scores[f]
        w_avg = safe_div(w["total"], w["count"])
        l_avg = safe_div(l["total"], l["count"])
        results["factor_sensitivity"][f] = {
            "winner_avg_score": round(w_avg, 4),
            "loser_avg_score": round(l_avg, 4),
            "differential": round(w_avg - l_avg, 4),
            "winner_samples": w["count"],
        }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS — {total_races} races, {total_runners} runners")
    print(f"{'='*70}")

    print(f"\n{'-'*70}")
    print("TOP-N ACCURACY")
    print(f"{'-'*70}")
    for n, v in top_n.items():
        wr = safe_div(v["win_hits"], v["races"]) * 100
        print(f"  Top {n}: {v['win_hits']}/{v['races']} winners ({wr:.1f}%)")

    print(f"\n{'-'*70}")
    print("BET SIMULATIONS ($5 flat)")
    print(f"{'-'*70}")
    for k, v in bet_sim.items():
        roi = safe_div(v["pnl"], v["bets"] * STAKE) * 100
        print(f"  {k:<20} {v['bets']:>7} bets  P&L ${v['pnl']:>+10.2f}  ROI {roi:>+6.1f}%")

    print(f"\n{'-'*70}")
    print("WIN CALIBRATION")
    print(f"{'-'*70}")
    for k, v in sorted(cal_bins_win.items()):
        actual = safe_div(v["wins"], v["count"]) * 100
        predicted = safe_div(v["pred_sum"], v["count"]) * 100
        print(f"  {k:<10} {v['count']:>7} runners  predicted {predicted:>5.1f}%  actual {actual:>5.1f}%  "
              f"diff {actual - predicted:>+5.1f}pp")

    print(f"\n{'-'*70}")
    print("VALUE RATING PERFORMANCE")
    print(f"{'-'*70}")
    for k in ["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0+"]:
        v = vr_bins[k]
        if v["count"] > 0:
            wr = safe_div(v["wins"], v["count"]) * 100
            roi = safe_div(v["pnl"], v["count"] * STAKE) * 100
            p_roi = safe_div(v["place_pnl"], v["count"] * STAKE) * 100
            print(f"  VR {k:<8} {v['count']:>7}  WinRate {wr:>5.1f}%  WinROI {roi:>+6.1f}%  PlaceROI {p_roi:>+6.1f}%")

    print(f"\n{'-'*70}")
    print("DISTANCE BREAKDOWN")
    print(f"{'-'*70}")
    for k, v in dist_stats.items():
        t1 = safe_div(v["top1_win"], v["races"]) * 100
        wr = safe_div(v["win_pnl"], v["win_bets"] * STAKE) * 100
        pr = safe_div(v["place_pnl"], v["place_bets"] * STAKE) * 100
        print(f"  {k:<25} {v['races']:>6} races  Top1 {t1:>5.1f}%  WinROI {wr:>+6.1f}%  PlaceROI {pr:>+6.1f}%")

    print(f"\n{'-'*70}")
    print("TOP 20 VENUES (by race count)")
    print(f"{'-'*70}")
    top_venues = sorted(venue_stats.items(), key=lambda x: -x[1]["races"])[:20]
    for k, v in top_venues:
        t1 = safe_div(v["top1_win"], v["races"]) * 100
        wr = safe_div(v["win_pnl"], v["win_bets"] * STAKE) * 100
        pr = safe_div(v["place_pnl"], v["place_bets"] * STAKE) * 100
        print(f"  {k:<25} {v['races']:>5}  Top1 {t1:>5.1f}%  WinROI {wr:>+6.1f}%  PlaceROI {pr:>+6.1f}%")

    print(f"\n{'-'*70}")
    print("FACTOR SENSITIVITY (winner vs loser avg score)")
    print(f"{'-'*70}")
    for f in sorted(all_factors):
        s = results["factor_sensitivity"][f]
        print(f"  {f:<20} winner={s['winner_avg_score']:.4f}  loser={s['loser_avg_score']:.4f}  "
              f"diff={s['differential']:>+.4f}  (n={s['winner_samples']})")

    await bt_engine.dispose()


asyncio.run(main())
