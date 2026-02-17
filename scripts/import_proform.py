"""Import Proform historical data into a separate backtest SQLite database.

Usage (local):
    python scripts/import_proform.py

Creates data/backtest.db with the same schema as punty.db, populated from
13 months of Proform data at D:\\Punty\\DatafromProform.

Data structure discovered:
- 2025/January → corrupt (2009 dates), skip
- 2025/February-December → valid 2025 data
- 2026/ top-level → the REAL January 2025 data (252 meetings) + Ratings
- 2026/ subfolders → exact duplicates of 2025, skip
"""

import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, date, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROFORM_ROOT = Path("D:/Punty/DatafromProform")
BACKTEST_DB = Path("data/backtest.db")


def slugify(name: str) -> str:
    """Convert venue name to URL-friendly slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def parse_flucs(flucs_str: str) -> dict:
    """Parse Proform Flucs string like 'opening,6.00;mid,5.50;starting,6.50;'."""
    result = {}
    if not flucs_str:
        return result
    for part in flucs_str.strip().split(";"):
        part = part.strip()
        if "," in part:
            key, val = part.split(",", 1)
            try:
                result[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return result


def format_record(rec: dict) -> str:
    """Format a Proform record dict as 'Starts-Firsts-Seconds-Thirds'."""
    if not rec:
        return None
    starts = rec.get("Starts", 0)
    if not starts:
        return None
    return json.dumps({
        "starts": starts,
        "wins": rec.get("Firsts", 0),
        "seconds": rec.get("Seconds", 0),
        "thirds": rec.get("Thirds", 0),
    })


def map_speed_position(settle_pos: int) -> str:
    """Map PredictedSettlePosition to speed_map_position label."""
    if settle_pos is None or settle_pos <= 0:
        return None
    if settle_pos <= 3:
        return "leader"
    elif settle_pos <= 6:
        return "on_pace"
    elif settle_pos <= 10:
        return "midfield"
    else:
        return "backmarker"


def get_data_sources():
    """Build list of (path, year_label) data sources to import, avoiding duplicates."""
    sources = []

    # 2025/February through December (skip January — corrupt 2009 dates)
    months_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    for month in months_order[1:]:  # Skip January
        path = PROFORM_ROOT / "2025" / month
        if path.is_dir() and (path / "meetings.json").exists():
            sources.append((path, f"2025/{month}"))

    # Real January 2025 data is in 2026/ top-level
    jan_path = PROFORM_ROOT / "2026"
    if (jan_path / "meetings.json").exists():
        sources.append((jan_path, "2025/January (from 2026 root)"))

    return sources


async def main():
    import aiosqlite
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import text

    # Remove old backtest DB
    if BACKTEST_DB.exists():
        BACKTEST_DB.unlink()
        logger.info(f"Removed old {BACKTEST_DB}")

    BACKTEST_DB.parent.mkdir(parents=True, exist_ok=True)

    # Create engine pointing to backtest.db
    db_url = f"sqlite+aiosqlite:///{BACKTEST_DB}"
    engine = create_async_engine(db_url, echo=False, connect_args={"timeout": 30})

    # Create all tables using our existing models
    from punty.models.database import Base
    from punty.models import meeting as _m, content as _c, pick as _p  # noqa: register models

    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA busy_timeout=30000"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Load ratings index (Jan 2025 only)
    ratings_by_key = {}  # (track, race_no, tab_no) -> rating dict
    ratings_dir = PROFORM_ROOT / "2026" / "Ratings"
    if ratings_dir.is_dir():
        logger.info("Loading ratings files...")
        for rf in sorted(ratings_dir.glob("*.json")):
            with open(rf, encoding="utf-8") as f:
                rdata = json.load(f)
            for item in rdata.get("PayLoad", []):
                key = (item["Track"], item["RaceNo"], item["TabNo"])
                ratings_by_key[key] = item
        logger.info(f"Loaded {len(ratings_by_key)} rating records")

    sources = get_data_sources()
    total_meetings = 0
    total_races = 0
    total_runners = 0
    skipped_trials = 0

    for source_path, label in sources:
        logger.info(f"\n{'='*60}")
        logger.info(f"Importing {label}")
        logger.info(f"{'='*60}")

        # Load meetings data
        with open(source_path / "meetings.json", encoding="utf-8") as f:
            meetings_data = json.load(f)

        # Load results data
        results_path = source_path / "results.json"
        results_by_meeting = {}
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                results_data = json.load(f)
            for r in results_data:
                mid = r.get("MeetingId")
                if mid:
                    results_by_meeting[str(mid)] = r

        month_meetings = 0
        month_races = 0
        month_runners = 0

        # Process in batches
        batch_size = 50
        for batch_start in range(0, len(meetings_data), batch_size):
            batch = meetings_data[batch_start:batch_start + batch_size]

            async with session_factory() as db:
                for meeting_data in batch:
                    # Skip barrier trials and jumps
                    if meeting_data.get("IsBarrierTrial"):
                        skipped_trials += 1
                        continue
                    if meeting_data.get("IsJumps"):
                        continue

                    track = meeting_data.get("Track", {})
                    venue_name = track.get("Name", "")
                    meeting_date_str = meeting_data.get("MeetingDate", "")
                    if not venue_name or not meeting_date_str:
                        continue

                    # Parse date
                    try:
                        meeting_date = datetime.fromisoformat(meeting_date_str.replace("T00:00:00", "")).date()
                    except ValueError:
                        continue

                    # Skip clearly wrong dates
                    if meeting_date.year < 2020:
                        continue

                    venue_slug = slugify(venue_name)
                    meeting_id = f"{venue_slug}-{meeting_date.isoformat()}"
                    pf_meeting_id = str(meeting_data.get("MeetingId", ""))

                    location = track.get("Location", "")
                    source_label = f"proform_{location}" if location else "proform"

                    # Get results for this meeting
                    meeting_results = results_by_meeting.get(pf_meeting_id, {})
                    results_by_race = {}
                    for rr in meeting_results.get("RaceResults", []):
                        rn = rr.get("RaceNumber")
                        if rn:
                            results_by_race[rn] = rr

                    # Create Meeting
                    await db.execute(text(
                        "INSERT OR IGNORE INTO meetings (id, venue, date, track_condition, rail_position, "
                        "source, meeting_type, selected, track_condition_locked, created_at, updated_at) "
                        "VALUES (:id, :venue, :date, :tc, :rail, :source, :mt, 0, 0, :now, :now)"
                    ), {
                        "id": meeting_id,
                        "venue": venue_name,
                        "date": meeting_date.isoformat(),
                        "tc": meeting_data.get("ExpectedCondition"),
                        "rail": meeting_data.get("RailPosition") or None,
                        "source": source_label,
                        "mt": "race",
                        "now": datetime.now(timezone.utc).isoformat(),
                    })

                    races = meeting_data.get("Races") or []
                    if not races:
                        continue

                    month_meetings += 1

                    for race_data in races:
                        race_num = race_data.get("Number")
                        if not race_num:
                            continue

                        race_id = f"{meeting_id}-r{race_num}"
                        distance = race_data.get("Distance", 0)
                        race_class = race_data.get("RaceClass", "")
                        prize_money = race_data.get("PrizeMoney", 0)

                        # Get result-level track condition (more accurate)
                        race_result = results_by_race.get(race_num, {})
                        track_condition = race_result.get("TrackConditionLabel") or meeting_data.get("ExpectedCondition")
                        winning_time = race_result.get("OfficialRaceTimeString")

                        # Determine results status
                        has_results = bool(race_result.get("Runners"))
                        results_status = "Paying" if has_results else None

                        await db.execute(text(
                            "INSERT OR IGNORE INTO races (id, meeting_id, race_number, name, distance, "
                            "class, prize_money, status, track_condition, results_status, winning_time, "
                            "age_restriction, weight_type, race_type, created_at, updated_at) "
                            "VALUES (:id, :mid, :rn, :name, :dist, :cls, :prize, :status, :tc, :rs, :wt, "
                            ":age, :weight_type, :race_type, :now, :now)"
                        ), {
                            "id": race_id,
                            "mid": meeting_id,
                            "rn": race_num,
                            "name": race_data.get("Name", ""),
                            "dist": distance,
                            "cls": race_class,
                            "prize": prize_money,
                            "status": "final" if has_results else "scheduled",
                            "tc": track_condition,
                            "rs": results_status,
                            "wt": winning_time,
                            "age": race_data.get("AgeRestrictions"),
                            "weight_type": race_data.get("WeightType"),
                            "race_type": race_data.get("SexRestrictions"),
                            "now": datetime.now(timezone.utc).isoformat(),
                        })

                        month_races += 1

                        # Build results lookup for this race
                        result_runners = {}
                        for rr in race_result.get("Runners", []):
                            tab = rr.get("TabNo")
                            if tab:
                                result_runners[tab] = rr

                        runners = race_data.get("Runners") or []
                        field_size = 0

                        for runner_data in runners:
                            tab_no = runner_data.get("TabNo")
                            if not tab_no:
                                continue

                            runner_id = f"{race_id}-{tab_no}"
                            horse_name = runner_data.get("Name", "")

                            # Check if scratched (emergency or no position + no result)
                            is_emergency = runner_data.get("EmergencyIndicator", False)
                            position = runner_data.get("Position")
                            result_runner = result_runners.get(tab_no, {})
                            result_position = result_runner.get("Position")

                            # Use result position if available, fall back to meetings position
                            finish_position = result_position or position
                            scratched = is_emergency and not finish_position

                            if not scratched:
                                field_size += 1

                            # Parse odds from Flucs
                            flucs = parse_flucs(result_runner.get("Flucs", ""))
                            starting_price = result_runner.get("Price") or runner_data.get("PriceSP")
                            opening_odds = flucs.get("opening")
                            current_odds = flucs.get("starting") or starting_price

                            # Estimate dividends
                            win_dividend = None
                            place_dividend = None
                            result_margin = result_runner.get("Margin") or runner_data.get("Margin")

                            if finish_position == 1 and starting_price:
                                try:
                                    win_dividend = float(starting_price)
                                except (ValueError, TypeError):
                                    pass

                            # Estimate place dividend: (SP - 1) / 3 + 1
                            place_count = 2 if field_size <= 7 else 3
                            if finish_position and finish_position <= place_count and starting_price:
                                try:
                                    sp = float(starting_price)
                                    place_dividend = round((sp - 1) / 3 + 1, 2)
                                except (ValueError, TypeError):
                                    pass

                            # Last five from Last10
                            last10 = (runner_data.get("Last10") or "").strip()
                            last_five = last10[:5] if last10 else None

                            # Career record
                            starts = runner_data.get("CareerStarts", 0)
                            wins = runner_data.get("CareerWins", 0)
                            seconds = runner_data.get("CareerSeconds", 0)
                            thirds = runner_data.get("CareerThirds", 0)
                            career_record = f"{starts}: {wins}-{seconds}-{thirds}" if starts else None

                            # Trainer/jockey
                            trainer_info = runner_data.get("Trainer", {})
                            jockey_info = runner_data.get("Jockey", {})
                            trainer_name = trainer_info.get("FullName", "") if isinstance(trainer_info, dict) else str(trainer_info)
                            jockey_name = jockey_info.get("FullName", "") if isinstance(jockey_info, dict) else str(jockey_info)
                            trainer_location = trainer_info.get("Location") if isinstance(trainer_info, dict) else None

                            # Stats
                            track_stats = format_record(runner_data.get("TrackRecord"))
                            distance_stats = format_record(runner_data.get("DistanceRecord"))
                            track_dist_stats = format_record(runner_data.get("TrackDistRecord"))
                            good_track_stats = format_record(runner_data.get("GoodRecord"))
                            soft_track_stats = format_record(runner_data.get("SoftRecord"))
                            heavy_track_stats = format_record(runner_data.get("HeavyRecord"))
                            first_up_stats = format_record(runner_data.get("FirstUpRecord"))
                            second_up_stats = format_record(runner_data.get("SecondUpRecord"))

                            # Ratings (Jan 2025 only)
                            rating = ratings_by_key.get((venue_name, race_num, tab_no), {})
                            pf_ai_score = rating.get("PFAIScore") or None
                            pf_ai_price = rating.get("PFAIPrice") or None
                            pf_ai_rank = rating.get("PFAIRank") or None
                            predicted_settle = rating.get("PredictedSettlePostion")
                            speed_map_pos = map_speed_position(predicted_settle)

                            # Stewards
                            stewards = result_runner.get("StewardsReports")
                            gear_changes = result_runner.get("GearChanges")

                            await db.execute(text(
                                "INSERT OR IGNORE INTO runners "
                                "(id, race_id, horse_name, saddlecloth, barrier, weight, jockey, trainer, "
                                "form, career_record, speed_map_position, current_odds, opening_odds, "
                                "scratched, horse_age, horse_sex, horse_colour, sire, dam, dam_sire, "
                                "career_prize_money, last_five, handicap_rating, "
                                "track_dist_stats, track_stats, distance_stats, "
                                "first_up_stats, second_up_stats, good_track_stats, soft_track_stats, "
                                "heavy_track_stats, trainer_location, stewards_comment, gear_changes, "
                                "finish_position, result_margin, starting_price, win_dividend, place_dividend, "
                                "pf_ai_score, pf_ai_price, pf_ai_rank, "
                                "created_at, updated_at) "
                                "VALUES (:id, :rid, :name, :tab, :bar, :wt, :jockey, :trainer, "
                                ":form, :career, :smp, :odds, :opening, "
                                ":scr, :age, :sex, :colour, :sire, :dam, :damsire, "
                                ":prize, :last5, :hcap, "
                                ":td_stats, :t_stats, :d_stats, "
                                ":fu_stats, :su_stats, :good_stats, :soft_stats, "
                                ":heavy_stats, :trainer_loc, :stewards, :gear_ch, "
                                ":fp, :margin, :sp, :wd, :pd, "
                                ":ai_score, :ai_price, :ai_rank, "
                                ":now, :now)"
                            ), {
                                "id": runner_id,
                                "rid": race_id,
                                "name": horse_name,
                                "tab": tab_no,
                                "bar": runner_data.get("Barrier"),
                                "wt": runner_data.get("Weight"),
                                "jockey": jockey_name,
                                "trainer": trainer_name,
                                "form": last10,
                                "career": career_record,
                                "smp": speed_map_pos,
                                "odds": current_odds,
                                "opening": opening_odds,
                                "scr": scratched,
                                "age": runner_data.get("Age"),
                                "sex": runner_data.get("Sex"),
                                "colour": runner_data.get("Colour"),
                                "sire": runner_data.get("Sire"),
                                "dam": runner_data.get("Dam"),
                                "damsire": runner_data.get("SireofDam"),
                                "prize": runner_data.get("PrizeMoney"),
                                "last5": last_five,
                                "hcap": runner_data.get("HandicapRating") or None,
                                "td_stats": track_dist_stats,
                                "t_stats": track_stats,
                                "d_stats": distance_stats,
                                "fu_stats": first_up_stats,
                                "su_stats": second_up_stats,
                                "good_stats": good_track_stats,
                                "soft_stats": soft_track_stats,
                                "heavy_stats": heavy_track_stats,
                                "trainer_loc": trainer_location,
                                "stewards": stewards,
                                "gear_ch": gear_changes,
                                "fp": finish_position,
                                "margin": str(result_margin) if result_margin is not None else None,
                                "sp": str(starting_price) if starting_price else None,
                                "wd": win_dividend,
                                "pd": place_dividend,
                                "ai_score": pf_ai_score if pf_ai_score else None,
                                "ai_price": pf_ai_price if pf_ai_price else None,
                                "ai_rank": pf_ai_rank if pf_ai_rank else None,
                                "now": datetime.now(timezone.utc).isoformat(),
                            })

                            month_runners += 1

                        # Update field_size on race
                        if field_size > 0:
                            await db.execute(text(
                                "UPDATE races SET field_size = :fs WHERE id = :id"
                            ), {"fs": field_size, "id": race_id})

                await db.commit()

        total_meetings += month_meetings
        total_races += month_races
        total_runners += month_runners
        logger.info(f"  {label}: {month_meetings} meetings, {month_races} races, {month_runners} runners")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"IMPORT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total meetings: {total_meetings}")
    logger.info(f"Total races: {total_races}")
    logger.info(f"Total runners: {total_runners}")
    logger.info(f"Skipped trials: {skipped_trials}")
    logger.info(f"Database: {BACKTEST_DB} ({BACKTEST_DB.stat().st_size / 1024 / 1024:.1f} MB)")

    # Verify with some counts
    async with session_factory() as db:
        result = await db.execute(text("SELECT COUNT(*) FROM meetings"))
        m_count = result.scalar()
        result = await db.execute(text("SELECT COUNT(*) FROM races"))
        r_count = result.scalar()
        result = await db.execute(text("SELECT COUNT(*) FROM runners"))
        run_count = result.scalar()
        result = await db.execute(text("SELECT COUNT(*) FROM runners WHERE finish_position IS NOT NULL"))
        settled_count = result.scalar()
        result = await db.execute(text("SELECT COUNT(*) FROM runners WHERE current_odds IS NOT NULL AND current_odds > 0"))
        odds_count = result.scalar()
        result = await db.execute(text("SELECT MIN(date), MAX(date) FROM meetings"))
        date_range = result.one()

    logger.info(f"\nVerification:")
    logger.info(f"  Meetings in DB: {m_count}")
    logger.info(f"  Races in DB: {r_count}")
    logger.info(f"  Runners in DB: {run_count}")
    logger.info(f"  With results: {settled_count}")
    logger.info(f"  With odds: {odds_count}")
    logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
