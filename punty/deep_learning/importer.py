"""Import Proform historical data into the deep learning SQLite DB.

Usage:
    python -m punty.deep_learning.importer --data-dir "C:\\path\\to\\DatafromProform\\2026"

Processes Form files from monthly subdirectories (January/Form/, etc.)
and Sectionals from the top-level Sectionals/ directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    HistoricalRace,
    HistoricalRunner,
    HistoricalSectional,
    get_engine,
    init_db,
    get_session,
)

logger = logging.getLogger(__name__)

# Venue-to-state mapping for Australian tracks
VENUE_STATE_MAP: dict[str, str] = {
    # NSW
    "randwick": "NSW", "rosehill": "NSW", "warwick farm": "NSW",
    "canterbury": "NSW", "newcastle": "NSW", "kembla grange": "NSW",
    "wyong": "NSW", "gosford": "NSW", "hawkesbury": "NSW",
    "scone": "NSW", "tamworth": "NSW", "mudgee": "NSW",
    "dubbo": "NSW", "bathurst": "NSW", "orange": "NSW",
    "wagga": "NSW", "albury": "NSW", "canberra": "ACT",
    "queanbeyan": "NSW", "moruya": "NSW", "nowra": "NSW",
    "port macquarie": "NSW", "coffs harbour": "NSW", "grafton": "NSW",
    "ballina": "NSW", "lismore": "NSW", "taree": "NSW",
    "muswellbrook": "NSW", "cessnock": "NSW", "glen innes": "NSW",
    "gilgandra": "NSW", "inverell": "NSW", "gundagai": "NSW",
    "bowraville": "NSW", "quirindi": "NSW", "coonamble": "NSW",
    "wellington": "NSW", "cowra": "NSW", "young": "NSW",
    "sapphire coast": "NSW", "moree": "NSW", "narromine": "NSW",
    "armidale": "NSW", "goulburn": "NSW", "broken hill": "NSW",
    "tuncurry": "NSW", "corowa": "NSW", "casino": "NSW",
    # VIC
    "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC",
    "sandown": "VIC", "cranbourne": "VIC", "pakenham": "VIC",
    "mornington": "VIC", "geelong": "VIC", "ballarat": "VIC",
    "bendigo": "VIC", "sale": "VIC", "wangaratta": "VIC",
    "wodonga": "VIC", "hamilton": "VIC", "stawell": "VIC",
    "werribee": "VIC", "kilmore": "VIC", "yarra glen": "VIC",
    "stony creek": "VIC", "healesville": "VIC", "woolamai": "VIC",
    "seymour": "VIC", "donald": "VIC", "echuca": "VIC",
    "terang": "VIC", "warrnambool": "VIC", "merton": "VIC",
    "hanging rock": "VIC", "burrumbeet": "VIC", "kyneton": "VIC",
    "colac": "VIC", "swan hill": "VIC", "mildura": "VIC",
    "tatura": "VIC", "ararat": "VIC", "avoca": "VIC",
    "caulfield heath": "VIC", "sportsbet pakenham": "VIC",
    # QLD
    "eagle farm": "QLD", "doomben": "QLD", "gold coast": "QLD",
    "sunshine coast": "QLD", "ipswich": "QLD", "toowoomba": "QLD",
    "rockhampton": "QLD", "mackay": "QLD", "townsville": "QLD",
    "cairns": "QLD", "atherton": "QLD", "roma": "QLD",
    "dalby": "QLD", "beaudesert": "QLD", "kilcoy": "QLD",
    "bundaberg": "QLD", "gladstone": "QLD", "emerald": "QLD",
    "longreach": "QLD", "mt isa": "QLD", "yeppoon": "QLD",
    "callaghan park": "QLD", "bell": "QLD",
    # SA
    "morphettville": "SA", "murray bridge": "SA", "gawler": "SA",
    "oakbank": "SA", "naracoorte": "SA", "port augusta": "SA",
    "balaklava": "SA", "murray bridge gh": "SA", "clare": "SA",
    "strathalbyn": "SA", "mount gambier": "SA", "port lincoln": "SA",
    "penola": "SA", "ceduna": "SA", "bordertown": "SA",
    # WA
    "ascot": "WA", "belmont": "WA", "pinjarra": "WA",
    "bunbury": "WA", "northam": "WA", "geraldton": "WA",
    "kalgoorlie": "WA", "albany": "WA", "esperance": "WA",
    "narrogin": "WA", "york": "WA", "lark hill": "WA",
    "carnarvon": "WA", "broome": "WA",
    # TAS
    "hobart": "TAS", "launceston": "TAS", "devonport": "TAS",
    "devonport synthetic": "TAS", "longford": "TAS", "king island": "TAS",
    # NT
    "fannie bay": "NT", "alice springs": "NT", "darwin": "NT",
    "pioneer park": "NT",
    # NZ
    "ellerslie": "NZ", "trentham": "NZ", "riccarton": "NZ",
    "otaki": "NZ", "te aroha": "NZ", "tauranga": "NZ",
    "ruakaka": "NZ", "hastings": "NZ", "awapuni": "NZ",
    "tauherenikau": "NZ", "riverton": "NZ", "greymouth": "NZ",
    "kumara": "NZ", "reefton": "NZ", "new plymouth": "NZ",
    # International
    "sha tin": "HK", "happy valley": "HK", "bahrain": "INT",
}


def _parse_filename(filename: str) -> tuple[date | None, str]:
    """Parse YYMMDD_Venue_Name.json into (date, venue_name)."""
    stem = Path(filename).stem  # e.g. "250101_Flemington"
    match = re.match(r"^(\d{6})_(.+)$", stem)
    if not match:
        return None, stem
    date_str, venue_raw = match.groups()
    try:
        meeting_date = datetime.strptime(date_str, "%y%m%d").date()
    except ValueError:
        return None, venue_raw
    venue = venue_raw.replace("_", " ")
    return meeting_date, venue


def _resolve_state(venue: str, track_data: dict | None = None) -> str:
    """Resolve venue to state. Uses track data from Form if available."""
    if track_data and track_data.get("State"):
        return track_data["State"]
    venue_lower = venue.lower().strip()
    # Strip sponsor prefixes
    for prefix in ("sportsbet ", "ladbrokes ", "bet365 ", "picklebet park ",
                   "southside ", "tab ", "aquis ", "neds "):
        if venue_lower.startswith(prefix):
            venue_lower = venue_lower[len(prefix):]
            break
    # Strip "park" leftover
    if venue_lower.startswith("park "):
        venue_lower = venue_lower[5:]
    return VENUE_STATE_MAP.get(venue_lower, "")


def _record_to_json(record: dict | None) -> str | None:
    """Convert record dict to JSON string, or None if empty."""
    if not record or not record.get("Starts"):
        return None
    return json.dumps(record)


def _parse_in_run(in_run: str | None) -> dict:
    """Parse InRun string: 'settling_down,2;m800,3;m400,2;finish,1;'"""
    result = {"settle": None, "m800": None, "m400": None}
    if not in_run:
        return result
    for part in in_run.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(",")
        if len(pieces) != 2:
            continue
        key, val = pieces[0].strip(), pieces[1].strip()
        try:
            pos = int(val)
        except (ValueError, TypeError):
            continue
        if key == "settling_down":
            result["settle"] = pos
        elif key == "m800":
            result["m800"] = pos
        elif key == "m400":
            result["m400"] = pos
    return result


def _parse_flucs(flucs: str | None) -> dict:
    """Parse flucs string: 'opening,2.40;mid,2.30;starting,2.60;'"""
    result = {"opening": None, "mid": None, "starting": None}
    if not flucs:
        return result
    for part in flucs.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(",")
        if len(pieces) != 2:
            continue
        key, val = pieces[0].strip(), pieces[1].strip()
        try:
            price = float(val)
        except (ValueError, TypeError):
            continue
        if key in result:
            result[key] = price
    return result


def _parse_time_to_secs(time_str: str | None) -> float | None:
    """Parse 'HH:MM:SS.nnnnnnn' to seconds."""
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        return float(time_str)
    except (ValueError, TypeError):
        return None


def _build_form_history(forms: list[dict]) -> str | None:
    """Build compact JSON summary of past form entries."""
    if not forms:
        return None
    history = []
    for f in forms[:10]:
        track = f.get("Track", {})
        entry = {
            "date": f.get("MeetingDate", "")[:10],
            "venue": track.get("Name", "") if isinstance(track, dict) else "",
            "state": track.get("State", "") if isinstance(track, dict) else "",
            "dist": f.get("Distance"),
            "class": f.get("RaceClass"),
            "cond": f.get("TrackCondition"),
            "pos": f.get("Position"),
            "margin": f.get("Margin"),
            "sp": f.get("PriceSP"),
            "starters": f.get("Starters"),
            "in_run": f.get("InRun"),
            "time": f.get("OfficialRaceTime"),
            "prep": f.get("PrepRuns"),
            "weight": f.get("Weight"),
            "barrier": f.get("Barrier"),
            "prize_money": f.get("PrizeMoney"),
        }
        history.append(entry)
    return json.dumps(history)


def import_form_file(
    session, filepath: Path, meeting_date: date, venue: str
) -> dict:
    """Import a single Form JSON file. Returns stats dict."""
    with open(filepath, encoding="utf-8") as f:
        runners_data = json.load(f)

    if not isinstance(runners_data, list):
        return {"runners": 0, "races": 0, "skipped": True}

    # Group runners by race_id
    races: dict[int, list[dict]] = defaultdict(list)
    for r in runners_data:
        race_id = r.get("RaceId")
        if race_id:
            races[race_id].append(r)

    stats = {"runners": 0, "races": 0, "skipped": False}

    for race_id, race_runners in races.items():
        # Check if already imported
        existing = session.execute(
            select(HistoricalRace).where(HistoricalRace.race_id == race_id)
        ).scalar_one_or_none()
        if existing:
            continue

        # Get state from first runner's form track data
        first_runner = race_runners[0]
        forms = first_runner.get("Forms", [])
        track_data = None
        if forms:
            last_form = forms[0]
            if isinstance(last_form.get("Track"), dict):
                track_data = last_form["Track"]

        state = _resolve_state(venue, track_data)
        location_type = track_data.get("Location", "") if track_data else ""

        # Determine race number from runners (Form files don't have it directly)
        # We'll use the race ordering within the file
        race_numbers = sorted(races.keys())
        race_num = race_numbers.index(race_id) + 1

        # Create race record
        race = HistoricalRace(
            race_id=race_id,
            meeting_date=meeting_date,
            venue=venue,
            state=state,
            country=first_runner.get("Country", "AUS"),
            location_type=location_type,
            race_number=race_num,
            distance=forms[0].get("Distance") if forms else None,
            race_class=forms[0].get("RaceClass") if forms else None,
            track_condition=forms[0].get("TrackCondition") if forms else None,
            track_condition_number=forms[0].get("TrackConditionNumber") if forms else None,
            field_size=len(race_runners),
            prize_money=forms[0].get("PrizeMoney") if forms else None,
            rail_position=forms[0].get("Rail") if forms else None,
            official_time=forms[0].get("OfficialRaceTime") if forms else None,
            official_time_secs=_parse_time_to_secs(
                forms[0].get("OfficialRaceTime") if forms else None
            ),
            age_restriction=forms[0].get("AgeRestrictions") if forms else None,
            sex_restriction=forms[0].get("SexRestrictions") if forms else None,
        )
        session.add(race)
        session.flush()  # Get race.id

        stats["races"] += 1

        for r in race_runners:
            forms_data = r.get("Forms", [])
            first_form = forms_data[0] if forms_data else {}
            in_run = _parse_in_run(first_form.get("InRun"))
            flucs = _parse_flucs(first_form.get("Flucs"))
            position = r.get("Position")

            # Derive prep_runs: runner-level is always 0, use first form entry
            prep_runs = r.get("PrepRuns") or first_form.get("PrepRuns") or 0

            runner = HistoricalRunner(
                race_fk=race.id,
                race_id=race_id,
                form_id=r.get("FormId"),
                runner_id=r.get("RunnerId"),
                horse_name=r.get("Name", ""),
                tab_no=r.get("TabNo"),
                barrier=r.get("Barrier"),
                original_barrier=r.get("OriginalBarrier"),
                weight=r.get("Weight"),
                age=r.get("Age"),
                sex=r.get("Sex"),
                country=r.get("Country"),
                jockey=r.get("Jockey", {}).get("FullName") if isinstance(r.get("Jockey"), dict) else None,
                jockey_id=r.get("Jockey", {}).get("JockeyId") if isinstance(r.get("Jockey"), dict) else None,
                jockey_claim=r.get("JockeyClaim"),
                trainer=r.get("Trainer", {}).get("FullName") if isinstance(r.get("Trainer"), dict) else None,
                trainer_id=r.get("Trainer", {}).get("TrainerId") if isinstance(r.get("Trainer"), dict) else None,
                career_starts=r.get("CareerStarts"),
                career_wins=r.get("CareerWins"),
                career_seconds=r.get("CareerSeconds"),
                career_thirds=r.get("CareerThirds"),
                win_pct=r.get("WinPct"),
                place_pct=r.get("PlacePct"),
                prize_money=r.get("PrizeMoney"),
                handicap_rating=r.get("HandicapRating"),
                last_10=r.get("Last10", "").strip() if r.get("Last10") else None,
                prep_runs=prep_runs,
                track_record=_record_to_json(r.get("TrackRecord")),
                distance_record=_record_to_json(r.get("DistanceRecord")),
                track_dist_record=_record_to_json(r.get("TrackDistRecord")),
                first_up_record=_record_to_json(r.get("FirstUpRecord")),
                second_up_record=_record_to_json(r.get("SecondUpRecord")),
                good_record=_record_to_json(r.get("GoodRecord")),
                soft_record=_record_to_json(r.get("SoftRecord")),
                heavy_record=_record_to_json(r.get("HeavyRecord")),
                firm_record=_record_to_json(r.get("FirmRecord")),
                synthetic_record=_record_to_json(r.get("SyntheticRecord")),
                group1_record=_record_to_json(r.get("Group1Record")),
                group2_record=_record_to_json(r.get("Group2Record")),
                group3_record=_record_to_json(r.get("Group3Record")),
                jockey_a2e_career=_record_to_json(r.get("JockeyA2E_Career")),
                jockey_a2e_last100=_record_to_json(r.get("JockeyA2E_Last100")),
                trainer_a2e_career=_record_to_json(r.get("TrainerA2E_Career")),
                trainer_a2e_last100=_record_to_json(r.get("TrainerA2E_Last100")),
                trainer_jockey_a2e_career=_record_to_json(r.get("TrainerJockeyA2E_Career")),
                trainer_jockey_a2e_last100=_record_to_json(r.get("TrainerJockeyA2E_Last100")),
                form_history=_build_form_history(r.get("Forms", [])),
                starting_price=r.get("PriceSP"),
                opening_odds=flucs["opening"],
                mid_odds=flucs["mid"],
                settle_pos=in_run["settle"],
                pos_800=in_run["m800"],
                pos_400=in_run["m400"],
                finish_position=position,
                margin=r.get("Margin"),
                won=position == 1 if position else False,
                placed=position is not None and 1 <= position <= 3,
            )
            session.add(runner)
            stats["runners"] += 1

    return stats


def import_sectionals_file(session, filepath: Path) -> dict:
    """Import a single Sectionals JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    payload = data.get("payLoad", data)
    if not isinstance(payload, list):
        return {"sectionals": 0}

    stats = {"sectionals": 0, "races_updated": 0}

    for race_data in payload:
        race_id = race_data.get("raceId")
        if not race_id:
            continue

        # Update race-level sectional data
        race = session.execute(
            select(HistoricalRace).where(HistoricalRace.race_id == race_id)
        ).scalar_one_or_none()
        if race:
            race.time_to_finish = race_data.get("timeToFinish")
            race.time_to_1200 = race_data.get("timeTo1200")
            race.time_to_1000 = race_data.get("timeTo1000")
            race.time_to_800 = race_data.get("timeTo800")
            race.time_to_600 = race_data.get("timeTo600")
            race.time_to_400 = race_data.get("timeTo400")
            race.time_to_200 = race_data.get("timeTo200")
            race.last_600 = race_data.get("last600Time")
            race.last_400 = race_data.get("last400Time")
            race.last_200 = race_data.get("last200Time")
            race.wind_direction = race_data.get("windDirection")
            race.wind_speed = race_data.get("windSpeed")
            stats["races_updated"] += 1

        # Import per-runner sectionals
        for rs in race_data.get("runnerSectionals", []):
            # Check for existing
            form_id = rs.get("formId")
            if form_id:
                existing = session.execute(
                    select(HistoricalSectional).where(
                        HistoricalSectional.form_id == form_id
                    )
                ).scalar_one_or_none()
                if existing:
                    continue

            sect = HistoricalSectional(
                race_fk=race.id if race else 0,
                race_id=race_id,
                form_id=form_id,
                runner_id=rs.get("runnerId"),
                tab_no=rs.get("tabNumber"),
                runner_name=rs.get("runnerName"),
                time_to_1200=rs.get("timeTo1200"),
                time_to_1000=rs.get("timeTo1000"),
                time_to_800=rs.get("timeTo800"),
                time_to_600=rs.get("timeTo600"),
                time_to_400=rs.get("timeTo400"),
                time_to_200=rs.get("timeTo200"),
                time_to_100=rs.get("timeTo100"),
                time_to_fin=rs.get("timeToFin"),
                last_1200=rs.get("last1200Time"),
                last_1000=rs.get("last1000Time"),
                last_800=rs.get("last800Time"),
                last_600=rs.get("last600Time"),
                last_400=rs.get("last400Time"),
                last_200=rs.get("last200Time"),
                last_100=rs.get("last100Time"),
                pos_1200=rs.get("pos1200"),
                pos_1000=rs.get("pos1000"),
                pos_800=rs.get("pos800"),
                pos_600=rs.get("pos600"),
                pos_400=rs.get("pos400"),
                pos_200=rs.get("pos200"),
                pos_100=rs.get("pos100"),
                pos_fin=rs.get("posFin"),
                marg_1200=rs.get("marg1200"),
                marg_1000=rs.get("marg1000"),
                marg_800=rs.get("marg800"),
                marg_600=rs.get("marg600"),
                marg_400=rs.get("marg400"),
                marg_200=rs.get("marg200"),
                marg_100=rs.get("marg100"),
                marg_fin=rs.get("margFin"),
                wides_800=rs.get("wides800"),
                wides_600=rs.get("wides600"),
                wides_400=rs.get("wides400"),
                wides_200=rs.get("wides200"),
                wides_fin=rs.get("widesFin"),
                meeting_rank_6f=rs.get("meetingRank6F"),
                meeting_rank_4f=rs.get("meetingRank4F"),
                meeting_rank_2f=rs.get("meetingRank2F"),
                meeting_rank_1f=rs.get("meetingRank1F"),
            )
            session.add(sect)
            stats["sectionals"] += 1

    return stats


def import_all(data_dir: str | Path, db_path: str | Path | None = None):
    """Import all Proform data from the given directory.

    Walks monthly subdirectories for Form data, uses top-level for Sectionals.
    Skips already-imported races (idempotent).
    """
    data_dir = Path(data_dir)
    engine = init_db(db_path)
    SessionFactory = sessionmaker(bind=engine)

    # Collect all Form files from monthly subdirs
    form_files: list[tuple[Path, date, str]] = []
    seen_filenames: set[str] = set()

    # Monthly directories first (authoritative)
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    for month in months:
        form_dir = data_dir / month / "Form"
        if not form_dir.is_dir():
            continue
        for fn in sorted(form_dir.iterdir()):
            if not fn.suffix == ".json":
                continue
            meeting_date, venue = _parse_filename(fn.name)
            if meeting_date and fn.name not in seen_filenames:
                form_files.append((fn, meeting_date, venue))
                seen_filenames.add(fn.name)

    # Top-level Form/ as fallback (for files not in monthly dirs)
    top_form = data_dir / "Form"
    if top_form.is_dir():
        for fn in sorted(top_form.iterdir()):
            if not fn.suffix == ".json":
                continue
            if fn.name in seen_filenames:
                continue
            meeting_date, venue = _parse_filename(fn.name)
            if meeting_date:
                form_files.append((fn, meeting_date, venue))
                seen_filenames.add(fn.name)

    print(f"Found {len(form_files)} Form files to import")

    # Import Form files in batches
    total_runners = 0
    total_races = 0
    batch_size = 50

    for i, (filepath, meeting_date, venue) in enumerate(form_files):
        session = SessionFactory()
        try:
            stats = import_form_file(session, filepath, meeting_date, venue)
            session.commit()
            total_runners += stats["runners"]
            total_races += stats["races"]
            if (i + 1) % batch_size == 0:
                print(
                    f"  [{i + 1}/{len(form_files)}] "
                    f"{total_races} races, {total_runners} runners"
                )
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to import {filepath.name}: {e}")
            print(f"  ERROR: {filepath.name}: {e}")
        finally:
            session.close()

    print(f"Form import complete: {total_races} races, {total_runners} runners")

    # Import Sectionals
    sect_dir = data_dir / "Sectionals"
    if sect_dir.is_dir():
        sect_files = sorted(sect_dir.glob("*.json"))
        print(f"Found {len(sect_files)} Sectionals files to import")
        total_sectionals = 0
        total_updated = 0

        for i, filepath in enumerate(sect_files):
            session = SessionFactory()
            try:
                stats = import_sectionals_file(session, filepath)
                session.commit()
                total_sectionals += stats["sectionals"]
                total_updated += stats["races_updated"]
                if (i + 1) % batch_size == 0:
                    print(
                        f"  [{i + 1}/{len(sect_files)}] "
                        f"{total_sectionals} sectionals, {total_updated} races updated"
                    )
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to import sectionals {filepath.name}: {e}")
                print(f"  ERROR: {filepath.name}: {e}")
            finally:
                session.close()

        print(
            f"Sectionals import complete: "
            f"{total_sectionals} records, {total_updated} races updated"
        )

    # Final summary
    session = SessionFactory()
    race_count = session.query(HistoricalRace).count()
    runner_count = session.query(HistoricalRunner).count()
    sect_count = session.query(HistoricalSectional).count()
    session.close()

    print(f"\n=== Import Summary ===")
    print(f"Races:      {race_count:,}")
    print(f"Runners:    {runner_count:,}")
    print(f"Sectionals: {sect_count:,}")
    print(f"DB path:    {Path(db_path) if db_path else 'data/deep_learning.db'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import Proform data")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to DatafromProform/2026 directory",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to deep_learning.db (default: data/deep_learning.db)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    import_all(args.data_dir, args.db_path)
