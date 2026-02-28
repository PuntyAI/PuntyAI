#!/usr/bin/env python3
"""ETL: Build analytics.duckdb from backtest.db + punty.db + Proform JSON.

Reads three data sources and creates a denormalized DuckDB database
optimized for fast analytical queries:

  1. backtest.db (SQLite) — 222K runners across 23K races
  2. punty.db (SQLite) — production picks and P&L
  3. Proform JSON (8.3GB) — speed maps (InRun) and A2E signals

Usage:
    python scripts/build_analytics.py                  # Full rebuild
    python scripts/build_analytics.py --picks-only     # Quick pick refresh
    python scripts/build_analytics.py --no-proform     # Skip Proform (faster)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb

from punty.venues import guess_state, normalize_venue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BACKTEST_DB = Path("data/backtest.db")
PUNTY_DB = Path("data/punty.db")
ANALYTICS_DB = Path("data/analytics.duckdb")
PROFORM_BASE = Path(r"D:\Punty\DatafromProform")

MONTH_DIRS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

DISTANCE_CATEGORIES = [
    (0, 1100, "sprint"),
    (1100, 1400, "short"),
    (1400, 1800, "middle"),
    (1800, 2200, "classic"),
    (2200, 99999, "staying"),
]

ODDS_BANDS = [
    (0, 2, "$1-$2"),
    (2, 4, "$2-$4"),
    (4, 6, "$4-$6"),
    (6, 10, "$6-$10"),
    (10, 20, "$10-$20"),
    (20, 50, "$20-$50"),
    (50, 9999, "$50+"),
]


def _distance_category(distance: int | None) -> str | None:
    if not distance:
        return None
    for lo, hi, cat in DISTANCE_CATEGORIES:
        if lo <= distance < hi:
            return cat
    return None


def _odds_band(odds: float | None) -> str | None:
    if not odds or odds <= 0:
        return None
    for lo, hi, label in ODDS_BANDS:
        if lo < odds <= hi:
            return label
    return "$50+" if odds > 50 else None


def _derive_settling_position(inrun: str, field_size: int) -> int | None:
    """Parse settling_down position from InRun string.

    Format: "settling_down,6;m800,6;m400,4;finish,3;"
    """
    if not inrun:
        return None
    for segment in inrun.split(";"):
        if segment.startswith("settling_down,"):
            try:
                return int(segment.split(",")[1])
            except (ValueError, IndexError):
                pass
            break
    return None


def _derive_finish_from_inrun(inrun: str) -> int | None:
    """Parse finish position from InRun string."""
    if not inrun:
        return None
    for segment in inrun.split(";"):
        if segment.startswith("finish,"):
            try:
                return int(segment.split(",")[1])
            except (ValueError, IndexError):
                pass
            break
    return None


def _speed_map_category(avg_settle: float, field_size: int) -> str | None:
    """Map average settle position to speed_map_position category."""
    if field_size <= 0:
        field_size = 12
    pct = avg_settle / field_size
    if pct <= 0.20:
        return "leader"
    elif pct <= 0.40:
        return "on_pace"
    elif pct <= 0.65:
        return "midfield"
    else:
        return "backmarker"


def _build_venue_state_table(conn: duckdb.DuckDBPyConnection, backtest_path: str):
    """Build a temporary venue→state lookup table using Python's guess_state().

    This runs guess_state() only once per unique venue (~400 venues) instead of
    per meeting (3K) or per runner (222K), making the ETL orders of magnitude faster.
    """
    venues = conn.execute(f"""
        SELECT DISTINCT venue FROM sqlite_scan('{backtest_path}', 'meetings')
        WHERE venue IS NOT NULL
    """).fetchall()

    venue_states = [(v, guess_state(v)) for (v,) in venues]
    conn.execute("CREATE OR REPLACE TABLE _venue_state (venue VARCHAR, state VARCHAR)")
    conn.executemany("INSERT INTO _venue_state VALUES ($1, $2)", venue_states)
    logger.info("  Venue→state lookup: %d venues", len(venue_states))


def build_core_tables(conn: duckdb.DuckDBPyConnection):
    """Load meetings, races, runners from backtest.db (or punty.db fallback).

    All computed columns (state, distance_category, odds_band) are done in
    pure SQL CASE expressions — no Python loops over 222K rows.

    When backtest.db doesn't exist, falls back to punty.db for core tables,
    filtered to meetings with approved/sent content (i.e. meetings we tipped on).
    """
    if BACKTEST_DB.exists():
        source_path = str(BACKTEST_DB.resolve()).replace("\\", "/")
        logger.info("Loading core tables from backtest.db...")
        content_filter = False
    elif PUNTY_DB.exists():
        source_path = str(PUNTY_DB.resolve()).replace("\\", "/")
        logger.info("backtest.db not found — using punty.db for core tables")
        content_filter = True
    else:
        logger.error("Neither backtest.db nor punty.db found")
        return

    backtest_path = source_path

    conn.execute("INSTALL sqlite; LOAD sqlite;")

    # Build venue→state lookup (fast: ~400 unique venues)
    _build_venue_state_table(conn, backtest_path)

    # Meetings — join with venue_state for state column
    # When using punty.db, filter to meetings with approved/sent content only
    content_join = ""
    if content_filter:
        content_join = f"""
        INNER JOIN (
            SELECT DISTINCT meeting_id
            FROM sqlite_scan('{backtest_path}', 'content')
            WHERE status IN ('approved', 'sent')
        ) c ON c.meeting_id = m.id"""

    conn.execute(f"""
    CREATE OR REPLACE TABLE meetings AS
    SELECT
        m.id,
        m.venue,
        m.date,
        m.track_condition,
        m.weather_condition,
        m.weather_temp,
        m.rail_position,
        m.meeting_type,
        COALESCE(vs.state, 'VIC') AS state
    FROM sqlite_scan('{backtest_path}', 'meetings') m
    LEFT JOIN _venue_state vs ON vs.venue = m.venue
    {content_join}
    """)
    meeting_count = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
    logger.info("  Meetings: %d", meeting_count)

    # Races — denormalized with meeting info + distance_category via SQL CASE
    conn.execute(f"""
    CREATE OR REPLACE TABLE races AS
    SELECT
        rc.id,
        rc.meeting_id,
        rc.race_number,
        rc.name AS race_name,
        rc.distance,
        rc."class" AS race_class,
        rc.prize_money,
        rc.field_size,
        rc.track_condition,
        rc.results_status,
        m.venue,
        m.state,
        m.date,
        CASE
            WHEN rc.distance < 1100 THEN 'sprint'
            WHEN rc.distance < 1400 THEN 'short'
            WHEN rc.distance < 1800 THEN 'middle'
            WHEN rc.distance < 2200 THEN 'classic'
            WHEN rc.distance >= 2200 THEN 'staying'
            ELSE NULL
        END AS distance_category
    FROM sqlite_scan('{backtest_path}', 'races') rc
    JOIN meetings m ON m.id = rc.meeting_id
    """)
    race_count = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    logger.info("  Races: %d", race_count)

    # Runners — heavily denormalized, all computed columns in SQL
    conn.execute(f"""
    CREATE OR REPLACE TABLE runners AS
    SELECT
        ru.id,
        ru.race_id,
        ru.horse_name,
        ru.saddlecloth,
        ru.barrier,
        ru.weight,
        ru.jockey,
        ru.trainer,
        ru.current_odds,
        ru.opening_odds,
        ru.place_odds,
        ru.finish_position,
        ru.win_dividend,
        ru.place_dividend,
        ru.speed_map_position,
        ru.horse_age,
        ru.horse_sex,
        ru.last_five,
        ru.days_since_last_run,
        ru.handicap_rating,
        ru.scratched,
        -- Denormalized from race/meeting
        rc.venue,
        rc.state,
        rc.date,
        rc.distance,
        rc.distance_category,
        rc.race_class,
        rc.field_size,
        rc.track_condition,
        rc.race_number,
        -- Computed: boolean flags
        CASE WHEN ru.finish_position = 1 AND (ru.scratched = false OR ru.scratched IS NULL)
             THEN true ELSE false END AS is_winner,
        CASE WHEN ru.finish_position <= 3 AND ru.finish_position > 0
             AND (ru.scratched = false OR ru.scratched IS NULL)
             THEN true ELSE false END AS is_placed,
        -- Computed: implied probability
        CASE WHEN ru.current_odds > 0 THEN ROUND(1.0 / ru.current_odds, 4) ELSE NULL END AS implied_prob,
        -- Computed: odds band
        CASE
            WHEN ru.current_odds IS NULL OR ru.current_odds <= 0 THEN NULL
            WHEN ru.current_odds <= 2 THEN '$1-$2'
            WHEN ru.current_odds <= 4 THEN '$2-$4'
            WHEN ru.current_odds <= 6 THEN '$4-$6'
            WHEN ru.current_odds <= 10 THEN '$6-$10'
            WHEN ru.current_odds <= 20 THEN '$10-$20'
            WHEN ru.current_odds <= 50 THEN '$20-$50'
            ELSE '$50+'
        END AS odds_band
    FROM sqlite_scan('{backtest_path}', 'runners') ru
    JOIN races rc ON rc.id = ru.race_id
    WHERE ru.scratched = false OR ru.scratched IS NULL
    """)
    runner_count = conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0]
    logger.info("  Runners: %d (non-scratched)", runner_count)

    # Drop temp table
    conn.execute("DROP TABLE IF EXISTS _venue_state")


def build_picks_table(conn: duckdb.DuckDBPyConnection):
    """Load production picks from punty.db."""
    logger.info("Loading picks from punty.db...")

    if not PUNTY_DB.exists():
        logger.warning("punty.db not found at %s — skipping picks", PUNTY_DB)
        conn.execute("""
        CREATE OR REPLACE TABLE picks (
            id VARCHAR, content_id VARCHAR, meeting_id VARCHAR, race_number INTEGER,
            horse_name VARCHAR, pick_type VARCHAR, bet_type VARCHAR, bet_stake FLOAT,
            odds_at_tip FLOAT, hit BOOLEAN, pnl FLOAT, settled BOOLEAN, settled_at TIMESTAMP,
            venue VARCHAR, state VARCHAR, date DATE
        )
        """)
        return

    punty_path = str(PUNTY_DB.resolve()).replace("\\", "/")

    # Schema may vary between local and production — try full query, fallback to minimal
    try:
        conn.execute(f"""
        CREATE OR REPLACE TABLE picks AS
        SELECT
            p.id,
            p.content_id,
            p.meeting_id,
            p.race_number,
            p.horse_name,
            p.pick_type,
            p.exotic_type,
            p.sequence_type,
            p.sequence_variant,
            p.bet_type,
            p.bet_stake,
            p.odds_at_tip,
            p.win_probability,
            p.place_probability,
            p.value_rating,
            p.confidence,
            p.is_puntys_pick,
            p.tracked_only,
            p.hit,
            p.pnl,
            p.settled,
            p.settled_at,
            p.created_at
        FROM sqlite_scan('{punty_path}', 'picks') p
        """)
    except Exception:
        # Fallback without newer columns
        logger.info("  Using fallback picks query (older schema)")
        conn.execute(f"""
        CREATE OR REPLACE TABLE picks AS
        SELECT
            p.id,
            p.content_id,
            p.meeting_id,
            p.race_number,
            p.horse_name,
            p.pick_type,
            p.bet_type,
            p.bet_stake,
            p.odds_at_tip,
            p.hit,
            p.pnl,
            p.settled,
            p.settled_at,
            p.created_at,
            NULL AS exotic_type,
            NULL AS sequence_type,
            NULL AS sequence_variant,
            NULL AS win_probability,
            NULL AS place_probability,
            NULL AS value_rating,
            NULL AS confidence,
            NULL AS is_puntys_pick,
            false AS tracked_only
        FROM sqlite_scan('{punty_path}', 'picks') p
        """)

    # Denormalize venue/state/date from meetings
    conn.execute("ALTER TABLE picks ADD COLUMN venue VARCHAR")
    conn.execute("ALTER TABLE picks ADD COLUMN state VARCHAR")
    conn.execute("ALTER TABLE picks ADD COLUMN date DATE")
    conn.execute(f"""
    UPDATE picks SET
        venue = m.venue,
        state = m.state,
        date = m.date
    FROM (
        SELECT id, venue, date,
               NULL AS state
        FROM sqlite_scan('{punty_path}', 'meetings')
    ) m
    WHERE picks.meeting_id = m.id
    """)

    # Fill state from venue
    pick_venues = conn.execute(
        "SELECT DISTINCT meeting_id, venue FROM picks WHERE venue IS NOT NULL"
    ).fetchall()
    for mid, venue in pick_venues:
        state = guess_state(venue)
        conn.execute("UPDATE picks SET state = $1 WHERE meeting_id = $2", [state, mid])

    pick_count = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
    settled_count = conn.execute("SELECT COUNT(*) FROM picks WHERE settled = true").fetchone()[0]
    logger.info("  Picks: %d (%d settled)", pick_count, settled_count)


def build_speed_maps(conn: duckdb.DuckDBPyConnection):
    """Parse InRun data from Proform Form JSON files into speed_maps table.

    Processes files one meeting at a time to avoid loading 8GB into memory.
    Each runner's past Forms[] contain InRun data with settle/finish positions.
    """
    logger.info("Building speed_maps from Proform Form data...")

    conn.execute("""
    CREATE OR REPLACE TABLE speed_maps (
        runner_id VARCHAR,
        race_id VARCHAR,
        horse_name VARCHAR,
        settle_position INTEGER,
        m800_position INTEGER,
        m400_position INTEGER,
        finish_position INTEGER,
        position_change INTEGER,
        speed_map_position VARCHAR,
        venue VARCHAR,
        date DATE
    )
    """)

    if not PROFORM_BASE.exists():
        logger.warning("Proform data not found at %s — speed_maps empty", PROFORM_BASE)
        return

    # Build runner_id lookup from runners table for joining
    existing_runners = set()
    for row in conn.execute("SELECT id FROM runners").fetchall():
        existing_runners.add(row[0])

    total_rows = 0
    files_processed = 0

    # Walk all year/month/Form directories
    for year_dir in sorted(PROFORM_BASE.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            form_dir = month_dir / "Form"
            if not form_dir.exists():
                continue

            for form_file in sorted(form_dir.glob("*.json")):
                try:
                    rows = _process_form_file(form_file, existing_runners)
                    if rows:
                        conn.executemany(
                            """INSERT INTO speed_maps VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                            rows,
                        )
                        total_rows += len(rows)
                    files_processed += 1
                    if files_processed % 100 == 0:
                        logger.info("  Processed %d files, %d speed map rows...",
                                    files_processed, total_rows)
                except Exception:
                    logger.exception("Error processing %s", form_file)
                    continue

    logger.info("  Speed maps: %d rows from %d files", total_rows, files_processed)


def _process_form_file(form_file: Path, existing_runners: set) -> list[tuple]:
    """Process a single Proform Form JSON file.

    Each file is a flat list of runners for one meeting. We extract InRun
    data from their Forms[] to build speed map rows.
    """
    # Parse venue and date from filename: YYMMDD_Venue.json
    fname = form_file.stem  # e.g. "250101_Ascot"
    parts = fname.split("_", 1)
    if len(parts) != 2:
        return []

    date_str = parts[0]
    venue_name = parts[1]

    try:
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        date_val = f"{year}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return []

    venue_norm = normalize_venue(venue_name)

    with open(form_file, "r", encoding="utf-8") as f:
        runners = json.load(f)

    if not isinstance(runners, list):
        return []

    rows = []
    for runner in runners:
        runner_id = str(runner.get("RunnerId", ""))
        race_id = str(runner.get("RaceId", ""))
        horse_name = runner.get("Name", "")

        forms = runner.get("Forms", [])
        if not forms:
            continue

        # Use the CURRENT race's InRun if this runner has a Position (finished)
        # But InRun is in Forms[] (past starts). The current race data is at top level.
        # We want InRun from the current race's form entry matching this RaceId
        # Actually, Forms[] is PAST form. The current race position/margin is at top level.
        # For speed map analysis, we want the InRun from past starts to build running style.

        # Aggregate settling positions from recent forms
        settle_positions = []
        m800_positions = []
        m400_positions = []
        finish_positions_inrun = []

        for form in forms[:10]:  # Last 10 starts
            inrun = form.get("InRun", "")
            if not inrun:
                continue
            for segment in inrun.split(";"):
                if not segment:
                    continue
                key_val = segment.split(",")
                if len(key_val) != 2:
                    continue
                key, val_str = key_val
                try:
                    val = int(val_str)
                except ValueError:
                    continue

                if key == "settling_down":
                    settle_positions.append(val)
                elif key == "m800":
                    m800_positions.append(val)
                elif key == "m400":
                    m400_positions.append(val)
                elif key == "finish":
                    finish_positions_inrun.append(val)

        if not settle_positions:
            continue

        avg_settle = round(sum(settle_positions) / len(settle_positions))
        avg_m800 = round(sum(m800_positions) / len(m800_positions)) if m800_positions else None
        avg_m400 = round(sum(m400_positions) / len(m400_positions)) if m400_positions else None
        avg_finish = round(sum(finish_positions_inrun) / len(finish_positions_inrun)) if finish_positions_inrun else None

        pos_change = (avg_settle - avg_finish) if avg_finish else None

        # Derive field_size from number of runners in this file with same RaceId
        field_size = sum(1 for r in runners if str(r.get("RaceId", "")) == race_id)
        smp = _speed_map_category(avg_settle, field_size) if avg_settle else None

        rows.append((
            runner_id, race_id, horse_name,
            avg_settle, avg_m800, avg_m400, avg_finish,
            pos_change, smp,
            venue_norm, date_val,
        ))

    return rows


def build_a2e_table(conn: duckdb.DuckDBPyConnection):
    """Extract A2E signals from Proform Form JSON into proform_a2e table.

    Processes files one at a time. Deduplicates by jockey+trainer combo.
    """
    logger.info("Building proform_a2e from Proform data...")

    conn.execute("""
    CREATE OR REPLACE TABLE proform_a2e (
        jockey VARCHAR,
        trainer VARCHAR,
        trainer_a2e_career FLOAT,
        trainer_pot_career FLOAT,
        trainer_strike_career FLOAT,
        trainer_runners_career INTEGER,
        jockey_a2e_career FLOAT,
        jockey_pot_career FLOAT,
        jockey_strike_career FLOAT,
        jockey_runners_career INTEGER,
        combo_a2e_career FLOAT,
        combo_pot_career FLOAT,
        combo_strike_career FLOAT,
        combo_runners_career INTEGER,
        trainer_a2e_last100 FLOAT,
        jockey_a2e_last100 FLOAT,
        combo_a2e_last100 FLOAT
    )
    """)

    if not PROFORM_BASE.exists():
        logger.warning("Proform data not found — proform_a2e empty")
        return

    # Collect best (most recent) A2E per jockey+trainer combo
    combos: dict[tuple[str, str], dict] = {}
    files_processed = 0

    for year_dir in sorted(PROFORM_BASE.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue
            form_dir = month_dir / "Form"
            if not form_dir.exists():
                continue

            for form_file in sorted(form_dir.glob("*.json"), reverse=True):
                try:
                    with open(form_file, "r", encoding="utf-8") as f:
                        runners = json.load(f)
                    if not isinstance(runners, list):
                        continue

                    for runner in runners:
                        jockey = runner.get("Jockey", {}).get("FullName", "")
                        trainer = runner.get("Trainer", {}).get("FullName", "")
                        if not jockey or not trainer:
                            continue

                        key = (jockey, trainer)
                        if key in combos:
                            continue  # Already have most recent

                        ta = runner.get("TrainerA2E_Career", {}) or {}
                        ja = runner.get("JockeyA2E_Career", {}) or {}
                        ca = runner.get("TrainerJockeyA2E_Career", {}) or {}
                        ta100 = runner.get("TrainerA2E_Last100", {}) or {}
                        ja100 = runner.get("JockeyA2E_Last100", {}) or {}
                        ca100 = runner.get("TrainerJockeyA2E_Last100", {}) or {}

                        # Skip zero-data combos
                        if not ca.get("Runners"):
                            continue

                        combos[key] = {
                            "jockey": jockey,
                            "trainer": trainer,
                            "trainer_a2e_career": ta.get("A2E", 0),
                            "trainer_pot_career": ta.get("PoT", 0),
                            "trainer_strike_career": ta.get("StrikeRate", 0),
                            "trainer_runners_career": ta.get("Runners", 0),
                            "jockey_a2e_career": ja.get("A2E", 0),
                            "jockey_pot_career": ja.get("PoT", 0),
                            "jockey_strike_career": ja.get("StrikeRate", 0),
                            "jockey_runners_career": ja.get("Runners", 0),
                            "combo_a2e_career": ca.get("A2E", 0),
                            "combo_pot_career": ca.get("PoT", 0),
                            "combo_strike_career": ca.get("StrikeRate", 0),
                            "combo_runners_career": ca.get("Runners", 0),
                            "trainer_a2e_last100": ta100.get("A2E", 0),
                            "jockey_a2e_last100": ja100.get("A2E", 0),
                            "combo_a2e_last100": ca100.get("A2E", 0),
                        }

                    files_processed += 1
                    if files_processed % 100 == 0:
                        logger.info("  Processed %d files, %d combos...",
                                    files_processed, len(combos))
                except Exception:
                    logger.exception("Error in %s", form_file)
                    continue

    # Insert all combos
    if combos:
        rows = [
            (
                d["jockey"], d["trainer"],
                d["trainer_a2e_career"], d["trainer_pot_career"],
                d["trainer_strike_career"], d["trainer_runners_career"],
                d["jockey_a2e_career"], d["jockey_pot_career"],
                d["jockey_strike_career"], d["jockey_runners_career"],
                d["combo_a2e_career"], d["combo_pot_career"],
                d["combo_strike_career"], d["combo_runners_career"],
                d["trainer_a2e_last100"], d["jockey_a2e_last100"],
                d["combo_a2e_last100"],
            )
            for d in combos.values()
        ]
        conn.executemany(
            "INSERT INTO proform_a2e VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)",
            rows,
        )

    logger.info("  A2E combos: %d from %d files", len(combos), files_processed)


def create_indexes(conn: duckdb.DuckDBPyConnection):
    """Create indexes on all filter columns for fast queries."""
    logger.info("Creating indexes...")

    indexes = [
        ("idx_runners_venue", "runners", "venue"),
        ("idx_runners_state", "runners", "state"),
        ("idx_runners_date", "runners", "date"),
        ("idx_runners_odds_band", "runners", "odds_band"),
        ("idx_runners_distance_cat", "runners", "distance_category"),
        ("idx_runners_track_cond", "runners", "track_condition"),
        ("idx_runners_smp", "runners", "speed_map_position"),
        ("idx_runners_race_id", "runners", "race_id"),
        ("idx_runners_is_winner", "runners", "is_winner"),
        ("idx_races_meeting_id", "races", "meeting_id"),
        ("idx_races_venue", "races", "venue"),
        ("idx_speed_maps_runner", "speed_maps", "runner_id"),
        ("idx_speed_maps_venue", "speed_maps", "venue"),
        ("idx_picks_settled", "picks", "settled"),
        ("idx_picks_bet_type", "picks", "bet_type"),
        ("idx_a2e_combo", "proform_a2e", "combo_a2e_career"),
    ]

    for idx_name, table, column in indexes:
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
        except Exception as e:
            logger.warning("Index %s failed: %s", idx_name, e)

    conn.execute("ANALYZE")
    logger.info("  Indexes created and statistics updated")


def main():
    parser = argparse.ArgumentParser(description="Build analytics DuckDB")
    parser.add_argument("--picks-only", action="store_true",
                        help="Only refresh the picks table (fast)")
    parser.add_argument("--no-proform", action="store_true",
                        help="Skip Proform JSON processing (faster)")
    args = parser.parse_args()

    start = time.time()

    # Remove existing DB for full rebuild
    if not args.picks_only and ANALYTICS_DB.exists():
        ANALYTICS_DB.unlink()
        logger.info("Removed existing %s", ANALYTICS_DB)

    ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(ANALYTICS_DB))

    try:
        if args.picks_only:
            build_picks_table(conn)
        else:
            build_core_tables(conn)
            build_picks_table(conn)

            if not args.no_proform:
                build_speed_maps(conn)
                build_a2e_table(conn)
            else:
                # Create empty tables so queries don't fail
                conn.execute("""
                CREATE OR REPLACE TABLE speed_maps (
                    runner_id VARCHAR, race_id VARCHAR, horse_name VARCHAR,
                    settle_position INTEGER, m800_position INTEGER,
                    m400_position INTEGER, finish_position INTEGER,
                    position_change INTEGER, speed_map_position VARCHAR,
                    venue VARCHAR, date DATE
                )""")
                conn.execute("""
                CREATE OR REPLACE TABLE proform_a2e (
                    jockey VARCHAR, trainer VARCHAR,
                    trainer_a2e_career FLOAT, trainer_pot_career FLOAT,
                    trainer_strike_career FLOAT, trainer_runners_career INTEGER,
                    jockey_a2e_career FLOAT, jockey_pot_career FLOAT,
                    jockey_strike_career FLOAT, jockey_runners_career INTEGER,
                    combo_a2e_career FLOAT, combo_pot_career FLOAT,
                    combo_strike_career FLOAT, combo_runners_career INTEGER,
                    trainer_a2e_last100 FLOAT, jockey_a2e_last100 FLOAT,
                    combo_a2e_last100 FLOAT
                )""")

            create_indexes(conn)

        # Print summary
        for table in ["meetings", "races", "runners", "picks", "speed_maps", "proform_a2e"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                logger.info("Table %-15s: %s rows", table, f"{count:,}")
            except Exception:
                pass

        db_size = ANALYTICS_DB.stat().st_size / (1024 * 1024)
        elapsed = time.time() - start
        logger.info("Done in %.1fs. Database: %.1f MB", elapsed, db_size)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
