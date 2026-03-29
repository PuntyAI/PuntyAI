#!/usr/bin/env python3
"""Fill the picks table in analytics.duckdb from production punty.db.

Run on the server after uploading analytics.duckdb:
    cd /opt/puntyai && source venv/bin/activate
    python scripts/fill_analytics_picks.py

This keeps the core tables (runners/races from backtest) intact and only
populates the picks table from the live production database.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
from punty.venues import guess_state

PUNTY_DB = Path("data/punty.db")
ANALYTICS_DB = Path("data/analytics.duckdb")


def main():
    if not ANALYTICS_DB.exists():
        print(f"ERROR: {ANALYTICS_DB} not found. Upload it first.")
        sys.exit(1)
    if not PUNTY_DB.exists():
        print(f"ERROR: {PUNTY_DB} not found.")
        sys.exit(1)

    conn = duckdb.connect(str(ANALYTICS_DB))
    conn.execute("INSTALL sqlite; LOAD sqlite;")

    pp = str(PUNTY_DB.resolve()).replace("\\", "/")

    # Drop and recreate picks
    conn.execute("DROP TABLE IF EXISTS picks")

    try:
        conn.execute(f"""
        CREATE TABLE picks AS
        SELECT
            p.id, p.content_id, p.meeting_id, p.race_number, p.horse_name,
            p.pick_type, p.exotic_type, p.sequence_type, p.sequence_variant,
            p.bet_type, p.bet_stake, p.odds_at_tip,
            p.win_probability, p.place_probability, p.value_rating,
            p.confidence, p.is_puntys_pick, p.tracked_only,
            p.hit, p.pnl, p.settled, p.settled_at, p.created_at
        FROM sqlite_scan('{pp}', 'picks') p
        """)
    except Exception:
        print("Full schema failed, trying fallback...")
        conn.execute(f"""
        CREATE TABLE picks AS
        SELECT
            p.id, p.content_id, p.meeting_id, p.race_number, p.horse_name,
            p.pick_type, p.bet_type, p.bet_stake, p.odds_at_tip,
            p.hit, p.pnl, p.settled, p.settled_at, p.created_at,
            NULL AS exotic_type, NULL AS sequence_type, NULL AS sequence_variant,
            NULL AS win_probability, NULL AS place_probability, NULL AS value_rating,
            NULL AS confidence, NULL AS is_puntys_pick, false AS tracked_only
        FROM sqlite_scan('{pp}', 'picks') p
        """)

    # Denormalize venue/state/date
    conn.execute("ALTER TABLE picks ADD COLUMN IF NOT EXISTS venue VARCHAR")
    conn.execute("ALTER TABLE picks ADD COLUMN IF NOT EXISTS state VARCHAR")
    conn.execute("ALTER TABLE picks ADD COLUMN IF NOT EXISTS date DATE")

    conn.execute(f"""
    UPDATE picks SET
        venue = m.venue,
        date = m.date
    FROM (
        SELECT id, venue, date
        FROM sqlite_scan('{pp}', 'meetings')
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

    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_settled ON picks(settled)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_bet_type ON picks(bet_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_meeting ON picks(meeting_id)")
    conn.execute("ANALYZE")

    total = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
    settled = conn.execute("SELECT COUNT(*) FROM picks WHERE settled = true").fetchone()[0]
    pnl = conn.execute("SELECT ROUND(SUM(COALESCE(pnl, 0)), 2) FROM picks WHERE settled = true").fetchone()[0]
    print(f"Picks loaded: {total:,} ({settled:,} settled, P&L: ${pnl:+,.2f})")

    db_size = os.path.getsize(str(ANALYTICS_DB)) / (1024 * 1024)
    print(f"Database: {db_size:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
