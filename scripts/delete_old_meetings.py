"""Delete meetings and all related data for Feb 4-5 2026 and any earlier dates."""
import sqlite3
import sys

DB_PATH = "data/punty.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")  # Disable FK checks for deletion order
    cur = conn.cursor()

    # Find all meetings on or before Feb 5, 2026
    cur.execute("SELECT id, venue, date FROM meetings WHERE date < '2026-02-07' ORDER BY date, venue")
    meetings = cur.fetchall()

    if not meetings:
        print("No meetings found on or before 2026-02-07.")
        return

    print(f"Found {len(meetings)} meeting(s) to delete:")
    for m in meetings:
        print(f"  {m[0]:45s} | {m[1]:25s} | {m[2]}")

    meeting_ids = [m[0] for m in meetings]
    placeholders = ",".join("?" * len(meeting_ids))

    # Count related data before deletion
    tables_to_check = [
        ("picks", "meeting_id"),
        ("live_updates", "meeting_id"),
        ("content", "meeting_id"),
        ("context_snapshots", "meeting_id"),
        ("race_memories", "meeting_id"),
        ("race_assessments", "meeting_id"),
        ("scheduled_jobs", "meeting_id"),
        ("races", "meeting_id"),
    ]

    # Get race IDs for runner/result counts
    cur.execute(f"SELECT id FROM races WHERE meeting_id IN ({placeholders})", meeting_ids)
    race_ids = [r[0] for r in cur.fetchall()]
    race_placeholders = ",".join("?" * len(race_ids)) if race_ids else "''"

    print(f"\nRelated data counts:")
    for table, col in tables_to_check:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IN ({placeholders})", meeting_ids)
            count = cur.fetchone()[0]
            if count > 0:
                print(f"  {table:25s}: {count}")
        except sqlite3.OperationalError as e:
            print(f"  {table:25s}: (table not found: {e})")

    if race_ids:
        cur.execute(f"SELECT COUNT(*) FROM runners WHERE race_id IN ({race_placeholders})", race_ids)
        print(f"  {'runners':25s}: {cur.fetchone()[0]}")
        try:
            cur.execute(f"SELECT COUNT(*) FROM results WHERE race_id IN ({race_placeholders})", race_ids)
            print(f"  {'results':25s}: {cur.fetchone()[0]}")
        except sqlite3.OperationalError:
            pass

    print(f"  {'meetings':25s}: {len(meetings)}")

    if "--dry-run" in sys.argv:
        print("\n[DRY RUN] No data was deleted.")
        conn.close()
        return

    # Delete in dependency order (children first)
    print("\nDeleting...")

    def safe_delete(table, col, ids):
        ph = ",".join("?" * len(ids))
        try:
            cur.execute(f"DELETE FROM {table} WHERE {col} IN ({ph})", ids)
            print(f"  Deleted {cur.rowcount} {table}")
        except sqlite3.OperationalError:
            print(f"  {table}: skipped (table not found)")

    # Meeting-referenced tables
    for table in ["picks", "live_updates", "content", "context_snapshots",
                   "race_memories", "race_assessments", "scheduled_jobs"]:
        safe_delete(table, "meeting_id", meeting_ids)

    # Race-referenced tables
    if race_ids:
        for table in ["results", "runners"]:
            safe_delete(table, "race_id", race_ids)

    safe_delete("races", "meeting_id", meeting_ids)
    safe_delete("meetings", "id", meeting_ids)

    conn.commit()
    print("\nDone! All data for meetings on or before 2026-02-07 has been removed.")

    # Verify
    cur.execute("SELECT COUNT(*) FROM meetings WHERE date <= '2026-02-07'")
    remaining = cur.fetchone()[0]
    print(f"Remaining meetings on/before 2026-02-07: {remaining}")

    conn.close()

if __name__ == "__main__":
    main()
