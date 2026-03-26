"""Get meeting IDs for today."""
import sqlite3
conn = sqlite3.connect("data/punty.db")
rows = conn.execute("""
    SELECT m.id, m.venue,
           (SELECT MIN(r.start_time) FROM races r WHERE r.meeting_id = m.id) as first_race
    FROM meetings m
    WHERE m.date = '2026-02-13' AND m.selected = 1
    ORDER BY first_race
""").fetchall()
for r in rows:
    print(f"{r[0]:<45} {r[1]:<25} R1={r[2]}")
conn.close()
