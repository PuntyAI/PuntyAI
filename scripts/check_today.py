"""Check today's meetings status."""
import sqlite3
conn = sqlite3.connect("data/punty.db")

rows = conn.execute("""
    SELECT m.id, m.venue, m.selected, m.track_condition,
           (SELECT MIN(r.start_time) FROM races r WHERE r.meeting_id = m.id) as first_race,
           (SELECT COUNT(*) FROM races r WHERE r.meeting_id = m.id) as num_races,
           (SELECT COUNT(*) FROM content c WHERE c.meeting_id = m.id AND c.content_type = 'early_mail') as has_early_mail,
           (SELECT COUNT(*) FROM content c WHERE c.meeting_id = m.id AND c.content_type = 'early_mail' AND c.status = 'approved') as approved_em
    FROM meetings m
    WHERE m.date = '2026-02-13'
    ORDER BY m.selected DESC, first_race
""").fetchall()

print("MEETINGS TODAY:")
print(f"{'ID':<35} {'Venue':<15} {'Sel':<5} {'Track':<12} {'First Race':<20} {'#R':<4} {'EM':<4} {'App':<4}")
for r in rows:
    print(f"{r[0]:<35} {r[1]:<15} {r[2]:<5} {str(r[3] or '-'):<12} {str(r[4] or '-'):<20} {r[5]:<4} {r[6]:<4} {r[7]:<4}")

# Check scheduler status
print("\nSCHEDULER JOBS:")
import subprocess
# just print the time
from datetime import datetime
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo("Australia/Melbourne"))
print(f"Current Melbourne time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

conn.close()
