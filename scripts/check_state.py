"""Check today's meeting state."""
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

conn = sqlite3.connect("data/punty.db")
rows = conn.execute("""
    SELECT m.id, m.venue, m.selected, m.track_condition,
           (SELECT MIN(r.start_time) FROM races r WHERE r.meeting_id = m.id) as first_race,
           (SELECT COUNT(*) FROM races r WHERE r.meeting_id = m.id) as num_races,
           (SELECT COUNT(*) FROM content c WHERE c.meeting_id = m.id AND c.content_type = 'early_mail') as has_em,
           (SELECT c.status FROM content c WHERE c.meeting_id = m.id AND c.content_type = 'early_mail' LIMIT 1) as em_status,
           (SELECT c.id FROM content c WHERE c.meeting_id = m.id AND c.content_type = 'early_mail' LIMIT 1) as em_id
    FROM meetings m
    WHERE m.date = '2026-02-13'
    ORDER BY first_race
""").fetchall()

now = datetime.now(ZoneInfo("Australia/Melbourne"))
print(f"Melbourne time: {now.strftime('%H:%M:%S')}")
print()
fmt = "{:<18} sel={} track={:<12} R1={:<20} races={} EM={} status={:<16} em_id={}"
for r in rows:
    print(fmt.format(r[1], r[2], str(r[3] or "-"), str(r[4] or "-"), r[5], r[6], str(r[7] or "none"), str(r[8] or "-")))
conn.close()
