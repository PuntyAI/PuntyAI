"""Check Nowra race results status."""
import sqlite3
conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("""
    SELECT r.race_number, r.results_status, r.name,
           COUNT(ru.id) total_runners,
           SUM(CASE WHEN ru.finish_position IS NOT NULL THEN 1 ELSE 0 END) with_pos
    FROM races r
    LEFT JOIN runners ru ON ru.race_id = r.id AND (ru.scratched = 0 OR ru.scratched IS NULL)
    WHERE r.meeting_id = 'nowra-2026-02-08'
    GROUP BY r.id
    ORDER BY r.race_number
""")
for r in c.fetchall():
    print(f"  R{r['race_number']} {r['results_status'] or 'None':10s} pos={r['with_pos']}/{r['total_runners']} {r['name']}")

# Check picks
c.execute("""
    SELECT pick_type, COUNT(*) cnt, SUM(CASE WHEN settled=1 THEN 1 ELSE 0 END) settled
    FROM picks WHERE meeting_id = 'nowra-2026-02-08'
    GROUP BY pick_type
""")
print("\nPick breakdown:")
for r in c.fetchall():
    print(f"  {r['pick_type']:15s} {r['cnt']} total, {r['settled']} settled")

conn.close()
