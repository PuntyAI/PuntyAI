"""Quick check: are all picks settled across all meetings?"""
import sqlite3
conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Unsettled picks where race has results
c.execute("""
    SELECT p.meeting_id, p.race_number, p.pick_type, p.horse_name, p.exotic_type, p.sequence_type
    FROM picks p
    JOIN races race ON race.id = p.meeting_id || '-r' || p.race_number
    WHERE p.settled = 0
      AND race.results_status IN ('Paying', 'Closed')
    ORDER BY p.meeting_id, p.race_number
""")
unsettled = c.fetchall()
print(f"Unsettled picks with results: {len(unsettled)}")
for u in unsettled:
    name = u["horse_name"] or u["exotic_type"] or u["sequence_type"] or "?"
    print(f"  {u['meeting_id']} R{u['race_number']} {u['pick_type']:12s} {name}")

# Also check unsettled sequences with null race_number
c.execute("""
    SELECT p.meeting_id, p.pick_type, p.sequence_type, p.sequence_variant
    FROM picks p
    WHERE p.settled = 0 AND p.pick_type IN ('sequence', 'big3_multi')
""")
unsettled_seq = c.fetchall()
print(f"\nUnsettled sequences: {len(unsettled_seq)}")
for s in unsettled_seq:
    print(f"  {s['meeting_id']} {s['sequence_type']} {s['sequence_variant']}")

# Per-meeting summary
print("\n=== PER-MEETING SUMMARY ===")
c.execute("""
    SELECT meeting_id, COUNT(*) total,
           SUM(CASE WHEN settled=1 THEN 1 ELSE 0 END) settled,
           SUM(CASE WHEN settled=0 THEN 1 ELSE 0 END) unsettled,
           SUM(CASE WHEN hit=1 THEN 1 ELSE 0 END) winners,
           COALESCE(SUM(pnl),0) pnl
    FROM picks GROUP BY meeting_id ORDER BY meeting_id
""")
for m in c.fetchall():
    flag = " ***" if m["unsettled"] > 0 else ""
    print(f"  {m['meeting_id']:40s} {m['settled']:3d}/{m['total']:3d} settled  {m['winners']:3d} won  pnl=${m['pnl']:8.2f}{flag}")

conn.close()
