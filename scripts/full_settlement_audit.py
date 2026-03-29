"""Full settlement audit: find ALL unsettled picks where races have results,
and verify all settled picks are correct."""
import sqlite3
import json

conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# ---- 1. Find all meetings with results ----
print("=== ALL MEETINGS ===")
c.execute("""
    SELECT m.id, m.venue, m.date,
           COUNT(DISTINCT r.id) as total_races,
           SUM(CASE WHEN r.results_status IN ('Paying', 'Closed') THEN 1 ELSE 0 END) as completed_races
    FROM meetings m
    JOIN races r ON r.meeting_id = m.id
    GROUP BY m.id
    ORDER BY m.date DESC
""")
meetings = c.fetchall()
for m in meetings:
    print(f"  {m['date']} {m['venue']:20s} {m['completed_races']}/{m['total_races']} races complete")

# ---- 2. Find unsettled picks where race has results ----
print("\n=== UNSETTLED PICKS WITH RESULTS AVAILABLE ===")
c.execute("""
    SELECT p.id, p.meeting_id, p.race_number, p.pick_type, p.horse_name,
           p.saddlecloth, p.bet_type, p.bet_stake, p.exotic_type, p.exotic_stake,
           p.sequence_type, p.sequence_variant,
           race.results_status
    FROM picks p
    JOIN races race ON race.id = p.meeting_id || '-r' || p.race_number
    WHERE p.settled = 0
      AND race.results_status IN ('Paying', 'Closed')
    ORDER BY p.meeting_id, p.race_number
""")
unsettled = c.fetchall()
print(f"  Found {len(unsettled)} unsettled picks with race results available")
for u in unsettled:
    name = u["horse_name"] or u["exotic_type"] or u["sequence_type"] or "?"
    print(f"    {u['meeting_id']} R{u['race_number']} {u['pick_type']:12s} {name}")

# Also check unsettled sequences (race_number might be NULL)
c.execute("""
    SELECT p.id, p.meeting_id, p.pick_type, p.sequence_type, p.sequence_variant,
           p.sequence_start_race, p.sequence_legs, p.exotic_stake
    FROM picks p
    WHERE p.settled = 0 AND p.pick_type = 'sequence' AND p.race_number IS NULL
""")
unsettled_seq = c.fetchall()
if unsettled_seq:
    print(f"\n  Plus {len(unsettled_seq)} unsettled sequences (no race_number):")
    for s in unsettled_seq:
        legs = json.loads(s["sequence_legs"]) if s["sequence_legs"] else []
        start = s["sequence_start_race"] or 0
        # Check if all legs have results
        all_done = True
        for i in range(len(legs)):
            race_id = f"{s['meeting_id']}-r{start + i}"
            c2 = conn.cursor()
            c2.execute("SELECT results_status FROM races WHERE id = ?", (race_id,))
            r = c2.fetchone()
            if not r or r["results_status"] not in ("Paying", "Closed"):
                all_done = False
                break
        status = "READY" if all_done else "waiting"
        print(f"    {s['meeting_id']} {s['sequence_type']} {s['sequence_variant']} "
              f"R{start}-R{start+len(legs)-1} [{status}]")

# ---- 3. Check races with results but NO picks at all ----
print("\n=== RACES WITH RESULTS BUT NO PICKS ===")
c.execute("""
    SELECT r.meeting_id, r.race_number, r.results_status
    FROM races r
    LEFT JOIN picks p ON p.meeting_id = r.meeting_id AND p.race_number = r.race_number
    WHERE r.results_status IN ('Paying', 'Closed')
    GROUP BY r.id
    HAVING COUNT(p.id) = 0
    ORDER BY r.meeting_id, r.race_number
""")
no_picks = c.fetchall()
if no_picks:
    for r in no_picks:
        print(f"  {r['meeting_id']} R{r['race_number']} — {r['results_status']} but 0 picks")
else:
    print("  None — all races with results have picks")

# ---- 4. Check for runners missing finish positions in completed races ----
print("\n=== COMPLETED RACES WITH MISSING FINISH POSITIONS ===")
c.execute("""
    SELECT r.id, r.meeting_id, r.race_number, r.results_status,
           COUNT(ru.id) as total_runners,
           SUM(CASE WHEN ru.finish_position IS NOT NULL THEN 1 ELSE 0 END) as with_position
    FROM races r
    JOIN runners ru ON ru.race_id = r.id AND (ru.scratched = 0 OR ru.scratched IS NULL)
    WHERE r.results_status IN ('Paying', 'Closed')
    GROUP BY r.id
    HAVING with_position < 4
    ORDER BY r.meeting_id, r.race_number
""")
missing_pos = c.fetchall()
if missing_pos:
    for r in missing_pos:
        print(f"  {r['meeting_id']} R{r['race_number']} — {r['results_status']}: "
              f"{r['with_position']}/{r['total_runners']} runners have positions")
else:
    print("  All completed races have sufficient finish position data")

# ---- 5. Summary per meeting ----
print("\n=== PER-MEETING SETTLEMENT SUMMARY ===")
c.execute("""
    SELECT p.meeting_id,
           COUNT(*) as total_picks,
           SUM(CASE WHEN p.settled = 1 THEN 1 ELSE 0 END) as settled,
           SUM(CASE WHEN p.settled = 0 THEN 1 ELSE 0 END) as unsettled,
           SUM(CASE WHEN p.hit = 1 THEN 1 ELSE 0 END) as winners,
           COALESCE(SUM(p.pnl), 0) as pnl
    FROM picks p
    GROUP BY p.meeting_id
    ORDER BY p.meeting_id
""")
for m in c.fetchall():
    flag = " *** UNSETTLED" if m["unsettled"] > 0 else ""
    print(f"  {m['meeting_id']:40s} {m['settled']:3d} settled, {m['unsettled']:2d} unsettled, "
          f"{m['winners']:3d} winners, P&L: ${m['pnl']:8.2f}{flag}")

conn.close()
