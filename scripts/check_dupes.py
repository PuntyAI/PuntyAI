import sqlite3

conn = sqlite3.connect('C:/projects/PuntyAI/data/punty.db')
c = conn.cursor()

# Find The Valley meeting
c.execute("SELECT id, venue, date FROM meetings WHERE venue LIKE '%Valley%' OR venue LIKE '%Cranbourne%' ORDER BY date DESC LIMIT 10")
print("=== MEETINGS ===")
for row in c.fetchall():
    print(row)

# Look for the specific meeting - "The Valley @ Southside Cranbourne"
c.execute("SELECT id, venue, date FROM meetings WHERE venue LIKE '%Valley%Cranbourne%' OR venue LIKE '%Cranbourne%' ORDER BY date DESC LIMIT 5")
print("\n=== CRANBOURNE SPECIFIC ===")
for row in c.fetchall():
    print(row)

# Get picks for recent Valley/Cranbourne Race 1
c.execute("""
    SELECT p.id, p.content_id, p.race_id, p.pick_type, p.horse_name, p.saddlecloth,
           p.tip_rank, p.bet_type, p.bet_stake, p.exotic_type, p.sequence_type, p.sequence_variant,
           p.created_at
    FROM picks p
    WHERE p.race_id LIKE '%cranbourne%r1' OR p.race_id LIKE '%valley%r1'
    ORDER BY p.race_id DESC, p.created_at DESC
""")
print("\n=== RACE 1 PICKS ===")
for row in c.fetchall():
    print(row)

conn.close()
