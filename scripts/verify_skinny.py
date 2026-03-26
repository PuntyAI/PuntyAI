"""Quick verification of skinny budget fix."""
import sqlite3

conn = sqlite3.connect("/opt/puntyai/data/punty.db")
cur = conn.cursor()

# Stats by variant
rows = cur.execute("""
    SELECT sequence_variant, COUNT(*), SUM(exotic_stake), SUM(pnl),
           SUM(CASE WHEN hit=1 THEN 1 ELSE 0 END)
    FROM picks
    WHERE pick_type = 'sequence' AND settled = 1
    GROUP BY sequence_variant
""").fetchall()

print("Variant      | Count | Staked     | PNL        | Wins")
print("-" * 60)
for r in rows:
    print(f"{r[0]:<12} | {r[1]:>5} | ${r[2]:>9.2f} | ${r[3]:>9.2f} | {r[4]:>4}")

# Winning skinny picks
print("\nWinning skinny picks:")
wins = cur.execute("""
    SELECT id, meeting_id, sequence_type, exotic_stake, pnl
    FROM picks
    WHERE pick_type = 'sequence' AND sequence_variant = 'skinny' AND hit = 1
""").fetchall()
for w in wins:
    print(f"  {w[0]} | {w[1]} | {w[2]} | ${w[3]:.2f} | pnl=${w[4]:.2f}")

conn.close()
