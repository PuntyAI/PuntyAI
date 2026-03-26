"""Check impact of removing Friday Feb 6 data on overall P&L."""
import sqlite3

conn = sqlite3.connect('data/punty.db')
cur = conn.cursor()

# Get meetings on Friday Feb 6
cur.execute("SELECT id, venue FROM meetings WHERE date = '2026-02-06' ORDER BY venue")
fri_meetings = cur.fetchall()
print("=== FRIDAY FEB 6 MEETINGS ===")
for m in fri_meetings:
    print(f"  {m[0]:45s} | {m[1]}")

fri_ids = [m[0] for m in fri_meetings]
if not fri_ids:
    print("  No meetings found!")
    conn.close()
    exit()

ph = ",".join("?" * len(fri_ids))

# Friday picks breakdown
print("\n=== FRIDAY PICKS P&L ===")
cur.execute(f"""
    SELECT pick_type,
           COUNT(*) as total,
           SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as won,
           ROUND(SUM(COALESCE(bet_stake, exotic_stake, 0)), 2) as staked,
           ROUND(SUM(COALESCE(pnl, 0)), 2) as pnl
    FROM picks
    WHERE meeting_id IN ({ph}) AND settled = 1
    GROUP BY pick_type
    ORDER BY pick_type
""", fri_ids)
fri_total_pnl = 0
fri_total_staked = 0
rows = cur.fetchall()
for r in rows:
    roi = (r[4] / r[3] * 100) if r[3] else 0
    print(f"  {r[0]:15s} | {r[2]}/{r[1]} won | Staked: ${r[3]:>8.2f} | P&L: ${r[4]:>8.2f} | ROI: {roi:>+.1f}%")
    fri_total_pnl += r[4]
    fri_total_staked += r[3]
fri_roi = (fri_total_pnl / fri_total_staked * 100) if fri_total_staked else 0
print(f"  {'FRIDAY TOTAL':15s} |              | Staked: ${fri_total_staked:>8.2f} | P&L: ${fri_total_pnl:>8.2f} | ROI: {fri_roi:>+.1f}%")

# Overall current P&L (all data)
print("\n=== CURRENT OVERALL P&L (ALL DATA) ===")
cur.execute("""
    SELECT pick_type,
           COUNT(*) as total,
           SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as won,
           ROUND(SUM(COALESCE(bet_stake, exotic_stake, 0)), 2) as staked,
           ROUND(SUM(COALESCE(pnl, 0)), 2) as pnl
    FROM picks WHERE settled = 1
    GROUP BY pick_type
    ORDER BY pick_type
""")
all_total_pnl = 0
all_total_staked = 0
for r in cur.fetchall():
    roi = (r[4] / r[3] * 100) if r[3] else 0
    print(f"  {r[0]:15s} | {r[2]}/{r[1]} won | Staked: ${r[3]:>8.2f} | P&L: ${r[4]:>8.2f} | ROI: {roi:>+.1f}%")
    all_total_pnl += r[4]
    all_total_staked += r[3]
all_roi = (all_total_pnl / all_total_staked * 100) if all_total_staked else 0
print(f"  {'ALL TOTAL':15s} |              | Staked: ${all_total_staked:>8.2f} | P&L: ${all_total_pnl:>8.2f} | ROI: {all_roi:>+.1f}%")

# Without Friday
without_pnl = all_total_pnl - fri_total_pnl
without_staked = all_total_staked - fri_total_staked
without_roi = (without_pnl / without_staked * 100) if without_staked else 0

print("\n=== COMPARISON ===")
print(f"  Current ROI:          {all_roi:>+.1f}%  (P&L: ${all_total_pnl:>+.2f})")
print(f"  Without Friday ROI:   {without_roi:>+.1f}%  (P&L: ${without_pnl:>+.2f})")
print(f"  Improvement:          {without_roi - all_roi:>+.1f}% ROI")
print(f"  Friday drag on P&L:   ${fri_total_pnl:>+.2f}")

conn.close()
