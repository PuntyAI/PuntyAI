import sqlite3
conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Get R8 race details
c.execute("""SELECT id, race_number, results_status, exotic_results FROM races
             WHERE meeting_id = 'sale-2026-02-11' AND race_number = 8""")
race = c.fetchone()
print(f"Race ID: {race['id']}")
print(f"Results status: {race['results_status']}")
print(f"Exotic results: {race['exotic_results']}")
print()

# Get ALL runners in R8 with finish data
c.execute("""SELECT saddlecloth, horse_name, finish_position, win_dividend, place_dividend,
             result_margin, current_odds, scratched
             FROM runners WHERE race_id = ? ORDER BY COALESCE(finish_position, 999)""", (race['id'],))
print("All Runners in R8:")
for r in c.fetchall():
    print(f"  No.{r['saddlecloth']} {r['horse_name']:20s} pos={r['finish_position']} "
          f"win=${r['win_dividend'] or 0} place=${r['place_dividend'] or 0} "
          f"odds=${r['current_odds'] or 0} scratched={r['scratched']}")
print()

# Get ALL picks for R8
c.execute("""SELECT id, pick_type, horse_name, saddlecloth, tip_rank, bet_type, bet_stake,
             odds_at_tip, hit, pnl, settled, settled_at, exotic_type, exotic_runners
             FROM picks WHERE meeting_id = 'sale-2026-02-11' AND race_number = 8
             ORDER BY tip_rank""")
print("Picks for R8:")
for p in c.fetchall():
    print(f"  id={p['id']} type={p['pick_type']} {p['horse_name'] or p['exotic_type'] or '':20s} "
          f"No.{p['saddlecloth']} rank={p['tip_rank']} bet={p['bet_type']} "
          f"stake=${p['bet_stake'] or 0} odds=${p['odds_at_tip'] or 0} "
          f"hit={p['hit']} pnl=${p['pnl'] or 0} settled={p['settled']} at={p['settled_at']}")
    if p['exotic_runners']:
        print(f"       exotic_runners={p['exotic_runners']}")
print()

# Check settlement timing - when was R8 settled vs when did results come in
c.execute("""SELECT settled_at FROM picks
             WHERE meeting_id = 'sale-2026-02-11' AND race_number = 8 AND settled = 1
             ORDER BY settled_at LIMIT 1""")
row = c.fetchone()
if row:
    print(f"R8 first settled at: {row['settled_at']}")

# Check when runners got results
c.execute("""SELECT updated_at FROM runners WHERE race_id = ? AND finish_position IS NOT NULL
             ORDER BY updated_at LIMIT 1""", (race['id'],))
row = c.fetchone()
if row:
    print(f"R8 runners first updated: {row['updated_at']}")

# Also check if the runner saddlecloth matches what pick expects
print("\nDirect match check:")
c.execute("""SELECT p.id, p.horse_name, p.saddlecloth, p.bet_type, p.hit, p.pnl,
             r.finish_position, r.win_dividend, r.place_dividend
             FROM picks p
             LEFT JOIN runners r ON r.race_id = ? AND r.saddlecloth = p.saddlecloth
             WHERE p.meeting_id = 'sale-2026-02-11' AND p.race_number = 8 AND p.pick_type = 'selection'
             ORDER BY p.tip_rank""", (race['id'],))
for row in c.fetchall():
    should_hit = False
    if row['bet_type'] in ('place', 'each_way') and row['finish_position'] and row['finish_position'] <= 3:
        should_hit = True
    if row['bet_type'] in ('win', 'saver_win') and row['finish_position'] == 1:
        should_hit = True
    mismatch = "MISMATCH!" if should_hit != bool(row['hit']) else "OK"
    print(f"  {row['horse_name']:20s} No.{row['saddlecloth']} bet={row['bet_type']} "
          f"pos={row['finish_position']} hit={row['hit']} pnl=${row['pnl'] or 0} "
          f"win_div=${row['win_dividend'] or 0} place_div=${row['place_dividend'] or 0} "
          f"-> {mismatch}")

conn.close()
