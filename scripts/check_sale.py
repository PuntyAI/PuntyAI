import sqlite3
conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Find Sale meetings
c.execute("SELECT id, date, venue FROM meetings WHERE LOWER(venue) LIKE '%sale%' ORDER BY date DESC LIMIT 5")
for m in c.fetchall():
    print(f"Meeting: {m['id']} | {m['venue']} | {m['date']}")

    # Check race results
    c.execute("SELECT race_number, results_status FROM races WHERE meeting_id = ? ORDER BY race_number", (m["id"],))
    races = c.fetchall()
    for r in races:
        print(f"  Race {r['race_number']}: results_status={r['results_status']}")

    # Check runners with results for each race
    c.execute("""
        SELECT r.race_number, ru.saddlecloth, ru.horse_name, ru.finish_position, ru.win_dividend, ru.place_dividend
        FROM runners ru
        JOIN races r ON ru.race_id = r.id
        WHERE r.meeting_id = ? AND ru.finish_position IS NOT NULL AND ru.finish_position <= 3
        ORDER BY r.race_number, ru.finish_position
    """, (m["id"],))
    print("  Top 3 finishers:")
    for ru in c.fetchall():
        print(f"    R{ru['race_number']}: {ru['finish_position']}. {ru['horse_name']} (No.{ru['saddlecloth']}) Win=${ru['win_dividend'] or 0} Place=${ru['place_dividend'] or 0}")

    # Check picks
    c.execute("""
        SELECT pick_type, horse_name, saddlecloth, race_number, tip_rank, bet_type, bet_stake, odds_at_tip, hit, pnl, settled
        FROM picks WHERE meeting_id = ? ORDER BY race_number, tip_rank
    """, (m["id"],))
    picks = c.fetchall()
    settled = sum(1 for p in picks if p["settled"])
    wins = sum(1 for p in picks if p["hit"])
    print(f"  Picks: {len(picks)} total, {settled} settled, {wins} winners")
    for p in picks:
        hit_str = "WIN" if p["hit"] else ("LOSS" if p["settled"] else "UNSETTLED")
        name = p["horse_name"] or p["pick_type"]
        print(f"    R{p['race_number']} {p['pick_type']:10s} {name:20s} No.{p['saddlecloth']} @${p['odds_at_tip'] or 0:.2f} {p['bet_type'] or ''} {hit_str} pnl=${p['pnl'] or 0:.2f}")
    print()

conn.close()
