"""Find all picks that were incorrectly settled as losses when they should have won."""
import sqlite3

conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Find all settled selection picks where the runner actually placed but pick shows loss
c.execute("""
    SELECT p.id, p.meeting_id, p.race_number, p.horse_name, p.saddlecloth,
           p.bet_type, p.bet_stake, p.odds_at_tip, p.place_odds_at_tip,
           p.hit, p.pnl, p.settled_at,
           r.finish_position, r.win_dividend, r.place_dividend
    FROM picks p
    JOIN races race ON race.id = p.meeting_id || '-r' || p.race_number
    JOIN runners r ON r.race_id = race.id AND r.saddlecloth = p.saddlecloth
    WHERE p.settled = 1
      AND p.pick_type = 'selection'
      AND p.hit = 0
      AND r.finish_position IS NOT NULL
      AND (
          (p.bet_type IN ('win', 'saver_win') AND r.finish_position = 1)
          OR (p.bet_type = 'place' AND r.finish_position <= 3)
          OR (p.bet_type = 'each_way' AND r.finish_position <= 3)
      )
    ORDER BY p.meeting_id, p.race_number
""")

misses = c.fetchall()
print(f"Found {len(misses)} incorrectly settled picks:\n")

total_lost_pnl = 0
fixes = []
for m in misses:
    bet_type = (m["bet_type"] or "win").lower().replace(" ", "_")
    stake = m["bet_stake"] or 1.0
    pos = m["finish_position"]
    won = pos == 1
    placed = pos is not None and pos <= 3

    win_odds = m["odds_at_tip"] or m["win_dividend"]
    place_odds = m["place_odds_at_tip"] or m["place_dividend"]

    if bet_type in ("win", "saver_win"):
        if won and win_odds:
            correct_pnl = round(win_odds * stake - stake, 2)
        else:
            correct_pnl = round(-stake, 2)
        correct_hit = won
    elif bet_type == "place":
        if placed and place_odds:
            correct_pnl = round(place_odds * stake - stake, 2)
        else:
            correct_pnl = round(-stake, 2)
        correct_hit = placed
    elif bet_type == "each_way":
        half = stake / 2
        if won and win_odds and place_odds:
            correct_pnl = round((win_odds * half - half) + (place_odds * half - half), 2)
            correct_hit = True
        elif placed and place_odds:
            correct_pnl = round(-half + (place_odds * half - half), 2)
            correct_hit = True
        else:
            correct_pnl = round(-stake, 2)
            correct_hit = False
    else:
        continue

    swing = correct_pnl - (m["pnl"] or 0)
    total_lost_pnl += swing
    print(f"  {m['meeting_id']} R{m['race_number']} {m['horse_name']:20s} No.{m['saddlecloth']} "
          f"{bet_type} pos={pos} | was: hit={m['hit']} pnl=${m['pnl'] or 0} | "
          f"should be: hit={correct_hit} pnl=${correct_pnl} (swing: +${swing:.2f})")
    fixes.append((correct_hit, correct_pnl, m["id"]))

print(f"\nTotal P&L swing if fixed: +${total_lost_pnl:.2f}")
print(f"\nApplying {len(fixes)} fixes...")

for hit, pnl, pick_id in fixes:
    c.execute("UPDATE picks SET hit = ?, pnl = ? WHERE id = ?", (hit, pnl, pick_id))

conn.commit()
print("Done! All fixes applied.")
conn.close()
