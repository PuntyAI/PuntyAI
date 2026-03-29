import sqlite3

conn = sqlite3.connect('data/punty.db')
c = conn.cursor()

# Check He'sthechief finish
c.execute("SELECT horse_name, saddlecloth, finish_position, win_dividend, place_dividend FROM runners WHERE race_id = 'tamworth-2026-02-13-r1' AND saddlecloth = 2")
print("R1 He'sthechief:", c.fetchone())

meetings = [
    'tamworth-2026-02-13', 'mackay-2026-02-13', 'bet365-park-kilmore-2026-02-13',
    'taree-2026-02-13', 'canterbury-park-2026-02-13', 'southside-cranbourne-2026-02-13'
]

for mid in meetings:
    c.execute("""
        SELECT p.race_number, p.horse_name, p.saddlecloth
        FROM picks p
        WHERE p.meeting_id = ? AND p.pick_type = 'big3'
        ORDER BY p.race_number
    """, (mid,))
    legs = c.fetchall()
    if not legs:
        continue

    all_hit = True
    results = []
    for rn, horse, sc in legs:
        race_id = f"{mid}-r{rn}"
        c.execute("SELECT horse_name, saddlecloth FROM runners WHERE race_id = ? AND finish_position = 1", (race_id,))
        winner = c.fetchone()
        if winner and str(winner[1]) == str(sc):
            results.append(f"  R{rn}: {horse} #{sc} WON")
        elif winner:
            all_hit = False
            results.append(f"  R{rn}: {horse} #{sc} LOST (winner: {winner[0]} #{winner[1]})")
        else:
            all_hit = False
            results.append(f"  R{rn}: {horse} #{sc} NO RESULT YET")

    status = "ALL HIT" if all_hit else "missed"
    print(f"\n{mid}: {status}")
    for r in results:
        print(r)

conn.close()
