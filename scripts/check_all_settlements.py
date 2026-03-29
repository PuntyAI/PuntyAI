"""Audit ALL settled picks across all meetings for incorrect settlements.
Uses the same logic as picks.py settlement code."""
import sqlite3
import json

conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

def find_dividend(exotic_divs, key):
    """Match dividend key flexibly (same as _find_dividend in picks.py)."""
    key = key.lower().strip()
    for k, v in exotic_divs.items():
        if k.lower().strip() == key:
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0
    # Fuzzy match
    for k, v in exotic_divs.items():
        if key in k.lower():
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0
    return 0.0

fixes = []

# ---- SELECTIONS (already fixed, verify) ----
print("=== CHECKING SELECTIONS ===")
c.execute("""
    SELECT p.id, p.meeting_id, p.race_number, p.horse_name, p.saddlecloth,
           p.bet_type, p.bet_stake, p.odds_at_tip, p.place_odds_at_tip,
           p.hit, p.pnl,
           r.finish_position, r.win_dividend, r.place_dividend
    FROM picks p
    JOIN races race ON race.id = p.meeting_id || '-r' || p.race_number
    JOIN runners r ON r.race_id = race.id AND r.saddlecloth = p.saddlecloth
    WHERE p.settled = 1 AND p.pick_type = 'selection' AND p.hit = 0
      AND r.finish_position IS NOT NULL
      AND (
          (p.bet_type IN ('win', 'saver_win') AND r.finish_position = 1)
          OR (p.bet_type = 'place' AND r.finish_position <= 3)
          OR (p.bet_type = 'each_way' AND r.finish_position <= 3)
      )
    ORDER BY p.meeting_id, p.race_number
""")
sel_misses = c.fetchall()
for m in sel_misses:
    bet_type = (m["bet_type"] or "win").lower().replace(" ", "_")
    stake = m["bet_stake"] or 1.0
    pos = m["finish_position"]
    won = pos == 1
    placed = pos is not None and pos <= 3
    win_odds = m["odds_at_tip"] or m["win_dividend"]
    place_odds = m["place_odds_at_tip"] or m["place_dividend"]

    if bet_type in ("win", "saver_win"):
        correct_pnl = round(win_odds * stake - stake, 2) if won and win_odds else round(-stake, 2)
        correct_hit = won
    elif bet_type == "place":
        correct_pnl = round(place_odds * stake - stake, 2) if placed and place_odds else round(-stake, 2)
        correct_hit = placed
    elif bet_type == "each_way":
        half = stake / 2
        if won and win_odds and place_odds:
            correct_pnl = round(win_odds * half + place_odds * half - stake, 2)
            correct_hit = True
        elif placed and place_odds:
            correct_pnl = round(place_odds * half - stake, 2)
            correct_hit = True
        else:
            correct_pnl = round(-stake, 2)
            correct_hit = False
    else:
        continue

    swing = correct_pnl - (m["pnl"] or 0)
    print(f"  {m['meeting_id']} R{m['race_number']} {m['horse_name']:20s} {bet_type} pos={pos} "
          f"was=${m['pnl'] or 0} -> ${correct_pnl} (+${swing:.2f})")
    fixes.append((correct_hit, correct_pnl, m["id"]))

if not sel_misses:
    print("  All selections correct.")

# ---- EXOTICS ----
print("\n=== CHECKING EXOTICS ===")
c.execute("""
    SELECT p.id, p.meeting_id, p.race_number, p.exotic_type, p.exotic_runners,
           p.exotic_stake, p.bet_stake, p.hit, p.pnl,
           race.exotic_results, race.id as race_id
    FROM picks p
    JOIN races race ON race.id = p.meeting_id || '-r' || p.race_number
    WHERE p.settled = 1 AND p.pick_type = 'exotic' AND p.hit = 0
    ORDER BY p.meeting_id, p.race_number
""")

for p in c.fetchall():
    exotic_type = (p["exotic_type"] or "").lower()
    exotic_runners = json.loads(p["exotic_runners"]) if p["exotic_runners"] else []
    stake = p["exotic_stake"] or p["bet_stake"] or 1.0
    exotic_divs = json.loads(p["exotic_results"]) if p["exotic_results"] else {}

    if not exotic_runners:
        continue

    # Get finish order for this race
    c2 = conn.cursor()
    c2.execute("""SELECT saddlecloth, finish_position FROM runners
                  WHERE race_id = ? AND finish_position IS NOT NULL AND finish_position <= 4
                  ORDER BY finish_position""", (p["race_id"],))
    finish_order = c2.fetchall()
    top_sc_int = [r["saddlecloth"] for r in finish_order]

    if not top_sc_int:
        continue

    # Check if legs format [[1], [5, 8], [8, 9]] vs flat [1, 5, 8, 9]
    is_legs_format = exotic_runners and isinstance(exotic_runners[0], list)

    hit = False
    dividend = 0.0
    combos = 1

    if is_legs_format:
        legs = exotic_runners
        required_positions = {"trifecta": 3, "exacta": 2, "quinella": 2, "first": 4}
        req_pos = 3
        for k, v in required_positions.items():
            if k in exotic_type:
                req_pos = v
                break

        if len(legs) >= req_pos and len(top_sc_int) >= req_pos:
            if "quinella" in exotic_type:
                a, b = top_sc_int[0], top_sc_int[1]
                hit = (a in legs[0] and b in legs[1]) or (b in legs[0] and a in legs[1])
            else:
                hit = all(top_sc_int[i] in legs[i] for i in range(req_pos))

        if hit:
            div_key = "first4" if "first" in exotic_type else exotic_type.split()[0]
            dividend = find_dividend(exotic_divs, div_key)

        combos = 1
        for leg in legs:
            combos *= len(leg)
    else:
        exotic_runners_int = [int(x) for x in exotic_runners if str(x).isdigit()]
        is_standout = "standout" in exotic_type
        is_boxed = "box" in exotic_type

        if "trifecta" in exotic_type:
            if len(top_sc_int) >= 3 and len(exotic_runners_int) >= 3:
                if is_standout:
                    standout = exotic_runners_int[0]
                    others = set(exotic_runners_int[1:])
                    hit = (top_sc_int[0] == standout and set(top_sc_int[1:3]).issubset(others))
                elif is_boxed or len(exotic_runners_int) > 3:
                    hit = set(top_sc_int[:3]).issubset(set(exotic_runners_int))
                else:
                    hit = list(top_sc_int[:3]) == list(exotic_runners_int[:3])
            if hit:
                dividend = find_dividend(exotic_divs, "trifecta")
        elif "exacta" in exotic_type:
            if len(top_sc_int) >= 2 and len(exotic_runners_int) >= 2:
                if is_standout:
                    standout = exotic_runners_int[0]
                    others = set(exotic_runners_int[1:])
                    hit = (top_sc_int[0] == standout and top_sc_int[1] in others)
                elif is_boxed or len(exotic_runners_int) > 2:
                    hit = set(top_sc_int[:2]).issubset(set(exotic_runners_int))
                else:
                    hit = list(top_sc_int[:2]) == list(exotic_runners_int[:2])
            if hit:
                dividend = find_dividend(exotic_divs, "exacta")
        elif "quinella" in exotic_type:
            if len(top_sc_int) >= 2 and len(exotic_runners_int) >= 2:
                hit = set(top_sc_int[:2]).issubset(set(exotic_runners_int))
            if hit:
                dividend = find_dividend(exotic_divs, "quinella")
        elif "first" in exotic_type and ("four" in exotic_type or "4" in exotic_type):
            if len(top_sc_int) >= 4 and len(exotic_runners_int) >= 4:
                if is_standout:
                    standout = exotic_runners_int[0]
                    others = set(exotic_runners_int[1:])
                    hit = (top_sc_int[0] == standout and set(top_sc_int[1:4]).issubset(others))
                elif is_boxed or len(exotic_runners_int) > 4:
                    hit = set(top_sc_int[:4]).issubset(set(exotic_runners_int))
                else:
                    hit = list(top_sc_int[:4]) == list(exotic_runners_int[:4])
            if hit:
                dividend = find_dividend(exotic_divs, "first4")

        # Calculate combos
        n = len(set(exotic_runners_int))
        if is_standout:
            others_n = n - 1
            if "trifecta" in exotic_type:
                combos = others_n * (others_n - 1) if others_n >= 2 else 1
            elif "exacta" in exotic_type:
                combos = others_n if others_n >= 1 else 1
            elif "first" in exotic_type:
                combos = others_n * (others_n - 1) * (others_n - 2) if others_n >= 3 else 1
        else:
            if "trifecta" in exotic_type:
                combos = n * (n - 1) * (n - 2) if n >= 3 else 1
            elif "exacta" in exotic_type:
                combos = n * (n - 1) if n >= 2 else 1
            elif "quinella" in exotic_type:
                combos = n * (n - 1) // 2 if n >= 2 else 1
            elif "first" in exotic_type:
                combos = n * (n - 1) * (n - 2) * (n - 3) if n >= 4 else 1

    if hit and dividend > 0:
        flexi_pct = stake / combos if combos > 0 else stake
        correct_pnl = round(dividend * flexi_pct - stake, 2)
        swing = correct_pnl - (p["pnl"] or 0)
        print(f"  MISS: {p['meeting_id']} R{p['race_number']} {p['exotic_type']} "
              f"runners={exotic_runners} top={top_sc_int[:4]} "
              f"div=${dividend} combos={combos} stake=${stake} "
              f"was=${p['pnl'] or 0} -> ${correct_pnl} (+${swing:.2f})")
        fixes.append((True, correct_pnl, p["id"]))

# ---- SEQUENCES ----
print("\n=== CHECKING SEQUENCES ===")
c.execute("""
    SELECT p.id, p.meeting_id, p.sequence_type, p.sequence_variant,
           p.sequence_legs, p.sequence_start_race, p.exotic_stake, p.bet_stake,
           p.hit, p.pnl
    FROM picks p
    WHERE p.settled = 1 AND p.pick_type = 'sequence' AND p.hit = 0
    ORDER BY p.meeting_id
""")

for p in c.fetchall():
    legs = json.loads(p["sequence_legs"]) if p["sequence_legs"] else []
    start_race = p["sequence_start_race"] or 1
    if not legs:
        continue

    all_hit = True
    all_resolved = True
    for i, leg in enumerate(legs):
        race_num = start_race + i
        race_id = f"{p['meeting_id']}-r{race_num}"
        c2 = conn.cursor()
        c2.execute("SELECT results_status FROM races WHERE id = ?", (race_id,))
        race_row = c2.fetchone()
        if not race_row or race_row["results_status"] not in ("Paying", "Closed"):
            all_resolved = False
            break
        c2.execute("SELECT saddlecloth FROM runners WHERE race_id = ? AND finish_position = 1", (race_id,))
        winner = c2.fetchone()
        if not winner or winner["saddlecloth"] not in leg:
            all_hit = False

    if not all_resolved:
        continue

    if all_hit:
        last_race_id = f"{p['meeting_id']}-r{start_race + len(legs) - 1}"
        c2.execute("SELECT exotic_results FROM races WHERE id = ?", (last_race_id,))
        race_row = c2.fetchone()
        exotic_divs = json.loads(race_row["exotic_results"]) if race_row and race_row["exotic_results"] else {}

        seq_type = (p["sequence_type"] or "").lower()
        dividend = 0.0
        for key in ("quaddie", "quadrella", "big6", "big 6"):
            if key in seq_type or seq_type == "":
                dividend = find_dividend(exotic_divs, key)
                if dividend > 0:
                    break

        stake = p["exotic_stake"] or p["bet_stake"] or 1.0
        if dividend > 0:
            combos = 1
            for leg in legs:
                combos *= len(leg)
            flexi_pct = stake / combos if combos > 0 else stake
            correct_pnl = round(dividend * flexi_pct - stake, 2)
            swing = correct_pnl - (p["pnl"] or 0)
            print(f"  MISS: {p['meeting_id']} {p['sequence_type']} {p['sequence_variant']} "
                  f"div=${dividend} combos={combos} stake=${stake} "
                  f"was=${p['pnl'] or 0} -> ${correct_pnl} (+${swing:.2f})")
            fixes.append((True, correct_pnl, p["id"]))

print(f"\n=== SUMMARY ===")
print(f"Total fixes: {len(fixes)}")

if fixes:
    total_swing = sum(f[1] - 0 for f in fixes)  # approximate
    for hit, pnl, pick_id in fixes:
        c.execute("UPDATE picks SET hit = ?, pnl = ? WHERE id = ?", (hit, pnl, pick_id))
    conn.commit()
    print("All fixes applied!")

# Final stats
print("\n=== UPDATED STATS ===")
for ptype in ("selection", "exotic", "sequence", "big3_multi"):
    c.execute("""SELECT COUNT(*) as total,
                        SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins,
                        COALESCE(SUM(pnl), 0) as pnl,
                        COALESCE(SUM(COALESCE(bet_stake, exotic_stake, 0)), 0) as staked
                 FROM picks WHERE settled = 1 AND pick_type = ?""", (ptype,))
    r = c.fetchone()
    sr = (r["wins"] / r["total"] * 100) if r["total"] else 0
    roi = (r["pnl"] / r["staked"] * 100) if r["staked"] else 0
    print(f"  {ptype:15s}: {r['wins']:3d}/{r['total']:3d} ({sr:5.1f}% SR) P&L: ${r['pnl']:8.2f} ROI: {roi:+.1f}%")

c.execute("""SELECT COUNT(*) as total,
                    SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as wins,
                    COALESCE(SUM(pnl), 0) as pnl,
                    COALESCE(SUM(bet_stake), 0) as staked
             FROM picks WHERE settled = 1 AND pick_type = 'selection' AND is_puntys_pick = 1""")
r = c.fetchone()
sr = (r["wins"] / r["total"] * 100) if r["total"] else 0
roi = (r["pnl"] / r["staked"] * 100) if r["staked"] else 0
print(f"  {'puntys_pick':15s}: {r['wins']:3d}/{r['total']:3d} ({sr:5.1f}% SR) P&L: ${r['pnl']:8.2f} ROI: {roi:+.1f}%")

conn.close()
