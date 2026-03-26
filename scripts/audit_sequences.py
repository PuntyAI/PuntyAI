"""Audit ALL sequence picks across all meetings.
Check: unsettled sequences where all legs have results, and verify settled ones."""
import sqlite3
import json

conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=== ALL SEQUENCE PICKS ===")
c.execute("""
    SELECT p.id, p.meeting_id, p.pick_type, p.sequence_type, p.sequence_variant,
           p.sequence_legs, p.sequence_start_race, p.exotic_stake, p.bet_stake,
           p.hit, p.pnl, p.settled
    FROM picks p
    WHERE p.pick_type IN ('sequence', 'big3_multi')
    ORDER BY p.meeting_id, p.sequence_type, p.sequence_variant
""")

picks = c.fetchall()
print(f"Total sequence/multi picks: {len(picks)}\n")

unsettled_ready = []
wrong_settlements = []

for p in picks:
    legs = json.loads(p["sequence_legs"]) if p["sequence_legs"] else []
    start_race = p["sequence_start_race"] or 1
    stake = p["exotic_stake"] or p["bet_stake"] or 1.0
    seq_label = f"{p['sequence_type'] or p['pick_type']} {p['sequence_variant'] or ''}"

    if not legs:
        status = "EMPTY" if not p["settled"] else ("SETTLED-EMPTY" if p["settled"] else "?")
        print(f"  {p['meeting_id']:40s} {seq_label:25s} {status} settled={p['settled']} hit={p['hit']} pnl=${p['pnl'] or 0:.2f}")
        continue

    # Check each leg
    all_resolved = True
    all_hit = True
    leg_details = []
    for i, leg in enumerate(legs):
        race_num = start_race + i
        race_id = f"{p['meeting_id']}-r{race_num}"
        c2 = conn.cursor()
        c2.execute("SELECT results_status FROM races WHERE id = ?", (race_id,))
        race_row = c2.fetchone()
        if not race_row or race_row["results_status"] not in ("Paying", "Closed"):
            all_resolved = False
            leg_details.append(f"R{race_num}:PENDING")
            continue

        c2.execute("SELECT saddlecloth FROM runners WHERE race_id = ? AND finish_position = 1", (race_id,))
        winner = c2.fetchone()
        winner_sc = winner["saddlecloth"] if winner else None
        hit = winner_sc is not None and winner_sc in leg
        if not hit:
            all_hit = False
        leg_details.append(f"R{race_num}:{'HIT' if hit else 'MISS'}(w={winner_sc},sel={leg})")

    # Calculate expected dividend if all hit
    expected_pnl = None
    if all_resolved and all_hit:
        last_race_id = f"{p['meeting_id']}-r{start_race + len(legs) - 1}"
        c2.execute("SELECT exotic_results FROM races WHERE id = ?", (last_race_id,))
        race_row = c2.fetchone()
        exotic_divs = json.loads(race_row["exotic_results"]) if race_row and race_row["exotic_results"] else {}

        seq_type = (p["sequence_type"] or "").lower()
        dividend = 0.0
        for key in ("quaddie", "quadrella", "early quaddie", "early_quaddie", "big6", "big 6"):
            for k, v in exotic_divs.items():
                if key in k.lower() or k.lower() in key:
                    try:
                        dividend = float(v)
                        if dividend > 0:
                            break
                    except (ValueError, TypeError):
                        pass
            if dividend > 0:
                break

        combos = 1
        for leg in legs:
            combos *= len(leg)
        flexi_pct = stake / combos if combos > 0 else stake
        expected_pnl = round(dividend * flexi_pct - stake, 2) if dividend > 0 else None

    settled_str = "SETTLED" if p["settled"] else "UNSETTLED"
    hit_str = "HIT" if p["hit"] else "MISS" if p["settled"] else "?"
    resolved_str = "ALL-DONE" if all_resolved else "PENDING"

    flag = ""
    if not p["settled"] and all_resolved:
        flag = " *** SHOULD BE SETTLED"
        unsettled_ready.append(p)
    if p["settled"] and all_resolved and all_hit and not p["hit"]:
        flag = " *** SHOULD BE HIT"
        wrong_settlements.append((p, expected_pnl))
    if p["settled"] and p["hit"] and not all_hit:
        flag = " *** MARKED HIT BUT LEGS DON'T ALL WIN"

    print(f"  {p['meeting_id']:40s} {seq_label:25s} {settled_str:10s} {hit_str:5s} "
          f"pnl=${p['pnl'] or 0:8.2f} stake=${stake:6.2f} {resolved_str} {flag}")
    if flag:
        for ld in leg_details:
            print(f"    {ld}")
        if expected_pnl is not None:
            print(f"    Expected PNL: ${expected_pnl}")

print(f"\n=== SUMMARY ===")
print(f"Unsettled but ready: {len(unsettled_ready)}")
print(f"Wrong settlements: {len(wrong_settlements)}")

if unsettled_ready or wrong_settlements:
    print("\n=== APPLYING FIXES ===")
    for p in unsettled_ready:
        legs = json.loads(p["sequence_legs"]) if p["sequence_legs"] else []
        start_race = p["sequence_start_race"] or 1
        stake = p["exotic_stake"] or p["bet_stake"] or 1.0

        all_hit = True
        for i, leg in enumerate(legs):
            race_num = start_race + i
            race_id = f"{p['meeting_id']}-r{race_num}"
            c2 = conn.cursor()
            c2.execute("SELECT saddlecloth FROM runners WHERE race_id = ? AND finish_position = 1", (race_id,))
            winner = c2.fetchone()
            if not winner or winner["saddlecloth"] not in leg:
                all_hit = False
                break

        if all_hit:
            last_race_id = f"{p['meeting_id']}-r{start_race + len(legs) - 1}"
            c2.execute("SELECT exotic_results FROM races WHERE id = ?", (last_race_id,))
            race_row = c2.fetchone()
            exotic_divs = json.loads(race_row["exotic_results"]) if race_row and race_row["exotic_results"] else {}
            dividend = 0.0
            for key in ("quaddie", "quadrella", "early quaddie", "early_quaddie", "big6", "big 6"):
                for k, v in exotic_divs.items():
                    if key in k.lower() or k.lower() in key:
                        try:
                            dividend = float(v)
                            if dividend > 0:
                                break
                        except (ValueError, TypeError):
                            pass
                if dividend > 0:
                    break
            combos = 1
            for leg in legs:
                combos *= len(leg)
            flexi_pct = stake / combos if combos > 0 else stake
            pnl = round(dividend * flexi_pct - stake, 2) if dividend > 0 else round(-stake, 2)
            print(f"  Settling {p['meeting_id']} {p['sequence_type']} {p['sequence_variant']}: HIT div=${dividend} pnl=${pnl}")
            c.execute("UPDATE picks SET settled = 1, hit = 1, pnl = ? WHERE id = ?", (pnl, p["id"]))
        else:
            pnl = round(-stake, 2)
            print(f"  Settling {p['meeting_id']} {p['sequence_type']} {p['sequence_variant']}: MISS pnl=${pnl}")
            c.execute("UPDATE picks SET settled = 1, hit = 0, pnl = ? WHERE id = ?", (pnl, p["id"]))

    for p, expected_pnl in wrong_settlements:
        if expected_pnl is not None:
            print(f"  Fixing {p['meeting_id']} {p['sequence_type']} {p['sequence_variant']}: was pnl=${p['pnl'] or 0} -> ${expected_pnl}")
            c.execute("UPDATE picks SET hit = 1, pnl = ? WHERE id = ?", (expected_pnl, p["id"]))

    conn.commit()
    print("Fixes applied!")

conn.close()
