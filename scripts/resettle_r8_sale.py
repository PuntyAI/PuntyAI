"""Re-settle R8 Sale 2026-02-11 — picks were settled before results were populated."""
import sqlite3

conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()

meeting_id = "sale-2026-02-11"
race_number = 8
race_id = f"{meeting_id}-r{race_number}"

# Load runners with finish positions
c.execute("SELECT saddlecloth, horse_name, finish_position, win_dividend, place_dividend FROM runners WHERE race_id = ?", (race_id,))
runners = {r["saddlecloth"]: dict(r) for r in c.fetchall() if r["saddlecloth"]}

print("Runners with results:")
for sc, r in sorted(runners.items()):
    print(f"  No.{sc} {r['horse_name']:20s} pos={r['finish_position']} win=${r['win_dividend'] or 0} place=${r['place_dividend'] or 0}")

# Load picks for R8
c.execute("""SELECT id, pick_type, horse_name, saddlecloth, tip_rank, bet_type, bet_stake,
             odds_at_tip, place_odds_at_tip, hit, pnl, exotic_type, exotic_runners
             FROM picks WHERE meeting_id = ? AND race_number = ?""", (meeting_id, race_number))
picks = [dict(p) for p in c.fetchall()]

print(f"\nRe-settling {len(picks)} picks...")

for pick in picks:
    if pick["pick_type"] == "selection":
        sc = pick["saddlecloth"]
        runner = runners.get(sc)
        if not runner:
            print(f"  SKIP {pick['horse_name']} No.{sc} — runner not found")
            continue

        pos = runner["finish_position"]
        bet_type = (pick["bet_type"] or "win").lower().replace(" ", "_")
        stake = pick["bet_stake"] or 1.0
        won = pos == 1
        placed = pos is not None and pos <= 3

        # Use fixed odds from tip time
        win_odds = pick["odds_at_tip"] or runner["win_dividend"]
        place_odds = pick["place_odds_at_tip"] or runner["place_dividend"]

        old_hit = pick["hit"]
        old_pnl = pick["pnl"]

        if bet_type in ("win", "saver_win"):
            hit = won
            if won and win_odds:
                pnl = round(win_odds * stake - stake, 2)
            else:
                pnl = round(-stake, 2)
        elif bet_type == "place":
            hit = placed
            if placed and place_odds:
                pnl = round(place_odds * stake - stake, 2)
            else:
                pnl = round(-stake, 2)
        elif bet_type == "each_way":
            half = stake / 2
            if won and win_odds and place_odds:
                pnl = round((win_odds * half - half) + (place_odds * half - half), 2)
                hit = True
            elif placed and place_odds:
                pnl = round(-half + (place_odds * half - half), 2)
                hit = True
            else:
                pnl = round(-stake, 2)
                hit = False
        elif bet_type == "exotics_only":
            hit = won
            pnl = 0.0
        else:
            print(f"  SKIP {pick['horse_name']} — unknown bet_type: {bet_type}")
            continue

        changed = (hit != old_hit) or (abs((pnl or 0) - (old_pnl or 0)) > 0.01)
        if changed:
            print(f"  FIX {pick['horse_name']:20s} No.{sc} {bet_type} pos={pos} "
                  f"hit: {old_hit}->{hit} pnl: ${old_pnl or 0}->${pnl}")
            c.execute("UPDATE picks SET hit = ?, pnl = ? WHERE id = ?", (hit, pnl, pick["id"]))
        else:
            print(f"  OK  {pick['horse_name']:20s} No.{sc} {bet_type} pos={pos} hit={hit} pnl=${pnl}")

    elif pick["pick_type"] == "exotic":
        # Check trifecta box [1, 2, 8, 5]
        import json
        exotic_runners = json.loads(pick["exotic_runners"]) if pick["exotic_runners"] else []
        exotic_type = (pick["exotic_type"] or "").lower()

        # Get top 3 finish order
        top3 = sorted(
            [(r["finish_position"], r["saddlecloth"]) for r in runners.values() if r["finish_position"] and r["finish_position"] <= 3],
            key=lambda x: x[0]
        )
        top3_scs = [sc for _, sc in top3]

        print(f"  Exotic: {pick['exotic_type']} runners={exotic_runners} top3={top3_scs}")

        # Trifecta box: all top 3 must be in our runners
        if "trifecta" in exotic_type:
            hit = all(sc in exotic_runners for sc in top3_scs[:3]) if len(top3_scs) >= 3 else False
            # Get trifecta dividend
            race_result = c.execute("SELECT exotic_results FROM races WHERE id = ?", (race_id,)).fetchone()
            if race_result and race_result["exotic_results"]:
                import json
                exotic_results = json.loads(race_result["exotic_results"])
                tri_div = float(exotic_results.get("trifecta", 0))
                stake = pick["bet_stake"] or pick["exotic_stake"] or 20.0
                if hit and tri_div:
                    # For box bet: 6 combos from 4 runners, $20/6 per combo
                    n_runners = len(exotic_runners)
                    combos = 1
                    for i in range(min(3, n_runners)):
                        combos *= (n_runners - i)
                    unit = stake / combos if combos else stake
                    pnl = round(tri_div * unit - stake, 2)
                else:
                    pnl = round(-stake, 2)
            else:
                pnl = round(-(pick["bet_stake"] or pick.get("exotic_stake") or 20.0), 2)

            old_hit = pick["hit"]
            old_pnl = pick["pnl"]
            changed = (hit != old_hit) or (abs((pnl or 0) - (old_pnl or 0)) > 0.01)
            if changed:
                print(f"    FIX exotic hit: {old_hit}->{hit} pnl: ${old_pnl}->${pnl}")
                c.execute("UPDATE picks SET hit = ?, pnl = ? WHERE id = ?", (hit, pnl, pick["id"]))
            else:
                print(f"    OK  exotic hit={hit} pnl=${pnl}")

conn.commit()
print("\nDone! Changes committed.")

# Verify
c.execute("""SELECT horse_name, saddlecloth, bet_type, hit, pnl, pick_type, exotic_type
             FROM picks WHERE meeting_id = ? AND race_number = ? ORDER BY tip_rank""",
          (meeting_id, race_number))
print("\nFinal state:")
for p in c.fetchall():
    name = p["horse_name"] or p["exotic_type"] or "exotic"
    status = "WIN" if p["hit"] else "LOSS"
    print(f"  {name:20s} No.{p['saddlecloth']} {p['bet_type'] or ''} {status} pnl=${p['pnl'] or 0}")

conn.close()
