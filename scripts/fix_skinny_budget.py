"""Retrospective fix: update all skinny sequence picks from $1 to $10 budget.

Recalculates P&L using the flexi formula:
  pnl = dividend × (total_stake / combos) - total_stake

Since this is linear in total_stake, scaling from $1 to $10 means:
  new_pnl = old_pnl × (10 / old_stake)

For unsettled or missed picks: pnl = -10.0

Run on server:
  cd /opt/puntyai && source venv/bin/activate
  python scripts/fix_skinny_budget.py
"""

import sqlite3
import sys

DB_PATH = "/opt/puntyai/data/punty.db"
NEW_STAKE = 10.0


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Find all skinny sequence picks
    rows = cur.execute("""
        SELECT id, meeting_id, sequence_type, exotic_stake, hit, pnl, settled
        FROM picks
        WHERE pick_type = 'sequence' AND sequence_variant = 'skinny'
        ORDER BY meeting_id, id
    """).fetchall()

    if not rows:
        print("No skinny sequence picks found.")
        return

    print(f"Found {len(rows)} skinny sequence picks to update.\n")
    print(f"{'ID':<16} {'Meeting':<30} {'Type':<10} {'Old Stake':>10} {'Old PNL':>10} {'New Stake':>10} {'New PNL':>10} {'Hit':>5}")
    print("-" * 113)

    total_old_pnl = 0.0
    total_new_pnl = 0.0
    updated = 0

    for row in rows:
        old_stake = row["exotic_stake"] or 1.0
        old_pnl = row["pnl"]
        is_hit = row["hit"]
        is_settled = row["settled"]

        # Calculate new PNL
        if not is_settled or old_pnl is None:
            # Not yet settled - just update the stake
            new_pnl = None
        elif not is_hit:
            # Missed - lost the full new stake
            new_pnl = -NEW_STAKE
        else:
            # Hit - scale proportionally
            # flexi formula: pnl = dividend × (stake / combos) - stake
            # This scales linearly with stake, so:
            new_pnl = round(old_pnl * (NEW_STAKE / old_stake), 2)

        print(
            f"{row['id']:<16} {row['meeting_id']:<30} {row['sequence_type'] or '?':<10} "
            f"${old_stake:>8.2f} ${old_pnl or 0:>8.2f} "
            f"${NEW_STAKE:>8.2f} ${new_pnl or 0:>8.2f} "
            f"{'YES' if is_hit else 'NO':>5}"
        )

        if old_pnl is not None:
            total_old_pnl += old_pnl
        if new_pnl is not None:
            total_new_pnl += new_pnl

        # Update the pick
        if new_pnl is not None:
            cur.execute(
                "UPDATE picks SET exotic_stake = ?, pnl = ? WHERE id = ?",
                (NEW_STAKE, new_pnl, row["id"])
            )
        else:
            cur.execute(
                "UPDATE picks SET exotic_stake = ? WHERE id = ?",
                (NEW_STAKE, row["id"])
            )
        updated += 1

    print("-" * 113)
    print(f"\nSummary:")
    print(f"  Picks updated: {updated}")
    print(f"  Old total PNL: ${total_old_pnl:+.2f}")
    print(f"  New total PNL: ${total_new_pnl:+.2f}")
    print(f"  Difference:    ${total_new_pnl - total_old_pnl:+.2f}")

    # Verify a few picks
    print(f"\nVerification (spot check 3 random settled picks):")
    verify = cur.execute("""
        SELECT id, exotic_stake, pnl, hit
        FROM picks
        WHERE pick_type = 'sequence' AND sequence_variant = 'skinny' AND settled = 1
        LIMIT 3
    """).fetchall()
    for v in verify:
        print(f"  {v['id']}: stake=${v['exotic_stake']:.2f}, pnl=${v['pnl']:.2f}, hit={v['hit']}")

    conn.commit()
    print(f"\nDone. {updated} picks committed to database.")
    conn.close()


if __name__ == "__main__":
    main()
