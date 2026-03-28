"""Audit and fix incorrectly settled exacta picks.

For each exacta marked as hit:
- Get exotic_runners (first runner = anchor that must win)
- Get actual finish positions from the race
- If anchor didn't finish 1st → incorrect hit, fix to hit=0, pnl=-stake

Run on server:
    cd /opt/puntyai && source venv/bin/activate
    python3 scripts/fix_exacta_settlements.py
"""

import asyncio
import json
import sys

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Use production DB path on server, local path otherwise
import os
if os.path.exists("/opt/puntyai/data/punty.db"):
    DB_URL = "sqlite+aiosqlite:///data/punty.db"
else:
    DB_URL = "sqlite+aiosqlite:///data/punty.db"


async def audit_exactas():
    engine = create_async_engine(DB_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Find all settled exacta hits
        result = await db.execute(text("""
            SELECT p.id, p.meeting_id, p.race_number, p.exotic_type,
                   p.exotic_runners, p.pnl, p.exotic_stake
            FROM picks p
            WHERE p.pick_type = 'exotic' AND p.hit = 1 AND p.settled = 1
            AND p.exotic_type LIKE '%Exacta%'
            ORDER BY p.pnl DESC
        """))
        exactas = result.fetchall()
        print(f"Found {len(exactas)} settled exacta hits to audit\n")

        fixes = []
        for row in exactas:
            pick_id, meeting_id, race_number, exotic_type, exotic_runners_raw, pnl, stake = row

            # Parse exotic_runners
            try:
                runners = json.loads(exotic_runners_raw) if exotic_runners_raw else []
            except (json.JSONDecodeError, TypeError):
                # Try comma-separated
                try:
                    runners = [int(x.strip()) for x in str(exotic_runners_raw).split(",") if x.strip().isdigit()]
                except Exception:
                    runners = []

            if not runners:
                print(f"  SKIP {pick_id}: no runners parsed from {exotic_runners_raw}")
                continue

            anchor = runners[0]  # First runner must finish 1st for exacta

            # Get actual finish positions
            race_id = f"{meeting_id}-r{race_number}"
            fp_result = await db.execute(text("""
                SELECT saddlecloth, finish_position, horse_name
                FROM runners
                WHERE race_id = :race_id AND finish_position IS NOT NULL
                ORDER BY finish_position
            """), {"race_id": race_id})
            finishers = fp_result.fetchall()

            if not finishers:
                print(f"  SKIP {pick_id}: no finish positions for {race_id}")
                continue

            # Find who actually won (finish_position = 1)
            winner_sc = None
            for sc, fp, name in finishers:
                if fp == 1:
                    winner_sc = sc
                    break

            if winner_sc is None:
                print(f"  SKIP {pick_id}: no 1st place finisher found for {race_id}")
                continue

            if anchor == winner_sc:
                print(f"  OK   {pick_id}: {exotic_type} anchor #{anchor} won — P&L ${pnl:+.2f}")
            else:
                winner_name = next((name for sc, fp, name in finishers if fp == 1), "?")
                print(f"  FIX  {pick_id}: {exotic_type} anchor #{anchor} DID NOT WIN "
                      f"(winner was #{winner_sc} {winner_name}) — was P&L ${pnl:+.2f}, "
                      f"fixing to -${stake or 0:.2f}")
                fixes.append((pick_id, stake or 0))

        print(f"\n{'='*60}")
        print(f"Audit complete: {len(exactas)} exactas checked, {len(fixes)} need fixing")

        if not fixes:
            print("No fixes needed.")
            return

        # Ask for confirmation
        if "--dry-run" in sys.argv:
            print("\nDRY RUN — no changes applied.")
            return

        print(f"\nApplying {len(fixes)} fixes...")
        for pick_id, stake in fixes:
            await db.execute(text("""
                UPDATE picks SET hit = 0, pnl = :pnl
                WHERE id = :pick_id
            """), {"pick_id": pick_id, "pnl": -stake})
            print(f"  Fixed {pick_id}: hit=0, pnl=-${stake:.2f}")

        await db.commit()
        print(f"\nDone. {len(fixes)} exacta settlements corrected.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(audit_exactas())
