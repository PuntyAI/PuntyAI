"""Check today's results and overall tracking."""
import asyncio
import json


async def run():
    from sqlalchemy import text
    from punty.models.database import async_session, init_db
    await init_db()
    async with async_session() as db:
        # Today's selections
        r = await db.execute(text("""
            SELECT p.meeting_id, p.race_number, p.tip_rank, p.horse_name,
                   p.saddlecloth, p.bet_type, p.bet_stake, p.odds_at_tip,
                   p.hit, p.pnl, p.settled, p.value_rating, p.win_probability
            FROM picks p
            WHERE p.pick_type = 'selection'
              AND p.meeting_id IN (SELECT id FROM meetings WHERE date = '2026-02-16')
            ORDER BY p.meeting_id, p.race_number, p.tip_rank
        """))
        sels = r.all()

        # Today's exotics
        r2 = await db.execute(text("""
            SELECT p.meeting_id, p.race_number, p.exotic_type, p.exotic_runners,
                   p.exotic_stake, p.hit, p.pnl, p.settled
            FROM picks p
            WHERE p.pick_type = 'exotic'
              AND p.meeting_id IN (SELECT id FROM meetings WHERE date = '2026-02-16')
            ORDER BY p.meeting_id, p.race_number
        """))
        exotics = r2.all()

        # Today's sequences
        r3 = await db.execute(text("""
            SELECT p.meeting_id, p.sequence_type, p.sequence_variant,
                   p.exotic_stake, p.hit, p.pnl, p.settled
            FROM picks p
            WHERE p.pick_type = 'sequence'
              AND p.meeting_id IN (SELECT id FROM meetings WHERE date = '2026-02-16')
            ORDER BY p.meeting_id, p.sequence_type, p.sequence_variant
        """))
        seqs = r3.all()

        # Big3
        r4 = await db.execute(text("""
            SELECT p.meeting_id, p.pick_type, p.horse_name, p.race_number,
                   p.hit, p.pnl, p.settled
            FROM picks p
            WHERE p.pick_type IN ('big3', 'big3_multi')
              AND p.meeting_id IN (SELECT id FROM meetings WHERE date = '2026-02-16')
            ORDER BY p.meeting_id, p.pick_type
        """))
        big3s = r4.all()

        print("=== TODAY 2026-02-16 SELECTIONS ===")
        print()
        current_mid = None
        sel_settled = 0
        sel_hits = 0
        sel_pnl = 0.0
        sel_staked = 0.0
        win_pnl = 0.0
        place_pnl = 0.0
        win_staked = 0.0
        place_staked = 0.0
        winners = []
        pending = 0

        for s in sels:
            mid, rn, rank, name, sc, btype, stake, odds, hit, pnl, settled, val, prob = s
            if mid != current_mid:
                if current_mid:
                    print()
                current_mid = mid
                print("--- %s ---" % mid)

            status = ""
            if settled:
                sel_settled += 1
                sel_pnl += (pnl or 0)
                sel_staked += (stake or 0)
                bt = (btype or "").lower()
                if "win" in bt and "each" not in bt:
                    win_pnl += (pnl or 0)
                    win_staked += (stake or 0)
                elif "place" in bt:
                    place_pnl += (pnl or 0)
                    place_staked += (stake or 0)
                elif "each" in bt:
                    win_pnl += (pnl or 0) / 2
                    place_pnl += (pnl or 0) / 2
                    win_staked += (stake or 0)
                    place_staked += (stake or 0)
                if hit:
                    sel_hits += 1
                    status = "HIT +$%.2f" % (pnl or 0)
                    if pnl and pnl > 0:
                        winners.append(
                            "%s R%d %s $%.2f %s +$%.2f"
                            % (mid[:20], rn, name[:15], odds or 0, btype, pnl)
                        )
                else:
                    status = "MISS -$%.2f" % abs(pnl or 0)
            else:
                status = "PENDING"
                pending += 1

            print(
                "  R%d #%d %-18s SC%-2s $%5.2f %-10s $%.2f  v=%.2f p=%.1f%%  %s"
                % (
                    rn, rank, name[:18], sc, odds or 0,
                    btype or "", stake or 0,
                    val or 0, (prob or 0) * 100, status,
                )
            )

        # Exotic summary
        print()
        print("=== TODAY EXOTICS ===")
        ex_settled = 0
        ex_hits = 0
        ex_pnl = 0.0
        ex_staked = 0.0
        ex_pending = 0
        for e in exotics:
            mid, rn, etype, runners, stake, hit, pnl, settled = e
            if settled:
                ex_settled += 1
                ex_pnl += (pnl or 0)
                ex_staked += (stake or 0)
                if hit:
                    ex_hits += 1
            else:
                ex_pending += 1
            status = "HIT" if hit else ("MISS" if settled else "PEND")
            print(
                "  %-25s R%d %-18s $%.2f %s pnl=$%+.2f"
                % (mid[:25], rn, etype or "", stake or 20, status, pnl or 0)
            )

        # Sequence summary
        print()
        print("=== TODAY SEQUENCES ===")
        seq_pnl = 0.0
        seq_staked = 0.0
        seq_settled = 0
        seq_hits = 0
        seq_pending = 0
        for s in seqs:
            mid, stype, svar, stake, hit, pnl, settled = s
            if settled:
                seq_settled += 1
                seq_pnl += (pnl or 0)
                seq_staked += (stake or 0)
                if hit:
                    seq_hits += 1
            else:
                seq_pending += 1
            status = "HIT" if hit else ("MISS" if settled else "PEND")
            print(
                "  %-25s %-12s %-10s $%.2f %s pnl=$%+.2f"
                % (mid[:25], stype or "", svar or "", stake or 0, status, pnl or 0)
            )

        # Big3
        print()
        print("=== TODAY BIG3 ===")
        b3_pnl = 0.0
        for b in big3s:
            mid, ptype, name, rn, hit, pnl, settled = b
            if settled:
                b3_pnl += (pnl or 0)
            status = "HIT" if hit else ("MISS" if settled else "PEND")
            print(
                "  %-25s %-12s R%s %-15s %s pnl=$%+.2f"
                % (mid[:25], ptype, rn or "?", (name or "")[:15], status, pnl or 0)
            )

        # SUMMARY
        print()
        print("=" * 70)
        print("TODAY SUMMARY (2026-02-16)")
        print("=" * 70)
        print(
            "  Selections: %d settled, %d hits (%.0f%%), PnL: $%+.2f on $%.2f staked (%d pending)"
            % (sel_settled, sel_hits, sel_hits * 100 / max(1, sel_settled), sel_pnl, sel_staked, pending)
        )
        if win_staked:
            print(
                "    Win/Saver/EW: PnL $%+.2f (%.0f%% ROI)"
                % (win_pnl, win_pnl / win_staked * 100)
            )
        if place_staked:
            print(
                "    Place/EW:     PnL $%+.2f (%.0f%% ROI)"
                % (place_pnl, place_pnl / place_staked * 100)
            )
        print(
            "  Exotics:    %d settled, %d hits, PnL: $%+.2f (%d pending)"
            % (ex_settled, ex_hits, ex_pnl, ex_pending)
        )
        print(
            "  Sequences:  %d settled, %d hits, PnL: $%+.2f (%d pending)"
            % (seq_settled, seq_hits, seq_pnl, seq_pending)
        )
        print("  Big3:       PnL: $%+.2f" % b3_pnl)
        total_pnl = sel_pnl + ex_pnl + seq_pnl + b3_pnl
        print()
        print("  TOTAL TODAY: $%+.2f" % total_pnl)

        if winners:
            print()
            print("  WINNERS:")
            for w in winners:
                print("    %s" % w)

        # OVERALL STATS
        print()
        print("=" * 70)
        print("OVERALL STATS (ALL TIME)")
        print("=" * 70)
        r5 = await db.execute(text("""
            SELECT pick_type, bet_type,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN settled THEN 1 ELSE 0 END) as settled,
                   SUM(CASE WHEN hit THEN 1 ELSE 0 END) as hits,
                   SUM(pnl) as pnl,
                   SUM(CASE WHEN pick_type='selection' THEN bet_stake
                            WHEN pick_type='exotic' THEN exotic_stake
                            WHEN pick_type='sequence' THEN exotic_stake
                            ELSE 0 END) as staked
            FROM picks
            WHERE settled = 1
            GROUP BY pick_type, bet_type
            ORDER BY pick_type, bet_type
        """))
        print("%-12s %-12s %6s %5s %8s %10s %8s" % ("Type", "BetType", "Settld", "Hits", "Hit%", "PnL", "ROI%"))
        print("-" * 65)
        grand_pnl = 0.0
        grand_staked = 0.0
        for row in r5.all():
            ptype, btype, cnt, settled, hits, pnl, staked = row
            pnl = pnl or 0
            staked = staked or 0
            hits = hits or 0
            roi = pnl / staked * 100 if staked else 0
            grand_pnl += pnl
            grand_staked += staked
            print("%-12s %-12s %6d %5d %7.1f%% $%+9.2f %+7.1f%%" % (
                ptype or "-", btype or "-", settled or 0, hits,
                hits * 100 / max(1, settled or 1), pnl, roi))
        print("-" * 65)
        print("%-25s %6s %5s %8s $%+9.2f %+7.1f%%" % (
            "GRAND TOTAL", "", "", "", grand_pnl,
            grand_pnl / grand_staked * 100 if grand_staked else 0))


asyncio.run(run())
