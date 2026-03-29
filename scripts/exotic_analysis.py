"""Deep analysis of exotic bet performance: overlap with selections, hit rates, and ROI."""
import asyncio
import json


async def run():
    from sqlalchemy import text
    from punty.models.database import async_session, init_db
    await init_db()
    async with async_session() as db:
        # Get all settled exotics
        r = await db.execute(text("""
            SELECT p.meeting_id, p.race_number, p.exotic_type, p.exotic_runners,
                   p.exotic_stake, p.hit, p.pnl, p.settled
            FROM picks p
            WHERE p.pick_type = 'exotic' AND p.settled = 1
        """))
        exotics = r.all()

        # Get all selections
        r2 = await db.execute(text("""
            SELECT p.meeting_id, p.race_number, p.saddlecloth, p.tip_rank,
                   p.horse_name, p.bet_type, p.hit, p.pnl, p.odds_at_tip,
                   p.value_rating, p.win_probability
            FROM picks p
            WHERE p.pick_type = 'selection' AND p.settled = 1
        """))
        selections = r2.all()

        # Build selection lookup
        sel_map = {}
        for s in selections:
            key = (s[0], s[1])
            if key not in sel_map:
                sel_map[key] = []
            sel_map[key].append({
                "sc": s[2], "rank": s[3], "name": s[4], "bet_type": s[5],
                "hit": s[6], "pnl": s[7], "odds": s[8], "value": s[9], "prob": s[10]
            })

        # Get race results
        r3 = await db.execute(text("""
            SELECT r.id, ru.saddlecloth, ru.finish_position, ru.win_dividend, ru.place_dividend
            FROM runners ru
            JOIN races r ON r.id = ru.race_id
            WHERE ru.finish_position IS NOT NULL AND ru.finish_position <= 4
        """))
        result_map = {}
        for row in r3.all():
            if row[0] not in result_map:
                result_map[row[0]] = {}
            result_map[row[0]][int(row[1])] = (row[2], row[3] or 0, row[4] or 0)

        def parse_runners(runners_json):
            """Parse exotic_runners JSON, flattening nested lists."""
            try:
                raw = json.loads(runners_json) if runners_json else []
                result = []
                for item in raw:
                    if isinstance(item, list):
                        result.extend(int(x) for x in item if str(x).isdigit())
                    else:
                        result.append(int(item))
                return result
            except Exception:
                return []

        def norm_type(t):
            t = t.lower().strip()
            if "trifecta" in t and ("box" in t or "boxed" in t):
                return "Trifecta Box"
            if "trifecta" in t and "standout" in t:
                return "Trifecta Standout"
            if "trifecta" in t:
                return "Trifecta"
            if "first" in t and "4" in t and "standout" in t:
                return "First4 Standout"
            if "first" in t and "4" in t:
                return "First4 Box"
            if "exacta" in t and "standout" in t:
                return "Exacta Standout"
            if "exacta" in t:
                return "Exacta"
            if "quinella" in t:
                return "Quinella"
            return t

        print("=== EXOTIC DEEP ANALYSIS ===")
        print()

        # 1. By type
        by_type = {}
        by_overlap = {}
        by_overlap_pct = {}

        for ex in exotics:
            mid, rn, etype, runners_json, stake, hit, pnl, settled = ex
            key = (mid, rn)
            sels = sel_map.get(key, [])

            ex_runners = parse_runners(runners_json)

            if not ex_runners:
                continue

            sel_scs = {s["sc"] for s in sels}
            overlap = len(set(ex_runners) & sel_scs)
            n_runners = len(ex_runners)
            overlap_pct = overlap / n_runners if n_runners else 0

            nt = norm_type(etype)

            if nt not in by_type:
                by_type[nt] = {"n": 0, "hit": 0, "pnl": 0.0, "stake": 0.0, "overlap_sum": 0}
            by_type[nt]["n"] += 1
            by_type[nt]["hit"] += (1 if hit else 0)
            by_type[nt]["pnl"] += (pnl or 0)
            by_type[nt]["stake"] += (stake or 20)
            by_type[nt]["overlap_sum"] += overlap

            ok = str(overlap)
            if ok not in by_overlap:
                by_overlap[ok] = {"n": 0, "hit": 0, "pnl": 0.0, "stake": 0.0}
            by_overlap[ok]["n"] += 1
            by_overlap[ok]["hit"] += (1 if hit else 0)
            by_overlap[ok]["pnl"] += (pnl or 0)
            by_overlap[ok]["stake"] += (stake or 20)

            if overlap_pct >= 1.0:
                band = "100%"
            elif overlap_pct >= 0.66:
                band = "66-99%"
            elif overlap_pct >= 0.50:
                band = "50-66%"
            elif overlap_pct >= 0.33:
                band = "33-50%"
            else:
                band = "<33%"
            if band not in by_overlap_pct:
                by_overlap_pct[band] = {"n": 0, "hit": 0, "pnl": 0.0, "stake": 0.0}
            by_overlap_pct[band]["n"] += 1
            by_overlap_pct[band]["hit"] += (1 if hit else 0)
            by_overlap_pct[band]["pnl"] += (pnl or 0)
            by_overlap_pct[band]["stake"] += (stake or 20)

        print("1. PERFORMANCE BY EXOTIC TYPE (normalized)")
        print("%-20s %5s %5s %8s %10s %8s %8s" % ("Type", "Count", "Hits", "Hit%", "PnL", "ROI%", "AvgOvlp"))
        print("-" * 65)
        for nt in sorted(by_type.keys(), key=lambda k: by_type[k]["n"], reverse=True):
            s = by_type[nt]
            roi = s["pnl"] / s["stake"] * 100 if s["stake"] else 0
            print("%-20s %5d %5d %7.1f%% $%+9.2f %+7.1f%% %7.1f" % (
                nt, s["n"], s["hit"],
                s["hit"] * 100 / s["n"] if s["n"] else 0,
                s["pnl"], roi,
                s["overlap_sum"] / s["n"] if s["n"] else 0))

        print()
        print("2. PERFORMANCE BY OVERLAP WITH SELECTIONS (# runners matching)")
        print("%-12s %5s %5s %8s %10s %8s" % ("Overlap", "Count", "Hits", "Hit%", "PnL", "ROI%"))
        print("-" * 50)
        for ok in sorted(by_overlap.keys()):
            s = by_overlap[ok]
            roi = s["pnl"] / s["stake"] * 100 if s["stake"] else 0
            print("%-12s %5d %5d %7.1f%% $%+9.2f %+7.1f%%" % (
                ok + " match", s["n"], s["hit"],
                s["hit"] * 100 / s["n"] if s["n"] else 0,
                s["pnl"], roi))

        print()
        print("3. PERFORMANCE BY OVERLAP PERCENTAGE")
        print("%-12s %5s %5s %8s %10s %8s" % ("Overlap%", "Count", "Hits", "Hit%", "PnL", "ROI%"))
        print("-" * 50)
        for band in ["100%", "66-99%", "50-66%", "33-50%", "<33%"]:
            s = by_overlap_pct.get(band, {"n": 0, "hit": 0, "pnl": 0.0, "stake": 0.0})
            if s["n"] == 0:
                continue
            roi = s["pnl"] / s["stake"] * 100 if s["stake"] else 0
            print("%-12s %5d %5d %7.1f%% $%+9.2f %+7.1f%%" % (
                band, s["n"], s["hit"],
                s["hit"] * 100 / s["n"] if s["n"] else 0,
                s["pnl"], roi))

        # 4. When exotics hit - what was the actual result vs our selections?
        print()
        print("4. HIT EXOTICS: RUNNER OVERLAP DETAIL")
        hit_exotics = [ex for ex in exotics if ex[5]]
        for ex in hit_exotics:
            mid, rn, etype, runners_json, stake, hit, pnl, settled = ex
            ex_runners = parse_runners(runners_json)

            key = (mid, rn)
            sels = sel_map.get(key, [])
            sel_info = {s["sc"]: "#%d %s" % (s["rank"], s["name"][:15]) for s in sels}

            race_id = "%s-r%d" % (mid, rn)
            race_results = result_map.get(race_id, {})

            runners_detail = []
            for sc in ex_runners:
                sel_str = sel_info.get(sc, "not-sel")
                pos = race_results.get(sc, (None,))[0] if race_results.get(sc) else None
                pos_str = "(%s)" % pos if pos else ""
                runners_detail.append("%d%s[%s]" % (sc, pos_str, sel_str))

            print("  %-25s R%d %-18s pnl=$%+.2f  %s" % (
                mid[:25], rn, norm_type(etype), pnl or 0,
                " | ".join(runners_detail)))

        # 5. Non-selection runners in exotics
        print()
        print("5. NON-SELECTION RUNNERS IN EXOTICS")
        non_sel_positions = []
        in_sel_positions = []
        for ex in exotics:
            mid, rn, etype, runners_json, stake, hit, pnl, settled = ex
            ex_runners = parse_runners(runners_json)
            if not ex_runners:
                continue
            key = (mid, rn)
            sels = sel_map.get(key, [])
            sel_scs = {s["sc"] for s in sels}
            race_id = "%s-r%d" % (mid, rn)
            race_results = result_map.get(race_id, {})

            for sc in ex_runners:
                res = race_results.get(sc)
                pos = res[0] if res else 99
                if sc in sel_scs:
                    in_sel_positions.append(pos)
                else:
                    non_sel_positions.append(pos)

        if non_sel_positions:
            non_top3 = sum(1 for p in non_sel_positions if p <= 3)
            in_top3 = sum(1 for p in in_sel_positions if p <= 3)
            print("  Non-selection runners in exotics: %d total" % len(non_sel_positions))
            print("  Place rate (top 3): non-sel %.1f%% vs sel %.1f%%" % (
                non_top3 * 100 / len(non_sel_positions),
                in_top3 * 100 / len(in_sel_positions) if in_sel_positions else 0))
            non_wins = sum(1 for p in non_sel_positions if p == 1)
            in_wins = sum(1 for p in in_sel_positions if p == 1)
            print("  Win rate: non-sel %.1f%% vs sel %.1f%%" % (
                non_wins * 100 / len(non_sel_positions),
                in_wins * 100 / len(in_sel_positions) if in_sel_positions else 0))

        # 6. Sequences
        print()
        print("6. SEQUENCE PERFORMANCE BY TYPE x VARIANT")
        r5 = await db.execute(text("""
            SELECT sequence_type, sequence_variant, COUNT(*), SUM(hit),
                   SUM(pnl), SUM(exotic_stake), AVG(exotic_stake)
            FROM picks
            WHERE pick_type = 'sequence' AND settled = 1
            GROUP BY sequence_type, sequence_variant
            ORDER BY sequence_type, sequence_variant
        """))
        print("%-12s %-12s %5s %5s %8s %10s %8s %10s" % (
            "Type", "Variant", "Count", "Hits", "Hit%", "PnL", "ROI%", "TotalStake"))
        print("-" * 75)
        for row in r5.all():
            n = row[2]
            hits = row[3] or 0
            pnl = row[4] or 0
            total_stake = row[5] or 0
            roi = pnl / total_stake * 100 if total_stake else 0
            print("%-12s %-12s %5d %5d %7.1f%% $%+9.2f %+7.1f%% $%9.2f" % (
                row[0] or "-", row[1] or "-", n, hits,
                hits * 100 / n if n else 0, pnl, roi, total_stake))

        # 7. Sequence leg hit rates
        print()
        print("7. SEQUENCE LEG HIT RATES")
        r6 = await db.execute(text("""
            SELECT sequence_type, sequence_variant, sequence_legs, sequence_start_race,
                   meeting_id, hit, pnl
            FROM picks
            WHERE pick_type = 'sequence' AND settled = 1 AND sequence_legs IS NOT NULL
        """))
        leg_hits = {}
        for row in r6.all():
            stype, svar, legs_json, start_race, mid, hit, pnl = row
            try:
                legs = json.loads(legs_json) if legs_json else []
            except Exception:
                continue
            if not legs:
                continue

            key = (stype or "-", svar or "-")
            if key not in leg_hits:
                leg_hits[key] = {"total": 0, "leg_hits": [0] * len(legs), "n_legs": len(legs)}

            leg_hits[key]["total"] += 1

            for i, leg in enumerate(legs):
                if i >= leg_hits[key]["n_legs"]:
                    break
                race_num = (start_race or 1) + i
                race_id = "%s-r%d" % (mid, race_num)
                race_res = result_map.get(race_id, {})

                winner_sc = None
                for sc, (pos, _, _) in race_res.items():
                    if pos == 1:
                        winner_sc = sc
                        break

                if winner_sc is not None:
                    leg_runners = []
                    for x in leg:
                        if isinstance(x, int):
                            leg_runners.append(x)
                        elif isinstance(x, str) and x.isdigit():
                            leg_runners.append(int(x))
                    if winner_sc in leg_runners:
                        leg_hits[key]["leg_hits"][i] += 1

        for key, data in sorted(leg_hits.items()):
            stype, svar = key
            total = data["total"]
            if total < 5:
                continue
            leg_strs = []
            for i, h in enumerate(data["leg_hits"]):
                leg_strs.append("L%d: %d/%d (%.0f%%)" % (i + 1, h, total, h * 100 / total if total else 0))
            print("  %s %s (%d bets): %s" % (stype, svar, total, " | ".join(leg_strs)))

        # 8. TODAY check
        print()
        print("8. TODAY (2026-02-16) - EXOTICS vs SELECTIONS vs RESULTS")
        r7 = await db.execute(text("""
            SELECT p.meeting_id, p.race_number, p.exotic_type, p.exotic_runners,
                   p.exotic_stake, p.hit, p.pnl, p.settled
            FROM picks p
            WHERE p.pick_type = 'exotic'
              AND p.meeting_id IN (SELECT id FROM meetings WHERE date = '2026-02-16')
            ORDER BY p.meeting_id, p.race_number
        """))
        today_exotics = r7.all()

        r8 = await db.execute(text("""
            SELECT p.meeting_id, p.race_number, p.saddlecloth, p.tip_rank,
                   p.horse_name, p.bet_type, p.hit, p.pnl, p.odds_at_tip
            FROM picks p
            WHERE p.pick_type = 'selection'
              AND p.meeting_id IN (SELECT id FROM meetings WHERE date = '2026-02-16')
            ORDER BY p.meeting_id, p.race_number, p.tip_rank
        """))
        today_sels = {}
        for row in r8.all():
            key = (row[0], row[1])
            if key not in today_sels:
                today_sels[key] = []
            today_sels[key].append({
                "sc": row[2], "rank": row[3], "name": row[4],
                "bet_type": row[5], "hit": row[6], "pnl": row[7], "odds": row[8]
            })

        for ex in today_exotics:
            mid, rn, etype, runners_json, stake, hit, pnl, settled = ex
            ex_runners = parse_runners(runners_json)

            key = (mid, rn)
            sels = today_sels.get(key, [])
            sel_scs = {s["sc"] for s in sels}
            overlap = len(set(ex_runners) & sel_scs)

            race_id = "%s-r%d" % (mid, rn)
            race_results = result_map.get(race_id, {})
            top4 = sorted(
                [(sc, pos) for sc, (pos, _, _) in race_results.items() if pos <= 4],
                key=lambda x: x[1]
            )

            status = "HIT" if hit else ("MISS" if settled else "PEND")
            result_str = ", ".join(["%d(%s)" % (sc, p) for sc, p in top4]) if top4 else "no results"

            print("  %-28s R%d %-18s ex=%s ovlp=%d/%d %s pnl=$%+.2f top4=[%s]" % (
                mid[:28], rn, norm_type(etype),
                str(ex_runners), overlap, len(ex_runners),
                status, pnl or 0, result_str))

        # 9. Missed opportunities
        print()
        print("9. TODAY - MISSED EXOTIC OPPORTUNITIES")
        today_exotic_keys = {(ex[0], ex[1]) for ex in today_exotics}
        for key in sorted(today_sels.keys()):
            if key in today_exotic_keys:
                continue
            mid, rn = key
            sels = today_sels[key]
            race_id = "%s-r%d" % (mid, rn)
            race_results = result_map.get(race_id, {})

            if not race_results:
                continue

            top3 = sorted(
                [(sc, pos) for sc, (pos, _, _) in race_results.items() if pos <= 3],
                key=lambda x: x[1]
            )
            sel_scs = [s["sc"] for s in sels]
            top3_scs = [sc for sc, _ in top3]

            if len(top3_scs) >= 3 and all(sc in sel_scs for sc in top3_scs):
                sel_names = {s["sc"]: s["name"][:15] for s in sels}
                print("  %-28s R%d: TRIFECTA BOX %s WOULD HIT! Result: %s" % (
                    mid[:28], rn, str(sel_scs),
                    ", ".join(["%d=%s(%s)" % (sc, sel_names.get(sc, "?"), pos) for sc, pos in top3])))
            elif len(top3_scs) >= 2 and all(sc in sel_scs for sc in top3_scs[:2]):
                sel_names = {s["sc"]: s["name"][:15] for s in sels}
                print("  %-28s R%d: EXACTA %d->%d WOULD HIT! Result: %s" % (
                    mid[:28], rn, top3_scs[0], top3_scs[1],
                    ", ".join(["%d=%s(%s)" % (sc, sel_names.get(sc, "?"), pos) for sc, pos in top3[:2]])))
            elif len(top3_scs) >= 2 and sum(1 for sc in top3_scs[:2] if sc in sel_scs) >= 1:
                sel_names = {s["sc"]: s["name"][:15] for s in sels}
                print("  %-28s R%d: QUINELLA partial (1 of 2 in sels). Result: %s" % (
                    mid[:28], rn,
                    ", ".join(["%d=%s(%s)" % (sc, sel_names.get(sc, "?"), pos) for sc, pos in top3[:2]])))


asyncio.run(run())
