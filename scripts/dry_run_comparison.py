"""Dry run: compare ranking formulas against actual results.

Formulas tested:
  OLD   = pure win_prob (pick by highest probability)
  SPLIT = #1 by pure prob, #2-3 by prob * clamp(value, 0.90, 1.20)
"""
import asyncio


async def run():
    from sqlalchemy import select
    from punty.models.database import async_session, init_db
    from punty.models.meeting import Meeting, Runner

    await init_db()

    async with async_session() as db:
        result = await db.execute(
            select(Meeting).where(Meeting.date == "2026-02-16")
        )
        meetings = result.scalars().all()

        SEP = "-" * 140
        print("=== DRY RUN: Old (pure prob) vs Split Formula (2026-02-16) ===")
        print("OLD = pure win_prob for all | SPLIT = #1 pure prob, #2-3 prob*clamp(value,0.90,1.20)")
        print("=" * 140)

        total_races = 0
        total_with_results = 0
        total_changes = 0

        # Per-formula stats
        stats = {
            "OLD": {"wins": 0, "places": 0, "top3_wins": 0, "top3_places": 0,
                    "pnl_win": 0.0, "pnl_place": 0.0, "odds": []},
            "SPLIT": {"wins": 0, "places": 0, "top3_wins": 0, "top3_places": 0,
                      "pnl_win": 0.0, "pnl_place": 0.0, "odds": []},
        }

        for meeting in meetings:
            from punty.context.builder import ContextBuilder
            builder = ContextBuilder(db)
            ctx = await builder.build_meeting_context(meeting.id)

            venue = ctx.get("meeting", {}).get("venue", meeting.venue)
            races = ctx.get("races", [])

            print()
            print(SEP)
            print("  %s (%d races)" % (venue.upper(), len(races)))
            print(SEP)

            for race in races:
                race_num = race.get("race_number", 0)
                race_id = "%s-r%d" % (meeting.id, race_num)
                runners = race.get("runners", [])

                # Get finish positions from DB
                finish_map = {}
                dividend_map = {}
                rr = await db.execute(
                    select(Runner).where(Runner.race_id == race_id)
                )
                for dbr in rr.scalars().all():
                    if dbr.saddlecloth and dbr.finish_position:
                        finish_map[int(dbr.saddlecloth)] = dbr.finish_position
                        dividend_map[int(dbr.saddlecloth)] = (
                            dbr.win_dividend or 0,
                            dbr.place_dividend or 0,
                        )

                candidates = []
                for r in runners:
                    if r.get("scratched"):
                        continue
                    sc = r.get("saddlecloth")
                    if not sc:
                        continue
                    odds = r.get("current_odds") or 0
                    if not odds or odds <= 1.0:
                        continue

                    win_prob = r.get("_win_prob_raw", 0)
                    value_rating = r.get("punty_value_rating", 1.0)

                    if win_prob <= 0:
                        continue

                    candidates.append({
                        "horse_name": r.get("horse_name", ""),
                        "saddlecloth": int(sc),
                        "odds": odds,
                        "win_prob": win_prob,
                        "value_rating": value_rating,
                    })

                if len(candidates) < 3:
                    continue

                total_races += 1
                has_results = bool(finish_map)
                if has_results:
                    total_with_results += 1

                mkt_fav = min(candidates, key=lambda c: c["odds"])

                # OLD: pure win_prob for all positions
                old_ranked = sorted(candidates, key=lambda c: c["win_prob"], reverse=True)

                # SPLIT: #1 by pure prob, #2-3 by prob * clamped value
                split_ranked = []
                # #1 = highest probability
                by_prob = sorted(candidates, key=lambda c: c["win_prob"], reverse=True)
                split_ranked.append(by_prob[0])
                used = {by_prob[0]["saddlecloth"]}

                # #2-3 = remaining ranked by prob * clamp(value, 0.90, 1.20)
                remaining = [c for c in candidates if c["saddlecloth"] not in used]
                remaining.sort(
                    key=lambda c: c["win_prob"] * max(0.90, min(c["value_rating"], 1.20)),
                    reverse=True,
                )
                for c in remaining[:3]:
                    split_ranked.append(c)

                old_top = old_ranked[:4]
                split_top = split_ranked[:4]

                changed = any(
                    old_top[i]["saddlecloth"] != split_top[i]["saddlecloth"]
                    for i in range(min(3, len(old_top), len(split_top)))
                )
                if changed:
                    total_changes += 1

                stats["OLD"]["odds"].append(old_top[0]["odds"])
                stats["SPLIT"]["odds"].append(split_top[0]["odds"])

                # Results tracking
                if has_results:
                    stake = 10.0
                    for label, top in [("OLD", old_top), ("SPLIT", split_top)]:
                        s = stats[label]
                        fp1 = finish_map.get(top[0]["saddlecloth"], 99)
                        div1 = dividend_map.get(top[0]["saddlecloth"], (0, 0))

                        if fp1 == 1:
                            s["wins"] += 1
                        if fp1 <= 3:
                            s["places"] += 1

                        # Top3 picks contain winner/placer
                        for i in range(min(3, len(top))):
                            if finish_map.get(top[i]["saddlecloth"], 99) == 1:
                                s["top3_wins"] += 1
                                break
                        for i in range(min(3, len(top))):
                            if finish_map.get(top[i]["saddlecloth"], 99) <= 3:
                                s["top3_places"] += 1
                                break

                        # PnL
                        if fp1 == 1 and div1[0] > 0:
                            s["pnl_win"] += div1[0] * stake - stake
                        else:
                            s["pnl_win"] -= stake
                        if fp1 <= 3 and div1[1] > 0:
                            s["pnl_place"] += div1[1] * stake - stake
                        else:
                            s["pnl_place"] -= stake

                # Print race detail
                marker = " *** CHANGED ***" if changed else ""
                print()
                print("  Race %d%s" % (race_num, marker))

                for i in range(min(4, len(old_top), len(split_top))):
                    o = old_top[i]
                    n = split_top[i]
                    rank = "Roughie" if i == 3 else "#%d" % (i + 1)
                    eq = "=" if o["saddlecloth"] == n["saddlecloth"] else "!"

                    o_res = ""
                    n_res = ""
                    if has_results:
                        for item, res_list in [(o, "o_res"), (n, "n_res")]:
                            fp = finish_map.get(item["saddlecloth"], 0)
                            res = ""
                            if fp == 1:
                                res = " WIN!"
                            elif fp == 2:
                                res = " 2nd"
                            elif fp == 3:
                                res = " 3rd"
                            elif fp:
                                res = " %dth" % fp
                            if res_list == "o_res":
                                o_res = res
                            else:
                                n_res = res

                    print(
                        "    %7s  OLD: %-18s $%5.2f p=%.3f v=%.2f%-5s %s  SPLIT: %-18s $%5.2f p=%.3f v=%.2f%-5s"
                        % (
                            rank,
                            o["horse_name"][:18], o["odds"], o["win_prob"], o["value_rating"], o_res,
                            eq,
                            n["horse_name"][:18], n["odds"], n["win_prob"], n["value_rating"], n_res,
                        )
                    )

        print()
        print("=" * 140)
        print("SUMMARY")
        print("=" * 140)
        print("  Total races:              %d" % total_races)
        print("  Races with results:       %d" % total_with_results)
        if total_races:
            print("  Races with changed top 3: %d (%d%%)" % (
                total_changes, total_changes * 100 // total_races))

        if total_with_results:
            print()
            print("  === RESULTS COMPARISON ===")
            print()
            print("  %-30s %12s %12s" % ("Metric", "OLD", "SPLIT"))
            print("  " + "-" * 56)
            for label in ["OLD", "SPLIT"]:
                s = stats[label]
            n = total_with_results
            for metric, key in [
                ("#1 pick wins", "wins"),
                ("#1 pick places (top 3)", "places"),
                ("Top3 contain winner", "top3_wins"),
                ("Top3 contain placer", "top3_places"),
            ]:
                ov = stats["OLD"][key]
                sv = stats["SPLIT"][key]
                print("  %-30s %5d/%d %3d%% %5d/%d %3d%%" % (
                    metric, ov, n, ov * 100 // n, sv, n, sv * 100 // n))

            print()
            print("  Simulated PnL ($10/race on #1 pick):")
            for label in ["OLD", "SPLIT"]:
                s = stats[label]
                print("    %s:  Win $%+8.2f  Place $%+8.2f  Combined $%+8.2f" % (
                    label, s["pnl_win"], s["pnl_place"], s["pnl_win"] + s["pnl_place"]))
            diff = (stats["SPLIT"]["pnl_win"] + stats["SPLIT"]["pnl_place"]) - \
                   (stats["OLD"]["pnl_win"] + stats["OLD"]["pnl_place"])
            print("    Diff: $%+.2f" % diff)


asyncio.run(run())
