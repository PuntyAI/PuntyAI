"""Dry run probability engine for Flemington today with detailed breakdown.

Shows per-track context multipliers, with/without comparison, and actual results.
"""

import asyncio
import json
import sys


async def main():
    from punty.models.database import async_session, init_db
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from punty.models.meeting import Meeting, Race, Runner
    from punty.probability import (
        calculate_race_probabilities, _load_context_profiles,
        _get_context_multipliers, _context_venue_type, _get_dist_bucket,
        _context_class_bucket, _get_state_for_venue, _track_key,
        load_dl_patterns_for_probability, _get_median_odds,
        DEFAULT_WEIGHTS, _CONTEXT_PROFILES,
    )

    await init_db()
    async with async_session() as db:
        # Load DL patterns
        dl_patterns = await load_dl_patterns_for_probability(db)
        print("DL patterns loaded:", len(dl_patterns))

        # Load context profiles
        ctx = _load_context_profiles()
        n_profiles = len(ctx.get("profiles", {})) if ctx else 0
        n_fallbacks = len(ctx.get("fallbacks", {})) if ctx else 0
        meta = ctx.get("metadata", {}) if ctx else {}
        print("Context profiles:", n_profiles, "profiles,", n_fallbacks, "fallbacks")
        if meta:
            print("  Built from:", meta.get("built_from", "?"), "runners")
            print("  Track profiles:", meta.get("track_profiles", 0))
            print("  Venue-type profiles:", meta.get("vtype_profiles", 0))
            print("  Min sample:", meta.get("min_sample", "?"))
            print("  Mult range:", meta.get("mult_range", "?"))

        # Load probability weights
        from punty.models.settings import AppSettings
        wt_result = await db.execute(
            select(AppSettings).where(AppSettings.key == "probability_weights")
        )
        wt_setting = wt_result.scalar_one_or_none()
        if wt_setting and wt_setting.value:
            raw_w = json.loads(wt_setting.value)
            weights = {k: v / 100.0 for k, v in raw_w.items()}
        else:
            weights = None

        # Load Flemington
        result = await db.execute(
            select(Meeting).where(
                Meeting.venue.ilike("%lemington%"),
                Meeting.date == "2026-02-14",
            ).options(selectinload(Meeting.races).selectinload(Race.runners))
        )
        meeting = result.scalar_one_or_none()
        if not meeting:
            print("No Flemington found!")
            return

        sep = "=" * 100
        dash = "-" * 100
        print()
        print(sep)
        print("FLEMINGTON -", meeting.date, "- Condition:", meeting.track_condition)
        print(sep)

        state = _get_state_for_venue(meeting.venue)
        vtype = _context_venue_type(meeting.venue, state)
        track = _track_key(meeting.venue)
        print("Track key:", track, "| Venue type:", vtype, "| State:", state)

        # Summary accumulators
        total_races = 0
        total_runners = 0
        correct_favs = 0
        total_favs = 0
        ctx_impact_summary = []

        for race in sorted(meeting.races, key=lambda r: r.race_number):
            active = [r for r in race.runners if not r.scratched]
            if not active:
                continue
            total_races += 1
            total_runners += len(active)

            print()
            print(dash)
            print("RACE %d: %s" % (race.race_number, race.name))
            dist = race.distance or 1400
            print("Distance: %dm | Class: %s | Prize: $%s" % (dist, race.class_, "{:,.0f}".format(race.prize_money or 0)))

            dbucket = _get_dist_bucket(dist)
            cbucket = _context_class_bucket(race.class_ or "")

            # Show context lookup chain
            track_key = "%s|%s|%s" % (track, dbucket, cbucket)
            vtype_key = "%s|%s|%s" % (vtype, dbucket, cbucket)
            print("Context keys: track=%s | vtype=%s" % (track_key, vtype_key))

            ctx_mults = _get_context_multipliers(race, meeting)
            if ctx_mults:
                # Determine which level matched
                profiles = ctx.get("profiles", {})
                fallbacks = ctx.get("fallbacks", {})
                matched = "?"
                if track_key in profiles:
                    matched = "PER-TRACK (%s)" % track_key
                elif vtype_key in profiles:
                    matched = "VENUE-TYPE (%s)" % vtype_key
                else:
                    for fb_key in ["%s|%s" % (track, dbucket), "%s|%s" % (vtype, dbucket), "%s|%s" % (dbucket, cbucket)]:
                        if fb_key in fallbacks:
                            matched = "FALLBACK (%s)" % fb_key
                            break
                n_runners = ctx_mults.pop("_n", None)  # remove _n if present
                print("Matched: %s (n=%s)" % (matched, n_runners or "?"))
                print("Multipliers:", json.dumps({k: round(v, 2) for k, v in sorted(ctx_mults.items())}))
            else:
                print("Context multipliers: NONE (no profile match)")

            # Calculate WITH context multipliers (normal)
            probs_with = calculate_race_probabilities(active, race, meeting, weights=weights, dl_patterns=dl_patterns)

            # Calculate WITHOUT context multipliers (temporarily disable)
            import punty.probability as prob_mod
            saved = prob_mod._CONTEXT_PROFILES
            prob_mod._CONTEXT_PROFILES = {}  # empty = no multipliers
            probs_without = calculate_race_probabilities(active, race, meeting, weights=weights, dl_patterns=dl_patterns)
            prob_mod._CONTEXT_PROFILES = saved  # restore

            # Sort by win probability (with context)
            sorted_runners = sorted(active, key=lambda r: probs_with.get(r.id) and probs_with[r.id].win_probability or 0, reverse=True)

            # Find actual result
            winner = None
            placers = []
            for r in active:
                fp = getattr(r, "finish_position", None)
                if fp == 1:
                    winner = r
                if fp and fp <= 3:
                    placers.append(r)

            has_results = winner is not None

            # Header
            hdr = "%-24s %3s %6s %6s %6s %6s | %6s %6s %+7s | %5s %5s %5s %5s %5s %5s %5s %5s %5s %5s" % (
                "Runner", "Pos", "Win%", "NoCtx", "Mkt%", "Value",
                "NoCtxV", "Delta", "Impact",
                "Mkt", "Form", "Class", "Pace", "Barr", "J/T", "Mvmt", "Wt", "Prof", "DL",
            )
            print()
            print(hdr)
            print("-" * len(hdr))

            for runner in sorted_runners:
                rp = probs_with.get(runner.id)
                rp_nc = probs_without.get(runner.id)
                if not rp or not rp_nc:
                    continue

                f = rp.factors
                name = (runner.horse_name or "?")[:23]
                fp = getattr(runner, "finish_position", None)
                pos_str = str(fp) if fp else "-"

                # Highlight winner
                marker = ""
                if fp == 1:
                    marker = " <-- WON"
                elif fp and fp <= 3:
                    marker = " <-- %s" % (["", "1st", "2nd", "3rd"][fp])

                delta = (rp.win_probability - rp_nc.win_probability) * 100
                val_delta = rp.value_rating - rp_nc.value_rating

                print("%-24s %3s %5.1f%% %5.1f%% %5.1f%% %5.2fx | %5.2fx %+5.2f %+6.1f%% | %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f%s" % (
                    name, pos_str,
                    rp.win_probability * 100, rp_nc.win_probability * 100,
                    rp.market_implied * 100, rp.value_rating,
                    rp_nc.value_rating, val_delta, delta,
                    f.get("market", 0), f.get("form", 0), f.get("class_fitness", 0),
                    f.get("pace", 0), f.get("barrier", 0), f.get("jockey_trainer", 0),
                    f.get("movement", 0), f.get("weight_carried", 0),
                    f.get("horse_profile", 0), f.get("deep_learning", 0),
                    marker,
                ))

            # Race summary
            if has_results:
                total_favs += 1
                fav_rp = probs_with.get(sorted_runners[0].id)
                if sorted_runners[0] == winner:
                    correct_favs += 1
                    print("  >> Model pick CORRECT: %s won!" % winner.horse_name)
                else:
                    winner_rp = probs_with.get(winner.id) if winner else None
                    winner_rank = next((i+1 for i, r in enumerate(sorted_runners) if r == winner), "?")
                    print("  >> Model pick WRONG: picked %s (%.1f%%), winner was %s (%.1f%%, ranked #%s)" % (
                        sorted_runners[0].horse_name, fav_rp.win_probability * 100 if fav_rp else 0,
                        winner.horse_name, winner_rp.win_probability * 100 if winner_rp else 0,
                        winner_rank,
                    ))

                # Check if winner was a value pick
                if winner:
                    w_rp = probs_with.get(winner.id)
                    w_nc = probs_without.get(winner.id)
                    if w_rp and w_nc:
                        ctx_impact_summary.append({
                            "race": race.race_number,
                            "winner": winner.horse_name,
                            "prob_with": w_rp.win_probability,
                            "prob_without": w_nc.win_probability,
                            "value_with": w_rp.value_rating,
                            "value_without": w_nc.value_rating,
                        })
            else:
                top = sorted_runners[0] if sorted_runners else None
                if top:
                    top_rp = probs_with[top.id]
                    print("  Top prob: %s (%.1f%%)" % (top.horse_name, top_rp.win_probability * 100))

        # Overall summary
        print()
        print(sep)
        print("SUMMARY")
        print(sep)
        print("Races: %d | Runners: %d" % (total_races, total_runners))

        if total_favs > 0:
            print("Model top pick correct: %d/%d (%.0f%%)" % (correct_favs, total_favs, correct_favs / total_favs * 100))

        if ctx_impact_summary:
            print()
            print("CONTEXT MULTIPLIER IMPACT ON WINNERS:")
            print("%-6s %-24s %8s %8s %+8s | %7s %7s %+7s" % (
                "Race", "Winner", "Ctx%", "NoCtx%", "Delta", "CtxVal", "NoVal", "Delta"))
            print("-" * 90)
            avg_delta = 0
            for s in ctx_impact_summary:
                d = (s["prob_with"] - s["prob_without"]) * 100
                vd = s["value_with"] - s["value_without"]
                avg_delta += d
                print("R%-5d %-24s %7.1f%% %7.1f%% %+7.1f%% | %6.2fx %6.2fx %+6.2f" % (
                    s["race"], s["winner"][:24],
                    s["prob_with"] * 100, s["prob_without"] * 100, d,
                    s["value_with"], s["value_without"], vd,
                ))
            avg_delta /= len(ctx_impact_summary)
            print()
            print("Average probability delta on winners: %+.1f%%" % avg_delta)


asyncio.run(main())
