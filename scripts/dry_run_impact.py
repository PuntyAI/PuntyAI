"""Dry-run impact analysis: compare data quality before/after fixes.

Runs on production to measure:
1. Field population rates (before = current DB, after = with derived fields)
2. Probability engine output changes per runner
3. Odds provider comparison (TAB vs SB)
"""
import asyncio
import json
import copy
import statistics
from datetime import date

from sqlalchemy import text, select
from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner
from punty.probability import calculate_race_probabilities, _get_weight_change_class


async def main():
    today = "2026-02-15"

    async with async_session() as db:
        # ── 1. Load all today's runners ──
        rows = await db.execute(text("""
            SELECT r.id as race_id, r.race_number, r.distance, r.class as class_,
                   r.track_condition, r.meeting_id,
                   m.venue,
                   ru.*
            FROM runners ru
            JOIN races r ON ru.race_id = r.id
            JOIN meetings m ON r.meeting_id = m.id
            WHERE m.date = :today AND NOT ru.scratched
            ORDER BY m.venue, r.race_number, ru.saddlecloth
        """), {"today": today})
        columns = rows.keys()
        all_rows = rows.fetchall()

        if not all_rows:
            print("No runners found for today!")
            return

        print(f"=== DRY RUN IMPACT ANALYSIS — {today} ===")
        print(f"Total runners: {len(all_rows)}")
        print()

        # Group by race for probability calculations
        races_data = {}
        for row in all_rows:
            d = dict(zip(columns, row))
            race_id = d["race_id"]
            if race_id not in races_data:
                races_data[race_id] = {
                    "race": {
                        "id": race_id,
                        "race_number": d["race_number"],
                        "distance": d["distance"],
                        "class_": d["class_"],
                        "track_condition": d["track_condition"],
                        "field_size": 0,
                    },
                    "meeting": {
                        "id": d["meeting_id"],
                        "venue": d["venue"],
                    },
                    "runners": [],
                }
            races_data[race_id]["runners"].append(d)
            races_data[race_id]["race"]["field_size"] = len(races_data[race_id]["runners"])

        # ── 2. Field Population Analysis ──
        print("=" * 70)
        print("FIELD POPULATION RATES")
        print("=" * 70)

        key_fields = [
            "current_odds", "place_odds", "opening_odds",
            "odds_tab", "odds_sportsbet", "odds_bet365", "odds_ladbrokes", "odds_betfair",
            "days_since_last_run", "class_stats", "track_stats", "track_dist_stats",
            "distance_stats", "first_up_stats", "second_up_stats",
            "good_track_stats", "soft_track_stats", "heavy_track_stats",
            "jockey_stats", "speed_map_position", "form_history",
            "last_five", "weight", "barrier", "handicap_rating",
        ]

        for field in key_fields:
            populated = sum(1 for row in all_rows if dict(zip(columns, row)).get(field))
            pct = (populated / len(all_rows)) * 100 if all_rows else 0
            marker = " *** EMPTY ***" if pct == 0 else " ** LOW **" if pct < 30 else ""
            print(f"  {field:30s}: {populated:4d}/{len(all_rows)} ({pct:5.1f}%){marker}")

        # ── 3. Odds Provider Comparison ──
        print()
        print("=" * 70)
        print("ODDS PROVIDER COMPARISON")
        print("=" * 70)

        tab_vs_sb = []
        tab_only = 0
        sb_only = 0
        both = 0
        neither = 0

        for row in all_rows:
            d = dict(zip(columns, row))
            tab = d.get("odds_tab")
            sb = d.get("odds_sportsbet")
            has_tab = tab and isinstance(tab, (int, float)) and tab > 1.0
            has_sb = sb and isinstance(sb, (int, float)) and sb > 1.0

            if has_tab and has_sb:
                both += 1
                ratio = tab / sb
                tab_vs_sb.append({"horse": d.get("horse_name", "?"), "tab": tab, "sb": sb, "ratio": ratio})
            elif has_tab:
                tab_only += 1
            elif has_sb:
                sb_only += 1
            else:
                neither += 1

        print(f"  Both TAB+SB:    {both}")
        print(f"  TAB only:       {tab_only}")
        print(f"  SB only:        {sb_only}")
        print(f"  Neither:        {neither}")

        if tab_vs_sb:
            ratios = [x["ratio"] for x in tab_vs_sb]
            mismatches = [x for x in tab_vs_sb if x["ratio"] > 2.0 or x["ratio"] < 0.5]
            print(f"\n  Avg TAB/SB ratio: {statistics.mean(ratios):.2f}")
            print(f"  Median ratio:     {statistics.median(ratios):.2f}")
            print(f"  Major mismatches (>2x): {len(mismatches)}")
            if mismatches:
                for m in sorted(mismatches, key=lambda x: x["ratio"], reverse=True)[:10]:
                    print(f"    {m['horse']:25s}: TAB=${m['tab']:.2f} vs SB=${m['sb']:.2f} (ratio={m['ratio']:.1f}x)")

        # ── 4. Weight Change Analysis ──
        print()
        print("=" * 70)
        print("WEIGHT CHANGE FROM FORM HISTORY (NEW)")
        print("=" * 70)

        weight_classes = {"weight_up_big": 0, "weight_up_small": 0, "weight_same": 0,
                          "weight_down_small": 0, "weight_down_big": 0, "": 0}
        for row in all_rows:
            d = dict(zip(columns, row))
            wc = _get_weight_change_class(d)
            weight_classes[wc] = weight_classes.get(wc, 0) + 1

        total = len(all_rows)
        classified = total - weight_classes.get("", 0)
        print(f"  Classified: {classified}/{total} ({classified/total*100:.1f}%)")
        for k, v in sorted(weight_classes.items(), key=lambda x: -x[1]):
            if k:
                print(f"    {k:25s}: {v:4d} ({v/total*100:.1f}%)")
        print(f"    {'(no data)':25s}: {weight_classes.get('', 0):4d} ({weight_classes.get('', 0)/total*100:.1f}%)")

        # ── 5. Derived Fields Impact ──
        print()
        print("=" * 70)
        print("DERIVED FIELDS IMPACT (days_since_last_run, class_stats)")
        print("=" * 70)

        has_days = sum(1 for row in all_rows if dict(zip(columns, row)).get("days_since_last_run"))
        has_class = sum(1 for row in all_rows if dict(zip(columns, row)).get("class_stats"))
        has_fh = sum(1 for row in all_rows if dict(zip(columns, row)).get("form_history"))

        print(f"  form_history coverage:       {has_fh}/{total} ({has_fh/total*100:.1f}%)")
        print(f"  days_since_last_run filled:  {has_days}/{total} ({has_days/total*100:.1f}%)")
        print(f"  class_stats filled:          {has_class}/{total} ({has_class/total*100:.1f}%)")
        print()
        print("  NOTE: These will be filled on next re-scrape via _fill_derived_fields().")
        print("  Current DB has OLD data (before the fix). Impact will be visible after scrape.")

        # ── 6. Probability Engine — Before vs Simulated After ──
        print()
        print("=" * 70)
        print("PROBABILITY ENGINE COMPARISON (Current DB → Simulated Derived Fields)")
        print("=" * 70)

        changes = []

        for race_id, race_data in sorted(races_data.items()):
            runners_orig = race_data["runners"]
            race = race_data["race"]
            meeting = race_data["meeting"]

            if len(runners_orig) < 2:
                continue

            # Run probability on current DB data
            try:
                probs_before = calculate_race_probabilities(runners_orig, race, meeting)
            except Exception as e:
                print(f"  ERROR on {race_id}: {e}")
                continue

            # Simulate derived fields
            runners_after = []
            for r in runners_orig:
                r2 = dict(r)
                fh_raw = r2.get("form_history")
                if fh_raw:
                    try:
                        fh = json.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
                        if isinstance(fh, list) and fh:
                            # Derive days_since_last_run
                            if not r2.get("days_since_last_run") and isinstance(fh[0], dict):
                                from datetime import datetime
                                d_str = fh[0].get("date")
                                if d_str:
                                    try:
                                        race_date = datetime.strptime(today, "%Y-%m-%d").date()
                                        last_date = datetime.strptime(d_str[:10], "%Y-%m-%d").date()
                                        r2["days_since_last_run"] = (race_date - last_date).days
                                    except (ValueError, TypeError):
                                        pass

                            # Derive class_stats
                            if not r2.get("class_stats"):
                                race_class = race.get("class_", "") or ""
                                from punty.context.combo_form import _bucket_class
                                race_bucket = _bucket_class(race_class)
                                starts = wins = seconds = thirds = 0
                                for start in fh:
                                    if isinstance(start, dict):
                                        sc = start.get("class", "")
                                        if _bucket_class(sc) == race_bucket and race_bucket:
                                            starts += 1
                                            pos = start.get("position")
                                            try:
                                                pos_int = int(pos)
                                                if pos_int == 1: wins += 1
                                                elif pos_int == 2: seconds += 1
                                                elif pos_int == 3: thirds += 1
                                            except (ValueError, TypeError):
                                                pass
                                if starts > 0:
                                    r2["class_stats"] = json.dumps({
                                        "starts": starts, "wins": wins,
                                        "seconds": seconds, "thirds": thirds
                                    })
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Simulate SB-primary odds
                sb = r2.get("odds_sportsbet")
                tab = r2.get("odds_tab")
                if sb and isinstance(sb, (int, float)) and sb > 1.0:
                    r2["current_odds"] = sb
                    # Derive place_odds if missing
                    if not r2.get("place_odds"):
                        r2["place_odds"] = round(1.0 + (sb - 1.0) / 3, 2)

                runners_after.append(r2)

            try:
                probs_after = calculate_race_probabilities(runners_after, race, meeting)
            except Exception as e:
                print(f"  ERROR (after) on {race_id}: {e}")
                continue

            # Compare
            venue = meeting["venue"]
            rn = race["race_number"]

            for runner in runners_orig:
                rid = runner["id"]
                if rid in probs_before and rid in probs_after:
                    before = probs_before[rid]
                    after = probs_after[rid]
                    wp_before = before.win_probability
                    wp_after = after.win_probability
                    diff = wp_after - wp_before
                    pct_change = (diff / wp_before * 100) if wp_before > 0 else 0

                    changes.append({
                        "venue": venue,
                        "race": rn,
                        "horse": runner.get("horse_name", "?"),
                        "saddlecloth": runner.get("saddlecloth", "?"),
                        "odds_before": runner.get("current_odds"),
                        "odds_after": runners_after[runners_orig.index(runner)].get("current_odds"),
                        "wp_before": wp_before,
                        "wp_after": wp_after,
                        "diff": diff,
                        "pct_change": pct_change,
                        "factors_before": {k: round(v, 3) for k, v in before.factors.items()},
                        "factors_after": {k: round(v, 3) for k, v in after.factors.items()},
                    })

        # Summary
        if not changes:
            print("  No changes computed.")
            return

        abs_diffs = [abs(c["diff"]) for c in changes]
        pct_changes = [abs(c["pct_change"]) for c in changes]

        print(f"\n  Runners analysed: {len(changes)}")
        print(f"  Mean absolute probability change: {statistics.mean(abs_diffs):.4f}")
        print(f"  Mean % change:                    {statistics.mean(pct_changes):.1f}%")
        print(f"  Max absolute change:              {max(abs_diffs):.4f}")
        print(f"  Runners with >1% change:          {sum(1 for c in changes if abs(c['pct_change']) > 1)}")
        print(f"  Runners with >5% change:          {sum(1 for c in changes if abs(c['pct_change']) > 5)}")
        print(f"  Runners with >10% change:         {sum(1 for c in changes if abs(c['pct_change']) > 10)}")

        # Top movers
        print()
        print("  TOP 15 BIGGEST MOVERS:")
        print(f"  {'Venue':15s} {'R':>2s} {'#':>2s} {'Horse':25s} {'Odds B→A':>15s} {'WP Before':>10s} {'WP After':>10s} {'Change':>8s} {'%':>7s}")
        print("  " + "-" * 98)

        top = sorted(changes, key=lambda x: abs(x["diff"]), reverse=True)[:15]
        for c in top:
            ob = f"${c['odds_before']:.1f}" if c['odds_before'] else "?"
            oa = f"${c['odds_after']:.1f}" if c['odds_after'] else "?"
            odds_str = f"{ob}→{oa}"
            sign = "+" if c["diff"] > 0 else ""
            sc = str(c['saddlecloth'])
            horse = str(c['horse'])
            print(f"  {c['venue']:15s} R{c['race']} #{sc:>2s} {horse:25s} {odds_str:>15s} {c['wp_before']:10.4f} {c['wp_after']:10.4f} {sign}{c['diff']:7.4f} {sign}{c['pct_change']:6.1f}%")

        # Factor-level analysis
        print()
        print("  FACTOR-LEVEL CHANGES (avg absolute change per factor):")
        print(f"  {'Factor':25s} {'Avg Δ':>8s} {'Max Δ':>8s} {'Changed':>8s}")
        print("  " + "-" * 55)

        factor_names = set()
        for c in changes:
            factor_names.update(c["factors_before"].keys())
            factor_names.update(c["factors_after"].keys())

        factor_diffs = {}
        for fname in sorted(factor_names):
            diffs = []
            for c in changes:
                fb = c["factors_before"].get(fname, 0.5)
                fa = c["factors_after"].get(fname, 0.5)
                diffs.append(abs(fa - fb))
            factor_diffs[fname] = diffs

        for fname in sorted(factor_names, key=lambda f: statistics.mean(factor_diffs[f]), reverse=True):
            diffs = factor_diffs[fname]
            avg = statistics.mean(diffs)
            mx = max(diffs)
            changed = sum(1 for d in diffs if d > 0.001)
            if avg > 0.0001:
                print(f"  {fname:25s} {avg:8.4f} {mx:8.4f} {changed:8d}")

        # Per-venue summary
        print()
        print("  PER-VENUE IMPACT:")
        venues = sorted(set(c["venue"] for c in changes))
        for v in venues:
            vc = [c for c in changes if c["venue"] == v]
            mean_chg = statistics.mean([abs(c["pct_change"]) for c in vc])
            max_chg = max([abs(c["pct_change"]) for c in vc])
            print(f"    {v:25s}: {len(vc)} runners, avg {mean_chg:.1f}% change, max {max_chg:.1f}%")


asyncio.run(main())
