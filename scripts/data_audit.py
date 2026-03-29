"""Comprehensive data audit: every field imported vs actual results over last 2 days."""
import asyncio
import json
import statistics
from collections import defaultdict
from sqlalchemy import text
from punty.models.database import async_session


async def main():
    async with async_session() as db:
        # ── ALL RUNNERS (last 2 days, not scratched) ──
        result = await db.execute(text("""
            SELECT
                r.race_id, r.horse_name, r.saddlecloth, r.barrier,
                r.weight, r.jockey, r.trainer,
                r.current_odds, r.opening_odds, r.place_odds,
                r.odds_tab, r.odds_sportsbet, r.odds_bet365, r.odds_ladbrokes, r.odds_betfair,
                r.finish_position, r.win_dividend, r.place_dividend, r.result_margin,
                r.speed_map_position, r.speed_value,
                r.form, r.last_five, r.career_record,
                r.horse_age, r.horse_sex, r.horse_colour,
                r.sire, r.dam, r.dam_sire,
                r.days_since_last_run, r.handicap_rating,
                r.track_dist_stats, r.track_stats, r.distance_stats,
                r.first_up_stats, r.second_up_stats,
                r.good_track_stats, r.soft_track_stats, r.heavy_track_stats,
                r.jockey_stats, r.class_stats,
                r.career_prize_money,
                r.scratched, r.scratching_reason,
                r.gear, r.gear_changes,
                r.stewards_comment, r.comment_long, r.comment_short,
                r.trainer_location,
                r.odds_flucs,
                r.form_history,
                rc.distance, rc.class as race_class, rc.prize_money,
                rc.results_status,
                m.venue, m.track_condition, m.date as meet_date
            FROM runners r
            JOIN races rc ON rc.id = r.race_id
            JOIN meetings m ON m.id = rc.meeting_id
            WHERE m.date >= date('now', '-2 days')
            AND r.scratched = 0
            ORDER BY m.date, m.venue, rc.race_number, r.saddlecloth
        """))
        rows = result.fetchall()
        cols = list(result.keys())
        total = len(rows)

        # Settled runners only (have finish position)
        settled = [r for r in rows if r[15] is not None]
        winners = [r for r in settled if r[15] == 1]
        placers = [r for r in settled if r[15] is not None and r[15] <= 3]

        print("=" * 100)
        print("COMPREHENSIVE DATA AUDIT — LAST 2 DAYS")
        print("=" * 100)
        print(f"Total runners: {total}")
        print(f"Settled (have finish pos): {len(settled)}")
        print(f"Winners: {len(winners)}")
        print(f"Venues: {sorted(set(r[57] for r in rows))}")
        print(f"Dates: {sorted(set(str(r[59]) for r in rows))}")

        # ═══════════════════════════════════════════════
        # SECTION 1: FIELD COVERAGE
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 1: FIELD COVERAGE (populated vs total)")
        print("=" * 100)

        col_idx = {c: i for i, c in enumerate(cols)}

        for i, col in enumerate(cols):
            non_null = sum(1 for r in rows if r[i] is not None and str(r[i]).strip() != '')
            pct = round(non_null / total * 100, 1) if total else 0
            flag = " *** EMPTY ***" if pct == 0 else (" ** LOW **" if pct < 50 else "")
            print(f"  {col:<25}: {non_null:>5}/{total} ({pct:>5.1f}%){flag}")

        # ═══════════════════════════════════════════════
        # SECTION 2: ODDS PROVIDER COMPARISON
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 2: ODDS PROVIDER COMPARISON")
        print("=" * 100)

        # 2a. Provider mismatches
        mismatches = []
        for r in settled:
            tab = r[10] or 0
            sb = r[11] or 0
            if tab > 1 and sb > 1:
                ratio = tab / sb
                if ratio > 3 or ratio < 0.33:
                    mismatches.append(r)

        print(f"\nMajor provider mismatches (TAB vs SB ratio >3x): {len(mismatches)}")
        for r in mismatches:
            tab = r[10] or 0
            sb = r[11] or 0
            ratio = tab / sb if sb > 0 else 0
            wdiv = r[16] or 0
            print(f"  {r[57]:<12} {r[0]:<30} {r[1]:<22} TAB=${tab:.2f} SB=${sb:.2f} ratio={ratio:.2f} pos={r[15]} wdiv=${wdiv:.2f}")

        # 2b. Which provider is closest to dividend for winners?
        print(f"\nWINNER ODDS ACCURACY (scraped odds vs actual dividend):")
        print(f"  {'Race':<30} {'Horse':<22} {'TAB':>7} {'SB':>7} {'Curr':>7} {'WDiv':>7} {'TABerr':>7} {'SBerr':>7} {'Best':>5}")
        tab_errors = []
        sb_errors = []
        curr_errors = []
        for r in winners:
            tab = r[10] or 0
            sb = r[11] or 0
            curr = r[7] or 0
            wdiv = r[16] or 0
            if wdiv > 0:
                te = abs(tab - wdiv) / wdiv * 100 if tab > 0 else None
                se = abs(sb - wdiv) / wdiv * 100 if sb > 0 else None
                ce = abs(curr - wdiv) / wdiv * 100 if curr > 0 else None
                if te is not None:
                    tab_errors.append(te)
                if se is not None:
                    sb_errors.append(se)
                if ce is not None:
                    curr_errors.append(ce)
                best = "TAB" if (te or 999) < (se or 999) else "SB"
                print(f"  {r[0]:<30} {r[1]:<22} ${tab:>6.2f} ${sb:>6.2f} ${curr:>6.2f} ${wdiv:>6.2f} {(str(round(te,0))+'%') if te else 'N/A':>7} {(str(round(se,0))+'%') if se else 'N/A':>7} {best:>5}")

        if tab_errors:
            print(f"\n  TAB avg error from dividend:     {sum(tab_errors)/len(tab_errors):.1f}% (n={len(tab_errors)})")
        if sb_errors:
            print(f"  SB avg error from dividend:      {sum(sb_errors)/len(sb_errors):.1f}% (n={len(sb_errors)})")
        if curr_errors:
            print(f"  current_odds avg error from div: {sum(curr_errors)/len(curr_errors):.1f}% (n={len(curr_errors)})")

        # 2c. Per-venue odds error
        print(f"\n  Per-venue TAB error from dividend (winners):")
        venue_tab_err = defaultdict(list)
        venue_sb_err = defaultdict(list)
        for r in winners:
            tab = r[10] or 0
            sb = r[11] or 0
            wdiv = r[16] or 0
            if wdiv > 0:
                if tab > 0:
                    venue_tab_err[r[57]].append(abs(tab - wdiv) / wdiv * 100)
                if sb > 0:
                    venue_sb_err[r[57]].append(abs(sb - wdiv) / wdiv * 100)
        for v in sorted(venue_tab_err):
            te = sum(venue_tab_err[v]) / len(venue_tab_err[v])
            se = sum(venue_sb_err.get(v, [0])) / max(len(venue_sb_err.get(v, [1])), 1)
            flag = " *** BAD ***" if te > 100 else ""
            print(f"    {v:<20}: TAB_err={te:.1f}%  SB_err={se:.1f}%  n={len(venue_tab_err[v])}{flag}")

        # ═══════════════════════════════════════════════
        # SECTION 3: SPEED MAP ACCURACY
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 3: SPEED MAP POSITION vs ACTUAL FINISH")
        print("=" * 100)

        smap_data = defaultdict(list)
        for r in settled:
            sm = r[19]
            fin = r[15]
            if sm and fin and fin <= 20:
                smap_data[sm].append(fin)

        for pos in ['leader', 'on_pace', 'midfield', 'backmarker']:
            if pos in smap_data:
                f = smap_data[pos]
                avg = sum(f) / len(f)
                wins = sum(1 for x in f if x == 1)
                places = sum(1 for x in f if x <= 3)
                top5 = sum(1 for x in f if x <= 5)
                print(f"  {pos:<12}: n={len(f):>3}, avg_finish={avg:.1f}, win%={wins/len(f)*100:.1f}%, place%={places/len(f)*100:.1f}%, top5%={top5/len(f)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 4: BARRIER ANALYSIS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 4: BARRIER vs FINISH POSITION")
        print("=" * 100)

        for lo, hi, label in [(1, 4, '1-4'), (5, 8, '5-8'), (9, 12, '9-12'), (13, 99, '13+')]:
            fins = [r[15] for r in settled if r[3] and lo <= r[3] <= hi and r[15] <= 20]
            if fins:
                avg = sum(fins) / len(fins)
                wins = sum(1 for f in fins if f == 1)
                places = sum(1 for f in fins if f <= 3)
                print(f"  {label:<6}: n={len(fins):>3}, avg_finish={avg:.1f}, win%={wins/len(fins)*100:.1f}%, place%={places/len(fins)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 5: WEIGHT ANALYSIS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 5: WEIGHT CARRIED vs FINISH POSITION")
        print("=" * 100)

        for lo, hi, label in [(0, 54, '<54'), (54, 56, '54-56'), (56, 58, '56-58'), (58, 60, '58-60'), (60, 99, '60+')]:
            fins = [r[15] for r in settled if r[4] and lo <= r[4] < hi and r[15] <= 20]
            if fins:
                avg = sum(fins) / len(fins)
                wins = sum(1 for f in fins if f == 1)
                places = sum(1 for f in fins if f <= 3)
                print(f"  {label:<6}: n={len(fins):>3}, avg_finish={avg:.1f}, win%={wins/len(fins)*100:.1f}%, place%={places/len(fins)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 6: FORM STRING ANALYSIS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 6: LAST FIVE FORM vs FINISH")
        print("=" * 100)

        # Categorize form quality
        def form_quality(last_five):
            if not last_five:
                return "unknown"
            wins = last_five.count('1')
            places = sum(1 for c in last_five if c in '123')
            if wins >= 2:
                return "strong (2+ wins)"
            elif places >= 3:
                return "good (3+ places)"
            elif places >= 1:
                return "fair (1-2 places)"
            else:
                return "poor (no places)"

        form_cats = defaultdict(list)
        for r in settled:
            lf = r[22]
            fin = r[15]
            if fin and fin <= 20:
                form_cats[form_quality(lf)].append(fin)

        for cat in ["strong (2+ wins)", "good (3+ places)", "fair (1-2 places)", "poor (no places)", "unknown"]:
            if cat in form_cats:
                f = form_cats[cat]
                avg = sum(f) / len(f)
                wins = sum(1 for x in f if x == 1)
                places = sum(1 for x in f if x <= 3)
                print(f"  {cat:<25}: n={len(f):>3}, avg_finish={avg:.1f}, win%={wins/len(f)*100:.1f}%, place%={places/len(f)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 7: TRACK/DISTANCE/CONDITION STATS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 7: STATS FIELDS — DO THEY PREDICT?")
        print("=" * 100)

        def parse_stats(s):
            """Parse stats string like '5: 2-1-0' into (starts, wins, seconds, thirds)."""
            if not s:
                return None
            try:
                parts = s.split(':')
                starts = int(parts[0].strip())
                record = parts[1].strip().split('-')
                return (starts, int(record[0]), int(record[1]), int(record[2]))
            except (ValueError, IndexError):
                return None

        def stats_win_rate(s):
            parsed = parse_stats(s)
            if not parsed or parsed[0] == 0:
                return None
            return parsed[1] / parsed[0]

        stat_fields = [
            (32, 'track_dist_stats', 'Track+Distance'),
            (33, 'track_stats', 'Track Only'),
            (34, 'distance_stats', 'Distance Only'),
            (35, 'first_up_stats', 'First Up'),
            (36, 'second_up_stats', 'Second Up'),
            (37, 'good_track_stats', 'Good Track'),
            (38, 'soft_track_stats', 'Soft Track'),
            (39, 'heavy_track_stats', 'Heavy Track'),
            (40, 'jockey_stats', 'Jockey'),
            (41, 'class_stats', 'Class'),
        ]

        for idx, field_name, label in stat_fields:
            populated = sum(1 for r in settled if r[idx] is not None and str(r[idx]).strip())
            # For runners with this stat, bucket by their historical win rate
            good_wr = []  # >25% win rate in this stat
            poor_wr = []  # <10% win rate
            mid_wr = []   # 10-25%
            no_data = []
            for r in settled:
                wr = stats_win_rate(r[idx])
                fin = r[15]
                if fin and fin <= 20:
                    if wr is None:
                        no_data.append(fin)
                    elif wr >= 0.25:
                        good_wr.append(fin)
                    elif wr < 0.10:
                        poor_wr.append(fin)
                    else:
                        mid_wr.append(fin)

            print(f"\n  {label} ({field_name}): populated={populated}/{len(settled)} ({populated/len(settled)*100:.0f}%)")
            for bucket_name, bucket in [("  >25% hist WR", good_wr), ("  10-25% hist WR", mid_wr), ("  <10% hist WR", poor_wr), ("  No data", no_data)]:
                if bucket:
                    avg = sum(bucket) / len(bucket)
                    wins = sum(1 for x in bucket if x == 1)
                    places = sum(1 for x in bucket if x <= 3)
                    print(f"    {bucket_name:<18}: n={len(bucket):>3}, avg_finish={avg:.1f}, win%={wins/len(bucket)*100:.1f}%, place%={places/len(bucket)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 8: HANDICAP RATING
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 8: HANDICAP RATING vs FINISH")
        print("=" * 100)

        for lo, hi, label in [(0, 50, '<50'), (50, 60, '50-60'), (60, 70, '60-70'), (70, 80, '70-80'), (80, 999, '80+')]:
            fins = [r[15] for r in settled if r[31] and lo <= r[31] < hi and r[15] <= 20]
            if fins:
                avg = sum(fins) / len(fins)
                wins = sum(1 for f in fins if f == 1)
                places = sum(1 for f in fins if f <= 3)
                print(f"  {label:<6}: n={len(fins):>3}, avg_finish={avg:.1f}, win%={wins/len(fins)*100:.1f}%, place%={places/len(fins)*100:.1f}%")

        no_hr = [r[15] for r in settled if not r[31] and r[15] and r[15] <= 20]
        if no_hr:
            print(f"  None  : n={len(no_hr):>3}, avg_finish={sum(no_hr)/len(no_hr):.1f}")

        # ═══════════════════════════════════════════════
        # SECTION 9: DAYS SINCE LAST RUN
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 9: DAYS SINCE LAST RUN vs FINISH")
        print("=" * 100)

        dslr_populated = sum(1 for r in settled if r[30] is not None)
        print(f"  Populated: {dslr_populated}/{len(settled)} ({dslr_populated/len(settled)*100:.1f}%)")

        for lo, hi, label in [(0, 15, '0-14'), (15, 29, '15-28'), (29, 61, '29-60'), (61, 121, '61-120'), (121, 9999, '120+')]:
            fins = [r[15] for r in settled if r[30] is not None and lo <= r[30] < hi and r[15] <= 20]
            if fins:
                avg = sum(fins) / len(fins)
                wins = sum(1 for f in fins if f == 1)
                places = sum(1 for f in fins if f <= 3)
                print(f"  {label:<7}: n={len(fins):>3}, avg_finish={avg:.1f}, win%={wins/len(fins)*100:.1f}%, place%={places/len(fins)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 10: CAREER PRIZE MONEY
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 10: CAREER PRIZE MONEY vs FINISH")
        print("=" * 100)

        for lo, hi, label in [(0, 10000, '<$10K'), (10000, 50000, '$10-50K'), (50000, 150000, '$50-150K'),
                               (150000, 500000, '$150-500K'), (500000, 99999999, '$500K+')]:
            fins = [r[15] for r in settled if r[42] is not None and lo <= r[42] < hi and r[15] <= 20]
            if fins:
                avg = sum(fins) / len(fins)
                wins = sum(1 for f in fins if f == 1)
                places = sum(1 for f in fins if f <= 3)
                print(f"  {label:<12}: n={len(fins):>3}, avg_finish={avg:.1f}, win%={wins/len(fins)*100:.1f}%, place%={places/len(fins)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 11: HORSE AGE & SEX
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 11: HORSE AGE & SEX vs FINISH")
        print("=" * 100)

        age_data = defaultdict(list)
        sex_data = defaultdict(list)
        for r in settled:
            fin = r[15]
            if fin and fin <= 20:
                age_data[r[24] or 'Unknown'].append(fin)
                sex_data[r[25] or 'Unknown'].append(fin)

        print("  Age:")
        for age in sorted(age_data):
            f = age_data[age]
            wins = sum(1 for x in f if x == 1)
            places = sum(1 for x in f if x <= 3)
            print(f"    {str(age):<8}: n={len(f):>3}, avg={sum(f)/len(f):.1f}, win%={wins/len(f)*100:.1f}%, place%={places/len(f)*100:.1f}%")

        print("  Sex:")
        for sex in sorted(sex_data):
            f = sex_data[sex]
            wins = sum(1 for x in f if x == 1)
            places = sum(1 for x in f if x <= 3)
            print(f"    {str(sex):<8}: n={len(f):>3}, avg={sum(f)/len(f):.1f}, win%={wins/len(f)*100:.1f}%, place%={places/len(f)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 12: GEAR & GEAR CHANGES
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 12: GEAR & GEAR CHANGES vs FINISH")
        print("=" * 100)

        gear_pop = sum(1 for r in settled if r[45] and str(r[45]).strip())
        gear_change_pop = sum(1 for r in settled if r[46] and str(r[46]).strip())
        print(f"  Gear populated: {gear_pop}/{len(settled)}")
        print(f"  Gear changes populated: {gear_change_pop}/{len(settled)}")

        has_changes = [r[15] for r in settled if r[46] and str(r[46]).strip() and r[15] and r[15] <= 20]
        no_changes = [r[15] for r in settled if (not r[46] or not str(r[46]).strip()) and r[15] and r[15] <= 20]
        if has_changes:
            wins = sum(1 for x in has_changes if x == 1)
            print(f"  With gear changes: n={len(has_changes)}, avg={sum(has_changes)/len(has_changes):.1f}, win%={wins/len(has_changes)*100:.1f}%")
        if no_changes:
            wins = sum(1 for x in no_changes if x == 1)
            print(f"  No gear changes:   n={len(no_changes)}, avg={sum(no_changes)/len(no_changes):.1f}, win%={wins/len(no_changes)*100:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 13: COMMENTS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 13: COMMENTS COVERAGE")
        print("=" * 100)

        comment_pop = sum(1 for r in settled if r[48] and str(r[48]).strip())
        stewards_pop = sum(1 for r in settled if r[47] and str(r[47]).strip())
        short_pop = sum(1 for r in settled if r[49] and str(r[49]).strip())
        print(f"  comment_long: {comment_pop}/{len(settled)} ({comment_pop/len(settled)*100:.1f}%)")
        print(f"  stewards_comment: {stewards_pop}/{len(settled)} ({stewards_pop/len(settled)*100:.1f}%)")
        print(f"  comment_short: {short_pop}/{len(settled)} ({short_pop/len(settled)*100:.1f}%)")

        # ═══════════════════════════════════════════════
        # SECTION 14: FORM HISTORY COVERAGE
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 14: FORM HISTORY (JSON) COVERAGE")
        print("=" * 100)

        fh_pop = sum(1 for r in rows if r[52] and str(r[52]).strip())
        print(f"  form_history populated: {fh_pop}/{total} ({fh_pop/total*100:.1f}%)")

        # Sample a form_history to see what fields are in it
        for r in rows:
            if r[52] and str(r[52]).strip():
                try:
                    fh = json.loads(r[52])
                    if isinstance(fh, list) and fh:
                        print(f"  Sample fields in form_history[0]: {list(fh[0].keys()) if isinstance(fh[0], dict) else type(fh[0])}")
                        print(f"  Total entries for {r[1]}: {len(fh)}")
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

        # ═══════════════════════════════════════════════
        # SECTION 15: TRACK CONDITION BREAKDOWN
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 15: TRACK CONDITION BREAKDOWN")
        print("=" * 100)

        cond_data = defaultdict(list)
        for r in settled:
            cond = r[58] or 'Unknown'
            fin = r[15]
            if fin and fin <= 20:
                cond_data[cond].append(fin)

        for cond in sorted(cond_data):
            f = cond_data[cond]
            print(f"  {cond:<25}: n={len(f)}, avg_finish={sum(f)/len(f):.1f}")

        # ═══════════════════════════════════════════════
        # SECTION 16: VENUE BREAKDOWN
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 16: VENUE BREAKDOWN (per venue data quality)")
        print("=" * 100)

        venues = defaultdict(list)
        for r in rows:
            venues[r[57]].append(r)

        for venue in sorted(venues):
            vrows = venues[venue]
            vset = [r for r in vrows if r[15] is not None]
            odds_tab_pop = sum(1 for r in vrows if r[10] and r[10] > 0)
            odds_sb_pop = sum(1 for r in vrows if r[11] and r[11] > 0)
            smap_pop = sum(1 for r in vrows if r[19])
            td_pop = sum(1 for r in vrows if r[32] and str(r[32]).strip())
            dslr_pop = sum(1 for r in vrows if r[30] is not None)
            print(f"  {venue:<20}: total={len(vrows)}, settled={len(vset)}, TAB={odds_tab_pop}, SB={odds_sb_pop}, speedmap={smap_pop}, trackdist={td_pop}, dslr={dslr_pop}")

        # ═══════════════════════════════════════════════
        # SECTION 17: RESULT MARGIN ANALYSIS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 17: RESULT MARGIN COVERAGE")
        print("=" * 100)

        margin_pop = sum(1 for r in settled if r[18] is not None)
        print(f"  result_margin populated: {margin_pop}/{len(settled)} ({margin_pop/len(settled)*100:.1f}%)")
        if margin_pop:
            margins = []
            for r in settled:
                if r[18] is not None:
                    try:
                        margins.append(float(r[18]))
                    except (ValueError, TypeError):
                        pass
            if margins:
                print(f"  margin range: {min(margins):.2f} to {max(margins):.2f}")
                print(f"  margin avg: {sum(margins)/len(margins):.2f}")

        # ═══════════════════════════════════════════════
        # SECTION 18: OUR PICKS vs RESULTS
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 18: OUR PICKS vs RESULTS (selections only)")
        print("=" * 100)

        picks_result = await db.execute(text("""
            SELECT
                p.meeting_id, p.race_number, p.horse_name, p.saddlecloth,
                p.tip_rank, p.bet_type, p.bet_stake,
                p.odds_at_tip, p.place_odds_at_tip,
                p.win_probability, p.place_probability, p.value_rating,
                p.hit, p.pnl, p.settled, p.confidence,
                p.is_puntys_pick,
                r.current_odds, r.opening_odds, r.place_odds,
                r.odds_tab, r.odds_sportsbet,
                r.finish_position, r.win_dividend, r.place_dividend,
                r.speed_map_position, r.barrier, r.weight,
                m.venue, m.track_condition
            FROM picks p
            LEFT JOIN runners r ON r.race_id = p.meeting_id || '-r' || p.race_number
                AND r.saddlecloth = p.saddlecloth
            LEFT JOIN meetings m ON m.id = p.meeting_id
            WHERE m.date >= date('now', '-2 days')
            AND p.pick_type = 'selection'
            ORDER BY m.date, m.venue, p.race_number, p.tip_rank
        """))
        picks = picks_result.fetchall()

        print(f"  Total selections: {len(picks)}")

        total_pnl = 0
        settled_count = 0
        for p in picks:
            venue = p[28] or ''
            settled_flag = p[14]
            pnl = p[13] or 0
            if settled_flag:
                total_pnl += pnl
                settled_count += 1

            odds_tip = p[7] or 0
            tab = p[20] or 0
            sb = p[21] or 0
            curr = p[17] or 0
            wdiv = p[23] or 0
            pdiv = p[24] or 0
            fin = p[22]
            wp = p[9] or 0
            pp = p[10] or 0
            val = p[11] or 0
            smap = p[25] or ''
            cond = (p[29] or '')[:12]

            print(f"  {venue:<14} R{p[1]} #{p[3] or 0:>2} {(p[2] or ''):<22} tip={p[4]} {(p[5] or ''):<10} WP={wp:.1%} PP={pp:.1%} val={val:.2f} tip${odds_tip:.2f} tab${tab:.2f} sb${sb:.2f} curr${curr:.2f} wdiv${wdiv:.2f} pdiv${pdiv:.2f} pos={fin or '-'} hit={p[12]} pnl${pnl:.2f} {smap} {cond}")

        print(f"\n  TOTAL P&L: ${total_pnl:.2f} from {settled_count} settled")

        # ═══════════════════════════════════════════════
        # SECTION 19: PROBABILITY CALIBRATION
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 19: PROBABILITY CALIBRATION (all settled runners)")
        print("=" * 100)

        # Run probability engine implied probs vs actual for ALL runners (not just picks)
        # Use current_odds as proxy for implied probability
        imp_buckets = defaultdict(list)
        for r in settled:
            curr = r[7]
            fin = r[15]
            if curr and curr > 1 and fin and fin <= 20:
                imp_prob = 1.0 / curr
                if imp_prob < 0.05:
                    imp_buckets['<5%'].append(1 if fin == 1 else 0)
                elif imp_prob < 0.10:
                    imp_buckets['5-10%'].append(1 if fin == 1 else 0)
                elif imp_prob < 0.15:
                    imp_buckets['10-15%'].append(1 if fin == 1 else 0)
                elif imp_prob < 0.25:
                    imp_buckets['15-25%'].append(1 if fin == 1 else 0)
                elif imp_prob < 0.40:
                    imp_buckets['25-40%'].append(1 if fin == 1 else 0)
                else:
                    imp_buckets['40%+'].append(1 if fin == 1 else 0)

        print("  Market implied probability vs actual win rate:")
        for bucket in ['<5%', '5-10%', '10-15%', '15-25%', '25-40%', '40%+']:
            if bucket in imp_buckets:
                hits = imp_buckets[bucket]
                actual = sum(hits) / len(hits) * 100
                print(f"    {bucket:<8}: n={len(hits):>3}, actual_win%={actual:.1f}%")

        # ═══════════════════════════════════════════════
        # SECTION 20: SUMMARY OF ISSUES
        # ═══════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("SECTION 20: CRITICAL ISSUES SUMMARY")
        print("=" * 100)

        issues = []

        # Check for completely empty fields
        for i, col in enumerate(cols):
            non_null = sum(1 for r in rows if r[i] is not None and str(r[i]).strip() != '')
            if non_null == 0:
                issues.append(f"EMPTY FIELD: {col} — 0% populated across {total} runners")

        # Check for fields <50% populated
        for i, col in enumerate(cols):
            non_null = sum(1 for r in rows if r[i] is not None and str(r[i]).strip() != '')
            pct = non_null / total * 100 if total else 0
            if 0 < pct < 50:
                issues.append(f"LOW COVERAGE: {col} — {pct:.0f}% populated ({non_null}/{total})")

        # Odds mismatches
        if mismatches:
            issues.append(f"ODDS MISMATCHES: {len(mismatches)} runners where TAB vs SB ratio >3x")

        for issue in issues:
            print(f"  - {issue}")

asyncio.run(main())
