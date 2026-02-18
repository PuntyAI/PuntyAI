"""Import status report for tomorrow's meetings + PF name mismatch check."""
import asyncio
from sqlalchemy import select, func
from punty.models.database import async_session
from punty.models.meeting import Meeting, Race, Runner


async def report():
    async with async_session() as db:
        q = await db.execute(select(Meeting).where(Meeting.date == "2026-02-19").order_by(Meeting.id))
        meetings = q.scalars().all()
        print("=" * 70)
        print("TOMORROW (2026-02-19) IMPORT STATUS REPORT")
        print("=" * 70)

        total_races = 0
        total_runners = 0
        total_with_form = 0
        total_with_odds = 0
        total_with_speedmap = 0
        total_scratched = 0

        for m in meetings:
            rq = await db.execute(select(Race).where(Race.meeting_id == m.id).order_by(Race.race_number))
            races = rq.scalars().all()

            rrq = await db.execute(select(Runner).where(Runner.race_id.like(f"{m.id}%")))
            runners = rrq.scalars().all()

            active = [r for r in runners if not r.scratched]
            scratched = [r for r in runners if r.scratched]
            with_odds = [r for r in active if r.current_odds and r.current_odds > 1.0]
            with_form = [r for r in active if r.form_history]
            with_sm = [r for r in active if r.speed_map_position]
            with_weight = [r for r in active if r.weight]
            with_jockey = [r for r in active if r.jockey]
            with_trainer = [r for r in active if r.trainer]
            with_barrier = [r for r in active if r.barrier]

            missing_jockey = [r for r in active if not r.jockey]
            missing_weight = [r for r in active if not r.weight]

            total_races += len(races)
            total_runners += len(active)
            total_with_form += len(with_form)
            total_with_odds += len(with_odds)
            total_with_speedmap += len(with_sm)
            total_scratched += len(scratched)

            print(f"\n  {m.venue} ({m.id}):")
            print(f"    Races: {len(races)}  |  Runners: {len(active)} active, {len(scratched)} scratched")
            if active:
                print(f"    Odds:     {len(with_odds):3d}/{len(active)} ({len(with_odds)/len(active)*100:.0f}%)")
                print(f"    Form:     {len(with_form):3d}/{len(active)} ({len(with_form)/len(active)*100:.0f}%)")
                print(f"    Speed:    {len(with_sm):3d}/{len(active)} ({len(with_sm)/len(active)*100:.0f}%)")
                print(f"    Weight:   {len(with_weight):3d}/{len(active)} ({len(with_weight)/len(active)*100:.0f}%)")
                print(f"    Jockey:   {len(with_jockey):3d}/{len(active)} ({len(with_jockey)/len(active)*100:.0f}%)")
                print(f"    Trainer:  {len(with_trainer):3d}/{len(active)} ({len(with_trainer)/len(active)*100:.0f}%)")
                print(f"    Barrier:  {len(with_barrier):3d}/{len(active)} ({len(with_barrier)/len(active)*100:.0f}%)")
            else:
                print("    No active runners")

            if missing_jockey:
                names = ", ".join(r.horse_name for r in missing_jockey[:5])
                print(f"    Missing jockey: {names}")
            if missing_weight:
                names = ", ".join(r.horse_name for r in missing_weight[:5])
                print(f"    Missing weight: {names}")

        if total_runners:
            print(f"\n  TOTALS:")
            print(f"    {total_races} races, {total_runners} active runners, {total_scratched} scratched")
            print(f"    Odds:     {total_with_odds}/{total_runners} ({total_with_odds/total_runners*100:.0f}%)")
            print(f"    Form:     {total_with_form}/{total_runners} ({total_with_form/total_runners*100:.0f}%)")
            print(f"    Speed:    {total_with_speedmap}/{total_runners} ({total_with_speedmap/total_runners*100:.0f}%)")

        print(f"\n  STILL NEEDED (filled by 5am morning scrape + pre-race job):")
        print(f"    - Full odds from TAB/Betfair (currently {total_with_odds}/{total_runners})")
        print(f"    - Speed maps from PuntingForm (currently {total_with_speedmap}/{total_runners})")
        print(f"    - Pre-race odds refresh (2.5h before first race)")
        print(f"    - Weather/track conditions update")

        # --- PF NAME MISMATCH CHECK ---
        print("\n" + "=" * 70)
        print("PF NAME MISMATCH CHECK")
        print("=" * 70)
        print("(Checking horse names that might cause lookup failures)")

        # Common problematic patterns
        issues = []
        for m in meetings:
            rrq = await db.execute(select(Runner).where(Runner.race_id.like(f"{m.id}%")))
            runners = rrq.scalars().all()
            for r in runners:
                if r.scratched:
                    continue
                name = r.horse_name or ""
                # Check for apostrophes, hyphens, special chars
                if "'" in name:
                    issues.append(("apostrophe", m.venue, r.race_id, name))
                if name != name.strip():
                    issues.append(("whitespace", m.venue, r.race_id, name))
                if "  " in name:
                    issues.append(("double_space", m.venue, r.race_id, name))
                # Check jockey name patterns that cause issues
                jockey = r.jockey or ""
                if "(a" in jockey.lower() or "(late" in jockey.lower():
                    # These are fine now after our fix
                    pass
                if not jockey:
                    issues.append(("no_jockey", m.venue, r.race_id, name))

        if issues:
            by_type = {}
            for issue_type, venue, race_id, name in issues:
                by_type.setdefault(issue_type, []).append((venue, race_id, name))
            for itype, items in by_type.items():
                print(f"\n  {itype}: {len(items)} runners")
                for venue, race_id, name in items[:10]:
                    rnum = race_id.split("-r")[-1] if "-r" in race_id else "?"
                    print(f"    {venue} R{rnum}: {name}")
                if len(items) > 10:
                    print(f"    ... and {len(items) - 10} more")
        else:
            print("  No name issues found")

        # Check PF data availability
        print("\n" + "=" * 70)
        print("PF DATA AVAILABILITY CHECK")
        print("=" * 70)
        try:
            from punty.scrapers.punting_form import PuntingFormScraper
            from punty.models.settings import get_api_key
            api_key = await get_api_key(db, "punting_form_api_key")
            if api_key:
                pf = PuntingFormScraper(api_key=api_key)
                for m in meetings:
                    venue = m.venue
                    try:
                        # Try to get PF meeting data
                        pf_data = await pf.get_meeting_data(venue, "2026-02-19")
                        if pf_data:
                            pf_races = pf_data.get("races", [])
                            print(f"  {venue}: PF has {len(pf_races)} races")
                            # Check horse name matches
                            for pf_race in pf_races[:2]:
                                pf_runners = pf_race.get("runners", [])
                                race_num = pf_race.get("race_number", 0)
                                race_id = f"{m.id}-r{race_num}"
                                db_rq = await db.execute(
                                    select(Runner).where(Runner.race_id == race_id)
                                )
                                db_runners = {r.horse_name.upper(): r for r in db_rq.scalars().all() if r.horse_name}
                                mismatches = 0
                                for pf_r in pf_runners:
                                    pf_name = (pf_r.get("horse_name") or pf_r.get("name") or "").upper()
                                    if pf_name and pf_name not in db_runners:
                                        # Try fuzzy match
                                        close = [n for n in db_runners if n.replace("'", "") == pf_name.replace("'", "")]
                                        if not close:
                                            mismatches += 1
                                            print(f"    R{race_num} PF '{pf_name}' not in DB: {list(db_runners.keys())[:5]}...")
                                if mismatches == 0:
                                    print(f"    R{race_num}: all names match")
                        else:
                            print(f"  {venue}: PF returned no data")
                    except Exception as e:
                        print(f"  {venue}: PF error - {e}")
                await pf.close()
            else:
                print("  No PF API key found")
        except Exception as e:
            print(f"  PF check error: {e}")


asyncio.run(report())
