"""Audit today's meeting data completeness."""
import sqlite3

conn = sqlite3.connect("data/punty.db")
conn.row_factory = sqlite3.Row

meetings = conn.execute(
    "SELECT id, venue, date, track_condition, weather, rail_position "
    "FROM meetings WHERE date = '2026-02-14' ORDER BY venue"
).fetchall()

print(f"=== MEETINGS FOR 2026-02-14: {len(meetings)} ===")

for m in meetings:
    mid = m["id"]
    venue = m["venue"]
    tc = m["track_condition"] or "?"
    wx = m["weather"] or "?"
    rail = m["rail_position"] or "?"
    print(f"\n{'='*60}")
    print(f"MEETING: {venue} ({mid})")
    print(f"  Track: {tc}  Weather: {wx}  Rail: {rail}")

    races = conn.execute(
        "SELECT id, race_number, name, distance FROM races "
        "WHERE meeting_id=? ORDER BY race_number", (mid,)
    ).fetchall()
    print(f"  Races: {len(races)}")

    for race in races:
        rid = race["id"]
        r = conn.execute("""
            SELECT COUNT(*) as total,
                SUM(CASE WHEN scratched=1 THEN 1 ELSE 0 END) as scratched,
                SUM(CASE WHEN current_odds IS NOT NULL AND current_odds > 0 THEN 1 ELSE 0 END) as has_odds,
                SUM(CASE WHEN speed_map_position IS NOT NULL THEN 1 ELSE 0 END) as has_smp,
                SUM(CASE WHEN form IS NOT NULL OR last_five IS NOT NULL THEN 1 ELSE 0 END) as has_form,
                SUM(CASE WHEN weight IS NOT NULL AND weight > 0 THEN 1 ELSE 0 END) as has_weight,
                SUM(CASE WHEN track_dist_stats IS NOT NULL THEN 1 ELSE 0 END) as has_td,
                SUM(CASE WHEN form_history IS NOT NULL THEN 1 ELSE 0 END) as has_fh,
                SUM(CASE WHEN jockey IS NOT NULL AND jockey != '' THEN 1 ELSE 0 END) as has_jockey,
                SUM(CASE WHEN trainer IS NOT NULL AND trainer != '' THEN 1 ELSE 0 END) as has_trainer
            FROM runners WHERE race_id=?
        """, (rid,)).fetchone()

        total = r["total"]
        scr = r["scratched"] or 0
        active = total - scr
        odds = r["has_odds"] or 0
        smp = r["has_smp"] or 0
        form = r["has_form"] or 0
        wt = r["has_weight"] or 0
        td = r["has_td"] or 0
        fh = r["has_fh"] or 0
        jk = r["has_jockey"] or 0
        tr = r["has_trainer"] or 0

        issues = []
        if odds < active:
            issues.append(f"odds:{odds}/{active}")
        if smp < active:
            issues.append(f"smp:{smp}/{active}")
        if form < active:
            issues.append(f"form:{form}/{active}")
        if wt < active:
            issues.append(f"wt:{wt}/{active}")
        if jk < active:
            issues.append(f"jockey:{jk}/{active}")
        if tr < active:
            issues.append(f"trainer:{tr}/{active}")
        if td == 0 and active > 0:
            issues.append(f"td_stats:NONE")
        if fh == 0 and active > 0:
            issues.append(f"form_hist:NONE")

        marker = "OK" if not issues else "GAPS: " + ", ".join(issues)
        print(f"    R{race['race_number']:>2} ({race['distance']}m): {active:>2} active ({scr} scr) | {marker}")

    # Content status
    content = conn.execute(
        "SELECT content_type, status FROM content WHERE meeting_id=? ORDER BY content_type",
        (mid,)
    ).fetchall()
    if content:
        for c in content:
            print(f"  Content: {c['content_type']} -> {c['status']}")
    else:
        print("  Content: NONE")

    # Selected for tipping?
    sel = conn.execute(
        "SELECT value FROM app_settings WHERE key='selected_meetings'"
    ).fetchone()
    if sel:
        import json
        selected = json.loads(sel["value"])
        is_selected = mid in selected
        print(f"  Selected: {'YES' if is_selected else 'NO'}")

conn.close()
