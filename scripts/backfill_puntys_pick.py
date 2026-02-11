"""Backfill is_puntys_pick flag on existing picks by re-parsing content.

Run on server:
  cd /opt/puntyai && source venv/bin/activate && python3 scripts/backfill_puntys_pick.py
"""

import asyncio
import re
import sqlite3

# Connect directly to avoid async complexity
DB_PATH = "data/punty.db"

# Regex to find Punty's Pick line and extract saddlecloth(s)
_PUNTYS_PICK = re.compile(
    r"\*?Punty'?s\s+Pick:?\*?\s*(.+)",
    re.IGNORECASE,
)
_PUNTYS_PICK_HORSE = re.compile(
    r"(?:^|[+])\s*(?:\*?[A-Z][A-Z\s'\u2019\-]+?\*?\s*)?\(No\.(\d+)\)",
    re.IGNORECASE,
)
_RACE_HEADER = re.compile(r"\*Race\s+(\d+)\s*[–\-—]", re.IGNORECASE)


def extract_puntys_picks(raw_content: str) -> dict[int, set[int]]:
    """Extract {race_number: {saddlecloth, ...}} for Punty's Pick lines."""
    result = {}
    race_splits = _RACE_HEADER.split(raw_content)
    if len(race_splits) < 3:
        return result

    for i in range(1, len(race_splits), 2):
        race_num = int(race_splits[i])
        section = race_splits[i + 1] if i + 1 < len(race_splits) else ""

        m = _PUNTYS_PICK.search(section)
        if m:
            pick_line = m.group(1)
            saddlecloths = set()
            for hm in _PUNTYS_PICK_HORSE.finditer(pick_line):
                saddlecloths.add(int(hm.group(1)))
            if saddlecloths:
                result[race_num] = saddlecloths

    return result


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Reset all is_puntys_pick flags
    cursor.execute("UPDATE picks SET is_puntys_pick = 0")
    print(f"Reset all picks to is_puntys_pick=0")

    # Get all approved/sent content with raw_content
    cursor.execute("""
        SELECT id, meeting_id, raw_content
        FROM content
        WHERE status IN ('approved', 'sent')
          AND content_type = 'early_mail'
          AND raw_content IS NOT NULL
    """)
    contents = cursor.fetchall()
    print(f"Processing {len(contents)} content entries...")

    total_marked = 0
    for content in contents:
        meeting_id = content["meeting_id"]
        raw = content["raw_content"]
        if not raw:
            continue

        picks_map = extract_puntys_picks(raw)
        for race_num, saddlecloths in picks_map.items():
            for sc in saddlecloths:
                cursor.execute("""
                    UPDATE picks
                    SET is_puntys_pick = 1
                    WHERE meeting_id = ?
                      AND race_number = ?
                      AND saddlecloth = ?
                      AND pick_type = 'selection'
                """, (meeting_id, race_num, sc))
                total_marked += cursor.rowcount

    conn.commit()
    print(f"Marked {total_marked} picks as Punty's Picks")

    # Verify
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN is_puntys_pick = 1 THEN 1 ELSE 0 END) as puntys_picks,
               SUM(CASE WHEN settled = 1 AND is_puntys_pick = 1 THEN 1 ELSE 0 END) as settled_pp,
               SUM(CASE WHEN settled = 1 AND is_puntys_pick = 1 AND hit = 1 THEN 1 ELSE 0 END) as pp_hits
        FROM picks
        WHERE pick_type = 'selection'
    """)
    row = cursor.fetchone()
    print(f"\nVerification:")
    print(f"  Total selections: {row['total']}")
    print(f"  Punty's Picks: {row['puntys_picks']}")
    print(f"  Settled PP: {row['settled_pp']}")
    print(f"  PP Hits: {row['pp_hits']}")
    if row['settled_pp'] > 0:
        rate = row['pp_hits'] / row['settled_pp'] * 100
        print(f"  PP Strike Rate: {rate:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
