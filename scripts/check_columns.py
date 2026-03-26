import sqlite3

c = sqlite3.connect("/opt/puntyai/data/punty.db")
r = c.cursor()

for table in ["runners", "races", "meetings"]:
    print(f"=== {table.upper()} ===")
    r.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in r.fetchall()]
    r.execute(f"SELECT COUNT(*) FROM {table}")
    total = r.fetchone()[0]
    for col in cols:
        try:
            r.execute(f"SELECT COUNT(*) FROM {table} WHERE [{col}] IS NOT NULL AND CAST([{col}] AS TEXT) != ''")
            count = r.fetchone()[0]
        except:
            count = -1
        pct = (count / total * 100) if total else 0
        flag = "" if pct > 50 else " << LOW" if pct > 0 else " << EMPTY"
        print(f"  {col:30s}: {count:>5}/{total} ({pct:>5.1f}%){flag}")
    print()

c.close()
