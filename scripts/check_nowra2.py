import sqlite3
conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT * FROM meetings WHERE id = 'nowra-2026-02-08'")
m = c.fetchone()
if m:
    for k in m.keys():
        print(f"  {k}: {m[k]}")
conn.close()
