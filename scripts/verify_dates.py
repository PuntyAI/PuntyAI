import sqlite3
c = sqlite3.connect("/opt/puntyai/data/punty.db")
r = c.cursor()
r.execute("SELECT MIN(date), COUNT(*) FROM meetings")
row = r.fetchone()
print(f"Earliest: {row[0]}, Total: {row[1]}")
r.execute("SELECT date, COUNT(*) FROM meetings GROUP BY date ORDER BY date")
for x in r.fetchall():
    print(f"  {x[0]}: {x[1]} meetings")
c.close()
