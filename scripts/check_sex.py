import sqlite3
c = sqlite3.connect("/opt/puntyai/data/punty.db")
r = c.cursor()
r.execute("SELECT sex, COUNT(*) FROM runners WHERE sex IS NOT NULL GROUP BY sex ORDER BY COUNT(*) DESC")
for row in r.fetchall():
    print(f"{row[0]:10s}: {row[1]}")
c.close()
