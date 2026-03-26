"""Show all unique exotic_results keys across all races."""
import sqlite3, json
conn = sqlite3.connect("data/punty.db", timeout=30)
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT id, exotic_results FROM races WHERE exotic_results IS NOT NULL AND exotic_results != '{}'")
all_keys = set()
samples = {}
for r in c.fetchall():
    exr = json.loads(r["exotic_results"])
    for k in exr:
        all_keys.add(k)
        if k not in samples:
            samples[k] = f"{r['id']}: {exr[k]}"
print("All exotic_results keys:")
for k in sorted(all_keys):
    print(f"  {k:30s} e.g. {samples.get(k, '?')}")
conn.close()
