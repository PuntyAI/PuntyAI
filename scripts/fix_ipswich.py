"""Fix Ipswich R7 — race was abandoned, void picks and fix status."""
import sqlite3

DB_PATH = "/opt/puntyai/data/punty.db"
conn = sqlite3.connect(DB_PATH)

# Mark race as abandoned
conn.execute(
    "UPDATE races SET results_status = 'Abandoned' WHERE id = 'ipswich-2026-02-12-r7'"
)
print("Race status: Paying -> Abandoned")

# Clear bogus dividend data (these were odds, not dividends)
conn.execute(
    "UPDATE runners SET win_dividend = NULL, place_dividend = NULL "
    "WHERE race_id = 'ipswich-2026-02-12-r7'"
)
print("Cleared bogus dividends from runners")

# Settle picks as void ($0 P&L)
result = conn.execute(
    "UPDATE picks SET settled = 1, hit = 0, pnl = 0.0, "
    "settled_at = datetime('now') "
    "WHERE meeting_id LIKE 'ipswich-2026-02-12%' AND race_number = 7 AND settled = 0"
)
print(f"Voided {result.rowcount} picks (settled with $0 P&L)")

conn.commit()
print("Done")
