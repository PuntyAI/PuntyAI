import sqlite3

c = sqlite3.connect("/opt/puntyai/data/punty.db")
r = c.cursor()

print("=== HORSE SEX VALUES ===")
r.execute("SELECT horse_sex, COUNT(*) FROM runners GROUP BY horse_sex ORDER BY COUNT(*) DESC")
for row in r.fetchall():
    print(f"  {str(row[0]):10s}: {row[1]}")

print("\n=== PLACE ODDS (should have data for place value) ===")
r.execute("SELECT COUNT(*) FROM runners WHERE place_odds IS NOT NULL AND place_odds != ''")
print(f"  With place_odds: {r.fetchone()[0]}")
r.execute("SELECT COUNT(*) FROM runners")
print(f"  Total runners: {r.fetchone()[0]}")

print("\n=== PUNTING FORM DATA ===")
for col in ["pf_speed_rank", "pf_settle", "pf_map_factor", "pf_jockey_factor"]:
    r.execute(f"SELECT COUNT(*) FROM runners WHERE [{col}] IS NOT NULL")
    print(f"  {col:20s}: {r.fetchone()[0]}")

print("\n=== EMPTY/LOW COLUMNS THAT MATTER FOR PICKS ===")
for col, label in [
    ("dam_sire", "Dam sire (pedigree)"),
    ("trainer_location", "Trainer location"),
    ("trainer_stats", "Trainer stats"),
    ("comment_short", "Short comment"),
    ("stewards_comment", "Stewards comment"),
    ("odds_bet365", "Bet365 odds"),
    ("odds_ladbrokes", "Ladbrokes odds"),
    ("odds_betfair", "Betfair odds"),
]:
    r.execute(f"SELECT COUNT(*) FROM runners WHERE [{col}] IS NOT NULL AND [{col}] != ''")
    print(f"  {label:25s}: {r.fetchone()[0]}")

c.close()
