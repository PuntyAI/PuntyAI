"""Build a training database from ALL Proform data.

Parses all 7,500+ Proform JSON files ONCE into a fast SQLite DB.
Future retrains just load from this DB — no more re-parsing JSON every time.

Output: scripts/_backtest_data/training.db
Tables:
  - races: meeting_id, date, venue, track_condition, race_number, distance, class, ...
  - runners: race_key, saddlecloth, horse_name, finish_position, margin, barrier, weight, ...
  - form_entries: runner_key, start_index, position, margin, distance, venue, condition, kri, ...
  - jockey_stats: jockey, venue_type, condition, distance_bucket, starts, wins, places
  - trainer_stats: trainer, venue_type, condition, distance_bucket, starts, wins, places
  - speed_benchmarks: venue, distance_bucket, condition_bucket, median_time, n_races
"""

import json
import os
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punty.calibration_engine import _distance_bucket, _condition_bucket, _class_bucket, _venue_type

PROFORM_BASE = "D:/Punty/DatafromProform"
DB_PATH = os.path.join(os.path.dirname(__file__), "_backtest_data", "training.db")


def safe_float(v, default=0.0):
    if v is None or v == "" or v == "None":
        return default
    try:
        return float(v)
    except:
        return default


def safe_int(v, default=0):
    if v is None or v == "" or v == "None":
        return default
    try:
        return int(float(v))
    except:
        return default


def parse_time(t):
    """Parse OfficialRaceTime like '00:01:02.3400000' to seconds."""
    if not t or t == "00:00:00":
        return 0
    try:
        parts = t.split(":")
        mins = int(parts[1])
        secs = float(parts[2])
        return mins * 60 + secs
    except:
        return 0


def main():
    t0 = time.time()
    print("=" * 60)
    print("BUILD TRAINING DATABASE FROM PROFORM")
    print("=" * 60)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create tables
    c.execute("""CREATE TABLE races (
        race_key TEXT PRIMARY KEY,
        meeting_date TEXT,
        venue TEXT,
        track_condition TEXT,
        distance INTEGER,
        race_class TEXT,
        race_number INTEGER,
        field_size INTEGER,
        dist_bucket TEXT,
        cond_bucket TEXT,
        cls_bucket TEXT,
        venue_type TEXT
    )""")

    c.execute("""CREATE TABLE runners (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_key TEXT,
        saddlecloth INTEGER,
        horse_name TEXT,
        jockey TEXT,
        trainer TEXT,
        barrier INTEGER,
        weight REAL,
        finish_position INTEGER,
        margin REAL,
        kri REAL,
        career_starts INTEGER,
        career_wins INTEGER,
        career_places INTEGER,
        last_five TEXT,
        days_since_run INTEGER,
        horse_age TEXT,
        horse_sex TEXT,
        FOREIGN KEY (race_key) REFERENCES races(race_key)
    )""")

    c.execute("""CREATE TABLE form_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        runner_id INTEGER,
        start_index INTEGER,
        position INTEGER,
        margin REAL,
        distance INTEGER,
        venue TEXT,
        track_condition TEXT,
        kri REAL,
        race_class TEXT,
        barrier INTEGER,
        weight REAL,
        race_time REAL,
        settled INTEGER,
        jockey TEXT,
        FOREIGN KEY (runner_id) REFERENCES runners(id)
    )""")

    c.execute("""CREATE TABLE speed_benchmarks (
        key TEXT PRIMARY KEY,
        venue TEXT,
        dist_bucket TEXT,
        cond_bucket TEXT,
        median_time REAL,
        avg_time REAL,
        n_races INTEGER
    )""")

    c.execute("""CREATE TABLE jt_context (
        key TEXT PRIMARY KEY,
        name TEXT,
        role TEXT,
        venue_type TEXT,
        cond_bucket TEXT,
        dist_bucket TEXT,
        starts INTEGER,
        wins INTEGER,
        places INTEGER,
        place_sr REAL
    )""")

    # Parse all Proform data
    total_files = 0
    total_runners = 0
    total_forms = 0
    total_races = 0
    race_keys_seen = set()

    # Speed + JT accumulators
    speed_data = defaultdict(list)
    jt_data = defaultdict(lambda: {"starts": 0, "wins": 0, "places": 0})

    for year in ["2025", "2026"]:
        year_path = os.path.join(PROFORM_BASE, year)
        if not os.path.isdir(year_path):
            continue
        for month_name in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_name)
            if not os.path.isdir(month_path):
                continue

            form_path = os.path.join(month_path, "Form")
            if not os.path.isdir(form_path):
                # Files might be directly in month folder
                json_files = [f for f in os.listdir(month_path) if f.endswith(".json") and f != "meetings.json" and f != "results.json"]
                if json_files:
                    form_path = month_path
                else:
                    continue

            json_files = [f for f in os.listdir(form_path) if f.endswith(".json")]

            for jf in json_files:
                filepath = os.path.join(form_path, jf)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except:
                    continue

                total_files += 1

                if isinstance(data, list):
                    runners_list = data
                elif isinstance(data, dict):
                    runners_list = data.get("Runners", [data])
                else:
                    continue

                for runner in runners_list:
                    if not isinstance(runner, dict):
                        continue

                    forms = runner.get("Forms", [])
                    if not forms:
                        continue

                    def _str(v):
                        if isinstance(v, dict):
                            return v.get("Name", "") or v.get("name", "") or str(v)
                        return str(v) if v else ""

                    name = _str(runner.get("Name", ""))
                    jockey = _str(runner.get("Jockey", ""))
                    trainer = _str(runner.get("Trainer", ""))
                    barrier = safe_int(runner.get("Barrier"))
                    weight = safe_float(runner.get("Weight"))
                    tab_no = safe_int(runner.get("TabNo"))
                    age = str(runner.get("Age", ""))
                    sex = str(runner.get("Sex", ""))
                    career_starts = safe_int(runner.get("CareerStarts"))
                    career_wins = safe_int(runner.get("CareerWins"))
                    career_secs = safe_int(runner.get("CareerSeconds"))

                    # Build race key from first form entry
                    for fi, form in enumerate(forms):
                        venue = form.get("Track", "")
                        if isinstance(venue, dict):
                            venue = venue.get("Name", "") or venue.get("name", "")
                        venue = str(venue) if venue else ""
                        date = form.get("MeetingDate", "")
                        if date:
                            date = date[:10]  # YYYY-MM-DD
                        distance = safe_int(form.get("Distance"))
                        race_class = form.get("RaceClass", "")
                        position = safe_int(form.get("Position"))
                        margin = safe_float(form.get("Margin"))
                        kri = safe_float(form.get("KRI"))
                        condition = form.get("TrackCondition", "")
                        race_time_str = form.get("OfficialRaceTime", "")
                        race_time = parse_time(race_time_str)
                        settled_pos = safe_int(form.get("Settled") or form.get("InRun", "").split("/")[0] if form.get("InRun") else 0)
                        form_jockey = form.get("Jockey", jockey)
                        form_barrier = safe_int(form.get("Barrier", barrier))
                        form_weight = safe_float(form.get("Weight", weight))

                        if not venue or not date or distance <= 0:
                            continue

                        # Build race key
                        race_key = f"{venue.lower().replace(' ', '-')}-{date}-r{fi}"

                        # Speed benchmark accumulation
                        if position == 1 and race_time > 10:
                            dist_b = _distance_bucket(distance)
                            cond_b = _condition_bucket(condition)
                            speed_key = f"{venue.lower()}|{dist_b}|{cond_b}"
                            speed_data[speed_key].append(race_time)

                        # J/T context accumulation
                        if form_jockey and position > 0:
                            vt = _venue_type(venue)
                            dist_b = _distance_bucket(distance)
                            cond_b = _condition_bucket(condition)
                            jk = f"{form_jockey}|{vt}|{cond_b}|{dist_b}"
                            jt_data[jk]["starts"] += 1
                            if position == 1:
                                jt_data[jk]["wins"] += 1
                            if position <= 3:
                                jt_data[jk]["places"] += 1

                        # Only store first form entry as "this race" result
                        if fi == 0:
                            total_runners += 1

                    # Store runner with first form as result
                    if forms:
                        first = forms[0]
                        venue = first.get("Track", "")
                        if isinstance(venue, dict):
                            venue = venue.get("Name", "") or venue.get("name", "")
                        venue = str(venue) if venue else ""
                        date = first.get("MeetingDate", "")[:10] if first.get("MeetingDate") else ""
                        distance = safe_int(first.get("Distance"))
                        race_class = first.get("RaceClass", "")
                        condition = first.get("TrackCondition", "")
                        position = safe_int(first.get("Position"))
                        margin_val = safe_float(first.get("Margin"))

                        if venue and date and distance > 0 and position > 0:
                            race_key = f"{venue.lower().replace(' ', '-')}-{date}"

                            if race_key not in race_keys_seen:
                                race_keys_seen.add(race_key)
                                total_races += 1
                                c.execute("""INSERT OR IGNORE INTO races VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
                                    race_key, date, venue, condition, distance, race_class, 0,
                                    0, _distance_bucket(distance), _condition_bucket(condition),
                                    _class_bucket(race_class), _venue_type(venue),
                                ))

                            c.execute("""INSERT INTO runners (race_key, saddlecloth, horse_name, jockey, trainer,
                                barrier, weight, finish_position, margin, kri, career_starts, career_wins,
                                career_places, last_five, days_since_run, horse_age, horse_sex)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                                race_key, tab_no, name, jockey, trainer,
                                barrier, weight, position, margin_val,
                                safe_float(first.get("KRI")),
                                career_starts, career_wins, career_secs,
                                (runner.get("Last10") or "")[:5],
                                0, age, sex,
                            ))
                            runner_id = c.lastrowid

                            # Store form history entries
                            for fi2, fh in enumerate(forms[:6]):
                                total_forms += 1
                                fh_venue = fh.get("Track", "")
                                if isinstance(fh_venue, dict):
                                    fh_venue = fh_venue.get("Name", "")
                                fh_cond = fh.get("TrackCondition", "")
                                if isinstance(fh_cond, dict):
                                    fh_cond = str(fh_cond.get("Name", ""))
                                fh_class = fh.get("RaceClass", "")
                                if isinstance(fh_class, dict):
                                    fh_class = str(fh_class.get("Name", ""))
                                fh_jockey = fh.get("Jockey", "")
                                if isinstance(fh_jockey, dict):
                                    fh_jockey = fh_jockey.get("Name", "")
                                c.execute("""INSERT INTO form_entries (runner_id, start_index, position, margin,
                                    distance, venue, track_condition, kri, race_class, barrier, weight,
                                    race_time, settled, jockey) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                                    runner_id, fi2,
                                    safe_int(fh.get("Position")),
                                    safe_float(fh.get("Margin")),
                                    safe_int(fh.get("Distance")),
                                    str(fh_venue or ""),
                                    str(fh_cond or ""),
                                    safe_float(fh.get("KRI")),
                                    str(fh_class or ""),
                                    safe_int(fh.get("Barrier")),
                                    safe_float(fh.get("Weight")),
                                    parse_time(fh.get("OfficialRaceTime", "")),
                                    0, str(fh_jockey or ""),
                                ))

                if total_files % 500 == 0:
                    conn.commit()
                    print(f"  {total_files} files, {total_runners} runners, {total_races} races...")

    conn.commit()

    # Store speed benchmarks
    print(f"\n  Storing {len(speed_data)} speed benchmarks...")
    for key, times in speed_data.items():
        parts = key.split("|")
        if len(parts) == 3 and len(times) >= 3:
            sorted_times = sorted(times)
            median = sorted_times[len(sorted_times) // 2]
            avg = sum(times) / len(times)
            c.execute("INSERT OR REPLACE INTO speed_benchmarks VALUES (?,?,?,?,?,?,?)",
                      (key, parts[0], parts[1], parts[2], median, avg, len(times)))
    conn.commit()

    # Store J/T context
    print(f"  Storing {len(jt_data)} J/T context entries...")
    for key, stats in jt_data.items():
        if stats["starts"] < 3:
            continue
        parts = key.split("|")
        if len(parts) == 4:
            psr = stats["places"] / stats["starts"] if stats["starts"] > 0 else 0
            c.execute("INSERT OR REPLACE INTO jt_context VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (key, parts[0], "jockey", parts[1], parts[2], parts[3],
                       stats["starts"], stats["wins"], stats["places"], psr))
    conn.commit()

    # Create indexes
    c.execute("CREATE INDEX idx_runners_race ON runners(race_key)")
    c.execute("CREATE INDEX idx_form_runner ON form_entries(runner_id)")
    c.execute("CREATE INDEX idx_races_date ON races(meeting_date)")
    conn.commit()

    # Summary
    db_size = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"TRAINING DB BUILT: {DB_PATH}")
    print(f"  Files parsed: {total_files}")
    print(f"  Races: {total_races}")
    print(f"  Runners: {total_runners}")
    print(f"  Form entries: {total_forms}")
    print(f"  Speed benchmarks: {len(speed_data)}")
    print(f"  J/T context: {len(jt_data)}")
    print(f"  DB size: {db_size:.1f}MB")
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
