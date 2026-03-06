"""Build condition-lookup tables for the tissue probability engine.

Queries data/backtest.db (221K runners, 22K races) to produce empirical
lookup tables that power the tissue-based probability model.

Output: punty/data/tissue_tables.json

Tables built:
  1. condition_multipliers  — (distance × condition × barrier_zone × pace) → multiplier
  2. career_bands           — career win rate band → multiplier
  3. form_recency           — last_five pattern → multiplier
  4. specialist_tables      — track/distance and condition specialist → multiplier
  5. first_up_second_up     — spell pattern → multiplier
  6. venue_overrides        — venue-specific barrier/distance combos
  7. field_size_adjustment   — field size → baseline win rate

Each entry stores: multiplier, n (sample size), win_rate, place_rate.
Entries below MIN_N are excluded; lookups use fallback chains.

Usage:
    python scripts/build_tissue_tables.py
"""

import duckdb
import json
import sys
from pathlib import Path

# Minimum sample size for a table entry to be included
MIN_N = 80
# For venue-specific overrides, need more data
MIN_N_VENUE = 150

DB_PATH = "data/backtest.db"
OUTPUT_PATH = Path("punty/data/tissue_tables.json")


def connect():
    return duckdb.connect(DB_PATH, read_only=True)


def _dist_bucket(distance: int) -> str:
    if distance <= 1100:
        return "sprint"
    if distance <= 1399:
        return "short"
    if distance <= 1799:
        return "middle"
    if distance <= 2199:
        return "classic"
    return "staying"


def _cond_bucket(cond: str) -> str:
    c = (cond or "").lower().strip()
    if "heavy" in c:
        return "heavy"
    if "soft" in c:
        return "soft"
    if "synthetic" in c or "synth" in c:
        return "synthetic"
    if "firm" in c:
        return "good"  # firm and good grouped
    return "good"


def _barrier_zone(barrier: int, field_size: int) -> str:
    if field_size <= 1:
        return "inside"
    relative = (barrier - 1) / max(field_size - 1, 1)
    if relative <= 0.30:
        return "inside"
    if relative <= 0.65:
        return "middle"
    return "outside"


def _rail_bucket(rail: str) -> str:
    """Classify rail position into inside/true/outside."""
    r = (rail or "").lower().strip()
    if not r or r == "0" or "true" in r:
        return "true"
    # Parse "+Xm" or "out Xm" patterns
    import re
    m = re.search(r'(?:out\s*|[+])(\d+)', r)
    if m:
        offset = int(m.group(1))
        if offset >= 6:
            return "outside"
        elif offset >= 3:
            return "mid_rail"
        return "true"
    return "true"


# ──────────────────────────────────────────────
# Table 1: Condition Multipliers
# (distance × condition × barrier_zone × pace_position) → multiplier
# ──────────────────────────────────────────────

def build_condition_multipliers(conn) -> dict:
    """Build the core cross-reference lookup table."""
    print("Building condition multipliers...")

    rows = conn.execute("""
        WITH base AS (
            SELECT
                r.finish_position,
                ra.distance,
                COALESCE(ra.track_condition, m.track_condition, 'Good') as track_cond,
                r.barrier,
                ra.field_size,
                r.speed_map_position,
                r.current_odds
            FROM runners r
            JOIN races ra ON ra.id = r.race_id
            JOIN meetings m ON m.id = ra.meeting_id
            WHERE r.finish_position IS NOT NULL
              AND r.scratched = 0
              AND r.current_odds > 1
              AND r.barrier > 0
              AND ra.distance IS NOT NULL
        )
        SELECT
            distance, track_cond, barrier, field_size,
            speed_map_position, finish_position
        FROM base
    """).fetchall()

    # Aggregate into buckets
    from collections import defaultdict
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "places": 0})

    for distance, track_cond, barrier, field_size, pace_pos, finish_pos in rows:
        dist = _dist_bucket(distance)
        cond = _cond_bucket(track_cond)
        bzone = _barrier_zone(barrier, field_size or 10)
        pace = pace_pos or "unknown"

        # 4-way key
        key4 = f"{dist}|{cond}|{bzone}|{pace}"
        buckets[key4]["n"] += 1
        if finish_pos == 1:
            buckets[key4]["wins"] += 1
        if finish_pos <= 3:
            buckets[key4]["places"] += 1

        # 3-way fallbacks
        for key3 in [f"{dist}|{cond}|{bzone}", f"{dist}|{cond}|{pace}",
                      f"{dist}|{bzone}|{pace}"]:
            buckets[key3]["n"] += 1
            if finish_pos == 1:
                buckets[key3]["wins"] += 1
            if finish_pos <= 3:
                buckets[key3]["places"] += 1

        # 2-way fallbacks
        for key2 in [f"{dist}|{cond}", f"{dist}|{bzone}", f"{dist}|{pace}",
                      f"{cond}|{bzone}", f"{cond}|{pace}"]:
            buckets[key2]["n"] += 1
            if finish_pos == 1:
                buckets[key2]["wins"] += 1
            if finish_pos <= 3:
                buckets[key2]["places"] += 1

        # 1-way
        for key1 in [dist, cond, bzone, pace]:
            buckets[key1]["n"] += 1
            if finish_pos == 1:
                buckets[key1]["wins"] += 1
            if finish_pos <= 3:
                buckets[key1]["places"] += 1

    # Calculate overall baseline win rate
    total_n = sum(1 for _ in rows)
    total_wins = sum(1 for r in rows if r[5] == 1)
    baseline_wr = total_wins / total_n if total_n > 0 else 0.10
    print(f"  Baseline win rate: {baseline_wr:.4f} ({total_wins}/{total_n})")

    # Convert to multipliers (relative to baseline)
    result = {}
    for key, data in buckets.items():
        if data["n"] < MIN_N:
            continue
        wr = data["wins"] / data["n"]
        pr = data["places"] / data["n"]
        mult = wr / baseline_wr if baseline_wr > 0 else 1.0
        result[key] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(pr, 4),
            "n": data["n"],
        }

    # Sort by key specificity (4-way first, then 3, 2, 1)
    sorted_result = dict(sorted(result.items(), key=lambda x: (-x[0].count("|"), x[0])))

    pipe_counts = {i: sum(1 for k in sorted_result if k.count("|") == i) for i in range(4)}
    print(f"  Entries: {len(sorted_result)} total — "
          f"4-way: {pipe_counts.get(3,0)}, 3-way: {pipe_counts.get(2,0)}, "
          f"2-way: {pipe_counts.get(1,0)}, 1-way: {pipe_counts.get(0,0)}")
    print(f"  Baseline: {baseline_wr:.4f}")

    return {"entries": sorted_result, "baseline_win_rate": round(baseline_wr, 5)}


# ──────────────────────────────────────────────
# Table 2: Career Bands
# ──────────────────────────────────────────────

def build_career_bands(conn) -> dict:
    """Career win rate → tissue multiplier."""
    print("Building career bands...")

    rows = conn.execute("""
        WITH parsed AS (
            SELECT
                r.finish_position,
                TRY_CAST(SPLIT_PART(r.career_record, ': ', 1) AS INT) as starts,
                TRY_CAST(SPLIT_PART(SPLIT_PART(r.career_record, ': ', 2), '-', 1) AS INT) as wins_career,
                r.current_odds
            FROM runners r
            WHERE r.finish_position IS NOT NULL
              AND r.scratched = 0
              AND r.current_odds > 1
              AND r.career_record IS NOT NULL
        )
        SELECT
            CASE
                WHEN starts < 3 THEN 'lightly_raced'
                WHEN 1.0 * wins_career / starts >= 0.30 THEN 'elite_30pct'
                WHEN 1.0 * wins_career / starts >= 0.20 THEN 'good_20pct'
                WHEN 1.0 * wins_career / starts >= 0.10 THEN 'average_10pct'
                WHEN wins_career > 0 THEN 'below_avg'
                ELSE 'maiden_career'
            END as band,
            COUNT(*) as n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as places,
            ROUND(AVG(current_odds), 2) as avg_odds
        FROM parsed
        WHERE starts IS NOT NULL AND starts > 0
        GROUP BY band
        ORDER BY 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*) DESC
    """).fetchall()

    # Also get baseline
    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    result = {}
    for band, n, wins, places, avg_odds in rows:
        wr = wins / n
        pr = places / n
        mult = wr / baseline if baseline > 0 else 1.0
        result[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(pr, 4),
            "n": n,
            "avg_odds": avg_odds,
        }
        print(f"  {band:20s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    # Career starts experience bands (separate dimension)
    starts_rows = conn.execute("""
        WITH parsed AS (
            SELECT
                r.finish_position,
                TRY_CAST(SPLIT_PART(r.career_record, ': ', 1) AS INT) as starts
            FROM runners r
            WHERE r.finish_position IS NOT NULL AND r.scratched = 0
              AND r.current_odds > 1 AND r.career_record IS NOT NULL
        )
        SELECT
            CASE
                WHEN starts <= 3 THEN 'starts_1_3'
                WHEN starts <= 10 THEN 'starts_4_10'
                WHEN starts <= 20 THEN 'starts_11_20'
                WHEN starts <= 40 THEN 'starts_21_40'
                ELSE 'starts_40plus'
            END as exp_band,
            COUNT(*) as n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as places
        FROM parsed
        WHERE starts IS NOT NULL AND starts > 0
        GROUP BY exp_band
    """).fetchall()

    experience = {}
    for band, n, wins, places in starts_rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        experience[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }

    return {"bands": result, "experience": experience, "baseline": round(baseline, 5)}


# ──────────────────────────────────────────────
# Table 3: Form Recency (last_five patterns)
# ──────────────────────────────────────────────

def build_form_recency(conn) -> dict:
    """Last-five form → tissue multiplier based on first char (most recent run)."""
    print("Building form recency...")

    rows = conn.execute("""
        SELECT
            CASE
                WHEN last_five LIKE '1%' THEN 'last_1st'
                WHEN last_five LIKE '2%' THEN 'last_2nd'
                WHEN last_five LIKE '3%' THEN 'last_3rd'
                WHEN last_five LIKE '4%' OR last_five LIKE '5%' THEN 'last_4th5th'
                WHEN last_five LIKE '6%' OR last_five LIKE '7%' OR last_five LIKE '8%' THEN 'last_mid'
                WHEN last_five LIKE 'x%' OR last_five LIKE 'X%' THEN 'last_x'
                ELSE 'last_back'
            END as form_band,
            COUNT(*) as n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as places
        FROM runners
        WHERE finish_position IS NOT NULL AND scratched = 0
          AND current_odds > 1 AND last_five IS NOT NULL AND LENGTH(last_five) >= 1
        GROUP BY form_band
    """).fetchall()

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    result = {}
    for band, n, wins, places in rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        result[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {band:15s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    # Also do 2-char pattern (last two runs together)
    rows2 = conn.execute("""
        SELECT
            CASE
                WHEN last_five LIKE '11%' THEN 'seq_11'
                WHEN last_five LIKE '12%' OR last_five LIKE '21%' THEN 'seq_top2'
                WHEN last_five LIKE '1%' THEN 'seq_1x'
                WHEN last_five LIKE '23%' OR last_five LIKE '32%' OR last_five LIKE '22%' THEN 'seq_2_3'
                WHEN SUBSTRING(last_five, 1, 1) IN ('1','2','3') AND SUBSTRING(last_five, 2, 1) IN ('1','2','3') THEN 'seq_top3_top3'
                ELSE 'seq_other'
            END as seq_band,
            COUNT(*) as n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as places
        FROM runners
        WHERE finish_position IS NOT NULL AND scratched = 0
          AND current_odds > 1 AND last_five IS NOT NULL AND LENGTH(last_five) >= 2
        GROUP BY seq_band
        HAVING COUNT(*) >= 100
    """).fetchall()

    sequences = {}
    for band, n, wins, places in rows2:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        sequences[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }

    # Form trend: improving vs declining
    trend_rows = conn.execute("""
        WITH form_trend AS (
            SELECT
                finish_position,
                CASE
                    WHEN LENGTH(last_five) >= 3
                        AND TRY_CAST(SUBSTRING(last_five, 1, 1) AS INT) IS NOT NULL
                        AND TRY_CAST(SUBSTRING(last_five, 2, 1) AS INT) IS NOT NULL
                        AND TRY_CAST(SUBSTRING(last_five, 3, 1) AS INT) IS NOT NULL
                    THEN CASE
                        WHEN TRY_CAST(SUBSTRING(last_five, 1, 1) AS INT) <
                             TRY_CAST(SUBSTRING(last_five, 2, 1) AS INT)
                         AND TRY_CAST(SUBSTRING(last_five, 2, 1) AS INT) <
                             TRY_CAST(SUBSTRING(last_five, 3, 1) AS INT)
                        THEN 'improving'
                        WHEN TRY_CAST(SUBSTRING(last_five, 1, 1) AS INT) >
                             TRY_CAST(SUBSTRING(last_five, 2, 1) AS INT)
                         AND TRY_CAST(SUBSTRING(last_five, 2, 1) AS INT) >
                             TRY_CAST(SUBSTRING(last_five, 3, 1) AS INT)
                        THEN 'declining'
                        ELSE 'mixed'
                    END
                    ELSE 'unknown'
                END as trend
            FROM runners
            WHERE finish_position IS NOT NULL AND scratched = 0
              AND current_odds > 1 AND last_five IS NOT NULL AND LENGTH(last_five) >= 3
        )
        SELECT trend, COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM form_trend
        WHERE trend != 'unknown'
        GROUP BY trend
    """).fetchall()

    trends = {}
    for trend, n, wins, places in trend_rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        trends[trend] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  trend_{trend:12s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    return {"recent_form": result, "sequences": sequences, "trends": trends}


# ──────────────────────────────────────────────
# Table 4: Specialist Tables (T/D and condition)
# ──────────────────────────────────────────────

def build_specialist_tables(conn) -> dict:
    """Track/distance specialist and condition specialist multipliers."""
    print("Building specialist tables...")

    # Track+distance specialist
    td_rows = conn.execute("""
        WITH parsed_td AS (
            SELECT r.finish_position,
                TRY_CAST(r.track_dist_stats->>'starts' AS INT) as td_starts,
                TRY_CAST(r.track_dist_stats->>'wins' AS INT) as td_wins,
                r.current_odds
            FROM runners r
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1
              AND r.scratched = 0 AND r.track_dist_stats IS NOT NULL
        )
        SELECT
            CASE
                WHEN td_starts IS NULL OR td_starts < 2 THEN 'td_no_form'
                WHEN td_starts >= 2 AND 1.0 * td_wins / td_starts >= 0.30 THEN 'td_specialist_30'
                WHEN td_starts >= 2 AND 1.0 * td_wins / td_starts >= 0.20 THEN 'td_specialist_20'
                WHEN td_wins > 0 THEN 'td_winner'
                ELSE 'td_no_win'
            END as td_band,
            COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed_td
        GROUP BY td_band
        HAVING COUNT(*) >= 100
    """).fetchall()

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    td_specialist = {}
    for band, n, wins, places in td_rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        td_specialist[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {band:20s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    # Condition specialist (good/soft/heavy track stats)
    cond_rows = conn.execute("""
        WITH cond AS (
            SELECT r.finish_position,
                COALESCE(ra.track_condition, m.track_condition) as race_cond,
                r.good_track_stats, r.soft_track_stats, r.heavy_track_stats
            FROM runners r
            JOIN races ra ON ra.id = r.race_id
            JOIN meetings m ON m.id = ra.meeting_id
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1 AND r.scratched = 0
        ),
        parsed AS (
            SELECT finish_position, race_cond,
                CASE
                    WHEN LOWER(race_cond) LIKE '%heavy%' THEN
                        TRY_CAST(heavy_track_stats->>'starts' AS INT)
                    WHEN LOWER(race_cond) LIKE '%soft%' THEN
                        TRY_CAST(soft_track_stats->>'starts' AS INT)
                    ELSE
                        TRY_CAST(good_track_stats->>'starts' AS INT)
                END as cond_starts,
                CASE
                    WHEN LOWER(race_cond) LIKE '%heavy%' THEN
                        TRY_CAST(heavy_track_stats->>'wins' AS INT)
                    WHEN LOWER(race_cond) LIKE '%soft%' THEN
                        TRY_CAST(soft_track_stats->>'wins' AS INT)
                    ELSE
                        TRY_CAST(good_track_stats->>'wins' AS INT)
                END as cond_wins
            FROM cond
        )
        SELECT
            CASE
                WHEN cond_starts IS NULL OR cond_starts < 3 THEN 'cond_no_form'
                WHEN 1.0 * cond_wins / cond_starts >= 0.25 THEN 'cond_specialist_25'
                WHEN 1.0 * cond_wins / cond_starts >= 0.15 THEN 'cond_specialist_15'
                WHEN cond_wins > 0 THEN 'cond_winner'
                ELSE 'cond_no_win'
            END as cond_band,
            COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed
        GROUP BY cond_band
        HAVING COUNT(*) >= 100
    """).fetchall()

    cond_specialist = {}
    for band, n, wins, places in cond_rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        cond_specialist[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {band:20s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    return {"track_distance": td_specialist, "condition": cond_specialist}


# ──────────────────────────────────────────────
# Table 5: First-Up / Second-Up
# ──────────────────────────────────────────────

def build_spell_tables(conn) -> dict:
    """First-up and second-up specialist data."""
    print("Building spell tables...")

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    fu_rows = conn.execute("""
        WITH parsed_fu AS (
            SELECT r.finish_position,
                TRY_CAST(r.first_up_stats->>'starts' AS INT) as fu_starts,
                TRY_CAST(r.first_up_stats->>'wins' AS INT) as fu_wins,
                r.days_since_last_run
            FROM runners r
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1
              AND r.scratched = 0 AND r.first_up_stats IS NOT NULL
        )
        SELECT
            CASE
                WHEN fu_starts IS NULL OR fu_starts < 2 THEN 'fu_few_starts'
                WHEN 1.0 * fu_wins / fu_starts >= 0.30 THEN 'fu_strong_30'
                WHEN 1.0 * fu_wins / fu_starts >= 0.15 THEN 'fu_ok_15'
                WHEN fu_wins > 0 THEN 'fu_weak'
                ELSE 'fu_never_won'
            END as fu_band,
            COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed_fu
        WHERE fu_starts IS NOT NULL AND fu_starts > 0
        GROUP BY fu_band
        HAVING COUNT(*) >= 100
    """).fetchall()

    first_up = {}
    for band, n, wins, places in fu_rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        first_up[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {band:20s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    # Second-up
    su_rows = conn.execute("""
        WITH parsed_su AS (
            SELECT r.finish_position,
                TRY_CAST(r.second_up_stats->>'starts' AS INT) as su_starts,
                TRY_CAST(r.second_up_stats->>'wins' AS INT) as su_wins
            FROM runners r
            WHERE r.finish_position IS NOT NULL AND r.current_odds > 1
              AND r.scratched = 0 AND r.second_up_stats IS NOT NULL
        )
        SELECT
            CASE
                WHEN su_starts IS NULL OR su_starts < 2 THEN 'su_few_starts'
                WHEN 1.0 * su_wins / su_starts >= 0.25 THEN 'su_strong_25'
                WHEN 1.0 * su_wins / su_starts >= 0.12 THEN 'su_ok_12'
                WHEN su_wins > 0 THEN 'su_weak'
                ELSE 'su_never_won'
            END as su_band,
            COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM parsed_su
        WHERE su_starts IS NOT NULL AND su_starts > 0
        GROUP BY su_band
        HAVING COUNT(*) >= 100
    """).fetchall()

    second_up = {}
    for band, n, wins, places in su_rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        second_up[band] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {band:20s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    return {"first_up": first_up, "second_up": second_up}


# ──────────────────────────────────────────────
# Table 6: Venue-Specific Overrides
# ──────────────────────────────────────────────

def build_venue_overrides(conn) -> dict:
    """Venue × distance × barrier zone overrides where data is sufficient."""
    print("Building venue overrides...")

    rows = conn.execute("""
        SELECT
            m.venue,
            ra.distance,
            r.barrier,
            ra.field_size,
            r.finish_position
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        JOIN meetings m ON m.id = ra.meeting_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND r.barrier > 0
          AND ra.distance IS NOT NULL
    """).fetchall()

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    from collections import defaultdict
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "places": 0})

    for venue, distance, barrier, field_size, finish_pos in rows:
        v = venue.lower().strip()
        dist = _dist_bucket(distance)
        bzone = _barrier_zone(barrier, field_size or 10)

        key = f"{v}|{dist}|{bzone}"
        buckets[key]["n"] += 1
        if finish_pos == 1:
            buckets[key]["wins"] += 1
        if finish_pos <= 3:
            buckets[key]["places"] += 1

        # Venue + distance fallback
        key2 = f"{v}|{dist}"
        buckets[key2]["n"] += 1
        if finish_pos == 1:
            buckets[key2]["wins"] += 1
        if finish_pos <= 3:
            buckets[key2]["places"] += 1

    result = {}
    for key, data in buckets.items():
        if data["n"] < MIN_N_VENUE:
            continue
        wr = data["wins"] / data["n"]
        pr = data["places"] / data["n"]
        mult = wr / baseline if baseline > 0 else 1.0
        result[key] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(pr, 4),
            "n": data["n"],
        }

    print(f"  Venue overrides: {len(result)} entries (n>={MIN_N_VENUE})")

    # Show most extreme overrides
    sorted_by_mult = sorted(result.items(), key=lambda x: x[1]["mult"])
    print("  Worst venue combos:")
    for k, v in sorted_by_mult[:5]:
        print(f"    {k:40s}: mult={v['mult']:.3f}  wr={v['win_rate']:.3%}  n={v['n']}")
    print("  Best venue combos:")
    for k, v in sorted_by_mult[-5:]:
        print(f"    {k:40s}: mult={v['mult']:.3f}  wr={v['win_rate']:.3%}  n={v['n']}")

    return result


# ──────────────────────────────────────────────
# Table 7: Field Size Adjustment
# ──────────────────────────────────────────────

def build_field_size_table(conn) -> dict:
    """Field size → expected baseline win rate (for normalisation)."""
    print("Building field size table...")

    rows = conn.execute("""
        SELECT
            ra.field_size,
            COUNT(*) n,
            SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND ra.field_size IS NOT NULL AND ra.field_size > 0
        GROUP BY ra.field_size
        HAVING COUNT(*) >= 50
        ORDER BY ra.field_size
    """).fetchall()

    result = {}
    for fs, n, wins, places in rows:
        wr = wins / n
        result[str(fs)] = {
            "expected_win_rate": round(wr, 4),
            "expected_place_rate": round(places / n, 4),
            "n": n,
        }
        if 4 <= fs <= 18:
            print(f"  field={fs:2d}: wr={wr:.3%}  pr={places/n:.3%}  n={n}")

    return result


# ──────────────────────────────────────────────
# Table 8: Odds Movement Signal
# ──────────────────────────────────────────────

def build_movement_table(conn) -> dict:
    """Odds movement (opening → current) as confirmation signal."""
    print("Building odds movement table...")

    rows = conn.execute("""
        SELECT
            CASE
                WHEN current_odds < opening_odds * 0.80 THEN 'steamer_20pct'
                WHEN current_odds < opening_odds * 0.95 THEN 'firmed_5_20'
                WHEN current_odds <= opening_odds * 1.05 THEN 'steady'
                WHEN current_odds <= opening_odds * 1.20 THEN 'drifted_5_20'
                ELSE 'blowout_20pct'
            END as movement,
            COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners
        WHERE finish_position IS NOT NULL AND scratched = 0
          AND current_odds > 1 AND opening_odds > 1
          AND opening_odds IS NOT NULL
        GROUP BY movement
        ORDER BY MIN(current_odds / opening_odds)
    """).fetchall()

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0
          AND current_odds > 1 AND opening_odds > 1 AND opening_odds IS NOT NULL
    """).fetchone()[0]

    result = {}
    for movement, n, wins, places in rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        result[movement] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {movement:20s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    return result


# ──────────────────────────────────────────────
# Table 9: Horse Profile (age × sex)
# ──────────────────────────────────────────────

def build_horse_profile_table(conn) -> dict:
    """Age × sex → tissue multiplier."""
    print("Building horse profile table...")

    rows = conn.execute("""
        SELECT
            CASE
                WHEN horse_age <= 2 THEN '2yo'
                WHEN horse_age = 3 THEN '3yo'
                WHEN horse_age = 4 THEN '4yo'
                WHEN horse_age = 5 THEN '5yo'
                WHEN horse_age = 6 THEN '6yo'
                WHEN horse_age >= 7 THEN '7yo_plus'
                ELSE 'unknown'
            END as age_band,
            CASE
                WHEN LOWER(horse_sex) IN ('m', 'c', 'r', 'h') THEN 'male'
                WHEN LOWER(horse_sex) IN ('f', 'mare') THEN 'female'
                WHEN LOWER(horse_sex) IN ('g', 'gelding') THEN 'gelding'
                ELSE 'other'
            END as sex_band,
            COUNT(*) n,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners
        WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
          AND horse_age IS NOT NULL
        GROUP BY age_band, sex_band
        HAVING COUNT(*) >= 100
    """).fetchall()

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    result = {}
    for age, sex, n, wins, places in rows:
        key = f"{age}|{sex}"
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        result[key] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        if n >= 500:
            print(f"  {key:15s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    return result


# ──────────────────────────────────────────────
# Table 10: Weight Carried
# ──────────────────────────────────────────────

def build_weight_table(conn) -> dict:
    """Weight relative to field average → multiplier."""
    print("Building weight table...")

    rows = conn.execute("""
        SELECT
            CASE
                WHEN r.weight < field.avg_wt - 3 THEN 'very_light'
                WHEN r.weight < field.avg_wt - 1 THEN 'light'
                WHEN r.weight <= field.avg_wt + 1 THEN 'average'
                WHEN r.weight <= field.avg_wt + 3 THEN 'heavy'
                ELSE 'very_heavy'
            END as wt_class,
            COUNT(*) n,
            SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins,
            SUM(CASE WHEN r.finish_position <= 3 THEN 1 ELSE 0 END) places
        FROM runners r
        JOIN (
            SELECT race_id, AVG(weight) as avg_wt
            FROM runners WHERE weight > 0 AND scratched = 0 GROUP BY race_id
        ) field ON field.race_id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.weight > 0
          AND r.current_odds > 1 AND r.scratched = 0
        GROUP BY wt_class
    """).fetchall()

    baseline = conn.execute("""
        SELECT 1.0 * SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) / COUNT(*)
        FROM runners WHERE finish_position IS NOT NULL AND scratched = 0 AND current_odds > 1
    """).fetchone()[0]

    result = {}
    for wt, n, wins, places in rows:
        wr = wins / n
        mult = wr / baseline if baseline > 0 else 1.0
        result[wt] = {
            "mult": round(mult, 3),
            "win_rate": round(wr, 4),
            "place_rate": round(places / n, 4),
            "n": n,
        }
        print(f"  {wt:15s}: mult={mult:.3f}  wr={wr:.3%}  n={n}")

    return result


# ──────────────────────────────────────────────
# Table 11: Class Level
# ──────────────────────────────────────────────

def build_class_table(conn) -> dict:
    """Race class × distance interaction."""
    print("Building class table...")

    rows = conn.execute("""
        SELECT
            CASE
                WHEN LOWER(ra.class) LIKE '%maiden%' THEN 'maiden'
                WHEN LOWER(ra.class) LIKE '%class 1%' OR LOWER(ra.class) LIKE '%cl1%' THEN 'class1'
                WHEN LOWER(ra.class) LIKE '%class 2%' OR LOWER(ra.class) LIKE '%cl2%' THEN 'class2'
                WHEN LOWER(ra.class) LIKE '%benchmark%' OR LOWER(ra.class) LIKE '%bm%' THEN
                    CASE
                        WHEN REGEXP_EXTRACT(LOWER(ra.class), '(?:bm|benchmark)\s*(\d+)', 1) != '' THEN
                            CASE
                                WHEN TRY_CAST(REGEXP_EXTRACT(LOWER(ra.class), '(?:bm|benchmark)\s*(\d+)', 1) AS INT) <= 58 THEN 'bm58'
                                WHEN TRY_CAST(REGEXP_EXTRACT(LOWER(ra.class), '(?:bm|benchmark)\s*(\d+)', 1) AS INT) <= 72 THEN 'bm72'
                                ELSE 'open'
                            END
                        ELSE 'bm58'
                    END
                WHEN LOWER(ra.class) LIKE '%group%' OR LOWER(ra.class) LIKE '%listed%'
                    OR LOWER(ra.class) LIKE '%stakes%' THEN 'open'
                WHEN LOWER(ra.class) LIKE '%restricted%' THEN 'restricted'
                ELSE 'other'
            END as class_band,
            CASE
                WHEN ra.distance <= 1200 THEN 'sprint_short'
                WHEN ra.distance <= 1600 THEN 'middle'
                ELSE 'staying'
            END as dist_band,
            COUNT(*) n,
            SUM(CASE WHEN r.finish_position = 1 THEN 1 ELSE 0 END) wins
        FROM runners r
        JOIN races ra ON ra.id = r.race_id
        WHERE r.finish_position IS NOT NULL AND r.scratched = 0
          AND r.current_odds > 1 AND ra.class IS NOT NULL
        GROUP BY class_band, dist_band
        HAVING COUNT(*) >= 200
    """).fetchall()

    # This table captures "in this class at this distance, what's the expected win rate?"
    # Used to adjust baseline expectations, not as a multiplier per se
    result = {}
    for cls, dist, n, wins in rows:
        key = f"{cls}|{dist}"
        wr = wins / n
        result[key] = {
            "expected_win_rate": round(wr, 4),
            "n": n,
        }
        print(f"  {key:25s}: wr={wr:.3%}  n={n}")

    return result


# ──────────────────────────────────────────────
# Main: Build All Tables
# ──────────────────────────────────────────────

def main():
    print(f"Building tissue tables from {DB_PATH}...")
    print(f"Minimum sample size: {MIN_N} (venue: {MIN_N_VENUE})")
    print()

    conn = connect()

    tables = {
        "version": "1.0",
        "source": "data/backtest.db — 221K Proform runners, 2024-2025",
        "min_n": MIN_N,
        "min_n_venue": MIN_N_VENUE,
    }

    tables["condition_multipliers"] = build_condition_multipliers(conn)
    print()
    tables["career_bands"] = build_career_bands(conn)
    print()
    tables["form_recency"] = build_form_recency(conn)
    print()
    tables["specialist_tables"] = build_specialist_tables(conn)
    print()
    tables["spell_tables"] = build_spell_tables(conn)
    print()
    tables["venue_overrides"] = build_venue_overrides(conn)
    print()
    tables["field_size"] = build_field_size_table(conn)
    print()
    tables["odds_movement"] = build_movement_table(conn)
    print()
    tables["horse_profile"] = build_horse_profile_table(conn)
    print()
    tables["weight_carried"] = build_weight_table(conn)
    print()
    tables["class_table"] = build_class_table(conn)

    conn.close()

    # Summary
    print()
    print("=" * 60)
    print("TISSUE TABLE SUMMARY")
    print("=" * 60)
    cm = tables["condition_multipliers"]
    print(f"  Condition multipliers: {len(cm['entries'])} entries")
    print(f"  Career bands:         {len(tables['career_bands']['bands'])} bands + {len(tables['career_bands']['experience'])} experience")
    print(f"  Form recency:         {len(tables['form_recency']['recent_form'])} + {len(tables['form_recency']['sequences'])} sequences + {len(tables['form_recency']['trends'])} trends")
    print(f"  Specialist:           {len(tables['specialist_tables']['track_distance'])} T/D + {len(tables['specialist_tables']['condition'])} cond")
    print(f"  Spell tables:         {len(tables['spell_tables']['first_up'])} FU + {len(tables['spell_tables']['second_up'])} SU")
    print(f"  Venue overrides:      {len(tables['venue_overrides'])} entries")
    print(f"  Field size:           {len(tables['field_size'])} sizes")
    print(f"  Odds movement:        {len(tables['odds_movement'])} bands")
    print(f"  Horse profile:        {len(tables['horse_profile'])} combos")
    print(f"  Weight carried:       {len(tables['weight_carried'])} bands")
    print(f"  Class table:          {len(tables['class_table'])} combos")

    # Fallback chain documentation
    tables["fallback_chains"] = {
        "condition_multipliers": [
            "dist|cond|barrier|pace",
            "dist|cond|barrier",
            "dist|cond|pace",
            "dist|barrier|pace",
            "dist|cond",
            "dist|barrier",
            "dist|pace",
            "cond|barrier",
            "dist",
            "cond",
        ],
        "venue_overrides": [
            "venue|dist|barrier",
            "venue|dist",
        ],
    }

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(tables, f, indent=2)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n  Output: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("  Done!")


if __name__ == "__main__":
    main()
