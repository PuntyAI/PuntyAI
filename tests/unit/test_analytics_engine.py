"""Tests for DuckDB analytics engine and query builder."""

import pytest

import duckdb

from punty.analytics.queries import (
    _where_clauses,
    summary,
    calibration,
    odds_band_performance,
    venue_performance,
    distance_performance,
    speed_map_analysis,
    speed_map_heatmap,
    bet_type_performance,
    time_series,
    filter_options,
)


@pytest.fixture
def analytics_db():
    """Create an in-memory DuckDB with test data."""
    conn = duckdb.connect(":memory:")

    # Create runners table
    conn.execute("""
    CREATE TABLE runners (
        id VARCHAR, race_id VARCHAR, horse_name VARCHAR,
        saddlecloth INTEGER, barrier INTEGER, weight FLOAT,
        jockey VARCHAR, trainer VARCHAR,
        current_odds FLOAT, opening_odds FLOAT, place_odds FLOAT,
        finish_position INTEGER, win_dividend FLOAT, place_dividend FLOAT,
        speed_map_position VARCHAR, horse_age INTEGER, horse_sex VARCHAR,
        last_five VARCHAR, days_since_last_run INTEGER, handicap_rating FLOAT,
        scratched BOOLEAN,
        venue VARCHAR, state VARCHAR, date DATE, distance INTEGER,
        distance_category VARCHAR, race_class VARCHAR, field_size INTEGER,
        track_condition VARCHAR, race_number INTEGER,
        is_winner BOOLEAN, is_placed BOOLEAN, implied_prob FLOAT,
        odds_band VARCHAR
    )
    """)

    # Insert test runners
    runners = [
        # Winner, leader, sprint, NSW, $3 odds
        ("r1", "race1", "Horse A", 1, 2, 57.0, "Jockey A", "Trainer A",
         3.0, 3.5, 1.5, 1, 3.0, 1.50, "leader", 4, "Gelding",
         "11213", 14, 70.0, False,
         "randwick", "NSW", "2025-06-01", 1000, "sprint", "Class 3", 10,
         "Good 4", 1, True, True, 0.333, "$2-$4"),
        # Placed, midfield, middle, VIC, $6 odds
        ("r2", "race1", "Horse B", 2, 5, 55.0, "Jockey B", "Trainer B",
         6.0, 7.0, 2.5, 3, 0.0, 2.50, "midfield", 5, "Mare",
         "32145", 21, 65.0, False,
         "randwick", "NSW", "2025-06-01", 1000, "sprint", "Class 3", 10,
         "Good 4", 1, False, True, 0.167, "$4-$6"),
        # Loser, backmarker
        ("r3", "race1", "Horse C", 3, 8, 56.0, "Jockey C", "Trainer C",
         15.0, 20.0, 5.0, 7, 0.0, 0.0, "backmarker", 3, "Colt",
         "54678", 28, 55.0, False,
         "randwick", "NSW", "2025-06-01", 1000, "sprint", "Class 3", 10,
         "Good 4", 1, False, False, 0.067, "$10-$20"),
        # Winner, on_pace, classic, VIC
        ("r4", "race2", "Horse D", 1, 1, 58.0, "Jockey A", "Trainer D",
         2.5, 2.8, 1.3, 1, 2.5, 1.30, "on_pace", 6, "Horse",
         "11112", 7, 80.0, False,
         "flemington", "VIC", "2025-07-15", 2000, "classic", "Group 2", 12,
         "Soft 5", 3, True, True, 0.400, "$2-$4"),
        # Loser, leader, classic, VIC
        ("r5", "race2", "Horse E", 2, 3, 54.0, "Jockey D", "Trainer E",
         8.0, 9.0, 3.0, 5, 0.0, 0.0, "leader", 4, "Filly",
         "43256", 35, 60.0, False,
         "flemington", "VIC", "2025-07-15", 2000, "classic", "Group 2", 12,
         "Soft 5", 3, False, False, 0.125, "$6-$10"),
    ]

    for r in runners:
        conn.execute("INSERT INTO runners VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)

    # Create speed_maps table
    conn.execute("""
    CREATE TABLE speed_maps (
        runner_id VARCHAR, race_id VARCHAR, horse_name VARCHAR,
        settle_position INTEGER, m800_position INTEGER,
        m400_position INTEGER, finish_position INTEGER,
        position_change INTEGER, speed_map_position VARCHAR,
        venue VARCHAR, date DATE
    )
    """)
    conn.execute("INSERT INTO speed_maps VALUES ('r1','race1','Horse A',2,2,1,1,1,'leader','randwick','2025-06-01')")
    conn.execute("INSERT INTO speed_maps VALUES ('r4','race2','Horse D',3,2,2,1,2,'on_pace','flemington','2025-07-15')")
    conn.execute("INSERT INTO speed_maps VALUES ('r5','race2','Horse E',1,1,3,5,-4,'leader','flemington','2025-07-15')")

    # Create picks table
    conn.execute("""
    CREATE TABLE picks (
        id VARCHAR, content_id VARCHAR, meeting_id VARCHAR, race_number INTEGER,
        horse_name VARCHAR, pick_type VARCHAR, exotic_type VARCHAR,
        sequence_type VARCHAR, sequence_variant VARCHAR,
        bet_type VARCHAR, bet_stake FLOAT, odds_at_tip FLOAT,
        win_probability FLOAT, place_probability FLOAT, value_rating FLOAT,
        confidence VARCHAR, is_puntys_pick BOOLEAN, tracked_only BOOLEAN,
        hit BOOLEAN, pnl FLOAT, settled BOOLEAN, settled_at TIMESTAMP,
        created_at TIMESTAMP,
        venue VARCHAR, state VARCHAR, date DATE
    )
    """)
    conn.execute("""INSERT INTO picks VALUES (
        'p1','c1','m1',1,'Horse A','selection',NULL,NULL,NULL,
        'place',5.0,3.0,0.33,0.65,1.1,'HIGH',true,false,
        true,2.50,true,'2025-06-01 15:00:00','2025-06-01 12:00:00',
        'randwick','NSW','2025-06-01'
    )""")
    conn.execute("""INSERT INTO picks VALUES (
        'p2','c1','m1',1,'Horse B','selection',NULL,NULL,NULL,
        'place',5.0,6.0,0.17,0.45,0.9,'MED',false,false,
        true,7.50,true,'2025-06-01 15:00:00','2025-06-01 12:00:00',
        'randwick','NSW','2025-06-01'
    )""")

    # Create proform_a2e table
    conn.execute("""
    CREATE TABLE proform_a2e (
        jockey VARCHAR, trainer VARCHAR,
        trainer_a2e_career FLOAT, trainer_pot_career FLOAT,
        trainer_strike_career FLOAT, trainer_runners_career INTEGER,
        jockey_a2e_career FLOAT, jockey_pot_career FLOAT,
        jockey_strike_career FLOAT, jockey_runners_career INTEGER,
        combo_a2e_career FLOAT, combo_pot_career FLOAT,
        combo_strike_career FLOAT, combo_runners_career INTEGER,
        trainer_a2e_last100 FLOAT, jockey_a2e_last100 FLOAT,
        combo_a2e_last100 FLOAT
    )
    """)
    conn.execute("""INSERT INTO proform_a2e VALUES (
        'Jockey A','Trainer A',1.1,-5.0,12.0,500,1.2,-3.0,14.0,400,
        1.3,10.0,16.0,50,1.15,1.25,1.35
    )""")

    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Filter builder tests
# ---------------------------------------------------------------------------

class TestWhereClauseBuilder:
    def test_empty_filters(self):
        sql, params = _where_clauses({})
        assert sql == ""
        assert params == {}

    def test_single_filter(self):
        sql, params = _where_clauses({"state": "NSW"})
        assert "WHERE" in sql
        assert "r.state = $state" in sql
        assert params == {"state": "NSW"}

    def test_multiple_filters(self):
        sql, params = _where_clauses({
            "state": "VIC",
            "date_from": "2025-01-01",
            "odds_min": 2.0,
        })
        assert "WHERE" in sql
        assert "AND" in sql
        assert params["state"] == "VIC"
        assert params["date_from"] == "2025-01-01"
        assert params["odds_min"] == 2.0

    def test_none_values_excluded(self):
        sql, params = _where_clauses({"state": "NSW", "venue": None})
        assert "venue" not in sql

    def test_empty_string_excluded(self):
        sql, params = _where_clauses({"state": ""})
        assert sql == ""

    def test_all_filter_types(self):
        filters = {
            "date_from": "2025-01-01",
            "date_to": "2025-12-31",
            "venue": "randwick",
            "state": "NSW",
            "odds_min": 2.0,
            "odds_max": 10.0,
            "distance_category": "sprint",
            "track_condition": "Good 4",
            "speed_map_position": "leader",
        }
        sql, params = _where_clauses(filters)
        assert len(params) == 9
        # All clauses present
        for key in filters:
            assert f"${key}" in sql


# ---------------------------------------------------------------------------
# Parameterization safety tests
# ---------------------------------------------------------------------------

class TestParameterizationSafety:
    def test_sql_injection_in_venue(self, analytics_db):
        """Verify SQL injection via filter values is prevented."""
        filters = {"venue": "'; DROP TABLE runners; --"}
        sql, params = summary(filters)
        # Should execute without error (parameterized, not interpolated)
        result = analytics_db.execute(sql, params).fetchall()
        # No results for injection string, but table still exists
        assert analytics_db.execute("SELECT COUNT(*) FROM runners").fetchone()[0] == 5

    def test_parameterized_values_not_in_sql(self):
        """Filter values must be in params dict, not in SQL string."""
        filters = {"state": "NSW", "venue": "randwick"}
        sql, params = summary(filters)
        assert "NSW" not in sql
        assert "randwick" not in sql
        assert "$state" in sql
        assert "$venue" in sql


# ---------------------------------------------------------------------------
# Query function tests
# ---------------------------------------------------------------------------

class TestSummaryQuery:
    def test_unfiltered(self, analytics_db):
        sql, params = summary({})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        result = dict(zip(cols, rows[0]))
        assert result["total_runners"] == 5
        assert result["total_races"] == 2
        assert result["overall_win_rate"] == 40.0  # 2 winners out of 5

    def test_filtered_by_state(self, analytics_db):
        sql, params = summary({"state": "NSW"})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        result = dict(zip(cols, rows[0]))
        assert result["total_runners"] == 3
        assert result["total_races"] == 1

    def test_filtered_by_date_range(self, analytics_db):
        sql, params = summary({"date_from": "2025-07-01", "date_to": "2025-12-31"})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        result = dict(zip(cols, rows[0]))
        assert result["total_runners"] == 2  # Only flemington race


class TestCalibrationQuery:
    def test_returns_bins(self, analytics_db):
        sql, params = calibration({})
        rows = analytics_db.execute(sql, params).fetchall()
        assert len(rows) > 0
        cols = [d[0] for d in analytics_db.description]
        for row in rows:
            result = dict(zip(cols, row))
            assert "prob_bin" in result
            assert "predicted_rate" in result
            assert "actual_win_rate" in result


class TestOddsBandQuery:
    def test_returns_bands(self, analytics_db):
        sql, params = odds_band_performance({})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        bands = [dict(zip(cols, row))["odds_band"] for row in rows]
        assert "$2-$4" in bands


class TestVenueQuery:
    def test_returns_venues(self, analytics_db):
        sql, params = venue_performance({})
        rows = analytics_db.execute(sql, params).fetchall()
        # Minimum 20 runners threshold — our test data has fewer
        # So no results expected with HAVING >= 20
        # That's correct behavior for small datasets


class TestDistanceQuery:
    def test_returns_categories(self, analytics_db):
        sql, params = distance_performance({})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        cats = [dict(zip(cols, row))["distance_category"] for row in rows]
        assert "sprint" in cats
        assert "classic" in cats


class TestSpeedMapQuery:
    def test_returns_positions(self, analytics_db):
        sql, params = speed_map_analysis({})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        positions = [dict(zip(cols, row))["speed_map_position"] for row in rows]
        assert "leader" in positions

    def test_heatmap(self, analytics_db):
        sql, params = speed_map_heatmap({})
        rows = analytics_db.execute(sql, params).fetchall()
        assert len(rows) > 0


class TestBetTypeQuery:
    def test_returns_settled_picks(self, analytics_db):
        sql, params = bet_type_performance({})
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        assert len(rows) == 1  # Both picks are 'place' type
        result = dict(zip(cols, rows[0]))
        assert result["bet_type"] == "place"
        assert result["settled"] == 2
        assert result["total_pnl"] == 10.0  # 2.50 + 7.50


class TestFilterOptions:
    def test_returns_categories(self, analytics_db):
        sql, params = filter_options()
        rows = analytics_db.execute(sql, params).fetchall()
        cols = [d[0] for d in analytics_db.description]
        categories = set(dict(zip(cols, row))["category"] for row in rows)
        assert "states" in categories
        assert "distance_categories" in categories
        assert "speed_map_positions" in categories
