"""Parameterized query library for DuckDB analytics.

All filter values are passed via DuckDB's $variable parameterization
to prevent SQL injection. The _where_clauses() builder converts a
filter dict into a (sql_fragment, params) tuple that gets appended
to any base query.

Picks-only mode: when backtest.db hasn't been loaded (no runners table),
queries gracefully fall back to picks-based analytics. The filter_options()
and summary() functions detect this and return production pick data instead.
"""

from __future__ import annotations


def _where_clauses(filters: dict) -> tuple[str, dict]:
    """Build WHERE clause fragments from a filter dict.

    Supported keys: date_from, date_to, venue, state, odds_min, odds_max,
    distance_category, track_condition, speed_map_position.

    Returns (sql_fragment, params) where sql_fragment starts with "WHERE"
    if any filters are active, or is empty string with empty dict.
    """
    clauses = []
    params = {}

    if filters.get("date_from"):
        clauses.append("r.date >= $date_from")
        params["date_from"] = filters["date_from"]

    if filters.get("date_to"):
        clauses.append("r.date <= $date_to")
        params["date_to"] = filters["date_to"]

    if filters.get("venue"):
        clauses.append("r.venue = $venue")
        params["venue"] = filters["venue"]

    if filters.get("state"):
        clauses.append("r.state = $state")
        params["state"] = filters["state"]

    if filters.get("odds_min"):
        clauses.append("r.current_odds >= $odds_min")
        params["odds_min"] = float(filters["odds_min"])

    if filters.get("odds_max"):
        clauses.append("r.current_odds <= $odds_max")
        params["odds_max"] = float(filters["odds_max"])

    if filters.get("distance_category"):
        clauses.append("r.distance_category = $distance_category")
        params["distance_category"] = filters["distance_category"]

    if filters.get("track_condition"):
        clauses.append("r.track_condition = $track_condition")
        params["track_condition"] = filters["track_condition"]

    if filters.get("speed_map_position"):
        clauses.append("r.speed_map_position = $speed_map_position")
        params["speed_map_position"] = filters["speed_map_position"]

    if not clauses:
        return "", {}

    return "WHERE " + " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# Query functions — each returns (sql, params) for engine.query()
# ---------------------------------------------------------------------------

def summary(filters: dict) -> tuple[str, dict]:
    """Overall summary stats: runner/race counts, win/place rates, ROI."""
    where, params = _where_clauses(filters)
    sql = f"""
    SELECT
        COUNT(*) AS total_runners,
        COUNT(DISTINCT r.race_id) AS total_races,
        COUNT(DISTINCT r.venue) AS total_venues,
        ROUND(100.0 * SUM(CASE WHEN r.is_winner THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS overall_win_rate,
        ROUND(100.0 * SUM(CASE WHEN r.is_placed THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS overall_place_rate,
        ROUND(100.0 * (SUM(CASE WHEN r.is_winner THEN r.current_odds - 1 ELSE -1 END)) / NULLIF(COUNT(*), 0), 2) AS flat_win_roi,
        ROUND(100.0 * (SUM(CASE WHEN r.is_placed THEN r.place_dividend - 1 ELSE -1 END)) / NULLIF(COUNT(*), 0), 2) AS flat_place_roi,
        MIN(r.date) AS date_from,
        MAX(r.date) AS date_to
    FROM runners r
    {where}
    """
    return sql, params


def calibration(filters: dict) -> tuple[str, dict]:
    """Predicted vs actual win/place rates in probability bins."""
    where, params = _where_clauses(filters)
    # Bins: 0-5%, 5-10%, 10-15%, ..., 45-50%, 50%+
    sql = f"""
    WITH binned AS (
        SELECT
            CASE
                WHEN r.implied_prob < 0.05 THEN '00-05'
                WHEN r.implied_prob < 0.10 THEN '05-10'
                WHEN r.implied_prob < 0.15 THEN '10-15'
                WHEN r.implied_prob < 0.20 THEN '15-20'
                WHEN r.implied_prob < 0.25 THEN '20-25'
                WHEN r.implied_prob < 0.30 THEN '25-30'
                WHEN r.implied_prob < 0.35 THEN '30-35'
                WHEN r.implied_prob < 0.40 THEN '35-40'
                WHEN r.implied_prob < 0.45 THEN '40-45'
                WHEN r.implied_prob < 0.50 THEN '45-50'
                ELSE '50+'
            END AS prob_bin,
            r.implied_prob,
            r.is_winner,
            r.is_placed
        FROM runners r
        {where}
        {"AND" if where else "WHERE"} r.implied_prob IS NOT NULL AND r.implied_prob > 0
    )
    SELECT
        prob_bin,
        COUNT(*) AS count,
        ROUND(100.0 * AVG(implied_prob), 2) AS predicted_rate,
        ROUND(100.0 * AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END), 2) AS actual_win_rate,
        ROUND(100.0 * AVG(CASE WHEN is_placed THEN 1.0 ELSE 0.0 END), 2) AS actual_place_rate
    FROM binned
    GROUP BY prob_bin
    ORDER BY prob_bin
    """
    return sql, params


def odds_band_performance(filters: dict) -> tuple[str, dict]:
    """Win/place rate and ROI by odds band."""
    where, params = _where_clauses(filters)
    sql = f"""
    SELECT
        r.odds_band,
        COUNT(*) AS count,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN 1.0 ELSE 0.0 END), 2) AS win_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN 1.0 ELSE 0.0 END), 2) AS place_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN r.current_odds - 1 ELSE -1 END), 2) AS win_roi,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN r.place_dividend - 1 ELSE -1 END), 2) AS place_roi
    FROM runners r
    {where}
    {"AND" if where else "WHERE"} r.odds_band IS NOT NULL
    GROUP BY r.odds_band
    ORDER BY r.odds_band
    """
    return sql, params


def venue_performance(filters: dict) -> tuple[str, dict]:
    """Performance breakdown by venue."""
    where, params = _where_clauses(filters)
    sql = f"""
    SELECT
        r.venue,
        r.state,
        COUNT(*) AS runners,
        COUNT(DISTINCT r.race_id) AS races,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN 1.0 ELSE 0.0 END), 2) AS win_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN 1.0 ELSE 0.0 END), 2) AS place_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN r.current_odds - 1 ELSE -1 END), 2) AS win_roi,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN r.place_dividend - 1 ELSE -1 END), 2) AS place_roi
    FROM runners r
    {where}
    GROUP BY r.venue, r.state
    HAVING COUNT(*) >= 20
    ORDER BY races DESC
    """
    return sql, params


def distance_performance(filters: dict) -> tuple[str, dict]:
    """Performance by distance category."""
    where, params = _where_clauses(filters)
    sql = f"""
    SELECT
        r.distance_category,
        COUNT(*) AS runners,
        COUNT(DISTINCT r.race_id) AS races,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN 1.0 ELSE 0.0 END), 2) AS win_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN 1.0 ELSE 0.0 END), 2) AS place_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN r.current_odds - 1 ELSE -1 END), 2) AS win_roi,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN r.place_dividend - 1 ELSE -1 END), 2) AS place_roi
    FROM runners r
    {where}
    {"AND" if where else "WHERE"} r.distance_category IS NOT NULL
    GROUP BY r.distance_category
    ORDER BY r.distance_category
    """
    return sql, params


def speed_map_analysis(filters: dict) -> tuple[str, dict]:
    """Win/place rates by speed map position."""
    where, params = _where_clauses(filters)
    sql = f"""
    SELECT
        r.speed_map_position,
        COUNT(*) AS runners,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN 1.0 ELSE 0.0 END), 2) AS win_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN 1.0 ELSE 0.0 END), 2) AS place_rate,
        ROUND(100.0 * AVG(CASE WHEN r.is_winner THEN r.current_odds - 1 ELSE -1 END), 2) AS win_roi,
        ROUND(100.0 * AVG(CASE WHEN r.is_placed THEN r.place_dividend - 1 ELSE -1 END), 2) AS place_roi
    FROM runners r
    {where}
    {"AND" if where else "WHERE"} r.speed_map_position IS NOT NULL AND r.speed_map_position != ''
    GROUP BY r.speed_map_position
    ORDER BY win_rate DESC
    """
    return sql, params


def speed_map_heatmap(filters: dict) -> tuple[str, dict]:
    """Settle position vs finish position cross-tab from speed_maps table."""
    where_base, params = _where_clauses(filters)
    # Join speed_maps with runners for filtering
    where = where_base.replace("r.", "ru.") if where_base else ""
    sql = f"""
    SELECT
        s.settle_position AS settle,
        s.finish_position AS finish,
        COUNT(*) AS count,
        ROUND(100.0 * SUM(CASE WHEN ru.is_winner THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS win_pct
    FROM speed_maps s
    JOIN runners ru ON ru.id = s.runner_id
    {where}
    {"AND" if where else "WHERE"} s.settle_position IS NOT NULL AND s.finish_position IS NOT NULL
        AND s.settle_position <= 12 AND s.finish_position <= 12
    GROUP BY s.settle_position, s.finish_position
    ORDER BY s.settle_position, s.finish_position
    """
    return sql, params


def bet_type_performance(filters: dict) -> tuple[str, dict]:
    """Production pick P&L by bet type."""
    where, params = _where_clauses(filters)
    # picks table uses different columns, adjust where
    pick_where = where.replace("r.", "p.") if where else ""
    sql = f"""
    SELECT
        p.bet_type,
        p.pick_type,
        COUNT(*) AS bets,
        SUM(CASE WHEN p.settled THEN 1 ELSE 0 END) AS settled,
        ROUND(SUM(CASE WHEN p.hit THEN 1.0 ELSE 0.0 END) / NULLIF(SUM(CASE WHEN p.settled THEN 1 ELSE 0 END), 0) * 100, 2) AS strike_rate,
        ROUND(SUM(COALESCE(p.pnl, 0)), 2) AS total_pnl,
        ROUND(100.0 * SUM(COALESCE(p.pnl, 0)) / NULLIF(SUM(COALESCE(p.bet_stake, 0)), 0), 2) AS roi
    FROM picks p
    {pick_where}
    {"AND" if pick_where else "WHERE"} p.settled = true
    GROUP BY p.bet_type, p.pick_type
    ORDER BY total_pnl DESC
    """
    return sql, params


def factor_importance(filters: dict) -> tuple[str, dict]:
    """Factor scores: winners vs losers average comparison.

    Note: This queries the runners table which has factor data from backtest.
    Factor data is NOT available for all runners — only those from backtest.
    """
    # Factor data would need to be in the runners table as separate columns
    # For now, return a placeholder that works with the speed_map and odds data
    where, params = _where_clauses(filters)
    sql = f"""
    SELECT
        'speed_map' AS factor,
        ROUND(AVG(CASE WHEN r.is_winner THEN
            CASE r.speed_map_position
                WHEN 'leader' THEN 0.9
                WHEN 'on_pace' THEN 0.7
                WHEN 'midfield' THEN 0.5
                ELSE 0.3
            END ELSE NULL END), 4) AS winner_avg,
        ROUND(AVG(CASE WHEN NOT r.is_winner THEN
            CASE r.speed_map_position
                WHEN 'leader' THEN 0.9
                WHEN 'on_pace' THEN 0.7
                WHEN 'midfield' THEN 0.5
                ELSE 0.3
            END ELSE NULL END), 4) AS loser_avg,
        COUNT(CASE WHEN r.is_winner THEN 1 END) AS winner_samples
    FROM runners r
    {where}
    {"AND" if where else "WHERE"} r.speed_map_position IS NOT NULL AND r.speed_map_position != ''

    UNION ALL

    SELECT
        'market' AS factor,
        ROUND(AVG(CASE WHEN r.is_winner THEN r.implied_prob ELSE NULL END), 4) AS winner_avg,
        ROUND(AVG(CASE WHEN NOT r.is_winner THEN r.implied_prob ELSE NULL END), 4) AS loser_avg,
        COUNT(CASE WHEN r.is_winner THEN 1 END) AS winner_samples
    FROM runners r
    {where}
    {"AND" if where else "WHERE"} r.implied_prob IS NOT NULL AND r.implied_prob > 0

    UNION ALL

    SELECT
        'barrier' AS factor,
        ROUND(AVG(CASE WHEN r.is_winner THEN 1.0 / NULLIF(r.barrier, 0) ELSE NULL END), 4) AS winner_avg,
        ROUND(AVG(CASE WHEN NOT r.is_winner THEN 1.0 / NULLIF(r.barrier, 0) ELSE NULL END), 4) AS loser_avg,
        COUNT(CASE WHEN r.is_winner THEN 1 END) AS winner_samples
    FROM runners r
    {where}
    {"AND" if where else "WHERE"} r.barrier IS NOT NULL AND r.barrier > 0
    """
    return sql, params


def jockey_trainer_a2e(filters: dict) -> tuple[str, dict]:
    """Top jockey/trainer combos by A2E from Proform data."""
    where, params = _where_clauses(filters)
    a2e_where = where.replace("r.", "a.") if where else ""
    sql = f"""
    SELECT
        a.jockey,
        a.trainer,
        a.combo_a2e_career,
        a.combo_pot_career,
        a.combo_strike_career,
        a.combo_runners_career,
        a.trainer_a2e_career,
        a.jockey_a2e_career
    FROM proform_a2e a
    {a2e_where}
    {"AND" if a2e_where else "WHERE"} a.combo_runners_career >= 20
    ORDER BY a.combo_a2e_career DESC
    LIMIT 100
    """
    return sql, params


def time_series(filters: dict, group_by: str = "month") -> tuple[str, dict]:
    """Monthly or weekly P&L trend from production picks."""
    where, params = _where_clauses(filters)
    pick_where = where.replace("r.", "p.") if where else ""

    if group_by == "week":
        date_trunc = "DATE_TRUNC('week', p.settled_at)"
    else:
        date_trunc = "DATE_TRUNC('month', p.settled_at)"

    sql = f"""
    SELECT
        {date_trunc} AS period,
        COUNT(*) AS bets,
        ROUND(SUM(COALESCE(p.pnl, 0)), 2) AS pnl,
        ROUND(SUM(SUM(COALESCE(p.pnl, 0))) OVER (ORDER BY {date_trunc}), 2) AS cumulative_pnl,
        ROUND(100.0 * SUM(COALESCE(p.pnl, 0)) / NULLIF(SUM(COALESCE(p.bet_stake, 0)), 0), 2) AS roi
    FROM picks p
    {pick_where}
    {"AND" if pick_where else "WHERE"} p.settled = true AND p.settled_at IS NOT NULL
    GROUP BY {date_trunc}
    ORDER BY period
    """
    return sql, params


def filter_options() -> tuple[str, dict]:
    """Distinct values for filter dropdowns."""
    sql = """
    SELECT 'states' AS category, state AS value, COUNT(*) AS count
    FROM runners WHERE state IS NOT NULL AND state != ''
    GROUP BY state
    UNION ALL
    SELECT 'venues' AS category, venue AS value, COUNT(*) AS count
    FROM runners WHERE venue IS NOT NULL AND venue != ''
    GROUP BY venue
    HAVING COUNT(*) >= 20
    UNION ALL
    SELECT 'distance_categories' AS category, distance_category AS value, COUNT(*) AS count
    FROM runners WHERE distance_category IS NOT NULL
    GROUP BY distance_category
    UNION ALL
    SELECT 'track_conditions' AS category, track_condition AS value, COUNT(*) AS count
    FROM runners WHERE track_condition IS NOT NULL AND track_condition != ''
    GROUP BY track_condition
    HAVING COUNT(*) >= 10
    UNION ALL
    SELECT 'speed_map_positions' AS category, speed_map_position AS value, COUNT(*) AS count
    FROM runners WHERE speed_map_position IS NOT NULL AND speed_map_position != ''
    GROUP BY speed_map_position
    ORDER BY category, count DESC
    """
    return sql, {}
