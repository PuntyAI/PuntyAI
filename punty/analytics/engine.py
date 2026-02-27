"""DuckDB connection manager for analytics queries.

Provides async-compatible query execution via a thread pool, since DuckDB
connections are not natively async. The database is opened read-only to
prevent accidental writes from the web layer.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

ANALYTICS_DB_PATH = Path("data/analytics.duckdb")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="duckdb")


def is_available() -> bool:
    """Check if the analytics DuckDB file exists."""
    return ANALYTICS_DB_PATH.exists()


@lru_cache(maxsize=1)
def _get_connection() -> duckdb.DuckDBPyConnection:
    """Get a cached read-only DuckDB connection."""
    if not is_available():
        raise FileNotFoundError(f"Analytics DB not found at {ANALYTICS_DB_PATH}")
    conn = duckdb.connect(str(ANALYTICS_DB_PATH), read_only=True)
    logger.info("Opened DuckDB analytics connection: %s", ANALYTICS_DB_PATH)
    return conn


def _execute_query(sql: str, params: dict | None = None) -> list[dict]:
    """Execute a parameterized query and return results as list of dicts.

    Uses DuckDB's $variable syntax for safe parameterization.
    """
    conn = _get_connection()
    try:
        if params:
            result = conn.execute(sql, params)
        else:
            result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception:
        logger.exception("DuckDB query failed: %s", sql[:200])
        raise


async def query(sql: str, params: dict | None = None) -> list[dict]:
    """Async wrapper: run a DuckDB query in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _execute_query, sql, params)


async def query_one(sql: str, params: dict | None = None) -> dict | None:
    """Run a query and return the first row, or None."""
    rows = await query(sql, params)
    return rows[0] if rows else None


def close():
    """Close the DuckDB connection and clear the cache."""
    try:
        conn = _get_connection()
        conn.close()
    except Exception:
        pass
    _get_connection.cache_clear()
