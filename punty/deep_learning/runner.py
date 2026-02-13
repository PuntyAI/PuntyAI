"""Deep learning pattern runner — orchestrates import, analysis, and export.

Runs all pattern analyses against the deep_learning.db, then writes
significant patterns as PatternInsight rows to the main production DB.

Usage:
    python -m punty.deep_learning.runner --data-dir "path/to/DatafromProform/2026"

Stages:
    1. Import Proform data → deep_learning.db (skip if already populated)
    2. Run 15 pattern analyses
    3. Write PatternInsight rows to main DB (async)
    4. Print summary report
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .importer import import_all
from .models import get_session, HistoricalRace, HistoricalRunner
from .patterns import run_all_analyses, Pattern

logger = logging.getLogger(__name__)


def _check_db_populated(db_path=None) -> tuple[int, int]:
    """Check if deep_learning.db has data."""
    session = get_session(db_path)
    try:
        races = session.query(HistoricalRace).count()
        runners = session.query(HistoricalRunner).count()
        return races, runners
    finally:
        session.close()


async def export_to_production(patterns: list[Pattern]):
    """Write patterns as PatternInsight rows to the main production DB.

    Uses async session since the main DB is async SQLite.
    Clears existing deep_learning patterns first (full refresh).
    """
    from punty.models.database import async_session
    from punty.memory.models import PatternInsight
    from sqlalchemy import delete

    async with async_session() as db:
        # Clear old deep learning patterns
        await db.execute(
            delete(PatternInsight).where(
                PatternInsight.pattern_type.like("deep_learning_%")
            )
        )

        # Write new patterns
        for p in patterns:
            insight = PatternInsight(
                pattern_type=p.pattern_type,
                pattern_key=p.dimension,
                sample_count=p.sample_size,
                hit_rate=p.win_rate,
                avg_pnl=p.edge,
                avg_odds=1.0 / p.win_rate if p.win_rate > 0 else 0,
                insight_text=p.description,
                conditions_json=json.dumps({
                    "confidence": p.confidence,
                    "p_value": round(p.p_value, 4),
                    "base_rate": round(p.base_rate, 4),
                    **p.metadata,
                }),
            )
            db.add(insight)

        await db.commit()
        logger.info(f"Exported {len(patterns)} patterns to production DB")


def run(
    data_dir: str | Path,
    db_path: str | Path | None = None,
    skip_import: bool = False,
    export: bool = True,
):
    """Full pipeline: import → analyse → export."""
    data_dir = Path(data_dir)

    # Stage 1: Import
    races, runners = _check_db_populated(db_path)
    if races > 0 and skip_import:
        print(f"DB already has {races:,} races, {runners:,} runners — skipping import")
    elif races > 0 and not skip_import:
        print(f"DB has {races:,} races, {runners:,} runners — re-importing (idempotent)")
        import_all(data_dir, db_path)
    else:
        print("Empty DB — running full import...")
        import_all(data_dir, db_path)

    # Stage 2: Analyse
    print("\n=== Running Pattern Analysis ===")
    patterns = run_all_analyses(db_path)

    # Print top patterns
    print("\n=== Top Patterns ===")
    for i, p in enumerate(patterns[:30]):
        print(
            f"  [{p.confidence}] {p.pattern_type}: {p.description} "
            f"(n={p.sample_size}, p={p.p_value:.4f})"
        )

    # Stage 3: Export
    if export and patterns:
        print(f"\nExporting {len(patterns)} patterns to production DB...")
        asyncio.run(export_to_production(patterns))
        print("Export complete.")
    elif not patterns:
        print("\nNo significant patterns found — nothing to export.")
    else:
        print(f"\nSkipping export (--no-export). {len(patterns)} patterns found.")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Patterns found:     {len(patterns)}")
    high = sum(1 for p in patterns if p.confidence == "HIGH")
    med = sum(1 for p in patterns if p.confidence == "MEDIUM")
    low = sum(1 for p in patterns if p.confidence == "LOW")
    print(f"  HIGH confidence:  {high}")
    print(f"  MEDIUM:           {med}")
    print(f"  LOW:              {low}")

    # Breakdown by type
    type_counts: dict[str, int] = {}
    for p in patterns:
        type_counts[p.pattern_type] = type_counts.get(p.pattern_type, 0) + 1
    print("\nBy analysis type:")
    for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")

    return patterns


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run deep learning analysis")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to DatafromProform/2026 directory",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to deep_learning.db (default: data/deep_learning.db)",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip import if DB already has data",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Don't export to production DB",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(args.data_dir, args.db_path, args.skip_import, not args.no_export)
