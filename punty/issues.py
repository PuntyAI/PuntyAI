"""Issue tracker — log persistent data quality and settlement issues.

Usage:
    from punty.issues import log_issue
    await log_issue(db, "settlement", "error",
        "Exotic hit but $0 P&L — missing dividend",
        meeting_id="cairns-2026-03-26", race_number=3,
        pick_id="pk-d1478bdf-019", amount=15.0)
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.models.issue import Issue

logger = logging.getLogger(__name__)


async def log_issue(
    db: AsyncSession,
    category: str,
    severity: str,
    title: str,
    description: str | None = None,
    meeting_id: str | None = None,
    race_number: int | None = None,
    pick_id: str | None = None,
    link: str | None = None,
    amount: float | None = None,
) -> Issue:
    """Log a persistent issue to the DB."""
    issue = Issue(
        category=category,
        severity=severity,
        title=title,
        description=description,
        meeting_id=meeting_id,
        race_number=race_number,
        pick_id=pick_id,
        link=link,
        amount=amount,
    )
    db.add(issue)
    await db.flush()

    level = logging.ERROR if severity == "error" else logging.WARNING if severity == "warning" else logging.INFO
    logger.log(level, f"[Issue #{issue.id}] {title}" + (f" ({meeting_id} R{race_number})" if meeting_id else ""))

    return issue
