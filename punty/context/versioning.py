"""Context versioning and snapshot management."""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Per-meeting locks to prevent concurrent snapshot creation
_snapshot_locks: dict[str, asyncio.Lock] = {}


async def create_context_snapshot(
    db: AsyncSession,
    meeting_id: str,
    force: bool = False,
) -> Optional[dict[str, Any]]:
    """Create a new context snapshot if data has changed.

    Args:
        db: Database session
        meeting_id: Meeting to snapshot
        force: Create snapshot even if no changes

    Returns:
        Snapshot dict if created, None if no changes
    """
    from punty.context.builder import ContextBuilder
    from punty.context.diff import detect_significant_changes
    from punty.models.content import ContextSnapshot

    # Acquire per-meeting lock to prevent concurrent version collisions
    lock = _snapshot_locks.setdefault(meeting_id, asyncio.Lock())

    async with lock:
        builder = ContextBuilder(db)

        # Build current context
        context = await builder.build_meeting_context(meeting_id)
        if not context:
            return None

        # Calculate hash
        context_json = builder.context_to_json(context)
        data_hash = hashlib.sha256(context_json.encode()).hexdigest()[:16]

        # Get previous snapshot
        previous = await get_latest_snapshot(db, meeting_id)

        # Check if data has changed
        if previous and previous.data_hash == data_hash and not force:
            logger.debug(f"No changes detected for {meeting_id}")
            return None

        # Detect significant changes
        significant_changes = []
        if previous:
            prev_context = json.loads(previous.snapshot_json)
            significant_changes = detect_significant_changes(prev_context, context)

        # Get next version number
        result = await db.execute(
            select(func.max(ContextSnapshot.version)).where(
                ContextSnapshot.meeting_id == meeting_id
            )
        )
        max_version = result.scalar() or 0
        new_version = max_version + 1

        # Create snapshot
        snapshot = ContextSnapshot(
            id=str(uuid.uuid4()),
            meeting_id=meeting_id,
            version=new_version,
            data_hash=data_hash,
            snapshot_json=context_json,
            significant_changes=json.dumps(significant_changes) if significant_changes else None,
        )

        db.add(snapshot)
        await db.commit()

        logger.info(
            f"Created context snapshot v{new_version} for {meeting_id} "
            f"(changes: {len(significant_changes)})"
        )

        return {
            "id": snapshot.id,
            "version": new_version,
            "data_hash": data_hash,
            "significant_changes": significant_changes,
            "context": context,
        }


async def get_latest_snapshot(
    db: AsyncSession,
    meeting_id: str,
) -> Optional[Any]:
    """Get the most recent context snapshot for a meeting."""
    from punty.models.content import ContextSnapshot

    result = await db.execute(
        select(ContextSnapshot)
        .where(ContextSnapshot.meeting_id == meeting_id)
        .order_by(ContextSnapshot.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_snapshot_by_version(
    db: AsyncSession,
    meeting_id: str,
    version: int,
) -> Optional[Any]:
    """Get a specific version of context snapshot."""
    from punty.models.content import ContextSnapshot

    result = await db.execute(
        select(ContextSnapshot).where(
            ContextSnapshot.meeting_id == meeting_id,
            ContextSnapshot.version == version,
        )
    )
    return result.scalar_one_or_none()


async def compare_snapshots(
    db: AsyncSession,
    meeting_id: str,
    version_a: int,
    version_b: int,
) -> dict[str, Any]:
    """Compare two context snapshots."""
    from punty.context.diff import detect_significant_changes

    snapshot_a = await get_snapshot_by_version(db, meeting_id, version_a)
    snapshot_b = await get_snapshot_by_version(db, meeting_id, version_b)

    if not snapshot_a or not snapshot_b:
        return {"error": "Snapshot not found"}

    context_a = json.loads(snapshot_a.snapshot_json)
    context_b = json.loads(snapshot_b.snapshot_json)

    changes = detect_significant_changes(context_a, context_b)

    return {
        "version_a": version_a,
        "version_b": version_b,
        "changes": changes,
        "changed": len(changes) > 0,
    }
