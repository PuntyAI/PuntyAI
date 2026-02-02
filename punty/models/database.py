"""Database setup and session management."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from punty.config import settings


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Initialize the database, creating all tables."""
    # Import models to ensure they're registered with Base
    from punty.models import meeting, content, settings, pick  # noqa: F401  # registers models with Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add columns that may be missing on existing databases
        _text = __import__("sqlalchemy").text
        for col in [
            "ALTER TABLE picks ADD COLUMN estimated_return_pct FLOAT",
            "ALTER TABLE picks ADD COLUMN bet_type VARCHAR(20)",
            "ALTER TABLE picks ADD COLUMN bet_stake FLOAT",
            "ALTER TABLE meetings ADD COLUMN meeting_type VARCHAR(20)",
            "ALTER TABLE runners ADD COLUMN form_history TEXT",
        ]:
            try:
                await conn.execute(_text(col))
            except Exception:
                pass  # Column already exists

        # Create indexes on existing tables (IF NOT EXISTS handled by SQLite)
        for idx in [
            "CREATE INDEX IF NOT EXISTS ix_meetings_date ON meetings(date)",
            "CREATE INDEX IF NOT EXISTS ix_races_meeting_id ON races(meeting_id)",
            "CREATE INDEX IF NOT EXISTS ix_runners_race_id ON runners(race_id)",
            "CREATE INDEX IF NOT EXISTS ix_runners_finish_position ON runners(race_id, finish_position)",
            "CREATE INDEX IF NOT EXISTS ix_picks_meeting_race ON picks(meeting_id, race_number)",
            "CREATE INDEX IF NOT EXISTS ix_picks_content_id ON picks(content_id)",
            "CREATE INDEX IF NOT EXISTS ix_picks_settled ON picks(settled)",
            "CREATE INDEX IF NOT EXISTS ix_content_meeting_id ON content(meeting_id)",
            "CREATE INDEX IF NOT EXISTS ix_content_status ON content(status)",
        ]:
            try:
                await conn.execute(_text(idx))
            except Exception:
                pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
