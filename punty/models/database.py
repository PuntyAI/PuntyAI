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
        ]:
            try:
                await conn.execute(_text(col))
            except Exception:
                pass  # Column already exists


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
