"""Database setup and session management."""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from punty.config import settings

logger = logging.getLogger(__name__)


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
    from punty.models import meeting, content, settings, pick  # noqa: F401
    from punty.memory import models as memory_models  # noqa: F401  # registers memory models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        _text = __import__("sqlalchemy").text

        # Create memory tables if they don't exist (for existing databases)
        for table_sql in [
            """CREATE TABLE IF NOT EXISTS race_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id VARCHAR NOT NULL,
                race_number INTEGER NOT NULL,
                race_id VARCHAR NOT NULL,
                context_json TEXT NOT NULL,
                runner_json TEXT NOT NULL,
                horse_name VARCHAR NOT NULL,
                saddlecloth INTEGER NOT NULL,
                tip_rank INTEGER NOT NULL,
                confidence VARCHAR,
                odds_at_tip FLOAT,
                bet_type VARCHAR,
                finish_position INTEGER,
                hit BOOLEAN,
                pnl FLOAT,
                sp_odds FLOAT,
                embedding_json TEXT,
                created_at DATETIME NOT NULL,
                settled_at DATETIME,
                UNIQUE(race_id, saddlecloth)
            )""",
            """CREATE TABLE IF NOT EXISTS pattern_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type VARCHAR NOT NULL,
                pattern_key VARCHAR NOT NULL,
                sample_count INTEGER DEFAULT 0,
                hit_rate FLOAT DEFAULT 0.0,
                avg_pnl FLOAT DEFAULT 0.0,
                avg_odds FLOAT DEFAULT 0.0,
                insight_text TEXT NOT NULL,
                conditions_json TEXT DEFAULT '{}',
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS race_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id VARCHAR NOT NULL UNIQUE,
                meeting_id VARCHAR NOT NULL,
                race_number INTEGER NOT NULL,
                track VARCHAR NOT NULL,
                distance INTEGER NOT NULL,
                race_class VARCHAR NOT NULL,
                going VARCHAR NOT NULL,
                rail_position VARCHAR,
                assessment_json TEXT NOT NULL,
                key_learnings TEXT NOT NULL,
                embedding_json TEXT,
                top_pick_hit BOOLEAN,
                any_pick_hit BOOLEAN,
                total_pnl FLOAT,
                created_at DATETIME NOT NULL
            )""",
        ]:
            try:
                await conn.execute(_text(table_sql))
            except Exception as e:
                err_msg = str(e).lower()
                if "already exists" not in err_msg:
                    logger.debug(f"Memory table note: {e}")

        # Add columns that may be missing on existing databases
        for col in [
            "ALTER TABLE picks ADD COLUMN estimated_return_pct FLOAT",
            "ALTER TABLE picks ADD COLUMN bet_type VARCHAR(20)",
            "ALTER TABLE picks ADD COLUMN bet_stake FLOAT",
            "ALTER TABLE meetings ADD COLUMN meeting_type VARCHAR(20)",
            "ALTER TABLE runners ADD COLUMN form_history TEXT",
            # Punting Form insights
            "ALTER TABLE runners ADD COLUMN pf_speed_rank INTEGER",
            "ALTER TABLE runners ADD COLUMN pf_settle FLOAT",
            "ALTER TABLE runners ADD COLUMN pf_map_factor FLOAT",
            "ALTER TABLE runners ADD COLUMN pf_jockey_factor FLOAT",
            # Data completeness tracking
            "ALTER TABLE meetings ADD COLUMN speed_map_complete BOOLEAN",
            # Fixed odds for settlement
            "ALTER TABLE picks ADD COLUMN place_odds_at_tip FLOAT",
            "ALTER TABLE runners ADD COLUMN place_odds FLOAT",
            # Additional runner stats columns
            "ALTER TABLE runners ADD COLUMN trainer_stats TEXT",
            "ALTER TABLE runners ADD COLUMN class_stats TEXT",
            "ALTER TABLE runners ADD COLUMN trainer_location VARCHAR(100)",
            # Post-race sectional times
            "ALTER TABLE races ADD COLUMN sectional_times TEXT",
            "ALTER TABLE races ADD COLUMN has_sectionals BOOLEAN",
            # Meeting code for sectional CSV downloads
            "ALTER TABLE meetings ADD COLUMN meet_code VARCHAR(20)",
            # Race assessment additional fields for better matching
            "ALTER TABLE race_assessments ADD COLUMN age_restriction VARCHAR(50)",
            "ALTER TABLE race_assessments ADD COLUMN sex_restriction VARCHAR(50)",
            "ALTER TABLE race_assessments ADD COLUMN weight_type VARCHAR(50)",
            "ALTER TABLE race_assessments ADD COLUMN field_size INTEGER",
            "ALTER TABLE race_assessments ADD COLUMN prize_money INTEGER",
            "ALTER TABLE race_assessments ADD COLUMN penetrometer FLOAT",
            "ALTER TABLE race_assessments ADD COLUMN state VARCHAR(10)",
            "ALTER TABLE race_assessments ADD COLUMN weather VARCHAR(50)",
            "ALTER TABLE race_assessments ADD COLUMN temperature INTEGER",
        ]:
            try:
                await conn.execute(_text(col))
            except Exception as e:
                # Log non-"column exists" errors for debugging
                err_msg = str(e).lower()
                if "duplicate column" not in err_msg and "already exists" not in err_msg:
                    logger.warning(f"DB migration warning: {col[:50]}... - {e}")

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
            # Memory system indexes
            "CREATE INDEX IF NOT EXISTS ix_race_memories_race_id ON race_memories(race_id)",
            "CREATE INDEX IF NOT EXISTS ix_race_memories_meeting_id ON race_memories(meeting_id)",
            "CREATE INDEX IF NOT EXISTS ix_race_memories_settled ON race_memories(settled_at)",
            "CREATE INDEX IF NOT EXISTS ix_pattern_insights_type_key ON pattern_insights(pattern_type, pattern_key)",
            # Race assessment indexes for RAG retrieval
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_meeting_id ON race_assessments(meeting_id)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_track ON race_assessments(track)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_distance ON race_assessments(distance)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_race_class ON race_assessments(race_class)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_going ON race_assessments(going)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_age ON race_assessments(age_restriction)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_sex ON race_assessments(sex_restriction)",
            "CREATE INDEX IF NOT EXISTS ix_race_assessments_state ON race_assessments(state)",
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
