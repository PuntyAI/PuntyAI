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


# Create async engine with SQLite timeout for concurrent access
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"timeout": 30},
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
    from punty.models import meeting, content, settings, pick, live_update  # noqa: F401
    from punty.memory import models as memory_models  # noqa: F401  # registers memory models

    async with engine.begin() as conn:
        _text = __import__("sqlalchemy").text

        # Enable WAL mode and busy timeout for concurrent access
        await conn.execute(_text("PRAGMA journal_mode=WAL"))
        await conn.execute(_text("PRAGMA busy_timeout=30000"))

        await conn.run_sync(Base.metadata.create_all)

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
            """CREATE TABLE IF NOT EXISTS live_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id VARCHAR(64) NOT NULL REFERENCES meetings(id),
                race_number INTEGER,
                update_type VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                tweet_id VARCHAR(64),
                parent_tweet_id VARCHAR(64),
                horse_name VARCHAR(100),
                odds FLOAT,
                pnl FLOAT,
                created_at DATETIME NOT NULL
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
            """CREATE TABLE IF NOT EXISTS settings_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key VARCHAR NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_by VARCHAR(200),
                action VARCHAR(20) NOT NULL,
                changed_at DATETIME NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS future_races (
                id VARCHAR(128) PRIMARY KEY,
                venue VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                race_number INTEGER,
                race_name VARCHAR(200) NOT NULL,
                group_level VARCHAR(20),
                distance INTEGER,
                prize_money INTEGER,
                state VARCHAR(10),
                scraped_at DATETIME NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS probability_tuning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_weights_json TEXT NOT NULL,
                new_weights_json TEXT NOT NULL,
                metrics_json TEXT DEFAULT '{}',
                reason VARCHAR(20) DEFAULT 'auto_tune',
                picks_analyzed INTEGER DEFAULT 0,
                created_at DATETIME NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS future_nominations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                future_race_id VARCHAR(128) NOT NULL REFERENCES future_races(id) ON DELETE CASCADE,
                horse_name VARCHAR(100) NOT NULL,
                trainer VARCHAR(100),
                jockey VARCHAR(100),
                barrier INTEGER,
                weight REAL,
                last_start VARCHAR(200),
                career_record VARCHAR(100)
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
            # Pace analysis insights
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
            # Twitter tweet ID for reply threading
            "ALTER TABLE content ADD COLUMN twitter_id VARCHAR(64)",
            # Probability model fields on picks
            "ALTER TABLE picks ADD COLUMN win_probability FLOAT",
            "ALTER TABLE picks ADD COLUMN place_probability FLOAT",
            "ALTER TABLE picks ADD COLUMN value_rating FLOAT",
            "ALTER TABLE picks ADD COLUMN confidence VARCHAR(10)",
            "ALTER TABLE picks ADD COLUMN recommended_stake FLOAT",
            # Manual track condition override lock
            "ALTER TABLE meetings ADD COLUMN track_condition_locked BOOLEAN DEFAULT 0",
            # Punty's Pick flag (best-bet recommendation per race)
            "ALTER TABLE picks ADD COLUMN is_puntys_pick BOOLEAN DEFAULT 0",
            # Conditions data
            "ALTER TABLE meetings ADD COLUMN rainfall REAL",
            "ALTER TABLE meetings ADD COLUMN irrigation BOOLEAN",
            "ALTER TABLE meetings ADD COLUMN going_stick REAL",
            "ALTER TABLE meetings ADD COLUMN weather_humidity INTEGER",
            # Blog columns on content table
            "ALTER TABLE content ADD COLUMN blog_title VARCHAR(200)",
            "ALTER TABLE content ADD COLUMN blog_slug VARCHAR(200) UNIQUE",
            "ALTER TABLE content ADD COLUMN blog_week_start DATE",
            # Probability factor breakdown on picks
            "ALTER TABLE picks ADD COLUMN factors_json TEXT",
        ]:
            try:
                await conn.execute(_text(col))
            except Exception as e:
                # Log non-"column exists" errors for debugging
                err_msg = str(e).lower()
                if "duplicate column" not in err_msg and "already exists" not in err_msg:
                    logger.warning(f"DB migration warning: {col[:50]}... - {e}")

        # --- Make content.meeting_id nullable (SQLite table rebuild) ---
        try:
            row = await conn.execute(_text(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='content'"
            ))
            create_sql = row.scalar()
            # Match both quoted and unquoted column names
            needs_migrate = (
                create_sql
                and 'meeting_id' in create_sql
                and 'NOT NULL' in create_sql
                and ('meeting_id VARCHAR(64) NOT NULL' in create_sql
                     or '"meeting_id" VARCHAR(64) NOT NULL' in create_sql)
            )
            if needs_migrate:
                logger.info("Migrating content table: making meeting_id nullable")
                new_sql = create_sql.replace(
                    'meeting_id VARCHAR(64) NOT NULL',
                    'meeting_id VARCHAR(64)',
                ).replace(
                    '"meeting_id" VARCHAR(64) NOT NULL',
                    '"meeting_id" VARCHAR(64)',
                )
                # SQLite table rebuild: rename -> recreate -> copy -> drop old
                await conn.execute(_text("ALTER TABLE content RENAME TO _content_old"))
                await conn.execute(_text(new_sql))
                await conn.execute(_text(
                    "INSERT INTO content SELECT * FROM _content_old"
                ))
                await conn.execute(_text("DROP TABLE _content_old"))
                logger.info("Content table migration complete: meeting_id now nullable")
        except Exception as e:
            logger.warning(f"Content meeting_id nullable migration: {e}")

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
            # Live updates indexes
            "CREATE INDEX IF NOT EXISTS ix_live_updates_meeting_id ON live_updates(meeting_id)",
            # Settlement query indexes
            "CREATE INDEX IF NOT EXISTS ix_races_meeting_status ON races(meeting_id, results_status)",
            "CREATE INDEX IF NOT EXISTS ix_picks_type_settled ON picks(meeting_id, pick_type, settled)",
            # Settings audit indexes
            "CREATE INDEX IF NOT EXISTS ix_settings_audit_key ON settings_audit(key)",
            "CREATE INDEX IF NOT EXISTS ix_settings_audit_changed_at ON settings_audit(changed_at)",
            # Future races indexes
            "CREATE INDEX IF NOT EXISTS ix_future_races_date ON future_races(date)",
            "CREATE INDEX IF NOT EXISTS ix_future_races_group ON future_races(group_level)",
            "CREATE INDEX IF NOT EXISTS ix_future_noms_race_id ON future_nominations(future_race_id)",
            # Blog indexes
            "CREATE INDEX IF NOT EXISTS ix_content_blog_slug ON content(blog_slug)",
            # Probability tuning log indexes
            "CREATE INDEX IF NOT EXISTS ix_tuning_log_created ON probability_tuning_log(created_at)",
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
