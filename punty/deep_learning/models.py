"""SQLAlchemy models for the deep learning historical database.

Separate SQLite DB (deep_learning.db) storing ~280K historical runners
imported from Proform data. Used for offline pattern analysis.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


DB_PATH = Path("data/deep_learning.db")


class Base(DeclarativeBase):
    pass


class HistoricalRace(Base):
    """One row per race across all imported meetings."""

    __tablename__ = "historical_races"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, unique=True, nullable=False)  # Proform RaceId
    meeting_date = Column(Date, nullable=False)
    venue = Column(String(100), nullable=False)
    state = Column(String(10))  # NSW, VIC, QLD, SA, WA, TAS, NT, ACT, NZ
    country = Column(String(10))  # AUS, NZL
    location_type = Column(String(5))  # M=Metro, P=Provincial, C=Country
    race_number = Column(Integer)
    race_name = Column(String(200))
    race_class = Column(String(100))
    distance = Column(Integer)  # meters
    track_condition = Column(String(20))  # G4, S5, H8, etc.
    track_condition_number = Column(Integer)
    field_size = Column(Integer)
    prize_money = Column(Float)
    rail_position = Column(String(100))
    official_time = Column(String(20))  # HH:MM:SS.nnnnnnn
    official_time_secs = Column(Float)  # Converted to seconds
    age_restriction = Column(String(30))
    sex_restriction = Column(String(30))
    weight_type = Column(String(30))

    # Sectional data (race-level leader times)
    time_to_finish = Column(Float)
    time_to_1200 = Column(Float)
    time_to_1000 = Column(Float)
    time_to_800 = Column(Float)
    time_to_600 = Column(Float)
    time_to_400 = Column(Float)
    time_to_200 = Column(Float)
    last_600 = Column(Float)
    last_400 = Column(Float)
    last_200 = Column(Float)

    # Weather (from sectionals)
    wind_direction = Column(String(10))
    wind_speed = Column(Float)

    runners = relationship("HistoricalRunner", back_populates="race")
    sectionals = relationship("HistoricalSectional", back_populates="race")

    __table_args__ = (
        Index("ix_hr_venue_date", "venue", "meeting_date"),
        Index("ix_hr_venue_dist_cond", "venue", "distance", "track_condition"),
        Index("ix_hr_date", "meeting_date"),
        Index("ix_hr_state", "state"),
    )


class HistoricalRunner(Base):
    """One row per runner per race. This is the main training data table."""

    __tablename__ = "historical_runners"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_fk = Column(Integer, ForeignKey("historical_races.id"), nullable=False)
    race_id = Column(Integer, nullable=False)  # Proform RaceId
    form_id = Column(Integer)  # Proform FormId (links to sectionals)
    runner_id = Column(Integer)  # Proform RunnerId (horse identity)

    # Identity
    horse_name = Column(String(100), nullable=False)
    tab_no = Column(Integer)
    barrier = Column(Integer)
    original_barrier = Column(Integer)
    weight = Column(Float)
    age = Column(Integer)
    sex = Column(String(20))
    country = Column(String(10))

    # Connections
    jockey = Column(String(100))
    jockey_id = Column(Integer)
    jockey_claim = Column(Float)
    trainer = Column(String(100))
    trainer_id = Column(Integer)

    # Career stats
    career_starts = Column(Integer)
    career_wins = Column(Integer)
    career_seconds = Column(Integer)
    career_thirds = Column(Integer)
    win_pct = Column(Float)
    place_pct = Column(Float)
    prize_money = Column(Float)
    handicap_rating = Column(Float)
    last_10 = Column(String(20))
    prep_runs = Column(Integer)

    # Condition records (JSON: {"Starts":N, "Firsts":N, ...})
    track_record = Column(Text)  # JSON
    distance_record = Column(Text)  # JSON
    track_dist_record = Column(Text)  # JSON
    first_up_record = Column(Text)  # JSON
    second_up_record = Column(Text)  # JSON
    good_record = Column(Text)  # JSON
    soft_record = Column(Text)  # JSON
    heavy_record = Column(Text)  # JSON
    firm_record = Column(Text)  # JSON
    synthetic_record = Column(Text)  # JSON

    # Group records
    group1_record = Column(Text)  # JSON
    group2_record = Column(Text)  # JSON
    group3_record = Column(Text)  # JSON

    # A2E stats (JSON: {"A2E":float, "PoT":float, ...})
    jockey_a2e_career = Column(Text)  # JSON
    jockey_a2e_last100 = Column(Text)  # JSON
    trainer_a2e_career = Column(Text)  # JSON
    trainer_a2e_last100 = Column(Text)  # JSON
    trainer_jockey_a2e_career = Column(Text)  # JSON
    trainer_jockey_a2e_last100 = Column(Text)  # JSON

    # Past form (JSON array of last N starts summary)
    form_history = Column(Text)  # JSON

    # Odds
    starting_price = Column(Float)
    opening_odds = Column(Float)
    mid_odds = Column(Float)

    # In-run positions (parsed from "settling_down,2;m800,3;...")
    settle_pos = Column(Integer)
    pos_800 = Column(Integer)
    pos_400 = Column(Integer)

    # === OUTCOME (labels) ===
    finish_position = Column(Integer)
    margin = Column(Float)
    won = Column(Boolean)
    placed = Column(Boolean)

    # Import tracking
    imported_at = Column(DateTime, default=datetime.utcnow)

    race = relationship("HistoricalRace", back_populates="runners")

    __table_args__ = (
        Index("ix_hrun_race", "race_id"),
        Index("ix_hrun_horse", "runner_id"),
        Index("ix_hrun_jockey", "jockey_id"),
        Index("ix_hrun_trainer", "trainer_id"),
        Index("ix_hrun_barrier", "barrier"),
        Index("ix_hrun_pos", "finish_position"),
        Index("ix_hrun_form_id", "form_id"),
    )


class HistoricalSectional(Base):
    """Per-runner sectional timing and positional data."""

    __tablename__ = "historical_sectionals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_fk = Column(Integer, ForeignKey("historical_races.id"), nullable=False)
    race_id = Column(Integer, nullable=False)
    form_id = Column(Integer)  # Links to HistoricalRunner.form_id
    runner_id = Column(Integer)
    tab_no = Column(Integer)
    runner_name = Column(String(100))

    # Cumulative times
    time_to_1200 = Column(Float)
    time_to_1000 = Column(Float)
    time_to_800 = Column(Float)
    time_to_600 = Column(Float)
    time_to_400 = Column(Float)
    time_to_200 = Column(Float)
    time_to_100 = Column(Float)
    time_to_fin = Column(Float)

    # Split times
    last_1200 = Column(Float)
    last_1000 = Column(Float)
    last_800 = Column(Float)
    last_600 = Column(Float)
    last_400 = Column(Float)
    last_200 = Column(Float)
    last_100 = Column(Float)

    # Positional calls
    pos_1200 = Column(Integer)
    pos_1000 = Column(Integer)
    pos_800 = Column(Integer)
    pos_600 = Column(Integer)
    pos_400 = Column(Integer)
    pos_200 = Column(Integer)
    pos_100 = Column(Integer)
    pos_fin = Column(Integer)

    # Margins at each call
    marg_1200 = Column(Float)
    marg_1000 = Column(Float)
    marg_800 = Column(Float)
    marg_600 = Column(Float)
    marg_400 = Column(Float)
    marg_200 = Column(Float)
    marg_100 = Column(Float)
    marg_fin = Column(Float)

    # Track width (3=rail, higher=wider)
    wides_800 = Column(Integer)
    wides_600 = Column(Integer)
    wides_400 = Column(Integer)
    wides_200 = Column(Integer)
    wides_fin = Column(Integer)

    # Meeting ranks (how this runner compares to others at same meeting)
    meeting_rank_6f = Column(Integer)
    meeting_rank_4f = Column(Integer)
    meeting_rank_2f = Column(Integer)
    meeting_rank_1f = Column(Integer)

    race = relationship("HistoricalRace", back_populates="sectionals")

    __table_args__ = (
        Index("ix_hsect_race", "race_id"),
        Index("ix_hsect_form", "form_id"),
        Index("ix_hsect_runner", "runner_id"),
    )


# --- Database setup ---

def get_engine(db_path: str | Path | None = None):
    """Create SQLite engine for the deep learning DB."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )


def get_session(db_path: str | Path | None = None) -> Session:
    """Get a sync session for the deep learning DB."""
    engine = get_engine(db_path)
    factory = sessionmaker(bind=engine)
    return factory()


def init_db(db_path: str | Path | None = None):
    """Create all tables in the deep learning DB."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
