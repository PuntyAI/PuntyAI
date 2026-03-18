"""ORM models for analytics and tracking tables."""

from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, Boolean, UniqueConstraint

from punty.models.database import Base


class TokenUsage(Base):
    """Tracks AI API token usage and costs per generation."""

    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(50), nullable=False)
    content_type = Column(String(30))
    meeting_id = Column(String(64))
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    reasoning_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class RaceMemory(Base):
    """Stores per-runner prediction memories for RAG retrieval."""

    __tablename__ = "race_memories"
    __table_args__ = (
        UniqueConstraint("race_id", "saddlecloth", name="uq_race_memories_race_sc"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(String, nullable=False)
    race_number = Column(Integer, nullable=False)
    race_id = Column(String, nullable=False)
    context_json = Column(Text, nullable=False)
    runner_json = Column(Text, nullable=False)
    horse_name = Column(String, nullable=False)
    saddlecloth = Column(Integer, nullable=False)
    tip_rank = Column(Integer, nullable=False)
    confidence = Column(String)
    odds_at_tip = Column(Float)
    bet_type = Column(String)
    finish_position = Column(Integer)
    hit = Column(Boolean)
    pnl = Column(Float)
    sp_odds = Column(Float)
    embedding_json = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    settled_at = Column(DateTime)


class PatternInsight(Base):
    """Stores discovered betting patterns and their performance metrics."""

    __tablename__ = "pattern_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_type = Column(String, nullable=False)
    pattern_key = Column(String, nullable=False)
    sample_count = Column(Integer, default=0)
    hit_rate = Column(Float, default=0.0)
    avg_pnl = Column(Float, default=0.0)
    avg_odds = Column(Float, default=0.0)
    insight_text = Column(Text, nullable=False)
    conditions_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
