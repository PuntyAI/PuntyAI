"""Memory models for storing race predictions and outcomes."""

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from punty.config import melb_now_naive
from punty.models.database import Base


class RaceMemory(Base):
    """Stores a prediction and its outcome for learning."""

    __tablename__ = "race_memories"
    __table_args__ = (
        UniqueConstraint("race_id", "saddlecloth", name="uq_race_memory_race_saddlecloth"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Race identification
    meeting_id: Mapped[str] = mapped_column(String, index=True)
    race_number: Mapped[int] = mapped_column(Integer)
    race_id: Mapped[str] = mapped_column(String, index=True)  # Not unique alone - composite with saddlecloth

    # Race context (stored as JSON)
    # Includes: track condition, distance, class, field size, tempo, rail position
    context_json: Mapped[str] = mapped_column(Text)

    # Runner context for the prediction (stored as JSON)
    # Includes: form, odds, barrier, jockey, speed map position, market movement
    runner_json: Mapped[str] = mapped_column(Text)

    # Prediction made
    horse_name: Mapped[str] = mapped_column(String)
    saddlecloth: Mapped[int] = mapped_column(Integer)
    tip_rank: Mapped[int] = mapped_column(Integer)  # 1=top pick, 2, 3, 4=roughie
    confidence: Mapped[str] = mapped_column(String, nullable=True)  # high/med/low
    odds_at_tip: Mapped[float] = mapped_column(Float, nullable=True)
    bet_type: Mapped[str] = mapped_column(String, nullable=True)  # win/place/each_way

    # Outcome
    finish_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hit: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sp_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Embedding for similarity search (JSON array of floats)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    @property
    def context(self) -> dict[str, Any]:
        """Get context as dictionary."""
        if self.context_json:
            return json.loads(self.context_json)
        return {}

    @context.setter
    def context(self, value: dict[str, Any]):
        """Set context from dictionary."""
        self.context_json = json.dumps(value)

    @property
    def runner(self) -> dict[str, Any]:
        """Get runner data as dictionary."""
        if self.runner_json:
            return json.loads(self.runner_json)
        return {}

    @runner.setter
    def runner(self, value: dict[str, Any]):
        """Set runner data from dictionary."""
        self.runner_json = json.dumps(value)

    @property
    def embedding(self) -> list[float] | None:
        """Get embedding as list of floats."""
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None

    @embedding.setter
    def embedding(self, value: list[float] | None):
        """Set embedding from list of floats."""
        if value is not None:
            self.embedding_json = json.dumps(value)
        else:
            self.embedding_json = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "race_id": self.race_id,
            "horse_name": self.horse_name,
            "saddlecloth": self.saddlecloth,
            "tip_rank": self.tip_rank,
            "confidence": self.confidence,
            "odds_at_tip": self.odds_at_tip,
            "bet_type": self.bet_type,
            "finish_position": self.finish_position,
            "hit": self.hit,
            "pnl": self.pnl,
            "sp_odds": self.sp_odds,
            "context": self.context,
            "runner": self.runner,
        }


class PatternInsight(Base):
    """Stores learned patterns and insights from accumulated memories."""

    __tablename__ = "pattern_insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Pattern identification
    pattern_type: Mapped[str] = mapped_column(String, index=True)
    # e.g., "track_condition", "distance_class", "barrier_bias", "market_move"

    pattern_key: Mapped[str] = mapped_column(String, index=True)
    # e.g., "heavy_1200_maiden", "barrier_1_caulfield", "heavy_support_first_up"

    # Pattern statistics
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    hit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    avg_odds: Mapped[float] = mapped_column(Float, default=0.0)

    # Human-readable insight
    insight_text: Mapped[str] = mapped_column(Text)

    # When this pattern applies
    conditions_json: Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive, onupdate=melb_now_naive)

    @property
    def conditions(self) -> dict[str, Any]:
        """Get conditions as dictionary."""
        if self.conditions_json:
            return json.loads(self.conditions_json)
        return {}

    @conditions.setter
    def conditions(self, value: dict[str, Any]):
        """Set conditions from dictionary."""
        self.conditions_json = json.dumps(value)


class RaceAssessment(Base):
    """Stores post-race LLM assessments for learning from predictions.

    After each race settles, the LLM reviews what happened vs what was predicted
    and extracts structured learnings. These are stored here for RAG retrieval
    when generating future Early Mail for similar races.
    """

    __tablename__ = "race_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Race identification
    race_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    meeting_id: Mapped[str] = mapped_column(String, index=True)
    race_number: Mapped[int] = mapped_column(Integer)

    # Filterable attributes (denormalized for fast SQL queries)
    track: Mapped[str] = mapped_column(String, index=True)
    distance: Mapped[int] = mapped_column(Integer, index=True)
    race_class: Mapped[str] = mapped_column(String, index=True)
    going: Mapped[str] = mapped_column(String, index=True)
    rail_position: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Full structured assessment from LLM (JSON)
    assessment_json: Mapped[str] = mapped_column(Text)

    # Summary text for embedding and quick retrieval
    key_learnings: Mapped[str] = mapped_column(Text)

    # Embedding for similarity search (JSON array of floats)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Prediction accuracy metrics
    top_pick_hit: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    any_pick_hit: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    total_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    @property
    def assessment(self) -> dict[str, Any]:
        """Get assessment as dictionary."""
        if self.assessment_json:
            return json.loads(self.assessment_json)
        return {}

    @assessment.setter
    def assessment(self, value: dict[str, Any]):
        """Set assessment from dictionary."""
        self.assessment_json = json.dumps(value)

    @property
    def embedding(self) -> list[float] | None:
        """Get embedding as list of floats."""
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None

    @embedding.setter
    def embedding(self, value: list[float] | None):
        """Set embedding from list of floats."""
        if value is not None:
            self.embedding_json = json.dumps(value)
        else:
            self.embedding_json = None

    def to_retrieval_dict(self) -> dict[str, Any]:
        """Convert to dictionary for RAG retrieval context."""
        return {
            "track": self.track,
            "distance": self.distance,
            "class": self.race_class,
            "going": self.going,
            "key_learnings": self.key_learnings,
            "top_pick_hit": self.top_pick_hit,
            "total_pnl": self.total_pnl,
        }
