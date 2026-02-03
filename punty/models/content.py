"""Models for content generation, context, and scheduling."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Index, String, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


class ContentStatus(str, Enum):
    """Status of generated content."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGENERATING = "regenerating"
    SCHEDULED = "scheduled"
    SENT = "sent"
    SUPERSEDED = "superseded"


class ContentType(str, Enum):
    """Types of content Punty can generate."""

    EARLY_MAIL = "early_mail"
    RACE_PREVIEW = "race_preview"
    RESULTS = "results"
    UPDATE_ALERT = "update_alert"
    ENGAGEMENT = "engagement"
    MEETING_WRAPUP = "meeting_wrapup"


class ContextSnapshot(Base):
    """A snapshot of data context at a point in time."""

    __tablename__ = "context_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    version: Mapped[int] = mapped_column(Integer)
    data_hash: Mapped[str] = mapped_column(String(64))  # Hash to detect changes
    snapshot_json: Mapped[str] = mapped_column(Text)  # Full context as JSON
    significant_changes: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON list of changes from previous
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "version": self.version,
            "data_hash": self.data_hash,
            "significant_changes": self.significant_changes,
            "created_at": self.created_at.isoformat(),
        }


# Import Meeting for relationship
from punty.models.meeting import Meeting  # noqa: E402


class Content(Base):
    """Generated content item."""

    __tablename__ = "content"
    __table_args__ = (
        Index("ix_content_meeting_id", "meeting_id"),
        Index("ix_content_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    race_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("races.id"), nullable=True
    )
    content_type: Mapped[str] = mapped_column(String(50))
    context_snapshot_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("context_snapshots.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default=ContentStatus.DRAFT.value)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=True)

    # Content
    raw_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    whatsapp_formatted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    twitter_formatted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Review
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Scheduling
    scheduled_send_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sent_to_whatsapp: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_to_twitter: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting")
    context_snapshot: Mapped[Optional["ContextSnapshot"]] = relationship("ContextSnapshot")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "race_id": self.race_id,
            "content_type": self.content_type,
            "context_snapshot_id": self.context_snapshot_id,
            "status": self.status,
            "requires_review": self.requires_review,
            "raw_content": self.raw_content,
            "whatsapp_formatted": self.whatsapp_formatted,
            "twitter_formatted": self.twitter_formatted,
            "review_notes": self.review_notes,
            "scheduled_send_at": (
                self.scheduled_send_at.isoformat() if self.scheduled_send_at else None
            ),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "sent_to_whatsapp": self.sent_to_whatsapp,
            "sent_to_twitter": self.sent_to_twitter,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ScheduledJob(Base):
    """Scheduled background job configuration."""

    __tablename__ = "scheduled_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    job_type: Mapped[str] = mapped_column(String(50))  # scrape_race_cards, scrape_speed_maps, etc.
    cron_expression: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    next_run: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_run: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "job_type": self.job_type,
            "cron_expression": self.cron_expression,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "enabled": self.enabled,
        }
