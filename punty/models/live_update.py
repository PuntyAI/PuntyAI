"""Model for live race-day updates (celebrations, pace analysis)."""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from punty.config import melb_now_naive
from punty.models.database import Base


class LiveUpdate(Base):
    """A live update posted during a race meeting (celebration or pace analysis)."""

    __tablename__ = "live_updates"
    __table_args__ = (
        Index("ix_live_updates_meeting_id", "meeting_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    race_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    update_type: Mapped[str] = mapped_column(String(20))  # "celebration" or "pace_analysis"
    content: Mapped[str] = mapped_column(Text)
    tweet_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    parent_tweet_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    horse_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    odds: Mapped[Optional[float]] = mapped_column(nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "race_number": self.race_number,
            "update_type": self.update_type,
            "content": self.content,
            "tweet_id": self.tweet_id,
            "horse_name": self.horse_name,
            "odds": self.odds,
            "pnl": self.pnl,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
