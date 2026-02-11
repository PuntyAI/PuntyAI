"""Pick model — tracks individual tips/bets from early mail for P&L settlement."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Index, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


class Pick(Base):
    """A single pick extracted from approved early mail content."""

    __tablename__ = "picks"
    __table_args__ = (
        Index("ix_picks_meeting_race", "meeting_id", "race_number"),
        Index("ix_picks_content_id", "content_id"),
        Index("ix_picks_settled", "settled"),
    )

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    content_id: Mapped[str] = mapped_column(String(64), ForeignKey("content.id"))
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    race_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    horse_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    saddlecloth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tip_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    odds_at_tip: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    place_odds_at_tip: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Pick type: selection | big3 | big3_multi | exotic | sequence
    pick_type: Mapped[str] = mapped_column(String(20))

    # Exotic fields
    exotic_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    exotic_runners: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    exotic_stake: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Sequence fields
    sequence_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    sequence_variant: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    sequence_legs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON [[saddlecloths]]
    sequence_start_race: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Big3 multi
    multi_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Selection bet type and stake
    bet_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # win, saver_win, place, each_way
    bet_stake: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # dollar amount

    # Estimated return percentage from AI
    estimated_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Probability model outputs
    win_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.0-1.0
    place_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.0-1.0
    value_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # >1.0 = value
    confidence: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # HIGH/MED/LOW
    recommended_stake: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Settlement
    hit: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settled: Mapped[bool] = mapped_column(Boolean, default=False)
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    # Relationships
    content: Mapped["Content"] = relationship("Content")
    meeting: Mapped["Meeting"] = relationship("Meeting")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content_id": self.content_id,
            "meeting_id": self.meeting_id,
            "race_number": self.race_number,
            "horse_name": self.horse_name,
            "saddlecloth": self.saddlecloth,
            "tip_rank": self.tip_rank,
            "odds_at_tip": self.odds_at_tip,
            "place_odds_at_tip": self.place_odds_at_tip,
            "pick_type": self.pick_type,
            "exotic_type": self.exotic_type,
            "exotic_runners": self.exotic_runners,
            "exotic_stake": self.exotic_stake,
            "sequence_type": self.sequence_type,
            "sequence_variant": self.sequence_variant,
            "sequence_legs": self.sequence_legs,
            "sequence_start_race": self.sequence_start_race,
            "multi_odds": self.multi_odds,
            "bet_type": self.bet_type,
            "bet_stake": self.bet_stake,
            "estimated_return_pct": self.estimated_return_pct,
            "win_probability": self.win_probability,
            "place_probability": self.place_probability,
            "value_rating": self.value_rating,
            "confidence": self.confidence,
            "recommended_stake": self.recommended_stake,
            "hit": self.hit,
            "pnl": self.pnl,
            "settled": self.settled,
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "created_at": self.created_at.isoformat(),
        }


# Avoid circular imports
from punty.models.content import Content  # noqa: E402
from punty.models.meeting import Meeting  # noqa: E402
