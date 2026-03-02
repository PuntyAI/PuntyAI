"""BetfairBet model — tracks lifecycle of automated Betfair place bets."""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


class BetfairBet(Base):
    """A single automated Betfair place bet, from queue through settlement."""

    __tablename__ = "betfair_bets"
    __table_args__ = (
        UniqueConstraint("meeting_id", "race_number", name="uq_betfair_bet_meeting_race"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # e.g. bf-sale-2026-03-02-r1
    pick_id: Mapped[Optional[str]] = mapped_column(String(16), ForeignKey("picks.id"), nullable=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    race_number: Mapped[int] = mapped_column(Integer)

    horse_name: Mapped[str] = mapped_column(String(100))
    saddlecloth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Betfair market data (resolved before placement)
    market_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    selection_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Bet parameters
    bet_type: Mapped[str] = mapped_column(String(10), default="place")  # "place" or "win"
    stake: Mapped[float] = mapped_column(Float, default=2.00)
    requested_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    matched_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Queue control
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Status: queued → placing → placed → settled (or failed/cancelled/lapsed)
    status: Mapped[str] = mapped_column(String(20), default="queued")

    # Betfair response
    bet_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    size_matched: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_price_matched: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Settlement
    hit: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settled: Mapped[bool] = mapped_column(Boolean, default=False)
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Scheduling
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # race_time - 5min
    placed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    # Relationships
    pick: Mapped[Optional["Pick"]] = relationship("Pick")
    meeting: Mapped["Meeting"] = relationship("Meeting")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pick_id": self.pick_id,
            "meeting_id": self.meeting_id,
            "race_number": self.race_number,
            "horse_name": self.horse_name,
            "saddlecloth": self.saddlecloth,
            "market_id": self.market_id,
            "selection_id": self.selection_id,
            "bet_type": self.bet_type,
            "stake": self.stake,
            "requested_odds": self.requested_odds,
            "matched_odds": self.matched_odds,
            "enabled": self.enabled,
            "status": self.status,
            "bet_id": self.bet_id,
            "size_matched": self.size_matched,
            "average_price_matched": self.average_price_matched,
            "error_message": self.error_message,
            "hit": self.hit,
            "pnl": self.pnl,
            "settled": self.settled,
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "placed_at": self.placed_at.isoformat() if self.placed_at else None,
            "created_at": self.created_at.isoformat(),
        }


# Avoid circular imports
from punty.models.pick import Pick  # noqa: E402
from punty.models.meeting import Meeting  # noqa: E402
