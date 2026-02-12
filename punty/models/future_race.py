"""Models for upcoming Group race nominations and forecasting."""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime, Date, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


class FutureRace(Base):
    """An upcoming race (Group 1/2/3 or Listed) discovered from PF API."""

    __tablename__ = "future_races"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # venue-date-rN
    venue: Mapped[str] = mapped_column(String(100))
    date: Mapped[date] = mapped_column(Date)
    race_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    race_name: Mapped[str] = mapped_column(String(200))
    group_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Group 1, Group 2, Listed
    distance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    state: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    nominations: Mapped[list["FutureNomination"]] = relationship(
        "FutureNomination", back_populates="future_race", cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "venue": self.venue,
            "date": self.date.isoformat() if self.date else None,
            "race_number": self.race_number,
            "race_name": self.race_name,
            "group_level": self.group_level,
            "distance": self.distance,
            "prize_money": self.prize_money,
            "state": self.state,
            "nominations": [n.to_dict() for n in (self.nominations or [])],
        }


class FutureNomination(Base):
    """A nominated horse for an upcoming Group race."""

    __tablename__ = "future_nominations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    future_race_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("future_races.id", ondelete="CASCADE"),
    )
    horse_name: Mapped[str] = mapped_column(String(100))
    trainer: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    jockey: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    barrier: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_start: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    career_record: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    future_race: Mapped["FutureRace"] = relationship("FutureRace", back_populates="nominations")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "horse_name": self.horse_name,
            "trainer": self.trainer,
            "jockey": self.jockey,
            "barrier": self.barrier,
            "weight": self.weight,
            "last_start": self.last_start,
            "career_record": self.career_record,
        }
