"""Models for race meetings, races, runners, and results."""

from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import String, Integer, Float, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.models.database import Base


class Meeting(Base):
    """A race meeting at a venue on a specific date."""

    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    venue: Mapped[str] = mapped_column(String(100))
    date: Mapped[date] = mapped_column(Date)
    track_condition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    weather: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    rail_position: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    races: Mapped[List["Race"]] = relationship(
        "Race", back_populates="meeting", cascade="all, delete-orphan"
    )

    def to_dict(self, include_races: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "venue": self.venue,
            "date": self.date.isoformat(),
            "track_condition": self.track_condition,
            "weather": self.weather,
            "rail_position": self.rail_position,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_races:
            data["races"] = [r.to_dict(include_runners=True) for r in self.races]
        return data


class Race(Base):
    """A single race within a meeting."""

    __tablename__ = "races"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    race_number: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    distance: Mapped[int] = mapped_column(Integer)  # meters
    class_: Mapped[Optional[str]] = mapped_column("class", String(100), nullable=True)
    prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled, running, finished
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="races")
    runners: Mapped[List["Runner"]] = relationship(
        "Runner", back_populates="race", cascade="all, delete-orphan"
    )
    results: Mapped[List["Result"]] = relationship(
        "Result", back_populates="race", cascade="all, delete-orphan"
    )

    def to_dict(self, include_runners: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "race_number": self.race_number,
            "name": self.name,
            "distance": self.distance,
            "class": self.class_,
            "prize_money": self.prize_money,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "status": self.status,
        }
        if include_runners:
            data["runners"] = [r.to_dict() for r in self.runners]
        return data


class Runner(Base):
    """A horse running in a race."""

    __tablename__ = "runners"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    race_id: Mapped[str] = mapped_column(String(64), ForeignKey("races.id"))
    horse_name: Mapped[str] = mapped_column(String(100))
    barrier: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # kg
    jockey: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    trainer: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    form: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # e.g., "1x23"
    career_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # e.g., "5-2-1-1"
    speed_map_position: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )  # leader, on_pace, midfield, backmarker
    current_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opening_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scratched: Mapped[bool] = mapped_column(default=False)
    scratching_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Form comments
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    race: Mapped["Race"] = relationship("Race", back_populates="runners")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "race_id": self.race_id,
            "horse_name": self.horse_name,
            "barrier": self.barrier,
            "weight": self.weight,
            "jockey": self.jockey,
            "trainer": self.trainer,
            "form": self.form,
            "career_record": self.career_record,
            "speed_map_position": self.speed_map_position,
            "current_odds": self.current_odds,
            "opening_odds": self.opening_odds,
            "scratched": self.scratched,
            "scratching_reason": self.scratching_reason,
            "comments": self.comments,
        }


class Result(Base):
    """Race result for a runner."""

    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    race_id: Mapped[str] = mapped_column(String(64), ForeignKey("races.id"))
    runner_id: Mapped[str] = mapped_column(String(64), ForeignKey("runners.id"))
    position: Mapped[int] = mapped_column(Integer)
    margin: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    finish_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    dividend_win: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dividend_place: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    race: Mapped["Race"] = relationship("Race", back_populates="results")
    runner: Mapped["Runner"] = relationship("Runner")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "race_id": self.race_id,
            "runner_id": self.runner_id,
            "position": self.position,
            "margin": self.margin,
            "finish_time": self.finish_time,
            "dividend_win": self.dividend_win,
            "dividend_place": self.dividend_place,
        }
