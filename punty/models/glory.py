"""Models for Group One Glory tipping competition."""

from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import Boolean, Index, String, Integer, Float, Date, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


class G1User(Base):
    """Group One Glory user (separate from Punty auth)."""

    __tablename__ = "g1_users"
    __table_args__ = (Index("ix_g1_users_email", "email"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(100))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    picks: Mapped[List["G1Pick"]] = relationship(
        "G1Pick", back_populates="user", cascade="all, delete-orphan"
    )
    honor_entries: Mapped[List["G1HonorBoard"]] = relationship(
        "G1HonorBoard", back_populates="user"
    )

    def to_dict(self, include_picks: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_picks:
            data["picks"] = [p.to_dict() for p in self.picks]
        return data


class G1Competition(Base):
    """A tipping competition (e.g., Autumn Carnival 2026)."""

    __tablename__ = "g1_competitions"
    __table_args__ = (Index("ix_g1_competitions_active", "is_active"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))  # "Autumn Carnival 2026"
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    prize_pool: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Total prize in cents
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    races: Mapped[List["G1Race"]] = relationship(
        "G1Race", back_populates="competition", cascade="all, delete-orphan"
    )
    honor_entries: Mapped[List["G1HonorBoard"]] = relationship(
        "G1HonorBoard", back_populates="competition"
    )

    def to_dict(self, include_races: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "is_active": self.is_active,
            "prize_pool": self.prize_pool,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_races:
            data["races"] = [r.to_dict() for r in self.races]
        return data


class G1Race(Base):
    """A Group 1 race in a competition."""

    __tablename__ = "g1_races"
    __table_args__ = (
        Index("ix_g1_races_competition_id", "competition_id"),
        Index("ix_g1_races_race_date", "race_date"),
        Index("ix_g1_races_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    competition_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_competitions.id"))
    external_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Racing Australia key
    race_name: Mapped[str] = mapped_column(String(200))  # "Lexus Melbourne Cup"
    venue: Mapped[str] = mapped_column(String(100))  # "Flemington"
    race_date: Mapped[datetime] = mapped_column(DateTime)
    distance: Mapped[int] = mapped_column(Integer)  # meters
    prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="nominations")  # nominations | final_field | open | closed | resulted
    tipping_closes_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    race_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    competition: Mapped["G1Competition"] = relationship("G1Competition", back_populates="races")
    horses: Mapped[List["G1Horse"]] = relationship(
        "G1Horse", back_populates="race", cascade="all, delete-orphan"
    )
    picks: Mapped[List["G1Pick"]] = relationship(
        "G1Pick", back_populates="race", cascade="all, delete-orphan"
    )

    def to_dict(self, include_horses: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "competition_id": self.competition_id,
            "external_id": self.external_id,
            "race_name": self.race_name,
            "venue": self.venue,
            "race_date": self.race_date.isoformat(),
            "distance": self.distance,
            "prize_money": self.prize_money,
            "status": self.status,
            "tipping_closes_at": self.tipping_closes_at.isoformat() if self.tipping_closes_at else None,
            "race_number": self.race_number,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_horses:
            data["horses"] = [h.to_dict() for h in self.horses]
        return data


class G1Horse(Base):
    """A horse in a Group 1 race."""

    __tablename__ = "g1_horses"
    __table_args__ = (
        Index("ix_g1_horses_race_id", "race_id"),
        Index("ix_g1_horses_finish_position", "race_id", "finish_position"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    race_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_races.id"))
    name: Mapped[str] = mapped_column(String(100))
    barrier: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    jockey: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    trainer: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    form: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "2-1-1-3"
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    career_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "15: 6-2-2"
    career_prize: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    saddlecloth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_scratched: Mapped[bool] = mapped_column(Boolean, default=False)
    scratching_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    finish_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1, 2, 3, etc after result

    # Additional form data
    horse_age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    horse_sex: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    sire: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dam: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_five: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    race: Mapped["G1Race"] = relationship("G1Race", back_populates="horses")
    picks: Mapped[List["G1Pick"]] = relationship("G1Pick", back_populates="horse")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "race_id": self.race_id,
            "name": self.name,
            "barrier": self.barrier,
            "weight": self.weight,
            "jockey": self.jockey,
            "trainer": self.trainer,
            "odds": self.odds,
            "form": self.form,
            "rating": self.rating,
            "career_record": self.career_record,
            "career_prize": self.career_prize,
            "saddlecloth": self.saddlecloth,
            "is_scratched": self.is_scratched,
            "scratching_reason": self.scratching_reason,
            "finish_position": self.finish_position,
            "horse_age": self.horse_age,
            "horse_sex": self.horse_sex,
            "sire": self.sire,
            "dam": self.dam,
            "last_five": self.last_five,
            "comment": self.comment,
        }


class G1Pick(Base):
    """A user's pick for a race."""

    __tablename__ = "g1_picks"
    __table_args__ = (
        UniqueConstraint("user_id", "race_id", name="uq_g1_picks_user_race"),
        Index("ix_g1_picks_user_id", "user_id"),
        Index("ix_g1_picks_race_id", "race_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_users.id"))
    race_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_races.id"))
    horse_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_horses.id"))
    points_earned: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Set after race result
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    user: Mapped["G1User"] = relationship("G1User", back_populates="picks")
    race: Mapped["G1Race"] = relationship("G1Race", back_populates="picks")
    horse: Mapped["G1Horse"] = relationship("G1Horse", back_populates="picks")

    def to_dict(self, include_horse: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "race_id": self.race_id,
            "horse_id": self.horse_id,
            "points_earned": self.points_earned,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_horse:
            data["horse"] = self.horse.to_dict()
        return data


class G1HonorBoard(Base):
    """Historical competition winners."""

    __tablename__ = "g1_honor_board"
    __table_args__ = (
        Index("ix_g1_honor_board_competition_id", "competition_id"),
        Index("ix_g1_honor_board_user_id", "user_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    competition_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_competitions.id"))
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("g1_users.id"))
    final_points: Mapped[int] = mapped_column(Integer)
    final_rank: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

    # Relationships
    competition: Mapped["G1Competition"] = relationship("G1Competition", back_populates="honor_entries")
    user: Mapped["G1User"] = relationship("G1User", back_populates="honor_entries")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "competition_id": self.competition_id,
            "user_id": self.user_id,
            "final_points": self.final_points,
            "final_rank": self.final_rank,
            "created_at": self.created_at.isoformat(),
            "competition_name": self.competition.name if self.competition else None,
            "user_display_name": self.user.display_name if self.user else None,
        }
