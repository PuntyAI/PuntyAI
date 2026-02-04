"""Models for race meetings, races, runners, and results."""

from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import Boolean, Index, String, Integer, Float, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


class Meeting(Base):
    """A race meeting at a venue on a specific date."""

    __tablename__ = "meetings"
    __table_args__ = (Index("ix_meetings_date", "date"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    venue: Mapped[str] = mapped_column(String(100))
    date: Mapped[date] = mapped_column(Date)
    track_condition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    weather: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    rail_position: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    meeting_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # race, trial, jumpout
    speed_map_complete: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # None=not checked, True=complete, False=incomplete

    # New GraphQL fields
    penetrometer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weather_condition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    weather_temp: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weather_wind_speed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weather_wind_dir: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    rail_bias_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
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
            "selected": self.selected,
            "source": self.source,
            "meeting_type": self.meeting_type,
            "speed_map_complete": self.speed_map_complete,
            "penetrometer": self.penetrometer,
            "weather_condition": self.weather_condition,
            "weather_temp": self.weather_temp,
            "weather_wind_speed": self.weather_wind_speed,
            "weather_wind_dir": self.weather_wind_dir,
            "rail_bias_comment": self.rail_bias_comment,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_races:
            data["races"] = [r.to_dict(include_runners=True) for r in self.races]
        return data


class Race(Base):
    """A single race within a meeting."""

    __tablename__ = "races"
    __table_args__ = (Index("ix_races_meeting_id", "meeting_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("meetings.id"))
    race_number: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    distance: Mapped[int] = mapped_column(Integer)  # meters
    class_: Mapped[Optional[str]] = mapped_column("class", String(100), nullable=True)
    prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")

    # New GraphQL fields
    track_condition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    race_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    age_restriction: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    weight_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    field_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Result fields
    winning_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    results_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    exotic_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
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
            "track_condition": self.track_condition,
            "race_type": self.race_type,
            "age_restriction": self.age_restriction,
            "weight_type": self.weight_type,
            "field_size": self.field_size,
            "winning_time": self.winning_time,
            "results_status": self.results_status,
            "exotic_results": self.exotic_results,
        }
        if include_runners:
            data["runners"] = [r.to_dict() for r in self.runners]
        return data


class Runner(Base):
    """A horse running in a race."""

    __tablename__ = "runners"
    __table_args__ = (
        Index("ix_runners_race_id", "race_id"),
        Index("ix_runners_finish_position", "race_id", "finish_position"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    race_id: Mapped[str] = mapped_column(String(64), ForeignKey("races.id"))
    horse_name: Mapped[str] = mapped_column(String(100))
    saddlecloth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    barrier: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    jockey: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    trainer: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    form: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    career_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    speed_map_position: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Punting Form insights
    pf_speed_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-25, lower = faster early speed
    pf_settle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Historical avg settling position
    pf_map_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # >1.0 = pace advantage, <1.0 = disadvantage
    pf_jockey_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Jockey effectiveness factor

    current_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opening_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scratched: Mapped[bool] = mapped_column(default=False)
    scratching_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Horse details
    horse_age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    horse_sex: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    horse_colour: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    sire: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dam: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dam_sire: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Performance
    career_prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_five: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    days_since_last_run: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    handicap_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Stats (JSON text columns)
    track_dist_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    track_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    distance_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_up_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    second_up_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    good_track_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    soft_track_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    heavy_track_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    jockey_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    class_stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Gear & stewards
    gear: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    gear_changes: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    stewards_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Form comments
    comment_long: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    comment_short: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Odds (multi-provider)
    odds_tab: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_sportsbet: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_bet365: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_ladbrokes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_betfair: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_flucs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Result fields
    finish_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    result_margin: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    starting_price: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    win_dividend: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    place_dividend: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sectional_400: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    sectional_800: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Full form history (JSON: list of past starts)
    form_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Trainer details
    trainer_location: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    race: Mapped["Race"] = relationship("Race", back_populates="runners")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "race_id": self.race_id,
            "horse_name": self.horse_name,
            "saddlecloth": self.saddlecloth,
            "barrier": self.barrier,
            "weight": self.weight,
            "jockey": self.jockey,
            "trainer": self.trainer,
            "form": self.form,
            "career_record": self.career_record,
            "speed_map_position": self.speed_map_position,
            "pf_speed_rank": self.pf_speed_rank,
            "pf_settle": self.pf_settle,
            "pf_map_factor": self.pf_map_factor,
            "pf_jockey_factor": self.pf_jockey_factor,
            "current_odds": self.current_odds,
            "opening_odds": self.opening_odds,
            "scratched": self.scratched,
            "scratching_reason": self.scratching_reason,
            "comments": self.comments,
            "horse_age": self.horse_age,
            "horse_sex": self.horse_sex,
            "horse_colour": self.horse_colour,
            "sire": self.sire,
            "dam": self.dam,
            "dam_sire": self.dam_sire,
            "career_prize_money": self.career_prize_money,
            "last_five": self.last_five,
            "days_since_last_run": self.days_since_last_run,
            "handicap_rating": self.handicap_rating,
            "speed_value": self.speed_value,
            "track_dist_stats": self.track_dist_stats,
            "track_stats": self.track_stats,
            "distance_stats": self.distance_stats,
            "first_up_stats": self.first_up_stats,
            "second_up_stats": self.second_up_stats,
            "good_track_stats": self.good_track_stats,
            "soft_track_stats": self.soft_track_stats,
            "heavy_track_stats": self.heavy_track_stats,
            "jockey_stats": self.jockey_stats,
            "class_stats": self.class_stats,
            "gear": self.gear,
            "gear_changes": self.gear_changes,
            "stewards_comment": self.stewards_comment,
            "comment_long": self.comment_long,
            "comment_short": self.comment_short,
            "odds_tab": self.odds_tab,
            "odds_sportsbet": self.odds_sportsbet,
            "odds_bet365": self.odds_bet365,
            "odds_ladbrokes": self.odds_ladbrokes,
            "odds_betfair": self.odds_betfair,
            "odds_flucs": self.odds_flucs,
            "trainer_location": self.trainer_location,
            "finish_position": self.finish_position,
            "result_margin": self.result_margin,
            "starting_price": self.starting_price,
            "win_dividend": self.win_dividend,
            "place_dividend": self.place_dividend,
            "sectional_400": self.sectional_400,
            "sectional_800": self.sectional_800,
            "form_history": self.form_history,
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)

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
