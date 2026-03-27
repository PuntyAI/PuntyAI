"""SQLAlchemy models for greyhound racing — VIC, NSW, QLD.

Separate table namespace (greyhound_*) to avoid any collision with thoroughbred tables.
Key differences from thoroughbred models:
  - Box number 1-8 (not saddlecloth 1-24), reserves 9-10
  - No jockey, no weight, no gear
  - Grades instead of classes (C0-C6, Gr1-5, Maiden, FFA, Masters, Novice)
  - Run times and split times are core performance metrics
  - Track type (oval vs straight) matters significantly
  - Iggy-Joey model predictions as a data source
"""

from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import Boolean, Index, String, Integer, Float, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from punty.config import melb_now_naive
from punty.models.database import Base


# ---------------------------------------------------------------------------
# Greyhound venue registry — VIC, NSW, QLD tracks
# ---------------------------------------------------------------------------

# fmt: off
GREYHOUND_VENUES: dict[str, dict] = {
    # VIC
    "Bendigo":       {"state": "VIC", "track_type": "oval", "distances": [400, 460, 545]},
    "Cranbourne":    {"state": "VIC", "track_type": "oval", "distances": [311, 520, 600, 699]},
    "Geelong":       {"state": "VIC", "track_type": "oval", "distances": [350, 457, 545, 600]},
    "Healesville":   {"state": "VIC", "track_type": "straight", "distances": [300, 350]},
    "Horsham":       {"state": "VIC", "track_type": "oval", "distances": [410, 485, 570]},
    "Meadows":       {"state": "VIC", "track_type": "oval", "distances": [295, 460, 525, 600, 725]},
    "Sale":          {"state": "VIC", "track_type": "oval", "distances": [440, 520, 650]},
    "Sandown Park":  {"state": "VIC", "track_type": "oval", "distances": [295, 460, 515, 595, 715]},
    "Shepparton":    {"state": "VIC", "track_type": "oval", "distances": [390, 450, 550]},
    "Traralgon":     {"state": "VIC", "track_type": "oval", "distances": [395, 455, 525]},
    "Warragul":      {"state": "VIC", "track_type": "oval", "distances": [400, 460, 555]},
    "Warrnambool":   {"state": "VIC", "track_type": "oval", "distances": [390, 450, 560]},
    "Ballarat":      {"state": "VIC", "track_type": "oval", "distances": [390, 450, 545]},
    # NSW
    "Bulli":         {"state": "NSW", "track_type": "oval", "distances": [340, 400, 472, 536]},
    "Casino":        {"state": "NSW", "track_type": "oval", "distances": [305, 400, 480]},
    "Dapto":         {"state": "NSW", "track_type": "oval", "distances": [297, 340, 400, 520]},
    "Dubbo":         {"state": "NSW", "track_type": "oval", "distances": [305, 400, 500]},
    "Garden City":   {"state": "NSW", "track_type": "oval", "distances": [305, 407, 500]},
    "Gosford":       {"state": "NSW", "track_type": "oval", "distances": [315, 405, 500, 600]},
    "Grafton":       {"state": "NSW", "track_type": "oval", "distances": [305, 407, 480]},
    "Gunnedah":      {"state": "NSW", "track_type": "oval", "distances": [305, 407, 500]},
    "Lismore":       {"state": "NSW", "track_type": "oval", "distances": [305, 407, 500]},
    "Maitland":      {"state": "NSW", "track_type": "oval", "distances": [315, 400, 485]},
    "Muswellbrook":  {"state": "NSW", "track_type": "oval", "distances": [280, 370, 462]},
    "Newcastle":     {"state": "NSW", "track_type": "oval", "distances": [335, 400, 480, 575]},
    "Nowra":         {"state": "NSW", "track_type": "oval", "distances": [305, 400, 520]},
    "Richmond":      {"state": "NSW", "track_type": "oval", "distances": [335, 400, 535, 618]},
    "Tamworth":      {"state": "NSW", "track_type": "oval", "distances": [280, 340, 407, 480]},
    "Temora":        {"state": "NSW", "track_type": "oval", "distances": [305, 407, 480]},
    "The Gardens":   {"state": "NSW", "track_type": "oval", "distances": [315, 407, 515, 600]},
    "Wagga":         {"state": "NSW", "track_type": "oval", "distances": [305, 400, 500]},
    "Wentworth Park":{"state": "NSW", "track_type": "oval", "distances": [295, 420, 520, 720]},
    # QLD
    "Albion Park":   {"state": "QLD", "track_type": "oval", "distances": [295, 350, 520, 600, 710]},
    "Bundaberg":     {"state": "QLD", "track_type": "oval", "distances": [310, 400, 490]},
    "Capalaba":      {"state": "QLD", "track_type": "oval", "distances": [340, 420, 500]},
    "Ipswich":       {"state": "QLD", "track_type": "oval", "distances": [326, 431, 520, 630]},
    "Rockhampton":   {"state": "QLD", "track_type": "oval", "distances": [305, 407, 500]},
    "Toowoomba":     {"state": "QLD", "track_type": "oval", "distances": [305, 400, 500]},
    "Townsville":    {"state": "QLD", "track_type": "oval", "distances": [305, 407, 500]},
}
# fmt: on


# ---------------------------------------------------------------------------
# Grade hierarchy for greyhound racing
# ---------------------------------------------------------------------------

GRADE_HIERARCHY = {
    "Maiden": 0,
    "C0": 1,
    "C1": 2,
    "C2": 3,
    "C3": 4,
    "C4": 5,
    "C5": 6,
    "C6": 7,
    "Novice": 2,  # roughly equivalent to C1
    "Masters": 5,
    "FFA": 8,     # Free For All — open class
    "Gr5": 9,
    "Gr4": 10,
    "Gr3": 11,
    "Gr2": 12,
    "Gr1": 13,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GreyhoundMeeting(Base):
    """A greyhound race meeting at a venue on a specific date.

    Analogous to thoroughbred Meeting but with greyhound-specific fields.
    ID format: 'gh-{venue_slug}-{date}' e.g. 'gh-sandown-park-2026-03-27'
    """

    __tablename__ = "greyhound_meetings"
    __table_args__ = (Index("ix_gh_meetings_date", "date"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    venue: Mapped[str] = mapped_column(String(100))
    date: Mapped[date] = mapped_column(Date)
    state: Mapped[str] = mapped_column(String(3))  # VIC, NSW, QLD
    track_condition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    weather: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    weather_temp: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    track_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # oval, straight

    # Distance category — short (< 400m), middle (400-530m), staying (530m+)
    distance_category: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    races: Mapped[List["GreyhoundRace"]] = relationship(
        "GreyhoundRace", back_populates="meeting", cascade="all, delete-orphan"
    )

    def to_dict(self, include_races: bool = False) -> dict:
        data = {
            "id": self.id,
            "venue": self.venue,
            "date": self.date.isoformat(),
            "state": self.state,
            "track_condition": self.track_condition,
            "weather": self.weather,
            "weather_temp": self.weather_temp,
            "track_type": self.track_type,
            "distance_category": self.distance_category,
            "selected": self.selected,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_races:
            data["races"] = [r.to_dict(include_runners=True) for r in self.races]
        return data


class GreyhoundRace(Base):
    """A single greyhound race within a meeting.

    ID format: 'gh-{venue_slug}-{date}-r{N}' e.g. 'gh-sandown-park-2026-03-27-r1'
    Grades: Maiden, C0-C6, Novice, Masters, FFA, Gr1-5
    """

    __tablename__ = "greyhound_races"
    __table_args__ = (Index("ix_gh_races_meeting_id", "meeting_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), ForeignKey("greyhound_meetings.id"))
    race_number: Mapped[int] = mapped_column(Integer)
    race_name: Mapped[str] = mapped_column(String(200))
    distance: Mapped[int] = mapped_column(Integer)  # metres
    grade: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # C1-C6, Gr1-5, Maiden, FFA, etc.
    prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    field_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # typically 8

    # Track specifics for this race
    track_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # oval, straight

    # Result fields
    winning_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # e.g. "29.85"
    results_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Open, Interim, Paying, Closed
    exotic_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON dict of dividends

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    meeting: Mapped["GreyhoundMeeting"] = relationship("GreyhoundMeeting", back_populates="races")
    runners: Mapped[List["GreyhoundRunner"]] = relationship(
        "GreyhoundRunner", back_populates="race", cascade="all, delete-orphan"
    )

    def to_dict(self, include_runners: bool = False) -> dict:
        data = {
            "id": self.id,
            "meeting_id": self.meeting_id,
            "race_number": self.race_number,
            "race_name": self.race_name,
            "distance": self.distance,
            "grade": self.grade,
            "prize_money": self.prize_money,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "field_size": self.field_size,
            "track_type": self.track_type,
            "winning_time": self.winning_time,
            "results_status": self.results_status,
            "exotic_results": self.exotic_results,
        }
        if include_runners:
            data["runners"] = [r.to_dict() for r in self.runners]
        return data


class GreyhoundRunner(Base):
    """A greyhound running in a race.

    Key differences from thoroughbred Runner:
      - box_number 1-8 (reserves 9-10), NOT saddlecloth
      - dog_name instead of horse_name
      - NO jockey, NO weight, NO gear/gear_changes
      - trainer is the only human connection
      - run_time and split_time are core metrics
      - Iggy-Joey model predictions (rated price, early speed)
      - Box win percentage is a critical factor (inside boxes advantage on ovals)
      - Form history includes times, not just positions

    ID format: 'gh-{venue_slug}-{date}-r{N}-b{box}' e.g. 'gh-sandown-park-2026-03-27-r1-b1'
    """

    __tablename__ = "greyhound_runners"
    __table_args__ = (
        Index("ix_gh_runners_race_id", "race_id"),
        Index("ix_gh_runners_finish", "race_id", "finish_position"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    race_id: Mapped[str] = mapped_column(String(64), ForeignKey("greyhound_races.id"))

    # Dog identity
    dog_name: Mapped[str] = mapped_column(String(100))
    box_number: Mapped[int] = mapped_column(Integer)  # 1-8, reserves 9-10
    trainer: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    trainer_location: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_reserve: Mapped[bool] = mapped_column(Boolean, default=False)  # box 9-10

    # Dog details
    dog_colour: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)  # BK, BD, BE, WBK, etc.
    dog_sex: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # D (dog), B (bitch)
    dog_age_months: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sire: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dam: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Odds
    current_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opening_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    place_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_tab: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_sportsbet: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_pointsbet: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_betfair: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_ladbrokes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Status
    scratched: Mapped[bool] = mapped_column(Boolean, default=False)
    scratching_reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Performance metrics — key differentiator from thoroughbreds
    run_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # race time in seconds
    split_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # sectional/first split
    best_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # PB at this distance

    # Box statistics — critical for greyhounds (inside box advantage on ovals)
    box_win_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # pre-calculated per track/distance

    # Form
    form: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # last starts e.g. "12341"
    last_five: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    career_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "45: 12-8-5"
    distance_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    track_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    grade_record: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    days_since_last_run: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    career_prize_money: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Form history — JSON list of recent starts with times, positions, margins
    # Format: [{"date": "2026-03-20", "venue": "Sandown", "distance": 515,
    #           "box": 3, "position": 1, "margin": "2.5L", "time": "29.85",
    #           "grade": "C5", "weight": 32.1, "track_condition": "Good"}]
    form_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Iggy-Joey model predictions (Betfair Data Scientists greyhound model)
    iggy_rated_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    iggy_early_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Result fields
    finish_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    result_margin: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    starting_price: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    win_dividend: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    place_dividend: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=melb_now_naive, onupdate=melb_now_naive
    )

    # Relationships
    race: Mapped["GreyhoundRace"] = relationship("GreyhoundRace", back_populates="runners")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "race_id": self.race_id,
            "dog_name": self.dog_name,
            "box_number": self.box_number,
            "trainer": self.trainer,
            "trainer_location": self.trainer_location,
            "is_reserve": self.is_reserve,
            "dog_colour": self.dog_colour,
            "dog_sex": self.dog_sex,
            "dog_age_months": self.dog_age_months,
            "sire": self.sire,
            "dam": self.dam,
            "current_odds": self.current_odds,
            "opening_odds": self.opening_odds,
            "place_odds": self.place_odds,
            "odds_tab": self.odds_tab,
            "odds_sportsbet": self.odds_sportsbet,
            "odds_pointsbet": self.odds_pointsbet,
            "odds_betfair": self.odds_betfair,
            "odds_ladbrokes": self.odds_ladbrokes,
            "scratched": self.scratched,
            "scratching_reason": self.scratching_reason,
            "run_time": self.run_time,
            "split_time": self.split_time,
            "best_time": self.best_time,
            "box_win_pct": self.box_win_pct,
            "form": self.form,
            "last_five": self.last_five,
            "career_record": self.career_record,
            "distance_record": self.distance_record,
            "track_record": self.track_record,
            "grade_record": self.grade_record,
            "days_since_last_run": self.days_since_last_run,
            "career_prize_money": self.career_prize_money,
            "form_history": self.form_history,
            "iggy_rated_price": self.iggy_rated_price,
            "iggy_early_speed": self.iggy_early_speed,
            "finish_position": self.finish_position,
            "result_margin": self.result_margin,
            "starting_price": self.starting_price,
            "win_dividend": self.win_dividend,
            "place_dividend": self.place_dividend,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_meeting_id(venue: str, race_date: date) -> str:
    """Generate meeting ID: gh-{venue_slug}-{date}."""
    slug = venue.lower().replace(" ", "-").replace("'", "")
    return f"gh-{slug}-{race_date.isoformat()}"


def make_race_id(meeting_id: str, race_number: int) -> str:
    """Generate race ID: {meeting_id}-r{N}."""
    return f"{meeting_id}-r{race_number}"


def make_runner_id(race_id: str, box_number: int) -> str:
    """Generate runner ID: {race_id}-b{box}."""
    return f"{race_id}-b{box_number}"


def venue_state(venue: str) -> Optional[str]:
    """Look up state for a greyhound venue. Returns None if unknown."""
    info = GREYHOUND_VENUES.get(venue)
    return info["state"] if info else None


def venue_track_type(venue: str) -> str:
    """Look up track type for a venue. Defaults to 'oval'."""
    info = GREYHOUND_VENUES.get(venue)
    return info.get("track_type", "oval") if info else "oval"
