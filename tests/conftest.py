"""Shared test fixtures for PuntyAI."""

import asyncio
import json
from datetime import date, datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from punty.models.database import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session


@pytest.fixture
def sample_meeting_data() -> dict:
    """Sample meeting data for testing."""
    return {
        "meeting": {
            "id": "test-venue-2026-02-06",
            "venue": "Test Venue",
            "date": "2026-02-06",
            "track_condition": "Good 3",
            "weather": "Fine",
            "rail_position": "True",
            "penetrometer": 3.5,
            "weather_condition": "Fine",
            "weather_temp": 25,
            "weather_wind_speed": 10,
            "weather_wind_dir": "NW",
            "rail_bias_comment": None,
        },
        "races": [
            {
                "race_number": 1,
                "name": "Test Race 1",
                "distance": 1200,
                "class": "Maiden",
                "prize_money": 50000,
                "start_time": "2026-02-06T12:00:00",
                "status": "Open",
                "track_condition": "Good 3",
                "race_type": "Flat",
                "age_restriction": "3YO+",
                "weight_type": "Handicap",
                "field_size": 10,
                "runners": [
                    {
                        "saddlecloth": 1,
                        "barrier": 1,
                        "horse_name": "Test Horse 1",
                        "jockey": "J. Smith",
                        "trainer": "T. Jones",
                        "weight": 58.0,
                        "scratched": False,
                        "current_odds": 3.50,
                        "place_odds": 1.45,
                        "opening_odds": 4.00,
                        "form": "1-21",
                        "last_five": "x1x2x",
                        "speed_map_position": "leader",
                        "pf_speed_rank": 2,
                        "pf_map_factor": 1.15,
                    },
                    {
                        "saddlecloth": 2,
                        "barrier": 2,
                        "horse_name": "Test Horse 2",
                        "jockey": "A. Brown",
                        "trainer": "B. White",
                        "weight": 56.5,
                        "scratched": False,
                        "current_odds": 5.00,
                        "place_odds": 1.80,
                        "opening_odds": 8.00,
                        "form": "32-1",
                        "last_five": "x321x",
                        "speed_map_position": "on_pace",
                        "pf_speed_rank": 5,
                        "pf_map_factor": 0.95,
                        "market_movement": {
                            "direction": "heavy_support",
                            "summary": "Heavily backed $8.00 -> $5.00",
                            "from": 8.00,
                            "to": 5.00,
                        },
                    },
                ],
                "analysis": {
                    "pace_scenario": "genuine_pace",
                    "likely_leaders": ["Test Horse 1"],
                    "backmarkers": [],
                    "market_movers": [
                        {
                            "horse": "Test Horse 2",
                            "direction": "in",
                            "movement": "heavy_support",
                            "summary": "Heavily backed $8.00 -> $5.00",
                            "from": 8.00,
                            "to": 5.00,
                        }
                    ],
                    "pace_advantaged": [{"horse": "Test Horse 1", "map_factor": 1.15}],
                    "pace_disadvantaged": [],
                },
            },
        ],
        "summary": {
            "total_races": 1,
            "total_runners": 2,
            "scratchings": 0,
            "favorites": [{"race": 1, "horse": "Test Horse 1", "odds": 3.50}],
            "roughies": [],
        },
    }


@pytest.fixture
def sample_context_with_market_movers(sample_meeting_data) -> dict:
    """Context data that includes market movers to test variable shadowing bug."""
    return sample_meeting_data


@pytest.fixture
def mock_ai_client():
    """Mock AI client that returns predictable responses."""
    client = MagicMock()
    client.generate_with_context = AsyncMock(
        return_value="*PUNTY EARLY MAIL* - Test content generated"
    )
    return client


@pytest.fixture
def sample_early_mail_content() -> str:
    """Sample early mail content for parser testing."""
    return """*PUNTY EARLY MAIL – Test Venue (06-02-2026)*
Rightio Legends —
Test opening paragraph.

*PUNTY'S BIG 3 + MULTI*
1) *TEST HORSE A* (Race 1, No.1) — $3.50
   Confidence: high
   Why: Test reason A
2) *TEST HORSE B* (Race 2, No.3) — $5.00
   Confidence: med
   Why: Test reason B
3) *TEST HORSE C* (Race 3, No.5) — $8.00
   Confidence: low
   Why: Test reason C
Multi (all three to win): 10U × ~140.00 = ~1400U collect

*Race 1 – Test Race*
Race type: Maiden, 1200m
Map & tempo: Genuine pace
Punty read: Test analysis

*Top 3 + Roughie ($20 pool)*
1. *TEST HORSE 1* (No.1) — $3.50 / $1.45
   Bet: $10 Win, return $35.00
   Confidence: high
   Why: Best horse in race
2. *TEST HORSE 2* (No.2) — $5.00 / $1.80
   Bet: $6 Place, return $10.80
   Confidence: med
   Why: Each way value
3. *TEST HORSE 3* (No.3) — $8.00 / $2.50
   Bet: $4 Each Way, return $42.00
   Confidence: low
   Why: Can sneak a place
Roughie: *LONGSHOT* (No.8) — $21.00 / $5.00
Bet: Exotics only
Why: Only if chaos

*Degenerate Exotic of the Race*
Exacta: 1, 2 — $20
Est. return: 100% on $20
Why: Clear top two

MAIN QUADDIE (R1–R4)
Skinny ($10): 1 / 3 / 5, 6 / 2 (4 combos × $2.50 = $10) — est. return: 250%
Balanced ($50): 1, 2 / 3, 4 / 5, 6 / 2, 7 (16 combos × $3.13 = $50.08) — est. return: 313%
"""


@pytest.fixture
def sample_form_history() -> list:
    """Sample form history data."""
    return [
        {
            "date": "2026-01-15",
            "venue": "Flemington",
            "distance": 1200,
            "class": "BM78",
            "track": "Good 4",
            "pos": "2",
            "margin": "0.5L",
            "settled": "3",
            "at400": "22.5",
        },
        {
            "date": "2026-01-01",
            "venue": "Caulfield",
            "distance": 1400,
            "class": "BM70",
            "track": "Soft 5",
            "pos": "1",
            "margin": "1.2L",
            "settled": "5",
            "at400": "22.1",
        },
    ]
