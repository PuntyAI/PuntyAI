"""Tests for public website routes."""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick
from punty.models.content import Content


@pytest.fixture
async def sample_meeting(db_session: AsyncSession) -> Meeting:
    """Create a sample meeting for testing."""
    meeting = Meeting(
        id="test-venue-2026-02-17",
        venue="Test Venue",
        date=date(2026, 2, 17),
        track_condition="Good 4",
        weather="Fine",
        selected=True,
    )
    db_session.add(meeting)
    await db_session.commit()
    return meeting


@pytest.fixture
async def sample_races(db_session: AsyncSession, sample_meeting: Meeting) -> list[Race]:
    """Create sample races for testing."""
    races = []
    for i in range(1, 4):
        race = Race(
            id=f"{sample_meeting.id}-r{i}",
            meeting_id=sample_meeting.id,
            race_number=i,
            name=f"Test Race {i}",
            distance=1200,
            results_status="Paying" if i < 3 else "Open",
            start_time=datetime(2026, 2, 17, 12 + i, 0, 0),
        )
        races.append(race)
        db_session.add(race)
    await db_session.commit()
    return races


@pytest.fixture
async def sample_runners(db_session: AsyncSession, sample_races: list[Race]) -> list[Runner]:
    """Create sample runners for testing."""
    runners = []
    for race in sample_races:
        for j in range(1, 4):
            runner = Runner(
                id=f"{race.id}-{j}",
                race_id=race.id,
                saddlecloth=j,
                horse_name=f"Horse {j}",
                finish_position=j if race.results_status == "Paying" else None,
                win_dividend=3.50 if j == 1 else None,
                place_dividend=1.50 if j <= 3 else None,
            )
            runners.append(runner)
            db_session.add(runner)
    await db_session.commit()
    return runners


@pytest.fixture
async def sample_content_for_picks(db_session: AsyncSession, sample_meeting: Meeting) -> Content:
    """Create content that picks can reference."""
    content = Content(
        id="content-for-picks",
        meeting_id=sample_meeting.id,
        content_type="early_mail",
        status="sent",
        raw_content="Test content for picks",
        created_at=datetime(2026, 2, 17, 8, 0, 0),
    )
    db_session.add(content)
    await db_session.commit()
    return content


@pytest.fixture
async def sample_picks(
    db_session: AsyncSession, sample_meeting: Meeting, sample_content_for_picks: Content
) -> list[Pick]:
    """Create sample picks for testing."""
    picks = []
    content_id = sample_content_for_picks.id

    # Selection picks - some hit, some miss
    for i, (hit, pnl, stake) in enumerate([
        (True, 25.0, 10.0),   # Winner
        (True, 5.0, 6.0),     # Place hit
        (False, -10.0, 10.0), # Loss
        (True, 15.0, 5.0),    # Winner
    ], start=1):
        pick = Pick(
            id=f"sel-{i}",
            content_id=content_id,
            meeting_id=sample_meeting.id,
            race_number=(i % 2) + 1,
            pick_type="selection",
            horse_name=f"Horse {i}",
            saddlecloth=i,
            hit=hit,
            pnl=pnl,
            bet_stake=stake,
            settled=True,
            settled_at=datetime(2026, 2, 17, 14, 0, 0),
            created_at=datetime(2026, 2, 17, 10, 0, 0),
        )
        picks.append(pick)
        db_session.add(pick)

    # Exotic pick - hit
    exotic = Pick(
        id="exotic-1",
        content_id=content_id,
        meeting_id=sample_meeting.id,
        race_number=1,
        pick_type="exotic",
        exotic_type="Exacta",
        hit=True,
        pnl=80.0,
        exotic_stake=20.0,
        settled=True,
        settled_at=datetime(2026, 2, 17, 14, 0, 0),
        created_at=datetime(2026, 2, 17, 10, 0, 0),
    )
    picks.append(exotic)
    db_session.add(exotic)

    # Sequence pick - miss
    sequence = Pick(
        id="seq-1",
        content_id=content_id,
        meeting_id=sample_meeting.id,
        pick_type="sequence",
        sequence_type="quaddie",
        hit=False,
        pnl=-50.0,
        exotic_stake=50.0,
        settled=True,
        settled_at=datetime(2026, 2, 17, 14, 0, 0),
        created_at=datetime(2026, 2, 17, 10, 0, 0),
    )
    picks.append(sequence)
    db_session.add(sequence)

    await db_session.commit()
    return picks


@pytest.fixture
async def sample_content(db_session: AsyncSession, sample_meeting: Meeting) -> Content:
    """Create sample content for testing."""
    content = Content(
        id="content-1",
        meeting_id=sample_meeting.id,
        content_type="early_mail",
        status="sent",
        raw_content="Test early mail content",
        created_at=datetime(2026, 2, 17, 8, 0, 0),
    )
    db_session.add(content)
    await db_session.commit()
    return content


class TestGetWinnerStats:
    """Tests for get_winner_stats function."""

    @pytest.mark.asyncio
    async def test_today_winners_counts_all_pick_types(
        self, db_session: AsyncSession, sample_meeting, sample_picks
    ):
        """Verify today's winners includes all pick types that hit."""
        from punty.public.routes import get_winner_stats

        # Patch the async_session to use our test session
        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            # Patch melb_today to return our test date
            with patch("punty.public.routes.melb_today", return_value=date(2026, 2, 17)):
                stats = await get_winner_stats()

        # Should count: 3 winning selections + 1 winning exotic = 4 winners
        # (sequence missed, 1 selection missed)
        assert stats["today_winners"] == 4

    @pytest.mark.asyncio
    async def test_alltime_winners_counts_all_pick_types(
        self, db_session: AsyncSession, sample_meeting, sample_picks
    ):
        """Verify all-time winners includes all pick types."""
        from punty.public.routes import get_winner_stats

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            with patch("punty.public.routes.melb_today", return_value=date(2026, 2, 17)):
                stats = await get_winner_stats()

        # Same as today since all picks are from today
        assert stats["alltime_winners"] == 4

    @pytest.mark.asyncio
    async def test_collected_includes_all_bet_types(
        self, db_session: AsyncSession, sample_meeting, sample_picks
    ):
        """Verify collected amount includes selections and exotics."""
        from punty.public.routes import get_winner_stats

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            with patch("punty.public.routes.melb_today", return_value=date(2026, 2, 17)):
                stats = await get_winner_stats()

        # Selection returns: (10+25) + (6+5) + (5+15) = 35 + 11 + 20 = 66
        # Exotic returns: 20 + 80 = 100
        # Total collected: 166
        expected_collected = (10 + 25) + (6 + 5) + (5 + 15) + (20 + 80)
        assert stats["alltime_winnings"] == expected_collected

    @pytest.mark.asyncio
    async def test_no_error_when_no_picks(self, db_session: AsyncSession, sample_meeting):
        """Verify function handles empty picks gracefully."""
        from punty.public.routes import get_winner_stats

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            with patch("punty.public.routes.melb_today", return_value=date(2026, 2, 17)):
                stats = await get_winner_stats()

        assert stats["today_winners"] == 0
        assert stats["alltime_winners"] == 0
        assert stats["alltime_winnings"] == 0


class TestGetTipsCalendar:
    """Tests for get_tips_calendar function."""

    @pytest.mark.asyncio
    async def test_returns_meetings_with_sent_content(
        self, db_session: AsyncSession, sample_meeting, sample_content
    ):
        """Verify calendar returns meetings that have sent early mail."""
        from punty.public.routes import get_tips_calendar

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            result = await get_tips_calendar(page=1, per_page=30)

        assert len(result["calendar"]) == 1
        assert result["calendar"][0]["meetings"][0]["venue"] == "Test Venue"

    @pytest.mark.asyncio
    async def test_excludes_meetings_without_sent_content(
        self, db_session: AsyncSession, sample_meeting
    ):
        """Verify calendar excludes meetings without sent content."""
        from punty.public.routes import get_tips_calendar

        # Create content that is NOT sent
        draft_content = Content(
            id="draft-1",
            meeting_id=sample_meeting.id,
            content_type="early_mail",
            status="pending_review",  # Not sent!
            raw_content="Draft content",
            created_at=datetime(2026, 2, 17, 8, 0, 0),
        )
        db_session.add(draft_content)
        await db_session.commit()

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            result = await get_tips_calendar(page=1, per_page=30)

        assert len(result["calendar"]) == 0

    @pytest.mark.asyncio
    async def test_pagination_works(self, db_session: AsyncSession):
        """Verify pagination returns correct pages."""
        from punty.public.routes import get_tips_calendar

        # Create multiple meetings with sent content (dates >= TIPS_START_DATE)
        for i in range(5):
            day = 17 + i
            meeting = Meeting(
                id=f"venue-{i}-2026-02-{day}",
                venue=f"Venue {i}",
                date=date(2026, 2, day),
                selected=True,
            )
            db_session.add(meeting)

            content = Content(
                id=f"content-{i}",
                meeting_id=meeting.id,
                content_type="early_mail",
                status="sent",
                raw_content=f"Content {i}",
                created_at=datetime(2026, 2, day, 8, 0, 0),
            )
            db_session.add(content)

        await db_session.commit()

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            # Get first page with 2 per page
            result = await get_tips_calendar(page=1, per_page=2)

        assert result["total_dates"] == 5
        assert result["total_pages"] == 3
        assert result["has_next"] == True
        assert result["has_prev"] == False


class TestGetMeetingTips:
    """Tests for get_meeting_tips function."""

    @pytest.mark.asyncio
    async def test_returns_early_mail_and_wrapup(
        self, db_session: AsyncSession, sample_meeting
    ):
        """Verify function returns both early mail and wrap-up."""
        from punty.public.routes import get_meeting_tips

        # Create early mail
        early_mail = Content(
            id="em-1",
            meeting_id=sample_meeting.id,
            content_type="early_mail",
            status="sent",
            raw_content="Early mail content",
            created_at=datetime(2026, 2, 17, 8, 0, 0),
        )
        db_session.add(early_mail)

        # Create wrap-up
        wrapup = Content(
            id="wu-1",
            meeting_id=sample_meeting.id,
            content_type="meeting_wrapup",
            status="sent",
            raw_content="Wrap-up content",
            created_at=datetime(2026, 2, 17, 18, 0, 0),
        )
        db_session.add(wrapup)
        await db_session.commit()

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            result = await get_meeting_tips(sample_meeting.id)

        assert result is not None
        # Content is now HTML formatted
        assert "<p>Early mail content</p>" in result["early_mail"]["content"]
        assert "<p>Wrap-up content</p>" in result["wrapup"]["content"]
        assert result["meeting"]["venue"] == "Test Venue"

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent_meeting(self, db_session: AsyncSession):
        """Verify function returns None for non-existent meeting."""
        from punty.public.routes import get_meeting_tips

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            result = await get_meeting_tips("nonexistent-meeting-id")

        assert result is None


class TestGetNextRace:
    """Tests for get_next_race function."""

    @pytest.mark.asyncio
    async def test_returns_no_next_when_no_meetings(self, db_session: AsyncSession):
        """Verify function handles no meetings gracefully."""
        from punty.public.routes import get_next_race

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            with patch("punty.public.routes.melb_today", return_value=date(2026, 2, 17)):
                with patch("punty.public.routes.melb_now") as mock_now:
                    mock_now.return_value = datetime(2026, 2, 17, 10, 0, 0)
                    result = await get_next_race()

        assert result["has_next"] == False



class TestGetRecentWinsPublic:
    """Tests for get_recent_wins_public function."""

    @pytest.mark.asyncio
    async def test_returns_recent_wins_with_celebrations(
        self, db_session: AsyncSession, sample_meeting, sample_picks
    ):
        """Verify function returns wins with Punty celebrations."""
        from punty.public.routes import get_recent_wins_public

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            result = await get_recent_wins_public(limit=10)

        # Should have 4 wins (3 selections + 1 exotic that hit)
        assert len(result["wins"]) == 4

        # Each win should have a celebration
        for win in result["wins"]:
            assert "celebration" in win
            assert win["pnl"] > 0

    @pytest.mark.asyncio
    async def test_respects_limit(
        self, db_session: AsyncSession, sample_meeting, sample_picks
    ):
        """Verify function respects the limit parameter."""
        from punty.public.routes import get_recent_wins_public

        with patch("punty.public.routes.async_session") as mock_session:
            mock_session.return_value.__aenter__.return_value = db_session

            result = await get_recent_wins_public(limit=2)

        assert len(result["wins"]) <= 2
