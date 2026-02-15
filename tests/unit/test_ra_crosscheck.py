"""Tests for RA Free Fields cross-check logic."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from punty.scrapers.orchestrator import _cross_check_ra_fields, _RA_AUTH_RUNNER_FIELDS, _RA_AUTH_RACE_FIELDS


def _make_meeting(venue="Newcastle", race_date=date(2026, 2, 16)):
    m = MagicMock()
    m.venue = venue
    m.date = race_date
    return m


def _make_runner(race_id, saddlecloth, horse_name="TEST HORSE", barrier=3, weight=57.0, jockey="J Smith", scratched=False):
    r = MagicMock()
    r.race_id = race_id
    r.saddlecloth = saddlecloth
    r.horse_name = horse_name
    r.barrier = barrier
    r.weight = weight
    r.jockey = jockey
    r.scratched = scratched
    return r


def _make_race(race_id, race_number=1, distance=1200, class_="Maiden", field_size=8, start_time=None):
    r = MagicMock()
    r.race_number = race_number
    r.distance = distance
    r.class_ = class_
    r.field_size = field_size
    r.start_time = start_time
    return r


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.flush = AsyncMock()
    return db


class TestCrossCheckMismatch:
    """Test that PF↔RA mismatches are detected and RA wins."""

    @pytest.mark.asyncio
    async def test_barrier_mismatch_ra_wins(self, mock_db):
        """RA barrier value overwrites PF barrier."""
        meeting = _make_meeting()
        meeting_id = "newcastle-2026-02-16"
        race_id = f"{meeting_id}-r1"

        db_runner = _make_runner(race_id, 1, "FAST HORSE", barrier=3, weight=57.0, jockey="J Smith")
        db_race = _make_race(race_id)

        ra_data = {
            "meeting": {},
            "races": [{"id": race_id, "race_number": 1, "distance": 1200, "class_": "Maiden", "field_size": 8, "start_time": None}],
            "runners": [{"race_id": race_id, "saddlecloth": 1, "barrier": 7, "weight": 57.0, "jockey": "J Smith", "scratched": False}],
        }

        # Mock db.get for race
        async def mock_get(model, id):
            if id == race_id:
                return db_race
            return None
        mock_db.get = AsyncMock(side_effect=mock_get)

        # Mock db.execute for runner query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_runner
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=ra_data):
            result = await _cross_check_ra_fields(mock_db, meeting, meeting_id)

        assert result["mismatches"] == 1
        assert db_runner.barrier == 7  # RA won

    @pytest.mark.asyncio
    async def test_weight_and_jockey_mismatch(self, mock_db):
        """Multiple field mismatches detected and corrected."""
        meeting = _make_meeting()
        meeting_id = "newcastle-2026-02-16"
        race_id = f"{meeting_id}-r1"

        db_runner = _make_runner(race_id, 1, "FAST HORSE", barrier=3, weight=57.0, jockey="J Smith")
        db_race = _make_race(race_id)

        ra_data = {
            "meeting": {},
            "races": [{"id": race_id, "race_number": 1, "distance": 1200, "class_": "Maiden", "field_size": 8, "start_time": None}],
            "runners": [{"race_id": race_id, "saddlecloth": 1, "barrier": 3, "weight": 58.5, "jockey": "K Lee(late alt)", "scratched": False}],
        }

        async def mock_get(model, id):
            if id == race_id:
                return db_race
            return None
        mock_db.get = AsyncMock(side_effect=mock_get)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_runner
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=ra_data):
            result = await _cross_check_ra_fields(mock_db, meeting, meeting_id)

        assert result["mismatches"] == 2
        assert db_runner.weight == 58.5
        assert db_runner.jockey == "K Lee(late alt)"

    @pytest.mark.asyncio
    async def test_race_distance_mismatch(self, mock_db):
        """RA race distance overwrites PF distance."""
        meeting = _make_meeting()
        meeting_id = "newcastle-2026-02-16"
        race_id = f"{meeting_id}-r1"

        db_race = _make_race(race_id, distance=1200)

        ra_data = {
            "meeting": {},
            "races": [{"id": race_id, "race_number": 1, "distance": 1350, "class_": "Maiden", "field_size": 8, "start_time": None}],
            "runners": [],
        }

        async def mock_get(model, id):
            if id == race_id:
                return db_race
            return None
        mock_db.get = AsyncMock(side_effect=mock_get)
        mock_db.execute = AsyncMock()

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=ra_data):
            result = await _cross_check_ra_fields(mock_db, meeting, meeting_id)

        assert result["mismatches"] == 1
        assert db_race.distance == 1350


class TestScratchingAuthority:
    """Test that RA can scratch but never un-scratch."""

    @pytest.mark.asyncio
    async def test_ra_scratches_runner(self, mock_db):
        """RA scratched=True overrides PF scratched=False."""
        meeting = _make_meeting()
        meeting_id = "newcastle-2026-02-16"
        race_id = f"{meeting_id}-r1"

        db_runner = _make_runner(race_id, 1, "FAST HORSE", scratched=False)
        db_race = _make_race(race_id)

        ra_data = {
            "meeting": {},
            "races": [{"id": race_id, "race_number": 1, "distance": 1200, "class_": "Maiden", "field_size": 8, "start_time": None}],
            "runners": [{"race_id": race_id, "saddlecloth": 1, "barrier": 3, "weight": 57.0, "jockey": "J Smith", "scratched": True}],
        }

        async def mock_get(model, id):
            if id == race_id:
                return db_race
            return None
        mock_db.get = AsyncMock(side_effect=mock_get)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_runner
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=ra_data):
            result = await _cross_check_ra_fields(mock_db, meeting, meeting_id)

        assert result["mismatches"] == 1
        assert db_runner.scratched is True

    @pytest.mark.asyncio
    async def test_ra_cannot_unscratch(self, mock_db):
        """PF scratched=True stays even if RA says scratched=False."""
        meeting = _make_meeting()
        meeting_id = "newcastle-2026-02-16"
        race_id = f"{meeting_id}-r1"

        db_runner = _make_runner(race_id, 1, "FAST HORSE", scratched=True)
        db_race = _make_race(race_id)

        ra_data = {
            "meeting": {},
            "races": [{"id": race_id, "race_number": 1, "distance": 1200, "class_": "Maiden", "field_size": 8, "start_time": None}],
            "runners": [{"race_id": race_id, "saddlecloth": 1, "barrier": 3, "weight": 57.0, "jockey": "J Smith", "scratched": False}],
        }

        async def mock_get(model, id):
            if id == race_id:
                return db_race
            return None
        mock_db.get = AsyncMock(side_effect=mock_get)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_runner
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=ra_data):
            result = await _cross_check_ra_fields(mock_db, meeting, meeting_id)

        # No mismatch logged because scratched=True stays
        assert db_runner.scratched is True


class TestGracefulHandling:
    """Test error and no-data scenarios."""

    @pytest.mark.asyncio
    async def test_ra_returns_no_data(self, mock_db):
        """RA returns None — function returns no_data status."""
        meeting = _make_meeting()

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=None):
            result = await _cross_check_ra_fields(mock_db, meeting, "test-id")

        assert result["status"] == "no_data"
        assert result["mismatches"] == 0

    @pytest.mark.asyncio
    async def test_ra_throws_exception(self, mock_db):
        """RA scraper raises — function catches and returns error status."""
        meeting = _make_meeting()

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, side_effect=Exception("Connection timeout")):
            result = await _cross_check_ra_fields(mock_db, meeting, "test-id")

        assert result["status"] == "error"
        assert result["mismatches"] == 0

    @pytest.mark.asyncio
    async def test_consistent_data_zero_mismatches(self, mock_db):
        """PF and RA agree on all fields — zero mismatches."""
        meeting = _make_meeting()
        meeting_id = "newcastle-2026-02-16"
        race_id = f"{meeting_id}-r1"

        db_runner = _make_runner(race_id, 1, "FAST HORSE", barrier=3, weight=57.0, jockey="J Smith", scratched=False)
        db_race = _make_race(race_id, distance=1200, class_="Maiden", field_size=8)

        ra_data = {
            "meeting": {},
            "races": [{"id": race_id, "race_number": 1, "distance": 1200, "class_": "Maiden", "field_size": 8, "start_time": None}],
            "runners": [{"race_id": race_id, "saddlecloth": 1, "barrier": 3, "weight": 57.0, "jockey": "J Smith", "scratched": False}],
        }

        async def mock_get(model, id):
            if id == race_id:
                return db_race
            return None
        mock_db.get = AsyncMock(side_effect=mock_get)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = db_runner
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("punty.scrapers.ra_fields.scrape_ra_fields", new_callable=AsyncMock, return_value=ra_data):
            result = await _cross_check_ra_fields(mock_db, meeting, meeting_id)

        assert result["status"] == "ok"
        assert result["mismatches"] == 0
