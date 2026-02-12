"""Unit tests for Punting Form API scraper."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from punty.scrapers.punting_form import (
    PuntingFormScraper,
    RUN_STYLE_MAP,
    _settle_to_position,
)


# ---- Test _settle_to_position ----

class TestSettleToPosition:
    def test_leader_in_small_field(self):
        assert _settle_to_position(1, 6) == "leader"

    def test_leader_in_large_field(self):
        assert _settle_to_position(1, 14) == "leader"

    def test_on_pace_position(self):
        assert _settle_to_position(3, 10) == "on_pace"

    def test_midfield_position(self):
        assert _settle_to_position(6, 10) == "midfield"

    def test_backmarker_position(self):
        assert _settle_to_position(9, 10) == "backmarker"

    def test_no_data_settle_25(self):
        assert _settle_to_position(25, 10) is None

    def test_no_data_settle_0(self):
        assert _settle_to_position(0, 10) is None

    def test_default_field_size(self):
        # When field_size is 0, defaults to 12
        assert _settle_to_position(1, 0) == "leader"

    def test_relative_to_field(self):
        # Same settle position, different in small vs large field
        # settle=4 in field of 5 = ratio 0.8 = backmarker
        assert _settle_to_position(4, 5) == "backmarker"
        # settle=4 in field of 14 = ratio 0.29 = on_pace
        assert _settle_to_position(4, 14) == "on_pace"


# ---- Test RUN_STYLE_MAP ----

class TestRunStyleMap:
    def test_leader(self):
        assert RUN_STYLE_MAP["l"] == "leader"

    def test_on_pace(self):
        assert RUN_STYLE_MAP["op"] == "on_pace"

    def test_midfield(self):
        assert RUN_STYLE_MAP["mf"] == "midfield"

    def test_backmarker(self):
        assert RUN_STYLE_MAP["bm"] == "backmarker"

    def test_boundary_op_mf(self):
        assert RUN_STYLE_MAP["op/mf"] == "on_pace"

    def test_boundary_mf_bm(self):
        assert RUN_STYLE_MAP["mf/bm"] == "midfield"


# ---- Sample API responses ----

SAMPLE_MEETINGS = [
    {
        "track": {"name": "Pakenham", "trackId": "1134", "state": "VIC"},
        "meetingId": "237271",
        "meetingDate": "2026-02-12T00:00:00",
        "tabMeeting": True,
    },
    {
        "track": {"name": "Goulburn", "trackId": "119", "state": "NSW"},
        "meetingId": "237270",
        "meetingDate": "2026-02-12T00:00:00",
        "tabMeeting": True,
    },
]

SAMPLE_SPEEDMAPS = [
    {
        "raceNo": 1,
        "meetingId": 237271,
        "items": [
            {
                "runnerName": "Good Feel",
                "tabNo": 2,
                "speed": 1,
                "settle": 1,
                "mapA2E": 1.05,
                "jockeyA2E": 0.95,
                "pfaiScore": 93,
                "pfaiPrice": 2.5,
                "pfaiRank": 1,
                "assessedPrice": 2.5,
                "ratedSettle": 1,
                "ratedRunStyle": 4,
            },
            {
                "runnerName": "Harry Met Sally",
                "tabNo": 3,
                "speed": 2,
                "settle": 2,
                "mapA2E": 0.98,
                "jockeyA2E": -0.78,
                "pfaiScore": 87,
                "pfaiPrice": 3.38,
                "pfaiRank": 2,
                "assessedPrice": 3.38,
                "ratedSettle": 2,
                "ratedRunStyle": 6,
            },
            {
                "runnerName": "Certain Impact",
                "tabNo": 1,
                "speed": 0,
                "settle": 25,
                "mapA2E": 0,
                "jockeyA2E": 1.01,
                "pfaiScore": 50,
                "pfaiPrice": 19.72,
                "pfaiRank": 5,
                "assessedPrice": 19.61,
                "ratedSettle": 1,
                "ratedRunStyle": 0,
            },
        ],
    },
]

SAMPLE_RATINGS = [
    {
        "raceNo": 1,
        "tabNo": 2,
        "runnerName": "Good Feel",
        "runStyle": "l         ",
        "predictedSettlePostion": 1,
    },
    {
        "raceNo": 1,
        "tabNo": 3,
        "runnerName": "Harry Met Sally",
        "runStyle": "bm        ",
        "predictedSettlePostion": 2,
    },
    {
        "raceNo": 1,
        "tabNo": 1,
        "runnerName": "Certain Impact",
        "runStyle": "no data   ",
        "predictedSettlePostion": 25,
    },
]


# ---- Test PuntingFormScraper ----

class TestPuntingFormScraper:
    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key-1234")

    @pytest.mark.asyncio
    async def test_from_settings_with_key(self):
        """Test from_settings loads API key."""
        mock_db = AsyncMock()
        with patch("punty.scrapers.punting_form.PuntingFormScraper.__init__", return_value=None) as mock_init:
            with patch("punty.models.settings.get_api_key", return_value="test-key"):
                scraper = await PuntingFormScraper.from_settings(mock_db)
                assert scraper is not None

    @pytest.mark.asyncio
    async def test_from_settings_no_key(self):
        """Test from_settings raises when no API key."""
        mock_db = AsyncMock()
        with patch("punty.models.settings.get_api_key", return_value=None):
            with pytest.raises(Exception, match="API key not configured"):
                await PuntingFormScraper.from_settings(mock_db)


class TestResolveVenue:
    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key")

    @pytest.mark.asyncio
    async def test_exact_match(self):
        with patch.object(self.scraper, "get_meetings", return_value=SAMPLE_MEETINGS):
            mid = await self.scraper.resolve_meeting_id("Pakenham", date(2026, 2, 12))
            assert mid == 237271

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        with patch.object(self.scraper, "get_meetings", return_value=SAMPLE_MEETINGS):
            mid = await self.scraper.resolve_meeting_id("pakenham", date(2026, 2, 12))
            assert mid == 237271

    @pytest.mark.asyncio
    async def test_sponsor_prefix_stripped(self):
        with patch.object(self.scraper, "get_meetings", return_value=SAMPLE_MEETINGS):
            mid = await self.scraper.resolve_meeting_id("Sportsbet Pakenham", date(2026, 2, 12))
            assert mid == 237271

    @pytest.mark.asyncio
    async def test_partial_match(self):
        with patch.object(self.scraper, "get_meetings", return_value=SAMPLE_MEETINGS):
            mid = await self.scraper.resolve_meeting_id("Goulburn", date(2026, 2, 12))
            assert mid == 237270

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self):
        with patch.object(self.scraper, "get_meetings", return_value=SAMPLE_MEETINGS):
            mid = await self.scraper.resolve_meeting_id("Flemington", date(2026, 2, 12))
            assert mid is None


class TestParseSpeedMapItems:
    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key")

    def test_basic_parsing_with_ratings(self):
        """Speed map items enriched with ratings runStyle."""
        ratings_by_tab = {
            2: {"runStyle": "l         ", "predictedSettlePostion": 1},
            3: {"runStyle": "bm        ", "predictedSettlePostion": 2},
        }
        items = SAMPLE_SPEEDMAPS[0]["items"][:2]  # Good Feel + Harry Met Sally

        positions = self.scraper._parse_speed_map_items(items, ratings_by_tab, field_size=6)

        assert len(positions) == 2

        # Good Feel: runStyle=l → leader
        gf = next(p for p in positions if p["horse_name"] == "Good Feel")
        assert gf["position"] == "leader"
        assert gf["saddlecloth"] == 2
        assert gf["pf_speed_rank"] == 1
        assert gf["pf_settle"] == 1.0
        assert gf["pf_map_factor"] == 1.05
        assert gf["pf_jockey_factor"] == 0.95
        assert gf["pf_ai_score"] == 93

        # Harry Met Sally: runStyle=bm → backmarker
        hms = next(p for p in positions if p["horse_name"] == "Harry Met Sally")
        assert hms["position"] == "backmarker"
        assert hms["pf_speed_rank"] == 2

    def test_no_data_runner_excluded(self):
        """Runner with speed=0, settle=25 and no runStyle is excluded."""
        ratings_by_tab = {
            1: {"runStyle": "no data   ", "predictedSettlePostion": 25},
        }
        # Only Certain Impact (speed=0, settle=25)
        items = [SAMPLE_SPEEDMAPS[0]["items"][2]]

        positions = self.scraper._parse_speed_map_items(items, ratings_by_tab, field_size=6)
        assert len(positions) == 0  # No valid position

    def test_fallback_to_settle_when_no_ratings(self):
        """When ratings don't have runStyle, use settle position."""
        ratings_by_tab = {}  # No ratings
        items = SAMPLE_SPEEDMAPS[0]["items"][:2]

        positions = self.scraper._parse_speed_map_items(items, ratings_by_tab, field_size=6)
        assert len(positions) == 2

        # Good Feel: settle=1 in field of 6 → leader
        gf = next(p for p in positions if p["horse_name"] == "Good Feel")
        assert gf["position"] == "leader"

        # Harry Met Sally: settle=2 in field of 6 → on_pace (ratio=0.33)
        hms = next(p for p in positions if p["horse_name"] == "Harry Met Sally")
        assert hms["position"] == "on_pace"

    def test_pf_ai_fields_extracted(self):
        """PF AI score/price/rank are included in output."""
        items = SAMPLE_SPEEDMAPS[0]["items"][:1]
        positions = self.scraper._parse_speed_map_items(items, {}, field_size=6)

        assert positions[0]["pf_ai_score"] == 93
        assert positions[0]["pf_ai_price"] == 2.5
        assert positions[0]["pf_ai_rank"] == 1
        assert positions[0]["pf_assessed_price"] == 2.5

    def test_zero_map_factor_stored_as_none(self):
        """mapA2E of 0 should be stored as None (no data)."""
        items = [SAMPLE_SPEEDMAPS[0]["items"][2]]  # Certain Impact (mapA2E=0)
        # Give it a valid position via ratings so it's not excluded
        ratings_by_tab = {1: {"runStyle": "mf        "}}

        positions = self.scraper._parse_speed_map_items(items, ratings_by_tab, field_size=6)
        assert len(positions) == 1
        assert positions[0]["pf_map_factor"] is None


class TestScrapeSpeedMaps:
    """Test the async generator scrape_speed_maps method."""

    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key")

    @pytest.mark.asyncio
    async def test_successful_scrape(self):
        """Full scrape yields correct events."""
        with patch.object(self.scraper, "resolve_meeting_id", return_value=237271):
            with patch.object(self.scraper, "get_speed_maps", return_value=SAMPLE_SPEEDMAPS):
                with patch.object(self.scraper, "get_ratings", return_value=SAMPLE_RATINGS):
                    with patch.object(self.scraper, "close", return_value=None):
                        events = []
                        async for event in self.scraper.scrape_speed_maps("Pakenham", date(2026, 2, 12), 1):
                            events.append(event)

        # Should have: resolving, race 1 result, complete
        assert any("[PF] Resolving" in e.get("label", "") for e in events)
        assert any("Race 1" in e.get("label", "") for e in events)
        assert any(e.get("status") == "complete" for e in events)

        # Race 1 event should have positions
        race1 = next(e for e in events if e.get("race_number") == 1)
        assert len(race1["positions"]) == 2  # Good Feel + Harry (Certain Impact excluded)

    @pytest.mark.asyncio
    async def test_meeting_not_found(self):
        """When venue doesn't match, yields error."""
        with patch.object(self.scraper, "resolve_meeting_id", return_value=None):
            events = []
            async for event in self.scraper.scrape_speed_maps("Unknown Venue", date(2026, 2, 12), 8):
                events.append(event)

        assert any("not found" in e.get("label", "").lower() for e in events)
        assert any(e.get("status") == "error" for e in events)

    @pytest.mark.asyncio
    async def test_api_error_handled(self):
        """API errors don't crash, yield error events."""
        with patch.object(self.scraper, "resolve_meeting_id", side_effect=Exception("API timeout")):
            events = []
            async for event in self.scraper.scrape_speed_maps("Pakenham", date(2026, 2, 12), 8):
                events.append(event)

        assert any(e.get("status") == "error" for e in events)

    @pytest.mark.asyncio
    async def test_ratings_failure_non_fatal(self):
        """Ratings API failing shouldn't prevent speed maps from working."""
        with patch.object(self.scraper, "resolve_meeting_id", return_value=237271):
            with patch.object(self.scraper, "get_speed_maps", return_value=SAMPLE_SPEEDMAPS):
                with patch.object(self.scraper, "get_ratings", side_effect=Exception("ratings failed")):
                    with patch.object(self.scraper, "close", return_value=None):
                        events = []
                        async for event in self.scraper.scrape_speed_maps("Pakenham", date(2026, 2, 12), 1):
                            events.append(event)

        # Should still get race data (from settle positions, not runStyle)
        race1 = next(e for e in events if e.get("race_number") == 1)
        assert len(race1["positions"]) >= 1  # At least Good Feel
