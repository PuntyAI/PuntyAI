"""Unit tests for Punting Form API scraper."""

import json
import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from punty.scrapers.punting_form import (
    PuntingFormScraper,
    RUN_STYLE_MAP,
    _settle_to_position,
    _record_to_json,
    _a2e_to_json,
    _parse_pf_start_time,
    _parse_last10,
    _pf_runner_to_dict,
    _pf_form_entry_to_history,
    _CONDITION_LABELS,
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
        assert any("Resolving" in e.get("label", "") for e in events)
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


# ---- Test _record_to_json ----

class TestRecordToJson:
    def test_valid_record(self):
        record = {"starts": 10, "firsts": 3, "seconds": 2, "thirds": 1}
        result = json.loads(_record_to_json(record))
        assert result == {"starts": 10, "wins": 3, "seconds": 2, "thirds": 1}

    def test_none_record(self):
        assert _record_to_json(None) is None

    def test_zero_starts(self):
        assert _record_to_json({"starts": 0, "firsts": 0, "seconds": 0, "thirds": 0}) is None

    def test_empty_dict(self):
        assert _record_to_json({}) is None


# ---- Test _a2e_to_json ----

class TestA2eToJson:
    def test_career_only(self):
        career = {"a2E": 1.15, "poT": 12.5, "strikeRate": 22.0, "wins": 50, "runners": 227}
        result = json.loads(_a2e_to_json(career, None))
        assert "career" in result
        assert result["career"]["a2e"] == 1.15
        assert result["career"]["strike_rate"] == 22.0

    def test_career_and_last100(self):
        career = {"a2E": 1.1, "poT": 10.0, "strikeRate": 20.0, "wins": 50, "runners": 250}
        last100 = {"a2E": 1.2, "poT": 15.0, "strikeRate": 25.0, "wins": 25, "runners": 100}
        result = json.loads(_a2e_to_json(career, last100))
        assert "career" in result
        assert "last100" in result
        assert result["last100"]["a2e"] == 1.2

    def test_with_combo_stats(self):
        career = {"a2E": 1.0, "poT": 10.0, "strikeRate": 15.0, "wins": 10, "runners": 67}
        combo_career = {"a2E": 1.5, "poT": 20.0, "strikeRate": 30.0, "wins": 3, "runners": 10}
        result = json.loads(_a2e_to_json(career, None, combo_career, None))
        assert "combo_career" in result
        assert result["combo_career"]["a2e"] == 1.5

    def test_empty_returns_none(self):
        assert _a2e_to_json(None, None) is None

    def test_zero_runners_excluded(self):
        assert _a2e_to_json({"runners": 0}, None) is None


# ---- Test _parse_pf_start_time ----

class TestParsePfStartTime:
    def test_us_format_converts_utc_to_melbourne(self):
        # 7:45 AM UTC → 18:45 AEDT (UTC+11) in February
        result = _parse_pf_start_time("2/12/2026 7:45:00 AM")
        assert isinstance(result, datetime)
        assert result.month == 2
        assert result.day == 12
        assert result.hour == 18
        assert result.minute == 45
        assert result.tzinfo is None  # stored as naive

    def test_pm_time_converts_utc_to_melbourne(self):
        # 2:30 PM UTC = 14:30 UTC → 01:30 next day AEDT
        result = _parse_pf_start_time("2/12/2026 2:30:00 PM")
        assert result.day == 13  # rolls to next day
        assert result.hour == 1
        assert result.minute == 30

    def test_typical_race_time(self):
        # Typical AU race: 2:50 AM UTC → 13:50 AEDT (1:50 PM Melbourne)
        result = _parse_pf_start_time("2/12/2026 2:50:00 AM")
        assert result.hour == 13
        assert result.minute == 50

    def test_none_input(self):
        assert _parse_pf_start_time(None) is None

    def test_empty_string(self):
        assert _parse_pf_start_time("") is None

    def test_invalid_format(self):
        assert _parse_pf_start_time("not-a-date") is None

    def test_iso_fallback_converts_utc_to_melbourne(self):
        # ISO without timezone assumed UTC: 7:45 UTC → 18:45 AEDT
        result = _parse_pf_start_time("2026-02-12T07:45:00")
        assert result is not None
        assert result.hour == 18


# ---- Test _parse_last10 ----

class TestParseLast10:
    def test_simple_form(self):
        form, last_five = _parse_last10("12345")
        assert form == "12345"
        assert last_five == "12345"

    def test_with_spell_breaks(self):
        form, last_five = _parse_last10("40x42")
        assert form == "4042"
        assert last_five == "4042"

    def test_with_spaces(self):
        form, last_five = _parse_last10("  12x34  ")
        assert form == "1234"
        assert last_five == "1234"

    def test_long_form(self):
        form, last_five = _parse_last10("1234567890")
        assert form == "1234567890"
        assert last_five == "12345"

    def test_none_input(self):
        form, last_five = _parse_last10(None)
        assert form == ""
        assert last_five == ""

    def test_empty_string(self):
        form, last_five = _parse_last10("")
        assert form == ""
        assert last_five == ""


# ---- Test _pf_runner_to_dict ----

SAMPLE_PF_RUNNER = {
    "name": "Good Feel",
    "tabNo": 2,
    "barrier": 5,
    "weightTotal": 58.5,
    "jockey": {"fullName": "Craig Williams"},
    "trainer": {"fullName": "Ciaron Maher", "location": "Pakenham"},
    "age": 4,
    "sex": "Gelding",
    "colour": "Bay",
    "sire": "Snitzel",
    "dam": "Lady Feel",
    "sireofDam": "Redoute's Choice",
    "last10": "12x31",
    "careerStarts": 15,
    "careerWins": 4,
    "careerSeconds": 3,
    "careerThirds": 2,
    "prizeMoney": 250000,
    "handicap": 62,
    "gearChanges": "Blinkers OFF",
    "trackRecord": {"starts": 5, "firsts": 2, "seconds": 1, "thirds": 0},
    "distanceRecord": {"starts": 8, "firsts": 3, "seconds": 2, "thirds": 1},
    "trackDistRecord": {"starts": 3, "firsts": 1, "seconds": 1, "thirds": 0},
    "firstUpRecord": {"starts": 4, "firsts": 1, "seconds": 1, "thirds": 0},
    "secondUpRecord": {"starts": 3, "firsts": 1, "seconds": 0, "thirds": 1},
    "goodRecord": {"starts": 10, "firsts": 3, "seconds": 2, "thirds": 1},
    "softRecord": {"starts": 3, "firsts": 1, "seconds": 1, "thirds": 0},
    "heavyRecord": {"starts": 2, "firsts": 0, "seconds": 0, "thirds": 1},
    "jockeyA2E_Career": {"a2E": 1.15, "poT": 12.5, "strikeRate": 22.0, "wins": 50, "runners": 227},
    "jockeyA2E_Last100": {"a2E": 1.2, "poT": 15.0, "strikeRate": 25.0, "wins": 25, "runners": 100},
    "trainerA2E_Career": {"a2E": 1.1, "poT": 10.0, "strikeRate": 18.0, "wins": 200, "runners": 1111},
    "trainerA2E_Last100": {"a2E": 1.3, "poT": 14.0, "strikeRate": 22.0, "wins": 22, "runners": 100},
    "trainerJockeyA2E_Career": {"a2E": 1.5, "poT": 20.0, "strikeRate": 30.0, "wins": 3, "runners": 10},
    "trainerJockeyA2E_Last100": None,
}


class TestPfRunnerToDict:
    def test_basic_fields(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "pak-2026-02-12-r1", "pak-2026-02-12")
        assert result["horse_name"] == "Good Feel"
        assert result["saddlecloth"] == 2
        assert result["barrier"] == 5
        assert result["weight"] == 58.5
        assert result["jockey"] == "Craig Williams"
        assert result["trainer"] == "Ciaron Maher"
        assert result["trainer_location"] == "Pakenham"

    def test_pedigree(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        assert result["sire"] == "Snitzel"
        assert result["dam"] == "Lady Feel"
        assert result["dam_sire"] == "Redoute's Choice"

    def test_career_record(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        assert result["career_record"] == "15: 4-3-2"

    def test_form_parsing(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        assert result["form"] == "1231"
        assert result["last_five"] == "1231"

    def test_stats_as_json(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        track = json.loads(result["track_stats"])
        assert track["starts"] == 5
        assert track["wins"] == 2

    def test_a2e_jockey_stats(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        jockey = json.loads(result["jockey_stats"])
        assert "career" in jockey
        assert "last100" in jockey
        assert "combo_career" in jockey
        assert jockey["career"]["a2e"] == 1.15

    def test_a2e_trainer_stats(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        trainer = json.loads(result["trainer_stats"])
        assert trainer["career"]["a2e"] == 1.1
        assert trainer["last100"]["a2e"] == 1.3

    def test_id_generation(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "pak-2026-02-12-r1", "pak-2026-02-12")
        assert result["id"] == "pak-2026-02-12-r1-2-good-feel"

    def test_gear_changes(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        assert result["gear_changes"] == "Blinkers OFF"

    def test_odds_default_none(self):
        result = _pf_runner_to_dict(SAMPLE_PF_RUNNER, "r1", "m1")
        assert result["current_odds"] is None
        assert result["opening_odds"] is None
        assert result["place_odds"] is None


# ---- Test _pf_form_entry_to_history ----

SAMPLE_PF_FORM_ENTRY = {
    "meetingDate": "2026-01-15T00:00:00",
    "track": {"name": "Flemington"},
    "distance": 1600,
    "raceClass": "BM78",
    "prizeMoney": 100000,
    "trackCondition": "Good 4",
    "starters": 12,
    "position": 1,
    "margin": 0.5,
    "weight": 58.0,
    "weightTotal": 58.5,
    "jockey": {"fullName": "Damien Oliver"},
    "barrier": 3,
    "priceSP": 4.50,
    "priceTAB": 4.20,
    "priceBF": 5.10,
    "flucs": "opening,5.00;mid,4.50;starting,4.20;",
    "inRun": "finish,1;settling_down,3;m800,2;m400,1;",
    "officialRaceTime": "1:34.56",
    "stewardsReport": "Held up on turn",
    "top4Finishers": [
        {"runnerName": "Good Feel", "position": 1},
        {"runnerName": "Second Runner", "position": 2},
        {"runnerName": "Third Runner", "position": 3},
    ],
    "isBarrierTrial": False,
}


class TestPfFormEntryToHistory:
    def test_basic_fields(self):
        result = _pf_form_entry_to_history(SAMPLE_PF_FORM_ENTRY)
        assert result["date"] == "2026-01-15"
        assert result["venue"] == "Flemington"
        assert result["distance"] == 1600
        assert result["class"] == "BM78"
        assert result["position"] == 1
        assert result["margin"] == 0.5

    def test_flucs_parsing(self):
        result = _pf_form_entry_to_history(SAMPLE_PF_FORM_ENTRY)
        assert result["flucs"]["opening"] == 5.00
        assert result["flucs"]["mid"] == 4.50
        assert result["flucs"]["starting"] == 4.20

    def test_in_run_parsing(self):
        result = _pf_form_entry_to_history(SAMPLE_PF_FORM_ENTRY)
        assert result["at800"] == "2"
        assert result["at400"] == "1"
        assert result["settled"] == "3"

    def test_top4(self):
        result = _pf_form_entry_to_history(SAMPLE_PF_FORM_ENTRY)
        assert len(result["top4"]) == 3
        assert result["top4"][0]["name"] == "Good Feel"

    def test_prices(self):
        result = _pf_form_entry_to_history(SAMPLE_PF_FORM_ENTRY)
        assert result["sp"] == 4.50
        assert result["sp_tab"] == 4.20
        assert result["sp_bf"] == 5.10

    def test_empty_flucs(self):
        entry = {**SAMPLE_PF_FORM_ENTRY, "flucs": ""}
        result = _pf_form_entry_to_history(entry)
        assert result["flucs"] is None

    def test_empty_in_run(self):
        entry = {**SAMPLE_PF_FORM_ENTRY, "inRun": ""}
        result = _pf_form_entry_to_history(entry)
        assert result["at800"] is None
        assert result["at400"] is None

    def test_barrier_trial_flag(self):
        entry = {**SAMPLE_PF_FORM_ENTRY, "isBarrierTrial": True}
        result = _pf_form_entry_to_history(entry)
        assert result["is_trial"] is True


# ---- Test conditions parsing ----

class TestConditionsParsing:
    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key")

    def test_parse_basic_condition(self):
        cond = {
            "track": "Pakenham",
            "trackCondition": "Good 4",
            "trackConditionNumber": 4,
            "rail": "Out 3m",
            "weather": "Fine",
            "penetrometer": "4.20",
            "wind": "15",
            "windDirection": "NW",
            "rainfall": "2.5",
            "irrigation": "6mm last 24hrs",
            "comment": "",
            "abandonded": False,
        }
        result = self.scraper._parse_condition(cond)
        assert result["condition"] == "Good 4"
        assert result["rail"] == "Out 3m"
        assert result["weather"] == "Fine"
        assert result["penetrometer"] == 4.2
        assert result["wind_speed"] == 15
        assert result["wind_direction"] == "NW"
        assert result["rainfall"] == "2.5"
        assert result["irrigation"] is True
        assert result["going_stick"] is None

    def test_nil_rainfall_is_none(self):
        cond = {"track": "Test", "trackCondition": "Good", "rainfall": "Nil", "irrigation": "Nil"}
        result = self.scraper._parse_condition(cond)
        assert result["rainfall"] is None
        assert result["irrigation"] is False

    def test_string_penetrometer(self):
        cond = {"track": "Test", "trackCondition": "Good", "penetrometer": "6.40"}
        result = self.scraper._parse_condition(cond)
        assert result["penetrometer"] == 6.4

    def test_going_stick_from_comment(self):
        cond = {
            "track": "Flemington",
            "trackCondition": "Good 4",
            "trackConditionNumber": 4,
            "comment": "Track is in good order. Going Stick: 11.8",
        }
        result = self.scraper._parse_condition(cond)
        assert result["going_stick"] == 11.8

    def test_condition_number_preferred_over_bare_label(self):
        """When API returns bare 'Good' but trackConditionNumber=4, use 'Good 4'."""
        cond = {
            "track": "Test",
            "trackCondition": "Good",
            "trackConditionNumber": 4,
        }
        result = self.scraper._parse_condition(cond)
        assert result["condition"] == "Good 4"

    def test_condition_number_preferred_soft(self):
        cond = {
            "track": "Test",
            "trackCondition": "Soft",
            "trackConditionNumber": 6,
        }
        result = self.scraper._parse_condition(cond)
        assert result["condition"] == "Soft 6"

    def test_condition_number_as_string(self):
        """API often returns trackConditionNumber as string, not int."""
        cond = {
            "track": "Test",
            "trackCondition": "Good",
            "trackConditionNumber": "4",
        }
        result = self.scraper._parse_condition(cond)
        assert result["condition"] == "Good 4"

    def test_condition_label_fallback(self):
        cond = {
            "track": "Test",
            "trackCondition": "",
            "trackConditionNumber": 8,
        }
        result = self.scraper._parse_condition(cond)
        assert result["condition"] == "Heavy 8"

    def test_abandoned(self):
        cond = {
            "track": "Test",
            "trackCondition": "Heavy 10",
            "trackConditionNumber": 10,
            "abandonded": True,  # PF typo
        }
        result = self.scraper._parse_condition(cond)
        assert result["abandoned"] is True

    @pytest.mark.asyncio
    async def test_get_conditions_for_venue_match(self):
        conditions = [
            {"track": "Pakenham", "trackCondition": "Good 4", "trackConditionNumber": 4},
            {"track": "Flemington", "trackCondition": "Soft 5", "trackConditionNumber": 5},
        ]
        self.scraper._conditions_cache = conditions
        result = await self.scraper.get_conditions_for_venue("Pakenham")
        assert result is not None
        assert result["condition"] == "Good 4"

    @pytest.mark.asyncio
    async def test_get_conditions_for_venue_no_match(self):
        conditions = [
            {"track": "Flemington", "trackCondition": "Soft 5", "trackConditionNumber": 5},
        ]
        self.scraper._conditions_cache = conditions
        result = await self.scraper.get_conditions_for_venue("Sandown")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_conditions_sponsor_prefix(self):
        conditions = [
            {"track": "Pakenham", "trackCondition": "Good 4", "trackConditionNumber": 4},
        ]
        self.scraper._conditions_cache = conditions
        result = await self.scraper.get_conditions_for_venue("Sportsbet Pakenham")
        assert result is not None
        assert result["condition"] == "Good 4"


# ---- Test scratchings ----

class TestScratchings:
    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key")

    @pytest.mark.asyncio
    async def test_filter_by_meeting_id(self):
        self.scraper._scratchings_cache = [
            {"meetingId": 237271, "raceNo": 1, "tabNo": 3, "deduction": 2.5, "timeStamp": "2026-02-12T10:00:00"},
            {"meetingId": 237271, "raceNo": 3, "tabNo": 7, "deduction": 1.0, "timeStamp": "2026-02-12T10:05:00"},
            {"meetingId": 237270, "raceNo": 1, "tabNo": 1, "deduction": 3.0, "timeStamp": "2026-02-12T09:00:00"},
        ]
        result = await self.scraper.get_scratchings_for_meeting(237271)
        assert len(result) == 2
        assert result[0]["race_number"] == 1
        assert result[0]["tab_no"] == 3
        assert result[0]["deduction"] == 2.5

    @pytest.mark.asyncio
    async def test_no_scratchings(self):
        self.scraper._scratchings_cache = []
        result = await self.scraper.get_scratchings_for_meeting(999999)
        assert result == []


# ---- Test scrape_meeting_data ----

SAMPLE_PF_FIELDS = {
    "track": {"name": "Pakenham", "abbrev": "PKM"},
    "railPosition": "True",
    "meetingId": 237271,
    "races": [
        {
            "number": 1,
            "name": "Maiden Plate",
            "distance": 1200,
            "raceClass": "MDN",
            "prizeMoney": "40000",
            "startTimeUTC": "2/12/2026 7:45:00 AM",
            "ageRestrictions": "3yo+",
            "weightType": "Set Weights",
            "runners": [SAMPLE_PF_RUNNER],
        }
    ],
}


class TestScrapeMeetingData:
    def setup_method(self):
        self.scraper = PuntingFormScraper(api_key="test-key")

    @pytest.mark.asyncio
    async def test_successful_meeting_scrape(self):
        with patch.object(self.scraper, "resolve_meeting_id", return_value=237271):
            with patch.object(self.scraper, "get_fields", return_value=SAMPLE_PF_FIELDS):
                with patch.object(self.scraper, "get_form", side_effect=Exception("no form")):
                    with patch.object(self.scraper, "get_scratchings_for_meeting", return_value=[]):
                        result = await self.scraper.scrape_meeting_data("Pakenham", date(2026, 2, 12))

        assert "meeting" in result
        assert "races" in result
        assert "runners" in result
        assert len(result["races"]) == 1
        assert len(result["runners"]) == 1

        race = result["races"][0]
        assert race["name"] == "Maiden Plate"
        assert race["distance"] == 1200
        assert race["class_"] == "MDN"
        assert race["prize_money"] == 40000
        assert race["age_restriction"] == "3yo+"

        runner = result["runners"][0]
        assert runner["horse_name"] == "Good Feel"
        assert runner["dam_sire"] == "Redoute's Choice"
        assert runner["trainer_location"] == "Pakenham"

    @pytest.mark.asyncio
    async def test_meeting_not_found(self):
        with patch.object(self.scraper, "resolve_meeting_id", return_value=None):
            result = await self.scraper.scrape_meeting_data("Unknown", date(2026, 2, 12))
        assert result["races"] == []
        assert result["runners"] == []

    @pytest.mark.asyncio
    async def test_scratchings_applied(self):
        scratchings = [{"race_number": 1, "tab_no": 2, "deduction": 2.0, "timestamp": None}]
        with patch.object(self.scraper, "resolve_meeting_id", return_value=237271):
            with patch.object(self.scraper, "get_fields", return_value=SAMPLE_PF_FIELDS):
                with patch.object(self.scraper, "get_form", side_effect=Exception("no form")):
                    with patch.object(self.scraper, "get_scratchings_for_meeting", return_value=scratchings):
                        result = await self.scraper.scrape_meeting_data("Pakenham", date(2026, 2, 12))

        runner = result["runners"][0]
        assert runner["scratched"] is True


# ---- Test _CONDITION_LABELS ----

class TestConditionLabels:
    def test_all_labels_present(self):
        assert len(_CONDITION_LABELS) == 10
        assert _CONDITION_LABELS[1] == "Firm 1"
        assert _CONDITION_LABELS[4] == "Good 4"
        assert _CONDITION_LABELS[7] == "Soft 7"
        assert _CONDITION_LABELS[10] == "Heavy 10"


# ---- Test Strike Rates ----

SAMPLE_STRIKE_RATE_PAYLOAD = [
    {
        "startDate": "2026-02-13T00:00:00",
        "entityId": 7232,
        "entityName": "Kelly Myers",
        "careerWins": 279,
        "careerStarts": 8063,
        "careerSeconds": 834,
        "careerThirds": 768,
        "last100Wins": 11,
        "last100Starts": 100,
        "last100Seconds": 12,
        "last100Thirds": 13,
        "careerExpectedWins": 303.4,
        "careerPL": -8349.08,
        "careerTurnvoer": 44614.5,
        "last100ExpectedWins": 13.39,
        "last100PL": -306.88,
        "last100Turnvoer": 2125.63,
    },
    {
        "startDate": "2026-02-13T00:00:00",
        "entityId": 5001,
        "entityName": "C J Waller",
        "careerWins": 4200,
        "careerStarts": 20000,
        "careerSeconds": 3000,
        "careerThirds": 2500,
        "last100Wins": 28,
        "last100Starts": 100,
        "last100Seconds": 18,
        "last100Thirds": 12,
        "careerExpectedWins": 3800.0,
        "careerPL": 5000.0,
        "careerTurnvoer": 100000.0,
        "last100ExpectedWins": 22.0,
        "last100PL": 800.0,
        "last100Turnvoer": 5000.0,
    },
]


class TestStrikeRates:
    def setup_method(self):
        from punty.scrapers.punting_form import _strike_rate_cache
        _strike_rate_cache.clear()
        self.scraper = PuntingFormScraper(api_key="test-key")

    @pytest.mark.asyncio
    async def test_get_strike_rates_parses_correctly(self):
        with patch.object(self.scraper, "_api_get", return_value=SAMPLE_STRIKE_RATE_PAYLOAD):
            result = await self.scraper.get_strike_rates(entity_type=1)

        assert len(result) == 2
        assert "kelly myers" in result
        assert "c j waller" in result

        km = result["kelly myers"]
        assert km["name"] == "Kelly Myers"
        assert km["career_starts"] == 8063
        assert km["career_wins"] == 279
        assert km["career_sr"] == round(279 / 8063 * 100, 1)
        assert km["last100_wins"] == 11
        assert km["last100_sr"] == 11.0
        assert km["career_a2e"] == round(279 / 303.4, 2)
        assert km["last100_a2e"] == round(11 / 13.39, 2)
        assert km["career_pl"] == -8349.08
        assert km["last100_pl"] == -306.88

    @pytest.mark.asyncio
    async def test_strike_rate_cache(self):
        with patch.object(self.scraper, "_api_get", return_value=SAMPLE_STRIKE_RATE_PAYLOAD) as mock_get:
            result1 = await self.scraper.get_strike_rates(entity_type=1)
            result2 = await self.scraper.get_strike_rates(entity_type=1)

        mock_get.assert_called_once()
        assert result1 is result2

    @pytest.mark.asyncio
    async def test_get_all_strike_rates(self):
        async def mock_api(path, params):
            return SAMPLE_STRIKE_RATE_PAYLOAD

        with patch.object(self.scraper, "_api_get", side_effect=mock_api):
            result = await self.scraper.get_all_strike_rates()

        assert "jockeys" in result
        assert "trainers" in result
        assert len(result["jockeys"]) == 2
        assert len(result["trainers"]) == 2

    @pytest.mark.asyncio
    async def test_hot_jockey_detection(self):
        """Jockey with last100_a2e >= 1.15 should be detectable."""
        with patch.object(self.scraper, "_api_get", return_value=SAMPLE_STRIKE_RATE_PAYLOAD):
            result = await self.scraper.get_strike_rates(entity_type=1)

        waller = result["c j waller"]
        assert waller["last100_a2e"] == round(28 / 22.0, 2)
        assert waller["last100_a2e"] >= 1.15  # HOT

    @pytest.mark.asyncio
    async def test_cold_jockey_detection(self):
        """Jockey with last100_a2e <= 0.80 should be detectable."""
        with patch.object(self.scraper, "_api_get", return_value=SAMPLE_STRIKE_RATE_PAYLOAD):
            result = await self.scraper.get_strike_rates(entity_type=1)

        km = result["kelly myers"]
        assert km["last100_a2e"] == round(11 / 13.39, 2)
        assert km["last100_a2e"] <= 0.85  # COLD or borderline

    @pytest.mark.asyncio
    async def test_empty_entity_name_skipped(self):
        data = [{"entityName": "", "careerWins": 1, "careerStarts": 10}]
        with patch.object(self.scraper, "_api_get", return_value=data):
            result = await self.scraper.get_strike_rates(entity_type=1)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_zero_starts_no_division_error(self):
        data = [{
            "entityName": "Test Jockey",
            "careerWins": 0, "careerStarts": 0,
            "careerSeconds": 0, "careerThirds": 0,
            "last100Wins": 0, "last100Starts": 0,
            "last100Seconds": 0, "last100Thirds": 0,
            "careerExpectedWins": 0, "careerPL": 0, "careerTurnvoer": 0,
            "last100ExpectedWins": 0, "last100PL": 0, "last100Turnvoer": 0,
        }]
        with patch.object(self.scraper, "_api_get", return_value=data):
            result = await self.scraper.get_strike_rates(entity_type=1)
        tj = result["test jockey"]
        assert tj["career_sr"] == 0
        assert tj["career_a2e"] == 0
        assert tj["last100_sr"] == 0
        assert tj["last100_a2e"] == 0
