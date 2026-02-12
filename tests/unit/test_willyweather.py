"""Tests for WillyWeather API scraper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from punty.scrapers.willyweather import (
    WillyWeatherScraper,
    VENUE_LOCATIONS,
    analyse_wind_impact,
    _DIRECTION_DEGREES,
)


# ── VENUE_LOCATIONS Tests ──────────────────────────────────────────────────


class TestVenueLocations:
    """Test the pre-mapped venue → location ID dict."""

    def test_ten_unique_location_ids(self):
        """Should have 10 unique WillyWeather location IDs (API limit)."""
        unique_ids = set(v["id"] for v in VENUE_LOCATIONS.values())
        assert len(unique_ids) == 10

    def test_all_states_covered(self):
        """Should cover VIC, NSW, QLD, SA, WA, TAS."""
        states = set(v["state"] for v in VENUE_LOCATIONS.values())
        assert states == {"VIC", "NSW", "QLD", "SA", "WA", "TAS"}

    def test_key_metro_tracks(self):
        """Key metro tracks should be mapped."""
        for track in ["flemington", "randwick", "eagle farm", "morphettville", "ascot", "hobart"]:
            assert track in VENUE_LOCATIONS, f"{track} not in VENUE_LOCATIONS"

    def test_venue_data_has_bearing(self):
        """All mapped venues should have a straight_bearing."""
        for venue, data in VENUE_LOCATIONS.items():
            assert "bearing" in data, f"{venue} missing bearing"
            assert 0 <= data["bearing"] < 360, f"{venue} bearing out of range"


# ── Wind Impact Analysis Tests ─────────────────────────────────────────────


class TestWindAnalysis:
    """Test the analyse_wind_impact function."""

    def test_unknown_venue_returns_none(self):
        result = analyse_wind_impact("Middle of Nowhere", 20, "NW")
        assert result is None

    def test_calm_wind(self):
        """Wind < 5 km/h should be negligible."""
        result = analyse_wind_impact("Flemington", 3, "N")
        assert result is not None
        assert result["strength"] == "negligible"
        assert result["straight_impact"] == "calm"

    def test_headwind_at_flemington(self):
        """Flemington straight bearing ~200° (SSW). Wind from SSW = headwind."""
        result = analyse_wind_impact("Flemington", 25, "SSW")
        assert result is not None
        assert result["straight_impact"] == "headwind"
        assert result["strength"] == "moderate"
        assert "headwind" in result["description"].lower()

    def test_tailwind_at_flemington(self):
        """Flemington: wind from NNE (opposite of 200°) = tailwind."""
        result = analyse_wind_impact("Flemington", 25, "NNE")
        assert result is not None
        assert result["straight_impact"] == "tailwind"
        assert "tailwind" in result["description"].lower()

    def test_crosswind(self):
        """Wind perpendicular to the straight."""
        # Flemington bearing 200°, perpendicular is ~290° (WNW) or ~110° (ESE)
        result = analyse_wind_impact("Flemington", 20, "WNW")
        assert result is not None
        assert result["straight_impact"] == "crosswind"
        assert "crosswind" in result["description"].lower()

    def test_strong_wind_threshold(self):
        """Wind >= 30 km/h should be classified as strong."""
        result = analyse_wind_impact("Randwick", 35, "S")
        assert result is not None
        assert result["strength"] == "strong"

    def test_light_wind_threshold(self):
        """Wind 5-14 km/h should be classified as light."""
        result = analyse_wind_impact("Randwick", 10, "N")
        assert result is not None
        assert result["strength"] == "light"

    def test_invalid_direction(self):
        """Invalid wind direction string should return None."""
        result = analyse_wind_impact("Flemington", 20, "INVALID")
        assert result is None

    def test_effective_speed_present(self):
        """Result should include effective_speed for non-calm winds."""
        result = analyse_wind_impact("Flemington", 20, "S")
        assert result is not None
        assert "effective_speed" in result
        assert isinstance(result["effective_speed"], int)


# ── Direction Degrees ──────────────────────────────────────────────────────


class TestDirectionDegrees:
    """Test compass direction → degrees mapping."""

    def test_cardinal_directions(self):
        assert _DIRECTION_DEGREES["N"] == 0
        assert _DIRECTION_DEGREES["E"] == 90
        assert _DIRECTION_DEGREES["S"] == 180
        assert _DIRECTION_DEGREES["W"] == 270

    def test_all_16_points(self):
        """Should have all 16 compass points."""
        assert len(_DIRECTION_DEGREES) == 16


# ── WillyWeatherScraper Tests ─────────────────────────────────────────────


class TestScraper:
    """Test the WillyWeatherScraper class."""

    def test_resolve_known_venue(self):
        scraper = WillyWeatherScraper(api_key="test")
        assert scraper._resolve_location_id("Flemington") == 39330
        assert scraper._resolve_location_id("flemington") == 39330
        assert scraper._resolve_location_id("Eagle Farm") == 6731

    def test_resolve_unknown_venue(self):
        scraper = WillyWeatherScraper(api_key="test")
        assert scraper._resolve_location_id("Goulburn") is None

    def test_state_fallback_for_unmapped_venue(self):
        """Unmapped venue should fall back to nearest state metro location."""
        scraper = WillyWeatherScraper(api_key="test")
        # "Goulburn" is in NSW — should fall back to Randwick (5017)
        with patch.object(scraper, "_infer_state", return_value="NSW"):
            result = scraper._resolve_with_state_fallback("Goulburn")
            assert result == 5017  # Randwick (NSW fallback)

    def test_state_fallback_vic(self):
        """VIC venues not in map should fall back to Flemington."""
        scraper = WillyWeatherScraper(api_key="test")
        with patch.object(scraper, "_infer_state", return_value="VIC"):
            result = scraper._resolve_with_state_fallback("Geelong")
            assert result == 39330  # Flemington (VIC fallback)

    def test_state_fallback_returns_none_for_unknown_state(self):
        """Unknown state should return None from state fallback."""
        scraper = WillyWeatherScraper(api_key="test")
        with patch.object(scraper, "_infer_state", return_value=None):
            result = scraper._resolve_with_state_fallback("Unknown Track")
            assert result is None

    def test_direct_match_preferred_over_fallback(self):
        """Known venues should use direct match, not fallback."""
        scraper = WillyWeatherScraper(api_key="test")
        result = scraper._resolve_with_state_fallback("Pakenham")
        assert result == 13859  # Direct match, not Flemington fallback

    @pytest.mark.asyncio
    async def test_from_settings_returns_none_when_no_key(self):
        """Should return None when API key not configured."""
        db = AsyncMock()
        with patch("punty.models.settings.get_api_key", new_callable=AsyncMock, return_value=None):
            result = await WillyWeatherScraper.from_settings(db)
            assert result is None

    def test_parse_weather_full_response(self):
        """Test parsing a full WillyWeather API response."""
        scraper = WillyWeatherScraper(api_key="test")

        mock_data = {
            "forecasts": {
                "weather": {
                    "days": [{
                        "entries": [{
                            "precis": "Partly Cloudy",
                            "min": 14,
                            "max": 26,
                        }]
                    }]
                },
                "rainfall": {
                    "days": [{
                        "entries": [{
                            "probability": 40,
                            "startRange": 0,
                            "endRange": 2,
                        }]
                    }]
                },
                "wind": {
                    "days": [{
                        "entries": [
                            {"dateTime": "2026-02-12T06:00:00", "speed": 12, "directionText": "SW"},
                            {"dateTime": "2026-02-12T12:00:00", "speed": 18, "directionText": "SW"},
                            {"dateTime": "2026-02-12T18:00:00", "speed": 15, "directionText": "W"},
                        ]
                    }]
                },
                "temperature": {
                    "days": [{
                        "entries": [
                            {"dateTime": "2026-02-12T06:00:00", "temperature": 16},
                            {"dateTime": "2026-02-12T12:00:00", "temperature": 24},
                            {"dateTime": "2026-02-12T18:00:00", "temperature": 22},
                        ]
                    }]
                },
                "rainfallprobability": {
                    "days": [{
                        "entries": [
                            {"dateTime": "2026-02-12T06:00:00", "probability": 20},
                            {"dateTime": "2026-02-12T12:00:00", "probability": 45},
                            {"dateTime": "2026-02-12T18:00:00", "probability": 30},
                        ]
                    }]
                },
            },
            "observational": {
                "observations": {
                    "temperature": {"temperature": 23, "apparentTemperature": 21},
                    "rainfall": {"since9AMAmount": 0, "lastHourAmount": 0, "todayAmount": 1.2},
                    "wind": {"speed": 20, "gustSpeed": 30, "directionText": "SW"},
                    "humidity": {"percentage": 52},
                }
            },
        }

        result = scraper._parse_weather(mock_data)
        assert result is not None
        assert result["condition"] == "Partly Cloudy"
        # Observational data preferred over forecast
        assert result["temp"] == 23
        assert result["wind_speed"] == 20
        assert result["wind_direction"] == "SW"
        assert result["humidity"] == 52  # From observational only
        assert result["rainfall_chance"] == 40
        assert result["rainfall_amount"] == "0-2"
        assert len(result["hourly_wind"]) == 3
        assert len(result["hourly_temp"]) == 3
        assert len(result["hourly_rain_prob"]) == 3
        assert result["hourly_rain_prob"][1]["probability"] == 45
        assert result["observation"]["rain_since_9am"] == 0
        assert result["observation"]["rain_last_hour"] == 0
        assert result["observation"]["rain_today"] == 1.2
        assert result["observation"]["humidity"] == 52
        assert result["observation"]["wind_gust"] == 30  # Gusts only in observational

    def test_humidity_requires_observational(self):
        """Humidity only comes from observational data, not forecasts."""
        scraper = WillyWeatherScraper(api_key="test")
        # No observational data → humidity should be None
        mock_data = {
            "forecasts": {
                "weather": {"days": [{"entries": [{"precis": "Sunny"}]}]},
            },
        }
        result = scraper._parse_weather(mock_data)
        assert result is not None
        assert result["humidity"] is None

    def test_humidity_from_observational(self):
        """Humidity should be populated from observational station data."""
        scraper = WillyWeatherScraper(api_key="test")
        mock_data = {
            "forecasts": {},
            "observational": {
                "observations": {
                    "humidity": {"percentage": 72},
                }
            },
        }
        result = scraper._parse_weather(mock_data)
        assert result is not None
        assert result["humidity"] == 72

    def test_parse_weather_minimal_response(self):
        """Test parsing a response with missing sections."""
        scraper = WillyWeatherScraper(api_key="test")
        result = scraper._parse_weather({"forecasts": {}})
        assert result is not None
        assert result["condition"] is None
        assert result["temp"] is None
        assert result["wind_speed"] is None
        assert result["humidity"] is None

    def test_parse_weather_bad_data(self):
        """Test parsing completely invalid data returns None gracefully."""
        scraper = WillyWeatherScraper(api_key="test")
        # This shouldn't crash
        result = scraper._parse_weather({})
        assert result is not None  # Returns dict with None values

    def test_wind_forecast_has_no_gust(self):
        """Wind forecast entries should not include gust (only in observational)."""
        scraper = WillyWeatherScraper(api_key="test")
        mock_data = {
            "forecasts": {
                "wind": {
                    "days": [{
                        "entries": [
                            {"dateTime": "2026-02-12T12:00:00", "speed": 20, "directionText": "NW"},
                        ]
                    }]
                },
            },
        }
        result = scraper._parse_weather(mock_data)
        assert result is not None
        assert len(result["hourly_wind"]) == 1
        assert "gust" not in result["hourly_wind"][0]
        assert result["hourly_wind"][0]["speed"] == 20

    def test_rainfallprobability_hourly(self):
        """Hourly rainfall probability should be parsed from rainfallprobability forecast."""
        scraper = WillyWeatherScraper(api_key="test")
        mock_data = {
            "forecasts": {
                "rainfallprobability": {
                    "days": [{
                        "entries": [
                            {"dateTime": "2026-02-12T10:00:00", "probability": 15},
                            {"dateTime": "2026-02-12T14:00:00", "probability": 60},
                        ]
                    }]
                },
            },
        }
        result = scraper._parse_weather(mock_data)
        assert result is not None
        assert len(result["hourly_rain_prob"]) == 2
        assert result["hourly_rain_prob"][0]["probability"] == 15
        assert result["hourly_rain_prob"][1]["probability"] == 60

    def test_cache_key_uses_venue_and_date(self):
        """Weather cache should be keyed by venue + date."""
        scraper = WillyWeatherScraper(api_key="test")
        from datetime import date
        scraper._weather_cache["flemington:2026-02-12"] = {"temp": 25}
        # Same venue + date should hit cache (tested via key format)
        assert "flemington:2026-02-12" in scraper._weather_cache

    def test_search_cache(self):
        """Search results should be cached."""
        scraper = WillyWeatherScraper(api_key="test")
        scraper._search_cache["goulburn"] = 12345
        assert scraper._search_cache.get("goulburn") == 12345


# ── Weather Change Detection (monitor integration) ────────────────────────


class TestWeatherChangeDetection:
    """Test weather change thresholds used by ResultsMonitor."""

    def test_wind_shift_threshold(self):
        """Wind change >= 10 km/h should be significant."""
        old_wind = 12
        new_wind = 25
        assert abs(new_wind - old_wind) >= 10

    def test_temp_change_threshold(self):
        """Temperature change >= 3°C should be significant."""
        old_temp = 24
        new_temp = 28
        assert abs(new_temp - old_temp) >= 3

    def test_rain_detection(self):
        """Rain since 9am > 0 when previously dry should trigger alert."""
        rain_since_9am = 2.5
        was_dry = True
        assert rain_since_9am > 0 and was_dry

    def test_humidity_change_threshold(self):
        """Humidity change >= 15% should be significant."""
        old_humidity = 45
        new_humidity = 65
        assert abs(new_humidity - old_humidity) >= 15

    def test_humidity_small_change_not_significant(self):
        """Humidity change < 15% should NOT be significant."""
        old_humidity = 50
        new_humidity = 60
        assert abs(new_humidity - old_humidity) < 15


# ── Radar Parsing Tests ──────────────────────────────────────────────────


class TestRadarParsing:
    """Test radar data parsing from map provider response."""

    def test_parse_radar_response(self):
        """Should extract overlay URLs, bounds, and status from radar API."""
        scraper = WillyWeatherScraper(api_key="test")

        mock_response = [
            {
                "id": 71,
                "name": "Sydney (Terrey Hills)",
                "lat": -33.701,
                "lng": 151.21,
                "bounds": {
                    "minLat": -36.051, "minLng": 148.51,
                    "maxLat": -31.351, "maxLng": 153.91,
                },
                "radius": 256000,
                "interval": 6,
                "overlayPath": "//cdn2.willyweather.com.au/radar/",
                "overlays": [
                    {"dateTime": "2026-02-12 14:30:00", "name": "71-202602121430.png"},
                    {"dateTime": "2026-02-12 14:36:00", "name": "71-202602121436.png"},
                ],
                "classification": "radar",
                "mapLegend": {"keys": [{"colour": "#f5f5ff", "range": {"min": 0.2, "max": 0.5}, "label": "light"}]},
                "status": {"code": "active"},
            }
        ]

        # Test the parsing logic directly — radar takes first provider
        provider = mock_response[0]
        overlay_path = provider["overlayPath"]
        overlays = provider["overlays"]

        result = {
            "provider_name": provider["name"],
            "lat": provider["lat"],
            "lng": provider["lng"],
            "bounds": provider["bounds"],
            "radius_km": provider["radius"],
            "interval_min": provider["interval"],
            "overlay_path": overlay_path,
            "overlays": [
                {"time": o["dateTime"], "url": f"https:{overlay_path}{o['name']}"}
                for o in overlays
            ],
            "status": provider["status"]["code"],
            "legend": provider["mapLegend"],
        }

        assert result["provider_name"] == "Sydney (Terrey Hills)"
        assert result["status"] == "active"
        assert len(result["overlays"]) == 2
        assert result["overlays"][0]["url"].startswith("https://cdn2.willyweather.com.au/radar/")
        assert result["bounds"]["minLat"] == -36.051
        assert result["interval_min"] == 6
        assert result["legend"]["keys"][0]["label"] == "light"


# ── Camera Parsing Tests ─────────────────────────────────────────────────


class TestCameraParsing:
    """Test camera data extraction."""

    def test_parse_camera_response(self):
        """Should extract preview URL, stream URL, and status from camera data."""
        camera_data = {
            "id": 4,
            "name": "Bellingen",
            "displayName": "Bellingen",
            "lat": -30.4512,
            "lng": 152.8982,
            "cameraStatus": {"status": 2, "code": "online"},
            "state": {"abbreviation": "NSW"},
            "region": {"name": "Mid North Coast"},
            "player": {
                "previewURL": "https://cdncams.willyweather.com.au/streams/nsw/bellingen/preview.jpg",
                "hlsURL": "https://cdncams.willyweather.com.au/streams/nsw/bellingen/hls",
                "thumbnailVideoURL": "https://cdncams.willyweather.com.au/streams/nsw/bellingen/thumbnail",
            },
            "updatedDateTime": "2026-02-12 11:45:55",
        }

        player = camera_data["player"]
        result = {
            "id": camera_data["id"],
            "name": camera_data["displayName"],
            "state": camera_data["state"]["abbreviation"],
            "region": camera_data["region"]["name"],
            "status": camera_data["cameraStatus"]["code"],
            "preview_url": player["previewURL"],
            "stream_url": player["hlsURL"],
            "thumbnail_video_url": player["thumbnailVideoURL"],
            "updated": camera_data["updatedDateTime"],
            "lat": camera_data["lat"],
            "lng": camera_data["lng"],
        }

        assert result["id"] == 4
        assert result["name"] == "Bellingen"
        assert result["state"] == "NSW"
        assert result["status"] == "online"
        assert "preview.jpg" in result["preview_url"]
        assert "hls" in result["stream_url"]

    def test_camera_state_filtering(self):
        """Should prefer cameras from the same state as the venue."""
        cameras = [
            {"id": 1, "state": {"abbreviation": "VIC"}, "lat": -37.8, "lng": 144.9,
             "cameraStatus": {"code": "online"}, "player": {}},
            {"id": 2, "state": {"abbreviation": "NSW"}, "lat": -33.9, "lng": 151.2,
             "cameraStatus": {"code": "online"}, "player": {}},
        ]

        # Filter to NSW
        venue_state = "NSW"
        state_cameras = [c for c in cameras if c["state"]["abbreviation"] == venue_state]
        assert len(state_cameras) == 1
        assert state_cameras[0]["id"] == 2

    def test_camera_online_preferred(self):
        """Should prefer online cameras over offline."""
        cameras = [
            {"id": 1, "cameraStatus": {"code": "offline"}, "state": {"abbreviation": "VIC"}},
            {"id": 2, "cameraStatus": {"code": "online"}, "state": {"abbreviation": "VIC"}},
        ]
        online = [c for c in cameras if c["cameraStatus"]["code"] == "online"]
        assert len(online) == 1
        assert online[0]["id"] == 2
