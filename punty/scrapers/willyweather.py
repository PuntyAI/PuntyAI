"""WillyWeather API scraper for real-time racecourse weather data.

Provides hourly wind, rainfall, temperature forecasts and current observations.
Supplements PuntingForm weather with fresher, more detailed data.

Requires API key stored in app_settings as 'willyweather_api_key'.
"""

import logging
import math
from datetime import date, datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.willyweather.com.au/v2"

# Pre-mapped venues → WillyWeather location IDs (max 10 tracked locations)
# Each entry: {location_id, straight_bearing (degrees from north, direction runners finish)}
VENUE_LOCATIONS: dict[str, dict] = {
    # VIC
    "flemington":    {"id": 39330, "bearing": 200, "state": "VIC"},
    "caulfield":     {"id": 39330, "bearing": 220, "state": "VIC"},  # shares Flemington location
    "moonee valley": {"id": 39330, "bearing": 180, "state": "VIC"},
    "sandown":       {"id": 39330, "bearing": 250, "state": "VIC"},
    "pakenham":      {"id": 13859, "bearing": 200, "state": "VIC"},
    "kilmore":       {"id": 11158, "bearing": 200, "state": "VIC"},  # uses Ballarat station as nearest
    "ballarat":      {"id": 11158, "bearing": 250, "state": "VIC"},
    # NSW
    "randwick":      {"id": 5017,  "bearing": 350, "state": "NSW"},
    "rosehill":      {"id": 5017,  "bearing": 200, "state": "NSW"},  # shares Randwick location
    "newcastle":     {"id": 2919,  "bearing": 120, "state": "NSW"},
    # QLD
    "eagle farm":    {"id": 6731,  "bearing": 160, "state": "QLD"},
    "doomben":       {"id": 6731,  "bearing": 100, "state": "QLD"},  # shares Eagle Farm location
    "gold coast":    {"id": 5983,  "bearing": 160, "state": "QLD"},
    # SA
    "morphettville": {"id": 9665,  "bearing": 350, "state": "SA"},
    # WA
    "ascot":         {"id": 16006, "bearing": 180, "state": "WA"},
    # TAS
    "hobart":        {"id": 10638, "bearing": 180, "state": "TAS"},
    "elwick":        {"id": 10638, "bearing": 180, "state": "TAS"},
}

# Cardinal direction → degrees mapping
_DIRECTION_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def analyse_wind_impact(venue: str, wind_speed: int, wind_direction: str) -> dict | None:
    """Analyse how wind affects racing at this venue based on straight bearing.

    Args:
        venue: Racecourse name
        wind_speed: Wind speed in km/h
        wind_direction: Cardinal direction (e.g. "NW", "SSE")

    Returns:
        Dict with straight_impact, strength, and description, or None if unknown.
    """
    venue_key = venue.lower().strip()
    venue_data = VENUE_LOCATIONS.get(venue_key)
    if not venue_data:
        # Strip sponsor prefix and retry
        clean = WillyWeatherScraper._strip_sponsor(venue)
        venue_data = VENUE_LOCATIONS.get(clean)
        if not venue_data:
            for key, data in VENUE_LOCATIONS.items():
                if key in clean or clean in key:
                    venue_data = data
                    break
    if not venue_data or not venue_data.get("bearing"):
        return None

    wind_deg = _DIRECTION_DEGREES.get(wind_direction.upper().strip())
    if wind_deg is None:
        return None

    if wind_speed < 5:
        return {
            "straight_impact": "calm",
            "strength": "negligible",
            "description": f"Light winds ({wind_speed}km/h) — negligible impact at {venue}",
        }

    straight_bearing = venue_data["bearing"]

    # Angle between wind direction and straight bearing
    # Wind FROM a direction means it blows TOWARD the opposite direction
    # Runners heading in straight_bearing direction face headwind from that direction
    wind_from_deg = wind_deg  # "NW wind" means wind blowing FROM NW
    # Headwind = wind blowing FROM the direction runners are heading
    angle_diff = abs(straight_bearing - wind_from_deg) % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Cosine component: 0° = pure headwind, 90° = crosswind, 180° = tailwind
    cos_component = math.cos(math.radians(angle_diff))

    if abs(cos_component) < 0.3:
        impact = "crosswind"
    elif cos_component > 0:
        impact = "headwind"
    else:
        impact = "tailwind"

    # Strength classification
    if wind_speed >= 30:
        strength = "strong"
    elif wind_speed >= 15:
        strength = "moderate"
    else:
        strength = "light"

    # Build description
    effective_speed = abs(int(wind_speed * cos_component))
    if impact == "crosswind":
        desc = (
            f"{strength.title()} {wind_speed}km/h {wind_direction} crosswind "
            f"at {venue} — may affect wide runners"
        )
    elif impact == "headwind":
        desc = (
            f"{strength.title()} {wind_speed}km/h {wind_direction} "
            f"({effective_speed}km/h headwind up the straight) at {venue} "
            f"— favours on-pace runners, harder for closers to sustain finishing burst"
        )
    else:
        desc = (
            f"{strength.title()} {wind_speed}km/h {wind_direction} "
            f"({effective_speed}km/h tailwind up the straight) at {venue} "
            f"— helps closers, runners can sustain a longer sprint"
        )

    return {
        "straight_impact": impact,
        "strength": strength,
        "effective_speed": effective_speed,
        "description": desc,
    }


class WillyWeatherScraper:
    """WillyWeather API scraper for racecourse weather."""

    # State → location ID fallback (nearest metro location per state)
    _STATE_FALLBACK: dict[str, int] = {
        "VIC": 39330,   # Flemington / Melbourne
        "NSW": 5017,    # Randwick / Sydney
        "QLD": 6731,    # Eagle Farm / Brisbane
        "SA": 9665,     # Morphettville / Adelaide
        "WA": 16006,    # Ascot / Perth
        "TAS": 10638,   # Hobart
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._search_cache: dict[str, int] = {}
        self._weather_cache: dict[str, dict] = {}  # key: "venue:date"

    @classmethod
    async def from_settings(cls, db) -> Optional["WillyWeatherScraper"]:
        """Create scraper with API key from app_settings. Returns None if not configured."""
        from punty.models.settings import get_api_key

        api_key = await get_api_key(db, "willyweather_api_key")
        if not api_key:
            return None
        return cls(api_key=api_key)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=f"{BASE_URL}/{self.api_key}",
                timeout=15.0,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def search_location(self, query: str) -> int | None:
        """Search for a WillyWeather location by name.

        Returns the location ID or None if not found.
        Results are cached for the session.
        """
        cache_key = query.lower().strip()
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        try:
            resp = await self.client.get("/search.json", params={"query": query})
            resp.raise_for_status()
            results = resp.json()
            if results and isinstance(results, list):
                location_id = results[0].get("id")
                if location_id:
                    self._search_cache[cache_key] = location_id
                    logger.info(f"WillyWeather search '{query}' → location {location_id}")
                    return location_id
        except Exception as e:
            logger.warning(f"WillyWeather search failed for '{query}': {e}")

        return None

    _SPONSOR_PREFIXES = [
        "ladbrokes", "tab", "bet365", "sportsbet", "neds", "pointsbet",
        "unibet", "betfair", "palmerbet", "bluebet", "topsport", "aquis",
        "picklebet",
    ]

    @staticmethod
    def _strip_sponsor(venue: str) -> str:
        """Strip sponsor prefix from venue name (e.g. 'bet365 Park Kilmore' → 'park kilmore')."""
        v = venue.lower().strip()
        for prefix in WillyWeatherScraper._SPONSOR_PREFIXES:
            if v.startswith(prefix + " "):
                return v[len(prefix) + 1:]
        return v

    def _resolve_location_id(self, venue: str) -> int | None:
        """Resolve venue name to WillyWeather location ID from pre-mapped dict."""
        venue_key = venue.lower().strip()
        venue_data = VENUE_LOCATIONS.get(venue_key)
        if venue_data:
            return venue_data["id"]
        # Try with sponsor prefix stripped
        clean = self._strip_sponsor(venue)
        if clean != venue_key:
            venue_data = VENUE_LOCATIONS.get(clean)
            if venue_data:
                return venue_data["id"]
            # Partial match: check if any known venue is contained in the clean name
            for key, data in VENUE_LOCATIONS.items():
                if key in clean or clean in key:
                    return data["id"]
        return None

    def _resolve_with_state_fallback(self, venue: str, state: str | None = None) -> int | None:
        """Resolve venue to location ID, falling back to nearest state location.

        For venues not in our pre-mapped list, uses the closest metro track
        location from the same state. Weather at a nearby metro track is usually
        close enough for betting decisions.
        """
        # Direct match first
        location_id = self._resolve_location_id(venue)
        if location_id:
            return location_id

        # Try to infer state from racing_knowledge
        if not state:
            state = self._infer_state(venue)

        if state and state in self._STATE_FALLBACK:
            fallback_id = self._STATE_FALLBACK[state]
            logger.info(f"Using state fallback for '{venue}' ({state}) → location {fallback_id}")
            return fallback_id

        return None

    @staticmethod
    def _infer_state(venue: str) -> str | None:
        """Try to infer state from racing_knowledge TRACKS data."""
        try:
            from punty.memory.racing_knowledge import TRACKS
            venue_lower = venue.lower().strip()
            clean = WillyWeatherScraper._strip_sponsor(venue)
            for track_name, track_data in TRACKS.items():
                tk = track_name.lower()
                if tk == venue_lower or tk == clean or tk in clean or clean in tk:
                    return track_data.get("state")
        except Exception:
            pass
        return None

    async def get_weather(self, venue: str, race_date: date | None = None) -> dict | None:
        """Fetch weather data for a racecourse venue.

        Args:
            venue: Racecourse name (e.g. "Flemington", "Randwick")
            race_date: Date for forecast (defaults to today)

        Returns:
            Standardized weather dict or None if unavailable.
        """
        today_str = (race_date or date.today()).isoformat()
        cache_key = f"{venue.lower().strip()}:{today_str}"
        if cache_key in self._weather_cache:
            return self._weather_cache[cache_key]

        # Resolve location ID: pre-mapped → state fallback → search (soft)
        location_id = self._resolve_with_state_fallback(venue)
        if not location_id:
            # Last resort: search API (may not be in plan)
            location_id = await self.search_location(venue)
        if not location_id:
            logger.warning(f"No WillyWeather location found for '{venue}'")
            return None

        try:
            resp = await self.client.get(
                f"/locations/{location_id}/weather.json",
                params={
                    "forecasts": "weather,rainfall,wind,temperature,rainfallprobability",
                    "observational": "true",
                    "days": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            result = self._parse_weather(data)
            if result:
                self._weather_cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"WillyWeather fetch failed for '{venue}': {e}")
            return None

    def _parse_weather(self, data: dict) -> dict | None:
        """Parse WillyWeather API response into standardized dict."""
        try:
            forecasts = data.get("forecasts", {})

            # Weather summary
            condition = None
            temp_min = None
            temp_max = None
            weather_days = forecasts.get("weather", {}).get("days", [])
            if weather_days:
                entries = weather_days[0].get("entries", [])
                if entries:
                    condition = entries[0].get("precis")
                    temp_min = entries[0].get("min")
                    temp_max = entries[0].get("max")

            # Rainfall
            rainfall_chance = None
            rainfall_amount = None
            rain_days = forecasts.get("rainfall", {}).get("days", [])
            if rain_days:
                entries = rain_days[0].get("entries", [])
                if entries:
                    rainfall_chance = entries[0].get("probability")
                    start = entries[0].get("startRange") or 0
                    end = entries[0].get("endRange") or 0
                    rainfall_amount = f"{start}-{end}"

            # Wind (hourly entries — forecast has speed/direction/directionText, no gust)
            hourly_wind = []
            wind_speed = None
            wind_direction = None
            wind_days = forecasts.get("wind", {}).get("days", [])
            if wind_days:
                entries = wind_days[0].get("entries", [])
                for entry in entries:
                    hourly_wind.append({
                        "time": entry.get("dateTime"),
                        "speed": entry.get("speed"),
                        "direction": entry.get("directionText"),
                    })
                # Use midday entry (or latest available) as representative
                if entries:
                    mid_idx = min(len(entries) - 1, len(entries) // 2)
                    wind_speed = entries[mid_idx].get("speed")
                    wind_direction = entries[mid_idx].get("directionText")

            # Temperature (hourly entries)
            hourly_temp = []
            current_temp = None
            temp_days = forecasts.get("temperature", {}).get("days", [])
            if temp_days:
                entries = temp_days[0].get("entries", [])
                for entry in entries:
                    hourly_temp.append({
                        "time": entry.get("dateTime"),
                        "temp": entry.get("temperature"),
                    })
                if entries:
                    mid_idx = min(len(entries) - 1, len(entries) // 2)
                    current_temp = entries[mid_idx].get("temperature")

            # Rainfall probability (hourly entries — separate from daily rainfall summary)
            hourly_rain_prob = []
            rain_prob_days = forecasts.get("rainfallprobability", {}).get("days", [])
            if rain_prob_days:
                entries = rain_prob_days[0].get("entries", [])
                for entry in entries:
                    hourly_rain_prob.append({
                        "time": entry.get("dateTime"),
                        "probability": entry.get("probability"),
                    })

            # Humidity — only available from observational data (no forecast type)
            humidity = None

            # Observational data (current actual conditions)
            observation = None
            obs_data = data.get("observational", {}).get("observations", {})
            if obs_data:
                temp_obs = obs_data.get("temperature", {})
                rain_obs = obs_data.get("rainfall", {})
                wind_obs = obs_data.get("wind", {})
                humidity_obs = obs_data.get("humidity", {})
                observation = {
                    "temp": temp_obs.get("temperature"),
                    "feels_like": temp_obs.get("apparentTemperature"),
                    "rain_since_9am": rain_obs.get("since9AMAmount"),
                    "rain_last_hour": rain_obs.get("lastHourAmount"),
                    "rain_today": rain_obs.get("todayAmount"),
                    "wind_speed": wind_obs.get("speed"),
                    "wind_gust": wind_obs.get("gustSpeed"),
                    "wind_direction": wind_obs.get("directionText"),
                    "humidity": humidity_obs.get("percentage"),
                }
                # Prefer actual observations over forecast where available
                if observation["temp"] is not None:
                    current_temp = observation["temp"]
                if observation["wind_speed"] is not None:
                    wind_speed = observation["wind_speed"]
                if observation["wind_direction"]:
                    wind_direction = observation["wind_direction"]
                if observation.get("humidity") is not None:
                    humidity = observation["humidity"]

            return {
                "condition": condition,
                "temp": int(current_temp) if current_temp is not None else None,
                "temp_min": temp_min,
                "temp_max": temp_max,
                "wind_speed": int(wind_speed) if wind_speed is not None else None,
                "wind_direction": wind_direction,
                "humidity": int(humidity) if humidity is not None else None,
                "rainfall_chance": rainfall_chance,
                "rainfall_amount": rainfall_amount,
                "hourly_wind": hourly_wind,
                "hourly_temp": hourly_temp,
                "hourly_rain_prob": hourly_rain_prob,
                "observation": observation,
            }

        except Exception as e:
            logger.warning(f"Failed to parse WillyWeather response: {e}")
            return None

    async def get_weather_at_time(
        self, venue: str, race_time: datetime
    ) -> dict | None:
        """Get weather forecast for a specific race time.

        Finds the hourly entry closest to the given race time.
        """
        weather = await self.get_weather(venue, race_time.date())
        if not weather:
            return None

        target_hour = race_time.hour

        # Find closest wind entry
        wind_at_time = None
        for entry in weather.get("hourly_wind", []):
            dt_str = entry.get("time", "")
            try:
                entry_hour = datetime.fromisoformat(dt_str).hour
                if entry_hour == target_hour:
                    wind_at_time = entry
                    break
            except (ValueError, TypeError):
                continue

        # Find closest temp entry
        temp_at_time = None
        for entry in weather.get("hourly_temp", []):
            dt_str = entry.get("time", "")
            try:
                entry_hour = datetime.fromisoformat(dt_str).hour
                if entry_hour == target_hour:
                    temp_at_time = entry
                    break
            except (ValueError, TypeError):
                continue

        # Find closest rain probability entry
        rain_prob_at_time = None
        for entry in weather.get("hourly_rain_prob", []):
            dt_str = entry.get("time", "")
            try:
                entry_hour = datetime.fromisoformat(dt_str).hour
                if entry_hour == target_hour:
                    rain_prob_at_time = entry.get("probability")
                    break
            except (ValueError, TypeError):
                continue

        obs = weather.get("observation") or {}
        return {
            "wind_speed": wind_at_time["speed"] if wind_at_time else weather["wind_speed"],
            "wind_direction": wind_at_time["direction"] if wind_at_time else weather["wind_direction"],
            "wind_gust": obs.get("wind_gust"),  # gusts only from observational
            "temp": temp_at_time["temp"] if temp_at_time else weather["temp"],
            "humidity": weather.get("humidity"),
            "condition": weather["condition"],
            "rainfall_chance": rain_prob_at_time if rain_prob_at_time is not None else weather["rainfall_chance"],
        }

    async def get_radar(self, venue: str) -> dict | None:
        """Get regional radar data for a venue's location.

        Returns radar overlay data including image paths and bounds,
        suitable for rendering an animated radar map.
        """
        location_id = self._resolve_with_state_fallback(venue)
        if not location_id:
            return None

        try:
            resp = await self.client.get(
                f"/locations/{location_id}/maps.json",
                params={"mapTypes": "regional-radar"},
            )
            resp.raise_for_status()
            providers = resp.json()

            if not providers or not isinstance(providers, list):
                return None

            # Use the first (closest) radar provider
            provider = providers[0]
            overlay_path = provider.get("overlayPath", "")
            overlays = provider.get("overlays", [])

            return {
                "provider_name": provider.get("name"),
                "lat": provider.get("lat"),
                "lng": provider.get("lng"),
                "bounds": provider.get("bounds"),
                "radius_km": provider.get("radius"),
                "interval_min": provider.get("interval"),
                "overlay_path": overlay_path,
                "overlays": [
                    {
                        "time": o.get("dateTime"),
                        "url": f"https:{overlay_path}{o['name']}" if o.get("name") else None,
                    }
                    for o in overlays
                ],
                "status": provider.get("status", {}).get("code"),
                "legend": provider.get("mapLegend"),
            }

        except Exception as e:
            logger.warning(f"WillyWeather radar fetch failed for '{venue}': {e}")
            return None

    async def get_nearest_camera(self, venue: str) -> dict | None:
        """Find the nearest sky camera to a racecourse.

        Uses the cameras API to find cameras, then picks the closest
        by comparing lat/lng with the venue's known location.

        Returns camera preview URL, stream URLs, and metadata.
        """
        location_id = self._resolve_with_state_fallback(venue)
        if not location_id:
            return None

        # Get venue coordinates from a lightweight location lookup
        venue_lat, venue_lng = None, None
        try:
            resp = await self.client.get(f"/locations/{location_id}.json")
            resp.raise_for_status()
            loc_data = resp.json()
            venue_lat = loc_data.get("lat")
            venue_lng = loc_data.get("lng")
        except Exception:
            pass

        try:
            resp = await self.client.get("/cameras.json")
            resp.raise_for_status()
            cameras = resp.json()

            if not cameras or not isinstance(cameras, list):
                return None

            # Filter to online sky cameras only
            online_cameras = [
                c for c in cameras
                if c.get("cameraStatus", {}).get("code") == "online"
            ]
            if not online_cameras:
                online_cameras = cameras  # fall back to all if none online

            # Find nearest camera by state match + distance
            venue_key = venue.lower().strip()
            venue_state = VENUE_LOCATIONS.get(venue_key, {}).get("state")
            if not venue_state:
                venue_state = self._infer_state(venue)

            # Filter by same state if possible
            state_cameras = [
                c for c in online_cameras
                if c.get("state", {}).get("abbreviation") == venue_state
            ] if venue_state else online_cameras

            if not state_cameras:
                state_cameras = online_cameras

            # Pick closest by lat/lng if we have coordinates
            best = state_cameras[0]
            if venue_lat is not None and venue_lng is not None:
                def _distance(cam):
                    clat = cam.get("lat", 0)
                    clng = cam.get("lng", 0)
                    return math.sqrt((clat - venue_lat) ** 2 + (clng - venue_lng) ** 2)
                best = min(state_cameras, key=_distance)

            player = best.get("player", {})
            return {
                "id": best.get("id"),
                "name": best.get("displayName") or best.get("name"),
                "state": best.get("state", {}).get("abbreviation"),
                "region": best.get("region", {}).get("name"),
                "status": best.get("cameraStatus", {}).get("code"),
                "preview_url": player.get("previewURL"),
                "stream_url": player.get("hlsURL"),
                "thumbnail_video_url": player.get("thumbnailVideoURL"),
                "updated": best.get("updatedDateTime"),
                "lat": best.get("lat"),
                "lng": best.get("lng"),
            }

        except Exception as e:
            logger.warning(f"WillyWeather camera fetch failed for '{venue}': {e}")
            return None
