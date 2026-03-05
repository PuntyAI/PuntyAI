"""Open-Meteo weather API for NZ racecourses.

Free, no API key required. Returns same standardized dict as WillyWeather.
Used as fallback when WillyWeather (AU-only) can't resolve a venue.
"""

import logging
from datetime import date, datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# NZ racecourse coordinates (lat, lon)
NZ_VENUES: dict[str, tuple[float, float]] = {
    "tauherenikau": (-41.12, 175.27),
    "trentham": (-41.12, 175.03),
    "otaki": (-40.76, 175.15),
    "awapuni": (-40.37, 175.60),
    "hastings": (-39.64, 176.85),
    "woodville": (-40.34, 175.87),
    "wanganui": (-39.93, 175.05),
    "ellerslie": (-36.88, 174.80),
    "te rapa": (-37.76, 175.27),
    "matamata": (-37.81, 175.77),
    "rotorua": (-38.14, 176.25),
    "tauranga": (-37.69, 176.17),
    "ruakaka": (-35.90, 174.52),
    "pukekohe": (-37.21, 174.90),
    "riccarton": (-43.54, 172.58),
    "addington": (-43.55, 172.61),
    "wingatui": (-45.91, 170.31),
    "ascot park": (-46.40, 168.35),
    "riverton": (-46.35, 168.02),
    "oamaru": (-45.10, 170.97),
    "cromwell": (-45.05, 169.20),
    "waikouaiti": (-45.62, 170.67),
    "kurow": (-44.73, 170.47),
    "waimate": (-44.73, 171.05),
    "timaru": (-44.40, 171.25),
    "new plymouth": (-39.06, 174.08),
    "hawera": (-39.59, 174.28),
    "wairoa": (-39.04, 177.41),
    "waverley": (-39.76, 174.63),
    "stratford": (-39.34, 174.28),
    "gore": (-46.10, 168.94),
    "invercargill": (-46.41, 168.36),
    "ashburton": (-43.90, 171.75),
    "rangiora": (-43.31, 172.59),
    "greymouth": (-42.45, 171.21),
    "reefton": (-42.12, 171.87),
    "kumara": (-42.63, 171.22),
    "westport": (-41.76, 171.60),
}


def _resolve_nz_coords(venue: str) -> tuple[float, float] | None:
    """Resolve a venue name to NZ lat/lon coordinates."""
    key = venue.lower().strip()
    if key in NZ_VENUES:
        return NZ_VENUES[key]
    # Partial match
    for name, coords in NZ_VENUES.items():
        if name in key or key in name:
            return coords
    return None


async def get_nz_weather(venue: str, race_date: date | None = None) -> dict | None:
    """Fetch weather for an NZ venue via Open-Meteo.

    Returns a dict matching WillyWeather's standardized format so it can
    be used as a drop-in replacement in the context builder.
    """
    coords = _resolve_nz_coords(venue)
    if not coords:
        return None

    lat, lon = coords
    target_date = race_date or date.today()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(BASE_URL, params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation_probability",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                    "weather_code",
                ]),
                "timezone": "Pacific/Auckland",
                "forecast_days": 1,
            })
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Open-Meteo fetch failed for '{venue}': {e}")
        return None

    return _parse_response(data, venue)


# WMO weather codes → human-readable conditions
_WMO_CODES = {
    0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow",
    80: "Light Showers", 81: "Showers", 82: "Heavy Showers",
    95: "Thunderstorm", 96: "Thunderstorm + Hail", 99: "Severe Thunderstorm",
}


def _wind_direction_text(degrees: float | None) -> str | None:
    """Convert wind direction degrees to cardinal text."""
    if degrees is None:
        return None
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = round(degrees / 22.5) % 16
    return directions[idx]


def _parse_response(data: dict, venue: str) -> dict | None:
    """Parse Open-Meteo response into WillyWeather-compatible format."""
    try:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        rain_prob = hourly.get("precipitation_probability", [])
        precip = hourly.get("precipitation", [])
        wind_speed = hourly.get("wind_speed_10m", [])
        wind_dir = hourly.get("wind_direction_10m", [])
        wind_gust = hourly.get("wind_gusts_10m", [])
        weather_code = hourly.get("weather_code", [])

        if not times:
            return None

        # Use midday (12:00) as representative, or closest available
        mid_idx = min(12, len(times) - 1)

        # Condition from WMO weather code
        condition = _WMO_CODES.get(weather_code[mid_idx] if weather_code else 0, "Unknown")

        # Build hourly arrays matching WillyWeather format
        hourly_wind = []
        hourly_temp = []
        hourly_rain_prob = []
        for i, t in enumerate(times):
            if i < len(wind_speed):
                hourly_wind.append({
                    "time": t,
                    "speed": wind_speed[i],
                    "direction": _wind_direction_text(wind_dir[i] if i < len(wind_dir) else None),
                })
            if i < len(temps):
                hourly_temp.append({"time": t, "temp": temps[i]})
            if i < len(rain_prob):
                hourly_rain_prob.append({"time": t, "probability": rain_prob[i]})

        # Max rainfall probability as "chance"
        rainfall_chance = max(rain_prob) if rain_prob else None
        total_precip = sum(p for p in precip if p) if precip else 0

        result = {
            "condition": condition,
            "temp": int(temps[mid_idx]) if temps else None,
            "temp_min": int(min(temps)) if temps else None,
            "temp_max": int(max(temps)) if temps else None,
            "wind_speed": int(wind_speed[mid_idx]) if wind_speed else None,
            "wind_direction": _wind_direction_text(wind_dir[mid_idx] if wind_dir else None),
            "humidity": int(humidity[mid_idx]) if humidity else None,
            "rainfall_chance": rainfall_chance,
            "rainfall_amount": f"0-{total_precip:.1f}" if total_precip else "0-0",
            "hourly_wind": hourly_wind,
            "hourly_temp": hourly_temp,
            "hourly_rain_prob": hourly_rain_prob,
            "observation": None,  # Open-Meteo doesn't have live observations
            "source": "open-meteo",
        }

        logger.info(
            f"Open-Meteo weather for {venue}: {condition}, "
            f"{result['temp']}°C, wind {result['wind_speed']}km/h {result['wind_direction']}, "
            f"rain {rainfall_chance}%"
        )
        return result

    except Exception as e:
        logger.warning(f"Failed to parse Open-Meteo response for '{venue}': {e}")
        return None
