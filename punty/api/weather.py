"""API endpoints for weather data."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.database import get_db

router = APIRouter()


@router.get("/status")
async def weather_status(db: AsyncSession = Depends(get_db)):
    """Check WillyWeather API configuration status."""
    from punty.scrapers.willyweather import WillyWeatherScraper

    ww = await WillyWeatherScraper.from_settings(db)
    if not ww:
        return {"configured": False, "message": "WillyWeather API key not configured"}

    await ww.close()
    return {"configured": True}


@router.get("/{meeting_id}")
async def get_meeting_weather(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get full weather data for a meeting including wind analysis and radar.

    Returns current conditions, hourly forecasts, wind impact analysis,
    radar overlay data, and nearest sky camera.
    """
    from punty.models.meeting import Meeting
    from punty.scrapers.willyweather import WillyWeatherScraper, analyse_wind_impact

    result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    ww = await WillyWeatherScraper.from_settings(db)
    if not ww:
        # Return stored weather fields even without API key
        wind_analysis = None
        if meeting.weather_wind_speed and meeting.weather_wind_dir:
            wind_analysis = analyse_wind_impact(
                meeting.venue, meeting.weather_wind_speed, meeting.weather_wind_dir
            )
        return {
            "source": "stored",
            "venue": meeting.venue,
            "condition": meeting.weather_condition,
            "temp": meeting.weather_temp,
            "wind_speed": meeting.weather_wind_speed,
            "wind_direction": meeting.weather_wind_dir,
            "humidity": getattr(meeting, "weather_humidity", None),
            "rainfall": meeting.rainfall,
            "wind_analysis": wind_analysis,
            "radar": None,
            "camera": None,
            "hourly_wind": None,
            "hourly_temp": None,
            "hourly_rain_prob": None,
            "observation": None,
        }

    try:
        weather = await ww.get_weather(meeting.venue, meeting.date)
        radar = await ww.get_radar(meeting.venue)
        camera = await ww.get_nearest_camera(meeting.venue)
    finally:
        await ww.close()

    wind_analysis = None
    if weather and weather.get("wind_speed") and weather.get("wind_direction"):
        wind_analysis = analyse_wind_impact(
            meeting.venue, weather["wind_speed"], weather["wind_direction"]
        )

    if not weather:
        return {
            "source": "unavailable",
            "venue": meeting.venue,
            "error": f"No weather data available for {meeting.venue}",
            "radar": radar,
            "camera": camera,
        }

    return {
        "source": "willyweather",
        "venue": meeting.venue,
        "condition": weather.get("condition"),
        "temp": weather.get("temp"),
        "temp_min": weather.get("temp_min"),
        "temp_max": weather.get("temp_max"),
        "wind_speed": weather.get("wind_speed"),
        "wind_direction": weather.get("wind_direction"),
        "humidity": weather.get("humidity"),
        "rainfall_chance": weather.get("rainfall_chance"),
        "rainfall_amount": weather.get("rainfall_amount"),
        "wind_analysis": wind_analysis,
        "hourly_wind": weather.get("hourly_wind"),
        "hourly_temp": weather.get("hourly_temp"),
        "hourly_rain_prob": weather.get("hourly_rain_prob"),
        "observation": weather.get("observation"),
        "radar": radar,
        "camera": camera,
    }


@router.post("/{meeting_id}/refresh")
async def refresh_meeting_weather(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Force a fresh weather fetch from WillyWeather and update meeting fields."""
    from punty.models.meeting import Meeting
    from punty.scrapers.willyweather import WillyWeatherScraper, analyse_wind_impact

    result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    ww = await WillyWeatherScraper.from_settings(db)
    if not ww:
        raise HTTPException(status_code=400, detail="WillyWeather API key not configured")

    try:
        # Clear cache to force fresh fetch
        ww._weather_cache.clear()
        weather = await ww.get_weather(meeting.venue, meeting.date)
    finally:
        await ww.close()

    if not weather:
        raise HTTPException(status_code=502, detail=f"Failed to fetch weather for {meeting.venue}")

    # Update meeting fields
    old_values = {
        "temp": meeting.weather_temp,
        "wind_speed": meeting.weather_wind_speed,
        "wind_dir": meeting.weather_wind_dir,
        "condition": meeting.weather_condition,
        "humidity": getattr(meeting, "weather_humidity", None),
    }

    if weather.get("temp") is not None:
        meeting.weather_temp = weather["temp"]
    if weather.get("wind_speed") is not None:
        meeting.weather_wind_speed = weather["wind_speed"]
    if weather.get("wind_direction"):
        meeting.weather_wind_dir = weather["wind_direction"]
    if weather.get("condition"):
        meeting.weather_condition = weather["condition"]
    if weather.get("humidity") is not None:
        meeting.weather_humidity = weather["humidity"]
    if weather.get("rainfall_chance") is not None:
        meeting.rainfall = f"{weather['rainfall_chance']}% chance, {weather['rainfall_amount']}mm"
    await db.commit()

    wind_analysis = None
    if weather.get("wind_speed") and weather.get("wind_direction"):
        wind_analysis = analyse_wind_impact(
            meeting.venue, weather["wind_speed"], weather["wind_direction"]
        )

    return {
        "status": "updated",
        "venue": meeting.venue,
        "condition": weather.get("condition"),
        "temp": weather.get("temp"),
        "wind_speed": weather.get("wind_speed"),
        "wind_direction": weather.get("wind_direction"),
        "humidity": weather.get("humidity"),
        "rainfall_chance": weather.get("rainfall_chance"),
        "wind_analysis": wind_analysis,
        "previous": old_values,
    }


@router.get("/{meeting_id}/radar")
async def get_meeting_radar(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get radar overlay data for a meeting venue.

    Returns radar images, bounds, and timing for rendering an animated radar map.
    """
    from punty.models.meeting import Meeting
    from punty.scrapers.willyweather import WillyWeatherScraper

    result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    ww = await WillyWeatherScraper.from_settings(db)
    if not ww:
        raise HTTPException(status_code=400, detail="WillyWeather API key not configured")

    try:
        radar = await ww.get_radar(meeting.venue)
    finally:
        await ww.close()

    if not radar:
        raise HTTPException(status_code=404, detail=f"No radar data for {meeting.venue}")

    return radar


@router.get("/{meeting_id}/camera")
async def get_meeting_camera(meeting_id: str, db: AsyncSession = Depends(get_db)):
    """Get nearest sky camera for a meeting venue.

    Returns camera preview image, stream URLs, and status.
    """
    from punty.models.meeting import Meeting
    from punty.scrapers.willyweather import WillyWeatherScraper

    result = await db.execute(select(Meeting).where(Meeting.id == meeting_id))
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    ww = await WillyWeatherScraper.from_settings(db)
    if not ww:
        raise HTTPException(status_code=400, detail="WillyWeather API key not configured")

    try:
        camera = await ww.get_nearest_camera(meeting.venue)
    finally:
        await ww.close()

    if not camera:
        raise HTTPException(status_code=404, detail=f"No cameras found near {meeting.venue}")

    return camera
