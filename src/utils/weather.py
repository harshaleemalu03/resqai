# =============================================================
# ResQAI — Weather Integration Module
# Fetches real-time weather data from OpenWeatherMap API
# =============================================================

import os
import requests
from typing import Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import OPENWEATHER_API_KEY

# OpenWeatherMap API base URL
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
ONECALL_URL = "https://api.openweathermap.org/data/2.5/onecall"

# Disaster-relevant weather thresholds
THRESHOLDS = {
    "extreme_wind_kmh": 90,       # Hurricane-force winds
    "heavy_rain_mm": 50,          # Heavy rainfall (per hour)
    "extreme_temp_high": 45,      # Extreme heat (°C)
    "extreme_temp_low": -20,      # Extreme cold (°C)
    "low_visibility_m": 200,      # Dangerous low visibility
}


def fetch_weather(city: str = None,
                  lat: float = None,
                  lon: float = None) -> Optional[Dict]:
    """
    Fetch current weather data from OpenWeatherMap.

    Args:
        city: City name (e.g., "Mumbai") 
        lat: Latitude coordinate
        lon: Longitude coordinate

    Returns:
        Parsed weather dict or None on failure
    """
    if not OPENWEATHER_API_KEY:
        return _mock_weather_data(city or "Unknown Location")

    params = {
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",  # Celsius
    }

    if city:
        params["q"] = city
    elif lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    else:
        return None

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return _parse_weather_response(data)
    except requests.exceptions.ConnectionError:
        return _mock_weather_data(city or f"{lat},{lon}")
    except requests.exceptions.Timeout:
        return _mock_weather_data(city or "Unknown")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return None  # City not found
        return _mock_weather_data(city or "Unknown")
    except Exception:
        return _mock_weather_data(city or "Unknown")


def _parse_weather_response(data: Dict) -> Dict:
    """
    Parse raw OpenWeatherMap API response into clean dict.

    Args:
        data: Raw API response JSON

    Returns:
        Clean weather dictionary
    """
    main = data.get("main", {})
    weather = data.get("weather", [{}])[0]
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})
    visibility = data.get("visibility", None)

    temp = main.get("temp", None)
    wind_speed = wind.get("speed", 0)
    wind_speed_kmh = round(wind_speed * 3.6, 1) if wind_speed else 0

    parsed = {
        "city": data.get("name", "Unknown"),
        "country": data.get("sys", {}).get("country", ""),
        "temp": round(temp, 1) if temp is not None else None,
        "feels_like": round(main.get("feels_like", temp), 1) if temp else None,
        "temp_min": round(main.get("temp_min", temp), 1) if temp else None,
        "temp_max": round(main.get("temp_max", temp), 1) if temp else None,
        "humidity": main.get("humidity", None),
        "pressure": main.get("pressure", None),
        "description": weather.get("description", "Unknown").title(),
        "icon": weather.get("icon", "01d"),
        "wind_speed": wind_speed,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_direction": wind.get("deg", None),
        "cloud_coverage": clouds.get("all", 0),
        "visibility_m": visibility,
        "alerts": [],
        "is_mock": False,
    }

    # Add disaster alerts based on conditions
    parsed["alerts"] = _generate_weather_alerts(parsed)

    return parsed


def _generate_weather_alerts(weather: Dict) -> list:
    """
    Generate disaster-relevant alerts from weather conditions.

    Args:
        weather: Parsed weather dict

    Returns:
        List of alert message strings
    """
    alerts = []
    wind_kmh = weather.get("wind_speed_kmh", 0)
    temp = weather.get("temp")
    visibility = weather.get("visibility_m")
    desc = weather.get("description", "").lower()

    if wind_kmh >= THRESHOLDS["extreme_wind_kmh"]:
        alerts.append(
            f"⚠️ EXTREME WINDS: {wind_kmh} km/h — Hurricane/cyclone risk"
        )
    elif wind_kmh >= 60:
        alerts.append(f"⚠️ Strong winds: {wind_kmh} km/h — Take precautions")

    if temp is not None:
        if temp >= THRESHOLDS["extreme_temp_high"]:
            alerts.append(f"🌡️ EXTREME HEAT: {temp}°C — Heat stroke risk")
        elif temp >= 38:
            alerts.append(f"🌡️ High temperature: {temp}°C — Stay hydrated")
        if temp <= THRESHOLDS["extreme_temp_low"]:
            alerts.append(f"❄️ EXTREME COLD: {temp}°C — Hypothermia risk")
        elif temp <= -10:
            alerts.append(f"❄️ Very cold: {temp}°C — Frostbite risk")

    if visibility is not None and visibility <= THRESHOLDS["low_visibility_m"]:
        alerts.append(
            f"🌫️ DANGEROUS VISIBILITY: {visibility}m — Travel not recommended"
        )

    if any(kw in desc for kw in ["thunderstorm", "lightning"]):
        alerts.append("⚡ THUNDERSTORM WARNING — Seek shelter immediately")
    if "snow" in desc or "blizzard" in desc:
        alerts.append("❄️ SNOWSTORM — Road travel may be dangerous")
    if "fog" in desc:
        alerts.append("🌫️ Foggy conditions — Reduced visibility on roads")

    return alerts


def _mock_weather_data(location: str) -> Dict:
    """
    Return mock weather data when API is unavailable.
    Used for demo/testing purposes.
    """
    return {
        "city": location or "Demo City",
        "country": "US",
        "temp": 28.5,
        "feels_like": 31.2,
        "temp_min": 24.0,
        "temp_max": 32.0,
        "humidity": 72,
        "pressure": 1013,
        "description": "Partly Cloudy",
        "icon": "02d",
        "wind_speed": 5.2,
        "wind_speed_kmh": 18.7,
        "wind_direction": 180,
        "cloud_coverage": 35,
        "visibility_m": 10000,
        "alerts": [],
        "is_mock": True,
    }


def get_weather_risk_level(weather: Dict) -> str:
    """
    Assess overall weather risk level for emergency planning.

    Args:
        weather: Parsed weather dict

    Returns:
        Risk level: CRITICAL, HIGH, MEDIUM, or LOW
    """
    if not weather:
        return "UNKNOWN"

    alerts = weather.get("alerts", [])
    wind_kmh = weather.get("wind_speed_kmh", 0)
    temp = weather.get("temp", 20)

    # Critical conditions
    if (wind_kmh >= 90 or
            (temp and (temp >= 45 or temp <= -20)) or
            any("EXTREME" in a for a in alerts)):
        return "CRITICAL"

    # High risk conditions
    if (wind_kmh >= 60 or
            (temp and (temp >= 38 or temp <= -10)) or
            len(alerts) >= 2):
        return "HIGH"

    # Medium risk
    if wind_kmh >= 30 or len(alerts) >= 1:
        return "MEDIUM"

    return "LOW"
