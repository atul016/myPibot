"""Tools: what Rocky can find out or do beyond its own senses.
  - the weather outside (Open-Meteo, no key) -- refreshed by openbot-mind every
    WEATHER_EVERY_S, so no conversation ever waits on the network for it; every
    prompt reads the cached line (journal.inject);
  - an encyclopedia lookup (Wikipedia), for a factual question it can't answer
    well itself (services/chat.py);
  - reminders it texts later ("remind me at 6 to call mom") -- kept in the
    session, sent by openbot-chat when due.
Network calls are short and fail soft: no answer is never an error anyone sees.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from urllib.parse import quote

import requests

from .state import STATE_DIR, atomic_write

LOCATION = os.environ.get("OPENBOT_LOCATION", "")  # "Boston, MA" -- with OPENBOT_LATLON; "" = no weather
LATLON = os.environ.get("OPENBOT_LATLON", "")      # "42.36,-71.06"
WEATHER_PATH = STATE_DIR / "weather.json"
WEATHER_EVERY_S = 1800
HEADERS = {"User-Agent": "OpenBot/1.0 (https://github.com/atul016/myPibot)"}  # Wikipedia asks clients to say who they are
WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
       80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 85: "snow showers", 86: "snow showers",
       95: "a thunderstorm", 96: "a thunderstorm with hail", 99: "a thunderstorm with hail"}


def now_words(now: dt.datetime | None = None) -> str:
    return f"{now or dt.datetime.now():%A, %B %-d, %-I:%M %p}"


def _f_c(f: float) -> str:
    return f"{f:.0f}°F ({(f - 32) * 5 / 9:.0f}°C)"


def weather_line(data: dict) -> str:
    """One line from Open-Meteo's answer: now, today, tomorrow."""
    cur, day = data["current"], data["daily"]
    days = [f"{name} {_f_c(day['temperature_2m_min'][i])} to {_f_c(day['temperature_2m_max'][i])}, "
            f"{WMO.get(day['weather_code'][i], 'mixed')}, {day['precipitation_probability_max'][i]}% chance of rain"
            for i, name in enumerate(("today", "tomorrow"))]
    return (f"{LOCATION} now: {_f_c(cur['temperature_2m'])}, {WMO.get(cur['weather_code'], 'mixed')}, "
            f"wind {cur['wind_speed_10m']:.0f} mph; " + "; ".join(days))


def refresh_weather() -> None:
    """Called by openbot-mind's loop: fetch at most every WEATHER_EVERY_S."""
    if not LATLON or time.time() - _cached().get("ts", 0) < WEATHER_EVERY_S:
        return
    lat, lon = LATLON.split(",")
    try:
        data = requests.get("https://api.open-meteo.com/v1/forecast", timeout=8, params={
            "latitude": lat.strip(), "longitude": lon.strip(), "timezone": "auto", "forecast_days": 2,
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"}).json()
        atomic_write(WEATHER_PATH, json.dumps({"ts": time.time(), "line": weather_line(data)}))
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        print(f"tools: weather failed: {e!r}")


def _cached() -> dict:
    try:
        return json.loads(WEATHER_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def weather() -> str | None:
    """The cached weather line, if it's fresh enough to trust (3 hours)."""
    w = _cached()
    return w.get("line") if time.time() - w.get("ts", 0) < 3 * 3600 else None


def lookup(topic: str) -> str | None:
    """"Mount Washington: Mount Washington is the highest peak..." -- Wikipedia's
    summary of the best match for `topic`, or None."""
    try:
        found = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=6, params={
            "action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"}).json()
        hits = found["query"]["search"]  # full-text: "opensearch" only matches titles starting with the words
        if not hits:
            return None
        page = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(hits[0]['title'])}",
                            headers=HEADERS, timeout=6).json()
        return f"{page['title']}: {page['extract'][:900]}"
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None


def remind_time(hhmm: str, now: dt.datetime | None = None) -> float | None:
    """"18:30" -> the next time it's 18:30 (today, or tomorrow if that's passed), as a timestamp."""
    now = now or dt.datetime.now()
    try:
        t = dt.datetime.strptime(hhmm.strip(), "%H:%M").time()
    except ValueError:
        return None
    at = dt.datetime.combine(now.date(), t)
    return (at if at > now else at + dt.timedelta(days=1)).timestamp()


def demo() -> None:
    now = dt.datetime(2026, 10, 3, 20, 15)
    assert now_words(now) == "Saturday, October 3, 8:15 PM"
    assert remind_time("21:00", now) == dt.datetime(2026, 10, 3, 21, 0).timestamp()
    assert remind_time("6:30", now) == dt.datetime(2026, 10, 4, 6, 30).timestamp()  # already past: tomorrow
    assert remind_time("soon", now) is None
    line = weather_line({"current": {"temperature_2m": 57.9, "weather_code": 0, "wind_speed_10m": 3.0},
                         "daily": {"temperature_2m_min": [50, 45], "temperature_2m_max": [70.6, 62],
                                   "weather_code": [1, 61], "precipitation_probability_max": [0, 80]}})
    assert "58°F (14°C), clear, wind 3 mph" in line and "tomorrow 45°F (7°C) to 62°F (17°C), light rain, 80%" in line, line


if __name__ == "__main__":
    demo()
    print("tools: ok")
