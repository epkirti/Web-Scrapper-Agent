"""Map-based area news — turn a tapped map point into news for that area.

Data layer only (no Streamlit here; the UI lives in ``app.py``).

Flow:
    (lat, lon)  --reverse geocode-->  area name  --news search-->  headlines

Reverse geocoding uses OpenStreetMap Nominatim (free, no key). News uses Serper's
Google News endpoint when a key is given, falling back to DuckDuckGo news.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

import httpx

_UA = {"User-Agent": "area-explorer/1.0 (research assistant)"}


# --------------------------------------------------------------------------- #
# Reverse geocoding: coordinates -> human area name
# --------------------------------------------------------------------------- #
def reverse_geocode(lat: float, lon: float, timeout: float = 10.0) -> dict:
    """Turn a tapped ``(lat, lon)`` into a place name via Nominatim (OSM).

    Returns ``{area, locality, state, country, detail, address}`` where:
      - ``area``    = best single string to search news for (locality + state)
      - ``detail``  = full human-readable display name from OSM
      - ``address`` = raw address-component dict (city/state/country/...)

    Free, no API key. Nominatim asks for <=1 req/sec and a real User-Agent — fine
    for interactive taps. Returns blanks (never raises) if the lookup fails.
    """
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "accept-language": "en",
                "zoom": 10,  # ~city/district granularity
            },
            headers=_UA,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"area": "", "locality": "", "state": "", "country": "",
                "detail": "", "address": {}, "category": "", "type": ""}

    addr = data.get("address") or {}
    # Most-specific populated place first, then broaden until we find something.
    locality = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or addr.get("county")
        or addr.get("state_district")
        or addr.get("suburb")
        or ""
    )
    state = addr.get("state") or ""
    country = addr.get("country") or ""

    if locality and state:
        area = f"{locality}, {state}"
    else:
        area = ", ".join(p for p in (locality, state, country) if p)

    return {
        "area": area or data.get("display_name", ""),
        "locality": locality,
        "state": state,
        "country": country,
        "detail": data.get("display_name", ""),
        "address": addr,
        "category": data.get("category", ""),  # e.g. 'place', 'boundary'
        "type": data.get("type", ""),          # e.g. 'city', 'town', 'administrative'
    }


# --------------------------------------------------------------------------- #
# Forward geocoding: a typed place name -> coordinates (for the search bar)
# --------------------------------------------------------------------------- #
def forward_geocode(query: str, timeout: float = 10.0) -> dict:
    """Turn a typed place name (e.g. "Indore" or "Eiffel Tower") into coordinates
    plus the same fields ``reverse_geocode`` returns (so callers treat them alike).

    Returns ``{}`` if nothing matched. Uses Nominatim's free search endpoint.
    """
    if not query.strip():
        return {}
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
                "accept-language": "en",
            },
            headers=_UA,
            timeout=timeout,
        )
        resp.raise_for_status()
        arr = resp.json()
    except Exception:
        return {}
    if not arr:
        return {}

    d = arr[0]
    addr = d.get("address") or {}
    locality = (
        addr.get("city") or addr.get("town") or addr.get("village")
        or addr.get("municipality") or addr.get("county")
        or addr.get("state_district") or addr.get("suburb")
        or d.get("name") or ""  # the matched feature's own name (e.g. "Tokyo")
    )
    state = addr.get("state") or ""
    country = addr.get("country") or ""
    if locality and state:
        area = f"{locality}, {state}"
    else:
        area = ", ".join(p for p in (locality, state, country) if p) or d.get("display_name", "")

    try:
        lat, lon = float(d["lat"]), float(d["lon"])
    except (KeyError, ValueError, TypeError):
        return {}

    return {
        "lat": lat,
        "lon": lon,
        "area": area,
        "locality": locality,
        "state": state,
        "country": country,
        "detail": d.get("display_name", ""),
        "address": addr,
        "category": d.get("category", ""),
        "type": d.get("type", ""),
    }


# --------------------------------------------------------------------------- #
# Weather (Open-Meteo — free, no API key)
# --------------------------------------------------------------------------- #
# WMO weather-interpretation codes -> human text (the common ones).
_WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Dense drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
    67: "Heavy freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm w/ hail", 99: "Severe thunderstorm w/ hail",
}


def fetch_weather(lat: float, lon: float, timeout: float = 15.0) -> dict:
    """Current weather at a point via Open-Meteo. Returns {} on failure.

    Keys: temp, feels, humidity, precip, wind (and a human ``condition``), all
    in metric units (°C, %, mm, km/h).
    """
    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        cur = resp.json().get("current") or {}
    except Exception:
        return {}
    if not cur:
        return {}
    code = cur.get("weather_code")
    return {
        "temp": cur.get("temperature_2m"),
        "feels": cur.get("apparent_temperature"),
        "humidity": cur.get("relative_humidity_2m"),
        "precip": cur.get("precipitation"),
        "wind": cur.get("wind_speed_10m"),
        "condition": _WMO.get(code, "—"),
    }


def fetch_elevation(lat: float, lon: float, timeout: float = 15.0) -> Optional[float]:
    """Ground elevation (metres above sea level) at a point via Open-Meteo."""
    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat, "longitude": lon},
            timeout=timeout,
        )
        resp.raise_for_status()
        el = resp.json().get("elevation")
        if isinstance(el, list) and el:
            return float(el[0])
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Overview (Wikipedia REST summary — free, no key)
# --------------------------------------------------------------------------- #
def fetch_wikipedia_summary(title: str, timeout: float = 15.0) -> dict:
    """Short encyclopedic summary for a place name. Returns {} if there's no
    (unambiguous) page. Keys: extract, url, thumb."""
    if not title.strip():
        return {}
    try:
        resp = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
            headers=_UA,
            timeout=timeout,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return {}
        d = resp.json()
    except Exception:
        return {}
    if d.get("type") == "disambiguation" or not d.get("extract"):
        return {}
    return {
        "extract": d.get("extract", ""),
        "url": ((d.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        "thumb": (d.get("thumbnail") or {}).get("source", ""),
    }


# --------------------------------------------------------------------------- #
# Generic topic info (any aspect of a place: history, economy, tourism, ...)
# --------------------------------------------------------------------------- #
def web_search(query: str, serper_api_key: str = "", max_results: int = 6,
               timeout: float = 20.0) -> list[dict]:
    """General web search → list of {title, url, snippet}.

    Serper's Google search when a key is given, else DuckDuckGo (no key).
    Used to gather context for the knowledge categories (history, economy, ...).
    """
    if not query.strip():
        return []
    if serper_api_key:
        try:
            resp = httpx.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
                timeout=timeout,
            )
            resp.raise_for_status()
            org = resp.json().get("organic", []) or []
            out = [
                {"title": o.get("title", ""), "url": o.get("link", ""),
                 "snippet": o.get("snippet", "")}
                for o in org if o.get("title")
            ]
            if out:
                return out[:max_results]
        except Exception:
            pass
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""),
             "url": r.get("href") or r.get("url", ""),
             "snippet": r.get("body", "")}
            for r in results if r.get("title")
        ][:max_results]
    except Exception:
        return []


def summarize_topic(groq_client, model: str, place: str, topic: str,
                    snippets: list[dict], wiki_extract: str = "") -> str:
    """Write a focused, factual summary about one ``topic`` of a ``place`` using
    ONLY the supplied web snippets (+ optional Wikipedia extract). Returns "" if
    there's nothing to work from or the call fails."""
    parts = []
    if wiki_extract:
        parts.append(f"Encyclopedia: {wiki_extract}")
    for s in snippets:
        line = f"- {s.get('title', '')}: {s.get('snippet', '')}".strip(" -:")
        if line:
            parts.append(line)
    context = "\n".join(parts)
    if not context.strip():
        return ""
    prompt = (
        f"Write a clear, factual summary about the {topic} of {place}. "
        "Use ONLY the information below — do not invent facts. Aim for 3-6 sentences. "
        "If the information is thin, say what is known and note the rest is unclear.\n\n"
        f"{context}\n\nSummary:"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Area name -> news headlines
# --------------------------------------------------------------------------- #
def fetch_area_news(
    area: str,
    serper_api_key: str = "",
    max_results: int = 10,
    timeout: float = 20.0,
) -> list[dict]:
    """Recent news for an area.

    Uses Serper's Google News endpoint when ``serper_api_key`` is set, otherwise
    (or on failure) falls back to DuckDuckGo news (no key). Each item is
    normalized to ``{title, url, source, date, snippet}``.
    """
    if not area.strip():
        return []
    query = f"{area} news"

    if serper_api_key:
        try:
            resp = httpx.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
                json={"q": query, "num": max_results},
                timeout=timeout,
            )
            resp.raise_for_status()
            items = resp.json().get("news", []) or []
            out = [
                {
                    "title": it.get("title", ""),
                    "url": it.get("link", ""),
                    "source": it.get("source", ""),
                    "date": it.get("date", ""),
                    "snippet": it.get("snippet", ""),
                }
                for it in items
                if it.get("title")
            ]
            if out:
                return out[:max_results]
        except Exception:
            pass  # fall through to DuckDuckGo

    # Fallback: DuckDuckGo news (no API key required).
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", "") or r.get("href", ""),
                "source": r.get("source", ""),
                "date": r.get("date", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
            if r.get("title")
        ][:max_results]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Optional: short AI summary of an area's headlines
# --------------------------------------------------------------------------- #
def summarize_news(groq_client, model: str, area: str, items: list[dict]) -> str:
    """A short, neutral 2-4 sentence summary built ONLY from the given headlines.

    Returns "" if there is nothing to summarize or the call fails.
    """
    if not items:
        return ""
    lines = "\n".join(
        f"- {it.get('title','')} ({it.get('source','')}, {it.get('date','')}): {it.get('snippet','')}"
        for it in items
    )
    prompt = (
        f"Below are recent news headlines about {area}. Write a short, neutral "
        "2-4 sentence summary of what is happening there. Use ONLY these headlines; "
        "do not invent any details.\n\n"
        f"{lines}\n\nSummary:"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
