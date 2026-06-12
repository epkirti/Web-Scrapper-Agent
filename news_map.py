"""Map-based area news — turn a tapped map point into news for that area.

Data layer only (no Streamlit here; the UI lives in ``app.py``).

Flow:
    (lat, lon)  --reverse geocode-->  area name  --news search-->  headlines

Reverse geocoding uses OpenStreetMap Nominatim (free, no key). News uses Serper's
Google News endpoint when a key is given, falling back to DuckDuckGo news.
"""

from __future__ import annotations

from typing import Optional

import httpx


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
            headers={"User-Agent": "web-research-news-map/1.0 (research assistant)"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"area": "", "locality": "", "state": "", "country": "", "detail": "", "address": {}}

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
    }


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
