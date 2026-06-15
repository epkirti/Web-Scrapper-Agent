"""Map-based area news — turn a tapped map point into news for that area.

Data layer only (no Streamlit here; the UI lives in ``app.py``).

Flow:
    (lat, lon)  --reverse geocode-->  area name  --news search-->  headlines

Reverse geocoding uses OpenStreetMap Nominatim (free, no key). News uses Serper's
Google News endpoint when a key is given, falling back to DuckDuckGo news.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Optional
from urllib.parse import quote

import httpx
from huggingface_hub import InferenceClient

try:  # optional; only used to sort news by date (handles "2 days ago" etc.)
    import dateparser as _dateparser
except Exception:  # pragma: no cover - falls back to ISO-only parsing
    _dateparser = None

_UA = {"User-Agent": "area-explorer/1.0 (research assistant)"}


# --------------------------------------------------------------------------- #
# News date parsing, de-duplication and recency sorting
# --------------------------------------------------------------------------- #
def _parse_news_dt(raw: str) -> Optional[_dt.datetime]:
    """Best-effort parse of a news date into a naive datetime for sorting.
    Handles ISO 8601 (DuckDuckGo) and relative/absolute text (Serper: "2 days
    ago", "Jun 10, 2024"). Returns None when it can't be parsed."""
    s = (raw or "").strip()
    if not s:
        return None
    try:  # fast path: ISO 8601
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    if _dateparser is not None:
        try:
            dt = _dateparser.parse(s, settings={"RETURN_AS_TIMEZONE_AWARE": False})
            if dt:
                return dt.replace(tzinfo=None)
        except Exception:
            pass
    return None


def _dedupe_sort_news(items: list[dict]) -> list[dict]:
    """Drop duplicate stories and order them latest-first.

    Dedup is by URL first, then by a normalized title (case/punctuation-insensitive)
    so the same story from one source isn't repeated. Each item's ``date`` is
    rewritten to a tidy "10 Jun 2024" when parseable; undated items sort last."""
    seen_url, seen_title, out = set(), set(), []
    for it in items:
        url = (it.get("url") or "").strip().rstrip("/").lower()
        norm = re.sub(r"[^a-z0-9]+", " ", (it.get("title") or "").lower()).strip()
        if (url and url in seen_url) or (norm and norm in seen_title):
            continue
        if url:
            seen_url.add(url)
        if norm:
            seen_title.add(norm)
        dt = _parse_news_dt(it.get("date", ""))
        new = dict(it)
        new["_dt"] = dt
        if dt is not None:
            new["date"] = dt.strftime("%d %b %Y")
        out.append(new)
    # Dated items first (newest → oldest), then undated ones.
    out.sort(key=lambda x: (x["_dt"] is not None, x["_dt"] or _dt.datetime.min),
             reverse=True)
    for it in out:
        it.pop("_dt", None)
    return out


# --------------------------------------------------------------------------- #
# LLM backend — Hugging Face Inference API, exposed as a Groq-compatible client
# --------------------------------------------------------------------------- #
# Model notes for the HF Inference API:
#  - meta-llama/Llama-2-7b-chat-hf is NOT served by any provider -> StopIteration.
#  - Gated models (meta-llama/*) need their license accepted on HF, else 403
#    (HfHubHTTPError) at inference time.
#  - Qwen/Qwen2.5-7B-Instruct is OPEN (no gate) and broadly served -> safe default.
# Overridable from the app sidebar.
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class _HFMessage:
    def __init__(self, content: str):
        self.content = content


class _HFChoice:
    def __init__(self, content: str):
        self.message = _HFMessage(content)


class _HFResponse:
    def __init__(self, content: str):
        self.choices = [_HFChoice(content)]


class _HFCompletions:
    def __init__(self, client: "InferenceClient", model: str):
        self._client, self._model = client, model

    def create(self, *, messages, model=None, temperature=0.7, max_tokens=512, **_):
        out = self._client.chat_completion(
            messages=messages,
            model=self._model,
            temperature=max(float(temperature or 0.0), 0.01),  # HF rejects temp=0
            max_tokens=int(max_tokens or 512),
        )
        return _HFResponse(out.choices[0].message.content or "")


class _HFChat:
    def __init__(self, client: "InferenceClient", model: str):
        self.completions = _HFCompletions(client, model)


class HFChatClient:
    """Groq/OpenAI-compatible chat client backed by the Hugging Face Inference API.

    Exposes ``.chat.completions.create(messages=..., temperature=..., max_tokens=...)``
    so existing Groq call sites keep working unchanged. The per-call ``model`` is
    ignored — the configured HF model (default ``meta-llama/Llama-2-7b-chat-hf``) is
    used. Needs an HF API token with access to the (gated) model.
    """

    def __init__(self, token: str, model: str = HF_MODEL):
        self._client = InferenceClient(token=token or None)
        self.chat = _HFChat(self._client, model)


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


# US AQI bands -> human label (Open-Meteo returns the US AQI scale).
def _aqi_label(aqi) -> str:
    if aqi is None:
        return ""
    try:
        a = float(aqi)
    except (TypeError, ValueError):
        return ""
    if a <= 50:
        return "Good"
    if a <= 100:
        return "Moderate"
    if a <= 150:
        return "Unhealthy for sensitive groups"
    if a <= 200:
        return "Unhealthy"
    if a <= 300:
        return "Very unhealthy"
    return "Hazardous"


def fetch_air_quality(lat: float, lon: float, timeout: float = 15.0) -> dict:
    """Current air quality (US AQI + PM2.5/PM10) via Open-Meteo. {} on failure."""
    try:
        resp = httpx.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat, "longitude": lon,
                "current": "us_aqi,pm2_5,pm10",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        cur = resp.json().get("current") or {}
    except Exception:
        return {}
    if not cur:
        return {}
    aqi = cur.get("us_aqi")
    return {"aqi": aqi, "label": _aqi_label(aqi),
            "pm2_5": cur.get("pm2_5"), "pm10": cur.get("pm10")}


def fetch_astro(lat: float, lon: float, timeout: float = 15.0) -> dict:
    """Local timezone + today's sunrise/sunset via Open-Meteo. {} on failure.

    Returns ``{timezone, sunrise, sunset}`` where sunrise/sunset are local
    ``HH:MM`` strings.
    """
    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "sunrise,sunset", "timezone": "auto", "forecast_days": 1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        j = resp.json()
    except Exception:
        return {}
    daily = j.get("daily") or {}

    def _hhmm(values):
        if isinstance(values, list) and values and isinstance(values[0], str) and "T" in values[0]:
            return values[0].split("T", 1)[1][:5]
        return ""

    return {
        "timezone": j.get("timezone", ""),
        "sunrise": _hhmm(daily.get("sunrise")),
        "sunset": _hhmm(daily.get("sunset")),
    }


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
                    snippets: list[dict], wiki_extract: str = "",
                    lang: str = "English") -> str:
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
        "If the information is thin, say what is known and note the rest is unclear.\n"
        f"Write the summary in {lang}.\n\n"
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
    time_filter: str = "",
) -> list[dict]:
    """Recent news for an area.

    Uses Serper's Google News endpoint when ``serper_api_key`` is set, otherwise
    (or on failure) falls back to DuckDuckGo news (no key). Each item is
    normalized to ``{title, url, source, date, snippet}``.

    ``time_filter`` limits recency: "" (any time), "d" (24h), "w" (week),
    "m" (month).
    """
    if not area.strip():
        return []
    query = f"{area} news"

    if serper_api_key:
        try:
            payload = {"q": query, "num": max_results}
            if time_filter in ("d", "w", "m"):
                payload["tbs"] = f"qdr:{time_filter}"
            resp = httpx.post(
                "https://google.serper.dev/news",
                headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
                json=payload,
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
            out = _dedupe_sort_news(out)
            if out:
                return out[:max_results]
        except Exception:
            pass  # fall through to DuckDuckGo

    # Fallback: DuckDuckGo news (no API key required).
    try:
        from ddgs import DDGS

        kw = {"max_results": max_results}
        if time_filter in ("d", "w", "m"):
            kw["timelimit"] = time_filter
        with DDGS() as ddgs:
            results = list(ddgs.news(query, **kw))
        out = _dedupe_sort_news([
            {
                "title": r.get("title", ""),
                "url": r.get("url", "") or r.get("href", ""),
                "source": r.get("source", ""),
                "date": r.get("date", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
            if r.get("title")
        ])
        return out[:max_results]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Optional: short AI summary of an area's headlines
# --------------------------------------------------------------------------- #
def _parse_name_list(raw: str, exclude: str, n: int) -> list[str]:
    """Turn an LLM's comma/newline list of place names into a clean unique list,
    dropping numbering, boilerplate prefixes and the parent place itself."""
    out, seen = [], set()
    for part in raw.replace("\n", ",").split(","):
        # Drop any leading list markers / numbering (e.g. "1.", "-", "•").
        name = part.strip().lstrip("0123456789.)-•*# \t").strip()
        low = name.lower()
        if (
            name and low not in seen and 1 < len(name) < 40
            and not low.startswith(("here", "sure", "the city", "the state",
                                    "some ", "okay", "i "))
            and exclude.strip().lower() != low
        ):
            seen.add(low)
            out.append(name)
    return out[:n]


def _ask_name_list(client, model: str, prompt: str, exclude: str, n: int) -> list[str]:
    """Run one LLM call and parse its reply into a clean place-name list."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        return []
    return _parse_name_list(raw, exclude, n)


def list_localities(client, model: str, city: str, n: int = 12) -> list[str]:
    """Ask the LLM for the well-known neighbourhoods/localities of a city, so the
    user can get news for a specific area (e.g. Bhawarkua, Vijay Nagar in Indore)
    even though map data doesn't reliably name them. Returns [] on failure."""
    if not city.strip():
        return []
    prompt = (
        f"List up to {n} well-known neighbourhoods, localities or areas WITHIN the "
        f"city of {city}. Return ONLY a comma-separated list of the area names — no "
        "numbering, no description, no extra words. If the place is not a city with "
        "distinct named localities, return nothing."
    )
    return _ask_name_list(client, model, prompt, city, n)


def list_cities(client, model: str, state: str, n: int = 12) -> list[str]:
    """Ask the LLM for the major cities/towns within a state or region, so the user
    can drill the area picker State → City → locality. Returns [] on failure."""
    if not state.strip():
        return []
    prompt = (
        f"List up to {n} of the largest and most well-known cities or towns WITHIN "
        f"the state/region of {state}. Return ONLY a comma-separated list of the "
        "city names — no numbering, no description, no extra words. If it is not a "
        "state or region with distinct cities, return nothing."
    )
    return _ask_name_list(client, model, prompt, state, n)


def summarize_news(groq_client, model: str, area: str, items: list[dict],
                   lang: str = "English") -> str:
    """A concise digest of the area's news, consolidating many publications into
    one place. Returns "" if there is nothing to summarize or the call fails."""
    if not items:
        return ""
    lines = "\n".join(
        f"- {it.get('title','')} ({it.get('source','')}, {it.get('date','')}): {it.get('snippet','')}"
        for it in items
    )
    prompt = (
        f"You are a news editor. Below are headlines about {area} from many "
        "publications, already ordered newest-first. Write a CONCISE digest (3-6 "
        "short bullet points) that consolidates them, so the reader does not have "
        "to open every article. Rules:\n"
        "- Each bullet must be a DISTINCT story — never repeat the same development, "
        "even if several headlines cover it; merge duplicates into one bullet.\n"
        "- Order bullets from most recent to older; include the date when known.\n"
        "- Lead with the most important/recent item; keep it neutral and factual.\n"
        "- Use ONLY these headlines — do not invent details.\n"
        f"Write the digest in {lang}.\n\n"
        f"{lines}\n\nConcise news digest:"
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
