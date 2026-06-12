"""Geography-for-students layer on top of the agentic web-research pipeline.

Flow: a clicked map coordinate (lat/lon) -> reverse-geocode to a real place ->
ask the existing ``ResearchAgent`` about that place's geography, phrased for a
class 6-10 student -> a short, cited, grade-appropriate explainer.

This reuses ``scraper.ResearchAgent`` completely unchanged. We only add two thin
pieces around it:
  1. ``reverse_geocode`` — turn coordinates into a readable "City, State, Country"
     using OpenStreetMap's Nominatim (free, no API key; 1 req/sec policy).
  2. ``build_student_query`` — shape the question so the pipeline's normal cited
     answer comes out at the right reading level and stays kid-safe.

Everything downstream (search, scrape, hybrid retrieval, reranking, citations,
fact-check) is the pipeline you already have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from groq import InternalServerError, RateLimitError

from news_map import web_search


def groq_complete(client, *, retries: int = 4, base_delay: float = 2.0, **kwargs):
    """``client.chat.completions.create`` with exponential backoff on Groq rate
    limits / 5xx, so the student UI doesn't fail on a transient RateLimitError."""
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, InternalServerError):
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# The physical / environmental layers scientists use to analyse a region. Each
# answer is organised under these so students see how the layers interact. The
# sub-elements are phrased plainly; the grade band (below) decides how much of the
# technical vocabulary is explained.
_GEO_LAYERS = (
    ("Climate & Weather",
     "temperature (seasonal averages and extremes), sunlight/solar radiation, "
     "humidity, and wind patterns"),
    ("Terrain & Landforms (geomorphology)",
     "elevation, slope, aspect (the direction a slope faces), and topography "
     "(mountains, valleys, plains and plateaus)"),
    ("Soil & Geology (edaphic factors)",
     "soil type (texture and nutrients), the underlying bedrock (parent material), "
     "and how easily water drains through it (permeability)"),
    ("Water Systems (hydrology)",
     "surface water (rivers, lakes, wetlands, seas), groundwater/aquifers, and "
     "drainage basins"),
    ("Natural Disturbances",
     "wildfires, flooding, and erosion that reshape the land over time"),
)


@dataclass
class PlaceInfo:
    """A clicked coordinate resolved to a human-readable place."""

    lat: float
    lon: float
    name: str                       # e.g. "Bhopal, Madhya Pradesh, India"
    raw: dict = field(default_factory=dict)
    is_water: bool = False          # click landed on ocean / unnamed area


def _reverse_geocoder(user_agent: str = "geo-edu-classroom"):
    """A Nominatim reverse-geocoder, rate-limited to respect the 1 req/sec policy."""
    geocoder = Nominatim(user_agent=user_agent, timeout=10)
    return RateLimiter(geocoder.reverse, min_delay_seconds=1.0)


def reverse_geocode(lat: float, lon: float, *, language: str = "en") -> PlaceInfo:
    """Turn a clicked coordinate into a readable place via OpenStreetMap Nominatim.

    ``language='en'`` keeps names English; pass ``'hi'`` for Hindi labels. Ocean /
    empty clicks return ``is_water=True`` with a coordinate-based name so the
    pipeline can still answer "what sea/region is this?".
    """
    reverse = _reverse_geocoder()
    loc = reverse((lat, lon), language=language, zoom=10, addressdetails=True)
    if loc is None:
        return PlaceInfo(
            lat=lat, lon=lon,
            name=f"the area near {lat:.3f}, {lon:.3f}",
            is_water=True,
        )
    addr = (loc.raw or {}).get("address", {})
    locality = (
        addr.get("city") or addr.get("town") or addr.get("village")
        or addr.get("county") or addr.get("state_district") or addr.get("suburb") or ""
    )
    state = addr.get("state") or ""
    country = addr.get("country") or ""
    # Keep names short and student-friendly: "Bhopal, Madhya Pradesh".
    if locality and state:
        name = f"{locality}, {state}"
    else:
        name = ", ".join(p for p in (locality, state, country) if p) or loc.address
    return PlaceInfo(lat=lat, lon=lon, name=name, raw=loc.raw or {})


def build_student_query(place: PlaceInfo, grade: int) -> str:
    """A geography question about ``place``, phrased for a class ``grade`` student.

    The reading level is folded into the question itself, so the pipeline's normal
    cited answer node produces grade-appropriate text without any change to
    ``scraper.py``.
    """
    layers = "\n".join(f"  - {title}: {detail}" for title, detail in _GEO_LAYERS)
    band = (
        "very simple words and short sentences, and explain any hard word in brackets"
        if grade <= 7
        else "clear, simple language, briefly explaining technical terms when first used"
    )
    return (
        f"Give a geography profile of {place.name} for a class {grade} school student "
        f"in India. Start by saying where it is located (state/region and country). "
        f"Then describe it under these labelled sections, showing how the layers "
        f"interact (skip a section only if there is genuinely no information for it):\n"
        f"{layers}\n"
        f"Finish with one or two interesting geography facts. Use {band}. Keep it "
        f"factual and suitable for an 11-16 year old: avoid politics, violence, "
        f"religious disputes, and any casualty or disaster-death details."
    )


def synthesize_geography(
    client,
    model: str,
    place: PlaceInfo,
    grade: int,
    *,
    serper_api_key: str = "",
    max_results: int = 8,
) -> dict:
    """FAST path: web-search snippets -> Groq writes a grade-tailored, layered,
    cited geography explainer. No scraping / embeddings / FAISS, so it answers in
    roughly one search + one LLM call instead of the full LangGraph pipeline.

    Search uses Serper (Google) when ``serper_api_key`` is set and otherwise falls
    back to DuckDuckGo automatically (``news_map.web_search``). Returns
    ``{answer, sources, snippets, student_question}``.
    """
    queries = [
        f"{place.name} geography climate rivers landforms",
        f"{place.name} physical features soil terrain elevation natural resources",
    ]
    seen: set[str] = set()
    snippets: list[dict] = []
    for q in queries:
        for s in web_search(q, serper_api_key=serper_api_key, max_results=max_results):
            url = s.get("url", "")
            if url and url not in seen:
                seen.add(url)
                snippets.append(s)

    question = build_student_query(place, grade)
    snippets = snippets[:8]  # keep the prompt small -> faster, fewer rate limits
    blocks = [
        f"[{i}] {s.get('title', '')}: {(s.get('snippet', '') or '')[:280]}"
        for i, s in enumerate(snippets, 1)
    ]
    context = "\n".join(blocks) if blocks else "(no search results)"
    band = (
        "very simple words and short sentences, and explain any hard word in brackets"
        if grade <= 7
        else "clear, simple language, explaining a technical term the first time you use it"
    )
    prompt = (
        f"You are a friendly geography teacher writing for a class {grade} student in "
        f"India. Write a short, engaging geography profile of {place.name}.\n"
        f"Structure:\n"
        f"- Start with ONE sentence on where it is and what it is.\n"
        f"- Then describe its geography warmly, using short **bold** mini-headings ONLY "
        f"for aspects you can actually describe — choose from: Location, Landscape & "
        f"Landforms, Climate & Weather, Soil, Water (rivers / lakes / sea), Natural "
        f"Events (like floods or erosion).\n"
        f"- End with 2-3 fun facts a student would enjoy.\n\n"
        f"RULES:\n"
        f"- Prefer the numbered SOURCES; put a citation like [1] after a fact taken "
        f"from them.\n"
        f"- You MAY add basic, well-known geography to keep it clear and complete, but "
        f"do NOT invent specific numbers, dates or named claims.\n"
        f"- NEVER write that information is missing, unavailable, or not in the sources "
        f"— simply leave out what you do not know, and skip any mini-heading you cannot "
        f"fill with real content.\n"
        f"- Write {band}. Keep it suitable for an 11-16 year old: avoid politics, "
        f"violence, religious disputes and casualty details.\n\n"
        f"SOURCES:\n{context}\n\nWrite the profile:"
    )
    answer = ""
    if snippets:
        try:
            resp = groq_complete(
                client, model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=900,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception:
            answer = ""

    if not answer:
        # Fallback: write from the model's own geography knowledge so the student
        # always gets a profile, even if search/synthesis was thin or rate-limited.
        fb = (
            f"Write a short, friendly geography profile of {place.name} for a class "
            f"{grade} student in India. Cover where it is, its landscape and landforms, "
            f"its climate, and nearby water (rivers, lakes or sea), then end with 2 fun "
            f"facts. Use {band}. Keep it factual and simple; avoid politics, violence "
            f"and casualty details."
        )
        try:
            resp = groq_complete(
                client, model=model,
                messages=[{"role": "user", "content": fb}],
                temperature=0.4, max_tokens=700,
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception:
            answer = ""

    return {
        "answer": answer,
        "sources": [s["url"] for s in snippets if s.get("url")],
        "snippets": snippets,
        "student_question": question,
    }
