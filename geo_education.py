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

from dataclasses import dataclass, field
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from scraper import ResearchAgent


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
    parts = [
        addr.get("city") or addr.get("town") or addr.get("village")
        or addr.get("county") or addr.get("suburb"),
        addr.get("state"),
        addr.get("country"),
    ]
    name = ", ".join(p for p in parts if p) or loc.address
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


def explain_location(
    agent: ResearchAgent,
    lat: float,
    lon: float,
    grade: int,
    *,
    language: str = "en",
) -> dict:
    """End-to-end (blocking): coordinate -> place -> grade-tailored research dict.

    Returns the pipeline's merged final state (``answer``, ``urls``,
    ``confidence``, ``retrieved_chunks``, ...) plus ``place``, ``grade`` and the
    ``student_question`` that was asked. For live UI progress, call
    ``reverse_geocode`` + ``build_student_query`` yourself and drive
    ``agent.stream_research(question)`` (see geo_app.py).
    """
    place = reverse_geocode(lat, lon, language=language)
    question = build_student_query(place, grade)
    result = agent.research(question)
    result["place"] = place
    result["grade"] = grade
    result["student_question"] = question
    return result
