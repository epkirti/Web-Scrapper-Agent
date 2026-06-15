"""Offline India geography for the Area Explorer cascade (State → District → Area).

Backed by ``data/india_geo.json.gz`` (built by ``build_geo_data.py`` from the
All-India PIN-code directory). No API key and no network needed — lists and
coordinates come straight from the bundled file. Returns empty / None gracefully
if the data file is absent, so the app still runs without it.
"""
import functools
import gzip
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "india_geo.json.gz")


@functools.lru_cache(maxsize=1)
def _data() -> dict:
    try:
        with gzip.open(_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def available() -> bool:
    """True when the bundled offline data loaded successfully."""
    return bool(_data())


def list_states() -> list:
    return sorted(_data().keys())


def list_districts(state: str) -> list:
    node = _data().get(state)
    return sorted(node["districts"].keys()) if node else []


def list_areas(state: str, district: str) -> list:
    node = (_data().get(state) or {}).get("districts", {}).get(district)
    return [a[0] for a in node["areas"]] if node else []


# Directional / generic place words that are also common words — never use these to
# match news (they'd match "north India", "central govt", "industrial area", etc.).
_GENERIC_PLACES = {
    "central", "east", "west", "north", "south", "new delhi", "north east",
    "north west", "south east", "south west", "city", "industrial area",
    "civil lines", "new colony", "main road", "station road", "bus stand",
}


@functools.lru_cache(maxsize=1)
def _area_district_count() -> dict:
    """How many distinct districts each (lowercased) locality name appears in. A name
    seen in just one district is distinctive enough to identify that district."""
    count: dict = {}
    for node in _data().values():
        for dn in node.get("districts", {}).values():
            for a in dn.get("areas", []):
                nm = a[0].lower()
                count[nm] = count.get(nm, 0) + 1
    return count


@functools.lru_cache(maxsize=256)
def local_keep_terms(state: str, district: str) -> frozenset:
    """Lowercased terms that mark a news item as belonging to ``district``: the
    district/city name plus its localities that are UNIQUE to it (appear in no other
    district) and aren't short/generic. Generic and data-driven — used to pin
    area news to the searched place without any hand-maintained landmark list."""
    terms = set()
    if district:
        terms.add(district.lower())
    node = (_data().get(state) or {}).get("districts", {}).get(district)
    if node:
        count = _area_district_count()
        for a in node.get("areas", []):
            nm = a[0].lower()
            if len(nm) >= 5 and nm not in _GENERIC_PLACES and count.get(nm, 0) == 1:
                terms.add(nm)
    return frozenset(terms)


def _coord(node: dict):
    la, lo = node.get("lat"), node.get("lon")
    return (la, lo) if la is not None else None


def locate(state: str, district: str = "", area: str = ""):
    """Coordinates of the deepest given level (area → district → state), falling
    back up the chain when a level has no stored coordinate. Returns None if the
    state is unknown / data is missing."""
    node = _data().get(state)
    if not node:
        return None
    if district:
        dnode = (node.get("districts") or {}).get(district)
        if not dnode:
            return _coord(node)
        if area:
            dc = _coord(dnode)
            for name, la, lo in dnode["areas"]:
                if name == area:
                    # Some PIN-code rows carry a mislabeled coordinate (a same-named
                    # locality elsewhere). If it's far from the district centroid,
                    # trust the centroid instead of pinning the wrong place.
                    if la is not None and (dc is None or
                                           (abs(la - dc[0]) < 1.5 and abs(lo - dc[1]) < 1.5)):
                        return (la, lo)
                    break
            return dc  # area has no / an implausible coord -> district centroid
        return _coord(dnode)
    return _coord(node)
