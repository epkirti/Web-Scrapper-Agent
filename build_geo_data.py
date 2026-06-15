"""Build the offline India geo bundle used by the Area Explorer's region cascade.

Input : an All-India PIN-code CSV with columns OfficeName, District, StateName,
        Latitude, Longitude (e.g. the data.gov.in directory mirrored on GitHub).
Output: data/india_geo.json.gz — a compact State → District → Area tree with
        coordinates, so the cascade (and its map pin) works fully offline with no
        API key. Re-run this only when refreshing the dataset.

Usage:  python build_geo_data.py path/to/pincode.csv
"""
import csv, gzip, json, os, re, sys
from collections import defaultdict

# Trailing post-office markers in OfficeName (B.O = branch office, S.O = sub
# office, H.O = head office, G.P.O, R.S = railway station, I.E = industrial estate).
_SUFFIX = re.compile(r"\s+(B\.?O|S\.?O|H\.?O|G\.?P\.?O\.?|R\.?S|I\.?E)\.?\s*$", re.I)


def _clean_area(name: str) -> str:
    name = _SUFFIX.sub("", (name or "").strip())
    return re.sub(r"\s{2,}", " ", name).strip()


def _title_state(s: str) -> str:
    s = (s or "").strip().title().replace(" And ", " and ").replace(" Of ", " of ")
    # The CSV spells this UT with a leading "The"; drop it for a clean label.
    return {"The Dadra and Nagar Haveli and Daman and Diu":
            "Dadra and Nagar Haveli and Daman and Diu"}.get(s, s)


def _title_district(d: str) -> str:
    return (d or "").strip().title().replace(" And ", " and ")


def _valid(lat, lon):
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    return (round(la, 4), round(lo, 4)) if (6 < la < 38 and 68 < lo < 98) else None


def build(csv_path: str, out_path: str) -> None:
    # state -> district -> {area_lower: [name, lat, lon]}  (dedupe, prefer real coords)
    tree: dict = defaultdict(lambda: defaultdict(dict))
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            state = _title_state(row["StateName"])
            district = _title_district(row["District"])
            area = _clean_area(row["OfficeName"])
            if not (state and district and area):
                continue
            coord = _valid(row.get("Latitude"), row.get("Longitude"))
            areas = tree[state][district]
            key = area.lower()
            if key not in areas:
                areas[key] = [area, coord[0] if coord else None, coord[1] if coord else None]
            elif coord and areas[key][1] is None:  # upgrade a coordless entry
                areas[key][1], areas[key][2] = coord

    def centroid(points):
        pts = [(la, lo) for la, lo in points if la is not None]
        if not pts:
            return None, None
        return (round(sum(p[0] for p in pts) / len(pts), 4),
                round(sum(p[1] for p in pts) / len(pts), 4))

    out: dict = {}
    for state, districts in tree.items():
        dout, dcenters = {}, []
        for district, areas in districts.items():
            alist = sorted(([a[0], a[1], a[2]] for a in areas.values()),
                           key=lambda x: x[0].lower())
            dlat, dlon = centroid([(a[1], a[2]) for a in alist])
            dout[district] = {"lat": dlat, "lon": dlon, "areas": alist}
            dcenters.append((dlat, dlon))
        slat, slon = centroid(dcenters)
        out[state] = {"lat": slat, "lon": slon, "districts": dout}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    n_d = sum(len(v["districts"]) for v in out.values())
    n_a = sum(len(d["areas"]) for v in out.values() for d in v["districts"].values())
    print(f"states={len(out)} districts={n_d} areas={n_a}")
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB gzipped)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else ".tmpdata/pin1.csv"
    build(src, os.path.join("data", "india_geo.json.gz"))
