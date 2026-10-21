"""Geography helpers: distances, county lookup and the simplified county map."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from .config import CONFIG_DIR, RAW_DIR

EARTH_KM = 6371.0088

# NDMA classes the 23 arid and semi-arid (ASAL) counties; the arid ones are mainly pastoral.
ARID = {"Turkana", "Marsabit", "Mandera", "Wajir", "Garissa", "Isiolo", "Samburu", "Tana River", "Baringo"}
SEMI_ARID = {"West Pokot", "Kitui", "Makueni", "Kajiado", "Narok", "Laikipia", "Lamu", "Kilifi", "Kwale",
             "Taita Taveta", "Embu", "Meru", "Tharaka-Nithi", "Nyeri"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def asal_class(county: str) -> str:
    if county in ARID:
        return "arid"
    if county in SEMI_ARID:
        return "semi-arid"
    return "non-ASAL"


def _rings(geometry: dict) -> list[list[list[float]]]:
    """Outer rings of a Polygon or MultiPolygon (holes are ignored for county lookup)."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    return [poly[0] for poly in geometry["coordinates"]]


def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker line simplification."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    norm = np.hypot(*seg)
    if norm == 0:
        dists = np.hypot(*(points - start).T)
    else:
        dists = np.abs(seg[0] * (start[1] - points[:, 1]) - seg[1] * (start[0] - points[:, 0])) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > tol:
        left = _simplify(points[: idx + 1], tol)
        right = _simplify(points[idx:], tol)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def build_county_assets(source=RAW_DIR / "ken_admin1.geojson", tol: float = 0.01) -> pd.DataFrame:
    """Write config/counties.csv and a light config/ken_counties.geojson from the HDX COD boundaries."""
    data = json.loads(source.read_text(encoding="utf-8"))
    rows, features = [], []
    for feat in data["features"]:
        p = feat["properties"]
        name, pcode = p["adm1_name"], p["adm1_pcode"]
        rows.append({
            "pcode": pcode, "county": name, "center_lat": round(p["center_lat"], 4),
            "center_lon": round(p["center_lon"], 4), "area_sqkm": round(p["area_sqkm"], 1),
            "asal": asal_class(name),
        })
        polys = []
        for ring in _rings(feat["geometry"]):
            simple = _simplify(np.asarray(ring, dtype=float), tol)
            if len(simple) >= 4:
                # d3-geo (Plotly geo maps) needs clockwise exterior rings; a counter-clockwise
                # ring would be drawn as "the whole globe except this county".
                x, y = simple[:, 0], simple[:, 1]
                if np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) > 0:
                    simple = simple[::-1]
                polys.append([np.round(simple, 4).tolist()])
        features.append({
            "type": "Feature", "id": pcode,
            "properties": {"pcode": pcode, "county": name},
            "geometry": {"type": "MultiPolygon", "coordinates": polys},
        })
    counties = pd.DataFrame(rows).sort_values("pcode").reset_index(drop=True)
    counties.to_csv(CONFIG_DIR / "counties.csv", index=False)
    (CONFIG_DIR / "ken_counties.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")), encoding="utf-8")
    return counties


@lru_cache(maxsize=1)
def counties() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "counties.csv")


@lru_cache(maxsize=1)
def county_geojson() -> dict:
    return json.loads((CONFIG_DIR / "ken_counties.geojson").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _county_paths() -> list[tuple[str, list[MplPath]]]:
    out = []
    for feat in county_geojson()["features"]:
        paths = [MplPath(np.asarray(poly[0])) for poly in feat["geometry"]["coordinates"]]
        out.append((feat["properties"]["pcode"], paths))
    return out


def locate_county(lat: float, lon: float) -> str | None:
    """County pcode containing the point; falls back to the nearest county centre within 60 km."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    for pcode, paths in _county_paths():
        if any(p.contains_point((lon, lat)) for p in paths):
            return pcode
    c = counties()
    d = haversine_km(lat, lon, c.center_lat.values, c.center_lon.values)
    i = int(np.argmin(d))
    return c.pcode.iloc[i] if d[i] < 60 else None


@lru_cache(maxsize=1)
def border_crossings() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "border_crossings.csv")


def distance_to_border_km(lat: float, lon: float) -> float:
    b = border_crossings()
    return float(np.min(haversine_km(lat, lon, b.latitude.values, b.longitude.values)))
