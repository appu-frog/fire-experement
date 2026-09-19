"""Demo API over a prepared GeoJSON catalogue.

Run: uvicorn service.app:app --host 0.0.0.0 --port 8000
Set FIRE_CATALOGUE to a GeoJSON FeatureCollection with `observed_at`,
`kind` (fire_point/burn_polygon), `severity_class`, and `area_ha` fields.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from shapely.geometry import box, shape

app = FastAPI(title="Fire Monitoring API", version="1.0")


class Query(BaseModel):
    date_from: str | None = None
    date_to: str | None = None
    polygon: dict | None = None
    bbox: list[float] | None = None


def catalogue() -> list[dict]:
    path = Path(os.getenv("FIRE_CATALOGUE", "service/catalogue.geojson"))
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("features", [])


def filtered(query: Query, kind: str | None = None) -> list[dict]:
    features = catalogue()
    result = []
    for feature in features:
        properties = feature.get("properties", {})
        observed = str(properties.get("observed_at", ""))
        if kind and properties.get("kind") != kind:
            continue
        if query.date_from and observed < query.date_from:
            continue
        if query.date_to and observed > query.date_to:
            continue
        geometry = feature.get("geometry")
        if geometry and query.bbox:
            if len(query.bbox) != 4:
                continue
            if not shape(geometry).intersects(box(*query.bbox)):
                continue
        if geometry and query.polygon and not shape(geometry).intersects(shape(query.polygon)):
            continue
        result.append(feature)
    return result


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return "<h1>Fire Monitoring API</h1><p>Use /docs for REST API documentation.</p>"


@app.post("/api/v1/query")
def query(payload: Query) -> dict:
    features = filtered(payload)
    return {"type": "FeatureCollection", "features": features}


@app.post("/api/v1/fire-points")
def fire_points(payload: Query) -> dict:
    return {"type": "FeatureCollection", "features": filtered(payload, "fire_point")}


@app.post("/api/v1/burn-polygons")
def burn_polygons(payload: Query) -> dict:
    return {"type": "FeatureCollection", "features": filtered(payload, "burn_polygon")}


@app.post("/api/v1/report")
def report(payload: Query) -> dict:
    features = filtered(payload, "burn_polygon")
    areas = {str(c): 0.0 for c in (1, 2, 3)}
    for feature in features:
        p = feature.get("properties", {})
        c = str(p.get("severity_class", ""))
        if c in areas:
            areas[c] += float(p.get("area_ha", 0.0))
    return {"area_total_ha": sum(areas.values()), "area_by_severity_ha": areas}
