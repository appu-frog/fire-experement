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
from pyproj import Geod

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


def query_geometry(query: Query):
    if query.polygon:
        return shape(query.polygon)
    if query.bbox and len(query.bbox) == 4:
        return box(*query.bbox)
    return None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><title>Fire Monitoring</title>
    <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'>
    <style>body{font-family:sans-serif;margin:0}header{padding:10px 16px}#map{height:75vh}.row{display:flex;gap:8px}</style></head>
    <body><header><h2>Мониторинг природных пожаров</h2><div class='row'>
    <input id='from' type='date'><input id='to' type='date'><button onclick='loadData()'>Показать</button>
    <a href='/docs'>REST API</a></div><p id='summary'></p></header><div id='map'></div>
    <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script><script>
    const map=L.map('map').setView([48.5,44.5],5); L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    let layer; async function loadData(){const payload={date_from:document.getElementById('from').value||null,date_to:document.getElementById('to').value||null,bbox:map.getBounds().toBBoxString().split(',').map(Number)};
    const [geo,report]=await Promise.all([fetch('/api/v1/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(r=>r.json()),fetch('/api/v1/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(r=>r.json())]);
    if(layer)map.removeLayer(layer);layer=L.geoJSON(geo,{style:f=>({color:['#888','#ffd54f','#ff8c42','#c62828'][f.properties.severity_class||0]}),pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:5,color:'#e53935'})}).addTo(map);
    document.getElementById('summary').textContent=`Площадь гари: ${report.area_total_ha.toFixed(1)} га`;}
    loadData();</script></body></html>"""


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
    clip = query_geometry(payload)
    geod = Geod(ellps="WGS84")
    for feature in features:
        p = feature.get("properties", {})
        c = str(p.get("severity_class", ""))
        if c in areas:
            geometry = shape(feature["geometry"])
            if clip is not None:
                geometry = geometry.intersection(clip)
                area_m2, _ = geod.geometry_area_perimeter(geometry)
                areas[c] += abs(area_m2) / 10000.0
            else:
                areas[c] += float(p.get("area_ha", 0.0))
    return {"area_total_ha": sum(areas.values()), "area_by_severity_ha": areas}
