from __future__ import annotations
import sys, os, json, math, urllib.request, re
from pathlib import Path
from typing import Any

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_repo_root, _backend_dir]:
    if _p not in sys.path: sys.path.insert(0, _p)

from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse

from microservices.api_soda_cleaner import (
    DATASET_CONFIGS, cache_service, serialize_entry,
    update_dataset_node, get_combined_datasets_snapshot
)
from microservices.routing import RouteOptimizer, GraphNode
from microservices.grapher import MapGrapher
from microservices.reporter import get_filtered_accidents, generate_report_chart
from microservices.osm_graph import (
    load_osm_graph, build_osm_nodes, snap_to_osm_node,
    find_safest_route_osm, compute_risk_scores, build_edge_index
)

app = FastAPI(title="Safeway API", version="1.0.0")
COMMUNITY_REPORTS = []

# ---------- shared helpers ----------

# ---------- Symbolic formula from invariant regression ----------
_sym_coefs = None

def _load_symbolic_formula():
    global _sym_coefs
    if _sym_coefs is not None:
        return _sym_coefs
    try:
        fp = Path(__file__).resolve().parent / "model" / "symbolic_formula.txt"
        if fp.exists():
            text = fp.read_text()
            r2_match = re.search(r'R²=([\d.]+)', text)
            r2 = float(r2_match.group(1)) if r2_match else 0.0
            fm_line = [l for l in text.split('\n') if l.startswith('risk=')][0]
            nums = re.findall(r'[-]?\d+\.\d+', fm_line)
            if nums:
                _sym_coefs = [r2] + [float(n) for n in nums]
                print(f"Loaded formula (R²={r2:.3f}, {len(nums)} coeffs)")
                return _sym_coefs
    except Exception as e:
        print(f"Formula load error: {e}")
    _sym_coefs = [0.0] + [0.0]*9
    return _sym_coefs

def _symbolic_risk(rain=0, severity=0, acc_density=0, degree=0, 
                   neighbor_acc=0, neighbor_sev=0, betweenness=0, mode_match=0):
    """Symbolic risk from 8 structural features (generalizes across cities)."""
    cf = _load_symbolic_formula()
    feats = [rain, severity, acc_density, degree, neighbor_acc, neighbor_sev, betweenness, mode_match]
    risk = cf[1] if len(cf) > 1 else 0.0
    for i in range(min(len(cf)-2, len(feats))):
        risk += cf[i+2] * feats[i]
    return max(0.0, risk)

def _severity(vehicles_str: str) -> float:
    v = str(vehicles_str).upper()
    if any(k in v for k in ("MUERTO","FALLECIDO","MORTAL")): return 4.0
    if any(k in v for k in ("HERIDO","LESIONADO")): return 2.0
    return 1.0

# OSRM helpers
OSRM_BASE = "https://router.project-osrm.org"

def _osrm_route(waypoints, alternatives=False):
    if len(waypoints) < 2: return None
    q = ";".join(f"{ln},{lt}" for lt, ln in waypoints)
    extra = "&alternatives=" + str(alternatives).lower() if alternatives else ""
    try:
        url = f"{OSRM_BASE}/route/v1/driving/{q}?overview=full&geometries=geojson&steps=false{extra}"
        req = urllib.request.Request(url, headers={'User-Agent':'SafeWay/1.0'})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        routes = []
        for rt in data.get("routes", []):
            coords = [(c[1], c[0]) for c in rt["geometry"]["coordinates"]]
            routes.append({"path": coords, "dist_km": round(rt.get("distance",0)/1000,2),
                          "dur_min": round(rt.get("duration",0)/60,1)})
        return routes if routes else None
    except Exception:
        return None

def _path_length_km(path):
    t = 0.0
    for i in range(1, len(path)):
        d = math.sqrt((path[i][0]-path[i-1][0])**2+(path[i][1]-path[i-1][1])**2)
        t += d * 111.0
    return t

CITY_CONFIG = {
    'palmira': {
        'dataset_id': 'sjpx-eqfp',
        'graphml': 'palmira_streets.graphml',
        'center': [3.54, -76.31],
        'zoom': 13,
    },
    'pereira': {
        'dataset_id': 'pg82-4qqr',
        'graphml': 'pereira_streets.graphml',
        'center': [4.814, -75.694],
        'zoom': 13,
    },
}

def _load_osm_data(city='palmira'):
    """Load OSM graph + snap accidents. Cached in memory."""
    cfg = CITY_CONFIG.get(city, CITY_CONFIG['palmira'])
    dataset_id = cfg['dataset_id']
    graphml_name = cfg['graphml']
    
    cache_key = f"{dataset_id}_{graphml_name}"
    if not hasattr(_load_osm_data, '_cache'):
        _load_osm_data._cache = {}
    if cache_key in _load_osm_data._cache:
        return _load_osm_data._cache[cache_key]
    
    graphml_path = os.path.join(_repo_root, 'data', graphml_name)
    G = load_osm_graph(graphml_path)
    
    accidents = []
    if dataset_id == 'pg82-4qqr':
        # Pereira: load from raw JSON file directly
        raw_path = os.path.join(_repo_root, 'data', 'raw_pg82-4qqr.json')
        if os.path.exists(raw_path):
            import re as _re
            with open(raw_path) as f:
                raw_list = json.load(f)
            for row in raw_list:
                coord = str(row.get('coordenadas', ''))
                if coord and '(' in coord:
                    parts = coord.strip('()').split()
                    if len(parts) == 2:
                        try:
                            lat, lng = float(parts[0]), float(parts[1])
                            fecha = str(row.get('fecha_del_hecho', ''))
                            hora = str(row.get('hora_hecho', ''))
                            victima = str(row.get('victima', ''))
                            h = 12
                            m = _re.search(r'T(\d{2}):', hora)
                            if m: h = int(m.group(1))
                            accidents.append({
                                'latitude': lat, 'longitude': lng,
                                'date_iso': fecha[:10] if fecha else '2023-01-01',
                                'time': f'{h:02d}:00', 'vehicles': victima,
                                'data_original': {
                                    'victima': victima,
                                    'lesionados_y_muertos': 'MUERTO',
                                    'condicion_de_la_victima': victima,
                                }
                            })
                        except: pass
    elif dataset_id in DATASET_CONFIGS:
        try:
            for r in cache_service.get_snapshot(dataset_id, max_rows=50000, force_refresh=False).processed:
                if r.get('latitude') is not None and r.get('longitude') is not None:
                    accidents.append(dict(r))
        except Exception:
            pass
    accidents.extend(COMMUNITY_REPORTS)
    
    nodes, ei, snapped = build_osm_nodes(accidents, G)
    result = (G, nodes, ei)
    _load_osm_data._cache[cache_key] = result
    return result

# ---------- endpoints ----------

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return path.read_text("utf-8") if path.exists() else "<h1>index.html not found</h1>"

@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, max_rows: int = Query(default=200, ge=1, le=5000),
                force_refresh: bool = False, id_field: str|None = None, location_field: str|None = None):
    return serialize_entry(cache_service.get_snapshot(dataset_id, max_rows, force_refresh, id_field, location_field))

@app.get("/datasets/combined")
def get_combined(dataset_ids: str = Query(default="7cci-nqqb"), max_rows: int = Query(default=50000, ge=1, le=100000),
                 force_refresh: bool = False):
    return get_combined_datasets_snapshot(dataset_ids, max_rows, force_refresh)

@app.get("/datasets/combined/graph")
def get_graph(dataset_ids: str = Query(default="sjpx-eqfp"), max_rows: int = Query(default=50000, ge=1, le=100000),
              target_year: int = 2024, rain_active: bool = False, target_hour: int | None = None,
              mode: str = Query(default="all"), city: str = Query(default="palmira")):
    try:
        G, nodes, ei = _load_osm_data(city)
    except Exception as e:
        return {"error": f"Failed to load OSM graph: {e}", "nodes": [], "edges": []}
    
    # Compute risks using current formula
    node_risks = compute_risk_scores(nodes, G, rain_active, target_hour or 12, mode)
    
    # Build response
    res_nodes = []
    for nd in nodes:
        acc_cnt = len(nd.accidents)
        danger = 1 if acc_cnt == 0 else min(95, max(10, int(10 + 85 * (acc_cnt / 30))))
        risk = node_risks.get(nd.osm_id, 0.0)
        res_nodes.append({
            "id": nd.id, "lat": nd.lat, "lng": nd.lng, "name": nd.label,
            "weight": acc_cnt, "predicted_risk": risk,
            "danger_percent": danger, "is_fallback": nd.is_fallback
        })
    
    # Sample edges: show ~2000 from the graph for visualization
    res_edges = []
    edge_list = list(G.edges())
    step = max(1, len(edge_list) // 2000)
    for i, (u, v) in enumerate(edge_list):
        if i % step == 0:
            res_edges.append({"source": f"osm_{u}", "target": f"osm_{v}"})
    
    return {"nodes": res_nodes, "edges": res_edges, "osm_nodes": len(G.nodes), "osm_edges": len(G.edges)}

@app.get("/datasets/combined/route")
def get_route(dataset_ids: str = Query(default="sjpx-eqfp"), max_rows: int = Query(default=50000, ge=1, le=100000),
              start_lat: float = Query(...), start_lng: float = Query(...),
              end_lat: float = Query(...), end_lng: float = Query(...),
              target_year: int = 2024, rain_active: bool = False, target_hour: int | None = None,
              mode: str = Query(default="all"), city: str = Query(default="palmira")):
    
    try:
        G, nodes, ei = _load_osm_data(city)
    except Exception:
        return _simple_fallback(start_lat, start_lng, end_lat, end_lng)
    
    # Compute risks
    node_risks = compute_risk_scores(nodes, G, rain_active, target_hour or 12, mode)
    
    # OSM Dijkstra for safe route
    safe_path, safe_dist_m = find_safest_route_osm(
        G, start_lat, start_lng, end_lat, end_lng, node_risks
    )
    
    # OSRM for fast route
    direct_routes = _osrm_route([(start_lat, start_lng), (end_lat, end_lng)], alternatives=False)
    
    # Score safe route
    safe_hz = _score_path(safe_path, node_risks, G)
    safe_km = round(safe_dist_m / 1000, 2) if safe_dist_m else round(_path_length_km(safe_path), 2)
    
    if direct_routes:
        fast = direct_routes[0]
        fast_hz = _score_path(fast["path"], node_risks, G)
        fastest = {"path": fast["path"], "hazard_score": fast_hz, "danger_percent": _danger_from_hz(fast_hz),
                   "dist_km": fast["dist_km"], "dur_min": fast["dur_min"]}
    else:
        fastest = {"path": safe_path, "hazard_score": safe_hz, "danger_percent": _danger_from_hz(safe_hz),
                   "dist_km": safe_km, "dur_min": round(safe_km/30*60, 1)}
    
    return {
        "safest": {"path": safe_path, "hazard_score": safe_hz,
                   "danger_percent": _danger_from_hz(safe_hz), "dist_km": safe_km,
                   "dur_min": round(safe_km/30*60, 1)},
        "fastest": {"path": fastest["path"], "hazard_score": fastest["hazard_score"],
                    "danger_percent": fastest["danger_percent"], "dist_km": fastest["dist_km"],
                    "dur_min": fastest["dur_min"]}
    }

def _score_path(path, node_risks, G):
    if len(path) < 2: return 0.0
    total = 0.0; hits = 0
    for lt, ln in path:
        nid = snap_to_osm_node(lt, ln, G, max_dist_m=100)
        if nid and nid in node_risks:
            total += node_risks[nid]
            hits += 1
    return round(total / max(hits, 1), 3)

def _danger_from_hz(hz):
    """Map hazard score to danger percentage (0-100)."""
    # Hazard scores range ~0 to ~3 with ×300 penalty
    hz = max(0, min(3, hz))
    return round(hz / 3.0 * 100)

def _simple_fallback(start_lat, start_lng, end_lat, end_lng):
    d_km = _path_length_km([(start_lat,start_lng),(end_lat,end_lng)])
    return {
        "safest": {"path": [[start_lat,start_lng],[end_lat,end_lng]], "hazard_score": 0,
                   "danger_percent": 5, "dist_km": round(d_km,2), "dur_min": round(d_km/30*60,1)},
        "fastest": {"path": [[start_lat,start_lng],[end_lat,end_lng]], "hazard_score": 0,
                    "danger_percent": 5, "dist_km": round(d_km,2), "dur_min": round(d_km/30*60,1)}
    }

@app.get("/datasets/export")
def export_records(dataset_ids: str = Query(default="7cci-nqqb"), max_rows: int = Query(default=50000, ge=1, le=100000),
                   start_year: int | None = None, end_year: int | None = None, rain_only: bool | None = None,
                   vehicle_type: str | None = None, city: str | None = None, export_format: str = "json"):
    snapshot = get_combined_datasets_snapshot(dataset_ids, max_rows)
    filtered = get_filtered_accidents(snapshot["tables"]["records"], start_year, end_year, rain_only, vehicle_type, city)
    if export_format.lower() == "csv":
        import csv, io
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(["id","dataset_id","row_id","latitude","longitude","location","is_fallback_coord","date_iso","time","vehicles"])
        for r in filtered:
            w.writerow([r.get("id"), r.get("dataset_id"), r.get("row_id"), r.get("latitude"), r.get("longitude"),
                        r.get("location"), r.get("is_fallback_coord"), r.get("date_iso"), r.get("time"), r.get("vehicles")])
        return Response(content=out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition":"attachment; filename=export.csv"})
    return {"count": len(filtered), "records": filtered}

@app.get("/datasets/chart.png")
def get_chart(dataset_ids: str = Query(default="7cci-nqqb"), max_rows: int = Query(default=50000, ge=1, le=100000),
              start_year: int | None = None, end_year: int | None = None, rain_only: bool | None = None,
              vehicle_type: str | None = None, city: str | None = None):
    snapshot = get_combined_datasets_snapshot(dataset_ids, max_rows)
    filtered = get_filtered_accidents(snapshot["tables"]["records"], start_year, end_year, rain_only, vehicle_type, city)
    return Response(content=generate_report_chart(filtered), media_type="image/png")

@app.get("/datasets/{dataset_id}/updates")
async def poll_updates(dataset_id: str, last_version: str | None = None, max_rows: int = Query(default=200, ge=1, le=5000),
                        timeout_seconds: int = Query(default=30, ge=1, le=300)):
    entry, changed, timed_out = await cache_service.wait_for_update(dataset_id, max_rows, last_version, timeout_seconds)
    resp = {"dataset_id": dataset_id, "max_rows": max_rows, "version": entry.version,
            "changed": changed, "timed_out": timed_out, "fetched_at": entry.fetched_at}
    if changed: resp["data"] = serialize_entry(entry)
    return resp

@app.put("/datasets/{dataset_id}/nodes/{row_id}")
def update_node(dataset_id: str, row_id: str, payload: dict[str, Any]):
    return update_dataset_node(dataset_id, row_id, payload)
