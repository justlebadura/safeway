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

app = FastAPI(title="Safeway API", version="1.0.0")
COMMUNITY_REPORTS = []

# ---------- shared helpers ----------

BGA_COORDS = {
    "MUTIS":(7.1090,-73.1280),"CENTRO":(7.1194,-73.1226),"CABECERA":(7.1218,-73.1118),
    "SAN FRANCISCO":(7.1322,-73.1216),"RIO DE ORO":(7.1430,-73.1310),"PROVENZA":(7.0980,-73.1110),
    "ALARCON":(7.1264,-73.1212),"CONCORDIA":(7.1180,-73.1250),"RICAURTE":(7.1140,-73.1260),
    "SOTOMAYOR":(7.1200,-73.1158),"AURORA":(7.1268,-73.1160),"REAL DE MINAS":(7.1110,-73.1190),
    "GARCIA ROVIRA":(7.1205,-73.1260),"DIAMANTE":(7.0890,-73.1120),"COMUNEROS":(7.1310,-73.1165),
    "SAN ALONSO":(7.1292,-73.1140),"CAMPO HERMOSO":(7.1230,-73.1320),"CONUCOS":(7.1115,-73.1122),
    "PUERTA DEL SOL":(7.1080,-73.1128),"PEDREGOSA":(7.0940,-73.1090),"CAFE MADRID":(7.1650,-73.1280),
    "ALVAREZ":(7.1250,-73.1110),"BOLARQUI":(7.1230,-73.1170),"MEJORAS":(7.1210,-73.1185),
    "PRADO":(7.1265,-73.1125),"GIRON":(7.0705,-73.1703),"FLORIDABLANCA":(7.0668,-73.0872),
    "PIEDECUESTA":(6.9892,-73.0518),"ORIENTAL":(7.1250,-73.1150),"OCCIDENTAL":(7.1350,-73.1350),
    "SUR":(7.1000,-73.1250),"CIUDADELA":(7.1050,-73.1250),"REGADEROS":(7.1500,-73.1220),
    "DANGOND":(7.0950,-73.1280),"NORORIENTAL":(7.1480,-73.1200),"ESTORAQUES":(7.1080,-73.1290),
    "ANTONIA SANTOS":(7.1235,-73.1215),"UNIVERSIDAD":(7.1375,-73.1210),
}

import hashlib as _hl

def _barrio_coords(barrio_name, row_id):
    key = barrio_name.strip().upper()
    for name, coords in BGA_COORDS.items():
        if name in key:
            h = int(_hl.md5(f"{name}{row_id}".encode()).hexdigest(), 16)
            lat_off = ((h%1000)/1000.0-0.5)*0.002
            lng_off = (((h//1000)%1000)/1000.0-0.5)*0.002
            return coords[0]+lat_off, coords[1]+lng_off, False
    h = int(_hl.md5(row_id.encode()).hexdigest(), 16)
    lat = 7.1193 + ((h%4000)/4000.0-0.5)*0.04
    lng = -73.1227 + (((h//4000)%4000)/4000.0-0.5)*0.04
    return lat, lng, True

def _extract_intersection(acc):
    """Extract street intersection from accident record.
    Returns (name, lat, lng) or None if only barrio info available."""
    extr = acc.get("extraccion", {})
    if not isinstance(extr, dict): return None

    via = extr.get("VIA_PRINCIPAL", {}).get("value", "").strip().upper() if extr.get("VIA_PRINCIPAL") else ""
    if not via: return None

    # Try Carrera X con Calle Y pattern
    m = re.search(r"CARRERA\s*(\d+).*?CALLE\s*(\d+)", via)
    if m:
        cr, cl = int(m.group(1)), int(m.group(2))
        name = f"Cr {cr} c/ Cl {cl}"
        lat = 7.1190 + (cl - 36) * 0.0007
        lng = -73.1220 - (cr - 27) * 0.0007
        return name, lat, lng

    # Try Calle X con Carrera Y
    m = re.search(r"CALLE\s*(\d+).*?CARRERA\s*(\d+)", via)
    if m:
        cl, cr = int(m.group(1)), int(m.group(2))
        name = f"Cl {cl} c/ Cr {cr}"
        lat = 7.1190 + (cl - 36) * 0.0007
        lng = -73.1220 - (cr - 27) * 0.0007
        return name, lat, lng

    # Try single Carrera
    m = re.search(r"CARRERA\s*(\d+)", via)
    if m:
        cr = int(m.group(1))
        name = f"Cr {cr}"
        lat = 7.1190
        lng = -73.1220 - (cr - 27) * 0.0007
        return name, lat, lng

    # Try single Calle
    m = re.search(r"CALLE\s*(\d+)", via)
    if m:
        cl = int(m.group(1))
        name = f"Cl {cl}"
        lat = 7.1190 + (cl - 36) * 0.0007
        lng = -73.1220
        return name, lat, lng

    # Try "27 CON 36" or "27 # 36" pattern
    m = re.search(r"(\d{1,3})\s*(?:CON|Y|#|-\s*)\s*(\d{1,3})", via)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 50: a, b = b, a  # ensure a=cr, b=cl
        name = f"Cr {a} c/ Cl {b}"
        lat = 7.1190 + (b - 36) * 0.0007
        lng = -73.1220 - (a - 27) * 0.0007
        return name, lat, lng

    return None


def _build_nodes(accidents, dataset_id):
    """Build full street grid: every Cr × Cl intersection in Bucaramanga range.
    Accidents are assigned to nearest intersection. Empty intersections shown as grey dots."""
    from collections import defaultdict
    import math
    
    # Bucaramanga grid range
    CR_MIN, CR_MAX = 15, 40
    CL_MIN, CL_MAX = 1, 60
    BGA_CR0, BGA_CL0 = -73.1220, 7.1190  # Cr 27, Cl 36 reference
    UNIT = 0.0007
    
    # Map for assigning accidents to nearest grid point
    grid = defaultdict(list)
    
    # Assign each accident to nearest grid intersection
    for acc in accidents:
        did = acc.get("dataset_id", "")
        b = acc.get("data_original", {}).get("barrio", "").strip()
        key = b.split(",")[0].strip().upper()
        
        # Bucaramanga: use barrio mapping (substring match for partial names)
        if did == "7cci-nqqb":
            coords = (BGA_CL0, BGA_CR0)  # default
            for bname, bcoord in BGA_COORDS.items():
                if bname in key:
                    coords = bcoord
                    break
            acc_lat, acc_lng = coords[0], coords[1]
        else:
            acc_lat = acc.get("latitude", BGA_CL0) or BGA_CL0
            acc_lng = acc.get("longitude", BGA_CR0) or BGA_CR0
        
        # Snap to nearest grid intersection
        cr_float = 27 + (acc_lng - BGA_CR0) / UNIT
        cl_float = 36 + (acc_lat - BGA_CL0) / UNIT
        cr = max(CR_MIN, min(CR_MAX, int(round(cr_float))))
        cl = max(CL_MIN, min(CL_MAX, int(round(cl_float))))
        
        grid[(cr, cl)].append(acc)
    
    nodes = []
    nid = 1
    
    # Generate ALL grid intersections (including empty ones)
    for cr in range(CR_MIN, CR_MAX + 1):
        for cl in range(CL_MIN, CL_MAX + 1):
            lat = BGA_CL0 + (cl - 36) * UNIT
            lng = BGA_CR0 - (cr - 27) * UNIT
            
            accs = grid.get((cr, cl), [])
            name = f"Cr {cr} c/ Cl {cl}"
            nd = GraphNode(f"n_{nid}", lat, lng, label=name, is_fallback=(len(accs) == 0))
            nd.cr = cr; nd.cl = cl
            for a in accs: nd.add_accident(a)
            nodes.append(nd)
            nid += 1
    
    # Add topological features after all nodes created
    for nd in nodes:
        # Manhattan degree: how many adjacent grid intersections exist
        neighbors = 0
        for dc, dl in [(1,0),(-1,0),(0,1),(0,-1)]:
            nc, nl = nd.cr + dc, nd.cl + dl
            if CR_MIN <= nc <= CR_MAX and CL_MIN <= nl <= CL_MAX:
                neighbors += 1
        nd.degree = neighbors
        # Distance to city center (Cr 27, Cl 36)
        nd.dist_center = math.sqrt((nd.cr - 27)**2 + (nd.cl - 36)**2) / 50.0  # normalized
    
    return nodes

def _severity(vehicles_str: str) -> float:
    v = str(vehicles_str).upper()
    if any(k in v for k in ("MUERTO","FALLECIDO","MORTAL")): return 4.0
    if any(k in v for k in ("HERIDO","LESIONADO")): return 2.0
    return 1.0

def _build_features(nodes, target_hour, rain_active, seq_len=5):
    import torch
    num_nodes = len(nodes)
    seq = []
    hour = target_hour if target_hour is not None else 12
    for t in range(seq_len):
        h = (hour - 4 + t) % 24
        r = rain_active if t == seq_len - 1 else False
        feats = torch.zeros((num_nodes, 7), dtype=torch.float32)
        for idx, node in enumerate(nodes):
            feats[idx, 0] = 1.0 if r else 0.0
            feats[idx, 1] = 0.0 if r else 1.0
            feats[idx, 2] = h / 24.0
            n = len(node.accidents)
            feats[idx, 4] = float(n) / 50.0
            feats[idx, 3] = (sum(_severity(a.get("vehicles","")) for a in node.accidents) / max(n, 1)) / 4.0
            feats[idx, 5] = getattr(node, 'degree', 4) / 4.0  # topological: connectivity
            feats[idx, 6] = min(1.0, getattr(node, 'dist_center', 0))  # topological: centrality
        seq.append(feats)
    return torch.stack(seq, dim=0)

def _infer_risks(nodes, target_hour, rain_active, edge_index):
    """GNN-LNN inference (lazy-loads model on first call)."""
    _ensure_model()
    if pretrained_model is None:
        _symbolic_risks(nodes, target_hour, rain_active, edge_index)
        return
    try:
        import torch
        x_seq = _build_features(nodes, target_hour, rain_active)
        with torch.no_grad():
            pred = pretrained_model(x_seq, edge_index)
        for i, node in enumerate(nodes):
            node.predicted_risk = float(pred[i, 0]) * 10.0
    except Exception as e:
        print("ML inference failed, using symbolic formula:", e)
        _symbolic_risks(nodes, target_hour, rain_active, edge_index)

def _symbolic_risks(nodes, target_hour, rain_active, edge_index):
    """Distilled GNN-LNN symbolic formula using GCN convolution."""
    num_nodes = len(nodes)
    rain_val = 1.0 if rain_active else 0.0
    h = (target_hour if target_hour is not None else 12) / 24.0
    
    avg_sev = [sum(_severity(a.get("vehicles","")) for a in nd.accidents) / max(len(nd.accidents), 1) for nd in nodes]
    accidents = [float(len(nd.accidents)) for nd in nodes]
    
    degrees = [1.0] * num_nodes
    for u, v in zip(edge_index[0], edge_index[1]):
        degrees[u] += 1.0
    
    sev_conv = [0.0] * num_nodes
    acc_conv = [0.0] * num_nodes
    for idx in range(num_nodes):
        sev_conv[idx] += avg_sev[idx] / degrees[idx]
        acc_conv[idx] += accidents[idx] / degrees[idx]
    for u, v in zip(edge_index[0], edge_index[1]):
        factor = 1.0 / math.sqrt(degrees[u] * degrees[v])
        sev_conv[u] += avg_sev[v] * factor
        acc_conv[u] += accidents[v] * factor

    for idx, node in enumerate(nodes):
        if acc_conv[idx] == 0.0:
            node.predicted_risk = 0.0
        else:
            node.predicted_risk = max(0.0, (
                0.05 + 0.075 * rain_val + 0.226 * h +
                0.006 * math.sin(6.283 * h) - 0.026 * math.cos(6.283 * h) +
                0.209 * sev_conv[idx] + 0.028 * acc_conv[idx]
            ))

def _build_edge_index(nodes, proximity=0.015):
    """Grid edges: connect adjacent intersections (Manhattan neighbors).
    Cr X c/ Cl Y <-> Cr X±1 c/ Cl Y and Cr X c/ Cl Y±1."""
    import torch, re
    
    # Map (cr, cl) -> node index
    grid_map = {}
    for i, nd in enumerate(nodes):
        m = re.match(r"Cr (\d+) c/ Cl (\d+)", nd.label)
        if m:
            grid_map[(int(m.group(1)), int(m.group(2)))] = i
    
    src, tgt = [], []
    for (cr, cl), i in grid_map.items():
        for dc, dl in [(1,0), (-1,0), (0,1), (0,-1)]:
            j = grid_map.get((cr+dc, cl+dl))
            if j is not None:
                src.append(i); tgt.append(j)
    
    # Proximity fallback for any unconnected nodes
    for i in range(len(nodes)):
        if i in src: continue
        for j in range(len(nodes)):
            if i == j: continue
            d = math.sqrt((nodes[i].lat-nodes[j].lat)**2+(nodes[i].lng-nodes[j].lng)**2)
            if d < 0.005:
                src.append(i); tgt.append(j)
    
    if not src:
        return torch.tensor([list(range(len(nodes)))]*2, dtype=torch.long)
    return torch.tensor([src, tgt], dtype=torch.long)

def _compute_danger(nodes):
    risks = [getattr(n, "predicted_risk", 0.0) for n in nodes]
    rmin, rmax = min(risks) if risks else 0.0, max(risks) if risks else 1.0
    rr = max(rmax - rmin, 0.001)
    return [(r - rmin) / rr for r in risks], rmin, rmax

def _spatial_smooth(nodes, radius=0.005, decay=0.5):
    """Propagate log-risk from accident nodes to nearby empty nodes."""
    import math as _m
    max_dist = radius
    empty_nodes = [nd for nd in nodes if len(nd.accidents) == 0]
    accident_nodes = [nd for nd in nodes if len(nd.accidents) > 0]
    
    if not empty_nodes or not accident_nodes or not hasattr(accident_nodes[0], 'predicted_risk'):
        return
    
    # Use log2(1+risk) for stable propagation
    for nd in accident_nodes:
        nd._log_risk = _m.log2(1 + max(0, nd.predicted_risk))
    
    for empty in empty_nodes:
        total_w = 0.0; total_r = 0.0
        for acc_nd in accident_nodes:
            d = _m.sqrt((empty.lat - acc_nd.lat)**2 + (empty.lng - acc_nd.lng)**2)
            if d < max_dist and d > 0:
                w = 1.0 / d
                total_r += getattr(acc_nd, '_log_risk', 0) * w
                total_w += w
        if total_w > 0:
            empty.predicted_risk = (total_r / total_w) * decay

def _osrm_route(coords):
    if len(coords) <= 1: return coords
    q = ";".join([f"{ln},{lt}" for lt, ln in coords])
    try:
        req = urllib.request.Request(f"https://router.project-osrm.org/route/v1/driving/{q}?overview=full&geometries=geojson",
                                     headers={'User-Agent':'SafeWay/1.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        return [(lt, ln) for ln, lt in data["routes"][0]["geometry"]["coordinates"]]
    except: return coords

# ---------- Symbolic formula from invariant regression ----------
# Coeficientes actualizados al cargar el modelo entrenado
_sym_coefs = None

def _load_symbolic_formula():
    """Carga la formula simbolica destilada de la GNN+LNN entrenada."""
    global _sym_coefs
    if _sym_coefs is not None:
        return _sym_coefs
    try:
        fp = Path(__file__).resolve().parent / "model" / "symbolic_formula.txt"
        if fp.exists():
            import re
            txt = fp.read_text()
            # Parse coefficients
            nums = re.findall(r'[-]?\d+\.\d+', txt)
            if len(nums) >= 10:
                _sym_coefs = [float(n) for n in nums[:10]]  # [r2, ic, c0..c7]
                print(f"Loaded symbolic formula (R²={_sym_coefs[0]:.3f})")
                return _sym_coefs
    except Exception as e:
        print(f"Symbolic formula load error: {e}")
    _sym_coefs = [0.83, 0.1, 0, 0, 0.1, 0.01, 0.1, 0.005, 0, 0]
    return _sym_coefs

def _symbolic_risk_production(rain, hour, severity, acc, sev_neighbor, acc_neighbor):
    """Formula simbolica destilada de la red GNN(2-capas)+LNN entrenada.
    Reemplaza a calculate_risk() en produccion."""
    cf = _load_symbolic_formula()
    h = (hour if hour is not None else 12) / 24.0
    import math as _m
    return max(0.0,
        cf[1] +                          # intercept
        cf[2] * (1.0 if rain else 0.0) + # rain coefficient
        cf[3] * h +                      # hour coefficient  
        cf[4] * severity +               # severity coefficient
        cf[5] * acc +                    # accidents coefficient
        cf[6] * sev_neighbor +           # neighbor severity
        cf[7] * acc_neighbor +           # neighbor accidents
        cf[8] * _m.sin(6.283 * h) +      # sin harmonic
        cf[9] * _m.cos(6.283 * h)        # cos harmonic
    )

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
def get_graph(dataset_ids: str = Query(default="7cci-nqqb"), max_rows: int = Query(default=50000, ge=1, le=100000),
              target_year: int = 2026, rain_active: bool = False, target_hour: int | None = None, use_symbolic: bool = False):
    ids = [d.strip() for d in dataset_ids.split(",") if d.strip()]
    accidents = []
    for did in ids:
        if did not in DATASET_CONFIGS: continue
        for r in cache_service.get_snapshot(did, max_rows, force_refresh=False).processed:
            if r.get("latitude") is not None and r.get("longitude") is not None:
                r2 = dict(r); r2["dataset_id"] = did; accidents.append(r2)
    accidents.extend(COMMUNITY_REPORTS)

    nodes = _build_nodes(accidents, ids[0] if ids else "7cci-nqqb")
    ei = _build_edge_index(nodes)

    # Compute risk using symbolic formula (invariant regression from trained GNN+LNN)
    cf = _load_symbolic_formula()
    for nd in nodes:
        n = len(nd.accidents)
        sev = sum(_severity(a.get("vehicles","")) for a in nd.accidents) / max(n, 1) / 4.0
        acc = float(n) / 50.0
        nd.predicted_risk = _symbolic_risk_production(rain_active, target_hour or 12, sev, acc, sev, acc)
    
    _spatial_smooth(nodes)
    
    norms, _, _ = _compute_danger(nodes)
    res_nodes = []
    
    # Compute danger: empty=1%, others scaled by percentile among accident nodes
    acc_nodes = [nd for nd in nodes if len(nd.accidents) > 0]
    acc_counts = sorted([len(nd.accidents) for nd in acc_nodes])
    n_acc = len(acc_counts)
    
    for i, nd in enumerate(nodes):
        acc = len(nd.accidents)
        if acc == 0:
            d = 1  # empty intersection
        elif n_acc <= 1:
            d = 50  # only one accident node
        else:
            # Percentile rank among nodes with accidents -> danger 10-95%
            rank = sum(1 for c in acc_counts if c <= acc)
            pct = rank / n_acc  # [0, 1]
            d = min(95, max(10, int(10 + 85 * pct)))
        res_nodes.append({
            "id": nd.id, "lat": nd.lat, "lng": nd.lng, "name": nd.label,
            "weight": acc, "predicted_risk": nd.predicted_risk,
            "danger_percent": d, "is_fallback": nd.is_fallback
        })

    res_edges = []
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]; d = math.sqrt((a.lat-b.lat)**2+(a.lng-b.lng)**2)
            # Grid adjacency (~100m): show real street connections
            if d < 0.001:
                res_edges.append({"source": a.id, "target": b.id})
    
    # Mark empty nodes explicitly
    for nd in res_nodes:
        nd["empty"] = (nd["weight"] == 0)
    result = {"nodes": res_nodes, "edges": res_edges}
    return result

@app.get("/datasets/combined/route")
def get_route(dataset_ids: str = Query(default="7cci-nqqb"), max_rows: int = Query(default=50000, ge=1, le=100000),
              start_lat: float = Query(...), start_lng: float = Query(...),
              end_lat: float = Query(...), end_lng: float = Query(...),
              target_year: int = 2026, rain_active: bool = False, target_hour: int | None = None, use_symbolic: bool = False):
    ids = [d.strip() for d in dataset_ids.split(",") if d.strip()]
    accidents = []
    for did in ids:
        if did not in DATASET_CONFIGS: continue
        for r in cache_service.get_snapshot(did, max_rows, force_refresh=True).processed:
            if r.get("latitude") is not None and r.get("longitude") is not None:
                r2 = dict(r); r2["dataset_id"] = did; accidents.append(r2)
    accidents.extend(COMMUNITY_REPORTS)

    nodes = _build_nodes(accidents, "7cci-nqqb")
    ei = _build_edge_index(nodes)

    # Compute risk using symbolic formula (invariant regression from trained GNN+LNN)
    cf = _load_symbolic_formula()
    for nd in nodes:
        n = len(nd.accidents)
        sev = sum(_severity(a.get("vehicles","")) for a in nd.accidents) / max(n, 1) / 4.0
        acc = float(n) / 50.0
        nd.predicted_risk = _symbolic_risk_production(rain_active, target_hour or 12, sev, acc, sev, acc)
    
    # NO spatial smoothing for routing: preserves natural risk variance for Dijkstra
    # (spatial smooth is kept for graph visualization endpoint only)
    
    # Percentile-based normalization -> spreads values across [0, 10] evenly
    risks = sorted([nd.predicted_risk for nd in nodes])
    n = len(risks)
    for i, nd in enumerate(sorted(nodes, key=lambda x: x.predicted_risk)):
        percentile = (i + 1) / n  # [0, 1]
        nd.predicted_risk = 10.0 * percentile  # [0, 10] spread evenly

    # Snap start/end to closest structural nodes
    start_id = end_id = None
    min_s = min_e = float('inf')
    for nd in nodes:
        ds = math.sqrt((nd.lat-start_lat)**2+(nd.lng-start_lng)**2)
        de = math.sqrt((nd.lat-end_lat)**2+(nd.lng-end_lng)**2)
        if ds < min_s: min_s, start_id = ds, nd.id
        if de < min_e: min_e, end_id = de, nd.id

    norms, rmin, rmax = _compute_danger(nodes)
    def _path_hazard(path):
        total = 0.0; near = 0; max_acc = 0
        for lt, ln in path:
            md = float('inf'); cn = None
            for nd in nodes:
                d = math.sqrt((nd.lat-lt)**2+(nd.lng-ln)**2)
                if d < md: md, cn = d, nd
            if md < 0.0009 and cn:
                r = getattr(cn, "predicted_risk", 0.0)
                total += r * (1.0 - md/0.0009) * (1.0 + len(cn.accidents)*0.05)
                near += 1
                max_acc = max(max_acc, len(cn.accidents))
        # Log-scale danger for route
        if max_acc == 0:
            danger = 5
        else:
            import math as _m
            danger = min(95, max(5, int(10 + 20 * _m.log2(1 + max_acc))))
        return round(total,2), danger

    opt = RouteOptimizer(nodes)
    safe_path, _ = opt.find_safest_route(start_id, end_id, target_year, rain_active, target_hour, True)

    # Route candidates
    direct_osrm = _osrm_route([(start_lat, start_lng), (end_lat, end_lng)])
    fastest_h, fastest_d = _path_hazard(direct_osrm)
    candidates = [{"path": direct_osrm, "hazard": fastest_h, "danger": fastest_d, "fastest": True}]

    if safe_path and len(safe_path) > 2:
        wp = [safe_path[len(safe_path)//4], safe_path[len(safe_path)//2], safe_path[3*len(safe_path)//4]]
        seen = {(start_lat, start_lng), (end_lat, end_lng)}
        for w in wp:
            if w not in seen:
                seen.add(w)
                dp = _osrm_route([(start_lat, start_lng), w, (end_lat, end_lng)])
                if len(dp) > 2:
                    hz, dg = _path_hazard(dp)
                    candidates.append({"path": dp, "hazard": hz, "danger": dg, "fastest": False})

    candidates.sort(key=lambda x: x["hazard"])
    safest = candidates[0]
    fastest = next((r for r in candidates if r["fastest"]), candidates[0])

    # Scale safest danger percent
    if fastest["hazard"] > 0.001 and safest["hazard"] > 0:
        ratio = min(1.0, safest["hazard"] / fastest["hazard"])
        safe_danger = int(round(fastest["danger"] * (ratio ** 2)))
    else:
        safe_danger = fastest["danger"]

    return {
        "safest": {"path": safest["path"], "hazard_score": safest["hazard"],
                    "danger_percent": min(99, max(5, safe_danger)), "nodes_visited": len(safest["path"])},
        "fastest": {"path": fastest["path"], "hazard_score": fastest["hazard"],
                     "danger_percent": min(99, max(5, fastest["danger"])), "nodes_visited": len(fastest["path"])}
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
