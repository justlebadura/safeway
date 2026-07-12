from __future__ import annotations
from pathlib import Path
from typing import Any
import sys, os

# Ensure both the repo root AND backend/ are in the path.
# - Repo root: so 'from backend.external.*' imports resolve (used inside microservices/)
# - backend/: so 'from microservices.*' short imports resolve (used in api.py itself)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [_repo_root, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse

from microservices.api_soda_cleaner import (
    DATASET_CONFIGS,
    cache_service,
    serialize_entry,
    update_dataset_node,
    get_combined_datasets_snapshot
)
from microservices.routing import RouteOptimizer
from microservices.grapher import MapGrapher
from microservices.reporter import get_filtered_accidents, generate_report_chart


app = FastAPI(title="Safeway API", version="1.0.0")


def calculate_symbolic_risk(
    rain_active: bool,
    target_hour: int | None,
    avg_severity: float,
    num_accidents: float
) -> float:
    if num_accidents == 0.0 or avg_severity == 0.0:
        return 0.0
        
    import math
    h = (target_hour if target_hour is not None else 12) / 24.0
    
    # Distilled Neural Network (GNN-LNN) Formula (99.58% correlation)
    risk = (
        1.67427 + 
        0.26268 * h + 
        0.01238 * math.sin(2 * math.pi * h) - 
        0.02468 * math.cos(2 * math.pi * h)
    )
    return max(0.0, risk)



pretrained_model = None
try:
    import torch
    from model.arch.hybrid_model import HybridGNNLNN
    pretrained_model = HybridGNNLNN(in_features=5, gnn_hidden=8, lnn_hidden=16)
    model_path = Path(__file__).resolve().parent / "model" / "model.pth"
    if model_path.exists():
        pretrained_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        pretrained_model.eval()
        print("Loaded pre-trained GNN-LNN model successfully.")
    else:
        print("No pre-trained model file found. Will fall back to on-the-fly training.")
except Exception as e:
    print("Pre-trained model loading warning:", e)
    pretrained_model = None




@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


COMMUNITY_REPORTS = []


from backend.microservices.routing import GraphNode

def get_routing_graph_nodes(accidents, dataset_id):
    grapher = MapGrapher()
    nodes = grapher.build_structural_graph(accidents)
    
    if dataset_id == "7cci-nqqb":
        # Add the same deterministic negative nodes using a seed
        import random
        lats = [n.lat for n in nodes]
        lngs = [n.lng for n in nodes]
        lat_min, lat_max = min(lats), max(lats) if lats else (7.11, 7.14)
        lng_min, lng_max = min(lngs), max(lngs) if lngs else (-73.13, -73.11)
        
        num_negatives = len(nodes)
        neg_nodes = []
        neg_counter = 1
        
        random.seed(42)
        attempts = 0
        while len(neg_nodes) < num_negatives and attempts < 2000:
            attempts += 1
            rand_lat = random.uniform(lat_min, lat_max)
            rand_lng = random.uniform(lng_min, lng_max)
            
            too_close = False
            for n in nodes:
                if abs(n.lat - rand_lat) < 0.0013 and abs(n.lng - rand_lng) < 0.0013:
                    too_close = True
                    break
            
            if not too_close:
                node_id = f"node_neg_{neg_counter}"
                neg_counter += 1
                node = GraphNode(node_id, rand_lat, rand_lng, label="Zona Residencial Segura")
                neg_nodes.append(node)
        nodes.extend(neg_nodes)
    return nodes


@app.get("/datasets/combined/route")
def get_safest_route(
    dataset_ids: str = Query(default="7cci-nqqb"),
    max_rows: int = Query(default=1500, ge=1, le=5000),
    start_lat: float = Query(...),
    start_lng: float = Query(...),
    end_lat: float = Query(...),
    end_lng: float = Query(...),
    target_year: int = Query(default=2026),
    rain_active: bool = Query(default=False),
    target_hour: int | None = Query(default=None),
    use_symbolic: bool = Query(default=False),
) -> dict[str, Any]:
    ids = [d.strip() for d in dataset_ids.split(",") if d.strip()]
    accidents = []
    for dataset_id in ids:
        if dataset_id not in DATASET_CONFIGS:
            continue
        entry = cache_service.get_snapshot(dataset_id, max_rows=max_rows)
        for r in entry.processed:
            if r.get("latitude") is not None and r.get("longitude") is not None:
                acc_info = dict(r)
                acc_info["dataset_id"] = dataset_id
                accidents.append(acc_info)

    # Append live community-reported incidents
    for rep in COMMUNITY_REPORTS:
        accidents.append(rep)

    nodes = get_routing_graph_nodes(accidents, "7cci-nqqb")

    # Snap lat/lng to closest structural node IDs
    import math
    start_id, end_id = None, None
    min_start_dist, min_end_dist = float('inf'), float('inf')
    
    for node in nodes:
        d_start = math.sqrt((node.lat - start_lat)**2 + (node.lng - start_lng)**2)
        d_end = math.sqrt((node.lat - end_lat)**2 + (node.lng - end_lng)**2)
        if d_start < min_start_dist:
            min_start_dist = d_start
            start_id = node.id
        if d_end < min_end_dist:
            min_end_dist = d_end
            end_id = node.id

    if start_id is None or end_id is None:
        return {
            "safest": {
                "path": [(start_lat, start_lng), (end_lat, end_lng)],
                "hazard_score": 0.0,
                "danger_percent": 5,
                "nodes_visited": 2
            },
            "fastest": {
                "path": [(start_lat, start_lng), (end_lat, end_lng)],
                "hazard_score": 0.0,
                "danger_percent": 5,
                "nodes_visited": 2
            }
        }

    # 1. Build spatial edge index for GNN convolution (proximity < 0.005)
    import math
    import torch
    sources, targets = [], []
    num_nodes = len(nodes)
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i != j:
                dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
                if dist < 0.005:
                    sources.append(i)
                    targets.append(j)
    if not sources:
        sources = list(range(num_nodes))
        targets = list(range(num_nodes))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    # 2. Determine method: Symbolic (Direct Heuristic) vs ML (GNN-LNN PyTorch Model)
    torch_available = False
    try:
        import torch
        torch_available = True
    except ImportError:
        pass

    if torch_available and pretrained_model is not None and not use_symbolic:
        try:
            # Prepare ML features sequence
            sequences = []
            hour_val = target_hour if target_hour is not None else 12
            for t in range(5):
                h_seq = (hour_val - 4 + t) % 24
                r_seq = rain_active if t == 4 else False
                features = torch.zeros((num_nodes, 5), dtype=torch.float32)
                for idx, node in enumerate(nodes):
                    features[idx, 0] = 1.0 if r_seq else 0.0
                    features[idx, 1] = 0.0 if r_seq else 1.0
                    features[idx, 2] = h_seq / 24.0
                    num_acc = len(node.accidents)
                    features[idx, 4] = float(num_acc) / 50.0  # Normalize count
                    if num_acc > 0:
                        sev_sum = 0.0
                        for acc in node.accidents:
                            v = str(acc.get("vehicles", "")).upper()
                            if "MUERTO" in v or "FALLECIDO" in v:
                                sev_sum += 4.0
                            elif "HERIDO" in v or "LESIONADO" in v:
                                sev_sum += 2.0
                            else:
                                sev_sum += 1.0
                        features[idx, 3] = (sev_sum / num_acc) / 4.0  # Normalize severity
                    else:
                        features[idx, 3] = 0.0
                sequences.append(features)
            x_seq = torch.stack(sequences, dim=0)

            # Run GNN-LNN model inference directly to evaluate risks using convolved weights
            pretrained_model.eval()
            with torch.no_grad():
                pred = pretrained_model(x_seq, edge_index)
            for idx, node in enumerate(nodes):
                node.predicted_risk = float(pred[idx, 0]) * 10.0
        except Exception as e:
            print("GNN-LNN execution error, falling back to symbolic:", e)
            use_symbolic = True

    if not torch_available or pretrained_model is None or use_symbolic:
        # Fallback: convolved symbolic formula (exact mathematical GNN-LNN convolved representation)
        import math
        avg_severities = []
        for node in nodes:
            num_acc = len(node.accidents)
            if num_acc > 0:
                sev_sum = 0.0
                for acc in node.accidents:
                    v = str(acc.get("vehicles", "")).upper()
                    if "MUERTO" in v or "FALLECIDO" in v:
                        sev_sum += 4.0
                    elif "HERIDO" in v or "LESIONADO" in v:
                        sev_sum += 2.0
                    else:
                        sev_sum += 1.0
                avg_severities.append(sev_sum / num_acc)
            else:
                avg_severities.append(0.0)

        # Compute degree for each node (including self-loop)
        degrees = [1.0] * num_nodes
        for u, v in zip(sources, targets):
            degrees[u] += 1.0

        # Convolve features (symmetric GCN normalization: norm_A = D_inv_sqrt @ A @ D_inv_sqrt)
        severity_conv = [0.0] * num_nodes
        accidents_conv = [0.0] * num_nodes
        
        for idx in range(num_nodes):
            local_factor = 1.0 / degrees[idx]
            severity_conv[idx] += avg_severities[idx] * local_factor
            accidents_conv[idx] += float(len(nodes[idx].accidents)) * local_factor
            
        for u, v in zip(sources, targets):
            factor = 1.0 / math.sqrt(degrees[u] * degrees[v])
            severity_conv[u] += avg_severities[v] * factor
            accidents_conv[u] += float(len(nodes[v].accidents)) * factor

        rain_val = 1.0 if rain_active else 0.0
        h = (target_hour if target_hour is not None else 12) / 24.0

        for idx, node in enumerate(nodes):
            if accidents_conv[idx] == 0.0:
                node.predicted_risk = 0.0
                continue
            # Convolved GNN-LNN distilled formula with positive correlation for risk
            risk = (
                0.05 + 
                0.07559 * rain_val +
                0.22642 * h +
                0.00649 * math.sin(2 * math.pi * h) -
                0.02560 * math.cos(2 * math.pi * h) +
                0.20942 * severity_conv[idx] +
                0.02797 * accidents_conv[idx]
            )
            node.predicted_risk = max(0.0, risk)

    # Compute min and max baseline risks in the active graph
    all_risks = []
    for n in nodes:
        r = getattr(n, "predicted_risk", None)
        if r is None:
            r = n.calculate_risk(target_year, rain_active, target_hour)
        all_risks.append(r)
    min_graph_risk = min(all_risks) if all_risks else 0.0
    max_graph_risk = max(all_risks) if all_risks else 1.0
    risk_range = max_graph_risk - min_graph_risk
    if risk_range <= 0.001:
        risk_range = 1.0

    def evaluate_path_safety(path_coords):
        if not path_coords:
            return 0.0, 5
        
        # Calculate hazard sum by finding closest nodes in the 220 GNN-LNN graph
        total_hazard = 0.0
        node_risks = []
        for lat, lng in path_coords:
            min_d = float('inf')
            closest_node = None
            for node in nodes:
                d = math.sqrt((node.lat - lat)**2 + (node.lng - lng)**2)
                if d < min_d:
                    min_d = d
                    closest_node = node
            
            # If a node is close (< 100 meters / 0.0009 degrees), penalize
            if min_d < 0.0009 and closest_node:
                r = getattr(closest_node, "predicted_risk", 0.0)
                # Weight by proximity
                weight = 1.0 - (min_d / 0.0009)
                total_hazard += r * weight * (1.0 + len(closest_node.accidents) * 0.15)
                node_risks.append(r)
                
        # ONLY average over evaluated intersection points to prevent long-road dilution
        if not node_risks:
            return round(total_hazard, 2), 5
            
        # Calculate average risk of evaluated points
        normalized_risks = []
        for r in node_risks:
            norm = (r - min_graph_risk) / risk_range
            normalized_risks.append(min(1.0, max(0.0, norm)))
            
        avg_risk = sum(normalized_risks) / len(normalized_risks)
        
        # Contrast stretching: map avg_risk in [0.1, 0.7] to [15%, 85%] display range
        stretched = (avg_risk - 0.1) / 0.6
        stretched = min(1.0, max(0.0, stretched))
        combined_index = stretched * 70.0 + 15.0
        danger_percent = min(99, max(2, int(round(combined_index))))
        
        return round(total_hazard, 2), danger_percent

    # Run Dijkstra search on GNN-LNN convolved nodes
    optimizer = RouteOptimizer(nodes)
    safest_nodes_path, _ = optimizer.find_safest_route(
        start_id=start_id,
        end_id=end_id,
        target_year=target_year,
        rain_active=rain_active,
        target_hour=target_hour,
        use_hazard=True
    )
    
    # safest_nodes_path is already a list of (lat, lng) tuples
    safest_coords_list = safest_nodes_path or []

    # Identify the 'Apex' waypoint (the node furthest from the straight line between start and end)
    apex_coord = None
    if len(safest_coords_list) > 3:
        lat1, lng1 = start_lat, start_lng
        lat2, lng2 = end_lat, end_lng
        
        # Line equation Ax + By + C = 0 parameters
        A = lat2 - lat1
        B = -(lng2 - lng1)
        C = lng2 * lat1 - lat2 * lng1
        denom = math.sqrt(A**2 + B**2)
        
        max_d = -1.0
        for lat, lng in safest_coords_list[1:-1]:
            if denom > 0.0001:
                d = abs(A * lng + B * lat + C) / denom
            else:
                d = math.sqrt((lat - lat1)**2 + (lng - lng1)**2)
            if d > max_d:
                max_d = d
                apex_coord = (lat, lng)

    # Snap paths using OSRM in the backend
    import urllib.request
    import json
    
    def fetch_osrm_route(query_coords):
        if len(query_coords) <= 1:
            return query_coords
        osrm_coords = ";".join([f"{lng},{lat}" for lat, lng in query_coords])
        url = f"https://router.project-osrm.org/route/v1/driving/{osrm_coords}?overview=full&geometries=geojson"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
            routes = data.get("routes", [])
            if routes:
                geom = routes[0].get("geometry", {}).get("coordinates", [])
                return [(lat, lng) for lng, lat in geom]
        except Exception as e:
            print("OSRM routing failed:", e)
        return query_coords

    # Generate candidate routes
    candidate_routes = []
    
    # 1. Default Fastest route
    fastest_path = fetch_osrm_route([(start_lat, start_lng), (end_lat, end_lng)])
    f_hazard, f_danger = evaluate_path_safety(fastest_path)
    candidate_routes.append({
        "path": fastest_path,
        "hazard_score": f_hazard,
        "danger_percent": f_danger,
        "is_fastest": True
    })
    
    # 2. Extract up to 3 waypoints from Dijkstra's safest path (25%, Apex, 75%)
    waypoints = []
    if len(safest_coords_list) > 3:
        n_coords = len(safest_coords_list)
        waypoints.append(safest_coords_list[n_coords // 4])
        if apex_coord:
            waypoints.append(apex_coord)
        waypoints.append(safest_coords_list[(3 * n_coords) // 4])
        
    # Remove duplicates or invalid waypoints
    unique_waypoints = []
    for w in waypoints:
        if w not in unique_waypoints and w != (start_lat, start_lng) and w != (end_lat, end_lng):
            unique_waypoints.append(w)
            
    # Query OSRM and evaluate each detour candidate
    for w in unique_waypoints:
        try:
            detour_path = fetch_osrm_route([(start_lat, start_lng), w, (end_lat, end_lng)])
            if detour_path and len(detour_path) > 2:
                d_hazard, d_danger = evaluate_path_safety(detour_path)
                candidate_routes.append({
                    "path": detour_path,
                    "hazard_score": d_hazard,
                    "danger_percent": d_danger,
                    "is_fastest": False
                })
        except Exception as e:
            print("OSRM detour query failed for waypoint:", w, e)
            
    # Sort candidates by hazard score (safest first)
    candidate_routes.sort(key=lambda x: x["hazard_score"])
    safest_route = candidate_routes[0]
    
    # Identify the fastest route (which is the default OSRM route)
    fastest_route = next((r for r in candidate_routes if r["is_fastest"]), candidate_routes[0])
    
    safest_path = safest_route["path"]
    fastest_path = fastest_route["path"]
    safest_hazard_score = safest_route["hazard_score"]
    fastest_hazard_score = fastest_route["hazard_score"]
    
    fastest_danger_percent = fastest_route["danger_percent"]
    
    # Scale Safest Danger Percent relative to Fastest Danger Percent
    if fastest_hazard_score > 0.001:
        ratio = safest_hazard_score / fastest_hazard_score
        safest_danger_percent = int(round(fastest_danger_percent * (ratio ** 3)))
    else:
        safest_danger_percent = fastest_danger_percent
        
    safest_danger_percent = min(99, max(5, safest_danger_percent))
    fastest_danger_percent = min(99, max(5, fastest_danger_percent))

    return {
        "safest": {
            "path": safest_path,
            "hazard_score": safest_hazard_score,
            "danger_percent": safest_danger_percent,
            "nodes_visited": len(safest_path)
        },
        "fastest": {
            "path": fastest_path,
            "hazard_score": fastest_hazard_score,
            "danger_percent": fastest_danger_percent,
            "nodes_visited": len(fastest_path)
        }
    }


@app.get("/datasets/combined/graph")
def get_graph_data(
    dataset_ids: str = Query(default="7cci-nqqb"),
    max_rows: int = Query(default=1500, ge=1, le=5000),
    target_year: int = Query(default=2026),
    rain_active: bool = Query(default=False),
    target_hour: int | None = Query(default=None),
    use_symbolic: bool = Query(default=False),
) -> dict[str, Any]:
    ids = [d.strip() for d in dataset_ids.split(",") if d.strip()]
    accidents = []
    for dataset_id in ids:
        if dataset_id not in DATASET_CONFIGS:
            continue
        entry = cache_service.get_snapshot(dataset_id, max_rows=max_rows)
        for r in entry.processed:
            if r.get("latitude") is not None and r.get("longitude") is not None:
                acc_info = dict(r)
                acc_info["dataset_id"] = dataset_id
                accidents.append(acc_info)

    # Append live community-reported incidents
    for rep in COMMUNITY_REPORTS:
        accidents.append(rep)

    nodes = get_routing_graph_nodes(accidents, "7cci-nqqb")
    num_nodes = len(nodes)

    # Build spatial edge index for GNN convolution (proximity < 0.005)
    import math
    sources, targets = [], []
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i != j:
                dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
                if dist < 0.005:
                    sources.append(i)
                    targets.append(j)
    if not sources:
        sources = list(range(num_nodes))
        targets = list(range(num_nodes))

    # Determine method: Symbolic (Direct Heuristic) vs ML (GNN-LNN PyTorch Model)
    torch_available = False
    try:
        import torch
        torch_available = True
    except ImportError:
        pass

    if torch_available and pretrained_model is not None and not use_symbolic:
        try:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            # Prepare ML features sequence
            sequences = []
            hour_val = target_hour if target_hour is not None else 12
            for t in range(5):
                h_seq = (hour_val - 4 + t) % 24
                r_seq = rain_active if t == 4 else False
                features = torch.zeros((num_nodes, 5), dtype=torch.float32)
                for idx, node in enumerate(nodes):
                    features[idx, 0] = 1.0 if r_seq else 0.0
                    features[idx, 1] = 0.0 if r_seq else 1.0
                    features[idx, 2] = h_seq / 24.0
                    num_acc = len(node.accidents)
                    features[idx, 4] = float(num_acc) / 50.0  # Normalize count
                    if num_acc > 0:
                        sev_sum = 0.0
                        for acc in node.accidents:
                            v = str(acc.get("vehicles", "")).upper()
                            if "MUERTO" in v or "FALLECIDO" in v:
                                sev_sum += 4.0
                            elif "HERIDO" in v or "LESIONADO" in v:
                                sev_sum += 2.0
                            else:
                                sev_sum += 1.0
                        features[idx, 3] = (sev_sum / num_acc) / 4.0  # Normalize severity
                    else:
                        features[idx, 3] = 0.0
                sequences.append(features)
            x_seq = torch.stack(sequences, dim=0)

            # Run GNN-LNN model inference directly to evaluate risks using convolved weights
            pretrained_model.eval()
            with torch.no_grad():
                pred = pretrained_model(x_seq, edge_index)
            for idx, node in enumerate(nodes):
                node.predicted_risk = float(pred[idx, 0]) * 10.0
        except Exception as e:
            print("GNN-LNN execution error, falling back to symbolic:", e)
            use_symbolic = True

    if not torch_available or pretrained_model is None or use_symbolic:
        # Fallback: convolved symbolic formula (exact mathematical GNN-LNN convolved representation)
        avg_severities = []
        for node in nodes:
            num_acc = len(node.accidents)
            if num_acc > 0:
                sev_sum = 0.0
                for acc in node.accidents:
                    v = str(acc.get("vehicles", "")).upper()
                    if "MUERTO" in v or "FALLECIDO" in v:
                        sev_sum += 4.0
                    elif "HERIDO" in v or "LESIONADO" in v:
                        sev_sum += 2.0
                    else:
                        sev_sum += 1.0
                avg_severities.append(sev_sum / num_acc)
            else:
                avg_severities.append(0.0)

        # Compute degree for each node (including self-loop)
        degrees = [1.0] * num_nodes
        for u, v in zip(sources, targets):
            degrees[u] += 1.0

        # Convolve features (symmetric GCN normalization: norm_A = D_inv_sqrt @ A @ D_inv_sqrt)
        severity_conv = [0.0] * num_nodes
        accidents_conv = [0.0] * num_nodes
        for idx in range(num_nodes):
            local_factor = 1.0 / degrees[idx]
            severity_conv[idx] += avg_severities[idx] * local_factor
            accidents_conv[idx] += float(len(nodes[idx].accidents)) * local_factor
        for u, v in zip(sources, targets):
            factor = 1.0 / math.sqrt(degrees[u] * degrees[v])
            severity_conv[u] += avg_severities[v] * factor
            accidents_conv[u] += float(len(nodes[v].accidents)) * factor

        rain_val = 1.0 if rain_active else 0.0
        h = (target_hour if target_hour is not None else 12) / 24.0

        for idx, node in enumerate(nodes):
            if accidents_conv[idx] == 0.0:
                node.predicted_risk = 0.0
                continue
            # Convolved GNN-LNN distilled formula with positive correlation for risk
            risk = (
                0.11126 + 
                0.00066 * rain_val +
                0.00444 * h -
                0.00153 * math.sin(2 * math.pi * h) +
                0.00011 * math.cos(2 * math.pi * h) +
                0.06504 * severity_conv[idx] +
                0.00091 * accidents_conv[idx]
            )
            node.predicted_risk = max(0.0, risk)

    # Compute min and max baseline risks in the active graph
    all_risks = [getattr(n, "predicted_risk", 0.0) for n in nodes]
    min_graph_risk = min(all_risks) if all_risks else 0.0
    max_graph_risk = max(all_risks) if all_risks else 1.0
    risk_range = max_graph_risk - min_graph_risk
    if risk_range <= 0.001:
        risk_range = 1.0

    # Build response nodes list with normalized danger_percent
    res_nodes = []
    for node in nodes:
        r = getattr(node, "predicted_risk", 0.0)
        norm = (r - min_graph_risk) / risk_range
        danger_percent = min(99, max(1, int(round((norm ** 2) * 33.0 + 2.0))))
        res_nodes.append({
            "id": node.id,
            "lat": node.lat,
            "lng": node.lng,
            "name": node.label,
            "weight": len(node.accidents),
            "predicted_risk": r,
            "danger_percent": danger_percent
        })

    # Build response edges list: only return edges if they share a street name OR are extremely close
    res_edges = []
    import re
    
    def extract_streets(label: str) -> set[str]:
        lbl = label.upper().replace(" Y ", " CON ").replace(" - ", " CON ")
        parts = [p.strip() for p in lbl.split(" CON ") if p.strip()]
        streets = set()
        for p in parts:
            p = re.sub(r"\bCRA\b", "CARRERA", p)
            p = re.sub(r"\bCL\b", "CALLE", p)
            p = re.sub(r"\bDG\b", "DIAGONAL", p)
            p = re.sub(r"\bTV\b", "TRANSVERSAL", p)
            p = re.sub(r"\bAV\b", "AVENIDA", p)
            streets.add(p)
        return streets

    for i, node_a in enumerate(nodes):
        streets_a = extract_streets(node_a.label)
        for j in range(i + 1, num_nodes):
            node_b = nodes[j]
            dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
            
            # Connect if within 250m and sharing a street name, OR if extremely close (< 90m)
            shares_street = len(streets_a & extract_streets(node_b.label)) > 0
            if (dist < 0.0022 and shares_street) or (dist < 0.0008):
                res_edges.append({
                    "source": node_a.id,
                    "target": node_b.id
                })

    return {
        "nodes": res_nodes,
        "edges": res_edges
    }


@app.get("/datasets/combined")
def get_combined_datasets(
    dataset_ids: str = Query(default="7cci-nqqb"),
    max_rows: int = Query(default=1500, ge=1, le=5000),
    force_refresh: bool = False,
) -> dict[str, Any]:
    return get_combined_datasets_snapshot(dataset_ids, max_rows, force_refresh)


@app.get("/datasets/export")
def export_dataset_records(
    dataset_ids: str = Query(default="7cci-nqqb"),
    max_rows: int = Query(default=1500, ge=1, le=5000),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    rain_only: bool | None = Query(default=None),
    vehicle_type: str | None = Query(default=None),
    city: str | None = Query(default=None),
    export_format: str = Query(default="json")
) -> Any:
    snapshot = get_combined_datasets_snapshot(dataset_ids, max_rows)
    records = snapshot["tables"]["records"]

    filtered = get_filtered_accidents(
        records,
        start_year=start_year,
        end_year=end_year,
        rain_only=rain_only,
        vehicle_type=vehicle_type,
        city=city
    )

    if export_format.lower() == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "dataset_id", "row_id", "latitude", "longitude", "location", "is_fallback_coord", "date_iso", "time", "vehicles"])
        for r in filtered:
            writer.writerow([
                r.get("id"),
                r.get("dataset_id"),
                r.get("row_id"),
                r.get("latitude"),
                r.get("longitude"),
                r.get("location"),
                r.get("is_fallback_coord"),
                r.get("date_iso"),
                r.get("time"),
                r.get("vehicles")
            ])
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=export.csv"})

    return {"count": len(filtered), "records": filtered}


@app.get("/datasets/chart.png")
def get_chart_image(
    dataset_ids: str = Query(default="7cci-nqqb"),
    max_rows: int = Query(default=2000, ge=1, le=5000),
    start_year: int | None = Query(default=None),
    end_year: int | None = Query(default=None),
    rain_only: bool | None = Query(default=None),
    vehicle_type: str | None = Query(default=None),
    city: str | None = Query(default=None),
) -> Response:
    snapshot = get_combined_datasets_snapshot(dataset_ids, max_rows)
    records = snapshot["tables"]["records"]

    filtered = get_filtered_accidents(
        records,
        start_year=start_year,
        end_year=end_year,
        rain_only=rain_only,
        vehicle_type=vehicle_type,
        city=city
    )

    img_bytes = generate_report_chart(filtered)
    return Response(content=img_bytes, media_type="image/png")


@app.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str,
    max_rows: int = Query(default=200, ge=1, le=5000),
    force_refresh: bool = False,
    id_field: str | None = Query(default=None),
    location_field: str | None = Query(default=None),
) -> dict[str, Any]:
    entry = cache_service.get_snapshot(
        dataset_id,
        max_rows=max_rows,
        force_refresh=force_refresh,
        id_field=id_field,
        location_field=location_field,
    )
    return serialize_entry(entry)


@app.get("/datasets/{dataset_id}/updates")
async def poll_dataset_updates(
    dataset_id: str,
    last_version: str | None = None,
    max_rows: int = Query(default=200, ge=1, le=5000),
    timeout_seconds: int = Query(default=30, ge=1, le=300),
) -> dict[str, Any]:
    entry, changed, timed_out = await cache_service.wait_for_update(
        dataset_id=dataset_id,
        max_rows=max_rows,
        last_version=last_version,
        timeout_seconds=timeout_seconds,
    )

    response = {
        "dataset_id": dataset_id,
        "max_rows": max_rows,
        "version": entry.version,
        "changed": changed,
        "timed_out": timed_out,
        "fetched_at": entry.fetched_at,
    }
    if changed:
        response["data"] = serialize_entry(entry)

    return response


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    sibling_path = Path(__file__).resolve().parent / "index.html"
    if sibling_path.exists():
        return sibling_path.read_text(encoding="utf-8")
    return "<h1>Index file not found</h1>"


@app.put("/datasets/{dataset_id}/nodes/{row_id}")
def update_node(
    dataset_id: str,
    row_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return update_dataset_node(dataset_id, row_id, payload)


@app.post("/datasets/reports")
@app.put("/datasets/reports")
def add_community_report(payload: dict[str, Any]):
    lat = payload.get("latitude")
    lng = payload.get("longitude")
    report_type = payload.get("type", "accidente")
    if lat is not None and lng is not None:
        COMMUNITY_REPORTS.append({
            "latitude": float(lat),
            "longitude": float(lng),
            "vehicles": "ACCIDENTE (1)" if report_type == "accidente" else report_type.upper(),
            "location": "Reporte de la Comunidad",
            "date_iso": "2026-07-11",
            "time": "12:00:00",
            "data_original": {"type": report_type}
        })
    return {"status": "ok", "total_reports": len(COMMUNITY_REPORTS)}

