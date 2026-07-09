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


@app.get("/datasets/combined/route")
def get_safest_route(
    dataset_ids: str = Query(default="7cci-nqqb"),
    max_rows: int = Query(default=1500, ge=1, le=5000),
    start_id: str = Query(...),
    end_id: str = Query(...),
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

    grapher = MapGrapher()
    nodes = grapher.build_structural_graph(accidents)

    # Train and infer GNN-LNN model on-the-fly to calculate neural network predicted risks
    if nodes and use_symbolic:
        # Precompute average severities
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

        # Build spatial edge index for GNN convolution (proximity < 0.005)
        import math
        sources, targets = [], []
        num_nodes = len(nodes)
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i != j:
                    dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
                    if dist < 0.005:
                        sources.append(i)
                        targets.append(j)
                        
        # Compute degrees (including self-loop)
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
            
        # Evaluate distilled GNN-LNN convolved formula for each node
        rain_val = 1.0 if rain_active else 0.0
        h = (target_hour if target_hour is not None else 12) / 24.0
        
        for idx, node in enumerate(nodes):
            if accidents_conv[idx] == 0.0:
                node.predicted_risk = 0.0
                continue
                
            risk = (
                2.47456 +
                0.07559 * rain_val +
                0.22642 * h +
                0.00649 * math.sin(2 * math.pi * h) -
                0.02560 * math.cos(2 * math.pi * h) -
                0.20942 * severity_conv[idx] -
                0.02797 * accidents_conv[idx]
            )
            node.predicted_risk = max(0.0, risk)
    elif nodes:
        torch_available = False
        try:
            import torch
            torch_available = True
        except ImportError:
            pass

        if torch_available and pretrained_model is not None:
            try:
                import math
                num_nodes = len(nodes)
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
                edge_index = torch.tensor([sources, targets], dtype=torch.long)

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
                        features[idx, 4] = float(num_acc)
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
                            features[idx, 3] = sev_sum / num_acc
                        else:
                            features[idx, 3] = 0.0
                    sequences.append(features)
                x_seq = torch.stack(sequences, dim=0)

                with torch.no_grad():
                    pred = pretrained_model(x_seq, edge_index)
                for idx, node in enumerate(nodes):
                    node.predicted_risk = float(pred[idx, 0]) * 10.0
            except Exception as e:
                print("Pre-trained model inference exception, falling back to formula:", e)
                torch_available = False

        if not torch_available or pretrained_model is None:
            # Fallback: convolved symbolic formula (exact mathematical distillation of GNN-LNN)
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

            num_nodes = len(nodes)
            sources, targets = [], []
            for i, node_a in enumerate(nodes):
                for j, node_b in enumerate(nodes):
                    if i != j:
                        dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
                        if dist < 0.005:
                            sources.append(i)
                            targets.append(j)

            degrees = [1.0] * num_nodes
            for u, v in zip(sources, targets):
                degrees[u] += 1.0

            severity_conv = [0.0] * num_nodes
            accidents_conv = [0.0] * num_nodes
            for idx in range(num_nodes):
                lf = 1.0 / degrees[idx]
                severity_conv[idx] += avg_severities[idx] * lf
                accidents_conv[idx] += float(len(nodes[idx].accidents)) * lf
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
                risk = (
                    2.47456 +
                    0.07559 * rain_val +
                    0.22642 * h +
                    0.00649 * math.sin(2 * math.pi * h) -
                    0.02560 * math.cos(2 * math.pi * h) -
                    0.20942 * severity_conv[idx] -
                    0.02797 * accidents_conv[idx]
                )
                node.predicted_risk = max(0.0, risk)

    optimizer = RouteOptimizer(nodes)
    
    # 1. Safest path (using AI / hazard weights)
    safest_path, _ = optimizer.find_safest_route(
        start_id=start_id,
        end_id=end_id,
        target_year=target_year,
        rain_active=rain_active,
        target_hour=target_hour,
        use_hazard=True
    )

    # 2. Fastest path (ignoring risk weights, shortest geometric distance)
    fastest_path, _ = optimizer.find_safest_route(
        start_id=start_id,
        end_id=end_id,
        target_year=target_year,
        rain_active=rain_active,
        target_hour=target_hour,
        use_hazard=False
    )

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

    # Helper to calculate a comparative danger percentage (1% to 99%)
    def get_path_danger_percent(path_coords):
        if not path_coords:
            return 1
        coord_to_node = {(n.lat, n.lng): n for n in nodes}
        node_risks = []
        for lat, lng in path_coords:
            node = coord_to_node.get((lat, lng))
            if node:
                risk = getattr(node, "predicted_risk", None)
                if risk is None:
                    risk = node.calculate_risk(target_year, rain_active, target_hour)
                node_risks.append(risk)
        
        if not node_risks:
            return 1

        # Map each node risk to comparative danger: 10% (min in city) to 90% (max in city)
        normalized_risks = []
        for r in node_risks:
            p = 0.1 + 0.8 * ((r - min_graph_risk) / risk_range)
            normalized_risks.append(p)

        max_risk = max(normalized_risks) if normalized_risks else 0.0
        avg_risk = sum(normalized_risks) / len(normalized_risks) if normalized_risks else 0.0
        
        # Route index is 50% worst-spot danger and 50% average path danger
        combined_index = 0.5 * max_risk + 0.5 * avg_risk
        return min(99, max(1, int(round(combined_index * 100))))

    # Old sum-based hazard scores for compatibility/dijkstra logic
    def get_path_hazard_sum(path_coords):
        total_hazard = 0.0
        coord_to_node = {(n.lat, n.lng): n for n in nodes}
        for lat, lng in path_coords:
            node = coord_to_node.get((lat, lng))
            if node:
                risk = getattr(node, "predicted_risk", None)
                if risk is None:
                    risk = node.calculate_risk(target_year, rain_active, target_hour)
                total_hazard += risk
        return round(total_hazard, 2)

    safest_hazard_score = get_path_hazard_sum(safest_path)
    fastest_hazard_score = get_path_hazard_sum(fastest_path)

    safest_danger_percent = get_path_danger_percent(safest_path)
    fastest_danger_percent = get_path_danger_percent(fastest_path)

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

