from __future__ import annotations
from pathlib import Path
from typing import Any

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets/combined/route")
def get_safest_route(
    dataset_ids: str = Query(default="stq8-drvp,7cci-nqqb,3v2w-chcq,sjpx-eqfp,dr5c-eewa"),
    max_rows: int = Query(default=200, ge=1, le=5000),
    start_id: str = Query(...),
    end_id: str = Query(...),
    target_year: int = Query(default=2026),
    rain_active: bool = Query(default=False),
    target_hour: int | None = Query(default=None)
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

    optimizer = RouteOptimizer(nodes)
    path, hazard = optimizer.find_safest_route(
        start_id=start_id,
        end_id=end_id,
        target_year=target_year,
        rain_active=rain_active,
        target_hour=target_hour
    )

    return {
        "path": path,
        "hazard_score": hazard,
        "nodes_visited": len(path)
    }


@app.get("/datasets/combined")
def get_combined_datasets(
    dataset_ids: str = Query(default="stq8-drvp,7cci-nqqb,3v2w-chcq,sjpx-eqfp,dr5c-eewa"),
    max_rows: int = Query(default=200, ge=1, le=5000),
    force_refresh: bool = False,
) -> dict[str, Any]:
    return get_combined_datasets_snapshot(dataset_ids, max_rows, force_refresh)


@app.get("/datasets/export")
def export_dataset_records(
    dataset_ids: str = Query(default="stq8-drvp,7cci-nqqb,3v2w-chcq,sjpx-eqfp,dr5c-eewa"),
    max_rows: int = Query(default=2000, ge=1, le=5000),
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
    dataset_ids: str = Query(default="stq8-drvp,7cci-nqqb,3v2w-chcq,sjpx-eqfp,dr5c-eewa"),
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

