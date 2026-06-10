from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from threading import Lock
from time import monotonic, time
from typing import Any, Callable

from fastapi import FastAPI, Query

from .pipeline import process_rows
from .soda_client import SodaClient


@dataclass
class CacheEntry:
    dataset_id: str
    max_rows: int
    version: str
    fetched_at: float
    rows: list[dict[str, Any]]
    processed: list[dict[str, Any]]


class DatasetCacheService:
    def __init__(
        self,
        refresh_interval_seconds: int = 60,
        poll_interval_seconds: float = 2.0,
        client_factory: Callable[[str], SodaClient] | None = None,
    ) -> None:
        self.refresh_interval_seconds = refresh_interval_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.client_factory = client_factory or SodaClient
        self._entries: dict[str, CacheEntry] = {}
        self._lock = Lock()

    def _cache_key(self, dataset_id: str, max_rows: int) -> str:
        return f"{dataset_id}:{max_rows}"

    def _is_stale(self, entry: CacheEntry) -> bool:
        return (time() - entry.fetched_at) >= self.refresh_interval_seconds

    def _build_version(self, processed: list[dict[str, Any]]) -> str:
        payload = json.dumps(processed, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def refresh_dataset(self, dataset_id: str, max_rows: int) -> CacheEntry:
        client = self.client_factory(dataset_id)
        rows = client.fetch_rows(select="*", where=None, limit=1000, max_rows=max_rows)
        processed = process_rows(rows)
        entry = CacheEntry(
            dataset_id=dataset_id,
            max_rows=max_rows,
            version=self._build_version(processed),
            fetched_at=time(),
            rows=rows,
            processed=processed,
        )

        with self._lock:
            self._entries[self._cache_key(dataset_id, max_rows)] = entry

        return entry

    def get_snapshot(self, dataset_id: str, max_rows: int, force_refresh: bool = False) -> CacheEntry:
        key = self._cache_key(dataset_id, max_rows)
        with self._lock:
            entry = self._entries.get(key)

        if force_refresh or entry is None or self._is_stale(entry):
            return self.refresh_dataset(dataset_id, max_rows)

        return entry

    async def wait_for_update(
        self,
        dataset_id: str,
        max_rows: int,
        last_version: str | None,
        timeout_seconds: int = 30,
    ) -> tuple[CacheEntry, bool, bool]:
        deadline = monotonic() + timeout_seconds

        while True:
            entry = self.get_snapshot(dataset_id, max_rows)
            if last_version is None or entry.version != last_version:
                return entry, True, False

            remaining = deadline - monotonic()
            if remaining <= 0:
                return entry, False, True

            await asyncio.sleep(min(self.poll_interval_seconds, remaining))


def serialize_entry(entry: CacheEntry) -> dict[str, Any]:
    payload = asdict(entry)
    relational_rows = []
    cleaned_rows = []
    extraction_rows = []

    for index, row in enumerate(entry.processed, start=1):
        relational_rows.append(
            {
                "id": index,
                "dataset_id": entry.dataset_id,
                "comparendo": row.get("comparendo"),
                "data_original": row.get("data_original"),
            }
        )
        cleaned_rows.append(
            {
                "id": index,
                "comparendo_id": index,
                "data_limpia": row.get("data_limpia"),
            }
        )
        extraction_rows.append(
            {
                "id": index,
                "comparendo_id": index,
                "extraccion": row.get("extraccion"),
            }
        )

    payload["cached"] = True
    payload["tables"] = {
        "dataset_metadata": [
            {
                "dataset_id": entry.dataset_id,
                "max_rows": entry.max_rows,
                "version": entry.version,
                "fetched_at": entry.fetched_at,
            }
        ],
        "comparendos": relational_rows,
        "comparendos_limpios": cleaned_rows,
        "extracciones": extraction_rows,
    }
    return payload


app = FastAPI(title="Safeway API", version="1.0.0")
cache_service = DatasetCacheService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str,
    max_rows: int = Query(default=200, ge=1, le=5000),
    force_refresh: bool = False,
) -> dict[str, Any]:
    entry = cache_service.get_snapshot(dataset_id, max_rows=max_rows, force_refresh=force_refresh)
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