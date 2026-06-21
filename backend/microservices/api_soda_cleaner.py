import re
import unicodedata
from datetime import datetime
from typing import Any, Dict

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from time import time
from typing import Callable, List

from external.pipeline import process_rows
from external.soda_client import SodaClient
from microservices.mapper import resolve_coordinates


DATASET_CONFIGS = {
    "stq8-drvp": {
        "id_field": "comparendo",
        "location_field": "lugar",
        "table_name": "comparendos",
        "clean_table_name": "comparendos_limpios",
        "fk_field": "comparendo_id",
    },
    "7cci-nqqb": {
        "id_field": "orden",
        "location_field": "barrio",
        "table_name": "accidentes",
        "clean_table_name": "accidentes_limpios",
        "fk_field": "accidente_id",
    },
    "3v2w-chcq": {
        "id_field": "CODIGO_ACCIDENTE",
        "location_field": "DIRECCION",
        "latitude_field": "LATITUD",
        "longitude_field": "LONGITUD",
        "table_name": "siniestros_bogota",
        "clean_table_name": "siniestros_bogota_limpios",
        "fk_field": "siniestro_id",
    },
    "sjpx-eqfp": {
        "id_field": "ipat",
        "location_field": "direccion",
        "latitude_field": "lat",
        "longitude_field": "long",
        "table_name": "siniestros_palmira",
        "clean_table_name": "siniestros_palmira_limpios",
        "fk_field": "siniestro_id",
    },
    "dr5c-eewa": {
        "id_field": "id_mt",
        "location_field": "tramo",
        "latitude_field": "latitud",
        "longitude_field": "longitud",
        "table_name": "mortalidad_vias",
        "clean_table_name": "mortalidad_vias_limpios",
        "fk_field": "mortalidad_id",
    },
    "sefb-a755": {
        "id_field": "fecha_accidente",
        "location_field": "sitio_exacto_accidente",
        "table_name": "accidentes_calle30",
        "clean_table_name": "accidentes_calle30_limpios",
        "fk_field": "accidente_c30_id",
    },
    "ixgc-yijx": {
        "id_field": "Fecha",
        "location_field": "Dirección_Reporte",
        "table_name": "lesionados_cali",
        "clean_table_name": "lesionados_cali_limpios",
        "fk_field": "lesionado_id",
    },
}
DEFAULT_CONFIG = {
    "id_field": "comparendo",
    "location_field": "lugar",
    "table_name": "comparendos",
    "clean_table_name": "comparendos_limpios",
    "fk_field": "comparendo_id",
}


def normalize_text(text: str) -> str:
    """Normaliza texto para facilitar reglas de extracción."""
    raw = text or ""
    raw = raw.strip().upper()
    raw = "".join(
        ch for ch in unicodedata.normalize("NFD", raw) if unicodedata.category(ch) != "Mn"
    )
    raw = re.sub(r"\s+", " ", raw)
    return raw


def clean_location_text(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\bNRO\b|\bNUMERO\b", "#", cleaned)
    cleaned = re.sub(r"\bNO\b", "#", cleaned)
    cleaned = re.sub(r"\s*#\s*", " # ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_date(text: str) -> str:
    candidate = _normalize_whitespace(text)
    if not candidate:
        return candidate

    try:
        dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        return candidate


def clean_soda_value(key: str, value: Any) -> Any:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    cleaned = _normalize_whitespace(value)
    if not cleaned:
        return None

    if cleaned.upper() in {"N/A", "NA", "NULL", "NONE", "-", "--", "---"}:
        return None

    lowered_key = key.lower()
    if lowered_key == "lugar":
        return clean_location_text(cleaned)
    if "fecha" in lowered_key:
        return _normalize_date(cleaned)

    return cleaned


def clean_soda_row(row: dict[str, Any], location_field: str = "lugar") -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if key == location_field and key != "lugar":
            cleaned = clean_soda_value("lugar", value)
        else:
            cleaned = clean_soda_value(key, value)
        result[key] = cleaned
    return result


def normalize_row_features(row: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Helper method to parse raw dates, times, and vehicles from SODA schemas."""
    config = DATASET_CONFIGS.get(dataset_id, DEFAULT_CONFIG)
    
    loc_field = config["location_field"]
    raw_location = row.get(loc_field, "")
    
    if dataset_id == "7cci-nqqb" and row.get("nombrecomuna"):
        raw_location = f"{raw_location}, {row.get('nombrecomuna')}"
        
    lat_field = config.get("latitude_field")
    lng_field = config.get("longitude_field")
    
    from microservices.mapper import parse_coordinate
    lat = parse_coordinate(row.get(lat_field)) if lat_field else None
    lng = parse_coordinate(row.get(lng_field)) if lng_field else None
    
    date_val = ""
    time_val = ""
    
    if dataset_id == "stq8-drvp":
        date_val = row.get("fecha") or row.get("fecha_hora") or ""
    elif dataset_id == "7cci-nqqb":
        date_val = row.get("fecha") or ""
        time_val = row.get("hora") or ""
    elif dataset_id == "3v2w-chcq":
        ts = row.get("FECHA_OCURRENCIA_ACC") or row.get("FECHA_ACCIDENTE") or row.get("FECHA_HORA_ACC")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts / 1000.0)
            date_val = dt.date().isoformat()
            if ts % 86400000 != 0:
                time_val = dt.time().isoformat()
        else:
            date_val = row.get("FECHA") or ""
            
        if "/" in str(date_val):
            parts = str(date_val).split("/")
            if len(parts) == 3:
                d, m, y = parts
                date_val = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                
        t_field = row.get("HORA_ACCIDENTE") or row.get("HORA") or ""
        if t_field and t_field != row.get("FECHA"):
            time_val = t_field
    elif dataset_id == "sjpx-eqfp":
        date_val = row.get("fecha") or ""
        time_val = row.get("hora") or ""
    elif dataset_id == "dr5c-eewa":
        date_val = "2015-2019"
        time_val = "NO REGISTRA"
    elif dataset_id == "sefb-a755":
        date_val = row.get("fecha_accidente") or ""
        time_val = row.get("hora_accidente") or ""
    elif dataset_id == "ixgc-yijx":
        date_val = row.get("Fecha") or ""
        time_val = row.get("Hora") or ""
        
    normalized_date = ""
    if date_val:
        if "/" in str(date_val):
            parts = str(date_val).split("T")[0].split(" ")[0].split("/")
            if len(parts) == 3:
                d, m, y = parts
                normalized_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        
        if not normalized_date:
            match = re.search(r"(\d{4}-\d{2}-\d{2})", str(date_val))
            if match:
                normalized_date = match.group(1)
            else:
                normalized_date = str(date_val).strip()

    normalized_time = ""
    if time_val:
        t_str = str(time_val).strip().lower()
        if "am" in t_str or "pm" in t_str:
            cleaned_t = t_str.replace("am", " am").replace("pm", " pm").replace("::", ":").replace(":", " ")
            parts = cleaned_t.split()
            if len(parts) >= 2:
                hms = parts[0].split(":")
                meridiem = parts[-1]
                if len(hms) >= 2:
                    h, m = int(hms[0]), int(hms[1])
                    if meridiem == "pm" and h < 12:
                        h += 12
                    elif meridiem == "am" and h == 12:
                        h = 0
                    normalized_time = f"{str(h).zfill(2)}:{str(m).zfill(2)}"
        if not normalized_time:
            match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", str(time_val))
            if match:
                normalized_time = match.group(1)
            else:
                normalized_time = str(time_val).strip()
            
    vehicles = []
    if dataset_id == "7cci-nqqb":
        vehicle_types = ["automovil", "campero", "camioneta", "micro", "buseta", "bus", "camion", "volqueta", "moto", "bicicleta", "peaton"]
        for vt in vehicle_types:
            count_str = row.get(vt)
            try:
                if count_str and int(count_str) > 0:
                    vehicles.append(f"{vt.upper()} ({count_str})")
            except (ValueError, TypeError):
                pass
    elif dataset_id == "3v2w-chcq":
        cant = row.get("CANTIDAD_VEHICULOS")
        if cant is not None and str(cant).strip():
            vehicles.append(f"TOTAL: {cant}")
        
        for k, v in row.items():
            if "VEHICULO" in k.upper() and v and str(v).strip():
                vehicles.append(str(v).strip().upper())
    elif dataset_id == "sjpx-eqfp":
        cond = row.get("condicion_de_la_victima")
        if cond and str(cond).strip():
            vehicles.append(str(cond).strip().upper())
    elif dataset_id == "dr5c-eewa":
        fallecidos = row.get("fallecidos")
        if fallecidos is not None and str(fallecidos).strip():
            vehicles.append(f"FALLECIDOS: {fallecidos}")
    elif dataset_id == "sefb-a755":
        clase = row.get("clase_accidente")
        grav = row.get("gravedad_accidente")
        parts_info = []
        if clase:
            parts_info.append(clase.upper())
        if grav:
            parts_info.append(grav.upper())
        if parts_info:
            vehicles.append(" - ".join(parts_info))
    elif dataset_id == "ixgc-yijx":
        actor = row.get("Tipo_Confirmado")
        if isinstance(actor, list):
            actor = ", ".join(actor)
        if actor:
            vehicles.append(str(actor).upper())
        for col in ["Automovil", "Moto", "Ciclista", "Peaton"]:
            val = row.get(col)
            if val and val != "." and val != "":
                vehicles.append(f"{col.upper()}: {val}")
            
    if not vehicles:
        vehicles = ["NO REGISTRA"]
        
    return {
        "raw_location": raw_location,
        "lat": lat,
        "lng": lng,
        "date_iso": normalized_date,
        "time": normalized_time,
        "vehicles": ", ".join(vehicles),
    }


def _load_edits(dataset_id: str) -> dict[str, Any]:
    path = Path("data") / f"edits_{dataset_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_edits(dataset_id: str, edits: dict[str, Any]) -> None:
    path = Path("data") / f"edits_{dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8")


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

    def refresh_dataset(
        self,
        dataset_id: str,
        max_rows: int,
        id_field: str | None = None,
        location_field: str | None = None,
        force_refresh: bool = False,
    ) -> CacheEntry:
        config = DATASET_CONFIGS.get(dataset_id, DEFAULT_CONFIG)
        effective_id = id_field or config["id_field"]
        effective_loc = location_field or config["location_field"]

        import os
        is_testing = "PYTEST_CURRENT_TEST" in os.environ

        # Check local disk cache first for speed optimization
        raw_cache_path = Path("data") / f"raw_{dataset_id}.json"
        rows = None
        if not is_testing and not force_refresh and raw_cache_path.exists():
            try:
                rows = json.loads(raw_cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if rows is None:
            # Intercept Bogota dataset to query ArcGIS feature server
            if dataset_id == "3v2w-chcq":
                import urllib.request
                url = f"https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services/HistoricoSiniestros/FeatureServer/0/query?where=1%3D1&outFields=%2A&resultRecordCount={max_rows}&f=json"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response:
                    res = json.loads(response.read().decode())
                    rows = [f["attributes"] for f in res.get("features", [])]
            elif dataset_id == "ixgc-yijx":
                # Download and parse Cali CSV
                import urllib.request
                import csv
                import codecs
                url = "https://datos.cali.gov.co/dataset/75c089ba-7df3-4816-b80f-c69c6e5362ae/resource/b5e009ef-8739-487d-bb0a-ffab613ce5cb/download/lesionados-en-santiago-de-cali-del-2016-2025.csv"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response:
                    reader = csv.DictReader(codecs.iterdecode(response, 'utf-8', errors='ignore'), delimiter=';')
                    rows = []
                    for r in reader:
                        rows.append(dict(r))
                        if len(rows) >= max_rows:
                            break
            else:
                soda_id = "rs3u-8r4q" if dataset_id == "dr5c-eewa" else ("yb9r-2dsi" if dataset_id == "sefb-a755" else dataset_id)
                client = self.client_factory(soda_id)
                rows = client.fetch_rows(select="*", where=None, limit=1000, max_rows=max_rows)
            
            # Write to disk cache
            if not is_testing:
                try:
                    raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_cache_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

        normalized_data = [normalize_row_features(row, dataset_id) for row in rows]
        virtual_rows = []
        for row, norm in zip(rows, normalized_data):
            v_row = dict(row)
            v_row[effective_loc] = norm["raw_location"]
            virtual_rows.append(v_row)

        processed = process_rows(virtual_rows, id_field=effective_id, location_field=effective_loc)
        for row, norm in zip(processed, normalized_data):
            row["latitude"] = norm["lat"]
            row["longitude"] = norm["lng"]
            row["date_iso"] = norm["date_iso"]
            row["time"] = norm["time"]
            row["vehicles"] = norm["vehicles"]

        # Apply local edits/geolocations
        edits = _load_edits(dataset_id)
        for row in processed:
            row_id = str(row["id"])
            if row_id in edits:
                row_edit = edits[row_id]
                if "data_limpia" in row_edit:
                    row["data_limpia"] = row_edit["data_limpia"]
                if "extraccion" in row_edit:
                    row["extraccion"] = row_edit["extraccion"]
                row["latitude"] = row_edit.get("latitude")
                row["longitude"] = row_edit.get("longitude")
                row["is_fallback_coord"] = False
            else:
                lat_res, lng_res, is_fb = resolve_coordinates(
                    row_id,
                    row.get("latitude"),
                    row.get("longitude"),
                    row.get("extraccion"),
                    dataset_id
                )
                row["latitude"] = lat_res
                row["longitude"] = lng_res
                row["is_fallback_coord"] = is_fb

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

    def get_snapshot(
        self,
        dataset_id: str,
        max_rows: int,
        force_refresh: bool = False,
        id_field: str | None = None,
        location_field: str | None = None,
    ) -> CacheEntry:
        key = self._cache_key(dataset_id, max_rows)
        with self._lock:
            entry = self._entries.get(key)

        if force_refresh or entry is None or self._is_stale(entry):
            return self.refresh_dataset(dataset_id, max_rows, id_field=id_field, location_field=location_field, force_refresh=force_refresh)

        return entry

    async def wait_for_update(
        self,
        dataset_id: str,
        max_rows: int,
        last_version: str | None,
        timeout_seconds: int = 30,
    ) -> tuple[CacheEntry, bool, bool]:
        import asyncio
        deadline = time() + timeout_seconds

        while True:
            entry = self.get_snapshot(dataset_id, max_rows)
            if last_version is None or entry.version != last_version:
                return entry, True, False

            remaining = deadline - time()
            if remaining <= 0:
                return entry, False, True

            await asyncio.sleep(min(self.poll_interval_seconds, remaining))


def serialize_entry(entry: CacheEntry) -> dict[str, Any]:
    payload = asdict(entry)
    relational_rows = []
    cleaned_rows = []
    extraction_rows = []

    config = DATASET_CONFIGS.get(entry.dataset_id, DEFAULT_CONFIG)
    table_name = config["table_name"]
    clean_table_name = config["clean_table_name"]
    fk_field = config["fk_field"]
    loc_field = config["location_field"]

    for index, row in enumerate(entry.processed, start=1):
        orig_loc = row.get("data_original", {}).get(loc_field, "")
        clean_loc = row.get("data_limpia", {}).get(loc_field, "")

        relational_rows.append(
            {
                "id": index,
                "dataset_id": entry.dataset_id,
                "row_id": row.get("id"),
                "data_original": row.get("data_original"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "location": orig_loc,
                "is_fallback_coord": row.get("is_fallback_coord", False),
                "date_iso": row.get("date_iso", ""),
                "time": row.get("time", ""),
                "vehicles": row.get("vehicles", "NO REGISTRA"),
                config["id_field"]: row.get("id"),
            }
        )
        cleaned_rows.append(
            {
                "id": index,
                fk_field: index,
                "record_id": index,
                "data_limpia": row.get("data_limpia"),
                "clean_location": clean_loc,
            }
        )
        extraction_rows.append(
            {
                "id": index,
                fk_field: index,
                "record_id": index,
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
        table_name: relational_rows,
        clean_table_name: cleaned_rows,
        "extracciones": extraction_rows,
        "records": relational_rows,
        "records_cleaned": cleaned_rows,
    }
    return payload


cache_service = DatasetCacheService()


def update_dataset_node(dataset_id: str, row_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    edits = _load_edits(dataset_id)
    edits[row_id] = {
        "data_limpia": payload.get("data_limpia"),
        "extraccion": payload.get("extraccion"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
    }
    _save_edits(dataset_id, edits)

    # Invalidate/refresh all active cache entries for this dataset
    keys_to_refresh = [key for key in cache_service._entries if key.startswith(f"{dataset_id}:")]
    for key in keys_to_refresh:
        _, max_rows = key.split(":")
        cache_service.refresh_dataset(dataset_id, max_rows=int(max_rows), force_refresh=True)

    return {"status": "success", "updated_row_id": row_id}


def get_combined_datasets_snapshot(dataset_ids: str, max_rows: int, force_refresh: bool = False) -> dict[str, Any]:
    ids = [d.strip() for d in dataset_ids.split(",") if d.strip()]
    
    combined_records = []
    combined_cleaned = []
    combined_extractions = []
    
    fetched_at = time()
    versions = []
    
    global_index = 1
    for dataset_id in ids:
        if dataset_id not in DATASET_CONFIGS:
            continue
        entry = cache_service.get_snapshot(
            dataset_id,
            max_rows=max_rows,
            force_refresh=force_refresh,
        )
        config = DATASET_CONFIGS[dataset_id]
        loc_field = config["location_field"]
        fk_field = config["fk_field"]
        
        versions.append(entry.version)
        
        for row in entry.processed:
            orig_loc = row.get("data_original", {}).get(loc_field, "")
            clean_loc = row.get("data_limpia", {}).get(loc_field, "")
            
            combined_records.append(
                {
                    "id": global_index,
                    "dataset_id": dataset_id,
                    "row_id": f"{dataset_id}:{row.get('id')}",
                    "data_original": row.get("data_original"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "location": orig_loc,
                    "is_fallback_coord": row.get("is_fallback_coord", False),
                    "date_iso": row.get("date_iso", ""),
                    "time": row.get("time", ""),
                    "vehicles": row.get("vehicles", "NO REGISTRA"),
                }
            )
            combined_cleaned.append(
                {
                    "id": global_index,
                    fk_field: global_index,
                    "record_id": global_index,
                    "data_limpia": row.get("data_limpia"),
                    "clean_location": clean_loc,
                }
            )
            combined_extractions.append(
                {
                    "id": global_index,
                    fk_field: global_index,
                    "record_id": global_index,
                    "extraccion": row.get("extraccion"),
                }
            )
            global_index += 1
            
    combined_version = hashlib.sha256(",".join(versions).encode("utf-8")).hexdigest()
    
    return {
        "dataset_id": "combined",
        "max_rows": max_rows,
        "version": combined_version,
        "fetched_at": fetched_at,
        "cached": True,
        "tables": {
            "dataset_metadata": [
                {
                    "dataset_id": "combined",
                    "max_rows": max_rows,
                    "version": combined_version,
                    "fetched_at": fetched_at,
                }
            ],
            "records": combined_records,
            "records_cleaned": combined_cleaned,
            "extracciones": combined_extractions,
        }
    }
