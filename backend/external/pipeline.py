from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from microservices.mapper import AddressExtractor
from external.soda_client import SodaClient


def process_rows(
    rows: list[dict[str, Any]],
    extractor: AddressExtractor | None = None,
    id_field: str = "comparendo",
    location_field: str = "lugar",
) -> list[dict[str, Any]]:
    from microservices.api_soda_cleaner import clean_soda_row
    extractor = extractor or AddressExtractor()

    processed: list[dict[str, Any]] = []
    for row in rows:
        cleaned_row = clean_soda_row(row, location_field=location_field)
        cleaned_location = str(cleaned_row.get(location_field) or "")
        extraction = extractor.extract(cleaned_location)
        if isinstance(extraction, list) and "UNKNOWN" in extraction:
            continue
        processed.append(
            {
                "id": row.get(id_field),
                "data_original": row,
                "data_limpia": cleaned_row,
                "extraccion": extraction,
            }
        )

    return processed


def run_pipeline(
    dataset_id: str,
    max_rows: int,
    output_file: Path,
    id_field: str = "comparendo",
    location_field: str = "lugar",
) -> list[dict[str, Any]]:
    client = SodaClient(dataset_id=dataset_id)
    extractor = AddressExtractor()

    rows = client.fetch_rows(
        select="*",
        where=None,
        limit=1000,
        max_rows=max_rows,
    )
    processed = process_rows(rows, extractor=extractor, id_field=id_field, location_field=location_field)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline SODA -> limpieza -> extracción")
    parser.add_argument("--dataset-id", default="stq8-drvp")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--output", default="outputs/extracciones.json")
    parser.add_argument("--id-field", default="comparendo", help="Campo usado como identificador de fila")
    parser.add_argument("--location-field", default="lugar", help="Campo de texto de ubicacion a extraer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = Path(args.output)
    rows = run_pipeline(
        args.dataset_id,
        args.max_rows,
        output_file,
        id_field=args.id_field,
        location_field=args.location_field,
    )
    print(f"Filas procesadas: {len(rows)}")
    print(f"Salida: {output_file}")


if __name__ == "__main__":
    main()
