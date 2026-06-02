from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cleaning import clean_location_text
from .extractor import AddressExtractor
from .soda_client import SodaClient


def run_pipeline(dataset_id: str, max_rows: int, output_file: Path) -> list[dict[str, Any]]:
    client = SodaClient(dataset_id=dataset_id)
    extractor = AddressExtractor()

    rows = client.fetch_rows(
        select="comparendo,lugar",
        where="lugar is not null",
        limit=1000,
        max_rows=max_rows,
    )

    processed: list[dict[str, Any]] = []
    for row in rows:
        raw_lugar = row.get("lugar", "")
        cleaned_lugar = clean_location_text(raw_lugar)
        extraction = extractor.extract(cleaned_lugar)
        processed.append(
            {
                "comparendo": row.get("comparendo"),
                "lugar_original": raw_lugar,
                "lugar_limpio": cleaned_lugar,
                "extraccion": extraction,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline SODA -> limpieza -> extracción")
    parser.add_argument("--dataset-id", default="stq8-drvp")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--output", default="outputs/extracciones.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = Path(args.output)
    rows = run_pipeline(args.dataset_id, args.max_rows, output_file)
    print(f"Filas procesadas: {len(rows)}")
    print(f"Salida: {output_file}")


if __name__ == "__main__":
    main()
