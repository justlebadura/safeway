import json
from pathlib import Path

from safeway import pipeline


class DummySodaClient:
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id

    def fetch_rows(self, select: str, where: str | None = None, limit: int = 1000, max_rows: int = 5000):
        return [
            {"comparendo": "1", "lugar": "CALLE 10 #5-20 CUCUTA"},
            {"comparendo": "2", "lugar": "sin datos de direccion"},
        ]


def test_run_pipeline_generates_output_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "SodaClient", DummySodaClient)

    output = tmp_path / "extracciones_test.json"
    rows = pipeline.run_pipeline("stq8-drvp", max_rows=2, output_file=output)

    assert len(rows) == 2
    assert output.exists()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data[0]["extraccion"]["VIA_PRINCIPAL"]["confidence"] >= 0.0
    assert data[1]["extraccion"] == ["UNKNOWN"]
