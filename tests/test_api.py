from __future__ import annotations

from fastapi.testclient import TestClient

from safeway import api


class SequencedSodaClient:
    sequences: list[list[dict[str, str]]] = []

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id

    def fetch_rows(
        self,
        select: str,
        where: str | None = None,
        limit: int = 1000,
        max_rows: int = 5000,
    ) -> list[dict[str, str]]:
        if len(self.sequences) > 1:
            return self.sequences.pop(0)
        return self.sequences[0]


def build_service(refresh_interval_seconds: int = 60, poll_interval_seconds: float = 0.01) -> api.DatasetCacheService:
    return api.DatasetCacheService(
        refresh_interval_seconds=refresh_interval_seconds,
        poll_interval_seconds=poll_interval_seconds,
        client_factory=SequencedSodaClient,
    )


def test_get_dataset_uses_cache_and_returns_processed_rows(monkeypatch) -> None:
    SequencedSodaClient.sequences = [
        [{"comparendo": "1", "lugar": "CALLE 10 #5-20 CUCUTA", "fecha": "2024-01-01T10:00:00.000"}]
    ]
    service = build_service(refresh_interval_seconds=999)
    monkeypatch.setattr(api, "cache_service", service)
    client = TestClient(api.app)

    response = client.get("/datasets/stq8-drvp", params={"max_rows": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "stq8-drvp"
    assert payload["processed"][0]["data_limpia"]["fecha"] == "2024-01-01T10:00:00"
    assert payload["processed"][0]["extraccion"]["VIA_PRINCIPAL"]["value"] == "CALLE 10"


def test_long_poll_returns_updated_data_when_version_changes(monkeypatch) -> None:
    SequencedSodaClient.sequences = [
        [{"comparendo": "1", "lugar": "CALLE 10 #5-20 CUCUTA"}],
        [{"comparendo": "1", "lugar": "CALLE 10 #5-20 CUCUTA"}],
        [{"comparendo": "1", "lugar": "AV LIBERTADORES #12-34 CUCUTA"}],
    ]
    service = build_service(refresh_interval_seconds=0, poll_interval_seconds=0.01)
    monkeypatch.setattr(api, "cache_service", service)
    client = TestClient(api.app)

    snapshot = client.get("/datasets/stq8-drvp", params={"max_rows": 1})
    version = snapshot.json()["version"]

    response = client.get(
        "/datasets/stq8-drvp/updates",
        params={"max_rows": 1, "last_version": version, "timeout_seconds": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["changed"] is True
    assert payload["timed_out"] is False
    assert payload["version"] != version
    assert payload["data"][0]["data_limpia"]["lugar"] == "AV LIBERTADORES # 12-34 CUCUTA"


def test_long_poll_times_out_without_change(monkeypatch) -> None:
    SequencedSodaClient.sequences = [
        [{"comparendo": "1", "lugar": "CALLE 10 #5-20 CUCUTA"}],
    ]
    service = build_service(refresh_interval_seconds=999, poll_interval_seconds=0.01)
    monkeypatch.setattr(api, "cache_service", service)
    client = TestClient(api.app)

    snapshot = client.get("/datasets/stq8-drvp", params={"max_rows": 1})
    version = snapshot.json()["version"]

    response = client.get(
        "/datasets/stq8-drvp/updates",
        params={"max_rows": 1, "last_version": version, "timeout_seconds": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["changed"] is False
    assert payload["timed_out"] is True
    assert "data" not in payload