from __future__ import annotations

import os
from typing import Any

import requests


class SodaClient:
    def __init__(
        self,
        dataset_id: str,
        domain: str = "www.datos.gov.co",
        app_token_env: str = "SODA_APP_TOKEN",
        timeout: int = 30,
    ) -> None:
        self.dataset_id = dataset_id
        self.base_url = f"https://{domain}/resource/{dataset_id}.json"
        self.timeout = timeout
        self.headers = {}
        token = os.getenv(app_token_env)
        if token:
            self.headers["X-App-Token"] = token

    def fetch_rows(
        self,
        select: str,
        where: str | None = None,
        limit: int = 1000,
        max_rows: int = 5000,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0

        while len(rows) < max_rows:
            params = {
                "$select": select,
                "$limit": min(limit, max_rows - len(rows)),
                "$offset": offset,
            }
            if where:
                params["$where"] = where

            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            rows.extend(batch)
            offset += len(batch)

        return rows
