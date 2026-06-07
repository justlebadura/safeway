import re
import unicodedata
from datetime import datetime
from typing import Any


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


def clean_soda_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: clean_soda_value(key, value) for key, value in row.items()}
