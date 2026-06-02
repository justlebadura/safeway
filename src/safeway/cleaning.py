import re
import unicodedata


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
