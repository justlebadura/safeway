from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

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


import hashlib

def parse_coordinate(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).strip().replace(",", ".")
        return float(cleaned)
    except ValueError:
        return None


FALLBACK_COORDINATES = {
    "CUCUTA": (7.89391, -72.50782),
    "VILLA DEL ROSARIO": (7.83389, -72.47417),
    "LOS PATIOS": (7.85972, -72.50806),
    "OCAÑA": (8.23778, -73.35333),
    "PAMPLONA": (7.37583, -72.64833),
    "SARDINATA": (8.08278, -72.80056),
    "TIBU": (8.63889, -72.73333),
    "CHINACOTA": (7.60833, -72.60056),
    "BUCARAMANGA": (7.12539, -73.1198),
    "BOGOTA": (4.60971, -74.08175),
    "PALMIRA": (3.51833, -76.30400),
    "BARRANQUILLA": (10.963889, -74.796389),
    "CALI": (3.451647, -76.531985),
}


def resolve_coordinates(
    row_id: str,
    latitude: float | None,
    longitude: float | None,
    extraccion: Any,
    dataset_id: str,
) -> tuple[float, float, bool]:
    if latitude is not None and longitude is not None:
        return latitude, longitude, False

    muni_name = None
    if isinstance(extraccion, dict) and extraccion.get("BARRIO_O_MUNICIPIO"):
        muni_name = extraccion["BARRIO_O_MUNICIPIO"].get("value")

    if dataset_id == "7cci-nqqb":
        default_city = "BUCARAMANGA"
    elif dataset_id == "3v2w-chcq":
        default_city = "BOGOTA"
    elif dataset_id == "sjpx-eqfp":
        default_city = "PALMIRA"
    elif dataset_id == "sefb-a755":
        default_city = "BARRANQUILLA"
    elif dataset_id == "ixgc-yijx":
        default_city = "CALI"
    else:
        default_city = "CUCUTA"

    city_key = (muni_name or default_city).upper()

    coords = FALLBACK_COORDINATES.get(city_key)
    if not coords:
        if city_key == "BOGOTA":
            coords = (4.60971, -74.08175)
        elif city_key == "PALMIRA":
            coords = (3.51833, -76.30400)
        else:
            coords = FALLBACK_COORDINATES.get(default_city)
    if not coords:
        coords = FALLBACK_COORDINATES.get("CUCUTA")

    h = int(hashlib.md5(row_id.encode("utf-8")).hexdigest(), 16)
    lat_off = ((h % 1000) / 1000.0 - 0.5) * 0.015
    lng_off = (((h // 1000) % 1000) / 1000.0 - 0.5) * 0.015

    return coords[0] + lat_off, coords[1] + lng_off, True


MUNICIPIOS_NORTE_SANTANDER = [
    "ABREGO",
    "ARBOLEDAS",
    "BOCHALEMA",
    "BUCARASICA",
    "CACOTA",
    "CACHIRA",
    "CHINACOTA",
    "CHITAGA",
    "CONVENCION",
    "CUCUTA",
    "CUCUTILLA",
    "DURANIA",
    "EL CARMEN",
    "EL TARRA",
    "EL ZULIA",
    "GRAMALOTE",
    "HACARI",
    "HERRAN",
    "LABATECA",
    "LA ESPERANZA",
    "LA PLAYA DE BELEN",
    "LOS PATIOS",
    "LOURDES",
    "MUTISCUA",
    "OCAÑA",
    "PAMPLONA",
    "PAMPLONITA",
    "PUERTO SANTANDER",
    "RAGONVALIA",
    "SALAZAR",
    "SAN CALIXTO",
    "SAN CAYETANO",
    "SANTIAGO",
    "SARDINATA",
    "SILOS",
    "TEORAMA",
    "TIBU",
    "TOLEDO",
    "VILLA CARO",
    "VILLA DEL ROSARIO",
]

MUNICIPIO_ALIASES = {
    "LA PLAYA": "LA PLAYA DE BELEN",
    "SAN JOSE DE CUCUTA": "CUCUTA",
}


def _normalized_municipalities() -> dict[str, str]:
    # Map normalized municipality tokens to a normalized canonical value.
    canonical = {normalize_text(name): normalize_text(name) for name in MUNICIPIOS_NORTE_SANTANDER}
    for alias, target in MUNICIPIO_ALIASES.items():
        canonical[normalize_text(alias)] = normalize_text(target)
    return canonical


MUNICIPIO_CANONICAL_BY_NORMALIZED = _normalized_municipalities()
MUNICIPIO_PATTERNS = sorted(MUNICIPIO_CANONICAL_BY_NORMALIZED.keys(), key=len, reverse=True)


ROAD_PATTERN = re.compile(
    r"\b(ANILLO VIAL|AUTOPISTA|AVENIDA|AV|CALLE|CL|CARRERA|CRA|CR|TRANSVERSAL|TV|DIAGONAL|DG|RUTA|VIA|VEREDA)\b(?:\s+[A-Z0-9]+){0,4}"
)
KM_PATTERN = re.compile(r"\bKM\s*\d+(?:\+\d+)?\b")
NUM_PATTERN_HASH = re.compile(r"\b\d{1,4}\s*#\s*\d{1,4}(?:\s*-\s*\d{1,4})?\b")
NUM_PATTERN_SIMPLE = re.compile(r"\b\d{1,4}(?:\s*-\s*\d{1,4})\b")
BARRIO_PATTERN = re.compile(r"\bBARRIO\s+([A-Z0-9\s]{3,40})")

REFERENCE_KEYWORDS = [
    "FRENTE A",
    "CERCA DE",
    "SECTOR",
    "PUENTE",
    "PEAJE",
    "GLORIETA",
    "TERMINAL",
    "HOSPITAL",
    "COLEGIO",
    "PARQUE",
    "UNIVERSIDAD",
    "ESTACION",
    "SENTIDO",
]


BLACKLIST_TERMS = {
    "OBSERVACION", "ADMINISTRATIVA", "SIN", "DATOS", "NO", "REGISTRA", "NINGUNO",
    "N/A", "PENDIENTE", "DESCONOCIDO", "DIRECCION", "RUTA", "N/D", "ADMINISTRATIVO"
}


def build_nlp() -> Language:
    nlp = spacy.blank("es")
    ruler = nlp.add_pipe("entity_ruler")
    assert isinstance(ruler, EntityRuler)

    patterns = []
    for municipality in MUNICIPIO_PATTERNS:
        patterns.append({"label": "LOC", "pattern": municipality})

    patterns.extend(
        [
            {"label": "ROAD", "pattern": "ANILLO VIAL"},
            {"label": "ROAD", "pattern": "AUTOPISTA"},
            {"label": "ROAD", "pattern": "AVENIDA"},
            {"label": "ROAD", "pattern": "CALLE"},
            {"label": "ROAD", "pattern": "CARRERA"},
            {"label": "ROAD", "pattern": "TRANSVERSAL"},
            {"label": "ROAD", "pattern": "DIAGONAL"},
            {"label": "LANDMARK", "pattern": "TERMINAL"},
            {"label": "LANDMARK", "pattern": "HOSPITAL"},
            {"label": "LANDMARK", "pattern": "PEAJE"},
            {"label": "LANDMARK", "pattern": "GLORIETA"},
        ]
    )
    ruler.add_patterns(patterns)
    return nlp


class AddressExtractor:
    REQUIRED_KEYS = [
        "VIA_PRINCIPAL",
        "NUMERO_O_KM",
        "REFERENCIA_SEMANTICA",
        "BARRIO_O_MUNICIPIO",
    ]

    def __init__(self) -> None:
        self.nlp = build_nlp()

    def extract(self, cleaned_text: str) -> dict[str, Any] | list[str]:
        text = (cleaned_text or "").strip().upper()
        if not text:
            return ["UNKNOWN"]

        doc = self.nlp(text)
        via = self._extract_via(text, doc)
        num_km = self._extract_num_or_km(text)
        ref = self._extract_reference(text)
        barrio_muni = self._extract_barrio_or_municipality(text, doc)

        if barrio_muni["value"] is None and text:
            words = text.split()
            has_blacklist = any(w in BLACKLIST_TERMS for w in words)
            if not has_blacklist and len(words) <= 3:
                barrio_muni = {"value": text, "confidence": 0.70}

        has_spatial_signal = any(
            item["value"] is not None for item in [via, num_km, ref, barrio_muni]
        )
        if not has_spatial_signal:
            return ["UNKNOWN"]

        return {
            "VIA_PRINCIPAL": via,
            "NUMERO_O_KM": num_km,
            "REFERENCIA_SEMANTICA": ref,
            "BARRIO_O_MUNICIPIO": barrio_muni,
        }

    def _extract_via(self, text: str, doc: Any) -> dict[str, Any]:
        match = ROAD_PATTERN.search(text)
        entity_match = next((ent.text for ent in doc.ents if ent.label_ == "ROAD"), None)

        if match:
            value = match.group(0).strip()
            score = 0.9 if entity_match and entity_match in value else 0.84
            return {"value": value, "confidence": round(score, 2)}
        if entity_match:
            return {"value": entity_match, "confidence": 0.78}
        return {"value": None, "confidence": 0.0}

    def _extract_num_or_km(self, text: str) -> dict[str, Any]:
        km_match = KM_PATTERN.search(text)
        if km_match:
            return {"value": km_match.group(0), "confidence": 0.93}

        hash_match = NUM_PATTERN_HASH.search(text)
        if hash_match:
            value = re.sub(r"\s+", " ", hash_match.group(0)).replace(" - ", "-")
            return {"value": value, "confidence": 0.88}

        simple_match = NUM_PATTERN_SIMPLE.search(text)
        if simple_match:
            value = simple_match.group(0).replace(" ", "")
            return {"value": value, "confidence": 0.74}

        return {"value": None, "confidence": 0.0}

    def _extract_reference(self, text: str) -> dict[str, Any]:
        for kw in REFERENCE_KEYWORDS:
            idx = text.find(kw)
            if idx >= 0:
                snippet = text[idx : idx + 55].strip(" ,;.")
                snippet = re.sub(r"\s+", " ", snippet)
                return {"value": snippet, "confidence": 0.76}
        return {"value": None, "confidence": 0.0}

    def _extract_barrio_or_municipality(self, text: str, doc: Any) -> dict[str, Any]:
        barrio_match = BARRIO_PATTERN.search(text)
        if barrio_match:
            raw = barrio_match.group(0).strip()
            value = re.sub(r"\s+", " ", raw)
            return {"value": value, "confidence": 0.81}

        municipality = next((ent.text for ent in doc.ents if ent.label_ == "LOC"), None)
        if municipality:
            normalized = normalize_text(municipality)
            canonical = MUNICIPIO_CANONICAL_BY_NORMALIZED.get(normalized, normalized)
            return {"value": canonical, "confidence": 0.95}

        for muni in MUNICIPIO_PATTERNS:
            if re.search(rf"\b{re.escape(muni)}\b", text):
                return {"value": MUNICIPIO_CANONICAL_BY_NORMALIZED[muni], "confidence": 0.9}

        return {"value": None, "confidence": 0.0}
