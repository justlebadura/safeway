from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

from .cleaning import normalize_text


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
