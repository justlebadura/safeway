from __future__ import annotations
import re, hashlib, math
from typing import Any

CITY_COORDS = {
    "CUCUTA": (7.89391, -72.50782), "BUCARAMANGA": (7.1193, -73.1227),
    "BOGOTA": (4.60971, -74.08175), "PALMIRA": (3.51833, -76.30400),
    "BARRANQUILLA": (10.963889, -74.796389), "CALI": (3.451647, -76.531985),
}
DEFAULT_CITY = {"7cci-nqqb":"BUCARAMANGA","3v2w-chcq":"BOGOTA","sjpx-eqfp":"PALMIRA",
                "sefb-a755":"BARRANQUILLA","ixgc-yijx":"CALI"}

BGA_CR_BASE, BGA_CL_BASE, BGA_UNIT = -73.1220, 7.1190, 0.0007

BGA_BARRIOS = {
    "CENTRO": (7.1194, -73.1226), "CABECERA DEL LLANO": (7.1218, -73.1118),
    "CABECERA": (7.1218, -73.1118), "SAN FRANCISCO": (7.1322, -73.1216),
    "RIO DE ORO I": (7.1430, -73.1310), "PROVENZA": (7.0980, -73.1110),
    "ALARCON": (7.1264, -73.1212), "LA CONCORDIA": (7.1180, -73.1250),
    "RICAURTE": (7.1140, -73.1260), "SOTOMAYOR": (7.1200, -73.1158),
    "LA AURORA": (7.1268, -73.1160), "REAL DE MINAS": (7.1110, -73.1190),
    "GARCIA ROVIRA": (7.1205, -73.1260), "DIAMANTE II": (7.0890, -73.1120),
    "COMUNEROS": (7.1310, -73.1165), "SAN ALONSO": (7.1292, -73.1140),
    "CAMPO HERMOSO": (7.1230, -73.1320), "CONUCOS": (7.1115, -73.1122),
    "PUERTA DEL SOL": (7.1080, -73.1128), "LA PEDREGOSA": (7.0940, -73.1090),
    "CAFE MADRID": (7.1650, -73.1280), "ALVAREZ": (7.1250, -73.1110),
    "BOLARQUI": (7.1230, -73.1170), "MEJORAS PUBLICAS": (7.1210, -73.1185),
    "EL PRADO": (7.1265, -73.1125), "MUTIS": (7.1090, -73.1280),
    "GIRON": (7.0705, -73.1703), "FLORIDABLANCA": (7.0668, -73.0872),
    "PIEDECUESTA": (6.9892, -73.0518),
}

def parse_coordinate(val: Any) -> float | None:
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    try: return float(str(val).strip().replace(",", "."))
    except: return None

def _extract_cr_cl(text: str):
    text = text.upper().replace("CON", " Y ").replace("#", " ")
    cr = re.search(r"(?:CARRERA|CRA|CR)\s*(\d+)", text)
    cl = re.search(r"(?:CALLE|CL)\s*(\d+)", text)
    if not cr or not cl:
        m = re.search(r"(\d{1,3})\s*Y\s*(\d{1,3})", text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > 50 and b <= 50: a, b = b, a
            return a, b
    if cr and cl: return int(cr.group(1)), int(cl.group(1))
    return None, None

def _bga_from_address(text: str, row_id: str):
    cr, cl = _extract_cr_cl(text)
    if cr and cl:
        lat = BGA_CL_BASE + (cl - 36) * BGA_UNIT
        lng = BGA_CR_BASE - (cr - 27) * BGA_UNIT
        h = int(hashlib.md5(f"{cr}{cl}{row_id}".encode()).hexdigest(), 16)
        return lat + ((h%100)/100.0-0.5)*BGA_UNIT, lng + ((h//100%100)/100.0-0.5)*BGA_UNIT
    cr = re.search(r"(?:CARRERA|CRA|CR)\s*(\d+)", text)
    if cr: return BGA_CL_BASE, BGA_CR_BASE - (int(cr.group(1))-27)*BGA_UNIT
    cl = re.search(r"(?:CALLE|CL)\s*(\d+)", text)
    if cl: return BGA_CL_BASE + (int(cl.group(1))-36)*BGA_UNIT, BGA_CR_BASE
    return None

def _bga_from_barrio(text: str, row_id: str):
    t = text.strip().upper()
    for barrio, coords in BGA_BARRIOS.items():
        if barrio in t:
            h = int(hashlib.md5(f"{barrio}{row_id}".encode()).hexdigest(), 16)
            lat_off = ((h%4000)/4000.0-0.5)*0.003
            lng_off = (((h//4000)%4000)/4000.0-0.5)*0.003
            return coords[0]+lat_off, coords[1]+lng_off
    return None

def resolve_coordinates(
    row_id: str, latitude: float | None, longitude: float | None,
    extraction: Any, dataset_id: str,
) -> tuple[float, float, bool]:
    if latitude is not None and longitude is not None:
        return latitude, longitude, False

    city = DEFAULT_CITY.get(dataset_id, "CUCUTA")
    
    if city == "BUCARAMANGA" and isinstance(extraction, dict):
        via = extraction.get("VIA_PRINCIPAL", {}).get("value", "")
        barrio = extraction.get("BARRIO_O_MUNICIPIO", {}).get("value", "")
        if via:
            est = _bga_from_address(str(via), row_id)
            if est and est[0] is not None: return est[0], est[1], True
        if barrio:
            est = _bga_from_barrio(str(barrio), row_id)
            if est and est[0] is not None: return est[0], est[1], True

    coords = CITY_COORDS.get(city, CITY_COORDS["CUCUTA"])
    h = int(hashlib.md5(row_id.encode()).hexdigest(), 16)
    return coords[0] + ((h%200)/200.0-0.5)*0.001, coords[1] + (((h//200)%200)/200.0-0.5)*0.001, True


class AddressExtractor:
    REQUIRED_KEYS = ["VIA_PRINCIPAL", "NUMERO_O_KM", "REFERENCIA_SEMANTICA", "BARRIO_O_MUNICIPIO"]
    
    def extract(self, text: str) -> dict[str, Any] | list[str]:
        text = (text or "").strip().upper()
        if not text or text in ("N/A","NA","NULL","NONE","-","SIN DATOS","NO REGISTRA"):
            return ["UNKNOWN"]

        via = None
        m = re.search(r"(?:CARRERA|CRA|CR|CALLE|CL|AVENIDA|AV|TRANSVERSAL|TV|DIAGONAL|DG|AUTOPISTA|ANILLO\s+VIAL|VIA|RUTA)\s*\d+[A-Z]*", text)
        if m: via = m.group(0).strip()
        if not via:
            m2 = re.search(r"(\d{1,3})\s*[Y#\-CON]\s*(\d{1,3})", text)
            if m2: via = f"CARRERA {m2.group(1)} CON CALLE {m2.group(2)}"
        if not via: via = text[:60]

        ref = None
        for kw in ["FRENTE A","CERCA DE","SECTOR","PUENTE","PEAJE","GLORIETA","TERMINAL","HOSPITAL","COLEGIO","PARQUE","UNIVERSIDAD","ESTACION"]:
            idx = text.find(kw)
            if idx >= 0: ref = text[idx:idx+50].strip(); break

        num = None
        km = re.search(r"KM\s*\d+", text)
        if km: num = km.group(0)
        if not num:
            nm = re.search(r"#\s*\d+[\-\d]*|\d+\s*#\s*\d+", text)
            if nm: num = nm.group(0)
        if not num:
            nm = re.search(r"\b\d{1,4}(?:\s*-\s*\d{1,4})\b", text)
            if nm: num = nm.group(0)

        return {
            "VIA_PRINCIPAL": {"value": via, "confidence": 0.85 if via else 0.0},
            "NUMERO_O_KM": {"value": num, "confidence": 0.85 if num else 0.0},
            "REFERENCIA_SEMANTICA": {"value": ref, "confidence": 0.70 if ref else 0.0},
            "BARRIO_O_MUNICIPIO": {"value": text[:80], "confidence": 0.50},
        }
