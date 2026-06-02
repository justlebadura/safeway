from safeway.cleaning import clean_location_text
from safeway.extractor import AddressExtractor


extractor = AddressExtractor()


def test_extract_returns_required_keys_with_confidence() -> None:
    text = clean_location_text("AV LIBERTADORES #12-34 CUCUTA")
    result = extractor.extract(text)

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "VIA_PRINCIPAL",
        "NUMERO_O_KM",
        "REFERENCIA_SEMANTICA",
        "BARRIO_O_MUNICIPIO",
    }

    for key, entity in result.items():
        assert "value" in entity, f"Falta value en {key}"
        assert "confidence" in entity, f"Falta confidence en {key}"
        assert 0.0 <= entity["confidence"] <= 1.0


def test_extract_returns_unknown_when_no_spatial_signal() -> None:
    text = clean_location_text("observacion administrativa")
    result = extractor.extract(text)
    assert result == ["UNKNOWN"]
