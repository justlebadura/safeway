from safeway.cleaning import clean_location_text


def test_clean_location_text_normalizes_case_accents_and_spaces() -> None:
    raw = "  Calle 10  NúMero  5-20   Cúcuta  "
    assert clean_location_text(raw) == "CALLE 10 # 5-20 CUCUTA"


def test_clean_location_text_replaces_no_token_with_hash() -> None:
    raw = "Cra 8 no 12-44"
    assert clean_location_text(raw) == "CRA 8 # 12-44"
