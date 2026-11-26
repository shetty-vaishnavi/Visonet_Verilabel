"""
Unit tests for MedicalFieldExtractor in Module 2.
"""

import pytest
from verilabel_ocr.models import OCRTextBlock, BoundingBox, FieldStatus
from verilabel_ocr.field_extractor import MedicalFieldExtractor


@pytest.fixture
def extractor():
    return MedicalFieldExtractor()


def create_block(block_id: int, text: str, y_offset: int = 10, confidence: float = 0.95) -> OCRTextBlock:
    return OCRTextBlock(
        id=block_id,
        text=text,
        confidence=confidence,
        bbox=BoundingBox(
            xmin=50,
            ymin=y_offset,
            xmax=400,
            ymax=y_offset + 30,
        )
    )


def test_extract_identifiers(extractor):
    blocks = [
        create_block(0, "CARDIO-FLEX VASCULAR BALLOON CATHETER", y_offset=20),
        create_block(1, "REF: CATH-2024-X8", y_offset=80),
        create_block(2, "LOT: LOT20240915-A", y_offset=120),
        create_block(3, "SN: SN-987214-E", y_offset=160),
    ]

    data = extractor.extract(blocks)

    assert data.device_name.is_present
    assert "CARDIO-FLEX" in data.device_name.value
    assert data.catalog_number.is_present
    assert data.catalog_number.value == "CATH-2024-X8"
    assert data.lot_number.is_present
    assert data.lot_number.value == "LOT20240915-A"
    assert data.serial_number.is_present
    assert data.serial_number.value == "SN-987214-E"


def test_extract_dates_with_iso_parsing(extractor):
    blocks = [
        create_block(0, "MFG DATE: 2024-09-15", y_offset=100),
        create_block(1, "EXPIRY DATE (EXP): 2027-09-14", y_offset=140),
    ]

    data = extractor.extract(blocks)

    assert data.mfg_date.is_present
    assert data.mfg_date.value == "2024-09-15"
    assert data.mfg_date.metadata.get("parsed_iso") == "2024-09-15"

    assert data.expiry_date.is_present
    assert data.expiry_date.value == "2027-09-14"
    assert data.expiry_date.metadata.get("parsed_iso") == "2027-09-14"


def test_missing_expiry_date(extractor):
    # TC02: Defective label where expiry date is omitted
    blocks = [
        create_block(0, "ORTHO-LOCK BONE SCREW SYSTEM", y_offset=20),
        create_block(1, "REF: ORTH-9021-S", y_offset=80),
        create_block(2, "LOT: LOT20241002-B", y_offset=120),
        # Notice: No EXP date here!
    ]

    data = extractor.extract(blocks)

    assert not data.expiry_date.is_present
    assert data.expiry_date.status == FieldStatus.MISSING
    assert data.catalog_number.is_present


def test_extract_warnings_and_storage(extractor):
    blocks = [
        create_block(0, "Store between 15°C and 25°C. Keep dry.", y_offset=150),
        create_block(1, "CAUTION: Federal law restricts this device to sale by a physician.", y_offset=250),
        create_block(2, "WARNING: Single Use Only. Do not reuse.", y_offset=290),
    ]

    data = extractor.extract(blocks)

    assert data.storage_conditions.is_present
    assert "15°C" in data.storage_conditions.value
    assert len(data.warnings) >= 2
    warning_texts = [w.value for w in data.warnings]
    assert any("CAUTION" in t for t in warning_texts)
    assert any("Single Use" in t for t in warning_texts)


def test_extract_manufacturer_and_country(extractor):
    blocks = [
        create_block(0, "Manufactured by: Brahmastra BioMedical Technologies Inc.", y_offset=300),
        create_block(1, "100 Innovation Blvd, Suite 400, Cambridge, MA 02142, USA", y_offset=330),
        create_block(2, "Country of Origin: Made in USA", y_offset=360),
    ]

    data = extractor.extract(blocks)

    assert data.manufacturer_name.is_present
    assert "Brahmastra BioMedical" in data.manufacturer_name.value
    assert data.country_of_origin.is_present
    assert "USA" in data.country_of_origin.value
