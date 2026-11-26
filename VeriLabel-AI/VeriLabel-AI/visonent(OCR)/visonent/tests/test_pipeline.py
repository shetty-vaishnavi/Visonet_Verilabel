"""
End-to-End Pipeline tests for Module 2: OCR & Document Processing.
"""

import os
from pathlib import Path
import pytest
from verilabel_ocr.pipeline import DocumentProcessingPipeline
from verilabel_ocr.config import PipelineConfig

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_labels"


@pytest.fixture(scope="module")
def pipeline():
    output_dir = Path(__file__).resolve().parent.parent / "test_ocr_output"
    config = PipelineConfig(output_dir=str(output_dir))
    return DocumentProcessingPipeline(config)


def test_pipeline_tc01_compliant_label(pipeline):
    """TC01: Verify fully compliant medical device label extraction."""
    sample_path = str(SAMPLE_DIR / "tc01_compliant_label.png")
    assert os.path.exists(sample_path)

    result = pipeline.process(sample_path)

    assert result.success is True
    assert result.total_blocks_detected > 0
    assert result.processing_time_ms > 0
    assert result.raw_text != ""

    summary = result.structured_data.get_summary()
    assert summary["catalog_ref"] is not None
    assert "CATH-2024" in summary["catalog_ref"]
    assert summary["lot_number"] is not None
    assert summary["expiry_date"] is not None
    assert summary["manufacturer"] is not None

    if result.annotated_image_path:
        assert os.path.exists(result.annotated_image_path)


def test_pipeline_tc02_missing_expiry(pipeline):
    """TC02: Verify detection of missing expiry date."""
    sample_path = str(SAMPLE_DIR / "tc02_missing_expiry.png")
    result = pipeline.process(sample_path)

    assert result.success is True
    assert not result.structured_data.expiry_date.is_present
    # Catalog and LOT should still be extracted
    assert result.structured_data.catalog_number.is_present
    assert "ORTH-9021" in result.structured_data.catalog_number.value


def test_pipeline_tc03_missing_warning(pipeline):
    """TC03: Verify detection of missing warnings and storage instructions."""
    sample_path = str(SAMPLE_DIR / "tc03_missing_warning.png")
    result = pipeline.process(sample_path)

    assert result.success is True
    assert len(result.structured_data.warnings) == 0
    assert not result.structured_data.storage_conditions.is_present


def test_pipeline_tc04_skewed_label(pipeline):
    """TC04: Verify deskewing on rotated label."""
    sample_path = str(SAMPLE_DIR / "tc04_skewed_compliant.png")
    result = pipeline.process(sample_path)

    assert result.success is True
    assert result.total_blocks_detected > 0
    assert abs(result.deskew_angle) > 0.0
