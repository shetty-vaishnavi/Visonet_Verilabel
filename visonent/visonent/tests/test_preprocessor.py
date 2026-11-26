"""
Unit tests for DocumentPreprocessor in Module 2.
"""

import os
import pytest
import numpy as np
import cv2

from verilabel_ocr.config import PreprocessConfig
from verilabel_ocr.preprocessor import DocumentPreprocessor


@pytest.fixture
def preprocessor():
    config = PreprocessConfig(enable_deskew=True, enable_clahe=True)
    return DocumentPreprocessor(config)


def test_load_document_from_file(preprocessor):
    sample_path = "sample_labels/tc01_compliant_label.png"
    assert os.path.exists(sample_path)
    img, meta = preprocessor.load_document(sample_path)
    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3
    assert meta["file_name"] == "tc01_compliant_label.png"
    assert meta["original_width"] == 1200
    assert meta["original_height"] == 800


def test_load_document_from_bytes(preprocessor):
    sample_path = "sample_labels/tc01_compliant_label.png"
    with open(sample_path, "rb") as f:
        img_bytes = f.read()
    img, meta = preprocessor.load_document(img_bytes)
    assert isinstance(img, np.ndarray)
    assert meta["source_type"] == "bytes"
    assert meta["original_width"] == 1200


def test_preprocess_enhancement(preprocessor):
    sample_path = "sample_labels/tc01_compliant_label.png"
    img, _ = preprocessor.load_document(sample_path)
    enhanced_color, ocr_ready, stats = preprocessor.preprocess(img)

    assert enhanced_color.shape[:2] == ocr_ready.shape[:2]
    assert len(ocr_ready.shape) == 2  # Grayscale / CLAHE processed
    assert "deskew_angle" in stats
    assert stats["final_width"] >= preprocessor.config.min_dimension


def test_deskew_angle_detection(preprocessor):
    # Test on known skewed sample
    skewed_path = "sample_labels/tc04_skewed_compliant.png"
    img, _ = preprocessor.load_document(skewed_path)
    angle = preprocessor.detect_skew_angle(img)
    # Generated with -5.5 degree rotation
    assert abs(angle) > 1.0  # Successfully detected non-zero skew
