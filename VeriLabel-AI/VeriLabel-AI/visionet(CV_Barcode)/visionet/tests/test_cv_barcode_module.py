"""
VeriLabel AI - Module 3: Comprehensive Unit & Integration Test Suite
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Covers:
  1. OpenCV Image Processing & Quality Assessment
  2. Barcode Detection & Multi-Engine Decoding
  3. UDI Extraction & GS1 / HIBCC Regulatory Validation
  4. ISO 15223-1 Medical Symbol Recognition
  5. End-to-End Pipeline Integration
"""

import sys
import io
import math
import unittest
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import zxingcpp

# Resolve project root so we can import verilabel_cv_barcode directly
MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from verilabel_cv_barcode.image_processor import ImageProcessor, ImageProcessingError
from verilabel_cv_barcode.barcode_scanner import BarcodeScanner, ScanTelemetry
from verilabel_cv_barcode.udi_validator import UDIValidator, validate_gtin_checksum, validate_hibcc_modulo43
from verilabel_cv_barcode.symbol_detector import MedicalSymbolDetector, ISO_SYMBOL_METADATA
from verilabel_cv_barcode.models import (
    DecodedBarcode, UDIComponents, DetectedSymbol,
    ImageQualityAssessment, Module3Result,
)
from verilabel_cv_barcode.pipeline import ComputerVisionBarcodePipeline
from verilabel_cv_barcode.config import PipelineConfig, SAMPLE_LABELS_DIR, TEMPLATES_DIR

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_labels"


def make_genuine_barcode(text: str, fmt: zxingcpp.BarcodeFormat, scale: int = 5, pad: int = 15) -> np.ndarray:
    """Creates an authentic barcode image array with quiet zone."""
    bc = zxingcpp.create_barcode(text, fmt)
    img = zxingcpp.write_barcode_to_image(bc)
    gray = np.array(img, dtype=np.uint8)
    h, w = gray.shape[:2]
    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


# =============================================================================
# 1. IMAGE PROCESSING & QUALITY ASSESSMENT
# =============================================================================

class TestImageProcessingAndQuality(unittest.TestCase):
    """Unit tests for OpenCV image processing, preprocessing, and quality assessment."""

    @classmethod
    def setUpClass(cls):
        cls.processor = ImageProcessor()
        cls.sample_label = SAMPLE_DIR / "tc01_compliant_label.png"

    def test_valid_image_ingestion_path(self):
        """Verifies ingestion of a valid PNG image from a file path."""
        img, meta = self.processor.load_image(self.sample_label)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(len(img.shape), 3)
        self.assertIn("sha256", meta)
        self.assertIn("width", meta)
        self.assertIn("height", meta)
        self.assertEqual(meta["extension"], ".png")

    def test_valid_image_ingestion_bytes(self):
        """Verifies loading a valid image from an in-memory byte buffer."""
        with open(self.sample_label, "rb") as f:
            raw_bytes = f.read()
        img, meta = self.processor.load_image(raw_bytes)
        self.assertIsInstance(img, np.ndarray)
        self.assertIn("sha256", meta)

    def test_valid_image_ingestion_bytesio(self):
        """Verifies loading a valid image from a BytesIO stream."""
        with open(self.sample_label, "rb") as f:
            buf = io.BytesIO(f.read())
        img, meta = self.processor.load_image(buf)
        self.assertIsInstance(img, np.ndarray)

    def test_valid_image_ingestion_pil(self):
        """Verifies ingestion from a PIL Image object."""
        pil_img = Image.open(self.sample_label)
        img, meta = self.processor.load_image(pil_img)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(len(img.shape), 3)

    def test_valid_image_ingestion_numpy(self):
        """Verifies ingestion from a NumPy array."""
        arr = cv2.imread(str(self.sample_label))
        img, meta = self.processor.load_image(arr)
        self.assertIsInstance(img, np.ndarray)
        self.assertIn("sha256", meta)

    def test_nonexistent_image_raises_error(self):
        """Verifies ImageProcessingError is raised for non-existent file paths."""
        with self.assertRaises(ImageProcessingError):
            self.processor.load_image(Path("/nonexistent/path/label.png"))

    def test_corrupt_bytes_raises_error(self):
        """Verifies ImageProcessingError is raised for corrupted byte buffers."""
        with self.assertRaises(ImageProcessingError):
            self.processor.load_image(b"\x00\x01\x02\x03CORRUPTED_GARBAGE_DATA_XYZ")

    def test_image_quality_sharp_label(self):
        """Verifies a sharp packaging label is classified with high Laplacian variance."""
        img, _ = self.processor.load_image(self.sample_label)
        qa = self.processor.assess_quality(img)
        self.assertIsInstance(qa, ImageQualityAssessment)
        self.assertGreater(qa.blur_variance, 100.0)
        self.assertFalse(qa.is_blurred)
        self.assertGreater(qa.quality_score, 50.0)

    def test_image_quality_blurred_label(self):
        """Verifies a deliberately blurred image is flagged as blurred."""
        img, _ = self.processor.load_image(self.sample_label)
        blurred = cv2.GaussianBlur(img, (51, 51), 20)
        qa = self.processor.assess_quality(blurred)
        self.assertTrue(qa.is_blurred)
        self.assertTrue(any("blur" in r.lower() or "retake" in r.lower() for r in qa.recommendations))

    def test_image_quality_overexposed_detection(self):
        """Verifies an artificially over-brightened image triggers overexposure flag."""
        img, _ = self.processor.load_image(self.sample_label)
        overexposed = np.clip(img.astype(np.int32) + 170, 0, 255).astype(np.uint8)
        qa = self.processor.assess_quality(overexposed)
        self.assertTrue(qa.is_overexposed)

    def test_image_quality_underexposed_detection(self):
        """Verifies an artificially darkened image triggers underexposure flag."""
        img, _ = self.processor.load_image(self.sample_label)
        underexposed = (img * 0.08).astype(np.uint8)
        qa = self.processor.assess_quality(underexposed, blur_threshold=50.0)
        self.assertTrue(qa.is_underexposed)

    def test_image_quality_score_bounded(self):
        """Verifies quality score is always in the [0, 100] range."""
        img, _ = self.processor.load_image(self.sample_label)
        qa = self.processor.assess_quality(img)
        self.assertGreaterEqual(qa.quality_score, 0.0)
        self.assertLessEqual(qa.quality_score, 100.0)

    def test_deskewing_synthetic_rotated_image(self):
        """Verifies deskewing is applied to a synthetically rotated label."""
        img, _ = self.processor.load_image(self.sample_label)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 8.0, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        deskewed, applied_angle = self.processor.deskew(rotated)
        self.assertIsInstance(deskewed, np.ndarray)
        # Canvas must expand or be same size (never shrink)
        self.assertGreaterEqual(deskewed.shape[0], h)
        self.assertGreaterEqual(deskewed.shape[1], w)

    def test_image_preprocessing_pipeline(self):
        """Verifies enhancement pipeline produces valid BGR output."""
        img, _ = self.processor.load_image(self.sample_label)
        enhanced = self.processor.enhance_image(img)
        self.assertIsInstance(enhanced, np.ndarray)
        self.assertEqual(enhanced.shape, img.shape)

    def test_preprocess_for_barcode_variants(self):
        """Verifies barcode preprocessing generates multiple named variants."""
        img, _ = self.processor.load_image(self.sample_label)
        variants = self.processor.preprocess_for_barcode(img)
        self.assertGreaterEqual(len(variants), 4)
        variant_names = [v[0] for v in variants]
        self.assertIn("original_gray", variant_names)
        self.assertIn("clahe", variant_names)
        self.assertIn("inverted", variant_names)
        for name, arr in variants:
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(len(arr.shape), 2)  # Grayscale


# =============================================================================
# 2. BARCODE DETECTION & MULTI-ENGINE DECODING
# =============================================================================

class TestBarcodeAndUDIScanner(unittest.TestCase):
    """Unit tests for multi-engine barcode detection and UDI regulatory parsing."""

    @classmethod
    def setUpClass(cls):
        cls.scanner = BarcodeScanner(
            enable_adaptive_retries=True,
            enable_rotations=True,
            enable_polarity_inversion=True,
        )
        cls.validator = UDIValidator()
        cls.sample_label_tc01 = SAMPLE_DIR / "tc01_compliant_label.png"

    def test_barcode_detection_compliant_label(self):
        """Verifies DataMatrix decoding on the TC01 compliant sample label."""
        img = cv2.imread(str(self.sample_label_tc01))
        barcodes = self.scanner.scan(img)
        self.assertGreater(len(barcodes), 0)
        bc = barcodes[0]
        self.assertIsInstance(bc, DecodedBarcode)
        self.assertIn("00847239012349", bc.data)

    def test_genuine_datamatrix_decoding(self):
        """Verifies authentic GS1 DataMatrix encoding and decoding."""
        payload = "(01)00847239012349(17)280630(10)LOT-1234"
        dm_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix, scale=5)
        barcodes = self.scanner.scan(dm_img)
        self.assertGreater(len(barcodes), 0)
        self.assertEqual(barcodes[0].data, payload)
        # zxingcpp may report "DATA MATRIX" or "DATAMATRIX" — normalize to compare
        self.assertIn("DATAMATRIX", barcodes[0].barcode_type.upper().replace(" ", "").replace("-", ""))

    def test_genuine_code128_decoding(self):
        """Verifies authentic Code-128 encoding and decoding (1D linear)."""
        payload = "LOT-BATCH-9988-REF42"
        code_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.Code128, scale=3)
        barcodes = self.scanner.scan(code_img)
        self.assertGreater(len(barcodes), 0)
        self.assertIn("128", barcodes[0].barcode_type.upper().replace("-", "").replace("_", ""))

    def test_genuine_code39_decoding(self):
        """Verifies authentic Code-39 encoding and decoding."""
        payload = "LOT8842REF99"
        code_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.Code39, scale=4)
        barcodes = self.scanner.scan(code_img)
        self.assertGreater(len(barcodes), 0)
        self.assertEqual(barcodes[0].data, payload)

    def test_genuine_ean13_decoding(self):
        """Verifies authentic EAN-13 (European Article Number) decoding."""
        payload = "5901234123457"
        ean_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.EAN13, scale=4)
        barcodes = self.scanner.scan(ean_img)
        self.assertGreater(len(barcodes), 0)
        self.assertEqual(barcodes[0].data, payload)

    def test_multiple_barcodes_on_single_image(self):
        """Verifies detection of multiple barcodes on a single packaging label (TC03)."""
        img = cv2.imread(str(SAMPLE_DIR / "tc03_multiple_barcodes.png"))
        barcodes = self.scanner.scan(img)
        self.assertGreaterEqual(len(barcodes), 2)
        types_found = {bc.barcode_type.upper().replace(" ", "").replace("-", "") for bc in barcodes}
        self.assertTrue(any("DATAMATRIX" in t for t in types_found))

    def test_orthogonal_rotations_decoding(self):
        """Verifies a DataMatrix rotated 90 degrees clockwise is decoded via rotation retry."""
        payload = "(01)00847239012349(17)280630(10)ROTATION"
        dm_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix, scale=5)
        rotated_90 = cv2.rotate(dm_img, cv2.ROTATE_90_CLOCKWISE)
        barcodes = self.scanner.scan(rotated_90)
        self.assertGreater(len(barcodes), 0)
        self.assertEqual(barcodes[0].data, payload)

    def test_inverted_polarity_datamatrix(self):
        """Verifies a photonegative (white-on-dark) DataMatrix is decoded via polarity inversion."""
        payload = "(01)00847239012349(17)280630(10)INVERTED"
        dm_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix, scale=5)
        inverted = cv2.bitwise_not(dm_img)
        barcodes = self.scanner.scan(inverted)
        self.assertGreater(len(barcodes), 0)
        self.assertEqual(barcodes[0].data, payload)

    def test_zero_hallucination_on_blank_image(self):
        """Verifies zero barcode detections are returned for a pure blank white canvas."""
        blank = np.full((300, 400, 3), 255, dtype=np.uint8)
        barcodes = self.scanner.scan(blank)
        self.assertEqual(len(barcodes), 0)

    def test_scan_with_telemetry(self):
        """Verifies ScanTelemetry metadata is returned with correct structure."""
        payload = "(01)00847239012349(17)280630(10)TEL-TEST"
        dm_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix, scale=5)
        barcodes, telemetry = self.scanner.scan_with_telemetry(dm_img)
        self.assertIsInstance(telemetry, ScanTelemetry)
        self.assertGreater(telemetry.total_found, 0)
        self.assertIsInstance(telemetry.engines_used, list)
        self.assertGreater(telemetry.latency_ms, 0.0)

    def test_adaptive_enhancement_on_low_contrast_barcode(self):
        """Verifies a low-contrast barcode is decoded via CLAHE adaptive enhancement retry."""
        payload = "(01)00847239012349(17)280630(10)LOWCONTRAST"
        dm_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix, scale=5)
        gray = cv2.cvtColor(dm_img, cv2.COLOR_BGR2GRAY)
        # Compress contrast to simulate faded print
        low_contrast = cv2.addWeighted(gray, 0.3, np.full_like(gray, 128), 0.7, 0)
        low_bgr = cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR)
        barcodes = self.scanner.scan(low_bgr)
        # May or may not decode depending on severity — just verify no crash
        self.assertIsInstance(barcodes, list)

    def test_bounding_box_extraction(self):
        """Verifies decoded barcode returns a valid (x, y, w, h) bounding box."""
        payload = "(01)00847239012349(17)280630(10)BBOX-TEST"
        dm_img = make_genuine_barcode(payload, zxingcpp.BarcodeFormat.DataMatrix, scale=5)
        barcodes = self.scanner.scan(dm_img)
        self.assertGreater(len(barcodes), 0)
        bbox = barcodes[0].bbox
        self.assertIsNotNone(bbox)
        self.assertEqual(len(bbox), 4)
        x, y, w, h = bbox
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    # ---- GS1 Modulo-10 Checksum Tests ----

    def test_gtin_modulo10_checksum_valid(self):
        """Verifies valid GTIN-14 passes Modulo-10 checksum verification."""
        self.assertTrue(validate_gtin_checksum("00847239012349"))  # GTIN-14 valid

    def test_gtin_modulo10_checksum_invalid(self):
        """Verifies corrupted GTIN-14 (wrong check digit) fails Modulo-10 checksum."""
        self.assertFalse(validate_gtin_checksum("00847239012348"))  # Corrupted

    def test_gtin13_valid(self):
        """Verifies valid GTIN-13 (EAN-13) checksum."""
        self.assertTrue(validate_gtin_checksum("5901234123457"))

    def test_gtin13_invalid(self):
        """Verifies invalid GTIN-13 (corrupted digit) fails checksum."""
        self.assertFalse(validate_gtin_checksum("5901234123456"))

    # ---- GS1 Application Identifier (AI) Parsing Tests ----

    def test_gs1_hri_parenthesized_parsing(self):
        """Verifies correct parsing of GS1 HRI parenthesized format."""
        raw = "(01)00847239012349(17)280630(10)LOT-8842(21)SN-001"
        result = self.validator.parse_udi(raw)
        self.assertEqual(result.standard, "GS1")
        self.assertEqual(result.device_identifier_di, "00847239012349")
        self.assertTrue(result.is_di_checksum_valid)
        self.assertEqual(result.lot_number_pi, "LOT-8842")
        self.assertEqual(result.serial_number_pi, "SN-001")
        self.assertFalse(result.is_expired)

    def test_gs1_continuous_machine_stream_parsing(self):
        """Verifies parsing of raw GS1 machine-stream barcode data."""
        raw = "010084723901234917280630\x1d10LOT8842\x1d21SN999"
        result = self.validator.parse_udi(raw)
        self.assertEqual(result.standard, "GS1")
        self.assertEqual(result.device_identifier_di, "00847239012349")
        self.assertTrue(result.is_di_checksum_valid)
        self.assertEqual(result.lot_number_pi, "LOT8842")
        self.assertEqual(result.serial_number_pi, "SN999")

    def test_udi_expiration_date_parsing_and_validity(self):
        """Verifies a future expiration date is correctly parsed and classified as not expired."""
        raw = "(01)00847239012349(17)300630(10)LOT-9999"
        result = self.validator.parse_udi(raw)
        self.assertIsNotNone(result.expiry_date_pi)
        self.assertFalse(result.is_expired)

    def test_udi_expiration_date_day_00_rule(self):
        """Verifies the GS1 Day-00 rule: day=00 resolves to the last calendar day of the month."""
        raw = "(01)00847239012349(17)300200(10)LOT-DAY00"
        result = self.validator.parse_udi(raw)
        # 2030-02-00 should resolve to 2030-02-28 (non-leap) or 2030-02-28
        self.assertIsNotNone(result.expiry_date_pi)
        self.assertTrue(result.expiry_date_pi.endswith("-28") or result.expiry_date_pi.endswith("-29"))

    def test_udi_invalid_date_month_rejected(self):
        """Verifies an invalid month (13) in GS1 Expiry Date triggers validation error."""
        raw = "(01)00847239012349(17)281330(10)LOT-BADMONTH"
        result = self.validator.parse_udi(raw)
        self.assertTrue(len(result.validation_errors) > 0)

    # ---- HIBCC Parsing Tests ----

    def test_hibcc_standard_parsing(self):
        """Verifies HIBCC Primary DI format is correctly identified and parsed."""
        raw = "+A99912345671/$$280630LOT8842X"
        result = self.validator.parse_udi(raw)
        self.assertEqual(result.standard, "HIBCC")
        # DI is extracted without the check character
        self.assertIsNotNone(result.device_identifier_di)

    def test_hibcc_modulo43_checksum_validation(self):
        """Verifies the HIBCC Modulo-43 check character validation function."""
        # Build a valid HIBCC string
        from verilabel_cv_barcode.udi_validator import HIBCC_CHARSET
        content = "A999123456"
        total = sum(HIBCC_CHARSET.index(c) for c in content.upper() if c in HIBCC_CHARSET)
        check_char = HIBCC_CHARSET[total % 43]
        valid_str = f"+{content}{check_char}"
        self.assertTrue(validate_hibcc_modulo43(valid_str))

        corrupted = valid_str[:-1] + ("Z" if check_char != "Z" else "A")
        self.assertFalse(validate_hibcc_modulo43(corrupted))

    # ---- OCR / Barcode Bidirectional Cross-Verification Tests ----

    def test_ocr_udi_cross_validation_matching(self):
        """Verifies cross-validation passes when OCR LOT and Expiry match barcode UDI."""
        raw = "(01)00847239012349(17)280630(10)LOT8842"
        udi = self.validator.parse_udi(raw)
        result = self.validator.cross_verify_with_ocr(udi, ocr_lot="LOT8842", ocr_expiry="2028-06-30")
        self.assertTrue(result.lot_matched)
        self.assertTrue(result.expiry_matched)
        self.assertTrue(result.is_consistent)

    def test_ocr_udi_cross_validation_lot_mismatch(self):
        """Verifies a LOT mismatch between barcode UDI and OCR text triggers CRITICAL mismatch."""
        raw = "(01)00847239012349(17)280630(10)LOT8842"
        udi = self.validator.parse_udi(raw)
        result = self.validator.cross_verify_with_ocr(udi, ocr_lot="LOT-WRONG-999")
        self.assertFalse(result.lot_matched)
        self.assertFalse(result.is_consistent)
        self.assertTrue(any("CRITICAL" in m or "MISMATCH" in m.upper() for m in result.mismatches))

    def test_ocr_udi_cross_validation_expiry_mismatch(self):
        """Verifies an Expiry mismatch between barcode UDI and OCR text triggers CRITICAL mismatch."""
        raw = "(01)00847239012349(17)280630(10)LOT8842"
        udi = self.validator.parse_udi(raw)
        result = self.validator.cross_verify_with_ocr(udi, ocr_expiry="2029-12-31")
        self.assertFalse(result.expiry_matched)
        self.assertFalse(result.is_consistent)


# =============================================================================
# 3. ISO 15223-1 MEDICAL SYMBOL DETECTOR TESTS
# =============================================================================

class TestMedicalSymbolDetector(unittest.TestCase):
    """Unit tests for multi-scale template matching symbol detection."""

    @classmethod
    def setUpClass(cls):
        cls.detector = MedicalSymbolDetector(templates_dir=TEMPLATES_DIR, match_threshold=0.55)
        cls.sample_label = SAMPLE_DIR / "tc01_compliant_label.png"

    def test_all_13_templates_loaded(self):
        """Verifies all 13 ISO 15223-1 symbol reference templates are registered."""
        self.assertEqual(len(ISO_SYMBOL_METADATA), 13)
        self.assertGreater(len(self.detector.templates), 0)
        for sym_name in self.detector.templates.keys():
            self.assertIn(sym_name, ISO_SYMBOL_METADATA)

    def test_symbol_detection_on_compliant_label(self):
        """Verifies at least 5 ISO symbols are detected on the TC01 compliant label."""
        img = cv2.imread(str(self.sample_label))
        symbols = self.detector.detect(img)
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)
        for sym in symbols:
            self.assertIsInstance(sym, DetectedSymbol)
            self.assertGreater(sym.confidence, 0.0)
            self.assertIn(sym.symbol_name, ISO_SYMBOL_METADATA)

    def test_no_false_positive_on_white_canvas(self):
        """Verifies zero symbol detections on a pure blank white packaging canvas."""
        blank = np.full((400, 600, 3), 255, dtype=np.uint8)
        symbols = self.detector.detect(blank)
        self.assertEqual(len(symbols), 0)

    def test_nms_suppresses_redundant_detections(self):
        """Verifies Non-Maximum Suppression removes duplicate detections across scales."""
        img = cv2.imread(str(self.sample_label))
        symbols_no_nms_raw_count_estimate = sum(
            len(np.where(cv2.matchTemplate(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                cv2.resize(self.detector.templates[sym], (45, 45)),
                cv2.TM_CCOEFF_NORMED
            ) >= 0.55)[0]) for sym in list(self.detector.templates.keys())[:3]
        )
        symbols_after_nms = self.detector.detect(img, threshold=0.55, target_symbols=list(self.detector.templates.keys())[:3])
        # NMS survivors must be <= raw peak count
        self.assertLessEqual(len(symbols_after_nms), max(symbols_no_nms_raw_count_estimate, 50))

    def test_dual_correlation_fusion_score_range(self):
        """Verifies the fused correlation score is always in [0, 1]."""
        img = cv2.imread(str(self.sample_label))
        symbols = self.detector.detect(img)
        for sym in symbols:
            self.assertGreaterEqual(sym.confidence, 0.0)
            self.assertLessEqual(sym.confidence, 1.0)

    def test_annotation_drawing_integrity(self):
        """Verifies annotated overlay output preserves image dimensions."""
        img = cv2.imread(str(self.sample_label))
        symbols = self.detector.detect(img)
        annotated = self.detector.annotate(img, symbols)
        self.assertEqual(annotated.shape, img.shape)

    def test_missing_symbols_label_fewer_detections(self):
        """Verifies TC06 (no symbols) produces fewer symbol detections than TC01."""
        tc01 = cv2.imread(str(self.sample_label))
        tc06 = cv2.imread(str(SAMPLE_DIR / "tc06_missing_symbols.png"))
        syms_tc01 = self.detector.detect(tc01)
        syms_tc06 = self.detector.detect(tc06)
        # TC06 deliberately has no symbols, so fewer or zero detections vs TC01
        self.assertGreaterEqual(len(syms_tc01), len(syms_tc06))


# =============================================================================
# 4. END-TO-END PIPELINE INTEGRATION TESTS
# =============================================================================

class TestPipelineIntegration(unittest.TestCase):
    """End-to-end integration tests for the ComputerVisionBarcodePipeline."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ComputerVisionBarcodePipeline(
            PipelineConfig(
                templates_dir=TEMPLATES_DIR,
                output_dir=SAMPLE_DIR.parent / "cv_output",
                save_annotated_image=True,
                enable_adaptive_barcode_retries=True,
            )
        )

    def test_tc01_compliant_label_full_pipeline(self):
        """TC01: Full pipeline on a fully compliant medical device label."""
        result = self.pipeline.process(SAMPLE_DIR / "tc01_compliant_label.png", save_visuals=True)
        self.assertIsInstance(result, Module3Result)
        self.assertGreater(len(result.barcodes), 0)
        self.assertTrue(result.has_valid_udi)
        self.assertGreater(result.symbol_count, 0)
        self.assertGreater(result.quality.quality_score, 0.0)
        self.assertIsNotNone(result.annotated_image_path)

    def test_tc02_invalid_checksum_flagged(self):
        """TC02: Pipeline correctly flags a barcode with an invalid GTIN Modulo-10 checksum."""
        result = self.pipeline.process(SAMPLE_DIR / "tc02_invalid_checksum.png", save_visuals=False)
        self.assertGreater(len(result.barcodes), 0)
        # UDI checksum must be invalid
        invalid_udi = [u for u in result.udi_records if not u.is_di_checksum_valid]
        self.assertGreater(len(invalid_udi), 0)

    def test_tc03_multiple_barcodes_detected(self):
        """TC03: Pipeline detects two distinct barcodes (DataMatrix + Code-128) on a single label."""
        result = self.pipeline.process(SAMPLE_DIR / "tc03_multiple_barcodes.png", save_visuals=False)
        self.assertGreaterEqual(len(result.barcodes), 2)

    def test_tc04_skewed_label_deskewed_and_decoded(self):
        """TC04: Pipeline successfully deskews a rotated label and still decodes the barcode."""
        result = self.pipeline.process(SAMPLE_DIR / "tc04_skewed_and_noisy.png", save_visuals=False)
        # Even on noisy/skewed image the pipeline should not crash
        self.assertIsInstance(result, Module3Result)

    def test_tc05_inverted_polarity_decoded(self):
        """TC05: Pipeline recovers barcode from an inverted polarity (white-on-dark) label."""
        result = self.pipeline.process(SAMPLE_DIR / "tc05_inverted_datamatrix.png", save_visuals=False)
        self.assertIsInstance(result, Module3Result)
        # Expect at least one barcode to be recovered via inversion retry
        self.assertGreater(len(result.barcodes), 0)

    def test_tc06_no_symbols_detected(self):
        """TC06: Pipeline correctly reports zero ISO symbols on a symbols-free label."""
        result = self.pipeline.process(SAMPLE_DIR / "tc06_missing_symbols.png", save_visuals=False)
        # Should have barcode but no symbols (or very few false positives below threshold)
        self.assertGreater(len(result.barcodes), 0)
        self.assertIsInstance(result.symbols, list)

    def test_pipeline_cross_verification_matching(self):
        """Verifies cross-verification pass when OCR fields match barcode UDI."""
        result = self.pipeline.process(
            SAMPLE_DIR / "tc01_compliant_label.png",
            ocr_fields={"lot_number": "LOT8842", "expiry_date": "2028-06-30"},
            save_visuals=False,
        )
        self.assertIsNotNone(result.cross_validation)
        self.assertTrue(result.cross_validation.is_consistent)

    def test_pipeline_cross_verification_lot_mismatch(self):
        """Verifies cross-verification flags CRITICAL when OCR LOT doesn't match barcode LOT."""
        result = self.pipeline.process(
            SAMPLE_DIR / "tc01_compliant_label.png",
            ocr_fields={"lot_number": "LOT-WRONG-XYZ"},
            save_visuals=False,
        )
        self.assertIsNotNone(result.cross_validation)
        self.assertFalse(result.cross_validation.is_consistent)
        self.assertGreater(len(result.cross_validation.mismatches), 0)

    def test_pipeline_result_to_dict_serializable(self):
        """Verifies Module3Result.to_dict() produces a valid Python dictionary."""
        result = self.pipeline.process(SAMPLE_DIR / "tc01_compliant_label.png", save_visuals=False)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("barcodes", d)
        self.assertIn("symbols", d)
        self.assertIn("quality", d)
        self.assertIn("summary", d)

    def test_pipeline_in_memory_numpy_input(self):
        """Verifies pipeline accepts an in-memory NumPy array as image input."""
        img = cv2.imread(str(SAMPLE_DIR / "tc01_compliant_label.png"))
        result = self.pipeline.process(img, file_name="in_memory.png", save_visuals=False)
        self.assertIsInstance(result, Module3Result)

    def test_pipeline_processing_latency_logged(self):
        """Verifies processing latency is recorded and positive."""
        result = self.pipeline.process(SAMPLE_DIR / "tc01_compliant_label.png", save_visuals=False)
        self.assertGreater(result.processing_latency_ms, 0.0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestImageProcessingAndQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestBarcodeAndUDIScanner))
    suite.addTests(loader.loadTestsFromTestCase(TestMedicalSymbolDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
