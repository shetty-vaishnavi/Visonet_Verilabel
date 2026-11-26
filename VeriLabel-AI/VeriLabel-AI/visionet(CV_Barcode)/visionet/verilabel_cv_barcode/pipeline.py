"""
VeriLabel AI - Module 3: Central Computer Vision & Barcode Pipeline Orchestrator
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Responsibilities:
- Unified execution pipeline:
    1. Document ingestion and quality diagnostics (ImageProcessor)
    2. Automated deskewing and enhancement
    3. Multi-engine barcode detection and decoding (BarcodeScanner)
    4. Regulatory UDI parsing and Modulo-10 checksum verification (UDIValidator)
    5. Multi-scale ISO 15223-1 symbol detection with NMS (MedicalSymbolDetector)
    6. Optional cross-verification with OCR fields (FDA 21 CFR 830.20 / EU MDR 23.2(h))
    7. Annotated visual overlay rendering and export (LabelVisualizer)
- Clean, decoupled API designed for seamless integration with Member 1 (Backend) and Member 4 (Frontend UI)
"""

import time
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np
import io
from PIL import Image

from .config import PipelineConfig, DEFAULT_OUTPUT_DIR
from .models import Module3Result, DecodedBarcode, UDIComponents, DetectedSymbol, CrossValidationResult
from .image_processor import ImageProcessor
from .barcode_scanner import BarcodeScanner
from .udi_validator import UDIValidator
from .symbol_detector import MedicalSymbolDetector
from .visualizer import LabelVisualizer

logger = logging.getLogger("VeriLabel.CV.Pipeline")


class ComputerVisionBarcodePipeline:
    """
    Production pipeline orchestrating OpenCV image processing, barcode decoding,
    UDI checksum validation, and ISO symbol recognition.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.image_processor = ImageProcessor(
            blur_threshold=self.config.blur_threshold,
            min_brightness=self.config.min_brightness,
            max_brightness=self.config.max_brightness,
        )
        self.barcode_scanner = BarcodeScanner(
            enable_adaptive_retries=self.config.enable_adaptive_barcode_retries,
            enable_rotations=self.config.enable_orthogonal_barcode_rotations,
            enable_polarity_inversion=self.config.enable_inverted_barcode_polarity,
        )
        self.udi_validator = UDIValidator()
        self.symbol_detector = MedicalSymbolDetector(
            templates_dir=self.config.templates_dir,
            match_threshold=self.config.symbol_match_threshold,
            scales=self.config.scales,
        )
        self.visualizer = LabelVisualizer()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
        file_name: Optional[str] = None,
        ocr_fields: Optional[Dict[str, str]] = None,
        save_visuals: Optional[bool] = None,
    ) -> Module3Result:
        """
        Executes full Module 3 verification pipeline.

        Args:
            source: Image input (file path, raw bytes, BytesIO, PIL Image, or NumPy array).
            file_name: Optional label filename for reporting.
            ocr_fields: Optional dictionary containing extracted OCR text fields for cross-verification:
                        e.g., {"lot_number": "BATCH123", "expiry_date": "2028-06-30", "serial_number": "SN99"}
            save_visuals: Whether to save annotated inspection overlay image to disk.

        Returns:
            Module3Result dataclass containing quality metrics, decoded barcodes,
            UDI components, detected ISO symbols, and execution metadata.
        """
        t0 = time.perf_counter()

        # Determine file name
        if file_name is None:
            if isinstance(source, (str, Path)):
                file_name = Path(source).name
            else:
                file_name = "in_memory_label.png"

        # 1. Ingestion & Integrity Check
        image, meta = self.image_processor.load_image(source)
        orig_h, orig_w = image.shape[:2]

        # 2. Quality Diagnostics (Blur, Exposure, Contrast, Skew)
        quality = self.image_processor.assess_quality(image)

        # 3. Deskewing (if enabled and skewed)
        working_image = image
        if self.config.enable_deskew and abs(quality.estimated_skew_angle) > 1.0:
            working_image, _ = self.image_processor.deskew(image, angle=quality.estimated_skew_angle)

        # 4. Barcode Scanning & Decoding
        barcodes = self.barcode_scanner.scan(working_image)

        # 5. UDI Extraction & Regulatory Standards Validation
        udi_records: List[UDIComponents] = []
        for bc in barcodes:
            parsed_udi = self.udi_validator.parse_udi(bc.data)
            udi_records.append(parsed_udi)

        # 6. Medical Symbol Detection (ISO 15223-1 & Regulatory Emblems)
        detected_symbols = self.symbol_detector.detect(working_image)

        # 7. Bidirectional Cross-Verification with OCR Fields (if provided)
        cross_val: Optional[CrossValidationResult] = None
        if ocr_fields and udi_records:
            primary_udi = udi_records[0]
            cross_val = self.udi_validator.cross_verify_with_ocr(
                udi=primary_udi,
                ocr_lot=ocr_fields.get("lot_number") or ocr_fields.get("lot") or ocr_fields.get("batch"),
                ocr_expiry=ocr_fields.get("expiry_date") or ocr_fields.get("expiry") or ocr_fields.get("exp"),
                ocr_serial=ocr_fields.get("serial_number") or ocr_fields.get("serial") or ocr_fields.get("sn"),
            )

        # 8. Visual Inspection Overlay Generation & Persistence
        annotated_path_str: Optional[str] = None
        should_save = save_visuals if save_visuals is not None else self.config.save_annotated_image

        if should_save:
            annotated_overlay = self.visualizer.create_overlay(
                image=working_image,
                barcodes=barcodes,
                udi_records=udi_records,
                symbols=detected_symbols,
                quality=quality,
            )
            out_stem = Path(file_name).stem
            out_file = self.config.output_dir / f"{out_stem}_annotated.png"
            self.visualizer.save_image(annotated_overlay, out_file)
            annotated_path_str = str(out_file)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return Module3Result(
            file_name=file_name,
            file_hash_sha256=meta["sha256"],
            image_width=orig_w,
            image_height=orig_h,
            quality=quality,
            barcodes=barcodes,
            udi_records=udi_records,
            symbols=detected_symbols,
            cross_validation=cross_val,
            annotated_image_path=annotated_path_str,
            processing_latency_ms=elapsed_ms,
        )
