"""
VeriLabel AI - Central Integration Pipeline
Team: Brahmastra | GITAM University
Role: Member 1 - Backend Developer & Integration

The primary integration facade orchestrating:
Module 1: Document Processing & Validation
Module 2: OCR & Field Identification
Module 3: Computer Vision & Symbol Detection
Module 4: Barcode & UDI Validation
Module 5: Compliance Rule Engine
Module 6: Report Generation & Presentation
Module 7: SQLite Database & Audit Traceability
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

from verilabel.config import AppConfig, config as default_config
from verilabel.exceptions import VeriLabelException, DocumentProcessingError
from verilabel.core.document_processor import DocumentProcessor
from verilabel.modules.ocr.base import BaseOCREngine, OCRResult
from verilabel.modules.ocr.paddle_engine import PaddleOCREngine
from verilabel.modules.ocr.fallback_engine import FallbackOCREngine
from verilabel.modules.ocr.field_extractor import MedicalFieldExtractor, ExtractedLabelFields
from verilabel.modules.vision.symbol_detector import MedicalSymbolDetector, DetectedSymbol
from verilabel.modules.barcode.barcode_scanner import BarcodeScanner, DecodedBarcode
from verilabel.modules.barcode.udi_validator import UDIValidator, UDIComponents
from verilabel.modules.compliance.engine import ComplianceEngine
from verilabel.modules.compliance.models import LabelComplianceReport
from verilabel.modules.reporting.generator import ReportGenerator
from verilabel.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VeriLabel.Pipeline")


@dataclass
class PipelineResult:
    """Comprehensive container returned by the VeriLabel pipeline to Streamlit frontend."""
    report: LabelComplianceReport
    annotated_image: np.ndarray
    dataframe: pd.DataFrame
    summary: Dict[str, Any]
    markdown_report: str
    validation_id: int
    raw_ocr: OCRResult
    fields: ExtractedLabelFields
    symbols: List[DetectedSymbol]
    barcodes: List[DecodedBarcode]
    udi: Optional[UDIComponents]
    file_metadata: Dict[str, Any]


class VeriLabelPipeline:
    """
    Unified compliance verification engine for medical device labels.
    Provides clean high-level methods for Streamlit frontend and REST APIs.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        symbol_detector: Optional[MedicalSymbolDetector] = None
    ):
        self.config = config or default_config

        # 1. Document Processor (Module 1)
        self.doc_processor = DocumentProcessor()

        # 2. OCR Engine (Module 2)
        if ocr_engine:
            self.ocr_engine = ocr_engine
        else:
            paddle = PaddleOCREngine()
            if paddle.is_available():
                self.ocr_engine = paddle
                logger.info("Using PaddleOCR Engine.")
            else:
                self.ocr_engine = FallbackOCREngine()
                logger.info("PaddleOCR not installed; using Fallback OCR Engine.")

        self.field_extractor = MedicalFieldExtractor()

        # 3. Vision & Symbol Detector (Module 3)
        self.symbol_detector = symbol_detector or MedicalSymbolDetector(
            templates_dir=self.config.symbols_dir,
            threshold=self.config.symbol_match_min
        )

        # 4. Barcode & UDI Scanner (Module 4)
        self.barcode_scanner = BarcodeScanner()
        self.udi_validator = UDIValidator()

        # 5. Compliance Engine (Module 5)
        self.compliance_engine = ComplianceEngine(rules_file=self.config.rules_file)

        # 6. Database Manager (Module 7)
        self.db = DatabaseManager(db_path=self.config.db_path)
        self.db.seed_demo_products()

    def process_label(
        self,
        file_path: Union[str, Path],
        product_code: Optional[str] = None,
        save_to_db: bool = True
    ) -> PipelineResult:
        """
        Executes the entire end-to-end verification pipeline:
        File -> Ingestion & Quality -> OCR -> Field Extraction -> Symbol Detection
        -> Barcode/UDI Validation -> Compliance Engine -> DB Persistence -> Report Generation.
        """
        start_time = time.perf_counter()
        path = Path(file_path)
        logger.info(f"Initiating label verification for: {path.name}")

        # Step 1: Ingestion & Preprocessing (Module 1)
        img, file_meta = self.doc_processor.load_document(path)
        preprocessed = self.doc_processor.preprocess_image(img)

        # Step 2: OCR & Text Extraction (Module 2)
        ocr_input = preprocessed.get("clahe", img)
        raw_ocr = self.ocr_engine.extract(ocr_input, file_path=path)

        # Step 3: Structured Regulatory Field Extraction (Module 2)
        extracted_fields = self.field_extractor.extract_fields(raw_ocr)

        # Step 4: Symbol Detection via Computer Vision (Module 3)
        detected_symbols = self.symbol_detector.detect_symbols(img)

        # Step 5: Barcode & UDI Extraction (Module 4)
        scanned_barcodes = self.barcode_scanner.scan(img)
        parsed_udi: Optional[UDIComponents] = None

        # Look for UDI payload from barcode scan or extracted text
        if scanned_barcodes:
            # Prefer 1D/2D barcode payload
            barcode_str = scanned_barcodes[0].data
            parsed_udi = self.udi_validator.parse_udi(barcode_str)
        elif extracted_fields.raw_text:
            # Fallback: check if UDI string appears in text
            import re
            udi_matches = re.findall(r"\((?:01|17|10|21)\)[0-9A-Za-z]+", extracted_fields.raw_text)
            if udi_matches:
                full_carrier = "".join(udi_matches)
                parsed_udi = self.udi_validator.parse_udi(full_carrier)

        # Step 6: Visual Annotations (Overlay symbols & barcodes on image)
        annotated_image = img.copy()
        if detected_symbols:
            annotated_image = self.symbol_detector.draw_detections(annotated_image, detected_symbols)
        if scanned_barcodes:
            annotated_image = self.barcode_scanner.draw_barcodes(annotated_image, scanned_barcodes)

        # Step 7: Compliance Rule Evaluation (Module 5)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        report = self.compliance_engine.evaluate(
            file_name=path.name,
            file_hash=file_meta["sha256"],
            fields=extracted_fields,
            symbols=detected_symbols,
            barcodes=scanned_barcodes,
            udi=parsed_udi,
            processing_time_ms=elapsed_ms
        )

        # Step 8: Database Audit & Persistence (Module 7)
        val_id = -1
        if save_to_db:
            val_id = self.db.save_validation_run(
                file_meta=file_meta,
                report=report,
                product_code=product_code
            )

        # Step 9: Report Generation & Presentation Models (Module 6)
        df = ReportGenerator.to_dataframe(report)
        summary = ReportGenerator.to_executive_summary(report)
        md_report = ReportGenerator.to_markdown(report)

        logger.info(
            f"Verification complete for '{path.name}' | Verdict: {report.verdict.value} | "
            f"Score: {report.compliance_score:.1f}% | Time: {elapsed_ms:.1f}ms"
        )

        return PipelineResult(
            report=report,
            annotated_image=annotated_image,
            dataframe=df,
            summary=summary,
            markdown_report=md_report,
            validation_id=val_id,
            raw_ocr=raw_ocr,
            fields=extracted_fields,
            symbols=detected_symbols,
            barcodes=scanned_barcodes,
            udi=parsed_udi,
            file_metadata=file_meta
        )

    def submit_human_review(
        self,
        validation_id: int,
        reviewer_name: str,
        reviewer_role: str,
        decision: str,
        review_notes: Optional[str] = None
    ) -> int:
        """Submits human review sign-off to SQLite audit history."""
        return self.db.record_human_review(
            validation_id=validation_id,
            reviewer_name=reviewer_name,
            reviewer_role=reviewer_role,
            decision=decision,
            review_notes=review_notes
        )

    def get_audit_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves historical audit entries for the Streamlit dashboard."""
        return self.db.list_validations(limit=limit)

    def get_validation_details(self, validation_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves complete validation run with review log."""
        return self.db.get_validation_record(validation_id)
