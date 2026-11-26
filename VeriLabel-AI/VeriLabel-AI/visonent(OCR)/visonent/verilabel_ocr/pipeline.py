"""
Top-level Document Processing Pipeline for VeriLabel AI Module 2.
Coordinates preprocessing, OCR inference, medical entity extraction,
visual overlay generation, and Member 3 Computer Vision / Barcode integration.
"""

from typing import Union, Optional, List, Dict, Any
import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .models import OCRResult, OCRTextBlock, MedicalLabelData
from .preprocessor import DocumentPreprocessor
from .ocr_engine import create_ocr_engine, OCREngine
from .field_extractor import MedicalFieldExtractor
from .visualizer import LabelVisualizer

# Dynamically link Member 3 modules from Backend
_backend_dir = Path(__file__).resolve().parent.parent.parent.parent / "visionet(Backend)" / "visionet"
if _backend_dir.exists() and str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

try:
    from verilabel.modules.barcode.barcode_scanner import BarcodeScanner
    from verilabel.modules.barcode.udi_validator import UDIValidator
    from verilabel.modules.vision.symbol_detector import MedicalSymbolDetector
    _BARCODE_AVAILABLE = True
except ImportError:
    _BARCODE_AVAILABLE = False

logger = logging.getLogger("verilabel_ocr.pipeline")


class DocumentProcessingPipeline:
    """
    Main entry point for Module 2: OCR and Document Processing,
    integrated with Member 3's Computer Vision and Barcode Module.
    Usage:
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process("path/to/label.png")
        print(result.to_json())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.preprocessor = DocumentPreprocessor(self.config.preprocess)
        self.ocr_engine = create_ocr_engine(self.config.ocr)
        self.extractor = MedicalFieldExtractor()
        self.visualizer = LabelVisualizer()

        # Member 3: Barcode, UDI, and Symbol Detection Integration
        self.barcode_scanner = BarcodeScanner() if _BARCODE_AVAILABLE else None
        self.udi_validator = UDIValidator() if _BARCODE_AVAILABLE else None
        self.symbol_detector = MedicalSymbolDetector() if _BARCODE_AVAILABLE else None

        os.makedirs(self.config.output_dir, exist_ok=True)

    def process(
        self,
        input_source: Union[str, bytes, np.ndarray, Image.Image],
        file_name: Optional[str] = None,
        save_visualization: bool = True,
        output_vis_path: Optional[str] = None,
    ) -> OCRResult:
        """
        Executes the full OCR and document processing workflow on an input document or image.
        Integrates text extraction, barcode decoding, UDI parsing, and ISO symbol detection.
        """
        start_time = time.perf_counter()
        logs: List[str] = []

        # Determine file name
        if not file_name:
            if isinstance(input_source, str):
                file_name = os.path.basename(input_source)
            else:
                file_name = f"document_{int(time.time())}.png"

        logs.append(f"Starting processing for '{file_name}'")

        try:
            # 1. Load document (handles PDF conversion if applicable)
            raw_image, load_meta = self.preprocessor.load_document(input_source)
            logs.append(f"Document loaded successfully. PDF: {load_meta.get('is_pdf')}, "
                        f"Orig size: {load_meta.get('original_width')}x{load_meta.get('original_height')}")

            # 2. Preprocess: Deskewing, Noise Reduction & CLAHE
            color_enhanced, ocr_ready, prep_stats = self.preprocessor.preprocess(raw_image)
            deskew_angle = prep_stats.get("deskew_angle", 0.0)
            logs.append(f"Preprocessing completed. Deskew angle: {deskew_angle}°, Final size: "
                        f"{prep_stats.get('final_width')}x{prep_stats.get('final_height')}")

            # 3. OCR Text Extraction (RapidOCR / PaddleOCR)
            text_blocks: List[OCRTextBlock] = self.ocr_engine.extract_text_blocks(ocr_ready)
            logs.append(f"OCR extraction completed. Total text blocks found: {len(text_blocks)}")

            # 4. Concatenate raw text
            full_text = "\n".join([b.text for b in text_blocks])

            # 5. Extract structured medical fields
            structured_data: MedicalLabelData = self.extractor.extract(text_blocks)
            logs.append("Medical field extraction completed.")

            # 6. Member 3: Barcode Detection, Decoding, and UDI Validation
            barcodes_dicts: List[Dict[str, Any]] = []
            udi_dict: Optional[Dict[str, Any]] = None
            scanned_barcodes = []

            if self.barcode_scanner:
                try:
                    scanned_barcodes = self.barcode_scanner.scan(raw_image)
                    barcodes_dicts = [b.to_dict() for b in scanned_barcodes]
                    logs.append(f"Barcode scanning completed. Barcodes found: {len(scanned_barcodes)}")

                    if scanned_barcodes and self.udi_validator:
                        for bc in scanned_barcodes:
                            cand = self.udi_validator.parse_udi(bc.data)
                            if cand.is_valid_format or cand.device_identifier_di:
                                udi_dict = cand.to_dict()
                                break
                        if not udi_dict and scanned_barcodes:
                            udi_dict = self.udi_validator.parse_udi(scanned_barcodes[0].data).to_dict()
                except Exception as e:
                    logger.warning(f"Barcode scan integration notice: {e}")

            # 7. Member 3: Medical Symbol Detection
            symbols_dicts: List[Dict[str, Any]] = []
            detected_symbols = []
            if self.symbol_detector:
                try:
                    detected_symbols = self.symbol_detector.detect_symbols(raw_image)
                    symbols_dicts = [s.to_dict() for s in detected_symbols]
                    logs.append(f"Symbol detection completed. Symbols found: {len(detected_symbols)}")
                except Exception as e:
                    logger.warning(f"Symbol detection integration notice: {e}")

            # 8. Generate color-coded visualization if requested
            annotated_path = None
            if save_visualization:
                annotated_img = self.visualizer.annotate(
                    image=color_enhanced,
                    structured_data=structured_data,
                    all_blocks=text_blocks,
                )

                # Overlay symbols and barcodes if detected
                if self.symbol_detector and detected_symbols:
                    annotated_img = self.symbol_detector.draw_detections(annotated_img, detected_symbols)
                if self.barcode_scanner and scanned_barcodes:
                    annotated_img = self.barcode_scanner.draw_barcodes(annotated_img, scanned_barcodes)

                if not output_vis_path:
                    base_name, _ = os.path.splitext(file_name)
                    vis_filename = f"{base_name}_annotated.png"
                    output_vis_path = os.path.join(self.config.output_dir, vis_filename)

                annotated_path = self.visualizer.save_annotated(annotated_img, output_vis_path)
                logs.append(f"Annotated visualization saved to '{annotated_path}'")

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logs.append(f"Pipeline finished in {elapsed_ms} ms")

            return OCRResult(
                success=True,
                file_name=file_name,
                file_path=input_source if isinstance(input_source, str) else None,
                processing_time_ms=elapsed_ms,
                image_width=prep_stats.get("final_width", raw_image.shape[1]),
                image_height=prep_stats.get("final_height", raw_image.shape[0]),
                deskew_angle=deskew_angle,
                raw_text=full_text,
                total_blocks_detected=len(text_blocks),
                text_blocks=text_blocks,
                structured_data=structured_data,
                annotated_image_path=annotated_path,
                barcodes=barcodes_dicts,
                udi=udi_dict,
                symbols=symbols_dicts,
                logs=logs,
            )

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception("OCR Pipeline error on %s", file_name)
            logs.append(f"ERROR: {str(e)}")

            return OCRResult(
                success=False,
                file_name=file_name,
                file_path=input_source if isinstance(input_source, str) else None,
                processing_time_ms=elapsed_ms,
                image_width=0,
                image_height=0,
                deskew_angle=0.0,
                raw_text="",
                total_blocks_detected=0,
                text_blocks=[],
                structured_data=MedicalLabelData(),
                annotated_image_path=None,
                barcodes=[],
                udi=None,
                symbols=[],
                logs=logs,
            )

    def process_batch(self, file_paths: List[str]) -> List[OCRResult]:
        """Processes a list of files sequentially and returns results."""
        results = []
        for path in file_paths:
            res = self.process(path)
            results.append(res)
        return results
