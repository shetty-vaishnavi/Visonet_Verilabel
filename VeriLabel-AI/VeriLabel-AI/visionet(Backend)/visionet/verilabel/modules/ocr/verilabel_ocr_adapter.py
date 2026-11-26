"""
VeriLabel AI - OCR Adapter for visonent(OCR) verilabel_ocr Pipeline
Team: Brahmastra | GITAM University
Member 3: Computer Vision and Barcode Module

Integrates the standalone verilabel_ocr pipeline from visonent(OCR) as an
executable BaseOCREngine for the central VeriLabel backend pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np

from verilabel.modules.ocr.base import BaseOCREngine, OCRResult, OCRWordBox
from verilabel.exceptions import OCRExtractionError

logger = logging.getLogger("VeriLabel.OCR.Adapter")

# Add OCR folder to sys.path if not present
_ocr_pkg_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "visonent(OCR)" / "visonent"
if _ocr_pkg_dir.exists() and str(_ocr_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_ocr_pkg_dir))


class VeriLabelOCRPipelineEngine(BaseOCREngine):
    """
    Bridge adapter executing the complete DocumentProcessingPipeline from the
    visonent(OCR) component within the central VeriLabel backend pipeline.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or "ocr_output"
        self._pipeline = None
        self._initialized = False

    def is_available(self) -> bool:
        """Checks if verilabel_ocr is importable."""
        try:
            import verilabel_ocr.pipeline  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_initialized(self):
        if self._initialized:
            return
        if not self.is_available():
            raise OCRExtractionError(
                "verilabel_ocr package from visonent(OCR) is not importable. "
                "Ensure visonent(OCR)/visonent is on sys.path or use PaddleOCREngine.",
                details={"engine": "verilabel_ocr"}
            )
        try:
            from verilabel_ocr.pipeline import DocumentProcessingPipeline
            from verilabel_ocr.config import PipelineConfig
            config = PipelineConfig(output_dir=self.output_dir)
            self._pipeline = DocumentProcessingPipeline(config)
            self._initialized = True
            logger.info("VeriLabelOCRPipelineEngine (from visonent) loaded successfully.")
        except Exception as e:
            raise OCRExtractionError(f"Failed to initialize verilabel_ocr pipeline: {e}") from e

    def extract(self, image: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> OCRResult:
        """
        Runs the full verilabel_ocr pipeline on the image and translates results into
        the backend's standard OCRResult container.
        """
        self._ensure_initialized()
        try:
            input_source = str(file_path) if file_path else image
            ocr_res = self._pipeline.process(input_source, save_visualization=False)

            word_boxes: List[OCRWordBox] = []
            lines: List[str] = []

            for block in ocr_res.text_blocks:
                b = block.bbox
                poly: List[Tuple[int, int]] = []
                if b.polygon and len(b.polygon) >= 4:
                    poly = [(int(p[0]), int(p[1])) for p in b.polygon]
                else:
                    poly = [(b.xmin, b.ymin), (b.xmax, b.ymin), (b.xmax, b.ymax), (b.xmin, b.ymax)]

                word_boxes.append(
                    OCRWordBox(
                        text=block.text,
                        confidence=block.confidence,
                        bbox=poly
                    )
                )
                lines.append(block.text)

            avg_conf = (
                float(np.mean([b.confidence for b in word_boxes]))
                if word_boxes else 0.0
            )

            return OCRResult(
                raw_text=ocr_res.raw_text,
                lines=lines,
                word_boxes=word_boxes,
                engine_name="verilabel_ocr (visonent)",
                average_confidence=avg_conf,
                metadata={
                    "total_detections": len(word_boxes),
                    "deskew_angle": ocr_res.deskew_angle,
                    "processing_time_ms": ocr_res.processing_time_ms
                }
            )
        except Exception as e:
            raise OCRExtractionError(f"VeriLabelOCRPipelineEngine extraction failed: {e}") from e
