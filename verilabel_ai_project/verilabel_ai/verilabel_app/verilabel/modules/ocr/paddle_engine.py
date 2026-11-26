"""
VeriLabel AI - Module 2: PaddleOCR Integration
Team: Brahmastra | GITAM University
Integrates PaddleOCR for high-accuracy text extraction from medical device labels.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

from verilabel.modules.ocr.base import BaseOCREngine, OCRResult, OCRWordBox
from verilabel.exceptions import OCRExtractionError

logger = logging.getLogger("VeriLabel.OCR.Paddle")


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR implementation for multilingual medical label text extraction.
    Supports angles, rotated text, and compact fonts common on medical labels.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, use_gpu: bool = False):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = None
        self._initialized = False

    def is_available(self) -> bool:
        """Checks if paddleocr is installed and importable."""
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False

    def _initialize(self):
        """Lazy initialization of PaddleOCR model weights."""
        if self._initialized:
            return
        if not self.is_available():
            raise OCRExtractionError(
                "PaddleOCR is not installed. Please install 'paddlepaddle' and 'paddleocr' or use the fallback OCR engine.",
                details={"engine": "PaddleOCR"}
            )
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False
            )
            self._initialized = True
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            raise OCRExtractionError(f"Failed to initialize PaddleOCR: {e}") from e

    def extract(self, image: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> OCRResult:
        """
        Runs PaddleOCR on the image and structures results into standard OCRResult.
        """
        self._initialize()

        try:
            # PaddleOCR expects BGR or RGB numpy array
            results = self._ocr.ocr(image, cls=self.use_angle_cls)
            if not results or not results[0]:
                return OCRResult(
                    raw_text="",
                    lines=[],
                    word_boxes=[],
                    engine_name="PaddleOCR",
                    average_confidence=0.0
                )

            page_results = results[0]
            word_boxes: List[OCRWordBox] = []
            lines: List[str] = []
            confidences: List[float] = []

            for item in page_results:
                # item format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('detected_text', confidence)]
                bbox_points, (text, conf) = item
                text = text.strip()
                if not text:
                    continue

                bbox = [(int(p[0]), int(p[1])) for p in bbox_points]
                word_boxes.append(OCRWordBox(text=text, confidence=float(conf), bbox=bbox))
                lines.append(text)
                confidences.append(float(conf))

            raw_text = "\n".join(lines)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return OCRResult(
                raw_text=raw_text,
                lines=lines,
                word_boxes=word_boxes,
                engine_name="PaddleOCR",
                average_confidence=avg_conf,
                metadata={"total_detections": len(word_boxes)}
            )

        except Exception as e:
            raise OCRExtractionError(f"PaddleOCR extraction failed: {e}") from e
