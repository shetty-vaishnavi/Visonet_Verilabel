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
        """Checks if paddleocr or rapidocr_onnxruntime is installed and importable."""
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def _initialize(self):
        """Lazy initialization of PaddleOCR or RapidOCR model weights."""
        if self._initialized:
            return
        if not self.is_available():
            raise OCRExtractionError(
                "Neither PaddleOCR nor RapidOCR is installed. Please install 'rapidocr-onnxruntime' or use the fallback OCR engine.",
                details={"engine": "PaddleOCR"}
            )

        # 1. Try native PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False
            )
            self._is_rapid = False
            self._initialized = True
            logger.info("PaddleOCR engine initialized successfully.")
            return
        except Exception:
            pass

        # 2. Try RapidOCR (PaddleOCR on ONNXRuntime)
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._is_rapid = True
            self._initialized = True
            logger.info("RapidOCR (PaddleOCR ONNX) engine initialized successfully.")
        except Exception as e:
            raise OCRExtractionError(f"Failed to initialize RapidOCR / PaddleOCR: {e}") from e

    def extract(self, image: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> OCRResult:
        """
        Runs PaddleOCR/RapidOCR on the image and structures results into standard OCRResult.
        """
        self._initialize()

        try:
            word_boxes: List[OCRWordBox] = []
            lines: List[str] = []
            confidences: List[float] = []

            if getattr(self, "_is_rapid", False):
                # RapidOCR returns (result, elapse_list)
                # result: list of [box_points, text, confidence]
                results, _ = self._ocr(image)
                if not results:
                    return OCRResult(
                        raw_text="",
                        lines=[],
                        word_boxes=[],
                        engine_name="RapidOCR (PaddleOCR ONNX)",
                        average_confidence=0.0
                    )

                for item in results:
                    if not item or len(item) < 3:
                        continue
                    box_points, text, conf = item[0], str(item[1]).strip(), float(item[2])
                    if not text:
                        continue
                    bbox = [(int(p[0]), int(p[1])) for p in box_points]
                    word_boxes.append(OCRWordBox(text=text, confidence=conf, bbox=bbox))
                    lines.append(text)
                    confidences.append(conf)

                engine_name = "RapidOCR (PaddleOCR ONNX)"
            else:
                # Native PaddleOCR
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
                for item in page_results:
                    bbox_points, (text, conf) = item
                    text = text.strip()
                    if not text:
                        continue
                    bbox = [(int(p[0]), int(p[1])) for p in bbox_points]
                    word_boxes.append(OCRWordBox(text=text, confidence=float(conf), bbox=bbox))
                    lines.append(text)
                    confidences.append(float(conf))

                engine_name = "PaddleOCR"

            raw_text = "\n".join(lines)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return OCRResult(
                raw_text=raw_text,
                lines=lines,
                word_boxes=word_boxes,
                engine_name=engine_name,
                average_confidence=avg_conf,
                metadata={"total_detections": len(word_boxes)}
            )

        except Exception as e:
            raise OCRExtractionError(f"OCR extraction failed: {e}") from e
