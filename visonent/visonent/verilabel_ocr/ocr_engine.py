"""
OCR Engine implementation for VeriLabel AI.
Supports:
  1. PaddleOCR (native PaddlePaddle backend)
  2. RapidOCR (PaddleOCR models executed on ONNXRuntime for ultra-fast, cross-platform inference)
  3. Fallback engine (OpenCV layout & morphological analyzer / Tesseract)
"""

from typing import List, Optional, Tuple, Any
import logging
import numpy as np
import cv2

from .config import OCRConfig
from .models import OCRTextBlock, BoundingBox

logger = logging.getLogger("verilabel_ocr.engine")


class OCREngine:
    """Base interface for OCR engines."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()

    def extract_text_blocks(self, image: np.ndarray) -> List[OCRTextBlock]:
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """
    Native PaddleOCR engine wrapper.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(config)
        self._ocr_instance = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR (lang=%s)", self.config.lang)
            self._ocr_instance = PaddleOCR(
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.lang,
                show_log=False,
            )
            self._initialized = True
        except Exception as e:
            logger.info("Native PaddleOCR not loaded (%s). Checking RapidOCR.", e)
            self._ocr_instance = None
            self._initialized = True

    def is_available(self) -> bool:
        self._ensure_initialized()
        return self._ocr_instance is not None

    def extract_text_blocks(self, image: np.ndarray) -> List[OCRTextBlock]:
        self._ensure_initialized()
        if self._ocr_instance is None:
            # Delegate to RapidOCR
            rapid = RapidOCREngine(self.config)
            return rapid.extract_text_blocks(image)

        results = self._ocr_instance.ocr(image, cls=self.config.use_angle_cls)
        blocks: List[OCRTextBlock] = []
        if not results or results[0] is None:
            return blocks

        page_results = results[0] if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list) else results

        for item in page_results:
            if not item or len(item) < 2:
                continue
            polygon, text_info = item[0], item[1]
            if not isinstance(text_info, (tuple, list)) or len(text_info) < 2:
                continue

            text = str(text_info[0]).strip()
            confidence = float(text_info[1])
            if confidence < self.config.drop_score or not text:
                continue

            poly_np = np.array(polygon, dtype=np.int32)
            xmin = int(np.min(poly_np[:, 0]))
            ymin = int(np.min(poly_np[:, 1]))
            xmax = int(np.max(poly_np[:, 0]))
            ymax = int(np.max(poly_np[:, 1]))

            bbox = BoundingBox(
                xmin=max(0, xmin),
                ymin=max(0, ymin),
                xmax=xmax,
                ymax=ymax,
                polygon=[[int(p[0]), int(p[1])] for p in polygon],
            )
            blocks.append(
                OCRTextBlock(
                    id=len(blocks),
                    text=text,
                    confidence=round(confidence, 4),
                    bbox=bbox,
                )
            )

        blocks.sort(key=lambda b: (b.bbox.ymin // 25, b.bbox.xmin))
        return blocks


class RapidOCREngine(OCREngine):
    """
    RapidOCR engine running PaddleOCR's deep neural models via ONNXRuntime.
    Provides identical PP-OCR recognition accuracy with high efficiency.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(config)
        self._ocr_instance = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("Initializing RapidOCR (PaddleOCR on ONNXRuntime)...")
            self._ocr_instance = RapidOCR()
            self._initialized = True
        except Exception as e:
            logger.warning("RapidOCR unavailable (%s). Using fallback engine.", e)
            self._ocr_instance = None
            self._initialized = True

    def is_available(self) -> bool:
        self._ensure_initialized()
        return self._ocr_instance is not None

    def extract_text_blocks(self, image: np.ndarray) -> List[OCRTextBlock]:
        self._ensure_initialized()
        if self._ocr_instance is None:
            fallback = FallbackOCREngine(self.config)
            return fallback.extract_text_blocks(image)

        # RapidOCR returns: (result, elapse_list)
        # where result is a list of [box_points, text, confidence]
        results, _ = self._ocr_instance(image)
        blocks: List[OCRTextBlock] = []

        if not results:
            return blocks

        for item in results:
            if not item or len(item) < 3:
                continue
            box_points = item[0]
            text = str(item[1]).strip()
            confidence = float(item[2])

            if confidence < self.config.drop_score or not text:
                continue

            pts = np.array(box_points, dtype=np.int32)
            xmin = int(np.min(pts[:, 0]))
            ymin = int(np.min(pts[:, 1]))
            xmax = int(np.max(pts[:, 0]))
            ymax = int(np.max(pts[:, 1]))

            bbox = BoundingBox(
                xmin=max(0, xmin),
                ymin=max(0, ymin),
                xmax=xmax,
                ymax=ymax,
                polygon=[[int(p[0]), int(p[1])] for p in box_points],
            )

            blocks.append(
                OCRTextBlock(
                    id=len(blocks),
                    text=text,
                    confidence=round(confidence, 4),
                    bbox=bbox,
                )
            )

        blocks.sort(key=lambda b: (b.bbox.ymin // 25, b.bbox.xmin))
        return blocks


class FallbackOCREngine(OCREngine):
    """
    Lightweight fallback engine using computer vision morphological layout analysis.
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(config)

    def extract_text_blocks(self, image: np.ndarray) -> List[OCRTextBlock]:
        blocks: List[OCRTextBlock] = []

        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                if conf < 0 or not text:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bbox = BoundingBox(
                    xmin=x, ymin=y, xmax=x + w, ymax=y + h,
                    polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                )
                blocks.append(OCRTextBlock(
                    id=len(blocks),
                    text=text,
                    confidence=round(conf / 100.0, 4),
                    bbox=bbox,
                ))
            if blocks:
                return blocks
        except Exception:
            pass

        # Morphological region detector
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 and h > 8 and w < image.shape[1] * 0.95:
                blocks.append(
                    OCRTextBlock(
                        id=len(blocks),
                        text="[Detected Text Region]",
                        confidence=0.85,
                        bbox=BoundingBox(
                            xmin=x, ymin=y, xmax=x + w, ymax=y + h,
                            polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        ),
                    )
                )

        blocks.sort(key=lambda b: (b.bbox.ymin // 30, b.bbox.xmin))
        return blocks


def create_ocr_engine(config: Optional[OCRConfig] = None) -> OCREngine:
    """Factory creating the primary OCR engine with automatic fallback cascade."""
    config = config or OCRConfig()
    if config.engine_mode == "direct":
        return FallbackOCREngine(config)

    # 1. Try native PaddleOCR
    paddle_engine = PaddleOCREngine(config)
    if paddle_engine.is_available():
        return paddle_engine

    # 2. Try RapidOCR (PaddleOCR on ONNXRuntime)
    rapid_engine = RapidOCREngine(config)
    if rapid_engine.is_available():
        return rapid_engine

    # 3. Fallback
    return FallbackOCREngine(config)
