"""
VeriLabel AI - Standalone Fallback & Testing OCR Engine
Team: Brahmastra | GITAM University
Provides reliable OCR text extraction for offline development, CI/CD, and testing.
Uses PyTesseract if available, or an intelligent text extractor with image region detection.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2

from verilabel.modules.ocr.base import BaseOCREngine, OCRResult, OCRWordBox

logger = logging.getLogger("VeriLabel.OCR.Fallback")


class FallbackOCREngine(BaseOCREngine):
    """
    Robust fallback OCR engine.
    - If pytesseract is installed and tesseract-ocr binary is present, it uses tesseract.
    - Otherwise, provides simulated label text detection using OpenCV morphological contour detection
      and pre-annotated test payload mappings.
    """

    def __init__(self, mocked_text_map: Optional[Dict[str, str]] = None):
        self.mocked_text_map = mocked_text_map or {}
        self.has_pytesseract = False
        try:
            import pytesseract
            self._pytesseract = pytesseract
            self.has_pytesseract = True
        except ImportError:
            self._pytesseract = None

    def is_available(self) -> bool:
        return True

    def register_mock_text(self, image_signature: str, text: str):
        """Allows test suites to register exact ground truth OCR text for synthetic images."""
        self.mocked_text_map[image_signature] = text

    def extract(self, image: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> OCRResult:
        """Extracts text using pytesseract, sidecar companion files, or simulated detection."""
        # 0. Check if companion text file exists (e.g., label_name.txt)
        if file_path:
            fp = Path(file_path)
            txt_sidecar = fp.with_suffix(".txt")
            if txt_sidecar.exists():
                try:
                    with open(txt_sidecar, "r", encoding="utf-8") as f:
                        sidecar_text = f.read().strip()
                    if sidecar_text:
                        lines = [l.strip() for l in sidecar_text.splitlines() if l.strip()]
                        word_boxes = []
                        for idx, line in enumerate(lines):
                            y_pos = int(50 + idx * 35)
                            words = line.split()
                            x_cursor = 40
                            for w in words:
                                w_len = len(w) * 12
                                bbox = [(x_cursor, y_pos), (x_cursor + w_len, y_pos),
                                        (x_cursor + w_len, y_pos + 25), (x_cursor, y_pos + 25)]
                                word_boxes.append(OCRWordBox(text=w, confidence=0.98, bbox=bbox))
                                x_cursor += w_len + 10
                        return OCRResult(
                            raw_text=sidecar_text,
                            lines=lines,
                            word_boxes=word_boxes,
                            engine_name="SidecarOCR",
                            average_confidence=0.98,
                            metadata={"source": "sidecar_text_file"}
                        )
                except Exception as e:
                    logger.debug(f"Failed to read sidecar text file: {e}")

        # Check if tesseract is usable
        if self.has_pytesseract:
            try:
                # Convert BGR to RGB
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
                data = self._pytesseract.image_to_data(rgb, output_type=self._pytesseract.Output.DICT)
                word_boxes: List[OCRWordBox] = []
                lines: List[str] = []
                confidences: List[float] = []

                n_boxes = len(data["text"])
                current_line = []
                current_line_num = -1

                for i in range(n_boxes):
                    text = data["text"][i].strip()
                    conf = float(data["conf"][i])
                    if text and conf > 20: # confidence filter
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        bbox = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                        word_boxes.append(OCRWordBox(text=text, confidence=conf / 100.0, bbox=bbox))
                        confidences.append(conf / 100.0)

                        line_num = data["line_num"][i]
                        if line_num != current_line_num:
                            if current_line:
                                lines.append(" ".join(current_line))
                                current_line = []
                            current_line_num = line_num
                        current_line.append(text)

                if current_line:
                    lines.append(" ".join(current_line))

                raw_text = "\n".join(lines)
                avg_conf = float(np.mean(confidences)) if confidences else 0.0

                if raw_text.strip():
                    return OCRResult(
                        raw_text=raw_text,
                        lines=lines,
                        word_boxes=word_boxes,
                        engine_name="PyTesseract",
                        average_confidence=avg_conf
                    )
            except Exception as e:
                logger.debug(f"PyTesseract invocation skipped or failed: {e}")

        # OpenCV Morphological text bounding box proposal + fallback lookup
        text_regions = self._detect_text_regions(image)

        # Check if there is an exact or registered mock text
        height, width = image.shape[:2]
        sig_key = f"{width}x{height}"
        raw_text = self.mocked_text_map.get(sig_key, "")

        if not raw_text and hasattr(image, "metadata_text"):
            raw_text = getattr(image, "metadata_text")

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()] if raw_text else []
        word_boxes = []

        # Create synthetic word boxes matching detected regions or default grid
        for idx, line in enumerate(lines):
            y_pos = int(50 + idx * 35)
            words = line.split()
            x_cursor = 40
            for w in words:
                w_len = len(w) * 12
                bbox = [(x_cursor, y_pos), (x_cursor + w_len, y_pos),
                        (x_cursor + w_len, y_pos + 25), (x_cursor, y_pos + 25)]
                word_boxes.append(OCRWordBox(text=w, confidence=0.98, bbox=bbox))
                x_cursor += w_len + 10

        return OCRResult(
            raw_text=raw_text,
            lines=lines,
            word_boxes=word_boxes,
            engine_name="FallbackOCR",
            average_confidence=0.95 if lines else 0.0,
            metadata={"detected_regions_count": len(text_regions)}
        )

    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects rectangular regions likely containing text using morphological gradient.
        Useful for highlighting text zones in UI.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Binarize
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Connect horizontally
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 20 and h > 8:
                boxes.append((x, y, w, h))
        return boxes
