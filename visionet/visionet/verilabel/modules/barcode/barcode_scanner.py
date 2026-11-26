"""
VeriLabel AI - Module 4: Barcode Scanner
Team: Brahmastra | GITAM University
Responsibilities:
- Detect and decode 1D & 2D barcodes (GS1-128, DataMatrix, QR, Code-128, EAN-13)
- Multi-engine architecture: OpenCV BarcodeDetector + QRCodeDetector + PyZbar fallback
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from verilabel.exceptions import BarcodeValidationError

logger = logging.getLogger("VeriLabel.Barcode.Scanner")


@dataclass
class DecodedBarcode:
    """Represents a decoded barcode on a medical device label."""
    data: str
    barcode_type: str  # e.g., 'EAN-13', 'CODE-128', 'DATAMATRIX', 'QRCODE'
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    polygon: Optional[List[Tuple[int, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "type": self.barcode_type,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3]
            } if self.bbox else None
        }


class BarcodeScanner:
    """
    Robust barcode scanner combining OpenCV Barcode/QR detectors with pyzbar.
    """

    def __init__(self):
        # Initialize OpenCV Barcode Detector
        self._cv_barcode = None
        if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
            try:
                self._cv_barcode = cv2.barcode.BarcodeDetector()
            except Exception as e:
                logger.debug(f"OpenCV BarcodeDetector init skipped: {e}")

        # Initialize OpenCV QR Detector
        self._cv_qr = cv2.QRCodeDetector()

        # Check Pyzbar
        self._has_pyzbar = False
        try:
            import pyzbar.pyzbar as pyzbar
            self._pyzbar = pyzbar
            self._has_pyzbar = True
        except ImportError:
            self._pyzbar = None

    def scan(self, image: np.ndarray) -> List[DecodedBarcode]:
        """
        Scans an image using all available barcode engines and returns unique detected codes.
        """
        results: List[DecodedBarcode] = []
        seen_payloads = set()

        # 1. Pyzbar (best for 1D and DataMatrix if installed)
        if self._has_pyzbar:
            try:
                pyzbar_results = self._scan_with_pyzbar(image)
                for code in pyzbar_results:
                    if code.data not in seen_payloads:
                        seen_payloads.add(code.data)
                        results.append(code)
            except Exception as e:
                logger.debug(f"Pyzbar scan warning: {e}")

        # 2. OpenCV Barcode Detector (EAN, UPC, Code 128, etc.)
        if self._cv_barcode is not None:
            try:
                cv_codes = self._scan_with_opencv_barcode(image)
                for code in cv_codes:
                    if code.data not in seen_payloads:
                        seen_payloads.add(code.data)
                        results.append(code)
            except Exception as e:
                logger.debug(f"OpenCV Barcode scan warning: {e}")

        # 3. OpenCV QR Detector
        try:
            qr_codes = self._scan_with_opencv_qr(image)
            for code in qr_codes:
                if code.data not in seen_payloads:
                    seen_payloads.add(code.data)
                    results.append(code)
        except Exception as e:
            logger.debug(f"OpenCV QR scan warning: {e}")

        return results

    def _scan_with_pyzbar(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Decodes barcodes via pyzbar."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        decoded = self._pyzbar.decode(gray)
        results = []

        for obj in decoded:
            data = obj.data.decode("utf-8", errors="replace").strip()
            if not data:
                continue
            b_type = obj.type
            rect = obj.rect
            bbox = (rect.left, rect.top, rect.width, rect.height)
            results.append(DecodedBarcode(data=data, barcode_type=b_type, bbox=bbox))

        return results

    def _scan_with_opencv_barcode(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Decodes barcodes via cv2.barcode.BarcodeDetector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        retval, decoded_info, decoded_type, points = self._cv_barcode.detectAndDecode(gray)

        results = []
        if retval:
            # OpenCV returns either string or tuple of strings
            infos = [decoded_info] if isinstance(decoded_info, str) else decoded_info
            types = [decoded_type] if isinstance(decoded_type, (str, int)) else decoded_type

            for i, text in enumerate(infos):
                if text and str(text).strip():
                    b_type = str(types[i]) if i < len(types) else "BARCODE"
                    bbox = None
                    if points is not None and len(points) > i:
                        pts = points[i].astype(int)
                        x, y, w, h = cv2.boundingRect(pts)
                        bbox = (x, y, w, h)
                    results.append(DecodedBarcode(data=str(text).strip(), barcode_type=b_type, bbox=bbox))

        return results

    def _scan_with_opencv_qr(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Decodes QR codes via cv2.QRCodeDetector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        results = []

        # 1. Try single detector (most reliable in OpenCV)
        val, pts, _ = self._cv_qr.detectAndDecode(gray)
        if val and val.strip():
            bbox = None
            if pts is not None and len(pts) > 0:
                p_int = pts.astype(int)
                x, y, w, h = cv2.boundingRect(p_int)
                bbox = (x, y, w, h)
            results.append(DecodedBarcode(data=val.strip(), barcode_type="QRCODE", bbox=bbox))
            return results

        # 2. Try multi-code detector
        retval, decoded_info, points, _ = self._cv_qr.detectAndDecodeMulti(gray)
        if retval and decoded_info:
            for i, text in enumerate(decoded_info):
                if text and text.strip():
                    bbox = None
                    if points is not None and len(points) > i:
                        pts_i = points[i].astype(int)
                        x, y, w, h = cv2.boundingRect(pts_i)
                        bbox = (x, y, w, h)
                    results.append(DecodedBarcode(data=text.strip(), barcode_type="QRCODE", bbox=bbox))
                elif points is not None and len(points) > i:
                    # Points found but text empty: crop region with margin and retry single decode
                    pts_i = points[i].astype(int)
                    x, y, w, h = cv2.boundingRect(pts_i)
                    margin = 20
                    y1 = max(0, y - margin)
                    y2 = min(gray.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(gray.shape[1], x + w + margin)
                    crop = gray[y1:y2, x1:x2]
                    c_val, _, _ = self._cv_qr.detectAndDecode(crop)
                    if c_val and c_val.strip():
                        results.append(DecodedBarcode(data=c_val.strip(), barcode_type="QRCODE", bbox=(x, y, w, h)))

        return results

    def draw_barcodes(self, image: np.ndarray, barcodes: List[DecodedBarcode]) -> np.ndarray:
        """Draws bounding boxes around scanned barcodes."""
        annotated = image.copy()
        for b in barcodes:
            if b.bbox:
                x, y, w, h = b.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 100, 0), 2)
                label = f"{b.barcode_type}: {b.data[:16]}"
                cv2.putText(annotated, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        return annotated
