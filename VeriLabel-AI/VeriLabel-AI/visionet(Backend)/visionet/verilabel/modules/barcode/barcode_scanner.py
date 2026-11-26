"""
VeriLabel AI - Module 4: Barcode Scanner & Multi-Engine Decoder
Team: Brahmastra | GITAM University
Member 3: Computer Vision and Barcode Module

Responsibilities:
- Detect and decode 1D & 2D barcodes on medical device packaging:
    • GS1 DataMatrix (ISO/IEC 16022) - Primary UDI carrier for medical devices
    • QR Code (ISO/IEC 18004) - High-capacity 2D carrier
    • GS1-128 / Code 128 (ISO/IEC 15417) - Primary 1D linear carrier
    • Code 39 (ISO/IEC 16388) - Standard linear healthcare barcode
    • EAN-13, EAN-8 (ISO/IEC 15420) - European packaging article numbers
    • UPC-A, UPC-E - Universal Product Codes
    • Interleaved 2 of 5 (ITF) - Outer shipping container codes
    • PDF417 - Two-dimensional stacked linear barcode
- Robust Multi-Engine Architecture:
    1. zxing-cpp: Native C++ high-accuracy decoder (DataMatrix, GS1-128, QR, Code-128, EAN, UPC)
    2. OpenCV BarcodeDetector: Native OpenCV 1D linear barcode decoder
    3. OpenCV QRCodeDetector: Dual-mode single and multi-target QR code engine
    4. pyzbar fallback: Native ZBar C-library wrapper (if installed)
- Adaptive Enhancement & Retry Heuristics:
    • CLAHE contrast equalization for uneven label lighting
    • Unsharp masking for blurred edges
    • Contrast stretching (Min-Max normalization)
    • Otsu binarization
    • Inverted photonegative polarity (for laser-etched metal or white-on-dark DataMatrix)
    • Orthogonal rotation sweeps (90°, 180°, 270°) for vertical packaging barcodes
    • Bicubic 2x upscaling for compact micro-barcodes
- Precise bounding box localization (x, y, w, h) and 4-point polygon contours
- Multi-barcode detection on a single label (e.g. 1D linear + 2D DataMatrix)
- Zero Hallucination Guarantee: Returns genuine decoded payload from actual decoding engines or an explicit empty list.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import cv2

logger = logging.getLogger("VeriLabel.Barcode.Scanner")


@dataclass
class DecodedBarcode:
    """Represents an authentic decoded barcode on a medical device packaging label."""
    data: str
    barcode_type: str  # e.g., 'DATAMATRIX', 'GS1-128', 'CODE-128', 'QRCODE', 'EAN-13', 'CODE-39'
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    polygon: Optional[List[Tuple[int, int]]] = None
    engine: str = "zxing-cpp"
    confidence: float = 1.0
    is_gs1: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "type": self.barcode_type,
            "engine": self.engine,
            "confidence": round(self.confidence, 3),
            "is_gs1": self.is_gs1,
            "error": self.error,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3]
            } if self.bbox else None,
            "polygon": self.polygon
        }


@dataclass
class ScanTelemetry:
    """Diagnostic telemetry recording the barcode scanning execution path."""
    total_found: int
    engines_used: List[str]
    retries_attempted: int
    successful_variant: Optional[str]
    latency_ms: float


class BarcodeScanner:
    """
    Production-grade multi-engine barcode detection and decoding system.
    Orchestrates zxing-cpp, OpenCV detectors, and pyzbar with adaptive enhancement retries.
    """

    SUPPORTED_FORMATS = [
        "DATAMATRIX",
        "QRCODE",
        "CODE-128",
        "CODE-39",
        "EAN-13",
        "EAN-8",
        "UPC-A",
        "UPC-E",
        "ITF",
        "PDF417",
    ]

    def __init__(self, enable_enhancement_retries: bool = True):
        self.enable_enhancement_retries = enable_enhancement_retries

        # 1. Initialize zxing-cpp (primary for DataMatrix & GS1)
        self._zxing = None
        try:
            import zxingcpp
            self._zxing = zxingcpp
            logger.info("zxing-cpp barcode engine loaded successfully.")
        except ImportError:
            logger.warning("zxing-cpp not installed. Falling back to OpenCV detectors.")

        # 2. Initialize OpenCV Barcode Detector
        self._cv_barcode = None
        if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
            try:
                self._cv_barcode = cv2.barcode.BarcodeDetector()
            except Exception as e:
                logger.debug(f"OpenCV BarcodeDetector initialization notice: {e}")

        # 3. Initialize OpenCV QR Detector
        try:
            self._cv_qr = cv2.QRCodeDetector()
        except Exception:
            self._cv_qr = None

        # 4. Initialize Pyzbar fallback
        self._pyzbar = None
        try:
            import pyzbar.pyzbar as pyzbar
            self._pyzbar = pyzbar
            logger.info("pyzbar fallback engine available.")
        except Exception:
            self._pyzbar = None

    def get_supported_formats(self) -> List[str]:
        """Returns the list of supported medical-device packaging barcode formats."""
        return list(self.SUPPORTED_FORMATS)

    def scan(self, image: np.ndarray) -> List[DecodedBarcode]:
        """
        Scans an image using all available barcode engines and returns unique decoded barcodes.
        If initial detection yields no results, triggers adaptive enhancement retries.
        """
        results, _ = self.scan_with_telemetry(image)
        return results

    def scan_with_telemetry(self, image: np.ndarray) -> Tuple[List[DecodedBarcode], ScanTelemetry]:
        """
        Scans image and returns decoded barcodes along with execution telemetry.
        """
        start_time = time.perf_counter()
        engines_used: List[str] = []
        if self._zxing:
            engines_used.append("zxing-cpp")
        if self._cv_barcode:
            engines_used.append("opencv_barcode")
        if self._cv_qr:
            engines_used.append("opencv_qr")
        if self._pyzbar:
            engines_used.append("pyzbar")

        if image is None or image.size == 0:
            logger.warning("BarcodeScanner received empty or None image array.")
            telemetry = ScanTelemetry(
                total_found=0,
                engines_used=engines_used,
                retries_attempted=0,
                successful_variant=None,
                latency_ms=0.0
            )
            return [], telemetry

        # Pass 1: Scan original image directly
        results = self._scan_single_image(image)
        if results or not self.enable_enhancement_retries:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            telemetry = ScanTelemetry(
                total_found=len(results),
                engines_used=engines_used,
                retries_attempted=0,
                successful_variant="original" if results else None,
                latency_ms=round(elapsed_ms, 2)
            )
            return results, telemetry

        # Pass 2: Adaptive Enhancement Retries (for difficult, low-contrast, or inverted barcodes)
        logger.debug("Initial barcode scan returned 0 results. Triggering adaptive enhancement retries...")
        enhanced_candidates = self._generate_enhanced_variants(image)
        retries_count = len(enhanced_candidates)

        for variant_name, enhanced_img in enhanced_candidates:
            res = self._scan_single_image(enhanced_img)
            if res:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.info(f"Barcode successfully recovered via '{variant_name}' enhancement retry.")
                telemetry = ScanTelemetry(
                    total_found=len(res),
                    engines_used=engines_used,
                    retries_attempted=retries_count,
                    successful_variant=variant_name,
                    latency_ms=round(elapsed_ms, 2)
                )
                return res, telemetry

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        telemetry = ScanTelemetry(
            total_found=0,
            engines_used=engines_used,
            retries_attempted=retries_count,
            successful_variant=None,
            latency_ms=round(elapsed_ms, 2)
        )
        return [], telemetry

    def _scan_single_image(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Runs all configured engines on a single image array and deduplicates results."""
        results: List[DecodedBarcode] = []
        seen_payloads: Set[str] = set()

        # Engine 1: zxing-cpp (Highest accuracy for DataMatrix, GS1-128, QR, Code-128)
        if self._zxing is not None:
            try:
                zx_results = self._scan_with_zxing(image)
                for code in zx_results:
                    if code.data and code.data not in seen_payloads:
                        seen_payloads.add(code.data)
                        results.append(code)
            except Exception as e:
                logger.debug(f"zxing-cpp scan error: {e}")

        # Engine 2: OpenCV QR Detector
        if self._cv_qr is not None:
            try:
                qr_results = self._scan_with_opencv_qr(image)
                for code in qr_results:
                    if code.data and code.data not in seen_payloads:
                        seen_payloads.add(code.data)
                        results.append(code)
            except Exception as e:
                logger.debug(f"OpenCV QR scan error: {e}")

        # Engine 3: OpenCV Barcode Detector (1D codes)
        if self._cv_barcode is not None:
            try:
                cv_codes = self._scan_with_opencv_barcode(image)
                for code in cv_codes:
                    if code.data and code.data not in seen_payloads:
                        seen_payloads.add(code.data)
                        results.append(code)
            except Exception as e:
                logger.debug(f"OpenCV Barcode scan error: {e}")

        # Engine 4: Pyzbar Fallback
        if self._pyzbar is not None:
            try:
                pyz_results = self._scan_with_pyzbar(image)
                for code in pyz_results:
                    if code.data and code.data not in seen_payloads:
                        seen_payloads.add(code.data)
                        results.append(code)
            except Exception as e:
                logger.debug(f"Pyzbar scan error: {e}")

        return results

    def _scan_with_zxing(self, image: np.ndarray) -> List[DecodedBarcode]:
        """
        Decodes barcodes using zxing-cpp with multi-orientation, inversion,
        and downscaling optimizations.
        """
        results: List[DecodedBarcode] = []
        try:
            barcodes = self._zxing.read_barcodes(
                image,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
                text_mode=self._zxing.TextMode.HRI
            )
        except Exception:
            # Fallback to default invocation if signature differs across minor versions
            barcodes = self._zxing.read_barcodes(image)

        for b in barcodes:
            raw_text = b.text.strip()
            if not raw_text:
                continue

            format_str = str(b.format).replace("BarcodeFormat.", "").replace(" ", "").upper()
            if "DATAMATRIX" in format_str:
                b_type = "DATAMATRIX"
            elif "QR" in format_str:
                b_type = "QRCODE"
            elif "CODE128" in format_str:
                b_type = "CODE-128"
            elif "EAN13" in format_str:
                b_type = "EAN-13"
            elif "EAN8" in format_str:
                b_type = "EAN-8"
            elif "UPCA" in format_str:
                b_type = "UPC-A"
            elif "UPCE" in format_str:
                b_type = "UPC-E"
            elif "CODE39" in format_str:
                b_type = "CODE-39"
            elif "ITF" in format_str:
                b_type = "ITF"
            elif "PDF417" in format_str:
                b_type = "PDF417"
            else:
                b_type = format_str or "BARCODE"

            # Check if GS1 formatted
            is_gs1 = bool(
                "(01)" in raw_text or
                (raw_text.startswith("01") and len(raw_text) >= 16 and raw_text[2:16].isdigit()) or
                (hasattr(b, "content_type") and "GS1" in str(b.content_type).upper())
            )

            # Extract bounding polygon & bounding box
            bbox = None
            poly = None
            try:
                pos = b.position
                pts = [
                    (int(pos.top_left.x), int(pos.top_left.y)),
                    (int(pos.top_right.x), int(pos.top_right.y)),
                    (int(pos.bottom_right.x), int(pos.bottom_right.y)),
                    (int(pos.bottom_left.x), int(pos.bottom_left.y)),
                ]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = (x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))
                poly = pts
            except Exception:
                pass

            results.append(
                DecodedBarcode(
                    data=raw_text,
                    barcode_type=b_type,
                    bbox=bbox,
                    polygon=poly,
                    engine="zxing-cpp",
                    confidence=1.0,
                    is_gs1=is_gs1
                )
            )

        return results

    def _scan_with_opencv_barcode(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Decodes 1D linear barcodes via cv2.barcode.BarcodeDetector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        retval, decoded_info, decoded_type, points = self._cv_barcode.detectAndDecode(gray)

        results: List[DecodedBarcode] = []
        if retval:
            infos = [decoded_info] if isinstance(decoded_info, str) else decoded_info
            types = [decoded_type] if isinstance(decoded_type, (str, int)) else decoded_type

            for i, text in enumerate(infos):
                if text and str(text).strip():
                    raw_type = str(types[i]).upper() if i < len(types) else "BARCODE"
                    b_type = "EAN-13" if "EAN_13" in raw_type or "EAN13" in raw_type else ("CODE-128" if "128" in raw_type else raw_type)
                    bbox = None
                    poly = None
                    if points is not None and len(points) > i:
                        pts = points[i].astype(int)
                        x, y, w, h = cv2.boundingRect(pts)
                        bbox = (x, y, w, h)
                        poly = [tuple(p) for p in pts]

                    results.append(
                        DecodedBarcode(
                            data=str(text).strip(),
                            barcode_type=b_type,
                            bbox=bbox,
                            polygon=poly,
                            engine="opencv_barcode"
                        )
                    )

        return results

    def _scan_with_opencv_qr(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Decodes QR codes via cv2.QRCodeDetector (single and multi-code)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        results: List[DecodedBarcode] = []

        # 1. Single code detector
        val, pts, _ = self._cv_qr.detectAndDecode(gray)
        if val and val.strip():
            bbox = None
            poly = None
            if pts is not None and len(pts) > 0:
                p_int = pts.reshape(-1, 2).astype(int)
                x, y, w, h = cv2.boundingRect(p_int)
                bbox = (x, y, w, h)
                poly = [tuple(p) for p in p_int]
            results.append(
                DecodedBarcode(
                    data=val.strip(),
                    barcode_type="QRCODE",
                    bbox=bbox,
                    polygon=poly,
                    engine="opencv_qr"
                )
            )
            return results

        # 2. Multi-code detector
        try:
            retval, decoded_info, points, _ = self._cv_qr.detectAndDecodeMulti(gray)
            if retval and decoded_info:
                for i, text in enumerate(decoded_info):
                    if text and text.strip():
                        bbox = None
                        poly = None
                        if points is not None and len(points) > i:
                            pts_i = points[i].reshape(-1, 2).astype(int)
                            x, y, w, h = cv2.boundingRect(pts_i)
                            bbox = (x, y, w, h)
                            poly = [tuple(p) for p in pts_i]
                        results.append(
                            DecodedBarcode(
                                data=text.strip(),
                                barcode_type="QRCODE",
                                bbox=bbox,
                                polygon=poly,
                                engine="opencv_qr_multi"
                            )
                        )
        except Exception:
            pass

        return results

    def _scan_with_pyzbar(self, image: np.ndarray) -> List[DecodedBarcode]:
        """Decodes barcodes via pyzbar fallback."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        decoded = self._pyzbar.decode(gray)
        results: List[DecodedBarcode] = []

        for obj in decoded:
            data = obj.data.decode("utf-8", errors="replace").strip()
            if not data:
                continue
            rect = obj.rect
            bbox = (rect.left, rect.top, rect.width, rect.height)
            poly = [tuple(p) for p in obj.polygon] if obj.polygon else None
            results.append(
                DecodedBarcode(
                    data=data,
                    barcode_type=str(obj.type).upper(),
                    bbox=bbox,
                    polygon=poly,
                    engine="pyzbar"
                )
            )

        return results

    def _generate_enhanced_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Generates contrast-enhanced, sharpened, inverted, and rotated variants
        for decoding difficult or low-contrast packaging barcodes.
        """
        variants: List[Tuple[str, np.ndarray]] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # 1. CLAHE Contrast Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(("clahe", clahe_img))

        # 2. Unsharp Masking (Sharpening)
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.5)
        sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
        variants.append(("sharpened", sharpened))

        # 3. Contrast Stretching (Min-Max normalization)
        min_v, max_v = int(np.min(gray)), int(np.max(gray))
        if max_v > min_v:
            stretched = ((gray.astype(np.float32) - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
            variants.append(("contrast_stretched", stretched))

        # 4. Otsu Binarization on CLAHE
        _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu_bin", otsu))

        # 5. Inverted Photonegative (Light on Dark) - Critical for metal or dark packaging
        inverted = 255 - gray
        variants.append(("inverted", inverted))
        variants.append(("inverted_clahe", 255 - clahe_img))

        # 6. Orthogonal Rotations (90, 180, 270 degrees)
        variants.append(("rot_90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)))
        variants.append(("rot_180", cv2.rotate(image, cv2.ROTATE_180)))
        variants.append(("rot_270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)))

        # 7. Bicubic 2x Upscaling for compact/low-res barcodes
        h, w = gray.shape[:2]
        if min(h, w) < 1200:
            upscaled = cv2.resize(clahe_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            variants.append(("upscaled_2x", upscaled))

        return variants

    def draw_barcodes(self, image: np.ndarray, barcodes: List[DecodedBarcode]) -> np.ndarray:
        """
        Draws precise bounding boxes or polygon contours around scanned barcodes
        with professional badges showing barcode format and decoded data.
        Returns annotated BGR numpy image.
        """
        annotated = image.copy()

        for b in barcodes:
            # Draw polygon if available, else bounding box
            if b.polygon and len(b.polygon) >= 4:
                pts = np.array(b.polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=(240, 130, 20), thickness=3, lineType=cv2.LINE_AA)
            elif b.bbox:
                x, y, w, h = b.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (240, 130, 20), 3)

            if b.bbox:
                x, y, w, h = b.bbox
                label = f"[{b.barcode_type}] {b.data[:20]}{'...' if len(b.data) > 20 else ''}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                badge_y1 = max(0, y - th - 8)
                badge_y2 = y
                badge_x2 = min(annotated.shape[1], x + tw + 8)
                cv2.rectangle(annotated, (x, badge_y1), (badge_x2, badge_y2), (200, 100, 10), -1)
                cv2.putText(
                    annotated,
                    label,
                    (x + 4, max(th + 2, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return annotated
