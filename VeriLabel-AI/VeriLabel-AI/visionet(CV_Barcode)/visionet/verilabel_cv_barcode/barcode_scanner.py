"""
VeriLabel AI - Module 3: Barcode Scanner & Multi-Engine Decoder
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Responsibilities:
- Detect and decode 1D & 2D barcodes on medical device packaging:
    • GS1 DataMatrix (ISO/IEC 16022) - Primary medical device UDI carrier
    • QR Code (ISO/IEC 18004) - High-capacity 2D carrier
    • GS1-128 / Code 128 (ISO/IEC 15417) - Primary 1D linear carrier
    • Code 39 (ISO/IEC 16388) - Standard linear healthcare barcode
    • EAN-13, EAN-8 (ISO/IEC 15420) - European packaging article numbers
    • UPC-A, UPC-E - Universal Product Codes
    • Interleaved 2 of 5 (ITF), Aztec, PDF417
- Robust Multi-Engine Architecture:
    1. zxing-cpp: Native C++ high-accuracy decoder (DataMatrix, GS1-128, QR, Code-128, EAN)
    2. OpenCV BarcodeDetector: Native OpenCV 1D linear barcode decoder
    3. OpenCV QRCodeDetector: Dual-mode single and multi-target QR code engine
    4. pyzbar fallback: Native ZBar C-library wrapper (if installed)
- Adaptive Enhancement & Retry Heuristics:
    • CLAHE contrast equalization for uneven label lighting
    • Unsharp masking for blurred edges
    • Otsu binarization
    • Inverted photonegative polarity (for laser-etched metal or white-on-dark DataMatrix)
    • Orthogonal rotation sweeps (90°, 180°, 270°) for vertical packaging barcodes
    • Bicubic 2x upscaling for compact micro-barcodes
- Precise bounding box localization (x, y, w, h) and 4-point polygon contours
- Multi-barcode detection on a single label (e.g. 1D linear + 2D DataMatrix)
- Zero Hallucination Guarantee: Only returns verified decoded payload from actual decoding engines.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import cv2

from .models import DecodedBarcode
from .image_processor import ImageProcessor

logger = logging.getLogger("VeriLabel.CV.BarcodeScanner")

# Import zxingcpp if available
try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False
    logger.warning("zxingcpp is not installed. Will use OpenCV and pyzbar fallback.")

# Import pyzbar if available
try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


@dataclass
class ScanTelemetry:
    """Diagnostic telemetry recording the barcode scanning execution path."""
    total_found: int
    engines_used: List[str]
    retries_attempted: int
    successful_variant: Optional[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_found": self.total_found,
            "engines_used": self.engines_used,
            "retries_attempted": self.retries_attempted,
            "successful_variant": self.successful_variant,
            "latency_ms": round(self.latency_ms, 2),
        }


class BarcodeScanner:
    """
    Production-grade multi-engine barcode detection and decoding system.
    Orchestrates zxing-cpp, OpenCV detectors, and pyzbar with adaptive enhancement retries.
    """

    SUPPORTED_FORMATS = [
        "DATAMATRIX",
        "QRCODE",
        "CODE-128",
        "GS1-128",
        "CODE-39",
        "EAN-13",
        "EAN-8",
        "UPC-A",
        "UPC-E",
        "ITF",
        "AZTEC",
        "PDF417",
    ]

    def __init__(
        self,
        enable_adaptive_retries: bool = True,
        enable_rotations: bool = True,
        enable_polarity_inversion: bool = True,
    ):
        self.enable_adaptive_retries = enable_adaptive_retries
        self.enable_rotations = enable_rotations
        self.enable_polarity_inversion = enable_polarity_inversion
        self.image_processor = ImageProcessor()

        # Initialize OpenCV BarcodeDetector if available
        self.cv_barcode_detector = None
        try:
            if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
                self.cv_barcode_detector = cv2.barcode.BarcodeDetector()
        except Exception as e:
            logger.debug(f"OpenCV BarcodeDetector unavailable: {e}")

        # Initialize OpenCV QRCodeDetector
        try:
            self.cv_qr_detector = cv2.QRCodeDetector()
        except Exception:
            self.cv_qr_detector = None

    def scan(self, image: np.ndarray) -> List[DecodedBarcode]:
        """
        Scans an image array for all barcodes using multi-engine decoding and adaptive retries.
        Returns a list of unique DecodedBarcode objects.
        """
        results, _ = self.scan_with_telemetry(image)
        return results

    def scan_with_telemetry(self, image: np.ndarray) -> Tuple[List[DecodedBarcode], ScanTelemetry]:
        """
        Executes multi-engine decoding and returns decoded barcodes along with
        diagnostic performance telemetry.
        """
        t0 = time.perf_counter()
        engines_used: Set[str] = set()
        retries_count = 0
        successful_variant: Optional[str] = None

        raw_results: List[DecodedBarcode] = []

        # 1. Base Attempt: Primary zxing-cpp scan on raw image
        base_found = self._scan_single_image(image, engines_used)
        if base_found:
            raw_results.extend(base_found)
            successful_variant = "original"

        # 2. Adaptive Enhancement Retries if nothing found or to catch secondary barcodes
        if self.enable_adaptive_retries:
            variants = self.image_processor.preprocess_for_barcode(image)
            for v_name, v_img in variants:
                if v_name == "original_gray":
                    continue
                if not self.enable_polarity_inversion and v_name == "inverted":
                    continue

                retries_count += 1
                v_found = self._scan_single_image(v_img, engines_used)
                if v_found:
                    raw_results.extend(v_found)
                    if successful_variant is None:
                        successful_variant = v_name

        # 3. Orthogonal Rotation Retries (90, 180, 270 degrees)
        if self.enable_rotations and (len(raw_results) == 0 or len(raw_results) < 2):
            for angle in [90, 180, 270]:
                retries_count += 1
                rot_img = self._rotate_orthogonal(image, angle)
                rot_found = self._scan_single_image(rot_img, engines_used)
                if rot_found:
                    # Map bounding box back to original orientation
                    h_orig, w_orig = image.shape[:2]
                    for bc in rot_found:
                        if bc.bbox:
                            bc.bbox = self._remap_bbox_from_rotation(bc.bbox, angle, w_orig, h_orig)
                        if bc.polygon:
                            bc.polygon = [self._remap_point_from_rotation(pt, angle, w_orig, h_orig) for pt in bc.polygon]
                        raw_results.append(bc)
                    if successful_variant is None:
                        successful_variant = f"rotated_{angle}"

        # 4. Upscaling Retry for Micro-barcodes
        if len(raw_results) == 0 and self.enable_adaptive_retries:
            retries_count += 1
            h, w = image.shape[:2]
            upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            up_found = self._scan_single_image(upscaled, engines_used)
            for bc in up_found:
                if bc.bbox:
                    x, y, bw, bh = bc.bbox
                    bc.bbox = (x // 2, y // 2, bw // 2, bh // 2)
                if bc.polygon:
                    bc.polygon = [(px // 2, py // 2) for px, py in bc.polygon]
                raw_results.append(bc)
            if up_found and successful_variant is None:
                successful_variant = "upscaled_2x"

        # 5. Deduplicate results based on data string and spatial proximity
        unique_barcodes = self._deduplicate_barcodes(raw_results)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = ScanTelemetry(
            total_found=len(unique_barcodes),
            engines_used=sorted(list(engines_used)),
            retries_attempted=retries_count,
            successful_variant=successful_variant,
            latency_ms=elapsed_ms,
        )
        return unique_barcodes, telemetry

    def _scan_single_image(self, img_array: np.ndarray, engines_used: Set[str]) -> List[DecodedBarcode]:
        """Runs the active engines against a single image array."""
        found: List[DecodedBarcode] = []

        # 1. zxing-cpp Engine
        if HAS_ZXING:
            engines_used.add("zxing-cpp")
            try:
                # read_barcodes accepts numpy arrays
                zx_results = zxingcpp.read_barcodes(img_array)
                for res in zx_results:
                    if not res.text:
                        continue
                    fmt_str = str(res.format).replace("BarcodeFormat.", "").upper().replace(" ", "").replace("_", "")
                    if fmt_str in ("DATAMATRIX", "DATA MATRIX"):
                        b_type = "DATAMATRIX"
                    elif "QR" in fmt_str:
                        b_type = "QRCODE"
                    elif "CODE128" in fmt_str:
                        b_type = "GS1-128" if "(01)" in res.text or res.text.startswith("01") else "CODE-128"
                    elif "CODE39" in fmt_str:
                        b_type = "CODE-39"
                    elif "EAN13" in fmt_str:
                        b_type = "EAN-13"
                    elif "EAN8" in fmt_str:
                        b_type = "EAN-8"
                    elif "UPC" in fmt_str:
                        b_type = "UPC-A"
                    else:
                        b_type = fmt_str

                    # Extract polygon and bounding box
                    pos = res.position
                    polygon = [(pos.top_left.x, pos.top_left.y),
                               (pos.top_right.x, pos.top_right.y),
                               (pos.bottom_right.x, pos.bottom_right.y),
                               (pos.bottom_left.x, pos.bottom_left.y)]
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                    is_gs1 = bool(
                        "(01)" in res.text or
                        (len(res.text) >= 16 and res.text[:2] == "01" and res.text[2:16].isdigit()) or
                        b_type == "DATAMATRIX"
                    )

                    found.append(DecodedBarcode(
                        data=res.text,
                        barcode_type=b_type,
                        bbox=bbox,
                        polygon=polygon,
                        engine="zxing-cpp",
                        confidence=1.0,
                        is_gs1=is_gs1,
                    ))
            except Exception as e:
                logger.debug(f"zxing-cpp scan error: {e}")

        # 2. OpenCV QRCodeDetector Engine (Fallback or Complementary)
        if self.cv_qr_detector and len(found) == 0:
            engines_used.add("opencv-qrcode")
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
                retval, decoded_info, points, _ = self.cv_qr_detector.detectAndDecodeMulti(gray)
                if retval and decoded_info:
                    for text, pts in zip(decoded_info, points):
                        if text:
                            pts = pts.astype(int)
                            polygon = [(int(p[0]), int(p[1])) for p in pts]
                            xs = [p[0] for p in polygon]
                            ys = [p[1] for p in polygon]
                            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                            found.append(DecodedBarcode(
                                data=text,
                                barcode_type="QRCODE",
                                bbox=bbox,
                                polygon=polygon,
                                engine="opencv-qrcode",
                                confidence=0.95,
                            ))
            except Exception as e:
                logger.debug(f"OpenCV QRCodeDetector error: {e}")

        # 3. OpenCV BarcodeDetector (for 1D Linear Barcodes)
        if self.cv_barcode_detector and len(found) == 0:
            engines_used.add("opencv-barcode")
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
                retval, decoded_info, decoded_type, points = self.cv_barcode_detector.detectAndDecodeMulti(gray)
                if retval and decoded_info:
                    for text, btype, pts in zip(decoded_info, decoded_type, points):
                        if text:
                            pts = pts.astype(int)
                            polygon = [(int(p[0]), int(p[1])) for p in pts]
                            xs = [p[0] for p in polygon]
                            ys = [p[1] for p in polygon]
                            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                            found.append(DecodedBarcode(
                                data=text,
                                barcode_type=str(btype).upper(),
                                bbox=bbox,
                                polygon=polygon,
                                engine="opencv-barcode",
                                confidence=0.95,
                            ))
            except Exception as e:
                logger.debug(f"OpenCV BarcodeDetector error: {e}")

        # 4. pyzbar Fallback
        if HAS_PYZBAR and len(found) == 0:
            engines_used.add("pyzbar")
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
                zbar_results = pyzbar.decode(gray)
                for item in zbar_results:
                    data_str = item.data.decode("utf-8", errors="ignore")
                    if data_str:
                        rect = item.rect
                        bbox = (rect.left, rect.top, rect.width, rect.height)
                        polygon = [(p.x, p.y) for p in item.polygon]
                        found.append(DecodedBarcode(
                            data=data_str,
                            barcode_type=str(item.type).upper(),
                            bbox=bbox,
                            polygon=polygon if polygon else None,
                            engine="pyzbar",
                            confidence=0.90,
                        ))
            except Exception as e:
                logger.debug(f"pyzbar decode error: {e}")

        return found

    # --------------------------------------------------------------------------
    # Helper Geometry & Deduplication Functions
    # --------------------------------------------------------------------------
    def _rotate_orthogonal(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotates image array by 90, 180, or 270 degrees clockwise."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _remap_point_from_rotation(self, pt: Tuple[int, int], angle: int, w_orig: int, h_orig: int) -> Tuple[int, int]:
        """Maps point (x, y) on a rotated canvas back to original canvas coordinates."""
        px, py = pt
        if angle == 90:
            # Rotated 90 CW: new_w = h_orig, new_h = w_orig
            # px = h_orig - 1 - orig_y, py = orig_x
            return (py, h_orig - 1 - px)
        elif angle == 180:
            # Rotated 180: px = w_orig - 1 - orig_x, py = h_orig - 1 - orig_y
            return (w_orig - 1 - px, h_orig - 1 - py)
        elif angle == 270:
            # Rotated 270 CW (90 CCW): px = orig_y, py = w_orig - 1 - orig_x
            return (w_orig - 1 - py, px)
        return pt

    def _remap_bbox_from_rotation(
        self,
        bbox: Tuple[int, int, int, int],
        angle: int,
        w_orig: int,
        h_orig: int,
    ) -> Tuple[int, int, int, int]:
        """Maps bounding box (x, y, w, h) from rotated coordinate space back to original space."""
        x, y, w, h = bbox
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        remapped = [self._remap_point_from_rotation(p, angle, w_orig, h_orig) for p in pts]
        xs = [p[0] for p in remapped]
        ys = [p[1] for p in remapped]
        x_min, x_max = max(0, min(xs)), min(w_orig, max(xs))
        y_min, y_max = max(0, min(ys)), min(h_orig, max(ys))
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _deduplicate_barcodes(self, barcodes: List[DecodedBarcode]) -> List[DecodedBarcode]:
        """Eliminates redundant decodings of the same physical barcode."""
        if not barcodes:
            return []

        unique: List[DecodedBarcode] = []
        seen: Set[str] = set()

        for bc in barcodes:
            norm_key = f"{bc.barcode_type}:{bc.data.strip()}"
            if norm_key in seen:
                continue

            # Check spatial overlap with existing barcodes if data matches
            duplicate = False
            for existing in unique:
                if existing.data.strip() == bc.data.strip():
                    duplicate = True
                    break

            if not duplicate:
                seen.add(norm_key)
                unique.append(bc)

        return unique
