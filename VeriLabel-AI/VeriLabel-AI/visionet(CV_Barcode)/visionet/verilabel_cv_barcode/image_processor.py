"""
VeriLabel AI - Module 3: OpenCV Image Processor & Preprocessor
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Responsibilities:
- Multi-format ingestion (Path, bytes, PIL, numpy, and 300 DPI PDF)
- Cryptographic SHA-256 integrity verification
- Packaging quality assessment:
    • Laplacian blur variance: σ²_Δ = Var(∇² I)
    • Mean luminance μ & contrast dynamic range
    • Over/under-exposure warnings
    • Unified 0-100 quality scoring
- Automated deskewing with boundary-expansion rotation
- Multi-technique image enhancement (CLAHE, bilateral denoising, unsharp sharpening)
- Specialized multi-variant preprocessing for low-contrast and inverted barcodes
"""

import io
import math
import hashlib
import logging
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image

from .config import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MIN_BRIGHTNESS,
    DEFAULT_MAX_BRIGHTNESS,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MAX_DESKEW_ANGLE,
    DEFAULT_TARGET_DPI,
)
from .models import ImageQualityAssessment

logger = logging.getLogger("VeriLabel.CV.ImageProcessor")


class ImageProcessingError(Exception):
    """Exception raised for errors during OpenCV image processing."""
    pass


class ImageProcessor:
    """
    OpenCV processing engine providing validation, diagnostics, deskewing, and enhancement.
    """

    def __init__(
        self,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
        max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    # --------------------------------------------------------------------------
    # 1. Image Ingestion & Cryptographic Hashing
    # --------------------------------------------------------------------------
    def load_image(
        self,
        source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads an image from various input sources into an authentic BGR NumPy array.
        Returns:
            Tuple of (bgr_image_array, metadata_dict)
        """
        file_hash = "unknown"
        ext = ".png"
        raw_bytes = None

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise ImageProcessingError(f"Image file does not exist: {path}")

            ext = path.suffix.lower()
            with open(path, "rb") as f:
                raw_bytes = f.read()

        elif isinstance(source, bytes):
            raw_bytes = source
        elif isinstance(source, io.BytesIO):
            raw_bytes = source.getvalue()
        elif isinstance(source, Image.Image):
            buf = io.BytesIO()
            source.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
            ext = ".png"
        elif isinstance(source, np.ndarray):
            if source.size == 0:
                raise ImageProcessingError("Empty numpy array supplied as image.")
            # Ensure 3-channel BGR
            if len(source.shape) == 2:
                bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            elif source.shape[2] == 4:
                bgr = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            else:
                bgr = source.copy()
            file_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
            h, w = bgr.shape[:2]
            return bgr, {
                "sha256": file_hash,
                "width": w,
                "height": h,
                "extension": ext,
                "channels": 3,
            }
        else:
            raise ImageProcessingError(f"Unsupported image input type: {type(source)}")

        # Compute SHA-256 for byte stream
        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Handle PDF rendering
        if ext == ".pdf":
            bgr = self._render_pdf_first_page(raw_bytes)
        else:
            # Decode image bytes using OpenCV
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ImageProcessingError("Failed to decode image bytes: corrupted or unsupported format.")

            if len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                bgr = img

        h, w = bgr.shape[:2]
        metadata = {
            "sha256": file_hash,
            "width": w,
            "height": h,
            "extension": ext,
            "channels": 3,
        }
        return bgr, metadata

    def _render_pdf_first_page(self, pdf_bytes: bytes) -> np.ndarray:
        """Renders first page of PDF at high resolution (300 DPI)."""
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_bytes)
            if len(pdf) == 0:
                raise ImageProcessingError("Supplied PDF document contains zero pages.")
            page = pdf[0]
            # 300 DPI corresponds to scale = 300 / 72 ~= 4.166
            pil_image = page.render(scale=300 / 72).to_pil()
            rgb = np.array(pil_image)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            logger.warning("pypdfium2 not installed; attempting PIL / fallback PDF render.")
            raise ImageProcessingError("pypdfium2 required for PDF rendering.")

    # --------------------------------------------------------------------------
    # 2. Packaging Quality Assessment (Blur, Exposure, Contrast)
    # --------------------------------------------------------------------------
    def assess_quality(
        self,
        image: np.ndarray,
        blur_threshold: Optional[float] = None,
    ) -> ImageQualityAssessment:
        """
        Computes objective physical quality diagnostics:
        - Laplacian variance for sharpness
        - Mean pixel intensity and standard deviation for illumination
        - Dynamic range for packaging contrast
        - Estimated skew angle
        - Composite 0-100 quality score
        """
        if blur_threshold is None:
            blur_threshold = self.blur_threshold

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Blur Detection via Laplacian Variance: σ² = Var(∇² I)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = float(laplacian.var())
        is_blurred = bool(blur_variance < blur_threshold)

        # 2. Illumination & Contrast Metrics
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        min_val, max_val = float(np.min(gray)), float(np.max(gray))
        dynamic_range = max_val - min_val

        # Overexposure / Underexposure check
        high_pixel_ratio = float(np.mean(gray > 250))
        low_pixel_ratio = float(np.mean(gray < 10))
        is_overexposed = bool(mean_lum > self.max_brightness or high_pixel_ratio > 0.20)
        is_underexposed = bool(mean_lum < self.min_brightness or low_pixel_ratio > 0.25)

        # 3. Estimated Skew Angle
        skew_angle = self.estimate_skew_angle(gray)

        # 4. Composite Quality Score Calculation (0.0 to 100.0)
        score = 100.0

        # Blur penalty
        if blur_variance < blur_threshold:
            blur_deficit = (blur_threshold - blur_variance) / blur_threshold
            score -= min(40.0, blur_deficit * 40.0)

        # Exposure penalty
        if is_overexposed or is_underexposed:
            score -= 25.0
        elif mean_lum < 70 or mean_lum > 200:
            score -= 10.0

        # Contrast penalty
        if dynamic_range < 100.0:
            score -= 15.0

        # Skew penalty
        if abs(skew_angle) > 5.0:
            score -= min(15.0, abs(skew_angle) * 1.5)

        score = max(0.0, min(100.0, score))

        # Recommendations generator
        recs: List[str] = []
        if is_blurred:
            recs.append(f"Image is blurred (Laplacian variance {blur_variance:.1f} < {blur_threshold}). Retake with steady autofocus.")
        if is_overexposed:
            recs.append("Severe glare or overexposure detected. Adjust packaging lighting to avoid specular reflection.")
        if is_underexposed:
            recs.append("Underexposed image. Increase packaging lighting before scanning.")
        if abs(skew_angle) > 3.0:
            recs.append(f"Packaging label is skewed by {skew_angle:.1f}°. Automated deskewing recommended.")
        if not recs:
            recs.append("Packaging image quality is optimal for UDI decoding and ISO symbol recognition.")

        return ImageQualityAssessment(
            blur_variance=blur_variance,
            is_blurred=is_blurred,
            brightness_mean=mean_lum,
            brightness_std=std_lum,
            contrast_dynamic_range=dynamic_range,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            estimated_skew_angle=skew_angle,
            quality_score=score,
            recommendations=recs,
        )

    # --------------------------------------------------------------------------
    # 3. Deskewing & Boundary-Expanding Rotation
    # --------------------------------------------------------------------------
    def estimate_skew_angle(self, gray: np.ndarray) -> float:
        """
        Estimates the dominant skew angle using Hough line transform.
        Returns angle in degrees (-45.0 to +45.0).
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return 0.0

        angles: List[float] = []
        for line in lines:
            coords = line.ravel()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            # Only consider near-horizontal lines (-45 to 45 degrees)
            if -DEFAULT_MAX_DESKEW_ANGLE <= angle <= DEFAULT_MAX_DESKEW_ANGLE:
                angles.append(angle)

        if not angles:
            return 0.0

        median_angle = float(np.median(angles))
        return median_angle

    def deskew(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        max_angle: float = DEFAULT_MAX_DESKEW_ANGLE,
    ) -> Tuple[np.ndarray, float]:
        """
        Rotates image by the estimated or given skew angle while expanding canvas bounds
        so no label content or corner markers are cropped.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if angle is None:
            angle = self.estimate_skew_angle(gray)

        if abs(angle) < 0.3 or abs(angle) > max_angle:
            return image, 0.0

        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # Build affine rotation matrix
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute new expanded canvas dimensions
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust transformation matrix with translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rotated = cv2.warpAffine(
            image,
            M,
            (nW, nH),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255,
        )
        return rotated, angle

    # --------------------------------------------------------------------------
    # 4. Multi-Stage Enhancement & Preprocessing Pipeline
    # --------------------------------------------------------------------------
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies professional multi-stage image enhancement:
        1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance
        2. Bilateral filtering for edge-preserving denoising
        3. Unsharp masking for character and symbol edge sharpness
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE on L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Bilateral filter
            filtered_l = cv2.bilateralFilter(cl, d=7, sigmaColor=50, sigmaSpace=50)

            merged_lab = cv2.merge((filtered_l, a, b))
            enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(image)
            enhanced = cv2.bilateralFilter(cl, d=7, sigmaColor=50, sigmaSpace=50)

        # Unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)
        return sharpened

    def preprocess_for_barcode(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Generates a sequence of specialized preprocessing variants tailored for
        challenging packaging barcodes (e.g., low contrast, glossy reflection, photonegative).
        Returns list of (variant_name, image_array).
        """
        variants: List[Tuple[str, np.ndarray]] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # 1. Raw grayscale
        variants.append(("original_gray", gray))

        # 2. CLAHE equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(("clahe", clahe_img))

        # 3. Otsu binarization
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu_binary", otsu))

        # 4. Photonegative / Inversion (for laser-etched metal or dark mode)
        inverted = cv2.bitwise_not(gray)
        variants.append(("inverted", inverted))

        # 5. Morphological closing (repairs small voids in dot-peen or printer defects)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        variants.append(("morph_close", morph))

        # 6. Unsharp sharpening
        gaussian = cv2.GaussianBlur(gray, (0, 0), 1.5)
        unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        variants.append(("unsharp", unsharp))

        return variants
