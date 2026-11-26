"""
VeriLabel AI - Module 1: Document Processing, Computer Vision & Ingestion
Team: Brahmastra | GITAM University
Member 3: Computer Vision and Barcode Module

Responsibilities:
- Multi-source document ingestion (file paths, raw bytes, numpy arrays, PIL images)
- Cryptographic SHA-256 hashing for FDA 21 CFR Part 11 audit trails
- Corrupt and invalid image detection with graceful error handling
- Automated image quality assessment:
    • Laplacian variance blur metric
    • Exposure & luminance analysis (under/over-exposure)
    • Contrast standard deviation & dynamic range
    • Resolution & aspect ratio validation
    • Composite quality score (0-100%)
- Resolution-adaptive scaling for low-resolution and oversized packaging labels
- High-precision deskewing & rotation correction (Probabilistic Hough Lines + Minimum Area Rect)
- Multi-variant OpenCV image preprocessing:
    • CLAHE (Contrast Limited Adaptive Histogram Equalization)
    • Bilateral edge-preserving noise reduction
    • Unsharp masking for micro-text and barcode edges
    • Otsu and Gaussian adaptive binarization
    • Inverted polarity generation for dark packaging / laser-etched DataMatrix
- Barcode-specific vs OCR-specific preprocessing pipelines
- PDF rendering via pypdfium2, PyMuPDF, or pdf2image at 300 DPI
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image

from verilabel.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_DOCUMENT_EXTENSIONS,
    ALL_ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES
)
from verilabel.exceptions import DocumentProcessingError
from verilabel.core.utils import calculate_file_hash, load_image_safely, save_image_safely


class ImagePreprocessor:
    """
    Dedicated computer vision preprocessing toolkit for medical device packaging labels.
    Provides deskewing, quality scoring, edge-preserving denoising, contrast enhancement,
    and polar inversion for both OCR and barcode recognition.
    """

    @staticmethod
    def detect_skew_angle(image: np.ndarray) -> float:
        """
        Estimates label skew angle in degrees using edge detection and Hough transform.
        Falls back to minimum area bounding box on contours if no Hough lines found.
        Returns rotation angle between -45.0 and +45.0 degrees.
        """
        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Downscale for fast, robust line detection
        max_dim = 1000
        h, w = gray.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = gray

        # Technique 1: Probabilistic Hough Line Transform on Canny edges
        edges = cv2.Canny(small, 50, 150, apertureSize=3)
        min_line_len = max(20, small.shape[1] // 8)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_line_len, maxLineGap=20)

        angles: List[float] = []
        if lines is not None and len(lines) > 0:
            for line in lines:
                coords = line.reshape(-1)
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = coords[:4]
                if x2 - x1 == 0:
                    continue
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Focus on near-horizontal text and border lines (-45 to +45 deg)
                if -45.0 <= angle <= 45.0:
                    angles.append(angle)

        if angles:
            return float(np.median(angles))

        # Technique 2: Minimum Area Rectangle on thresholded contours (fallback)
        try:
            _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contour_angles: List[float] = []
            for cnt in contours:
                if cv2.contourArea(cnt) > (small.shape[0] * small.shape[1] * 0.005):
                    rect = cv2.minAreaRect(cnt)
                    box_angle = rect[-1]
                    if box_angle < -45:
                        box_angle = 90 + box_angle
                    elif box_angle > 45:
                        box_angle = box_angle - 90
                    if -45.0 <= box_angle <= 45.0 and abs(box_angle) > 0.3:
                        contour_angles.append(box_angle)
            if contour_angles:
                return float(np.median(contour_angles))
        except Exception:
            pass

        return 0.0

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Rotates image around center while expanding canvas boundaries to prevent corner clipping.
        Preserves original aspect ratio and fills new borders with clean neutral background.
        """
        if abs(angle) < 0.1:
            return image.copy()

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]

        border_val = bg_color if len(image.shape) == 3 else bg_color[0]

        return cv2.warpAffine(
            image,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )

    @staticmethod
    def sharpen(image: np.ndarray, amount: float = 1.2) -> np.ndarray:
        """
        Applies unsharp masking to enhance text readability and barcode edges
        without amplifying high-frequency noise.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), 2.5)
        sharpened = cv2.addWeighted(image, 1.0 + (amount * 0.5), blurred, -(amount * 0.5), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Enhances local contrast across varied packaging surfaces and glossy reflections.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(gray)

    @staticmethod
    def denoise_bilateral(image: np.ndarray, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
        """
        Bilateral filtering smooths surface noise/grain while preserving sharp text and symbol boundaries.
        """
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def adaptive_threshold(gray: np.ndarray, block_size: int = 15, c: int = 4) -> np.ndarray:
        """
        Gaussian adaptive binarization for uneven label lighting conditions.
        """
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )

    @staticmethod
    def otsu_threshold(gray: np.ndarray) -> np.ndarray:
        """
        Global Otsu thresholding for binarization of high-contrast text and barcodes.
        """
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """
        Inverts pixel values (255 - pixel).
        Essential for laser-etched, metal-printed, or dark-background DataMatrix barcodes.
        """
        return 255 - image


class DocumentProcessor:
    """
    Production-grade document processor handling file validation, format conversions,
    quality assessment diagnostics, and multi-technique image preprocessing for VeriLabel AI.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        auto_deskew: bool = True,
        min_dimension: int = 600,
        max_dimension: int = 3200
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.auto_deskew = auto_deskew
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validates file existence, supported extension, and file size limits.
        Computes SHA-256 hash for FDA 21 CFR Part 11 audit traceability.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentProcessingError(
                f"File does not exist: {file_path}",
                details={"file_path": str(file_path)}
            )

        extension = path.suffix.lower()
        if extension not in ALL_ALLOWED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file format '{extension}'. Supported formats: {', '.join(sorted(ALL_ALLOWED_EXTENSIONS))}",
                details={"extension": extension, "allowed": list(ALL_ALLOWED_EXTENSIONS)}
            )

        file_size = path.stat().st_size
        if file_size <= 0:
            raise DocumentProcessingError(
                "Uploaded file is empty (0 bytes).",
                details={"file_size": file_size}
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            raise DocumentProcessingError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit ({MAX_FILE_SIZE_BYTES / (1024*1024):.2f} MB).",
                details={"file_size": file_size, "max_size": MAX_FILE_SIZE_BYTES}
            )

        file_hash = calculate_file_hash(path)

        return {
            "valid": True,
            "file_name": path.name,
            "extension": extension,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024.0, 2),
            "sha256": file_hash,
            "is_pdf": extension in ALLOWED_DOCUMENT_EXTENSIONS
        }

    def load_document(
        self,
        input_source: Union[str, Path, bytes, bytearray, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validates and loads document into an OpenCV BGR numpy array from diverse sources:
        - File path (str or Path)
        - In-memory bytes / bytearray (with PDF magic bytes detection)
        - PIL Image
        - Numpy ndarray
        Executes automated deskewing (if enabled) and comprehensive quality diagnostics.
        Returns: (image_array, metadata_dict)
        """
        metadata: Dict[str, Any] = {
            "source_type": type(input_source).__name__,
            "is_pdf": False,
            "deskew_applied": False,
            "deskew_angle": 0.0,
        }

        # Case 1: File path
        if isinstance(input_source, (str, Path)):
            path = Path(input_source)
            file_meta = self.validate_file(path)
            metadata.update(file_meta)

            if file_meta["is_pdf"]:
                img = self._convert_pdf_to_image(path)
            else:
                img = load_image_safely(path)

        # Case 2: In-memory raw bytes
        elif isinstance(input_source, (bytes, bytearray)):
            import hashlib
            metadata["file_name"] = f"buffer_{hashlib.md5(input_source[:64]).hexdigest()[:8]}.png"
            metadata["file_size_bytes"] = len(input_source)
            metadata["file_size_kb"] = round(len(input_source) / 1024.0, 2)
            metadata["sha256"] = hashlib.sha256(input_source).hexdigest()

            # Check PDF magic bytes '%PDF'
            if input_source.startswith(b"%PDF"):
                metadata["is_pdf"] = True
                metadata["extension"] = ".pdf"
                img = self._convert_pdf_bytes_to_image(input_source)
            else:
                metadata["extension"] = ".png"
                nparr = np.frombuffer(input_source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise DocumentProcessingError(
                        "Failed to decode image from provided byte buffer. Data may be corrupted.",
                        details={"buffer_size": len(input_source)}
                    )

        # Case 3: PIL Image
        elif isinstance(input_source, Image.Image):
            metadata["file_name"] = "pil_image.png"
            metadata["extension"] = ".png"
            metadata["sha256"] = "in_memory_pil"
            rgb_arr = np.array(input_source.convert("RGB"))
            img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        # Case 4: Numpy ndarray
        elif isinstance(input_source, np.ndarray):
            metadata["file_name"] = "numpy_array.png"
            metadata["extension"] = ".png"
            metadata["sha256"] = "in_memory_ndarray"
            if len(input_source.shape) == 2:
                img = cv2.cvtColor(input_source, cv2.COLOR_GRAY2BGR)
            elif input_source.shape[2] == 4:
                img = cv2.cvtColor(input_source, cv2.COLOR_BGRA2BGR)
            else:
                img = input_source.copy()
        else:
            raise DocumentProcessingError(
                f"Unsupported input source type '{type(input_source).__name__}'.",
                details={"type": str(type(input_source))}
            )

        if img is None or img.size == 0:
            raise DocumentProcessingError("Loaded image is empty or failed to decode.")

        # Adaptive resolution normalization
        orig_h, orig_w = img.shape[:2]
        metadata["original_width"] = orig_w
        metadata["original_height"] = orig_h

        if min(orig_h, orig_w) < self.min_dimension:
            scale = self.min_dimension / float(min(orig_h, orig_w))
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            metadata["upscaled"] = True
            metadata["scale_factor"] = round(scale, 3)
        elif max(orig_h, orig_w) > self.max_dimension:
            scale = self.max_dimension / float(max(orig_h, orig_w))
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            metadata["downscaled"] = True
            metadata["scale_factor"] = round(scale, 3)

        # Deskewing
        deskew_angle = 0.0
        if self.auto_deskew:
            deskew_angle = ImagePreprocessor.detect_skew_angle(img)
            if abs(deskew_angle) > 0.4 and abs(deskew_angle) <= 45.0:
                img = ImagePreprocessor.rotate_image(img, deskew_angle)
                metadata["deskew_applied"] = True
            else:
                metadata["deskew_applied"] = False
        metadata["deskew_angle"] = round(deskew_angle, 2)

        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # Quality metrics
        quality = self.assess_image_quality(img)

        metadata.update({
            "height": height,
            "width": width,
            "channels": channels,
            "aspect_ratio": round(width / float(height), 3),
            "quality": quality
        })

        return img, metadata

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assesses comprehensive image quality indicators:
        - Blur detection via Laplacian variance (benchmark > 100.0)
        - Mean pixel brightness (exposure analysis: < 40 underexposed, > 220 overexposed)
        - Contrast standard deviation (benchmark > 35.0)
        - Dynamic range (max - min pixel intensity)
        - Composite quality score (0.0 - 100.0)
        """
        if image is None or image.size == 0:
            return {
                "blur_metric": 0.0,
                "is_blurry": True,
                "mean_brightness": 0.0,
                "contrast_std": 0.0,
                "dynamic_range": 0,
                "is_under_exposed": False,
                "is_over_exposed": False,
                "quality_score": 0.0,
                "quality_acceptable": False
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Blur metric: Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < 100.0

        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        dynamic_range = int(int(np.max(gray)) - int(np.min(gray)))

        is_under_exposed = mean_brightness < 40.0
        is_over_exposed = mean_brightness > 220.0

        # Calculate composite quality score (0-100)
        # Blur subscore (0-40 pts): 40 pts if laplacian_var >= 300, scaled below
        blur_subscore = min(40.0, (laplacian_var / 300.0) * 40.0)
        # Contrast subscore (0-35 pts): 35 pts if contrast_std >= 60, scaled below
        contrast_subscore = min(35.0, (contrast_std / 60.0) * 35.0)
        # Exposure subscore (0-25 pts): optimal brightness is ~128
        exp_diff = abs(mean_brightness - 128.0)
        exp_subscore = max(0.0, 25.0 - (exp_diff / 128.0) * 25.0)

        quality_score = round(blur_subscore + contrast_subscore + exp_subscore, 1)
        quality_acceptable = bool(not is_blurry and not is_under_exposed and contrast_std >= 20.0)

        return {
            "blur_metric": round(laplacian_var, 2),
            "is_blurry": bool(is_blurry),
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(contrast_std, 2),
            "dynamic_range": dynamic_range,
            "is_under_exposed": bool(is_under_exposed),
            "is_over_exposed": bool(is_over_exposed),
            "quality_score": quality_score,
            "quality_acceptable": quality_acceptable
        }

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies a comprehensive suite of preprocessing techniques optimized for text & symbol extraction:
        1. Original image
        2. Grayscale
        3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        4. Bilateral edge-preserving denoised image
        5. Sharpened unsharp-masked image
        6. Otsu Threshold (binarized)
        7. Adaptive Gaussian Threshold
        8. Inverted (negative)
        """
        if image is None or image.size == 0:
            raise DocumentProcessingError("Cannot preprocess empty image array.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # CLAHE for local contrast equalization across varied packaging surfaces
        clahe_enhanced = ImagePreprocessor.apply_clahe(gray, clip_limit=2.5, tile_grid=(8, 8))

        # Bilateral filter preserves sharp edges while smoothing background texture
        denoised = ImagePreprocessor.denoise_bilateral(clahe_enhanced, d=7, sigma_color=50.0, sigma_space=50.0)

        # Unsharp sharpening for small typography and symbol edges
        sharpened = ImagePreprocessor.sharpen(clahe_enhanced, amount=1.2)

        # Otsu thresholding for clean black-white text segmentation
        otsu_bin = ImagePreprocessor.otsu_threshold(denoised)

        # Adaptive Gaussian thresholding for uneven packaging lighting
        adaptive_bin = ImagePreprocessor.adaptive_threshold(denoised, block_size=15, c=4)

        # Inverted image (light-on-dark)
        inverted = ImagePreprocessor.invert(gray)

        return {
            "original": image,
            "gray": gray,
            "clahe": clahe_enhanced,
            "denoised": denoised,
            "sharpened": sharpened,
            "otsu_binary": otsu_bin,
            "adaptive_binary": adaptive_bin,
            "inverted": inverted
        }

    def preprocess_for_barcode(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Generates specialized preprocessing variants tailored specifically for 1D and 2D barcode decoding.
        Barcodes require sharp module borders and must NOT be degraded by aggressive smoothing.
        Returns a sequence of (variant_name, image_array) tuples for multi-pass decoding retries.
        """
        if image is None or image.size == 0:
            return []

        variants: List[Tuple[str, np.ndarray]] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # 1. CLAHE Contrast Equalization
        clahe = ImagePreprocessor.apply_clahe(gray, clip_limit=3.0, tile_grid=(8, 8))
        variants.append(("clahe", clahe))

        # 2. Sharpening (Unsharp masking)
        sharpened = ImagePreprocessor.sharpen(gray, amount=1.6)
        variants.append(("sharpened", sharpened))

        # 3. Contrast Stretching (Min-Max normalization)
        min_val, max_val = np.min(gray), np.max(gray)
        if max_val > min_val:
            stretched = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            variants.append(("contrast_stretched", stretched))

        # 4. Otsu Binarization on CLAHE
        otsu = ImagePreprocessor.otsu_threshold(clahe)
        variants.append(("otsu_bin", otsu))

        # 5. Inversion (Photonegative) - Critical for laser-etched metal or white-on-dark DataMatrix
        inverted = ImagePreprocessor.invert(gray)
        variants.append(("inverted", inverted))
        variants.append(("inverted_clahe", ImagePreprocessor.invert(clahe)))

        # 6. Orthogonal Rotations (90, 180, 270 degrees)
        variants.append(("rot_90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)))
        variants.append(("rot_180", cv2.rotate(image, cv2.ROTATE_180)))
        variants.append(("rot_270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)))

        # 7. Rescaled 2x (for small or low-resolution DataMatrix codes)
        h, w = gray.shape[:2]
        if min(h, w) < 1200:
            upscaled = cv2.resize(clahe, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            variants.append(("upscaled_2x", upscaled))

        return variants

    def _convert_pdf_to_image(self, pdf_path: Path) -> np.ndarray:
        """
        Converts a PDF label page to high-resolution BGR image array.
        Supports pypdfium2 (primary, fast), PyMuPDF (fitz), or pdf2image.
        """
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(pdf_path))
            if len(pdf) == 0:
                raise DocumentProcessingError(f"PDF document is empty: {pdf_path}")
            page = pdf[0]
            pil_image = page.render(scale=300 / 72.0).to_pil()
            rgb_arr = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            pass

        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                raise DocumentProcessingError(f"PDF document is empty: {pdf_path}")
            page = doc[0]
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except ImportError:
            pass

        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=300)
            if not images:
                raise DocumentProcessingError(f"No pages extracted from PDF: {pdf_path}")
            pil_img = images[0]
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise DocumentProcessingError(
            "PDF processing requires pypdfium2, PyMuPDF (fitz), or pdf2image. Please install one or upload PNG/JPG labels.",
            details={"pdf_path": str(pdf_path)}
        )

    def _convert_pdf_bytes_to_image(self, pdf_bytes: Union[bytes, bytearray]) -> np.ndarray:
        """
        Renders in-memory PDF bytes to a high-resolution BGR image array.
        """
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(bytes(pdf_bytes))
            if len(pdf) == 0:
                raise DocumentProcessingError("In-memory PDF document is empty.")
            page = pdf[0]
            pil_image = page.render(scale=300 / 72.0).to_pil()
            rgb_arr = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to render PDF bytes: {e}")
