"""
VeriLabel AI - Module 1: Document Processing & Ingestion
Team: Brahmastra | GITAM University
Responsibilities:
- Validate file types and size
- Extract file metadata & SHA-256 hash
- Preprocess image for OCR and Computer Vision (CLAHE, denoising, binarization)
- Perform document quality checks (blur / brightness detection)
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import cv2

from verilabel.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_DOCUMENT_EXTENSIONS,
    ALL_ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES
)
from verilabel.exceptions import DocumentProcessingError
from verilabel.core.utils import calculate_file_hash, load_image_safely, save_image_safely


class DocumentProcessor:
    """
    Handles ingestion, format validation, metadata extraction,
    and image preprocessing for medical device labels.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validates file existence, extension, and size.
        Returns a dictionary of file specifications.
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

    def load_document(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validates and loads the document into an OpenCV BGR image array.
        Handles PDF rendering if needed, or loads standard image formats.
        Returns: (image_array, metadata)
        """
        meta = self.validate_file(file_path)
        path = Path(file_path)

        if meta["is_pdf"]:
            img = self._convert_pdf_to_image(path)
        else:
            img = load_image_safely(path)

        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]

        # Quality metrics
        quality = self.assess_image_quality(img)

        meta.update({
            "height": height,
            "width": width,
            "channels": channels,
            "aspect_ratio": round(width / float(height), 3),
            "quality": quality
        })

        return img, meta

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assesses image quality indicators:
        - Blur detection via Laplacian variance
        - Brightness and contrast statistics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Blur metric: Laplacian variance (standard computer vision approach)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 100.0  # Common benchmark

        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        is_under_exposed = mean_brightness < 40.0
        is_over_exposed = mean_brightness > 220.0

        return {
            "blur_metric": round(float(laplacian_var), 2),
            "is_blurry": bool(is_blurry),
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(contrast_std, 2),
            "is_under_exposed": bool(is_under_exposed),
            "is_over_exposed": bool(is_over_exposed),
            "quality_acceptable": not (is_blurry or is_under_exposed)
        }

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Applies a suite of preprocessing techniques optimized for text & symbol extraction:
        1. Grayscale
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Denoised image (Bilateral filter)
        4. Otsu Threshold (Binarized for OCR)
        5. Adaptive Threshold
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # CLAHE for local contrast enhancement (crucial for medical labels on varied packages)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)

        # Bilateral filter preserves edges while smoothing background noise
        denoised = cv2.bilateralFilter(clahe_enhanced, d=7, sigmaColor=50, sigmaSpace=50)

        # Otsu thresholding for clean black-white text segmentation
        _, otsu_bin = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive thresholding for uneven lighting
        adaptive_bin = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )

        return {
            "original": image,
            "gray": gray,
            "clahe": clahe_enhanced,
            "denoised": denoised,
            "otsu_binary": otsu_bin,
            "adaptive_binary": adaptive_bin
        }

    def _convert_pdf_to_image(self, pdf_path: Path) -> np.ndarray:
        """
        Converts a PDF label to an image array.
        Supports pypdf/pymupdf if present, or provides clean fallback.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                raise DocumentProcessingError(f"PDF document is empty: {pdf_path}")
            page = doc[0]
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except ImportError:
            pass

        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=200)
            if not images:
                raise DocumentProcessingError(f"No pages extracted from PDF: {pdf_path}")
            pil_img = images[0]
            open_cv_image = np.array(pil_img)
            # Convert RGB to BGR
            return open_cv_image[:, :, ::-1].copy()
        except ImportError:
            pass

        raise DocumentProcessingError(
            "PDF processing requires PyMuPDF (fitz) or pdf2image. Please install one or upload PNG/JPG labels.",
            details={"pdf_path": str(pdf_path)}
        )
