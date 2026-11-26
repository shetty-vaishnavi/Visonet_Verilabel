"""
Document & Image Preprocessing Pipeline for Medical Device Labels.
Supports PDF rendering, deskewing, noise reduction, and contrast enhancement.
"""

from typing import Tuple, Dict, Any, Union, Optional
import os
import cv2
import numpy as np
from PIL import Image

from .config import PreprocessConfig


class DocumentPreprocessor:
    """
    Robust image preprocessor designed for medical device labels.
    Handles skew, lighting variations, glossy reflection, and PDF conversion.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def load_document(self, input_source: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads document from a file path, bytes, PIL image, or numpy array.
        Renders PDF pages if input is a PDF file.
        Returns (BGR numpy image, metadata dict).
        """
        metadata: Dict[str, Any] = {
            "source_type": type(input_source).__name__,
            "is_pdf": False,
            "original_pages": 1,
        }

        # Case 1: File path
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Input file not found: {input_source}")
            
            ext = os.path.splitext(input_source)[1].lower()
            metadata["file_extension"] = ext
            metadata["file_name"] = os.path.basename(input_source)

            if ext == ".pdf":
                metadata["is_pdf"] = True
                image = self._render_pdf_page(input_source, page_index=0)
            else:
                # Use cv2.imdecode to avoid path encoding issues on Windows
                with open(input_source, "rb") as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to decode image from path: {input_source}")

        # Case 2: Raw bytes
        elif isinstance(input_source, (bytes, bytearray)):
            # Check for PDF magic bytes (%PDF)
            if input_source.startswith(b"%PDF"):
                metadata["is_pdf"] = True
                image = self._render_pdf_bytes(input_source, page_index=0)
            else:
                nparr = np.frombuffer(input_source, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image from provided byte buffer.")

        # Case 3: PIL Image
        elif isinstance(input_source, Image.Image):
            rgb_img = np.array(input_source.convert("RGB"))
            image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # Case 4: Numpy ndarray
        elif isinstance(input_source, np.ndarray):
            if len(input_source.shape) == 2:
                image = cv2.cvtColor(input_source, cv2.COLOR_GRAY2BGR)
            elif input_source.shape[2] == 4:
                image = cv2.cvtColor(input_source, cv2.COLOR_BGRA2BGR)
            else:
                image = input_source.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_source)}")

        orig_h, orig_w = image.shape[:2]
        metadata["original_width"] = orig_w
        metadata["original_height"] = orig_h

        return image, metadata

    def _render_pdf_page(self, pdf_path: str, page_index: int = 0) -> np.ndarray:
        """Render a PDF page to a high-resolution BGR numpy image."""
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_path)
            if page_index >= len(pdf):
                page_index = 0
            page = pdf[page_index]
            # 300 DPI scale factor: 300 / 72 ≈ 4.166
            scale = self.config.pdf_render_dpi / 72.0
            pil_image = page.render(scale=scale).to_pil()
            rgb_arr = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Error rendering PDF via pypdfium2: {e}")

    def _render_pdf_bytes(self, pdf_bytes: bytes, page_index: int = 0) -> np.ndarray:
        """Render in-memory PDF bytes to high-resolution BGR image."""
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_bytes)
            if page_index >= len(pdf):
                page_index = 0
            page = pdf[page_index]
            scale = self.config.pdf_render_dpi / 72.0
            pil_image = page.render(scale=scale).to_pil()
            rgb_arr = np.array(pil_image.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Error rendering PDF bytes: {e}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes the full preprocessing pipeline.
        Returns:
            - enhanced_bgr: Color-enhanced deskewed image for visual rendering & bboxes
            - ocr_ready: Contrast-enhanced grayscale/binarized image optimized for OCR text detection
            - stats: Metadata and measurements (deskew angle, dimension changes)
        """
        stats: Dict[str, Any] = {}
        processed_color = image.copy()

        # Step 1: Scale if image is too small for small font recognition
        h, w = processed_color.shape[:2]
        if min(h, w) < self.config.min_dimension:
            scale = self.config.min_dimension / float(min(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            processed_color = cv2.resize(processed_color, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            stats["resized_scale"] = round(scale, 3)
        elif max(h, w) > self.config.max_dimension:
            scale = self.config.max_dimension / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            processed_color = cv2.resize(processed_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
            stats["downscaled_scale"] = round(scale, 3)

        # Step 2: Deskewing
        deskew_angle = 0.0
        if self.config.enable_deskew:
            deskew_angle = self.detect_skew_angle(processed_color)
            if abs(deskew_angle) > 0.4 and abs(deskew_angle) <= self.config.max_deskew_angle:
                processed_color = self.rotate_image(processed_color, deskew_angle)
                stats["deskew_applied"] = True
            else:
                stats["deskew_applied"] = False
        stats["deskew_angle"] = round(deskew_angle, 2)

        # Step 3: Noise reduction on color image
        if self.config.enable_noise_reduction:
            denoised_color = cv2.bilateralFilter(
                processed_color,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )
        else:
            denoised_color = processed_color

        # Step 4: Grayscale & Contrast Limited Adaptive Histogram Equalization (CLAHE)
        gray = cv2.cvtColor(denoised_color, cv2.COLOR_BGR2GRAY)
        
        if self.config.enable_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size,
            )
            ocr_ready = clahe.apply(gray)
        else:
            ocr_ready = gray

        # Step 5: Optional adaptive thresholding if enabled
        if self.config.enable_adaptive_thresh:
            ocr_ready = cv2.adaptiveThreshold(
                ocr_ready,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_block_size,
                self.config.adaptive_c,
            )

        final_h, final_w = denoised_color.shape[:2]
        stats["final_width"] = final_w
        stats["final_height"] = final_h

        return denoised_color, ocr_ready, stats

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Estimates text skew angle in degrees using edge detection and Hough transform.
        Returns rotation angle between -45 and +45 degrees.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Downscale for fast angle detection
        max_dim = 1000
        h, w = gray.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            small = gray

        # Detect text line edges
        edges = cv2.Canny(small, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=small.shape[1] // 8, maxLineGap=20)
        
        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines:
            coords = line.reshape(-1)
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = coords[:4]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Focus on near-horizontal text lines
            if -45.0 <= angle <= 45.0:
                angles.append(angle)

        if not angles:
            return 0.0

        median_angle = float(np.median(angles))
        return median_angle

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image around center while preserving full boundaries."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix translation
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
