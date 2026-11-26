"""
VeriLabel AI - Module 3: Visualizer & Annotated Overlay Generator
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Responsibilities:
- Draw high-contrast, color-coded visual inspection overlays:
    • Emerald Green for valid UDI barcodes with GTIN checksum verification badge
    • Crimson Red for corrupted / invalid UDI barcodes with alert badges
    • Cyan / Blue for ISO 15223-1 medical device symbols with ISO standard clause codes
    • Amber / Gold for regulatory emblems (CE Mark, FDA Rx Only)
- Top diagnostic banner overlay detailing quality score, blur variance, and detection counts
- Side-by-side comparison canvas (Original vs. Annotated Overlay)
- Export to PNG / JPG for frontend UI, audit reports, and PDF inclusion
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from .config import (
    COLOR_BARCODE_VALID,
    COLOR_BARCODE_INVALID,
    COLOR_SYMBOL_ISO,
    COLOR_SYMBOL_REGULATORY,
    COLOR_TEXT_BG,
    COLOR_TEXT_FG,
)
from .models import DecodedBarcode, UDIComponents, DetectedSymbol, ImageQualityAssessment

logger = logging.getLogger("VeriLabel.CV.Visualizer")


class LabelVisualizer:
    """
    Renders publication-ready regulatory visual inspection overlays onto packaging labels.
    """

    def __init__(self, font_scale: float = 0.45, line_thickness: int = 2):
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def create_overlay(
        self,
        image: np.ndarray,
        barcodes: List[DecodedBarcode],
        udi_records: List[UDIComponents],
        symbols: List[DetectedSymbol],
        quality: Optional[ImageQualityAssessment] = None,
        include_banner: bool = True,
    ) -> np.ndarray:
        """
        Renders complete color-coded inspection overlay on the image.
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]

        # 1. Annotate Detected Barcodes & UDI Information
        udi_map: Dict[str, UDIComponents] = {u.raw_carrier.strip(): u for u in udi_records}

        for bc in barcodes:
            raw_data = bc.data.strip()
            udi_info = udi_map.get(raw_data)

            # Determine color and status
            is_valid = udi_info.is_di_checksum_valid if udi_info else (bc.error is None)
            color = COLOR_BARCODE_VALID if is_valid else COLOR_BARCODE_INVALID

            # Draw polygon or bounding box
            if bc.polygon and len(bc.polygon) >= 4:
                pts = np.array(bc.polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=self.line_thickness)
            elif bc.bbox:
                bx, by, bw, bh = bc.bbox
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, self.line_thickness)

            # Draw badge label
            if bc.bbox:
                bx, by, bw, bh = bc.bbox
                status_text = "UDI VALID" if is_valid else "INVALID CHECKSUM"
                di_text = f"DI: {udi_info.device_identifier_di}" if (udi_info and udi_info.device_identifier_di) else bc.barcode_type
                badge_str = f"[{bc.barcode_type}] {di_text} | {status_text}"

                self._draw_badge(canvas, badge_str, (bx, by), color)

        # 2. Annotate Detected ISO 15223-1 Symbols
        for sym in symbols:
            sx, sy, sw, sh = sym.bbox
            color = COLOR_SYMBOL_REGULATORY if sym.symbol_name in ("ce_mark", "rx_only") else COLOR_SYMBOL_ISO

            # Outer bounding rectangle
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, self.line_thickness)

            # Text badge
            badge_str = f"{sym.display_name} ({sym.iso_code}) {int(sym.confidence * 100)}%"
            self._draw_badge(canvas, badge_str, (sx, sy), color)

        # 3. Add Diagnostic Information Banner at the top
        if include_banner and quality is not None:
            canvas = self._add_diagnostic_banner(canvas, quality, len(barcodes), len(symbols))

        return canvas

    def _draw_badge(
        self,
        canvas: np.ndarray,
        text: str,
        anchor: Tuple[int, int],
        border_color: Tuple[int, int, int],
    ) -> None:
        """Draws a high-contrast badge with background fill and border."""
        x, y = anchor
        (t_w, t_h), _ = cv2.getTextSize(text, self.font, self.font_scale, 1)

        badge_y = max(y - 5, t_h + 6)
        badge_x = max(2, x)

        # Draw filled background rectangle
        cv2.rectangle(canvas, (badge_x, badge_y - t_h - 4), (badge_x + t_w + 8, badge_y + 3), COLOR_TEXT_BG, -1)
        # Draw border
        cv2.rectangle(canvas, (badge_x, badge_y - t_h - 4), (badge_x + t_w + 8, badge_y + 3), border_color, 1)
        # Draw text
        cv2.putText(canvas, text, (badge_x + 4, badge_y - 2), self.font, self.font_scale, COLOR_TEXT_FG, 1, cv2.LINE_AA)

    def _add_diagnostic_banner(
        self,
        image: np.ndarray,
        quality: ImageQualityAssessment,
        num_barcodes: int,
        num_symbols: int,
    ) -> np.ndarray:
        """Prepends a sleek diagnostic telemetry banner at the top of the canvas."""
        h, w = image.shape[:2]
        banner_height = 42

        banner = np.full((banner_height, w, 3), (20, 24, 33), dtype=np.uint8)

        # Build telemetry text
        q_color = (46, 204, 113) if quality.quality_score >= 80 else ((0, 165, 255) if quality.quality_score >= 60 else (46, 46, 231))
        t1 = f"VeriLabel AI Module 3 | Quality: {quality.quality_score:.1f}/100"
        t2 = f"Blur Var: {quality.blur_variance:.1f} {'(BLURRED)' if quality.is_blurred else '(SHARP)'}"
        t3 = f"Barcodes: {num_barcodes} | ISO Symbols: {num_symbols}"

        full_banner_text = f"{t1}   •   {t2}   •   {t3}"
        cv2.putText(banner, full_banner_text, (15, 26), self.font, 0.44, (240, 240, 245), 1, cv2.LINE_AA)

        # Top indicator line
        cv2.line(banner, (0, 0), (w, 0), q_color, 2)
        cv2.line(banner, (0, banner_height - 1), (w, banner_height - 1), (50, 60, 75), 1)

        return np.vstack((banner, image))

    def create_side_by_side(self, original: np.ndarray, annotated: np.ndarray) -> np.ndarray:
        """
        Creates a side-by-side comparison canvas showing the original image and annotated overlay.
        """
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]

        target_h = max(h1, h2)
        scale1 = target_h / h1
        scale2 = target_h / h2

        img1_resized = cv2.resize(original, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(annotated, (int(w2 * scale2), target_h))

        # Add title headers
        header_h = 32
        w_comb = img1_resized.shape[1] + img2_resized.shape[1]
        header = np.full((header_h, w_comb, 3), (18, 22, 30), dtype=np.uint8)

        cv2.putText(header, "Original Packaging Label", (20, 22), self.font, 0.50, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(header, "Module 3 Inspection Overlay (CV & Barcode)", (img1_resized.shape[1] + 20, 22), self.font, 0.50, (46, 204, 113), 1, cv2.LINE_AA)

        combined_body = np.hstack((img1_resized, img2_resized))
        # Draw divider line
        cv2.line(combined_body, (img1_resized.shape[1], 0), (img1_resized.shape[1], target_h), (60, 70, 85), 2)

        return np.vstack((header, combined_body))

    def save_image(self, image: np.ndarray, output_path: Path) -> Path:
        """Saves image to disk, ensuring directory exists."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path
