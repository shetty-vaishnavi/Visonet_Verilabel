"""
VeriLabel AI - Module 3: Computer Vision & Medical Device Symbol Detector
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Responsibilities:
- Detect 13 standard medical device packaging symbols:
    • 11 ISO 15223-1 Symbols:
        - 5.1.1: Manufacturer (factory building)
        - 5.1.3: Date of manufacture (factory with date indicator)
        - 5.1.4: Use-by date (hourglass)
        - 5.1.5: Batch code [LOT]
        - 5.1.6: Catalogue number [REF]
        - 5.1.7: Serial number [SN]
        - 5.2.1: Sterile [STERILE]
        - 5.3.7: Temperature limit (thermometer with limits)
        - 5.4.2: Do not re-use (number 2 in circle with slash)
        - 5.4.3: Consult instructions for use (book with 'i')
        - 5.4.4: Caution (triangle with exclamation mark)
    • European CE Conformity Marking (EU-CE)
    • FDA Prescription Only Symbol (FDA-Rx)
- Multi-Scale Resolution Pyramid (0.35x to 1.50x scaling)
- Dual-Correlation Fusion: Normalized cross-correlation on grayscale luminance + Canny edge maps
- Two-Pass Non-Maximum Suppression (NMS):
    • Pass 1: Intra-symbol scale suppression (IoU >= 0.30)
    • Pass 2: Inter-symbol conflict resolution (IoU >= 0.40)
- Zero Hallucination Guarantee: Blank or uniform packaging canvases return zero detections
- Visual inspection overlay rendering with high-contrast bounding boxes and badges
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from .config import (
    TEMPLATES_DIR,
    DEFAULT_SYMBOL_MATCH_THRESHOLD,
    MULTI_SCALE_LEVELS,
    CORRELATION_GRAY_WEIGHT,
    CORRELATION_EDGE_WEIGHT,
    NMS_INTRA_IOU_THRESHOLD,
    NMS_INTER_IOU_THRESHOLD,
    COLOR_SYMBOL_ISO,
    COLOR_SYMBOL_REGULATORY,
)
from .models import DetectedSymbol, BoundingBox

logger = logging.getLogger("VeriLabel.CV.SymbolDetector")

# ISO 15223-1 Standard Reference Mapping
ISO_SYMBOL_METADATA: Dict[str, Dict[str, str]] = {
    "manufacturer": {"code": "5.1.1", "name": "Manufacturer", "description": "Indicates the medical device manufacturer"},
    "date_of_manufacture": {"code": "5.1.3", "name": "Date of manufacture", "description": "Indicates the date when the medical device was manufactured"},
    "use_by_date": {"code": "5.1.4", "name": "Use-by date", "description": "Indicates the date after which the medical device is not to be used"},
    "batch_code": {"code": "5.1.5", "name": "Batch code", "description": "Indicates the manufacturer’s batch code / LOT"},
    "catalogue_number": {"code": "5.1.6", "name": "Catalogue number", "description": "Indicates the manufacturer’s catalogue / REF number"},
    "serial_number": {"code": "5.1.7", "name": "Serial number", "description": "Indicates the manufacturer’s serial number"},
    "sterile": {"code": "5.2.1", "name": "Sterile", "description": "Indicates a medical device that has been subjected to a sterilization process"},
    "temperature_limit": {"code": "5.3.7", "name": "Temperature limit", "description": "Indicates upper and lower temperature limits to which the medical device may safely be exposed"},
    "do_not_reuse": {"code": "5.4.2", "name": "Do not re-use", "description": "Indicates a medical device that is intended for one single use only"},
    "consult_ifu": {"code": "5.4.3", "name": "Consult instructions for use", "description": "Indicates the need for the user to consult instructions for use"},
    "caution": {"code": "5.4.4", "name": "Caution", "description": "Indicates that caution is necessary when operating the device"},
    "ce_mark": {"code": "EU-CE", "name": "CE Conformity", "description": "European conformity marking with optional notified body number"},
    "rx_only": {"code": "FDA-Rx", "name": "Prescription Only", "description": "Caution: Federal law restricts this device to sale by or on order of a physician"},
}


class MedicalSymbolDetector:
    """
    High-performance multi-scale template matching detector for ISO 15223-1 symbols.
    Features:
    - Adaptive multi-scale resolution pyramid
    - Auto-Canny edge matching blended with normalized cross-correlation
    - Intra-symbol and cross-symbol Non-Maximum Suppression (NMS)
    - Zero hallucination on uniform or blank packaging surfaces
    """

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        match_threshold: float = DEFAULT_SYMBOL_MATCH_THRESHOLD,
        scales: Optional[List[float]] = None,
    ):
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self.match_threshold = match_threshold
        self.scales = scales or list(MULTI_SCALE_LEVELS)
        self.templates: Dict[str, np.ndarray] = {}
        self.template_edges: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Loads and precomputes Canny edge representations for all registered ISO symbol templates."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        for sym_name in ISO_SYMBOL_METADATA.keys():
            tpl_path = self.templates_dir / f"{sym_name}.png"
            if tpl_path.exists():
                tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    self.templates[sym_name] = tpl
                    # Precompute Canny edges for template
                    edges = cv2.Canny(tpl, 50, 150)
                    self.template_edges[sym_name] = edges
            else:
                logger.debug(f"Template image missing for symbol: {sym_name}")

        logger.info(f"Loaded {len(self.templates)} ISO symbol templates from {self.templates_dir}")

    def detect(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        target_symbols: Optional[List[str]] = None,
    ) -> List[DetectedSymbol]:
        """
        Executes multi-scale dual correlation template matching across the image.
        Returns deduplicated, NMS-filtered list of DetectedSymbol instances.
        """
        thresh = threshold if threshold is not None else self.match_threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Zero Hallucination Guarantee: Check for blank/uniform image
        img_variance = float(np.var(gray))
        if img_variance < 10.0:
            return []

        # 2. Precompute image Canny edge map
        img_edges = cv2.Canny(gray, 50, 150)

        raw_detections: List[DetectedSymbol] = []
        symbols_to_search = target_symbols or list(self.templates.keys())

        (img_h, img_w) = gray.shape[:2]

        for sym_name in symbols_to_search:
            if sym_name not in self.templates:
                continue

            tpl_base = self.templates[sym_name]
            tpl_edge_base = self.template_edges[sym_name]
            meta = ISO_SYMBOL_METADATA.get(sym_name, {"code": "ISO", "name": sym_name})

            for scale in self.scales:
                # Resize template
                new_w = int(tpl_base.shape[1] * scale)
                new_h = int(tpl_base.shape[0] * scale)

                # Skip if template is larger than image or smaller than 12px
                if new_w >= img_w or new_h >= img_h or new_w < 12 or new_h < 12:
                    continue

                tpl_scaled = cv2.resize(tpl_base, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
                tpl_edge_scaled = cv2.resize(tpl_edge_base, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

                # Normalized Cross-Correlation on Grayscale
                res_gray = cv2.matchTemplate(gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)

                # Normalized Cross-Correlation on Edges
                res_edge = cv2.matchTemplate(img_edges, tpl_edge_scaled, cv2.TM_CCOEFF_NORMED)

                # Fuse correlation scores: 60% grayscale + 40% edge
                fused_res = (CORRELATION_GRAY_WEIGHT * res_gray) + (CORRELATION_EDGE_WEIGHT * res_edge)

                # Find candidate peaks above threshold
                loc = np.where(fused_res >= thresh)
                for pt in zip(*loc[::-1]):
                    score = float(fused_res[pt[1], pt[0]])
                    raw_detections.append(DetectedSymbol(
                        symbol_name=sym_name,
                        display_name=meta["name"],
                        iso_code=meta["code"],
                        confidence=score,
                        bbox=(int(pt[0]), int(pt[1]), new_w, new_h),
                        scale=scale,
                    ))

        # 3. Two-Pass Non-Maximum Suppression (NMS)
        return self._apply_two_pass_nms(raw_detections)

    def _apply_two_pass_nms(self, detections: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Executes two-pass NMS:
        - Pass 1: Intra-symbol scale suppression (IoU >= 0.30)
        - Pass 2: Inter-symbol conflict resolution (IoU >= 0.40)
        """
        if not detections:
            return []

        # Sort all detections descending by confidence score
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # Pass 1: Intra-symbol suppression
        pass1_survivors: List[DetectedSymbol] = []
        by_symbol: Dict[str, List[DetectedSymbol]] = {}
        for d in sorted_dets:
            by_symbol.setdefault(d.symbol_name, []).append(d)

        for sym_name, sym_dets in by_symbol.items():
            kept_for_sym: List[DetectedSymbol] = []
            for candidate in sym_dets:
                cand_box = BoundingBox(*candidate.bbox)
                suppress = False
                for existing in kept_for_sym:
                    ex_box = BoundingBox(*existing.bbox)
                    if cand_box.iou(ex_box) >= NMS_INTRA_IOU_THRESHOLD:
                        suppress = True
                        break
                if not suppress:
                    kept_for_sym.append(candidate)
            pass1_survivors.extend(kept_for_sym)

        # Pass 2: Inter-symbol conflict resolution (higher confidence wins on overlapping bbox)
        pass1_survivors = sorted(pass1_survivors, key=lambda d: d.confidence, reverse=True)
        final_survivors: List[DetectedSymbol] = []

        for candidate in pass1_survivors:
            cand_box = BoundingBox(*candidate.bbox)
            suppress = False
            for existing in final_survivors:
                ex_box = BoundingBox(*existing.bbox)
                if cand_box.iou(ex_box) >= NMS_INTER_IOU_THRESHOLD:
                    suppress = True
                    break
            if not suppress:
                final_survivors.append(candidate)

        return final_survivors

    def annotate(self, image: np.ndarray, detected_symbols: List[DetectedSymbol]) -> np.ndarray:
        """
        Draws high-contrast bounding boxes, ISO clause badges, and confidence badges
        onto a copy of the input image canvas.
        """
        canvas = image.copy()
        for sym in detected_symbols:
            x, y, w, h = sym.bbox
            color = COLOR_SYMBOL_REGULATORY if sym.symbol_name in ("ce_mark", "rx_only") else COLOR_SYMBOL_ISO

            # Draw outer rectangle
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

            # Draw badge label
            badge_text = f"{sym.display_name} ({sym.iso_code}) {int(sym.confidence * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1

            (t_w, t_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
            badge_y = max(y - 6, t_h + 4)
            badge_x = x

            # Background rectangle for text badge
            cv2.rectangle(canvas, (badge_x, badge_y - t_h - 4), (badge_x + t_w + 6, badge_y + 2), (24, 24, 27), -1)
            cv2.rectangle(canvas, (badge_x, badge_y - t_h - 4), (badge_x + t_w + 6, badge_y + 2), color, 1)

            # Text string
            cv2.putText(canvas, badge_text, (badge_x + 3, badge_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return canvas
