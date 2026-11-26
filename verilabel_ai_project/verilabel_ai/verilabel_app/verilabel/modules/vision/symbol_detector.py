"""
VeriLabel AI - Module 3: Computer Vision & Medical Device Symbol Detector
Team: Brahmastra | GITAM University
Responsibilities:
- Detect ISO 15223-1 symbols using multi-scale template matching
- Provide Non-Maximum Suppression (NMS) for overlapping detections
- Verify detected symbols against expected symbols from product claims
- Generate visual overlay masks with bounding boxes for the frontend
"""

import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from verilabel.config import SYMBOLS_DIR, DEFAULT_SYMBOL_MATCH_THRESHOLD, MULTI_SCALE_LEVELS
from verilabel.exceptions import SymbolDetectionError
from verilabel.core.utils import load_image_safely

logger = logging.getLogger("VeriLabel.Vision.SymbolDetector")

# ISO 15223-1 Standard Reference Mapping
ISO_SYMBOL_METADATA = {
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
    "rx_only": {"code": "FDA-Rx", "name": "Prescription Only", "description": "Caution: Federal law restricts this device to sale by or on order of a physician"}
}


@dataclass
class DetectedSymbol:
    """Represents an identified ISO 15223-1 medical device symbol."""
    symbol_name: str
    display_name: str
    iso_code: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    scale: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol_name": self.symbol_name,
            "display_name": self.display_name,
            "iso_code": self.iso_code,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3]
            },
            "scale": round(self.scale, 2)
        }


class MedicalSymbolDetector:
    """
    Multi-scale template matching detector for ISO 15223-1 medical device symbols.
    Designed with pluggable hooks for YOLOv8/v11 integration (Member 3).
    """

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        threshold: float = DEFAULT_SYMBOL_MATCH_THRESHOLD,
        scales: Optional[List[float]] = None
    ):
        self.templates_dir = Path(templates_dir) if templates_dir else SYMBOLS_DIR
        self.threshold = threshold
        self.scales = scales or MULTI_SCALE_LEVELS
        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self):
        """Loads all reference symbol templates from disk."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        for img_file in self.templates_dir.glob("*.png"):
            symbol_key = img_file.stem
            try:
                tpl = load_image_safely(img_file)
                # Convert to grayscale
                gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                self.templates[symbol_key] = gray_tpl
            except Exception as e:
                logger.error(f"Error loading template {img_file}: {e}")

        logger.info(f"Loaded {len(self.templates)} symbol templates for detection.")

    def detect_symbols(self, image: np.ndarray, custom_threshold: Optional[float] = None) -> List[DetectedSymbol]:
        """
        Executes multi-scale template matching across all loaded ISO symbols.
        Applies Non-Maximum Suppression to filter redundant detections.
        """
        if not self.templates:
            self._load_templates()

        threshold = custom_threshold if custom_threshold is not None else self.threshold

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        img_h, img_w = image_gray.shape

        all_detections: List[Dict[str, Any]] = []

        # Edge maps improve template matching invariance to background lighting
        img_edges = cv2.Canny(image_gray, 50, 150)

        for symbol_name, tpl_gray in self.templates.items():
            tpl_h, tpl_w = tpl_gray.shape
            tpl_edges = cv2.Canny(tpl_gray, 50, 150)

            best_val = -1.0
            symbol_candidates = []

            for scale in self.scales:
                # Scale template
                scaled_w = int(tpl_w * scale)
                scaled_h = int(tpl_h * scale)

                if scaled_w >= img_w or scaled_h >= img_h or scaled_w < 15 or scaled_h < 15:
                    continue

                scaled_tpl = cv2.resize(tpl_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                scaled_edges = cv2.resize(tpl_edges, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

                # Match using correlation on grayscale
                res_gray = cv2.matchTemplate(image_gray, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                # Match using correlation on edges
                res_edge = cv2.matchTemplate(img_edges, scaled_edges, cv2.TM_CCOEFF_NORMED)

                # Blend score (0.6 grayscale + 0.4 edge)
                res = (0.6 * res_gray) + (0.4 * res_edge)

                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    score = float(res[pt[1], pt[0]])
                    symbol_candidates.append({
                        "name": symbol_name,
                        "score": score,
                        "bbox": (int(pt[0]), int(pt[1]), scaled_w, scaled_h),
                        "scale": scale
                    })

            # Apply NMS per symbol
            filtered = self._apply_nms(symbol_candidates, iou_thresh=0.3)
            all_detections.extend(filtered)

        # Global cross-symbol NMS to avoid conflicting symbols matching the same spot
        final_detections = self._apply_nms(all_detections, iou_thresh=0.4)

        # Build DetectedSymbol dataclasses
        results: List[DetectedSymbol] = []
        for det in final_detections:
            meta = ISO_SYMBOL_METADATA.get(det["name"], {"code": "ISO-GEN", "name": det["name"]})
            results.append(
                DetectedSymbol(
                    symbol_name=det["name"],
                    display_name=meta["name"],
                    iso_code=meta["code"],
                    confidence=det["score"],
                    bbox=det["bbox"],
                    scale=det["scale"]
                )
            )

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.3) -> List[Dict[str, Any]]:
        """Applies Non-Maximum Suppression to remove overlapping candidate boxes."""
        if not detections:
            return []

        # Sort by score descending
        dets = sorted(detections, key=lambda x: x["score"], reverse=True)
        picked = []

        while dets:
            current = dets.pop(0)
            picked.append(current)

            remaining = []
            for item in dets:
                iou = self._compute_iou(current["bbox"], item["bbox"])
                if iou < iou_thresh:
                    remaining.append(item)
            dets = remaining

        return picked

    def _compute_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Computes Intersection over Union (IoU) of two bounding boxes (x, y, w, h)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        unionArea = float(boxAArea + boxBArea - interArea)
        if unionArea <= 0:
            return 0.0

        return interArea / unionArea

    def draw_detections(self, image: np.ndarray, detections: List[DetectedSymbol]) -> np.ndarray:
        """
        Draws bounding boxes and labels around detected symbols for UI inspection.
        Returns annotated BGR image.
        """
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det.bbox
            # Green bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)

            label = f"{det.display_name} ({det.confidence*100:.0f}%)"
            # Background tag for readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - text_h - 6)), (x + text_w + 4, y), (0, 160, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x + 2, max(text_h + 2, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return annotated
