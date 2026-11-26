"""
Visualizer and Annotation Utility for VeriLabel AI OCR.
Generates color-coded overlays and bounding box annotations for Streamlit & compliance reports.
"""

from typing import Optional, List, Dict, Tuple
import os
import cv2
import numpy as np

from .models import MedicalLabelData, OCRTextBlock, ExtractedField, BoundingBox


class LabelVisualizer:
    """
    Renders visual feedback on medical device labels with color-coded regulatory badges.
    """

    # Color palette (BGR format for OpenCV)
    COLORS = {
        "device_name": (0, 180, 0),          # Green
        "trade_name": (0, 150, 0),           # Dark Green
        "catalog_number": (0, 200, 100),     # Sea Green
        "lot_number": (220, 100, 0),         # Blue
        "serial_number": (200, 120, 30),     # Light Blue
        "mfg_date": (180, 130, 0),           # Deep Cyan
        "expiry_date": (255, 69, 0),         # Royal Blue
        "manufacturer_name": (0, 140, 255),  # Orange
        "manufacturer_address": (0, 165, 255), # Light Orange
        "country_of_origin": (0, 190, 255),
        "storage_conditions": (180, 0, 180), # Purple
        "warning": (0, 0, 220),              # Red
        "regulatory_statement": (100, 0, 180),
        "udi_text_candidate": (150, 50, 100),
        "general_text": (200, 200, 200),     # Light Gray
    }

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha  # Transparency factor for highlight fill

    def annotate(
        self,
        image: np.ndarray,
        structured_data: MedicalLabelData,
        all_blocks: Optional[List[OCRTextBlock]] = None,
        draw_unmapped_blocks: bool = True,
    ) -> np.ndarray:
        """
        Draws semi-transparent bounding boxes and labels onto the image.
        Returns a new annotated BGR image.
        """
        output = image.copy()
        overlay = image.copy()

        # Step 1: Draw unmapped text blocks with faint gray bounding boxes
        if draw_unmapped_blocks and all_blocks:
            for block in all_blocks:
                x1, y1, x2, y2 = block.bbox.to_rect()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLORS["general_text"], 1)

        # Step 2: Collect all extracted fields with their designated colors
        fields_to_draw: List[Tuple[ExtractedField, Tuple[int, int, int], str]] = []

        def add_field(field_obj: ExtractedField, label_tag: str):
            if field_obj and field_obj.is_present and field_obj.bbox:
                color = self.COLORS.get(field_obj.field_name, (0, 255, 0))
                fields_to_draw.append((field_obj, color, label_tag))

        add_field(structured_data.device_name, "DEVICE")
        add_field(structured_data.catalog_number, "REF")
        add_field(structured_data.lot_number, "LOT")
        add_field(structured_data.serial_number, "SN")
        add_field(structured_data.mfg_date, "MFG")
        add_field(structured_data.expiry_date, "EXP")
        add_field(structured_data.manufacturer_name, "MFR")
        add_field(structured_data.manufacturer_address, "ADDR")
        add_field(structured_data.country_of_origin, "ORIGIN")
        add_field(structured_data.storage_conditions, "STORAGE")

        for w in structured_data.warnings:
            add_field(w, "WARNING")

        for s in structured_data.regulatory_statements:
            add_field(s, "STATEMENT")

        for u in structured_data.udi_text_candidates:
            add_field(u, "UDI")

        # Step 3: Draw filled color overlays
        for field_obj, color, label_tag in fields_to_draw:
            box = field_obj.bbox
            x1, y1, x2, y2 = box.to_rect()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Blend overlay with base image for transparency
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)

        # Step 4: Draw solid borders and text badges on top
        for field_obj, color, label_tag in fields_to_draw:
            box = field_obj.bbox
            x1, y1, x2, y2 = box.to_rect()

            # Outer border
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Badge pill at top-left of box
            badge_text = f"{label_tag}: {field_obj.value[:18]}..." if len(str(field_obj.value)) > 20 else f"{label_tag}: {field_obj.value}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - th - 6)
            badge_y2 = y1
            badge_x2 = min(output.shape[1], x1 + tw + 8)

            cv2.rectangle(output, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                output,
                badge_text,
                (x1 + 4, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return output

    def save_annotated(self, image: np.ndarray, output_path: str) -> str:
        """Saves annotated image to disk and returns absolute path."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, image)
        return os.path.abspath(output_path)
