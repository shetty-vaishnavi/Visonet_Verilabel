"""
Data models and schemas for Module 2: OCR and Document Processing.
Built with Pydantic v2 for robust validation and seamless JSON export.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict


class FieldStatus(str, Enum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    SUSPICIOUS = "SUSPICIOUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected text/symbol on label."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    polygon: Optional[List[List[int]]] = None  # 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.xmin + self.width // 2, self.ymin + self.height // 2)

    def to_rect(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class OCRTextBlock(BaseModel):
    """A single text line or token detected by the OCR engine."""
    id: int
    text: str
    confidence: float
    bbox: BoundingBox

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExtractedField(BaseModel):
    """Structured information extracted from label text."""
    field_name: str
    value: Optional[str] = None
    raw_text: Optional[str] = None
    status: FieldStatus = FieldStatus.PRESENT
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None
    source_block_ids: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_present(self) -> bool:
        return self.status == FieldStatus.PRESENT and bool(self.value and self.value.strip())


class MedicalLabelData(BaseModel):
    """
    Standardized medical device label data extracted by Module 2.
    Designed to feed directly into Module 5 (Compliance Validation Engine)
    and Module 4 (Streamlit Frontend).
    """
    # Product identification
    device_name: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="device_name", status=FieldStatus.MISSING)
    )
    trade_name: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="trade_name", status=FieldStatus.MISSING)
    )
    model_number: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="model_number", status=FieldStatus.MISSING)
    )
    catalog_number: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="catalog_number", status=FieldStatus.MISSING)
    )

    # Manufacturing & Traceability
    lot_number: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="lot_number", status=FieldStatus.MISSING)
    )
    serial_number: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="serial_number", status=FieldStatus.MISSING)
    )
    mfg_date: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="mfg_date", status=FieldStatus.MISSING)
    )
    expiry_date: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="expiry_date", status=FieldStatus.MISSING)
    )

    # Manufacturer Info
    manufacturer_name: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="manufacturer_name", status=FieldStatus.MISSING)
    )
    manufacturer_address: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="manufacturer_address", status=FieldStatus.MISSING)
    )
    country_of_origin: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="country_of_origin", status=FieldStatus.MISSING)
    )

    # Handling, Storage & Regulatory
    storage_conditions: ExtractedField = Field(
        default_factory=lambda: ExtractedField(field_name="storage_conditions", status=FieldStatus.MISSING)
    )
    warnings: List[ExtractedField] = Field(default_factory=list)
    regulatory_statements: List[ExtractedField] = Field(default_factory=list)
    udi_text_candidates: List[ExtractedField] = Field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a simplified key-value mapping of all extracted fields."""
        summary = {
            "device_name": self.device_name.value if self.device_name.is_present else None,
            "trade_name": self.trade_name.value if self.trade_name.is_present else None,
            "catalog_ref": self.catalog_number.value if self.catalog_number.is_present else None,
            "lot_number": self.lot_number.value if self.lot_number.is_present else None,
            "serial_number": self.serial_number.value if self.serial_number.is_present else None,
            "mfg_date": self.mfg_date.value if self.mfg_date.is_present else None,
            "expiry_date": self.expiry_date.value if self.expiry_date.is_present else None,
            "manufacturer": self.manufacturer_name.value if self.manufacturer_name.is_present else None,
            "storage": self.storage_conditions.value if self.storage_conditions.is_present else None,
            "warnings_count": len(self.warnings),
            "warnings_list": [w.value for w in self.warnings if w.is_present],
            "regulatory_statements": [s.value for s in self.regulatory_statements if s.is_present],
        }
        return summary


class OCRResult(BaseModel):
    """Complete result object returned by DocumentProcessingPipeline."""
    success: bool
    file_name: str
    file_path: Optional[str] = None
    processing_time_ms: float
    image_width: int
    image_height: int
    deskew_angle: float = 0.0
    
    # Text contents
    raw_text: str
    total_blocks_detected: int
    text_blocks: List[OCRTextBlock] = Field(default_factory=list)
    
    # Structured medical device entities
    structured_data: MedicalLabelData
    
    # Visual inspection asset
    annotated_image_path: Optional[str] = None
    
    # Member 3: Computer Vision, Barcode & UDI Integration
    barcodes: List[Dict[str, Any]] = Field(default_factory=list)
    udi: Optional[Dict[str, Any]] = None
    symbols: List[Dict[str, Any]] = Field(default_factory=list)

    # Diagnostic logs
    logs: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)
