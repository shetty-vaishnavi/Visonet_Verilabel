"""
VeriLabel AI - Module 3: Data Schemas & Models
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field


@dataclass
class BoundingBox:
    """Bounding box definition in (x, y, width, height) format."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def iou(self, other: "BoundingBox") -> float:
        """Computes Intersection-over-Union (IoU) with another bounding box."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        union = self.area + other.area - inter
        return float(inter / union) if union > 0 else 0.0


@dataclass
class ImageQualityAssessment:
    """Diagnostic metrics representing physical packaging image quality."""
    blur_variance: float
    is_blurred: bool
    brightness_mean: float
    brightness_std: float
    contrast_dynamic_range: float
    is_overexposed: bool
    is_underexposed: bool
    estimated_skew_angle: float
    quality_score: float  # 0.0 to 100.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blur_variance": round(self.blur_variance, 2),
            "is_blurred": self.is_blurred,
            "brightness_mean": round(self.brightness_mean, 2),
            "brightness_std": round(self.brightness_std, 2),
            "contrast_dynamic_range": round(self.contrast_dynamic_range, 2),
            "is_overexposed": self.is_overexposed,
            "is_underexposed": self.is_underexposed,
            "estimated_skew_angle": round(self.estimated_skew_angle, 2),
            "quality_score": round(self.quality_score, 1),
            "recommendations": self.recommendations,
        }


@dataclass
class DecodedBarcode:
    """Represents an authentic decoded barcode on a medical device packaging label."""
    data: str
    barcode_type: str  # e.g., 'DATAMATRIX', 'GS1-128', 'CODE-128', 'QRCODE', 'EAN-13'
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    polygon: Optional[List[Tuple[int, int]]] = None
    engine: str = "zxing-cpp"
    confidence: float = 1.0
    is_gs1: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "type": self.barcode_type,
            "engine": self.engine,
            "confidence": round(self.confidence, 3),
            "is_gs1": self.is_gs1,
            "error": self.error,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            } if self.bbox else None,
            "polygon": self.polygon,
        }


@dataclass
class UDIComponents:
    """Structured elements of a parsed Unique Device Identifier."""
    raw_carrier: str
    standard: str = "GS1"  # "GS1", "HIBCC", or "UNKNOWN"
    device_identifier_di: Optional[str] = None
    is_di_checksum_valid: bool = False
    expiry_date_pi: Optional[str] = None
    lot_number_pi: Optional[str] = None
    serial_number_pi: Optional[str] = None
    manufacturing_date_pi: Optional[str] = None
    additional_id_pi: Optional[str] = None
    is_valid_format: bool = False
    is_expired: bool = False
    parsed_elements: Dict[str, str] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectedSymbol:
    """Represents an identified ISO 15223-1 medical device packaging symbol."""
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
                "height": self.bbox[3],
            },
            "scale": round(self.scale, 2),
        }


@dataclass
class CrossValidationResult:
    """Bidirectional comparison between Barcode UDI data and Human-Readable / OCR fields."""
    lot_matched: Optional[bool] = None
    expiry_matched: Optional[bool] = None
    serial_matched: Optional[bool] = None
    lot_ocr: Optional[str] = None
    lot_barcode: Optional[str] = None
    expiry_ocr: Optional[str] = None
    expiry_barcode: Optional[str] = None
    mismatches: List[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return len(self.mismatches) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lot_matched": self.lot_matched,
            "expiry_matched": self.expiry_matched,
            "serial_matched": self.serial_matched,
            "lot_ocr": self.lot_ocr,
            "lot_barcode": self.lot_barcode,
            "expiry_ocr": self.expiry_ocr,
            "expiry_barcode": self.expiry_barcode,
            "is_consistent": self.is_consistent,
            "mismatches": self.mismatches,
        }


@dataclass
class Module3Result:
    """Unified result container for Module 3: Computer Vision & Barcode processing."""
    file_name: str
    file_hash_sha256: str
    image_width: int
    image_height: int
    quality: ImageQualityAssessment
    barcodes: List[DecodedBarcode] = field(default_factory=list)
    udi_records: List[UDIComponents] = field(default_factory=list)
    symbols: List[DetectedSymbol] = field(default_factory=list)
    cross_validation: Optional[CrossValidationResult] = None
    annotated_image_path: Optional[str] = None
    processing_latency_ms: float = 0.0

    @property
    def has_barcodes(self) -> bool:
        return len(self.barcodes) > 0

    @property
    def has_valid_udi(self) -> bool:
        return any(u.is_di_checksum_valid for u in self.udi_records)

    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_hash_sha256": self.file_hash_sha256,
            "dimensions": {"width": self.image_width, "height": self.image_height},
            "quality": self.quality.to_dict(),
            "barcodes": [b.to_dict() for b in self.barcodes],
            "udi_records": [u.to_dict() for u in self.udi_records],
            "symbols": [s.to_dict() for s in self.symbols],
            "cross_validation": self.cross_validation.to_dict() if self.cross_validation else None,
            "annotated_image_path": self.annotated_image_path,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "summary": {
                "barcode_count": len(self.barcodes),
                "valid_udi_count": sum(1 for u in self.udi_records if u.is_di_checksum_valid),
                "symbol_count": len(self.symbols),
                "image_quality_score": round(self.quality.quality_score, 1),
                "is_blurred": self.quality.is_blurred,
                "is_consistent": self.cross_validation.is_consistent if self.cross_validation else True,
            },
        }
