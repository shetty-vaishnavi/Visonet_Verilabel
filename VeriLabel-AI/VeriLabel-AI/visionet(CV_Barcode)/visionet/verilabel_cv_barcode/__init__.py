"""
VeriLabel AI - Module 3: Computer Vision & Barcode Processing
Platform: Medical Device Label Verification & Compliance Platform
Team: Brahmastra | Institution: GITAM University
Module Owner: Member 3 (Computer Vision and Barcode Module)
"""

from .config import PipelineConfig, TEMPLATES_DIR, SAMPLE_LABELS_DIR
from .models import (
    BoundingBox,
    ImageQualityAssessment,
    DecodedBarcode,
    UDIComponents,
    DetectedSymbol,
    CrossValidationResult,
    Module3Result,
)
from .image_processor import ImageProcessor, ImageProcessingError
from .barcode_scanner import BarcodeScanner, ScanTelemetry
from .udi_validator import UDIValidator, validate_gtin_checksum, validate_hibcc_modulo43
from .symbol_detector import MedicalSymbolDetector, ISO_SYMBOL_METADATA
from .visualizer import LabelVisualizer
from .pipeline import ComputerVisionBarcodePipeline

__version__ = "1.0.0"
__author__ = "Member 3 (Team Brahmastra, GITAM University)"

__all__ = [
    "ComputerVisionBarcodePipeline",
    "PipelineConfig",
    "ImageProcessor",
    "ImageProcessingError",
    "BarcodeScanner",
    "ScanTelemetry",
    "UDIValidator",
    "validate_gtin_checksum",
    "validate_hibcc_modulo43",
    "MedicalSymbolDetector",
    "ISO_SYMBOL_METADATA",
    "LabelVisualizer",
    "BoundingBox",
    "ImageQualityAssessment",
    "DecodedBarcode",
    "UDIComponents",
    "DetectedSymbol",
    "CrossValidationResult",
    "Module3Result",
    "TEMPLATES_DIR",
    "SAMPLE_LABELS_DIR",
]
