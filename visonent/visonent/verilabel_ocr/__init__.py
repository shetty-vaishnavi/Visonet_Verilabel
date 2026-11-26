"""
VeriLabel AI - Module 2: OCR and Document Processing
====================================================
Automated document preprocessing, PaddleOCR text extraction,
and regulatory-compliant medical device field identification.
"""

from .models import (
    BoundingBox,
    OCRTextBlock,
    ExtractedField,
    MedicalLabelData,
    OCRResult,
    FieldStatus,
)
from .config import OCRConfig, PreprocessConfig
from .preprocessor import DocumentPreprocessor
from .ocr_engine import OCREngine, PaddleOCREngine
from .field_extractor import MedicalFieldExtractor
from .visualizer import LabelVisualizer
from .pipeline import DocumentProcessingPipeline

__all__ = [
    "BoundingBox",
    "OCRTextBlock",
    "ExtractedField",
    "MedicalLabelData",
    "OCRResult",
    "FieldStatus",
    "OCRConfig",
    "PreprocessConfig",
    "DocumentPreprocessor",
    "OCREngine",
    "PaddleOCREngine",
    "MedicalFieldExtractor",
    "LabelVisualizer",
    "DocumentProcessingPipeline",
]

__version__ = "1.0.0"
