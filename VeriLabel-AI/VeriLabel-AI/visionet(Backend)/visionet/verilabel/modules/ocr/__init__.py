"""
VeriLabel AI - OCR Modules
"""

from .base import BaseOCREngine, OCRResult, OCRWordBox
from .paddle_engine import PaddleOCREngine
from .fallback_engine import FallbackOCREngine
from .field_extractor import MedicalFieldExtractor, ExtractedLabelFields
from .verilabel_ocr_adapter import VeriLabelOCRPipelineEngine

__all__ = [
    "BaseOCREngine",
    "OCRResult",
    "OCRWordBox",
    "PaddleOCREngine",
    "FallbackOCREngine",
    "MedicalFieldExtractor",
    "ExtractedLabelFields",
    "VeriLabelOCRPipelineEngine",
]
