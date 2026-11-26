"""
VeriLabel AI - OCR Engine Abstract Base Class
Team: Brahmastra | GITAM University
Defines standard contracts for all OCR implementations (PaddleOCR, Tesseract, Fallback).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np


@dataclass
class OCRWordBox:
    """Represents a detected word or text phrase with spatial coordinates and confidence."""
    text: str
    confidence: float
    bbox: List[Tuple[int, int]]  # 4 vertices: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0

    def __post_init__(self):
        if self.bbox and len(self.bbox) == 4:
            xs = [pt[0] for pt in self.bbox]
            ys = [pt[1] for pt in self.bbox]
            self.x_min = min(xs)
            self.y_min = min(ys)
            self.x_max = max(xs)
            self.y_max = max(ys)


@dataclass
class OCRResult:
    """Unified container for all text extracted from a medical device label."""
    raw_text: str
    lines: List[str]
    word_boxes: List[OCRWordBox]
    engine_name: str
    average_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "raw_text": self.raw_text,
            "total_lines": len(self.lines),
            "total_words": len(self.word_boxes),
            "average_confidence": round(self.average_confidence, 4),
            "lines": self.lines
        }


class BaseOCREngine(ABC):
    """Abstract interface for medical device label OCR engines."""

    @abstractmethod
    def extract(self, image: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> OCRResult:
        """
        Extracts structured text and bounding boxes from an input OpenCV image.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the underlying engine dependencies and model weights are installed."""
        pass
