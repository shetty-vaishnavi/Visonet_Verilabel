"""
VeriLabel AI - Module 3: Computer Vision & Barcode Configuration
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# Base paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
TEMPLATES_DIR = MODULE_DIR / "templates"
SAMPLE_LABELS_DIR = PROJECT_ROOT / "sample_labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cv_output"

# Quality & Preprocessing Thresholds
DEFAULT_BLUR_THRESHOLD: float = 100.0       # Laplacian variance below this is flagged as blurred
DEFAULT_MIN_BRIGHTNESS: float = 40.0        # Mean pixel intensity below this is underexposed
DEFAULT_MAX_BRIGHTNESS: float = 230.0       # Mean pixel intensity above this is overexposed
DEFAULT_MIN_CONTRAST: float = 30.0          # Dynamic range std-dev below this is low contrast
DEFAULT_MAX_DESKEW_ANGLE: float = 45.0      # Maximum deskew rotation angle in degrees
DEFAULT_TARGET_DPI: int = 300               # Target resolution for PDF rendering

# Multi-scale Symbol Detection Configuration
DEFAULT_SYMBOL_MATCH_THRESHOLD: float = 0.58
MULTI_SCALE_LEVELS: List[float] = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.50]
CORRELATION_GRAY_WEIGHT: float = 0.60
CORRELATION_EDGE_WEIGHT: float = 0.40
NMS_INTRA_IOU_THRESHOLD: float = 0.30       # Suppression for same symbol at varying scales
NMS_INTER_IOU_THRESHOLD: float = 0.40       # Suppression for overlapping distinct symbols

# Barcode Scanning Configuration
DEFAULT_BARCODE_FORMATS: List[str] = [
    "DATAMATRIX",
    "QRCODE",
    "CODE-128",
    "GS1-128",
    "CODE-39",
    "EAN-13",
    "EAN-8",
    "UPC-A",
    "UPC-E",
    "ITF",
    "AZTEC",
    "PDF417",
]

# Visualization Color Scheme (BGR)
COLOR_BARCODE_VALID: Tuple[int, int, int] = (46, 204, 113)     # Emerald Green
COLOR_BARCODE_INVALID: Tuple[int, int, int] = (46, 46, 231)    # Crimson Red
COLOR_SYMBOL_ISO: Tuple[int, int, int] = (245, 150, 48)         # Cyan / Sky Blue (BGR)
COLOR_SYMBOL_REGULATORY: Tuple[int, int, int] = (0, 165, 255)   # Amber Orange
COLOR_TEXT_BG: Tuple[int, int, int] = (24, 24, 27)             # Dark Slate
COLOR_TEXT_FG: Tuple[int, int, int] = (255, 255, 255)          # White


@dataclass
class PipelineConfig:
    """Configurable options for the Computer Vision & Barcode Pipeline."""
    templates_dir: Path = TEMPLATES_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS
    max_brightness: float = DEFAULT_MAX_BRIGHTNESS
    symbol_match_threshold: float = DEFAULT_SYMBOL_MATCH_THRESHOLD
    scales: List[float] = field(default_factory=lambda: list(MULTI_SCALE_LEVELS))
    enable_deskew: bool = True
    enable_enhancement: bool = True
    enable_adaptive_barcode_retries: bool = True
    enable_orthogonal_barcode_rotations: bool = True
    enable_inverted_barcode_polarity: bool = True
    save_annotated_image: bool = True
