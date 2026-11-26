"""
VeriLabel AI - Configuration & Constants
Team: Brahmastra | GITAM University
Project: Medical Device Label Compliance Platform
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VERILABEL_DIR = BASE_DIR / "verilabel"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
SYMBOLS_DIR = VERILABEL_DIR / "modules" / "vision" / "templates"
RULES_FILE = VERILABEL_DIR / "modules" / "compliance" / "rules.json"
DATABASE_PATH = BASE_DIR / "verilabel_audit.db"

# Supported File Types
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
ALLOWED_DOCUMENT_EXTENSIONS: Set[str] = {".pdf"}
ALL_ALLOWED_EXTENSIONS: Set[str] = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_DOCUMENT_EXTENSIONS)

# Max file size: 25 MB
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024

# Compliance Scoring & Thresholds
COMPLIANCE_SCORE_PASS: float = 95.0
COMPLIANCE_SCORE_CONDITIONAL: float = 80.0

# Computer Vision & Template Matching Thresholds
DEFAULT_SYMBOL_MATCH_THRESHOLD: float = 0.52
MULTI_SCALE_LEVELS: List[float] = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.25]

# OCR Settings
OCR_CONFIDENCE_THRESHOLD: float = 0.50

# Regulatory Standards References
STANDARDS = {
    "FDA": "FDA 21 CFR Part 801 (Medical Device Labeling) & Part 830 (Unique Device Identification)",
    "EU_MDR": "EU MDR 2017/745 Annex I, Chapter III (Requirements regarding device information)",
    "ISO": "ISO 15223-1:2021 (Symbols to be used with information to be supplied by the manufacturer)",
    "GS1": "GS1 General Specifications - Healthcare UDI Barcode Standards"
}

@dataclass
class AppConfig:
    """Runtime configuration container with sensible defaults."""
    base_dir: Path = BASE_DIR
    db_path: Path = DATABASE_PATH
    symbols_dir: Path = SYMBOLS_DIR
    rules_file: Path = RULES_FILE
    ocr_confidence_min: float = OCR_CONFIDENCE_THRESHOLD
    symbol_match_min: float = DEFAULT_SYMBOL_MATCH_THRESHOLD
    debug_mode: bool = False
    enable_gpu: bool = False

# Global Default Config
config = AppConfig()
