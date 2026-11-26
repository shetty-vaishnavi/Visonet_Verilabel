"""
Configuration dataclasses for Module 2: OCR and Document Processing.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessConfig:
    """Settings for image and document preprocessing."""
    # PDF rendering
    pdf_render_dpi: int = 300
    
    # Deskewing
    enable_deskew: bool = True
    max_deskew_angle: float = 45.0
    
    # Noise reduction & contrast enhancement
    enable_noise_reduction: bool = True
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    
    # Binarization / thresholding
    enable_adaptive_thresh: bool = False
    adaptive_block_size: int = 15
    adaptive_c: int = 8
    
    # Scaling bounds for OCR quality
    min_dimension: int = 1000
    max_dimension: int = 4000


@dataclass
class OCRConfig:
    """Settings for OCR engine execution."""
    # Primary language
    lang: str = "en"
    
    # Detection & Recognition
    use_angle_cls: bool = True
    drop_score: float = 0.35  # Confidence cutoff below which tokens are rejected
    
    # Engine mode: "auto", "paddle", or "direct"
    engine_mode: str = "auto"
    
    # Bounding box padding for cropped visualization
    box_padding: int = 4


@dataclass
class PipelineConfig:
    """Overall pipeline orchestration config."""
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    save_annotated_image: bool = True
    output_dir: str = "ocr_output"
