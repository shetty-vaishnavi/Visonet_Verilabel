"""
VeriLabel AI - Medical Device Label Compliance Platform
Team: Brahmastra | GITAM University
"""

from verilabel.pipeline import VeriLabelPipeline, PipelineResult
from verilabel.config import AppConfig, config
from verilabel.exceptions import VeriLabelException

__version__ = "1.0.0"
__all__ = ["VeriLabelPipeline", "PipelineResult", "AppConfig", "config", "VeriLabelException"]
