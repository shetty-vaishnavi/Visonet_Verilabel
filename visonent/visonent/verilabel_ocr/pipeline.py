"""
Top-level Document Processing Pipeline for VeriLabel AI Module 2.
Coordinates preprocessing, OCR inference, medical entity extraction,
and visual overlay generation.
"""

from typing import Union, Optional, List, Dict, Any
import os
import time
import logging
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .models import OCRResult, OCRTextBlock, MedicalLabelData
from .preprocessor import DocumentPreprocessor
from .ocr_engine import create_ocr_engine, OCREngine
from .field_extractor import MedicalFieldExtractor
from .visualizer import LabelVisualizer

logger = logging.getLogger("verilabel_ocr.pipeline")


class DocumentProcessingPipeline:
    """
    Main entry point for Module 2: OCR and Document Processing.
    Usage:
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process("path/to/label.png")
        print(result.to_json())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.preprocessor = DocumentPreprocessor(self.config.preprocess)
        self.ocr_engine = create_ocr_engine(self.config.ocr)
        self.extractor = MedicalFieldExtractor()
        self.visualizer = LabelVisualizer()

        os.makedirs(self.config.output_dir, exist_ok=True)

    def process(
        self,
        input_source: Union[str, bytes, np.ndarray, Image.Image],
        file_name: Optional[str] = None,
        save_visualization: bool = True,
        output_vis_path: Optional[str] = None,
    ) -> OCRResult:
        """
        Executes the full OCR and document processing workflow on an input document or image.
        """
        start_time = time.perf_counter()
        logs: List[str] = []

        # Determine file name
        if not file_name:
            if isinstance(input_source, str):
                file_name = os.path.basename(input_source)
            else:
                file_name = f"document_{int(time.time())}.png"

        logs.append(f"Starting processing for '{file_name}'")

        try:
            # 1. Load document (handles PDF conversion if applicable)
            raw_image, load_meta = self.preprocessor.load_document(input_source)
            logs.append(f"Document loaded successfully. PDF: {load_meta.get('is_pdf')}, "
                        f"Orig size: {load_meta.get('original_width')}x{load_meta.get('original_height')}")

            # 2. Preprocess: Deskewing, Noise Reduction & CLAHE
            color_enhanced, ocr_ready, prep_stats = self.preprocessor.preprocess(raw_image)
            deskew_angle = prep_stats.get("deskew_angle", 0.0)
            logs.append(f"Preprocessing completed. Deskew angle: {deskew_angle}°, Final size: "
                        f"{prep_stats.get('final_width')}x{prep_stats.get('final_height')}")

            # 3. OCR Text Extraction (PaddleOCR)
            text_blocks: List[OCRTextBlock] = self.ocr_engine.extract_text_blocks(ocr_ready)
            logs.append(f"OCR extraction completed. Total text blocks found: {len(text_blocks)}")

            # 4. Concatenate raw text
            full_text = "\n".join([b.text for b in text_blocks])

            # 5. Extract structured medical fields
            structured_data: MedicalLabelData = self.extractor.extract(text_blocks)
            logs.append("Medical field extraction completed.")

            # 6. Generate color-coded visualization if requested
            annotated_path = None
            if save_visualization:
                annotated_img = self.visualizer.annotate(
                    image=color_enhanced,
                    structured_data=structured_data,
                    all_blocks=text_blocks,
                )

                if not output_vis_path:
                    base_name, _ = os.path.splitext(file_name)
                    vis_filename = f"{base_name}_annotated.png"
                    output_vis_path = os.path.join(self.config.output_dir, vis_filename)

                annotated_path = self.visualizer.save_annotated(annotated_img, output_vis_path)
                logs.append(f"Annotated visualization saved to '{annotated_path}'")

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logs.append(f"Pipeline finished in {elapsed_ms} ms")

            return OCRResult(
                success=True,
                file_name=file_name,
                file_path=input_source if isinstance(input_source, str) else None,
                processing_time_ms=elapsed_ms,
                image_width=prep_stats.get("final_width", raw_image.shape[1]),
                image_height=prep_stats.get("final_height", raw_image.shape[0]),
                deskew_angle=deskew_angle,
                raw_text=full_text,
                total_blocks_detected=len(text_blocks),
                text_blocks=text_blocks,
                structured_data=structured_data,
                annotated_image_path=annotated_path,
                logs=logs,
            )

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception("OCR Pipeline error on %s", file_name)
            logs.append(f"ERROR: {str(e)}")

            return OCRResult(
                success=False,
                file_name=file_name,
                file_path=input_source if isinstance(input_source, str) else None,
                processing_time_ms=elapsed_ms,
                image_width=0,
                image_height=0,
                deskew_angle=0.0,
                raw_text="",
                total_blocks_detected=0,
                text_blocks=[],
                structured_data=MedicalLabelData(),
                annotated_image_path=None,
                logs=logs,
            )

    def process_batch(self, file_paths: List[str]) -> List[OCRResult]:
        """Processes a list of files sequentially and returns results."""
        results = []
        for path in file_paths:
            res = self.process(path)
            results.append(res)
        return results
