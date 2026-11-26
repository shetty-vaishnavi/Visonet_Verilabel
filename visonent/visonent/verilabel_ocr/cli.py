"""
Command-Line Interface (CLI) for VeriLabel AI Module 2: OCR & Document Processing.
Usage:
    python -m verilabel_ocr.cli --input sample_labels/tc01_compliant_label.png --visualize
    python -m verilabel_ocr.cli --dir sample_labels/ --output-dir results/
"""

import argparse
import json
import os
import sys
import glob

from .pipeline import DocumentProcessingPipeline
from .config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="VeriLabel AI - Module 2: OCR & Medical Device Document Processing"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to input medical device label image or PDF file."
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Directory containing label images or PDFs to batch process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="ocr_result.json",
        help="Path to save output JSON result. (Default: ocr_result.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ocr_output",
        help="Directory to save visualizations and outputs. (Default: ocr_output)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate color-coded bounding-box visualization."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only high-level summary of extracted medical fields to console."
    )

    args = parser.parse_args()

    if not args.input and not args.dir:
        parser.print_help()
        print("\nError: Please specify either --input <file> or --dir <folder>.")
        sys.exit(1)

    config = PipelineConfig(output_dir=args.output_dir)
    pipeline = DocumentProcessingPipeline(config)

    files_to_process = []
    if args.input:
        files_to_process.append(args.input)
    elif args.dir:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.tif"):
            files_to_process.extend(glob.glob(os.path.join(args.dir, ext)))

    if not files_to_process:
        print(f"No valid image or PDF files found.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"  VeriLabel AI - Module 2: OCR & Document Processing  ")
    print(f"  Processing {len(files_to_process)} document(s)...")
    print(f"=======================================================\n")

    all_results = []
    for file_path in files_to_process:
        print(f"[>] Processing: {file_path}")
        result = pipeline.process(file_path, save_visualization=args.visualize)
        all_results.append(result)

        if result.success:
            print(f"    [+] Status: SUCCESS ({result.processing_time_ms} ms)")
            print(f"    [+] Text Blocks: {result.total_blocks_detected}")
            print(f"    [+] Deskew Angle: {result.deskew_angle} deg")

            summary = result.structured_data.get_summary()
            print("    [+] Extracted Medical Fields:")
            for k, v in summary.items():
                if v:
                    safe_v = str(v).replace('\u2103', 'C').replace('°', ' deg')
                    safe_v = safe_v.encode('ascii', errors='replace').decode('ascii')
                    print(f"        - {k:22}: {safe_v}")

            if result.annotated_image_path:
                print(f"    [+] Annotated Image: {result.annotated_image_path}")
        else:
            print(f"    [-] Status: FAILED")
            for log in result.logs:
                print(f"        {log}")
        print()

    # Save output JSON
    if len(all_results) == 1:
        out_data = all_results[0].to_dict()
    else:
        out_data = [r.to_dict() for r in all_results]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"[OK] All results successfully saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
