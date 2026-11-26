"""
VeriLabel AI - Module 3: Command Line Interface (CLI)
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Usage:
    # Process a single label and save visual overlay
    python -m verilabel_cv_barcode.cli --input sample_labels/tc01_compliant_label.png --visualize

    # Process all labels in a folder and export JSON report
    python -m verilabel_cv_barcode.cli --batch sample_labels/ --output cv_output/ --json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

from .pipeline import ComputerVisionBarcodePipeline
from .config import PipelineConfig


def format_cli_summary(res) -> str:
    """Formats Module3Result into clean, readable terminal output."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f" VeriLabel AI - Module 3: Computer Vision & Barcode Inspection")
    lines.append(f" File: {res.file_name} ({res.image_width}x{res.image_height}) | Latency: {res.processing_latency_ms:.1f}ms")
    lines.append("=" * 72)

    # Image Quality Diagnostics
    q = res.quality
    blur_badge = "[BLURRED - RETAKE]" if q.is_blurred else "[SHARP]"
    lines.append(f"\n[1] Image Quality Diagnostics:")
    lines.append(f"    • Quality Score : {q.quality_score:.1f} / 100.0")
    lines.append(f"    • Sharpness     : Laplacian Var = {q.blur_variance:.1f} {blur_badge}")
    lines.append(f"    • Illumination  : Mean = {q.brightness_mean:.1f}, Std = {q.brightness_std:.1f}")
    lines.append(f"    • Dynamic Range : {q.contrast_dynamic_range:.1f}")
    lines.append(f"    • Skew Angle    : {q.estimated_skew_angle:.1f} degrees")
    if q.recommendations:
        lines.append(f"    • Advisory      : {q.recommendations[0]}")

    # Barcodes & UDI
    lines.append(f"\n[2] Barcode & UDI Extraction ({len(res.barcodes)} found):")
    if not res.barcodes:
        lines.append("    • No machine-readable barcodes detected.")
    else:
        for idx, (bc, udi) in enumerate(zip(res.barcodes, res.udi_records), start=1):
            check_badge = "[VALID CHECKSUM]" if udi.is_di_checksum_valid else "[CORRUPTED CHECKSUM]"
            lines.append(f"    ({idx}) Format: {bc.barcode_type} | Engine: {bc.engine}")
            lines.append(f"        Payload : {bc.data}")
            if udi.device_identifier_di:
                lines.append(f"        DI/GTIN : {udi.device_identifier_di} {check_badge}")
            if udi.lot_number_pi:
                lines.append(f"        Lot (10): {udi.lot_number_pi}")
            if udi.expiry_date_pi:
                expired_badge = "[EXPIRED]" if udi.is_expired else "[VALID FUTURE]"
                lines.append(f"        EXP (17): {udi.expiry_date_pi} {expired_badge}")
            if udi.serial_number_pi:
                lines.append(f"        SN  (21): {udi.serial_number_pi}")
            if udi.validation_errors:
                lines.append(f"        ERRORS  : {', '.join(udi.validation_errors)}")

    # ISO 15223-1 Symbols
    lines.append(f"\n[3] ISO 15223-1 Medical Device Symbols ({len(res.symbols)} detected):")
    if not res.symbols:
        lines.append("    • No regulatory symbols detected.")
    else:
        for s in res.symbols:
            lines.append(f"    • {s.display_name:<24} | Clause: {s.iso_code:<8} | Confidence: {int(s.confidence * 100)}% | Scale: {s.scale:.2f}")

    if res.annotated_image_path:
        lines.append(f"\n[4] Visual Overlay Saved to: {res.annotated_image_path}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="VeriLabel AI - Module 3: Computer Vision & Barcode CLI")
    parser.add_argument("--input", "-i", type=str, help="Path to label image or PDF file")
    parser.add_argument("--batch", "-b", type=str, help="Directory containing label images to batch process")
    parser.add_argument("--output", "-o", type=str, default="cv_output", help="Output directory for visual overlays")
    parser.add_argument("--visualize", "-v", action="store_true", default=True, help="Generate and save visual inspection overlays")
    parser.add_argument("--json", "-j", action="store_true", help="Output results in JSON format")
    parser.add_argument("--no-deskew", action="store_true", help="Disable automated image deskewing")

    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.print_help()
        sys.exit(1)

    cfg = PipelineConfig(
        output_dir=Path(args.output),
        enable_deskew=not args.no_deskew,
        save_annotated_image=args.visualize,
    )
    pipeline = ComputerVisionBarcodePipeline(cfg)

    # Process Single File
    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"Error: Input file does not exist: {in_path}", file=sys.stderr)
            sys.exit(1)

        result = pipeline.process(in_path, save_visuals=args.visualize)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(format_cli_summary(result))

    # Process Batch Directory
    elif args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists() or not batch_dir.is_dir():
            print(f"Error: Batch directory does not exist: {batch_dir}", file=sys.stderr)
            sys.exit(1)

        valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"}
        files = [p for p in batch_dir.iterdir() if p.suffix.lower() in valid_exts]

        print(f"\nProcessing {len(files)} packaging labels from: {batch_dir}...")
        batch_results = []
        for p in files:
            res = pipeline.process(p, save_visuals=args.visualize)
            batch_results.append(res.to_dict())
            status_icon = "PASS" if res.has_valid_udi else "WARN"
            print(f"[{status_icon}] {p.name:<30} | Q: {res.quality.quality_score:.1f} | Barcodes: {len(res.barcodes)} | Symbols: {len(res.symbols)} | {res.processing_latency_ms:.1f}ms")

        if args.json:
            out_json = Path(args.output) / "batch_cv_results.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(batch_results, f, indent=2)
            print(f"\nBatch JSON report exported to: {out_json}")


if __name__ == "__main__":
    main()
