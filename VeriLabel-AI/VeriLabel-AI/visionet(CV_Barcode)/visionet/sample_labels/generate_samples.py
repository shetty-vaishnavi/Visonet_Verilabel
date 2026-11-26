"""
VeriLabel AI - Module 3: Synthetic Test Label Generator
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Generates 6 authentic, production-grade synthetic medical device labels covering
HLD Section 11 test scenarios:
- TC01: Compliant Label (Valid GS1 DataMatrix + ISO Symbols)
- TC02: Corrupted UDI Checksum (Invalid GTIN Modulo-10 Check Digit)
- TC03: Multiple Barcodes (2D DataMatrix + 1D Code-128 on single label)
- TC04: Skewed & Noisy Label (Rotated by 12 degrees with sensor noise)
- TC05: Inverted Polarity Label (Photonegative / Metal Laser-Etched DataMatrix)
- TC06: Missing Symbols Label (Barcode present, ISO symbols omitted)
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import zxingcpp

# Resolve paths
SAMPLE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SAMPLE_DIR.parent / "verilabel_cv_barcode" / "templates"


def make_genuine_barcode(text: str, fmt: zxingcpp.BarcodeFormat, scale: int = 4, pad: int = 12) -> np.ndarray:
    """Creates mathematically valid barcode image array with mandatory quiet zone."""
    bc = zxingcpp.create_barcode(text, fmt)
    img = zxingcpp.write_barcode_to_image(bc)
    gray = np.array(img, dtype=np.uint8)
    h, w = gray.shape[:2]
    upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    padded = cv2.copyMakeBorder(upscaled, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def load_template_safe(sym_name: str, size: int = 40) -> np.ndarray:
    """Loads a symbol template and resizes to specified dimensions."""
    p = TEMPLATES_DIR / f"{sym_name}.png"
    if p.exists():
        tpl = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if tpl is not None:
            return cv2.resize(tpl, (size, size), interpolation=cv2.INTER_AREA)
    # Fallback blank with text
    box = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(box, (1, 1), (size - 2, size - 2), (0, 0, 0), 1)
    cv2.putText(box, sym_name[:3].upper(), (4, size // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return box


def paste_overlay(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Pastes patch onto canvas at (x, y) with boundary clamping."""
    h_c, w_c = canvas.shape[:2]
    h_p, w_p = patch.shape[:2]
    x2 = min(x + w_p, w_c)
    y2 = min(y + h_p, h_c)
    w_act = x2 - x
    h_act = y2 - y
    if w_act > 0 and h_act > 0:
        canvas[y:y2, x:x2] = patch[:h_act, :w_act]


def create_base_medical_label(
    product_name: str = "AEROVENT Coronary Stent System",
    catalog_no: str = "REF 8842-A",
    lot_no: str = "LOT 8842",
    mfg_date: str = "MFG 2024-06-15",
    exp_date: str = "EXP 2028-06-30",
    mfr_name: str = "Apex BioMedical Devices Inc.",
    mfr_addr: str = "104 Healthcare Blvd, Boston, MA 02115, USA",
) -> np.ndarray:
    """Draws realistic medical packaging canvas with typography and layout."""
    W, H = 820, 480
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # Outer border and header
    cv2.rectangle(canvas, (10, 10), (W - 10, H - 10), (0, 0, 0), 2)
    cv2.rectangle(canvas, (10, 10), (W - 10, 55), (25, 45, 85), -1)

    # Header text
    cv2.putText(canvas, product_name, (25, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Sub-header / Indications
    cv2.putText(canvas, "Sterile Hydrophilic Delivery Device • Single Use Only", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (20, 95), (W - 20, 95), (200, 200, 200), 1)

    # Product parameters column
    cv2.putText(canvas, catalog_no, (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, lot_no, (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, mfg_date, (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, exp_date, (25, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Storage: Store between 15C and 30C", (25, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    # Manufacturer block
    cv2.putText(canvas, mfr_name, (25, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, mfr_addr, (25, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

    return canvas


def generate_all_samples():
    """Generates all 6 sample labels and saves to sample_labels/ directory."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Generating Synthetic Medical Device Labels (HLD TC01 - TC06)...")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TC01: Fully Compliant Label
    # -------------------------------------------------------------------------
    tc01 = create_base_medical_label()
    udi_tc01 = "(01)00847239012349(17)280630(10)LOT8842"
    bc01 = make_genuine_barcode(udi_tc01, zxingcpp.BarcodeFormat.DataMatrix, scale=4, pad=12)
    paste_overlay(tc01, bc01, 490, 105)
    cv2.putText(tc01, udi_tc01, (430, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1)

    # Paste ISO symbols
    symbols_01 = [
        ("sterile", 280, 115),
        ("do_not_reuse", 340, 115),
        ("caution", 400, 115),
        ("temperature_limit", 280, 175),
        ("manufacturer", 340, 175),
        ("use_by_date", 400, 175),
        ("ce_mark", 280, 235),
        ("rx_only", 340, 235),
    ]
    for sym_name, sx, sy in symbols_01:
        tpl = load_template_safe(sym_name, size=46)
        paste_overlay(tc01, tpl, sx, sy)

    p01 = SAMPLE_DIR / "tc01_compliant_label.png"
    cv2.imwrite(str(p01), tc01)
    print(f"[OK] Generated: {p01.name}")

    # -------------------------------------------------------------------------
    # TC02: Corrupted Barcode Checksum (GTIN check digit corrupted from 9 to 8)
    # -------------------------------------------------------------------------
    tc02 = create_base_medical_label()
    udi_tc02 = "(01)00847239012348(17)280630(10)LOT8842"  # Invalid Modulo-10
    bc02 = make_genuine_barcode(udi_tc02, zxingcpp.BarcodeFormat.DataMatrix, scale=4, pad=12)
    paste_overlay(tc02, bc02, 490, 105)
    cv2.putText(tc02, udi_tc02, (430, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1)

    for sym_name, sx, sy in symbols_01:
        tpl = load_template_safe(sym_name, size=46)
        paste_overlay(tc02, tpl, sx, sy)

    p02 = SAMPLE_DIR / "tc02_invalid_checksum.png"
    cv2.imwrite(str(p02), tc02)
    print(f"[OK] Generated: {p02.name}")

    # -------------------------------------------------------------------------
    # TC03: Multiple Barcodes (2D DataMatrix + 1D Code-128 linear barcode)
    # -------------------------------------------------------------------------
    tc03 = create_base_medical_label()
    # DataMatrix
    paste_overlay(tc03, bc01, 490, 105)
    # 1D Code-128 barcode at bottom right - scale=3 and pad=15 for reliable decode
    code128_data = "REF8842-A-LOT8842"
    bc_128 = make_genuine_barcode(code128_data, zxingcpp.BarcodeFormat.Code128, scale=3, pad=15)
    paste_overlay(tc03, bc_128, 25, 290)
    cv2.putText(tc03, f"Linear: {code128_data}", (25, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    for sym_name, sx, sy in symbols_01[:5]:
        tpl = load_template_safe(sym_name, size=46)
        paste_overlay(tc03, tpl, sx, sy)

    p03 = SAMPLE_DIR / "tc03_multiple_barcodes.png"
    cv2.imwrite(str(p03), tc03)
    print(f"[OK] Generated: {p03.name}")

    # -------------------------------------------------------------------------
    # TC04: Skewed & Noisy Packaging Label (Rotated 12° with sensor noise)
    # -------------------------------------------------------------------------
    tc04_clean = tc01.copy()
    h, w = tc04_clean.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 12.0, 1.0)
    tc04_skewed = cv2.warpAffine(tc04_clean, M, (w, h), borderValue=(240, 240, 240))
    # Add subtle gaussian noise
    noise = np.random.normal(0, 10, tc04_skewed.shape).astype(np.int16)
    tc04_noisy = np.clip(tc04_skewed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    p04 = SAMPLE_DIR / "tc04_skewed_and_noisy.png"
    cv2.imwrite(str(p04), tc04_noisy)
    print(f"[OK] Generated: {p04.name}")

    # -------------------------------------------------------------------------
    # TC05: Inverted Polarity Label (Photonegative / Metal Laser Etched)
    # -------------------------------------------------------------------------
    tc05 = cv2.bitwise_not(tc01)
    p05 = SAMPLE_DIR / "tc05_inverted_datamatrix.png"
    cv2.imwrite(str(p05), tc05)
    print(f"[OK] Generated: {p05.name}")

    # -------------------------------------------------------------------------
    # TC06: Missing Symbols Label (Barcode present, symbols omitted)
    # -------------------------------------------------------------------------
    tc06 = create_base_medical_label()
    paste_overlay(tc06, bc01, 490, 105)
    cv2.putText(tc06, udi_tc01, (430, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 30, 30), 1)
    # Notice: NO symbols pasted! Demonstrates symbol absence detection.

    p06 = SAMPLE_DIR / "tc06_missing_symbols.png"
    cv2.imwrite(str(p06), tc06)
    print(f"[OK] Generated: {p06.name}")

    print("=" * 70)
    print(f"All 6 sample test labels successfully generated in: {SAMPLE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_samples()
