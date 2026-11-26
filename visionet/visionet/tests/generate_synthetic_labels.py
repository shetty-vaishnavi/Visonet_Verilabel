"""
VeriLabel AI - Synthetic Medical Device Label Generator
Generates realistic medical device label images and ground-truth metadata
for the 6 standard test cases defined in HLD Section 11:
- TC01: All mandatory fields available (Compliant)
- TC02: Expiry date is missing (Missing field detected)
- TC03: UDI format is incorrect / corrupted checksum (Format error detected)
- TC04: Required symbol is missing (Symbol warning generated)
- TC05: Warning information is missing (Missing warning detected)
- TC06: Valid barcode is present (Barcode extracted and checked)
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import List, Tuple, Optional
import numpy as np
import cv2

from verilabel.config import SAMPLE_DATA_DIR, SYMBOLS_DIR
from verilabel.core.utils import calculate_file_hash

OUTPUT_DIR = SAMPLE_DATA_DIR / "labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_GTIN_14 = "00847239012349"  # Checksum 9 is mathematically valid
INVALID_GTIN_14 = "00847239012345"  # Checksum 5 is corrupted


def load_symbol_template(symbol_name: str) -> Optional[np.ndarray]:
    """Loads a reference symbol template image."""
    path = SYMBOLS_DIR / f"{symbol_name}.png"
    if path.exists():
        return cv2.imread(str(path))
    return None


def overlay_symbol(
    canvas: np.ndarray,
    symbol_name: str,
    pos: Tuple[int, int],
    size: Tuple[int, int] = (52, 52)
):
    """Pastes an ISO symbol template onto canvas preserving square aspect ratio."""
    sym = load_symbol_template(symbol_name)
    if sym is None:
        return

    resized = cv2.resize(sym, size, interpolation=cv2.INTER_AREA)
    x, y = pos
    w, h = size
    if y + h <= canvas.shape[0] and x + w <= canvas.shape[1]:
        canvas[y:y+h, x:x+w] = resized


def generate_qr_code(payload: str, size: int = 150) -> np.ndarray:
    """Generates a QR code image with quiet border."""
    encoder = cv2.QRCodeEncoder_create()
    raw = encoder.encode(payload)
    scaled = cv2.resize(raw, (size - 30, size - 30), interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(scaled, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def draw_label_base(width: int = 850, height: int = 650) -> np.ndarray:
    """Creates a realistic medical packaging label background."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # Outer border
    cv2.rectangle(canvas, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)

    # Header blue accent bar
    cv2.rectangle(canvas, (12, 12), (width - 12, 70), (220, 240, 255), -1)
    cv2.line(canvas, (10, 70), (width - 10, 70), (0, 100, 180), 2)

    # Footer separator
    cv2.line(canvas, (10, height - 85), (width - 10, height - 85), (200, 200, 200), 1)

    return canvas


def create_tc01_compliant() -> Tuple[Path, Path]:
    """TC01: Fully compliant medical device label."""
    img = draw_label_base()
    height, width = img.shape[:2]

    # Product Title
    cv2.putText(img, "CardioPulse Catheter System 6Fr", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 50, 120), 2, cv2.LINE_AA)
    cv2.putText(img, "Sterile Intravascular Delivery Catheter", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)

    # Manufacturer & Address
    overlay_symbol(img, "manufacturer", (25, 120), (52, 52))
    cv2.putText(img, "Manufactured by: Brahmastra MedTech Corp.", (85, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "100 Innovation Way, Bengaluru, Karnataka, India", (85, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

    # Traceability Column 1: REF & LOT
    overlay_symbol(img, "catalogue_number", (25, 185), (52, 52))
    cv2.putText(img, "REF 90210-A", (85, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "batch_code", (25, 245), (52, 52))
    cv2.putText(img, "LOT BATCH-8842", (85, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Dates: MFG & EXP
    cv2.putText(img, "MFG 2026-01-15", (25, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "use_by_date", (25, 335), (52, 52))
    cv2.putText(img, "EXP 2028-06-30", (85, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Sterility & Single Use
    overlay_symbol(img, "sterile", (25, 400), (52, 52))
    overlay_symbol(img, "do_not_reuse", (90, 400), (52, 52))
    overlay_symbol(img, "temperature_limit", (155, 400), (52, 52))
    cv2.putText(img, "Store between 2 C and 25 C. Keep dry.", (25, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Warnings
    cv2.putText(img, "Caution: Federal law restricts this device to sale by or on order of physician.", (25, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Sterile unless package is opened or damaged. Single use only.", (25, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Consult instructions for use before operation.", (25, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # CE Mark & Rx
    overlay_symbol(img, "ce_mark", (25, height - 75), (52, 52))
    cv2.putText(img, "CE 0123", (90, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    overlay_symbol(img, "rx_only", (200, height - 75), (52, 52))

    # UDI QR Code (Right side)
    udi_payload = f"(01){VALID_GTIN_14}(17)280630(10)BATCH-8842"
    qr_img = generate_qr_code(udi_payload, size=150)
    qr_x, qr_y = 650, 160
    img[qr_y:qr_y+150, qr_x:qr_x+150] = qr_img
    cv2.putText(img, "UDI Carrier", (685, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(img, f"(01){VALID_GTIN_14}", (615, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "(17)280630(10)BATCH-8842", (615, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    img_path = OUTPUT_DIR / "TC01_compliant.png"
    txt_path = OUTPUT_DIR / "TC01_compliant.txt"
    cv2.imwrite(str(img_path), img)

    raw_text = (
        "CardioPulse Catheter System 6Fr\n"
        "Sterile Intravascular Delivery Catheter\n"
        "Manufactured by: Brahmastra MedTech Corp.\n"
        "100 Innovation Way, Bengaluru, Karnataka, India\n"
        "REF 90210-A\n"
        "LOT BATCH-8842\n"
        "MFG 2026-01-15\n"
        "EXP 2028-06-30\n"
        "STERILE EO\n"
        "Store between 2°C and 25°C. Keep dry.\n"
        "Caution: Federal law restricts this device to sale by or on order of physician. Rx Only.\n"
        "Sterile unless package is opened or damaged. Single use only. Do not reuse.\n"
        "Consult instructions for use before operation.\n"
        "CE 0123\n"
        f"(01){VALID_GTIN_14}(17)280630(10)BATCH-8842\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    return img_path, txt_path


def create_tc02_missing_expiry() -> Tuple[Path, Path]:
    """TC02: Expiry date is missing (Defect introduced)."""
    img = draw_label_base()
    height, width = img.shape[:2]

    cv2.putText(img, "CardioPulse Catheter System 6Fr", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 50, 120), 2, cv2.LINE_AA)
    overlay_symbol(img, "manufacturer", (25, 120), (52, 52))
    cv2.putText(img, "Manufactured by: Brahmastra MedTech Corp.", (85, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "catalogue_number", (25, 185), (52, 52))
    cv2.putText(img, "REF 90210-A", (85, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "batch_code", (25, 245), (52, 52))
    cv2.putText(img, "LOT BATCH-8842", (85, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(img, "MFG 2026-01-15", (25, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    # NOTICE: Expiry date text and hourglass symbol omitted!

    overlay_symbol(img, "sterile", (25, 400), (52, 52))
    overlay_symbol(img, "do_not_reuse", (90, 400), (52, 52))
    cv2.putText(img, "Store between 2 C and 25 C. Keep dry.", (25, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Caution: Federal law restricts this device. Rx Only.", (25, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "ce_mark", (25, height - 75), (52, 52))
    cv2.putText(img, "CE 0123", (90, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # UDI without expiry
    udi_payload = f"(01){VALID_GTIN_14}(10)BATCH-8842"
    qr_img = generate_qr_code(udi_payload, size=150)
    img[160:310, 650:800] = qr_img

    img_path = OUTPUT_DIR / "TC02_missing_expiry.png"
    txt_path = OUTPUT_DIR / "TC02_missing_expiry.txt"
    cv2.imwrite(str(img_path), img)

    raw_text = (
        "CardioPulse Catheter System 6Fr\n"
        "Manufactured by: Brahmastra MedTech Corp.\n"
        "REF 90210-A\n"
        "LOT BATCH-8842\n"
        "MFG 2026-01-15\n"
        "STERILE EO\n"
        "Store between 2°C and 25°C. Keep dry.\n"
        "Caution: Federal law restricts this device. Rx Only.\n"
        "Single use only. Do not reuse.\n"
        "CE 0123\n"
        f"(01){VALID_GTIN_14}(10)BATCH-8842\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    return img_path, txt_path


def create_tc03_invalid_udi() -> Tuple[Path, Path]:
    """TC03: UDI format has invalid GTIN check digit (Defect introduced)."""
    img = draw_label_base()
    height, width = img.shape[:2]

    cv2.putText(img, "CardioPulse Catheter System 6Fr", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 50, 120), 2, cv2.LINE_AA)
    overlay_symbol(img, "manufacturer", (25, 120), (52, 52))
    cv2.putText(img, "Manufactured by: Brahmastra MedTech Corp.", (85, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "batch_code", (25, 245), (52, 52))
    cv2.putText(img, "LOT BATCH-8842", (85, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "use_by_date", (25, 335), (52, 52))
    cv2.putText(img, "EXP 2028-06-30", (85, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "sterile", (25, 400), (52, 52))
    overlay_symbol(img, "do_not_reuse", (90, 400), (52, 52))
    cv2.putText(img, "Store between 2 C and 25 C. Keep dry.", (25, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Caution: Federal law restricts this device. Rx Only.", (25, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    overlay_symbol(img, "ce_mark", (25, height - 75), (52, 52))
    cv2.putText(img, "CE 0123", (90, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # INVALID UDI: GTIN ending in '5' instead of valid '9'
    invalid_udi = f"(01){INVALID_GTIN_14}(17)280630(10)BATCH-8842"
    qr_img = generate_qr_code(invalid_udi, size=150)
    img[160:310, 650:800] = qr_img
    cv2.putText(img, "INVALID UDI CHECKSUM", (610, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)

    img_path = OUTPUT_DIR / "TC03_invalid_udi.png"
    txt_path = OUTPUT_DIR / "TC03_invalid_udi.txt"
    cv2.imwrite(str(img_path), img)

    raw_text = (
        "CardioPulse Catheter System 6Fr\n"
        "Manufactured by: Brahmastra MedTech Corp.\n"
        "REF 90210-A\n"
        "LOT BATCH-8842\n"
        "MFG 2026-01-15\n"
        "EXP 2028-06-30\n"
        "STERILE EO\n"
        "Store between 2°C and 25°C. Keep dry.\n"
        "Caution: Federal law restricts this device. Rx Only.\n"
        "Single use only. Do not reuse.\n"
        "CE 0123\n"
        f"(01){INVALID_GTIN_14}(17)280630(10)BATCH-8842\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    return img_path, txt_path


def create_tc04_missing_symbol() -> Tuple[Path, Path]:
    """TC04: Sterile text claim present, but ISO Sterile symbol missing."""
    img = draw_label_base()
    height, width = img.shape[:2]

    cv2.putText(img, "CardioPulse Catheter System 6Fr", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 50, 120), 2, cv2.LINE_AA)
    overlay_symbol(img, "manufacturer", (25, 120), (52, 52))
    cv2.putText(img, "Manufactured by: Brahmastra MedTech Corp.", (85, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "batch_code", (25, 245), (52, 52))
    cv2.putText(img, "LOT BATCH-8842", (85, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "use_by_date", (25, 335), (52, 52))
    cv2.putText(img, "EXP 2028-06-30", (85, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # NOTICE: Only 'do_not_reuse' symbol is placed; 'sterile' symbol is missing!
    overlay_symbol(img, "do_not_reuse", (25, 400), (52, 52))
    cv2.putText(img, "Store between 2 C and 25 C. Keep dry.", (25, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Caution: Federal law restricts this device. Rx Only.", (25, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "ce_mark", (25, height - 75), (52, 52))
    cv2.putText(img, "CE 0123", (90, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    udi_payload = f"(01){VALID_GTIN_14}(17)280630(10)BATCH-8842"
    qr_img = generate_qr_code(udi_payload, size=150)
    img[160:310, 650:800] = qr_img

    img_path = OUTPUT_DIR / "TC04_missing_symbol.png"
    txt_path = OUTPUT_DIR / "TC04_missing_symbol.txt"
    cv2.imwrite(str(img_path), img)

    raw_text = (
        "CardioPulse Catheter System 6Fr\n"
        "Manufactured by: Brahmastra MedTech Corp.\n"
        "REF 90210-A\n"
        "LOT BATCH-8842\n"
        "MFG 2026-01-15\n"
        "EXP 2028-06-30\n"
        "STERILE EO\n"
        "Store between 2°C and 25°C. Keep dry.\n"
        "Caution: Federal law restricts this device. Rx Only.\n"
        "Single use only. Do not reuse.\n"
        "CE 0123\n"
        f"(01){VALID_GTIN_14}(17)280630(10)BATCH-8842\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    return img_path, txt_path


def create_tc05_missing_warnings() -> Tuple[Path, Path]:
    """TC05: Essential safety warnings and cautions omitted."""
    img = draw_label_base()
    height, width = img.shape[:2]

    cv2.putText(img, "CardioPulse Catheter System 6Fr", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 50, 120), 2, cv2.LINE_AA)
    overlay_symbol(img, "manufacturer", (25, 120), (52, 52))
    cv2.putText(img, "Manufactured by: Brahmastra MedTech Corp.", (85, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "batch_code", (25, 245), (52, 52))
    cv2.putText(img, "LOT BATCH-8842", (85, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    overlay_symbol(img, "use_by_date", (25, 335), (52, 52))
    cv2.putText(img, "EXP 2028-06-30", (85, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # NOTICE: All warnings and caution statements omitted!
    cv2.putText(img, "Store between 2 C and 25 C. Keep dry.", (25, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    overlay_symbol(img, "ce_mark", (25, height - 75), (52, 52))
    cv2.putText(img, "CE 0123", (90, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    udi_payload = f"(01){VALID_GTIN_14}(17)280630(10)BATCH-8842"
    qr_img = generate_qr_code(udi_payload, size=150)
    img[160:310, 650:800] = qr_img

    img_path = OUTPUT_DIR / "TC05_missing_warnings.png"
    txt_path = OUTPUT_DIR / "TC05_missing_warnings.txt"
    cv2.imwrite(str(img_path), img)

    raw_text = (
        "CardioPulse Catheter System 6Fr\n"
        "Manufactured by: Brahmastra MedTech Corp.\n"
        "REF 90210-A\n"
        "LOT BATCH-8842\n"
        "MFG 2026-01-15\n"
        "EXP 2028-06-30\n"
        "Store between 2°C and 25°C. Keep dry.\n"
        "CE 0123\n"
        f"(01){VALID_GTIN_14}(17)280630(10)BATCH-8842\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    return img_path, txt_path


def generate_all_test_cases():
    """Generates synthetic labels and sidecar text for all 6 test cases."""
    print("Generating synthetic labels for TC01 to TC06...")
    create_tc01_compliant()
    create_tc02_missing_expiry()
    create_tc03_invalid_udi()
    create_tc04_missing_symbol()
    create_tc05_missing_warnings()
    print("Successfully generated all synthetic labels in:", OUTPUT_DIR)


if __name__ == "__main__":
    generate_all_test_cases()
