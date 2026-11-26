"""
Synthetic Medical Device Label Generator for VeriLabel AI Testing & Benchmarking.
Generates test cases matching the High-Level Design (HLD) specifications:
  - TC01: Compliant Label (All mandatory fields present)
  - TC02: Defective Label (Missing Expiry Date)
  - TC03: Defective Label (Missing Safety Warnings & Storage Conditions)
  - TC04: Preprocessing Stress Test (Rotated / Skewed Compliant Label)
"""

import os
from PIL import Image, ImageDraw, ImageFont


def draw_label(
    output_path: str,
    device_name: str = "CARDIO-FLEX VASCULAR BALLOON CATHETER",
    catalog_no: str = "CATH-2024-X8",
    lot_no: str = "LOT20240915-A",
    serial_no: str = "SN-987214-E",
    mfg_date: str = "2024-09-15",
    expiry_date: str = "2027-09-14",
    manufacturer: str = "Brahmastra BioMedical Technologies Inc.",
    mfr_address: str = "100 Innovation Blvd, Suite 400, Cambridge, MA 02142, USA",
    country: str = "Made in USA",
    storage_temp: str = "Store between 15°C and 25°C. Keep dry.",
    warnings: list = None,
    regulatory_tags: list = None,
    udi_text: str = "(01)00850012345678(17)270914(10)LOT20240915A(21)SN987214E",
    rotate_angle: float = 0.0,
):
    if warnings is None:
        warnings = [
            "CAUTION: Federal law restricts this device to sale by or on the order of a physician.",
            "WARNING: Single Use Only. Do not reuse or resterilize.",
            "Do not use if package is opened or damaged.",
        ]
    if regulatory_tags is None:
        regulatory_tags = ["STERILE EO", "RX ONLY", "LATEX FREE", "REF", "LOT"]

    width, height = 1200, 800
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Outer border (Medical device sterile packaging style)
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(0, 0, 0), width=4)
    draw.rectangle([(26, 26), (width - 26, height - 26)], outline=(30, 80, 160), width=2)

    # Use default bitmap font or truetype
    try:
        font_title = ImageFont.truetype("arial.ttf", 32)
        font_sub = ImageFont.truetype("arial.ttf", 22)
        font_body = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_title = font_sub = font_body = font_bold = font_small = ImageFont.load_default()

    # Header: Brand & Device Name
    draw.rectangle([(30, 30), (width - 30, 100)], fill=(240, 245, 255))
    draw.text((45, 45), device_name, fill=(10, 40, 120), font=font_title)

    # Regulatory Tags Row
    tag_x = 45
    for tag in regulatory_tags:
        bbox = draw.textbbox((tag_x, 115), f" {tag} ", font=font_bold)
        draw.rectangle([(bbox[0] - 4, bbox[1] - 2), (bbox[2] + 4, bbox[3] + 2)], outline=(30, 80, 160), width=2)
        draw.text((tag_x, 115), f" {tag} ", fill=(30, 80, 160), font=font_bold)
        tag_x = bbox[2] + 25

    draw.line([(30, 155), (width - 30, 155)], fill=(200, 200, 200), width=2)

    # Left Column: Product Identifiers & Traceability
    y = 175
    draw.text((45, y), f"REF / CATALOG NO: {catalog_no}", fill=(0, 0, 0), font=font_bold)
    y += 38
    draw.text((45, y), f"LOT / BATCH NO: {lot_no}", fill=(0, 0, 0), font=font_bold)
    y += 38
    draw.text((45, y), f"SERIAL NO (SN): {serial_no}", fill=(0, 0, 0), font=font_body)
    y += 38

    if mfg_date:
        draw.text((45, y), f"MFG DATE: {mfg_date}", fill=(0, 0, 0), font=font_body)
        y += 38

    if expiry_date:
        draw.text((45, y), f"EXPIRY DATE (EXP): {expiry_date}", fill=(180, 0, 0), font=font_bold)
        y += 38
    else:
        # Intentionally missing for TC02
        pass

    # Right Column: Storage Conditions & Handling
    rx = 620
    ry = 175
    draw.text((rx, ry), "STORAGE & HANDLING CONDITIONS:", fill=(0, 0, 0), font=font_bold)
    ry += 32
    if storage_temp:
        draw.text((rx, ry), storage_temp, fill=(0, 0, 0), font=font_body)
        ry += 28
        draw.text((rx, ry), "Protect from sunlight and humidity < 75%.", fill=(0, 0, 0), font=font_body)
    else:
        # Intentionally missing for TC03
        pass

    # Middle Separator Line
    draw.line([(30, 360), (width - 30, 360)], fill=(200, 200, 200), width=2)

    # Safety Warnings Section
    wy = 375
    if warnings:
        draw.text((45, wy), "WARNINGS, PRECAUTIONS & CONTRAINDICATIONS:", fill=(180, 0, 0), font=font_bold)
        wy += 30
        for w in warnings:
            draw.text((45, wy), f"• {w}", fill=(40, 40, 40), font=font_body)
            wy += 28
    else:
        # Intentionally missing for TC03
        pass

    # Manufacturer & Country Info
    draw.line([(30, 520), (width - 30, 520)], fill=(200, 200, 200), width=2)
    my = 535
    draw.text((45, my), f"Manufactured by: {manufacturer}", fill=(0, 0, 0), font=font_bold)
    my += 28
    draw.text((45, my), f"Address: {mfr_address}", fill=(50, 50, 50), font=font_body)
    my += 28
    draw.text((45, my), f"Country of Origin: {country}", fill=(50, 50, 50), font=font_body)

    # Bottom Footer: Simulated Barcode & UDI Text
    draw.line([(30, 640), (width - 30, 640)], fill=(200, 200, 200), width=2)

    # Draw pseudo-barcode stripes
    bx = 45
    bar_y = 655
    bar_height = 55
    for i in range(120):
        stripe_w = 2 if (i * 7 + 3) % 5 == 0 else 4 if (i * 3 + 1) % 4 == 0 else 1
        draw.rectangle([(bx, bar_y), (bx + stripe_w, bar_y + bar_height)], fill=(0, 0, 0))
        bx += stripe_w + 3

    # Human-readable UDI string below barcode
    draw.text((45, 725), f"UDI (HRI): {udi_text}", fill=(0, 0, 0), font=font_small)

    # Rotate if requested (for skew testing TC04)
    if rotate_angle != 0.0:
        img = img.rotate(rotate_angle, expand=True, fillcolor=(255, 255, 255))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Generated: {output_path}")


def generate_all_samples(output_dir: str = "sample_labels"):
    os.makedirs(output_dir, exist_ok=True)

    # TC01: Compliant Medical Device Label
    draw_label(
        os.path.join(output_dir, "tc01_compliant_label.png"),
        device_name="CARDIO-FLEX VASCULAR BALLOON CATHETER",
        expiry_date="2027-09-14",
        rotate_angle=0.0,
    )

    # TC02: Defective Label - Missing Expiry Date
    draw_label(
        os.path.join(output_dir, "tc02_missing_expiry.png"),
        device_name="ORTHO-LOCK BONE SCREW SYSTEM",
        catalog_no="ORTH-9021-S",
        lot_no="LOT20241002-B",
        expiry_date="",  # Missing!
        rotate_angle=0.0,
    )

    # TC03: Defective Label - Missing Warnings & Storage Instructions
    draw_label(
        os.path.join(output_dir, "tc03_missing_warning.png"),
        device_name="DERMA-STITCH ABSORBABLE SUTURE",
        catalog_no="SUT-400-P",
        lot_no="LOT20241110-C",
        storage_temp="",  # Missing!
        warnings=[],      # Missing!
        rotate_angle=0.0,
    )

    # TC04: Preprocessing Stress Test - Skewed Compliant Label (rotated by 5.5 degrees)
    draw_label(
        os.path.join(output_dir, "tc04_skewed_compliant.png"),
        device_name="NEURO-PULSE DEEP BRAIN STIMULATOR LEAD",
        catalog_no="NEURO-501-L",
        lot_no="LOT20241105-D",
        expiry_date="2028-11-04",
        rotate_angle=-5.5,  # Skewed
    )


if __name__ == "__main__":
    generate_all_samples()
