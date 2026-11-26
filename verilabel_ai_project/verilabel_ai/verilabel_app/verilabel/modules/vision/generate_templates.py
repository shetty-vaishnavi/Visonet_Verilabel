"""
VeriLabel AI - ISO 15223-1 Reference Symbol Template Generator
Generates clean, standardized medical device symbol templates for template matching.
"""

from pathlib import Path
import numpy as np
import cv2

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def create_base_canvas(size: int = 80) -> np.ndarray:
    """Creates a white square canvas."""
    return np.full((size, size, 3), 255, dtype=np.uint8)


def generate_lot_template() -> np.ndarray:
    """ISO 15223-1 5.1.5: Batch code [LOT]."""
    img = create_base_canvas()
    cv2.rectangle(img, (8, 15), (72, 65), (0, 0, 0), 3)
    cv2.putText(img, "LOT", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_ref_template() -> np.ndarray:
    """ISO 15223-1 5.1.6: Catalogue / Model number [REF]."""
    img = create_base_canvas()
    cv2.rectangle(img, (8, 15), (72, 65), (0, 0, 0), 3)
    cv2.putText(img, "REF", (13, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_sn_template() -> np.ndarray:
    """ISO 15223-1 5.1.7: Serial number [SN]."""
    img = create_base_canvas()
    cv2.rectangle(img, (12, 15), (68, 65), (0, 0, 0), 3)
    cv2.putText(img, "SN", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_single_use_template() -> np.ndarray:
    """ISO 15223-1 5.4.2: Do not re-use (Single use). Number 2 in circle with slash."""
    img = create_base_canvas()
    center = (40, 40)
    radius = 30
    cv2.circle(img, center, radius, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "2", (26, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (18, 62), (62, 18), (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_caution_template() -> np.ndarray:
    """ISO 15223-1 5.4.4: Caution (Triangle with exclamation mark)."""
    img = create_base_canvas()
    pts = np.array([[40, 12], [10, 68], [70, 68]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    # Exclamation mark
    cv2.line(img, (40, 28), (40, 50), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (40, 58), 2, (0, 0, 0), -1)
    return img


def generate_use_by_template() -> np.ndarray:
    """ISO 15223-1 5.1.4: Use-by date (Hourglass symbol)."""
    img = create_base_canvas()
    pts_top = np.array([[16, 16], [64, 16], [40, 40]], np.int32)
    pts_bottom = np.array([[40, 40], [64, 64], [16, 64]], np.int32)
    cv2.polylines(img, [pts_top], isClosed=True, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts_bottom], isClosed=True, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.line(img, (14, 16), (66, 16), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (14, 64), (66, 64), (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_manufacturer_template() -> np.ndarray:
    """ISO 15223-1 5.1.1: Manufacturer (Factory symbol)."""
    img = create_base_canvas()
    # Factory roof ridges
    pts = np.array([
        [15, 65], [15, 40], [25, 40], [25, 30], [35, 40],
        [35, 30], [45, 40], [45, 30], [55, 40], [65, 40],
        [65, 65]
    ], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.line(img, (15, 65), (65, 65), (0, 0, 0), 3, cv2.LINE_AA)
    # Chimney / door
    cv2.rectangle(img, (35, 48), (45, 65), (0, 0, 0), -1)
    return img


def generate_ce_mark_template() -> np.ndarray:
    """CE Conformity Marking."""
    img = create_base_canvas()
    cv2.putText(img, "CE", (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 4, cv2.LINE_AA)
    return img


def generate_sterile_template() -> np.ndarray:
    """ISO 15223-1 5.2.1: Sterile."""
    img = create_base_canvas(size=80)
    cv2.rectangle(img, (6, 20), (74, 60), (0, 0, 0), 3)
    cv2.putText(img, "STERILE", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def generate_consult_ifu_template() -> np.ndarray:
    """ISO 15223-1 5.4.3: Consult instructions for use (Book with 'i')."""
    img = create_base_canvas()
    # Open book outline
    cv2.rectangle(img, (15, 18), (65, 62), (0, 0, 0), 3)
    cv2.putText(img, "i", (34, 50), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_temperature_limit_template() -> np.ndarray:
    """ISO 15223-1 5.3.7: Temperature limit (Thermometer)."""
    img = create_base_canvas()
    # Thermometer stem
    cv2.line(img, (40, 20), (40, 52), (0, 0, 0), 4, cv2.LINE_AA)
    # Bulb
    cv2.circle(img, (40, 58), 9, (0, 0, 0), -1)
    # Upper/lower limit arrows
    cv2.line(img, (24, 22), (32, 22), (0, 0, 0), 2)
    cv2.line(img, (24, 52), (32, 52), (0, 0, 0), 2)
    return img


def generate_rx_only_template() -> np.ndarray:
    """Prescription only symbol (Rx)."""
    img = create_base_canvas()
    cv2.putText(img, "Rx", (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def generate_all_templates() -> dict:
    """Generates and writes all reference symbol templates to disk."""
    templates = {
        "batch_code": generate_lot_template(),
        "catalogue_number": generate_ref_template(),
        "serial_number": generate_sn_template(),
        "do_not_reuse": generate_single_use_template(),
        "caution": generate_caution_template(),
        "use_by_date": generate_use_by_template(),
        "manufacturer": generate_manufacturer_template(),
        "ce_mark": generate_ce_mark_template(),
        "sterile": generate_sterile_template(),
        "consult_ifu": generate_consult_ifu_template(),
        "temperature_limit": generate_temperature_limit_template(),
        "rx_only": generate_rx_only_template()
    }

    for name, img in templates.items():
        file_path = TEMPLATE_DIR / f"{name}.png"
        cv2.imwrite(str(file_path), img)

    return templates


if __name__ == "__main__":
    generate_all_templates()
    print("All ISO 15223-1 symbol templates successfully generated.")
