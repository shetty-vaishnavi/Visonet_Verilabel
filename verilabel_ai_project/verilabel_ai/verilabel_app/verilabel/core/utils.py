"""
VeriLabel AI - Common Utilities
Team: Brahmastra | GITAM University
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import cv2


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """Calculate SHA-256 hash of a file for integrity tracking and audit trails."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_safely(image_path: Union[str, Path]) -> np.ndarray:
    """
    Safely load an image from disk, properly handling Windows paths and Unicode.
    Uses numpy fromfile + cv2.imdecode to prevent silent cv2.imread None returns.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    try:
        # Avoid cv2.imread path encoding issues on Windows
        with open(path, "rb") as f:
            bytes_data = f.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image: {image_path}")
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}") from e


def save_image_safely(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """Safely save an image handling Windows paths."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if not ext:
        ext = ".png"
    success, encoded_img = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"Failed to encode image for: {output_path}")
    with open(path, "wb") as f:
        f.write(encoded_img.tobytes())


def parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Parses a wide variety of medical device date formats into a standard datetime object.
    Common formats:
    - 2026-11-30, 2026/11/30
    - 30/11/2026, 30-11-2026
    - 11/30/2026
    - 2026-11 (Year-Month)
    - Nov 2026, November 2026, 30 Nov 2026
    - YYMMDD (GS1 AI 17 / AI 11)
    """
    if not date_str:
        return None

    cleaned = date_str.strip().replace(".", "-").replace("/", "-")

    # Common explicit formats
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%m",
        "%d %b %Y",
        "%d %B %Y",
        "%b %Y",
        "%B %Y",
        "%Y%m%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    # Try GS1 6-digit format YYMMDD (e.g. 261130 -> 2026-11-30)
    digits_only = re.sub(r"\D", "", date_str)
    if len(digits_only) == 6:
        try:
            # 20YY
            yy = int(digits_only[:2])
            mm = int(digits_only[2:4])
            dd = int(digits_only[4:6])
            # If day is 00 in GS1, it means last day of month
            if dd == 0:
                dd = 28 # simplified safe fallback
            year = 2000 + yy if yy < 70 else 1900 + yy
            return datetime(year, mm, min(dd, 28))
        except (ValueError, IndexError):
            pass

    return None


def calculate_gs1_check_digit(digits_str: str) -> int:
    """
    Computes the GS1 Modulo 10 check digit for GTIN-8, GTIN-12, GTIN-13, GTIN-14, SSCC, etc.
    Input must be digits string excluding the final check digit.
    """
    digits = [int(c) for c in digits_str if c.isdigit()]
    if not digits:
        return -1

    # Weights alternate 3 and 1 from right to left
    weights = [3 if i % 2 == 0 else 1 for i in range(len(digits))]
    weights.reverse()

    weighted_sum = sum(d * w for d, w in zip(digits, weights))
    check_digit = (10 - (weighted_sum % 10)) % 10
    return check_digit


def validate_gtin_checksum(gtin: str) -> bool:
    """
    Validates the GS1 Modulo 10 checksum of a GTIN (14, 13, 12, or 8 digits).
    Returns True if valid, False otherwise.
    """
    cleaned = re.sub(r"\D", "", str(gtin))
    if len(cleaned) not in (8, 12, 13, 14):
        return False

    body = cleaned[:-1]
    expected_check = calculate_gs1_check_digit(body)
    actual_check = int(cleaned[-1])
    return expected_check == actual_check


def format_iso_date(dt: Optional[datetime]) -> Optional[str]:
    """Formats datetime as ISO 8601 YYYY-MM-DD."""
    return dt.strftime("%Y-%m-%d") if dt else None
