"""
VeriLabel AI - Module 4: UDI (Unique Device Identification) Validator
Team: Brahmastra | GITAM University
Implements FDA 21 CFR 830 and EU MDR 2017/745 UDI validation standards.
Parses GS1 Application Identifiers (AI) and verifies GS1 Modulo-10 checksums.
"""

import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple

from verilabel.core.utils import validate_gtin_checksum, parse_date_string, format_iso_date
from verilabel.modules.ocr.field_extractor import ExtractedLabelFields


@dataclass
class UDIComponents:
    """Structured elements of a parsed Unique Device Identifier."""
    raw_carrier: str
    standard: str = "GS1"  # "GS1", "HIBCC", or "UNKNOWN"
    device_identifier_di: Optional[str] = None
    is_di_checksum_valid: bool = False
    expiry_date_pi: Optional[str] = None
    lot_number_pi: Optional[str] = None
    serial_number_pi: Optional[str] = None
    manufacturing_date_pi: Optional[str] = None
    is_valid_format: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UDIValidator:
    """
    Validates Medical Device UDI formats according to FDA and EU MDR regulations.
    Parses both Human Readable Interpretation (HRI) format: (01)...(17)...(10)...
    and machine barcode data strings: 01008472390123451726113010BATCH123
    """

    def parse_udi(self, raw_code: str) -> UDIComponents:
        """
        Parses raw barcode or text into UDIComponents and validates structure and checksum.
        """
        cleaned = raw_code.strip()
        result = UDIComponents(raw_carrier=cleaned)

        if not cleaned:
            result.validation_errors.append("Empty UDI carrier string.")
            return result

        # Check if HIBCC format (starts with '+')
        if cleaned.startswith("+"):
            return self._parse_hibcc(cleaned)

        # Default: Parse as GS1 format
        return self._parse_gs1(cleaned)

    def _parse_gs1(self, text: str) -> UDIComponents:
        """Parses GS1 AI formatted UDI string."""
        comp = UDIComponents(raw_carrier=text, standard="GS1")

        # 1. Check parenthesized format, e.g. (01)00847239012345(17)261130(10)LOT123
        if "(" in text and ")" in text:
            ai_matches = re.findall(r"\(([0-9]{2,4})\)([^\(\)]+)", text)
            for ai, val in ai_matches:
                val = val.strip()
                if ai == "01":  # GTIN / DI (14 digits)
                    comp.device_identifier_di = val
                elif ai == "17":  # Expiry date (YYMMDD)
                    comp.expiry_date_pi = val
                elif ai == "10":  # Batch / Lot
                    comp.lot_number_pi = val
                elif ai == "21":  # Serial number
                    comp.serial_number_pi = val
                elif ai == "11":  # Production date
                    comp.manufacturing_date_pi = val
        else:
            # Unparenthesized continuous GS1 barcode stream
            # (01) is 14 digits fixed length
            if text.startswith("01") and len(text) >= 16:
                comp.device_identifier_di = text[2:16]
                remainder = text[16:]
                # Scan remaining AIs
                idx = 0
                while idx < len(remainder):
                    ai2 = remainder[idx:idx+2]
                    if ai2 == "17" and len(remainder) >= idx + 8:
                        comp.expiry_date_pi = remainder[idx+2:idx+8]
                        idx += 8
                    elif ai2 == "11" and len(remainder) >= idx + 8:
                        comp.manufacturing_date_pi = remainder[idx+2:idx+8]
                        idx += 8
                    elif ai2 == "10":
                        # Lot is variable length up to 20 chars
                        comp.lot_number_pi = remainder[idx+2:idx+22]
                        break
                    elif ai2 == "21":
                        # Serial is variable length up to 20 chars
                        comp.serial_number_pi = remainder[idx+2:idx+22]
                        break
                    else:
                        idx += 1
            elif len(text) in (12, 13, 14) and text.isdigit():
                # Direct GTIN / DI barcode
                comp.device_identifier_di = text

        # Validate Device Identifier (DI)
        if comp.device_identifier_di:
            di_digits = re.sub(r"\D", "", comp.device_identifier_di)
            if len(di_digits) in (8, 12, 13, 14):
                if validate_gtin_checksum(di_digits):
                    comp.is_di_checksum_valid = True
                else:
                    comp.validation_errors.append(
                        f"GTIN Checksum failed for DI '{comp.device_identifier_di}'. Check digit is invalid."
                    )
            else:
                comp.validation_errors.append(
                    f"Invalid DI length ({len(di_digits)} digits). Expected 14, 13, or 12 digits."
                )
        else:
            comp.validation_errors.append("Missing mandatory Device Identifier (AI 01 / DI).")

        # Validate Expiry Date PI
        if comp.expiry_date_pi:
            parsed_exp = parse_date_string(comp.expiry_date_pi)
            if not parsed_exp:
                comp.validation_errors.append(
                    f"Invalid UDI Expiry Date format: '{comp.expiry_date_pi}'. Expected YYMMDD."
                )
            else:
                comp.expiry_date_pi = format_iso_date(parsed_exp)

        # Overall format validity
        comp.is_valid_format = bool(comp.device_identifier_di and comp.is_di_checksum_valid and not comp.validation_errors)
        return comp

    def _parse_hibcc(self, text: str) -> UDIComponents:
        """Parses Health Industry Business Communications Council (HIBCC) UDI string."""
        comp = UDIComponents(raw_carrier=text, standard="HIBCC")
        # Format: +$$$######...
        if len(text) >= 6:
            comp.device_identifier_di = text[:10]
            comp.is_di_checksum_valid = True
            comp.is_valid_format = True
        else:
            comp.validation_errors.append("Invalid HIBCC UDI string length.")
        return comp

    def cross_validate_with_ocr(
        self,
        udi: UDIComponents,
        ocr_fields: ExtractedLabelFields
    ) -> List[Dict[str, Any]]:
        """
        Cross-validates information between UDI Barcode and Human Readable OCR Text.
        Discrepancies indicate critical medical device labeling errors.
        """
        discrepancies = []

        # 1. Lot / Batch cross-verification
        if udi.lot_number_pi and ocr_fields.lot_number:
            clean_udi_lot = re.sub(r"[\s\-]", "", udi.lot_number_pi).upper()
            clean_ocr_lot = re.sub(r"[\s\-]", "", ocr_fields.lot_number).upper()
            if clean_udi_lot != clean_ocr_lot:
                discrepancies.append({
                    "field": "lot_number",
                    "severity": "CRITICAL",
                    "message": f"Discrepancy: UDI barcode LOT '{udi.lot_number_pi}' does not match label text LOT '{ocr_fields.lot_number}'."
                })

        # 2. Expiry date cross-verification
        if udi.expiry_date_pi and ocr_fields.expiry_date:
            udi_dt = parse_date_string(udi.expiry_date_pi)
            ocr_dt = parse_date_string(ocr_fields.expiry_date)
            if udi_dt and ocr_dt:
                # Compare year and month
                if (udi_dt.year, udi_dt.month) != (ocr_dt.year, ocr_dt.month):
                    discrepancies.append({
                        "field": "expiry_date",
                        "severity": "CRITICAL",
                        "message": f"Discrepancy: UDI barcode Expiry '{udi.expiry_date_pi}' does not match label text Expiry '{ocr_fields.expiry_date}'."
                    })

        return discrepancies
