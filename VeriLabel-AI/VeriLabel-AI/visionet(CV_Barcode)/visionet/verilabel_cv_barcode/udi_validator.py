"""
VeriLabel AI - Module 3: UDI (Unique Device Identification) Validator
Team: Brahmastra | GITAM University
Role: Member 3 - Computer Vision and Barcode Module

Implements FDA 21 CFR Part 830, EU MDR 2017/745 Article 27, and ISO 15223-1 UDI regulatory standards.
Supports:
  1. GS1 Application Identifiers (AI) in Human-Readable Interpretation (HRI) and machine streams:
     • (01) GTIN / Device Identifier (DI)
     • (17) Expiration Date (PI)
     • (11) Production / Manufacturing Date (PI)
     • (10) Batch / Lot Number (PI)
     • (21) Serial Number (PI)
     • (240) Additional Product Identification (PI)
  2. Health Industry Business Communications Council (HIBCC) formats:
     • Primary Data Structure (DI): LIC + Product Code + Unit of Measure + Modulo-43 Check Character
     • Secondary Data Structure (PI): Expiration date, lot/batch, serial number
     • Concatenated Primary + Secondary formats
Performs:
  • Mathematical GS1 Modulo-10 checksum validation on GTIN-14, GTIN-13, GTIN-12, GTIN-8
  • Mathematical HIBCC Modulo-43 check character validation
  • Calendar date validation (YYMMDD format, day 00 month-end rule, elapsed date detection)
  • Bidirectional OCR vs. Barcode cross-verification under FDA 21 CFR 830.20 and EU MDR 23.2(h)
"""

import re
import calendar
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from .models import UDIComponents, CrossValidationResult


# HIBCC Modulo-43 standard alphanumeric character set
HIBCC_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"
HIBCC_CHAR_MAP = {c: i for i, c in enumerate(HIBCC_CHARSET)}


def validate_gtin_checksum(gtin: str) -> bool:
    """
    Validates GS1 GTIN-14, GTIN-13, GTIN-12, or GTIN-8 check digit using official GS1 Modulo-10:
    Weights alternate 3, 1, 3, 1... moving leftwards from the position before the check digit.
    Formula:
        Sum = Σ (d_i * weight_i)
        Check_Digit = (10 - (Sum % 10)) % 10
    """
    cleaned = re.sub(r"\D", "", gtin.strip())
    if len(cleaned) not in (8, 12, 13, 14):
        return False

    digits = [int(c) for c in cleaned]
    expected_check = digits[-1]
    core_digits = digits[:-1]

    # Moving right to left, weights are alternating 3, 1, 3, 1...
    total = 0
    weight = 3
    for d in reversed(core_digits):
        total += d * weight
        weight = 1 if weight == 3 else 3

    calculated_check = (10 - (total % 10)) % 10
    return bool(calculated_check == expected_check)


def validate_hibcc_modulo43(hibcc_str: str) -> bool:
    """
    Validates HIBCC Modulo-43 check character.
    The last character is the check character, computed from all preceding characters.
    """
    cleaned = hibcc_str.strip()
    if len(cleaned) < 3:
        return False

    # Strip leading '+' if present
    content = cleaned[1:] if cleaned.startswith("+") else cleaned
    core_text = content[:-1].upper()
    expected_char = content[-1].upper()

    total = 0
    for char in core_text:
        if char not in HIBCC_CHAR_MAP:
            return False
        total += HIBCC_CHAR_MAP[char]

    calculated_char = HIBCC_CHARSET[total % 43]
    return bool(calculated_char == expected_char)


class UDIValidator:
    """
    Validates Medical Device UDI formats according to FDA and EU MDR regulations.
    Parses both Human Readable Interpretation (HRI) format: (01)...(17)...(10)...
    and raw machine barcode data streams: 01008472390123491728063010BATCH8842
    """

    # GS1 Application Identifier specifications: (fixed_or_max_length, is_fixed, description)
    GS1_AI_SPECS = {
        "01": (14, True, "Global Trade Item Number (GTIN / DI)"),
        "11": (6, True, "Production Date (YYMMDD)"),
        "17": (6, True, "Expiration Date (YYMMDD)"),
        "10": (20, False, "Batch / Lot Number"),
        "21": (20, False, "Serial Number"),
        "240": (30, False, "Additional Product Identification"),
    }

    def parse_udi(self, raw_code: str) -> UDIComponents:
        """
        Parses raw barcode or text into UDIComponents and validates structure and checksums.
        Distinguishes between GS1 and HIBCC standards cleanly.
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
        """Parses GS1 AI formatted UDI string (both parenthesized and continuous stream)."""
        result = UDIComponents(raw_carrier=text, standard="GS1")

        # 1. Parse parenthesized HRI format: (01)00847239012349(17)280630(10)LOT123
        ai_matches = re.findall(r"\((\d{2,3})\)([^\(\)]+)", text)
        if ai_matches:
            for ai, val in ai_matches:
                result.parsed_elements[ai] = val.strip()
        else:
            # 2. Parse continuous machine stream: 01008472390123491728063010LOT123
            self._parse_continuous_gs1_stream(text, result)

        if not result.parsed_elements:
            result.validation_errors.append("No valid GS1 Application Identifiers found in string.")
            return result

        # Map known AIs to attributes
        # (01) GTIN / Device Identifier
        if "01" in result.parsed_elements:
            di = result.parsed_elements["01"]
            result.device_identifier_di = di
            if len(di) == 14 and di.isdigit():
                if validate_gtin_checksum(di):
                    result.is_di_checksum_valid = True
                else:
                    result.validation_errors.append(f"GTIN-14 Device Identifier '{di}' failed Modulo-10 checksum.")
            else:
                result.validation_errors.append(f"GTIN Device Identifier '{di}' must be exactly 14 numeric digits.")

        # (17) Expiry Date
        if "17" in result.parsed_elements:
            raw_exp = result.parsed_elements["17"]
            parsed_iso, is_valid, is_past = self._validate_and_format_gs1_date(raw_exp)
            if is_valid:
                result.expiry_date_pi = parsed_iso
                result.is_expired = is_past
                if is_past:
                    result.validation_warnings.append(f"Product has expired based on UDI Expiry Date: {parsed_iso}")
            else:
                result.validation_errors.append(f"Invalid GS1 Expiry Date format: '{raw_exp}' (expected YYMMDD).")

        # (11) Manufacturing Date
        if "11" in result.parsed_elements:
            raw_mfg = result.parsed_elements["11"]
            parsed_iso, is_valid, _ = self._validate_and_format_gs1_date(raw_mfg)
            if is_valid:
                result.manufacturing_date_pi = parsed_iso
            else:
                result.validation_errors.append(f"Invalid GS1 Manufacturing Date format: '{raw_mfg}' (expected YYMMDD).")

        # (10) Batch / Lot
        if "10" in result.parsed_elements:
            result.lot_number_pi = result.parsed_elements["10"]

        # (21) Serial Number
        if "21" in result.parsed_elements:
            result.serial_number_pi = result.parsed_elements["21"]

        # (240) Additional ID
        if "240" in result.parsed_elements:
            result.additional_id_pi = result.parsed_elements["240"]

        # Valid overall format if Device Identifier exists
        result.is_valid_format = bool(result.device_identifier_di and result.is_di_checksum_valid)
        return result

    def _parse_continuous_gs1_stream(self, stream: str, result: UDIComponents) -> None:
        """
        Parses continuous unparenthesized GS1 barcode data streams.
        Handles fixed-length AIs (01, 11, 17) and variable-length AIs (10, 21, 240)
        delimited by ASCII 29 (<GS>) or end-of-string.
        """
        idx = 0
        length = len(stream)
        # ASCII 29 is the standard GS1 group separator (FNC1)
        GS = "\x1d"

        while idx < length:
            # Skip any group separator
            if stream[idx] == GS:
                idx += 1
                continue

            # Check 2-digit AI matches
            ai_cand = stream[idx : idx + 2]
            if ai_cand in self.GS1_AI_SPECS:
                ai = ai_cand
                idx += 2
                max_len, is_fixed, _ = self.GS1_AI_SPECS[ai]

                if is_fixed:
                    val = stream[idx : idx + max_len]
                    idx += max_len
                    result.parsed_elements[ai] = val
                else:
                    # Variable length: look for group separator or end
                    val_chars: List[str] = []
                    while idx < length and stream[idx] != GS and len(val_chars) < max_len:
                        val_chars.append(stream[idx])
                        idx += 1
                    result.parsed_elements[ai] = "".join(val_chars)
            else:
                # Check 3-digit AI matches (e.g. 240)
                ai_cand_3 = stream[idx : idx + 3]
                if ai_cand_3 in self.GS1_AI_SPECS:
                    ai = ai_cand_3
                    idx += 3
                    max_len, is_fixed, _ = self.GS1_AI_SPECS[ai]
                    val_chars: List[str] = []
                    while idx < length and stream[idx] != GS and len(val_chars) < max_len:
                        val_chars.append(stream[idx])
                        idx += 1
                    result.parsed_elements[ai] = "".join(val_chars)
                else:
                    # Unknown sequence; advance by 1
                    idx += 1

    def _validate_and_format_gs1_date(self, yymmdd: str) -> Tuple[Optional[str], bool, bool]:
        """
        Validates GS1 YYMMDD date format and applies the GS1 Day-00 Month-End Rule.
        Returns:
            Tuple of (formatted_ISO_string, is_valid, is_past)
        """
        cleaned = re.sub(r"\D", "", yymmdd.strip())
        if len(cleaned) != 6:
            return None, False, False

        yy = int(cleaned[:2])
        mm = int(cleaned[2:4])
        dd = int(cleaned[4:6])

        if not (1 <= mm <= 12):
            return None, False, False

        # Convert YY to 4-digit century
        # Under GS1 rules: 51-99 is 1951-1999; 00-50 is 2000-2050
        year = 2000 + yy if yy <= 50 else 1900 + yy

        # GS1 Day-00 Rule: If day is 00, it denotes the last day of the month
        if dd == 0:
            _, last_day = calendar.monthrange(year, mm)
            dd = last_day
        else:
            _, max_days = calendar.monthrange(year, mm)
            if not (1 <= dd <= max_days):
                return None, False, False

        try:
            dt = datetime(year, mm, dd, tzinfo=timezone.utc)
            iso_str = dt.strftime("%Y-%m-%d")
            now = datetime.now(timezone.utc)
            is_past = dt < now
            return iso_str, True, is_past
        except ValueError:
            return None, False, False

    def _parse_hibcc(self, text: str) -> UDIComponents:
        """Parses Health Industry Business Communications Council (HIBCC) format."""
        result = UDIComponents(raw_carrier=text, standard="HIBCC")

        # Strip leading '+'
        content = text[1:].strip()

        # Check for concatenated format with '/'
        parts = content.split("/")
        primary_part = parts[0]
        secondary_part = parts[1] if len(parts) > 1 else None

        # Validate primary check character
        if len(primary_part) >= 4:
            primary_full = f"+{primary_part}"
            if validate_hibcc_modulo43(primary_full):
                result.is_di_checksum_valid = True
                result.device_identifier_di = primary_part[:-1]
                result.parsed_elements["PRIMARY_DI"] = result.device_identifier_di
            else:
                result.validation_errors.append(f"HIBCC Primary DI failed Modulo-43 check character validation.")
                result.device_identifier_di = primary_part[:-1]

        # Parse secondary part (Expiry, LOT)
        if secondary_part:
            # Secondary check character validation
            sec_full = f"+{secondary_part}"
            if validate_hibcc_modulo43(sec_full):
                sec_content = secondary_part[:-1]
                result.parsed_elements["SECONDARY_PI"] = sec_content
                # Attempt to parse expiration (often first 4-5 digits: YYJJJ or MMYY)
                if len(sec_content) >= 5 and sec_content[:5].isdigit():
                    result.expiry_date_pi = f"20{sec_content[:2]}-{sec_content[2:5]}"
                if len(sec_content) > 5:
                    result.lot_number_pi = sec_content[5:]
            else:
                result.validation_warnings.append("HIBCC Secondary PI failed Modulo-43 check character validation.")

        result.is_valid_format = bool(result.device_identifier_di and result.is_di_checksum_valid)
        return result

    # --------------------------------------------------------------------------
    # Bidirectional OCR vs. Barcode Cross-Verification
    # --------------------------------------------------------------------------
    def cross_verify_with_ocr(
        self,
        udi: UDIComponents,
        ocr_lot: Optional[str] = None,
        ocr_expiry: Optional[str] = None,
        ocr_serial: Optional[str] = None,
    ) -> CrossValidationResult:
        """
        Cross-verifies barcode UDI extracted production identifiers (PI) against
        the human-readable text extracted by OCR.
        Mandated under FDA 21 CFR 830.20 and EU MDR 2017/745 Annex I 23.2(h).
        """
        mismatches: List[str] = []
        lot_matched: Optional[bool] = None
        exp_matched: Optional[bool] = None
        sn_matched: Optional[bool] = None

        # 1. LOT / Batch Cross-Verification
        if udi.lot_number_pi and ocr_lot:
            clean_b_lot = re.sub(r"[^A-Za-z0-9]", "", udi.lot_number_pi).upper()
            clean_o_lot = re.sub(r"[^A-Za-z0-9]", "", ocr_lot).upper()
            if clean_b_lot == clean_o_lot or clean_b_lot in clean_o_lot or clean_o_lot in clean_b_lot:
                lot_matched = True
            else:
                lot_matched = False
                mismatches.append(
                    f"CRITICAL LOT MISMATCH: Barcode UDI LOT '{udi.lot_number_pi}' does not match Printed Label LOT '{ocr_lot}'."
                )

        # 2. Expiry Date Cross-Verification
        if udi.expiry_date_pi and ocr_expiry:
            # Normalize dates to YYYY-MM
            clean_b_exp = udi.expiry_date_pi[:7]
            clean_o_exp = ocr_expiry[:7] if len(ocr_expiry) >= 7 else ocr_expiry
            if clean_b_exp == clean_o_exp:
                exp_matched = True
            else:
                exp_matched = False
                mismatches.append(
                    f"CRITICAL EXPIRY MISMATCH: Barcode UDI Expiry '{udi.expiry_date_pi}' does not match Printed Label Expiry '{ocr_expiry}'."
                )

        # 3. Serial Number Cross-Verification
        if udi.serial_number_pi and ocr_serial:
            clean_b_sn = re.sub(r"[^A-Za-z0-9]", "", udi.serial_number_pi).upper()
            clean_o_sn = re.sub(r"[^A-Za-z0-9]", "", ocr_serial).upper()
            if clean_b_sn == clean_o_sn:
                sn_matched = True
            else:
                sn_matched = False
                mismatches.append(
                    f"CRITICAL SERIAL MISMATCH: Barcode UDI SN '{udi.serial_number_pi}' does not match Printed Label SN '{ocr_serial}'."
                )

        return CrossValidationResult(
            lot_matched=lot_matched,
            expiry_matched=exp_matched,
            serial_matched=sn_matched,
            lot_ocr=ocr_lot,
            lot_barcode=udi.lot_number_pi,
            expiry_ocr=ocr_expiry,
            expiry_barcode=udi.expiry_date_pi,
            mismatches=mismatches,
        )
