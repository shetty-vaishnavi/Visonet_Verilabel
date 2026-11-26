"""
VeriLabel AI - Module 2: Medical Device Label Field Extractor
Team: Brahmastra | GITAM University
Parses raw OCR text into structured medical regulatory fields according to FDA 21 CFR 801,
EU MDR Annex I Chapter III, and ISO 15223-1 standards.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple
from verilabel.core.utils import parse_date_string, format_iso_date
from verilabel.modules.ocr.base import OCRResult


@dataclass
class ExtractedLabelFields:
    """Standardized medical device label data model."""
    product_name: Optional[str] = None
    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    manufacturing_date: Optional[str] = None
    expiry_date: Optional[str] = None
    lot_number: Optional[str] = None
    serial_number: Optional[str] = None
    catalog_number: Optional[str] = None
    udi_di: Optional[str] = None
    udi_carrier: Optional[str] = None
    storage_conditions: Optional[str] = None
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    is_sterile: bool = False
    sterilization_method: Optional[str] = None
    is_single_use: bool = False
    warnings: List[str] = field(default_factory=list)
    instructions_for_use: Optional[str] = None
    ce_mark_number: Optional[str] = None
    is_rx_only: bool = False
    raw_text: str = ""
    field_confidences: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MedicalFieldExtractor:
    """
    Extracts structured medical device label attributes using regex patterns,
    fuzzy keyword matching, and context analysis.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # Date patterns
        self.exp_pattern = re.compile(
            r"(?:EXP(?:IRY)?|USE[\s\-]BY|USE\s+BEFORE|VERWENDUNG\s+BIS)[\s\:\.\-]+([A-Za-z0-9\/\.\-]+)",
            re.IGNORECASE
        )
        self.mfg_pattern = re.compile(
            r"(?:MFG|MANUF(?:ACTURING)?|DATE\s+OF\s+MANUFACTURE|HERSTELLUNGSDATUM)[\s\:\.\-]+([A-Za-z0-9\/\.\-]+)",
            re.IGNORECASE
        )

        # Batch / Lot pattern
        self.lot_pattern = re.compile(
            r"(?:LOT|BATCH|CHARGE)[\s\:\.\#\-]+([A-Za-z0-9\-]+)",
            re.IGNORECASE
        )

        # Catalog / REF pattern
        self.ref_pattern = re.compile(
            r"(?:REF|CAT(?:ALOG)?(?:\s+NO)?|PART\s+NO|MODEL)[\s\:\.\#\-]+([A-Za-z0-9\-]+)",
            re.IGNORECASE
        )

        # Serial Number (SN)
        self.sn_pattern = re.compile(
            r"(?:SN|SERIAL\s*(?:NO)?|SERIENNUMMER)[\s\:\.\#\-]+([A-Za-z0-9\-]+)",
            re.IGNORECASE
        )

        # Temperature limits (e.g. 2°C - 8°C, 15°C to 25°C, 2-8 C, Store between 2C and 8C)
        self.temp_range_pattern = re.compile(
            r"(\-?\d+(?:\.\d+)?)\s*(?:°?C)?\s*(?:to|\-|–|until)\s*(\-?\d+(?:\.\d+)?)\s*°?C",
            re.IGNORECASE
        )
        self.temp_max_pattern = re.compile(
            r"(?:Store\s+below|Do\s+not\s+exceed|Max(?:imum)?)\s*(\-?\d+(?:\.\d+)?)\s*°?C",
            re.IGNORECASE
        )

        # CE Mark with 4 digit Notified Body
        self.ce_pattern = re.compile(
            r"\bCE\s*([0-9]{4})\b",
            re.IGNORECASE
        )

        # Rx Only
        self.rx_pattern = re.compile(
            r"(?:Rx\s*Only|Prescription\s*Only|Caution:\s*Federal\s*law\s*restricts)",
            re.IGNORECASE
        )

        # Sterility patterns
        self.sterile_pattern = re.compile(
            r"\b(STERILE(?:\s*(?:EO|R|A|STEAM))?)",
            re.IGNORECASE
        )

        # Single use patterns
        self.single_use_pattern = re.compile(
            r"(?:Do\s*not\s*re[\-\s]?use|Single\s*use\s*only|For\s*single\s*use)",
            re.IGNORECASE
        )

        # Manufacturer indicator
        self.mfr_indicator_pattern = re.compile(
            r"(?:Manufactured\s+by|Manufacturer|Mfr|Hersteller)[\s\:\-]+([^\n\r]+)",
            re.IGNORECASE
        )

    def extract_fields(self, ocr_data: OCRResult) -> ExtractedLabelFields:
        """Parses OCRResult into ExtractedLabelFields."""
        raw_text = ocr_data.raw_text
        lines = ocr_data.lines if ocr_data.lines else raw_text.splitlines()

        fields = ExtractedLabelFields(raw_text=raw_text)

        # 1. Product Name (usually the most prominent line, or line following 'Product:' or 'Device:')
        fields.product_name = self._extract_product_name(lines)
        if fields.product_name:
            fields.field_confidences["product_name"] = 0.90

        # 2. Manufacturer Name & Address
        mfr_name, mfr_addr = self._extract_manufacturer(lines, raw_text)
        fields.manufacturer_name = mfr_name
        fields.manufacturer_address = mfr_addr
        if mfr_name:
            fields.field_confidences["manufacturer_name"] = 0.88

        # 3. Expiry Date
        exp_match = self.exp_pattern.search(raw_text)
        if exp_match:
            raw_exp = exp_match.group(1).strip()
            parsed_exp = parse_date_string(raw_exp)
            fields.expiry_date = format_iso_date(parsed_exp) or raw_exp
            fields.field_confidences["expiry_date"] = 0.95

        # 4. Manufacturing Date
        mfg_match = self.mfg_pattern.search(raw_text)
        if mfg_match:
            raw_mfg = mfg_match.group(1).strip()
            parsed_mfg = parse_date_string(raw_mfg)
            fields.manufacturing_date = format_iso_date(parsed_mfg) or raw_mfg
            fields.field_confidences["manufacturing_date"] = 0.92

        # 5. Lot / Batch
        lot_match = self.lot_pattern.search(raw_text)
        if lot_match:
            raw_lot = lot_match.group(1).strip()
            # If OCR captures redundant label prefix, e.g. 'LOTBATCH-8842' -> 'BATCH-8842'
            clean_lot = re.sub(r"^(?:LOT|BATCH|CHARGE)[\s\:\.\#\-]*", "", raw_lot, flags=re.IGNORECASE)
            fields.lot_number = clean_lot if clean_lot else raw_lot
            fields.field_confidences["lot_number"] = 0.95

        # 6. Catalog / REF
        ref_match = self.ref_pattern.search(raw_text)
        if ref_match:
            raw_ref = ref_match.group(1).strip()
            clean_ref = re.sub(r"^(?:REF|CAT(?:ALOG)?|PART|MODEL)[\s\:\.\#\-]*", "", raw_ref, flags=re.IGNORECASE)
            fields.catalog_number = clean_ref if clean_ref else raw_ref
            fields.field_confidences["catalog_number"] = 0.95

        # 7. Serial Number
        sn_match = self.sn_pattern.search(raw_text)
        if sn_match:
            fields.serial_number = sn_match.group(1).strip()
            fields.field_confidences["serial_number"] = 0.92

        # 8. Storage & Temperature
        temp_match = self.temp_range_pattern.search(raw_text)
        if temp_match:
            fields.temperature_min_c = float(temp_match.group(1))
            fields.temperature_max_c = float(temp_match.group(2))
            fields.storage_conditions = f"Store between {fields.temperature_min_c}°C and {fields.temperature_max_c}°C"
            fields.field_confidences["storage_conditions"] = 0.95
        else:
            max_temp_match = self.temp_max_pattern.search(raw_text)
            if max_temp_match:
                fields.temperature_max_c = float(max_temp_match.group(1))
                fields.storage_conditions = f"Store below {fields.temperature_max_c}°C"
                fields.field_confidences["storage_conditions"] = 0.90
            elif "keep dry" in raw_text.lower():
                fields.storage_conditions = "Keep dry / protect from moisture"
                fields.field_confidences["storage_conditions"] = 0.85

        # 9. Sterility
        if "non-sterile" in raw_text.lower() or "not sterile" in raw_text.lower():
            fields.is_sterile = False
            fields.field_confidences["is_sterile"] = 0.95
        else:
            sterile_match = self.sterile_pattern.search(raw_text)
            if sterile_match:
                fields.is_sterile = True
                fields.sterilization_method = sterile_match.group(1).upper()
                fields.field_confidences["is_sterile"] = 0.95

        # 10. Single Use
        if self.single_use_pattern.search(raw_text):
            fields.is_single_use = True
            fields.field_confidences["is_single_use"] = 0.95

        # 11. Rx Only
        if self.rx_pattern.search(raw_text):
            fields.is_rx_only = True
            fields.field_confidences["is_rx_only"] = 0.95

        # 12. CE Mark
        ce_match = self.ce_pattern.search(raw_text)
        if ce_match:
            fields.ce_mark_number = f"CE {ce_match.group(1)}"
            fields.field_confidences["ce_mark_number"] = 0.98

        # 13. Warnings & Precautions
        fields.warnings = self._extract_warnings(raw_text, lines)

        # 14. Instructions for use (IFU)
        fields.instructions_for_use = self._extract_ifu_statement(raw_text)

        return fields

    def _extract_product_name(self, lines: List[str]) -> Optional[str]:
        """Identifies product name from header lines or explicit labels."""
        for line in lines:
            stripped = line.strip()
            # Explicit prefix
            prefix_match = re.match(r"^(?:Product|Device|Trade\s+Name|Name)[\s\:\-]+(.+)$", stripped, re.IGNORECASE)
            if prefix_match:
                return prefix_match.group(1).strip()

        # Otherwise pick first non-empty line that doesn't start with standard identifiers
        skip_words = ("REF", "LOT", "SN", "EXP", "MFG", "STERILE", "CE", "CAUTION", "WARNING", "MANUFACTURED")
        for line in lines[:4]:
            stripped = line.strip()
            if len(stripped) > 3 and not any(stripped.upper().startswith(kw) for kw in skip_words):
                return stripped
        return None

    def _extract_manufacturer(self, lines: List[str], raw_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts manufacturer name and address."""
        mfr_match = self.mfr_indicator_pattern.search(raw_text)
        if mfr_match:
            name = mfr_match.group(1).strip()
            # Check if next line contains address/country
            return name, None

        # Look for factory icon or company suffixes
        for i, line in enumerate(lines):
            if any(s in line.upper() for s in ["INC.", "LTD.", "LLC", "CORP.", "GMBH", "B.V.", "AG", "PHARMA"]):
                # Often preceded or followed by address
                addr = lines[i + 1].strip() if i + 1 < len(lines) else None
                return line.strip(), addr

        return None, None

    def _extract_warnings(self, raw_text: str, lines: List[str]) -> List[str]:
        """Extracts cautionary statements and warnings."""
        warnings: List[str] = []
        warning_keywords = [
            "caution", "warning", "do not use if package", "contraindication",
            "sterile unless opened or damaged", "do not re-sterilize", "pyrogen free",
            "keep away from direct sunlight", "contains latex", "latex-free", "non-pyrogenic"
        ]

        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in warning_keywords):
                warnings.append(line.strip())

        return list(dict.fromkeys(warnings))  # preserve order, remove duplicates

    def _extract_ifu_statement(self, raw_text: str) -> Optional[str]:
        """Detects references to instructions for use (IFU/eIFU)."""
        ifu_patterns = [
            r"(?:consult\s+instructions\s+for\s+use|see\s+instructions|ifu\s+available\s+at[^\n\r]+|www\.[^\s]+\.com\/ifu)",
            r"(?:read\s+package\s+insert|refer\s+to\s+manual)"
        ]
        for p in ifu_patterns:
            match = re.search(p, raw_text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None
