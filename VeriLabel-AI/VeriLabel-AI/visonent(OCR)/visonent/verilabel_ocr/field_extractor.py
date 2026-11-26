"""
Medical Device Field Extraction Engine for VeriLabel AI.
Extracts mandatory regulatory fields (FDA 21 CFR 801, EU MDR, ISO 15223-1)
using semantic heuristics, regex patterns, and spatial layout tracking.
"""

from typing import List, Optional, Tuple, Dict, Any
import re
from datetime import datetime
from dateutil import parser as date_parser

from .models import (
    OCRTextBlock,
    ExtractedField,
    MedicalLabelData,
    FieldStatus,
    BoundingBox,
)


class MedicalFieldExtractor:
    """
    Identifies and structures medical device label fields from OCR text blocks.
    Maps fields to standardized regulatory entities with confidence tracking and bounding boxes.
    """

    # --- Compound Prefix Regex Patterns for Medical Identifiers ---
    REF_PREFIX = re.compile(
        r"^(?:REF(?:\s*[/|]\s*CAT(?:ALOG)?(?:\s*NO\.?)?)?|CAT(?:ALOG)?(?:\s*NO\.?)?|MODEL(?:\s*NO\.?)?|PART(?:\s*NO\.?)?)[\s/:\-]*",
        re.IGNORECASE,
    )
    LOT_PREFIX = re.compile(
        r"^(?:LOT(?:\s*[/|]\s*BATCH(?:\s*NO\.?)?)?|BATCH(?:\s*NO\.?)?|BN)[\s/:\-]*",
        re.IGNORECASE,
    )
    SN_PREFIX = re.compile(
        r"^(?:SERIAL(?:\s*NO\.?)?(?:\s*\(\s*SN\s*\))?|SN(?:\s*NO\.?)?)[\s/:\-]*",
        re.IGNORECASE,
    )

    # Date markers & patterns
    EXP_PREFIX_PATTERN = re.compile(
        r"\b(?:EXP(?:IRY)?(?:\s*DATE)?(?:\s*\(\s*EXP\s*\))?|USE[\s\-]*(?:BY|BEFORE)|EXPIRATION(?:\s*DATE)?)\b[\s:]*",
        re.IGNORECASE,
    )
    MFG_PREFIX_PATTERN = re.compile(
        r"\b(?:MFG(?:[\s\.\-]*(?:DATE|DT))?|MFD|DOM|DATE\s*OF\s*MANUFACTURE|MANUFACTURED)\b[\s:]*",
        re.IGNORECASE,
    )
    
    DATE_VAL_PATTERN = re.compile(
        r"(\b\d{4}[\-\/\.]\d{1,2}[\-\/\.]\d{1,2}\b|"  # YYYY-MM-DD
        r"\b\d{1,2}[\-\/\.]\d{1,2}[\-\/\.]\d{4}\b|"  # DD-MM-YYYY or MM-DD-YYYY
        r"\b\d{4}[\-\/\.]\d{1,2}\b|"                  # YYYY-MM
        r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[\s\-\.]*\d{4}\b|"  # MMM YYYY
        r"\b\d{1,2}[\s\-\.](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[\s\-\.]*\d{4}\b)",  # DD MMM YYYY
        re.IGNORECASE,
    )

    # Storage conditions
    STORAGE_TEMP_PATTERN = re.compile(
        r"(?:store(?:\s*between|\s*at)?[\s:]*)?(-?\d{1,2}\s*°?\s*C|\d{1,2}\s*°?\s*F)\s*(?:to|-|–)\s*(-?\d{1,2}\s*°?\s*C|\d{1,2}\s*°?\s*F)|"
        r"(?:store\s*(?:below|under)\s*(-?\d{1,2}\s*°?\s*[CF]))",
        re.IGNORECASE,
    )
    STORAGE_KEYWORDS = [
        "keep dry", "protect from sunlight", "protect from light",
        "do not freeze", "store in a cool dry place", "temperature limit",
        "store at", "humidity", "keep away from heat"
    ]

    # Safety Warnings & Regulatory Precautions
    WARNING_KEYWORDS = [
        "caution", "warning", "precaution", "danger", "single use only",
        "do not reuse", "do not re-use", "non-sterile", "sterile eo", "sterile r",
        "rx only", "prescription only", "for professional use only",
        "do not resterilize", "contains phthalates", "latex free", "natural rubber latex",
        "if package is damaged", "see instructions for use"
    ]

    # Standalone regulatory tags to exclude from device name candidates
    REGULATORY_TAG_EXCLUSIONS = {
        "STERILE EO", "STERILE R", "STERILE", "NON-STERILE",
        "RX ONLY", "RX-ONLY", "LATEX FREE", "REF", "LOT", "SN",
        "CE", "MD", "UDI", "QTY", "EC REP"
    }

    # Manufacturer indicators
    MFR_PREFIX_PATTERN = re.compile(
        r"(?:manufactured\s+by|mfr(?:\s*name)?|fabricant|manufacturer|distributed\s+by)[\s:]*",
        re.IGNORECASE,
    )
    CORP_SUFFIX_PATTERN = re.compile(
        r"\b(?:inc\.?|llc\.?|ltd\.?|gmbh|corp\.?|corporation|technologies|healthcare|laboratories|biomedical|devices)\b",
        re.IGNORECASE,
    )

    # UDI text pattern (GS1 Application Identifiers)
    UDI_PATTERN = re.compile(
        r"\(01\)\d{14}(?:\(17\)\d{6})?(?:\(10\)[A-Z0-9]+)?(?:\(21\)[A-Z0-9]+)?|"
        r"\b01\d{14}17\d{6}10[A-Z0-9]+",
        re.IGNORECASE,
    )

    def extract(self, blocks: List[OCRTextBlock]) -> MedicalLabelData:
        data = MedicalLabelData()
        if not blocks:
            return data

        used_block_ids = set()

        # 1. Catalog / REF
        ref_field = self._extract_prefixed_field(blocks, self.REF_PREFIX, "catalog_number", ["REF", "CATALOG", "CAT NO", "MODEL"])
        if ref_field:
            data.catalog_number = ref_field
            used_block_ids.update(ref_field.source_block_ids)

        # 2. LOT Number
        lot_field = self._extract_prefixed_field(blocks, self.LOT_PREFIX, "lot_number", ["LOT", "BATCH", "BN"])
        if lot_field:
            data.lot_number = lot_field
            used_block_ids.update(lot_field.source_block_ids)

        # 3. Serial Number
        sn_field = self._extract_prefixed_field(blocks, self.SN_PREFIX, "serial_number", ["SERIAL", "SN"])
        if sn_field:
            data.serial_number = sn_field
            used_block_ids.update(sn_field.source_block_ids)

        # 4. Dates: Expiry and Manufacturing
        exp_field, mfg_field = self._extract_dates(blocks)
        if exp_field:
            data.expiry_date = exp_field
            used_block_ids.update(exp_field.source_block_ids)
        if mfg_field:
            data.mfg_date = mfg_field
            used_block_ids.update(mfg_field.source_block_ids)

        # 5. Manufacturer Information
        mfr_name_field, mfr_addr_field, country_field = self._extract_manufacturer_info(blocks)
        if mfr_name_field:
            data.manufacturer_name = mfr_name_field
            used_block_ids.update(mfr_name_field.source_block_ids)
        if mfr_addr_field:
            data.manufacturer_address = mfr_addr_field
        if country_field:
            data.country_of_origin = country_field

        # 6. Storage Conditions
        storage_field = self._extract_storage_conditions(blocks)
        if storage_field:
            data.storage_conditions = storage_field
            used_block_ids.update(storage_field.source_block_ids)

        # 7. Warnings and Regulatory Statements
        warnings_list, statements_list = self._extract_warnings_and_statements(blocks)
        data.warnings = warnings_list
        data.regulatory_statements = statements_list
        for item in warnings_list + statements_list:
            used_block_ids.update(item.source_block_ids)

        # 8. UDI Candidates
        data.udi_text_candidates = self._extract_udi_candidates(blocks)

        # 9. Product Name / Device Name
        dev_name_field, trade_name_field = self._infer_device_name(blocks, used_block_ids)
        if dev_name_field:
            data.device_name = dev_name_field
        if trade_name_field:
            data.trade_name = trade_name_field

        return data

    def _extract_prefixed_field(
        self, blocks: List[OCRTextBlock], prefix_re: re.Pattern, field_name: str, keyword_cues: List[str]
    ) -> Optional[ExtractedField]:
        """
        Extracts value after a known compound prefix like 'REF / CATALOG NO: CATH-123'
        or 'LOT / BATCH NO: LOT20240915'.
        """
        for i, block in enumerate(blocks):
            text = block.text.strip()
            upper_text = text.upper()

            # Check if line contains any keyword cue
            if not any(k in upper_text for k in keyword_cues):
                continue

            # Strip compound prefix
            # E.g., 'REF / CATALOG NO: CATH-2024-X8' -> 'CATH-2024-X8'
            cleaned = re.sub(
                r"^.*?(?:REF(?:\s*[/|]\s*CAT(?:ALOG)?(?:\s*NO\.?)?)?|CAT(?:ALOG)?(?:\s*NO\.?)?|LOT(?:\s*[/|]\s*BATCH(?:\s*NO\.?)?)?|BATCH(?:\s*NO\.?)?|SERIAL(?:\s*NO\.?)?(?:\s*\(\s*SN\s*\))?|SN(?:\s*NO\.?)?)[\s/:\-]+",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            if cleaned and cleaned != text and len(cleaned) >= 3:
                return ExtractedField(
                    field_name=field_name,
                    value=cleaned,
                    raw_text=text,
                    status=FieldStatus.PRESENT,
                    confidence=block.confidence,
                    bbox=block.bbox,
                    source_block_ids=[block.id],
                )

            # If value is on the NEXT block
            if i + 1 < len(blocks):
                next_text = blocks[i + 1].text.strip()
                if len(next_text) >= 3 and not any(k in next_text.upper() for k in ["MFG", "EXP", "LOT", "REF", "SN", "MADE"]):
                    comb_bbox = self._combine_bboxes(block.bbox, blocks[i + 1].bbox)
                    return ExtractedField(
                        field_name=field_name,
                        value=next_text,
                        raw_text=f"{text} {next_text}",
                        status=FieldStatus.PRESENT,
                        confidence=min(block.confidence, blocks[i + 1].confidence),
                        bbox=comb_bbox,
                        source_block_ids=[block.id, blocks[i + 1].id],
                    )

        return None

    def _extract_dates(
        self, blocks: List[OCRTextBlock]
    ) -> Tuple[Optional[ExtractedField], Optional[ExtractedField]]:
        """Identifies Expiry date and Manufacturing date."""
        exp_field = None
        mfg_field = None

        for i, block in enumerate(blocks):
            text = block.text

            # Check EXPIRY date
            if not exp_field:
                exp_match = self.EXP_PREFIX_PATTERN.search(text)
                if exp_match or any(k in text.upper() for k in ["EXP", "USE BY", "USE-BY"]):
                    date_match = self.DATE_VAL_PATTERN.search(text)
                    if date_match:
                        raw_val = date_match.group(0)
                        iso_val = self._parse_date_to_iso(raw_val)
                        exp_field = ExtractedField(
                            field_name="expiry_date",
                            value=raw_val,
                            raw_text=text,
                            status=FieldStatus.PRESENT,
                            confidence=block.confidence,
                            bbox=block.bbox,
                            source_block_ids=[block.id],
                            metadata={"parsed_iso": iso_val},
                        )
                    elif i + 1 < len(blocks):
                        next_text = blocks[i + 1].text
                        next_date_match = self.DATE_VAL_PATTERN.search(next_text)
                        if next_date_match:
                            raw_val = next_date_match.group(0)
                            iso_val = self._parse_date_to_iso(raw_val)
                            comb_bbox = self._combine_bboxes(block.bbox, blocks[i + 1].bbox)
                            exp_field = ExtractedField(
                                field_name="expiry_date",
                                value=raw_val,
                                raw_text=f"{text} {next_text}",
                                status=FieldStatus.PRESENT,
                                confidence=min(block.confidence, blocks[i + 1].confidence),
                                bbox=comb_bbox,
                                source_block_ids=[block.id, blocks[i + 1].id],
                                metadata={"parsed_iso": iso_val},
                            )

            # Check MANUFACTURING date
            if not mfg_field:
                mfg_match = self.MFG_PREFIX_PATTERN.search(text)
                if mfg_match or any(k in text.upper() for k in ["MFG", "MFD", "DOM", "MANUFACTURED"]):
                    date_match = self.DATE_VAL_PATTERN.search(text)
                    if date_match:
                        raw_val = date_match.group(0)
                        iso_val = self._parse_date_to_iso(raw_val)
                        mfg_field = ExtractedField(
                            field_name="mfg_date",
                            value=raw_val,
                            raw_text=text,
                            status=FieldStatus.PRESENT,
                            confidence=block.confidence,
                            bbox=block.bbox,
                            source_block_ids=[block.id],
                            metadata={"parsed_iso": iso_val},
                        )
                    elif i + 1 < len(blocks):
                        next_text = blocks[i + 1].text
                        next_date_match = self.DATE_VAL_PATTERN.search(next_text)
                        if next_date_match:
                            raw_val = next_date_match.group(0)
                            iso_val = self._parse_date_to_iso(raw_val)
                            comb_bbox = self._combine_bboxes(block.bbox, blocks[i + 1].bbox)
                            mfg_field = ExtractedField(
                                field_name="mfg_date",
                                value=raw_val,
                                raw_text=f"{text} {next_text}",
                                status=FieldStatus.PRESENT,
                                confidence=min(block.confidence, blocks[i + 1].confidence),
                                bbox=comb_bbox,
                                source_block_ids=[block.id, blocks[i + 1].id],
                                metadata={"parsed_iso": iso_val},
                            )

        return exp_field, mfg_field

    def _parse_date_to_iso(self, date_str: str) -> Optional[str]:
        try:
            clean_str = re.sub(r"[^\w\-\/\.]", " ", date_str).strip()
            dt = date_parser.parse(clean_str, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _extract_manufacturer_info(
        self, blocks: List[OCRTextBlock]
    ) -> Tuple[Optional[ExtractedField], Optional[ExtractedField], Optional[ExtractedField]]:
        mfr_name_field = None
        mfr_addr_field = None
        country_field = None

        country_regex = re.compile(
            r"\b(Made in|Country of Origin:?)\s*([A-Za-z\s]+)\b|"
            r"\b(USA|Germany|Japan|Switzerland|India|China|UK|Ireland)\b",
            re.IGNORECASE,
        )

        for i, block in enumerate(blocks):
            text = block.text

            # Check Country
            if not country_field:
                c_match = country_regex.search(text)
                if c_match:
                    val = c_match.group(2) or c_match.group(0)
                    country_field = ExtractedField(
                        field_name="country_of_origin",
                        value=val.strip(),
                        raw_text=text,
                        status=FieldStatus.PRESENT,
                        confidence=block.confidence,
                        bbox=block.bbox,
                        source_block_ids=[block.id],
                    )

            # Check Manufacturer header
            if not mfr_name_field:
                mfr_match = self.MFR_PREFIX_PATTERN.search(text)
                if mfr_match:
                    prefix_end = mfr_match.end()
                    remainder = text[prefix_end:].strip()
                    if remainder:
                        mfr_name_field = ExtractedField(
                            field_name="manufacturer_name",
                            value=remainder,
                            raw_text=text,
                            status=FieldStatus.PRESENT,
                            confidence=block.confidence,
                            bbox=block.bbox,
                            source_block_ids=[block.id],
                        )
                    elif i + 1 < len(blocks):
                        next_block = blocks[i + 1]
                        comb_bbox = self._combine_bboxes(block.bbox, next_block.bbox)
                        mfr_name_field = ExtractedField(
                            field_name="manufacturer_name",
                            value=next_block.text.strip(),
                            raw_text=f"{text} {next_block.text}",
                            status=FieldStatus.PRESENT,
                            confidence=min(block.confidence, next_block.confidence),
                            bbox=comb_bbox,
                            source_block_ids=[block.id, next_block.id],
                        )
                elif self.CORP_SUFFIX_PATTERN.search(text):
                    mfr_name_field = ExtractedField(
                        field_name="manufacturer_name",
                        value=text.strip(),
                        raw_text=text,
                        status=FieldStatus.PRESENT,
                        confidence=block.confidence,
                        bbox=block.bbox,
                        source_block_ids=[block.id],
                    )

        # Address detection
        if mfr_name_field and mfr_name_field.source_block_ids:
            last_id = max(mfr_name_field.source_block_ids)
            addr_blocks = []
            for b in blocks:
                if b.id in [last_id + 1, last_id + 2]:
                    if any(c in b.text.lower() for c in ["street", "st.", "road", "ave", "blvd", "box", "zip", "po box", "suite", ","]):
                        addr_blocks.append(b)
            if addr_blocks:
                addr_text = " ".join([b.text for b in addr_blocks])
                comb_bbox = addr_blocks[0].bbox
                for ab in addr_blocks[1:]:
                    comb_bbox = self._combine_bboxes(comb_bbox, ab.bbox)
                mfr_addr_field = ExtractedField(
                    field_name="manufacturer_address",
                    value=addr_text,
                    raw_text=addr_text,
                    status=FieldStatus.PRESENT,
                    confidence=sum(b.confidence for b in addr_blocks) / len(addr_blocks),
                    bbox=comb_bbox,
                    source_block_ids=[b.id for b in addr_blocks],
                )

        return mfr_name_field, mfr_addr_field, country_field

    def _extract_storage_conditions(self, blocks: List[OCRTextBlock]) -> Optional[ExtractedField]:
        matched_blocks = []
        storage_notes = []

        for block in blocks:
            text = block.text
            lower_text = text.lower()

            temp_match = self.STORAGE_TEMP_PATTERN.search(text)
            if temp_match:
                matched_blocks.append(block)
                storage_notes.append(text.strip())
                continue

            if any(k in lower_text for k in self.STORAGE_KEYWORDS):
                matched_blocks.append(block)
                storage_notes.append(text.strip())

        if not matched_blocks:
            return None

        combined_text = "; ".join(storage_notes)
        comb_bbox = matched_blocks[0].bbox
        for b in matched_blocks[1:]:
            comb_bbox = self._combine_bboxes(comb_bbox, b.bbox)

        return ExtractedField(
            field_name="storage_conditions",
            value=combined_text,
            raw_text=combined_text,
            status=FieldStatus.PRESENT,
            confidence=sum(b.confidence for b in matched_blocks) / len(matched_blocks),
            bbox=comb_bbox,
            source_block_ids=[b.id for b in matched_blocks],
        )

    def _extract_warnings_and_statements(
        self, blocks: List[OCRTextBlock]
    ) -> Tuple[List[ExtractedField], List[ExtractedField]]:
        """Identifies warnings, contraindications, and regulatory symbols/statements."""
        warnings: List[ExtractedField] = []
        statements: List[ExtractedField] = []

        header_regex = re.compile(
            r"^(?:WARNINGS?|PRECAUTIONS?|CONTRAINDICATIONS?|CAUTIONS?)[\s,/&:—\-]*$",
            re.IGNORECASE,
        )

        for block in blocks:
            lower = block.text.lower()
            text_strip = block.text.strip()

            # Ignore pure section header titles or placeholder omission text
            if header_regex.match(text_strip) or "omitted" in lower:
                continue

            matched_kws = [k for k in self.WARNING_KEYWORDS if k in lower]

            if matched_kws:
                is_warning = any(w in lower for w in ["caution", "warning", "precaution", "danger", "do not"])
                target_list = warnings if is_warning else statements

                target_list.append(
                    ExtractedField(
                        field_name="warning" if is_warning else "regulatory_statement",
                        value=text_strip,
                        raw_text=block.text,
                        status=FieldStatus.PRESENT,
                        confidence=block.confidence,
                        bbox=block.bbox,
                        source_block_ids=[block.id],
                        metadata={"matched_keywords": matched_kws},
                    )
                )

        return warnings, statements

    def _extract_udi_candidates(self, blocks: List[OCRTextBlock]) -> List[ExtractedField]:
        candidates: List[ExtractedField] = []
        for block in blocks:
            match = self.UDI_PATTERN.search(block.text)
            if match:
                candidates.append(
                    ExtractedField(
                        field_name="udi_text_candidate",
                        value=match.group(0),
                        raw_text=block.text,
                        status=FieldStatus.PRESENT,
                        confidence=block.confidence,
                        bbox=block.bbox,
                        source_block_ids=[block.id],
                    )
                )
        return candidates

    def _infer_device_name(
        self, blocks: List[OCRTextBlock], used_block_ids: set
    ) -> Tuple[Optional[ExtractedField], Optional[ExtractedField]]:
        dev_name = None
        trade_name = None

        if not blocks:
            return None, None

        max_y = max(b.bbox.ymax for b in blocks)
        top_threshold = max_y * 0.40

        candidates = []
        for b in blocks:
            cleaned = re.sub(r"[^\w\s]", "", b.text).strip().upper()
            if b.id in used_block_ids:
                continue
            if b.bbox.ymin > top_threshold:
                continue
            if len(b.text.strip()) < 3:
                continue
            if cleaned in self.REGULATORY_TAG_EXCLUSIONS:
                continue
            candidates.append(b)

        # Explicit "Device Name:" or "Product:" prefix
        for b in candidates:
            p_match = re.search(r"(?:Device\s*Name|Product(?:\s*Name)?|Trade\s*Name)[\s:]*(.*)", b.text, re.IGNORECASE)
            if p_match:
                val = p_match.group(1).strip()
                if val:
                    dev_name = ExtractedField(
                        field_name="device_name",
                        value=val,
                        raw_text=b.text,
                        status=FieldStatus.PRESENT,
                        confidence=b.confidence,
                        bbox=b.bbox,
                        source_block_ids=[b.id],
                    )
                    used_block_ids.add(b.id)
                    break

        # Fallback to the largest / top-most unused text block
        if not dev_name and candidates:
            candidates.sort(key=lambda b: (b.bbox.height, -b.bbox.ymin), reverse=True)
            top_block = candidates[0]
            dev_name = ExtractedField(
                field_name="device_name",
                value=top_block.text.strip(),
                raw_text=top_block.text,
                status=FieldStatus.PRESENT,
                confidence=top_block.confidence,
                bbox=top_block.bbox,
                source_block_ids=[top_block.id],
                metadata={"inferred_from_layout": True},
            )
            used_block_ids.add(top_block.id)

        return dev_name, trade_name

    def _combine_bboxes(self, b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
        return BoundingBox(
            xmin=min(b1.xmin, b2.xmin),
            ymin=min(b1.ymin, b2.ymin),
            xmax=max(b1.xmax, b2.xmax),
            ymax=max(b1.ymax, b2.ymax),
        )
