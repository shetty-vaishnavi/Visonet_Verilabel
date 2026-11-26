"""
VeriLabel AI - Module 5: Compliance Validation Engine
Team: Brahmastra | GITAM University
Evaluates extracted medical label data, detected symbols, and UDI barcodes against
FDA 21 CFR 801/830, EU MDR 2017/745, and ISO 15223-1 regulatory compliance rules.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from verilabel.config import RULES_FILE, COMPLIANCE_SCORE_PASS, COMPLIANCE_SCORE_CONDITIONAL
from verilabel.exceptions import ComplianceRuleError
from verilabel.core.utils import parse_date_string
from verilabel.modules.ocr.field_extractor import ExtractedLabelFields
from verilabel.modules.vision.symbol_detector import DetectedSymbol
from verilabel.modules.barcode.barcode_scanner import DecodedBarcode
from verilabel.modules.barcode.udi_validator import UDIComponents, UDIValidator
from verilabel.modules.compliance.models import (
    ComplianceRuleDefinition,
    RuleSeverity,
    RuleCheckOutcome,
    LabelComplianceReport,
    ComplianceVerdict
)

logger = logging.getLogger("VeriLabel.Compliance.Engine")


class ComplianceEngine:
    """
    Executes rule-based compliance checks and computes audit verdicts and metrics.
    """

    def __init__(self, rules_file: Optional[Path] = None):
        self.rules_file = Path(rules_file) if rules_file else RULES_FILE
        self.rules: Dict[str, ComplianceRuleDefinition] = {}
        self._load_rules()

    def _load_rules(self):
        """Loads and parses rules from JSON configuration."""
        if not self.rules_file.exists():
            raise ComplianceRuleError(
                f"Compliance rules file not found: {self.rules_file}",
                details={"path": str(self.rules_file)}
            )

        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                rule_def = ComplianceRuleDefinition(**item)
                self.rules[rule_def.rule_id] = rule_def

            logger.info(f"Loaded {len(self.rules)} compliance rules.")
        except Exception as e:
            raise ComplianceRuleError(f"Failed to load compliance rules: {e}") from e

    def evaluate(
        self,
        file_name: str,
        file_hash: str,
        fields: ExtractedLabelFields,
        symbols: List[DetectedSymbol],
        barcodes: List[DecodedBarcode],
        udi: Optional[UDIComponents] = None,
        processing_time_ms: float = 0.0,
        expected_product_data: Optional[Dict[str, Any]] = None
    ) -> LabelComplianceReport:
        """
        Executes all active rules against the ingested label data.
        Returns a comprehensive LabelComplianceReport.
        """
        checks: List[RuleCheckOutcome] = []
        symbol_names = {s.symbol_name for s in symbols}

        # 1. MAND-001: Product Name
        r_mand1 = self.rules.get("MAND-001")
        if r_mand1:
            passed = bool(fields.product_name and len(fields.product_name.strip()) > 2)
            checks.append(RuleCheckOutcome(
                rule_id=r_mand1.rule_id,
                rule_name=r_mand1.name,
                category=r_mand1.category,
                standard=r_mand1.standard,
                severity=r_mand1.severity,
                passed=passed,
                observed_value=fields.product_name or "MISSING",
                expected_condition="Prominent product or trade name must be present on label",
                message="Product name successfully identified." if passed else "Mandatory product name is missing from the label.",
                remediation_hint=r_mand1.remediation_hint if not passed else None
            ))

        # 2. MAND-002: Manufacturer Name
        r_mand2 = self.rules.get("MAND-002")
        if r_mand2:
            passed = bool(fields.manufacturer_name or "manufacturer" in symbol_names)
            checks.append(RuleCheckOutcome(
                rule_id=r_mand2.rule_id,
                rule_name=r_mand2.name,
                category=r_mand2.category,
                standard=r_mand2.standard,
                severity=r_mand2.severity,
                passed=passed,
                observed_value=fields.manufacturer_name or ("ISO Manufacturer Symbol" if "manufacturer" in symbol_names else "MISSING"),
                expected_condition="Legal manufacturer name or factory symbol must be present",
                message="Manufacturer identification verified." if passed else "Mandatory manufacturer identification is missing.",
                remediation_hint=r_mand2.remediation_hint if not passed else None
            ))

        # 3. MAND-003: Expiry Date Presence
        r_mand3 = self.rules.get("MAND-003")
        if r_mand3:
            has_text_exp = bool(fields.expiry_date)
            has_udi_exp = bool(udi and udi.expiry_date_pi)
            passed = has_text_exp or has_udi_exp
            observed = fields.expiry_date or (udi.expiry_date_pi if udi else None) or "MISSING"
            checks.append(RuleCheckOutcome(
                rule_id=r_mand3.rule_id,
                rule_name=r_mand3.name,
                category=r_mand3.category,
                standard=r_mand3.standard,
                severity=r_mand3.severity,
                passed=passed,
                observed_value=observed,
                expected_condition="Clear expiration / use-by date must be present",
                message=f"Expiration date '{observed}' identified." if passed else "Mandatory expiration / use-by date is missing.",
                remediation_hint=r_mand3.remediation_hint if not passed else None
            ))

        # 4. MAND-004: Batch / Lot Number Presence
        r_mand4 = self.rules.get("MAND-004")
        if r_mand4:
            has_text_lot = bool(fields.lot_number)
            has_udi_lot = bool(udi and udi.lot_number_pi)
            passed = has_text_lot or has_udi_lot
            observed = fields.lot_number or (udi.lot_number_pi if udi else None) or "MISSING"
            checks.append(RuleCheckOutcome(
                rule_id=r_mand4.rule_id,
                rule_name=r_mand4.name,
                category=r_mand4.category,
                standard=r_mand4.standard,
                severity=r_mand4.severity,
                passed=passed,
                observed_value=observed,
                expected_condition="Lot or batch number must be present for device traceability",
                message=f"Batch/Lot '{observed}' identified." if passed else "Mandatory batch or lot number is missing.",
                remediation_hint=r_mand4.remediation_hint if not passed else None
            ))

        # 5. MAND-005: Warnings & Precautions
        r_mand5 = self.rules.get("MAND-005")
        if r_mand5:
            # If product is sterile or single use, warning/caution is critical
            has_warnings = len(fields.warnings) > 0 or "caution" in symbol_names or fields.is_rx_only or fields.is_single_use
            checks.append(RuleCheckOutcome(
                rule_id=r_mand5.rule_id,
                rule_name=r_mand5.name,
                category=r_mand5.category,
                standard=r_mand5.standard,
                severity=r_mand5.severity,
                passed=has_warnings,
                observed_value=f"{len(fields.warnings)} warning statements found" if fields.warnings else ("Caution/Rx symbol present" if has_warnings else "NONE"),
                expected_condition="Device safety precautions, warnings, or caution statements required",
                message="Safety warnings or precautionary indicators verified." if has_warnings else "No safety warnings or precautionary information detected on label.",
                remediation_hint=r_mand5.remediation_hint if not has_warnings else None
            ))

        # 6. MAND-006: Storage Conditions
        r_mand6 = self.rules.get("MAND-006")
        if r_mand6:
            has_storage = bool(fields.storage_conditions or "temperature_limit" in symbol_names or (fields.temperature_max_c is not None))
            observed = fields.storage_conditions or ("Temperature symbol detected" if "temperature_limit" in symbol_names else "NONE")
            checks.append(RuleCheckOutcome(
                rule_id=r_mand6.rule_id,
                rule_name=r_mand6.name,
                category=r_mand6.category,
                standard=r_mand6.standard,
                severity=r_mand6.severity,
                passed=has_storage,
                observed_value=observed,
                expected_condition="Storage temperature or handling condition instructions required",
                message="Storage instructions verified." if has_storage else "No storage conditions or temperature limits specified.",
                remediation_hint=r_mand6.remediation_hint if not has_storage else None
            ))

        # 7. DATE-001: Expiration Date Future Validity
        r_date1 = self.rules.get("DATE-001")
        if r_date1 and fields.expiry_date:
            exp_dt = parse_date_string(fields.expiry_date)
            if exp_dt:
                now = datetime.now()
                is_future = exp_dt > now
                checks.append(RuleCheckOutcome(
                    rule_id=r_date1.rule_id,
                    rule_name=r_date1.name,
                    category=r_date1.category,
                    standard=r_date1.standard,
                    severity=r_date1.severity,
                    passed=is_future,
                    observed_value=fields.expiry_date,
                    expected_condition="Expiration date must be in the future relative to current date",
                    message="Expiration date is currently valid." if is_future else f"CRITICAL: Device has expired! Expiration date '{fields.expiry_date}' is in the past.",
                    remediation_hint=r_date1.remediation_hint if not is_future else None
                ))

        # 8. DATE-002: Chronological Date Order
        r_date2 = self.rules.get("DATE-002")
        if r_date2 and fields.expiry_date and fields.manufacturing_date:
            exp_dt = parse_date_string(fields.expiry_date)
            mfg_dt = parse_date_string(fields.manufacturing_date)
            if exp_dt and mfg_dt:
                order_ok = exp_dt > mfg_dt
                checks.append(RuleCheckOutcome(
                    rule_id=r_date2.rule_id,
                    rule_name=r_date2.name,
                    category=r_date2.category,
                    standard=r_date2.standard,
                    severity=r_date2.severity,
                    passed=order_ok,
                    observed_value=f"MFG: {fields.manufacturing_date} | EXP: {fields.expiry_date}",
                    expected_condition="Manufacturing date must strictly precede expiration date",
                    message="Date sequence verified." if order_ok else "CRITICAL: Manufacturing date is later than or equal to expiration date.",
                    remediation_hint=r_date2.remediation_hint if not order_ok else None
                ))

        # 9. UDI-001: UDI Barcode Carrier Presence
        r_udi1 = self.rules.get("UDI-001")
        if r_udi1:
            has_carrier = bool(barcodes or (udi and udi.device_identifier_di))
            checks.append(RuleCheckOutcome(
                rule_id=r_udi1.rule_id,
                rule_name=r_udi1.name,
                category=r_udi1.category,
                standard=r_udi1.standard,
                severity=r_udi1.severity,
                passed=has_carrier,
                observed_value=f"{len(barcodes)} barcode(s) scanned" if barcodes else ("UDI decoded" if udi else "NONE"),
                expected_condition="Machine-readable UDI barcode carrier must be present on label",
                message="UDI barcode carrier successfully detected." if has_carrier else "No UDI barcode carrier detected on label.",
                remediation_hint=r_udi1.remediation_hint if not has_carrier else None
            ))

        # 10. UDI-002: UDI DI Checksum Validity
        r_udi2 = self.rules.get("UDI-002")
        if r_udi2 and udi:
            checksum_ok = udi.is_di_checksum_valid
            checks.append(RuleCheckOutcome(
                rule_id=r_udi2.rule_id,
                rule_name=r_udi2.name,
                category=r_udi2.category,
                standard=r_udi2.standard,
                severity=r_udi2.severity,
                passed=checksum_ok,
                observed_value=f"DI: {udi.device_identifier_di} (Checksum {'Valid' if checksum_ok else 'Invalid'})",
                expected_condition="UDI Device Identifier must pass GS1 Modulo-10 checksum validation",
                message="GS1 Modulo-10 checksum verified." if checksum_ok else f"CRITICAL: Invalid GTIN check digit in DI '{udi.device_identifier_di}'.",
                remediation_hint=r_udi2.remediation_hint if not checksum_ok else None
            ))

        # 11. UDI-003 & UDI-004: UDI Cross-Validation with Text
        if udi:
            validator = UDIValidator()
            discrepancies = validator.cross_validate_with_ocr(udi, fields)
            r_udi3 = self.rules.get("UDI-003")
            if r_udi3:
                lot_disc = next((d for d in discrepancies if d["field"] == "lot_number"), None)
                checks.append(RuleCheckOutcome(
                    rule_id=r_udi3.rule_id,
                    rule_name=r_udi3.name,
                    category=r_udi3.category,
                    standard=r_udi3.standard,
                    severity=r_udi3.severity,
                    passed=(lot_disc is None),
                    observed_value=f"UDI LOT: {udi.lot_number_pi} vs Text LOT: {fields.lot_number}",
                    expected_condition="UDI Barcode Lot number must match label text Lot number",
                    message="Lot number consistency verified across barcode and text." if not lot_disc else lot_disc["message"],
                    remediation_hint=r_udi3.remediation_hint if lot_disc else None
                ))

            r_udi4 = self.rules.get("UDI-004")
            if r_udi4:
                exp_disc = next((d for d in discrepancies if d["field"] == "expiry_date"), None)
                checks.append(RuleCheckOutcome(
                    rule_id=r_udi4.rule_id,
                    rule_name=r_udi4.name,
                    category=r_udi4.category,
                    standard=r_udi4.standard,
                    severity=r_udi4.severity,
                    passed=(exp_disc is None),
                    observed_value=f"UDI EXP: {udi.expiry_date_pi} vs Text EXP: {fields.expiry_date}",
                    expected_condition="UDI Barcode Expiration date must match label text Expiration date",
                    message="Expiration date consistency verified across barcode and text." if not exp_disc else exp_disc["message"],
                    remediation_hint=r_udi4.remediation_hint if exp_disc else None
                ))

        # 12. SYM-001: Sterile Claim Symbol Verification
        r_sym1 = self.rules.get("SYM-001")
        if r_sym1 and fields.is_sterile:
            has_sterile_sym = "sterile" in symbol_names
            checks.append(RuleCheckOutcome(
                rule_id=r_sym1.rule_id,
                rule_name=r_sym1.name,
                category=r_sym1.category,
                standard=r_sym1.standard,
                severity=r_sym1.severity,
                passed=has_sterile_sym,
                observed_value="ISO Sterile symbol detected" if has_sterile_sym else "Sterile symbol NOT detected",
                expected_condition="Devices claiming sterility must display the official ISO 15223-1 [STERILE] symbol",
                message="Sterile symbol presence verified." if has_sterile_sym else "Device text claims sterility, but ISO 15223-1 STERILE symbol was not detected on label artwork.",
                remediation_hint=r_sym1.remediation_hint if not has_sterile_sym else None
            ))

        # 13. SYM-002: Single-Use Symbol Verification
        r_sym2 = self.rules.get("SYM-002")
        if r_sym2 and fields.is_single_use:
            has_su_sym = "do_not_reuse" in symbol_names
            checks.append(RuleCheckOutcome(
                rule_id=r_sym2.rule_id,
                rule_name=r_sym2.name,
                category=r_sym2.category,
                standard=r_sym2.standard,
                severity=r_sym2.severity,
                passed=has_su_sym,
                observed_value="Do Not Re-use symbol detected" if has_su_sym else "Symbol NOT detected",
                expected_condition="Single-use devices must bear the ISO 15223-1 'Do Not Re-use' (2 with slash) symbol",
                message="Single-use symbol verified." if has_su_sym else "Label claims single use, but ISO 15223-1 Do Not Re-use symbol is missing.",
                remediation_hint=r_sym2.remediation_hint if not has_su_sym else None
            ))

        # 14. SYM-003: Temperature Limit Symbol Verification
        r_sym3 = self.rules.get("SYM-003")
        if r_sym3 and (fields.temperature_max_c is not None or fields.temperature_min_c is not None):
            has_temp_sym = "temperature_limit" in symbol_names
            checks.append(RuleCheckOutcome(
                rule_id=r_sym3.rule_id,
                rule_name=r_sym3.name,
                category=r_sym3.category,
                standard=r_sym3.standard,
                severity=r_sym3.severity,
                passed=has_temp_sym,
                observed_value="Temperature symbol detected" if has_temp_sym else "Symbol NOT detected",
                expected_condition="Devices with temperature restrictions must display ISO 15223-1 thermometer symbol",
                message="Temperature limit symbol verified." if has_temp_sym else "Temperature limits specified in text, but ISO temperature symbol is missing.",
                remediation_hint=r_sym3.remediation_hint if not has_temp_sym else None
            ))

        # 15. SYM-004: CE Mark Verification
        r_sym4 = self.rules.get("SYM-004")
        if r_sym4 and fields.ce_mark_number:
            has_ce_sym = "ce_mark" in symbol_names
            checks.append(RuleCheckOutcome(
                rule_id=r_sym4.rule_id,
                rule_name=r_sym4.name,
                category=r_sym4.category,
                standard=r_sym4.standard,
                severity=r_sym4.severity,
                passed=has_ce_sym,
                observed_value="CE mark symbol detected" if has_ce_sym else f"Text '{fields.ce_mark_number}' found but CE emblem not detected",
                expected_condition="CE mark emblem must be present on label artwork",
                message="CE conformity emblem verified." if has_ce_sym else "CE mark text stated, but CE visual emblem was not detected.",
                remediation_hint=r_sym4.remediation_hint if not has_ce_sym else None
            ))

        # Calculate metrics and score
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.passed)
        failed_checks = total_checks - passed_checks

        crit_count = sum(1 for c in checks if not c.passed and c.severity == RuleSeverity.CRITICAL)
        major_count = sum(1 for c in checks if not c.passed and c.severity == RuleSeverity.MAJOR)
        minor_count = sum(1 for c in checks if not c.passed and c.severity == RuleSeverity.MINOR)

        # Scoring algorithm
        # Penalty: Critical = 25, Major = 10, Minor = 3
        penalty = (crit_count * 25.0) + (major_count * 10.0) + (minor_count * 3.0)
        score = max(0.0, round(100.0 - penalty, 2))

        # Verdict
        if crit_count == 0 and major_count == 0 and score >= COMPLIANCE_SCORE_PASS:
            verdict = ComplianceVerdict.COMPLIANT
        elif crit_count == 0 and score >= COMPLIANCE_SCORE_CONDITIONAL:
            verdict = ComplianceVerdict.CONDITIONAL_PASS
        else:
            verdict = ComplianceVerdict.NON_COMPLIANT

        report = LabelComplianceReport(
            file_name=file_name,
            file_hash=file_hash,
            verdict=verdict,
            compliance_score=score,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_violations_count=crit_count,
            major_violations_count=major_count,
            minor_violations_count=minor_count,
            checks=checks,
            extracted_fields=fields.to_dict(),
            detected_symbols=[s.to_dict() for s in symbols],
            barcodes_scanned=[b.to_dict() for b in barcodes],
            udi_details=udi.to_dict() if udi else None,
            processing_time_ms=round(processing_time_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        return report
