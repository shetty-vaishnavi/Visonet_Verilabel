"""
VeriLabel AI - Module 6: Report Generation Engine
Team: Brahmastra | GITAM University
Produces exportable compliance reports in multiple formats:
- Pandas DataFrames (for Streamlit UI tables)
- Markdown / HTML (for downloadable / printable audit reports)
- JSON & CSV (for enterprise integrations & automated pipelines)
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd

from verilabel.modules.compliance.models import LabelComplianceReport, RuleSeverity


class ReportGenerator:
    """
    Transforms raw LabelComplianceReport into executive summaries,
    presentation-ready DataFrames, Markdown, and CSV files.
    """

    @staticmethod
    def to_dataframe(report: LabelComplianceReport) -> pd.DataFrame:
        """
        Converts the checks array into a Pandas DataFrame formatted for Streamlit data tables.
        """
        rows = []
        for c in report.checks:
            status_symbol = "PASSED" if c.passed else f"FAILED ({c.severity.value})"
            rows.append({
                "Rule ID": c.rule_id,
                "Check Name": c.rule_name,
                "Category": c.category,
                "Standard": c.standard,
                "Severity": c.severity.value,
                "Status": status_symbol,
                "Observed Value": c.observed_value or "N/A",
                "Message": c.message,
                "Suggested Remediation": c.remediation_hint or "None"
            })

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def to_executive_summary(report: LabelComplianceReport) -> Dict[str, Any]:
        """Extracts executive KPI metrics for dashboard summary cards."""
        pass_rate = round((report.passed_checks / report.total_checks) * 100.0, 1) if report.total_checks > 0 else 0.0

        return {
            "file_name": report.file_name,
            "verdict": report.verdict.value,
            "compliance_score": report.compliance_score,
            "pass_rate_percent": pass_rate,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "critical_errors": report.critical_violations_count,
            "major_warnings": report.major_violations_count,
            "minor_warnings": report.minor_violations_count,
            "processing_time_ms": report.processing_time_ms,
            "timestamp": report.timestamp
        }

    @staticmethod
    def to_markdown(report: LabelComplianceReport) -> str:
        """Generates a complete, publication-quality Markdown compliance report."""
        summary = ReportGenerator.to_executive_summary(report)

        verdict_badge = {
            "COMPLIANT": "PASS - COMPLIANT",
            "CONDITIONAL_PASS": "CONDITIONAL PASS",
            "NON_COMPLIANT": "CRITICAL NON-COMPLIANCE"
        }.get(report.verdict.value, report.verdict.value)

        lines = [
            f"# VeriLabel AI - Medical Device Label Compliance Audit Report",
            f"**Platform:** VeriLabel AI Compliance Verification Platform",
            f"**Team:** Brahmastra | GITAM (Deemed to be University), Bengaluru",
            f"**Generated At:** {report.timestamp}",
            f"**File Evaluated:** `{report.file_name}` (SHA-256: `{report.file_hash[:16]}...`)",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| **Overall Verdict** | **{verdict_badge}** |",
            f"| **Compliance Score** | **{report.compliance_score:.1f}%** |",
            f"| **Checks Passed** | {report.passed_checks} / {report.total_checks} ({summary['pass_rate_percent']}%) |",
            f"| **Critical Violations** | {report.critical_violations_count} |",
            f"| **Major Warnings** | {report.major_violations_count} |",
            f"| **Minor Advisories** | {report.minor_violations_count} |",
            f"| **Processing Latency** | {report.processing_time_ms:.2f} ms |",
            "",
            "---",
            "",
            "## 2. Extracted Regulatory Information",
            "",
            "| Field | Extracted Label Value |",
            "| :--- | :--- |"
        ]

        fields = report.extracted_fields
        field_labels = [
            ("product_name", "Product / Device Name"),
            ("manufacturer_name", "Manufacturer"),
            ("manufacturer_address", "Manufacturer Address"),
            ("manufacturing_date", "Manufacturing Date (MFG)"),
            ("expiry_date", "Use-By / Expiration Date (EXP)"),
            ("lot_number", "Batch / Lot Number (LOT)"),
            ("serial_number", "Serial Number (SN)"),
            ("catalog_number", "Catalogue / REF Number"),
            ("storage_conditions", "Storage Conditions"),
            ("is_sterile", "Sterile Claim"),
            ("is_single_use", "Single Use Claim"),
            ("ce_mark_number", "CE Conformity ID"),
            ("is_rx_only", "Prescription Only (Rx)")
        ]

        for key, label in field_labels:
            val = fields.get(key)
            if val is not None:
                lines.append(f"| **{label}** | {val} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Detected ISO 15223-1 Symbols",
            "",
            "| Symbol Name | Standard Code | Confidence | Bounding Box (x, y, w, h) |",
            "| :--- | :--- | :--- | :--- |"
        ])

        if report.detected_symbols:
            for s in report.detected_symbols:
                bb = s["bbox"]
                bbox_str = f"({bb['x']}, {bb['y']}, {bb['width']}, {bb['height']})"
                lines.append(f"| {s['display_name']} | {s['iso_code']} | {s['confidence']*100:.1f}% | {bbox_str} |")
        else:
            lines.append("| *No standard ISO symbols detected* | - | - | - |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Barcode & UDI Integrity Assessment",
            ""
        ])

        if report.udi_details and report.udi_details.get("device_identifier_di"):
            udi = report.udi_details
            lines.extend([
                f"- **Carrier Format:** {udi.get('standard', 'GS1')}",
                f"- **Device Identifier (DI):** `{udi.get('device_identifier_di')}`",
                f"- **GTIN Checksum Valid:** {'YES' if udi.get('is_di_checksum_valid') else 'NO - CHECKSUM ERROR'}",
                f"- **Production Identifiers (PI):**",
                f"  - Lot Number: `{udi.get('lot_number_pi') or 'N/A'}`",
                f"  - Expiration Date: `{udi.get('expiry_date_pi') or 'N/A'}`",
                f"  - Serial Number: `{udi.get('serial_number_pi') or 'N/A'}`",
            ])
            if udi.get("validation_errors"):
                lines.append(f"- **UDI Validation Errors:** {', '.join(udi['validation_errors'])}")
        elif report.barcodes_scanned:
            lines.append(f"Scanned {len(report.barcodes_scanned)} barcode(s):")
            for b in report.barcodes_scanned:
                lines.append(f"- `{b['type']}`: `{b['data']}`")
        else:
            lines.append("⚠️ **No machine-readable UDI barcode found on label.**")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Detailed Compliance Checklist & Remediation Matrix",
            "",
            "| Rule ID | Requirement | Severity | Status | Observed Value | Suggested Remediation |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])

        for c in report.checks:
            badge = "PASS" if c.passed else f"FAIL ({c.severity.value})"
            obs = c.observed_value.replace("|", "/") if c.observed_value else "N/A"
            rem = c.remediation_hint.replace("|", "/") if c.remediation_hint else "None"
            lines.append(f"| `{c.rule_id}` | {c.rule_name} | {c.severity.value} | {badge} | {obs} | {rem} |")

        lines.extend([
            "",
            "---",
            "",
            "## 6. Human Expert Review & Traceability",
            "",
            "- **Review Requirement:** Under FDA 21 CFR and EU MDR guidelines, AI-generated compliance reports require human expert review prior to commercial release.",
            "- **Audit Record:** All findings, file hashes, and evaluation logs are persistently indexed in SQLite audit history.",
            "",
            "*End of VeriLabel AI Audit Report.*"
        ])

        return "\n".join(lines)

    @staticmethod
    def export_csv(report: LabelComplianceReport, output_path: Union[str, Path]) -> Path:
        """Exports compliance checks as CSV file."""
        df = ReportGenerator.to_dataframe(report)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def export_json(report: LabelComplianceReport, output_path: Union[str, Path]) -> Path:
        """Exports full report as JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        return path
