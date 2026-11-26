"""
VeriLabel AI - End-to-End Platform Demonstration
Team: Brahmastra | GITAM University
Demonstrates the full medical device label verification workflow.
"""

import sys
import json
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from verilabel.pipeline import VeriLabelPipeline
from verilabel.modules.reporting.generator import ReportGenerator


def main():
    print("\n" + "=" * 80)
    print(" VERILABEL AI - MEDICAL DEVICE LABEL COMPLIANCE PLATFORM")
    print(" Team: Brahmastra | GITAM University | Backend Demonstration")
    print("=" * 80)

    pipeline = VeriLabelPipeline()
    sample_label = BASE_DIR / "sample_data" / "labels" / "TC01_compliant.png"

    if not sample_label.exists():
        print(f"Sample label not found at {sample_label}. Generating synthetic labels first...")
        from tests.generate_synthetic_labels import generate_all_test_cases
        generate_all_test_cases()

    print(f"\n[1] Ingesting label document: {sample_label.name}")
    print("    Running Preprocessing -> OCR -> Field Extraction -> Symbol Detection -> UDI Validation...")

    result = pipeline.process_label(sample_label, product_code="DEV-CARDIO-01", save_to_db=True)

    print("\n" + "=" * 80)
    print(" [2] EXECUTIVE COMPLIANCE SUMMARY")
    print("=" * 80)
    summary = result.summary
    print(f"  • File Analyzed        : {summary['file_name']}")
    print(f"  • Compliance Verdict   : {summary['verdict']}")
    print(f"  • Compliance Score     : {summary['compliance_score']}%")
    print(f"  • Passed Checks        : {summary['passed_checks']} / {summary['total_checks']} ({summary['pass_rate_percent']}%)")
    print(f"  • Critical Violations  : {summary['critical_errors']}")
    print(f"  • Major Warnings       : {summary['major_warnings']}")
    print(f"  • Processing Latency   : {summary['processing_time_ms']:.2f} ms")

    print("\n" + "=" * 80)
    print(" [3] EXTRACTED REGULATORY ATTRIBUTES")
    print("=" * 80)
    fields = result.fields
    print(f"  • Product Name         : {fields.product_name}")
    print(f"  • Manufacturer         : {fields.manufacturer_name}")
    print(f"  • Expiration Date      : {fields.expiry_date}")
    print(f"  • Manufacturing Date   : {fields.manufacturing_date}")
    print(f"  • Batch / Lot Number   : {fields.lot_number}")
    print(f"  • Catalogue / REF      : {fields.catalog_number}")
    print(f"  • Storage Condition    : {fields.storage_conditions}")
    print(f"  • Sterility Status     : {'STERILE (' + str(fields.sterilization_method) + ')' if fields.is_sterile else 'NON-STERILE'}")
    print(f"  • Single-Use Only      : {'YES' if fields.is_single_use else 'NO'}")
    print(f"  • CE Mark ID           : {fields.ce_mark_number or 'N/A'}")
    print(f"  • Prescription Status  : {'Rx Only (Prescription Device)' if fields.is_rx_only else 'OTC / Standard'}")

    print("\n" + "=" * 80)
    print(" [4] COMPUTER VISION & BARCODE ASSESSMENT")
    print("=" * 80)
    print(f"  • Detected ISO Symbols : {len(result.symbols)} symbols recognized")
    for s in result.symbols[:6]:
        print(f"    - {s.display_name} (Code: {s.iso_code}, Conf: {s.confidence*100:.1f}%)")

    if result.udi:
        print(f"  • UDI Device Identifier: {result.udi.device_identifier_di} (Checksum: {'VALID' if result.udi.is_di_checksum_valid else 'INVALID'})")
        print(f"  • UDI Lot PI           : {result.udi.lot_number_pi}")
        print(f"  • UDI Expiry PI        : {result.udi.expiry_date_pi}")

    print("\n" + "=" * 80)
    print(" [5] COMPLIANCE RULE CHECKLIST")
    print("=" * 80)
    print(f"{'Rule ID':<10} | {'Requirement':<32} | {'Severity':<8} | {'Status':<6} | {'Observed Value'}")
    print("-" * 80)
    for c in result.report.checks:
        status_str = "PASS" if c.passed else "FAIL"
        obs = (c.observed_value or "N/A")[:20]
        print(f"{c.rule_id:<10} | {c.rule_name[:32]:<32} | {c.severity.value:<8} | {status_str:<6} | {obs}")

    print("\n" + "=" * 80)
    print(" [6] HUMAN EXPERT WORKFLOW SIGN-OFF")
    print("=" * 80)
    print("  Submitting Qualified Expert Approval into SQLite Audit Ledger...")
    review_id = pipeline.submit_human_review(
        validation_id=result.validation_id,
        reviewer_name="Dr. Sarah Jenkins",
        reviewer_role="Director of Regulatory Affairs & Quality Assurance",
        decision="APPROVED",
        review_notes="Label satisfies 21 CFR 801, 830 and EU MDR Annex I. Barcode format and GTIN checksum verified."
    )
    print(f"  Audit Review #{review_id} logged successfully with status: APPROVED")

    print("\n" + "=" * 80)
    print(" [7] AUDIT TRAIL QUERY")
    print("=" * 80)
    history = pipeline.get_audit_history(limit=5)
    for h in history:
        print(f"  Run #{h['validation_id']} | {h['file_name']:<22} | Verdict: {h['verdict']:<16} | Score: {h['compliance_score']}% | Review: {h['latest_review_status']}")

    # Export reports
    export_dir = BASE_DIR / "sample_data" / "reports"
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_file = export_dir / "audit_report.csv"
    json_file = export_dir / "audit_report.json"
    ReportGenerator.export_csv(result.report, csv_file)
    ReportGenerator.export_json(result.report, json_file)
    print(f"\n[8] Reports Exported:")
    print(f"  • CSV Report  : {csv_file}")
    print(f"  • JSON Report : {json_file}")
    print("\n" + "=" * 80)
    print(" VERILABEL AI BACKEND DEMONSTRATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
