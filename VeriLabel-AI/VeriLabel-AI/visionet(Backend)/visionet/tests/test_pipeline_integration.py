"""
VeriLabel AI - Integration & Functional Test Suite
Team: Brahmastra | GITAM University
Implements the test matrix specified in HLD Section 11:
TC01: All mandatory fields available -> Checks pass (COMPLIANT)
TC02: Expiry date missing -> Missing field detected (NON_COMPLIANT)
TC03: UDI format incorrect -> Format error detected (NON_COMPLIANT)
TC04: Required symbol missing -> Symbol warning generated (NON_COMPLIANT or CONDITIONAL)
TC05: Warning information missing -> Missing warning detected (NON_COMPLIANT)
TC06: Valid barcode present -> Barcode extracted and checked (Pass)
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from verilabel.pipeline import VeriLabelPipeline
from verilabel.modules.compliance.models import ComplianceVerdict, RuleSeverity
from verilabel.config import SAMPLE_DATA_DIR


class TestVeriLabelPipeline(unittest.TestCase):
    """End-to-end verification of medical device label compliance checks."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = VeriLabelPipeline()
        cls.labels_dir = SAMPLE_DATA_DIR / "labels"

    def test_tc01_compliant_label(self):
        """TC01: Fully compliant label with all mandatory data and valid UDI."""
        file_path = self.labels_dir / "TC01_compliant.png"
        result = self.pipeline.process_label(file_path, save_to_db=True)

        self.assertIsNotNone(result.report)
        self.assertEqual(result.report.verdict, ComplianceVerdict.COMPLIANT)
        self.assertGreaterEqual(result.report.compliance_score, 95.0)
        self.assertEqual(result.report.critical_violations_count, 0)
        self.assertEqual(result.report.major_violations_count, 0)

        # Verify extracted attributes
        self.assertEqual(result.fields.product_name, "CardioPulse Catheter System 6Fr")
        self.assertIsNotNone(result.fields.expiry_date)
        self.assertEqual(result.fields.lot_number, "BATCH-8842")

        # Verify UDI
        self.assertIsNotNone(result.udi)
        self.assertTrue(result.udi.is_di_checksum_valid)

    def test_tc02_missing_expiry(self):
        """TC02: Missing expiration date must trigger CRITICAL defect."""
        file_path = self.labels_dir / "TC02_missing_expiry.png"
        result = self.pipeline.process_label(file_path, save_to_db=True)

        self.assertIsNotNone(result.report)
        self.assertEqual(result.report.verdict, ComplianceVerdict.NON_COMPLIANT)
        self.assertGreater(result.report.critical_violations_count, 0)

        # Check that MAND-003 failed
        mand3_check = next((c for c in result.report.checks if c.rule_id == "MAND-003"), None)
        self.assertIsNotNone(mand3_check)
        self.assertFalse(mand3_check.passed)
        self.assertEqual(mand3_check.severity, RuleSeverity.CRITICAL)

    def test_tc03_invalid_udi_checksum(self):
        """TC03: Corrupted UDI check digit must trigger UDI-002 failure."""
        file_path = self.labels_dir / "TC03_invalid_udi.png"
        result = self.pipeline.process_label(file_path, save_to_db=True)

        self.assertIsNotNone(result.report)
        self.assertEqual(result.report.verdict, ComplianceVerdict.NON_COMPLIANT)

        # Check UDI-002 failed
        udi2_check = next((c for c in result.report.checks if c.rule_id == "UDI-002"), None)
        self.assertIsNotNone(udi2_check)
        self.assertFalse(udi2_check.passed)
        self.assertEqual(udi2_check.severity, RuleSeverity.CRITICAL)

    def test_tc04_missing_symbol(self):
        """TC04: Sterile text claim without Sterile symbol must trigger SYM-001 warning."""
        file_path = self.labels_dir / "TC04_missing_symbol.png"
        result = self.pipeline.process_label(file_path, save_to_db=True)

        self.assertIsNotNone(result.report)
        sym1_check = next((c for c in result.report.checks if c.rule_id == "SYM-001"), None)
        self.assertIsNotNone(sym1_check)
        self.assertFalse(sym1_check.passed)
        self.assertEqual(sym1_check.severity, RuleSeverity.MAJOR)

    def test_tc05_missing_warnings(self):
        """TC05: Omitted safety warnings must trigger MAND-005 failure."""
        file_path = self.labels_dir / "TC05_missing_warnings.png"
        result = self.pipeline.process_label(file_path, save_to_db=True)

        self.assertIsNotNone(result.report)
        mand5_check = next((c for c in result.report.checks if c.rule_id == "MAND-005"), None)
        self.assertIsNotNone(mand5_check)
        self.assertFalse(mand5_check.passed)
        self.assertEqual(mand5_check.severity, RuleSeverity.CRITICAL)

    def test_tc06_valid_barcode_presence(self):
        """TC06: Valid barcode extracted and verified."""
        file_path = self.labels_dir / "TC01_compliant.png"
        result = self.pipeline.process_label(file_path, save_to_db=False)

        self.assertGreater(len(result.barcodes), 0)
        self.assertIsNotNone(result.udi)
        self.assertTrue(result.udi.is_valid_format)

    def test_human_review_workflow(self):
        """Verifies recording human expert review sign-off."""
        file_path = self.labels_dir / "TC01_compliant.png"
        result = self.pipeline.process_label(file_path, save_to_db=True)

        review_id = self.pipeline.submit_human_review(
            validation_id=result.validation_id,
            reviewer_name="Dr. A. Sharma",
            reviewer_role="Lead Regulatory Affairs Specialist",
            decision="APPROVED",
            review_notes="Label meets all FDA 21 CFR 801 and ISO 15223-1 requirements."
        )
        self.assertGreater(review_id, 0)

        details = self.pipeline.get_validation_details(result.validation_id)
        self.assertEqual(details["latest_review_status"], "APPROVED")
        self.assertEqual(len(details["reviews"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
