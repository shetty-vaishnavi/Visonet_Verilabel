"""
VeriLabel AI - Compliance Models & Schemas
Team: Brahmastra | GITAM University
Pydantic schemas for compliance rules, violations, and validation outcomes.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RuleSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class ComplianceVerdict(str, Enum):
    COMPLIANT = "COMPLIANT"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    NON_COMPLIANT = "NON_COMPLIANT"


class ComplianceRuleDefinition(BaseModel):
    """Schema for a regulatory compliance rule definition."""
    rule_id: str
    category: str  # e.g., "MANDATORY_FIELDS", "UDI_VALIDATION", "SYMBOL_VERIFICATION", "DATE_INTEGRITY"
    standard: str  # e.g., "FDA 21 CFR 801", "EU MDR 2017/745", "ISO 15223-1"
    name: str
    description: str
    severity: RuleSeverity
    remediation_hint: str


class RuleCheckOutcome(BaseModel):
    """Result of evaluating a single rule against a label."""
    rule_id: str
    rule_name: str
    category: str
    standard: str
    severity: RuleSeverity
    passed: bool
    observed_value: Optional[str] = None
    expected_condition: str
    message: str
    remediation_hint: Optional[str] = None


class LabelComplianceReport(BaseModel):
    """Full comprehensive compliance report generated for a medical device label."""
    label_id: Optional[int] = None
    file_name: str
    file_hash: str
    verdict: ComplianceVerdict
    compliance_score: float = Field(ge=0.0, le=100.0)
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_violations_count: int
    major_violations_count: int
    minor_violations_count: int
    checks: List[RuleCheckOutcome]
    extracted_fields: Dict[str, Any]
    detected_symbols: List[Dict[str, Any]]
    barcodes_scanned: List[Dict[str, Any]]
    udi_details: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    timestamp: str
