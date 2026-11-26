"""
VeriLabel AI - Module 7: Database Models (SQLAlchemy ORM)
Team: Brahmastra | GITAM University
Schema for medical device label audits, compliance validation runs,
traceability, and human expert approvals.
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def get_utc_now():
    return datetime.now(timezone.utc)


class ProductReference(Base):
    """Master repository of registered medical devices and their baseline specifications."""
    __tablename__ = "product_references"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_code = Column(String(64), unique=True, nullable=False, index=True)
    trade_name = Column(String(255), nullable=False)
    manufacturer = Column(String(255), nullable=False)
    gtin_di = Column(String(14), nullable=True, index=True)
    is_sterile = Column(Boolean, default=False)
    is_single_use = Column(Boolean, default=True)
    temp_min_c = Column(Float, nullable=True)
    temp_max_c = Column(Float, nullable=True)
    expected_symbols = Column(Text, nullable=True)  # JSON or comma-separated
    created_at = Column(DateTime, default=get_utc_now)

    validations = relationship("ValidationRun", back_populates="product")


class LabelUpload(Base):
    """Audit log of all uploaded medical label documents."""
    __tablename__ = "label_uploads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256
    file_size_kb = Column(Float, nullable=False)
    extension = Column(String(16), nullable=False)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=get_utc_now, nullable=False)

    validations = relationship("ValidationRun", back_populates="upload", cascade="all, delete-orphan")


class ValidationRun(Base):
    """Historical records of compliance evaluation executions."""
    __tablename__ = "validation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    upload_id = Column(Integer, ForeignKey("label_uploads.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("product_references.id"), nullable=True)

    verdict = Column(String(32), nullable=False)  # COMPLIANT, CONDITIONAL_PASS, NON_COMPLIANT
    compliance_score = Column(Float, nullable=False)
    total_checks = Column(Integer, nullable=False)
    passed_checks = Column(Integer, nullable=False)
    failed_checks = Column(Integer, nullable=False)
    critical_count = Column(Integer, default=0)
    major_count = Column(Integer, default=0)
    minor_count = Column(Integer, default=0)
    processing_time_ms = Column(Float, nullable=False)
    report_json = Column(Text, nullable=False)  # Full serialized report
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    upload = relationship("LabelUpload", back_populates="validations")
    product = relationship("ProductReference", back_populates="validations")
    violations = relationship("ViolationLog", back_populates="validation", cascade="all, delete-orphan")
    reviews = relationship("HumanReview", back_populates="validation", cascade="all, delete-orphan")


class ViolationLog(Base):
    """Granular log of individual rule violations per validation run."""
    __tablename__ = "violation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    validation_id = Column(Integer, ForeignKey("validation_runs.id"), nullable=False, index=True)
    rule_id = Column(String(32), nullable=False, index=True)
    rule_name = Column(String(128), nullable=False)
    category = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)
    observed_value = Column(String(255), nullable=True)
    expected_condition = Column(Text, nullable=False)
    message = Column(Text, nullable=False)
    remediation_hint = Column(Text, nullable=True)

    validation = relationship("ValidationRun", back_populates="violations")


class HumanReview(Base):
    """Workflow table for qualified medical/regulatory expert review and final sign-off."""
    __tablename__ = "human_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    validation_id = Column(Integer, ForeignKey("validation_runs.id"), nullable=False, index=True)
    reviewer_name = Column(String(128), nullable=False)
    reviewer_role = Column(String(128), nullable=False)  # e.g., 'Regulatory Affairs Lead', 'QA Specialist'
    decision = Column(String(32), nullable=False)  # 'APPROVED', 'REJECTED', 'CHANGES_REQUESTED'
    review_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, default=get_utc_now, nullable=False)

    validation = relationship("ValidationRun", back_populates="reviews")
