"""
VeriLabel AI - Module 7: Database Manager
Team: Brahmastra | GITAM University
Provides clean transactional operations, data persistence, and traceability queries.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session

from verilabel.config import DATABASE_PATH
from verilabel.database.models import (
    Base,
    ProductReference,
    LabelUpload,
    ValidationRun,
    ViolationLog,
    HumanReview
)
from verilabel.modules.compliance.models import LabelComplianceReport, RuleCheckOutcome
from verilabel.exceptions import DatabaseError

logger = logging.getLogger("VeriLabel.Database")


class DatabaseManager:
    """Manages SQLite connections, migrations, and audit transactions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def create_tables(self):
        """Creates database schema if not present."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database schema initialized at {self.db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database tables: {e}") from e

    def get_session(self) -> Session:
        """Returns a new database session."""
        return self.SessionLocal()

    def save_validation_run(
        self,
        file_meta: Dict[str, Any],
        report: LabelComplianceReport,
        product_code: Optional[str] = None
    ) -> int:
        """
        Atomically records file upload metadata, compliance validation run,
        and all detected violations into the database.
        Returns the new validation_id.
        """
        session = self.get_session()
        try:
            # 1. Record Upload
            upload = LabelUpload(
                file_name=file_meta.get("file_name", report.file_name),
                file_hash=file_meta.get("sha256", report.file_hash),
                file_size_kb=file_meta.get("file_size_kb", 0.0),
                extension=file_meta.get("extension", ".png"),
                image_width=file_meta.get("width"),
                image_height=file_meta.get("height"),
                uploaded_at=datetime.now(timezone.utc)
            )
            session.add(upload)
            session.flush()

            # 2. Match Product Reference if provided
            product_id = None
            if product_code:
                prod = session.query(ProductReference).filter_by(product_code=product_code).first()
                if prod:
                    product_id = prod.id

            # 3. Record Validation Run
            val_run = ValidationRun(
                upload_id=upload.id,
                product_id=product_id,
                verdict=report.verdict.value,
                compliance_score=report.compliance_score,
                total_checks=report.total_checks,
                passed_checks=report.passed_checks,
                failed_checks=report.failed_checks,
                critical_count=report.critical_violations_count,
                major_count=report.major_violations_count,
                minor_count=report.minor_violations_count,
                processing_time_ms=report.processing_time_ms,
                report_json=report.model_dump_json(),
                created_at=datetime.now(timezone.utc)
            )
            session.add(val_run)
            session.flush()

            # 4. Record Granular Violations
            for check in report.checks:
                if not check.passed:
                    violation = ViolationLog(
                        validation_id=val_run.id,
                        rule_id=check.rule_id,
                        rule_name=check.rule_name,
                        category=check.category,
                        severity=check.severity.value,
                        observed_value=check.observed_value,
                        expected_condition=check.expected_condition,
                        message=check.message,
                        remediation_hint=check.remediation_hint
                    )
                    session.add(violation)

            session.commit()
            report.label_id = val_run.id
            logger.info(f"Validation run #{val_run.id} saved with verdict {report.verdict.value}")
            return val_run.id

        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to persist validation record: {e}") from e
        finally:
            session.close()

    def record_human_review(
        self,
        validation_id: int,
        reviewer_name: str,
        reviewer_role: str,
        decision: str,
        review_notes: Optional[str] = None
    ) -> int:
        """
        Records human expert sign-off decision (APPROVED, REJECTED, CHANGES_REQUESTED).
        """
        valid_decisions = {"APPROVED", "REJECTED", "CHANGES_REQUESTED"}
        dec_upper = decision.strip().upper()
        if dec_upper not in valid_decisions:
            raise ValueError(f"Invalid decision '{decision}'. Must be one of: {valid_decisions}")

        session = self.get_session()
        try:
            val_run = session.query(ValidationRun).filter_by(id=validation_id).first()
            if not val_run:
                raise DatabaseError(f"ValidationRun ID {validation_id} does not exist.")

            review = HumanReview(
                validation_id=validation_id,
                reviewer_name=reviewer_name.strip(),
                reviewer_role=reviewer_role.strip(),
                decision=dec_upper,
                review_notes=review_notes.strip() if review_notes else None,
                reviewed_at=datetime.now(timezone.utc)
            )
            session.add(review)
            session.commit()
            logger.info(f"Human review logged for validation #{validation_id}: {dec_upper} by {reviewer_name}")
            return review.id

        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Failed to record human review: {e}") from e
        finally:
            session.close()

    def get_validation_record(self, validation_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a validation run by ID along with violations and human reviews."""
        session = self.get_session()
        try:
            val = session.query(ValidationRun).filter_by(id=validation_id).first()
            if not val:
                return None

            report_data = json.loads(val.report_json)

            # Get human reviews
            reviews = [
                {
                    "id": r.id,
                    "reviewer_name": r.reviewer_name,
                    "reviewer_role": r.reviewer_role,
                    "decision": r.decision,
                    "notes": r.review_notes,
                    "timestamp": r.reviewed_at.isoformat()
                }
                for r in val.reviews
            ]

            return {
                "validation_id": val.id,
                "file_name": val.upload.file_name,
                "file_hash": val.upload.file_hash,
                "verdict": val.verdict,
                "compliance_score": val.compliance_score,
                "created_at": val.created_at.isoformat(),
                "processing_time_ms": val.processing_time_ms,
                "report": report_data,
                "reviews": reviews,
                "latest_review_status": reviews[-1]["decision"] if reviews else "PENDING_REVIEW"
            }
        finally:
            session.close()

    def list_validations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent validation runs for dashboard display."""
        session = self.get_session()
        try:
            runs = (
                session.query(ValidationRun)
                .order_by(desc(ValidationRun.created_at))
                .limit(limit)
                .all()
            )
            output = []
            for r in runs:
                latest_decision = r.reviews[-1].decision if r.reviews else "PENDING_REVIEW"
                output.append({
                    "validation_id": r.id,
                    "file_name": r.upload.file_name,
                    "verdict": r.verdict,
                    "compliance_score": r.compliance_score,
                    "critical_errors": r.critical_count,
                    "major_warnings": r.major_count,
                    "latest_review_status": latest_decision,
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            return output
        finally:
            session.close()

    def seed_demo_products(self):
        """Seeds baseline reference products into the database."""
        session = self.get_session()
        try:
            if session.query(ProductReference).count() == 0:
                demo_prods = [
                    ProductReference(
                        product_code="DEV-CARDIO-01",
                        trade_name="CardioPulse Catheter System",
                        manufacturer="Brahmastra MedTech Corp.",
                        gtin_di="00847239012345",
                        is_sterile=True,
                        is_single_use=True,
                        temp_min_c=2.0,
                        temp_max_c=25.0,
                        expected_symbols='["sterile", "do_not_reuse", "manufacturer", "use_by_date", "batch_code", "ce_mark"]'
                    ),
                    ProductReference(
                        product_code="DEV-ORTHO-02",
                        trade_name="Titanium Bone Screw Set 3.5mm",
                        manufacturer="Apex Orthopedics Ltd.",
                        gtin_di="00847239056789",
                        is_sterile=True,
                        is_single_use=True,
                        temp_min_c=None,
                        temp_max_c=30.0,
                        expected_symbols='["sterile", "do_not_reuse", "catalogue_number", "batch_code", "ce_mark"]'
                    )
                ]
                session.add_all(demo_prods)
                session.commit()
                logger.info("Seeded demo product references.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding demo products: {e}")
        finally:
            session.close()
