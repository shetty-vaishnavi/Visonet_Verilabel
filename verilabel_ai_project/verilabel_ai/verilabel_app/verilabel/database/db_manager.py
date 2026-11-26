"""
VeriLabel AI - Module 7: Database Manager
Team: Brahmastra | GITAM University
Provides clean transactional operations, data persistence, and traceability queries.
"""

import json
import logging
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker, Session

from verilabel.config import DATABASE_PATH
from verilabel.database.models import (
    Base,
    ProductReference,
    LabelUpload,
    ValidationRun,
    ViolationLog,
    HumanReview,
    User
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

    @staticmethod
    def _hash_password(password: str) -> str:
        """Computes a SHA-256 hash with a fixed salt for prototype demo accounts."""
        salt = "verilabel_2026_brahmastra_salt"
        return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()

    def seed_default_users(self):
        """Seeds default auditor and inspector accounts for seamless demo login."""
        session = self.get_session()
        try:
            if session.query(User).count() == 0:
                default_users = [
                    User(
                        email="sarah.jenkins@brahmastra.medtech",
                        name="Dr. Sarah Jenkins",
                        role="Lead Regulatory Auditor",
                        organization="Brahmastra MedTech Corp.",
                        password_hash=self._hash_password("Auditor@2026")
                    ),
                    User(
                        email="alex.chen@gitam.edu",
                        name="Alex Chen",
                        role="QA Inspection Specialist",
                        organization="GITAM Bengaluru Medical AI Lab",
                        password_hash=self._hash_password("Auditor@2026")
                    ),
                    User(
                        email="marcus.vance@fda-review.org",
                        name="Marcus Vance",
                        role="Senior Compliance Officer",
                        organization="Global Health Regulatory Standards Board",
                        password_hash=self._hash_password("Auditor@2026")
                    )
                ]
                session.add_all(default_users)
                session.commit()
                logger.info("Seeded default demo auditor accounts.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding default users: {e}")
        finally:
            session.close()

    @staticmethod
    def _serialize_user(user: User, auth_provider: Optional[str] = None) -> Dict[str, Any]:
        """Builds a clean user payload without exposing internal local-account suffixes."""
        is_local_email = bool(user.email and user.email.endswith("@local.verilabel"))
        payload: Dict[str, Any] = {
            "id": user.id,
            "username": user.name,
            "name": user.name,
            "email": None if is_local_email else user.email,
            "role": user.role,
            "organization": user.organization or "",
        }
        if auth_provider:
            payload["auth_provider"] = auth_provider
        return payload

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Fetches an active user record by primary key."""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.id == int(user_id)).first()
            return self._serialize_user(user) if user else None
        finally:
            session.close()

    def create_user(
        self,
        email: Optional[str] = None,
        name: Optional[str] = None,
        role: str = "Regulatory Auditor",
        organization: Optional[str] = None,
        password: str = "",
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a new user account with username (>=3 chars) and password (>=6 chars)."""
        clean_username = (username or name or "").strip()
        if len(clean_username) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(password or "") < 6:
            raise ValueError("Password must be at least 6 characters.")

        session = self.get_session()
        try:
            existing_name = (
                session.query(User)
                .filter(func.lower(User.name) == clean_username.lower())
                .first()
            )
            if existing_name:
                raise ValueError("An account with this username already exists.")

            clean_email = (email or "").strip().lower()
            if clean_email:
                existing_email = session.query(User).filter(User.email == clean_email).first()
                if existing_email:
                    raise ValueError("An auditor account with this email address already exists.")
                stored_email = clean_email
            else:
                slug = "".join(
                    ch if (ch.isalnum() or ch in "._-") else "_"
                    for ch in clean_username.lower()
                ).strip("_") or "user"
                candidate_email = f"{slug}@local.verilabel"
                while session.query(User).filter(User.email == candidate_email).first():
                    candidate_email = f"{slug}_{secrets.token_hex(3)}@local.verilabel"
                stored_email = candidate_email

            user = User(
                email=stored_email,
                name=clean_username,
                role=(role or "Regulatory Auditor").strip(),
                organization=(organization or "").strip() or None,
                password_hash=self._hash_password(password),
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return self._serialize_user(user)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def authenticate_user(self, identifier: str, password: str) -> Optional[Dict[str, Any]]:
        """Verifies credentials by email or username (User.name) and returns user details if valid."""
        ident = (identifier or "").strip()
        if not ident or not password:
            return None
        ident_lower = ident.lower()
        target_hash = self._hash_password(password)
        session = self.get_session()
        try:
            user = session.query(User).filter(User.email == ident_lower).first()
            if user and user.password_hash == target_hash:
                return self._serialize_user(user)

            candidates = session.query(User).filter(func.lower(User.name) == ident_lower).all()
            for cand in candidates:
                if cand.password_hash == target_hash:
                    return self._serialize_user(cand)
            return None
        finally:
            session.close()

    def get_or_create_oauth_user(
        self, email: str, name: str, organization: str = "Google Account"
    ) -> Dict[str, Any]:
        """Finds or creates a real SQLite user record for a verified OAuth identity."""
        norm_email = email.strip().lower()
        clean_name = (name or "").strip() or norm_email.split("@")[0]
        session = self.get_session()
        try:
            user = session.query(User).filter(User.email == norm_email).first()
            if not user:
                user = User(
                    email=norm_email,
                    name=clean_name,
                    role="Regulatory Auditor",
                    organization=organization,
                    password_hash=self._hash_password(secrets.token_urlsafe(32)),
                )
                session.add(user)
                session.commit()
                session.refresh(user)
            return self._serialize_user(user, auth_provider="google")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

