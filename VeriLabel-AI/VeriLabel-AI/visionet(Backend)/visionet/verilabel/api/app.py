"""
VeriLabel AI - REST API (FastAPI)
Team: Brahmastra | GITAM University
Exposes RESTful endpoints for label compliance verification, human approval workflows,
and audit trail retrieval.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from verilabel.pipeline import VeriLabelPipeline
from verilabel.modules.compliance.models import ComplianceVerdict

app = FastAPI(
    title="VeriLabel AI - Compliance API",
    description="Automated Medical Device Label Compliance & Traceability Platform API",
    version="1.0.0"
)

# Shared pipeline instance
pipeline = VeriLabelPipeline()


class HumanReviewRequest(BaseModel):
    reviewer_name: str
    reviewer_role: str
    decision: str  # 'APPROVED', 'REJECTED', 'CHANGES_REQUESTED'
    review_notes: Optional[str] = None


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "VeriLabel AI Compliance Platform",
        "ocr_engine": pipeline.ocr_engine.__class__.__name__,
        "database": str(pipeline.config.db_path)
    }


@app.get("/api/v1/rules")
def get_compliance_rules():
    """Returns active compliance rules."""
    return [r.model_dump() for r in pipeline.compliance_engine.rules.values()]


@app.post("/api/v1/labels/validate")
async def validate_label(
    file: UploadFile = File(...),
    product_code: Optional[str] = Form(None)
):
    """
    Upload a medical device label image or PDF to run automated compliance checks.
    """
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = pipeline.process_label(tmp_path, product_code=product_code, save_to_db=True)
        return {
            "validation_id": result.validation_id,
            "file_name": file.filename,
            "verdict": result.report.verdict.value,
            "compliance_score": result.report.compliance_score,
            "summary": result.summary,
            "report": result.report.model_dump(),
            "extracted_fields": result.fields.to_dict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/api/v1/labels/{validation_id}")
def get_validation_result(validation_id: int):
    """Fetches details and audit logs for a specific validation run."""
    record = pipeline.get_validation_details(validation_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Validation ID {validation_id} not found.")
    return record


@app.post("/api/v1/labels/{validation_id}/review")
def submit_review(validation_id: int, review: HumanReviewRequest):
    """Submits human expert review and approval decision."""
    try:
        review_id = pipeline.submit_human_review(
            validation_id=validation_id,
            reviewer_name=review.reviewer_name,
            reviewer_role=review.reviewer_role,
            decision=review.decision,
            review_notes=review.review_notes
        )
        return {
            "review_id": review_id,
            "validation_id": validation_id,
            "status": "REVIEW_RECORDED",
            "decision": review.decision.upper()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/history")
def get_audit_history(limit: int = 50):
    """Returns recent label audit records."""
    return pipeline.get_audit_history(limit=limit)
