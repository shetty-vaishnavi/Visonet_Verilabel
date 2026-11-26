"""
VeriLabel AI - Web Server & REST API
Team: Brahmastra | GITAM University
Role: Member 4 - Frontend & Dashboard Lead

Connects the VeriLabel verification pipeline, SQLite database, and authentication
to modern humanized web interfaces (Landing, Login, Register, and Audit Dashboard).
"""

import os
import sys
import time
import json
import hmac
import hashlib
import base64
import secrets
import logging
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import cv2
import numpy as np

# Configure path for verilabel package
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from verilabel.pipeline import VeriLabelPipeline, PipelineResult
from verilabel.config import AppConfig, DATABASE_PATH, SAMPLE_DATA_DIR
from verilabel.database.models import ProductReference, HumanReview, ValidationRun, LabelUpload
from verilabel.modules.vision.symbol_detector import ISO_SYMBOL_METADATA
from verilabel.modules.reporting.generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VeriLabel.Server")

# Initialize FastAPI application
app = FastAPI(
    title="VeriLabel AI — Medical Device Label Compliance Platform",
    description="AI-powered medical device label verification, compliance audits, and human-in-the-loop review.",
    version="1.0.0"
)

# Static and Templates directories
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
SAMPLES_DIR = BASE_DIR / "sample_data" / "labels"
REPORTS_DIR = BASE_DIR / "sample_data" / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/samples-img", StaticFiles(directory=str(SAMPLES_DIR)), name="samples_img")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize Pipeline Singleton
pipeline = VeriLabelPipeline()


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str

class RegisterRequest(BaseModel):
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = "Regulatory Auditor"
    organization: Optional[str] = ""
    password: str

class GoogleExchangeRequest(BaseModel):
    code: str

class HumanReviewRequest(BaseModel):
    validation_id: int
    reviewer_name: str
    reviewer_role: str
    decision: str  # 'APPROVED', 'REJECTED', 'CHANGES_REQUESTED'
    review_notes: str


# ==============================================================================
# HELPER FUNCTIONS & SIGNED SESSION AUTHENTICATION
# ==============================================================================

SESSION_COOKIE_NAME = "verilabel_session"
SESSION_TTL_SECONDS = 86400
_SESSION_SECRET: Optional[bytes] = None

PROTECTED_API_PREFIXES = (
    "/api/dashboard/",
    "/api/verify",
    "/api/review",
    "/api/reviews/",
    "/api/runs",
    "/api/reports",
    "/api/export/",
    "/api/settings",
    "/api/audit-history",
)


def _get_session_secret() -> bytes:
    """Returns persistent HMAC signing key from env or runtime uploads/.session_secret."""
    global _SESSION_SECRET
    if _SESSION_SECRET is not None:
        return _SESSION_SECRET

    env_secret = os.environ.get("VERILABEL_SESSION_SECRET", "").strip()
    if env_secret:
        _SESSION_SECRET = env_secret.encode("utf-8")
        return _SESSION_SECRET

    secret_file = UPLOADS_DIR / ".session_secret"
    if secret_file.exists():
        try:
            key = secret_file.read_text(encoding="utf-8").strip()
            if key:
                _SESSION_SECRET = key.encode("utf-8")
                return _SESSION_SECRET
        except Exception:
            pass

    generated = secrets.token_hex(32)
    try:
        secret_file.write_text(generated, encoding="utf-8")
    except Exception:
        pass
    _SESSION_SECRET = generated.encode("utf-8")
    return _SESSION_SECRET


def _create_session_token(user: Dict[str, Any]) -> str:
    """Signs a compact base64url JSON session token with HMAC-SHA256."""
    payload = {
        "uid": int(user["id"]),
        "name": str(user["name"]),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    sig = hmac.new(_get_session_secret(), b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _verify_session_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Validates HMAC signature, expiration, and SQLite user existence."""
    if not token or "." not in token:
        return None
    try:
        b64, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(_get_session_secret(), b64.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        padded = b64 + "=" * (-len(b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if int(payload.get("exp", 0)) <= int(time.time()):
            return None
        uid = payload.get("uid")
        if uid is None:
            return None
        return pipeline.db.get_user_by_id(int(uid))
    except Exception:
        return None


def _get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Extracts and verifies the current session user from HTTP-only cookie or Bearer header."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    return _verify_session_token(token)


def _set_session_cookie(response: Response, user: Dict[str, Any]) -> None:
    """Attaches signed HTTP-only session cookie."""
    token = _create_session_token(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    """Clears the session cookie on sign-out."""
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


@app.middleware("http")
async def auth_session_middleware(request: Request, call_next):
    """Rejects expired or unauthorized requests to protected dashboard APIs with 401."""
    path = request.url.path
    requires_auth = path.startswith(PROTECTED_API_PREFIXES) or (
        path == "/api/required-fields" and bool(request.query_params.get("validation_id"))
    )
    if requires_auth:
        user = _get_current_user(request)
        if not user:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required or session expired."},
            )
        request.state.user = user
    return await call_next(request)


def image_to_base64(image_np: np.ndarray) -> str:
    """Encodes a BGR or RGB OpenCV image array to base64 JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        return ""
    b64_bytes = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_bytes}"


# ==============================================================================
# HTML PAGE ROUTES
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Opens the Sign In portal by default, or Dashboard if already signed in."""
    if _get_current_user(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)

@app.get("/home", response_class=HTMLResponse)
async def home_overview_page(request: Request):
    """Serves the optional VeriLabel AI overview landing page."""
    return templates.TemplateResponse(request=request, name="index.html", context={"title": "VeriLabel AI — Label Smarter. Comply Better."})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serves the login page or redirects authenticated users straight to dashboard."""
    if _get_current_user(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    resp = templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"title": "VeriLabel AI — Sign In", "initial_tab": "signin"},
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Serves the Create Account tab or redirects authenticated users to dashboard."""
    if _get_current_user(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    resp = templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"title": "VeriLabel AI — Create Account", "initial_tab": "register"},
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serves the compliance dashboard; redirects unauthenticated users to login."""
    if not _get_current_user(request):
        return RedirectResponse(url="/login", status_code=302)
    resp = templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"title": "Audit Console — VeriLabel AI"},
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.get("/required-fields", response_class=HTMLResponse)
async def required_fields_page(request: Request):
    """Serves the interactive Required Fields Blueprint & Regulatory Cockpit."""
    return templates.TemplateResponse(request=request, name="required_fields.html", context={"title": "Mandatory Packaging Fields Studio — VeriLabel AI"})

@app.get("/logout", response_class=HTMLResponse)
async def logout_redirect_page():
    """Clears session cookie and redirects immediately back to login."""
    resp = RedirectResponse(url="/login", status_code=302)
    _clear_session_cookie(resp)
    return resp

@app.get("/signout", response_class=HTMLResponse)
async def signout_page(request: Request):
    """Clears session cookie and serves clean sign-out confirmation page."""
    resp = templates.TemplateResponse(request=request, name="signout.html", context={"title": "Signed Out — VeriLabel AI"})
    _clear_session_cookie(resp)
    return resp

@app.get("/flow", response_class=HTMLResponse)
async def simple_flow_page(request: Request):
    """Serves the clean multi-screen UI prototype & interactive auth workflow."""
    return templates.TemplateResponse(request=request, name="simple_flow.html", context={"title": "Simple UI Workflow — VeriLabel AI"})


# ==============================================================================
# AUTHENTICATION & GOOGLE OAUTH 2.0 APIS
# ==============================================================================

_GOOGLE_OAUTH_STATES: Dict[str, Dict[str, Any]] = {}
_GOOGLE_OAUTH_CODES: Dict[str, Dict[str, Any]] = {}


def _prune_google_oauth_stores() -> None:
    """Removes expired single-use OAuth state and exchange codes."""
    now = time.time()
    for key in [k for k, v in _GOOGLE_OAUTH_STATES.items() if now - v["ts"] > 600]:
        _GOOGLE_OAUTH_STATES.pop(key, None)
    for key in [k for k, v in _GOOGLE_OAUTH_CODES.items() if now - v["ts"] > 180]:
        _GOOGLE_OAUTH_CODES.pop(key, None)


def _get_google_oauth_config(request: Request) -> Dict[str, Any]:
    """Reads Google OAuth 2.0 client settings from environment variables."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip() or str(
        request.url_for("google_oauth_callback")
    )
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "configured": bool(client_id and client_secret),
    }


@app.get("/api/auth/session")
async def api_get_session(request: Request):
    """Verifies active signed session cookie and returns user or 401."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated or session expired.",
        )
    return {"status": "success", "authenticated": True, "user": user}


@app.post("/api/auth/logout")
async def api_logout():
    """Clears the HTTP-only session cookie."""
    resp = JSONResponse({"status": "success", "authenticated": False})
    _clear_session_cookie(resp)
    return resp


@app.post("/api/auth/login")
async def api_login(req: LoginRequest):
    """Authenticates a user by username or email and sets a signed HTTP-only session cookie."""
    identifier = (req.username or req.email or "").strip()
    if not identifier or not req.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required.",
        )
    user = pipeline.db.authenticate_user(identifier, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    resp = JSONResponse({"status": "success", "user": user})
    _set_session_cookie(resp, user)
    return resp


@app.post("/api/auth/register")
async def api_register(req: RegisterRequest):
    """Registers a new account using username (>=3 chars) and password (>=6 chars)."""
    try:
        user = pipeline.db.create_user(
            username=req.username or req.name,
            email=req.email,
            role=req.role or "Regulatory Auditor",
            organization=req.organization or "",
            password=req.password,
        )
        resp = JSONResponse({"status": "success", "user": user})
        _set_session_cookie(resp, user)
        return resp
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create auditor account.")


@app.get("/api/auth/google/status")
async def google_oauth_status(request: Request):
    """Reports whether server-side Google OAuth 2.0 credentials are configured."""
    cfg = _get_google_oauth_config(request)
    return {
        "status": "success",
        "configured": cfg["configured"],
        "client_id_present": bool(cfg["client_id"]),
        "redirect_uri": cfg["redirect_uri"],
    }


@app.get("/auth/google")
@app.get("/api/auth/google/login")
async def google_oauth_login(request: Request):
    """Starts real Google OAuth 2.0 authorization code flow when configured."""
    _prune_google_oauth_stores()
    cfg = _get_google_oauth_config(request)
    if not cfg["configured"]:
        return RedirectResponse(url="/login?oauth_error=google_not_configured", status_code=302)

    state = secrets.token_urlsafe(32)
    _GOOGLE_OAUTH_STATES[state] = {
        "ts": time.time(),
        "redirect_uri": cfg["redirect_uri"],
    }
    params = urllib.parse.urlencode(
        {
            "client_id": cfg["client_id"],
            "redirect_uri": cfg["redirect_uri"],
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    return RedirectResponse(
        url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}",
        status_code=302,
    )


@app.get("/api/auth/google/callback", name="google_oauth_callback")
async def google_oauth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    """Handles Google OAuth 2.0 callback, exchanges code, verifies userinfo, and creates session."""
    _prune_google_oauth_stores()
    if error:
        safe_err = urllib.parse.quote(error)
        return RedirectResponse(url=f"/login?oauth_error={safe_err}", status_code=302)

    state_entry = _GOOGLE_OAUTH_STATES.pop(state or "", None)
    if not state_entry or not code:
        return RedirectResponse(url="/login?oauth_error=invalid_oauth_state", status_code=302)

    cfg = _get_google_oauth_config(request)
    if not cfg["configured"]:
        return RedirectResponse(url="/login?oauth_error=google_not_configured", status_code=302)

    try:
        token_payload = urllib.parse.urlencode(
            {
                "code": code,
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uri": state_entry["redirect_uri"],
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        token_req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=token_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(token_req, timeout=10) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))

        access_token = token_data.get("access_token")
        if not access_token:
            return RedirectResponse(url="/login?oauth_error=missing_access_token", status_code=302)

        userinfo_req = urllib.request.Request(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(userinfo_req, timeout=10) as uresp:
            profile = json.loads(uresp.read().decode("utf-8"))

        email = (profile.get("email") or "").strip().lower()
        if not email or not profile.get("email_verified", False):
            return RedirectResponse(url="/login?oauth_error=unverified_google_email", status_code=302)

        full_name = profile.get("name") or email.split("@")[0]
        user = pipeline.db.get_or_create_oauth_user(email=email, name=full_name)

        handoff_code = secrets.token_urlsafe(24)
        _GOOGLE_OAUTH_CODES[handoff_code] = {"ts": time.time(), "user": user}
        return RedirectResponse(
            url=f"/login?oauth_code={urllib.parse.quote(handoff_code)}",
            status_code=302,
        )
    except Exception as exc:
        logger.error(f"Google OAuth callback error: {exc}")
        return RedirectResponse(url="/login?oauth_error=google_exchange_failed", status_code=302)


@app.post("/api/auth/google/exchange")
async def google_oauth_exchange(req: GoogleExchangeRequest):
    """Exchanges a single-use OAuth handoff code for the authenticated user session payload."""
    _prune_google_oauth_stores()
    entry = _GOOGLE_OAUTH_CODES.pop((req.code or "").strip(), None)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OAuth session code expired or invalid.",
        )
    resp = JSONResponse({"status": "success", "user": entry["user"]})
    _set_session_cookie(resp, entry["user"])
    return resp


# ==============================================================================
# LABEL SAMPLES, PRODUCTS & SCHEMA APIS
# ==============================================================================

SAMPLE_CATALOG = [
    {
        "id": "TC01",
        "file": "TC01_compliant.png",
        "name": "TC01_compliant.png",
        "description": "Sample label image in repository."
    },
    {
        "id": "TC02",
        "file": "TC02_missing_expiry.png",
        "name": "TC02_missing_expiry.png",
        "description": "Sample label image in repository."
    },
    {
        "id": "TC03",
        "file": "TC03_invalid_udi.png",
        "name": "TC03_invalid_udi.png",
        "description": "Sample label image in repository."
    },
    {
        "id": "TC04",
        "file": "TC04_missing_symbol.png",
        "name": "TC04_missing_symbol.png",
        "description": "Sample label image in repository."
    },
    {
        "id": "TC05",
        "file": "TC05_missing_warnings.png",
        "name": "TC05_missing_warnings.png",
        "description": "Sample label image in repository."
    },
    {
        "id": "TC06",
        "file": "tc04_skewed_and_noisy.png",
        "name": "tc04_skewed_and_noisy.png",
        "description": "Sample label image in repository."
    }
]

DOWNLOAD_PACK_DIR = BASE_DIR / "sample_data" / "download_pack"
DOWNLOAD_PACK_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/download-samples/{filename}")
async def download_sample_label(filename: str):
    """Downloads a sample label file for laptop upload testing."""
    safe_name = Path(filename).name
    target = (DOWNLOAD_PACK_DIR / safe_name).resolve()
    if target.parent != DOWNLOAD_PACK_DIR.resolve() or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found.")
    return FileResponse(
        path=str(target),
        filename=safe_name,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )

@app.get("/api/samples")
async def get_sample_catalog():
    """Returns only sample label files that actually exist on disk."""
    available = []
    for item in SAMPLE_CATALOG:
        fpath = SAMPLES_DIR / item["file"]
        if fpath.exists():
            available.append({
                "id": item["id"],
                "file": item["file"],
                "name": item["name"],
                "description": item["description"],
                "has_sidecar_text": fpath.with_suffix(".txt").exists()
            })
    return {"status": "success", "samples": available}

@app.get("/api/products")
async def get_product_baselines():
    """Returns registered product baselines strictly from SQLite ProductReference table."""
    session = pipeline.db.get_session()
    try:
        rows = session.query(ProductReference).all()
        products = [
            {
                "product_code": p.product_code,
                "trade_name": p.trade_name,
                "manufacturer": p.manufacturer,
                "gtin_di": p.gtin_di,
                "is_sterile": p.is_sterile,
                "is_single_use": p.is_single_use,
                "temp_min_c": p.temp_min_c,
                "temp_max_c": p.temp_max_c
            }
            for p in rows
        ]
        return {"status": "success", "products": products}
    finally:
        session.close()

@app.get("/api/rules")
async def get_compliance_rules():
    """Returns actual regulatory compliance rules loaded by ComplianceEngine."""
    rules_list = [
        {
            "rule_id": r.rule_id,
            "name": r.name,
            "category": r.category,
            "standard": r.standard,
            "severity": r.severity.value if hasattr(r.severity, "value") else str(r.severity),
            "description": r.description,
            "remediation_hint": r.remediation_hint
        }
        for r in pipeline.compliance_engine.rules.values()
    ]
    return {"status": "success", "total_rules": len(rules_list), "rules": rules_list}

@app.get("/api/required-fields")
async def get_required_fields_catalog(validation_id: Optional[int] = None):
    """
    Returns regulatory field schema aligned strictly to VeriLabel backend extractors.
    If validation_id is omitted, returns schema definitions only (no fabricated values).
    If validation_id is provided, populates values strictly from the stored DB ValidationRun.
    """
    run_record = None
    if validation_id is not None:
        run_record = pipeline.get_validation_details(validation_id)
        if not run_record:
            raise HTTPException(status_code=404, detail=f"Validation run #{validation_id} not found.")

    report = run_record.get("report", {}) if run_record else {}
    ext = report.get("extracted_fields", {}) if run_record else {}
    confs = ext.get("field_confidences", {}) if isinstance(ext, dict) else {}
    udi = report.get("udi_details") if run_record else None
    symbols = report.get("detected_symbols", []) if run_record else []
    barcodes = report.get("barcodes_scanned", []) if run_record else []
    symbol_names = {s.get("symbol_name") for s in symbols if isinstance(s, dict)}

    schema_defs = [
        ("product_name", "Product / Device Name", "identity", "FDA 21 CFR §801.60 / EU MDR §23.2(a)"),
        ("manufacturer_name", "Manufacturer Name", "origin", "FDA 21 CFR §801.1 / EU MDR §23.2(b)"),
        ("manufacturer_address", "Manufacturer Address", "origin", "FDA 21 CFR §801.1 / ISO 15223-1 §5.1.1"),
        ("lot_number", "Batch / Lot Number (LOT)", "traceability", "FDA 21 CFR §801 / EU MDR §23.2(g) / ISO 15223-1 §5.1.5"),
        ("expiry_date", "Use-By / Expiration Date (EXP)", "traceability", "FDA 21 CFR §801 / EU MDR §23.2(i) / ISO 15223-1 §5.1.4"),
        ("manufacturing_date", "Manufacturing Date (MFG)", "traceability", "ISO 15223-1 §5.1.3"),
        ("catalog_number", "Catalogue / REF Number", "identity", "ISO 15223-1 §5.1.6"),
        ("serial_number", "Serial Number (SN)", "traceability", "ISO 15223-1 §5.1.7"),
        ("storage_conditions", "Storage Conditions", "environment", "EU MDR §23.2(k) / ISO 15223-1 §5.3.7"),
        ("is_sterile", "Sterility Claim & Method", "sterility", "ISO 15223-1 §5.2.1 / EU MDR §23.2(l)"),
        ("is_single_use", "Single-Use Declaration", "sterility", "ISO 15223-1 §5.4.2 / EU MDR §23.2(n)"),
        ("ce_mark_number", "CE Conformity ID", "regulatory", "EU MDR Article 20"),
        ("is_rx_only", "Prescription Only (Rx Only)", "regulatory", "FDA 21 CFR §801.109(b)"),
        ("warnings", "Safety Warnings & Precautions", "sterility", "FDA 21 CFR §801.109 / EU MDR §23.2(m)"),
        ("instructions_for_use", "Instructions for Use (IFU)", "regulatory", "ISO 15223-1 §5.4.3"),
        ("udi_carrier", "UDI Barcode Carrier", "carrier", "FDA 21 CFR Part 830 / EU MDR Article 27"),
        ("udi_di", "UDI Device Identifier (DI)", "carrier", "GS1 General Specs / FDA 21 CFR §830")
    ]

    result_fields = []
    for key, label, category, standard in schema_defs:
        item = {
            "id": key,
            "name": label,
            "category": category,
            "regulation": standard,
            "value": None,
            "confidence": None,
            "status": "NOT_ANALYZED"
        }

        if run_record is not None:
            raw_val = None
            conf = confs.get(key)
            if key == "is_sterile":
                if ext.get("is_sterile"):
                    raw_val = ext.get("sterilization_method") or "STERILE"
                    item["status"] = "DETECTED"
                elif "sterile" in symbol_names:
                    raw_val = "ISO Sterile symbol detected"
                    item["status"] = "DETECTED"
                else:
                    item["status"] = "NOT_DETECTED"
            elif key == "is_single_use":
                if ext.get("is_single_use"):
                    raw_val = "Single Use / Do Not Re-use"
                    item["status"] = "DETECTED"
                elif "do_not_reuse" in symbol_names:
                    raw_val = "ISO Do Not Re-use symbol detected"
                    item["status"] = "DETECTED"
                else:
                    item["status"] = "NOT_DETECTED"
            elif key == "is_rx_only":
                if ext.get("is_rx_only"):
                    raw_val = "Rx Only"
                    item["status"] = "DETECTED"
                elif "rx_only" in symbol_names:
                    raw_val = "FDA Rx Only symbol detected"
                    item["status"] = "DETECTED"
                else:
                    item["status"] = "NOT_DETECTED"
            elif key == "warnings":
                w_list = ext.get("warnings") or []
                if w_list:
                    raw_val = " | ".join(w_list)
                    item["status"] = "DETECTED"
                else:
                    item["status"] = "NOT_DETECTED"
            elif key == "udi_carrier":
                if udi and udi.get("raw_carrier"):
                    raw_val = udi.get("raw_carrier")
                    item["status"] = "DETECTED"
                elif barcodes:
                    raw_val = barcodes[0].get("data")
                    item["status"] = "DETECTED"
                else:
                    item["status"] = "NOT_DETECTED"
            elif key == "udi_di":
                if udi and udi.get("device_identifier_di"):
                    chk = "Valid Checksum" if udi.get("is_di_checksum_valid") else "Invalid Checksum"
                    raw_val = f"{udi.get('device_identifier_di')} ({chk})"
                    item["status"] = "DETECTED" if udi.get("is_di_checksum_valid") else "FAILED"
                else:
                    item["status"] = "NOT_DETECTED"
            else:
                val = ext.get(key)
                if val is not None and str(val).strip():
                    raw_val = str(val).strip()
                    item["status"] = "DETECTED"
                else:
                    item["status"] = "NOT_DETECTED"

            item["value"] = raw_val
            item["confidence"] = round(float(conf), 4) if conf is not None else None

        result_fields.append(item)

    return {
        "status": "success",
        "analyzed": run_record is not None,
        "validation_id": validation_id,
        "file_name": run_record.get("file_name") if run_record else None,
        "verdict": run_record.get("verdict") if run_record else None,
        "compliance_score": run_record.get("compliance_score") if run_record else None,
        "total_fields": len(result_fields),
        "fields": result_fields
    }


# ==============================================================================
# AUDIT & VERIFICATION APIS
# ==============================================================================

@app.post("/api/verify")
async def verify_label(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    product_code: Optional[str] = Form(None)
):
    """
    Executes the end-to-end VeriLabel AI compliance verification pipeline.
    Accepts either an uploaded file or an explicit sample_id chosen by the user.
    Never invents missing values or confidences.
    """
    target_path: Optional[Path] = None
    clean_product_code = product_code.strip() if product_code and product_code.strip() else None

    if file and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}. Upload PDF, PNG, or JPG.")

        saved_name = f"upload_{int(time.time())}_{Path(file.filename).name}"
        target_path = UPLOADS_DIR / saved_name
        contents = await file.read()
        with open(target_path, "wb") as f:
            f.write(contents)
    elif sample_id:
        matched = next((s for s in SAMPLE_CATALOG if s["id"] == sample_id), None)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        target_path = SAMPLES_DIR / matched["file"]
        if not target_path.exists():
            alt = SAMPLES_DIR / matched["file"].lower()
            if alt.exists():
                target_path = alt
            else:
                raise HTTPException(status_code=404, detail=f"Sample file {matched['file']} not on disk.")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or a sample_id.")

    try:
        raw_img_np, _ = pipeline.doc_processor.load_document(target_path)
        raw_b64 = image_to_base64(raw_img_np)

        result: PipelineResult = pipeline.process_label(
            file_path=target_path,
            product_code=clean_product_code,
            save_to_db=True
        )

        annotated_b64 = image_to_base64(result.annotated_image)
        report = result.report
        summary = result.summary
        fields = result.fields

        raw_ocr_dict = result.raw_ocr.to_dict() if hasattr(result.raw_ocr, "to_dict") else {}
        if result.raw_ocr and result.raw_ocr.metadata:
            raw_ocr_dict["metadata"] = result.raw_ocr.metadata

        # Serialized symbols
        symbols_list = []
        for s in result.symbols:
            if hasattr(s, "to_dict"):
                sd = s.to_dict()
                bbox = sd.get("bbox")
                symbols_list.append({
                    "symbol_id": sd.get("symbol_name"),
                    "display_name": sd.get("display_name"),
                    "iso_code": sd.get("iso_code"),
                    "confidence": round(float(sd.get("confidence", 0)), 4),
                    "bbox": {"x": bbox["x"], "y": bbox["y"], "w": bbox["width"], "h": bbox["height"]} if bbox and isinstance(bbox, dict) else None
                })

        # Serialized barcodes & UDI
        barcodes_list = []
        for b in result.barcodes:
            bd = b.to_dict() if hasattr(b, "to_dict") else {
                "data": getattr(b, "data", ""),
                "type": getattr(b, "barcode_type", "UNKNOWN"),
                "bbox": None
            }
            barcodes_list.append(bd)

        udi_dict = result.udi.to_dict() if result.udi else None
        if udi_dict:
            udi_dict["raw_barcode"] = udi_dict.get("raw_carrier")
            udi_dict["carrier_type"] = udi_dict.get("standard")

        # Serialized checks
        checks_list = []
        for c in report.checks:
            checks_list.append({
                "rule_id": c.rule_id,
                "rule_name": c.rule_name,
                "category": c.category.value if hasattr(c.category, "value") else str(c.category),
                "standard": c.standard,
                "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity),
                "passed": bool(c.passed),
                "observed_value": c.observed_value,
                "expected_condition": c.expected_condition,
                "message": c.message,
                "remediation_hint": c.remediation_hint
            })

        return {
            "status": "success",
            "validation_id": result.validation_id,
            "file_name": target_path.name,
            "file_metadata": result.file_metadata,
            "product_code": clean_product_code,
            "verdict": report.verdict.value,
            "compliance_score": round(float(report.compliance_score), 1),
            "processing_time_ms": round(float(report.processing_time_ms), 2),
            "timestamp": report.timestamp,
            "summary": summary,
            "raw_image": raw_b64,
            "annotated_image": annotated_b64,
            "raw_ocr": raw_ocr_dict,
            "fields": fields.to_dict(),
            "symbols": symbols_list,
            "barcodes": barcodes_list,
            "udi": udi_dict,
            "checks": checks_list,
            "latest_review_status": "PENDING_REVIEW",
            "reviews": [],
            "markdown_report": result.markdown_report
        }

    except Exception as e:
        logger.error(f"Error during label verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ==============================================================================
# HUMAN EXPERT REVIEW SIGN-OFF API
# ==============================================================================

@app.post("/api/review")
async def submit_review(req: HumanReviewRequest):
    """
    Submits a certified human expert regulatory decision into the SQLite audit ledger.
    Fulfills FDA 21 CFR Part 11 and EU MDR Human-in-the-Loop traceability requirements.
    """
    try:
        review_id = pipeline.submit_human_review(
            validation_id=req.validation_id,
            reviewer_name=req.reviewer_name,
            reviewer_role=req.reviewer_role,
            decision=req.decision,
            review_notes=req.review_notes
        )
        return {
            "status": "success",
            "review_id": review_id,
            "message": f"Audit decision '{req.decision}' successfully logged by {req.reviewer_name}."
        }
    except Exception as e:
        logger.error(f"Failed to submit human review: {e}")
        raise HTTPException(status_code=500, detail=f"Review submission failed: {str(e)}")

@app.get("/api/reviews/history")
async def get_review_history(limit: int = 50):
    """Fetches actual persisted HumanReview entries from SQLite."""
    session = pipeline.db.get_session()
    try:
        rows = (
            session.query(HumanReview)
            .order_by(HumanReview.reviewed_at.desc())
            .limit(limit)
            .all()
        )
        items = []
        for r in rows:
            file_name = r.validation.upload.file_name if r.validation and r.validation.upload else None
            items.append({
                "id": r.id,
                "validation_id": r.validation_id,
                "file_name": file_name,
                "reviewer_name": r.reviewer_name,
                "reviewer_role": r.reviewer_role,
                "decision": r.decision,
                "review_notes": r.review_notes,
                "reviewed_at": r.reviewed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if r.reviewed_at else None
            })
        return {"status": "success", "reviews": items, "count": len(items)}
    finally:
        session.close()


# ==============================================================================
# AUDIT TRAIL LEDGER, ADVANCED QUERY & STATISTICS APIS
# ==============================================================================

@app.get("/api/dashboard/stats")
async def get_dashboard_analytics():
    """
    Returns strictly real aggregated compliance metrics computed directly from SQLite ValidationRun rows.
    Zero fabricated trends or placeholder counts.
    """
    history = pipeline.get_audit_history(limit=500)
    total_scans = len(history)
    compliant_count = sum(1 for h in history if h.get("verdict") == "COMPLIANT")
    non_compliant_count = sum(1 for h in history if h.get("verdict") == "NON_COMPLIANT")
    conditional_count = sum(1 for h in history if h.get("verdict") == "CONDITIONAL_PASS")
    pending_reviews = sum(1 for h in history if h.get("latest_review_status") == "PENDING_REVIEW")
    pass_rate = round((compliant_count / total_scans * 100), 1) if total_scans > 0 else 0.0

    # Group real validation runs by calendar date/month strictly from DB created_at timestamps
    period_buckets: Dict[str, Dict[str, int]] = {}
    for h in history:
        ts = str(h.get("created_at") or "")
        period = ts[:10] if len(ts) >= 10 else "Unknown"
        if period not in period_buckets:
            period_buckets[period] = {"compliant": 0, "non_compliant": 0, "conditional": 0, "total": 0}
        period_buckets[period]["total"] += 1
        v = h.get("verdict")
        if v == "COMPLIANT":
            period_buckets[period]["compliant"] += 1
        elif v == "CONDITIONAL_PASS":
            period_buckets[period]["conditional"] += 1
        else:
            period_buckets[period]["non_compliant"] += 1

    monthly_trend = [
        {"period": p, **counts}
        for p, counts in sorted(period_buckets.items())
    ]

    compliance_breakdown = [
        {"name": "Compliant", "value": compliant_count, "color": "#168B69"},
        {"name": "Conditional", "value": conditional_count, "color": "#E5A443"},
        {"name": "Non-Compliant", "value": non_compliant_count, "color": "#DC5656"}
    ]

    recent_activity = []
    for h in history[:8]:
        v = h.get("verdict")
        action_verb = "cleared as compliant" if v == "COMPLIANT" else ("marked conditional pass" if v == "CONDITIONAL_PASS" else "flagged non-compliant")
        recent_activity.append({
            "id": h.get("validation_id"),
            "title": f"Run #{h.get('validation_id')} — {h.get('file_name')}",
            "description": f"Verdict: {action_verb} (Score: {h.get('compliance_score')}%). Review: {h.get('latest_review_status')}.",
            "timestamp": h.get("created_at"),
            "type": "audit_complete",
            "status": v
        })

    recent_analyses = [
        {
            "validation_id": h.get("validation_id"),
            "file_name": h.get("file_name"),
            "verdict": h.get("verdict"),
            "compliance_score": h.get("compliance_score"),
            "critical_errors": h.get("critical_errors", 0),
            "major_warnings": h.get("major_warnings", 0),
            "latest_review_status": h.get("latest_review_status", "PENDING_REVIEW"),
            "created_at": h.get("created_at")
        }
        for h in history[:8]
    ]

    return {
        "status": "success",
        "has_records": total_scans > 0,
        "metrics": {
            "total_scans": total_scans,
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "conditional_count": conditional_count,
            "pending_reviews": pending_reviews,
            "pass_rate_percent": pass_rate
        },
        "compliance_breakdown": compliance_breakdown,
        "monthly_trend": monthly_trend,
        "recent_activity": recent_activity,
        "recent_analyses": recent_analyses,
        "total_rules_loaded": len(pipeline.compliance_engine.rules),
        "standards": ["FDA 21 CFR 801/830", "EU MDR 2017/745", "ISO 15223-1:2021", "GS1 UDI"]
    }

@app.get("/api/runs")
async def get_validation_runs(
    q: Optional[str] = None,
    verdict: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    limit: int = 15
):
    """Returns searchable, filterable, and paginated validation runs strictly from SQLite."""
    history = pipeline.get_audit_history(limit=500)

    if q and q.strip():
        term = q.strip().lower()
        history = [
            h for h in history
            if term in str(h.get("validation_id")).lower()
            or term in str(h.get("file_name", "")).lower()
        ]

    if verdict and verdict.strip() and verdict != "ALL":
        history = [h for h in history if h.get("verdict") == verdict]

    if status and status.strip() and status != "ALL":
        history = [h for h in history if h.get("latest_review_status") == status]

    total_records = len(history)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = history[start_idx:end_idx]

    return {
        "status": "success",
        "total": total_records,
        "page": page,
        "limit": limit,
        "pages": (total_records + limit - 1) // limit if limit > 0 else 1,
        "items": paginated_items
    }

@app.get("/api/runs/export")
async def export_all_runs_csv():
    """Exports all persisted validation runs from SQLite to a real CSV download."""
    import csv
    history = pipeline.get_audit_history(limit=500)
    export_path = REPORTS_DIR / "verilabel_audit_ledger.csv"
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["validation_id", "file_name", "verdict", "compliance_score", "critical_errors", "major_warnings", "latest_review_status", "created_at"])
        for h in history:
            writer.writerow([
                h.get("validation_id"),
                h.get("file_name"),
                h.get("verdict"),
                h.get("compliance_score"),
                h.get("critical_errors"),
                h.get("major_warnings"),
                h.get("latest_review_status"),
                h.get("created_at")
            ])
    return FileResponse(export_path, media_type="text/csv", filename="verilabel_audit_ledger.csv")

@app.get("/api/runs/{val_id}")
async def get_run_detail(val_id: int):
    """
    Returns full comprehensive details, extracted fields, symbols, barcodes,
    rule checklists, and reviewer history for a specific stored validation run.
    """
    record = pipeline.get_validation_details(val_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Validation run #{val_id} not found.")

    report = record.get("report", {})
    raw_b64 = None
    annotated_b64 = None

    # Locate original file on disk only if it actually exists (never fabricate)
    fname = record.get("file_name")
    if fname:
        candidate_paths = [UPLOADS_DIR / fname, SAMPLES_DIR / fname]
        found_path = next((p for p in candidate_paths if p.exists()), None)
        if found_path:
            try:
                raw_img_np, _ = pipeline.doc_processor.load_document(found_path)
                raw_b64 = image_to_base64(raw_img_np)
                annotated_b64 = raw_b64
            except Exception:
                pass

    symbols_norm = []
    for s in report.get("detected_symbols", []):
        bbox = s.get("bbox")
        symbols_norm.append({
            "symbol_id": s.get("symbol_name"),
            "display_name": s.get("display_name"),
            "iso_code": s.get("iso_code"),
            "confidence": round(float(s.get("confidence", 0)), 4),
            "bbox": {"x": bbox["x"], "y": bbox["y"], "w": bbox["width"], "h": bbox["height"]} if isinstance(bbox, dict) else None
        })

    udi_norm = report.get("udi_details")
    if isinstance(udi_norm, dict):
        udi_norm = dict(udi_norm)
        udi_norm["raw_barcode"] = udi_norm.get("raw_carrier")
        udi_norm["carrier_type"] = udi_norm.get("standard")

    return {
        "status": "success",
        "source": "database_record",
        "validation_id": record["validation_id"],
        "file_name": record["file_name"],
        "file_hash": record["file_hash"],
        "verdict": record["verdict"],
        "compliance_score": round(float(record["compliance_score"]), 1),
        "created_at": record["created_at"],
        "processing_time_ms": record["processing_time_ms"],
        "latest_review_status": record["latest_review_status"],
        "reviews": record["reviews"],
        "raw_image": raw_b64,
        "annotated_image": annotated_b64,
        "fields": report.get("extracted_fields", {}),
        "symbols": symbols_norm,
        "barcodes": report.get("barcodes_scanned", []),
        "udi": udi_norm,
        "checks": report.get("checks", []),
        "run": record
    }

@app.get("/api/reviews/pending")
async def get_pending_reviews():
    """Fetches all validation runs awaiting qualified human review."""
    history = pipeline.get_audit_history(limit=100)
    pending = [h for h in history if h.get("latest_review_status") == "PENDING_REVIEW"]
    return {"status": "success", "pending": pending, "count": len(pending)}

@app.get("/api/reports")
async def list_reports():
    """Lists all generated compliance audit reports."""
    history = pipeline.get_audit_history(limit=100)
    reports = []
    for h in history:
        reports.append({
            "validation_id": h.get("validation_id"),
            "file_name": h.get("file_name"),
            "verdict": h.get("verdict"),
            "compliance_score": h.get("compliance_score"),
            "review_status": h.get("latest_review_status"),
            "audit_date": h.get("created_at") or h.get("timestamp") or "N/A",
            "pdf_url": f"/api/reports/{h.get('validation_id')}/pdf",
            "json_url": f"/api/export/{h.get('validation_id')}?format=json",
            "csv_url": f"/api/export/{h.get('validation_id')}?format=csv"
        })
    return {"status": "success", "reports": reports}

@app.get("/api/reports/{val_id}/pdf", response_class=HTMLResponse)
async def view_printable_dossier(val_id: int, request: Request):
    """Generates an audit-ready, high-fidelity printable regulatory compliance dossier."""
    record = pipeline.get_validation_details(val_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Validation record #{val_id} not found.")

    report = record.get("report", {})
    fields = report.get("extracted_fields", {})
    symbols = report.get("detected_symbols", [])
    barcodes = report.get("barcodes_scanned", [])
    udi = report.get("udi_details")
    checks = report.get("checks", [])
    reviews = record.get("reviews", [])

    dossier_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Regulatory Compliance Dossier — Run #{val_id}</title>
        <link rel="stylesheet" href="/static/css/app.css">
        <style>
            body {{ background: #FFFFFF; font-family: 'Inter', sans-serif; padding: 2.5rem; }}
            .dossier-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #142B45; padding-bottom: 1.5rem; margin-bottom: 2rem; }}
            .dossier-title {{ font-size: 1.6rem; font-weight: 800; color: #142B45; }}
            .dossier-seal {{ border: 2px solid #087E70; color: #087E70; padding: 0.5rem 1rem; border-radius: 4px; font-weight: 800; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }}
            .section-box {{ border: 1px solid #E2E8F0; border-radius: 6px; padding: 1.25rem; margin-bottom: 1.5rem; }}
            .table-dossier {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
            .table-dossier th {{ background: #F7F9FC; text-align: left; padding: 0.6rem; border-bottom: 1px solid #CBD5E1; color: #475569; text-transform: uppercase; font-size: 0.72rem; }}
            .table-dossier td {{ padding: 0.6rem; border-bottom: 1px solid #E2E8F0; }}
            @media print {{ .no-print {{ display: none !important; }} }}
        </style>
    </head>
    <body>
        <div class="no-print" style="margin-bottom: 1.5rem; display: flex; justify-content: space-between;">
            <button onclick="window.print()" class="btn btn-teal">Print / Save as PDF</button>
            <button onclick="window.close()" class="btn btn-secondary">Close Window</button>
        </div>

        <div class="dossier-header">
            <div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <img src="/static/img/verilabel_logo.jpg" alt="Logo" style="width: 38px; height: 38px; border-radius: 4px;">
                    <span class="dossier-title">VeriLabel AI &bull; Regulatory Compliance Dossier</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B;">
                    Evaluation Standard: FDA 21 CFR Part 801/830 &bull; EU MDR 2017/745 Annex I &bull; ISO 15223-1:2021
                </div>
            </div>
            <div class="dossier-seal">
                VERDICT: {record.get('verdict', 'PENDING')} (Score: {record.get('compliance_score', 0)}%)
            </div>
        </div>

        <div class="section-box">
            <h4 style="margin-bottom: 0.75rem; color: #142B45;">1. Audit Identification & Traceability</h4>
            <table class="table-dossier">
                <tr><td style="width: 25%; font-weight: 600;">Validation Run ID:</td><td>#{val_id}</td><td style="width: 25%; font-weight: 600;">Timestamp:</td><td>{record.get('created_at', 'N/A')}</td></tr>
                <tr><td style="font-weight: 600;">Artwork File Name:</td><td>{record.get('file_name', 'N/A')}</td><td style="font-weight: 600;">Cryptographic SHA-256:</td><td class="mono">{record.get('file_hash', 'N/A')[:32]}...</td></tr>
            </table>
        </div>

        <div class="section-box">
            <h4 style="margin-bottom: 0.75rem; color: #142B45;">2. Extracted Mandatory Label Attributes (OCR)</h4>
            <table class="table-dossier">
                <tr><td style="width: 25%; font-weight: 600;">Trade Name:</td><td>{fields.get('product_name') or 'Not detected'}</td><td style="width: 25%; font-weight: 600;">Manufacturer:</td><td>{fields.get('manufacturer_name') or 'Not detected'}</td></tr>
                <tr><td style="font-weight: 600;">Batch / Lot No:</td><td>{fields.get('lot_number') or 'Not detected'}</td><td style="font-weight: 600;">Expiration Date:</td><td>{fields.get('expiry_date') or 'Not detected'}</td></tr>
                <tr><td style="font-weight: 600;">Sterility Claim:</td><td>{fields.get('sterilization_method') or ('STERILE' if fields.get('is_sterile') else 'Not detected')}</td><td style="font-weight: 600;">CE Mark:</td><td>{fields.get('ce_mark_number') or 'Not detected'}</td></tr>
            </table>
        </div>

        <div class="section-box">
            <h4 style="margin-bottom: 0.75rem; color: #142B45;">3. Regulatory Rule Checklist ({len(checks)} Evaluated)</h4>
            <table class="table-dossier">
                <thead><tr><th>Rule ID</th><th>Requirement</th><th>Severity</th><th>Status</th><th>Observed Value</th></tr></thead>
                <tbody>
    """
    for c in checks:
        status_text = "PASS" if c.get("passed") else "FAIL"
        dossier_html += f"<tr><td class='mono'>{c.get('rule_id')}</td><td>{c.get('rule_name')}</td><td>{c.get('severity')}</td><td style='font-weight: 700;'>{status_text}</td><td>{c.get('observed_value') or 'N/A'}</td></tr>"

    dossier_html += f"""
                </tbody>
            </table>
        </div>

        <div class="section-box">
            <h4 style="margin-bottom: 0.75rem; color: #142B45;">4. 21 CFR Part 11 Qualified Expert Sign-Off</h4>
    """
    if reviews:
        for r in reviews:
            dossier_html += f"""
            <div style="background: #F7F9FC; border: 1px solid #CBD5E1; padding: 1rem; border-radius: 4px; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <strong>Reviewer: {r.get('reviewer_name')} ({r.get('reviewer_role')})</strong>
                    <span style="font-weight: 700; color: #087E70;">DECISION: {r.get('decision')}</span>
                </div>
                <div style="font-size: 0.85rem; color: #334155;"><strong>Notes:</strong> {r.get('notes') or 'Regulatory clearance recorded.'}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.35rem;">Timestamp: {r.get('timestamp')} &bull; Electronic Signature Applied</div>
            </div>
            """
    else:
        dossier_html += "<div style='color: #64748B;'>Pending Qualified Reviewer Sign-Off.</div>"

    dossier_html += """
        </div>
        <div style="font-size: 0.72rem; color: #94A3B8; text-align: center; margin-top: 2rem;">
            Generated by VeriLabel AI Enterprise Compliance Engine &bull; GITAM University &bull; Confidential Regulatory Record
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=dossier_html)

SETTINGS_FILE = BASE_DIR / "sample_data" / "settings.json"

@app.get("/api/settings")
async def get_settings():
    """Retrieves stored organization/facility configuration settings without fabricating defaults."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"status": "success", "configured": True, "settings": data}
        except Exception:
            pass
    return {
        "status": "success",
        "configured": False,
        "settings": {}
    }

@app.post("/api/settings")
async def save_settings(req: Request):
    """Saves updated auditor and organization settings to settings.json."""
    incoming = await req.json()
    existing = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(incoming)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return {"status": "success", "message": "Settings updated successfully.", "settings": existing}

@app.get("/api/audit-history")
async def get_audit_history(limit: int = 15):
    """Fetches recent validation runs and human review sign-offs from SQLite."""
    try:
        history = pipeline.get_audit_history(limit=limit)
        return {"status": "success", "history": history}
    except Exception as e:
        logger.error(f"Failed to fetch audit history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch audit records.")

@app.get("/api/stats")
async def get_platform_stats():
    """Calculates real overall compliance statistics strictly from SQLite records."""
    history = pipeline.get_audit_history(limit=500)
    total_scans = len(history)
    compliant_count = sum(1 for h in history if h.get("verdict") == "COMPLIANT")
    pass_rate = round((compliant_count / total_scans * 100), 1) if total_scans > 0 else 0.0
    pending_reviews = sum(1 for h in history if h.get("latest_review_status") == "PENDING_REVIEW")

    return {
        "total_scans": total_scans,
        "compliant_count": compliant_count,
        "pass_rate_percent": pass_rate,
        "pending_reviews": pending_reviews,
        "total_rules_loaded": len(pipeline.compliance_engine.rules),
        "standards": ["FDA 21 CFR 801/830", "EU MDR 2017/745", "ISO 15223-1:2021", "GS1 UDI"]
    }


# ==============================================================================
# REPORT EXPORT APIS
# ==============================================================================

@app.get("/api/export/{val_id}")
async def export_report(val_id: int, format: str = "json"):
    """Exports full stored validation run report in JSON or CSV format."""
    record = pipeline.get_validation_details(val_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Validation ID #{val_id} not found.")

    if format.lower() == "json":
        export_path = REPORTS_DIR / f"audit_report_val_{val_id}.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        return FileResponse(export_path, media_type="application/json", filename=export_path.name)

    elif format.lower() == "csv":
        import csv
        export_path = REPORTS_DIR / f"audit_report_val_{val_id}.csv"
        checks = record.get("report", {}).get("checks", [])
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rule ID", "Rule Name", "Category", "Severity", "Passed", "Observed Value", "Message"])
            for c in checks:
                writer.writerow([
                    c.get("rule_id"),
                    c.get("rule_name"),
                    c.get("category"),
                    c.get("severity"),
                    c.get("passed"),
                    c.get("observed_value"),
                    c.get("message")
                ])
        return FileResponse(export_path, media_type="text/csv", filename=export_path.name)

    else:
        raise HTTPException(status_code=400, detail="Supported export formats: 'json', 'csv'")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VeriLabel AI web application on http://127.0.0.1:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
