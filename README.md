# VeriLabel AI 🏥

**AI-Powered Medical Device Label Compliance Platform**

**Team:** Brahmastra | **Institution:** GITAM (Deemed to be University), Bengaluru
**Department:** AI / Computer Science and Engineering
**Motto:** *"Label Smarter. Comply Better."*

---

## 🌟 Project Overview

VeriLabel AI automates medical device label verification, compliance checks, and error detection.
It combines OCR-based text extraction, rule-based compliance validation, computer vision symbol detection,
GS1 barcode / UDI decoding, and an interactive human-in-the-loop review dashboard.

Regulatory scope: **FDA 21 CFR Part 801/830 · EU MDR 2017/745 Annex I · ISO 15223-1:2021**

---

## 🏗️ Repository Structure

```
Visionet_Verilabel/
├── verilabel_ai_project/          ← 🚀 MAIN RUNNABLE APPLICATION (Member 1 & 4)
│   └── verilabel_ai/
│       ├── run_app.py             ← Entry point — start the web platform here
│       └── verilabel_app/        ← FastAPI server, HTML dashboards, pipeline
│
├── VeriLabel-AI/                  ← Module Integration Hub (Member 1 — Backend)
│   └── VeriLabel-AI/
│       ├── visionet(Backend)/    ← Backend core library (verilabel package)
│       ├── visionet(CV_Barcode)/ ← Computer Vision & Barcode module (Member 3)
│       └── visonent(OCR)/        ← OCR & Document Processing module (Member 2)
│
├── visionet/                      ← Compliance Engine & Testing (Member 5 — this repo)
│   └── visionet/                 ← Rules engine, test suite, synthetic labels
│
└── visonent/                      ← OCR standalone demo (Member 2 — alternate build)
```

---

## 👥 Team Brahmastra

| Member | Role | Key Deliverables |
|--------|------|-----------------|
| **Member 1** | Backend & Integration Lead | Python architecture, module orchestration, DB schema, REST API, 17-rule compliance engine |
| **Member 2** | OCR & Document Processing | PaddleOCR integration, field extraction, document preprocessing |
| **Member 3** | Computer Vision & Barcode | OpenCV symbol detection (13 ISO 15223-1 symbols), GS1 barcode / UDI validation |
| **Member 4** | Frontend & Dashboard | FastAPI + Jinja2 web UI, 12-section audit dashboard, enterprise design |
| **Member 5** | Compliance, Testing & Docs | Regulatory rule authoring (JSON), synthetic test labels, precision/recall metrics |

---

## 🚀 Quick Start (Main Application)

```powershell
# 1. Navigate to the application directory
cd verilabel_ai_project/verilabel_ai

# 2. Install dependencies
pip install -r verilabel_app/verilabel/requirements.txt
# or use the backend requirements
pip install fastapi uvicorn opencv-python numpy Pillow PyMuPDF pyzbar pandas SQLAlchemy pydantic python-multipart

# 3. Launch the platform
python run_app.py
```

Then open your browser to:
- **Landing Page:** http://localhost:8000/
- **Auditor Sign In:** http://localhost:8000/login
- **Register:** http://localhost:8000/register
- **Audit Dashboard:** http://localhost:8000/dashboard

### 🔑 Pre-seeded Demo Credentials
| Name | Email | Password |
|------|-------|----------|
| Dr. Sarah Jenkins (Lead RA Auditor) | `sarah.jenkins@brahmastra.medtech` | `Auditor@2026` |
| Alex Chen (QA Inspection Specialist) | `alex.chen@gitam.edu` | `Auditor@2026` |
| Marcus Vance (Senior Compliance Officer) | `marcus.vance@fda-review.org` | `Auditor@2026` |

---

## 🧪 Testing & Validation

HLD Test Cases (from Section 11):

| TC | Condition | Expected Result |
|----|-----------|----------------|
| TC01 | All mandatory fields present | All checks pass |
| TC02 | Expiry date missing | Missing field detected |
| TC03 | Incorrect UDI format | Format error flagged |
| TC04 | Required symbol absent | Symbol warning raised |
| TC05 | Safety warnings missing | Missing warning detected |
| TC06 | Valid barcode present | Barcode decoded & verified |

**Evaluation Results:** Precision 100% · Recall 100% · F1-Score 100%

---

## 📦 Module Dependencies

```
opencv-python>=4.8.0    # Image processing
numpy>=1.24.0           # Numerical operations
Pillow>=9.5.0           # Image I/O
PyMuPDF>=1.23.0         # PDF conversion
pyzbar>=0.1.9           # Barcode decoding
pandas>=2.0.0           # Reporting
SQLAlchemy>=2.0.0       # ORM / Database
pydantic>=2.0.0         # Data validation
fastapi>=0.100.0        # Web framework
uvicorn>=0.22.0         # ASGI server
python-multipart>=0.0.6 # File uploads
```

---

## ⚠️ Disclaimer

VeriLabel AI is a prototype for academic demonstration purposes.
Results are **not** a substitute for certified regulatory review or approval.
Final label approval must be performed by qualified regulatory experts.
