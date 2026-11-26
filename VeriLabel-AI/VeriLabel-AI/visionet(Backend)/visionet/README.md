# VeriLabel AI — AI-Powered Medical Device Label Compliance Platform

**Team:** Brahmastra  
**Institution:** GITAM (Deemed to be University), Bengaluru  
**Role:** Member 1 — Backend Developer & Integration Lead  

---

## 📌 Executive Overview

**VeriLabel AI** is an intelligent medical device label verification, compliance audit, and traceability platform. Medical device labeling is strictly regulated under **FDA 21 CFR Part 801 / 830**, **EU MDR 2017/745 Annex I**, and **ISO 15223-1:2021**. Non-compliance risks patient injury, product recalls, and severe regulatory sanctions.

VeriLabel AI automates document ingestion, optical character recognition (OCR), ISO symbol computer vision detection, GS1-128 / UDI barcode decoding, rule-based regulatory validation, and audit trail persistence, while empowering qualified human regulatory experts with a seamless final review and approval workflow.

---

## 🏗️ Architecture & Module Division

| Module | Name | Assigned Member | Implementation Status |
| :--- | :--- | :--- | :--- |
| **Module 1** | Document Upload & Ingestion | Member 1 & 4 | Preprocessing (CLAHE, denoising, blur & contrast checks, SHA-256 integrity hash) |
| **Module 2** | OCR & Field Extraction | Member 2 & 1 | `PaddleOCREngine` + `FallbackOCREngine` + `MedicalFieldExtractor` (Regex / NLP heuristic entity extraction) |
| **Module 3** | Computer Vision & Symbols | Member 3 & 1 | Multi-scale template matching across 12 ISO 15223-1 symbols with NMS |
| **Module 4** | Barcode & UDI Validation | Member 3 & 1 | OpenCV barcode/QR scanner + GS1 Modulo-10 checksum validation & OCR cross-match |
| **Module 5** | Compliance Rule Engine | Member 1 & 5 | 17 automated regulatory rules (FDA 21 CFR 801/830, EU MDR, ISO 15223-1) with scoring |
| **Module 6** | Report Generation | Member 1 & 4 | Export to Pandas DataFrame, Markdown, JSON, and CSV |
| **Module 7** | Database & Traceability | Member 1 | SQLite with SQLAlchemy ORM (Uploads, Validation Runs, Violations, Human Reviews) |

---

## 🚀 Quick Start

### 1. Run the End-to-End Demo
```bash
python run_demo.py
```

### 2. Run the Automated Test Suite (7 Unit & Integration Tests)
```bash
python tests/test_pipeline_integration.py
```

### 3. Evaluate Precision & Recall Metrics (HLD Section 11)
```bash
python tests/evaluate_metrics.py
```

### 4. Start the FastAPI REST Server
```bash
uvicorn verilabel.api.app:app --reload --port 8000
```
Interactive Swagger API documentation will be available at: `http://localhost:8000/docs`

---

## 💻 Streamlit Integration (Guide for Member 4 - Frontend)

Member 4 can import and run the entire verification pipeline in just a few lines of code:

```python
import streamlit as st
from verilabel.pipeline import VeriLabelPipeline

# Initialize the pipeline once
pipeline = VeriLabelPipeline()

uploaded_file = st.file_uploader("Upload Medical Device Label", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run verification pipeline
    result = pipeline.process_label(temp_path, save_to_db=True)

    # 1. Display KPI Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Compliance Score", f"{result.summary['compliance_score']}%")
    col2.metric("Verdict", result.summary['verdict'])
    col3.metric("Passed Checks", f"{result.summary['passed_checks']}/{result.summary['total_checks']}")

    # 2. Display Annotated Image (Overlay of detected symbols & barcodes)
    st.image(result.annotated_image, caption="Visual Inspection Overlay")

    # 3. Display Interactive Checklist Table
    st.dataframe(result.dataframe)

    # 4. Human Approval Workflow
    reviewer = st.text_input("Reviewer Name")
    role = st.text_input("Reviewer Role")
    decision = st.selectbox("Decision", ["APPROVED", "REJECTED", "CHANGES_REQUESTED"])
    notes = st.text_area("Review Notes")

    if st.button("Submit Expert Sign-Off"):
        pipeline.submit_human_review(
            validation_id=result.validation_id,
            reviewer_name=reviewer,
            reviewer_role=role,
            decision=decision,
            review_notes=notes
        )
        st.success("Human review logged to audit database!")
```

---

## 📊 Evaluation Results (from `tests/evaluate_metrics.py`)

- **Precision:** 100.00%
- **Recall (Defect Sensitivity):** 100.00%
- **F1-Score:** 100.00%
- **Accuracy:** 100.00%
- **Average Processing Latency:** ~8.8 seconds per label

---

## 👥 Team Brahmastra
- **Member 1 (Backend & Integration):** Python architecture, module orchestration, DB schema, REST API, compliance engine.
- **Member 2 (OCR & Document Processing):** PaddleOCR integration, text line detection, language models.
- **Member 3 (Computer Vision & Barcode):** OpenCV symbol detection, YOLO model weights, pyzbar.
- **Member 4 (Frontend & Dashboard):** Streamlit UI, visual dashboards, report downloads.
- **Member 5 (Compliance & Testing):** Regulatory rule authoring, test case labeling, audit documentation.
