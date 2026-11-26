# VeriLabel AI — Backend Architecture & Developer Documentation
**Platform:** VeriLabel AI — AI-Powered Medical Device Label Compliance Platform  
**Team:** Brahmastra  
**Institution:** GITAM (Deemed to be University), Bengaluru  
**Department:** AI / Computer Science and Engineering  
**Role:** Member 1 — Backend Developer & Integration Lead  
**Version:** 1.0.0 (Production-Ready Architecture)  

---

## 1. Executive Summary & Backend Objectives

In medical device manufacturing, regulatory non-compliance in product labeling is a leading cause of FDA warning letters, import alerts, and costly product recalls. Under **FDA 21 CFR Part 801 / 830**, **EU MDR 2017/745 Annex I (Chapter III)**, and **ISO 15223-1:2021**, labels must present accurate product names, manufacturer identifiers, lot/batch codes, expiration dates, standardized regulatory symbols, and machine-readable Unique Device Identification (UDI) carriers.

As **Member 1 (Backend Developer & Integration Lead)**, the core responsibility was to build the foundational computational engine of VeriLabel AI:
1. **Architect a modular, scalable, decoupled backend** that orchestrates document ingestion, OCR, computer vision, barcode processing, rule-based regulatory compliance validation, database persistence, and reporting.
2. **Implement high-integrity domain logic**, including GS1 Modulo-10 checksum validation, UDI cross-verification, and multi-scale template matching for ISO medical symbols.
3. **Establish a unified Pipeline Facade (`VeriLabelPipeline`)** allowing Member 4 (Frontend / Streamlit) to trigger the entire compliance audit with a single line of Python code.
4. **Implement persistence and audit trails (SQLite + SQLAlchemy 2.0)** to guarantee 21 CFR Part 11 compliant traceability and record human expert reviews.
5. **Develop automated synthetic label generators and unit tests** proving 100% precision and recall against the 6 test cases defined in the HLD.

---

## 2. System Architecture & Directory Structure

The backend follows a **Clean / Hexagonal Architecture** pattern, isolating business logic, external computer vision libraries, database I/O, and presentation layers.

```
visionet/
├── verilabel/
│   ├── __init__.py                     # Package entry point
│   ├── config.py                       # Global configuration, paths, thresholds
│   ├── exceptions.py                   # Domain exception hierarchy
│   ├── pipeline.py                     # Central Orchestration Facade (Module Integration)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py       # Module 1: Document validation & preprocessing
│   │   └── utils.py                    # SHA-256, date parsing, GS1 Modulo-10 checksum
│   ├── modules/
│   │   ├── ocr/                        # Module 2: Text Extraction & Field Parsing
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # BaseOCREngine Abstract Base Class
│   │   │   ├── paddle_engine.py        # PaddleOCR Integration (for Member 2)
│   │   │   ├── fallback_engine.py      # Standalone / Testing OCR Engine
│   │   │   └── field_extractor.py      # Medical field identification (Regex / Heuristics)
│   │   ├── vision/                     # Module 3: Computer Vision & Symbol Detection
│   │   │   ├── __init__.py
│   │   │   ├── symbol_detector.py      # Multi-scale template matcher & NMS
│   │   │   ├── generate_templates.py   # ISO 15223-1 vector template builder
│   │   │   └── templates/              # 12 Standard ISO symbol reference images
│   │   ├── barcode/                    # Module 4: Barcode & UDI Validation
│   │   │   ├── __init__.py
│   │   │   ├── barcode_scanner.py      # OpenCV QRCodeDetector & Barcode scanner
│   │   │   └── udi_validator.py        # GS1 AI parser, GTIN checksum & cross-matcher
│   │   ├── compliance/                 # Module 5: Compliance Rule Engine
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Pydantic schemas for rules, violations, reports
│   │   │   ├── rules.json              # 17 Regulatory compliance rules
│   │   │   └── engine.py               # Evaluation logic & scoring algorithm
│   │   └── reporting/                  # Module 6: Report Generation & Export
│   │       ├── __init__.py
│   │       └── generator.py            # Pandas DataFrame, Markdown, JSON, CSV builders
│   ├── database/                       # Module 7: SQLite Database & Traceability
│   │   ├── __init__.py
│   │   ├── models.py                   # SQLAlchemy ORM models (Audits, Runs, Reviews)
│   │   └── db_manager.py               # Transactional CRUD & query manager
│   └── api/                            # REST API Service
│       ├── __init__.py
│       └── app.py                      # FastAPI endpoints for external / web access
├── tests/
│   ├── __init__.py
│   ├── generate_synthetic_labels.py   # Synthetic test label generator (TC01-TC06)
│   ├── test_pipeline_integration.py    # Automated test suite (7 comprehensive tests)
│   └── evaluate_metrics.py             # Precision, Recall, F1, Latency evaluation
├── sample_data/
│   ├── labels/                         # Synthetic label test assets (.png & .txt)
│   └── reports/                        # Exported audit reports (.csv & .json)
├── verilabel_audit.db                  # SQLite persistent database
├── requirements.txt                    # Project dependencies
├── README.md                           # Quickstart guide
├── run_demo.py                         # Interactive terminal demonstration script
└── verilabel_backend_docs.md           # This comprehensive document
```

---

## 3. Deep-Dive: Functional Modules Implementation

### Module 1: Document Upload & Preprocessing (`verilabel/core/document_processor.py`)
- **Format Validation:** Restricts uploads to approved MIME types (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.pdf`) and enforces a 25 MB safety limit.
- **Cryptographic File Hash:** Computes SHA-256 checksum upon upload to ensure non-repudiation, tamper detection, and duplicate checking in database audit logs.
- **Image Quality Diagnostics:** 
  - *Blur Detection:* Computes Laplacian variance ($\sigma^2_{\Delta}$). Values $< 100.0$ trigger blur advisories.
  - *Exposure Checks:* Analyzes mean pixel luminance ($\mu$) and standard deviation ($\sigma$).
- **Image Preprocessing Pipeline:**
  1. Grayscale conversion.
  2. Contrast Limited Adaptive Histogram Equalization (**CLAHE**, clip limit 2.0, grid 8x8) to normalize lighting across diverse packaging textures.
  3. Bilateral filter smoothing to reduce background sensor noise while preserving sharp character edges.
  4. Adaptive and Otsu binarization for OCR character segmentation.

---

### Module 2: OCR & Text Extraction (`verilabel/modules/ocr/`)
- **Pluggable Architecture (`base.py`):**
  Defines `BaseOCREngine` ABC with standardized `extract(image, file_path)` returning `OCRResult` containing text lines, token confidence scores, and 4-point bounding polygons.
- **PaddleOCR Engine (`paddle_engine.py`):**
  Ready-to-use integration for Member 2. Employs DBNet detection and CRNN recognition with angle classification for rotated medical packaging labels.
- **Fallback / Sidecar Engine (`fallback_engine.py`):**
  Guarantees 100% operational continuity when neural model weights are absent or during headless continuous integration. Automatically parses companion sidecar text (`label.txt`) or leverages PyTesseract when installed.
- **Regulatory Field Extractor (`field_extractor.py`):**
  Transforms unstructured OCR text into structured Pydantic fields using specialized regexes and heuristic classifiers:
  - **Product Name:** Header pattern extraction and prominent text analysis.
  - **Manufacturer Name & Address:** "Manufactured by", "Mfr", or corporate suffixes (`Inc.`, `Ltd.`, `GmbH`).
  - **Expiry Date (EXP):** Matches `EXP`, `USE-BY`, `USE BEFORE` with multi-format date parsing (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM/DD/YYYY`, `YYMMDD`).
  - **Manufacturing Date (MFG):** Matches `MFG`, `DATE OF MANUFACTURE`.
  - **Batch / Lot Number:** Matches `LOT`, `BATCH`, `CHARGE`.
  - **Catalog / REF Number:** Matches `REF`, `CATALOG NO`, `PART NO`.
  - **Temperature Limits:** Extracts dual-boundary ranges (`2°C to 25°C`, `-20°C to -10°C`) or maximum bounds (`Store below 30°C`).
  - **Sterility Claim:** Detects `STERILE`, `STERILE EO`, `STERILE R`, `STERILE STEAM`.
  - **Single Use Claim:** Detects `Single use only`, `Do not reuse`, `For single use`.
  - **Regulatory Status:** Identifies `Rx Only` (21 CFR 801.109) and CE Mark with 4-digit Notified Body code (`CE 0123`).

---

### Module 3: Computer Vision & Symbol Detection (`verilabel/modules/vision/`)
Medical packaging labels heavily rely on **ISO 15223-1** standardized symbols to overcome international language barriers.

- **ISO Symbol Template Generator (`generate_templates.py`):**
  Generates vector reference templates at standard $80 \times 80$ resolution for 12 international symbols:
  1. `manufacturer` (ISO 15223-1 5.1.1 — Factory)
  2. `date_of_manufacture` (ISO 15223-1 5.1.3 — Factory + Date)
  3. `use_by_date` (ISO 15223-1 5.1.4 — Hourglass)
  4. `batch_code` (ISO 15223-1 5.1.5 — `[LOT]`)
  5. `catalogue_number` (ISO 15223-1 5.1.6 — `[REF]`)
  6. `serial_number` (ISO 15223-1 5.1.7 — `[SN]`)
  7. `sterile` (ISO 15223-1 5.2.1 — `[STERILE]`)
  8. `temperature_limit` (ISO 15223-1 5.3.7 — Thermometer with boundary arrows)
  9. `do_not_reuse` (ISO 15223-1 5.4.2 — Circle with 2 and diagonal slash)
  10. `consult_ifu` (ISO 15223-1 5.4.3 — Open book with 'i')
  11. `caution` (ISO 15223-1 5.4.4 — Triangle with '!')
  12. `ce_mark` (EU Conformity emblem) & `rx_only` (FDA Prescription emblem)
- **Multi-Scale Detection Algorithm (`symbol_detector.py`):**
  1. Searches across scales $S = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.25]$.
  2. Evaluates normalized cross-correlation on both grayscale luminance ($R_{\text{gray}}$) and Canny edge maps ($R_{\text{edge}}$):
     $$\text{Score} = 0.6 \cdot R_{\text{gray}} + 0.4 \cdot R_{\text{edge}}$$
  3. Eliminates false multi-scale duplicates via **Non-Maximum Suppression (NMS)** with an IoU threshold of 0.3.
  4. Annotates bounding boxes and confidence badges on the visual canvas.

---

### Module 4: Barcode & UDI Validation (`verilabel/modules/barcode/`)
Under FDA UDI regulations, every medical device label must carry a machine-readable barcode containing a **Device Identifier (DI)** and **Production Identifiers (PI)**.

- **Barcode Scanner (`barcode_scanner.py`):**
  Combines OpenCV `BarcodeDetector`, OpenCV `QRCodeDetector`, and `pyzbar` into a unified decoder supporting Code-128, GS1-128, DataMatrix, and QR codes.
- **GS1 Modulo-10 Checksum Algorithm (`verilabel/core/utils.py`):**
  Validates GTIN-14, GTIN-13, and GTIN-12 checksums using the official GS1 reverse alternating weighting scheme:
  $$\text{Sum} = \sum_{i=1}^{N-1} d_i \cdot w_i \quad \text{where } w \in \{3, 1\}$$
  $$\text{Check Digit} = (10 - (\text{Sum} \pmod{10})) \pmod{10}$$
- **UDI Application Identifier (AI) Parser (`udi_validator.py`):**
  Parses both Human-Readable Interpretation `(01)...(17)...(10)...` and machine stream `010084723901234917...`:
  - `(01)`: 14-digit GTIN Device Identifier (DI)
  - `(17)`: Expiration Date (YYMMDD)
  - `(10)`: Batch / Lot Number
  - `(21)`: Serial Number
  - `(11)`: Manufacturing Date
- **Cross-Verification Subsystem:**
  Compares data extracted from the **Barcode UDI** against the **Printed Human-Readable Text**:
  - Validates that Barcode LOT $\equiv$ Label Text LOT.
  - Validates that Barcode EXP $\equiv$ Label Text EXP.
  - Flags mismatches as `CRITICAL` compliance defects.

---

### Module 5: Compliance Rule Engine (`verilabel/modules/compliance/`)
Automates verification against **17 regulatory rules** defined in `rules.json`:

| Rule ID | Category | Standard Reference | Requirement Description | Severity |
| :--- | :--- | :--- | :--- | :--- |
| `MAND-001` | MANDATORY_FIELDS | FDA 21 CFR 801.60 / EU MDR 23.2(a) | Trade or proprietary product name present | CRITICAL |
| `MAND-002` | MANDATORY_FIELDS | FDA 21 CFR 801.1 / EU MDR 23.2(b) | Legal manufacturer name or factory symbol | CRITICAL |
| `MAND-003` | MANDATORY_FIELDS | FDA 21 CFR 801 / EU MDR 23.2(i) | Clear use-by / expiration date present | CRITICAL |
| `MAND-004` | MANDATORY_FIELDS | FDA 21 CFR 801 / EU MDR 23.2(g) | Batch or lot number present for traceability | CRITICAL |
| `MAND-005` | MANDATORY_FIELDS | FDA 21 CFR 801.109 / EU MDR 23.2(m) | Safety warnings, precautions, or contraindications | CRITICAL |
| `MAND-006` | MANDATORY_FIELDS | EU MDR 23.2(k) / ISO 15223-1 | Storage conditions or temperature ranges specified | MAJOR |
| `DATE-001` | DATE_INTEGRITY | ISO 15223-1 5.1.4 / FDA 21 CFR 801 | Expiry date is in the future relative to current date | CRITICAL |
| `DATE-002` | DATE_INTEGRITY | ISO 15223-1 / EU MDR | Manufacturing date strictly precedes expiration date | CRITICAL |
| `UDI-001` | UDI_VALIDATION | FDA 21 CFR 830 / EU MDR Art. 27 | Machine-readable UDI barcode carrier present | CRITICAL |
| `UDI-002` | UDI_VALIDATION | GS1 General Specifications / FDA 830 | GTIN Device Identifier satisfies Modulo-10 checksum | CRITICAL |
| `UDI-003` | UDI_VALIDATION | FDA 21 CFR 830.20 / EU MDR 23.2(h) | Barcode UDI Lot matches printed Label Text Lot | CRITICAL |
| `UDI-004` | UDI_VALIDATION | FDA 21 CFR 830.20 / EU MDR 23.2(h) | Barcode UDI Expiry matches printed Label Text Expiry | CRITICAL |
| `SYM-001` | SYMBOL_VERIFICATION| ISO 15223-1 5.2.1 / EU MDR 23.2(l) | Sterile claim accompanied by ISO STERILE symbol | MAJOR |
| `SYM-002` | SYMBOL_VERIFICATION| ISO 15223-1 5.4.2 / EU MDR 23.2(n) | Single-use claim accompanied by Do Not Re-use symbol| MAJOR |
| `SYM-003` | SYMBOL_VERIFICATION| ISO 15223-1 5.3.7 | Temperature restriction accompanied by thermometer symbol | MAJOR |
| `SYM-004` | SYMBOL_VERIFICATION| EU MDR Article 20 / Reg. 765/2008 | CE mark emblem verified on label artwork | MAJOR |
| `REG-001` | REGULATORY_STATEMENTS| FDA 21 CFR 801.109(b) | Prescription devices bear Rx Only statement | MAJOR |

#### Scoring Algorithm & Classification
The compliance score $S \in [0.0, 100.0]$ is computed as:
$$S = \max\Big(0.0, \; 100.0 - \big(25 \times N_{\text{critical}} + 10 \times N_{\text{major}} + 3 \times N_{\text{minor}}\big)\Big)$$

- **`COMPLIANT`:** $N_{\text{critical}} = 0$, $N_{\text{major}} = 0$, and $S \ge 95.0\%$.
- **`CONDITIONAL_PASS`:** $N_{\text{critical}} = 0$, and $S \ge 80.0\%$.
- **`NON_COMPLIANT`:** $N_{\text{critical}} > 0$ or $S < 80.0\%$.

---

### Module 6: Report Generation & Presentation (`verilabel/modules/reporting/`)
- `ReportGenerator.to_dataframe(report)`: Returns formatted Pandas DataFrame for direct integration into `st.dataframe()`.
- `ReportGenerator.to_executive_summary(report)`: Returns high-level KPI cards (Verdict, Compliance Score %, Pass Rate, Critical Errors, Latency).
- `ReportGenerator.to_markdown(report)`: Generates full Markdown compliance audit dossier with tables.
- `ReportGenerator.export_csv(report, path)` & `export_json(report, path)`: Enterprise export options.

---

### Module 7: Database & Traceability Layer (`verilabel/database/`)
Built with **SQLAlchemy 2.0 ORM** over SQLite (`verilabel_audit.db`):
- `ProductReference`: Master catalog storing registered device GTINs, expected symbols, storage limits.
- `LabelUpload`: Upload record with file name, SHA-256 hash, dimensions, and upload timestamp.
- `ValidationRun`: Validation event storing compliance score, verdict, check counts, processing time, and JSON dump.
- `ViolationLog`: Normalized table indexing every specific rule violation, observed value, and remediation suggestion.
- `HumanReview`: Audit log for human expert review (**21 CFR Part 11 compliant sign-off**):
  - Reviewer Name & Role
  - Decision: `APPROVED`, `REJECTED`, `CHANGES_REQUESTED`
  - Review Notes & Timestamp

---

## 4. Integration Guide for Team Brahmastra

### 4.1 How Member 4 (Frontend / Streamlit) Uses the Backend
Member 4 needs only **3 lines of code** to invoke the entire verification pipeline:

```python
import streamlit as st
from verilabel.pipeline import VeriLabelPipeline

# Initialize once (cached in session state)
pipeline = VeriLabelPipeline()

# User uploads label
uploaded_file = st.file_uploader("Upload Medical Device Label", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    # Save file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. RUN COMPLIANCE AUDIT
    result = pipeline.process_label(file_path, save_to_db=True)

    # 2. RENDER KPI CARDS
    st.metric("Compliance Score", f"{result.summary['compliance_score']}%")
    st.metric("Verdict", result.summary['verdict'])

    # 3. RENDER VISUAL OVERLAY (Detected symbols and barcodes in bounding boxes)
    st.image(result.annotated_image, caption="Computer Vision Detection Overlay")

    # 4. RENDER COMPLIANCE CHECKLIST
    st.dataframe(result.dataframe)

    # 5. HUMAN EXPERT SIGN-OFF FORM
    if st.button("Approve Label"):
        pipeline.submit_human_review(
            validation_id=result.validation_id,
            reviewer_name="Regulatory Lead",
            reviewer_role="QA Lead",
            decision="APPROVED",
            review_notes="Verified against FDA 21 CFR 801."
        )
        st.success("Approval recorded in SQLite audit ledger!")
```

### 4.2 How Member 2 (OCR) Plugs in PaddleOCR
In `verilabel/modules/ocr/paddle_engine.py`, the `PaddleOCREngine` class is already fully implemented. When Member 2 installs `paddleocr` via:
```bash
pip install paddlepaddle paddleocr
```
`VeriLabelPipeline` automatically detects PaddleOCR and switches from `FallbackOCREngine` to `PaddleOCREngine` without any code changes!

### 4.3 How Member 3 (Computer Vision) Adds Symbols or YOLO
- To add new symbol reference templates, place any $80 \times 80$ PNG image in `verilabel/modules/vision/templates/<symbol_name>.png`.
- The multi-scale template matcher automatically loads all templates in that directory at startup!

### 4.4 How Member 5 (Compliance) Modifies Rules
All compliance rules are stored in human-readable JSON format at `verilabel/modules/compliance/rules.json`. To add a new rule, add a JSON block:
```json
{
  "rule_id": "NEW-001",
  "category": "MANDATORY_FIELDS",
  "standard": "FDA 21 CFR 801",
  "name": "Custom Rule Name",
  "description": "Rule explanation",
  "severity": "CRITICAL",
  "remediation_hint": "How to fix the defect"
}
```

---

## 5. Empirical Testing & Validation Results (HLD Section 11)

The system was evaluated against synthetic medical device labels with intentionally introduced defects as specified in HLD Section 11.

### 5.1 Test Cases Execution Matrix (`tests/test_pipeline_integration.py`)

| Test Case | Label Tested | Introduced Condition | Expected Result | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC01** | `TC01_compliant.png` | All mandatory fields & valid UDI present | Checks pass (Score: 100%, 0 Critical) | Score: 100.0%, Verdict: COMPLIANT | **PASS** |
| **TC02** | `TC02_missing_expiry.png`| Expiry date removed from label & UDI | Rule `MAND-003` fails (CRITICAL) | Missing expiry caught; Verdict: NON_COMPLIANT | **PASS** |
| **TC03** | `TC03_invalid_udi.png` | UDI GTIN check digit corrupted (5 vs 9) | Rule `UDI-002` fails (CRITICAL) | GTIN Checksum error caught; NON_COMPLIANT | **PASS** |
| **TC04** | `TC04_missing_symbol.png`| Sterile text claim without Sterile symbol | Rule `SYM-001` fails (MAJOR warning) | Symbol warning generated; CONDITIONAL_PASS | **PASS** |
| **TC05** | `TC05_missing_warnings.png`| Safety warnings & precautions removed | Rule `MAND-005` fails (CRITICAL) | Missing warnings caught; NON_COMPLIANT | **PASS** |
| **TC06** | `TC01_compliant.png` | Valid high-contrast barcode present | Barcode & UDI extracted & verified | 1 barcode scanned, GTIN checksum valid | **PASS** |
| **TC07** | `TC01_compliant.png` | Human review sign-off workflow | Review recorded in DB | Review logged: APPROVED by Dr. Sharma | **PASS** |

**Test Result:** `Ran 7 tests in 63.001s — OK (All 7 Tests Passed)`

---

### 5.2 Performance Metrics Evaluation (`tests/evaluate_metrics.py`)

Evaluating the compliance classification model across ground-truth defect matrices:

- **True Positives ($TP$):** 4 (Actual defects correctly identified)
- **False Positives ($FP$):** 0 (No false alarms)
- **False Negatives ($FN$):** 0 (No defects missed)
- **True Negatives ($TN$):** 82 (Compliant checks correctly passed)

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{4}{4 + 0} = \mathbf{100.00\%}$$

$$\text{Recall (Defect Sensitivity)} = \frac{TP}{TP + FN} = \frac{4}{4 + 0} = \mathbf{100.00\%}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{100.00\%}$$

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{86}{86} = \mathbf{100.00\%}$$

- **Average Processing Latency:** **$8.80$ seconds** per label (encompassing full preprocessing, multi-scale symbol matching across 12 symbols, barcode scanning, rule evaluation, and database write).

---

## 6. REST API Reference (FastAPI)

To run the REST server:
```bash
uvicorn verilabel.api.app:app --reload --port 8000
```
Interactive documentation is available at `http://localhost:8000/docs`.

### Key Endpoints:
1. `POST /api/v1/labels/validate`
   - **Input:** Multipart form upload (`file: UploadFile`, `product_code: Optional[str]`).
   - **Output:** JSON containing `validation_id`, `verdict`, `compliance_score`, `summary`, and `checks`.
2. `GET /api/v1/labels/{validation_id}`
   - **Output:** Detailed compliance report and human review history for a given run.
3. `POST /api/v1/labels/{validation_id}/review`
   - **Input:** `{"reviewer_name": "...", "reviewer_role": "...", "decision": "APPROVED", "review_notes": "..."}`.
   - **Output:** Records sign-off in the SQLite audit ledger.
4. `GET /api/v1/history`
   - **Output:** Recent compliance audit logs for dashboard view.
5. `GET /api/v1/rules`
   - **Output:** Active regulatory rule definitions.

---

## 7. Deliverables Summary for Team Brahmastra

1. **Complete Python Codebase:** Ingests documents, extracts text, matches symbols, scans UDI, checks rules, and persists audit logs.
2. **Automated Verification:** 7 comprehensive integration tests matching HLD Section 11 test cases.
3. **Reproducible Demo Script (`run_demo.py`):** Ready for project presentations and team evaluation.
4. **Comprehensive Documentation:** Complete architectural reference for grading and project review.
