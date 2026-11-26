# VeriLabel AI — Module 3: Computer Vision & Barcode Module

**Platform:** Medical Device Label Verification & Compliance
**Team:** Brahmastra | **Institution:** GITAM (Deemed to be University), Bengaluru
**Module Owner:** Member 3 — Computer Vision and Barcode Module
**Version:** 1.0.0 (Production-Ready)

---

## 1. Overview & Scope

Module 3 is the **computer vision and machine-readable barcode inspection engine** of the VeriLabel AI platform. It operates on pre-ingested packaging label images and delivers three regulatory outputs:

1. **OpenCV Packaging Quality Diagnostics** — Laplacian blur, illumination, contrast, deskew
2. **Multi-Engine Barcode Decoding & UDI Validation** — GS1 DataMatrix, Code-128, QR, EAN-13 with Modulo-10 GTIN checksum and bidirectional cross-verification against OCR fields
3. **ISO 15223-1 Symbol Detection** — Multi-scale template matching with Dual Correlation Fusion and two-pass NMS across all 13 standard medical packaging symbols

```
                    ┌──────────────────────────────┐
                    │  Packaging Label Image (BGR)  │
                    └───────────────┬──────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────┐
            │          ImageProcessor                   │
            │  • SHA-256 integrity hash                 │
            │  • Laplacian blur & illumination check    │
            │  • CLAHE + bilateral + unsharp masking    │
            │  • Hough-line deskew (canvas expansion)   │
            └───────────────────┬───────────────────────┘
                    ┌───────────┴────────────┐
                    ▼                        ▼
     ┌──────────────────────┐  ┌───────────────────────────┐
     │   BarcodeScanner     │  │   MedicalSymbolDetector   │
     │  • zxingcpp (C++)    │  │  • 13 ISO 15223-1 TPLs    │
     │  • OpenCV QR/1D      │  │  • Multi-scale pyramid    │
     │  • pyzbar fallback   │  │  • Dual Correlation NCC   │
     │  • Rotation retries  │  │  • Two-pass NMS           │
     │  • Polarity inversion│  └───────────────┬───────────┘
     └──────────┬───────────┘                  │
                ▼                              │
     ┌──────────────────────┐                  │
     │    UDIValidator      │                  │
     │  • GS1 AI Parsing    │                  │
     │  • GTIN Modulo-10    │                  │
     │  • HIBCC Mod-43      │                  │
     │  • OCR cross-verify  │                  │
     └──────────┬───────────┘                  │
                └──────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │         Module3Result          │
              │  • Barcodes & UDI Components   │
              │  • ISO Symbols Detected        │
              │  • Cross-Validation Report     │
              │  • Annotated Overlay Image     │
              │  • Quality Score & Metrics     │
              └────────────────────────────────┘
```

---

## 2. Directory Structure

```
visionet(CV_Barcode)/
└── visionet/
    ├── verilabel_cv_barcode/
    │   ├── __init__.py              # Package exports & public API
    │   ├── config.py                # Thresholds, paths, colour palette
    │   ├── models.py                # Pydantic/dataclass schemas (Module3Result, etc.)
    │   ├── image_processor.py       # OpenCV ingestion, quality assessment, deskewing
    │   ├── barcode_scanner.py       # Multi-engine barcode detection & decoding
    │   ├── udi_validator.py         # GS1 / HIBCC UDI parsing & cross-verification
    │   ├── symbol_detector.py       # ISO 15223-1 multi-scale template matcher
    │   ├── visualizer.py            # Annotated overlay & side-by-side renderer
    │   ├── pipeline.py              # Central orchestration facade
    │   ├── cli.py                   # Command-line interface
    │   ├── generate_templates.py    # ISO symbol vector template builder
    │   └── templates/               # 13 ISO 15223-1 PNG reference templates
    ├── sample_labels/               # 6 synthetic test labels (TC01–TC06)
    │   └── generate_samples.py      # Automated label generator
    ├── tests/
    │   └── test_cv_barcode_module.py  # 59 unit & integration tests
    ├── cv_output/                   # Annotated overlay outputs (auto-generated)
    ├── requirements.txt             # Module dependencies
    └── README.md                    # This documentation
```

---

## 3. Quick Start

### Run Unit & Integration Tests (59 Tests)
```bash
# From the VeriLabel-AI root directory:
set PYTHONPATH=.\visionet(CV_Barcode)\visionet
python -m pytest visionet(CV_Barcode)/visionet/tests/test_cv_barcode_module.py -v
```
**Expected result: `59 passed`**

### Process a Single Label (CLI)
```bash
set PYTHONPATH=.\visionet(CV_Barcode)\visionet
python -m verilabel_cv_barcode.cli --input visionet(CV_Barcode)/visionet/sample_labels/tc01_compliant_label.png
```

### Batch Process a Directory (CLI)
```bash
python -m verilabel_cv_barcode.cli --batch visionet(CV_Barcode)/visionet/sample_labels/ --output cv_output/ --json
```

### Regenerate All 6 Synthetic Test Labels
```bash
python visionet(CV_Barcode)/visionet/sample_labels/generate_samples.py
```

### Python API Integration
```python
from verilabel_cv_barcode.pipeline import ComputerVisionBarcodePipeline
from verilabel_cv_barcode.config import PipelineConfig

pipeline = ComputerVisionBarcodePipeline()

result = pipeline.process(
    "sample_labels/tc01_compliant_label.png",
    ocr_fields={"lot_number": "LOT8842", "expiry_date": "2028-06-30"},
)

print(f"Barcodes found : {len(result.barcodes)}")
print(f"Valid UDI      : {result.has_valid_udi}")
print(f"Symbols found  : {result.symbol_count}")
print(f"Quality score  : {result.quality.quality_score:.1f}/100")
print(f"Cross-verify   : {'PASS' if result.cross_validation.is_consistent else 'FAIL'}")
```

---

## 4. Technical Implementation Deep-Dive

### 4.1 OpenCV Image Processing & Quality Assessment (`image_processor.py`)

| Operation | Algorithm | Purpose |
|-----------|-----------|---------|
| **Blur Detection** | Laplacian Variance $\sigma^2_\Delta = \text{Var}(\nabla^2 I)$ | Flag images below 100.0 as blurred |
| **Exposure Check** | Mean luminance $\mu$, overexposed pixel ratio | Detect glare / underexposure |
| **Dynamic Range** | Min-max pixel intensity spread | Flag low-contrast labels |
| **Skew Estimation** | Hough Line Transform (median angle) | Detect packaging orientation |
| **Deskewing** | Affine rotation with canvas expansion | Correct skewed labels without cropping |
| **Enhancement** | CLAHE (L*a*b*) + bilateral filter + unsharp mask | Normalize lighting and sharpen |
| **Barcode Prep** | 6 preprocessing variants | CLAHE, Otsu, Inverted, Morph-Close, Unsharp |

### 4.2 Barcode Scanner Architecture (`barcode_scanner.py`)

```
zxing-cpp (Primary)  →  DataMatrix, GS1-128, QR, Code-128, Code-39, EAN-13
OpenCV QRCodeDetector →  QR Code fallback
OpenCV BarcodeDetector → 1D linear barcode fallback
pyzbar (Optional)    →  ZBar C-library backup

Adaptive Retry Chain:
  1. Original image (all engines)
  2. CLAHE enhanced → all engines
  3. Otsu binary → all engines
  4. Inverted polarity → all engines    (laser-etched metal labels)
  5. Morphological close → all engines  (printer defect repair)
  6. Unsharp masked → all engines
  7. Orthogonal rotations: 90°, 180°, 270°
  8. 2x bicubic upscale (micro-barcode fallback)
```

### 4.3 GS1 Modulo-10 GTIN Checksum Validation

$$\text{Sum} = \sum_{i=1}^{N-1} d_i \cdot w_i \quad \text{where weights} \in \{3, 1, 3, 1, \ldots\}\ (\text{right-to-left})$$

$$\text{Check Digit} = \left(10 - (\text{Sum} \bmod 10)\right) \bmod 10$$

Supported: **GTIN-14, GTIN-13 (EAN-13), GTIN-12 (UPC-A), GTIN-8 (EAN-8)**

### 4.4 ISO 15223-1 Symbol Detection Algorithm (`symbol_detector.py`)

| Symbol | ISO Clause | Regulatory Standard |
|--------|-----------|---------------------|
| Manufacturer | 5.1.1 | ISO 15223-1 |
| Date of Manufacture | 5.1.3 | ISO 15223-1 |
| Use-by Date | 5.1.4 | ISO 15223-1 |
| Batch Code `[LOT]` | 5.1.5 | ISO 15223-1 |
| Catalogue Number `[REF]` | 5.1.6 | ISO 15223-1 |
| Serial Number `[SN]` | 5.1.7 | ISO 15223-1 |
| Sterile `[STERILE]` | 5.2.1 | ISO 15223-1 |
| Temperature Limit | 5.3.7 | ISO 15223-1 |
| Do Not Re-use | 5.4.2 | ISO 15223-1 |
| Consult IFU | 5.4.3 | ISO 15223-1 |
| Caution | 5.4.4 | ISO 15223-1 |
| CE Mark | EU-CE | EU MDR Article 20 |
| Rx Only | FDA-Rx | FDA 21 CFR 801.109 |

**Detection Algorithm:**

1. **Multi-scale pyramid** across scales $S = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.50]$
2. **Dual Correlation Fusion** (grayscale NCC + Canny edge NCC):

$$\text{Score} = 0.60 \cdot R_{\text{gray}} + 0.40 \cdot R_{\text{edge}}$$

3. **Two-Pass Non-Maximum Suppression (NMS):**
   - Pass 1: Intra-symbol scale suppression at IoU ≥ 0.30
   - Pass 2: Inter-symbol conflict resolution at IoU ≥ 0.40
4. **Zero Hallucination Guarantee:** Blank canvases (variance < 10) return zero detections

---

## 5. HLD Test Cases — Verification Matrix

| TC | Label Type | Barcode Expected | Symbol Expected | Validation |
|:--:|------------|:----------------:|:---------------:|------------|
| **TC01** | Fully Compliant | ✅ DataMatrix (valid checksum) | ✅ 8+ symbols | Pipeline PASS |
| **TC02** | Invalid UDI Checksum | ✅ DataMatrix (corrupted digit) | ✅ Present | Checksum FAIL flagged |
| **TC03** | Multiple Barcodes | ✅ DataMatrix + Code-128 | ✅ Subset | 2+ barcodes decoded |
| **TC04** | Skewed + Noisy | ✅ Recovered post-deskew | ✅ Present | Auto-deskew applies |
| **TC05** | Inverted Polarity | ✅ Recovered via inversion | (inverted) | Polarity retry succeeds |
| **TC06** | Missing Symbols | ✅ DataMatrix present | ❌ None | Symbol absence detected |

---

## 6. Backend Integration Guide (Member 1)

The module exposes a clean, decoupled `ComputerVisionBarcodePipeline` facade:

```python
from verilabel_cv_barcode import ComputerVisionBarcodePipeline, PipelineConfig
from pathlib import Path

pipeline = ComputerVisionBarcodePipeline(PipelineConfig(
    templates_dir=Path("verilabel_cv_barcode/templates"),
    symbol_match_threshold=0.58,
    enable_adaptive_barcode_retries=True,
))

# From the VeriLabel backend pipeline.py, pass the loaded image:
result = pipeline.process(
    source=image_array,         # np.ndarray from DocumentProcessor
    file_name="label_XYZ.png",
    ocr_fields={                # From Member 2 (OCR) ExtractedLabelFields
        "lot_number": extracted.lot_number,
        "expiry_date": extracted.expiry_date,
        "serial_number": extracted.serial_number,
    },
    save_visuals=True,
)

# result.barcodes          → List[DecodedBarcode]
# result.udi_records       → List[UDIComponents]  (with GTIN checksum status)
# result.symbols           → List[DetectedSymbol] (with ISO clause codes)
# result.cross_validation  → CrossValidationResult (LOT / EXP match status)
# result.to_dict()         → Full JSON-serializable audit report
```

---

## 7. Test Coverage Summary

```
59 tests | 4 test classes | 100% PASS rate

TestImageProcessingAndQuality (15 tests)
  - Multi-format ingestion (Path, bytes, BytesIO, PIL, numpy)
  - SHA-256 cryptographic hash verification
  - Blur / overexposure / underexposure classification
  - Quality score bounded [0, 100]
  - Deskew canvas expansion
  - Enhancement pipeline output integrity
  - Barcode preprocessing variant generation

TestBarcodeAndUDIScanner (28 tests)
  - Genuine DataMatrix, Code-128, Code-39, EAN-13 decoding
  - Multi-barcode detection (TC03)
  - Orthogonal rotation recovery (90°, 180°, 270°)
  - Inverted polarity (photonegative) decoding
  - Zero hallucination on blank canvas
  - GTIN-14 / GTIN-13 Modulo-10 valid & invalid
  - GS1 HRI parenthesized parsing
  - GS1 continuous machine-stream parsing
  - GS1 Day-00 month-end rule
  - HIBCC Modulo-43 check character validation
  - Cross-verification: LOT match / mismatch
  - Cross-verification: Expiry match / mismatch
  - ScanTelemetry metadata structure

TestMedicalSymbolDetector (7 tests)
  - All 13 ISO template registrations
  - Symbol detection on compliant label
  - Zero false positives on blank canvas
  - NMS redundancy suppression
  - Dual correlation score range [0, 1]
  - Annotation overlay dimension integrity
  - Missing symbols → fewer detections

TestPipelineIntegration (11 tests)
  - TC01–TC06 end-to-end HLD scenarios
  - Cross-verification PASS and FAIL paths
  - In-memory NumPy array input
  - JSON serialization of Module3Result
  - Processing latency logging
```
