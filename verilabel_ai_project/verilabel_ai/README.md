# VeriLabel AI — Enterprise Medical Device Regulatory Audit Platform
**Team Brahmastra &bull; GITAM (Deemed to be University), Bengaluru**  
*Motto: "Label Smarter. Comply Better."*

---

## 🌟 Executive Overview
VeriLabel AI has been upgraded into a complete, enterprise-grade, human-in-the-loop regulatory intelligence platform for medical device packaging audits. Built to comply with **FDA 21 CFR Part 801 / 830**, **EU MDR 2017/745 Annex I (Chapter III)**, and **ISO 15223-1:2021**, the platform eliminates manual review bottlenecks and packaging recall exposure.

---

## 🎨 Enterprise Design System
The user interface has been elevated following high-trust design standards inspired by **V7 Labs, Veryfi, LandingAI, Linear, and Stripe**:
- **Navy (`#142B45`):** Dark collapsible sidebar, deep header anchors, and primary title typography.
- **Teal (`#087E70`):** Clinical badges, verification highlights, active navigation accents, and approval buttons.
- **Blue (`#2563EB`):** Interactive primary actions and focused controls.
- **Background (`#F7F9FC`):** Ultra-clean, crisp clinical canvas.
- **White (`#FFFFFF`):** Elevated cards, data panels, and inspection surfaces.
- **Warning (`#E5A443`):** Amber advisory badges and conditional compliance alerts.
- **Error (`#DC5656`):** Crimson alerts for mandatory clause violations and checksum failures.
- **Success (`#168B69`):** Forest emerald for verified clearance states.

---

## 📑 12 Dedicated Workspace Navigation Sections
The modern collapsible sidebar provides instantaneous navigation across three core operational tiers:

### WORKSPACE
1. **Overview Dashboard (`#dashboard`):** Real-time analytics, total labels analyzed, compliance overview doughnut chart, monthly audit trend bar chart, recent analyses table, and quick upload CTA.
2. **Label Ingestion (`#ingestion`):** Drag-and-drop zone (PDF, PNG, JPG, JPEG), browse files, file size & type validation, preview card, device baseline selector, and 6 pre-loaded benchmark test cases.
3. **New AI Analysis (`#new-analysis`):** Multi-stage animated pipeline runner tracking Preprocessing, OCR, Symbol Vision, UDI Checksum, and Regulatory Rules.
4. **OCR Extraction (`#ocr`):** Split-screen viewer (original packaging artwork on left; structured extracted attributes, confidence scores, and missing field alerts on right).
5. **AI Visual Inspection (`#visual` - Signature Feature):** Interactive packaging canvas with actual bounding box overlays, view mode toggles (*Overlays*, *Raw Artwork*, *CLAHE High-Contrast*), zoom/pan controls, and clickable symbol cards showing ISO 15223-1 code, confidence %, and regulatory definition.
6. **Barcode / UDI Analysis (`#barcode`):** Carrier payload breakdown, parsed Application Identifiers `(01)` DI, `(17)` EXP, `(10)` LOT, GS1 Modulo-10 checksum validation badge, and automated cross-reconciliation with label text.

### VERIFICATION
7. **Compliance Results (`#compliance`):** Overall verdict banner with score gauge, applicable regulatory citations, and filterable 17-rule checklist (All, Critical, Major) with observed findings and remediation hints.
8. **Human Review Center (`#review`):** Dedicated reviewer workspace with pending review queue, decision selector (*Approved for Commercial Release*, *Conditional Approval*, *Rejected*), justification notes, and 21 CFR Part 11 legally binding electronic signature affirmation.
9. **21 CFR Part 11 Audit Ledger (`#ledger`):** Searchable, filterable historical transaction ledger with pagination, verdict filters, review status filters, and detailed record modal.

### MANAGEMENT
10. **Reports (`#reports`):** Full list of generated audit dossiers with instant links to **View PDF Dossier**, download JSON, and download CSV.
11. **Analysis History (`#history`):** Complete historical archive with one-click "Open Cockpit" to jump directly into detailed findings.
12. **Settings (`#settings`):** Auditor profile, institutional manufacturing facility code, governing regulatory jurisdiction, and 21 CFR Part 11 electronic records policy.

---

## 🚀 How to Run the Platform

From the workspace root directory:
```powershell
python run_app.py
```

Then access the web platform in your browser:
- **Platform Overview (Landing Page):** [http://localhost:8000/](http://localhost:8000/)
- **Auditor Sign In:** [http://localhost:8000/login](http://localhost:8000/login)
- **Terminal Registration:** [http://localhost:8000/register](http://localhost:8000/register)
- **Regulatory Cockpit & Dashboard:** [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

### Evaluator Personas (1-Click Sign-In on Login Page)
- **Dr. Sarah Jenkins** &bull; Lead Regulatory Auditor &bull; `sarah.jenkins@brahmastra.medtech` (Pass: `Auditor@2026`)
- **Alex Chen** &bull; QA Inspection Specialist &bull; `alex.chen@gitam.edu` (Pass: `Auditor@2026`)
- **Marcus Vance** &bull; Senior Compliance Officer &bull; `marcus.vance@fda-review.org` (Pass: `Auditor@2026`)
