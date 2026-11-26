"""
run_demo.py — Interactive Demo for Module 3: Computer Vision & Barcode
=======================================================================
Team: Brahmastra | GITAM University
Member 3: Computer Vision and Barcode Module

Usage (from project root):
    $env:PYTHONPATH=".\visionet(CV_Barcode)\visionet"
    python .\visionet(CV_Barcode)\visionet\run_demo.py
"""

import sys
import os
import time
import dataclasses
import json
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

# ── Colour helpers (ANSI) ────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

PASS = f"{GREEN}✅ PASS{RESET}"
FAIL = f"{RED}❌ FAIL{RESET}"
WARN = f"{YELLOW}⚠  WARN{RESET}"


def divider(char="─", width=70):
    print(f"{DIM}{char * width}{RESET}")


def header(text: str):
    print()
    divider("═")
    print(f"  {BOLD}{CYAN}{text}{RESET}")
    divider("═")


def section(text: str):
    print(f"\n{BOLD}{YELLOW}  {text}{RESET}")
    divider()


def metric(label: str, value, ok: bool | None = None):
    icon = ""
    if ok is True:
        icon = f"{GREEN}✅{RESET} "
    elif ok is False:
        icon = f"{RED}❌{RESET} "
    print(f"    {icon}{label:<35} {BOLD}{value}{RESET}")


# ── Test Case definitions ────────────────────────────────────────────────────
SAMPLE_DIR = MODULE_DIR / "sample_labels"
OUTPUT_DIR = MODULE_DIR / "demo_output"

TEST_CASES = [
    {
        "id": "TC-01",
        "file": "tc01_compliant_label.png",
        "title": "Fully Compliant Label",
        "description": "GS1 DataMatrix with valid GTIN-14, 8 ISO 15223-1 symbols",
        "expect_barcodes": 1,
        "expect_valid_udi": True,
        "expect_symbols_min": 1,
        "expect_quality_min": 50,
    },
    {
        "id": "TC-02",
        "file": "tc02_invalid_checksum.png",
        "title": "Invalid Checksum Label",
        "description": "DataMatrix barcode with corrupted GTIN-14 check digit",
        "expect_barcodes": 1,
        "expect_valid_udi": False,
        "expect_symbols_min": 1,
        "expect_quality_min": 50,
    },
    {
        "id": "TC-03",
        "file": "tc03_multiple_barcodes.png",
        "title": "Multiple Barcodes",
        "description": "GS1 DataMatrix + Code-128 on the same label",
        "expect_barcodes": 2,
        "expect_valid_udi": True,
        "expect_symbols_min": 1,
        "expect_quality_min": 50,
    },
    {
        "id": "TC-04",
        "file": "tc04_skewed_and_noisy.png",
        "title": "Skewed & Noisy Label",
        "description": "TC-01 rotated 12° with Gaussian noise — tests deskew + denoise",
        "expect_barcodes": 1,
        "expect_valid_udi": True,
        "expect_symbols_min": 0,
        "expect_quality_min": 30,
    },
    {
        "id": "TC-05",
        "file": "tc05_inverted_datamatrix.png",
        "title": "Inverted (Photonegative) Label",
        "description": "Photonegative of TC-01 — tests inverted-image fallback scan",
        "expect_barcodes": 1,
        "expect_valid_udi": True,
        "expect_symbols_min": 0,
        "expect_quality_min": 30,
    },
    {
        "id": "TC-06",
        "file": "tc06_missing_symbols.png",
        "title": "Missing ISO Symbols",
        "description": "DataMatrix only — no ISO 15223-1 symbols present on label",
        "expect_barcodes": 1,
        "expect_valid_udi": True,
        "expect_symbols_min": 0,
        "expect_quality_min": 50,
    },
]


# ── Main demo logic ──────────────────────────────────────────────────────────
def run_demo():
    header("VeriLabel AI — Module 3: Computer Vision & Barcode Demo")
    print(f"  {DIM}Team Brahmastra | GITAM University{RESET}")
    print(f"  Pipeline: {CYAN}ComputerVisionBarcodePipeline{RESET}")
    print(f"  Test Cases: {BOLD}{len(TEST_CASES)}{RESET} labels (TC-01 to TC-06)")

    # ── Import pipeline ──────────────────────────────────────────────────────
    section("🔧  Initialising Pipeline")
    try:
        from verilabel_cv_barcode import ComputerVisionBarcodePipeline
        t0 = time.time()
        pipeline = ComputerVisionBarcodePipeline()
        init_ms = (time.time() - t0) * 1000
        print(f"    {GREEN}Pipeline loaded in {init_ms:.0f} ms{RESET}")
        print(f"    Templates dir : {pipeline.templates_dir}")
        print(f"    Templates loaded: {len(pipeline.symbol_detector.templates)} symbols")
    except ImportError as e:
        print(f"  {RED}Import error: {e}{RESET}")
        print(f"  {YELLOW}Hint: set PYTHONPATH to this module's directory first.{RESET}")
        sys.exit(1)

    # ── Check sample labels exist ────────────────────────────────────────────
    section("📂  Checking Sample Labels")
    missing = []
    for tc in TEST_CASES:
        p = SAMPLE_DIR / tc["file"]
        if p.exists():
            print(f"    {GREEN}✓{RESET}  {tc['file']}")
        else:
            print(f"    {RED}✗{RESET}  {tc['file']}  {RED}(MISSING){RESET}")
            missing.append(tc["file"])

    if missing:
        print(f"\n  {YELLOW}⚠  Some sample labels are missing.{RESET}")
        print(f"  Run:  python {SAMPLE_DIR}/generate_samples.py")
        regen = input("\n  Auto-regenerate now? [y/N]: ").strip().lower()
        if regen == "y":
            print("  Generating …")
            import subprocess
            subprocess.run(
                [sys.executable, str(SAMPLE_DIR / "generate_samples.py")],
                check=True,
            )
            print(f"  {GREEN}Done.{RESET}")
        else:
            print(f"  {RED}Aborting demo — missing labels.{RESET}")
            sys.exit(1)

    # ── Create output directory ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run each test case ───────────────────────────────────────────────────
    all_pass = True
    summary_rows = []

    for tc in TEST_CASES:
        header(f"{tc['id']} — {tc['title']}")
        print(f"  {DIM}{tc['description']}{RESET}")

        image_path = str(SAMPLE_DIR / tc["file"])
        t_start = time.time()

        try:
            result = pipeline.process(
                image_path=image_path,
                ocr_text="",
                output_dir=str(OUTPUT_DIR),
            )
        except Exception as exc:
            print(f"  {RED}Pipeline error: {exc}{RESET}")
            all_pass = False
            summary_rows.append({"tc": tc["id"], "status": "ERROR", "detail": str(exc)})
            continue

        elapsed_ms = (time.time() - t_start) * 1000

        # Evaluate assertions
        q_ok = result.quality_score >= tc["expect_quality_min"]
        b_ok = result.barcodes_found >= tc["expect_barcodes"]
        s_ok = result.symbols_detected >= tc["expect_symbols_min"]

        has_valid_udi = any(
            u.checksum_valid for u in result.udi_results
        ) if result.udi_results else False
        u_ok = (has_valid_udi == tc["expect_valid_udi"])

        tc_pass = q_ok and b_ok and s_ok and u_ok
        if not tc_pass:
            all_pass = False

        # Print metrics
        section("  📊  Results")
        metric("Quality Score", f"{result.quality_score:.1f}/100  [{result.quality_grade}]", q_ok)
        metric("Barcodes Found", result.barcodes_found, b_ok)
        metric("Symbols Detected", result.symbols_detected, s_ok)

        if result.udi_results:
            u = result.udi_results[0]
            metric("GTIN-14", u.gtin or "N/A")
            metric("Lot Number", u.lot_number or "N/A")
            metric("Expiry Date", u.expiry_date or "N/A")
            metric("Serial Number", u.serial or "N/A")
            metric("Checksum Valid", u.checksum_valid, u_ok)
            metric("UDI Standard", u.udi_standard)
        else:
            metric("UDI Validation", "No UDI found", False if tc["expect_valid_udi"] else None)

        metric("Cross-Verification", result.cross_verification.status)
        metric("Processing Time", f"{elapsed_ms:.0f} ms")

        if result.annotated_image_path:
            metric("Annotated Output", Path(result.annotated_image_path).name)

        if result.errors:
            section("  ⚠  Warnings")
            for e in result.errors:
                print(f"    {YELLOW}{e}{RESET}")

        # Barcode details
        if result.barcode_results:
            section("  🔍  Barcode Details")
            for i, bc in enumerate(result.barcode_results, 1):
                print(f"    [{i}] Format: {CYAN}{bc.format}{RESET}")
                txt = bc.text[:60] + "…" if len(bc.text) > 60 else bc.text
                print(f"        Text  : {txt}")
                print(f"        Region: {bc.position}")

        # Symbol details
        if result.symbol_results:
            section("  🏷  Symbol Details")
            for sym in result.symbol_results:
                bar = "█" * int(sym.confidence * 20)
                print(f"    {sym.name:<30} {CYAN}{bar}{RESET} {sym.confidence:.2f}")

        # Verdict
        section("  📋  Verdict")
        status = PASS if tc_pass else FAIL
        print(f"    {tc['id']} overall: {status}")

        summary_rows.append({
            "tc": tc["id"],
            "title": tc["title"],
            "status": "PASS" if tc_pass else "FAIL",
            "quality": f"{result.quality_score:.1f}",
            "barcodes": result.barcodes_found,
            "symbols": result.symbols_detected,
            "udi_valid": has_valid_udi,
            "time_ms": f"{elapsed_ms:.0f}",
        })

        input(f"\n  {DIM}Press ENTER to continue to next test case …{RESET}")

    # ── Final Summary ────────────────────────────────────────────────────────
    header("Demo Complete — Summary")

    col_w = [8, 32, 8, 10, 10, 10, 12, 10]
    cols  = ["TC", "Title", "Status", "Quality", "Barcodes", "Symbols", "UDI Valid", "Time (ms)"]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(f"\n  {BOLD}" + row_fmt.format(*cols) + RESET)
    divider()

    passed = 0
    for row in summary_rows:
        status_col = f"{GREEN}PASS{RESET}" if row["status"] == "PASS" else f"{RED}FAIL{RESET}"
        print("  " + row_fmt.format(
            row["tc"],
            row.get("title", "")[:30],
            status_col,
            row.get("quality", "—"),
            str(row.get("barcodes", "—")),
            str(row.get("symbols", "—")),
            str(row.get("udi_valid", "—")),
            row.get("time_ms", "—"),
        ))
        if row["status"] == "PASS":
            passed += 1

    divider()
    total = len(summary_rows)
    overall = PASS if all_pass else FAIL
    print(f"\n  Overall: {overall}  —  {BOLD}{passed}/{total}{RESET} test cases passed")
    print(f"  Annotated labels saved to: {CYAN}{OUTPUT_DIR}{RESET}")

    # Save JSON summary
    summary_path = OUTPUT_DIR / "demo_summary.json"
    with open(summary_path, "w") as f:
        # Strip ANSI from status before saving
        clean_rows = []
        for r in summary_rows:
            clean_rows.append({k: v for k, v in r.items()})
        json.dump({"passed": passed, "total": total, "results": clean_rows}, f, indent=2)
    print(f"  JSON summary    saved to: {CYAN}{summary_path}{RESET}")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Demo interrupted by user.{RESET}\n")
        sys.exit(0)
