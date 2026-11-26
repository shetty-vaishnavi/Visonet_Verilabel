"""
VeriLabel AI - Application Launcher
Team: Brahmastra | GITAM University
Platform: Medical Device Label Compliance Platform
Motto: "Label Smarter. Comply Better."

Run this script to launch the full VeriLabel AI platform:
    python run_app.py
"""

import sys
import os
from pathlib import Path

# Add app directory to sys.path
APP_DIR = Path(__file__).resolve().parent / "verilabel_app"
if not APP_DIR.exists():
    APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.chdir(str(APP_DIR))

import uvicorn

def main():
    print("=" * 80)
    print("   __      __        _ _           _           _        _   ")
    print("   \\ \\    / /       (_) |         | |   /\\    (_)      | |  ")
    print("    \\ \\  / /__ _ __  _| |     __ _| |  /  \\    _ ______| |  ")
    print("     \\ \\/ / _ \\ '__| | | |    / _` | | / /\\ \\  | |______| |  ")
    print("      \\  /  __/ |  | | | |___| (_| | |/ ____ \\ | |      |_|  ")
    print("       \\/ \\___|_|  |_|_|______\\__,_|_/_/    \\_\\|_|      (_)  ")
    print("=" * 80)
    print(" VERILABEL AI — AI-POWERED MEDICAL DEVICE LABEL COMPLIANCE PLATFORM")
    print(" Team: Brahmastra | GITAM (Deemed to be University), Bengaluru")
    print(" Motto: 'Label Smarter. Comply Better.'")
    print("=" * 80)
    print("\n[+] Initializing backend pipelines & database...")
    print("[+] Pre-seeded Demo Auditor Credentials:")
    print("    • Dr. Sarah Jenkins (Lead RA Auditor)    : sarah.jenkins@brahmastra.medtech | Pass: Auditor@2026")
    print("    • Alex Chen (QA Inspection Specialist)  : alex.chen@gitam.edu              | Pass: Auditor@2026")
    print("    • Marcus Vance (Senior Compliance Off.) : marcus.vance@fda-review.org      | Pass: Auditor@2026")
    print("\n[+] Web Application Endpoints:")
    print("    • Landing Page      : http://localhost:8000/")
    print("    • Auditor Sign In   : http://localhost:8000/login")
    print("    • Registration      : http://localhost:8000/register")
    print("    • Audit Console     : http://localhost:8000/dashboard")
    print("\n[+] Starting server on http://127.0.0.1:8000 (Press Ctrl+C to stop)...")
    print("=" * 80 + "\n")

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
