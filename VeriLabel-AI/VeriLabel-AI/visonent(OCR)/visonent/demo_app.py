"""
Standalone Interactive Web Application & Demo Runner for Module 2: OCR & Document Processing.
Built with FastAPI and responsive modern UI to inspect OCR extractions, bounding boxes,
and structured medical fields in real time.

Usage:
    python demo_app.py --server --port 8501
    python demo_app.py --file sample_labels/tc01_compliant_label.png
"""

import os
import sys
import argparse
import base64
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from verilabel_ocr.pipeline import DocumentProcessingPipeline
from verilabel_ocr.config import PipelineConfig

# Initialize pipeline
pipeline = DocumentProcessingPipeline(PipelineConfig(output_dir="ocr_output"))

app = FastAPI(title="VeriLabel AI - OCR & Document Processing Explorer")

# Ensure static and output directories exist
os.makedirs("ocr_output", exist_ok=True)
os.makedirs("sample_labels", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="ocr_output"), name="outputs")
app.mount("/samples", StaticFiles(directory="sample_labels"), name="samples")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriLabel AI - Module 2 (OCR & Document Processing)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">

    <!-- Top Navigation Bar -->
    <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20 text-lg">
                    VL
                </div>
                <div>
                    <h1 class="text-lg font-bold text-white tracking-wide">VeriLabel AI <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-900/60 text-cyan-300 border border-cyan-700/50 ml-2">Module 2</span></h1>
                    <p class="text-xs text-slate-400">OCR & Document Processing • Team Brahmastra</p>
                </div>
            </div>
            <div class="flex items-center space-x-4 text-xs">
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/70 border border-emerald-700/50 text-emerald-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    PaddleOCR / RapidOCR Active
                </span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 py-8">
        
        <!-- Upload & Preset Row -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            
            <!-- Upload Box -->
            <div class="lg:col-span-2 bg-slate-800/60 border border-slate-700/70 rounded-2xl p-6 shadow-xl">
                <h2 class="text-base font-semibold text-white mb-2 flex items-center gap-2">
                    <svg class="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
                    Upload Medical Device Label (PDF or Image)
                </h2>
                <p class="text-xs text-slate-400 mb-4">Supports PDF, PNG, JPG, and TIFF. Multi-page PDFs render at 300 DPI with deskewing.</p>

                <div id="dropzone" class="border-2 border-dashed border-slate-600 hover:border-cyan-500 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-900/40">
                    <input type="file" id="fileInput" class="hidden" accept=".png,.jpg,.jpeg,.pdf,.tif">
                    <div class="flex flex-col items-center justify-center space-y-2">
                        <svg class="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
                        <p class="text-sm font-medium text-slate-300">Drag & drop your label file here, or <span class="text-cyan-400 underline">browse</span></p>
                        <p class="text-xs text-slate-500" id="fileNameDisplay">No file selected</p>
                    </div>
                </div>

                <div class="mt-4 flex justify-end">
                    <button id="processBtn" class="px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium text-sm hover:from-cyan-400 hover:to-blue-500 transition shadow-lg shadow-cyan-500/25 flex items-center gap-2 disabled:opacity-50" disabled>
                        <span>Run OCR & Preprocessing</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                    </button>
                </div>
            </div>

            <!-- Quick Presets -->
            <div class="bg-slate-800/60 border border-slate-700/70 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                    <h2 class="text-base font-semibold text-white mb-2 flex items-center gap-2">
                        <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/></svg>
                        Test Case Presets (HLD)
                    </h2>
                    <p class="text-xs text-slate-400 mb-4">Click to test pre-generated synthetic labels:</p>

                    <div class="space-y-2">
                        <button onclick="loadSample('tc01_compliant_label.png')" class="w-full text-left px-3.5 py-2.5 rounded-xl bg-slate-900/60 hover:bg-slate-700/60 border border-slate-700/50 hover:border-cyan-500/60 transition text-xs flex items-center justify-between">
                            <span class="font-medium text-slate-200">TC01: Compliant Label</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">Pass</span>
                        </button>
                        <button onclick="loadSample('tc02_missing_expiry.png')" class="w-full text-left px-3.5 py-2.5 rounded-xl bg-slate-900/60 hover:bg-slate-700/60 border border-slate-700/50 hover:border-amber-500/60 transition text-xs flex items-center justify-between">
                            <span class="font-medium text-slate-200">TC02: Missing Expiry</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800">Defect</span>
                        </button>
                        <button onclick="loadSample('tc03_missing_warning.png')" class="w-full text-left px-3.5 py-2.5 rounded-xl bg-slate-900/60 hover:bg-slate-700/60 border border-slate-700/50 hover:border-red-500/60 transition text-xs flex items-center justify-between">
                            <span class="font-medium text-slate-200">TC03: Missing Warning</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-red-950 text-red-400 border border-red-800">Defect</span>
                        </button>
                        <button onclick="loadSample('tc04_skewed_compliant.png')" class="w-full text-left px-3.5 py-2.5 rounded-xl bg-slate-900/60 hover:bg-slate-700/60 border border-slate-700/50 hover:border-purple-500/60 transition text-xs flex items-center justify-between">
                            <span class="font-medium text-slate-200">TC04: Skewed Label (Deskew)</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-purple-950 text-purple-400 border border-purple-800">Stress</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Processing Status Spinner -->
        <div id="loader" class="hidden text-center py-12">
            <div class="inline-block animate-spin rounded-full h-10 w-10 border-4 border-cyan-500 border-t-transparent mb-3"></div>
            <p class="text-sm text-cyan-300 font-medium">Running Document Preprocessing & OCR Extraction...</p>
        </div>

        <!-- Results Grid -->
        <div id="resultsSection" class="hidden space-y-6">
            
            <!-- Summary Stats Banner -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-4">
                    <p class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Status</p>
                    <p id="resStatus" class="text-lg font-bold text-emerald-400 mt-1">SUCCESS</p>
                </div>
                <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-4">
                    <p class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Latency</p>
                    <p id="resLatency" class="text-lg font-bold text-white mt-1">-- ms</p>
                </div>
                <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-4">
                    <p class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Text Blocks</p>
                    <p id="resBlocks" class="text-lg font-bold text-cyan-400 mt-1">--</p>
                </div>
                <div class="bg-slate-800/80 border border-slate-700 rounded-xl p-4">
                    <p class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Deskew Corrected</p>
                    <p id="resDeskew" class="text-lg font-bold text-purple-400 mt-1">0.0 deg</p>
                </div>
            </div>

            <!-- Main Visual & Extracted Fields Column -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                <!-- Visual Overlay (Annotated Image) -->
                <div class="lg:col-span-7 bg-slate-800/60 border border-slate-700/70 rounded-2xl p-6 shadow-xl">
                    <h3 class="text-sm font-semibold text-slate-200 mb-3 flex items-center justify-between">
                        <span>Annotated Label Visual Preview</span>
                        <a id="downloadVisLink" href="#" target="_blank" class="text-xs text-cyan-400 hover:underline">Full Resolution ↗</a>
                    </h3>
                    <div class="rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center min-h-[350px]">
                        <img id="annotatedImage" src="" alt="Annotated OCR Preview" class="w-full h-auto object-contain max-h-[580px]">
                    </div>
                </div>

                <!-- Structured Medical Entities -->
                <div class="lg:col-span-5 bg-slate-800/60 border border-slate-700/70 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                    <div>
                        <h3 class="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
                            <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                            Structured Regulatory Fields (FDA / EU MDR)
                        </h3>

                        <div id="fieldsContainer" class="space-y-3 text-xs">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <div class="mt-6 pt-4 border-t border-slate-700 flex justify-between items-center">
                        <button onclick="downloadJson()" class="px-4 py-2 rounded-xl bg-slate-700 hover:bg-slate-600 text-xs font-medium text-slate-200 transition flex items-center gap-1.5">
                            <svg class="w-3.5 h-3.5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                            Export JSON for Module 5
                        </button>
                    </div>
                </div>
            </div>

        </div>

    </main>

    <script>
        let currentFile = null;
        let currentResult = null;

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileNameDisplay = document.getElementById('fileNameDisplay');
        const processBtn = document.getElementById('processBtn');

        dropzone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('border-cyan-400');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('border-cyan-400');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-cyan-400');
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        function handleFileSelect(file) {
            currentFile = file;
            fileNameDisplay.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            processBtn.disabled = false;
        }

        processBtn.addEventListener('click', async () => {
            if (!currentFile) return;

            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            processBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const resp = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });
                const result = await resp.json();
                renderResult(result);
            } catch (err) {
                alert('OCR Error: ' + err.message);
            } finally {
                document.getElementById('loader').classList.add('hidden');
                processBtn.disabled = false;
            }
        });

        async function loadSample(sampleName) {
            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            try {
                const resp = await fetch(`/api/process_sample?sample=${sampleName}`);
                const result = await resp.json();
                renderResult(result);
            } catch (err) {
                alert('Sample processing error: ' + err.message);
            } finally {
                document.getElementById('loader').classList.add('hidden');
            }
        }

        function renderResult(result) {
            currentResult = result;
            document.getElementById('resultsSection').classList.remove('hidden');

            document.getElementById('resStatus').innerText = result.success ? 'SUCCESS' : 'FAILED';
            document.getElementById('resLatency').innerText = `${result.processing_time_ms} ms`;
            document.getElementById('resBlocks').innerText = result.total_blocks_detected;
            document.getElementById('resDeskew').innerText = `${result.deskew_angle} deg`;

            if (result.annotated_image_path) {
                const filename = result.annotated_image_path.split(/[\\\\/]/).pop();
                const visUrl = `/outputs/${filename}?t=${new Date().getTime()}`;
                document.getElementById('annotatedImage').src = visUrl;
                document.getElementById('downloadVisLink').href = visUrl;
            }

            // Render structured fields
            const data = result.structured_data;
            const container = document.getElementById('fieldsContainer');
            container.innerHTML = '';

            const fieldDefs = [
                { key: 'Device Name', val: data.device_name?.value, present: data.device_name?.is_present, color: 'text-emerald-400' },
                { key: 'Catalog / REF', val: data.catalog_number?.value, present: data.catalog_number?.is_present, color: 'text-cyan-400' },
                { key: 'LOT / Batch', val: data.lot_number?.value, present: data.lot_number?.is_present, color: 'text-blue-400' },
                { key: 'Serial No (SN)', val: data.serial_number?.value, present: data.serial_number?.is_present, color: 'text-indigo-400' },
                { key: 'Expiry Date', val: data.expiry_date?.value, present: data.expiry_date?.is_present, color: 'text-amber-400' },
                { key: 'MFG Date', val: data.mfg_date?.value, present: data.mfg_date?.is_present, color: 'text-cyan-300' },
                { key: 'Manufacturer', val: data.manufacturer_name?.value, present: data.manufacturer_name?.is_present, color: 'text-orange-400' },
                { key: 'Storage Conditions', val: data.storage_conditions?.value, present: data.storage_conditions?.is_present, color: 'text-purple-400' },
            ];

            fieldDefs.forEach(f => {
                const row = document.createElement('div');
                row.className = 'p-2.5 rounded-xl bg-slate-900/60 border border-slate-700/60 flex items-center justify-between';
                row.innerHTML = `
                    <div class="flex flex-col">
                        <span class="text-slate-400 text-[11px] font-medium uppercase tracking-wider">${f.key}</span>
                        <span class="${f.present ? f.color : 'text-slate-500 italic'} font-semibold mt-0.5">${f.present ? f.val : 'Not Found / Missing'}</span>
                    </div>
                    <span class="text-[10px] font-bold px-2 py-0.5 rounded-full ${f.present ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-red-950 text-red-400 border border-red-800'}">
                        ${f.present ? 'DETECTED' : 'MISSING'}
                    </span>
                `;
                container.appendChild(row);
            });

            // Warnings
            if (data.warnings && data.warnings.length > 0) {
                const wCard = document.createElement('div');
                wCard.className = 'p-2.5 rounded-xl bg-red-950/20 border border-red-800/40 text-red-300';
                wCard.innerHTML = `<span class="text-[11px] uppercase tracking-wider font-bold text-red-400 block mb-1">Safety Warnings (${data.warnings.length})</span>` +
                    data.warnings.map(w => `<p class="text-[11px] mb-1 leading-relaxed">• ${w.value}</p>`).join('');
                container.appendChild(wCard);
            }

            // Member 3: Barcodes & UDI Rendering
            if (result.barcodes && result.barcodes.length > 0) {
                const bcCard = document.createElement('div');
                bcCard.className = 'p-3 rounded-xl bg-blue-950/30 border border-blue-700/50 text-blue-200 mt-3';
                let bcHtml = `<span class="text-[11px] uppercase tracking-wider font-bold text-blue-400 block mb-1">Decoded Barcodes & UDI Carrier (${result.barcodes.length})</span>`;
                result.barcodes.forEach(b => {
                    bcHtml += `<p class="text-[11px] font-mono text-cyan-300">• [${b.type}] ${b.data}</p>`;
                });
                if (result.udi && result.udi.device_identifier_di) {
                    bcHtml += `<div class="mt-2 pt-2 border-t border-blue-800/40 text-[11px]">
                        <span class="text-amber-400 font-semibold">UDI DI (GTIN):</span> ${result.udi.device_identifier_di} 
                        <span class="text-xs px-1.5 py-0.5 rounded ${result.udi.is_di_checksum_valid ? 'bg-emerald-900 text-emerald-300' : 'bg-red-900 text-red-300'}">${result.udi.is_di_checksum_valid ? 'CHECKSUM VALID' : 'CHECKSUM INVALID'}</span>
                    </div>`;
                    if (result.udi.lot_number_pi) bcHtml += `<div class="text-[11px]"><span class="text-slate-400">LOT:</span> ${result.udi.lot_number_pi}</div>`;
                    if (result.udi.expiry_date_pi) bcHtml += `<div class="text-[11px]"><span class="text-slate-400">EXPIRY:</span> ${result.udi.expiry_date_pi}</div>`;
                }
                bcCard.innerHTML = bcHtml;
                container.appendChild(bcCard);
            }

            // Member 3: Medical Symbols Rendering
            if (result.symbols && result.symbols.length > 0) {
                const symCard = document.createElement('div');
                symCard.className = 'p-3 rounded-xl bg-emerald-950/25 border border-emerald-700/40 text-emerald-200 mt-3';
                let symHtml = `<span class="text-[11px] uppercase tracking-wider font-bold text-emerald-400 block mb-1">Detected ISO 15223-1 Symbols (${result.symbols.length})</span>`;
                result.symbols.forEach(s => {
                    symHtml += `<span class="inline-block text-[10px] font-medium px-2 py-1 rounded bg-slate-800 border border-emerald-700/50 mr-1 mb-1 text-emerald-300">${s.display_name} (${(s.confidence*100).toFixed(0)}%)</span>`;
                });
                symCard.innerHTML = symHtml;
                container.appendChild(symCard);
            }
        }

        function downloadJson() {
            if (!currentResult) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentResult, null, 2));
            const dlAnchor = document.createElement('a');
            dlAnchor.setAttribute("href", dataStr);
            dlAnchor.setAttribute("download", `ocr_result_${Date.now()}.json`);
            dlAnchor.click();
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE


@app.post("/api/process")
async def process_upload(file: UploadFile = File(...)):
    contents = await file.read()
    result = pipeline.process(contents, file_name=file.filename)
    return JSONResponse(content=result.to_dict())


@app.get("/api/process_sample")
def process_sample(sample: str):
    sample_path = os.path.join("sample_labels", sample)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample not found")
    result = pipeline.process(sample_path)
    return JSONResponse(content=result.to_dict())


def main():
    parser = argparse.ArgumentParser(description="VeriLabel AI Module 2 Demo")
    parser.add_argument("--server", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Server port (Default: 8501)")
    parser.add_argument("--file", type=str, help="Process a single file and print results")
    args = parser.parse_args()

    if args.file:
        res = pipeline.process(args.file)
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"\n=======================================================")
        print(f"  VeriLabel AI - Module 2: OCR & Document Processing  ")
        print(f"  Launching Interactive Web UI on http://localhost:{args.port}")
        print(f"=======================================================\n")
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
