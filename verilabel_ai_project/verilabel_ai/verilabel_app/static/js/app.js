/**
 * VeriLabel AI — Backend & Database Connected Client Logic
 * Strictly displays values returned by FastAPI endpoints or persisted in SQLite.
 * Never fabricates OCR text, confidence scores, verdicts, or dashboard statistics.
 */

// Global Application State (all analysis values start null until backend returns data)
let activeRunId = null;
let activeAuditData = null;
let currentZoomLevel = 1.0;
let currentVisualMode = "annotated";
let ingestSelectedFile = null;
let selectedSampleId = null;
let selectedSampleFile = null;
let cachedLedgerItems = [];
let cachedRequiredFields = [];
let cachedSamples = [];
let activeFieldSearchQuery = "";
let activeFieldStatusFilter = "all";
let currentAuthenticatedUser = null;

// Chart.js instances
let complianceDonutChart = null;
let auditTrendChart = null;

let isRedirectingToLogin = false;

function handleUnauthorizedRedirect() {
    if (isRedirectingToLogin) return;
    isRedirectingToLogin = true;
    try {
        localStorage.removeItem("verilabel_user");
        sessionStorage.removeItem("verilabel_last_signout");
    } catch (e) {}
    window.location.replace("/login");
}

// Intercept any 401 Unauthorized API response across the dashboard
const _nativeFetch = window.fetch.bind(window);
window.fetch = async function (...args) {
    const response = await _nativeFetch(...args);
    if (response.status === 401) {
        handleUnauthorizedRedirect();
    }
    return response;
};

document.addEventListener("DOMContentLoaded", async function () {
    const validUser = await verifyActiveSessionOrRedirect();
    if (!validUser) return;

    initAuditorSession(validUser);
    handleHashNavigation();
    loadProductBaselines();
    loadSampleCatalog();
    refreshDashboardData();
    loadLedgerData();
    loadReportsData();
    loadPendingReviewQueue();
    loadHumanReviewHistory();
    loadSettingsData();
    loadRequiredFieldsForRun(null);
    handleQuerySelection();
});

// ==============================================================================
// 1. NAVIGATION & URL STATE
// ==============================================================================

function toggleSidebar() {
    const sidebar = document.getElementById("app-sidebar");
    if (sidebar) sidebar.classList.toggle("collapsed");
}

function navigateTo(viewId, event) {
    if (event) event.preventDefault();

    document.querySelectorAll(".page-view").forEach(v => v.classList.remove("active"));
    document.querySelectorAll(".sidebar-nav-item, .tab-nav-link, .nav-item").forEach(n => n.classList.remove("active"));

    const target = document.getElementById(viewId);
    if (target) target.classList.add("active");

    const activeNav = document.querySelectorAll(`[data-view="${viewId}"]`);
    activeNav.forEach(el => el.classList.add("active"));

    const topbarTitle = document.getElementById("topbar-active-page");
    if (topbarTitle && activeNav.length > 0) {
        const text = activeNav[0].querySelector(".nav-label, .nav-item-text")?.textContent || "Console";
        topbarTitle.textContent = text;
    }

    if (viewId === "view-dashboard") {
        setTimeout(() => redrawCharts(), 50);
    }

    const hashSlug = viewId.replace("view-", "");
    if (window.location.hash !== `#${hashSlug}`) {
        history.replaceState(null, "", `#${hashSlug}`);
    }
}

function handleHashNavigation() {
    const hash = window.location.hash.replace("#", "").trim();
    if (hash) {
        const targetId = `view-${hash}`;
        if (document.getElementById(targetId)) {
            navigateTo(targetId);
        }
    }
}

function handleQuerySelection() {
    const params = new URLSearchParams(window.location.search);
    const valId = params.get("validation_id");
    const sampleParam = params.get("sample");

    if (valId && !isNaN(Number(valId))) {
        loadRunIntoWorkspace(Number(valId));
    } else if (sampleParam) {
        navigateTo("view-ingestion");
        // Selection waits until sample catalog loads cleanly
        setTimeout(() => selectSampleFromRepo(sampleParam), 300);
    }
}

// ==============================================================================
// 2. AUDITOR SESSION MANAGEMENT (NO INVENTED USER FALLBACK)
// ==============================================================================

async function verifyActiveSessionOrRedirect() {
    try {
        const res = await _nativeFetch("/api/auth/session", { credentials: "same-origin" });
        if (res.status === 401 || !res.ok) {
            handleUnauthorizedRedirect();
            return null;
        }
        const data = await res.json();
        if (!data || !data.authenticated || !data.user) {
            handleUnauthorizedRedirect();
            return null;
        }
        try {
            localStorage.setItem("verilabel_user", JSON.stringify(data.user));
        } catch (e) {}
        return data.user;
    } catch (e) {
        return null;
    }
}

function initAuditorSession(validatedUser) {
    let user = validatedUser || null;
    if (!user) {
        try {
            const stored = localStorage.getItem("verilabel_user");
            if (stored) user = JSON.parse(stored);
        } catch (e) {
            console.warn("Session parse warning:", e);
        }
    }

    currentAuthenticatedUser = user;

    const nameEl = document.getElementById("topbar-user-name");
    const avatarLetter = document.getElementById("sidebar-avatar-letter");
    const authBtnLabel = document.getElementById("btn-auth-action-label");
    const authBtn = document.getElementById("btn-auth-action");
    const welcomeTitle = document.getElementById("dashboard-welcome-title");
    const officerInput = document.getElementById("center-review-officer");

    if (user && user.name) {
        if (nameEl) nameEl.textContent = user.name;
        if (avatarLetter) avatarLetter.textContent = user.name.charAt(0).toUpperCase();
        if (authBtnLabel) authBtnLabel.textContent = "Sign Out";
        if (authBtn) authBtn.style.color = "var(--color-error)";
        if (welcomeTitle) welcomeTitle.textContent = `Welcome, ${user.name}`;
        if (officerInput) officerInput.value = `${user.name} (${user.role || "Auditor"})`;
    } else {
        handleUnauthorizedRedirect();
    }
}

function handleHeaderAuthAction() {
    logoutAuditor();
}

async function logoutAuditor() {
    try {
        await _nativeFetch("/api/auth/logout", { method: "POST", credentials: "same-origin" });
    } catch (e) {}
    try {
        localStorage.removeItem("verilabel_user");
        sessionStorage.removeItem("verilabel_last_signout");
    } catch (e) {}
    window.location.replace("/login");
}

// ==============================================================================
// 3. TOAST NOTIFICATIONS
// ==============================================================================

function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    let icon = "✓";
    if (type === "error") icon = "✕";
    if (type === "warning") icon = "⚠";
    if (type === "info") icon = "ℹ";

    toast.innerHTML = `<span style="font-size: 1rem; font-weight: 800;">${icon}</span><span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        toast.style.transition = "all 0.25s ease";
        setTimeout(() => toast.remove(), 250);
    }, 3500);
}

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

// ==============================================================================
// 4. OVERVIEW DASHBOARD ANALYTICS (STRICTLY FROM SQLITE)
// ==============================================================================

async function refreshDashboardData() {
    try {
        const res = await fetch("/api/dashboard/stats");
        const data = await res.json();
        if (!res.ok || data.status !== "success") return;

        const m = data.metrics;
        const hasRecords = m.total_scans > 0;

        const setEl = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setEl("kpi-total-scans", m.total_scans);
        setEl("kpi-compliant-scans", m.compliant_count);
        setEl("kpi-non-compliant-scans", m.non_compliant_count);
        setEl("kpi-pending-reviews", m.pending_reviews);
        setEl("kpi-pass-rate", hasRecords ? `${m.pass_rate_percent}%` : "0%");
        setEl("sidebar-pending-count", m.pending_reviews);
        setEl("sidebar-review-badge", m.pending_reviews);

        const emptyBanner = document.getElementById("dashboard-empty-banner");
        if (emptyBanner) {
            emptyBanner.style.display = hasRecords ? "none" : "block";
        }

        // Recent Table
        const tbody = document.getElementById("dashboard-recent-table");
        if (tbody) {
            tbody.innerHTML = "";
            if (!hasRecords || data.recent_analyses.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--color-text-muted); padding: 1.25rem;">No validations saved yet.</td></tr>`;
            } else {
                data.recent_analyses.slice(0, 6).forEach(item => {
                    const tr = document.createElement("tr");
                    let badgeClass = item.verdict === "COMPLIANT" ? "badge-compliant" : (item.verdict === "CONDITIONAL_PASS" ? "badge-warning" : "badge-non-compliant");
                    tr.innerHTML = `
                        <td class="mono" style="font-weight: 700; color: var(--color-navy);">#${item.validation_id}</td>
                        <td style="font-weight: 600;">${escapeHtml(item.file_name)}</td>
                        <td><span class="badge ${badgeClass}">${escapeHtml(item.verdict)}</span></td>
                        <td style="font-weight: 700;">${item.compliance_score}%</td>
                        <td><span class="badge ${item.latest_review_status === 'APPROVED' ? 'badge-compliant' : (item.latest_review_status === 'REJECTED' ? 'badge-non-compliant' : 'badge-neutral')}">${escapeHtml(item.latest_review_status)}</span></td>
                        <td><button type="button" class="btn btn-secondary btn-sm" onclick="loadRunIntoWorkspace(${item.validation_id})">Inspect</button></td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        }

        // Activity Feed
        const feed = document.getElementById("dashboard-activity-feed");
        if (feed) {
            feed.innerHTML = "";
            if (!hasRecords || data.recent_activity.length === 0) {
                feed.innerHTML = `<div style="color: var(--color-text-muted); font-size: 0.82rem; text-align: center; padding: 1.5rem 0;">No validation activity recorded in database yet.</div>`;
            } else {
                data.recent_activity.forEach(act => {
                    const item = document.createElement("div");
                    item.style.padding = "0.75rem";
                    item.style.background = "var(--color-surface-sub)";
                    item.style.borderRadius = "var(--radius-sm)";
                    item.style.borderLeft = `3px solid ${act.status === 'COMPLIANT' ? 'var(--color-success)' : (act.status === 'CONDITIONAL_PASS' ? 'var(--color-warning)' : 'var(--color-error)')}`;
                    item.innerHTML = `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                            <strong style="font-size: 0.82rem; color: var(--color-navy);">${escapeHtml(act.title)}</strong>
                            <span style="font-size: 0.72rem; color: var(--color-text-muted);">${escapeHtml(act.timestamp)}</span>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--color-text-body);">${escapeHtml(act.description)}</div>
                    `;
                    feed.appendChild(item);
                });
            }
        }

        initCharts(hasRecords, data.compliance_breakdown, data.monthly_trend);
    } catch (err) {
        console.error("Dashboard stats fetch error:", err);
        showToast("Error loading statistics from backend database.", "error");
    }
}

function initCharts(hasRecords, breakdown, trend) {
    const donutCtx = document.getElementById("chart-compliance-donut");
    const donutEmpty = document.getElementById("chart-donut-empty");
    const trendCtx = document.getElementById("chart-audit-trend");
    const trendEmpty = document.getElementById("chart-trend-empty");

    if (complianceDonutChart) {
        complianceDonutChart.destroy();
        complianceDonutChart = null;
    }
    if (auditTrendChart) {
        auditTrendChart.destroy();
        auditTrendChart = null;
    }

    if (!hasRecords) {
        if (donutEmpty) donutEmpty.style.display = "flex";
        if (trendEmpty) trendEmpty.style.display = "flex";
        return;
    }

    if (donutEmpty) donutEmpty.style.display = "none";
    if (trendEmpty) trendEmpty.style.display = "none";

    if (donutCtx && typeof Chart !== "undefined") {
        complianceDonutChart = new Chart(donutCtx, {
            type: "doughnut",
            data: {
                labels: breakdown.map(b => b.name),
                datasets: [{
                    data: breakdown.map(b => b.value),
                    backgroundColor: ["#168B69", "#E5A443", "#DC5656"],
                    borderWidth: 2,
                    borderColor: "#FFFFFF"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
                cutout: "65%"
            }
        });
    }

    if (trendCtx && typeof Chart !== "undefined" && trend && trend.length > 0) {
        auditTrendChart = new Chart(trendCtx, {
            type: "bar",
            data: {
                labels: trend.map(t => t.period),
                datasets: [
                    { label: "Compliant", data: trend.map(t => t.compliant), backgroundColor: "#168B69", borderRadius: 4 },
                    { label: "Conditional", data: trend.map(t => t.conditional || 0), backgroundColor: "#E5A443", borderRadius: 4 },
                    { label: "Non-Compliant", data: trend.map(t => t.non_compliant), backgroundColor: "#DC5656", borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { grid: { display: false } }, y: { beginAtZero: true, ticks: { stepSize: 1 }, grid: { color: "#DFE7F0" } } },
                plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } }
            }
        });
    }
}

function redrawCharts() {
    if (complianceDonutChart) complianceDonutChart.resize();
    if (auditTrendChart) auditTrendChart.resize();
}

// ==============================================================================
// 5. LABEL INGESTION (LOAD REAL PRODUCTS & SAMPLES; REQUIRE EXPLICIT ANALYZE)
// ==============================================================================

async function loadProductBaselines() {
    const select = document.getElementById("ingest-product-code");
    if (!select) return;
    try {
        const res = await fetch("/api/products");
        const data = await res.json();
        if (res.ok && data.status === "success" && Array.isArray(data.products)) {
            select.innerHTML = `<option value="">None (General Regulatory Check)</option>`;
            data.products.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.product_code;
                opt.textContent = `${p.product_code} — ${p.trade_name}`;
                select.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn("Product baseline load error:", e);
    }
}

async function loadSampleCatalog() {
    const container = document.getElementById("sample-catalog-buttons");
    if (!container) return;
    try {
        const res = await fetch("/api/samples");
        const data = await res.json();
        if (!res.ok || data.status !== "success") {
            container.innerHTML = `<span style="font-size: 0.8rem; color: var(--color-error);">Failed to load sample files from backend.</span>`;
            return;
        }
        cachedSamples = data.samples || [];
        if (cachedSamples.length === 0) {
            container.innerHTML = `<span style="font-size: 0.8rem; color: var(--color-text-muted);">No sample label files found on server.</span>`;
            return;
        }
        container.innerHTML = "";
        cachedSamples.forEach(s => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "tc-pill-button";
            btn.id = `btn-case-${s.id}`;
            btn.onclick = () => selectSampleFromRepo(s.id);
            btn.innerHTML = `<span>${escapeHtml(s.id)}: ${escapeHtml(s.file)}</span>`;
            container.appendChild(btn);
        });
    } catch (e) {
        container.innerHTML = `<span style="font-size: 0.8rem; color: var(--color-error);">Error loading sample list.</span>`;
    }
}

function updateAnalyzeButtonState() {
    const btn = document.getElementById("btn-execute-audit");
    const label = document.getElementById("ingest-selected-label-name");
    const hasSelection = Boolean(ingestSelectedFile || selectedSampleId);

    if (btn) btn.disabled = !hasSelection;
    if (label) {
        if (ingestSelectedFile) {
            label.style.color = "var(--color-blue)";
            label.textContent = `${ingestSelectedFile.name} (Uploaded file — ready to Analyze)`;
        } else if (selectedSampleId) {
            label.style.color = "var(--color-blue)";
            label.textContent = `${selectedSampleId}: ${selectedSampleFile || ""} (Repository sample — ready to Analyze)`;
        } else {
            label.style.color = "var(--color-text-muted)";
            label.textContent = "No label selected yet";
        }
    }
}

function selectSampleFromRepo(sampleId) {
    const match = cachedSamples.find(s => s.id === sampleId);
    if (!match) return;

    selectedSampleId = match.id;
    selectedSampleFile = match.file;
    ingestSelectedFile = null;

    const previewCard = document.getElementById("ingest-preview-card");
    if (previewCard) previewCard.style.display = "none";

    document.querySelectorAll(".tc-pill-button").forEach(b => b.classList.remove("active"));
    const activeBtn = document.getElementById(`btn-case-${sampleId}`);
    if (activeBtn) activeBtn.classList.add("active");

    const dropText = document.getElementById("ingest-drop-text");
    if (dropText) dropText.innerHTML = `Selected repository file <strong>${escapeHtml(match.file)}</strong>. Click <strong>Analyze</strong> below to run verification.`;

    updateAnalyzeButtonState();
}

function handleIngestFileSelected(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (file.size > 25 * 1024 * 1024) {
        showToast("File exceeds maximum allowable size of 25 MB.", "error");
        return;
    }

    ingestSelectedFile = file;
    selectedSampleId = null;
    selectedSampleFile = null;

    document.querySelectorAll(".tc-pill-button").forEach(btn => btn.classList.remove("active"));

    const previewCard = document.getElementById("ingest-preview-card");
    if (previewCard) previewCard.style.display = "flex";

    const fileName = document.getElementById("ingest-file-name");
    if (fileName) fileName.textContent = file.name;

    const fileDetails = document.getElementById("ingest-file-details");
    if (fileDetails) fileDetails.textContent = `Size: ${(file.size / 1024).toFixed(1)} KB • Type: ${file.type || "Document"}`;

    updateAnalyzeButtonState();
}

function clearSelectedIngestFile() {
    ingestSelectedFile = null;
    selectedSampleId = null;
    selectedSampleFile = null;

    const fileInput = document.getElementById("ingest-file-input");
    if (fileInput) fileInput.value = "";

    const previewCard = document.getElementById("ingest-preview-card");
    if (previewCard) previewCard.style.display = "none";

    document.querySelectorAll(".tc-pill-button").forEach(btn => btn.classList.remove("active"));

    const dropText = document.getElementById("ingest-drop-text");
    if (dropText) dropText.textContent = "Click to upload or drag & drop packaging label document";

    updateAnalyzeButtonState();
}

const ingestDropzone = document.getElementById("ingest-dropzone");
if (ingestDropzone) {
    ingestDropzone.addEventListener("dragover", e => {
        e.preventDefault();
        ingestDropzone.classList.add("drag-over");
    });
    ingestDropzone.addEventListener("dragleave", () => ingestDropzone.classList.remove("drag-over"));
    ingestDropzone.addEventListener("drop", e => {
        e.preventDefault();
        ingestDropzone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleIngestFileSelected({ target: { files: e.dataTransfer.files } });
        }
    });
}

// ==============================================================================
// 6. ANALYSIS EXECUTION (ONLY ON EXPLICIT CLICK OF "Analyze")
// ==============================================================================

async function executeAuditWorkflow() {
    if (!ingestSelectedFile && !selectedSampleId) {
        showToast("Please upload a label file or select a sample file first.", "warning");
        return;
    }

    const drawer = document.getElementById("ingestion-progress-drawer");
    const btn = document.getElementById("btn-execute-audit");
    const progressFill = document.getElementById("audit-progress-fill");
    const stageTitle = document.getElementById("runner-stage-title");
    const stageDesc = document.getElementById("runner-stage-desc");

    if (drawer) drawer.style.display = "block";
    if (btn) btn.disabled = true;
    if (progressFill) progressFill.style.width = "40%";
    if (stageTitle) stageTitle.textContent = "Running VeriLabel Backend Verification Pipeline...";
    if (stageDesc) stageDesc.textContent = "Submitting label to POST /api/verify and awaiting backend output...";

    const formData = new FormData();
    const productCode = document.getElementById("ingest-product-code")?.value || "";
    if (productCode) formData.append("product_code", productCode);

    if (ingestSelectedFile) {
        formData.append("file", ingestSelectedFile);
    } else if (selectedSampleId) {
        formData.append("sample_id", selectedSampleId);
    }

    try {
        const res = await fetch("/api/verify", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (progressFill) progressFill.style.width = "100%";

        if (res.ok && data.status === "success") {
            if (drawer) drawer.style.display = "none";
            activeRunId = data.validation_id;
            activeAuditData = data;

            populateAllViews(data);
            showToast(`Analysis saved as Run #${data.validation_id} (${data.verdict}, ${data.compliance_score}%).`, "success");

            refreshDashboardData();
            loadLedgerData();
            loadReportsData();
            loadPendingReviewQueue();
            navigateTo("view-compliance");
        } else {
            if (drawer) drawer.style.display = "none";
            showToast(`Analysis failed: ${data.detail || "Backend error"}`, "error");
        }
    } catch (err) {
        if (drawer) drawer.style.display = "none";
        console.error("Verification error:", err);
        showToast(`Failed to reach backend verification endpoint: ${err.message}`, "error");
    } finally {
        updateAnalyzeButtonState();
    }
}

// ==============================================================================
// 7. POPULATE ALL VIEWS STRICTLY FROM ACTUAL BACKEND/DB DATA
// ==============================================================================

function formatFieldOrMissing(val) {
    if (val === null || val === undefined || String(val).trim() === "") {
        return `<span style="color: var(--color-text-muted); font-style: italic;">Not detected</span>`;
    }
    return escapeHtml(String(val));
}

function populateAllViews(data) {
    const f = data.fields || {};
    const verdict = data.verdict;
    const score = data.compliance_score;

    // 1. OCR Extraction View
    const ocrEmpty = document.getElementById("ocr-empty-state");
    const ocrBody = document.getElementById("ocr-results-body");
    if (ocrEmpty) ocrEmpty.style.display = "none";
    if (ocrBody) ocrBody.style.display = "block";

    const ocrImg = document.getElementById("ocr-canvas-img");
    const ocrNoImg = document.getElementById("ocr-no-img-msg");
    if (ocrImg) {
        if (data.raw_image || data.annotated_image) {
            ocrImg.src = data.raw_image || data.annotated_image;
            ocrImg.style.display = "block";
            if (ocrNoImg) ocrNoImg.style.display = "none";
        } else {
            ocrImg.style.display = "none";
            if (ocrNoImg) ocrNoImg.style.display = "block";
        }
    }

    const rawOcr = data.raw_ocr || null;
    const ocrBadge = document.getElementById("ocr-engine-badge");
    const rawText = rawOcr ? rawOcr.raw_text : (f.raw_text || "");
    const ocrDiag = document.getElementById("ocr-diagnostic-alert");

    if (ocrBadge) {
        if (rawOcr && rawOcr.engine_name) {
            const confStr = rawOcr.average_confidence > 0 ? ` (${(rawOcr.average_confidence * 100).toFixed(1)}% avg conf)` : "";
            ocrBadge.textContent = `${rawOcr.engine_name}${confStr}`;
        } else {
            ocrBadge.textContent = "Stored Extraction Record";
        }
    }

    if (ocrDiag) {
        if (!rawText || !rawText.trim()) {
            ocrDiag.style.display = "block";
            ocrDiag.innerHTML = `<strong>OCR Notice:</strong> The backend OCR engine (${escapeHtml(rawOcr?.engine_name || "FallbackOCR")}) returned no text for this document (no local OCR model such as PaddleOCR/Tesseract was active, or no text was detected). Text fields therefore report <em>Not detected</em>.`;
        } else {
            ocrDiag.style.display = "none";
        }
    }

    const rawTextBox = document.getElementById("ocr-raw-text-box");
    if (rawTextBox) {
        rawTextBox.textContent = rawText && rawText.trim() ? rawText : "Not detected (no raw OCR text returned by backend).";
    }

    const setHtml = (id, html) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    };

    setHtml("ocr-field-product", formatFieldOrMissing(f.product_name));
    setHtml("ocr-field-manufacturer", formatFieldOrMissing(f.manufacturer_name ? (f.manufacturer_address ? `${f.manufacturer_name} (${f.manufacturer_address})` : f.manufacturer_name) : null));
    setHtml("ocr-field-lot", formatFieldOrMissing(f.lot_number));
    setHtml("ocr-field-expiry", formatFieldOrMissing(f.expiry_date));
    setHtml("ocr-field-mfg", formatFieldOrMissing(f.manufacturing_date));
    setHtml("ocr-field-catalog", formatFieldOrMissing(f.catalog_number));
    setHtml("ocr-field-serial", formatFieldOrMissing(f.serial_number));
    setHtml("ocr-field-sterile", f.is_sterile ? `<span class="badge badge-compliant">${escapeHtml(f.sterilization_method || "STERILE")}</span>` : `<span style="color: var(--color-text-muted); font-style: italic;">Not detected</span>`);
    setHtml("ocr-field-single-use", f.is_single_use ? "Do Not Re-use / Single Use" : `<span style="color: var(--color-text-muted); font-style: italic;">Not detected</span>`);
    setHtml("ocr-field-storage", formatFieldOrMissing(f.storage_conditions));
    setHtml("ocr-field-ce", formatFieldOrMissing(f.ce_mark_number));
    setHtml("ocr-field-rx", f.is_rx_only ? "Rx Only" : `<span style="color: var(--color-text-muted); font-style: italic;">Not detected</span>`);
    setHtml("ocr-field-ifu", formatFieldOrMissing(f.instructions_for_use));
    setHtml("ocr-field-warnings", (f.warnings && f.warnings.length > 0) ? escapeHtml(f.warnings.join(" | ")) : `<span style="color: var(--color-text-muted); font-style: italic;">Not detected</span>`);

    // 2. Visual Inspection View
    const visEmpty = document.getElementById("visual-empty-state");
    const visBody = document.getElementById("visual-results-body");
    if (visEmpty) visEmpty.style.display = "none";
    if (visBody) visBody.style.display = "grid";

    const visualImg = document.getElementById("visual-inspection-img");
    const visNoImg = document.getElementById("visual-no-img-msg");
    if (visualImg) {
        const chosenImg = currentVisualMode === "raw" ? (data.raw_image || data.annotated_image) : (data.annotated_image || data.raw_image);
        if (chosenImg) {
            visualImg.src = chosenImg;
            visualImg.style.display = "block";
            if (visNoImg) visNoImg.style.display = "none";
        } else {
            visualImg.style.display = "none";
            if (visNoImg) visNoImg.style.display = "block";
        }
    }

    const symbols = data.symbols || [];
    const symCount = document.getElementById("visual-symbols-detected-count");
    if (symCount) symCount.textContent = `${symbols.length} Detected Symbol${symbols.length === 1 ? "" : "s"}`;

    const symbolsContainer = document.getElementById("visual-symbol-cards-container");
    if (symbolsContainer) {
        symbolsContainer.innerHTML = "";
        if (symbols.length > 0) {
            symbols.forEach(sym => {
                const card = document.createElement("div");
                card.style.background = "var(--color-surface-sub)";
                card.style.border = "1px solid var(--color-border)";
                card.style.borderRadius = "var(--radius-sm)";
                card.style.padding = "0.75rem";
                card.style.cursor = "pointer";
                card.onclick = () => showSymbolDetailModal(sym);

                const bboxText = sym.bbox ? `[x:${sym.bbox.x}, y:${sym.bbox.y}, w:${sym.bbox.w}, h:${sym.bbox.h}]` : "Not reported";
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: var(--color-navy); font-size: 0.85rem;">${escapeHtml(sym.display_name || sym.symbol_id)}</strong>
                            <div class="mono" style="font-size: 0.72rem; color: var(--color-text-muted);">ISO Code: ${escapeHtml(sym.iso_code || "N/A")}</div>
                        </div>
                        <span class="badge badge-info">${(Number(sym.confidence || 0) * 100).toFixed(1)}% Score</span>
                    </div>
                    <div style="font-size: 0.7rem; color: var(--color-text-muted); margin-top: 0.35rem; font-family: var(--font-mono);">BBox: ${bboxText}</div>
                `;
                symbolsContainer.appendChild(card);
            });
        } else {
            symbolsContainer.innerHTML = `<div style="color: var(--color-text-muted); padding: 1.5rem; text-align: center;">Not detected — No ISO 15223-1 symbols met detection threshold on this image.</div>`;
        }
    }

    // 3. Barcode & UDI View
    const barEmpty = document.getElementById("barcode-empty-state");
    const barBody = document.getElementById("barcode-results-body");
    if (barEmpty) barEmpty.style.display = "none";
    if (barBody) barBody.style.display = "grid";

    const udi = data.udi;
    const barcodes = data.barcodes || [];

    setHtml("barcode-type-badge", escapeHtml(udi?.carrier_type || udi?.standard || (barcodes.length > 0 ? barcodes[0].type : "Not detected")));
    setHtml("barcode-raw-payload", formatFieldOrMissing(udi?.raw_barcode || udi?.raw_carrier || (barcodes[0]?.data || null)));
    setHtml("barcode-di-val", formatFieldOrMissing(udi?.device_identifier_di));
    setHtml("barcode-pi-lot", formatFieldOrMissing(udi?.lot_number_pi));
    setHtml("barcode-pi-expiry", formatFieldOrMissing(udi?.expiry_date_pi));

    const chkBadge = document.getElementById("barcode-checksum-badge");
    if (chkBadge) {
        if (!udi || !udi.device_identifier_di) {
            chkBadge.innerHTML = `<span class="badge badge-neutral">Not detected</span>`;
        } else if (udi.is_di_checksum_valid) {
            chkBadge.innerHTML = `<span class="badge badge-compliant">VALID GS1 MODULO-10</span>`;
        } else {
            chkBadge.innerHTML = `<span class="badge badge-non-compliant">INVALID CHECKSUM</span>`;
        }
    }

    const scannedListEl = document.getElementById("barcode-scanned-list");
    if (scannedListEl) {
        scannedListEl.textContent = barcodes.length > 0 ? barcodes.map(b => `${b.type}: ${b.data}`).join(" | ") : "Not detected";
    }

    // UDI Cross-Reconciliation strictly from real backend checks & UDI validation_errors
    const recContainer = document.getElementById("barcode-reconciliation-list");
    if (recContainer) {
        recContainer.innerHTML = "";
        const udiChecks = (data.checks || []).filter(c => c.rule_id && c.rule_id.startsWith("UDI-"));
        if (udiChecks.length === 0) {
            recContainer.innerHTML = `<div style="color: var(--color-text-muted); font-size: 0.82rem;">Not evaluated — No UDI checks present in report.</div>`;
        } else {
            udiChecks.forEach(c => {
                const box = document.createElement("div");
                box.style.padding = "0.85rem";
                box.style.borderRadius = "var(--radius-sm)";
                box.style.background = c.passed ? "var(--color-success-soft)" : "var(--color-error-soft)";
                box.style.border = `1px solid ${c.passed ? "var(--color-success-bd)" : "var(--color-error-bd)"}`;
                box.innerHTML = `
                    <div style="font-weight: 700; font-size: 0.85rem; color: ${c.passed ? "var(--color-success)" : "var(--color-error)"}; margin-bottom: 0.2rem;">
                        ${escapeHtml(c.rule_id)}: ${escapeHtml(c.rule_name)} (${c.passed ? "PASSED" : "FAILED"})
                    </div>
                    <div style="font-size: 0.78rem; color: var(--color-text-body);">${escapeHtml(c.message)}</div>
                    <div class="mono" style="font-size: 0.72rem; color: var(--color-text-muted); margin-top: 0.2rem;">Observed: ${escapeHtml(c.observed_value || "Not detected")}</div>
                `;
                recContainer.appendChild(box);
            });
        }
    }

    // 4. Compliance Results View
    const compEmpty = document.getElementById("compliance-empty-state");
    const compBody = document.getElementById("compliance-results-body");
    if (compEmpty) compEmpty.style.display = "none";
    if (compBody) compBody.style.display = "block";

    const ribbon = document.getElementById("compliance-verdict-ribbon");
    const ribBadge = document.getElementById("compliance-ribbon-badge");
    const ribTitle = document.getElementById("compliance-ribbon-title");
    const ribMeta = document.getElementById("compliance-ribbon-meta");
    const ribScore = document.getElementById("compliance-ribbon-score");

    if (ribScore) ribScore.textContent = `${score}%`;
    if (ribMeta) {
        const procMs = data.processing_time_ms ? `Processing time: ${data.processing_time_ms} ms • ` : "";
        ribMeta.textContent = `Run #${data.validation_id} • File: ${data.file_name} • ${procMs}${(data.checks || []).length} rules evaluated`;
    }

    if (ribbon && ribBadge && ribTitle && ribScore) {
        if (verdict === "COMPLIANT") {
            ribbon.style.background = "var(--color-success-soft)";
            ribbon.style.borderColor = "var(--color-success-bd)";
            ribBadge.className = "badge badge-compliant";
            ribBadge.textContent = "COMPLIANT";
            ribTitle.textContent = "Compliance Verdict: Compliant";
            ribTitle.style.color = "var(--color-success)";
            ribScore.style.color = "var(--color-success)";
        } else if (verdict === "CONDITIONAL_PASS") {
            ribbon.style.background = "var(--color-warning-soft)";
            ribbon.style.borderColor = "var(--color-warning-bd)";
            ribBadge.className = "badge badge-warning";
            ribBadge.textContent = "CONDITIONAL_PASS";
            ribTitle.textContent = "Compliance Verdict: Conditional Pass";
            ribTitle.style.color = "var(--color-warning)";
            ribScore.style.color = "var(--color-warning)";
        } else {
            ribbon.style.background = "var(--color-error-soft)";
            ribbon.style.borderColor = "var(--color-error-bd)";
            ribBadge.className = "badge badge-non-compliant";
            ribBadge.textContent = "NON_COMPLIANT";
            ribTitle.textContent = "Compliance Verdict: Non-Compliant";
            ribTitle.style.color = "var(--color-error)";
            ribScore.style.color = "var(--color-error)";
        }
    }

    const rulesHeading = document.getElementById("compliance-rules-heading");
    if (rulesHeading) {
        rulesHeading.textContent = `Evaluated Compliance Rules (${(data.checks || []).length} checked)`;
    }
    renderRulesTable(data.checks || []);

    // 5. Required Fields View Sync
    loadRequiredFieldsForRun(data.validation_id, data);

    // 6. Human Review Center Active Run Setup (NEVER prefill decision or notes!)
    const revBadge = document.getElementById("review-run-badge");
    const noRunAlert = document.getElementById("review-no-run-alert");
    const submitBtn = document.getElementById("btn-center-submit-review");
    if (revBadge) revBadge.textContent = `Run #${data.validation_id} (${verdict})`;
    if (noRunAlert) noRunAlert.style.display = "none";
    if (submitBtn) submitBtn.disabled = false;
}

function renderRulesTable(checks) {
    const tbody = document.getElementById("compliance-rules-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!checks || checks.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--color-text-muted); padding: 1.25rem;">No compliance check records returned.</td></tr>`;
        return;
    }

    checks.forEach(c => {
        const tr = document.createElement("tr");
        const statusBadge = c.passed
            ? `<span class="badge badge-compliant">PASS</span>`
            : (c.severity === "CRITICAL" ? `<span class="badge badge-non-compliant">FAIL</span>` : `<span class="badge badge-warning">WARN</span>`);

        const sevColor = c.severity === "CRITICAL" ? "var(--color-error)" : (c.severity === "MAJOR" ? "var(--color-warning)" : "var(--color-text-muted)");
        const sevBadge = `<span style="color: ${sevColor}; font-weight: 800; font-size: 0.72rem;">${escapeHtml(c.severity)}</span>`;

        tr.innerHTML = `
            <td class="mono" style="font-weight: 700;">${escapeHtml(c.rule_id)}</td>
            <td>
                <div style="font-weight: 700; color: var(--color-navy);">${escapeHtml(c.rule_name)}</div>
                <div style="font-size: 0.74rem; color: var(--color-text-muted);">${escapeHtml(c.standard || "")} &bull; ${escapeHtml(c.expected_condition)}</div>
            </td>
            <td>${sevBadge}</td>
            <td>${statusBadge}</td>
            <td>
                <div style="font-size: 0.82rem; color: var(--color-text-main);">${escapeHtml(c.observed_value || "Not detected")}</div>
                <div style="font-size: 0.74rem; color: var(--color-text-muted); margin-top: 0.15rem;">${escapeHtml(c.message)}</div>
                ${!c.passed && c.remediation_hint ? `<div style="font-size: 0.72rem; color: var(--color-warning); margin-top: 0.2rem;">Remediation: ${escapeHtml(c.remediation_hint)}</div>` : ""}
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function filterRules(severity) {
    if (!activeAuditData || !activeAuditData.checks) return;
    if (severity === "ALL") {
        renderRulesTable(activeAuditData.checks);
    } else {
        renderRulesTable(activeAuditData.checks.filter(c => c.severity === severity));
    }
}

// ==============================================================================
// 8. VISUAL CONTROLS & SYMBOL MODAL
// ==============================================================================

function setVisualMode(mode) {
    currentVisualMode = mode;
    const img = document.getElementById("visual-inspection-img");
    if (!img || !activeAuditData) return;

    if (mode === "raw") {
        if (activeAuditData.raw_image) {
            img.src = activeAuditData.raw_image;
            showToast("Showing raw unannotated image.", "info");
        } else {
            showToast("Raw image not available for this record.", "warning");
        }
    } else {
        if (activeAuditData.annotated_image) {
            img.src = activeAuditData.annotated_image;
            showToast("Showing annotated detection overlays.", "info");
        } else {
            showToast("Annotated image not available for this record.", "warning");
        }
    }
}

function zoomImage(factor) {
    currentZoomLevel = Math.max(0.5, Math.min(3.0, currentZoomLevel * factor));
    const img = document.getElementById("visual-inspection-img");
    if (img) img.style.transform = `scale(${currentZoomLevel})`;
}

function resetZoom() {
    currentZoomLevel = 1.0;
    const img = document.getElementById("visual-inspection-img");
    if (img) img.style.transform = "scale(1.0)";
}

function showSymbolDetailModal(sym) {
    const bboxStr = sym.bbox ? `[x:${sym.bbox.x}, y:${sym.bbox.y}, w:${sym.bbox.w}, h:${sym.bbox.h}]` : "Not reported";
    openDetailModalWithContent(`
        <h4 style="color: var(--color-navy); margin-bottom: 0.5rem; font-size: 1.1rem; font-weight: 700;">${escapeHtml(sym.display_name || sym.symbol_id)}</h4>
        <div style="margin-bottom: 1rem;"><span class="badge badge-info">ISO Code: ${escapeHtml(sym.iso_code || "N/A")}</span></div>
        <div class="field-inspection-row"><span class="field-label-text">Match Score:</span><span class="field-value-text" style="font-weight: 700;">${(Number(sym.confidence || 0) * 100).toFixed(2)}%</span></div>
        <div class="field-inspection-row"><span class="field-label-text">Bounding Coordinates:</span><span class="field-value-text mono">${bboxStr}</span></div>
    `);
}

// ==============================================================================
// 9. HUMAN REVIEW CENTER (CONFIRMED PERSISTENCE TO /api/review)
// ==============================================================================

async function loadPendingReviewQueue() {
    try {
        const res = await fetch("/api/reviews/pending");
        const data = await res.json();
        if (!res.ok || data.status !== "success") return;

        const countBadge = document.getElementById("center-pending-queue-badge");
        if (countBadge) countBadge.textContent = `${data.count} Pending`;

        const container = document.getElementById("center-pending-list");
        if (!container) return;
        container.innerHTML = "";

        if (data.pending.length === 0) {
            container.innerHTML = `<div style="color: var(--color-text-muted); font-size: 0.82rem; padding: 0.75rem;">No runs currently awaiting review.</div>`;
            return;
        }

        data.pending.slice(0, 15).forEach(run => {
            const card = document.createElement("div");
            card.style.background = "var(--color-surface-sub)";
            card.style.border = "1px solid var(--color-border)";
            card.style.borderRadius = "var(--radius-sm)";
            card.style.padding = "0.75rem";
            card.style.cursor = "pointer";
            card.onclick = () => loadRunIntoWorkspace(run.validation_id);

            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: var(--color-navy); font-size: 0.85rem;">Run #${run.validation_id} — ${escapeHtml(run.file_name)}</strong>
                    <span class="badge ${run.verdict === 'COMPLIANT' ? 'badge-compliant' : (run.verdict === 'CONDITIONAL_PASS' ? 'badge-warning' : 'badge-non-compliant')}">${escapeHtml(run.verdict)}</span>
                </div>
                <div style="font-size: 0.75rem; color: var(--color-text-muted); margin-top: 0.25rem;">
                    Score: ${run.compliance_score}% • Click to inspect &amp; select for review
                </div>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.warn("Pending review load error:", e);
    }
}

async function loadHumanReviewHistory() {
    const container = document.getElementById("center-history-list");
    if (!container) return;
    try {
        const res = await fetch("/api/reviews/history");
        const data = await res.json();
        if (!res.ok || data.status !== "success") return;

        container.innerHTML = "";
        if (!data.reviews || data.reviews.length === 0) {
            container.innerHTML = `<div style="color: var(--color-text-muted); font-size: 0.82rem; padding: 0.5rem;">No human review decisions have been recorded in the database yet.</div>`;
            return;
        }

        data.reviews.forEach(rev => {
            const row = document.createElement("div");
            row.style.background = "var(--color-surface-sub)";
            row.style.border = "1px solid var(--color-border)";
            row.style.borderRadius = "var(--radius-xs)";
            row.style.padding = "0.65rem";
            row.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: var(--color-navy);">Run #${rev.validation_id} (${escapeHtml(rev.file_name || "")})</strong>
                    <span class="badge ${rev.decision === 'APPROVED' ? 'badge-compliant' : 'badge-non-compliant'}">${escapeHtml(rev.decision)}</span>
                </div>
                <div style="font-size: 0.75rem; color: var(--color-text-body); margin-top: 0.2rem;">
                    By <strong>${escapeHtml(rev.reviewer_name)}</strong> (${escapeHtml(rev.reviewer_role)}) &bull; ${escapeHtml(rev.reviewed_at || "")}
                </div>
                ${rev.review_notes ? `<div style="font-size: 0.75rem; color: var(--color-text-muted); margin-top: 0.15rem;">"${escapeHtml(rev.review_notes)}"</div>` : ""}
            `;
            container.appendChild(row);
        });
    } catch (e) {
        container.innerHTML = `<div style="color: var(--color-error); font-size: 0.8rem;">Error loading review history.</div>`;
    }
}

async function submitCenterHumanReview(e) {
    e.preventDefault();
    if (!activeRunId) {
        showToast("No validation run selected. Analyze a label or select a saved run first.", "warning");
        return;
    }
    if (!currentAuthenticatedUser || !currentAuthenticatedUser.name) {
        showToast("Please sign in as an auditor before submitting a legally binding review.", "error");
        return;
    }

    const decision = document.getElementById("center-review-decision").value;
    const notes = document.getElementById("center-review-notes").value.trim();
    const sigChecked = document.getElementById("center-review-sig").checked;

    if (!decision) {
        showToast("Please select a regulatory determination.", "warning");
        return;
    }
    if (!notes) {
        showToast("Please enter review justification notes.", "warning");
        return;
    }
    if (!sigChecked) {
        showToast("Electronic signature checkbox must be checked.", "warning");
        return;
    }

    // Ask the user before mutating stored database state
    const confirmed = window.confirm(
        `Confirm Human Review Submission:\n\nValidation Run: #${activeRunId}\nDecision: ${decision}\nReviewer: ${currentAuthenticatedUser.name} (${currentAuthenticatedUser.role || "Auditor"})\n\nThis will permanently store a 21 CFR Part 11 review record in the database. Continue?`
    );
    if (!confirmed) return;

    const btn = document.getElementById("btn-center-submit-review");
    btn.disabled = true;

    try {
        const res = await fetch("/api/review", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                validation_id: activeRunId,
                reviewer_name: currentAuthenticatedUser.name,
                reviewer_role: currentAuthenticatedUser.role || "Regulatory Auditor",
                decision: decision,
                review_notes: notes
            })
        });

        const data = await res.json();
        if (res.ok && data.status === "success") {
            showToast(data.message || `Review saved for Run #${activeRunId}.`, "success");
            document.getElementById("center-review-notes").value = "";
            document.getElementById("center-review-sig").checked = false;
            loadLedgerData();
            loadReportsData();
            loadPendingReviewQueue();
            loadHumanReviewHistory();
            refreshDashboardData();
        } else {
            showToast(`Review submission failed: ${data.detail || "Server error"}`, "error");
        }
    } catch (err) {
        showToast("Network error recording review signature.", "error");
    } finally {
        btn.disabled = false;
    }
}

// ==============================================================================
// 10. AUDIT LEDGER LOAD, FILTERING & WORKSPACE RESTORE
// ==============================================================================

async function loadLedgerData() {
    try {
        const res = await fetch("/api/runs?limit=200");
        const data = await res.json();
        if (!res.ok || data.status !== "success") return;

        cachedLedgerItems = data.items || [];
        renderLedgerTable(cachedLedgerItems);
        renderHistoryArchive(cachedLedgerItems);
        populateRunDropdowns(cachedLedgerItems);
    } catch (e) {
        console.warn("Ledger load error:", e);
    }
}

function populateRunDropdowns(items) {
    const sel = document.getElementById("fields-run-selector");
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = `<option value="">-- Select a saved validation --</option>`;
    items.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r.validation_id;
        opt.textContent = `#${r.validation_id} — ${r.file_name} (${r.verdict}, ${r.compliance_score}%)`;
        sel.appendChild(opt);
    });
    if (activeRunId) sel.value = String(activeRunId);
    else if (prev) sel.value = prev;
}

function renderLedgerTable(items) {
    const emptyRow = `<tr><td colspan="7" style="text-align: center; color: var(--color-text-muted); padding: 1.25rem;">No validation records found in database.</td></tr>`;
    const tbody = document.getElementById("ledger-tbody-full");
    if (tbody) {
        tbody.innerHTML = "";
        if (!items || items.length === 0) {
            tbody.innerHTML = emptyRow;
        } else {
            items.forEach(r => tbody.appendChild(createLedgerRow(r)));
        }
    }

    const countLabel = document.getElementById("ledger-record-count");
    if (countLabel) countLabel.textContent = `Showing ${items ? items.length : 0} saved record(s)`;
}

function createLedgerRow(r) {
    const tr = document.createElement("tr");
    const verdictBadge = r.verdict === "COMPLIANT"
        ? `<span class="badge badge-compliant">COMPLIANT</span>`
        : (r.verdict === "CONDITIONAL_PASS" ? `<span class="badge badge-warning">CONDITIONAL_PASS</span>` : `<span class="badge badge-non-compliant">NON_COMPLIANT</span>`);

    const reviewBadge = r.latest_review_status === "APPROVED"
        ? `<span class="badge badge-compliant">APPROVED</span>`
        : (r.latest_review_status === "REJECTED" ? `<span class="badge badge-non-compliant">REJECTED</span>` : (r.latest_review_status === "CHANGES_REQUESTED" ? `<span class="badge badge-warning">CHANGES_REQUESTED</span>` : `<span class="badge badge-neutral">PENDING_REVIEW</span>`));

    tr.innerHTML = `
        <td class="mono" style="font-weight: 700; color: var(--color-navy);">#${r.validation_id}</td>
        <td style="font-weight: 600;">${escapeHtml(r.file_name)}</td>
        <td>${verdictBadge}</td>
        <td style="font-weight: 700;">${r.compliance_score}%</td>
        <td>${reviewBadge}</td>
        <td style="font-size: 0.78rem; color: var(--color-text-muted);">${escapeHtml(r.created_at || "N/A")}</td>
        <td>
            <div style="display: flex; gap: 0.4rem;">
                <button type="button" class="btn btn-secondary btn-sm" onclick="loadRunIntoWorkspace(${r.validation_id})">Inspect</button>
                <a href="/api/reports/${r.validation_id}/pdf" target="_blank" class="btn btn-secondary btn-sm" style="color: var(--color-blue);">Report</a>
            </div>
        </td>
    `;
    return tr;
}

function filterLedgerTable() {
    const q = (document.getElementById("ledger-search-input-full")?.value || "").toLowerCase();
    const verdict = document.getElementById("ledger-filter-verdict")?.value || "ALL";
    const review = document.getElementById("ledger-filter-review")?.value || "ALL";

    const filtered = cachedLedgerItems.filter(item => {
        const matchesQ = !q || item.file_name.toLowerCase().includes(q) || String(item.validation_id).includes(q);
        const matchesV = verdict === "ALL" || item.verdict === verdict;
        const matchesR = review === "ALL" || item.latest_review_status === review;
        return matchesQ && matchesV && matchesR;
    });

    renderLedgerTable(filtered);
}

async function loadRunIntoWorkspace(validationId, targetView = "view-compliance") {
    try {
        const res = await fetch(`/api/runs/${validationId}`);
        const data = await res.json();
        if (res.ok && data.status === "success") {
            activeRunId = validationId;
            activeAuditData = data;
            populateAllViews(data);
            showToast(`Loaded saved Run #${validationId} from database.`, "info");
            if (targetView) navigateTo(targetView);
        } else {
            showToast(`Failed to load Run #${validationId}: ${data.detail || "Not found"}`, "error");
        }
    } catch (e) {
        showToast(`Error loading saved record #${validationId}.`, "error");
    }
}

function exportLedgerCsv() {
    window.location.href = "/api/runs/export";
}

// ==============================================================================
// 11. REPORTS & HISTORY
// ==============================================================================

async function loadReportsData() {
    const tbody = document.getElementById("reports-tbody");
    if (!tbody) return;
    try {
        const res = await fetch("/api/reports");
        const data = await res.json();
        if (!res.ok || data.status !== "success") return;

        tbody.innerHTML = "";
        if (!data.reports || data.reports.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--color-text-muted); padding: 1.25rem;">No compliance reports generated yet. Run an analysis first.</td></tr>`;
            return;
        }

        data.reports.forEach(rep => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="mono" style="font-weight: 700; color: var(--color-navy);">#${rep.validation_id}</td>
                <td style="font-weight: 600;">${escapeHtml(rep.file_name)}</td>
                <td><span class="badge ${rep.verdict === 'COMPLIANT' ? 'badge-compliant' : (rep.verdict === 'CONDITIONAL_PASS' ? 'badge-warning' : 'badge-non-compliant')}">${escapeHtml(rep.verdict)}</span></td>
                <td style="font-weight: 700;">${rep.compliance_score}%</td>
                <td><span class="badge ${rep.review_status === 'APPROVED' ? 'badge-compliant' : 'badge-neutral'}">${escapeHtml(rep.review_status)}</span></td>
                <td style="font-size: 0.78rem; color: var(--color-text-muted); white-space: nowrap;">${escapeHtml(rep.audit_date || "N/A")}</td>
                <td style="white-space: nowrap;">
                    <div style="display: inline-flex; align-items: center; gap: 0.35rem; flex-wrap: nowrap;">
                        <a href="${rep.pdf_url}" target="_blank" class="btn btn-primary btn-sm">Dossier</a>
                        <a href="${rep.json_url}" class="btn btn-secondary btn-sm">JSON</a>
                        <a href="${rep.csv_url}" class="btn btn-secondary btn-sm">CSV</a>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn("Reports load error:", e);
    }
}

function renderHistoryArchive(items) {
    const tbody = document.getElementById("history-archive-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!items || items.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--color-text-muted); padding: 1.25rem;">No analysis history saved in database yet.</td></tr>`;
        return;
    }

    items.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="mono" style="font-weight: 700; color: var(--color-navy);">#${r.validation_id}</td>
            <td style="font-weight: 600;">${escapeHtml(r.file_name)}</td>
            <td><span class="badge ${r.verdict === 'COMPLIANT' ? 'badge-compliant' : (r.verdict === 'CONDITIONAL_PASS' ? 'badge-warning' : 'badge-non-compliant')}">${escapeHtml(r.verdict)}</span></td>
            <td style="font-weight: 700;">${r.compliance_score}%</td>
            <td><span class="badge badge-non-compliant">${r.critical_errors ?? 0}</span></td>
            <td><span class="badge badge-warning">${r.major_warnings ?? 0}</span></td>
            <td><span class="badge ${r.latest_review_status === 'APPROVED' ? 'badge-compliant' : 'badge-neutral'}">${escapeHtml(r.latest_review_status)}</span></td>
            <td style="font-size: 0.78rem; color: var(--color-text-muted);">${escapeHtml(r.created_at || "N/A")}</td>
            <td>
                <button type="button" class="btn btn-secondary btn-sm" onclick="loadRunIntoWorkspace(${r.validation_id})">Inspect</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// ==============================================================================
// 12. SETTINGS (REAL BACKEND STORAGE + USER CONFIRMATION)
// ==============================================================================

async function loadSettingsData() {
    try {
        const res = await fetch("/api/settings");
        const data = await res.json();
        if (!res.ok || data.status !== "success") return;

        const s = data.settings || {};
        const setVal = (id, v) => {
            const el = document.getElementById(id);
            if (el) el.value = v || "";
        };

        setVal("settings-auditor-name", s.auditor_name);
        setVal("settings-auditor-email", s.auditor_email);
        setVal("settings-auditor-role", s.auditor_role);
        setVal("settings-org-name", s.organization_name);
        setVal("settings-facility-id", s.facility_id);
        setVal("settings-jurisdiction", s.jurisdiction);
    } catch (e) {
        console.warn("Settings load error:", e);
    }
}

async function saveAuditorSettings(e) {
    e.preventDefault();
    const confirmed = window.confirm("Save changes to persistent settings storage on the server?");
    if (!confirmed) return;

    const payload = {
        auditor_name: document.getElementById("settings-auditor-name")?.value || "",
        auditor_email: document.getElementById("settings-auditor-email")?.value || "",
        auditor_role: document.getElementById("settings-auditor-role")?.value || "",
        organization_name: document.getElementById("settings-org-name")?.value || "",
        facility_id: document.getElementById("settings-facility-id")?.value || "",
        jurisdiction: document.getElementById("settings-jurisdiction")?.value || ""
    };

    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok && data.status === "success") {
            showToast("Settings saved to backend storage.", "success");
        } else {
            showToast("Failed to save settings.", "error");
        }
    } catch (err) {
        showToast("Error communicating with settings API.", "error");
    }
}

// ==============================================================================
// 13. GLOBAL SEARCH & DETAIL MODALS
// ==============================================================================

function handleGlobalSearch(e) {
    const term = e.target.value.trim().toLowerCase();
    if (!term) return;
    if (term.startsWith("run-") || /^\d+$/.test(term)) {
        const num = Number(term.replace("run-", ""));
        if (num > 0) loadRunIntoWorkspace(num);
    }
}

function openDetailModalWithContent(htmlContent) {
    const modal = document.getElementById("detail-modal-overlay");
    const body = document.getElementById("modal-run-body");
    if (modal && body) {
        body.innerHTML = htmlContent;
        modal.classList.add("open");
    }
}

function closeDetailModal(e) {
    if (e && e.target !== e.currentTarget && !e.target.classList.contains("modal-close-btn") && !e.target.closest(".modal-close-btn") && e.target.tagName !== "BUTTON") {
        return;
    }
    const modal = document.getElementById("detail-modal-overlay");
    if (modal) modal.classList.remove("open");
}

// ==============================================================================
// 14. REQUIRED FIELDS INSPECTOR (100% BACKEND & DB DRIVEN)
// ==============================================================================

async function loadRequiredFieldsForRun(validationId, preloadedRunData = null) {
    const emptyState = document.getElementById("fields-empty-state");
    const workspace = document.getElementById("fields-results-workspace");
    const runBadge = document.getElementById("fields-active-run-badge");
    const verdictPill = document.getElementById("fields-verdict-pill");
    const previewImg = document.getElementById("fields-label-preview-img");
    const noImgMsg = document.getElementById("fields-no-image-msg");
    const metaBox = document.getElementById("fields-file-meta-box");

    if (!validationId) {
        cachedRequiredFields = [];
        if (emptyState) emptyState.style.display = "block";
        if (workspace) workspace.style.display = "none";
        if (runBadge) {
            runBadge.className = "badge badge-neutral";
            runBadge.textContent = "No analysis yet";
        }
        renderRequiredFieldsDeck();
        return;
    }

    try {
        const res = await fetch(`/api/required-fields?validation_id=${validationId}`);
        const data = await res.json();
        if (!res.ok || data.status !== "success") {
            showToast(`Could not load fields for Run #${validationId}`, "error");
            return;
        }

        cachedRequiredFields = data.fields || [];
        if (emptyState) emptyState.style.display = "none";
        if (workspace) workspace.style.display = "block";

        if (runBadge) {
            runBadge.className = "badge badge-info";
            runBadge.textContent = `Run #${validationId} • ${data.file_name || ""}`;
        }
        if (verdictPill) {
            verdictPill.className = `badge ${data.verdict === "COMPLIANT" ? "badge-compliant" : (data.verdict === "CONDITIONAL_PASS" ? "badge-warning" : "badge-non-compliant")}`;
            verdictPill.textContent = `${data.verdict} (${data.compliance_score}%)`;
        }

        const runSource = preloadedRunData || activeAuditData;
        if (previewImg && noImgMsg) {
            if (runSource && (runSource.annotated_image || runSource.raw_image)) {
                previewImg.src = runSource.annotated_image || runSource.raw_image;
                previewImg.style.display = "block";
                noImgMsg.style.display = "none";
            } else {
                previewImg.style.display = "none";
                noImgMsg.style.display = "block";
            }
        }

        if (metaBox) {
            const fm = runSource?.file_metadata;
            const q = fm?.quality;
            metaBox.innerHTML = `
                <div><strong>Document:</strong> ${escapeHtml(data.file_name || "N/A")}</div>
                ${runSource?.file_hash ? `<div class="mono"><strong>SHA-256:</strong> ${escapeHtml(runSource.file_hash.slice(0, 24))}...</div>` : ""}
                ${q ? `<div><strong>Image Quality (DocProcessor):</strong> Blur var ${q.blur_metric}, Brightness ${q.mean_brightness} (${q.quality_acceptable ? "Acceptable" : "Low quality warning"})</div>` : ""}
            `;
        }

        const sel = document.getElementById("fields-run-selector");
        if (sel) sel.value = String(validationId);

        renderRequiredFieldsDeck();
    } catch (e) {
        console.error("Failed loading required fields:", e);
    }
}

function handleFieldsRunSelect(event) {
    const valId = event.target.value;
    if (!valId) {
        activeRunId = null;
        activeAuditData = null;
        loadRequiredFieldsForRun(null);
        return;
    }
    loadRunIntoWorkspace(Number(valId), "view-fields");
}

function renderRequiredFieldsDeck() {
    const deck = document.getElementById("required-fields-deck");
    if (!deck) return;
    deck.innerHTML = "";

    const filtered = cachedRequiredFields.filter(f => {
        if (activeFieldStatusFilter !== "all" && f.status !== activeFieldStatusFilter) return false;
        if (activeFieldSearchQuery) {
            const q = activeFieldSearchQuery.toLowerCase();
            return f.name.toLowerCase().includes(q)
                || (f.regulation || "").toLowerCase().includes(q)
                || (f.value || "").toLowerCase().includes(q);
        }
        return true;
    });

    if (filtered.length === 0) {
        deck.innerHTML = `<div style="text-align: center; padding: 1.5rem; color: var(--color-text-muted); font-size: 0.84rem;">No fields match your filter.</div>`;
        return;
    }

    filtered.forEach(f => {
        const card = document.createElement("div");
        const statusCls = f.status === "DETECTED" ? "status-compliant" : (f.status === "FAILED" ? "status-error" : "status-warning");
        card.className = `field-card-item ${statusCls}`;
        card.onclick = () => inspectField(f.id);

        let statusBadge = `<span class="badge badge-neutral">Not detected</span>`;
        if (f.status === "DETECTED") statusBadge = `<span class="badge badge-compliant">Detected</span>`;
        else if (f.status === "FAILED") statusBadge = `<span class="badge badge-non-compliant">Failed</span>`;
        else if (f.status === "NOT_ANALYZED") statusBadge = `<span class="badge badge-neutral">Not analyzed</span>`;

        const valueText = f.value ? escapeHtml(f.value) : `<span style="color: var(--color-text-muted); font-style: italic;">Not detected</span>`;
        const confHtml = f.confidence !== null && f.confidence !== undefined
            ? `Extractor confidence: <strong style="color: var(--color-blue);">${(f.confidence * 100).toFixed(1)}%</strong>`
            : `Confidence: Not reported`;

        card.innerHTML = `
            <div class="field-card-top-row">
                <div>
                    <div class="field-card-name">${escapeHtml(f.name)}</div>
                    <div class="field-card-reg">${escapeHtml(f.regulation)}</div>
                </div>
                <div>${statusBadge}</div>
            </div>
            <div class="field-card-value-box">
                <span>${valueText}</span>
            </div>
            <div class="field-card-meta-row">
                <span>${confHtml}</span>
                <span style="color: var(--color-blue); font-weight: 600;">Inspect &rarr;</span>
            </div>
        `;
        deck.appendChild(card);
    });
}

function handleFieldSearch(event) {
    activeFieldSearchQuery = event.target.value.trim();
    renderRequiredFieldsDeck();
}

function handleFieldStatusFilter(event) {
    activeFieldStatusFilter = event.target.value;
    renderRequiredFieldsDeck();
}

function inspectField(fieldId) {
    const field = cachedRequiredFields.find(x => x.id === fieldId);
    if (!field) return;

    const modal = document.getElementById("field-inspector-modal");
    if (!modal) return;

    document.getElementById("modal-field-title").textContent = field.name;
    document.getElementById("modal-field-reg").textContent = field.regulation;
    document.getElementById("modal-field-val").textContent = field.value || "Not detected";
    document.getElementById("modal-field-source").textContent = activeRunId ? `Run #${activeRunId}` : "No run selected";
    document.getElementById("modal-field-conf").textContent = field.confidence !== null && field.confidence !== undefined
        ? `${(field.confidence * 100).toFixed(1)}%`
        : "Not reported";

    const statusEl = document.getElementById("modal-field-status");
    statusEl.textContent = field.status;
    statusEl.className = `badge ${field.status === "DETECTED" ? "badge-compliant" : (field.status === "FAILED" ? "badge-non-compliant" : "badge-neutral")}`;

    modal.style.display = "flex";
}

function closeFieldModal() {
    const modal = document.getElementById("field-inspector-modal");
    if (modal) modal.style.display = "none";
}
