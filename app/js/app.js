/**
 * NephroScan AI — Frontend Application Logic
 * Pure Vanilla JavaScript Client
 */

(function () {
    'use strict';

    // ---------------------------------------------------------------------------
    // Configuration & State
    // ---------------------------------------------------------------------------
    const STORAGE_KEY_API_URL = 'nephroscan_api_url';
    
    // Auto-detect base API URL (handles same-origin if hosted by FastAPI or standalone fallback)
    function getDefaultApiUrl() {
        if (window.location.protocol.startsWith('http')) {
            return window.location.origin;
        }
        return 'http://127.0.0.1:8000';
    }

    const state = {
        apiUrl: localStorage.getItem(STORAGE_KEY_API_URL) || getDefaultApiUrl(),
        currentFile: null,
        currentFileBlob: null,
        isAnalyzing: false,
        lastResult: null,
        healthCheckTimer: null,
    };

    // Clinical knowledge base for UI insights
    const CLINICAL_INSIGHTS = {
        Normal: {
            summary: "Normal renal parenchyma demonstrated with preserved corticomedullary differentiation and absence of focal calculi, cystic lesions, or space-occupying neoplasms.",
            reference: "Homogeneous renal cortex and medulla with smooth renal contours. Renal sinus fat and collecting system appear anatomically unremarkable with no pelvicalyceal dilatation.",
            icon: "check-circle",
            colorClass: "theme-normal",
            dotColor: "var(--class-normal)",
        },
        Cyst: {
            summary: "Well-circumscribed fluid-attenuation lesion identified. Consistent with Bosniak Category benign/simple or complex renal cystic lesion.",
            reference: "Typically exhibits smooth thin walls, homogeneous water-density (<20 Hounsfield Units), with lack of internal calcification or pathological contrast enhancement.",
            icon: "alert-circle",
            colorClass: "theme-cyst",
            dotColor: "var(--class-cyst)",
        },
        Stone: {
            summary: "Hyperdense calcified focus identified within the renal calyces or pelviureteric junction, indicative of nephrolithiasis / urolithiasis.",
            reference: "Radio-opaque density (>200-400 HU on unenhanced CT). Evaluation recommended for secondary signs including hydronephrosis, perinephric stranding, or ureteral obstruction.",
            icon: "disc",
            colorClass: "theme-stone",
            dotColor: "var(--class-stone)",
        },
        Tumor: {
            summary: "Space-occupying solid or heterogeneous renal cortical mass detected, suspicious for Renal Cell Carcinoma (RCC) or oncocytoma.",
            reference: "Characterized by soft-tissue attenuation, irregular margins, internal heterogeneity, necrosis, or vascular invasion. Immediate formal radiologic staging recommended.",
            icon: "alert-triangle",
            colorClass: "theme-tumor",
            dotColor: "var(--class-tumor)",
        }
    };

    // ---------------------------------------------------------------------------
    // DOM Elements
    // ---------------------------------------------------------------------------
    const elements = {
        // Status & Nav
        statusPill: document.getElementById('backendStatusPill'),
        statusDot: document.getElementById('statusDot'),
        statusLabel: document.getElementById('statusLabel'),
        latencyTag: document.getElementById('latencyTag'),
        btnOpenSettings: document.getElementById('btnOpenSettings'),
        
        // Settings Modal
        settingsModal: document.getElementById('settingsModal'),
        btnCloseSettings: document.getElementById('btnCloseSettings'),
        apiUrlInput: document.getElementById('apiUrlInput'),
        btnSaveSettings: document.getElementById('btnSaveSettings'),
        btnResetApiUrl: document.getElementById('btnResetApiUrl'),
        modalApiStatus: document.getElementById('modalApiStatus'),
        modalModelStatus: document.getElementById('modalModelStatus'),

        // Input & Upload
        dropZone: document.getElementById('dropZone'),
        dropPrompt: document.getElementById('dropPrompt'),
        fileInput: document.getElementById('fileInput'),
        previewContainer: document.getElementById('previewContainer'),
        imagePreview: document.getElementById('imagePreview'),
        previewDimensions: document.getElementById('previewDimensions'),
        previewSize: document.getElementById('previewSize'),
        btnRemoveImage: document.getElementById('btnRemoveImage'),
        scannerLine: document.getElementById('scannerLine'),
        sampleCards: document.querySelectorAll('.sample-card'),
        btnAnalyze: document.getElementById('btnAnalyze'),
        analyzeBtnText: document.getElementById('analyzeBtnText'),
        analyzeSpinner: document.getElementById('analyzeSpinner'),

        // Results
        resultsPanel: document.getElementById('resultsPanel'),
        emptyState: document.getElementById('emptyState'),
        resultsContent: document.getElementById('resultsContent'),
        headlineCard: document.getElementById('headlineCard'),
        inferenceTimeVal: document.getElementById('inferenceTimeVal'),
        predClassBadge: document.getElementById('predClassBadge'),
        predClassIcon: document.getElementById('predClassIcon'),
        predClassName: document.getElementById('predClassName'),
        confidenceVal: document.getElementById('confidenceVal'),
        confidenceMiniFill: document.getElementById('confidenceMiniFill'),
        clinicalSummary: document.getElementById('clinicalSummary'),
        probBarsContainer: document.getElementById('probBarsContainer'),
        insightsCard: document.getElementById('insightsCard'),
        insightsText: document.getElementById('insightsText'),
        btnCopyReport: document.getElementById('btnCopyReport'),
        copyReportText: document.getElementById('copyReportText'),
        btnDownloadReport: document.getElementById('btnDownloadReport'),
        
        // Errors
        errorBanner: document.getElementById('errorBanner'),
        errorTitle: document.getElementById('errorTitle'),
        errorMessage: document.getElementById('errorMessage'),
        btnDismissError: document.getElementById('btnDismissError'),
    };

    // ---------------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------------
    function init() {
        refreshIcons();
        setupEventListeners();
        checkBackendHealth();
        // Periodic health check every 15 seconds
        state.healthCheckTimer = setInterval(checkBackendHealth, 15000);
    }

    function refreshIcons() {
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    // ---------------------------------------------------------------------------
    // Event Listeners
    // ---------------------------------------------------------------------------
    function setupEventListeners() {
        // Drag & Drop
        elements.dropZone.addEventListener('click', (e) => {
            if (e.target !== elements.btnRemoveImage && !elements.btnRemoveImage.contains(e.target)) {
                elements.fileInput.click();
            }
        });

        elements.fileInput.addEventListener('change', handleFileSelect);

        ['dragenter', 'dragover'].forEach(eventName => {
            elements.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                elements.dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            elements.dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                elements.dropZone.classList.remove('dragover');
            });
        });

        elements.dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files && dt.files.length > 0) {
                processImageFile(dt.files[0]);
            }
        });

        // Remove image
        elements.btnRemoveImage.addEventListener('click', (e) => {
            e.stopPropagation();
            resetImageInput();
        });

        // Sample Cards Click
        elements.sampleCards.forEach(card => {
            card.addEventListener('click', () => {
                const samplePath = card.getAttribute('data-path');
                const sampleLabel = card.getAttribute('data-label');
                loadSampleImage(samplePath, sampleLabel, card);
            });
        });

        // Analyze Button Click
        elements.btnAnalyze.addEventListener('click', runInference);

        // Keyboard Shortcut: Ctrl/Cmd + Enter to trigger analysis
        window.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                if (!elements.btnAnalyze.disabled && !state.isAnalyzing) {
                    runInference();
                }
            }
        });

        // Export Actions
        elements.btnCopyReport.addEventListener('click', copyDiagnosticReport);
        elements.btnDownloadReport.addEventListener('click', downloadDiagnosticReport);

        // Error Dismiss
        elements.btnDismissError.addEventListener('click', hideError);

        // Settings Modal
        elements.btnOpenSettings.addEventListener('click', openSettingsModal);
        elements.btnCloseSettings.addEventListener('click', closeSettingsModal);
        elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === elements.settingsModal) closeSettingsModal();
        });
        elements.btnSaveSettings.addEventListener('click', saveSettings);
        elements.btnResetApiUrl.addEventListener('click', resetSettingsToDefault);
    }

    // ---------------------------------------------------------------------------
    // Health Check & Connectivity
    // ---------------------------------------------------------------------------
    async function checkBackendHealth() {
        const startTime = performance.now();
        const endpoint = `${state.apiUrl.replace(/\/$/, '')}/health`;

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 6000);

            const response = await fetch(endpoint, {
                method: 'GET',
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            });
            clearTimeout(timeoutId);

            const latency = Math.round(performance.now() - startTime);

            if (response.ok) {
                const data = await response.json();
                updateHealthStatusUI('healthy', 'Backend Connected', latency, data.model_loaded);
            } else {
                updateHealthStatusUI('degraded', `HTTP ${response.status}`, latency, false);
            }
        } catch (error) {
            updateHealthStatusUI('offline', 'API Offline / Unreachable', null, false);
        }
    }

    function updateHealthStatusUI(status, label, latency, modelLoaded) {
        elements.statusDot.className = `status-dot ${status}`;
        elements.statusLabel.textContent = label;
        
        if (latency !== null) {
            elements.latencyTag.textContent = `${latency} ms`;
            elements.latencyTag.classList.remove('hidden');
        } else {
            elements.latencyTag.classList.add('hidden');
        }

        // Modal fields
        elements.modalApiStatus.textContent = label;
        elements.modalApiStatus.style.color = status === 'healthy' ? 'var(--class-normal)' : 'var(--class-tumor)';
        elements.modalModelStatus.textContent = modelLoaded ? 'Loaded in Memory (Ready)' : 'Unavailable';
        elements.modalModelStatus.style.color = modelLoaded ? 'var(--class-normal)' : 'var(--class-tumor)';
    }

    // ---------------------------------------------------------------------------
    // File & Sample Handling
    // ---------------------------------------------------------------------------
    function handleFileSelect(e) {
        if (e.target.files && e.target.files.length > 0) {
            processImageFile(e.target.files[0]);
        }
    }

    function processImageFile(file) {
        hideError();
        
        // Validate extension
        const validExts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'];
        const fileName = file.name.toLowerCase();
        const isValidExt = validExts.some(ext => fileName.endsWith(ext));

        if (!isValidExt) {
            showError('Invalid File Type', 'Please upload a valid CT image file (JPG, PNG, BMP, or TIFF).');
            return;
        }

        // Validate size (10 MB max)
        const maxBytes = 10 * 1024 * 1024;
        if (file.size > maxBytes) {
            showError('File Too Large', 'Selected CT scan exceeds the 10MB size limit.');
            return;
        }

        state.currentFile = file;

        // Clear sample active states
        elements.sampleCards.forEach(c => c.classList.remove('active'));

        // Display preview
        const reader = new FileReader();
        reader.onload = function (event) {
            elements.imagePreview.src = event.target.result;
            elements.dropPrompt.classList.add('hidden');
            elements.previewContainer.classList.remove('hidden');
            
            // Image resolution & size details
            const img = new Image();
            img.onload = function () {
                elements.previewDimensions.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
            };
            img.src = event.target.result;

            const sizeKb = (file.size / 1024).toFixed(1);
            elements.previewSize.textContent = `${sizeKb} KB`;

            elements.btnAnalyze.disabled = false;
            refreshIcons();
        };
        reader.readAsDataURL(file);
    }

    async function loadSampleImage(samplePath, sampleLabel, cardElement) {
        hideError();
        try {
            // Highlight active sample
            elements.sampleCards.forEach(c => c.classList.remove('active'));
            cardElement.classList.add('active');

            const response = await fetch(samplePath);
            if (!response.ok) throw new Error('Could not load sample image file');
            const blob = await response.blob();
            
            const file = new File([blob], `${sampleLabel.toLowerCase()}_sample.jpg`, { type: 'image/jpeg' });
            processImageFile(file);
        } catch (error) {
            showError('Sample Load Error', `Failed to load sample CT scan: ${error.message}`);
        }
    }

    function resetImageInput() {
        state.currentFile = null;
        elements.fileInput.value = '';
        elements.imagePreview.src = '';
        elements.previewContainer.classList.add('hidden');
        elements.dropPrompt.classList.remove('hidden');
        elements.btnAnalyze.disabled = true;
        elements.sampleCards.forEach(c => c.classList.remove('active'));
        hideScanner();
        refreshIcons();
    }

    // ---------------------------------------------------------------------------
    // Inference Execution
    // ---------------------------------------------------------------------------
    async function runInference() {
        if (!state.currentFile || state.isAnalyzing) return;

        hideError();
        setAnalyzingState(true);
        const startTime = performance.now();

        const formData = new FormData();
        formData.append('file', state.currentFile, state.currentFile.name);

        const endpoint = `${state.apiUrl.replace(/\/$/, '')}/predict`;

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 35000); // 35s to handle cold starts

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
            });
            clearTimeout(timeoutId);

            const latency = Math.round(performance.now() - startTime);

            if (response.ok) {
                const data = await response.json();
                state.lastResult = { ...data, latencyMs: latency, filename: state.currentFile.name };
                renderResults(state.lastResult);
            } else if (response.status === 503) {
                showError('Model Service Unavailable', 'The deep learning model is still initializing on the server. Please retry in a few seconds.');
            } else if (response.status === 400 || response.status === 422) {
                const errJson = await response.json().catch(() => ({}));
                showError('Validation Error', errJson.detail || 'The uploaded image could not be processed.');
            } else {
                showError(`Server Error (${response.status})`, 'An unexpected error occurred during model inference.');
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                showError('Request Timeout', 'The inference request timed out. If the backend is hosted on a free tier, it may be waking from cold sleep.');
            } else {
                showError('Connection Error', `Failed to communicate with inference endpoint (${endpoint}). Verify server is running.`);
            }
        } finally {
            setAnalyzingState(false);
        }
    }

    function setAnalyzingState(isAnalyzing) {
        state.isAnalyzing = isAnalyzing;
        elements.btnAnalyze.disabled = isAnalyzing;
        
        if (isAnalyzing) {
            elements.analyzeBtnText.textContent = 'Processing CT Scan...';
            elements.analyzeSpinner.classList.remove('hidden');
            showScanner();
        } else {
            elements.analyzeBtnText.textContent = 'Run Diagnostic Analysis';
            elements.analyzeSpinner.classList.add('hidden');
            hideScanner();
        }
    }

    function showScanner() {
        elements.scannerLine.classList.remove('hidden');
    }

    function hideScanner() {
        elements.scannerLine.classList.add('hidden');
    }

    // ---------------------------------------------------------------------------
    // Results Rendering
    // ---------------------------------------------------------------------------
    function renderResults(result) {
        const { prediction, confidence, probabilities, latencyMs } = result;
        const info = CLINICAL_INSIGHTS[prediction] || CLINICAL_INSIGHTS.Normal;

        // Hide empty state, show results
        elements.emptyState.classList.add('hidden');
        elements.resultsContent.classList.remove('hidden');

        // Headline & Theme
        elements.headlineCard.className = `prediction-headline-card ${info.colorClass}`;
        elements.predClassName.textContent = prediction;
        elements.predClassIcon.setAttribute('data-lucide', info.icon);
        elements.inferenceTimeVal.textContent = `${latencyMs}ms`;

        // Confidence gauge
        const confPercent = (confidence * 100).toFixed(2);
        elements.confidenceVal.textContent = `${confPercent}%`;
        elements.confidenceMiniFill.style.width = `${confPercent}%`;

        // Clinical summary & insights
        elements.clinicalSummary.textContent = info.summary;
        elements.insightsText.textContent = info.reference;

        // Probability Distribution Bars
        renderProbabilityBars(probabilities, prediction);

        refreshIcons();
    }

    function renderProbabilityBars(probabilities, topClass) {
        elements.probBarsContainer.innerHTML = '';

        // Sort classes by probability descending
        const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

        sortedEntries.forEach(([className, prob]) => {
            const percent = (prob * 100).toFixed(2);
            const isTop = className === topClass;
            const classInfo = CLINICAL_INSIGHTS[className] || { dotColor: 'var(--primary)' };

            const item = document.createElement('div');
            item.className = `prob-item ${isTop ? 'is-top' : ''}`;

            item.innerHTML = `
                <div class="prob-item-header">
                    <span class="prob-class-name">
                        <span class="prob-class-dot" style="background-color: ${classInfo.dotColor}"></span>
                        ${className}
                    </span>
                    <span class="prob-item-val">${percent}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-bar-fill" style="width: 0%; background: ${classInfo.dotColor}"></div>
                </div>
            `;

            elements.probBarsContainer.appendChild(item);

            // Animate progress bar fill smoothly
            requestAnimationFrame(() => {
                setTimeout(() => {
                    const fill = item.querySelector('.prob-bar-fill');
                    if (fill) fill.style.width = `${percent}%`;
                }, 50);
            });
        });
    }

    // ---------------------------------------------------------------------------
    // Report Exporting
    // ---------------------------------------------------------------------------
    function copyDiagnosticReport() {
        if (!state.lastResult) return;

        const res = state.lastResult;
        const textReport = 
`==================================================
NEPHROSCAN AI — CT DIAGNOSTIC INFERENCE REPORT
==================================================
Date/Time:        ${new Date().toISOString()}
Scan Slice:       ${res.filename}
Primary Finding:  ${res.prediction}
Confidence:       ${(res.confidence * 100).toFixed(2)}%
Latency:          ${res.latencyMs} ms

Softmax Class Probability Distribution:
${Object.entries(res.probabilities).map(([k, v]) => `  - ${k.padEnd(8)}: ${(v * 100).toFixed(2)}%`).join('\n')}

Clinical Summary:
${CLINICAL_INSIGHTS[res.prediction]?.summary || ''}

DISCLAIMER:
${res.disclaimer || 'For educational and research demonstration only. Not a medical device.'}
==================================================`;

        navigator.clipboard.writeText(textReport).then(() => {
            elements.copyReportText.textContent = 'Copied!';
            setTimeout(() => {
                elements.copyReportText.textContent = 'Copy Report';
            }, 2000);
        }).catch(err => {
            showError('Clipboard Error', 'Could not copy to clipboard.');
        });
    }

    function downloadDiagnosticReport() {
        if (!state.lastResult) return;

        const reportData = {
            system: 'NephroScan AI',
            model: 'EfficientNetB0',
            timestamp: new Date().toISOString(),
            input_file: state.lastResult.filename,
            prediction: state.lastResult.prediction,
            confidence: state.lastResult.confidence,
            probabilities: state.lastResult.probabilities,
            inference_latency_ms: state.lastResult.latencyMs,
            clinical_insights: CLINICAL_INSIGHTS[state.lastResult.prediction] || {},
            disclaimer: state.lastResult.disclaimer,
        };

        const jsonStr = JSON.stringify(reportData, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nephroscan_${state.lastResult.prediction.toLowerCase()}_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ---------------------------------------------------------------------------
    // Settings & Errors
    // ---------------------------------------------------------------------------
    function openSettingsModal() {
        elements.apiUrlInput.value = state.apiUrl;
        elements.settingsModal.classList.remove('hidden');
        checkBackendHealth();
        refreshIcons();
    }

    function closeSettingsModal() {
        elements.settingsModal.classList.add('hidden');
    }

    function saveSettings() {
        const val = elements.apiUrlInput.value.trim().replace(/\/$/, '');
        if (val) {
            state.apiUrl = val;
            localStorage.setItem(STORAGE_KEY_API_URL, val);
        }
        closeSettingsModal();
        checkBackendHealth();
    }

    function resetSettingsToDefault() {
        const def = getDefaultApiUrl();
        elements.apiUrlInput.value = def;
        state.apiUrl = def;
        localStorage.removeItem(STORAGE_KEY_API_URL);
        checkBackendHealth();
    }

    function showError(title, message) {
        elements.errorTitle.textContent = title;
        elements.errorMessage.textContent = message;
        elements.errorBanner.classList.remove('hidden');
        refreshIcons();
    }

    function hideError() {
        elements.errorBanner.classList.add('hidden');
    }

    // Bootstrap
    document.addEventListener('DOMContentLoaded', init);
})();
