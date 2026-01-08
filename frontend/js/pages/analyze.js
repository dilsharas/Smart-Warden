/**
 * Analyze Page Component
 */

const AnalyzePage = {
    /**
     * Render analyze page
     */
    render: function() {
        const container = document.getElementById('pageContainer');
        const currentAnalysis = StateManager.getValue('currentAnalysis');
        const contractCode = currentAnalysis?.contractCode || '';

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">🔍 Analyze Contract</h1>
                    <p class="page-description">Upload or paste your Solidity smart contract code for AI-powered security analysis</p>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">Contract Input</h2>
                    </div>
                    <div class="card-body">
                        <div class="form-group">
                            <label class="form-label">Upload File</label>
                            <div id="uploadZone" class="upload-zone">
                                <div class="upload-content">
                                    <p>📁 Drag and drop .sol file here or click to browse</p>
                                    <input type="file" id="fileInput" accept=".sol" style="display: none;">
                                </div>
                            </div>
                            <small class="text-muted">Maximum file size: 10MB</small>
                        </div>

                        <div style="text-align: center; margin: var(--spacing-lg) 0;">
                            <span style="color: var(--color-text-secondary);">OR</span>
                        </div>

                        <div class="form-group">
                            <label class="form-label">Paste Code</label>
                            <textarea id="contractCode" class="form-textarea" placeholder="pragma solidity ^0.8.0;&#10;&#10;contract MyContract {&#10;    // Your code here&#10;}" style="font-family: var(--font-mono); font-size: var(--font-size-sm);">${Helpers.escapeHtml(contractCode)}</textarea>
                            <small class="text-muted">Paste your Solidity contract code here</small>
                        </div>

                        <div class="form-group">
                            <label class="form-label">⚙️ Analysis Options</label>
                            
                            <div style="background-color: var(--color-bg-secondary); padding: var(--spacing-lg); border-radius: var(--radius-md); margin-bottom: var(--spacing-lg);">
                                <h4 style="margin-top: 0; margin-bottom: var(--spacing-md);">Analysis Methods</h4>
                                <div style="display: flex; flex-direction: column; gap: var(--spacing-md);">
                                    <label style="display: flex; align-items: center; gap: var(--spacing-md); cursor: pointer;">
                                        <input type="checkbox" id="aiAnalysis" checked>
                                        <span>🤖 AI-Powered Analysis</span>
                                    </label>
                                    <label style="display: flex; align-items: center; gap: var(--spacing-md); cursor: pointer;">
                                        <input type="checkbox" id="patternAnalysis" checked>
                                        <span>🔍 Pattern-Based Analysis</span>
                                    </label>
                                    <label style="display: flex; align-items: center; gap: var(--spacing-md); cursor: pointer;">
                                        <input type="checkbox" id="externalTools" checked>
                                        <span>🔧 External Tools (Slither, Mythril)</span>
                                    </label>
                                </div>
                            </div>

                            <div style="background-color: var(--color-bg-secondary); padding: var(--spacing-lg); border-radius: var(--radius-md); margin-bottom: var(--spacing-lg);">
                                <h4 style="margin-top: 0; margin-bottom: var(--spacing-md);">AI Model Selection</h4>
                                <div style="display: flex; flex-direction: column; gap: var(--spacing-md);">
                                    <div class="form-group" style="margin-bottom: 0;">
                                        <label class="form-label">Model Type</label>
                                        <select id="modelType" class="form-select">
                                            <option value="binary">Binary Classification (Vulnerable/Safe)</option>
                                            <option value="multiclass">Multi-class Classification (Severity Levels)</option>
                                            <option value="ensemble">Ensemble (Combined Models)</option>
                                        </select>
                                    </div>
                                    <div class="form-group" style="margin-bottom: 0;">
                                        <label class="form-label">Confidence Threshold</label>
                                        <div style="display: flex; align-items: center; gap: var(--spacing-md);">
                                            <input type="range" id="confidenceThreshold" class="form-input" min="0" max="100" value="50" style="flex: 1;">
                                            <span id="confidenceValue" style="min-width: 50px; text-align: right; font-weight: bold;">50%</span>
                                        </div>
                                        <small class="text-muted">Only report vulnerabilities with confidence above this threshold</small>
                                    </div>
                                </div>
                            </div>

                            <div style="background-color: var(--color-bg-secondary); padding: var(--spacing-lg); border-radius: var(--radius-md);">
                                <h4 style="margin-top: 0; margin-bottom: var(--spacing-md);">Analysis Configuration</h4>
                                <div style="display: flex; flex-direction: column; gap: var(--spacing-md);">
                                    <div class="form-group" style="margin-bottom: 0;">
                                        <label class="form-label">Analysis Timeout (seconds)</label>
                                        <input type="number" id="analysisTimeout" class="form-input" min="5" max="300" value="30" placeholder="30">
                                        <small class="text-muted">Maximum time to wait for analysis (5-300 seconds)</small>
                                    </div>
                                    <div class="form-group" style="margin-bottom: 0;">
                                        <label class="form-label">Report Format</label>
                                        <select id="reportFormat" class="form-select">
                                            <option value="detailed">Detailed Report (All findings)</option>
                                            <option value="summary">Summary Report (Critical only)</option>
                                            <option value="minimal">Minimal Report (Vulnerabilities only)</option>
                                        </select>
                                    </div>
                                    <label style="display: flex; align-items: center; gap: var(--spacing-md); cursor: pointer;">
                                        <input type="checkbox" id="includeRecommendations" checked>
                                        <span>Include Fix Recommendations</span>
                                    </label>
                                    <label style="display: flex; align-items: center; gap: var(--spacing-md); cursor: pointer;">
                                        <input type="checkbox" id="parallelAnalysis" checked>
                                        <span>Parallel Analysis (Faster, uses more resources)</span>
                                    </label>
                                </div>
                            </div>
                        </div>

                        <div style="display: flex; gap: var(--spacing-md); margin-top: var(--spacing-lg);">
                            <button id="analyzeBtn" class="btn btn-primary" style="flex: 1;">
                                🚀 Analyze Contract
                            </button>
                            <button id="clearBtn" class="btn btn-secondary">
                                🗑️ Clear
                            </button>
                        </div>
                    </div>
                </div>

                <div id="progressContainer" style="display: none; margin-top: var(--spacing-lg);">
                    <div class="card">
                        <div class="card-body">
                            <p>Analyzing contract...</p>
                            <div class="progress-bar">
                                <div id="progressFill" class="progress-fill" style="width: 0%;"></div>
                            </div>
                            <p id="progressText" style="text-align: center; margin-top: var(--spacing-md); color: var(--color-text-secondary);">0%</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.attachEventListeners();
        this.setupSyntaxHighlighting();
    },

    /**
     * Attach event listeners
     */
    attachEventListeners: function() {
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.getElementById('uploadZone');
        const contractCode = document.getElementById('contractCode');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const clearBtn = document.getElementById('clearBtn');
        const confidenceThreshold = document.getElementById('confidenceThreshold');
        const confidenceValue = document.getElementById('confidenceValue');
        const analysisTimeout = document.getElementById('analysisTimeout');

        // File upload
        if (uploadZone) {
            uploadZone.addEventListener('click', () => fileInput.click());
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.style.backgroundColor = 'var(--color-primary)';
                uploadZone.style.color = 'white';
            });
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.style.backgroundColor = '';
                uploadZone.style.color = '';
            });
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.style.backgroundColor = '';
                uploadZone.style.color = '';
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }

        // Analyze button
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeContract());
        }

        // Clear button
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                if (contractCode) {
                    contractCode.value = '';
                    contractCode.focus();
                }
            });
        }

        // Confidence threshold slider
        if (confidenceThreshold) {
            confidenceThreshold.addEventListener('input', (e) => {
                if (confidenceValue) {
                    confidenceValue.textContent = `${e.target.value}%`;
                }
                StateManager.updateValue('analysisOptions.confidenceThreshold', parseInt(e.target.value));
            });
        }

        // Analysis timeout validation
        if (analysisTimeout) {
            analysisTimeout.addEventListener('change', (e) => {
                let value = parseInt(e.target.value);
                if (value < 5) value = 5;
                if (value > 300) value = 300;
                e.target.value = value;
                StateManager.updateValue('analysisOptions.timeout', value);
            });
        }

        // Model type change
        const modelType = document.getElementById('modelType');
        if (modelType) {
            modelType.addEventListener('change', (e) => {
                StateManager.updateValue('analysisOptions.modelType', e.target.value);
            });
        }

        // Report format change
        const reportFormat = document.getElementById('reportFormat');
        if (reportFormat) {
            reportFormat.addEventListener('change', (e) => {
                StateManager.updateValue('analysisOptions.reportFormat', e.target.value);
            });
        }

        // Checkboxes for options
        const checkboxes = ['aiAnalysis', 'patternAnalysis', 'externalTools', 'includeRecommendations', 'parallelAnalysis'];
        checkboxes.forEach(id => {
            const checkbox = document.getElementById(id);
            if (checkbox) {
                checkbox.addEventListener('change', (e) => {
                    StateManager.updateValue(`analysisOptions.${id}`, e.target.checked);
                });
            }
        });

        // Update analyze button state
        if (contractCode) {
            contractCode.addEventListener('input', () => {
                const code = contractCode.value.trim();
                analyzeBtn.disabled = code.length === 0;
            });
            analyzeBtn.disabled = contractCode.value.trim().length === 0;
        }
    },

    /**
     * Handle file upload
     */
    handleFileUpload: async function(file) {
        if (!Validators.isSolidityFile(file.name)) {
            App.showNotification('Only .sol files are allowed', 'error');
            return;
        }

        if (!Validators.isFileSizeValid(file, 10)) {
            App.showNotification('File size exceeds 10MB limit', 'error');
            return;
        }

        try {
            const content = await Helpers.readFileAsText(file);
            const contractCode = document.getElementById('contractCode');
            if (contractCode) {
                contractCode.value = content;
                contractCode.dispatchEvent(new Event('input'));
                App.showNotification(`Loaded ${file.name}`, 'success');
            }
        } catch (error) {
            App.showNotification(`Error reading file: ${error.message}`, 'error');
        }
    },

    /**
     * Setup syntax highlighting
     */
    setupSyntaxHighlighting: function() {
        const contractCode = document.getElementById('contractCode');
        if (contractCode && window.hljs) {
            contractCode.addEventListener('input', () => {
                // Simple syntax highlighting for textarea
                // In production, use a proper code editor like CodeMirror or Monaco
            });
        }
    },

    /**
     * Analyze contract
     */
    analyzeContract: async function() {
        const contractCode = document.getElementById('contractCode');
        const code = contractCode.value.trim();

        if (!Validators.isContractCodeValid(code)) {
            App.showNotification('Please enter valid contract code', 'error');
            return;
        }

        // Get analysis options from page
        const pageOptions = {
            aiAnalysis: document.getElementById('aiAnalysis').checked,
            patternAnalysis: document.getElementById('patternAnalysis').checked,
            externalTools: document.getElementById('externalTools').checked,
            modelType: document.getElementById('modelType').value,
            confidenceThreshold: parseInt(document.getElementById('confidenceThreshold').value),
            timeout: parseInt(document.getElementById('analysisTimeout').value),
            reportFormat: document.getElementById('reportFormat').value,
            includeRecommendations: document.getElementById('includeRecommendations').checked,
            parallelAnalysis: document.getElementById('parallelAnalysis').checked
        };

        // Get external tools selection from sidebar
        const sidebarOptions = {
            include_slither: document.getElementById('slitherTool')?.checked || false,
            include_mythril: document.getElementById('mythrilTool')?.checked || false,
            include_binary_model: document.getElementById('binaryModel')?.checked || false,
            include_multiclass_model: document.getElementById('multiclassModel')?.checked || false
        };

        // Merge options
        const options = {
            ...pageOptions,
            ...sidebarOptions
        };

        // Validate at least one analysis method is selected
        if (!options.aiAnalysis && !options.patternAnalysis && !options.externalTools) {
            App.showNotification('Please select at least one analysis method', 'error');
            return;
        }

        // Validate timeout
        if (options.timeout < 5 || options.timeout > 300) {
            App.showNotification('Analysis timeout must be between 5 and 300 seconds', 'error');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        const progressContainer = document.getElementById('progressContainer');

        analyzeBtn.disabled = true;
        if (progressContainer) {
            progressContainer.style.display = 'block';
        }

        EventBus.emit(Events.ANALYSIS_STARTED);

        try {
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 30;
                if (progress > 90) progress = 90;
                this.updateProgress(progress);
            }, 500);

            // Call API with options - pass options as second parameter
            const result = await APIService.analyzeContract(code, { options });

            clearInterval(progressInterval);
            this.updateProgress(100);

            if (result.success) {
                // Store result in state
                const analysis = {
                    id: Helpers.generateId(),
                    contractCode: code,
                    timestamp: new Date(),
                    riskScore: result.data.risk_score || 0,
                    vulnerabilities: result.data.vulnerabilities || [],
                    analysisTime: result.data.analysis_time || 0,
                    toolResults: result.data.tool_results || {},
                    toolsUsed: result.data.tools_used || [],
                    analysisMethod: result.data.analysis_method || 'Pattern Analysis',
                    analysisOptions: options
                };

                StateManager.setCurrentAnalysis(analysis);
                App.updateAnalysisStats();

                setTimeout(() => {
                    EventBus.emit(Events.ANALYSIS_COMPLETED, analysis);
                }, 500);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            App.showNotification(`Analysis error: ${error.message}`, 'error');
            EventBus.emit(Events.ANALYSIS_FAILED, error);
        } finally {
            analyzeBtn.disabled = false;
            if (progressContainer) {
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 1000);
            }
        }
    },

    /**
     * Update progress
     */
    updateProgress: function(percent) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        if (progressFill) {
            progressFill.style.width = `${percent}%`;
        }
        if (progressText) {
            progressText.textContent = `${Math.round(percent)}%`;
        }
    }
};

// Add upload zone styles
const uploadZoneStyles = `
.upload-zone {
    border: 2px dashed var(--color-border);
    border-radius: var(--radius-lg);
    padding: var(--spacing-2xl);
    text-align: center;
    cursor: pointer;
    transition: all var(--transition-fast);
    background-color: var(--color-bg-secondary);
}

.upload-zone:hover {
    border-color: var(--color-primary);
    background-color: var(--color-primary);
    color: white;
}

.upload-content {
    pointer-events: none;
}

.upload-content p {
    margin: 0;
    font-size: var(--font-size-lg);
}
`;

if (!document.querySelector('style[data-upload-zone-styles]')) {
    const style = document.createElement('style');
    style.setAttribute('data-upload-zone-styles', 'true');
    style.textContent = uploadZoneStyles;
    document.head.appendChild(style);
}
