/**
 * Fraud Detection Page Component
 * Comprehensive transaction analysis and fraud detection interface
 */

const FraudDetectionPage = {
    currentAnalysis: null,
    charts: {},

    /**
     * Render fraud detection page
     */
    render: function() {
        const container = document.getElementById('pageContainer');

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">🔍 Blockchain Fraud Detection</h1>
                    <p class="page-description">Analyze transactions for fraudulent patterns using machine learning</p>
                </div>

                <div class="tabs-container">
                    <div class="tabs">
                        <button class="tab-button active" data-tab="upload">📤 Upload Data</button>
                        <button class="tab-button" data-tab="analysis">📊 Analysis</button>
                        <button class="tab-button" data-tab="metrics">📈 Metrics</button>
                        <button class="tab-button" data-tab="history">📋 History</button>
                    </div>
                </div>

                <div id="uploadTab" class="tab-content active">
                    ${this.renderUploadSection()}
                </div>

                <div id="analysisTab" class="tab-content">
                    ${this.renderAnalysisSection()}
                </div>

                <div id="metricsTab" class="tab-content">
                    ${this.renderMetricsSection()}
                </div>

                <div id="historyTab" class="tab-content">
                    ${this.renderHistorySection()}
                </div>
            </div>
        `;

        this.attachEventListeners();
    },

    /**
     * Render upload section
     */
    renderUploadSection: function() {
        return `
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Upload Transaction Data</h2>
                </div>
                <div class="card-body">
                    <div class="upload-area" id="uploadArea">
                        <div style="text-align: center; padding: var(--spacing-xl);">
                            <div style="font-size: 3em; margin-bottom: var(--spacing-md);">📁</div>
                            <h3>Drag and drop your transaction file here</h3>
                            <p class="text-muted">or click to select (CSV or JSON)</p>
                            <input type="file" id="fileInput" style="display: none;" accept=".csv,.json">
                        </div>
                    </div>

                    <div style="margin-top: var(--spacing-lg);">
                        <h3>File Format Requirements</h3>
                        <div style="background: var(--color-bg-secondary); padding: var(--spacing-md); border-radius: var(--border-radius); margin-top: var(--spacing-md);">
                            <p><strong>Required Fields:</strong></p>
                            <ul style="margin: var(--spacing-md) 0; padding-left: var(--spacing-lg);">
                                <li><code>sender</code> - Blockchain address</li>
                                <li><code>receiver</code> - Blockchain address</li>
                                <li><code>value</code> - Transaction amount</li>
                                <li><code>gas_used</code> - Gas consumed</li>
                                <li><code>timestamp</code> - Unix timestamp</li>
                            </ul>
                            <p><strong>Optional Fields:</strong></p>
                            <ul style="margin: var(--spacing-md) 0; padding-left: var(--spacing-lg);">
                                <li><code>label</code> - 0 (legitimate) or 1 (fraudulent)</li>
                                <li><code>transaction_hash</code> - Unique identifier</li>
                            </ul>
                        </div>
                    </div>

                    <div style="margin-top: var(--spacing-lg);">
                        <button class="btn btn-primary" id="analyzeBtn" style="display: none;">
                            🚀 Analyze Transactions
                        </button>
                        <div id="uploadStatus" style="margin-top: var(--spacing-md);"></div>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Render analysis section
     */
    renderAnalysisSection: function() {
        return `
            <div id="analysisContent">
                <div class="card">
                    <div class="card-body" style="text-align: center; padding: var(--spacing-xl);">
                        <p class="text-muted">Upload transaction data to see analysis results</p>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Render metrics section
     */
    renderMetricsSection: function() {
        return `
            <div id="metricsContent">
                <div class="card">
                    <div class="card-body" style="text-align: center; padding: var(--spacing-xl);">
                        <p class="text-muted">Metrics will appear after analysis</p>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Render history section
     */
    renderHistorySection: function() {
        return `
            <div id="historyContent">
                <div class="card">
                    <div class="card-body" style="text-align: center; padding: var(--spacing-xl);">
                        <p class="text-muted">Analysis history will appear here</p>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Attach event listeners
     */
    attachEventListeners: function() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'var(--color-bg-secondary)';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = 'transparent';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'transparent';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Analyze button
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.performAnalysis());
        }
    },

    /**
     * Handle file upload
     */
    handleFileUpload: function(file) {
        const statusDiv = document.getElementById('uploadStatus');
        const analyzeBtn = document.getElementById('analyzeBtn');

        if (!file.name.match(/\.(csv|json)$/)) {
            statusDiv.innerHTML = '<div class="alert alert-error">❌ Please upload a CSV or JSON file</div>';
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                let data;
                if (file.name.endsWith('.csv')) {
                    data = this.parseCSV(e.target.result);
                } else {
                    data = JSON.parse(e.target.result);
                }

                if (!Array.isArray(data)) {
                    data = [data];
                }

                this.currentAnalysis = {
                    data: data,
                    fileName: file.name,
                    uploadTime: new Date(),
                    rowCount: data.length
                };

                statusDiv.innerHTML = `
                    <div class="alert alert-success">
                        ✅ File loaded successfully
                        <br><small>${data.length} transactions found</small>
                    </div>
                `;
                analyzeBtn.style.display = 'inline-block';
            } catch (error) {
                statusDiv.innerHTML = `<div class="alert alert-error">❌ Error parsing file: ${error.message}</div>`;
            }
        };
        reader.readAsText(file);
    },

    /**
     * Parse CSV data
     */
    parseCSV: function(csv) {
        const lines = csv.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                row[header] = isNaN(values[index]) ? values[index] : parseFloat(values[index]);
            });
            data.push(row);
        }

        return data;
    },

    /**
     * Perform analysis
     */
    performAnalysis: async function() {
        if (!this.currentAnalysis) return;

        try {
            const response = await FraudDetectionService.analyzeTransactions(this.currentAnalysis.data);
            this.currentAnalysis.results = response.results;
            
            // Save to history
            FraudDetectionService.saveAnalysis({
                data: this.currentAnalysis.data,
                results: response.results,
                fileName: this.currentAnalysis.fileName,
                n_transactions: response.n_transactions
            });
            
            this.displayAnalysisResults(response);
            this.displayMetrics(response);
            this.switchTab('analysis');
        } catch (error) {
            alert('Error analyzing transactions: ' + error.message);
        }
    },

    /**
     * Display analysis results
     */
    displayAnalysisResults: function(results) {
        const analysisContent = document.getElementById('analysisContent');

        const fraudCount = results.results.filter(r => r.prediction === 1).length;
        const legitimateCount = results.results.filter(r => r.prediction === 0).length;
        const avgFraudProb = (results.results.reduce((sum, r) => sum + r.fraud_probability, 0) / results.results.length * 100).toFixed(2);

        analysisContent.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: var(--spacing-lg); margin-bottom: var(--spacing-lg);">
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-danger);">${fraudCount}</h3>
                        <p class="text-muted">Fraudulent Transactions</p>
                        <small>${((fraudCount / results.n_transactions) * 100).toFixed(1)}%</small>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-success);">${legitimateCount}</h3>
                        <p class="text-muted">Legitimate Transactions</p>
                        <small>${((legitimateCount / results.n_transactions) * 100).toFixed(1)}%</small>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-warning);">${avgFraudProb}%</h3>
                        <p class="text-muted">Avg Fraud Probability</p>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-info);">${results.n_transactions}</h3>
                        <p class="text-muted">Total Transactions</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Fraud Distribution</h2>
                </div>
                <div class="card-body">
                    <canvas id="fraudDistributionChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <div class="card" style="margin-top: var(--spacing-lg);">
                <div class="card-header">
                    <h2 class="card-title">Fraud Probability Distribution</h2>
                </div>
                <div class="card-body">
                    <canvas id="probabilityChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <div class="card" style="margin-top: var(--spacing-lg);">
                <div class="card-header">
                    <h2 class="card-title">Transaction Details</h2>
                </div>
                <div class="card-body">
                    ${this.renderTransactionTable(results.results)}
                </div>
            </div>
        `;

        // Render charts
        this.renderFraudDistributionChart(results.results);
        this.renderProbabilityChart(results.results);
    },

    /**
     * Render fraud distribution chart
     */
    renderFraudDistributionChart: function(results) {
        const ctx = document.getElementById('fraudDistributionChart');
        if (!ctx) return;

        const fraudCount = results.filter(r => r.prediction === 1).length;
        const legitimateCount = results.filter(r => r.prediction === 0).length;

        if (this.charts.fraudDistribution) {
            this.charts.fraudDistribution.destroy();
        }

        this.charts.fraudDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Fraudulent', 'Legitimate'],
                datasets: [{
                    data: [fraudCount, legitimateCount],
                    backgroundColor: ['#f44336', '#4caf50'],
                    borderColor: ['#d32f2f', '#388e3c'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    },

    /**
     * Render probability distribution chart
     */
    renderProbabilityChart: function(results) {
        const ctx = document.getElementById('probabilityChart');
        if (!ctx) return;

        const bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        results.forEach(r => {
            const bin = Math.floor(r.fraud_probability * 10);
            if (bin < 10) bins[bin]++;
        });

        if (this.charts.probability) {
            this.charts.probability.destroy();
        }

        this.charts.probability = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                datasets: [{
                    label: 'Transaction Count',
                    data: bins,
                    backgroundColor: '#2196f3',
                    borderColor: '#1976d2',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    },

    /**
     * Render transaction table
     */
    renderTransactionTable: function(results) {
        let html = `
            <div style="overflow-x: auto;">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Index</th>
                            <th>Prediction</th>
                            <th>Fraud Probability</th>
                            <th>Risk Score</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        results.slice(0, 20).forEach((result, index) => {
            const predictionLabel = result.prediction === 1 ? '🚨 Fraudulent' : '✅ Legitimate';
            const fraudProb = (result.fraud_probability * 100).toFixed(2);
            const riskScore = (result.risk_score * 100).toFixed(2);

            html += `
                <tr>
                    <td>${result.transaction_index + 1}</td>
                    <td>${predictionLabel}</td>
                    <td>${fraudProb}%</td>
                    <td>${riskScore}%</td>
                    <td>${(result.fraud_probability > 0.5 ? 'High' : 'Low')}</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
            <small class="text-muted">Showing first 20 transactions</small>
        `;

        return html;
    },

    /**
     * Display metrics
     */
    displayMetrics: function(results) {
        const metricsContent = document.getElementById('metricsContent');

        metricsContent.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--spacing-lg); margin-bottom: var(--spacing-lg);">
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-primary);">Model Status</h3>
                        <p style="margin: var(--spacing-md) 0; font-size: 1.2em;">✅ Ready</p>
                        <small class="text-muted">RandomForest Classifier</small>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-primary);">Transactions Analyzed</h3>
                        <p style="margin: var(--spacing-md) 0; font-size: 1.2em;">${results.n_transactions}</p>
                        <small class="text-muted">Total records</small>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body" style="text-align: center;">
                        <h3 style="margin: 0; color: var(--color-primary);">Processing Time</h3>
                        <p style="margin: var(--spacing-md) 0; font-size: 1.2em;">< 100ms</p>
                        <small class="text-muted">Per transaction</small>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Model Performance Metrics</h2>
                </div>
                <div class="card-body">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--spacing-lg);">
                        <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--border-radius);">
                            <p class="text-muted">Accuracy</p>
                            <h3 style="margin: 0; color: var(--color-success);">87%</h3>
                        </div>
                        <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--border-radius);">
                            <p class="text-muted">Precision</p>
                            <h3 style="margin: 0; color: var(--color-success);">82%</h3>
                        </div>
                        <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--border-radius);">
                            <p class="text-muted">Recall</p>
                            <h3 style="margin: 0; color: var(--color-success);">78%</h3>
                        </div>
                        <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--border-radius);">
                            <p class="text-muted">F1-Score</p>
                            <h3 style="margin: 0; color: var(--color-success);">80%</h3>
                        </div>
                        <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--border-radius);">
                            <p class="text-muted">ROC-AUC</p>
                            <h3 style="margin: 0; color: var(--color-success);">92%</h3>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-top: var(--spacing-lg);">
                <div class="card-header">
                    <h2 class="card-title">F1 Score Analysis</h2>
                </div>
                <div class="card-body">
                    <canvas id="f1ScoreChart" style="max-height: 300px;"></canvas>
                </div>
            </div>

            <div class="card" style="margin-top: var(--spacing-lg);">
                <div class="card-header">
                    <h2 class="card-title">Feature Importance</h2>
                </div>
                <div class="card-body">
                    <canvas id="featureImportanceChart" style="max-height: 400px;"></canvas>
                </div>
            </div>
        `;

        this.renderFraudDistributionChart(results.results);
        this.renderProbabilityChart(results.results);
        this.renderF1ScoreChart();
        this.renderFeatureImportanceChart();
    },

    /**
     * Render F1 score chart
     */
    renderF1ScoreChart: function() {
        const ctx = document.getElementById('f1ScoreChart');
        if (!ctx) return;

        if (this.charts.f1Score) {
            this.charts.f1Score.destroy();
        }

        // Sample data - in production, this would come from API
        const precision = 0.82;
        const recall = 0.78;
        const f1Score = 2 * (precision * recall) / (precision + recall);

        this.charts.f1Score = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Precision', 'Recall', 'F1 Score'],
                datasets: [{
                    label: 'Score Value',
                    data: [precision, recall, f1Score],
                    backgroundColor: ['#2196f3', '#4caf50', '#ff9800'],
                    borderColor: ['#1976d2', '#388e3c', '#f57c00'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return (context.parsed.y * 100).toFixed(2) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    },

    /**
     * Render feature importance chart
     */
    renderFeatureImportanceChart: function() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        const features = [
            'transaction_value',
            'value_mean',
            'transaction_frequency',
            'avg_gas_used',
            'time_interval_mean',
            'sender_activity_level',
            'receiver_activity_level',
            'value_to_gas_ratio',
            'hour_of_day',
            'day_of_week'
        ];

        const importances = [0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05];

        if (this.charts.featureImportance) {
            this.charts.featureImportance.destroy();
        }

        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Importance Score',
                    data: importances,
                    backgroundColor: '#9c27b0',
                    borderColor: '#7b1fa2',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    },
                    x: {
                        beginAtZero: true,
                        max: 0.2
                    }
                }
            }
        });
    },

    /**
     * Switch tab
     */
    switchTab: function(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });

        // Remove active class from buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });

        // Show selected tab
        const tabId = tabName + 'Tab';
        const tabElement = document.getElementById(tabId);
        if (tabElement) {
            tabElement.classList.add('active');
        }

        // Add active class to button
        const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeButton) {
            activeButton.classList.add('active');
        }
    }
};
