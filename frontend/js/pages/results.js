/**
 * Results Page Component
 */

const ResultsPage = {
    /**
     * Render results page
     */
    render: function() {
        const container = document.getElementById('pageContainer');
        const analysis = StateManager.getCurrentAnalysis();

        if (!analysis) {
            container.innerHTML = `
                <div class="page-container">
                    <div class="page-header">
                        <h1 class="page-title">📊 Results</h1>
                    </div>
                    <div class="card">
                        <div class="alert alert-info">
                            <p>No analysis results available. Please analyze a contract first.</p>
                        </div>
                        <button class="btn btn-primary" onclick="Router.navigate('analyze')">
                            Go to Analyze
                        </button>
                    </div>
                </div>
            `;
            return;
        }

        // Ensure vulnerabilities is an array
        let vulnerabilities = analysis.vulnerabilities || [];
        if (!Array.isArray(vulnerabilities)) {
            if (typeof vulnerabilities === 'object' && vulnerabilities !== null) {
                vulnerabilities = Object.values(vulnerabilities);
            } else {
                vulnerabilities = [];
            }
        }

        const riskInfo = Formatters.formatRiskScore(analysis.riskScore);

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">📊 Analysis Results</h1>
                    <p class="page-description">Analysis completed in ${Formatters.formatAnalysisTime(analysis.analysisTime)}</p>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--spacing-lg); margin-bottom: var(--spacing-lg);">
                    <div class="card">
                        <div class="card-body" style="text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: var(--spacing-md);">
                                <svg width="120" height="120" viewBox="0 0 120 120" style="margin: 0 auto; display: block;">
                                    <circle cx="60" cy="60" r="50" fill="none" stroke="${riskInfo.color}" stroke-width="8" opacity="0.2"/>
                                    <circle cx="60" cy="60" r="50" fill="none" stroke="${riskInfo.color}" stroke-width="8" stroke-dasharray="${(analysis.riskScore / 100) * 314} 314" transform="rotate(-90 60 60)"/>
                                    <text x="60" y="65" text-anchor="middle" font-size="32" font-weight="bold" fill="${riskInfo.color}">${analysis.riskScore}</text>
                                </svg>
                            </div>
                            <h3 style="margin: 0; color: ${riskInfo.color};">${riskInfo.level}</h3>
                            <p class="text-muted">Risk Score</p>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-body">
                            <h3>Summary</h3>
                            <div style="display: flex; flex-direction: column; gap: var(--spacing-md);">
                                <div>
                                    <span class="text-muted">Total Vulnerabilities:</span>
                                    <strong>${vulnerabilities.length}</strong>
                                </div>
                                <div>
                                    <span class="text-muted">Critical:</span>
                                    <strong>${vulnerabilities.filter(v => v.severity === 'Critical' || v.severity === 'critical').length}</strong>
                                </div>
                                <div>
                                    <span class="text-muted">High:</span>
                                    <strong>${vulnerabilities.filter(v => v.severity === 'High' || v.severity === 'high').length}</strong>
                                </div>
                                <div>
                                    <span class="text-muted">Medium:</span>
                                    <strong>${vulnerabilities.filter(v => v.severity === 'Medium' || v.severity === 'medium').length}</strong>
                                </div>
                                <div>
                                    <span class="text-muted">Low:</span>
                                    <strong>${vulnerabilities.filter(v => v.severity === 'Low' || v.severity === 'low').length}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">Vulnerabilities</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderVulnerabilitiesTable(vulnerabilities)}
                    </div>
                </div>

                <div style="display: flex; gap: var(--spacing-md); margin-top: var(--spacing-lg);">
                    <button class="btn btn-primary" onclick="ResultsPage.exportResults('json')">
                        📥 Export JSON
                    </button>
                    <button class="btn btn-secondary" onclick="ResultsPage.exportResults('csv')">
                        📥 Export CSV
                    </button>
                    <button class="btn btn-secondary" onclick="Router.navigate('analyze')">
                        🔄 Analyze Another
                    </button>
                </div>
            </div>
        `;

        this.attachEventListeners();
    },

    /**
     * Render vulnerabilities table
     */
    renderVulnerabilitiesTable: function(vulnerabilities) {
        // Ensure vulnerabilities is an array
        if (!Array.isArray(vulnerabilities)) {
            if (typeof vulnerabilities === 'object' && vulnerabilities !== null) {
                vulnerabilities = Object.values(vulnerabilities);
            } else {
                vulnerabilities = [];
            }
        }

        if (vulnerabilities.length === 0) {
            return '<div class="alert alert-success"><p>✅ No vulnerabilities detected!</p></div>';
        }

        let html = '<table class="table"><thead><tr><th>Type</th><th>Severity</th><th>Confidence</th><th>Line</th><th>Description</th></tr></thead><tbody>';

        vulnerabilities.forEach(vuln => {
            const severity = Formatters.formatSeverity(vuln.severity);
            html += `
                <tr class="severity-${vuln.severity}">
                    <td><strong>${Formatters.formatVulnerabilityType(vuln.type)}</strong></td>
                    <td><span class="badge badge-${vuln.severity}">${severity.label}</span></td>
                    <td>${Formatters.formatConfidence(vuln.confidence)}</td>
                    <td>${vuln.lineNumber || 'N/A'}</td>
                    <td>${Helpers.truncate(vuln.description, 50)}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        return html;
    },

    /**
     * Export results
     */
    exportResults: function(format) {
        const analysis = StateManager.getCurrentAnalysis();
        if (!analysis) return;

        // Ensure vulnerabilities is an array
        let vulnerabilities = analysis.vulnerabilities || [];
        if (!Array.isArray(vulnerabilities)) {
            if (typeof vulnerabilities === 'object' && vulnerabilities !== null) {
                vulnerabilities = Object.values(vulnerabilities);
            } else {
                vulnerabilities = [];
            }
        }

        let content, filename, mimeType;

        if (format === 'json') {
            content = Formatters.formatJson(analysis);
            filename = `analysis-${Helpers.generateId()}.json`;
            mimeType = 'application/json';
        } else if (format === 'csv') {
            const data = vulnerabilities.map(v => ({
                Type: v.type,
                Severity: v.severity,
                Confidence: v.confidence,
                Line: v.lineNumber,
                Description: v.description
            }));
            content = Formatters.formatCsv(data);
            filename = `analysis-${Helpers.generateId()}.csv`;
            mimeType = 'text/csv';
        }

        if (content) {
            Helpers.downloadFile(content, filename, mimeType);
            App.showNotification(`Exported as ${format.toUpperCase()}`, 'success');
        }
    },

    /**
     * Attach event listeners
     */
    attachEventListeners: function() {
        // Event listeners attached inline in render
    }
};
