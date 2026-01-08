/**
 * Comparison Page Component
 */

const ComparisonPage = {
    /**
     * Render comparison page
     */
    render: function() {
        const container = document.getElementById('pageContainer');
        const analysis = StateManager.getCurrentAnalysis();

        if (!analysis) {
            container.innerHTML = `
                <div class="page-container">
                    <div class="page-header">
                        <h1 class="page-title">⚖️ Tool Comparison</h1>
                    </div>
                    <div class="card">
                        <div class="alert alert-info">
                            <p>No analysis results available. Please analyze a contract first.</p>
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">⚖️ Tool Comparison</h1>
                    <p class="page-description">Compare results from different analysis tools</p>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">Analysis Tool Results</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderComparisonTable(analysis)}
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">Tools Used</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderToolsUsed(analysis)}
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">Vulnerability Summary</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderVulnerabilitySummary(analysis)}
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Render comparison table
     */
    renderComparisonTable: function(analysis) {
        const toolResults = analysis.toolResults || {};
        
        let html = '<table class="table"><thead><tr><th>Tool</th><th>Status</th><th>Findings</th><th>Execution Time</th></tr></thead><tbody>';

        // Slither
        if (toolResults.slither && typeof toolResults.slither === 'object') {
            const slither = toolResults.slither;
            html += `
                <tr>
                    <td><strong>🔍 Slither</strong></td>
                    <td>${slither.available ? '✅ Available' : '❌ Unavailable'}</td>
                    <td>${slither.findings || 0} vulnerabilities</td>
                    <td>${slither.execution_time ? slither.execution_time.toFixed(2) + 's' : 'N/A'}</td>
                </tr>
            `;
        }

        // Mythril
        if (toolResults.mythril && typeof toolResults.mythril === 'object') {
            const mythril = toolResults.mythril;
            html += `
                <tr>
                    <td><strong>⚡ Mythril</strong></td>
                    <td>${mythril.available ? '✅ Available' : '❌ Unavailable'}</td>
                    <td>${mythril.findings || 0} vulnerabilities</td>
                    <td>${mythril.execution_time ? mythril.execution_time.toFixed(2) + 's' : 'N/A'}</td>
                </tr>
            `;
        }

        // Binary Model
        if (toolResults.binary_model && typeof toolResults.binary_model === 'object') {
            const binary = toolResults.binary_model;
            const isVuln = binary.binary_prediction?.is_vulnerable ? 'Vulnerable' : 'Safe';
            const confidence = binary.binary_prediction?.confidence ? (binary.binary_prediction.confidence * 100).toFixed(1) : 'N/A';
            html += `
                <tr>
                    <td><strong>🤖 Binary Classifier</strong></td>
                    <td>${isVuln}</td>
                    <td>Confidence: ${confidence}%</td>
                    <td>N/A</td>
                </tr>
            `;
        }

        // Multiclass Model
        if (toolResults.multiclass_model && typeof toolResults.multiclass_model === 'object') {
            const multiclass = toolResults.multiclass_model;
            const vulnType = multiclass.multiclass_prediction?.vulnerability_type || 'Unknown';
            const confidence = multiclass.multiclass_prediction?.confidence ? (multiclass.multiclass_prediction.confidence * 100).toFixed(1) : 'N/A';
            html += `
                <tr>
                    <td><strong>🤖 Multi-class Classifier</strong></td>
                    <td>${vulnType}</td>
                    <td>Confidence: ${confidence}%</td>
                    <td>N/A</td>
                </tr>
            `;
        }

        html += '</tbody></table>';
        return html;
    },

    /**
     * Render tools used
     */
    renderToolsUsed: function(analysis) {
        let toolsUsed = analysis.toolsUsed || [];
        const analysisMethod = analysis.analysisMethod || 'Unknown';

        // Ensure toolsUsed is an array
        if (!Array.isArray(toolsUsed)) {
            if (typeof toolsUsed === 'object' && toolsUsed !== null) {
                toolsUsed = Object.values(toolsUsed);
            } else {
                toolsUsed = [];
            }
        }

        let html = `
            <div style="display: flex; flex-direction: column; gap: var(--spacing-md);">
                <div>
                    <h4>Analysis Method</h4>
                    <p>${Helpers.escapeHtml(analysisMethod)}</p>
                </div>
                <div>
                    <h4>Tools Used (${toolsUsed.length})</h4>
                    <ul style="margin-top: var(--spacing-md);">
        `;

        toolsUsed.forEach(tool => {
            html += `<li>✓ ${Helpers.escapeHtml(tool)}</li>`;
        });

        html += `
                    </ul>
                </div>
            </div>
        `;

        return html;
    },

    /**
     * Render vulnerability summary
     */
    renderVulnerabilitySummary: function(analysis) {
        let vulnerabilities = analysis.vulnerabilities || [];
        
        // Ensure vulnerabilities is an array
        if (!Array.isArray(vulnerabilities)) {
            if (typeof vulnerabilities === 'object' && vulnerabilities !== null) {
                // If it's an object, try to extract values
                vulnerabilities = Object.values(vulnerabilities);
            } else {
                vulnerabilities = [];
            }
        }
        
        if (vulnerabilities.length === 0) {
            return '<p class="text-muted">No vulnerabilities detected by any tool.</p>';
        }

        let html = '<div style="display: flex; flex-direction: column; gap: var(--spacing-md);">';

        vulnerabilities.forEach((vuln, index) => {
            const severity = vuln.severity || 'Medium';
            const severityColor = severity === 'Critical' ? '#e74c3c' : severity === 'High' ? '#e67e22' : severity === 'Medium' ? '#f39c12' : '#27ae60';
            
            html += `
                <div style="border-left: 4px solid ${severityColor}; padding-left: var(--spacing-md); padding-top: var(--spacing-sm); padding-bottom: var(--spacing-sm);">
                    <h5 style="margin: 0 0 var(--spacing-sm) 0;">${index + 1}. ${Helpers.escapeHtml(vuln.type || 'Unknown')} <span style="color: ${severityColor}; font-weight: bold;">${severity}</span></h5>
                    <p style="margin: 0 0 var(--spacing-sm) 0; color: var(--color-text-secondary);">${Helpers.escapeHtml(vuln.description || 'No description')}</p>
                    <p style="margin: 0; font-size: 0.9em; color: var(--color-text-secondary);">
                        <strong>Confidence:</strong> ${vuln.confidence ? (vuln.confidence * 100).toFixed(1) : 'N/A'}% | 
                        <strong>Source:</strong> ${Helpers.escapeHtml(vuln.source || 'Unknown')}
                    </p>
                </div>
            `;
        });

        html += '</div>';
        return html;
    }
};
