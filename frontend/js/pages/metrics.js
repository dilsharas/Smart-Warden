/**
 * Metrics Page Component
 */

const MetricsPage = {
    /**
     * Render metrics page
     */
    render: function() {
        const container = document.getElementById('pageContainer');
        const history = StateManager.getAnalysisHistory();

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">📈 Performance Metrics</h1>
                    <p class="page-description">Analysis performance and accuracy metrics</p>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: var(--spacing-lg); margin-bottom: var(--spacing-lg);">
                    <div class="card">
                        <div class="card-body" style="text-align: center;">
                            <h3 style="margin: 0; color: var(--color-primary);">${history.length}</h3>
                            <p class="text-muted">Total Analyses</p>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-body" style="text-align: center;">
                            <h3 style="margin: 0; color: var(--color-warning);">${history.filter(a => a.riskScore >= 60).length}</h3>
                            <p class="text-muted">High Risk Contracts</p>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-body" style="text-align: center;">
                            <h3 style="margin: 0; color: var(--color-success);">${history.filter(a => a.riskScore < 40).length}</h3>
                            <p class="text-muted">Safe Contracts</p>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-body" style="text-align: center;">
                            <h3 style="margin: 0; color: var(--color-info);">${history.length > 0 ? (history.reduce((sum, a) => sum + a.analysisTime, 0) / history.length / 1000).toFixed(2) : 0}s</h3>
                            <p class="text-muted">Avg Analysis Time</p>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">Risk Score Distribution</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderRiskDistribution(history)}
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">Analysis History</h2>
                    </div>
                    <div class="card-body">
                        ${this.renderAnalysisHistory(history)}
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Render risk distribution
     */
    renderRiskDistribution: function(history) {
        if (history.length === 0) {
            return '<p class="text-muted">No analysis data available</p>';
        }

        const critical = history.filter(a => a.riskScore >= 80).length;
        const high = history.filter(a => a.riskScore >= 60 && a.riskScore < 80).length;
        const medium = history.filter(a => a.riskScore >= 40 && a.riskScore < 60).length;
        const low = history.filter(a => a.riskScore < 40).length;

        const total = history.length;

        return `
            <div style="display: flex; flex-direction: column; gap: var(--spacing-lg);">
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                        <span>Critical (80-100)</span>
                        <strong>${critical} (${((critical/total)*100).toFixed(1)}%)</strong>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${(critical/total)*100}%; background-color: #f44336;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                        <span>High (60-79)</span>
                        <strong>${high} (${((high/total)*100).toFixed(1)}%)</strong>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${(high/total)*100}%; background-color: #ff9800;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                        <span>Medium (40-59)</span>
                        <strong>${medium} (${((medium/total)*100).toFixed(1)}%)</strong>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${(medium/total)*100}%; background-color: #9c27b0;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--spacing-sm);">
                        <span>Low (0-39)</span>
                        <strong>${low} (${((low/total)*100).toFixed(1)}%)</strong>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${(low/total)*100}%; background-color: #4caf50;"></div>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Render analysis history
     */
    renderAnalysisHistory: function(history) {
        if (history.length === 0) {
            return '<p class="text-muted">No analysis history available</p>';
        }

        let html = '<table class="table"><thead><tr><th>Time</th><th>Risk Score</th><th>Vulnerabilities</th><th>Analysis Time</th></tr></thead><tbody>';

        history.slice(0, 10).forEach(analysis => {
            const riskInfo = Formatters.formatRiskScore(analysis.riskScore);
            html += `
                <tr>
                    <td>${Formatters.formatRelativeTime(analysis.timestamp)}</td>
                    <td><span style="color: ${riskInfo.color}; font-weight: bold;">${analysis.riskScore}</span></td>
                    <td>${analysis.vulnerabilities.length}</td>
                    <td>${Formatters.formatAnalysisTime(analysis.analysisTime)}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        return html;
    }
};
