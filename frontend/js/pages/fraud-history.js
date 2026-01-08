/**
 * Fraud Detection History Page
 * View and manage past fraud detection analyses
 */

const FraudHistoryPage = {
    currentPage: 1,
    itemsPerPage: 10,

    /**
     * Render fraud history page
     */
    render: function() {
        const container = document.getElementById('pageContainer');
        const history = FraudDetectionService.getAnalysisHistory();

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">📋 Analysis History</h1>
                    <p class="page-description">View and manage your fraud detection analysis history</p>
                </div>

                <div style="display: flex; gap: var(--spacing-md); margin-bottom: var(--spacing-lg);">
                    <button class="btn btn-primary" id="exportAllBtn">📥 Export All</button>
                    <button class="btn btn-secondary" id="clearAllBtn">🗑️ Clear All</button>
                </div>

                ${history.length === 0 ? this.renderEmptyState() : this.renderHistoryTable(history)}
            </div>
        `;

        this.attachEventListeners();
    },

    /**
     * Render empty state
     */
    renderEmptyState: function() {
        return `
            <div class="card">
                <div class="card-body" style="text-align: center; padding: var(--spacing-2xl);">
                    <div style="font-size: 3em; margin-bottom: var(--spacing-md);">📭</div>
                    <h2>No Analysis History</h2>
                    <p class="text-muted">Start by analyzing transaction data to build your history</p>
                    <button class="btn btn-primary" onclick="Router.navigate('fraud-detection')" style="margin-top: var(--spacing-lg);">
                        🚀 Start Analysis
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Render history table
     */
    renderHistoryTable: function(history) {
        const startIdx = (this.currentPage - 1) * this.itemsPerPage;
        const endIdx = startIdx + this.itemsPerPage;
        const paginatedHistory = history.slice(startIdx, endIdx);
        const totalPages = Math.ceil(history.length / this.itemsPerPage);

        let html = `
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Analysis Records</h2>
                    <span class="text-muted">${history.length} total analyses</span>
                </div>
                <div class="card-body">
                    <div style="overflow-x: auto;">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Date & Time</th>
                                    <th>Transactions</th>
                                    <th>Fraudulent</th>
                                    <th>Legitimate</th>
                                    <th>Fraud Rate</th>
                                    <th>Avg Probability</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
        `;

        paginatedHistory.forEach((analysis, index) => {
            const timestamp = new Date(analysis.timestamp);
            const dateStr = timestamp.toLocaleDateString();
            const timeStr = timestamp.toLocaleTimeString();
            
            const fraudCount = analysis.results ? analysis.results.filter(r => r.prediction === 1).length : 0;
            const legitimateCount = analysis.results ? analysis.results.filter(r => r.prediction === 0).length : 0;
            const totalCount = fraudCount + legitimateCount;
            const fraudRate = totalCount > 0 ? ((fraudCount / totalCount) * 100).toFixed(1) : 0;
            
            const avgProb = analysis.results ? 
                (analysis.results.reduce((sum, r) => sum + r.fraud_probability, 0) / analysis.results.length * 100).toFixed(1) : 0;

            html += `
                <tr>
                    <td>
                        <div>${dateStr}</div>
                        <small class="text-muted">${timeStr}</small>
                    </td>
                    <td>${totalCount}</td>
                    <td><span style="color: var(--color-danger); font-weight: 600;">${fraudCount}</span></td>
                    <td><span style="color: var(--color-success); font-weight: 600;">${legitimateCount}</span></td>
                    <td>${fraudRate}%</td>
                    <td>${avgProb}%</td>
                    <td>
                        <button class="btn btn-secondary" style="padding: var(--spacing-sm) var(--spacing-md); font-size: 0.875rem;" onclick="FraudHistoryPage.viewDetails(${analysis.id})">View</button>
                        <button class="btn btn-secondary" style="padding: var(--spacing-sm) var(--spacing-md); font-size: 0.875rem;" onclick="FraudHistoryPage.exportAnalysis(${analysis.id})">Export</button>
                        <button class="btn btn-danger" style="padding: var(--spacing-sm) var(--spacing-md); font-size: 0.875rem;" onclick="FraudHistoryPage.deleteAnalysis(${analysis.id})">Delete</button>
                    </td>
                </tr>
            `;
        });

        html += `
                            </tbody>
                        </table>
                    </div>

                    ${totalPages > 1 ? this.renderPagination(totalPages) : ''}
                </div>
            </div>

            <div class="card" style="margin-top: var(--spacing-lg);">
                <div class="card-header">
                    <h2 class="card-title">Summary Statistics</h2>
                </div>
                <div class="card-body">
                    ${this.renderSummaryStats(history)}
                </div>
            </div>
        `;

        return html;
    },

    /**
     * Render pagination
     */
    renderPagination: function(totalPages) {
        let html = `
            <div style="display: flex; justify-content: center; gap: var(--spacing-md); margin-top: var(--spacing-lg);">
        `;

        for (let i = 1; i <= totalPages; i++) {
            const isActive = i === this.currentPage;
            html += `
                <button class="btn ${isActive ? 'btn-primary' : 'btn-secondary'}" 
                        onclick="FraudHistoryPage.goToPage(${i})"
                        style="min-width: 40px;">
                    ${i}
                </button>
            `;
        }

        html += `</div>`;
        return html;
    },

    /**
     * Render summary statistics
     */
    renderSummaryStats: function(history) {
        const totalAnalyses = history.length;
        const totalTransactions = history.reduce((sum, a) => sum + (a.results ? a.results.length : 0), 0);
        const totalFraudulent = history.reduce((sum, a) => sum + (a.results ? a.results.filter(r => r.prediction === 1).length : 0), 0);
        const totalLegitimate = totalTransactions - totalFraudulent;
        const avgFraudRate = totalTransactions > 0 ? ((totalFraudulent / totalTransactions) * 100).toFixed(1) : 0;

        return `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--spacing-lg);">
                <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--radius-md);">
                    <p class="text-muted">Total Analyses</p>
                    <h3 style="margin: 0; color: var(--color-primary);">${totalAnalyses}</h3>
                </div>
                <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--radius-md);">
                    <p class="text-muted">Total Transactions</p>
                    <h3 style="margin: 0; color: var(--color-primary);">${totalTransactions}</h3>
                </div>
                <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--radius-md);">
                    <p class="text-muted">Fraudulent Detected</p>
                    <h3 style="margin: 0; color: var(--color-danger);">${totalFraudulent}</h3>
                </div>
                <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--radius-md);">
                    <p class="text-muted">Legitimate Confirmed</p>
                    <h3 style="margin: 0; color: var(--color-success);">${totalLegitimate}</h3>
                </div>
                <div style="text-align: center; padding: var(--spacing-md); background: var(--color-bg-secondary); border-radius: var(--radius-md);">
                    <p class="text-muted">Average Fraud Rate</p>
                    <h3 style="margin: 0; color: var(--color-warning);">${avgFraudRate}%</h3>
                </div>
            </div>
        `;
    },

    /**
     * Attach event listeners
     */
    attachEventListeners: function() {
        const exportAllBtn = document.getElementById('exportAllBtn');
        const clearAllBtn = document.getElementById('clearAllBtn');

        if (exportAllBtn) {
            exportAllBtn.addEventListener('click', () => this.exportAllAnalyses());
        }

        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => this.clearAllAnalyses());
        }
    },

    /**
     * Go to page
     */
    goToPage: function(page) {
        this.currentPage = page;
        this.render();
    },

    /**
     * View analysis details
     */
    viewDetails: function(analysisId) {
        const history = FraudDetectionService.getAnalysisHistory();
        const analysis = history.find(a => a.id === analysisId);

        if (!analysis) {
            alert('Analysis not found');
            return;
        }

        // Store in state and navigate to fraud detection page
        StateManager.updateValue('selectedAnalysis', analysis);
        Router.navigate('fraud-detection');
    },

    /**
     * Export single analysis
     */
    exportAnalysis: function(analysisId) {
        const history = FraudDetectionService.getAnalysisHistory();
        const analysis = history.find(a => a.id === analysisId);

        if (!analysis) {
            alert('Analysis not found');
            return;
        }

        FraudDetectionService.exportResults(analysis, 'json');
    },

    /**
     * Delete analysis
     */
    deleteAnalysis: function(analysisId) {
        if (!confirm('Are you sure you want to delete this analysis?')) {
            return;
        }

        let history = FraudDetectionService.getAnalysisHistory();
        history = history.filter(a => a.id !== analysisId);
        localStorage.setItem('fraudDetectionHistory', JSON.stringify(history));

        this.render();
    },

    /**
     * Export all analyses
     */
    exportAllAnalyses: function() {
        const history = FraudDetectionService.getAnalysisHistory();

        if (history.length === 0) {
            alert('No analyses to export');
            return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `fraud-detection-history-${timestamp}.json`;
        const content = JSON.stringify(history, null, 2);
        const blob = new Blob([content], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    },

    /**
     * Clear all analyses
     */
    clearAllAnalyses: function() {
        if (!confirm('Are you sure you want to clear all analysis history? This cannot be undone.')) {
            return;
        }

        localStorage.setItem('fraudDetectionHistory', JSON.stringify([]));
        this.render();
    }
};
