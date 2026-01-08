/**
 * Results Table Component
 */

const ResultsTable = {
    /**
     * Render results table
     */
    render: function(vulnerabilities) {
        if (!vulnerabilities || vulnerabilities.length === 0) {
            return '<p>No vulnerabilities found</p>';
        }

        let html = '<table class="results-table"><thead><tr>';
        html += '<th>Type</th><th>Severity</th><th>Confidence</th><th>Description</th>';
        html += '</tr></thead><tbody>';

        vulnerabilities.forEach(vuln => {
            html += '<tr>';
            html += `<td>${Helpers.escapeHtml(vuln.type || 'Unknown')}</td>`;
            html += `<td><span class="severity-${(vuln.severity || 'Medium').toLowerCase()}">${vuln.severity || 'Medium'}</span></td>`;
            html += `<td>${((vuln.confidence || 0) * 100).toFixed(0)}%</td>`;
            html += `<td>${Helpers.escapeHtml(vuln.description || 'No description')}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table>';
        return html;
    }
};
