/**
 * Charts Component
 */

const Charts = {
    /**
     * Create vulnerability distribution chart
     */
    createVulnerabilityChart: function(containerId, vulnerabilities) {
        if (!vulnerabilities || vulnerabilities.length === 0) {
            return;
        }

        const severityCount = {
            'Critical': 0,
            'High': 0,
            'Medium': 0,
            'Low': 0
        };

        vulnerabilities.forEach(vuln => {
            const severity = vuln.severity || 'Medium';
            if (severityCount.hasOwnProperty(severity)) {
                severityCount[severity]++;
            }
        });

        const ctx = document.getElementById(containerId);
        if (!ctx || !window.Chart) return;

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(severityCount),
                datasets: [{
                    data: Object.values(severityCount),
                    backgroundColor: [
                        '#e74c3c',
                        '#e67e22',
                        '#f39c12',
                        '#27ae60'
                    ]
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
     * Create risk score trend chart
     */
    createRiskTrendChart: function(containerId, history) {
        if (!history || history.length === 0) {
            return;
        }

        const labels = history.slice(-10).map((_, i) => `Analysis ${i + 1}`);
        const data = history.slice(-10).map(h => h.riskScore || 0);

        const ctx = document.getElementById(containerId);
        if (!ctx || !window.Chart) return;

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Risk Score',
                    data: data,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: true
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
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
};
