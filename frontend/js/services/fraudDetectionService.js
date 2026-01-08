/**
 * Fraud Detection Service
 * Handles communication with the fraud detection API
 */

const FraudDetectionService = {
    baseURL: 'http://127.0.0.1:5000/api/fraud-detection',

    /**
     * Analyze transactions for fraud
     */
    analyzeTransactions: async function(transactions) {
        try {
            const response = await fetch(`${this.baseURL}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    transactions: transactions
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error analyzing transactions:', error);
            throw error;
        }
    },

    /**
     * Get model status
     */
    getModelStatus: async function() {
        try {
            const response = await fetch(`${this.baseURL}/models/status`);

            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting model status:', error);
            throw error;
        }
    },

    /**
     * Health check
     */
    healthCheck: async function() {
        try {
            const response = await fetch(`${this.baseURL}/health`);

            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error checking health:', error);
            throw error;
        }
    },

    /**
     * Get analysis history
     */
    getAnalysisHistory: function() {
        const history = localStorage.getItem('fraudDetectionHistory');
        return history ? JSON.parse(history) : [];
    },

    /**
     * Save analysis to history
     */
    saveAnalysis: function(analysis) {
        const history = this.getAnalysisHistory();
        history.unshift({
            ...analysis,
            timestamp: new Date().toISOString(),
            id: Date.now()
        });
        // Keep only last 50 analyses
        history.splice(50);
        localStorage.setItem('fraudDetectionHistory', JSON.stringify(history));
    },

    /**
     * Export analysis results
     */
    exportResults: function(results, format = 'json') {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `fraud-detection-${timestamp}.${format}`;

        let content;
        let type;

        if (format === 'json') {
            content = JSON.stringify(results, null, 2);
            type = 'application/json';
        } else if (format === 'csv') {
            content = this.convertToCSV(results);
            type = 'text/csv';
        }

        const blob = new Blob([content], { type });
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
     * Convert results to CSV
     */
    convertToCSV: function(results) {
        let csv = 'Transaction Index,Prediction,Fraud Probability,Risk Score,Confidence\n';

        results.results.forEach(result => {
            const prediction = result.prediction === 1 ? 'Fraudulent' : 'Legitimate';
            const fraudProb = (result.fraud_probability * 100).toFixed(2);
            const riskScore = (result.risk_score * 100).toFixed(2);
            const confidence = result.fraud_probability > 0.5 ? 'High' : 'Low';

            csv += `${result.transaction_index},${prediction},${fraudProb}%,${riskScore}%,${confidence}\n`;
        });

        return csv;
    }
};
