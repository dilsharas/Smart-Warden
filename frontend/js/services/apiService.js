/**
 * API Service for backend communication
 */

const APIService = {
    baseUrl: 'http://127.0.0.1:5000',
    timeout: 30000,

    /**
     * Set base URL
     */
    setBaseUrl: function(url) {
        this.baseUrl = url;
    },

    /**
     * Set timeout
     */
    setTimeout: function(ms) {
        this.timeout = ms;
    },

    /**
     * Make HTTP request
     */
    request: async function(method, endpoint, data = null, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        const config = {
            method,
            headers,
            signal: AbortSignal.timeout(this.timeout)
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            config.body = JSON.stringify(data);
        }

        try {
            EventBus.emit(Events.API_REQUEST_STARTED, { method, endpoint });
            
            const response = await fetch(url, config);
            const responseData = await response.json();

            if (!response.ok) {
                throw new Error(responseData.error || `HTTP ${response.status}`);
            }

            EventBus.emit(Events.API_REQUEST_COMPLETED, { method, endpoint, status: response.status });
            return { success: true, data: responseData, status: response.status };
        } catch (error) {
            EventBus.emit(Events.API_REQUEST_FAILED, { method, endpoint, error: error.message });
            return { success: false, error: error.message, status: 0 };
        }
    },

    /**
     * GET request
     */
    get: async function(endpoint, options = {}) {
        return this.request('GET', endpoint, null, options);
    },

    /**
     * POST request
     */
    post: async function(endpoint, data, options = {}) {
        return this.request('POST', endpoint, data, options);
    },

    /**
     * PUT request
     */
    put: async function(endpoint, data, options = {}) {
        return this.request('PUT', endpoint, data, options);
    },

    /**
     * DELETE request
     */
    delete: async function(endpoint, options = {}) {
        return this.request('DELETE', endpoint, null, options);
    },

    /**
     * Check API health
     */
    checkHealth: async function() {
        const result = await this.get('/health');
        const isHealthy = result.success && result.data?.status === 'healthy';
        StateManager.setApiStatus({ connected: isHealthy });
        return isHealthy;
    },

    /**
     * Get models status
     */
    getModelsStatus: async function() {
        const result = await this.get('/api/models/status');
        if (result.success) {
            StateManager.setApiStatus({ modelsLoaded: result.data?.models_loaded });
        }
        return result;
    },

    /**
     * Analyze contract
     */
    analyzeContract: async function(contractCode, options = {}) {
        if (!Validators.isContractCodeValid(contractCode)) {
            return { success: false, error: 'Invalid contract code' };
        }

        const payload = {
            contract_code: contractCode,
            ...options
        };

        return this.post('/api/analyze', payload);
    },

    /**
     * Get analysis result
     */
    getAnalysisResult: async function(analysisId) {
        return this.get(`/api/analyze/${analysisId}`);
    },

    /**
     * Get analysis history
     */
    getAnalysisHistory: async function(limit = 10) {
        return this.get(`/api/analyze/history?limit=${limit}`);
    },

    /**
     * Export analysis result
     */
    exportAnalysis: async function(analysisId, format = 'json') {
        return this.get(`/api/analyze/${analysisId}/export?format=${format}`);
    },

    /**
     * Get tool comparison
     */
    getToolComparison: async function(analysisId) {
        return this.get(`/api/analyze/${analysisId}/comparison`);
    },

    /**
     * Get metrics
     */
    getMetrics: async function(options = {}) {
        const params = new URLSearchParams(options);
        return this.get(`/api/metrics?${params.toString()}`);
    },

    /**
     * Retry request with exponential backoff
     */
    retryRequest: async function(fn, maxAttempts = 3, delay = 1000) {
        for (let i = 0; i < maxAttempts; i++) {
            try {
                return await fn();
            } catch (error) {
                if (i === maxAttempts - 1) throw error;
                await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
            }
        }
    }
};

// Initialize API service
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Check API health on startup
        APIService.checkHealth().catch(err => {
            console.error('API health check failed:', err);
            StateManager.setApiStatus({ connected: false });
        });
    });
} else {
    APIService.checkHealth().catch(err => {
        console.error('API health check failed:', err);
        StateManager.setApiStatus({ connected: false });
    });
}
