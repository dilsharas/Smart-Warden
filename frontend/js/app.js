/**
 * Main Application Entry Point
 */

const App = {
    /**
     * Initialize application
     */
    init: function() {
        console.log('🚀 Smart Contract AI Analyzer - Initializing...');

        // Initialize services
        this.initializeServices();

        // Setup event listeners
        this.setupEventListeners();

        // Setup theme toggle
        this.setupThemeToggle();

        // Setup quick actions
        this.setupQuickActions();

        // Update footer time
        this.updateFooterTime();
        setInterval(() => this.updateFooterTime(), 1000);

        // Check API health immediately on startup
        this.checkApiHealth();
        
        // Check API health periodically (every 30 seconds)
        setInterval(() => this.checkApiHealth(), 30000);

        // Register pages
        this.registerPages();

        // Initialize router
        Router.init();

        console.log('✅ Application initialized successfully');
    },

    /**
     * Initialize services
     */
    initializeServices: function() {
        // State manager is auto-initialized
        // API service is auto-initialized
        console.log('✅ Services initialized');
    },

    /**
     * Setup event listeners
     */
    setupEventListeners: function() {
        // API status changes
        EventBus.on(Events.API_STATUS_CHANGED, (status) => {
            this.updateStatusIndicators(status);
        });

        // Analysis started
        EventBus.on(Events.ANALYSIS_STARTED, () => {
            this.showNotification('Analysis started...', 'info');
        });

        // Analysis completed
        EventBus.on(Events.ANALYSIS_COMPLETED, (result) => {
            this.showNotification('Analysis completed!', 'success');
            Router.navigate('results');
        });

        // Analysis failed
        EventBus.on(Events.ANALYSIS_FAILED, (error) => {
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        });

        // API request failed
        EventBus.on(Events.API_REQUEST_FAILED, (data) => {
            console.error('API request failed:', data);
        });

        // Error occurred
        EventBus.on(Events.ERROR_OCCURRED, (error) => {
            this.showErrorBoundary(error);
        });
    },

    /**
     * Setup theme toggle
     */
    setupThemeToggle: function() {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                StateManager.toggleTheme();
                this.updateThemeIcon();
            });
        }
        this.updateThemeIcon();
    },

    /**
     * Update theme icon
     */
    updateThemeIcon: function() {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            const theme = StateManager.getTheme();
            themeToggle.textContent = theme === 'light' ? '🌙' : '☀️';
        }
    },

    /**
     * Setup quick actions
     */
    setupQuickActions: function() {
        const clearHistoryBtn = document.getElementById('clearHistoryBtn');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to clear analysis history?')) {
                    StateManager.clearAnalysisHistory();
                    this.updateAnalysisStats();
                    this.showNotification('Analysis history cleared', 'success');
                }
            });
        }

        const loadSampleBtn = document.getElementById('loadSampleBtn');
        if (loadSampleBtn) {
            loadSampleBtn.addEventListener('click', () => {
                this.loadSampleContract();
            });
        }

        // Setup external tools selection
        this.setupExternalToolsSelection();

        // Setup AI models selection
        this.setupAIModelsSelection();
    },

    /**
     * Setup external tools selection
     */
    setupExternalToolsSelection: function() {
        const slitherTool = document.getElementById('slitherTool');
        const mythrilTool = document.getElementById('mythrilTool');

        if (slitherTool) {
            slitherTool.addEventListener('change', (e) => {
                StateManager.updateValue('externalTools.slither', e.target.checked);
            });
        }

        if (mythrilTool) {
            mythrilTool.addEventListener('change', (e) => {
                StateManager.updateValue('externalTools.mythril', e.target.checked);
            });
        }
    },

    /**
     * Setup AI models selection
     */
    setupAIModelsSelection: function() {
        const binaryModel = document.getElementById('binaryModel');
        const multiclassModel = document.getElementById('multiclassModel');

        if (binaryModel) {
            binaryModel.addEventListener('change', (e) => {
                StateManager.updateValue('aiModels.binary', e.target.checked);
            });
        }

        if (multiclassModel) {
            multiclassModel.addEventListener('change', (e) => {
                StateManager.updateValue('aiModels.multiclass', e.target.checked);
            });
        }
    },

    /**
     * Load sample contract
     */
    loadSampleContract: function() {
        const sampleCode = `pragma solidity ^0.8.0;

contract VulnerableExample {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable to reentrancy
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // Bad randomness
    function randomNumber() public view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp))) % 100;
    }
}`;

        // Store in state for analyze page to use
        StateManager.updateValue('currentAnalysis', {
            contractCode: sampleCode,
            isExample: true
        });

        this.showNotification('Sample contract loaded', 'success');
        Router.navigate('analyze');
    },

    /**
     * Update status indicators
     */
    updateStatusIndicators: function(status) {
        const apiStatus = document.getElementById('apiStatus');
        const modelStatus = document.getElementById('modelStatus');
        const toolsStatus = document.getElementById('toolsStatus');

        if (apiStatus) {
            apiStatus.className = `status-badge ${status.connected ? 'status-online' : 'status-offline'}`;
            apiStatus.title = status.connected ? 'API Connected' : 'API Disconnected';
        }

        if (modelStatus) {
            modelStatus.className = `status-badge ${status.modelsLoaded ? 'status-online' : 'status-offline'}`;
            modelStatus.title = status.modelsLoaded ? 'Models Loaded' : 'Models Not Loaded';
        }

        if (toolsStatus) {
            toolsStatus.className = `status-badge ${status.toolsAvailable ? 'status-online' : 'status-offline'}`;
            toolsStatus.title = status.toolsAvailable ? 'Tools Available' : 'Tools Unavailable';
        }
    },

    /**
     * Update analysis statistics
     */
    updateAnalysisStats: function() {
        const history = StateManager.getAnalysisHistory();
        const statsContainer = document.getElementById('analysisStats');

        if (statsContainer) {
            const totalAnalyses = history.length;
            const vulnerableCount = history.filter(a => a.riskScore >= 40).length;
            const vulnerabilityRate = totalAnalyses > 0 ? ((vulnerableCount / totalAnalyses) * 100).toFixed(1) : 0;

            statsContainer.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Total Analyses</span>
                    <span class="stat-value">${totalAnalyses}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Vulnerabilities Found</span>
                    <span class="stat-value">${vulnerableCount}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Vulnerability Rate</span>
                    <span class="stat-value">${vulnerabilityRate}%</span>
                </div>
            `;
        }
    },

    /**
     * Update footer time
     */
    updateFooterTime: function() {
        const footerTime = document.getElementById('footerTime');
        if (footerTime) {
            footerTime.textContent = Formatters.formatDateTime(new Date());
        }
    },

    /**
     * Check API health
     */
    checkApiHealth: async function() {
        try {
            const status = {
                connected: false,
                modelsLoaded: false,
                toolsAvailable: false
            };
            
            // Check main API health
            try {
                const healthResult = await APIService.get('/health');
                status.connected = healthResult.success && healthResult.data?.status === 'healthy';
                console.log('API Health:', status.connected);
            } catch (e) {
                console.warn('Failed to check API health:', e);
                status.connected = false;
            }
            
            // Check models status
            try {
                const modelsResult = await APIService.get('/api/models/status');
                if (modelsResult.success && modelsResult.data) {
                    status.modelsLoaded = modelsResult.data.models_loaded >= 2 || 
                                         (modelsResult.data.binary_classifier?.loaded && 
                                          modelsResult.data.multiclass_classifier?.loaded);
                    console.log('Models Status:', status.modelsLoaded, modelsResult.data);
                } else {
                    status.modelsLoaded = false;
                }
            } catch (e) {
                console.warn('Failed to check models status:', e);
                status.modelsLoaded = false;
            }
            
            // Check tools status
            try {
                const toolsResult = await APIService.get('/api/tools/status');
                if (toolsResult.success && toolsResult.data) {
                    status.toolsAvailable = toolsResult.data.tools_available || 
                                           (toolsResult.data.slither?.available && 
                                            toolsResult.data.mythril?.available);
                    console.log('Tools Status:', status.toolsAvailable, toolsResult.data);
                } else {
                    status.toolsAvailable = false;
                }
            } catch (e) {
                console.warn('Failed to check tools status:', e);
                status.toolsAvailable = false;
            }
            
            StateManager.setApiStatus(status);
            this.updateStatusIndicators(status);
        } catch (error) {
            console.error('Error checking API health:', error);
        }
    },

    /**
     * Show notification
     */
    showNotification: function(message, type = 'info') {
        const alertClass = `alert alert-${type}`;
        const notification = document.createElement('div');
        notification.className = alertClass;
        notification.innerHTML = `
            <span>${Helpers.escapeHtml(message)}</span>
            <button class="btn-close" onclick="this.parentElement.remove()" style="background: none; border: none; cursor: pointer; font-size: 1.5rem;">×</button>
        `;

        const container = document.body;
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '9999';
        notification.style.maxWidth = '400px';

        container.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 5000);

        EventBus.emit(Events.NOTIFICATION_SHOWN, { message, type });
    },

    /**
     * Show error boundary
     */
    showErrorBoundary: function(error) {
        const errorBoundary = document.getElementById('errorBoundary');
        const errorMessage = document.getElementById('errorMessage');

        if (errorBoundary && errorMessage) {
            errorMessage.textContent = error.message || 'An unexpected error occurred';
            errorBoundary.style.display = 'flex';

            const retryBtn = document.getElementById('errorRetryBtn');
            const reloadBtn = document.getElementById('errorReloadBtn');

            if (retryBtn) {
                retryBtn.onclick = () => {
                    errorBoundary.style.display = 'none';
                    EventBus.emit(Events.ERROR_CLEARED);
                };
            }

            if (reloadBtn) {
                reloadBtn.onclick = () => {
                    window.location.reload();
                };
            }
        }
    },

    /**
     * Register pages
     */
    registerPages: function() {
        Router.register('analyze', () => AnalyzePage.render());
        Router.register('results', () => ResultsPage.render());
        Router.register('comparison', () => ComparisonPage.render());
        Router.register('metrics', () => MetricsPage.render());
        Router.register('about', () => AboutPage.render());
        
        // Fraud detection pages
        Router.register('fraud-detection', () => FraudDetectionPage.render());
        Router.register('fraud-metrics', () => FraudMetricsPage.render());
        Router.register('fraud-history', () => FraudHistoryPage.render());
    }
};

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => App.init());
} else {
    App.init();
}

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    EventBus.emit(Events.ERROR_OCCURRED, event.error);
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    EventBus.emit(Events.ERROR_OCCURRED, new Error(event.reason));
});
