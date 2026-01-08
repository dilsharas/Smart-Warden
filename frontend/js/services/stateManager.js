/**
 * State Manager for application state management
 */

const StateManager = {
    state: {
        currentPage: 'analyze',
        analysisHistory: [],
        currentAnalysis: null,
        userPreferences: {
            theme: 'light',
            autoExport: false,
            exportFormat: 'json'
        },
        analysisOptions: {
            aiAnalysis: true,
            patternAnalysis: true,
            externalTools: true,
            modelType: 'binary',
            confidenceThreshold: 50,
            timeout: 30,
            reportFormat: 'detailed',
            includeRecommendations: true,
            parallelAnalysis: true
        },
        uiState: {
            sidebarOpen: true,
            selectedVulnerability: null,
            filterSeverity: []
        },
        apiStatus: {
            connected: false,
            modelsLoaded: false,
            toolsAvailable: false
        }
    },

    /**
     * Initialize state from storage
     */
    init: function() {
        const savedState = StorageService.getLocal('app:state');
        if (savedState) {
            this.state = Helpers.merge(this.state, savedState);
        }
        // Ensure analysisHistory is always an array
        if (!Array.isArray(this.state.analysisHistory)) {
            this.state.analysisHistory = [];
        }
        this.loadPreferences();
    },

    /**
     * Get current state
     */
    getState: function() {
        return Helpers.deepClone(this.state);
    },

    /**
     * Get specific state value
     */
    getValue: function(path) {
        const keys = path.split('.');
        let value = this.state;
        for (let key of keys) {
            value = value[key];
            if (value === undefined) return null;
        }
        return value;
    },

    /**
     * Set state value
     */
    setState: function(updates) {
        const oldState = Helpers.deepClone(this.state);
        this.state = Helpers.merge(this.state, updates);
        this.saveState();
        EventBus.emit(Events.STATE_CHANGED, { oldState, newState: this.state });
    },

    /**
     * Update specific value
     */
    updateValue: function(path, value) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        let obj = this.state;
        
        for (let key of keys) {
            if (!obj[key]) obj[key] = {};
            obj = obj[key];
        }
        
        obj[lastKey] = value;
        this.saveState();
        EventBus.emit(Events.STATE_CHANGED, { path, value });
    },

    /**
     * Save state to storage
     */
    saveState: function() {
        StorageService.setLocal('app:state', this.state);
    },

    /**
     * Load preferences
     */
    loadPreferences: function() {
        const theme = StorageService.getLocal(StorageKeys.THEME, 'light');
        const preferences = StorageService.getLocal(StorageKeys.USER_PREFERENCES, {});
        
        this.state.userPreferences = Helpers.merge(this.state.userPreferences, preferences);
        this.applyTheme(theme);
    },

    /**
     * Save preferences
     */
    savePreferences: function() {
        StorageService.setLocal(StorageKeys.USER_PREFERENCES, this.state.userPreferences);
    },

    /**
     * Apply theme
     */
    applyTheme: function(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.state.userPreferences.theme = theme;
        this.savePreferences();
        EventBus.emit(Events.THEME_CHANGED, { theme });
    },

    /**
     * Get current theme
     */
    getTheme: function() {
        return this.state.userPreferences.theme;
    },

    /**
     * Toggle theme
     */
    toggleTheme: function() {
        const currentTheme = this.getTheme();
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme(newTheme);
    },

    /**
     * Set current page
     */
    setCurrentPage: function(page) {
        this.updateValue('currentPage', page);
        EventBus.emit(Events.PAGE_CHANGED, { page });
    },

    /**
     * Get current page
     */
    getCurrentPage: function() {
        return this.state.currentPage;
    },

    /**
     * Add analysis to history
     */
    addAnalysisToHistory: function(analysis) {
        // Ensure analysisHistory is an array
        if (!Array.isArray(this.state.analysisHistory)) {
            this.state.analysisHistory = [];
        }
        this.state.analysisHistory.unshift(analysis);
        // Keep only last 50 analyses
        if (this.state.analysisHistory.length > 50) {
            this.state.analysisHistory.pop();
        }
        this.saveState();
    },

    /**
     * Get analysis history
     */
    getAnalysisHistory: function() {
        return this.state.analysisHistory;
    },

    /**
     * Clear analysis history
     */
    clearAnalysisHistory: function() {
        this.state.analysisHistory = [];
        this.state.currentAnalysis = null;
        this.saveState();
        EventBus.emit(Events.SESSION_CLEARED);
    },

    /**
     * Set current analysis
     */
    setCurrentAnalysis: function(analysis) {
        this.state.currentAnalysis = analysis;
        this.addAnalysisToHistory(analysis);
        this.saveState();
    },

    /**
     * Get current analysis
     */
    getCurrentAnalysis: function() {
        return this.state.currentAnalysis;
    },

    /**
     * Set API status
     */
    setApiStatus: function(status) {
        this.state.apiStatus = Helpers.merge(this.state.apiStatus, status);
        this.saveState();
        EventBus.emit(Events.API_STATUS_CHANGED, this.state.apiStatus);
    },

    /**
     * Get API status
     */
    getApiStatus: function() {
        return this.state.apiStatus;
    },

    /**
     * Set selected vulnerability
     */
    setSelectedVulnerability: function(vulnId) {
        this.updateValue('uiState.selectedVulnerability', vulnId);
    },

    /**
     * Get selected vulnerability
     */
    getSelectedVulnerability: function() {
        return this.state.uiState.selectedVulnerability;
    },

    /**
     * Set severity filter
     */
    setSeverityFilter: function(severities) {
        this.updateValue('uiState.filterSeverity', severities);
    },

    /**
     * Get severity filter
     */
    getSeverityFilter: function() {
        return this.state.uiState.filterSeverity;
    },

    /**
     * Toggle sidebar
     */
    toggleSidebar: function() {
        const isOpen = this.state.uiState.sidebarOpen;
        this.updateValue('uiState.sidebarOpen', !isOpen);
        EventBus.emit(Events.SIDEBAR_TOGGLED, { isOpen: !isOpen });
    },

    /**
     * Reset state
     */
    reset: function() {
        this.state = {
            currentPage: 'analyze',
            analysisHistory: [],
            currentAnalysis: null,
            userPreferences: {
                theme: 'light',
                autoExport: false,
                exportFormat: 'json'
            },
            analysisOptions: {
                aiAnalysis: true,
                patternAnalysis: true,
                externalTools: true,
                modelType: 'binary',
                confidenceThreshold: 50,
                timeout: 30,
                reportFormat: 'detailed',
                includeRecommendations: true,
                parallelAnalysis: true
            },
            uiState: {
                sidebarOpen: true,
                selectedVulnerability: null,
                filterSeverity: []
            },
            apiStatus: {
                connected: false,
                modelsLoaded: false,
                toolsAvailable: false
            }
        };
        this.saveState();
        EventBus.emit(Events.SESSION_CLEARED);
    }
};

// Initialize state manager when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => StateManager.init());
} else {
    StateManager.init();
}
