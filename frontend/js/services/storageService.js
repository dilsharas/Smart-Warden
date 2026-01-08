/**
 * Storage Service for managing local and session storage
 */

const StorageService = {
    /**
     * Set item in localStorage
     */
    setLocal: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.error('Error saving to localStorage:', error);
            return false;
        }
    },

    /**
     * Get item from localStorage
     */
    getLocal: function(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Error reading from localStorage:', error);
            return defaultValue;
        }
    },

    /**
     * Remove item from localStorage
     */
    removeLocal: function(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('Error removing from localStorage:', error);
            return false;
        }
    },

    /**
     * Clear all localStorage
     */
    clearLocal: function() {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error('Error clearing localStorage:', error);
            return false;
        }
    },

    /**
     * Set item in sessionStorage
     */
    setSession: function(key, value) {
        try {
            sessionStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.error('Error saving to sessionStorage:', error);
            return false;
        }
    },

    /**
     * Get item from sessionStorage
     */
    getSession: function(key, defaultValue = null) {
        try {
            const item = sessionStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Error reading from sessionStorage:', error);
            return defaultValue;
        }
    },

    /**
     * Remove item from sessionStorage
     */
    removeSession: function(key) {
        try {
            sessionStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('Error removing from sessionStorage:', error);
            return false;
        }
    },

    /**
     * Clear all sessionStorage
     */
    clearSession: function() {
        try {
            sessionStorage.clear();
            return true;
        } catch (error) {
            console.error('Error clearing sessionStorage:', error);
            return false;
        }
    },

    /**
     * Check if localStorage is available
     */
    isLocalStorageAvailable: function() {
        try {
            const test = '__test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch {
            return false;
        }
    },

    /**
     * Check if sessionStorage is available
     */
    isSessionStorageAvailable: function() {
        try {
            const test = '__test__';
            sessionStorage.setItem(test, test);
            sessionStorage.removeItem(test);
            return true;
        } catch {
            return false;
        }
    },

    /**
     * Get all keys from localStorage
     */
    getLocalKeys: function() {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            keys.push(localStorage.key(i));
        }
        return keys;
    },

    /**
     * Get all keys from sessionStorage
     */
    getSessionKeys: function() {
        const keys = [];
        for (let i = 0; i < sessionStorage.length; i++) {
            keys.push(sessionStorage.key(i));
        }
        return keys;
    },

    /**
     * Get localStorage size in bytes
     */
    getLocalStorageSize: function() {
        let size = 0;
        for (let key in localStorage) {
            if (localStorage.hasOwnProperty(key)) {
                size += localStorage[key].length + key.length;
            }
        }
        return size;
    },

    /**
     * Get sessionStorage size in bytes
     */
    getSessionStorageSize: function() {
        let size = 0;
        for (let key in sessionStorage) {
            if (sessionStorage.hasOwnProperty(key)) {
                size += sessionStorage[key].length + key.length;
            }
        }
        return size;
    }
};

// Storage keys
const StorageKeys = {
    // User preferences
    THEME: 'app:theme',
    SIDEBAR_OPEN: 'app:sidebar:open',
    USER_PREFERENCES: 'app:preferences',

    // Analysis data
    ANALYSIS_HISTORY: 'analysis:history',
    CURRENT_ANALYSIS: 'analysis:current',
    ANALYSIS_RESULTS: 'analysis:results',

    // UI state
    CURRENT_PAGE: 'ui:current:page',
    SELECTED_VULNERABILITY: 'ui:selected:vulnerability',
    FILTER_SEVERITY: 'ui:filter:severity',

    // API configuration
    API_BASE_URL: 'api:base:url',
    API_TIMEOUT: 'api:timeout',

    // Session
    SESSION_ID: 'session:id',
    SESSION_CREATED: 'session:created'
};
