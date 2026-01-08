/**
 * Event Bus for application-wide event management
 */

const EventBus = {
    events: {},

    /**
     * Subscribe to an event
     */
    on: function(eventName, callback) {
        if (!this.events[eventName]) {
            this.events[eventName] = [];
        }
        this.events[eventName].push(callback);
        
        // Return unsubscribe function
        return () => this.off(eventName, callback);
    },

    /**
     * Unsubscribe from an event
     */
    off: function(eventName, callback) {
        if (!this.events[eventName]) return;
        this.events[eventName] = this.events[eventName].filter(cb => cb !== callback);
    },

    /**
     * Emit an event
     */
    emit: function(eventName, data) {
        if (!this.events[eventName]) return;
        this.events[eventName].forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in event listener for ${eventName}:`, error);
            }
        });
    },

    /**
     * Subscribe to event once
     */
    once: function(eventName, callback) {
        const wrappedCallback = (data) => {
            callback(data);
            this.off(eventName, wrappedCallback);
        };
        this.on(eventName, wrappedCallback);
    },

    /**
     * Clear all listeners for an event
     */
    clear: function(eventName) {
        if (eventName) {
            delete this.events[eventName];
        } else {
            this.events = {};
        }
    }
};

// Common events
const Events = {
    // Navigation
    PAGE_CHANGED: 'page:changed',
    NAVIGATION_REQUESTED: 'navigation:requested',

    // Analysis
    ANALYSIS_STARTED: 'analysis:started',
    ANALYSIS_PROGRESS: 'analysis:progress',
    ANALYSIS_COMPLETED: 'analysis:completed',
    ANALYSIS_FAILED: 'analysis:failed',

    // API
    API_REQUEST_STARTED: 'api:request:started',
    API_REQUEST_COMPLETED: 'api:request:completed',
    API_REQUEST_FAILED: 'api:request:failed',
    API_STATUS_CHANGED: 'api:status:changed',

    // State
    STATE_CHANGED: 'state:changed',
    SESSION_CLEARED: 'session:cleared',

    // UI
    THEME_CHANGED: 'theme:changed',
    SIDEBAR_TOGGLED: 'sidebar:toggled',
    MODAL_OPENED: 'modal:opened',
    MODAL_CLOSED: 'modal:closed',

    // Errors
    ERROR_OCCURRED: 'error:occurred',
    ERROR_CLEARED: 'error:cleared',

    // Notifications
    NOTIFICATION_SHOWN: 'notification:shown',
    NOTIFICATION_HIDDEN: 'notification:hidden'
};
