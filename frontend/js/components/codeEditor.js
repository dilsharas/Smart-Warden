/**
 * Code Editor Component
 */

const CodeEditor = {
    /**
     * Initialize code editor
     */
    init: function(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;

        // Add syntax highlighting support
        if (window.hljs) {
            element.addEventListener('input', () => {
                this.highlightCode(element);
            });
        }
    },

    /**
     * Highlight code
     */
    highlightCode: function(element) {
        if (!window.hljs) return;
        
        try {
            element.classList.add('hljs');
            window.hljs.highlightElement(element);
        } catch (e) {
            console.warn('Code highlighting failed:', e);
        }
    },

    /**
     * Get code content
     */
    getCode: function(elementId) {
        const element = document.getElementById(elementId);
        return element ? element.value : '';
    },

    /**
     * Set code content
     */
    setCode: function(elementId, code) {
        const element = document.getElementById(elementId);
        if (element) {
            element.value = code;
            this.highlightCode(element);
        }
    }
};
