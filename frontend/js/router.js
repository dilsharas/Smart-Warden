/**
 * Router for single-page application navigation
 */

const Router = {
    routes: {},
    currentPage: null,

    /**
     * Register a route
     */
    register: function(path, component) {
        this.routes[path] = component;
    },

    /**
     * Register multiple routes
     */
    registerMultiple: function(routes) {
        Object.assign(this.routes, routes);
    },

    /**
     * Navigate to a page
     */
    navigate: function(path) {
        // Remove leading hash if present
        if (path.startsWith('#')) {
            path = path.substring(1);
        }

        // Remove leading slash if present
        if (path.startsWith('/')) {
            path = path.substring(1);
        }

        // Default to analyze if empty
        if (!path) {
            path = 'analyze';
        }

        // Check if route exists
        if (!this.routes[path]) {
            console.warn(`Route not found: ${path}`);
            path = 'analyze';
        }

        // Update URL
        window.history.pushState({ page: path }, '', `#/${path}`);

        // Render page
        this.renderPage(path);

        // Update state
        StateManager.setCurrentPage(path);

        // Update active nav link
        this.updateActiveNavLink(path);
    },

    /**
     * Render page
     */
    renderPage: function(path) {
        const component = this.routes[path];
        if (!component) return;

        const container = document.getElementById('pageContainer');
        if (!container) return;

        try {
            container.innerHTML = '';
            if (typeof component === 'function') {
                component();
            } else if (typeof component === 'string') {
                container.innerHTML = component;
            }
            this.currentPage = path;
        } catch (error) {
            console.error(`Error rendering page ${path}:`, error);
            this.showErrorPage(error);
        }
    },

    /**
     * Update active nav link
     */
    updateActiveNavLink: function(path) {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            const linkPage = link.dataset.page;
            if (linkPage === path) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    },

    /**
     * Show error page
     */
    showErrorPage: function(error) {
        const container = document.getElementById('pageContainer');
        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">⚠️ Error</h1>
                </div>
                <div class="card">
                    <div class="alert alert-error">
                        <p>${Helpers.escapeHtml(error.message)}</p>
                    </div>
                    <button class="btn btn-primary" onclick="Router.navigate('analyze')">
                        Back to Analyze
                    </button>
                </div>
            </div>
        `;
    },

    /**
     * Initialize router
     */
    init: function() {
        // Handle browser back/forward
        window.addEventListener('popstate', (e) => {
            const page = e.state?.page || 'analyze';
            this.renderPage(page);
            this.updateActiveNavLink(page);
        });

        // Handle nav link clicks
        document.addEventListener('click', (e) => {
            const navLink = e.target.closest('.nav-link');
            if (navLink) {
                e.preventDefault();
                const page = navLink.dataset.page;
                this.navigate(page);
            }
        });

        // Load initial page from URL
        const hash = window.location.hash.substring(2) || 'analyze';
        this.navigate(hash);
    }
};

// Don't auto-initialize - let app.js control the initialization order

