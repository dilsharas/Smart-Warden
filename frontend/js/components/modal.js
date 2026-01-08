/**
 * Modal Component
 */

class Modal {
    constructor(options = {}) {
        this.id = Helpers.generateId();
        this.title = options.title || '';
        this.content = options.content || '';
        this.buttons = options.buttons || [];
        this.onClose = options.onClose || null;
        this.size = options.size || 'md'; // sm, md, lg
        this.closeButton = options.closeButton !== false;
        this.backdrop = options.backdrop !== false;
        this.element = null;
    }

    /**
     * Render modal
     */
    render() {
        const sizeClass = `modal-${this.size}`;
        const backdropClass = this.backdrop ? 'modal-backdrop' : '';

        let buttonsHtml = '';
        if (this.buttons.length > 0) {
            buttonsHtml = '<div class="modal-footer">';
            this.buttons.forEach(btn => {
                const btnClass = `btn btn-${btn.type || 'secondary'}`;
                buttonsHtml += `<button class="${btnClass}" data-action="${btn.action}">${btn.label}</button>`;
            });
            buttonsHtml += '</div>';
        }

        const closeButtonHtml = this.closeButton ? '<button class="modal-close" aria-label="Close">&times;</button>' : '';

        const html = `
            <div class="modal-overlay ${backdropClass}" data-modal-id="${this.id}">
                <div class="modal ${sizeClass}">
                    ${closeButtonHtml}
                    ${this.title ? `<div class="modal-header"><h2>${Helpers.escapeHtml(this.title)}</h2></div>` : ''}
                    <div class="modal-body">${this.content}</div>
                    ${buttonsHtml}
                </div>
            </div>
        `;

        const container = document.getElementById('modalContainer');
        const wrapper = document.createElement('div');
        wrapper.innerHTML = html;
        this.element = wrapper.firstElementChild;
        container.appendChild(this.element);

        this.attachEventListeners();
        EventBus.emit(Events.MODAL_OPENED, { modalId: this.id });
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Close button
        const closeBtn = this.element.querySelector('.modal-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.close());
        }

        // Backdrop click
        if (this.backdrop) {
            this.element.addEventListener('click', (e) => {
                if (e.target === this.element) {
                    this.close();
                }
            });
        }

        // Button actions
        const buttons = this.element.querySelectorAll('[data-action]');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                const buttonConfig = this.buttons.find(b => b.action === action);
                if (buttonConfig && buttonConfig.onClick) {
                    buttonConfig.onClick();
                }
                if (buttonConfig && buttonConfig.closeOnClick !== false) {
                    this.close();
                }
            });
        });

        // Escape key
        this.escapeHandler = (e) => {
            if (e.key === 'Escape' && this.closeButton) {
                this.close();
            }
        };
        document.addEventListener('keydown', this.escapeHandler);
    }

    /**
     * Close modal
     */
    close() {
        if (this.element) {
            this.element.remove();
        }
        document.removeEventListener('keydown', this.escapeHandler);
        if (this.onClose) {
            this.onClose();
        }
        EventBus.emit(Events.MODAL_CLOSED, { modalId: this.id });
    }

    /**
     * Update content
     */
    updateContent(content) {
        this.content = content;
        const body = this.element.querySelector('.modal-body');
        if (body) {
            body.innerHTML = content;
        }
    }

    /**
     * Show modal
     */
    show() {
        this.render();
    }

    /**
     * Hide modal
     */
    hide() {
        this.close();
    }
}

// Add modal styles
const modalStyles = `
.modal-container {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: var(--z-modal);
    pointer-events: none;
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    z-index: var(--z-modal);
}

.modal-overlay.modal-backdrop {
    background-color: rgba(0, 0, 0, 0.7);
}

.modal {
    background-color: var(--color-bg);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    max-height: 90vh;
    overflow-y: auto;
    position: relative;
    animation: modalSlideIn 0.3s ease-out;
}

.modal-sm {
    width: 90%;
    max-width: 400px;
}

.modal-md {
    width: 90%;
    max-width: 600px;
}

.modal-lg {
    width: 90%;
    max-width: 800px;
}

@keyframes modalSlideIn {
    from {
        transform: translateY(-50px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.modal-close {
    position: absolute;
    top: var(--spacing-md);
    right: var(--spacing-md);
    background: none;
    border: none;
    font-size: var(--font-size-2xl);
    cursor: pointer;
    color: var(--color-text-secondary);
    transition: color var(--transition-fast);
}

.modal-close:hover {
    color: var(--color-text);
}

.modal-header {
    padding: var(--spacing-lg);
    border-bottom: 1px solid var(--color-border);
}

.modal-header h2 {
    margin: 0;
    font-size: var(--font-size-xl);
}

.modal-body {
    padding: var(--spacing-lg);
}

.modal-footer {
    padding: var(--spacing-lg);
    border-top: 1px solid var(--color-border);
    display: flex;
    gap: var(--spacing-md);
    justify-content: flex-end;
}

.modal-footer .btn {
    margin: 0;
}
`;

// Inject modal styles
if (!document.querySelector('style[data-modal-styles]')) {
    const style = document.createElement('style');
    style.setAttribute('data-modal-styles', 'true');
    style.textContent = modalStyles;
    document.head.appendChild(style);
}
