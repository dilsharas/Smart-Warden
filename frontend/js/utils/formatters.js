/**
 * Formatting Utilities for Smart Contract AI Analyzer
 */

const Formatters = {
    /**
     * Format risk score with color
     */
    formatRiskScore: function(score) {
        if (score >= 80) return { value: score, level: 'Critical', color: '#f44336' };
        if (score >= 60) return { value: score, level: 'High', color: '#ff9800' };
        if (score >= 40) return { value: score, level: 'Medium', color: '#9c27b0' };
        if (score >= 20) return { value: score, level: 'Low', color: '#4caf50' };
        return { value: score, level: 'Safe', color: '#2e8b57' };
    },

    /**
     * Format severity with styling
     */
    formatSeverity: function(severity) {
        const severityMap = {
            'critical': { label: 'Critical', color: '#f44336', bg: '#ffebee' },
            'high': { label: 'High', color: '#ff9800', bg: '#fff3e0' },
            'medium': { label: 'Medium', color: '#9c27b0', bg: '#f3e5f5' },
            'low': { label: 'Low', color: '#4caf50', bg: '#e8f5e9' }
        };
        return severityMap[severity.toLowerCase()] || severityMap['low'];
    },

    /**
     * Format confidence percentage
     */
    formatConfidence: function(confidence) {
        return `${Math.round(confidence)}%`;
    },

    /**
     * Format vulnerability type
     */
    formatVulnerabilityType: function(type) {
        const typeMap = {
            'reentrancy': 'Reentrancy',
            'integer_overflow': 'Integer Overflow',
            'integer_underflow': 'Integer Underflow',
            'unchecked_call': 'Unchecked Call',
            'delegatecall': 'Delegatecall',
            'timestamp_dependency': 'Timestamp Dependency',
            'bad_randomness': 'Bad Randomness',
            'front_running': 'Front Running',
            'access_control': 'Access Control',
            'logic_error': 'Logic Error'
        };
        return typeMap[type.toLowerCase()] || type;
    },

    /**
     * Format analysis time
     */
    formatAnalysisTime: function(milliseconds) {
        if (milliseconds < 1000) {
            return `${Math.round(milliseconds)}ms`;
        } else if (milliseconds < 60000) {
            return `${(milliseconds / 1000).toFixed(2)}s`;
        } else {
            return `${(milliseconds / 60000).toFixed(2)}m`;
        }
    },

    /**
     * Format file size
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Format number with commas
     */
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    /**
     * Format percentage
     */
    formatPercentage: function(value, decimals = 1) {
        return `${(value * 100).toFixed(decimals)}%`;
    },

    /**
     * Format currency
     */
    formatCurrency: function(amount, currency = 'USD') {
        const formatter = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        });
        return formatter.format(amount);
    },

    /**
     * Format date and time
     */
    formatDateTime: function(date) {
        const d = new Date(date);
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hours = String(d.getHours()).padStart(2, '0');
        const minutes = String(d.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day} ${hours}:${minutes}`;
    },

    /**
     * Format date only
     */
    formatDateOnly: function(date) {
        const d = new Date(date);
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    },

    /**
     * Format time only
     */
    formatTimeOnly: function(date) {
        const d = new Date(date);
        const hours = String(d.getHours()).padStart(2, '0');
        const minutes = String(d.getMinutes()).padStart(2, '0');
        const seconds = String(d.getSeconds()).padStart(2, '0');
        return `${hours}:${minutes}:${seconds}`;
    },

    /**
     * Format relative time
     */
    formatRelativeTime: function(date) {
        const now = new Date();
        const diff = now - new Date(date);
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        const weeks = Math.floor(days / 7);
        const months = Math.floor(days / 30);
        const years = Math.floor(days / 365);

        if (seconds < 60) return 'just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;
        if (weeks < 4) return `${weeks}w ago`;
        if (months < 12) return `${months}mo ago`;
        return `${years}y ago`;
    },

    /**
     * Format code with line numbers
     */
    formatCodeWithLineNumbers: function(code) {
        return code
            .split('\n')
            .map((line, index) => `${String(index + 1).padStart(4, ' ')} | ${line}`)
            .join('\n');
    },

    /**
     * Format JSON with indentation
     */
    formatJson: function(obj, indent = 2) {
        return JSON.stringify(obj, null, indent);
    },

    /**
     * Format CSV from array
     */
    formatCsv: function(data) {
        if (!Array.isArray(data) || data.length === 0) return '';

        const headers = Object.keys(data[0]);
        const rows = data.map(row =>
            headers.map(header => {
                const value = row[header];
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            }).join(',')
        );

        return [headers.join(','), ...rows].join('\n');
    },

    /**
     * Format HTML table from array
     */
    formatHtmlTable: function(data, headers = null) {
        if (!Array.isArray(data) || data.length === 0) return '<table></table>';

        const cols = headers || Object.keys(data[0]);
        let html = '<table class="table"><thead><tr>';

        cols.forEach(col => {
            html += `<th>${Helpers.escapeHtml(col)}</th>`;
        });

        html += '</tr></thead><tbody>';

        data.forEach(row => {
            html += '<tr>';
            cols.forEach(col => {
                const value = row[col] || '';
                html += `<td>${Helpers.escapeHtml(String(value))}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        return html;
    },

    /**
     * Format markdown to HTML
     */
    formatMarkdownToHtml: function(markdown) {
        let html = markdown
            .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
            .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
            .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/gm, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gm, '<em>$1</em>')
            .replace(/\n\n/gm, '</p><p>')
            .replace(/^- (.*?)$/gm, '<li>$1</li>')
            .replace(/(<li>.*?<\/li>)/s, '<ul>$1</ul>');

        return `<p>${html}</p>`;
    },

    /**
     * Format bytes to hex string
     */
    formatBytesToHex: function(bytes) {
        return Array.from(bytes)
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');
    },

    /**
     * Format hex string to bytes
     */
    formatHexToBytes: function(hex) {
        const bytes = [];
        for (let i = 0; i < hex.length; i += 2) {
            bytes.push(parseInt(hex.substr(i, 2), 16));
        }
        return bytes;
    },

    /**
     * Format camelCase to Title Case
     */
    formatCamelCaseToTitleCase: function(str) {
        return str
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, str => str.toUpperCase())
            .trim();
    },

    /**
     * Format snake_case to Title Case
     */
    formatSnakeCaseToTitleCase: function(str) {
        return str
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    },

    /**
     * Format phone number
     */
    formatPhoneNumber: function(phone) {
        const cleaned = phone.replace(/\D/g, '');
        if (cleaned.length === 10) {
            return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
        }
        return phone;
    },

    /**
     * Format URL
     */
    formatUrl: function(url) {
        try {
            const urlObj = new URL(url);
            return urlObj.href;
        } catch {
            return url;
        }
    },

    /**
     * Format status badge
     */
    formatStatusBadge: function(status) {
        const statusMap = {
            'online': { label: 'Online', color: '#4caf50' },
            'offline': { label: 'Offline', color: '#f44336' },
            'loading': { label: 'Loading', color: '#ff9800' },
            'error': { label: 'Error', color: '#f44336' },
            'success': { label: 'Success', color: '#4caf50' },
            'warning': { label: 'Warning', color: '#ff9800' },
            'info': { label: 'Info', color: '#2196f3' }
        };
        return statusMap[status.toLowerCase()] || { label: status, color: '#999' };
    }
};
