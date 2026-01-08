/**
 * Validation Utilities for Smart Contract AI Analyzer
 */

const Validators = {
    /**
     * Validate email
     */
    isEmail: function(email) {
        const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return regex.test(email);
    },

    /**
     * Validate URL
     */
    isUrl: function(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    },

    /**
     * Validate Solidity contract code
     */
    isSolidityCode: function(code) {
        if (!code || typeof code !== 'string') return false;
        // Check for basic Solidity patterns
        return /pragma\s+solidity|contract\s+\w+|function\s+\w+/.test(code);
    },

    /**
     * Validate contract code is not empty
     */
    isContractCodeValid: function(code) {
        if (!code || typeof code !== 'string') return false;
        const trimmed = code.trim();
        return trimmed.length > 0 && trimmed.length < 1000000; // Max 1MB
    },

    /**
     * Validate file is .sol
     */
    isSolidityFile: function(filename) {
        return filename.toLowerCase().endsWith('.sol');
    },

    /**
     * Validate file size
     */
    isFileSizeValid: function(file, maxSizeMB = 10) {
        const maxBytes = maxSizeMB * 1024 * 1024;
        return file.size <= maxBytes;
    },

    /**
     * Validate risk score
     */
    isRiskScoreValid: function(score) {
        return typeof score === 'number' && score >= 0 && score <= 100;
    },

    /**
     * Validate severity level
     */
    isSeverityValid: function(severity) {
        const validSeverities = ['critical', 'high', 'medium', 'low'];
        return validSeverities.includes(severity.toLowerCase());
    },

    /**
     * Validate confidence score
     */
    isConfidenceValid: function(confidence) {
        return typeof confidence === 'number' && confidence >= 0 && confidence <= 100;
    },

    /**
     * Validate line number
     */
    isLineNumberValid: function(lineNumber) {
        return typeof lineNumber === 'number' && lineNumber > 0;
    },

    /**
     * Validate vulnerability object
     */
    isVulnerabilityValid: function(vuln) {
        return (
            vuln &&
            typeof vuln === 'object' &&
            typeof vuln.id === 'string' &&
            typeof vuln.type === 'string' &&
            this.isSeverityValid(vuln.severity) &&
            this.isConfidenceValid(vuln.confidence) &&
            typeof vuln.description === 'string'
        );
    },

    /**
     * Validate analysis result
     */
    isAnalysisResultValid: function(result) {
        return (
            result &&
            typeof result === 'object' &&
            typeof result.id === 'string' &&
            typeof result.contractCode === 'string' &&
            this.isRiskScoreValid(result.riskScore) &&
            Array.isArray(result.vulnerabilities) &&
            result.vulnerabilities.every(v => this.isVulnerabilityValid(v))
        );
    },

    /**
     * Validate API response
     */
    isApiResponseValid: function(response) {
        return (
            response &&
            typeof response === 'object' &&
            typeof response.success === 'boolean'
        );
    },

    /**
     * Validate JSON
     */
    isValidJson: function(jsonString) {
        try {
            JSON.parse(jsonString);
            return true;
        } catch {
            return false;
        }
    },

    /**
     * Validate CSV
     */
    isValidCsv: function(csvString) {
        try {
            // Basic CSV validation
            const lines = csvString.trim().split('\n');
            if (lines.length < 1) return false;
            const headerCount = lines[0].split(',').length;
            return lines.every(line => line.split(',').length === headerCount);
        } catch {
            return false;
        }
    },

    /**
     * Validate hex string
     */
    isHexString: function(str) {
        return /^0x[0-9a-fA-F]*$/.test(str);
    },

    /**
     * Validate Ethereum address
     */
    isEthereumAddress: function(address) {
        return /^0x[a-fA-F0-9]{40}$/.test(address);
    },

    /**
     * Validate port number
     */
    isPortValid: function(port) {
        const portNum = parseInt(port);
        return portNum > 0 && portNum < 65536;
    },

    /**
     * Validate IP address
     */
    isIpAddress: function(ip) {
        const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
        if (!ipv4Regex.test(ip)) return false;
        return ip.split('.').every(part => {
            const num = parseInt(part);
            return num >= 0 && num <= 255;
        });
    },

    /**
     * Validate date
     */
    isValidDate: function(date) {
        return date instanceof Date && !isNaN(date);
    },

    /**
     * Validate date string
     */
    isValidDateString: function(dateString) {
        const date = new Date(dateString);
        return this.isValidDate(date);
    },

    /**
     * Validate time range
     */
    isValidTimeRange: function(startTime, endTime) {
        const start = new Date(startTime);
        const end = new Date(endTime);
        return this.isValidDate(start) && this.isValidDate(end) && start < end;
    },

    /**
     * Validate number range
     */
    isInRange: function(value, min, max) {
        return typeof value === 'number' && value >= min && value <= max;
    },

    /**
     * Validate string length
     */
    isStringLengthValid: function(str, minLength = 0, maxLength = Infinity) {
        return typeof str === 'string' && str.length >= minLength && str.length <= maxLength;
    },

    /**
     * Validate array length
     */
    isArrayLengthValid: function(arr, minLength = 0, maxLength = Infinity) {
        return Array.isArray(arr) && arr.length >= minLength && arr.length <= maxLength;
    },

    /**
     * Validate object has required keys
     */
    hasRequiredKeys: function(obj, requiredKeys) {
        return requiredKeys.every(key => key in obj);
    },

    /**
     * Validate no special characters
     */
    hasNoSpecialChars: function(str) {
        return /^[a-zA-Z0-9_-]*$/.test(str);
    },

    /**
     * Validate alphanumeric
     */
    isAlphanumeric: function(str) {
        return /^[a-zA-Z0-9]*$/.test(str);
    },

    /**
     * Validate contains only numbers
     */
    isNumeric: function(str) {
        return /^\d+$/.test(str);
    },

    /**
     * Validate contains only letters
     */
    isAlpha: function(str) {
        return /^[a-zA-Z]*$/.test(str);
    }
};
