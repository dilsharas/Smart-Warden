/**
 * About Page Component
 */

const AboutPage = {
    /**
     * Render about page
     */
    render: function() {
        const container = document.getElementById('pageContainer');

        container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <h1 class="page-title">ℹ️ About</h1>
                    <p class="page-description">Smart Contract AI Analyzer - Advanced Security Analysis</p>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">About This Application</h2>
                    </div>
                    <div class="card-body">
                        <p>
                            The Smart Contract AI Analyzer is an advanced security analysis tool for Ethereum smart contracts.
                            It combines artificial intelligence, pattern-based detection, and external security tools to provide
                            comprehensive vulnerability analysis.
                        </p>
                        <h3>Features</h3>
                        <ul>
                            <li>🤖 AI-Powered Vulnerability Detection</li>
                            <li>🔍 Pattern-Based Analysis</li>
                            <li>⚖️ Tool Comparison (Slither, Mythril)</li>
                            <li>📊 Performance Metrics and Analytics</li>
                            <li>📥 Multiple Export Formats (JSON, CSV, PDF)</li>
                            <li>⚡ Real-time Analysis Progress</li>
                        </ul>
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">System Information</h2>
                    </div>
                    <div class="card-body">
                        <table class="table">
                            <tbody>
                                <tr>
                                    <td><strong>Application Version</strong></td>
                                    <td>1.0.0</td>
                                </tr>
                                <tr>
                                    <td><strong>Frontend Framework</strong></td>
                                    <td>Vanilla JavaScript</td>
                                </tr>
                                <tr>
                                    <td><strong>Backend API</strong></td>
                                    <td>Flask (Python)</td>
                                </tr>
                                <tr>
                                    <td><strong>ML Framework</strong></td>
                                    <td>scikit-learn</td>
                                </tr>
                                <tr>
                                    <td><strong>Browser Support</strong></td>
                                    <td>Chrome, Firefox, Safari, Edge (Latest)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">API Documentation</h2>
                    </div>
                    <div class="card-body">
                        <p>The backend API provides the following endpoints:</p>
                        <ul>
                            <li><code>POST /api/analyze</code> - Analyze a smart contract</li>
                            <li><code>GET /api/models/status</code> - Get AI models status</li>
                            <li><code>GET /health</code> - Check API health</li>
                            <li><code>GET /swagger</code> - Interactive API documentation</li>
                        </ul>
                        <p style="margin-top: var(--spacing-lg);">
                            <button class="btn btn-primary" onclick="window.open('http://localhost:5000/swagger', '_blank')">
                                📖 View Full API Documentation
                            </button>
                        </p>
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">Vulnerability Types</h2>
                    </div>
                    <div class="card-body">
                        <p>The analyzer detects the following types of vulnerabilities:</p>
                        <ul>
                            <li><strong>Reentrancy</strong> - Recursive calls that can drain funds</li>
                            <li><strong>Integer Overflow/Underflow</strong> - Arithmetic operation overflows</li>
                            <li><strong>Unchecked Call</strong> - External calls without proper error handling</li>
                            <li><strong>Delegatecall</strong> - Unsafe delegatecall usage</li>
                            <li><strong>Timestamp Dependency</strong> - Reliance on block.timestamp</li>
                            <li><strong>Bad Randomness</strong> - Weak random number generation</li>
                            <li><strong>Front Running</strong> - Transaction ordering vulnerabilities</li>
                            <li><strong>Access Control</strong> - Improper permission checks</li>
                            <li><strong>Logic Errors</strong> - Flawed business logic</li>
                        </ul>
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">Getting Started</h2>
                    </div>
                    <div class="card-body">
                        <ol>
                            <li>Navigate to the <strong>Analyze</strong> page</li>
                            <li>Upload a .sol file or paste your contract code</li>
                            <li>Select analysis options (AI, Pattern, External Tools)</li>
                            <li>Click <strong>Analyze Contract</strong></li>
                            <li>Review results on the <strong>Results</strong> page</li>
                            <li>Compare tools on the <strong>Comparison</strong> page</li>
                            <li>Export results in your preferred format</li>
                        </ol>
                    </div>
                </div>

                <div class="card" style="margin-top: var(--spacing-lg);">
                    <div class="card-header">
                        <h2 class="card-title">Support & Feedback</h2>
                    </div>
                    <div class="card-body">
                        <p>For issues, feature requests, or feedback, please contact the development team.</p>
                        <p style="margin-top: var(--spacing-lg);">
                            <button class="btn btn-secondary" onclick="App.showNotification('Support contact information coming soon', 'info')">
                                📧 Contact Support
                            </button>
                        </p>
                    </div>
                </div>
            </div>
        `;
    }
};
