/**
 * Risk Gauge Component
 */

const RiskGauge = {
    /**
     * Render risk gauge
     */
    render: function(riskScore) {
        const score = Math.min(100, Math.max(0, riskScore || 0));
        const color = score > 70 ? '#e74c3c' : score > 40 ? '#f39c12' : '#27ae60';
        const percentage = score;

        return `
            <div class="risk-gauge">
                <svg viewBox="0 0 200 120" class="gauge-svg">
                    <defs>
                        <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" style="stop-color:#27ae60;stop-opacity:1" />
                            <stop offset="50%" style="stop-color:#f39c12;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#e74c3c;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                    <path d="M 20 100 A 80 80 0 0 1 180 100" stroke="url(#gaugeGradient)" stroke-width="20" fill="none" stroke-linecap="round"/>
                    <line x1="100" y1="100" x2="100" y2="30" stroke="${color}" stroke-width="3" transform="rotate(${(percentage / 100) * 160 - 80} 100 100)"/>
                </svg>
                <div class="gauge-label">
                    <span class="gauge-value">${score}</span>
                    <span class="gauge-unit">/100</span>
                </div>
            </div>
        `;
    }
};
