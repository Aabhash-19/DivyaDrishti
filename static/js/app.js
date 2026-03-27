// ============================================
// Main Application Logic
// Real-time Updates & Animations
// ============================================

class DivyaDrishtiApp {
    constructor() {
        // DOM Elements
        this.detectedCount = document.getElementById('detectedCount');
        this.trackingCount = document.getElementById('trackingCount');
        this.fpsValue = document.getElementById('fpsValue');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.statusText = document.getElementById('statusText');
        this.statusIcon = document.getElementById('statusIcon');
        this.detectionsList = document.getElementById('detectionsList');
        this.videoFeed = document.getElementById('videoFeed');
        this.resetBtn = document.getElementById('resetBtn');
        this.refreshBtn = document.getElementById('refreshBtn');

        // Active model badge (read-only, auto-updated)
        this.activeModelBadge = document.getElementById('activeModelBadge');
        this.activeModelIcon = document.getElementById('activeModelIcon');
        this.activeModelText = document.getElementById('activeModelText');

        // State
        this.frameCount = 0;
        this.lastDetections = [];
        this.currentActiveModel = 'general';

        this.init();
    }

    init() {
        this.startUpdates();
        this.startFPSCounter();

        this.resetBtn.addEventListener('click', () => this.resetTracking());
        this.refreshBtn.addEventListener('click', () => location.reload());

        this.videoFeed.addEventListener('load', () => {
            this.frameCount++;
        });
    }

    startUpdates() {
        setInterval(() => this.updateStatus(), 1000);
        setInterval(() => this.updateDetections(), 1000);
    }

    startFPSCounter() {
        setInterval(() => {
            this.animateCounter(this.fpsValue, this.frameCount);
            this.fpsCounter.textContent = this.frameCount;
            this.frameCount = 0;
        }, 1000);
    }

    async updateStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            // Stream status
            if (data.streaming) {
                this.statusText.textContent = 'LIVE';
                this.statusIcon.textContent = '🟢';
            } else {
                this.statusText.textContent = 'OFFLINE';
                this.statusIcon.textContent = '🔴';
            }

            // Tracking count
            this.animateCounter(this.trackingCount, data.objects_tracking || 0);

            // Auto active model badge
            if (data.active_model && data.active_model !== this.currentActiveModel) {
                this.updateActiveModelBadge(data.active_model);
            }
        } catch (error) {
            console.error('Status update error:', error);
        }
    }

    updateActiveModelBadge(activeModel) {
        this.currentActiveModel = activeModel;

        const configs = {
            general: {
                icon: '🌍',
                text: 'General',
                bg: 'rgba(138,43,226,0.2)',
                border: 'rgba(138,43,226,0.3)'
            },
            currency: {
                icon: '💰',
                text: 'Currency',
                bg: 'rgba(255,215,0,0.2)',
                border: 'rgba(255,215,0,0.4)'
            },
            both: {
                icon: '✨',
                text: 'General + Currency',
                bg: 'rgba(0,200,100,0.2)',
                border: 'rgba(0,200,100,0.4)'
            }
        };

        const cfg = configs[activeModel] || configs.general;

        if (this.activeModelBadge) {
            this.activeModelBadge.style.background = cfg.bg;
            this.activeModelBadge.style.borderColor = cfg.border;
        }
        if (this.activeModelIcon) this.activeModelIcon.textContent = cfg.icon;
        if (this.activeModelText) this.activeModelText.textContent = cfg.text;
    }

    async updateDetections() {
        try {
            const response = await fetch('/detections');
            const data = await response.json();

            if (data.success) {
                this.animateCounter(this.detectedCount, data.count || 0);
                this.renderDetections(data.detections || []);
            }
        } catch (error) {
            console.error('Detections update error:', error);
        }
    }

    renderDetections(detections) {
        if (detections.length === 0) {
            this.detectionsList.innerHTML = `
                <div class="detection-card pending">
                    <div class="detection-info">
                        <div class="detection-name">No objects detected</div>
                        <div class="detection-meta">Hold objects steady for 3 seconds</div>
                    </div>
                </div>
            `;
            return;
        }

        const detectionsChanged = JSON.stringify(detections) !== JSON.stringify(this.lastDetections);

        if (detectionsChanged) {
            this.detectionsList.innerHTML = detections.map((det, index) => {
                const isCurrency = det.model_source === 'currency';
                const sourceIcon = isCurrency ? '💰 ' : '';
                const cardStyle = isCurrency
                    ? 'border-left: 3px solid gold;'
                    : '';
                return `
                <div class="detection-card" style="animation-delay: ${index * 0.1}s; ${cardStyle}">
                    <div class="detection-info">
                        <div class="detection-name">${sourceIcon}${this.capitalizeFirst(det.class)}</div>
                        <div class="detection-meta">
                            ${isCurrency ? 'Currency Model · ' : ''}Visible for: ${det.visible_for || 3}+ seconds
                        </div>
                    </div>
                    <div class="detection-confidence">
                        ${(det.confidence * 100).toFixed(1)}%
                    </div>
                </div>`;
            }).join('');

            this.lastDetections = detections;
        }
    }

    animateCounter(element, targetValue) {
        const currentValue = parseInt(element.textContent) || 0;
        if (currentValue === targetValue) return;

        element.classList.add('updating');
        const duration = 300;
        const steps = 20;
        const increment = (targetValue - currentValue) / steps;
        const stepDuration = duration / steps;

        let step = 0;
        const timer = setInterval(() => {
            step++;
            const newValue = Math.round(currentValue + (increment * step));
            element.textContent = newValue;

            if (step >= steps) {
                clearInterval(timer);
                element.textContent = targetValue;
                element.classList.remove('updating');
            }
        }, stepDuration);
    }

    async resetTracking() {
        try {
            const response = await fetch('/reset_tracking');
            const data = await response.json();

            if (data.success) {
                this.resetBtn.textContent = '✓ Reset!';
                this.resetBtn.style.background = 'var(--success-color)';

                setTimeout(() => {
                    this.resetBtn.textContent = '🔄 Reset Tracking';
                    this.resetBtn.style.background = '';
                }, 2000);

                this.updateStatus();
                this.updateDetections();
            }
        } catch (error) {
            console.error('Reset tracking error:', error);
        }
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Global function called by the HTML buttons
async function setModel(name) {
    try {
        const res = await fetch('/set_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: name })
        });
        const data = await res.json();
        if (data.success) {
            updateModelUI(data.active_model);
        } else {
            alert('Could not switch model: ' + (data.error || 'unknown error'));
        }
    } catch (e) {
        console.error('setModel error:', e);
    }
}

function updateModelUI(activeModel) {
    const label = document.getElementById('activeModelLabel');
    const genBtn = document.getElementById('modelGeneralBtn');
    const curBtn = document.getElementById('modelCurrencyBtn');
    const badge  = document.getElementById('activeModelBadge');
    const bIcon  = document.getElementById('activeModelIcon');
    const bText  = document.getElementById('activeModelText');

    const isCurrency = activeModel === 'currency';

    if (label) label.textContent = isCurrency ? 'Currency (INR Notes & Coins)' : 'General (Objects)';

    if (genBtn) { genBtn.className = isCurrency ? 'btn btn-secondary' : 'btn btn-primary'; }
    if (curBtn) { curBtn.className = isCurrency ? 'btn btn-primary' : 'btn btn-secondary'; }

    if (badge) {
        badge.style.background = isCurrency ? 'rgba(255,215,0,0.2)' : 'rgba(138,43,226,0.2)';
        badge.style.borderColor = isCurrency ? 'rgba(255,215,0,0.4)' : 'rgba(138,43,226,0.3)';
    }
    if (bIcon) bIcon.textContent = isCurrency ? '💰' : '🌍';
    if (bText) bText.textContent = isCurrency ? 'Currency' : 'General';
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DivyaDrishtiApp();
});
