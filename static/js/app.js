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

        // State
        this.frameCount = 0;
        this.lastDetections = [];

        this.init();
    }

    init() {
        // Start periodic updates
        this.startUpdates();

        // FPS counter
        this.startFPSCounter();

        // Button listeners
        this.resetBtn.addEventListener('click', () => this.resetTracking());
        this.refreshBtn.addEventListener('click', () => location.reload());

        // Video feed load counter
        this.videoFeed.addEventListener('load', () => {
            this.frameCount++;
        });
    }

    startUpdates() {
        // Update status every second
        setInterval(() => this.updateStatus(), 1000);

        // Update detections every second
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

            if (data.success) {
                // Update stream status
                if (data.streaming) {
                    this.statusText.textContent = 'LIVE';
                    this.statusIcon.textContent = '🟢';
                } else {
                    this.statusText.textContent = 'OFFLINE';
                    this.statusIcon.textContent = '🔴';
                }

                // Update tracking count
                this.animateCounter(this.trackingCount, data.objects_tracking || 0);
            }
        } catch (error) {
            console.error('Status update error:', error);
        }
    }

    async updateDetections() {
        try {
            const response = await fetch('/detections');
            const data = await response.json();

            if (data.success) {
                // Update detected count
                this.animateCounter(this.detectedCount, data.count || 0);

                // Update detections list
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

        // Check if detections changed
        const detectionsChanged = JSON.stringify(detections) !== JSON.stringify(this.lastDetections);

        if (detectionsChanged) {
            this.detectionsList.innerHTML = detections.map((det, index) => `
                <div class="detection-card" style="animation-delay: ${index * 0.1}s">
                    <div class="detection-info">
                        <div class="detection-name">${this.capitalizeFirst(det.class)}</div>
                        <div class="detection-meta">
                            Visible for: ${det.visible_for || 3}+ seconds
                        </div>
                    </div>
                    <div class="detection-confidence">
                        ${(det.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `).join('');

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
                // Show success feedback
                this.resetBtn.textContent = '✓ Reset!';
                this.resetBtn.style.background = 'var(--success-color)';

                setTimeout(() => {
                    this.resetBtn.textContent = '🔄 Reset Tracking';
                    this.resetBtn.style.background = '';
                }, 2000);

                // Update displays
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

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DivyaDrishtiApp();
});
