// ============================================
// TTS Controls - Voice Settings Panel
// ============================================

class TTSControls {
    constructor() {
        this.toggle = document.getElementById('ttsToggle');
        this.volumeSlider = document.getElementById('volumeSlider');
        this.rateSlider = document.getElementById('rateSlider');
        this.testBtn = document.getElementById('testTtsBtn');
        this.statusBadge = document.getElementById('ttsStatusBadge');
        this.statusText = document.getElementById('ttsStatusText');
        this.statusIcon = document.getElementById('ttsStatusIcon');
        this.status = document.getElementById('ttsStatus');
        this.speakingIndicator = document.getElementById('speakingIndicator');

        this.isEnabled = true;
        this.isSpeaking = false;

        this.init();
    }

    init() {
        // Load TTS settings
        this.loadSettings();

        // Toggle switch
        this.toggle.addEventListener('click', () => this.toggleTTS());

        // Volume slider
        this.volumeSlider.addEventListener('input', (e) => {
            document.getElementById('volumeValue').textContent = e.target.value + '%';
            this.updateSettings();
        });

        // Rate slider
        this.rateSlider.addEventListener('input', (e) => {
            document.getElementById('rateValue').textContent = e.target.value + ' WPM';
            this.updateSettings();
        });

        // Test button
        this.testBtn.addEventListener('click', () => this.testTTS());
    }

    async loadSettings() {
        try {
            const response = await fetch('/tts/settings');
            const data = await response.json();

            if (data.success) {
                this.isEnabled = data.settings.enabled;
                this.updateToggleUI();

                // Update sliders
                if (data.settings.volume !== undefined) {
                    const volumePercent = Math.round(data.settings.volume * 100);
                    this.volumeSlider.value = volumePercent;
                    document.getElementById('volumeValue').textContent = volumePercent + '%';
                }

                if (data.settings.rate !== undefined) {
                    this.rateSlider.value = data.settings.rate;
                    document.getElementById('rateValue').textContent = data.settings.rate + ' WPM';
                }
            }
        } catch (error) {
            console.error('Failed to load TTS settings:', error);
        }
    }

    async toggleTTS() {
        try {
            const response = await fetch('/tts/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: !this.isEnabled })
            });

            const data = await response.json();
            if (data.success) {
                this.isEnabled = data.tts_enabled;
                this.updateToggleUI();
            }
        } catch (error) {
            console.error('Failed to toggle TTS:', error);
        }
    }

    updateToggleUI() {
        if (this.isEnabled) {
            this.toggle.classList.add('active');
            this.statusText.textContent = 'TTS Active';
            this.statusIcon.textContent = '🔊';
            this.status.textContent = 'Active';
            this.statusBadge.style.background = 'rgba(16, 185, 129, 0.2)';
        } else {
            this.toggle.classList.remove('active');
            this.statusText.textContent = 'TTS Off';
            this.statusIcon.textContent = '🔇';
            this.status.textContent = 'Disabled';
            this.statusBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        }
    }

    async updateSettings() {
        try {
            const volume = this.volumeSlider.value / 100;
            const rate = parseInt(this.rateSlider.value);

            await fetch('/tts/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ volume, rate })
            });
        } catch (error) {
            console.error('Failed to update TTS settings:', error);
        }
    }

    async testTTS() {
        if (this.isSpeaking) return;

        try {
            this.isSpeaking = true;
            this.showSpeaking();

            const response = await fetch('/tts/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: 'Hello! Text to speech is working perfectly.' })
            });

            const data = await response.json();
            if (data.success) {
                // Show speaking animation for 3 seconds
                setTimeout(() => {
                    this.isSpeaking = false;
                    this.hideSpeaking();
                }, 3000);
            }
        } catch (error) {
            console.error('Failed to test TTS:', error);
            this.isSpeaking = false;
            this.hideSpeaking();
        }
    }

    showSpeaking() {
        this.speakingIndicator.style.display = 'block';
        this.status.textContent = 'Speaking...';
        this.testBtn.disabled = true;
        this.testBtn.style.opacity = '0.6';
    }

    hideSpeaking() {
        this.speakingIndicator.style.display = 'none';
        this.status.textContent = this.isEnabled ? 'Active' : 'Disabled';
        this.testBtn.disabled = false;
        this.testBtn.style.opacity = '1';
    }
}

// Initialize TTS controls when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new TTSControls();
});
