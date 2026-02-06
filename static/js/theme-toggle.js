// ============================================
// Theme Toggle - Dark/Light Mode
// ============================================

class ThemeToggle {
    constructor() {
        this.toggle = document.getElementById('themeToggle');
        this.icon = document.getElementById('themeIcon');
        this.currentTheme = localStorage.getItem('theme') || 'light';
        
        this.init();
    }
    
    init() {
        // Set initial theme
        this.setTheme(this.currentTheme, false);
        
        // Add click listener
        this.toggle.addEventListener('click', () => this.toggleTheme());
    }
    
    setTheme(theme, animate = true) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        this.currentTheme = theme;
        
        // Update icon
        if (theme === 'dark') {
            this.icon.textContent = '🌙';
            this.icon.className = 'theme-icon moon-icon';
        } else {
            this.icon.textContent = '☀️';
            this.icon.className = 'theme-icon sun-icon';
        }
        
        // Add rotation animation
        if (animate) {
            this.toggle.classList.add('rotating');
            setTimeout(() => this.toggle.classList.remove('rotating'), 500);
        }
    }
    
    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
}

// Initialize theme toggle when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ThemeToggle();
});
