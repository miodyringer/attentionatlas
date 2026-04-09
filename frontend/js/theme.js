// Theme management
function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    body.setAttribute('data-theme', newTheme);

    const themeIcon = document.getElementById('theme-icon');
    themeIcon.textContent = newTheme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';

    localStorage.setItem('theme', newTheme);
}

// Load saved theme on page load
window.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    const themeIcon = document.getElementById('theme-icon');
    themeIcon.textContent = savedTheme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';
});
