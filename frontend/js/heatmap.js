// Heatmap rendering using Canvas API

// Main render function
function renderHeatmap(canvas, attentionMatrix, options = {}) {
    if (!canvas || !attentionMatrix) return;

    const ctx = canvas.getContext('2d');
    const size = attentionMatrix.length;

    // Set canvas size
    const canvasSize = options.canvasSize || 512;
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // Calculate cell size
    const cellSize = canvasSize / size;

    // Disable image smoothing for crisp pixels
    ctx.imageSmoothingEnabled = false;

    // Get current theme for color scheme
    const isDark = document.body.getAttribute('data-theme') === 'dark';

    // Draw heatmap
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const value = attentionMatrix[i][j];
            const color = getHeatmapColor(value, isDark);

            ctx.fillStyle = color;
            ctx.fillRect(
                j * cellSize,
                i * cellSize,
                cellSize,
                cellSize
            );
        }
    }
}

// Color mapping (attention weight → color)
function getHeatmapColor(value, isDark = false) {
    // Clamp value between 0 and 1
    value = Math.max(0, Math.min(1, value));

    // Apply power function to make high values more prominent
    // This makes the yellow show up more aggressively
    value = Math.pow(value, 0.5); // Square root makes the scale more aggressive

    if (isDark) {
        // Dark mode: Purple to Yellow
        return interpolateColor(
            { r: 88, g: 28, b: 135 },    // Deep purple (low attention)
            { r: 156, g: 89, b: 182 },   // Medium purple (mid attention)
            { r: 241, g: 196, b: 15 },   // Bright yellow (high attention)
            value
        );
    } else {
        // Light mode: Purple to Yellow
        return interpolateColor(
            { r: 147, g: 51, b: 234 },   // Purple (low attention)
            { r: 192, g: 132, b: 252 },  // Light purple (mid attention)
            { r: 250, g: 204, b: 21 },   // Yellow (high attention)
            value
        );
    }
}

// Interpolate between three colors based on value
function interpolateColor(low, mid, high, value) {
    let color;

    if (value < 0.5) {
        // Interpolate between low and mid
        const t = value * 2;
        color = {
            r: Math.round(low.r + (mid.r - low.r) * t),
            g: Math.round(low.g + (mid.g - low.g) * t),
            b: Math.round(low.b + (mid.b - low.b) * t)
        };
    } else {
        // Interpolate between mid and high
        const t = (value - 0.5) * 2;
        color = {
            r: Math.round(mid.r + (high.r - mid.r) * t),
            g: Math.round(mid.g + (high.g - mid.g) * t),
            b: Math.round(mid.b + (high.b - mid.b) * t)
        };
    }

    return `rgb(${color.r}, ${color.g}, ${color.b})`;
}

// Setup interactive features (hover tooltip)
function setupHeatmapInteraction(canvas, attentionMatrix) {
    if (!canvas || !attentionMatrix) return;

    const size = attentionMatrix.length;
    let tooltip = document.getElementById('heatmap-tooltip');

    // Create tooltip if it doesn't exist
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'heatmap-tooltip';
        tooltip.className = 'heatmap-tooltip';
        document.body.appendChild(tooltip);
    }

    // Mouse move handler
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Calculate which cell we're hovering over
        const cellSize = canvas.width / size;
        const col = Math.floor(x / cellSize);
        const row = Math.floor(y / cellSize);

        if (row >= 0 && row < size && col >= 0 && col < size) {
            const value = attentionMatrix[row][col];

            // Show tooltip
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
            tooltip.innerHTML = `
                <strong>From:</strong> ${row}<br>
                <strong>To:</strong> ${col}<br>
                <strong>Attention:</strong> ${value.toFixed(4)}
            `;
        } else {
            tooltip.style.display = 'none';
        }
    });

    // Mouse leave handler
    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

// Clear heatmap
function clearHeatmap(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
