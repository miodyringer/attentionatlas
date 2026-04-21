// Visualization control logic and data processing

// Global visualization state
let visualizationState = {
    data: null,
    mode: 'heatmap',
    layer: 0,
    head: 0,
    resolution: 'auto',
    aggregation: 'none',
    // Token importance state
    selectedLayers: [],
    selectedHeads: [],
    tokenRange: null,
    importanceMode: 'incoming',
    // Section filter state
    sectionFilter: {
        focusMode: null,  // null, 'document', 'prompt', 'response'
        tokenBoundaries: null
    },
    // Text comparison state
    comparison: {
        originalText: '',
        modifiedText: '',
        originalData: null,
        modifiedData: null,
        selectedLayers: [],
        selectedHeads: []
    },
    // Timeline state
    timeline: {
        currentLayer: 0,
        aggregationMode: 'mean-heads',
        selectedHead: 0
    }
};

// Initialize visualization after analysis
function initializeVisualization(attentionData) {
    if (!attentionData || !attentionData.attention_pattern) {
        showVisualizationError('No attention data available');
        return;
    }

    // Store data in state
    visualizationState.data = attentionData;

    // Store token boundaries from analyzed text structure
    if (currentAnalyzedTextStructure && currentAnalyzedTextStructure.boundaries) {
        visualizationState.sectionFilter.tokenBoundaries = currentAnalyzedTextStructure.boundaries;
        // Show section filter controls
        document.getElementById('section-filter-group').style.display = 'block';
    } else {
        visualizationState.sectionFilter.tokenBoundaries = null;
        visualizationState.sectionFilter.focusMode = null;
        // Hide section filter controls
        document.getElementById('section-filter-group').style.display = 'none';
    }

    // Reset timeline state
    visualizationState.timeline.currentLayer = 0;
    visualizationState.timeline.aggregationMode = 'mean-heads';
    visualizationState.timeline.selectedHead = 0;

    // Show visualization container
    document.getElementById('visualization-container').style.display = 'block';

    // Populate controls
    populateControls(attentionData);

    // Setup keyboard navigation
    setupKeyboardNavigation();

    // Initial render
    updateVisualization();
}

// Populate control dropdowns based on data shape
function populateControls(data) {
    const [numLayers, numHeads, numTokens, contextLen] = data.shape;
    // Use the smaller of numTokens and contextLen for resolution display
    const displaySize = Math.min(numTokens, contextLen);

    // Layer selector
    const layerSelect = document.getElementById('layer-select');
    layerSelect.innerHTML = '';
    for (let i = 0; i < numLayers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Layer ${i}`;
        layerSelect.appendChild(option);
    }

    // Layer slider
    const layerSlider = document.getElementById('layer-slider');
    if (layerSlider) {
        layerSlider.max = numLayers - 1;
        layerSlider.value = visualizationState.layer;
        const layerDisplay = document.getElementById('layer-display');
        if (layerDisplay) {
            layerDisplay.textContent = visualizationState.layer;
        }
    }

    // Head selector
    const headSelect = document.getElementById('head-select');
    headSelect.innerHTML = '';
    for (let i = 0; i < numHeads; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Head ${i}`;
        headSelect.appendChild(option);
    }

    // Resolution selector
    const resolutionSelect = document.getElementById('resolution-select');
    resolutionSelect.innerHTML = '';

    // Auto resolution (smart default)
    const autoOption = document.createElement('option');
    autoOption.value = 'auto';
    autoOption.textContent = 'Auto';
    resolutionSelect.appendChild(autoOption);

    // Original resolution (only if not too large)
    if (displaySize <= 256) {
        const originalOption = document.createElement('option');
        originalOption.value = 'original';
        originalOption.textContent = `Original (${numTokens}x${contextLen})`;
        resolutionSelect.appendChild(originalOption);
    }

    // Fixed resolutions
    [32, 64, 128, 256].forEach(res => {
        if (res < displaySize || displaySize > 256) {
            const option = document.createElement('option');
            option.value = res;
            option.textContent = `${res}x${res}`;
            resolutionSelect.appendChild(option);
        }
    });

    // Set smart default
    if (displaySize > 256) {
        visualizationState.resolution = 128;
        resolutionSelect.value = 128;
    } else if (displaySize > 128) {
        visualizationState.resolution = 64;
        resolutionSelect.value = 64;
    } else {
        visualizationState.resolution = 'original';
        resolutionSelect.value = 'original';
    }
}

// Update visualization when controls change
function updateVisualization() {
    const data = visualizationState.data;
    if (!data) return;

    // Get current settings
    const layer = parseInt(visualizationState.layer);
    const head = parseInt(visualizationState.head);
    const resolution = visualizationState.resolution;
    const aggregation = visualizationState.aggregation;

    // Process data based on settings
    const matrix = processAttentionData(data, { layer, head, resolution, aggregation });

    // Render heatmap
    const canvas = document.getElementById('attention-heatmap');
    renderHeatmap(canvas, matrix, { canvasSize: 512 });

    // Setup interaction
    setupHeatmapInteraction(canvas, matrix);

    // Update control states based on aggregation
    updateControlStates();
}

// Process attention data based on options
function processAttentionData(data, options) {
    let matrix;

    // Step 1: Get base matrix based on aggregation mode
    switch (options.aggregation) {
        case 'heads':
            // Mean across all heads in selected layer
            matrix = aggregateHeads(data.attention_pattern, options.layer);
            break;

        case 'layers':
            // Mean across all layers for selected head
            matrix = aggregateLayers(data.attention_pattern, options.head);
            break;

        case 'none':
        default:
            // Single head from single layer
            matrix = data.attention_pattern[options.layer][options.head];
            break;
    }

    // Step 2: Apply resolution pooling if needed
    const numTokens = matrix.length;  // Query tokens (rows)
    const contextLen = matrix[0].length;  // Key tokens (columns)
    const maxDim = Math.max(numTokens, contextLen);

    let targetSize;
    if (options.resolution === 'auto') {
        if (maxDim > 256) {
            targetSize = 128;
        } else if (maxDim > 128) {
            targetSize = 64;
        } else {
            targetSize = maxDim;
        }
    } else if (options.resolution === 'original') {
        // No pooling - keep original rectangular shape
        return matrix;
    } else {
        targetSize = parseInt(options.resolution);
    }

    // Pool to square matrix if target size is smaller than dimensions
    if (targetSize < numTokens || targetSize < contextLen) {
        matrix = meanPool2D(matrix, targetSize);
    }

    return matrix;
}

// Aggregate attention across all heads in a layer
function aggregateHeads(tensor, layerIdx) {
    const numHeads = tensor[layerIdx].length;
    const numTokens = tensor[layerIdx][0].length;  // Query tokens
    const contextLen = tensor[layerIdx][0][0].length;  // Key tokens (context)

    // Initialize result matrix
    const result = Array(numTokens).fill().map(() => Array(contextLen).fill(0));

    // Sum across heads
    for (let h = 0; h < numHeads; h++) {
        for (let i = 0; i < numTokens; i++) {
            for (let j = 0; j < contextLen; j++) {
                result[i][j] += tensor[layerIdx][h][i][j];
            }
        }
    }

    // Average
    for (let i = 0; i < numTokens; i++) {
        for (let j = 0; j < contextLen; j++) {
            result[i][j] /= numHeads;
        }
    }

    return result;
}

// Aggregate attention across all layers for a head
function aggregateLayers(tensor, headIdx) {
    const numLayers = tensor.length;
    const numTokens = tensor[0][headIdx].length;  // Query tokens
    const contextLen = tensor[0][headIdx][0].length;  // Key tokens (context)

    // Initialize result matrix
    const result = Array(numTokens).fill().map(() => Array(contextLen).fill(0));

    // Sum across layers
    for (let l = 0; l < numLayers; l++) {
        for (let i = 0; i < numTokens; i++) {
            for (let j = 0; j < contextLen; j++) {
                result[i][j] += tensor[l][headIdx][i][j];
            }
        }
    }

    // Average
    for (let i = 0; i < numTokens; i++) {
        for (let j = 0; j < contextLen; j++) {
            result[i][j] /= numLayers;
        }
    }

    return result;
}

// Mean pooling for resolution reduction
function meanPool2D(matrix, targetSize) {
    const rows = matrix.length;
    const cols = matrix[0].length;

    const poolSizeRow = rows / targetSize;
    const poolSizeCol = cols / targetSize;

    const pooled = [];
    for (let i = 0; i < targetSize; i++) {
        pooled[i] = [];
        for (let j = 0; j < targetSize; j++) {
            let sum = 0;
            let count = 0;

            const rowStart = Math.floor(i * poolSizeRow);
            const rowEnd = Math.floor((i + 1) * poolSizeRow);
            const colStart = Math.floor(j * poolSizeCol);
            const colEnd = Math.floor((j + 1) * poolSizeCol);

            for (let pi = rowStart; pi < rowEnd; pi++) {
                for (let pj = colStart; pj < colEnd; pj++) {
                    if (pi < rows && pj < cols) {
                        sum += matrix[pi][pj];
                        count++;
                    }
                }
            }

            pooled[i][j] = count > 0 ? sum / count : 0;
        }
    }

    return pooled;
}

// Update control states based on aggregation mode
function updateControlStates() {
    const aggregation = visualizationState.aggregation;
    const headSelect = document.getElementById('head-select');
    const layerSelect = document.getElementById('layer-select');

    // Disable/enable controls based on aggregation
    if (aggregation === 'heads') {
        headSelect.disabled = true;
        layerSelect.disabled = false;
    } else if (aggregation === 'layers') {
        headSelect.disabled = false;
        layerSelect.disabled = true;
    } else {
        headSelect.disabled = false;
        layerSelect.disabled = false;
    }
}

// Control change handlers
function onLayerChange(value) {
    visualizationState.layer = parseInt(value);
    debounceUpdate();
}

function onLayerSliderChange(layer) {
    const layerValue = parseInt(layer);
    visualizationState.layer = layerValue;

    // Update display
    document.getElementById('layer-display').textContent = layer;

    // Sync hidden select for compatibility
    const layerSelect = document.getElementById('layer-select');
    if (layerSelect) {
        layerSelect.value = layer;
    }

    // Re-render
    debounceUpdate();
}

function onHeadChange(value) {
    visualizationState.head = parseInt(value);
    debounceUpdate();
}

function onResolutionChange(value) {
    visualizationState.resolution = value === 'auto' || value === 'original' ? value : parseInt(value);
    debounceUpdate();
}

function onAggregationChange(value) {
    visualizationState.aggregation = value;

    // Update control visibility based on aggregation mode
    const layerSlider = document.getElementById('layer-slider');
    const layerSliderGroup = layerSlider ? layerSlider.parentElement : null;
    const headSelect = document.getElementById('head-select');
    const headSelectGroup = headSelect ? headSelect.parentElement : null;

    if (layerSliderGroup && headSelectGroup) {
        if (value === 'heads') {
            // Mean across heads: enable layer, disable head
            layerSliderGroup.style.opacity = '1';
            if (layerSlider) layerSlider.disabled = false;
            headSelectGroup.style.opacity = '0.5';
            if (headSelect) headSelect.disabled = true;
        } else if (value === 'layers') {
            // Mean across layers: disable layer, enable head
            layerSliderGroup.style.opacity = '0.5';
            if (layerSlider) layerSlider.disabled = true;
            headSelectGroup.style.opacity = '1';
            if (headSelect) headSelect.disabled = false;
        } else {
            // No aggregation: enable both
            layerSliderGroup.style.opacity = '1';
            if (layerSlider) layerSlider.disabled = false;
            headSelectGroup.style.opacity = '1';
            if (headSelect) headSelect.disabled = false;
        }
    }

    updateVisualization(); // No debounce for aggregation - instant feedback
}

// Word grouping toggle handler
// Debounce updates to avoid excessive re-renders
let updateTimeout;
function debounceUpdate() {
    clearTimeout(updateTimeout);
    updateTimeout = setTimeout(updateVisualization, 100);
}

// Show visualization error
function showVisualizationError(message) {
    const container = document.getElementById('visualization-container');
    container.innerHTML = `<p style="color: var(--text-secondary); text-align: center;">${message}</p>`;
    container.style.display = 'block';
}

// Setup keyboard navigation
function setupKeyboardNavigation() {
    // Remove previous listener if it exists
    if (window.visualizationKeyHandler) {
        document.removeEventListener('keydown', window.visualizationKeyHandler);
    }

    // Create new handler
    window.visualizationKeyHandler = function(e) {
        // Only handle arrows when visualization is active
        const vizContainer = document.getElementById('visualization-container');
        if (!vizContainer || vizContainer.style.display === 'none') return;

        const data = visualizationState.data;
        if (!data) return;

        const mode = visualizationState.mode;

        // Timeline mode: left/right for layers, up/down for heads
        if (mode === 'timeline') {
            const aggregationMode = visualizationState.timeline.aggregationMode;

            switch(e.key) {
                case 'ArrowLeft':
                    // Previous layer
                    e.preventDefault();
                    const currentLayer = visualizationState.timeline.currentLayer;
                    const newLayer = Math.max(0, currentLayer - 1);
                    if (newLayer !== currentLayer) {
                        visualizationState.timeline.currentLayer = newLayer;
                        document.getElementById('timeline-layer-slider').value = newLayer;
                        document.getElementById('timeline-layer-display').textContent = newLayer;
                        renderTimeline();
                    }
                    break;

                case 'ArrowRight':
                    // Next layer
                    e.preventDefault();
                    const currentLayer2 = visualizationState.timeline.currentLayer;
                    const maxLayer = data.shape[0] - 1;
                    const newLayer2 = Math.min(maxLayer, currentLayer2 + 1);
                    if (newLayer2 !== currentLayer2) {
                        visualizationState.timeline.currentLayer = newLayer2;
                        document.getElementById('timeline-layer-slider').value = newLayer2;
                        document.getElementById('timeline-layer-display').textContent = newLayer2;
                        renderTimeline();
                    }
                    break;

                case 'ArrowUp':
                    // Previous head (only in single-head mode)
                    if (aggregationMode === 'single-head') {
                        e.preventDefault();
                        const currentHead = visualizationState.timeline.selectedHead;
                        const newHead = Math.max(0, currentHead - 1);
                        if (newHead !== currentHead) {
                            visualizationState.timeline.selectedHead = newHead;
                            document.getElementById('timeline-head-select').value = newHead;
                            renderTimeline();
                        }
                    }
                    break;

                case 'ArrowDown':
                    // Next head (only in single-head mode)
                    if (aggregationMode === 'single-head') {
                        e.preventDefault();
                        const currentHead2 = visualizationState.timeline.selectedHead;
                        const maxHead = data.shape[1] - 1;
                        const newHead2 = Math.min(maxHead, currentHead2 + 1);
                        if (newHead2 !== currentHead2) {
                            visualizationState.timeline.selectedHead = newHead2;
                            document.getElementById('timeline-head-select').value = newHead2;
                            renderTimeline();
                        }
                    }
                    break;
            }
            return;
        }

        // Heatmap mode: original keyboard navigation
        const layerSelect = document.getElementById('layer-select');
        const headSelect = document.getElementById('head-select');
        const layerSlider = document.getElementById('layer-slider');
        const layerDisplay = document.getElementById('layer-display');

        switch(e.key) {
            case 'ArrowUp':
                // Previous head
                if (!headSelect.disabled) {
                    e.preventDefault();
                    const currentHead = parseInt(visualizationState.head);
                    const newHead = Math.max(0, currentHead - 1);
                    if (newHead !== currentHead) {
                        visualizationState.head = newHead;
                        headSelect.value = newHead;
                        updateVisualization();
                    }
                }
                break;

            case 'ArrowDown':
                // Next head
                if (!headSelect.disabled) {
                    e.preventDefault();
                    const currentHead = parseInt(visualizationState.head);
                    const maxHead = data.shape[1] - 1;
                    const newHead = Math.min(maxHead, currentHead + 1);
                    if (newHead !== currentHead) {
                        visualizationState.head = newHead;
                        headSelect.value = newHead;
                        updateVisualization();
                    }
                }
                break;

            case 'ArrowLeft':
                // Previous layer
                if (!layerSelect.disabled && layerSlider && !layerSlider.disabled) {
                    e.preventDefault();
                    const currentLayer = parseInt(visualizationState.layer);
                    const newLayer = Math.max(0, currentLayer - 1);
                    if (newLayer !== currentLayer) {
                        visualizationState.layer = newLayer;
                        layerSelect.value = newLayer;

                        // Update slider and display
                        layerSlider.value = newLayer;
                        if (layerDisplay) {
                            layerDisplay.textContent = newLayer;
                        }

                        updateVisualization();
                    }
                }
                break;

            case 'ArrowRight':
                // Next layer
                if (!layerSelect.disabled && layerSlider && !layerSlider.disabled) {
                    e.preventDefault();
                    const currentLayer = parseInt(visualizationState.layer);
                    const maxLayer = data.shape[0] - 1;
                    const newLayer = Math.min(maxLayer, currentLayer + 1);
                    if (newLayer !== currentLayer) {
                        visualizationState.layer = newLayer;
                        layerSelect.value = newLayer;

                        // Update slider and display
                        layerSlider.value = newLayer;
                        if (layerDisplay) {
                            layerDisplay.textContent = newLayer;
                        }

                        updateVisualization();
                    }
                }
                break;
        }
    };

    // Add listener
    document.addEventListener('keydown', window.visualizationKeyHandler);
}

// ===== TOKEN IMPORTANCE FUNCTIONS =====

// Mode switching
function onVisualizationModeChange(mode) {
    visualizationState.mode = mode;

    // Hide all containers
    document.getElementById('heatmap-container').style.display = 'none';
    document.getElementById('token-importance-container').style.display = 'none';
    document.getElementById('model-comparison-container').style.display = 'none';
    document.getElementById('timeline-container').style.display = 'none';

    // Hide/show heatmap controls
    const heatmapControls = ['layer-select', 'head-select', 'resolution-select', 'aggregation-select'];
    const controlDisplay = mode === 'heatmap' ? 'flex' : 'none';
    heatmapControls.forEach(id => {
        document.getElementById(id).parentElement.style.display = controlDisplay;
    });

    // Show appropriate container and initialize
    if (mode === 'heatmap') {
        document.getElementById('heatmap-container').style.display = 'flex';
        updateVisualization();
    } else if (mode === 'token-importance') {
        document.getElementById('token-importance-container').style.display = 'block';
        initializeTokenImportance();
    } else if (mode === 'model-comparison') {
        document.getElementById('model-comparison-container').style.display = 'block';
        initializeTextComparison();
    } else if (mode === 'timeline') {
        document.getElementById('timeline-container').style.display = 'block';
        initializeTimeline();
    }
}

// Initialize token importance view
function initializeTokenImportance() {
}

// Initialize token importance view
function initializeTokenImportance() {
    const data = visualizationState.data;

    console.log('Initializing token importance, data:', data);

    if (!data) {
        console.error('No data available');
        return;
    }

    if (!data.tokens) {
        console.error('No tokens in data');
        showVisualizationError('Tokens not available. Please re-analyze the text.');
        return;
    }

    console.log('Number of tokens:', data.tokens.length);
    console.log('Shape:', data.shape);

    // Populate layer checkboxes
    populateLayerCheckboxes(data.shape[0]);

    // Populate head checkboxes
    populateHeadCheckboxes(data.shape[1]);

    // Render tokens
    renderTokens(data.tokens);
}

// Populate layer checkboxes
function populateLayerCheckboxes(numLayers) {
    const container = document.getElementById('layer-checkboxes');
    if (!container) {
        console.error('layer-checkboxes container not found');
        return;
    }

    console.log('Populating', numLayers, 'layers');
    container.innerHTML = '';

    // Add select/deselect all buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'checkbox-buttons';

    const selectAllBtn = document.createElement('button');
    selectAllBtn.textContent = 'Select All';
    selectAllBtn.className = 'checkbox-button';
    selectAllBtn.onclick = () => toggleAllLayers(true);

    const deselectAllBtn = document.createElement('button');
    deselectAllBtn.textContent = 'Deselect All';
    deselectAllBtn.className = 'checkbox-button';
    deselectAllBtn.onclick = () => toggleAllLayers(false);

    buttonContainer.appendChild(selectAllBtn);
    buttonContainer.appendChild(deselectAllBtn);
    container.appendChild(buttonContainer);

    // Add hint if only one layer
    if (numLayers === 1) {
        const hint = document.createElement('p');
        hint.style.fontSize = '0.8rem';
        hint.style.color = 'var(--text-secondary)';
        hint.style.marginBottom = '0.5rem';
        hint.textContent = 'Tip: Set "Attention Layer" to -1 before analyzing to get all layers';
        container.appendChild(hint);
    }

    // Create a wrapper for the checkbox grid
    const checkboxGrid = document.createElement('div');
    checkboxGrid.className = 'checkbox-grid';

    for (let i = 0; i < numLayers; i++) {
        const label = document.createElement('label');
        label.className = 'checkbox-label';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = i;
        checkbox.id = `layer-check-${i}`;
        checkbox.checked = i === 0; // Default: first layer selected
        checkbox.onchange = () => updateLayerSelection();

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${i}`));
        checkboxGrid.appendChild(label);
    }

    container.appendChild(checkboxGrid);

    // Initialize selection
    visualizationState.selectedLayers = [0];
    console.log('Layer checkboxes populated');
}

// Populate head checkboxes
function populateHeadCheckboxes(numHeads) {
    const container = document.getElementById('head-checkboxes');
    if (!container) {
        console.error('head-checkboxes container not found');
        return;
    }

    console.log('Populating', numHeads, 'heads');
    container.innerHTML = '';

    // Add select/deselect all buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'checkbox-buttons';

    const selectAllBtn = document.createElement('button');
    selectAllBtn.textContent = 'Select All';
    selectAllBtn.className = 'checkbox-button';
    selectAllBtn.onclick = () => toggleAllHeads(true);

    const deselectAllBtn = document.createElement('button');
    deselectAllBtn.textContent = 'Deselect All';
    deselectAllBtn.className = 'checkbox-button';
    deselectAllBtn.onclick = () => toggleAllHeads(false);

    buttonContainer.appendChild(selectAllBtn);
    buttonContainer.appendChild(deselectAllBtn);
    container.appendChild(buttonContainer);

    // Create a wrapper for the checkbox grid
    const checkboxGrid = document.createElement('div');
    checkboxGrid.className = 'checkbox-grid';

    for (let i = 0; i < numHeads; i++) {
        const label = document.createElement('label');
        label.className = 'checkbox-label';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = i;
        checkbox.id = `head-check-${i}`;
        checkbox.checked = i === 0; // Default: first head selected
        checkbox.onchange = () => updateHeadSelection();

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${i}`));
        checkboxGrid.appendChild(label);
    }

    container.appendChild(checkboxGrid);

    // Initialize selection
    visualizationState.selectedHeads = [0];
    console.log('Head checkboxes populated');
}

// Update layer selection from checkboxes
function updateLayerSelection() {
    const checkboxes = document.querySelectorAll('#layer-checkboxes input[type="checkbox"]');
    visualizationState.selectedLayers = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.value));

    // Re-render if range selected
    if (visualizationState.tokenRange) {
        updateTokenImportance();
    }
}

// Update head selection from checkboxes
function updateHeadSelection() {
    const checkboxes = document.querySelectorAll('#head-checkboxes input[type="checkbox"]');
    visualizationState.selectedHeads = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.value));

    // Re-render if range selected
    if (visualizationState.tokenRange) {
        updateTokenImportance();
    }
}

// Render tokens with click-and-drag selection
function renderTokens(tokens) {
    const container = document.getElementById('token-list');
    if (!container) {
        console.error('token-list container not found');
        return;
    }

    console.log('Rendering', tokens.length, 'tokens');
    console.log('First few tokens:', tokens.slice(0, 5));

    container.innerHTML = '';

    // Check if we have structured text to display
    if (currentAnalyzedTextStructure && currentAnalyzedTextStructure.hasContext) {
        // Render structured sections instead of plain token list
        renderStructuredTokenDisplay(container, tokens);
        return;
    }

    // Determine visible token range based on section filter
    const boundaries = visualizationState.sectionFilter.tokenBoundaries;
    const focusMode = visualizationState.sectionFilter.focusMode;

    let visibleTokens = tokens;
    let tokenOffset = 0;  // Offset to maintain correct indices

    if (boundaries && focusMode) {
        // Filter tokens to show only selected section
        const { start, end } = getSectionRange(boundaries, focusMode);
        visibleTokens = tokens.slice(start, end);
        tokenOffset = start;
    }

    // Original token rendering with drag selection
    let isDragging = false;
    let startIdx = null;

    visibleTokens.forEach((token, displayIdx) => {
        const actualIdx = displayIdx + tokenOffset;  // Real index in full token array
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = token;
        span.dataset.index = actualIdx;

        // Mouse down: start selection
        span.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
            startIdx = actualIdx;
            visualizationState.tokenRange = { start: actualIdx, end: actualIdx };
            updateTokenSelection();
        });

        // Mouse enter: extend selection while dragging
        span.addEventListener('mouseenter', () => {
            if (isDragging) {
                visualizationState.tokenRange = {
                    start: Math.min(startIdx, actualIdx),
                    end: Math.max(startIdx, actualIdx)
                };
                updateTokenSelection();
            }
        });

        container.appendChild(span);
    });

    console.log('Tokens rendered, container children:', container.children.length);

    // Mouse up: end selection
    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            // Calculate importance
            updateTokenImportance();
        }
    });
}

// Update visual selection
function updateTokenSelection() {
    const tokens = document.querySelectorAll('#token-list .token');
    const range = visualizationState.tokenRange;

    tokens.forEach((token, idx) => {
        if (range && idx >= range.start && idx <= range.end) {
            token.classList.add('selected');
        } else {
            token.classList.remove('selected');
        }
    });

    // Update info display
    document.getElementById('selection-info').style.display = 'block';
    document.getElementById('selection-range').textContent =
        `${range.start} - ${range.end}`;
}

// Calculate and display token importance
function updateTokenImportance() {
    const data = visualizationState.data;
    const range = visualizationState.tokenRange;
    const mode = visualizationState.importanceMode;

    if (!data || !range) return;

    // Check if we have selected layers and heads
    if (visualizationState.selectedLayers.length === 0 || visualizationState.selectedHeads.length === 0) {
        return;
    }

    // Calculate importance scores
    const scores = calculateTokenImportance(
        data.attention_pattern,
        range,
        visualizationState.selectedLayers,
        visualizationState.selectedHeads,
        mode
    );

    // Apply colors to tokens
    applyTokenColors(scores);
}

// Calculate importance scores from attention
function calculateTokenImportance(attention, range, layers, heads, mode) {
    const seqLen = attention[0][0].length;
    const scores = new Array(seqLen).fill(0);
    let count = 0;

    // Aggregate across selected layers and heads
    for (const layer of layers) {
        for (const head of heads) {
            const attnMatrix = attention[layer][head];

            if (mode === 'incoming') {
                // Previous → Selected: Look at queries of selected, keys of previous
                // Average attention from previous tokens to selected range
                for (let i = range.start; i <= range.end; i++) {
                    for (let j = 0; j < i; j++) {
                        scores[j] += attnMatrix[i][j];
                    }
                }
            } else {
                // Selected → Following: Look at queries of following, keys of selected
                // Show importance of selected tokens FOR following tokens
                for (let i = range.end + 1; i < seqLen; i++) {
                    for (let j = range.start; j <= range.end; j++) {
                        scores[i] += attnMatrix[i][j];  // Accumulate for following token i
                    }
                }
            }

            count++;
        }
    }

    // Normalize by count and selected range size
    const rangeSize = range.end - range.start + 1;
    for (let i = 0; i < seqLen; i++) {
        scores[i] /= (count * rangeSize);
    }

    return scores;
}

// Calculate word-level importance scores from attention
// Apply colors to tokens based on scores
function applyTokenColors(scores) {
    const tokens = document.querySelectorAll('#token-list .token');
    const isDark = document.body.getAttribute('data-theme') === 'dark';

    // Normalize scores to 0-1
    const maxScore = Math.max(...scores);
    const normalizedScores = scores.map(s => maxScore > 0 ? s / maxScore : 0);

    tokens.forEach((token, idx) => {
        const value = normalizedScores[idx];
        // Apply power function like heatmap
        const adjustedValue = Math.pow(value, 0.5);
        const color = getHeatmapColor(adjustedValue, isDark);
        token.style.backgroundColor = color;
    });
}

// Handler for importance mode change
function onImportanceModeChange(mode) {
    visualizationState.importanceMode = mode;
    if (visualizationState.tokenRange) {
        updateTokenImportance();
    }
}

// Toggle all layers
function toggleAllLayers(select) {
    const checkboxes = document.querySelectorAll('#layer-checkboxes input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = select;
    });
    updateLayerSelection();
}

// Toggle all heads
function toggleAllHeads(select) {
    const checkboxes = document.querySelectorAll('#head-checkboxes input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = select;
    });
    updateHeadSelection();
}

// ===== TEXT COMPARISON FUNCTIONS =====

// Initialize text comparison mode
async function initializeTextComparison() {
    const data = visualizationState.data;
    if (!data) return;

    // Store original text and data
    visualizationState.comparison.originalText = document.getElementById('analyze-text').value;
    visualizationState.comparison.originalData = data;

    // Fetch model names
    try {
        const models = await getModels();
        visualizationState.comparison.primaryModel = models.primary_model;
        visualizationState.comparison.compareModel = models.compare_model;

        // Update headers with model names
        const originalHeader = document.querySelector('#model-comparison-container .comparison-column:nth-child(1) h3');
        const modifiedHeader = document.querySelector('#model-comparison-container .comparison-column:nth-child(2) h3');

        if (originalHeader) originalHeader.textContent = models.primary_model || 'Primary Model';
        if (modifiedHeader) modifiedHeader.textContent = models.compare_model || 'Comparison Model';
    } catch (error) {
        console.error('Failed to fetch model names:', error);
    }

    // Populate checkboxes
    populateComparisonCheckboxes(data.shape[0], data.shape[1]);

    // Automatically trigger comparison
    await compareAttentionAutomatic();
}

// Populate checkboxes for comparison mode
function populateComparisonCheckboxes(numLayers, numHeads) {
    // Layer checkboxes
    const layerContainer = document.getElementById('comparison-layer-checkboxes');
    layerContainer.innerHTML = '';

    const layerButtons = document.createElement('div');
    layerButtons.className = 'checkbox-buttons';
    layerButtons.innerHTML = `
        <button class="checkbox-button" onclick="toggleComparisonLayers(true)">Select All</button>
        <button class="checkbox-button" onclick="toggleComparisonLayers(false)">Deselect All</button>
    `;
    layerContainer.appendChild(layerButtons);

    // Create a wrapper for the checkbox grid
    const layerGrid = document.createElement('div');
    layerGrid.className = 'checkbox-grid';

    for (let i = 0; i < numLayers; i++) {
        const label = document.createElement('label');
        label.className = 'checkbox-label';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = i;
        checkbox.checked = i === 0;
        checkbox.onchange = updateComparisonLayerSelection;
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${i}`));
        layerGrid.appendChild(label);
    }

    layerContainer.appendChild(layerGrid);

    // Head checkboxes
    const headContainer = document.getElementById('comparison-head-checkboxes');
    headContainer.innerHTML = '';

    const headButtons = document.createElement('div');
    headButtons.className = 'checkbox-buttons';
    headButtons.innerHTML = `
        <button class="checkbox-button" onclick="toggleComparisonHeads(true)">Select All</button>
        <button class="checkbox-button" onclick="toggleComparisonHeads(false)">Deselect All</button>
    `;
    headContainer.appendChild(headButtons);

    // Create a wrapper for the checkbox grid
    const headGrid = document.createElement('div');
    headGrid.className = 'checkbox-grid';

    for (let i = 0; i < numHeads; i++) {
        const label = document.createElement('label');
        label.className = 'checkbox-label';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = i;
        checkbox.checked = i === 0;
        checkbox.onchange = updateComparisonHeadSelection;
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${i}`));
        headGrid.appendChild(label);
    }

    headContainer.appendChild(headGrid);

    // Initialize selections
    visualizationState.comparison.selectedLayers = [0];
    visualizationState.comparison.selectedHeads = [0];
}

// Update layer selection for comparison
function updateComparisonLayerSelection() {
    const checkboxes = document.querySelectorAll('#comparison-layer-checkboxes input[type="checkbox"]');
    visualizationState.comparison.selectedLayers = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.value));

    // Re-render if comparison data exists
    if (visualizationState.comparison.modifiedData) {
        renderComparison();
    }
}

// Update head selection for comparison
function updateComparisonHeadSelection() {
    const checkboxes = document.querySelectorAll('#comparison-head-checkboxes input[type="checkbox"]');
    visualizationState.comparison.selectedHeads = Array.from(checkboxes)
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.value));

    // Re-render if comparison data exists
    if (visualizationState.comparison.modifiedData) {
        renderComparison();
    }
}

// Toggle all comparison layers
function toggleComparisonLayers(select) {
    const checkboxes = document.querySelectorAll('#comparison-layer-checkboxes input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = select);
    updateComparisonLayerSelection();
}

// Toggle all comparison heads
function toggleComparisonHeads(select) {
    const checkboxes = document.querySelectorAll('#comparison-head-checkboxes input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = select);
    updateComparisonHeadSelection();
}

// Compare attention (triggered by button)
// Automatic comparison on mode switch
async function compareAttentionAutomatic() {
    const compareText = visualizationState.comparison.originalText;

    if (!compareText) {
        console.error('No text to compare');
        return;
    }

    try {
        // Get attention layer from main analyze input
        const attnLayer = parseInt(document.getElementById('attn-layer').value);

        // Call COMPARE API with comparison model
        const data = await callCompareAPI(compareText, attnLayer);
        visualizationState.comparison.modifiedData = data;

        // Render comparison
        renderComparison();

        // Show comparison display
        document.getElementById('comparison-display').style.display = 'grid';
    } catch (error) {
        alert('Error analyzing with comparison model: ' + error.message);
    }
}

// Render side-by-side comparison
function renderComparison() {
    const originalData = visualizationState.comparison.originalData;
    const modifiedData = visualizationState.comparison.modifiedData;
    const layers = visualizationState.comparison.selectedLayers;
    const heads = visualizationState.comparison.selectedHeads;

    if (!originalData || !modifiedData || layers.length === 0 || heads.length === 0) {
        return;
    }

    // Calculate average attention for each token in both sequences
    const originalScores = calculateAverageAttention(
        originalData.attention_pattern,
        layers,
        heads
    );
    const modifiedScores = calculateAverageAttention(
        modifiedData.attention_pattern,
        layers,
        heads
    );

    // Render original tokens
    renderComparisonTokens(
        'original-tokens',
        originalData.tokens,
        originalScores
    );

    // Render modified tokens
    renderComparisonTokens(
        'modified-tokens',
        modifiedData.tokens,
        modifiedScores
    );
}

// Calculate average attention for a sequence
function calculateAverageAttention(attention, layers, heads) {
    const seqLen = attention[0][0].length;
    const scores = new Array(seqLen).fill(0);

    // Average incoming attention (how much each token is attended to)
    for (const layer of layers) {
        for (const head of heads) {
            const attnMatrix = attention[layer][head];
            for (let col = 0; col < seqLen; col++) {
                for (let row = 0; row < seqLen; row++) {
                    scores[col] += attnMatrix[row][col];
                }
            }
        }
    }

    // Normalize
    const count = layers.length * heads.length * seqLen;
    return scores.map(s => s / count);
}

// Render tokens for comparison (read-only, no selection)
function renderComparisonTokens(containerId, tokens, scores) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const isDark = document.body.getAttribute('data-theme') === 'dark';

    // Normalize scores
    const maxScore = Math.max(...scores);
    const normalizedScores = scores.map(s => maxScore > 0 ? s / maxScore : 0);

    tokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = token;

        // Apply color based on attention
        const value = normalizedScores[idx];
        const adjustedValue = Math.pow(value, 0.5);
        const color = getHeatmapColor(adjustedValue, isDark);
        span.style.backgroundColor = color;

        container.appendChild(span);
    });
}

// ============================================================================
// TIMELINE MODE FUNCTIONS
// ============================================================================

// Initialize timeline view
function initializeTimeline() {
    const data = visualizationState.data;
    if (!data) return;

    const numLayers = data.shape[0];
    const numHeads = data.shape[1];

    // Setup layer slider
    const slider = document.getElementById('timeline-layer-slider');
    slider.max = numLayers - 1;
    slider.value = 0;
    visualizationState.timeline.currentLayer = 0;
    document.getElementById('timeline-layer-display').textContent = '0';

    // Setup head selector
    const headSelect = document.getElementById('timeline-head-select');
    headSelect.innerHTML = '';
    for (let i = 0; i < numHeads; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Head ${i}`;
        headSelect.appendChild(option);
    }

    // Render initial timeline
    renderTimeline();
}

// Handle layer slider change
function onTimelineLayerChange(layer) {
    visualizationState.timeline.currentLayer = parseInt(layer);
    document.getElementById('timeline-layer-display').textContent = layer;
    renderTimeline();
}

// Handle aggregation mode change
function onTimelineAggregationChange(mode) {
    visualizationState.timeline.aggregationMode = mode;

    // Show/hide head selector based on mode
    const headSelectGroup = document.getElementById('timeline-head-select-group');
    if (mode === 'single-head') {
        headSelectGroup.style.display = 'flex';
    } else {
        headSelectGroup.style.display = 'none';
    }

    renderTimeline();
}

// Handle head selection change
function onTimelineHeadChange(head) {
    visualizationState.timeline.selectedHead = parseInt(head);
    renderTimeline();
}

// Main timeline rendering dispatcher
function renderTimeline() {
    const data = visualizationState.data;
    const { currentLayer, aggregationMode, selectedHead } = visualizationState.timeline;

    if (!data) return;

    if (aggregationMode === 'mean-heads') {
        renderMeanHeadsTimeline(data, currentLayer);
    } else if (aggregationMode === 'single-head') {
        renderSingleHeadTimeline(data, currentLayer, selectedHead);
    } else if (aggregationMode === 'all-heads') {
        renderAllHeadsTimeline(data, currentLayer);
    }
}

// Render timeline with mean attention across all heads
function renderMeanHeadsTimeline(data, layer) {
    const tokens = data.tokens;
    const numHeads = data.shape[1];
    const numTokens = data.num_tokens;

    // Calculate mean attention across all heads for this layer
    const meanAttention = new Array(numTokens).fill(0).map(() => new Array(numTokens).fill(0));

    for (let head = 0; head < numHeads; head++) {
        const headAttention = data.attention_pattern[layer][head];
        for (let i = 0; i < numTokens; i++) {
            for (let j = 0; j < numTokens; j++) {
                meanAttention[i][j] += headAttention[i][j];
            }
        }
    }

    // Average by number of heads
    for (let i = 0; i < numTokens; i++) {
        for (let j = 0; j < numTokens; j++) {
            meanAttention[i][j] /= numHeads;
        }
    }

    // Calculate incoming attention for each token
    const incomingScores = calculateIncomingAttention(meanAttention, numTokens);

    // Reset timeline-tokens container to flex display
    const timelineTokensContainer = document.getElementById('timeline-tokens');
    timelineTokensContainer.style.display = 'flex';
    timelineTokensContainer.style.gridTemplateColumns = '';

    // Render colored tokens
    renderColoredTokens(tokens, incomingScores, 'timeline-tokens');
}

// Render timeline for a single head
function renderSingleHeadTimeline(data, layer, head) {
    const tokens = data.tokens;
    const numTokens = data.num_tokens;

    // Get attention for specific layer and head
    const attention = data.attention_pattern[layer][head];

    // Calculate incoming attention for each token
    const incomingScores = calculateIncomingAttention(attention, numTokens);

    // Reset timeline-tokens container to flex display
    const timelineTokensContainer = document.getElementById('timeline-tokens');
    timelineTokensContainer.style.display = 'flex';
    timelineTokensContainer.style.gridTemplateColumns = '';

    // Render colored tokens
    renderColoredTokens(tokens, incomingScores, 'timeline-tokens');
}

// Render timeline showing all heads in a grid
function renderAllHeadsTimeline(data, layer) {
    const tokens = data.tokens;
    const numHeads = data.shape[1];
    const numTokens = data.num_tokens;

    const timelineTokensContainer = document.getElementById('timeline-tokens');
    timelineTokensContainer.innerHTML = '';
    timelineTokensContainer.style.display = 'grid';

    // Calculate grid dimensions (try for roughly square grid)
    const cols = Math.ceil(Math.sqrt(numHeads));
    timelineTokensContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    timelineTokensContainer.style.gap = '1rem';

    // Render each head in a grid cell
    for (let head = 0; head < numHeads; head++) {
        const headContainer = document.createElement('div');
        headContainer.className = 'timeline-head-cell';

        const headLabel = document.createElement('div');
        headLabel.className = 'timeline-head-label';
        headLabel.textContent = `Head ${head}`;
        headContainer.appendChild(headLabel);

        const headTokens = document.createElement('div');
        headTokens.className = 'token-list-readonly';
        headTokens.id = `timeline-head-${head}`;
        headContainer.appendChild(headTokens);

        timelineTokensContainer.appendChild(headContainer);

        // Calculate and render this head's attention
        const attention = data.attention_pattern[layer][head];
        const incomingScores = calculateIncomingAttention(attention, numTokens);
        renderColoredTokens(tokens, incomingScores, `timeline-head-${head}`);
    }
}

// Calculate incoming attention for each token (column sum)
function calculateIncomingAttention(attentionMatrix, numTokens) {
    const scores = new Array(numTokens).fill(0);

    // Sum attention received by each token (column sum)
    for (let toToken = 0; toToken < numTokens; toToken++) {
        for (let fromToken = 0; fromToken < numTokens; fromToken++) {
            scores[toToken] += attentionMatrix[fromToken][toToken];
        }
    }

    return scores;
}

// Render structured token display with sections
function renderStructuredTokenDisplay(container, allTokens) {
    container.innerHTML = '';
    container.style.display = 'block';

    // We need to split tokens based on the original structure
    // For now, just display all tokens as response since we have the full generated text
    const responseSection = document.createElement('div');
    responseSection.className = 'token-section-wrapper';

    // Add section label
    const label = document.createElement('div');
    label.className = 'token-section-label';
    label.textContent = 'Full Text (click and drag to select)';
    responseSection.appendChild(label);

    // Add tokens container
    const tokensContainer = document.createElement('div');
    tokensContainer.className = 'token-section-content';
    responseSection.appendChild(tokensContainer);

    container.appendChild(responseSection);

    // Render tokens with drag selection in the section
    let isDragging = false;
    let startIdx = null;

    allTokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = token;
        span.dataset.index = idx;

        span.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
            startIdx = idx;
            visualizationState.tokenRange = { start: idx, end: idx };
            updateTokenSelection();
        });

        span.addEventListener('mouseenter', () => {
            if (isDragging) {
                const endIdx = idx;
                visualizationState.tokenRange = {
                    start: Math.min(startIdx, endIdx),
                    end: Math.max(startIdx, endIdx)
                };
                updateTokenSelection();
            }
        });

        tokensContainer.appendChild(span);
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            updateTokenImportance();
        }
    });
}

// Render tokens with colors based on attention scores
function renderColoredTokens(tokens, scores, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    container.style.display = 'flex';
    container.style.flexWrap = 'wrap';
    container.style.gap = '0.25rem';

    // Normalize scores to [0, 1] range
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const range = maxScore - minScore;

    const isDark = document.body.getAttribute('data-theme') === 'dark';

    tokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = token;

        // Normalize and apply power function for better visual distribution
        const normalizedScore = range > 0 ? (scores[idx] - minScore) / range : 0;
        const enhancedScore = Math.pow(normalizedScore, 0.5);

        // Get color from heatmap color function (reuse existing gradient)
        const bgColor = getHeatmapColor(enhancedScore, isDark);
        span.style.backgroundColor = bgColor;

        // Choose text color based on background brightness
        const textColor = enhancedScore > 0.5 ? '#000000' : (isDark ? '#e5e7eb' : '#374151');
        span.style.color = textColor;

        // Add tooltip showing attention score
        span.title = `Incoming attention: ${scores[idx].toFixed(3)}`;

        container.appendChild(span);
    });
}

// ===== Section Filtering Utilities =====

/**
 * Get token range for a section
 * @param {Object} boundaries - Token boundaries object
 * @param {string} section - Section name ('document', 'prompt', 'response')
 * @returns {Object} Range with start and end indices
 */
function getSectionRange(boundaries, section) {
    const start = boundaries[`${section}_start`];
    const end = boundaries[`${section}_end`];
    return { start, end };
}

/**
 * Set section focus mode for token filtering
 * @param {Event} event - Click event
 * @param {string|null} section - Section to focus on, or null for all
 */
function setSectionFocus(event, section) {
    if (event) event.preventDefault();

    if (!visualizationState.sectionFilter.tokenBoundaries) {
        console.warn('No token boundaries available for section filtering');
        return;
    }

    visualizationState.sectionFilter.focusMode = section;

    // Update button states
    document.querySelectorAll('.section-filter-buttons .filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if ((section === null && btn.dataset.section === 'all') ||
            btn.dataset.section === section) {
            btn.classList.add('active');
        }
    });

    // Re-render tokens with filter applied
    if (visualizationState.mode === 'token-importance' && visualizationState.data) {
        renderTokens(visualizationState.data.tokens);
    }
}

// Render delta tokens showing attention changes
