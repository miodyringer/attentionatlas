// Global state
let currentGeneratedText = '';
let currentGeneratedResponse = ''; // Store only the response part
let currentAttentionData = null;
let uploadedDocuments = []; // Store multiple uploaded documents
let currentAnalyzedTextStructure = null; // Store structure of analyzed text
let currentTokenBoundaries = null; // Store token boundaries for section-aware visualization

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = '2.8rem'; // Reset to initial height (matches new padding)
    if (textarea.scrollHeight > textarea.clientHeight) {
        textarea.style.height = textarea.scrollHeight + 'px';
    }
}

// Initialize auto-resize for textareas
window.addEventListener('DOMContentLoaded', () => {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            autoResizeTextarea(this);
        });
        // Don't initialize height on load - let CSS handle it
    });

    // Initialize file upload
    setupFileUpload();
});

// Tab switching
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Generate text
async function generateText() {
    const userPrompt = document.getElementById('prompt').value.trim();
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxTokens = parseInt(document.getElementById('max-tokens').value);

    if (!userPrompt && uploadedDocuments.length === 0) {
        showError('generate-error', 'Please enter a prompt or upload a document');
        return;
    }

    // Combine all uploaded documents
    const combinedDocumentContext = uploadedDocuments.length > 0
        ? uploadedDocuments.map(doc => doc.content).join('\n\n---\n\n')
        : null;

    hideError('generate-error');
    document.getElementById('generate-loading').classList.add('active');
    document.getElementById('generate-output-card').style.display = 'none';

    // Scroll to loading spinner immediately
    setTimeout(() => {
        document.getElementById('generate-loading').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }, 100);

    try {
        // Send document context separately to backend
        const data = await callGenerateAPI(userPrompt, temperature, maxTokens, combinedDocumentContext);
        currentGeneratedText = data.answer;
        currentGeneratedResponse = data.metadata.new_text; // Store only the new response tokens

        // Store token boundaries if available
        if (data.metadata.token_boundaries) {
            currentTokenBoundaries = data.metadata.token_boundaries;
        }

        // Display output with sections if document context exists
        const outputBox = document.getElementById('generated-text');
        outputBox.innerHTML = ''; // Clear previous content

        if (data.metadata.has_document_context && combinedDocumentContext) {
            // Create structured display
            const contextSection = document.createElement('div');
            contextSection.className = 'context-section';
            contextSection.innerHTML = `<div class="section-label">Document Context (${data.metadata.context_token_count} tokens)</div><div class="context-preview collapsed">${truncateText(combinedDocumentContext, 150)}</div><button class="expand-button" onclick="toggleContextPreview(this)">Show full context</button>`;
            outputBox.appendChild(contextSection);

            const promptSection = document.createElement('div');
            promptSection.className = 'prompt-section';
            promptSection.innerHTML = `<div class="section-label">Your Question (${data.metadata.user_prompt_token_count} tokens)</div><div class="prompt-text">${userPrompt}</div>`;
            outputBox.appendChild(promptSection);

            const responseSection = document.createElement('div');
            responseSection.className = 'response-section';
            responseSection.innerHTML = `<div class="section-label">Model Response (${data.metadata.generated_tokens} tokens)</div><div class="response-text">${currentGeneratedResponse}</div>`;
            outputBox.appendChild(responseSection);
        } else {
            // Simple display for non-context generations
            outputBox.textContent = data.answer;
            currentGeneratedResponse = data.answer;
        }

        const metadata = data.metadata;
        document.getElementById('generate-metadata').innerHTML = `
            <div class="metadata-item"><strong>Model:</strong> ${data.model}</div>
            <div class="metadata-item"><strong>Temperature:</strong> ${metadata.temperature}</div>
            <div class="metadata-item"><strong>Total Tokens:</strong> ${metadata.total_tokens}</div>
            <div class="metadata-item"><strong>Prefill Tokens:</strong> ${metadata.prefill_tokens || metadata.prompt_tokens}</div>
            ${metadata.has_document_context ? `<div class="metadata-item"><strong>Context:</strong> ${metadata.context_token_count} tokens</div>` : ''}
            ${metadata.has_document_context ? `<div class="metadata-item"><strong>Question:</strong> ${metadata.user_prompt_token_count} tokens</div>` : ''}
            <div class="metadata-item"><strong>Response:</strong> ${metadata.generated_tokens} tokens</div>
        `;

        document.getElementById('generate-output-card').style.display = 'block';
    } catch (error) {
        showError('generate-error', `Error: ${error.message}`);
    } finally {
        document.getElementById('generate-loading').classList.remove('active');
    }
}

// Use generated text in visualization tab
function useGeneratedText() {
    const analyzeTextarea = document.getElementById('analyze-text');
    const structuredDiv = document.getElementById('analyze-text-structured');
    const noTextMessage = document.getElementById('no-text-message');
    const analyzeButton = document.getElementById('analyze-button');

    const combinedDocumentContext = uploadedDocuments.map(doc => doc.content).join('\n\n---\n\n');

    // Enable the analyze button
    analyzeButton.disabled = false;

    // Hide "no text" message
    if (noTextMessage) {
        noTextMessage.style.display = 'none';
    }

    // Carry over the text structure for visualization
    if (uploadedDocuments.length > 0) {
        currentAnalyzedTextStructure = {
            hasContext: true,
            documentContext: combinedDocumentContext,
            userPrompt: document.getElementById('prompt').value.trim(),
            response: currentGeneratedResponse, // Use only the response, not full text
            boundaries: currentTokenBoundaries // Include token boundaries
        };

        // Show structured display
        structuredDiv.style.display = 'block';
        structuredDiv.innerHTML = '';

        // Build structured sections
        const contextSection = document.createElement('div');
        contextSection.className = 'context-section';
        contextSection.innerHTML = `<div class="section-label">Document Context</div><div class="context-preview collapsed">${truncateText(combinedDocumentContext, 150)}</div><button class="expand-button" onclick="toggleAnalyzeContextPreview(this)">Show full context</button>`;
        structuredDiv.appendChild(contextSection);

        const promptSection = document.createElement('div');
        promptSection.className = 'prompt-section';
        promptSection.innerHTML = `<div class="section-label">User Question</div><div class="prompt-text">${currentAnalyzedTextStructure.userPrompt}</div>`;
        structuredDiv.appendChild(promptSection);

        const responseSection = document.createElement('div');
        responseSection.className = 'response-section';
        responseSection.innerHTML = `<div class="section-label">Model Response</div><div class="response-text">${currentGeneratedResponse}</div>`;
        structuredDiv.appendChild(responseSection);

        // Store full text in textarea (hidden) for analyze function
        analyzeTextarea.value = currentGeneratedText;
    } else {
        currentAnalyzedTextStructure = null;
        // Show simple display
        structuredDiv.style.display = 'block';
        structuredDiv.innerHTML = `<div class="response-section"><div class="section-label">Generated Text</div><div class="response-text">${currentGeneratedText}</div></div>`;
        analyzeTextarea.value = currentGeneratedText;
    }

    // Switch tabs first so the content is visible
    document.querySelectorAll('.tab').forEach((tab, index) => {
        tab.classList.remove('active');
        if (index === 1) tab.classList.add('active');
    });
    document.querySelectorAll('.tab-content').forEach((content, index) => {
        content.classList.remove('active');
        if (index === 1) content.classList.add('active');
    });
}

// Analyze text
async function analyzeText() {
    const text = document.getElementById('analyze-text').value.trim();
    const attnLayer = parseInt(document.getElementById('attn-layer').value);

    if (!text) {
        showError('visualize-error', 'Please generate text first in the Generate tab');
        return;
    }

    hideError('visualize-error');
    document.getElementById('analyze-loading').classList.add('active');
    document.getElementById('analyze-output-card').style.display = 'none';

    try {
        const data = await callAnalyzeAPI(text, attnLayer);
        currentAttentionData = data;

        const numLayers = data.shape[0];
        const numHeads = data.shape[1];
        const seqLen = data.shape[2];

        document.getElementById('analyze-metadata').innerHTML = `
            <div class="metadata-item"><strong>Shape:</strong> [${data.shape.join(', ')}]</div>
            <div class="metadata-item"><strong>Layers:</strong> ${numLayers}</div>
            <div class="metadata-item"><strong>Heads:</strong> ${numHeads}</div>
            <div class="metadata-item"><strong>Sequence Length:</strong> ${seqLen}</div>
            <div class="metadata-item"><strong>Total Tokens:</strong> ${data.num_tokens}</div>
            ${data.prefill_tokens ? `<div class="metadata-item"><strong>Prefill Tokens:</strong> ${data.prefill_tokens}</div>` : ''}
            ${data.prefill_tokens ? `<div class="metadata-item"><strong>Decode Tokens:</strong> ${data.num_tokens - data.prefill_tokens}</div>` : ''}
        `;

        document.getElementById('analyze-output-card').style.display = 'block';

        // Initialize visualization
        if (typeof initializeVisualization === 'function') {
            initializeVisualization(currentAttentionData);
        }
    } catch (error) {
        showError('visualize-error', `Error: ${error.message}`);
    } finally {
        document.getElementById('analyze-loading').classList.remove('active');
    }
}

// Utility functions
function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    errorElement.textContent = message;
    errorElement.classList.add('active');
}

function hideError(elementId) {
    const errorElement = document.getElementById(elementId);
    errorElement.classList.remove('active');
}

// File upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('file-upload-area');

    if (!fileInput || !uploadArea) return;

    // Click handler
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            // Handle multiple files
            Array.from(files).forEach(file => handleFile(file));
        }
    });
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        // Handle multiple files
        Array.from(files).forEach(file => handleFile(file));
    }
}

async function handleFile(file) {
    const filename = file.name;
    const extension = filename.split('.').pop().toLowerCase();

    // Check if file already uploaded
    if (uploadedDocuments.some(doc => doc.filename === filename)) {
        alert(`File "${filename}" is already uploaded.`);
        return;
    }

    try {
        let text = '';

        if (extension === 'pdf') {
            // Send to backend for PDF extraction
            text = await extractPDF(file);
        } else if (['txt', 'md', 'py', 'js', 'java', 'cpp', 'html', 'css'].includes(extension)) {
            // Read text files directly in browser
            text = await readTextFile(file);
        } else {
            alert('Unsupported file type. Please use txt, md, pdf, or code files.');
            return;
        }

        // Estimate token count (rough: ~4 chars per token)
        const estimatedTokens = Math.ceil(text.length / 4);

        // Warn if likely too long
        if (estimatedTokens > 512) {
            const proceed = confirm(
                `This document has ~${estimatedTokens} tokens. ` +
                `Performance is best with < 512 tokens. Continue anyway?`
            );
            if (!proceed) return;
        }

        // Add to uploaded documents array
        uploadedDocuments.push({
            filename: filename,
            content: text,
            size: file.size,
            tokens: estimatedTokens
        });

        // Update UI to show all uploaded files
        updateUploadedFilesList();

        console.log('Document loaded:', filename, 'with', estimatedTokens, 'tokens');

    } catch (error) {
        alert('Error reading file: ' + error.message);
    }
}

// Update the UI to show list of uploaded files
function updateUploadedFilesList() {
    const container = document.getElementById('files-container');
    const listElement = document.getElementById('uploaded-files-list');

    if (uploadedDocuments.length === 0) {
        listElement.style.display = 'none';
        return;
    }

    listElement.style.display = 'block';
    container.innerHTML = '';

    uploadedDocuments.forEach((doc, index) => {
        const fileCard = document.createElement('div');
        fileCard.className = 'uploaded-file-card';
        fileCard.innerHTML = `
            <div class="file-card-info">
                <div class="file-card-name">📄 ${doc.filename}</div>
                <div class="file-card-meta">${formatFileSize(doc.size)} | ~${doc.tokens} tokens</div>
            </div>
            <button class="remove-file-btn" onclick="removeFile(${index})" title="Remove file">×</button>
        `;
        container.appendChild(fileCard);
    });
}

// Remove a specific file
function removeFile(index) {
    uploadedDocuments.splice(index, 1);
    updateUploadedFilesList();
}

// Clear all uploaded files
function clearAllFiles() {
    if (uploadedDocuments.length === 0) return;
    if (confirm('Remove all uploaded documents?')) {
        uploadedDocuments = [];
        updateUploadedFilesList();
    }
}

function readTextFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

async function extractPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/extract_pdf`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`PDF extraction failed: ${response.status}`);
    }

    const result = await response.json();
    return result.text;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Helper functions for context display
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function toggleContextPreview(button) {
    const preview = button.previousElementSibling;
    const combinedDocumentContext = uploadedDocuments.map(doc => doc.content).join('\n\n---\n\n');
    if (preview.classList.contains('collapsed')) {
        preview.classList.remove('collapsed');
        preview.textContent = combinedDocumentContext;
        button.textContent = 'Hide full context';
    } else {
        preview.classList.add('collapsed');
        preview.textContent = truncateText(combinedDocumentContext, 150);
        button.textContent = 'Show full context';
    }
}

// Display analyzed text with structure
function displayAnalyzedText() {
    const container = document.getElementById('analyzed-text-display');
    container.innerHTML = '';

    if (currentAnalyzedTextStructure && currentAnalyzedTextStructure.hasContext) {
        // Structured display with context
        const contextSection = document.createElement('div');
        contextSection.className = 'context-section';
        contextSection.innerHTML = `<div class="section-label">Document Context</div><div class="context-preview collapsed">${truncateText(currentAnalyzedTextStructure.documentContext, 150)}</div><button class="expand-button" onclick="toggleAnalyzeContextPreview(this)">Show full context</button>`;
        container.appendChild(contextSection);

        const promptSection = document.createElement('div');
        promptSection.className = 'prompt-section';
        promptSection.innerHTML = `<div class="section-label">User Question</div><div class="prompt-text">${currentAnalyzedTextStructure.userPrompt}</div>`;
        container.appendChild(promptSection);

        const responseSection = document.createElement('div');
        responseSection.className = 'response-section';
        responseSection.innerHTML = `<div class="section-label">Model Response</div><div class="response-text">${currentAnalyzedTextStructure.response}</div>`;
        container.appendChild(responseSection);
    } else if (currentAttentionData && currentAttentionData.tokens) {
        // Show the analyzed text even if no structure
        const analyzedText = currentAttentionData.tokens.join('');
        const textSection = document.createElement('div');
        textSection.className = 'response-section';
        textSection.innerHTML = `<div class="section-label">Analyzed Text (${currentAttentionData.num_tokens} tokens)</div><div class="response-text">${analyzedText}</div>`;
        container.appendChild(textSection);
    }
}

function toggleAnalyzeContextPreview(button) {
    const preview = button.previousElementSibling;
    if (preview.classList.contains('collapsed')) {
        preview.classList.remove('collapsed');
        preview.textContent = currentAnalyzedTextStructure.documentContext;
        button.textContent = 'Hide full context';
    } else {
        preview.classList.add('collapsed');
        preview.textContent = truncateText(currentAnalyzedTextStructure.documentContext, 150);
        button.textContent = 'Show full context';
    }
}
