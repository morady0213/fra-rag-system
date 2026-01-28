/**
 * FRA RAG System - Frontend JavaScript
 * =====================================
 * This file handles all frontend interactions.
 * Backend API calls are stubbed - replace with actual endpoints.
 */

// ============================================
// Configuration
// ============================================

const API_BASE_URL = 'http://localhost:7860/api'; // Update with actual backend URL

// ============================================
// State Management
// ============================================

const state = {
    messages: [],
    sources: [],
    isLoading: false,
    theme: localStorage.getItem('theme') || 'light',
    settings: {
        language: 'ar',
        numSources: 5,
        hybridSearch: true,
        reranking: true,
        reactAgent: false,
        entityFilter: 'all',
        docTypeFilter: 'all',
        topicFilter: 'all'
    }
};

// ============================================
// DOM Elements
// ============================================

const elements = {
    // Chat
    chatMessages: document.getElementById('chatMessages'),
    userInput: document.getElementById('userInput'),
    sendBtn: document.getElementById('sendBtn'),
    
    // Sources
    sourcesPanel: document.getElementById('sourcesPanel'),
    sourcesContent: document.getElementById('sourcesContent'),
    showSourcesBtn: document.getElementById('showSourcesBtn'),
    closeSourcesBtn: document.getElementById('closeSourcesBtn'),
    
    // Settings
    language: document.getElementById('language'),
    numSources: document.getElementById('numSources'),
    numSourcesValue: document.getElementById('numSourcesValue'),
    hybridSearch: document.getElementById('hybridSearch'),
    reranking: document.getElementById('reranking'),
    reactAgent: document.getElementById('reactAgent'),
    entityFilter: document.getElementById('entityFilter'),
    docTypeFilter: document.getElementById('docTypeFilter'),
    topicFilter: document.getElementById('topicFilter'),
    
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadStatus: document.getElementById('uploadStatus'),
    
    // Buttons
    themeToggle: document.getElementById('themeToggle'),
    statsBtn: document.getElementById('statsBtn'),
    browseChunksBtn: document.getElementById('browseChunksBtn'),
    
    // Modals
    statsModal: document.getElementById('statsModal'),
    statsContent: document.getElementById('statsContent'),
    closeStatsModal: document.getElementById('closeStatsModal'),
    chunksModal: document.getElementById('chunksModal'),
    chunksContent: document.getElementById('chunksContent'),
    closeChunksModal: document.getElementById('closeChunksModal'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay')
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initEventListeners();
    loadSettings();
});

function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function updateThemeIcon() {
    elements.themeToggle.textContent = state.theme === 'dark' ? '☀️' : '🌙';
}

function initEventListeners() {
    // Chat
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.userInput.addEventListener('keydown', handleInputKeydown);
    elements.userInput.addEventListener('input', autoResizeTextarea);
    
    // Sources
    elements.showSourcesBtn.addEventListener('click', toggleSourcesPanel);
    elements.closeSourcesBtn.addEventListener('click', closeSourcesPanel);
    
    // Settings
    elements.language.addEventListener('change', updateSettings);
    elements.numSources.addEventListener('input', updateNumSources);
    elements.hybridSearch.addEventListener('change', updateSettings);
    elements.reranking.addEventListener('change', updateSettings);
    elements.reactAgent.addEventListener('change', updateSettings);
    elements.entityFilter.addEventListener('change', updateSettings);
    elements.docTypeFilter.addEventListener('change', updateSettings);
    elements.topicFilter.addEventListener('change', updateSettings);
    
    // Upload
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileUpload);
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Theme
    elements.themeToggle.addEventListener('click', toggleTheme);
    
    // Stats & Chunks
    elements.statsBtn.addEventListener('click', showStats);
    elements.browseChunksBtn.addEventListener('click', browseChunks);
    elements.closeStatsModal.addEventListener('click', () => closeModal('statsModal'));
    elements.closeChunksModal.addEventListener('click', () => closeModal('chunksModal'));
    
    // Close modals on backdrop click
    elements.statsModal.addEventListener('click', (e) => {
        if (e.target === elements.statsModal) closeModal('statsModal');
    });
    elements.chunksModal.addEventListener('click', (e) => {
        if (e.target === elements.chunksModal) closeModal('chunksModal');
    });
}

// ============================================
// Chat Functions
// ============================================

async function sendMessage() {
    const message = elements.userInput.value.trim();
    if (!message || state.isLoading) return;
    
    // Add user message
    addMessage(message, 'user');
    elements.userInput.value = '';
    autoResizeTextarea();
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        state.isLoading = true;
        elements.sendBtn.disabled = true;
        
        // API call to backend
        const response = await sendToBackend(message);
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        addMessage(response.answer, 'bot', response.sources);
        
        // Update sources panel
        updateSources(response.sources);
        
    } catch (error) {
        removeTypingIndicator();
        addMessage('حدث خطأ أثناء معالجة الطلب. يرجى المحاولة مرة أخرى.', 'bot');
        console.error('Error:', error);
    } finally {
        state.isLoading = false;
        elements.sendBtn.disabled = false;
    }
}

function addMessage(content, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = type === 'user' ? '👤' : '🤖';
    
    let feedbackHTML = '';
    if (type === 'bot') {
        feedbackHTML = `
            <div class="message-feedback">
                <button class="feedback-btn" data-feedback="positive" title="مفيد">👍</button>
                <button class="feedback-btn" data-feedback="negative" title="غير مفيد">👎</button>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <p>${formatMessage(content)}</p>
            ${feedbackHTML}
        </div>
    `;
    
    // Add feedback handlers
    if (type === 'bot') {
        const feedbackBtns = messageDiv.querySelectorAll('.feedback-btn');
        feedbackBtns.forEach(btn => {
            btn.addEventListener('click', () => handleFeedback(btn, content));
        });
    }
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Save to state
    state.messages.push({ content, type, sources, timestamp: new Date() });
}

function formatMessage(text) {
    // Convert markdown-like formatting to HTML
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    elements.chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const typingMessage = elements.chatMessages.querySelector('.typing-message');
    if (typingMessage) typingMessage.remove();
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    elements.userInput.style.height = 'auto';
    elements.userInput.style.height = Math.min(elements.userInput.scrollHeight, 150) + 'px';
}

// ============================================
// Sources Functions
// ============================================

function updateSources(sources) {
    if (!sources || sources.length === 0) {
        elements.sourcesContent.innerHTML = '<p class="text-muted">لا توجد مصادر متاحة</p>';
        return;
    }
    
    state.sources = sources;
    
    let html = '';
    sources.forEach((source, index) => {
        html += `
            <div class="source-card" data-index="${index}">
                <div class="source-card-header" onclick="toggleSourceCard(${index})">
                    <span class="source-number">[${index + 1}]</span>
                    <span class="source-name">📄 ${source.source || source.metadata?.source || 'مستند'}</span>
                </div>
                <div class="source-card-body">
                    ${source.content || source.text || ''}
                </div>
            </div>
        `;
    });
    
    elements.sourcesContent.innerHTML = html;
}

function toggleSourceCard(index) {
    const card = document.querySelector(`.source-card[data-index="${index}"]`);
    if (card) card.classList.toggle('open');
}

function toggleSourcesPanel() {
    elements.sourcesPanel.classList.toggle('open');
}

function closeSourcesPanel() {
    elements.sourcesPanel.classList.remove('open');
}

// ============================================
// Settings Functions
// ============================================

function updateSettings() {
    state.settings = {
        language: elements.language.value,
        numSources: parseInt(elements.numSources.value),
        hybridSearch: elements.hybridSearch.checked,
        reranking: elements.reranking.checked,
        reactAgent: elements.reactAgent.checked,
        entityFilter: elements.entityFilter.value,
        docTypeFilter: elements.docTypeFilter.value,
        topicFilter: elements.topicFilter.value
    };
    saveSettings();
}

function updateNumSources() {
    elements.numSourcesValue.textContent = elements.numSources.value;
    updateSettings();
}

function saveSettings() {
    localStorage.setItem('fraSettings', JSON.stringify(state.settings));
}

function loadSettings() {
    const saved = localStorage.getItem('fraSettings');
    if (saved) {
        state.settings = JSON.parse(saved);
        applySettings();
    }
}

function applySettings() {
    elements.language.value = state.settings.language;
    elements.numSources.value = state.settings.numSources;
    elements.numSourcesValue.textContent = state.settings.numSources;
    elements.hybridSearch.checked = state.settings.hybridSearch;
    elements.reranking.checked = state.settings.reranking;
    elements.reactAgent.checked = state.settings.reactAgent;
    elements.entityFilter.value = state.settings.entityFilter;
    elements.docTypeFilter.value = state.settings.docTypeFilter;
    elements.topicFilter.value = state.settings.topicFilter;
}

// ============================================
// Theme Functions
// ============================================

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('theme', state.theme);
    updateThemeIcon();
}

// ============================================
// Upload Functions
// ============================================

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    uploadFiles(files);
}

function handleFileUpload(e) {
    const files = e.target.files;
    uploadFiles(files);
}

async function uploadFiles(files) {
    if (files.length === 0) return;
    
    elements.uploadStatus.innerHTML = `<span class="text-warning">جاري الرفع... (${files.length} ملفات)</span>`;
    
    try {
        // TODO: Replace with actual API call
        const response = await uploadToBackend(files);
        elements.uploadStatus.innerHTML = `<span class="text-success">✓ تم رفع ${files.length} ملفات بنجاح</span>`;
    } catch (error) {
        elements.uploadStatus.innerHTML = `<span class="text-danger">✗ فشل في الرفع</span>`;
        console.error('Upload error:', error);
    }
}

// ============================================
// Modal Functions
// ============================================

function openModal(modalId) {
    document.getElementById(modalId).classList.add('open');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('open');
}

async function showStats() {
    openModal('statsModal');
    elements.statsContent.innerHTML = '<p>جاري التحميل...</p>';
    
    try {
        // TODO: Replace with actual API call
        const stats = await getStatsFromBackend();
        elements.statsContent.innerHTML = formatStats(stats);
    } catch (error) {
        elements.statsContent.innerHTML = '<p class="text-danger">فشل في تحميل الإحصائيات</p>';
    }
}

function formatStats(stats) {
    return `
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-value">${stats.totalDocuments || 0}</span>
                <span class="stat-label">المستندات</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.totalChunks || 0}</span>
                <span class="stat-label">الأجزاء</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.collectionSize || '0 MB'}</span>
                <span class="stat-label">حجم البيانات</span>
            </div>
        </div>
    `;
}

async function browseChunks() {
    openModal('chunksModal');
    elements.chunksContent.innerHTML = '<p>جاري التحميل...</p>';
    
    try {
        // TODO: Replace with actual API call
        const chunks = await getChunksFromBackend();
        elements.chunksContent.innerHTML = formatChunks(chunks);
    } catch (error) {
        elements.chunksContent.innerHTML = '<p class="text-danger">فشل في تحميل الأجزاء</p>';
    }
}

function formatChunks(chunks) {
    if (!chunks || chunks.length === 0) {
        return '<p class="text-muted">لا توجد أجزاء مخزنة</p>';
    }
    
    let html = '<div class="chunks-list">';
    chunks.forEach((chunk, i) => {
        html += `
            <div class="chunk-item">
                <div class="chunk-header">
                    <strong>جزء ${i + 1}</strong>
                    <span class="text-muted">${chunk.source || ''}</span>
                </div>
                <div class="chunk-text">${chunk.text?.substring(0, 200)}...</div>
            </div>
        `;
    });
    html += '</div>';
    return html;
}

// ============================================
// Feedback Functions
// ============================================

function handleFeedback(btn, messageContent) {
    const feedback = btn.dataset.feedback;
    
    // Toggle active state
    btn.classList.toggle('active');
    
    // Remove active from sibling
    const sibling = btn.parentElement.querySelector(`.feedback-btn:not([data-feedback="${feedback}"])`);
    if (sibling) sibling.classList.remove('active');
    
    // Save feedback
    saveFeedback(messageContent, feedback);
}

async function saveFeedback(message, feedback) {
    try {
        // TODO: Replace with actual API call
        console.log('Feedback saved:', { message, feedback });
    } catch (error) {
        console.error('Failed to save feedback:', error);
    }
}

// ============================================
// Loading Functions
// ============================================

function showLoading() {
    elements.loadingOverlay.classList.add('show');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('show');
}

// ============================================
// API Functions (STUB - Replace with actual calls)
// ============================================

/**
 * Send message to backend
 * @param {string} message - User message
 * @returns {Promise<{answer: string, sources: Array}>}
 */
async function sendToBackend(message) {
    // TODO: Replace with actual API call
    // Example:
    // const response = await fetch(`${API_BASE_URL}/chat`, {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify({
    //         message,
    //         ...state.settings
    //     })
    // });
    // return response.json();
    
    // MOCK RESPONSE - Remove this in production
    await new Promise(resolve => setTimeout(resolve, 1500));
    return {
        answer: `هذه إجابة تجريبية على سؤالك: "${message}"\n\nيرجى توصيل الواجهة الخلفية للحصول على إجابات حقيقية.`,
        sources: [
            { source: 'مستند-تجريبي-1.pdf', content: 'هذا نص تجريبي من المستند الأول...' },
            { source: 'مستند-تجريبي-2.docx', content: 'هذا نص تجريبي من المستند الثاني...' }
        ]
    };
}

/**
 * Upload files to backend
 * @param {FileList} files - Files to upload
 */
async function uploadToBackend(files) {
    // TODO: Replace with actual API call
    // const formData = new FormData();
    // for (const file of files) {
    //     formData.append('files', file);
    // }
    // const response = await fetch(`${API_BASE_URL}/upload`, {
    //     method: 'POST',
    //     body: formData
    // });
    // return response.json();
    
    // MOCK - Remove in production
    await new Promise(resolve => setTimeout(resolve, 2000));
    return { success: true };
}

/**
 * Get system statistics
 */
async function getStatsFromBackend() {
    // TODO: Replace with actual API call
    await new Promise(resolve => setTimeout(resolve, 500));
    return {
        totalDocuments: 15,
        totalChunks: 342,
        collectionSize: '24.5 MB'
    };
}

/**
 * Get stored chunks
 */
async function getChunksFromBackend() {
    // TODO: Replace with actual API call
    await new Promise(resolve => setTimeout(resolve, 500));
    return [
        { source: 'قانون-التأمين.pdf', text: 'نص تجريبي للجزء الأول من المستند...' },
        { source: 'لائحة-الترخيص.docx', text: 'نص تجريبي للجزء الثاني من المستند...' }
    ];
}

// Make toggleSourceCard available globally
window.toggleSourceCard = toggleSourceCard;
