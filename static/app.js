document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const modelSelect = document.getElementById('modelSelect');
    const clearBtn = document.getElementById('clearBtn');
    const chatContainer = document.getElementById('chatContainer');
    const welcomeHero = document.getElementById('welcomeHero');
    const messageList = document.getElementById('messageList');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const suggestionCards = document.querySelectorAll('.suggestion-card');

    // Conversation State
    let conversationHistory = [];
    let isGenerating = false;

    // Configure Marked JS
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (e) {}
                }
                return code;
            },
            breaks: true
        });
    }

    // 1. Fetch available Groq models
    async function loadModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (data.models && data.models.length > 0) {
                modelSelect.innerHTML = '';
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name || m.id;
                    modelSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.warn('Could not load models from server, using default:', err);
        }
    }

    loadModels();

    // 2. Textarea Auto-resize & Input State
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
        sendBtn.disabled = !userInput.value.trim() || isGenerating;
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (userInput.value.trim() && !isGenerating) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // 3. Quick Suggestion Cards
    suggestionCards.forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt && !isGenerating) {
                userInput.value = prompt;
                userInput.dispatchEvent(new Event('input'));
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
    });

    // 4. Clear Conversation
    clearBtn.addEventListener('click', () => {
        if (isGenerating) return;
        conversationHistory = [];
        messageList.innerHTML = '';
        welcomeHero.style.display = 'flex';
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;
    });

    // 5. Submit Message Form & Stream Response
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || isGenerating) return;

        // Hide welcome hero on first message
        if (welcomeHero.style.display !== 'none') {
            welcomeHero.style.display = 'none';
        }

        // Add user message
        appendMessage('user', text);
        conversationHistory.push({ role: 'user', content: text });

        // Clear input field
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;
        isGenerating = true;

        // Create empty assistant row with typing indicator
        const assistantRow = createMessageRow('assistant');
        const contentDiv = assistantRow.querySelector('.message-content');
        contentDiv.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        messageList.appendChild(assistantRow);
        scrollToBottom();

        let accumulatedContent = '';

        try {
            const selectedModel = modelSelect.value;
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    top_k: 3
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // Keep last incomplete line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const payload = line.replace('data: ', '').trim();
                        if (payload === '[DONE]') {
                            break;
                        }
                        try {
                            const parsed = JSON.parse(payload);
                            if (parsed.error) {
                                accumulatedContent += `\n\n**Error:** ${parsed.error}`;
                            } else if (parsed.content) {
                                accumulatedContent += parsed.content;
                            }
                            renderAssistantMessage(contentDiv, accumulatedContent);
                            scrollToBottom();
                        } catch (pErr) {
                            console.error('Error parsing SSE chunk:', pErr);
                        }
                    }
                }
            }

            // Push final content to history
            if (accumulatedContent) {
                conversationHistory.push({ role: 'assistant', content: accumulatedContent });
            } else {
                contentDiv.innerHTML = '<em>No response received from Groq.</em>';
            }

        } catch (err) {
            console.error('Chat stream error:', err);
            contentDiv.innerHTML = `<span style="color:#ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${err.message || 'Failed to communicate with server'}</span>`;
        } finally {
            isGenerating = false;
            sendBtn.disabled = !userInput.value.trim();
        }
    });

    // Helper functions
    function appendMessage(role, text) {
        const row = createMessageRow(role);
        const contentDiv = row.querySelector('.message-content');
        if (role === 'user') {
            contentDiv.textContent = text;
        } else {
            renderAssistantMessage(contentDiv, text);
        }
        messageList.appendChild(row);
        scrollToBottom();
    }

    function createMessageRow(role) {
        const row = document.createElement('div');
        row.className = `message-row ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = role === 'user' 
            ? '<i class="fa-solid fa-user"></i>' 
            : '<i class="fa-solid fa-robot"></i>';

        const content = document.createElement('div');
        content.className = 'message-content';

        row.appendChild(avatar);
        row.appendChild(content);
        return row;
    }

    function renderAssistantMessage(element, markdownText) {
        if (typeof marked !== 'undefined') {
            element.innerHTML = marked.parse(markdownText);
            if (typeof hljs !== 'undefined') {
                element.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
        } else {
            element.textContent = markdownText;
        }
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
