// script.js (Frontend - Improved Link Handling)

console.log("INFO: Script.js loaded - Version with Improved Link Handling");

const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');

// URL của backend
const BACKEND_API_URL = 'http://localhost:5000/chat';

// System prompt cho backend (nếu cần)
const SYSTEM_PROMPT_FOR_BACKEND = "Bạn là một trợ lý AI hữu ích chuyên hỗ trợ về các vấn đề pháp luật Việt Nam.";

// Gắn event listeners
if (sendButton && typeof sendButton.addEventListener === 'function') {
    sendButton.addEventListener('click', sendMessage);
} else {
    console.error("CRITICAL ERROR: sendButton not found!");
}

if (userInput && typeof userInput.addEventListener === 'function') {
    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
} else {
    console.error("CRITICAL ERROR: userInput not found!");
}

// Hàm gửi tin nhắn
function sendMessage() {
    if (!userInput) {
        console.error("ERROR: userInput is null!");
        return;
    }
    
    const messageText = userInput.value.trim();
    
    if (messageText === '') {
        showLoading(false);
        return;
    }

    displayMessage(messageText, 'user');
    userInput.value = '';
    showLoading(true);
    getBotResponseFromBackend(messageText);
}

// Hàm gọi backend
async function getBotResponseFromBackend(userInputText) {
    console.log(`Calling backend with: "${userInputText}"`);
    
    try {
        const response = await fetch(BACKEND_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_message: userInputText,
                system_prompt: SYSTEM_PROMPT_FOR_BACKEND
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Backend error ${response.status}:`, errorText);
            throw new Error(`Lỗi kết nối backend (${response.status})`);
        }

        const data = await response.json();
        console.log("Backend response:", data);

        let botReply = "Xin lỗi, tôi không thể nhận được phản hồi từ máy chủ.";
        if (data && data.bot_reply) {
            botReply = data.bot_reply.trim();
        } else if (data && data.error) {
            botReply = `Lỗi từ máy chủ: ${data.error}`;
        }
        
        showLoading(false);
        displayMessage(botReply, 'bot');

    } catch (error) {
        console.error("Error calling backend:", error);
        showLoading(false);
        displayMessage(`❗ Lỗi: ${error.message}`, 'bot');
    }
}

// Hàm hiển thị tin nhắn
function displayMessage(text, sender) {
    console.log(`Displaying ${sender} message`);
    
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);

    if (sender === 'bot') {
        const botTextElement = document.createElement('p');
        
        // Xử lý định dạng văn bản - chỉ format nếu cần thiết
        let formattedText = formatBotResponse(text);
        botTextElement.innerHTML = formattedText;
        messageElement.appendChild(botTextElement);
    } else {
        // Tin nhắn người dùng giữ nguyên
        messageElement.textContent = text;
    }

    if (chatBox) {
        chatBox.appendChild(messageElement);
        scrollToBottom();
    }
}

// Hàm format phản hồi bot - cải thiện xử lý link
function formatBotResponse(text) {
    if (!text) return '';
    
    console.log("Original text:", text);
    
    // Bước 1: Xử lý các định dạng cơ bản
    text = text.replace(/\\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.*?)__/g, '<u>$1</u>');
    
    // Bước 2: Xử lý links - CHỈ format những URL chưa được format
    // Kiểm tra xem text đã có link HTML chưa
    const hasHtmlLinks = /<a\s+[^>]*href\s*=\s*["'][^"']*["'][^>]*>/i.test(text);
    
    if (!hasHtmlLinks) {
        console.log("No HTML links found, processing plain URLs...");
        // Chỉ xử lý các URL thuần túy (không nằm trong HTML tags)
        text = text.replace(/(https?:\/\/[^\s<>"()[\]{}]+)/gi, function(match) {
            // Loại bỏ dấu câu ở cuối URL
            let cleanUrl = match.replace(/[.,;:!?)\]}]+$/, '');
            console.log(`Converting URL to link: ${cleanUrl}`);
            return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" style="color: #007bff; text-decoration: underline;">${cleanUrl}</a>`;
        });
    } else {
        console.log("HTML links already present, skipping URL conversion");
    }
    
    // Bước 3: Xử lý xuống dòng
    text = text.replace(/\n/g, '<br>');
    
    console.log("Formatted text:", text);
    return text;
}

// Hàm hiển thị loading
function showLoading(isLoading) {
    if (loadingIndicator) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
        if (isLoading) scrollToBottom();
    }
}

// Hàm cuộn xuống dưới
function scrollToBottom() {
    if (chatBox) {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// Khởi tạo
if (chatBox) {
    scrollToBottom();
}
