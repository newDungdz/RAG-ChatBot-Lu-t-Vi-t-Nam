// script.js (Frontend - Gọi Backend - Hiển thị phản hồi với hỗ trợ định dạng và link)

console.log("INFO: Script.js loaded - Version for Backend Communication - With Text Formatting Support and Link Detection");

const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');

// Kiểm tra các DOM elements
console.log("INFO: chatBox element:", chatBox);
console.log("INFO: userInput element:", userInput);
console.log("INFO: sendButton element:", sendButton);
console.log("INFO: loadingIndicator element:", loadingIndicator);

// --- URL CỦA BACKEND PYTHON ---
const BACKEND_API_URL = 'http://localhost:5000/chat'; // Đảm bảo backend chạy ở port 5000

// --- SYSTEM PROMPT (Optional - Backend có thể dùng hoặc bỏ qua) ---
const SYSTEM_PROMPT_FOR_BACKEND = "Bạn là một trợ lý AI hữu ích chuyên hỗ trợ về các vấn đề pháp luật Việt Nam. Hãy trả lời câu hỏi một cách rõ ràng, chính xác, tập trung vào luật pháp và đưa ra thông tin tham khảo nếu có thể. Luôn giữ thái độ trung lập và chuyên nghiệp. Bạn có thể sử dụng **in đậm**, __gạch chân__ và \\t tab để định dạng văn bản trả lời.";

// --- Gắn Event Listeners ---
if (sendButton && typeof sendButton.addEventListener === 'function') {
    console.log("INFO: Attaching click listener to sendButton.");
    sendButton.addEventListener('click', sendMessage);
} else {
    console.error("CRITICAL ERROR: sendButton not found or not an element!");
}

if (userInput && typeof userInput.addEventListener === 'function') {
    console.log("INFO: Attaching keypress listener to userInput.");
    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            console.log("INFO: Enter key pressed, calling sendMessage.");
            sendMessage();
        }
    });
} else {
    console.error("CRITICAL ERROR: userInput not found or not an element!");
}

// --- Các Hàm Xử Lý ---
function sendMessage() {
    console.log("---- INFO: sendMessage function CALLED ----");

    if (!userInput) {
        console.error("ERROR in sendMessage: userInput is null!");
        return;
    }
    const messageText = userInput.value.trim();
    console.log(` INFO in sendMessage: Message text from input: "${messageText}"`);

    if (messageText === '') {
        console.log("INFO in sendMessage: Message text is empty, returning.");
        showLoading(false);
        return;
    }

    displayMessage(messageText, 'user');

    if (userInput && typeof userInput.value !== 'undefined') {
        userInput.value = '';
    }
    
    showLoading(true);
    getBotResponseFromBackend(messageText);
}

async function getBotResponseFromBackend(userInputText) {
    console.log(`---- INFO: getBotResponseFromBackend CALLED with: "${userInputText}" ----`);
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
        console.log(" INFO in getBotResponseFromBackend: Fetch request sent to backend. Response status:", response.status);

        if (!response.ok) {
            let errorBodyText = await response.text();
            console.error(`ERROR in getBotResponseFromBackend: Backend responded with status ${response.status}. Body:`, errorBodyText);
            let errorJson = {};
            try { errorJson = JSON.parse(errorBodyText); } catch (e) {}
            const errorMessage = errorJson.error || errorJson.message || `Lỗi kết nối backend (${response.status})`;
            throw new Error(errorMessage);
        }

        const data = await response.json();
        console.log(" INFO in getBotResponseFromBackend: Data received from backend:", data);

        let botReply = "Xin lỗi, tôi không thể nhận được phản hồi từ máy chủ.";
        if (data && data.bot_reply) {
            botReply = data.bot_reply.trim();
        } else if (data && data.error) {
            botReply = `Lỗi từ máy chủ: ${data.error}`;
        }
        
        console.log(` INFO in getBotResponseFromBackend: Parsed bot reply: "${botReply}"`);
        showLoading(false);
        displayMessage(botReply, 'bot');

    } catch (error) {
        console.error("CRITICAL ERROR in getBotResponseFromBackend (catch block):", error);
        showLoading(false);
        displayMessage(`❗ Lỗi: ${error.message}`, 'bot');
    }
}

function displayMessage(text, sender) {
    console.log(`INFO: Displaying ${sender} message`);
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);

    if (sender === 'bot') {
        const botTextElement = document.createElement('p');
        
        // Xử lý định dạng văn bản
        let formattedText = formatText(text);
        
        // Sử dụng innerHTML để hiển thị HTML đã được định dạng
        botTextElement.innerHTML = formattedText;
        
        messageElement.appendChild(botTextElement);
    } else {
        // Tin nhắn người dùng giữ nguyên dạng text
        messageElement.textContent = text;
    }

    if (chatBox) {
        chatBox.appendChild(messageElement);
        scrollToBottom(); // Cuộn xuống dưới sau khi thêm tin nhắn
    } else {
        console.error("ERROR in displayMessage: chatBox is null!");
    }
}

// Hàm xử lý định dạng văn bản với hỗ trợ link
function formatText(text) {
    if (!text) return '';
    
    // Xử lý tabs
    text = text.replace(/\\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
    
    // Xử lý bold - hỗ trợ cả **text** và __text__
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Xử lý underline
    text = text.replace(/__(.*?)__/g, '<u>$1</u>');
    
    // Xử lý links - Tìm và chuyển đổi HTTPS URLs thành clickable links
    // Tránh xử lý URLs đã được format thành HTML links
    text = text.replace(/(https:\/\/[^\s<>"()[\]{}]+)/gi, function(match, url, offset, string) {
        // Loại bỏ các dấu câu ở cuối URL nếu có
        let cleanUrl = url.replace(/[.,;:!?)\]}]+$/, '');
        
        // Kiểm tra xem URL này có nằm trong thẻ <a> đã tồn tại không
        const beforeText = string.substring(0, offset);
        const afterText = string.substring(offset + match.length);
        
        // Tìm thẻ <a> mở gần nhất trước URL này
        const lastOpenTag = beforeText.lastIndexOf('<a ');
        const lastCloseTag = beforeText.lastIndexOf('</a>');
        
        // Nếu có thẻ <a> mở sau thẻ đóng cuối cùng, nghĩa là URL đang nằm trong thẻ <a>
        if (lastOpenTag > lastCloseTag) {
            console.log(`INFO: URL already inside <a> tag, skipping: ${cleanUrl}`);
            return match; // Trả về URL gốc không thay đổi
        }
        
        console.log(`INFO: Processing URL: ${cleanUrl}`);
        
        // Tạo link với target="_blank" để mở trong tab mới
        return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" style="color: #007bff; text-decoration: underline;">${cleanUrl}</a>`;
    });
    
    // Thay thế xuống dòng
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

function showLoading(isLoading) {
    if (loadingIndicator) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
        if (isLoading) scrollToBottom();
    } else {
        console.warn("WARN in showLoading: loadingIndicator is null!");
    }
}

function scrollToBottom() {
    if (chatBox) {
        // Đảm bảo cuộn xuống dưới cùng
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// --- Khởi chạy ---
if (chatBox) {
    scrollToBottom();
    console.log("INFO: Initial scrollToBottom called.");
} else {
    console.warn("WARN: chatBox is null at script end, initial scrollToBottom not called.");
}
console.log("INFO: End of script.js execution. Ready for backend communication. Formatting support and link detection enabled.");