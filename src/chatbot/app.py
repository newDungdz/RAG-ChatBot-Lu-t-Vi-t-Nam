# app.py (Backend Server with Chat Memory)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import os
import json
from datetime import datetime
from retrival_pipeline import RAGLawRetrieval
import re
from elasticsearch import Elasticsearch

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Simple list-based chat memory storage
chat_history = []
MAX_MESSAGES = 50  # Maximum number of messages to keep in memory

def add_message_to_history(role, content):
    """Add a message to chat history"""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }
    chat_history.append(message)
    
    # Keep only last MAX_MESSAGES to prevent memory overflow
    if len(chat_history) > MAX_MESSAGES:
        chat_history[:] = chat_history[-MAX_MESSAGES:]
    
    print(f"INFO: Added {role} message to history. Total messages: {len(chat_history)}")

def format_conversation_context(max_messages=10):
    """Format recent messages for context"""
    if not chat_history:
        return ""
    
    # Get last max_messages for context
    recent_messages = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history
    
    context = "Lịch sử cuộc trò chuyện gần đây:\n"
    for msg in recent_messages:
        role_display = "Người dùng" if msg['role'] == 'user' else "Trợ lý"
        context += f"{role_display}: {msg['content']}\n"
    context += "\nCâu hỏi hiện tại:\n"
    
    return context

def clear_chat_history():
    """Clear all chat history"""
    global chat_history
    chat_history = []
    print("INFO: Chat history cleared")

# os.environ['GOOGLE_API_KEY'] = "AIzaSyC_IGEJNCZJrQanC1eAfiOGSrd0rfU_yHs"

try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"INFO: Google API Key configured successfully from environment variable (...{GOOGLE_API_KEY[-5:]}).")
except KeyError:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable not found.")
    # Để server vẫn chạy cho mục đích debug giao diện, ta sẽ không exit()

# --- ĐỊNH NGHĨA SYSTEM PROMPT ---
SYSTEM_PROMPT = """Bạn là một trợ lý pháp lý. Hãy trả lời toàn diện câu hỏi pháp lý sau, chỉ dựa trên các văn bản pháp lý được cung cấp. 
Khi người dùng cần tư vấn , sử dụng tools get_specific_law_article_info để lấy các văn bản pháp luật liên quan, sau đó hãy xây dụng câu trả lời bằng cách suy luận từng bước theo hướng dẫn.

Bạn có thể tham khảo lịch sử cuộc trò chuyện để hiểu rõ hơn ngữ cảnh của câu hỏi hiện tại.

HƯỚNG DẪN:
1. **Bước 1 - Xác định điều luật áp dụng:** Kiểm tra từng văn bản pháp lý được cung cấp để xác định những điều luật, khoản, mục hoặc quy định nào có thể liên quan trực tiếp đến câu hỏi.
2. **Bước 2 - Phân tích áp dụng:** Với mỗi điều luật liên quan, hãy trích dẫn rõ nội dung pháp lý và giải thích tại sao và bằng cách nào điều đó có thể áp dụng để trả lời câu hỏi.
3. **Bước 3 - Đánh giá mức độ đầy đủ của thông tin:** Nếu các văn bản không đủ để trả lời toàn diện, hãy nêu rõ hạn chế này trong phân tích.
4. **Bước 4 - Kết luận ngắn gọn:** Sau phần phân tích chi tiết, hãy đưa ra một câu trả lời tóm tắt, súc tích để tổng hợp lại nội dung chính và đưa ra kết luận pháp lý. 

YÊU CẦU:
- Sau toàn bộ câu trả lời, ghi ra các link web của văn bản pháp lý được sử dụng trong câu trả lời để người dùng có thể tra cứu, với định dạng:
  "
  Bạn có thể kiểm tra các văn bản pháp luật liên quan tại đây:
  + [Tên văn bản]: [Link URL]
  + [Tên văn bản]: [Link URL]
  "
- Không sử dụng bất kỳ kiến thức nào nằm ngoài các văn bản được cung cấp.
- Phải trích dẫn rõ ràng điều luật/khoản/mục trong văn bản. Khi nêu tên 1 văn bản pháp luật, 
- Không suy diễn hoặc bổ sung thông tin pháp lý không có trong văn bản.
- Cấu trúc câu trả lời bao gồm các kết quả suy luận từ các bước hướng dẫn, với các xây dụng câu tự nhiên, giống như người thật giải thích
"""

def setup_es_client():
    CLOUD_ID="Legal_RAG_data:YXNpYS1zb3V0aGVhc3QxLmdjcC5lbGFzdGljLWNsb3VkLmNvbTo0NDMkYWJhZmZjOGQxNjA3NGY0Y2EwMzc4NGFhNDdlMmM1MjckNzg2YjMzY2I1NGFjNDNiZTg1NTljZDgxNTJlODJmNDA="
    
    # Connect to Elasticsearch
    if (os.environ.get('LOCAL_MODE', 'True').lower() in ('true', '1', 'yes')):
        es = Elasticsearch([{'host': os.environ.get('ELASTICSEARCH_HOST', "elasticsearch"), 'port': int(os.environ.get('ELASTICSEARCH_PORT', 9200)), 'scheme': 'http'}])
        print(f"LOCAL_MODE bật, kết nối với elasticsearch local thành công với {os.environ.get('ELASTICSEARCH_HOST', 'elasticsearch')} , {os.environ.get('ELASTICSEARCH_PORT', '9200')}")
    else:
        es = Elasticsearch(
            cloud_id=CLOUD_ID,
            api_key=("lQRSIZcBDy4SfGpi8c3q", "iKwdTKOvjEz31ahN9r7eug")
        )
        print(f"LOCAL_MODE tắt, kết nối với elasticsearch cloud thành công với {CLOUD_ID}")
    return es

retrieval_flow = RAGLawRetrieval(
    es_client=setup_es_client(),
    embedding_model = 'intfloat/multilingual-e5-small',
    query_process_model='gemini-2.0-flash-lite',
    # es_index='chunks_intfloat_multilingual-e5-small',
)

# --- ĐỊNH NGHĨA TOOL (Hàm Python và Khai báo cho Gemini) ---

def get_law_article_details_implementation(query):
    """
    Hàm này được Gemini gọi thông qua Tool Calling.
    Hiện tại, nó giả lập việc lấy thông tin chi tiết của một điều luật.
    BẠN SẼ THAY THẾ PHẦN GIẢ LẬP DƯỚI ĐÂY BẰNG LOGIC TRUY VẤN DỮ LIỆU THẬT CỦA MÌNH.
    """
    print(f"\n🐍 [BACKEND PYTHON - TOOL EXECUTOR]: Tool 'get_law_article_details_implementation' called.")
    print(f"   Parameters received from Gemini: query = {query}'")

    results = retrieval_flow.process_query(query, top_k_categories=2, top_k_chunks=20)
    context = ""
    for chunk in results["final_chunks"]:
        context += f"""
        - Văn bản: {chunk['doc_title']}, Link văn bản:{chunk['doc_link']}, Nội dung: {chunk['content']}
        """

    print(f"   Result from YOUR data query (or simulation): {context}")
    return context

# Khai báo Tool cho Gemini (giữ nguyên phần này)
get_law_article_tool_declaration = genai.protos.FunctionDeclaration(
    name="get_specific_law_article_info",
    description="""
    Lấy thông tin chi tiết của các điều luật liên quan đến câu hỏi của người dùng
    , dùng mọi khi người dùng cần tư vấn ở bất kỳ câu hỏi nào
    """,
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Câu hỏi của người dùng"),
        },
        required=["query"]
    )
)
law_tool_definition = genai.protos.Tool(function_declarations=[get_law_article_tool_declaration])

# Khởi tạo Model Gemini (giữ nguyên)
GEMINI_MODEL = None
MODEL_NAME_TO_USE = "gemini-2.0-flash"
try:
    if 'GOOGLE_API_KEY' in os.environ:
        GEMINI_MODEL = genai.GenerativeModel(
            model_name=MODEL_NAME_TO_USE,
            system_instruction=SYSTEM_PROMPT,
            tools=[law_tool_definition]
        )
        print(f"INFO: Gemini Model '{MODEL_NAME_TO_USE}' initialized successfully.")
    else:
        print("WARNING: Gemini Model NOT initialized because GOOGLE_API_KEY is missing.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize Gemini Model '{MODEL_NAME_TO_USE}': {e}")

# Add this function to your app.py

def remove_duplicate_urls(text):
    """
    Remove duplicate URLs in Markdown links where the URL appears as both text and link
    Example: 
    "[https://example.com](https://example.com)" 
    becomes 
    "https://example.com"
    """
    if not text:
        return text
    
    # Pattern to match Markdown links where URL is duplicated as text and link
    # [URL](URL) -> URL
    markdown_duplicate_pattern = r'\[(https?://[^\]]+)\]\(\1\)'
    
    # Replace duplicated Markdown links with just the URL
    cleaned_text = re.sub(markdown_duplicate_pattern, r'\1', text)
    
    return cleaned_text


# Update your /chat endpoint
@app.route('/chat', methods=['POST'])
def handle_chat_request():
    print("\n--- [BACKEND PYTHON - /chat ENDPOINT]: Received new request ---")
    
    if not GEMINI_MODEL:
        print("ERROR: Gemini Model is not available (likely due to missing API Key).")
        return jsonify({"error": "Lỗi máy chủ: Model AI chưa sẵn sàng (thiếu API Key?)."}), 500
    
    try:
        data = request.get_json()
        if not data or 'user_message' not in data:
            return jsonify({"error": "Yêu cầu không hợp lệ: Thiếu 'user_message'."}), 400
        
        user_message = data['user_message']
        print(f"   User message: \"{user_message}\"")
        
        # Get conversation context from chat history
        conversation_context = format_conversation_context()
        
        # Add conversation context to the user message
        contextual_message = conversation_context + user_message
        
        # Add user message to history
        add_message_to_history('user', user_message)
        
        # Khởi tạo chat session với system prompt
        chat_session = GEMINI_MODEL.start_chat(
            enable_automatic_function_calling=False
        )
        
        response = chat_session.send_message(contextual_message)
        
        while True:
            called_function_info = None
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call and part.function_call.name:
                        called_function_info = part.function_call
                        break
            
            if called_function_info:
                tool_name = called_function_info.name
                tool_args = dict(called_function_info.args)
                print(f"🧠 [GEMINI]: Tool call requested: '{tool_name}' with args: {tool_args}")
                
                if tool_name == "get_specific_law_article_info":
                    # Gọi hàm triển khai tool của bạn
                    result_from_your_function = get_law_article_details_implementation(
                        tool_args.get("query")
                    )
                    # Gửi kết quả lại cho Gemini
                    response = chat_session.send_message(genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name, 
                            response={"content": result_from_your_function}
                        )
                    ))
                else:
                    print(f"WARNING: Gemini requested an unknown tool: '{tool_name}'. Ignoring.")
                    break
            else:
                print("   No (more) tool call requested by Gemini.")
                break
        
        final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text)
        
        # POST-PROCESSING: Remove duplicate links
        final_text = remove_duplicate_urls(final_text)
        print(f"🧹 [POST-PROCESSING]: Duplicate links removed")
        
        # Add bot response to history
        add_message_to_history('assistant', final_text)
        
        print(f"🤖 [BACKEND - FINAL RESPONSE TO CLIENT]: \"{final_text}\"")
        print(f"   Total messages in chat history: {len(chat_history)}")
        
        return jsonify({
            "bot_reply": final_text,
            "message_count": len(chat_history)
        })
        
    except Exception as e:
        print(f"ERROR in /chat: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500

# Simple endpoint to get chat history
@app.route('/chat/history', methods=['GET'])
def get_chat_history_endpoint():
    try:
        return jsonify({
            "messages": chat_history,
            "total_messages": len(chat_history)
        })
    except Exception as e:
        print(f"ERROR in /chat/history: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500

# Simple endpoint to clear chat history
@app.route('/chat/clear', methods=['POST'])
def clear_chat_history_endpoint():
    try:
        clear_chat_history()
        return jsonify({
            "message": "Lịch sử trò chuyện đã được xóa",
            "total_messages": len(chat_history)
        })
    except Exception as e:
        print(f"ERROR in /chat/clear: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500

# Simple endpoint to get chat statistics
@app.route('/chat/stats', methods=['GET'])
def get_chat_stats():
    try:
        return jsonify({
            "total_messages": len(chat_history),
            "max_messages": MAX_MESSAGES
        })
    except Exception as e:
        print(f"ERROR in /chat/stats: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500

# Serve index.html
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve JS and CSS
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    print("INFO: Starting Flask backend server with simple list-based chat memory...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)