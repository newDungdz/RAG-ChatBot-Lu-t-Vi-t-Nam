import json
import time
import requests
import os

# Your OpenRouter API Key
OPENROUTER_API_KEY = "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"

# The OpenRouter API endpoint for chat completions
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROVISIONING_API_KEYS =[
"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",
"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",
"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",
"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"
]
# Rate limiting delays (in seconds)
RATE_LIMIT_DELAYS = {
    "openai/gpt-4o-mini": 2,  # 2 seconds between requests
    "meta-llama/llama-3.1-8b-instruct:free": 2,  # 3 seconds for free tier
    "default": 2
}

DATA_LIMIT = 120  # Limit the number of questions to process
MODEL_NAME = "meta-llama/llama-3.1-8b-instruct:free"  # Single model to use
REQUESTS_PER_KEY = 5  # Rotate key every 5 requests

def create_api_key(name, label, provisioning_api_key):
    """Create a new API key for the process"""
    url = "https://openrouter.ai/api/v1/keys"

    payload = {
        "name": name,
        "label": label,
        # "limit": 1000  # Optional credit limit
    }

    headers = {
        "Authorization": f"Bearer {provisioning_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"API key {name} created successfully")
        
        # Return both the API key and the hash for deletion
        return {
            'key': result.get('key', None),
            'hash': result.get('data', None).get('hash', None),
            'name': name
        }
    except Exception as e:
        print(f"Error creating API key {name}: {e}")
        return None

def delete_api_key(api_key_info, provisioning_api_key):
    """Delete an API key after use using the hash"""
    if not api_key_info or not api_key_info.get('hash'):
        print("Error: No hash provided for API key deletion")
        return False
        
    url = f"https://openrouter.ai/api/v1/keys/{api_key_info['hash']}"
    
    headers = {
        "Authorization": f"Bearer {provisioning_api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        print(f"API key {api_key_info.get('name', 'unknown')} deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting API key {api_key_info.get('name', 'unknown')}: {e}")
        return False

def rag_openrouter_response(query, retrieval_results, model_type="openai/gpt-4o-mini", max_tokens=2000, api_key=None):
    """
    Generate a legal response using OpenRouter GPT based on query and retrieval results.
    
    Args:
        query (str): The legal question to answer
        retrieval_results (dict): list of retrieval results containing legal contexts
        model_type (str): The OpenRouter model to use (default: "openai/gpt-4o-mini")
        max_tokens (int): Maximum tokens for the response
        api_key (str): API key to use for this request
    
    Returns:
        str: The generated answer or error message
    """
    
    # Use provided API key or fallback to default
    used_api_key = api_key if api_key else OPENROUTER_API_KEY
    
    # Build context from retrieval results
    context = ""
    for chunk in retrieval_results:
        context += f"""
        Nguồn: {chunk['doc_title']}, Điều {chunk['dieu_number']}, Nội dung: {chunk['content']}"""
    
    # Create the prompt
    prompt = f"""Bạn là một trợ lý pháp lý. Hãy trả lời toàn diện câu hỏi pháp lý sau, chỉ dựa trên các ngữ cảnh pháp lý được cung cấp, bằng cách suy luận từng bước theo hướng dẫn.

    CÂU HỎI PHÁP LÝ:
    {query}

    CÁC NGỮ CẢNH PHÁP LÝ LIÊN QUAN:
    {context}

    HƯỚNG DẪN:
    1. **Bước 1 - Xác định điều luật áp dụng:** Kiểm tra từng ngữ cảnh pháp lý được cung cấp để xác định những điều luật, khoản, mục hoặc quy định nào có thể liên quan trực tiếp đến câu hỏi.
    2. **Bước 2 - Phân tích áp dụng:** Với mỗi điều luật liên quan, hãy trích dẫn rõ nội dung pháp lý và giải thích tại sao và bằng cách nào điều đó có thể áp dụng để trả lời câu hỏi.
    3. **Bước 3 - Đánh giá mức độ đầy đủ của thông tin:** Nếu các ngữ cảnh không đủ để trả lời toàn diện, hãy nêu rõ hạn chế này trong phân tích.
    4. **Bước 4 - Kết luận ngắn gọn:** Sau phần phân tích chi tiết, hãy đưa ra một câu trả lời tóm tắt, súc tích để tổng hợp lại nội dung chính và đưa ra kết luận pháp lý.

    YÊU CẦU:
    - Không sử dụng bất kỳ kiến thức nào nằm ngoài các ngữ cảnh được cung cấp.
    - Phải trích dẫn rõ ràng điều luật/khoản/mục trong ngữ cảnh.
    - Không suy diễn hoặc bổ sung thông tin pháp lý không có trong ngữ cảnh.
    - Không cần liệt kê các bước hướng dẫn trong câu trả lời, chỉ đưa ra kết quả của các bước 2, 3, 4, xây dựng output cuối cùng tự nhiên, như người thật nói
    """
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {used_api_key}",
        "Content-Type": "application/json",
    }
    
    # Prepare payload
    payload = {
        "model": model_type,
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý pháp lý chuyên nghiệp, chỉ trả lời dựa trên các tài liệu pháp lý được cung cấp."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,  # Lower temperature for more consistent legal responses
    }
    
    try:
        # Make the API request
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload))
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse the response
        api_response = response.json()
        
        # Extract the assistant's message
        if "choices" in api_response and len(api_response["choices"]) > 0:
            answer = api_response["choices"][0]["message"]["content"].strip()
            return answer
        else:
            return "Lỗi: Không thể lấy phản hồi từ mô hình AI."
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if hasattr(response, 'content'):
            print(f"Response content: {response.content.decode()}")
        return f"Lỗi HTTP: {http_err}"
        
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return f"Lỗi kết nối: {req_err}"
        
    except KeyError as key_err:
        print(f"KeyError: {key_err}")
        print(f"API Response structure: {api_response if 'api_response' in locals() else 'No response received'}")
        return "Lỗi: Cấu trúc phản hồi từ API không như mong đợi."
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Lỗi không xác định: {e}"

def openrouter_check_limit():
    response = requests.get(
        url="https://openrouter.ai/api/v1/auth/key",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
    )
    print(json.dumps(response.json(), indent=2))

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def pre_answering(question, model, retrieval_results, api_key=None):
    """
    Prepares the answer to a question using the OpenRouter model.

    Args:
        question (str): The question to be answered.
        model (str): The model to use for answering.
        retrieval_results (list): The retrieval results for context.
        api_key (str): API key to use for this request.

    Returns:
        dict: The answer result dictionary.
    """
    begin_time = time.time()
    answer = rag_openrouter_response(question, retrieval_results, model, api_key=api_key)
    result = {
        "model": model,
        "generated_answer": answer,
        "generation_time": time.time() - begin_time
    }
    return result

class KeyManager:
    """Manages API key rotation every N requests"""
    
    def __init__(self, provisioning_keys, requests_per_key=5):
        self.provisioning_keys = provisioning_keys
        self.requests_per_key = requests_per_key
        self.current_key_index = 0
        self.current_api_key_info = None
        self.request_count = 0
        
    def get_current_api_key(self):
        """Get the current API key, creating a new one if needed"""
        # Check if we need to rotate or create initial key
        if (self.current_api_key_info is None or 
            self.request_count >= self.requests_per_key):
            
            # Delete old key if exists
            if self.current_api_key_info is not None:
                print(f"Rotating key after {self.request_count} requests")
                self._delete_current_key()
            
            # Create new key
            self._create_new_key()
            self.request_count = 0
        
        self.request_count += 1
        return self.current_api_key_info['key'] if self.current_api_key_info else None
    
    def _create_new_key(self):
        """Create a new API key using current provisioning key"""
        provisioning_key = self.provisioning_keys[self.current_key_index]
        run_name = f"rotate_key_{self.current_key_index}_{int(time.time())}"
        
        print(f"Creating new API key using provisioning key {self.current_key_index + 1}/{len(self.provisioning_keys)}")
        
        self.current_api_key_info = create_api_key(
            run_name, 
            f"Rotated key {self.current_key_index}", 
            provisioning_key
        )
        
        if not self.current_api_key_info or not self.current_api_key_info.get('key'):
            print(f"Failed to create API key with provisioning key {self.current_key_index}")
            # Try next provisioning key
            self.current_key_index = (self.current_key_index + 1) % len(self.provisioning_keys)
            return self._create_new_key()
        
        # Move to next provisioning key for next rotation
        self.current_key_index = (self.current_key_index + 1) % len(self.provisioning_keys)
    
    def _delete_current_key(self):
        """Delete the current API key"""
        if self.current_api_key_info:
            # Use the provisioning key that was used to create this key
            prev_index = (self.current_key_index - 1) % len(self.provisioning_keys)
            provisioning_key = self.provisioning_keys[prev_index]
            delete_api_key(self.current_api_key_info, provisioning_key)
            self.current_api_key_info = None
    
    def cleanup(self):
        """Clean up any remaining API key"""
        if self.current_api_key_info:
            print("Final cleanup of API key")
            self._delete_current_key()

def process_questions_with_rotation(model_name, questions_data):
    """
    Process questions with key rotation every N requests.
    
    Args:
        model_name (str): The model to use
        questions_data (list): List of questions and retrieval results
    
    Returns:
        list: List of processed answers
    """
    answer_data = []
    delay = RATE_LIMIT_DELAYS.get(model_name, RATE_LIMIT_DELAYS["default"])
    
    # Initialize key manager
    key_manager = KeyManager(PROVISIONING_API_KEYS, REQUESTS_PER_KEY)
    
    print(f"Starting processing with model {model_name} with key rotation every {REQUESTS_PER_KEY} requests")
    
    try:
        for i, data in enumerate(questions_data):
            print(f"Processing question {i+1}/{len(questions_data)} for model {model_name}")
            
            try:
                # Get current API key (will rotate if needed)
                current_api_key = key_manager.get_current_api_key()
                
                if not current_api_key:
                    print(f"Failed to get API key for question {i+1}")
                    continue
                
                print(f"Using API key for request {key_manager.request_count} (Key rotation: {(key_manager.request_count-1) % REQUESTS_PER_KEY + 1}/{REQUESTS_PER_KEY})")
                
                answer = pre_answering(
                    data["question"], 
                    model_name, 
                    data["relevant_chunks"], 
                    api_key=current_api_key
                )
                
                answer_data.append({
                    'id': data['id'],
                    'question': data['question'],
                    'answer_generation': answer,
                    'model': model_name,
                    'key_rotation_info': {
                        'request_number': key_manager.request_count,
                        'provisioning_key_index': key_manager.current_key_index
                    }
                })
                
                print(f"Question {i+1} completed for model {model_name}")
                
                # Rate limiting delay
                if i < len(questions_data) - 1:  # Don't delay after the last question
                    print(f"Waiting {delay} seconds for rate limiting...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error processing question {i+1} for model {model_name}: {e}")
                continue
    
    finally:
        # Always cleanup keys
        key_manager.cleanup()
    
    print(f"Completed processing {len(answer_data)} questions for model {model_name}")
    return answer_data

def main():
    print(f"Starting process with model {MODEL_NAME}, key rotation every {REQUESTS_PER_KEY} requests")
    
    # Load retrieval data
    try:
        retrieval_data = read_json_file("query_with_truth.json")
        limited_data = retrieval_data[:DATA_LIMIT]  # Limit to DATA_LIMIT questions
        print(f"Loaded {len(limited_data)} questions from retrieval data")
    except Exception as e:
        print(f"Error loading retrieval data: {e}")
        return
    
    try:
        # Process all questions with key rotation
        all_answers = process_questions_with_rotation(MODEL_NAME, limited_data)
        
        print(f"Processing completed. Total answers generated: {len(all_answers)}")
        
        # Save results
        output_filename = "answers_llama_openrouter.json"
        save_to_json({
            "total_answers": len(all_answers),
            "model_used": MODEL_NAME,
            "data_limit": DATA_LIMIT,
            "requests_per_key": REQUESTS_PER_KEY,
            "total_provisioning_keys": len(PROVISIONING_API_KEYS),
            "timestamp": time.time(),
            "answers": all_answers
        }, output_filename)
        
        print(f"Results saved to {output_filename}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Total answers: {len(all_answers)}")
        print(f"  Key rotation: Every {REQUESTS_PER_KEY} requests")
        print(f"  Provisioning keys used: {len(PROVISIONING_API_KEYS)}")
        
    except Exception as e:
        print(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()