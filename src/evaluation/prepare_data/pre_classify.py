import json, os
from typing import List, Dict
import time
from tqdm import tqdm
import google.generativeai as genai

API_KEY_LIST = ["AIzaSyCKtN98H-n2idRhIgWpvzcw-4cqdzik9rE",
                "AIzaSyAhZsYmuI9Waxj1o4ZXcT6lCYszhmVpWcM",
                "AIzaSyClqpWZjhwiFJ7kXJdalC-HOQ4GzNbGkq8",
                "AIzaSyAdis532XF3hKGdIlZ7PjvT0U4pi1FhWDw",
                "AIzaSyA8HYpppKJ84wAveUCdeMy6mTgETPNbQmw",]
API_KEY_IDX = 0
API_KEY = API_KEY_LIST[API_KEY_IDX]

GEMINI_MODEL = "gemini-2.0-flash-lite"

def get_legal_categories():
    """
    Get unique legal categories from Elasticsearch
    
    Returns:
    --------
    list
        List of unique legal categories
    """
    with open("note\\law_category_fulltext.txt", "r", encoding="utf-8") as file:
        categories = file.readlines()
        categories = [cate.strip() for cate in categories if cate.strip() != ""]
    return categories

legal_categories_fulltext = get_legal_categories()
with open("note\\law_category.txt", "r") as file:
    categories = file.readlines()
    categories = [cate.strip() for cate in categories]  
legal_categories = categories

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def generate_analysis_prompt(query, categories, categories_fulltext, top_k , enable_expansion = True):
    if enable_expansion: 
        return f"""Bạn cần thực hiện 2 nhiệm vụ cho câu truy vấn pháp lý sau: "{query}"

            NHIỆM VỤ 1: ĐÁNH GIÁ VÀ MỞ RỘNG TRUY VẤN
            Đánh giá xem câu truy vấn có cần mở rộng hay không dựa trên:
            - Câu truy vấn quá ngắn (ít hơn 5 từ)
            - Câu truy vấn quá trừu tượng hoặc mơ hồ
            - Sử dụng từ viết tắt
            - Thiếu ngữ cảnh pháp lý cụ thể
            - Cần thêm từ khóa liên quan để tìm kiếm hiệu quả hơn

            Nếu xác định rằng truy vấn cần mở rộng, hãy viết lại một câu truy vấn đầy đủ hơn, tuân theo các yêu cầu:
            1. Giữ nguyên ý định và nội dung chính của câu truy vấn gốc
            2. Bổ sung từ khóa pháp lý hoặc thuật ngữ chuyên ngành liên quan trực tiếp
            3. Làm rõ ngữ cảnh nếu đang thiếu (thời gian, hành vi, đối tượng…)
            4. Không thêm thông tin ngoài phạm vi truy vấn ban đầu
            5. Không sử dụng từ viết tắt — phải viết đầy đủ
            6. Câu truy vấn mở rộng phải tự nhiên, giống như một người thật hỏi

            NHIỆM VỤ 2: PHÂN LOẠI TRUY VẤN
            Phân loại câu truy vấn (sử dụng bản mở rộng nếu có) vào các danh mục pháp lý sau: {categories}

            Bối cảnh danh mục:
            {categories_fulltext}

            Chọn top {top_k} danh mục phù hợp nhất với độ tin cậy từ 0 đến 1, chú ý.

            TRẢ LỜI BẰNG JSON FORMAT:
            {{
                "expansion": {{
                    "original_query": "{query}",
                    "expanded_query": "câu truy vấn đã mở rộng (hoặc giống nguyên bản nếu không cần mở rộng)",
                    "needs_expansion": true/false,
                    "expansion_reason": "lý do cần/không cần mở rộng"
                }},
                "classification": [
                    {{"category": "tên danh mục", "confidence": 0.x}},
                    {{"category": "tên danh mục", "confidence": 0.x}}
                ]
            }}

            Chỉ trả về JSON, không giải thích thêm."""
    else:
        # Simple classification-only prompt (original style)
        return f"""Phân loại câu truy vấn pháp lý sau vào các danh mục pháp lý: {categories}

        Câu truy vấn: "{query}"

        Bối cảnh danh mục:
        {categories_fulltext}

        Chọn top {top_k} danh mục phù hợp nhất với độ tin cậy từ 0 đến 1.

        TRẢ LỜI BẰNG JSON FORMAT:
        {{
            "expansion": {{
                "original_query": "{query}",
                "expanded_query": "{query}",
                "needs_expansion": false,
                "expansion_reason": "Query expansion disabled"
            }},
            "classification": [
                {{"category": "tên danh mục", "confidence": 0.x}},
                {{"category": "tên danh mục", "confidence": 0.x}}
            ]
        }}

        Chỉ trả về JSON, không giải thích thêm."""

def expand_and_categories_query(query: str, gemini_model, top_k: int = 3, enable_expansion = True) -> Dict:
    """
    MERGED FUNCTION: Expand query if needed and classify into legal categories using Gemini API.
    
    Parameters:
    -----------
    query : str
        Original user query
    top_k : int
        Number of top categories to return
        
    Returns:
    --------
    dict
        Dictionary containing both expansion and classification results
    """
    try:
        # Validate inputs
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        # Prepare categories for classification
        categories = ", ".join(legal_categories)
        categories_fulltext = ", ".join(legal_categories_fulltext)
        
        # Combined prompt for both expansion and classification
        combined_prompt = generate_analysis_prompt(query, categories, categories_fulltext, top_k, enable_expansion)

        # Make single API request
        response = gemini_model.generate_content(combined_prompt,
                                                    generation_config=genai.types.GenerationConfig(
                                                            temperature=0  # <-- adjust this value
                                                        ))
        response_text = response.text.strip()
        
        # Parse combined response
        try:
            # Handle potential JSON formatting issues
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```") and response_text.endswith("```"):
                response_text = response_text[3:-3].strip()
            
            result = json.loads(response_text)
            
            # Validate structure
            if not isinstance(result, dict) or "expansion" not in result or "classification" not in result:
                raise ValueError("Invalid response structure")
            
            # Validate expansion result
            expansion = result["expansion"]
            if not isinstance(expansion, dict):
                raise ValueError("Invalid expansion format")
            
            # Validate classification result
            classification = result["classification"]
            if not isinstance(classification, list):
                raise ValueError("Invalid classification format")
            # Filter and sort classification results
            valid_categories = [
                {"category": c["category"], "confidence": float(c["confidence"])}
                for c in classification
                if isinstance(c, dict) and c.get("category") in legal_categories
            ]
            valid_categories = sorted(valid_categories, key=lambda x: x["confidence"], reverse=True)[:top_k]
            
            return {
                "expanded_query": expansion.get("expanded_query", query),
                "classification": valid_categories
            }

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Error parsing combined API response: {str(e)}")

    except Exception as e:
        print(f"Error in combined query expansion and classification: {str(e)}")


def process_single_query(query, top_k = 3):
    # query['question']
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    results = expand_and_categories_query(query['question'], gemini_model, top_k)
    print(results)
    query['expanded_query'] = results['expanded_query']
    # Add timing information to the query data
    query['category'] = results['classification']
    return query

# Example usage with timing and rate limiting
def switch_api_key():
    global API_KEY, API_KEY_IDX
    API_KEY_IDX += 1
    if(API_KEY_IDX == len(API_KEY_LIST)): API_KEY_IDX = 0
    API_KEY = API_KEY_LIST[API_KEY_IDX]

if __name__ == "__main__":
    # Rate limiting configuration
    DELAY_BETWEEN_QUERIES = 0.0  # seconds between queries
    BATCH_SIZE = 20  # process queries in batches
    BATCH_DELAY = 5.0  # additional delay between batches (seconds)
    
    # Initialize the retrieval flow with Elasticsearch connection
    query_data = sorted(read_json_file("processed_qna_data.json"), key=lambda x: x['id'])
    
    query_data = [data for data in query_data if len(data['retrieved_laws']) > 2]
    
    all_data = []
    # limit = 400
    query_data = query_data[201:]
    total_start_time = time.time()
    
    print(f"Starting to process {len(query_data)} queries with rate limiting...")
    print(f"Delay between queries: {DELAY_BETWEEN_QUERIES}s")
    print(f"Batch size: {BATCH_SIZE}, Batch delay: {BATCH_DELAY}s")
    

    for i, query in enumerate(tqdm(query_data, "Processing query", len(query_data))):
        # Rate limiting: add delay between queries
        if i > 0:  # Don't delay before the first query
            time.sleep(DELAY_BETWEEN_QUERIES)
        # Additional batch delay
        if i > 0 and i % BATCH_SIZE == 0:
            switch_api_key()
            print(f"\nBatch {i//BATCH_SIZE} completed, Switch to {API_KEY}. Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)
        
        try:
            query = process_single_query(query)
            all_data.append(query)
            
        except Exception as e:
            switch_api_key()
            print(f"Error processing query {i+1}: {e}, retry with {API_KEY_IDX}: {API_KEY}")
            query = process_single_query(query)
            # Still add the query with error information
            all_data.append(query)
    
    # Calculate overall statistics
    total_processing_time = time.time() - total_start_time
    successful_queries = [q for q in all_data if 'error' not in q]
    
    if successful_queries:
        # Add timing statistics to the output data
        
        # Save timing summary along with the data
        output_data = all_data
        output_file = f"pre_retrive_query_{GEMINI_MODEL.lower().replace('/', '_')}.json"
        save_to_json(output_data, output_file)
        print(f"\nResults saved to {output_file}")

# Utility function to get all unique categories from Elasticsearch
# Example usage for performance analysis
# analyze_retrieval_performance("pre_retrive_query_with_timing.json")