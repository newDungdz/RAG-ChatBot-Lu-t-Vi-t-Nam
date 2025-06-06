import json
import time
import requests
import random, re, os
import subprocess
from ollama import Client
from tqdm import tqdm

def generate_prompt(query, retrieval_results):
    """
    Generate Prompt for the LLM
    """
    def estimate_tokens(text):
        # Roughly 1.3 tokens per word (common heuristic for Vietnamese LLMs)
        words = re.findall(r'\w+', text)
        estimated_tokens = int(len(words) * 1.3)
        return estimated_tokens
    context = ""
    for chunk in retrieval_results:
        context += f"""
        - {chunk.get('dieu_title', '')} - {chunk['content']} 
        """
            
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
    - Không cần liệt kê các bước hướng dẫn trong câu trả lời, chỉ đưa ra kết quả của các bước 2, 3, 4
    """

    token_count = estimate_tokens(prompt)
    print(f"Query have {token_count} tokens")
    return prompt

def run_ollama(prompt, client: Client, model="llama3:8b-instruct-q4_K_M"):
    """
    Chạy Ollama với prompt đã cho sử dụng Ollama Python client
    """
    try:
        # Sử dụng Ollama client để gửi prompt
        chat_completion = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            options={"temperature": 0.3}  # Có thể điều chỉnh temperature theo nhu cầu
        )
        
        # Trả về response từ model
        return chat_completion["message"]["content"].strip()
        
    except Exception as e:
        print(f"Lỗi khi chạy ollama: {e}")
        return None

def read_json_file(json_file_path: str):
    """Read JSON file and return data."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    """Save data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_question_local_llm(retrieval_data, model_type="llama3:8b-instruct-q4_K_M", max_questions=None):
    """
    Process questions using local LLM (Ollama) with the same output format as Gemini version.
    """
    questions = retrieval_data
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"Processing {len(questions)} questions with local LLM")
    print(f"Using model: {model_type}")
    
    results = []
    successful = 0
    failed = 0
    ollama_client = Client()
    for i, data in tqdm(enumerate(questions), "Processing question", len(questions)):
        question = data["question"]
        retrieval_results = data["relevant_chunks"]
                
        # Generate prompt using the same function as Gemini
        prompt = generate_prompt(question, retrieval_results)
        
        # Process the question
        begin_time = time.time()
        try:
            answer = run_ollama(prompt, ollama_client, model_type)
            generation_time = time.time() - begin_time
            
            if answer is not None:
                result = {
                    'task_id': i,
                    'question': question,
                    'truth_answer': data['answer'],
                    'model': model_type,
                    'api_key_index': None,  # Not applicable for local LLM
                    'generated_answer': answer,
                    'generation_time': generation_time,
                    'status': 'success',
                    'contexts': retrieval_results
                }
                successful += 1
                # print(f"Completed question {i+1}")
            else:
                result = {
                    'task_id': i,
                    'question': question,
                    'truth_answer': data['answer'],
                    'model': model_type,
                    'api_key_index': None,
                    'generated_answer': None,
                    'generation_time': time.time() - begin_time,
                    'status': 'error',
                    'error': 'Ollama returned None response',
                    'contexts': retrieval_results
                }
                failed += 1
                print(f"Error on question {i+1}: Ollama returned None response")
            
        except Exception as e:
            result = {
                'task_id': i,
                'question': question,
                'truth_answer': data['answer'],
                'model': model_type,
                'api_key_index': None,
                'generated_answer': None,
                'generation_time': time.time() - begin_time,
                'status': 'error',
                'error': str(e),
                'contexts': retrieval_results
            }
            failed += 1
            print(f"Error on question {i+1}: {e}")
        
        results.append(result)
        
        # Progress update every 5 questions
        if (i + 1) % 5 == 0:
            print(f"Progress: {i+1}/{len(questions)} questions completed")
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    return results    

def main():
    """Main function to run the single process RAG system."""
    
    # Load retrieval data
    try:
        retrieval_data = read_json_file("data/evaluation/answer_evaluation/query_with_truth.json")
        print(f"Loaded {len(retrieval_data)} questions")
    except FileNotFoundError:
        print("Error: query_with_truth.json not found")
        return
    
    retrieval_dict = {data['question'] : data for data in retrieval_data}
    
    # Process questions with single process
    max_questions = None  # Limit for testing, set to None for all questions
    retrieval_data = retrieval_data[150:]
        
    # answered_data = read_json_file("answers_local_llm.json")
    # question_datas = {data['question']: data for data in answered_data}
    process_data = retrieval_data
    answer_data = {}
    # print(question_datas.keys())
    # for questions in question_datas.keys():
    #     if questions not in retrieval_dict.keys():
    #         continue
    #     if retrieval_dict[questions]['answer'] != question_datas[questions]['truth_answer']:
    #     # if question_datas[questions]['answer_generation']['generated_answer'] is None:
    #         process_data.append(retrieval_dict[questions])
    #     else:
    #         answer_data[retrieval_dict[questions]['id']] = question_datas[questions]
    # print(answer_data)
    print(len(process_data))
    start_time = time.time()
    results = process_question_local_llm(
        process_data, 
        max_questions=max_questions,
        model_type="llama3.1:8b-instruct-q4_K_M"
    )

    total_time = time.time() - start_time
    
    # Prepare output data
    for result in results:
        answer_data[retrieval_dict[result['question']]['id']] = {
            'question': result['question'],
            'truth_answer': result['truth_answer'],
            'answer_generation': {
                'model': result['model'],
                'generated_answer': result['generated_answer'],
                'generation_time': result['generation_time'],
            },
            'contexts': result['contexts']
        }
    
    save_data = []
    for i in answer_data.keys():
        new_data = answer_data[i]
        new_data = {
            'id': i,
            'question': new_data['question'],
            'truth_answer': new_data['truth_answer'],
            'answer_generation': new_data['answer_generation'],
            'contexts': new_data['contexts']
        }
        save_data.append(new_data)
    save_data = sorted(save_data, key=lambda x: x['id'])
    # Save results
    output_filename = "answers_local_llm.json"
    save_to_json(save_data, output_filename)
    
    print(f"\n=== Final Results ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per question: {total_time/len(results):.2f} seconds")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()