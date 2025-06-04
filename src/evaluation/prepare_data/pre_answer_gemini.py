import json
import time
import requests
import google.generativeai as genai
import random, re, os
from tqdm.auto import tqdm
import multiprocessing as mp
from multiprocessing import Manager, Queue, Process
import threading
from queue import Empty

# API Keys configuration
GEMINI_API_KEY_LIST = [
                "AIzaSyCKtN98H-n2idRhIgWpvzcw-4cqdzik9rE",
                "AIzaSyAhZsYmuI9Waxj1o4ZXcT6lCYszhmVpWcM",
                "AIzaSyClqpWZjhwiFJ7kXJdalC-HOQ4GzNbGkq8",
                "AIzaSyAdis532XF3hKGdIlZ7PjvT0U4pi1FhWDw",
                "AIzaSyA8HYpppKJ84wAveUCdeMy6mTgETPNbQmw",
]
# Model configuration
MODEL_TYPE = "gemini-2.0-flash"

# Rate limiting configuration
REQUEST_DELAY = 0  # Delay between requests in seconds

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
        - Văn bản: {chunk.get('dieu_title', '')} - Nội dung: {chunk['content']}
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
    - Không cần liệt kê các bước hướng dẫn trong câu trả lời, chỉ đưa ra kết quả của các bước 2, 3, 4, xây dựng output cuối cùng tự nhiên, như người thật nói
    """

    token_count = estimate_tokens(prompt)
    return prompt

def rag_gemini_response(query, retrieval_results, model_type, api_key):
    """
    Generate response using Gemini with specified API key and model.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_type)

    prompt = generate_prompt(query, retrieval_results)
    generation_config = genai.types.GenerationConfig(
        temperature=0.1
    )
    # Generate answer with Gemini
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
        request_options={'timeout': 30}  # Additional timeout option
    )
    answer = response.text.strip()

    return answer

def worker_process(api_key, api_key_idx, task_queue, result_queue, progress_queue):
    """
    Worker process function that processes tasks using a specific API key.
    """
    print(f"Worker {api_key_idx} started")
    
    while True:
        try:
            # Get task from queue with timeout
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill to stop worker
                break
                
            i, data = task
            question = data["question"]
            retrieval_results = data["retrival_results"]
            
            # Apply rate limiting delay
            time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))
            
            begin_time = time.time()
            try:
                answer = rag_gemini_response(question, retrieval_results, MODEL_TYPE, api_key)
                generation_time = time.time() - begin_time
                
                result = {
                    'task_id': i,
                    'question': question,
                    'truth_answer': data['answer'],
                    'model': MODEL_TYPE,
                    'api_key_index': api_key_idx,
                    'generated_answer': answer,
                    'generation_time': generation_time,
                    'status': 'success',
                    'contexts': retrieval_results
                }
                
            except Exception as e:
                result = {
                    'task_id': i,
                    'question': question,
                    'truth_answer': data['answer'],
                    'model': MODEL_TYPE,
                    'api_key_index': api_key_idx,
                    'generated_answer': None,
                    'generation_time': time.time() - begin_time,
                    'status': 'error',
                    'error': str(e),
                    'contexts': retrieval_results
                }
                print(f"Worker {api_key_idx} - Error on question {i}: {e}")
            
            # Put result in result queue
            result_queue.put(result)
            
            # Signal progress update
            progress_queue.put(1)
            
        except Empty:
            # No more tasks, continue waiting
            continue
        except Exception as e:
            print(f"Worker {api_key_idx} encountered error: {e}")
            break
    
    print(f"Worker {api_key_idx} finished")

def progress_monitor(progress_queue, total_tasks, pbar):
    """
    Monitor progress updates from worker processes and update tqdm bar.
    """
    completed = 0
    while completed < total_tasks:
        try:
            progress_queue.get(timeout=1)
            completed += 1
            pbar.update(1)
        except Empty:
            continue
        except Exception as e:
            print(f"Progress monitor error: {e}")
            break

def process_questions_multiprocess(retrieval_data, max_questions=None):
    """
    Process questions using multiprocessing with one process per API key.
    """
    questions = retrieval_data
    if max_questions:
        questions = questions[:max_questions]

    print(f"Processing {len(questions)} questions with multiprocessing")
    print(f"Using model: {MODEL_TYPE}")
    print(f"Number of processes: {len(GEMINI_API_KEY_LIST)}")

    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()

    # Fill task queue
    for i, data in enumerate(questions):
        task_queue.put((i, data))

    # Create and start worker processes
    processes = []
    for idx, api_key in enumerate(GEMINI_API_KEY_LIST):
        p = Process(
            target=worker_process, 
            args=(api_key, idx, task_queue, result_queue, progress_queue)
        )
        p.start()
        processes.append(p)

    # Create progress bar
    pbar = tqdm(total=len(questions), desc="Processing questions")
    
    # Start progress monitor in separate thread
    progress_thread = threading.Thread(
        target=progress_monitor, 
        args=(progress_queue, len(questions), pbar)
    )
    progress_thread.start()

    # Collect results
    results = []
    for _ in range(len(questions)):
        result = result_queue.get()
        results.append(result)

    # Send poison pills to stop workers
    for _ in processes:
        task_queue.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Wait for progress monitor to finish
    progress_thread.join()
    
    # Close progress bar
    pbar.close()

    # Count successful and failed
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")

    return results

def read_json_file(json_file_path: str):
    """Read JSON file and return data."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    """Save data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    """Main function to run the multiprocess RAG system."""
    print("Starting Multiprocess Gemini RAG System")
    print(f"Using model: {MODEL_TYPE}")
    print(f"API keys available: {len(GEMINI_API_KEY_LIST)}")
    print(f"Rate limit delay: {REQUEST_DELAY}s per request")

    # Load retrieval data
    try:
        retrieval_data = read_json_file("pre_retrive_query.json")['queries']
        print(f"Loaded {len(retrieval_data)} questions")
    except FileNotFoundError:
        print("Error: pre_retrive_query.json not found")
        return
    
    qna_data = read_json_file("qna.json")
    qna_dict = {data['question']: data for data in qna_data}

    process_data = []
    for data in retrieval_data:
        data['answer'] = qna_dict[data['question']]['answer']
        process_data.append(data)

    retrieval_dict = {data['question']: data for data in process_data}

    # Process questions with multiprocessing
    max_questions = None  # Limit for testing, set to None for all questions

    print(f"Processing {len(process_data)} questions")
    
    results = process_questions_multiprocess(
        process_data,
        max_questions=max_questions
    )

    # Prepare output data
    answer_data = {}
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
    
    total_time =  sum([data['answer_generation']['generation_time'] for data in save_data])
    
    time_data = {
        "query_count": len(results),
        "total_time": total_time,
        "average_time": total_time / len(results)
    }

    save_data = sorted(save_data, key=lambda x: x['id'])
    
    # Save results
    save_to_json(save_data, "gemini_real_answer.json")
    save_to_json(time_data, "time_data.json")
    
    print(f"\n=== Final Results ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per question: {total_time/len(results):.2f} seconds")

if __name__ == "__main__":
    # Set multiprocessing start method (important for some systems)
    mp.set_start_method('spawn', force=True)
    main()