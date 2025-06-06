import json, os
from typing import List, Dict
import time
from tqdm import tqdm
from retrival_pipeline import RAGLawRetrieval, rag_gemini_response, rag_openrouter_response, API_KEY

API_KEY_LIST = ["AIzaSyBq4HTkU_PWUyHh7NmOuFPSjgzMQI86CCo"
                "AIzaSyCKtN98H-n2idRhIgWpvzcw-4cqdzik9rE",
                "AIzaSyAhZsYmuI9Waxj1o4ZXcT6lCYszhmVpWcM",
                "AIzaSyClqpWZjhwiFJ7kXJdalC-HOQ4GzNbGkq8",
                "AIzaSyAdis532XF3hKGdIlZ7PjvT0U4pi1FhWDw",
                "AIzaSyA8HYpppKJ84wAveUCdeMy6mTgETPNbQmw",]
API_KEY_IDX = 0

embedding_model_list = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
]

choose_model = embedding_model_list[1]
retrieval_flow = RAGLawRetrieval(
    es_host='localhost',
    es_port=9200,
    es_index=f'chunks_{choose_model.replace("/", "_").lower()}',
    embedding_model = choose_model,
    query_process_model="gemini-2.0-flash-lite"
)

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_single_query(query):
    query_start_time = time.time()
    # query['question']
    query['expanded_query'] = {
        'original_query': query['question'],
        'expanded_query': query['expanded_query']
    }
    results = retrieval_flow.process_query(query, top_k_categories=2, top_bm25=50, top_k_chunks=20)
    query['expanded_query'] = results['step0_query_expansion']
    query['retrival_time'] = round(time.time() - query_start_time, 4)
    
    # Add timing information to the query data
    query['category'] = results['step1_top_categories']
    query['retrival_results'] = results['final_chunks']
    # query['processing_order'] = i + 1
    # query['processing_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
    # Print timing info for monitoring
    # print(f"Query {i+1}: {query['retrival_time']}s, {len(query['retrival_results'])} chunks")
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
    BATCH_SIZE = 10  # process queries in batches
    BATCH_DELAY = 0.0  # additional delay between batches (seconds)
    
    # Initialize the retrieval flow with Elasticsearch connection
    query_data = sorted(read_json_file("data/json_data/QnA/processed_qna_data.json"), key=lambda x: x['id'])
    # query_data = sorted(read_json_file("classified_query.json"), key=lambda x: x['id'])
    
    # Process queries with detailed timing tracking and rate limiting
    # preload_data = sorted(read_json_file("pre_retrive_query_with_timing.json"), key=lambda x: x['id'])
    all_data = []
    # limit = 400
    # query_data = query_data[:3]
    total_start_time = time.time()
    
    print(f"Starting to process {len(query_data)} queries with rate limiting...")
    print(f"Delay between queries: {DELAY_BETWEEN_QUERIES}s")
    print(f"Batch size: {BATCH_SIZE}, Batch delay: {BATCH_DELAY}s")
    
    API_KEY = API_KEY_LIST[API_KEY_IDX]
    
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
    
    # Calculate actual processing time (excluding delays)
    total_retrieval_time = sum(q['retrival_time'] for q in successful_queries) if successful_queries else 0
    total_delay_time = total_processing_time - total_retrieval_time
    
    if successful_queries:
        avg_retrieval_time = sum(q['retrival_time'] for q in successful_queries) / len(successful_queries)
        min_retrieval_time = min(q['retrival_time'] for q in successful_queries)
        max_retrieval_time = max(q['retrival_time'] for q in successful_queries)
        
        print(f"\n=== TIMING STATISTICS ===")
        print(f"Total queries processed: {len(query_data)}")
        print(f"Successful queries: {len(successful_queries)}")
        print(f"Failed queries: {len(query_data) - len(successful_queries)}")
        print(f"Total wall-clock time: {round(total_processing_time, 2)} seconds")
        print(f"Total retrieval time: {round(total_retrieval_time, 2)} seconds")
        print(f"Total delay time: {round(total_delay_time, 2)} seconds")
        print(f"Average retrieval time per query: {round(avg_retrieval_time, 4)} seconds")
        print(f"Min retrieval time: {round(min_retrieval_time, 4)} seconds")
        print(f"Max retrieval time: {round(max_retrieval_time, 4)} seconds")
        print(f"Effective processing rate: {round(len(successful_queries) / total_processing_time, 2)} queries/second")
        print(f"Pure processing rate: {round(len(successful_queries) / total_retrieval_time, 2)} queries/second")
        
        # Add timing statistics to the output data
        timing_summary = {
            "processing_summary": {
                "total_queries": len(query_data),
                "successful_queries": len(successful_queries),
                "failed_queries": len(query_data) - len(successful_queries),
                "total_wall_clock_time_seconds": round(total_processing_time, 2),
                "total_retrieval_time_seconds": round(total_retrieval_time, 2),
                "total_delay_time_seconds": round(total_delay_time, 2),
                "average_retrival_time": round(avg_retrieval_time, 4),
                "min_retrival_time": round(min_retrieval_time, 4),
                "max_retrival_time": round(max_retrieval_time, 4),
                "effective_queries_per_second": round(len(successful_queries) / total_processing_time, 2),
                "pure_queries_per_second": round(len(successful_queries) / total_retrieval_time, 2),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "rate_limiting": {
                    "delay_between_queries": DELAY_BETWEEN_QUERIES,
                    "batch_size": BATCH_SIZE,
                    "batch_delay": BATCH_DELAY
                }
            }
        }
        
        # Save timing summary along with the data
        output_data = {
            "timing_summary": timing_summary,
            "queries": all_data
        }
        output_file = f"pre_retrive_query_{choose_model.lower().replace('/', '_')}.json"
        save_to_json(output_data, output_file)
        print(f"\nResults saved to {output_file}")
        
    else:
        print("No successful queries to analyze.")
        # Still save the data even if all queries failed
        output_data = {
            "timing_summary": {
                "processing_summary": {
                    "total_queries": len(query_data),
                    "successful_queries": 0,
                    "failed_queries": len(query_data),
                    "total_wall_clock_time_seconds": round(total_processing_time, 2),
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "rate_limiting": {
                        "delay_between_queries": DELAY_BETWEEN_QUERIES,
                        "batch_size": BATCH_SIZE,
                        "batch_delay": BATCH_DELAY
                    }
                }
            },
            "queries": all_data
        }
        save_to_json(output_data, "pre_retrive_query_with_timing.json")
        print(f"Results (with errors) saved to 'pre_retrive_query_with_timing.json'")

# Utility function to get all unique categories from Elasticsearch

def analyze_retrieval_performance(json_file_path: str):
    """
    Analyze retrieval performance from the saved JSON file with timing data
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing retrieval results with timing
    """
    data = read_json_file(json_file_path)
    
    if "timing_summary" in data and "queries" in data:
        queries = data["queries"]
        timing_summary = data["timing_summary"]["processing_summary"]
        
        print("=== RETRIEVAL PERFORMANCE ANALYSIS ===")
        print(f"Dataset: {timing_summary['total_queries']} queries")
        print(f"Success Rate: {timing_summary['successful_queries']}/{timing_summary['total_queries']} ({round(timing_summary['successful_queries']/timing_summary['total_queries']*100, 1)}%)")
        
        # Show both effective and pure processing rates
        if "effective_queries_per_second" in timing_summary:
            print(f"Effective Processing Speed: {timing_summary['effective_queries_per_second']} queries/second (including delays)")
            if "pure_queries_per_second" in timing_summary:
                print(f"Pure Processing Speed: {timing_summary['pure_queries_per_second']} queries/second (excluding delays)")
        else:
            print(f"Processing Speed: {timing_summary['queries_per_second']} queries/second")
            
        print(f"Average Retrieval Time: {timing_summary['average_retrival_time']} seconds")
        print(f"Time Range: {timing_summary['min_retrival_time']}s - {timing_summary['max_retrival_time']}s")
        
        # Show rate limiting info if available
        if "rate_limiting" in timing_summary:
            rate_limit = timing_summary["rate_limiting"]
            print(f"\n=== RATE LIMITING CONFIGURATION ===")
            print(f"Delay between queries: {rate_limit['delay_between_queries']}s")
            print(f"Batch size: {rate_limit['batch_size']}")
            print(f"Batch delay: {rate_limit['batch_delay']}s")
        
        # Analyze step-by-step timing if available
        successful_queries = [q for q in queries if 'error' not in q and 'step_timings' in q]
        if successful_queries:
            print("\n=== STEP-BY-STEP TIMING ANALYSIS ===")
            step_names = successful_queries[0]['step_timings'].keys()
            
            for step in step_names:
                step_times = [q['step_timings'][step] for q in successful_queries]
                avg_time = sum(step_times) / len(step_times)
                min_time = min(step_times)
                max_time = max(step_times)
                print(f"{step}: avg={round(avg_time, 4)}s, min={round(min_time, 4)}s, max={round(max_time, 4)}s")
        
        # Analyze chunk retrieval statistics
        chunk_counts = [q['total_chunks_retrieved'] for q in successful_queries if 'total_chunks_retrieved' in q]
        if chunk_counts:
            print(f"\n=== CHUNK RETRIEVAL STATISTICS ===")
            print(f"Average chunks per query: {round(sum(chunk_counts)/len(chunk_counts), 1)}")
            print(f"Chunk count range: {min(chunk_counts)} - {max(chunk_counts)}")
    
    else:
        print("Invalid file format. Expected timing data structure not found.")

# Advanced rate limiting function for different scenarios
def process_queries_with_adaptive_rate_limiting(
    retrieval_flow, 
    query_data, 
    base_delay=1.0, 
    max_delay=10.0, 
    backoff_factor=2.0,
    success_threshold=0.9
):
    """
    Process queries with adaptive rate limiting that adjusts based on success rate
    
    Parameters:
    -----------
    retrieval_flow : RAGLawRetrieval
        Initialized retrieval flow
    query_data : list
        List of queries to process
    base_delay : float
        Base delay between queries in seconds
    max_delay : float
        Maximum delay between queries in seconds
    backoff_factor : float
        Factor to increase delay when encountering errors
    success_threshold : float
        Success rate threshold to maintain
    """
    all_data = []
    current_delay = base_delay
    recent_successes = []  # Track recent success/failure
    window_size = 10  # Window for calculating recent success rate
    
    print(f"Starting adaptive rate-limited processing...")
    print(f"Base delay: {base_delay}s, Max delay: {max_delay}s")
    
    for i, query in enumerate(tqdm(query_data, "Processing with adaptive rate limiting")):
        # Apply current delay
        if i > 0:
            time.sleep(current_delay)
        
        query_start_time = time.time()
        success = False
        
        try:
            results = retrieval_flow.process_query(query['question'], top_k_categories=3, top_bm25=50, top_k_chunks=20)
            query['retrival_time'] = round(time.time() - query_start_time, 4)
            query['category'] = results['step1_top_categories']
            query['retrival_results'] = results['final_chunks']
            query['processing_delay'] = current_delay
            success = True
            
        except Exception as e:
            print(f"Error in query {i+1}: {e}")
            query['retrival_time'] = 0
            query['error'] = str(e)
            query['processing_delay'] = current_delay
            success = False
        
        # Track recent successes
        recent_successes.append(success)
        if len(recent_successes) > window_size:
            recent_successes.pop(0)
        
        # Adaptive delay adjustment
        if len(recent_successes) >= window_size:
            recent_success_rate = sum(recent_successes) / len(recent_successes)
            
            if recent_success_rate < success_threshold:
                # Increase delay if success rate is low
                current_delay = min(current_delay * backoff_factor, max_delay)
                print(f"Low success rate ({recent_success_rate:.2f}), increasing delay to {current_delay}s")
            elif recent_success_rate > success_threshold and current_delay > base_delay:
                # Decrease delay if success rate is high
                current_delay = max(current_delay / backoff_factor, base_delay)
                print(f"High success rate ({recent_success_rate:.2f}), decreasing delay to {current_delay}s")
        
        all_data.append(query)
    
    return all_data

# Example usage for performance analysis
# analyze_retrieval_performance("pre_retrive_query_with_timing.json")