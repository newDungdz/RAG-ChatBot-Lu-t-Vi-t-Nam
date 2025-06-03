import os
import json
import time
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from docx import Document
import google.generativeai as genai
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_analysis.log"),
        logging.StreamHandler()
    ]
)

# ==== CONFIGURATION ====
# Store your API key in an environment variable
GENAI_API_KEY = "AIzaSyC_IGEJNCZJrQanC1eAfiOGSrd0rfU_yHs"
if not GENAI_API_KEY:
    logging.warning("GENAI_API_KEY not found in environment variables. Checking for config file...")
    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
            GENAI_API_KEY = config.get("GENAI_API_KEY")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("No API key found. Please set GENAI_API_KEY environment variable or create a config.json file.")
        exit(1)

# Define constants
CHECKPOINT_DIR = "results/checkpoints"
RESULTS_DIR = "results"
MAX_RETRIES = 3
BATCH_LIMIT = 120  # Maximum number of documents to process in a batch
NUM_PROCESSES = 5  # Adjust based on API rate limits

# ==== FILE HANDLING FUNCTIONS ====
def read_json_file(json_file_path: str) -> Dict:
    """Read and parse a JSON file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as dictionary
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading JSON file {json_file_path}: {e}")
        return {}

def save_to_json(data: Any, output_path: str) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        output_path: Path where to save the JSON file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.debug(f"Successfully saved: {output_path}")
    except Exception as e:
        logging.error(f"Error saving JSON to {output_path}: {e}")

# ==== DOCUMENT PARSING ====
def parse_docx_with_formatting(doc_path: str) -> str:
    """Parse a .docx file and preserve formatting like bold text.
    
    Args:
        doc_path: Path to the DOCX file
        
    Returns:
        Parsed text with markdown-style formatting
    """
    try:
        doc = Document(doc_path)
        text = ""
        
        for para in doc.paragraphs:
            line = ""
            bold_active = False
            
            for run in para.runs:
                if run.bold:
                    if not bold_active:
                        line += "**"
                        bold_active = True
                    line += run.text
                else:
                    if bold_active:
                        line += "**"
                        bold_active = False
                    line += run.text
                    
            if bold_active:
                line += "**"  # Close any unclosed bold at the end
                
            text += '\n' + line
            
        return text
    except Exception as e:
        logging.error(f"Error parsing DOCX file {doc_path}: {e}")
        return ""

# ==== PROMPT GENERATION ====
def build_legal_analysis_prompt(text: str) -> str:
    """Generate a prompt for legal document analysis.
    
    Args:
        text: Text of the legal document
        
    Returns:
        Formatted prompt for the AI model
    """
    return f"""
    Phân tích van bản pháp luật sửa đổi sau:

    VĂN BẢN:    
    {text}
    
    
    YÊU CẦU:
    1. Nêu văn bản nào đang được sửa đổi (tên luật cũ) và số luật được sửa đổi của văn bản đó đó (ví dụ 57/2014/QH13).
    2. Những Điều, Khoản, Điểm cụ thể nào bị sửa đổi.
    3. Nội dung sửa đổi cụ thể (bị thay thế, thêm, bỏ…).
    
    Trả về kết quả theo định dạng JSON:
    {{
        "title": "Tên văn bản được sửa đổi",
        "doc_name": "Số văn bản được sửa đổi",
        "doc_data": [
            {{
                "metadata": {{
                    "dieu": "số Điều",
                    "khoan": "số Khoản (nếu có, không thì để null)",
                    "diem": "số Điểm (nếu có, không thì để null)",
                    "type": "Thay thế/Thêm/Bỏ"
                }},
                "content": "Nội dung sửa đổi cụ thể, ghi đúng như trong văn bản, kể cả tiêu đề, không thêm hay bỏ một từ nào"
            }}
        ]
    }}
    """

# ==== PROCESSING FUNCTIONS ====
def process_document(doc_item_tuple: Tuple[int, Dict], api_key: str) -> Optional[Dict]:
    """Process a single document with the Gemini AI model.
    
    Args:
        doc_item_tuple: Tuple containing (index, document metadata)
        api_key: API key for Gemini
        
    Returns:
        Analysis result or None if processing failed
    """
    idx, doc_metadata = doc_item_tuple
    doc_name = doc_metadata.get("doc_name", "")
    
    # Add jitter to avoid API rate limiting when multiple processes start simultaneously
    time.sleep(random.uniform(0.1, 1.0))
    
    # Check for existing checkpoint
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{doc_name.replace('.docx', '').replace('.doc', '')}.json")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                logging.info(f"Process {os.getpid()}: Loaded checkpoint for document {idx}: {doc_name}")
                return checkpoint_data
        except Exception as e:
            logging.warning(f"Process {os.getpid()}: Failed to load checkpoint for document {idx}: {e}")
    
    # Initialize Gemini for this process
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    
    # Normalize document name
    if doc_name.endswith('.doc'):
        doc_name = doc_name[:-4] + '.docx'
    if not doc_name.endswith('.docx'):
        logging.warning(f"Process {os.getpid()}: Not a DOCX file: {doc_name}")
        return None
    
    # Determine document path
    doc_type = "luat"  # This could be parameterized
    doc_folder_path = os.path.join("data", "docs", "usable_doc_file", doc_type)
    docx_path = os.path.join(doc_folder_path, doc_name)
    
    # Check if file exists
    if not os.path.exists(docx_path):
        logging.error(f"Process {os.getpid()}: File not found: {docx_path}")
        return None
    
    # Parse document
    content = parse_docx_with_formatting(docx_path)
    if not content:
        logging.error(f"Process {os.getpid()}: Failed to parse document {doc_name}")
        return None
    
    # Process with retry logic
    success = False
    retries = 0
    analysis = None
    
    while not success and retries < MAX_RETRIES:
        try:
            prompt = build_legal_analysis_prompt(content)
            response = model.generate_content(prompt)
            
            try:
                # Handle potential JSON formatting issues
                response_text = response.text.strip()
                
                if response_text.startswith("```json") and response_text.endswith("```"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```") and response_text.endswith("```"):
                    response_text = response_text[3:-3].strip()
                
                # Parse JSON
                analysis = json.loads(response_text)
                
                # Create result entry with metadata
                result = {
                    "id": idx,
                    "doc_name": doc_name,
                    "original_metadata": doc_metadata,
                    "analysis": analysis
                }
                
                # Save individual checkpoint
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                save_to_json(result, checkpoint_file)
                
                success = True
                return result
                
            except json.JSONDecodeError as json_err:
                logging.error(f"Process {os.getpid()}: JSON error for document {idx}: {json_err}")
                # Save response text to a checkpoint file for debugging
                txt_checkpoint_file = os.path.join(CHECKPOINT_DIR, f"doc_{idx}_{doc_name.replace('.docx', '')}_response.txt")
                try:
                    with open(txt_checkpoint_file, "w", encoding="utf-8") as txt_file:
                        txt_file.write(response.text)
                    logging.info(f"Process {os.getpid()}: Saved response text to {txt_checkpoint_file}")
                except Exception as txt_err:
                    logging.error(f"Process {os.getpid()}: Failed to save response text to {txt_checkpoint_file}: {txt_err}")
                logging.debug(f"Process {os.getpid()}: Response text: {response.text[:500]}...")
                retries += 1
                
        except Exception as e:
            logging.error(f"Process {os.getpid()}: Error processing document {idx}, retry {retries}: {e}")
            retries += 1
            
            # Exponential backoff
            wait_time = 2 ** retries
            logging.info(f"Process {os.getpid()}: Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    # If all retries failed
    if not success:
        logging.error(f"Process {os.getpid()}: Failed to process document {idx} after {MAX_RETRIES} retries")
    
    return None

# ==== MAIN EXECUTION ====
def main():
    """Main execution function."""
    # Create output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Define paths
    metadata_path = os.path.join("data", "json_data", "doc_metadata", "amend_law", "amended_laws.json")
    results_path = os.path.join(RESULTS_DIR, "legal_analysis_results.json")
    
    # Read document metadata
    doc_metadata = read_json_file(metadata_path)
    logging.info(f"Loaded metadata for {len(doc_metadata)} documents")
    
    # Check for existing results
    results = []
    if os.path.exists(results_path):
        try:
            results = read_json_file(results_path)
            logging.info(f"Loaded {len(results)} existing results")
        except Exception as e:
            logging.error(f"Failed to load existing results: {e}")
    
    # Determine which documents need processing
    processed_ids = {r.get("id") for r in results}
    docs_to_process = []
    
    for idx, doc in enumerate(doc_metadata):
        if idx not in processed_ids:
            docs_to_process.append((idx, doc))
    
    logging.info(f"Found {len(docs_to_process)} documents to process")
    
    # Apply batch limit
    docs_to_process = docs_to_process[:BATCH_LIMIT]
    if not docs_to_process:
        logging.info("No documents to process. Exiting.")
        return
    
    logging.info(f"Will process {len(docs_to_process)} documents with {NUM_PROCESSES} processes")
    
    # Using a Manager to share results between processes
    with Manager() as manager:
        shared_results = manager.list(results)
        
        # Create a pool of worker processes
        with Pool(processes=NUM_PROCESSES) as pool:
            # Create a partial function with the API key
            process_func = partial(process_document, api_key=GENAI_API_KEY)
            
            # Process documents in parallel with progress bar
            for result in tqdm(
                pool.imap_unordered(process_func, docs_to_process),
                total=len(docs_to_process),
                desc="Processing documents"
            ):
                if result:
                    shared_results.append(result)
                    
                    # Save checkpoint periodically
                    if len(shared_results) % 5 == 0:
                        save_to_json(list(shared_results), results_path)
                        logging.info(f"Saved progress: {len(shared_results)} results processed")
        
        # Convert manager list to regular list
        results = list(shared_results)
    
    # Save final results
    save_to_json(results, results_path)
    logging.info("Processing complete!")
    
    # Generate summary statistics
    num_processed = len(results)
    num_total = len(doc_metadata)
    logging.info(f"Processed {num_processed}/{num_total} documents ({(num_processed/num_total)*100:.2f}%)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user. Results saved to checkpoint files.")
    except Exception as e:
        logging.critical(f"Critical error: {e}", exc_info=True)