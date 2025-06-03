import json
import re
import os
from tqdm import tqdm
import sys

# --- Fuzzy Matching Imports ---
from rapidfuzz import fuzz, process

# ==============================================================================
# SECTION 2: RAPIDFUZZ LAW RETRIEVAL WITH AMENDMENT SEPARATION
# ==============================================================================

def expand_abbreviations_local(title: str) -> str:
    """
    Expand common Vietnamese legal abbreviations.
    """
    ABBREVIATION_MAP = {
        "BHXH": "Bảo hiểm xã hội", "ATGT": "An toàn giao thông", "SHTT": "Sở hữu trí tuệ",
        "VPHC": "vi phạm hành chính", "BLHS": "Bộ luật Hình sự", "BLTTDS": "Bộ luật Tố tụng dân sự",
        "CSGT": "Cảnh sát giao thông", "NĐ-CP": "Nghị định Chính phủ", "QĐ": "Quyết định",
        "TT": "Thông tư", "UBND": "Ủy ban nhân dân"
    }
    for abbr, full_text in ABBREVIATION_MAP.items():
        title = re.sub(r'\b' + re.escape(abbr) + r'\b', full_text, title, flags=re.IGNORECASE)
    return title

def normalize_law_title(title: str) -> str:
    """
    Normalize law titles for better matching.
    """
    title = title.strip().lower()
    # Remove extra whitespaces
    title = re.sub(r'\s+', ' ', title)
    # Standardize common patterns
    title = re.sub(r'số\s*(\d+)', r'số \1', title)  # Standardize "số X" format
    title = re.sub(r'(\d+)/(\d+)', r'\1/\2', title)  # Remove spaces in decree numbers
    # Remove common prefixes that might vary
    title = re.sub(r'^(ban hành|về|quy định)\s+', '', title)
    return title

def is_valid_law_type(title: str) -> bool:
    """
    Check if the law title is a valid type (Luật or Bộ luật only).
    Excludes Nghị định, Thông tư, and other document types.
    """
    title_lower = title.lower().strip()
    
    # Check for valid law types
    valid_patterns = [
        r'\bluật\b',
        r'\bbộ luật\b'
    ]
    
    # Check for invalid law types to exclude
    invalid_patterns = [
        r'\bnghị định\b',
        r'\bthông tư\b',
        r'\bquyết định\b',
        r'\bchỉ thị\b',
        r'\bquy chế\b',
        r'\bquy định\b'
    ]
    
    # First check if it contains invalid patterns
    for pattern in invalid_patterns:
        if re.search(pattern, title_lower):
            return False
    
    # Then check if it contains valid patterns
    for pattern in valid_patterns:
        if re.search(pattern, title_lower):
            return True
    
    return False

def is_amendment_law(title: str) -> bool:
    """
    Check if a law title indicates it's an amendment law (Luật sửa đổi).
    """
    title_lower = title.lower().strip()
    
    # Patterns that indicate amendment laws
    amendment_patterns = [
        r'\bsửa đổi\b',
        r'\bbổ sung\b',
        r'\bsửa đổi,?\s*bổ sung\b',
        r'\bsửa đổi\s*và\s*bổ sung\b',
        r'\bthay đổi\b',
        r'\bcập nhật\b'
    ]
    
    for pattern in amendment_patterns:
        if re.search(pattern, title_lower):
            return True
    
    return False

def extract_base_law_from_amendment(title: str) -> str:
    """
    Extract the base law name from an amendment law title.
    For example: "Luật sửa đổi Bộ luật Dân sự" -> "Bộ luật Dân sự"
    """
    title_lower = title.lower().strip()
    
    # Remove amendment indicators
    amendment_patterns = [
        r'\bluật\s+sửa đổi,?\s*bổ sung\s+',
        r'\bluật\s+sửa đổi\s+và\s+bổ sung\s+',
        r'\bluật\s+sửa đổi\s+',
        r'\bluật\s+bổ sung\s+',
        r'\bsửa đổi,?\s*bổ sung\s+',
        r'\bsửa đổi\s+và\s+bổ sung\s+',
        r'\bsửa đổi\s+',
        r'\bbổ sung\s+'
    ]
    
    result = title_lower
    for pattern in amendment_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Clean up the result
    result = result.strip()
    
    # If the result doesn't start with "luật" or "bộ luật", try to preserve the original structure
    if not re.match(r'^(bộ\s+)?luật\b', result):
        # Try a different approach - look for the law being amended after common phrases
        match = re.search(r'(?:sửa đổi|bổ sung).*?((?:bộ\s+)?luật\s+[^,]+)', title_lower)
        if match:
            result = match.group(1).strip()
    
    return result if result else title_lower

def separate_law_types(title_index: dict) -> tuple:
    """
    Separate the title index into different categories:
    - Normal Luật
    - Normal Bộ luật  
    - Amendment Luật
    - Amendment Bộ luật
    Returns (luat_index, bo_luat_index, amendment_luat_index, amendment_bo_luat_index)
    """
    luat_index = {}
    bo_luat_index = {}
    amendment_luat_index = {}
    amendment_bo_luat_index = {}
    
    for title, chunks in title_index.items():
        is_amendment = is_amendment_law(title)
        
        if 'bộ luật' in title.lower():
            if is_amendment:
                amendment_bo_luat_index[title] = chunks
            else:
                bo_luat_index[title] = chunks
        elif 'luật' in title.lower():
            if is_amendment:
                amendment_luat_index[title] = chunks
            else:
                luat_index[title] = chunks
    
    return luat_index, bo_luat_index, amendment_luat_index, amendment_bo_luat_index

def create_law_title_index(law_data: list) -> dict:
    """
    Create an index of law titles for faster fuzzy matching using rapidfuzz.
    Only includes valid law types (Luật and Bộ luật).
    """
    print("Creating law title index for rapidfuzz matching...")
    title_index = {}
    excluded_count = 0
    
    for chunk in law_data:
        meta_data = chunk.get('meta_data', {})
        doc_title = meta_data.get('doc_title', '')
        if doc_title:
            # Check if this is a valid law type
            if not is_valid_law_type(doc_title):
                excluded_count += 1
                continue
                
            normalized_title = normalize_law_title(doc_title)
            if normalized_title not in title_index:
                title_index[normalized_title] = []
            title_index[normalized_title].append(chunk)
    
    print(f"Created index with {len(title_index)} unique law titles")
    print(f"Excluded {excluded_count} non-law documents (Nghị định, Thông tư, etc.)")
    return title_index

def rapidfuzz_match_law_title_with_amendments(query_title: str, indexes: dict, threshold: int = 80) -> tuple:
    """
    Use rapidfuzz to find the best matching law title, considering amendments.
    indexes should contain: 'luat', 'bo_luat', 'amendment_luat', 'amendment_bo_luat'
    Returns (best_match_title, similarity_score, index_type) or (None, 0, None) if no good match.
    """
    normalized_query = normalize_law_title(query_title)
    expanded_query = normalize_law_title(expand_abbreviations_local(query_title))
    
    # Determine if the query is looking for an amendment
    query_is_amendment = is_amendment_law(query_title)
    
    # Try both original and expanded query
    queries_to_try = [normalized_query, expanded_query]
    
    # If query is an amendment, also try matching against the base law
    if query_is_amendment:
        base_law = extract_base_law_from_amendment(query_title)
        if base_law != normalized_query:
            queries_to_try.append(normalize_law_title(base_law))
    
    best_match = None
    best_score = 0
    best_index_type = None
    
    # Define search priority based on query type
    if query_is_amendment:
        # For amendment queries, prioritize amendment indexes first
        search_order = ['amendment_luat', 'amendment_bo_luat', 'luat', 'bo_luat']
    else:
        # For normal queries, prioritize normal indexes first
        search_order = ['luat', 'bo_luat', 'amendment_luat', 'amendment_bo_luat']
    
    for query in queries_to_try:
        if not query.strip():
            continue
        
        for index_type in search_order:
            title_index = indexes.get(index_type, {})
            if not title_index:
                continue
                
            # Try different matching strategies
            for scorer in [fuzz.ratio, fuzz.token_sort_ratio, fuzz.partial_ratio]:
                try:
                    match_result = process.extractOne(
                        query, 
                        title_index.keys(), 
                        scorer=scorer,
                        score_cutoff=threshold
                    )
                    
                    if match_result and match_result[1] > best_score:
                        best_match = match_result[0]
                        best_score = match_result[1]
                        best_index_type = index_type
                        
                        # If we found a very good match in the preferred index, stop searching
                        if best_score >= 90 and index_type in search_order[:2]:
                            break
                            
                except Exception as e:
                    print(f"Error in rapidfuzz matching with {scorer.__name__}: {e}")
            
            # Early termination for very good matches in preferred indexes
            if best_score >= 90 and index_type in search_order[:2]:
                break
        
        # Early termination for very good matches
        if best_score >= 95:
            break
    
    return best_match, best_score, best_index_type

def keyword_based_matching_with_amendments(query_title: str, indexes: dict) -> tuple:
    """
    Fallback method using keyword-based matching, considering amendments.
    Returns (best_match_title, similarity_score, index_type)
    """
    normalized_query = normalize_law_title(query_title)
    expanded_query = normalize_law_title(expand_abbreviations_local(query_title))
    query_is_amendment = is_amendment_law(query_title)
    
    # Extract key terms from the query
    key_words = set()
    for query in [normalized_query, expanded_query]:
        if not query.strip():
            continue
            
        # Extract law type keywords
        law_types = re.findall(r'\b(?:luật|bộ luật)\b', query)
        key_words.update(law_types)
        
        # Extract numbers (years, decree numbers)
        numbers = re.findall(r'\b\d{4}\b|\b\d+/\d+\b', query)
        key_words.update(numbers)
        
        # Extract other significant words (remove common words)
        words = query.split()
        significant_words = [w for w in words if len(w) > 3 and w not in ['của', 'về', 'theo', 'trong', 'năm', 'sửa', 'đổi', 'bổ', 'sung']]
        key_words.update(significant_words[:3])  # Take top 3 significant words
    
    if not key_words:
        return None, 0, None
    
    best_match = None
    best_score = 0
    best_index_type = None
    
    # Define search priority based on query type
    if query_is_amendment:
        search_order = ['amendment_luat', 'amendment_bo_luat', 'luat', 'bo_luat']
    else:
        search_order = ['luat', 'bo_luat', 'amendment_luat', 'amendment_bo_luat']
    
    for index_type in search_order:
        title_index = indexes.get(index_type, {})
        if not title_index:
            continue
            
        for title in title_index.keys():
            score = 0
            for keyword in key_words:
                if keyword in title:
                    score += 1
            
            # Normalize score by number of keywords
            normalized_score = (score / len(key_words)) * 100
            if normalized_score > best_score:
                best_match = title
                best_score = normalized_score
                best_index_type = index_type
    
    return best_match, best_score, best_index_type

def law_retrieval_rapidfuzz_with_amendments(query_data: dict, indexes: dict, fuzzy_threshold: int = 80):
    """
    Law retrieval using rapidfuzz matching with amendment separation.
    """
    original_title = query_data.get('ten_luat', '')
    if not original_title:
        print(f"[FAILED] No law title found in query: {query_data}")
        return None
    
    print(f"\n[SEARCHING] Query: {query_data}")
    print(f"[SEARCHING] Original title: '{original_title}'")
    
    # Check if query is for an amendment
    if is_amendment_law(original_title):
        print(f"[INFO] Detected amendment law query")
        base_law = extract_base_law_from_amendment(original_title)
        print(f"[INFO] Extracted base law: '{base_law}'")
    
    # Try rapidfuzz matching first
    best_match_title, similarity_score, index_type = rapidfuzz_match_law_title_with_amendments(
        original_title, indexes, fuzzy_threshold
    )
    
    # If rapidfuzz matching fails, try keyword-based matching
    if not best_match_title:
        print(f"[RAPIDFUZZ] No match above threshold {fuzzy_threshold}. Trying keyword matching...")
        best_match_title, similarity_score, index_type = keyword_based_matching_with_amendments(
            original_title, indexes
        )
        
        if similarity_score < 80:  # Minimum threshold for keyword matching
            print(f"[FAILED] No suitable match found. Best score: {similarity_score:.1f}")
            return None
    
    print(f"[FOUND DOC] Best match: '{best_match_title}' (score: {similarity_score:.1f}, type: {index_type})")
    
    # Get chunks for the matched title from the appropriate index
    matching_chunks = indexes[index_type][best_match_title]
    
    # Find specific article/clause within the matched document
    desired_dieu = query_data.get('dieu')
    desired_khoan = query_data.get('khoan')
    desired_diem = query_data.get('diem')
    if(desired_dieu is None):
        print(f"[FAILED] Điều is None")
        return
    print(f"[SEARCHING] Looking for - Điều: {desired_dieu}, Khoản: {desired_khoan}, Điểm: {desired_diem}")
    
    # Filter chunks based on specific citation requirements
    relevant_chunks = []
    for chunk in matching_chunks:
        meta_data = chunk.get('meta_data', {})
        chunk_dieu = meta_data.get('dieu')
        chunk_khoan_list = meta_data.get('khoan', [])
        
        # Check if this chunk matches the desired article
        if desired_dieu and str(chunk_dieu) != str(desired_dieu):
            continue
            
        # If we need a specific clause, check if it's in this chunk
        if desired_khoan:
            if not chunk_khoan_list or str(desired_khoan) not in [str(k) for k in chunk_khoan_list]:
                continue
        
        relevant_chunks.append(chunk)
    
    if relevant_chunks:
        # Return the first matching chunk's ID
        chunk_id = relevant_chunks[0]['meta_data']['chunk_id']
        print(f"[SUCCESS] Found chunk: {chunk_id}")
        return chunk_id
    else:
        print(f"[FAILED] No chunks found for matched document")
        return None

# --- Helper functions for loading/saving data ---
def read_json_file(json_file_path: str):
    """Read JSON file with error handling."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as file: 
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file_path}: {e}")
        sys.exit(1)

def save_to_json(data, output_path):
    """Save data to JSON file with error handling."""
    try:
        with open(output_path, "w", encoding="utf-8") as f: 
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved processed data to: {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

# ==============================================================================
# SECTION 3: MAIN EXECUTION (UPDATED FOR AMENDMENT SEPARATION)
# ==============================================================================
if __name__ == "__main__":
    print("Legal Citation Parser with Amendment Law Separation + RapidFuzz")
    print("=" * 70)
    
    # --- Step 1: Load Data & Create Title Index ---
    qna_file = os.path.join("data", "json_data", "QnA", "qna_relevants_law.json")
    chunks_file = os.path.join("data", "json_data", "flatten_chunk_data", "combined_law_amend_flatten.json")
    output_file = "processed_qna_data.json"
    
    print("Loading data files...")
    qna_data = sorted(read_json_file(qna_file), key=lambda x: x['id'])
    qna_origin = sorted(read_json_file("qna.json"), key=lambda x: x['id'])
    chunks_data = read_json_file(chunks_file)
    
    print(f"Loaded {len(qna_data)} QnA entries and {len(chunks_data)} law chunks")
    
    # Create title index for rapidfuzz matching (only valid law types)
    title_index = create_law_title_index(chunks_data)
    
    # Separate law types including amendments
    luat_index, bo_luat_index, amendment_luat_index, amendment_bo_luat_index = separate_law_types(title_index)
    
    print(f"Separated into:")
    print(f"  - {len(luat_index)} normal Luật titles")
    print(f"  - {len(bo_luat_index)} normal Bộ luật titles")
    print(f"  - {len(amendment_luat_index)} amendment Luật titles")
    print(f"  - {len(amendment_bo_luat_index)} amendment Bộ luật titles")
    
    # Create indexes dictionary for easier passing
    indexes = {
        'luat': luat_index,
        'bo_luat': bo_luat_index,
        'amendment_luat': amendment_luat_index,
        'amendment_bo_luat': amendment_bo_luat_index
    }
    
    # --- Step 2: Main Processing Loop ---
    all_processed_data = []
    print("\nStarting processing with AMENDMENT SEPARATION...")
    for data in tqdm(qna_data, desc="Processing QnA"):
        retrieved_law_info = set()
        processed_law_set = set()
        
        # Process the new format: relevant_laws list
        for law_entry in data.get("relevant_laws", []):
            law_title = law_entry.get("title", "")
            dieu = law_entry.get("dieu")
            khoan = law_entry.get("khoan")
            relevance = law_entry.get("relevance", 0)
            
            # Skip if no title or already processed
            if not law_title:
                continue
                
            # Create a unique identifier for this law citation
            law_identifier = f"{law_title}_dieu_{dieu}_khoan_{khoan}"
            if law_identifier in processed_law_set:
                continue
                
            # Skip if not a valid law type
            if not is_valid_law_type(law_title):
                print(f"[SKIPPED] Invalid law type: {law_title}")
                continue

            print(f"\n{'='*70}")
            print(f"Processing law: '{law_title}' - Điều {dieu} - Khoản {khoan} | id: {data['id']}")
            
            # Create query data from the structured information
            query_data = {
                'ten_luat': law_title,
                'dieu': str(dieu) if dieu is not None else None,
                'khoan': str(khoan) if khoan is not None else None
            }
            
            # Perform retrieval with amendment awareness
            chunk_id = law_retrieval_rapidfuzz_with_amendments(query_data, indexes, fuzzy_threshold=80)
            
            if chunk_id:
                retrieved_law_info.add( chunk_id )
                print(f"[RESULT] SUCCESS: {chunk_id}")
            else:
                print(f"[RESULT] FAILED: No chunk found for {query_data}")
                
            processed_law_set.add(law_identifier)
        
        # Add retrieved laws to the data
        data['retrieved_laws'] = list(retrieved_law_info)
        del data['relevant_laws']
        data['question'] = qna_origin[data['id']-1]['question']
        all_processed_data.append(data)

    # --- Step 3: Save Results ---
    save_to_json(all_processed_data, output_file)
    
    # --- Step 4: Summary Statistics ---
    total_laws = sum(len(data.get("relevant_laws", [])) for data in qna_data)
    total_retrieved = sum(len(data.get("retrieved_laws", [])) for data in all_processed_data)
    success_rate = (total_retrieved / total_laws * 100) if total_laws > 0 else 0
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total law citations processed: {total_laws}")
    print(f"Successfully retrieved chunks: {total_retrieved}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {output_file}")
    print("\nThis enhanced version:")
    print("- Separates amendment laws (Luật sửa đổi) from normal laws")
    print("- Prioritizes appropriate indexes based on query type")
    print("- Extracts base law names from amendment titles")
    print("- Uses smart matching strategies for both normal and amendment laws")
    print("- Filters out Nghị định, Thông tư, and other non-law documents")
    print("- Uses RapidFuzz for fast fuzzy string matching")
    print("- Handles structured law citations directly")