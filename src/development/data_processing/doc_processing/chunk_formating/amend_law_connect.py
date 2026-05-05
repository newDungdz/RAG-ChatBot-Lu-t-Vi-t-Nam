from docx import Document
import re, os, json, unicodedata
from tqdm import tqdm

doc_type = "luat"

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_chunk_data(law_chunk_data, doc_code, dieu, khoan):
    for chunk in law_chunk_data:
        metadata = chunk['meta_data']
        if(metadata['doc_code'] == doc_code and metadata['dieu'] == dieu):
            if(khoan is None): 
                return chunk
            elif(khoan in metadata['khoan']): 
                return chunk
    return None

def remove_vietnamese_diacritics(text):
    normalized_text = unicodedata.normalize('NFD', text)
    text_without_diacritics = ''.join(
        char for char in normalized_text if unicodedata.category(char) != 'Mn'
    )
    return text_without_diacritics

def translate_amend_type(vietnamese_type):
    """
    Translate Vietnamese amendment types to English
    
    Args:
        vietnamese_type: Vietnamese amendment type
    
    Returns:
        English amendment type
    """
    translation_map = {
        "Thay thế": "replace",
        "Thêm": "add", 
        "Bỏ": "remove"
    }
    return translation_map.get(vietnamese_type, vietnamese_type.lower() if vietnamese_type else None)

def process_content_by_amend_type(content, amend_type, original_content=None):
    """
    Process content based on amendment type
    
    Args:
        content: New/amended content
        amend_type: Type of amendment (in English)
        original_content: Original content being amended
    
    Returns:
        Processed content
    """
    # TODO: Add your content processing logic here
    
    if amend_type == "replace":  # Replace
        # TODO: Process replacement content
        # Example: You might want to highlight changes, merge content, etc.
        processed_content = content
        
    elif amend_type == "add":  # Add
        # TODO: Process addition content
        # Example: You might want to mark it as new, combine with existing, etc.
        processed_content = f"[THÊM MỚI] {content}"
        
    elif amend_type == "remove":  # Remove
        # TODO: Process removal content
        # Example: You might want to mark as deprecated, keep for reference, etc.
        processed_content = f"[ĐÃ BỎ] {content}"
        
    else:
        # Handle other types
        processed_content = content
    
    return processed_content

def apply_amend_type_logic(chunk, amend_type, original_chunk=None):
    """
    Apply specific logic based on amendment type
    
    Args:
        chunk: The base chunk to modify
        amend_type: Type of amendment (replace, add, remove, etc.)
        original_chunk: Original chunk being amended (if available)
    
    Returns:
        Modified chunk with amendment-specific logic applied
    """
    # TODO: Add your custom logic here based on amend_type
    
    if amend_type == "replace":  # Replace
        # Process content for replacement
        chunk['content'] = process_content_by_amend_type(
            chunk['content'], amend_type, 
            original_chunk.get('content') if original_chunk else None
        )
        # TODO: Add specific replacement logic here
        
    elif amend_type == "add":  # Add
        # Add logic for addition amendments  
        # Process content for addition
        chunk['content'] = process_content_by_amend_type(chunk['content'], amend_type)
        # TODO: Add specific addition logic here
        
    elif amend_type == "remove":  # Remove
        # Add logic for removal amendments
        # Don't mark the current amend chunk as expired
        # Process content for removal
        chunk['content'] = process_content_by_amend_type(
            chunk['content'], amend_type,
            original_chunk.get('content') if original_chunk else None
        )
        # TODO: Add specific removal logic here
        
    else:
        # Handle other amendment types
        chunk['content'] = process_content_by_amend_type(chunk['content'], amend_type)
        # TODO: Add logic for other amendment types
    
    return chunk

def create_chunk(doc_code, doc_title, doc_link, doc_type, doc_issue_date, category,
                 fixed_metadata, content, analysis_doc_name, fixed_chunk=None):
    """
    Create a standardized chunk with flexible parameters
    
    Args:
        doc_code: Document code
        doc_title: Document title
        doc_link: Document link
        doc_type: Document type
        doc_issue_date: Document issue date
        category: Document category
        fixed_metadata: Metadata for the fixed content
        content: The actual content
        analysis_doc_name: Name from analysis document
        fixed_chunk: Optional reference chunk for global_ref
    
    Returns:
        Dictionary representing the chunk with amendment logic applied
    """
    chunk_id = f"{doc_code}_fixed_{analysis_doc_name}_d_{fixed_metadata['dieu']}"

    # Translate Vietnamese amendment type to English
    vietnamese_amend_type = fixed_metadata.get('type')
    english_amend_type = translate_amend_type(vietnamese_amend_type)

    # Handle global_ref requirements and chunk type
    global_ref = []
    chunk_type = 'amend'
    expired = False

    if fixed_chunk:
        # The amendment chunk will reference the original chunk's ID
        global_ref = [fixed_chunk['meta_data']['chunk_id']]

        # Update the found chunk's global_ref with this amendment
        if 'global_ref' not in fixed_chunk['meta_data']:
            fixed_chunk['meta_data']['global_ref'] = []
            
        # The reference added to the original chunk is now just the amendment's chunk_id (a string)
        amend_ref = chunk_id
        
        # Append the string ID if it's not already in the list
        if amend_ref not in fixed_chunk['meta_data']['global_ref']:
            fixed_chunk['meta_data']['global_ref'].append(amend_ref)
        
        # Handle cases where the original "dieu" becomes fully obsolete
        is_full_replacement = english_amend_type == "replace" and fixed_metadata.get('khoan') is None
        is_full_removal = english_amend_type == "remove" and fixed_metadata.get('khoan') is None

        # If the article is fully replaced or removed, the original is now expired.
        if is_full_replacement or is_full_removal:
            fixed_chunk['meta_data']['expired'] = True

        # If it's a full replacement, the new chunk becomes the 'original' version.
        if is_full_replacement:
            chunk_type = 'original'
            
        # Handle remove logic for entire dieu
        if english_amend_type == "remove" and fixed_metadata.get('khoan') is None:
            # Removing entire dieu - mark the referenced chunk as expired
            fixed_chunk['meta_data']['expired'] = True
    else:
        # Check if global_ref is required based on amendment type
        is_new_dieu = english_amend_type == "add" and fixed_metadata.get('khoan') is None

        if not is_new_dieu:
            # Change chunk type to original when global_ref can't be found
            chunk_type = 'original'
            
            # Add mark to content when global_ref can't be found
            amend_type_vietnamese = {
                'replace': 'Thay thế',
                'add': 'Thêm',
                'remove': 'Bỏ'
            }.get(english_amend_type, vietnamese_amend_type)
            
            content = f"[{amend_type_vietnamese} của văn bản {analysis_doc_name}] {content}"
            
            # For remove type, if ref can't be found, mark amend chunk as expired
            if english_amend_type == "remove":
                expired = True
    
    base_chunk = {
        'meta_data': {
            'chunk_id': chunk_id,
            'chunk_type': chunk_type,
            'amend_type': english_amend_type,
            'doc_code': doc_code,
            'doc_title': doc_title,
            'doc_link': doc_link,
            'doc_type': doc_type,
            'doc_issue_date': doc_issue_date,
            'category': category,
            'global_ref': global_ref, # This ref points from the amend chunk to the original
            'expired': expired
        },
        'content': content
    }
    
    # Apply amendment-specific logic
    if english_amend_type:
        base_chunk = apply_amend_type_logic(base_chunk, english_amend_type, fixed_chunk)
    
    return base_chunk

def extract_doc_info(doc):
    """Extract and process document information"""
    metadata = doc['original_metadata']
    doc_type = remove_vietnamese_diacritics(metadata['doc_name'].split("-")[0].lower())
    doc_code = metadata['doc_name'].split(".")[0].split("-")[-3:]
    doc_code = "/".join(doc_code)
    
    return {
        'doc_code': doc_code,
        'doc_type': doc_type,
        'doc_title': metadata['title'],
        'doc_link': metadata['link'],
        'doc_issue_date': metadata['issue_date'],
        'category': metadata['category']
    }

def process_doc_analysis(doc_info, analysis, law_chunk_data):
    """Process document analysis and create chunks"""
    chunks = []
    
    # Handle list type analysis (legacy format)
    if isinstance(analysis, list):
        for analysis_item in analysis:
            for fixed in analysis_item['doc_data']:
                fixed_chunk = get_chunk_data(
                    law_chunk_data, 
                    analysis_item['doc_name'], 
                    fixed['metadata']['dieu'], 
                    fixed['metadata'].get('khoan')
                )
                
                chunk = create_chunk(
                    doc_info['doc_code'],
                    doc_info['doc_title'],
                    doc_info['doc_link'],
                    doc_info['doc_type'],
                    doc_info['doc_issue_date'],
                    doc_info['category'],
                    fixed['metadata'],
                    fixed['content'],
                    analysis_item['doc_name'],
                    fixed_chunk
                )
                chunks.append(chunk)
    
    # Handle dictionary type analysis (new format)
    else:
        for fixed in analysis['doc_data']:
            fixed_chunk = get_chunk_data(
                law_chunk_data, 
                analysis['doc_name'], 
                fixed['metadata']['dieu'], 
                fixed['metadata'].get('khoan')
            )
            
            chunk = create_chunk(
                doc_info['doc_code'],
                doc_info['doc_title'],
                doc_info['doc_link'],
                doc_info['doc_type'],
                doc_info['doc_issue_date'],
                doc_info['category'],
                fixed['metadata'],
                fixed['content'],
                analysis['doc_name'],
                fixed_chunk
            )
            chunks.append(chunk)
    
    return chunks

def validate_amendment_data(chunk, amend_type):
    """
    Validate amendment data based on type
    
    Args:
        chunk: Chunk to validate
        amend_type: Amendment type (in English)
    
    Returns:
        Boolean indicating if data is valid, and list of validation errors
    """
    errors = []
    
    # TODO: Add your validation logic here
    if amend_type == "replace" and not chunk['meta_data'].get('global_ref'):
        errors.append("Replacement amendment missing original reference")
    
    if amend_type == "remove" and not chunk['meta_data'].get('global_ref'):
        errors.append("Removal amendment missing target reference")
    
    # Add more validation rules as needed
    
    return len(errors) == 0, errors

def get_amendment_priority(amend_type):
    """
    Get processing priority for different amendment types
    
    Args:
        amend_type: Amendment type (in English)
    
    Returns:
        Integer priority (lower = higher priority)
    """
    # TODO: Define your priority system
    priority_map = {
        "remove": 1,   # Process removals first
        "replace": 2,  # Then replacements
        "add": 3,      # Then additions
    }
    
    return priority_map.get(amend_type, 999)

def extract_qh_number(doc_code):
    """
    Extract QH number from doc_code
    
    Args:
        doc_code: Document code in format like "35/2005/QH11"
    
    Returns:
        QH number as integer, or None if not found
    """
    try:
        # Use regex to find QH followed by numbers
        qh_match = re.search(r'QH(\d+)', doc_code)
        if qh_match:
            return int(qh_match.group(1))
        return None
    except Exception as e:
        print(f"Error extracting QH number from {doc_code}: {e}")
        return None

def mark_expired_by_qh_number(chunks, qh_threshold=12):
    """
    Mark chunks as expired based on QH number
    
    Args:
        chunks: List of chunks to process
        qh_threshold: QH number threshold (chunks with QH < threshold will be marked as expired)
    
    Returns:
        Number of chunks marked as expired
    """
    expired_count = 0
    
    for chunk in chunks:
        doc_code = chunk['meta_data'].get('doc_code', '')
        qh_number = extract_qh_number(doc_code)
        
        if qh_number is not None and qh_number < qh_threshold:
            # Mark as expired if not already expired
            if not chunk['meta_data'].get('expired', False):
                chunk['meta_data']['expired'] = True
                expired_count += 1
                print(f"Marked as expired: {chunk['meta_data']['chunk_id']} (QH{qh_number})")
    
    return expired_count

def flatten_chunk(doc_data, law_chunk_data):
    """
    Main function to flatten chunk data with modular approach
    
    Args:
        doc_data: List of documents to process
        law_chunk_data: Reference law chunk data
    
    Returns:
        List of flattened chunks
    """
    chunk_data = []
    
    for doc in doc_data:
        try:
            # Extract document information
            doc_info = extract_doc_info(doc)
            print(f"Processing: {doc_info['doc_code']}, {doc_info['doc_type']}")
            
            # Process analysis and create chunks
            chunks = process_doc_analysis(doc_info, doc['analysis'], law_chunk_data)
            chunk_data.extend(chunks)
            
        except Exception as e:
            print(f"Error processing document: {e}")
            continue
    
    return chunk_data

if __name__ == "__main__":
    # Load data
    read_json_file_path = "data/json_data/nested_chunked_data/amend_law_chunked.json"
    amend_law_data = read_json_file(read_json_file_path)
    flatten_bo_luat = read_json_file("data/json_data/flatten_chunk_data/bo_luat_flatten.json")
    flatten_luat = read_json_file("data/json_data/flatten_chunk_data/luat_flatten.json")
    law_chunk_data = flatten_bo_luat + flatten_luat

    # Process chunks
    flatten_amend_law = flatten_chunk(amend_law_data, law_chunk_data)

    # Optional: Validate chunks
    for chunk in flatten_amend_law:
        amend_type = chunk['meta_data'].get('amend_type')
        if amend_type:
            is_valid, errors = validate_amendment_data(chunk, amend_type)
            if not is_valid:
                print(f"Validation errors for {chunk['meta_data']['chunk_id']}: {errors}")

    # 1. Save only amendment data
    save_to_json(flatten_amend_law, "data/json_data/flatten_chunk_data/amend_law_flatten.json")
    print(f"Processed {len(flatten_amend_law)} amendment chunks successfully!")

    # 2. Save combined data (amended law_chunk_data + amendment chunks)
    print("--- Starting final combination and save ---")

    # DEBUG: Check the lengths of the lists right before combining
    print(f"Length of original law_chunk_data (modified): {len(law_chunk_data)}")
    print(f"Length of new flatten_amend_law: {len(flatten_amend_law)}")

    combined_data = law_chunk_data + flatten_amend_law
    print(f"Length of combined_data after concatenation: {len(combined_data)}")

    # Use a try-except block to catch any error during sort or save
    try:
        print("Attempting to sort combined data...")
        # Use the SAFE sort key
        combined_data.sort(key=lambda x: x.get('meta_data', {}).get('chunk_type', 'original') != 'amend')
        print("Sorting completed successfully.")

        # NEW: Mark chunks as expired based on QH number
        print("--- Starting QH-based expiration marking ---")
        expired_count = mark_expired_by_qh_number(combined_data, qh_threshold=13)
        print(f"Marked {expired_count} chunks as expired based on QH number < 13")

        print("Attempting to save the final combined file...")
        # Save the sorted combined data
        save_to_json(combined_data, "data/json_data/flatten_chunk_data/combined_law_amend_flatten.json")
        print("Final combined file saved successfully!")
        print(f"Saved sorted combined data with {len(combined_data)} total chunks.")

    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"AN ERROR OCCURRED during the final step: {e}")
        print("The combined file was NOT saved correctly.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")