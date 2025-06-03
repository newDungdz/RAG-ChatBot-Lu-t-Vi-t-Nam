from docx import Document # Though not used in the current logic, it was in the original snippet.
import re, os, json, unicodedata # unicodedata is not used, re is not used. os is not used.
from tqdm import tqdm # For progress bars
from datetime import datetime # For handling dates

# This variable was in your original code.
doc_type = "luat"

# Commented out sentence transformer parts from your original code
# from sentence_transformers import SentenceTransformer
# model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
# model = SentenceTransformer(model_name)

def read_json_file(json_file_path: str):
    """Reads a JSON file and returns its content."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    """Saves data to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Commented out sentence transformer function from your original code
# def encode_content(text: str):
#     """Encode text using the sup-SimCSE model via sentence-transformers."""
#     embedding = model.encode(text, convert_to_tensor=False).tolist()
#     return embedding

def flatten_chunk(doc_data, max_words=800):
    """
    Flattens nested document data into chunks, with a word limit for each chunk.

    - Each "điều" (article) is treated as a separate unit.
    - If a "điều" has "khoản" (clauses), they are added sequentially to the "điều" content.
    - If the combined word count of the "điều" and its "khoản" exceeds max_words,
      a new chunk is started for the same "điều", continuing with the remaining "khoản".
    - "Điều" from different articles are never mixed in the same chunk.
    - The 'khoan' field in meta_data will be a list of clause numbers included in that chunk.
    """
    chunk_data = [] # This list will store all the generated chunks

    # Iterate through each document in the input data
    for doc in tqdm(doc_data, desc="Processing documents"):
        metadata = doc.get('doc_metadata', {}) # Get document metadata

        # Filter out documents issued before the year 2000
        try:
            issue_date_str = metadata.get('issue_date')
            if issue_date_str: # Check if issue_date exists
                issue_year = datetime.strptime(issue_date_str, "%d/%m/%Y").year
                if issue_year < 2000:
                    # print(f"Skipping document issued in {issue_year}: {metadata.get('title')}")
                    continue # Skip to the next document
            else:
                # print(f"Skipping document due to missing issue date: {metadata.get('title')}")
                continue # Skip if no issue date
        except (ValueError, TypeError) as e:
            # print(f"Invalid or missing date format for {metadata.get('title')}: {metadata.get('issue_date')}. Error: {e}")
            continue # Skip if date format is invalid

        # Generate a document code from the document name
        doc_name = metadata.get('doc_name', '')
        doc_code_parts = doc_name.split(".")[0].split("-")
        # Take the last 3 parts for the code, or the whole thing if fewer than 3
        doc_code = "/".join(doc_code_parts[-3:]) if len(doc_code_parts) >= 3 else "-".join(doc_code_parts)
        # print(f"Processing doc_code: {doc_code}")

        # Iterate through chapters ("chuong") in the document
        for chuong in doc.get('doc_data', []):
            # Iterate through articles ("dieu") in the chapter
            for dieu in chuong.get('dieu', []):
                base_content = dieu.get('content', '') # Content of the article itself
                base_word_count = len(base_content.split()) # Word count of the article content
                dieu_number = str(dieu.get('number', '')) # Convert to string

                # Case 1: The article ("điều") has no clauses ("khoản")
                if not dieu.get('khoan'):
                    chunk = {
                        'meta_data': {
                            'chunk_id': f"{doc_code}_d_{dieu_number}",
                            'chunk_type': 'original',
                            'doc_code': doc_code,
                            'doc_title': metadata.get('title'),
                            'doc_link': metadata.get('link'),
                            'doc_type': metadata.get('doc_type'),
                            'doc_issue_date': metadata.get('issue_date'),
                            'category': metadata.get('category'),
                            'expired': metadata.get('expired', False),
                            # 'self_ref': dieu.get('related', []),
                            'global_ref': [], # Placeholder for global references
                            'chuong': chuong.get('chuong'),
                            'muc': chuong.get('muc'), # Section within a chapter
                            'dieu': dieu_number,
                            'khoan': None, # No clauses in this chunk
                        },
                        'content': base_content,
                        # 'embedding': encode_content(base_content) # If using embeddings
                    }
                    chunk_data.append(chunk)
                    continue # Move to the next article

                # Case 2: The article has clauses; build chunks respecting the word limit
                current_chunk_khoan_text_parts = [] # Stores text of clauses for the current chunk
                current_chunk_khoan_numbers = []    # Stores numbers of clauses for the current chunk
                current_word_count = base_word_count # Start with word count of the article
                chunk_part_counter = 1 # To number parts if an article is split

                # Iterate through each clause ("khoan") in the article
                for khoan_idx, khoan in enumerate(dieu.get('khoan', [])):
                    khoan_content = khoan.get('content', '')
                    khoan_number = str(khoan.get('number', '')) # Convert to string
                    # Format the clause text: "Khoản [number]. [content]"
                    khoan_full_text = f"Khoản {khoan_content}"
                    khoan_word_count = len(khoan_full_text.split())

                    # Check if adding this clause would exceed the word limit
                    # AND if there's already content in the current_chunk_khoan_text_parts (to avoid creating empty initial chunks)
                    if current_word_count + khoan_word_count > max_words and current_chunk_khoan_text_parts:
                        # Finalize and save the current chunk
                        # Content is the base article content + collected clauses
                        final_chunk_content = base_content + "\n" + "\n".join(current_chunk_khoan_text_parts)
                        khoan_ids_for_chunk_id = "_".join(map(str, current_chunk_khoan_numbers))
                        chunk_id = f"{doc_code}_d_{dieu_number}"
                        # Add a part number if the article is split and this isn't the only part
                        # A bit complex to determine if it's the *only* part without looking ahead,
                        # so we add _part_X if chunk_part_counter > 1 or if there are more khoans to process
                        is_last_khoan_of_dieu = (khoan_idx == len(dieu.get('khoan', [])) - 1)
                        if chunk_part_counter > 1 or not is_last_khoan_of_dieu : # if it's not the first part or if there are more parts to come for this dieu
                             chunk_id += f"_part_{chunk_part_counter}"
                        chunk = {
                            'meta_data': {
                                'chunk_id': chunk_id,
                                'chunk_type': 'original',
                                'doc_code': doc_code,
                                'doc_title': metadata.get('title'),
                                'doc_link': metadata.get('link'),
                                'doc_type': metadata.get('doc_type'),
                                'doc_issue_date': metadata.get('issue_date'),
                                'category': metadata.get('category'),
                                'expired': metadata.get('expired', False),
                                # 'self_ref': dieu.get('related', []),
                                'global_ref': [],
                                'chuong': chuong.get('chuong'),
                                'muc': chuong.get('muc'),
                                'dieu': dieu_number,
                                'khoan': current_chunk_khoan_numbers, # List of clause numbers in this chunk
                            },
                            'content': final_chunk_content,
                            # 'embedding': encode_content(final_chunk_content)
                        }
                        chunk_data.append(chunk)
                        chunk_part_counter += 1

                        # Reset for the new chunk (which will start with the current clause)
                        current_chunk_khoan_text_parts = []
                        current_chunk_khoan_numbers = []
                        current_word_count = base_word_count # Reset to base article word count

                    # Add the current clause to the new or ongoing chunk
                    current_chunk_khoan_text_parts.append(khoan_full_text)
                    current_chunk_khoan_numbers.append(khoan_number) # Now appending string instead of int
                    current_word_count += khoan_word_count

                # Save the last remaining chunk for the article (if any clauses were processed)
                if current_chunk_khoan_text_parts:
                    final_chunk_content = base_content + "\n" + "\n".join(current_chunk_khoan_text_parts)
                    khoan_ids_for_chunk_id = "_".join(map(str, current_chunk_khoan_numbers))
                    chunk_id = f"{doc_code}_d_{dieu_number}"
                    # Add part number if this article was split into multiple chunks
                    if chunk_part_counter > 1:
                        chunk_id += f"_part_{chunk_part_counter}"

                    chunk = {
                        'meta_data': {
                            'chunk_id': chunk_id,
                            'chunk_type': 'original',
                            'doc_code': doc_code,
                            'doc_title': metadata.get('title'),
                            'doc_link': metadata.get('link'),
                            'doc_type': metadata.get('doc_type'),
                            'doc_issue_date': metadata.get('issue_date'),
                            'category': metadata.get('category'),
                            'expired': metadata.get('expired', False),
                            # 'self_ref': dieu.get('related', []),
                            'global_ref': [],
                            'chuong': chuong.get('chuong'),
                            'muc': chuong.get('muc'),
                            'dieu': dieu_number,
                            'khoan': current_chunk_khoan_numbers, # List of clause numbers
                        },
                        'content': final_chunk_content,
                        # 'embedding': encode_content(final_chunk_content)
                    }
                    chunk_data.append(chunk)
    return chunk_data

# --- Example Usage ---
if __name__ == "__main__":
    read_json_file_path = f"data/json_data/nested_chunked_data/{doc_type}_data_chunked.json"
    law_data = read_json_file(read_json_file_path)

    # Flatten the chunk data
    flattened_data = flatten_chunk(law_data, max_words=512)
    # Save the flattened data to a JSON file
    output_path = f"data/json_data/flatten_chunk_data/{doc_type}_flatten.json"

    save_to_json(flattened_data, output_path)

    print(f"Flattened data saved to {output_path}")