from docx import Document
import re, os, json, unicodedata
from tqdm import tqdm

doc_type = "luat"


def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
def bool_parse(doc: Document, merge_threshold: int = 4) -> str:
    """Parse .docx and wrap consecutive bold text runs with '**' once."""
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
            bold_active = False
        text += '\n' + line
    pattern = re.compile(r'\*\*(.{1,' + str(merge_threshold) + r'})\*\*')
    merged_text = re.sub(pattern, r'\1', text)
    
    return merged_text.strip() # Use strip() to remove leading/trailing whitespace
def extract_chuong_muc_tail(chunk: str):
    def clean_text(text):
        return ' '.join(text.replace("**", " ").split())

    def extract_section(pattern, text):
        matches = list(pattern.finditer(text))
        if not matches:
            return None, text

        match = matches[-1]
        start, end = match.span()

        # Extract the next line(s) as title
        remaining = text[end:].lstrip()
        title_lines = []
        for line in remaining.splitlines():
            if not line.strip():
                break
            # if re.match(r"^\s*(Chương|Mục)\s", line, re.IGNORECASE):  # stop if next header
            #     break
            title_lines.append(line.strip())

        full_text = match.group(1) + ' ' + ' '.join(title_lines)
        cleaned_text = clean_text(full_text)

        # Remove header and title lines from the chunk
        remove_end = end + sum(len(line) + 1 for line in title_lines)  # +1 for \n
        new_text = text[:start] + text[remove_end:]

        return cleaned_text, new_text
    chuong_pattern = re.compile(r"(?:\n|\A)\s*\**\s*(Chương\s+[IVXLCDM\d]+.*?)\s*\**\s*(?:\n|$)", re.IGNORECASE)
    muc_pattern = re.compile(r"(?:\n|\A)\s*\**\s*(Mục\s+\d+.*?)\s*\**\s*(?:\n|$)", re.IGNORECASE)

    chuong_data, chunk = extract_section(chuong_pattern, chunk)
    muc_data, chunk = extract_section(muc_pattern, chunk)

    return chuong_data, muc_data, chunk.strip()


def parse_dieu_structure(text: str):
    # Extract Điều root title (before first Khoản)
    dieu_match = re.search(r"^((?:\*\*)?Điều\s+\d+\..+?(?:\*\*)?)(?=^\d+[.-])", text, re.DOTALL | re.MULTILINE)
    dieu_content = dieu_match.group(1).strip() if dieu_match else text.strip()

    # Extract Khoản sections (1., 2., ...)
    khoan_pattern = r"^(\d+)[.-].*?(?=^\d+[.-]|\Z)"
    khoan_matches = re.finditer(khoan_pattern, text, re.DOTALL | re.MULTILINE)

    khoan_list = []

    for khoan_match in khoan_matches:
        khoan_number = khoan_match.group(1).strip()
        khoan_text = khoan_match.group(0).strip()

        # Just collect Khoản level, ignore splitting into Điểm
        khoan_list.append({
            "number": int(khoan_number),
            "content": khoan_text
        })

    if not khoan_list:
        khoan_list = None

    return dieu_content, khoan_list

def remove_last_section(content: str):
    """
    Removes everything after a horizontal rule (like "__________") in the content.
    """
    # Match a line with 3 or more underscores (possibly with surrounding spaces)
    separator_pattern = re.compile(r"\n[_\s]{3,}\n")
    
    match = separator_pattern.search(content)
    if match:
        return content[:match.start()].strip()
    
    return content.strip()

def extract_related_dieu(chunk: str):
    """Extract related Điều numbers from the chunk."""
    # Pattern to match: '**Điều <num>. ...**'
    pattern = r"Điều\s+(\d+)"
    matches = re.findall(pattern, chunk)
    seen = set()
    matches = [int(x) for x in matches if not (x in seen or seen.add(x))]
    return matches

def remove_diacritics(text: str) -> str:
    # Normalize to decomposed form (NFD) to separate base characters and diacritics
    normalized = unicodedata.normalize('NFD', text)
    # Keep only ASCII characters, removing diacritics
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')

def chunk_by_dieu(text: str) -> list[str]:    
    # Match Dieu lines in bold or plain at the beginning of a line in diacritic-removed text
    pattern = r"(?=^(\s*\*{0,2})Điều\s+\d+\.?.*?$)"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))

    # If no matches, return the entire original text as a single chunk (or empty list if text is empty)
    if not matches:
        return [text.strip()] if text.strip() else []

    chunks = []
    # Capture text before the first Dieu line, if any
    if matches[0].start() > 0:
        leading_text = text[:matches[0].start()].strip()
        if leading_text:
            chunks.append(leading_text)

    # Split into chunks based on Dieu lines, using original text
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks

doc_id = 0

def setup_chunk_data(doc_metadata, doc_folder_path):
    all_docs = os.listdir(doc_folder_path)
    docx_name = doc_metadata["doc_name"]
    if(docx_name.endswith('.doc')): docx_name = docx_name[:-4] + '.docx'
    if(not docx_name.endswith('.docx')):
        print(f"not a docx file {docx_name}")
        return None
    if(docx_name) not in all_docs:
        print(f"File not found: {docx_name}")
        return None
    docx_path = os.path.join(doc_folder_path, docx_name)
    doc = Document(docx_path)
    parsed_text = bool_parse(doc)
    # print(parsed_text)
    dieu_chunks = chunk_by_dieu(parsed_text)
    # print(dieu_chunks)
    dieu_chunks[-1] = remove_last_section(dieu_chunks[-1])
    
    content = []
    
    
    chuong_data = {
        "chuong": None,
        "muc": None,
        "dieu": []
    }
    
    for chunk in dieu_chunks:
        temp_chuong, temp_muc, processed_chunk = extract_chuong_muc_tail(chunk)
        dieu_data = extract_related_dieu(chunk)
        if(dieu_data == []):
            dieu_data.append(None)
        dieu_text, subsections = parse_dieu_structure(processed_chunk)
        chuong_data['dieu'].append({
            "number": dieu_data[0],
            "related": dieu_data[1:],
            "content": dieu_text,
            "khoan": subsections
        })
        if temp_chuong is not None or temp_muc is not None:
            content.append(chuong_data)
            chuong_data = {
                "chuong": temp_chuong if temp_chuong is not None else chuong_data["chuong"],
                "muc": temp_muc,
                "dieu": []
            }
        # print(chunk_metadata)
    content.append(chuong_data)
    global doc_id
    all_doc_data = {
        "doc_metadata": {
            "id": doc_id,
            "title": doc_metadata["title"],
            "doc_name": doc_metadata["doc_name"],
            "link": doc_metadata["link"],
            "issue_date": doc_metadata["issue_date"],
            "category": doc_metadata["category"],
            "doc_type": doc_folder_path.split("\\")[-1]
            # "summary": doc_metadata["summary"]
            },
        "doc_data": content
    }
    doc_id += 1
    return all_doc_data



doc_folder_path = f"data\\docs\\usable_doc_file\\{doc_type}"

all_doc_data = []


doc_metadata = read_json_file(f"data\\json_data\\doc_metadata\\normal_law\\{doc_type}_data.json")
# doc_metadata = read_json_file("data\\json_data\\doc_metadata\\amend_law\\normal_laws.json")

single_doc_mode = False
error_data = []
if(single_doc_mode):
    doc_path = "data\\usable_doc_file\\luat\\Luật-28-2023-QH15.docx"
    doc_name = os.path.basename(doc_path)
    doc_data = next((doc for doc in doc_metadata if doc["doc_name"] == doc_name), None)
    if not doc_data:
        doc_data = next((doc for doc in doc_metadata if doc["doc_name"] == doc_name.replace("docx", "doc")), None)
        if not doc_data:
            print(f"Metadata for {doc_name} not found.")
            exit()
    data = setup_chunk_data(doc_data, doc_folder_path)
    all_doc_data.append(data)
else:
    data = None
    with tqdm(total=len(doc_metadata), desc="Processing documents") as pbar:
        for doc_data in doc_metadata:
            pbar.set_description(f"Processing {doc_data['doc_name']}")
            try:
                data = setup_chunk_data(doc_data, doc_folder_path)
            except Exception as e:
                print(e)
                error_data.append(doc_data)
            if data is None:
                pbar.update(1)
                continue
            pbar.set_postfix({"Len": len(data['doc_data'])})
            all_doc_data.append(data)
            pbar.update(1)
save_to_json(error_data, "error_data.json")
save_to_json(all_doc_data, f"data\\json_data\\nested_chunked_data\\new_{doc_type}_data_chunked.json")



