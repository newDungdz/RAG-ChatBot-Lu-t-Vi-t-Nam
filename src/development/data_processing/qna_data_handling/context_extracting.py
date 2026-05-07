import json, re

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def parse_dieu_structure(text: str):
    pattern = r"(?:khoản\s+(\d+)?\s*,?\s+)?Điều\s+(\d+)\s+([\D\s]+?)\s*(?:năm\s*)?(\d{4})"

    matches = re.findall(pattern, text)

    results = []
    for m in matches:
        results.append({
            "khoan": int(m[0]) if m[0] else None,
            "dieu": int(m[1]),
            "doc_title": m[2].strip(),
            "year": int(m[3])
        })

    print(results)

data = read_json_file("data/fine_tuning/QnA_data.json")
for item in data[:5]:
    parse_dieu_structure(item["article_content_markdown"])
    