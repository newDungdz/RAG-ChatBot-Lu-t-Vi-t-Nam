import json
import unicodedata
from collections import Counter

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")

def remove_diacritics(text: str) -> str:
    # Normalize to decomposed form (NFD) to separate base characters and diacritics
    text = text.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize('NFD', text)
    # Keep only ASCII characters, removing diacritics
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')

with open("note\\law_category_fulltext.txt", "r", encoding="utf-8") as f:
    categories = [remove_diacritics(line.replace(" ", "-").strip().lower()) for line in f.readlines()]
    

# print(categories_dict)

ignore_txt = "hỏi"

question_data = []

# print(categories_dict)
with open("note\\gen_question.txt", "r", encoding="utf-8") as f:
    count = 0
    cur_category = ""
    for line in f.readlines():
        if line == "\n" or ignore_txt in line: continue
        if remove_diacritics(line.replace(" ", "-").strip().lower()) in categories:
            # print(remove_diacritics(line.replace(" ", "-").strip().lower()))
            cur_category = remove_diacritics(line.replace(" ", "-").strip().lower())
            continue
        question_data.append({
            "label": [cur_category],
            "text": line.strip()
        })

fine_tune_data = read_json_file("output.json")
question_set = set()
for data in fine_tune_data:
    # print(len(data['analysis']))
    cur_data = {
        "label": [],
        "text": data['question_snippet']
    }
    for law_data in data['analysis']:
        # print(law_data['category'])
        if(law_data['category'] is None): continue
        if(law_data['category'] == "vi-pham-hanh-chinh"): law_data['category'] = "hanh-chinh"
        if((law_data['category'], data['question_snippet']) not in question_set):
            cur_data["label"].append(law_data['category'])
            question_set.add((law_data['category'], data['question_snippet']))
    question_data.append(cur_data)
        

save_to_json(question_data, "questions.json")

# print(categories_dict)