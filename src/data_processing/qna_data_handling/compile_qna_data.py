import json

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")

qna_links = read_json_file("data\json_data\QnA\QnA_links.json")
qna_details = read_json_file("data\json_data\QnA\QnA_info.json")

all_data = []

for detail in qna_details:
    # Find matching link data by id
    if(detail is None): continue
    matching_link = next((link for link in qna_links if link.get('id') == detail.get('question_id')), None)
    if matching_link:
        # Merge the data
        merged_item = {
            "id" : detail['question_id'],
            "question": matching_link['title'],
            "answer": detail['answer'].replace("Trả lời:", "").strip(),
            "related_docs": detail['related_links']
        }
        all_data.append(merged_item)

# Save the merged data to a new JSON file
output_path = "data\json_data\QnA\merged_qna_data.json"
save_to_json(all_data, output_path)