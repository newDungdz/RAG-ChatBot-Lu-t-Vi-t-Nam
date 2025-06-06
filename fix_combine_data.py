import json

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

answer_data = read_json_file("answers_mistral_openrouter.json")
context_data = read_json_file("query_with_truth.json")
context_dict = {item["question"]: item for item in context_data}


new_data = []
for item in answer_data:
    context_sample = context_dict.get(item["question"])
    new_data.append({
        "id": item["id"],
        "question": item["question"],
        "truth_answer": context_sample["answer"],
        "answer_generation": item["answer_generation"],
        "contexts": context_sample["relevant_chunks"]
    })

save_to_json(new_data, "combined_data.json")