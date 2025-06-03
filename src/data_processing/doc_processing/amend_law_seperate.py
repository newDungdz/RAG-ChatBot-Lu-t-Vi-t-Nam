import json
from pathlib import Path

# Define file paths
input_file = Path("data/json_data/doc_metadata/normal_law/luat_data.json")
output_file_amended = Path("data/json_data/doc_metadata/amend_law/amended_laws.json")
output_file_normal = Path("data/json_data/doc_metadata/amend_law/normal_laws.json")

# Load the JSON data
with input_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Separate the data based on the title
amended_laws = []
normal_laws = []

for item in data:
    if "sửa đổi" in item.get("title", "").lower():
        amended_laws.append(item)
    else:
        normal_laws.append(item)

# Save the amended laws to a JSON file
output_file_amended.parent.mkdir(parents=True, exist_ok=True)
with output_file_amended.open("w", encoding="utf-8") as f:
    json.dump(amended_laws, f, ensure_ascii=False, indent=4)

# Save the normal laws to a JSON file
with output_file_normal.open("w", encoding="utf-8") as f:
    json.dump(normal_laws, f, ensure_ascii=False, indent=4)

print("Data has been separated and saved successfully.")