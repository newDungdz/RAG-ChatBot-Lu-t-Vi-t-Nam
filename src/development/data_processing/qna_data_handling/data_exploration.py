import argparse
import json
from collections import Counter
from pathlib import Path

def count_doc_types(json_path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {json_path}")
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of records")

    counts = Counter()
    for item in data:
        if isinstance(item, dict):
            doc_type = item.get("doc_type")
            if doc_type is not None:
                counts[doc_type] += 1

    return counts


def main():
    input_file = "data/fine_tune/QnA_data.json"

    counts = count_doc_types(input_file)
    for doc_type, count in counts.most_common():
        print(f"{doc_type}: {count}")

if __name__ == "__main__":
    main()