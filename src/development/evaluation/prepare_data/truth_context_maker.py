from elasticsearch import Elasticsearch
import json
from tqdm import tqdm
import subprocess

def read_json_file(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
def save_to_json(data, output_path):
    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

ELASTICSEARCH_HOST = "http://localhost:9200" 
es_index='chunks_intfloat_multilingual-e5-base', 

processed_datas = read_json_file("data/evalations/eval_bm25_250/pre_retrive_query.json")['queries']

qna_data = read_json_file("qna.json")

qna_data = {data['question'] : data for data in qna_data}

client = Elasticsearch(
        ELASTICSEARCH_HOST,
        max_retries=3,
        retry_on_timeout=True
    )
max_chunk_len = 10

all_data = []

for pr_datas in tqdm(processed_datas, "Processing", len(processed_datas)):
    if(len(pr_datas['retrieved_laws']) < 4):
        print("Too little relevant data, skiping...")
        continue
    relevant_chunk_ids = pr_datas['retrieved_laws']
    chunks_data = []
    related_query = {
                    "query": {
                        "terms": {
                            "meta_data.chunk_id": relevant_chunk_ids
                        }
                    },
                    "_source": ["content", "meta_data"],
                    "size": 10  # Increased size to handle multiple related chunks
                }
    response = client.search(index=es_index, body=related_query)

    # Process and filter related chunks
    chunks_data = {
        'id': pr_datas['id'],
        'question': pr_datas['question'],
        'answer': qna_data[pr_datas['question']]['answer'],
        'relevant_chunks': []
    }
    for hit in response['hits']['hits']:
        source = hit['_source']
        meta_data = source['meta_data']
        
        chunk = {
            "content": source['content'],
            "doc_issue_date": meta_data.get('doc_issue_date', ''),
            "doc_title": meta_data.get('doc_title', ''),
            "chunk_id": meta_data.get('chunk_id', ''),
            "dieu_number": meta_data.get('dieu', ''),
        }
        chunks_data['relevant_chunks'].append(chunk)
    remain_chunk = max_chunk_len - len(pr_datas['retrieved_laws'])
    for noise_chunk in pr_datas['retrival_results']:
        if(remain_chunk <= 0): break
        if(noise_chunk['chunk_id'] not in relevant_chunk_ids):
            chunks_data['relevant_chunks'].append(noise_chunk)
            remain_chunk -= 1
    all_data.append(chunks_data)

print(len(all_data))
save_to_json(all_data, "query_with_truth.json")

