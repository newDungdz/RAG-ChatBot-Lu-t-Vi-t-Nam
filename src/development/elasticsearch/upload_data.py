from elasticsearch import Elasticsearch, helpers
import json
import os

# chunks_sentence-transformers_distiluse-base-multilingual-cased-v2
# chunks_intfloat_multilingual-e5-base
# chunks_intfloat_multilingual-e5-small
# chunks_VoVanPhuc_sup-SimCSE-VietNamese-phobert-base

# --- Configuration ---
ELASTICSEARCH_HOST = "http://localhost:9200"
INDEX_NAME = "chunks".lower()
FOLDER_UPLOAD_MODE = False
FILE_FOLDER = "data\\json_data\\chunked_data"
JSON_FILE_PATH = "data\\json_data\\flatten_chunk_data\\embedded\\chunks_embeddings_intfloat_multilingual-e5-small.json"


CLOUD_ID="Legal_RAG_data:YXNpYS1zb3V0aGVhc3QxLmdjcC5lbGFzdGljLWNsb3VkLmNvbTo0NDMkYWJhZmZjOGQxNjA3NGY0Y2EwMzc4NGFhNDdlMmM1MjckNzg2YjMzY2I1NGFjNDNiZTg1NTljZDgxNTJlODJmNDA="

# --- Index mapping (will be dynamically updated with embedding dims) ---
INDEX_MAPPING_TEMPLATE = {
    "mappings": {
        "properties": {
            "content": {
                "type": "text"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 768,  # Will be updated dynamically
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "int8_hnsw",
                    "m": 16,
                    "ef_construction": 100
                }
            },
            "meta_data": {
                "properties": {
                    "amend_type": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "category": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "chuong": {"type": "keyword"},
                    "dieu": {"type": "keyword"},
                    "doc_code": {"type": "keyword"},
                    "doc_issue_date": {
                        "type": "date",
                        "format": "dd/MM/yyyy||yyyy-MM-dd"
                    },
                    "doc_link": {"type": "keyword"},
                    "doc_title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "doc_type": {"type": "keyword"},
                    "expired": {"type": "boolean"},
                    "global_ref": {"type": "keyword"},
                    "khoan": {"type": "keyword"},
                    "muc": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "self_ref": {"type": "keyword"}
                }
            }
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}

# --- Elasticsearch Connection ---
try:
    # es_client = Elasticsearch(
    #             cloud_id=CLOUD_ID,
    #             api_key=("lQRSIZcBDy4SfGpi8c3q", "iKwdTKOvjEz31ahN9r7eug")
    # )
    es_client = Elasticsearch(
        ELASTICSEARCH_HOST,
        max_retries=3,
        retry_on_timeout=True
    )
    if not es_client.ping():
        raise ValueError("Connection to Elasticsearch failed!")
    print("Successfully connected to Elasticsearch.")
except Exception as e:
    print(f"Could not connect to Elasticsearch: {e}")
    exit()

# --- Function to detect embedding dimensions and create index ---
def detect_embedding_dims_and_create_index(documents_list, index_name):
    """
    Detects embedding dimensions from the first document and creates index with correct mapping.
    """
    try:
        if not es_client.indices.exists(index=index_name):
            # Detect embedding dimensions from first document
            embedding_dims = 768  # default
            for doc in documents_list:
                if "embedding" in doc and isinstance(doc["embedding"], list):
                    embedding_dims = len(doc["embedding"])
                    print(f"Detected embedding dimensions: {embedding_dims}")
                    break
            
            # Update mapping with detected dimensions
            mapping = INDEX_MAPPING_TEMPLATE.copy()
            mapping["mappings"]["properties"]["embedding"]["dims"] = embedding_dims
            
            print(f"Index '{index_name}' does not exist. Creating it with {embedding_dims}D embeddings...")
            es_client.indices.create(index=index_name, body=mapping)
            print(f"Successfully created index '{index_name}'.")
        else:
            print(f"Index '{index_name}' already exists.")
    except Exception as e:
        print(f"Error creating index '{index_name}': {e}")
        raise

# --- Function to Generate Bulk Actions ---
def generate_actions(documents_list, index_name):
    """
    Generates actions for the Elasticsearch bulk API.
    Each document in the list will be an 'index' action.
    """
    for doc in documents_list:
        doc_id = None
        # Check for ID in meta_data.chunk_id (based on your mapping structure)
        if "meta_data" in doc and "chunk_id" in doc["meta_data"]:
            doc_id = str(doc["meta_data"]["chunk_id"])
        # Fallback to old structure if it exists
        elif "doc_metadata" in doc and "id" in doc["doc_metadata"]:
            doc_id = str(doc["doc_metadata"]["id"])
        elif "question_id" in doc:
            doc_id = str(doc['question_id'])
        
        action = {
            "_index": index_name,
            "_source": doc
        }
        if doc_id:
            action["_id"] = doc_id

        yield action

# --- Main Script Logic ---
def main():
    # Load documents from the JSON file first
    try:
        if FOLDER_UPLOAD_MODE:
            documents_to_index = []
            for json_file in os.listdir(FILE_FOLDER):
                json_path = os.path.join(FILE_FOLDER, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                json_data = [doc for doc in json_data if doc is not None]
                documents_to_index.extend(json_data)
                print(f"Successfully loaded {len(json_data)} documents from {json_path}")
        else:
            with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                documents_to_index = json.load(f)
            documents_to_index = [doc for doc in documents_to_index if doc is not None]
            print(f"Successfully loaded {len(documents_to_index)} documents from {JSON_FILE_PATH}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not isinstance(documents_to_index, list):
        print("Error: The JSON file should contain a list of documents.")
        return

    if not documents_to_index:
        print("No documents found in the JSON file to index.")
        return

    # Now create index with detected embedding dimensions
    detect_embedding_dims_and_create_index(documents_to_index, INDEX_NAME)
    
    # Perform bulk indexing
    print(f"Starting bulk indexing to '{INDEX_NAME}'...")
    try:
        successes, errors = helpers.bulk(
            es_client.options(request_timeout=60),
            generate_actions(documents_to_index, INDEX_NAME),
            chunk_size=100,
            raise_on_error=False,
        )

        print(f"Successfully indexed {successes} documents.")
        if errors:
            print(f"Encountered {len(errors)} errors during bulk indexing:")
            for i, error in enumerate(errors[:10]):
                print(f"  Error {i+1}: {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors.")
    except helpers.BulkIndexError as e:
        print(f"Bulk indexing failed with {len(e.errors)} errors.")
        for i, error_detail in enumerate(e.errors[:10]):
            print(f"  Error {i+1}: {error_detail}")
        if len(e.errors) > 10:
            print(f"  ... and {len(e.errors) - 10} more errors.")
    except Exception as e:
        print(f"An unexpected error occurred during bulk indexing: {e}")

if __name__ == "__main__":
    main()