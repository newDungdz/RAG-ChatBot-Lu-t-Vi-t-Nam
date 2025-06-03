from elasticsearch import Elasticsearch
import sys

# --- Configuration ---
ELASTICSEARCH_HOST = "http://localhost:9200"  # Replace if your Elasticsearch is elsewhere
INDEX_NAME = "chunks"                      # The name of your Elasticsearch index

# chunks_sentence-transformers_distiluse-base-multilingual-cased-v2
# chunks_intfloat_multilingual-e5-base
# chunks_intfloat_multilingual-e5-small
# chunks_VoVanPhuc_sup-SimCSE-VietNamese-phobert-base

# --- Elasticsearch Connection ---
try:
    es_client = Elasticsearch(
        ELASTICSEARCH_HOST,
        max_retries=3,
        retry_on_timeout=True
    )
    # Test connection
    if not es_client.ping():
        raise ValueError("Connection to Elasticsearch failed!")
    print("Successfully connected to Elasticsearch.")
except Exception as e:
    print(f"Could not connect to Elasticsearch: {e}")
    sys.exit(1)

# --- Function to Delete All Documents ---
def delete_all_documents(index_name):
    """
    Deletes all documents in the specified Elasticsearch index using delete_by_query.
    """
    try:
        # Check if the index exists
        if not es_client.indices.exists(index=index_name):
            print(f"Index '{index_name}' does not exist. Nothing to delete.")
            return

        # Perform delete_by_query with a match_all query
        response = es_client.delete_by_query(
            index=index_name,
            body={"query": {"match_all": {}}},
            wait_for_completion=True,  # Wait for the operation to complete
            request_timeout=  60     # Set a timeout for the operation
        )

        print(f"Successfully deleted {response['deleted']} documents from index '{index_name}'.")
        if response['deleted'] == 0:
            print(f"No documents were found in index '{index_name}'.")
        if response['failures']:
            print("Some documents could not be deleted due to failures:")
            for failure in response['failures']:
                print(f"  - {failure}")

    except Exception as e:
        print(f"An error occurred while deleting documents: {e}")

# --- Main Script Logic ---
if __name__ == "__main__":
    # Check if index exists before attempting deletion
    if es_client.indices.exists(index=INDEX_NAME):
        print(f"Starting deletion of all documents in index '{INDEX_NAME}'...")
        delete_all_documents(INDEX_NAME)
    else:
        print(f"Index '{INDEX_NAME}' does not exist. Nothing to delete.")