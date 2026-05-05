# System Overview

This project builds a Retrieval-Augmented Generation (RAG) system for Vietnamese legal documents sourced from:
https://luatvietnam.vn/

The main objective is to create a precise and controllable semantic retrieval layer, tailored for legal texts with complex hierarchical structures (Chapter – Article – Clause), where accuracy and traceability are critical.

<img width="1200" height="585" alt="Data Processing Pipeline" src="https://github.com/user-attachments/assets/6c618a52-08ae-4afe-b857-e8792dfb0e44" />
<p align="center">
  <em>Figure 1: Dataset Processing Pipeline</em>
</p>
<img width="1084" height="652" alt=" RAG System Architecture" src="https://github.com/user-attachments/assets/b60f45de-57cb-4ddb-8f79-8dc7db9538ee" />
<p align="center">
  <em>Figure 2: RAG System Architecture</em>
</p>
---

# Usage Guide

## Requirements

- Python environment (e.g., VSCode)
- Docker installed

---

## Environment Setup

Create a file named `.env` inside `src/chatbot` with the following content:
```env
GOOGLE_API_KEY=<your Gemini API Key>  
FLASK_ENV=development  
ELASTICSEARCH_HOST=elasticsearch  
ELASTICSEARCH_PORT=9200  
LOCAL_MODE=True  
```

## Running Elasticsearch Locally

1. Open the file:
   src/chatbot/.env

2. Set:
   LOCAL_MODE=True

3. Download dataset from Google Drive:
   https://drive.google.com/drive/folders/176TyeZvesvSbiMOnK3cTo8hBISdtJfgL

4. Choose any JSON file  
   Recommended:
   chunks_embeddings_intfloat_multilingual-e5-small.json

5. Update the embedding model in your Dockerfile

   Find this line:
   ```docker
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
   ```
   Replace the model name if you are using a different embedding model.


## Start Elasticsearch

Run:
```bash
docker-compose up --build
```

## Upload Data to Elasticsearch

1. Navigate to:
   src/elasticsearch/upload_data.py

2. Open the script and update:
   ```python
   JSON_FILE_PATH
   ```
   Set it to the path of your downloaded JSON file.

4. Install Elasticsearch Python client (if needed):
```bash
pip install elasticsearch
```
4. Run the script to upload data into Elasticsearch.

## Optional: Enable Kibana

- Open docker-compose.yml
- Uncomment the Kibana container
- Restart Docker

You can now use Kibana to explore and analyze indexed data.

## Access the Application

After everything is running, open the URL shown in Docker logs.

Default:
http://localhost:5000

---

# Project Resources

## Google Drive (Full Project)

https://drive.google.com/drive/folders/10aCBigAKkBziuXMHkBcd70D96c3vAqDM

Contents include:
- Raw dataset and processed (chunked) dataset
- Installation video and demo video
- Detailed project documentation

---

## Project Report

https://docs.google.com/document/d/1RaDzGh5KaLjDyYfc6lCLI4380iur3DR4d6obFOoL7lo/edit?usp=sharing
