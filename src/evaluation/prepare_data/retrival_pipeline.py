import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
import json, os
import google.generativeai as genai
import logging
from typing import List, Dict
import time, requests

API_KEY = "AIzaSyCKtN98H-n2idRhIgWpvzcw-4cqdzik9rE"
# AIzaSyBq4HTkU_PWUyHh7NmOuFPSjgzMQI86CCo
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"



class RAGLawRetrieval:
    def __init__(self, es_host='localhost', es_port=9200, es_index='chunks', 
                embedding_model='VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', query_process_model = 'meta-llama/llama-3.3-8b-instruct:free'):
        """
        Initialize the Elasticsearch Legal Retrieval Flow
        
        Parameters:
        -----------
        es_host : str
            Elasticsearch host
        es_port : int
            Elasticsearch port
        es_index : str
            Elasticsearch index name
        embedding_model : str
            Sentence transformer model for embeddings
        classify_tech : str
            Choose LLM or bert model for query classification ('llm', 'bert')
        """
        # Connect to Elasticsearch
        self.es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': 'http'}])
        self.index_name = es_index
        
        # Get legal categories from Elasticsearch
        self.legal_categories_fulltext = self._get_legal_categories()
        with open("note\\law_category.txt", "r") as file:
            categories = file.readlines()
            categories = [cate.strip() for cate in categories]  
        self.legal_categories = categories

        # Load models
        print("Using Classifier:", query_process_model)
        print("Using Embeder:", embedding_model)
        if("gemini" in query_process_model):
            genai.configure(api_key=API_KEY)
            self.gemini_model = genai.GenerativeModel(query_process_model)
            self._expand_and_classify_query = self._expand_and_classify_query_with_gemini
        else:
            self.openrouter_model = query_process_model
            self._expand_and_classify_query = self._expand_and_classify_query_with_openrouter
        self.sentence_encoder = SentenceTransformer(embedding_model)
        
        # Configure Gemini for query expansion
        
    
    def _get_legal_categories(self):
        """
        Get unique legal categories from Elasticsearch
        
        Returns:
        --------
        list
            List of unique legal categories
        """
        with open("note\\law_category_fulltext.txt", "r", encoding="utf-8") as file:
            categories = file.readlines()
            categories = [cate.strip() for cate in categories if cate.strip() != ""]
        return categories
    
    def _load_query_classifier(self):
        """
        Load a lightweight sentence classification model for query classification

        Returns:
        --------
        dict
            Dictionary containing tokenizer and model
        """
        model_name = "distilbert-base-uncased"
        model_dir = f"{model_name}"
        
        # Check if model already exists locally
        if not os.path.exists(model_dir):
            print(f"Model not found locally. Downloading to {model_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.legal_categories_fulltext)
            )
            # Save the downloaded model and tokenizer
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
        else:
            print(f"Loading model from local directory {model_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=len(self.legal_categories_fulltext)
            )

        return {"tokenizer": tokenizer, "model": model}

    def _generate_content_openrouter(self, prompt: str, max_tokens: int = 500) -> str:
            """
            Generate content using OpenRouter API as replacement for Gemini calls.
            
            Parameters:
            -----------
            prompt : str
                The prompt to send to the model
            max_tokens : int
                Maximum number of tokens to generate (default: 500)
                
            Returns:
            --------
            str
                Generated content from the model, or empty string if error occurs
            """
            try:
                # Validate inputs
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Prompt must be a non-empty string")
                    
                # OpenRouter API configuration
                OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
                
                if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
                    print("ERROR: Please replace 'YOUR_OPENROUTER_API_KEY' with your actual OpenRouter API key.")
                    return None

                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    # Optional: You can set a site URL or app name for OpenRouter to identify your application
                    # "HTTP-Referer": "YOUR_SITE_URL", # e.g., https://yourapp.com
                    # "X-Title": "YOUR_APP_NAME", # e.g., My Awesome App
                }

                # Construct the payload according to the OpenAI chat completions format
                # which OpenRouter generally follows.
                payload = {
                    "model": self.openrouter_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    # You can add other parameters here, like temperature, top_p, etc.
                    "temperature": 0.7,
                }

                # Make the POST request to the OpenRouter API
                response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload))

                # Raise an exception for HTTP errors (4xx or 5xx)
                response.raise_for_status()

                # Return the JSON response
                api_response =  response.json()
            
                # Extract the generated content
                try:
                    generated_content = api_response["choices"][0]["message"]["content"]
                    return generated_content.strip()
                except (KeyError, IndexError) as e:
                    logging.error(f"Could not extract content from OpenRouter response: {e}")
                    return ""
                    
            except requests.exceptions.HTTPError as http_err:
                logging.error(f"HTTP error occurred in OpenRouter request: {http_err}")
                if hasattr(http_err, 'response') and http_err.response:
                    logging.error(f"Response content: {http_err.response.content.decode()}")
                return ""
            except requests.exceptions.RequestException as req_err:
                logging.error(f"Request error occurred in OpenRouter request: {req_err}")
                return ""
            except Exception as e:
                logging.error(f"Unexpected error in OpenRouter request: {e}")
                return ""

    def generate_analysis_prompt(self, query, categories, categories_fulltext, top_k , enable_expansion = True):
        if enable_expansion: 
            return f"""Bạn cần thực hiện 2 nhiệm vụ cho câu truy vấn pháp lý sau: "{query}"

                NHIỆM VỤ 1: ĐÁNH GIÁ VÀ MỞ RỘNG TRUY VẤN
                Đánh giá xem câu truy vấn có cần mở rộng hay không dựa trên:
                - Câu truy vấn quá ngắn (ít hơn 5 từ)
                - Câu truy vấn quá trừu tượng hoặc mơ hồ
                - Sử dụng từ viết tắt
                - Thiếu ngữ cảnh pháp lý cụ thể
                - Cần thêm từ khóa liên quan để tìm kiếm hiệu quả hơn

                Nếu xác định rằng truy vấn cần mở rộng, hãy viết lại một câu truy vấn đầy đủ hơn, tuân theo các yêu cầu:
                1. Giữ nguyên ý định và nội dung chính của câu truy vấn gốc
                2. Bổ sung từ khóa pháp lý hoặc thuật ngữ chuyên ngành liên quan trực tiếp
                3. Làm rõ ngữ cảnh nếu đang thiếu (thời gian, hành vi, đối tượng…)
                4. Không thêm thông tin ngoài phạm vi truy vấn ban đầu
                5. Không sử dụng từ viết tắt — phải viết đầy đủ
                6. Câu truy vấn mở rộng phải tự nhiên, giống như một người thật hỏi

                NHIỆM VỤ 2: PHÂN LOẠI TRUY VẤN
                Phân loại câu truy vấn (sử dụng bản mở rộng nếu có) vào các danh mục pháp lý sau: {categories}

                Bối cảnh danh mục:
                {categories_fulltext}

                Chọn top {top_k} danh mục phù hợp nhất với độ tin cậy từ 0 đến 1, chú ý.

                TRẢ LỜI BẰNG JSON FORMAT:
                {{
                    "expansion": {{
                        "original_query": "{query}",
                        "expanded_query": "câu truy vấn đã mở rộng (hoặc giống nguyên bản nếu không cần mở rộng)",
                        "needs_expansion": true/false,
                        "expansion_reason": "lý do cần/không cần mở rộng"
                    }},
                    "classification": [
                        {{"category": "tên danh mục", "confidence": 0.x}},
                        {{"category": "tên danh mục", "confidence": 0.x}}
                    ]
                }}

                Chỉ trả về JSON, không giải thích thêm."""
        else:
            # Simple classification-only prompt (original style)
            return f"""Phân loại câu truy vấn pháp lý sau vào các danh mục pháp lý: {categories}

            Câu truy vấn: "{query}"

            Bối cảnh danh mục:
            {categories_fulltext}

            Chọn top {top_k} danh mục phù hợp nhất với độ tin cậy từ 0 đến 1.

            TRẢ LỜI BẰNG JSON FORMAT:
            {{
                "expansion": {{
                    "original_query": "{query}",
                    "expanded_query": "{query}",
                    "needs_expansion": false,
                    "expansion_reason": "Query expansion disabled"
                }},
                "classification": [
                    {{"category": "tên danh mục", "confidence": 0.x}},
                    {{"category": "tên danh mục", "confidence": 0.x}}
                ]
            }}

            Chỉ trả về JSON, không giải thích thêm."""
            
    def _expand_and_classify_query_with_openrouter(self, query: str, top_k: int, enable_expansion: bool = True) -> Dict:
        """
        MERGED FUNCTION: Expand query if needed and classify into legal categories using OpenRouter API.
        
        Parameters:
        -----------
        query : str
            Original user query
        top_k : int
            Number of top categories to return
        enable_expansion : bool
            Whether to enable query expansion (default: True)
            
        Returns:
        --------
        dict
            Dictionary containing both expansion and classification results:
            {
                "expansion": {
                    "original_query": str,
                    "expanded_query": str,
                    "needs_expansion": bool,
                    "expansion_reason": str
                },
                "classification": [
                    {"category": str, "confidence": float}
                ]
            }
        """
        try:
            # Validate inputs
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string")
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError("top_k must be a positive integer")

            # Prepare categories for classification
            categories = ", ".join(self.legal_categories)
            categories_fulltext = ", ".join(self.legal_categories_fulltext)
            
            # Create different prompts based on expansion setting
            combined_prompt = self.generate_analysis_prompt(query, categories, categories_fulltext, top_k, enable_expansion)

            # Make single API request
            response_text = self._generate_content_openrouter(combined_prompt, max_tokens=800)
            
            if not response_text:
                logging.error("No response from OpenRouter API")
                return self._get_fallback_result(query, top_k)
            
            # Parse combined response
            try:
                # Handle potential JSON formatting issues
                if response_text.startswith("```json") and response_text.endswith("```"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```") and response_text.endswith("```"):
                    response_text = response_text[3:-3].strip()
                
                result = json.loads(response_text)
                
                # Validate structure
                if not isinstance(result, dict) or "expansion" not in result or "classification" not in result:
                    raise ValueError("Invalid response structure")
                
                # Validate expansion result
                expansion = result["expansion"]
                if not isinstance(expansion, dict):
                    raise ValueError("Invalid expansion format")
                
                # Validate classification result
                classification = result["classification"]
                if not isinstance(classification, list):
                    raise ValueError("Invalid classification format")
                
                # Filter and sort classification results
                valid_categories = [
                    {"category": c["category"], "confidence": float(c["confidence"])}
                    for c in classification
                    if isinstance(c, dict) and c.get("category") in self.legal_categories
                ]
                valid_categories = sorted(valid_categories, key=lambda x: x["confidence"], reverse=True)[:top_k]
                
                return {
                    "expansion": {
                        "original_query": expansion.get("original_query", query),
                        "expanded_query": expansion.get("expanded_query", query),
                        "needs_expansion": expansion.get("needs_expansion", False),
                        "expansion_reason": expansion.get("expansion_reason", "")
                    },
                    "classification": valid_categories
                }

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logging.error(f"Error parsing combined API response: {str(e)}")
                return self._get_fallback_result(query, top_k)

        except Exception as e:
            logging.error(f"Error in combined query expansion and classification: {str(e)}")
            return self._get_fallback_result(query, top_k)

    def _expand_and_classify_query_with_gemini(self, query: str, top_k: int, enable_expansion = True) -> Dict:
        """
        MERGED FUNCTION: Expand query if needed and classify into legal categories using Gemini API.
        
        Parameters:
        -----------
        query : str
            Original user query
        top_k : int
            Number of top categories to return
            
        Returns:
        --------
        dict
            Dictionary containing both expansion and classification results
        """
        try:
            # Validate inputs
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string")
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError("top_k must be a positive integer")

            # Prepare categories for classification
            categories = ", ".join(self.legal_categories)
            categories_fulltext = ", ".join(self.legal_categories_fulltext)
            
            # Combined prompt for both expansion and classification
            combined_prompt = self.generate_analysis_prompt(query, categories, categories_fulltext, top_k, enable_expansion)

            # Make single API request
            response = self.gemini_model.generate_content(combined_prompt,
                                                        generation_config=genai.types.GenerationConfig(
                                                                temperature=0.2  # <-- adjust this value
                                                            ))
            response_text = response.text.strip()
            
            # Parse combined response
            try:
                # Handle potential JSON formatting issues
                if response_text.startswith("```json") and response_text.endswith("```"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```") and response_text.endswith("```"):
                    response_text = response_text[3:-3].strip()
                
                result = json.loads(response_text)
                
                # Validate structure
                if not isinstance(result, dict) or "expansion" not in result or "classification" not in result:
                    raise ValueError("Invalid response structure")
                
                # Validate expansion result
                expansion = result["expansion"]
                if not isinstance(expansion, dict):
                    raise ValueError("Invalid expansion format")
                
                # Validate classification result
                classification = result["classification"]
                if not isinstance(classification, list):
                    raise ValueError("Invalid classification format")
                
                # Filter and sort classification results
                valid_categories = [
                    {"category": c["category"], "confidence": float(c["confidence"])}
                    for c in classification
                    if isinstance(c, dict) and c.get("category") in self.legal_categories
                ]
                valid_categories = sorted(valid_categories, key=lambda x: x["confidence"], reverse=True)[:top_k]
                
                return {
                    "expansion": {
                        "original_query": expansion.get("original_query", query),
                        "expanded_query": expansion.get("expanded_query", query),
                        "needs_expansion": expansion.get("needs_expansion", False),
                        "expansion_reason": expansion.get("expansion_reason", "")
                    },
                    "classification": valid_categories
                }

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logging.error(f"Error parsing combined API response: {str(e)}")
                return self._get_fallback_result(query, top_k)

        except Exception as e:
            logging.error(f"Error in combined query expansion and classification: {str(e)}")
            return self._get_fallback_result(query, top_k)

    def _get_fallback_result(self, query: str, top_k: int) -> Dict:
        """
        Fallback method when API calls fail - provides basic expansion and error classification
        """
        needs_expansion = len(query.split()) < 5
        return {
            "expansion": {
                "original_query": query,
                "expanded_query": query,
                "needs_expansion": needs_expansion,
                "expansion_reason": "Fallback: API call failed, using original query"
            },
            "classification": [{"category": "error", "confidence": 0.0}] * min(top_k, 1)
        }
    
    def process_query(self, query, top_k_categories=3, top_bm25=100, top_k_chunks=5, enable_expanson = True):
        """
        Process a legal query through the retrieval flow with merged query expansion and classification
        
        Parameters:
        -----------
        query : str
            User's legal query
        top_k_categories : int
            Number of top categories to consider
        top_bm25 : int
            Number of candidates from BM25 search
        top_k_chunks : int
            Number of top chunks to retrieve after reranking
            
        Returns:
        --------
        dict
            Dictionary containing results from each step of the flow
        """
        results = {}
        
        if(isinstance(query, str)):   
            # STEP 1: Query Expansion and Classification 
            combined_result = self._expand_and_classify_query(query, top_k_categories, enable_expanson)
        else:
            combined_result = {
                'expansion': query['expanded_query'],
                'classification': query['category']
            }
        
        
        # Store results from merged operation
        results["step0_query_expansion"] = combined_result["expansion"]
        results["step1_top_categories"] = combined_result["classification"]
        
        # Use expanded query for subsequent steps
        search_query = combined_result["expansion"]["expanded_query"]
        top_categories = [classify for classify in combined_result["classification"] if float(classify['confidence']) > 0.5]
        
        # Step 2: BM25 Search in Combined Chunks of all chosen Categories
        bm25_candidates = self._bm25_search(search_query, top_categories, max_candidates=top_bm25)
        results["step2_bm25_candidates"] = bm25_candidates
        
        # Step 3: Rerank using Elasticsearch Vector Search
        reranked_chunks = self._vector_rerank_with_elasticsearch(search_query, bm25_candidates, top_k_chunks)
        results["step3_reranked_chunks"] = reranked_chunks
        
        # Step 4: Retrieve and Append Related Chunks
        final_chunks = self._append_related_chunks(reranked_chunks)
        results["final_chunks"] = final_chunks
        
        return results
    
    
    def _bm25_search(self, query, top_categories, max_candidates=1000):
        """
        Step 2: BM25 Search in Combined Chunks of all chosen Categories
        
        Parameters:
        -----------
        query : str
            User's legal query
        top_categories : list
            List of top categories with confidence scores
        max_candidates : int
            Maximum number of candidate chunks to return
            
        Returns:
        --------
        list
            List of candidate chunks with BM25 scores
        """
        category_filters = [{"term": {"meta_data.category": category_info["category"]}} 
            for category_info in top_categories]

        # category_filters = [{"term": {"meta_data.category": category_info["category"]}} 
        #            for category_info in top_categories]

        es_query = {
        "size": max_candidates,
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "content": query
                    }
                },
                "filter": [
                    {
                        "bool": {
                            "should": category_filters
                        }
                    },
                    {
                        "term": {
                            "meta_data.chunk_type": "original"
                        }
                    },
                    {
                        "term": {
                            "meta_data.expired": False
                        }
                    },
                    {
                        "bool": {
                            "must": {
                                "exists": {
                                    "field": "meta_data.chuong"
                                }
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["content", "meta_data", "embedding"]  # Include embedding field
    }

        
        # Execute search
        response = self.es.search(index=self.index_name, body=es_query)
        # print(category_filters)
        # print(response)
        # Process search results
        candidates = []
        
        for hit in response['hits']['hits']:
            score = hit['_score']
            source = hit['_source']
            meta_data = source['meta_data']
            content = source['content']
            embedding = source.get('embedding', None)  # Get embedding from Elasticsearch
            
            # Create candidate for each matching chunk
            candidate = {
                "doc_id": hit['_id'],
                "doc_title": meta_data.get('doc_title', ''),
                "chunk_id": meta_data.get('chunk_id', ''),
                "dieu_number": meta_data.get('dieu', ''),
                "chuong": meta_data.get('chuong', ''),
                "muc": meta_data.get('muc', ''),
                "content": content,
                "bm25_score": score,
                "related": meta_data.get('global_ref', []) if isinstance(meta_data.get('global_ref', []), list) else 
                           [meta_data.get('global_ref')] if meta_data.get('global_ref') else []
            }
            candidates.append(candidate)
        
        # Sort by BM25 score and take top candidates
        candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
        return candidates[:max_candidates]
    
    def _vector_rerank_with_elasticsearch(self, query, candidates, top_k):
        """
        Step 3: Rerank Chunks Using Elasticsearch Vector Search (NEW METHOD)
        
        This method uses Elasticsearch's vector search capabilities instead of 
        computing embeddings and similarities locally.
        
        Parameters:
        -----------
        query : str
            User's legal query
        candidates : list
            List of candidate chunks from BM25 search
        top_k : int
            Number of top chunks to return after reranking
            
        Returns:
        --------
        list
            List of top k chunks after vector-based reranking
        """
        if not candidates:
            return []
        
        try:
            if('intfloat' in self.index_name):
                query = f"query: {query}"
            # Generate query embedding
            query_embedding = self.sentence_encoder.encode([query], normalize_embeddings=True)[0].tolist()
            
            # Extract document IDs from candidates for filtering
            candidate_doc_ids = [candidate["doc_id"] for candidate in candidates]
            
            # Perform vector search on the candidates using Elasticsearch
            vector_search_query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "script_score": {
                                    "query": {
                                        "bool": {
                                            "filter": [
                                                {
                                                    "ids": {
                                                        "values": candidate_doc_ids
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {
                                            "query_vector": query_embedding
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "_source": ["content", "meta_data", "embedding"]
            }
            
            # Execute vector search
            response = self.es.search(index=self.index_name, body=vector_search_query)
            
            # Process results and combine with original candidate data
            reranked_chunks = []
            for hit in response['hits']['hits']:
                vector_score = hit['_score']
                doc_id = hit['_id']
                
                # Find the original candidate to preserve all information
                original_candidate = next((c for c in candidates if c["doc_id"] == doc_id), None)
                
                if original_candidate:
                    # Create reranked chunk with vector similarity score
                    reranked_chunk = original_candidate.copy()
                    reranked_chunk["vector_similarity_score"] = float(vector_score - 1.0)  # Subtract 1.0 added in script
                    reranked_chunk["combined_score"] = float(vector_score)  # Keep the raw ES score for sorting
                    reranked_chunks.append(reranked_chunk)
            
            # print(f"Vector reranking completed: {len(reranked_chunks)} chunks reranked")
            return reranked_chunks
            
        except Exception as e:
            logging.error(f"Error in vector reranking with Elasticsearch: {str(e)}")
            # Fallback to original embedding-based reranking
            print("Falling back to local embedding similarity calculation...")
            return self._rerank_chunks_fallback(query, candidates, top_k)
    
    def _append_related_chunks(self, top_chunks):
        """
        Step 4: Retrieve Related Chunks and Append to Original Chunks
        
        Parameters:
        -----------
        top_chunks : list
            List of top chunks after reranking
            
        Returns:
        --------
        list
            List of chunks with related content appended
        """
        enhanced_chunks = []
        
        for chunk in top_chunks:
            enhanced_chunk = chunk.copy()  # Create a copy to avoid modifying original
            related_refs = chunk.get("related", [])
            
            if not related_refs:
                enhanced_chunks.append(enhanced_chunk)
                continue
            
            # Construct query to find related chunks by chunk_id
            related_query = {
                "query": {
                    "terms": {
                        "meta_data.chunk_id": related_refs
                    }
                },
                "_source": ["content", "meta_data"],
                "size": 50  # Increased size to handle multiple related chunks
            }
            
            # Execute search for related chunks
            try:
                response = self.es.search(index=self.index_name, body=related_query)
                
                # Process and filter related chunks
                related_chunks_data = []
                for hit in response['hits']['hits']:
                    source = hit['_source']
                    meta_data = source['meta_data']
                    
                    related_chunk_data = {
                        "content": source['content'],
                        "doc_issue_date": meta_data.get('doc_issue_date', ''),
                        "doc_title": meta_data.get('doc_title', ''),
                        "chunk_id": meta_data.get('chunk_id', ''),
                        "dieu_number": meta_data.get('dieu', ''),
                    }
                    related_chunks_data.append(related_chunk_data)
                
                # Filter to get only the latest related chunk by doc_issue_date
                if related_chunks_data:
                    # Sort by doc_issue_date in descending order (latest first)
                    # Handle cases where doc_issue_date might be empty or None
                    def parse_date_for_sorting(date_str):
                        if not date_str:
                            return '0000-00-00'  # Default for empty dates
                        
                        # Handle dd/MM/yyyy format
                        if '/' in date_str and len(date_str.split('/')) == 3:
                            try:
                                day, month, year = date_str.split('/')
                                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            except:
                                return '0000-00-00'
                        
                        # Handle yyyy-MM-dd format (already in sortable format)
                        elif '-' in date_str and len(date_str.split('-')) == 3:
                            return date_str
                        
                        return '0000-00-00'  # Default for unrecognized formats
                    
                    related_chunks_data.sort(
                        key=lambda x: parse_date_for_sorting(x['doc_issue_date']), 
                        reverse=True
                    )
                    
                    # Get the latest related chunk
                    latest_related = related_chunks_data[0]
                    
                    # Append related content to original chunk content
                    if latest_related['content'].strip():
                        enhanced_chunk['content'] = (
                            chunk['content'] + 
                            "\n\nThông tin sửa đổi: " + 
                            latest_related['content']
                        )
                        
                        # Optionally add metadata about the related chunk
                        enhanced_chunk['related_info'] = {
                            'doc_title': latest_related['doc_title'],
                            'chunk_id': latest_related['chunk_id'],
                            'dieu_number': latest_related['dieu_number'],
                            'doc_issue_date': latest_related['doc_issue_date']
                        }
            
            except Exception as e:
                print(f"Error retrieving related chunks for chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
def rag_gemini_response(query, retrieval_results, model_type = "gemini-2.0-flash"):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_type)
    
    context = ""
    for chunk in retrieval_results["final_chunks"]:
        context += f"""
        - {chunk['doc_title']}, Nội dung: {chunk['content']}"""
    
    prompt = f"""Là một trợ lý pháp lý, hãy cung cấp câu trả lời toàn diện cho câu hỏi pháp lý sau dựa CHỈ trên các ngữ cảnh pháp lý được cung cấp.

            CÂU HỎI PHÁP LÝ:
            {query}

            CÁC NGỮ CẢNH PHÁP LÝ LIÊN QUAN:
            {context}

            HƯỚNG DẪN:
            1. Trả lời CHỈ dựa trên các ngữ cảnh pháp lý được cung cấp.
            2. Nếu các ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi một cách đầy đủ, hãy thừa nhận sự hạn chế này trong câu trả lời của bạn.
            3. Trích dẫn cụ thể các luật, điều khoản và quy định từ các ngữ cảnh khi có thể.
            4. Cấu trúc câu trả lời rõ ràng với các điểm pháp lý liên quan.
            5. Không tự ý tạo ra hoặc suy diễn thông tin pháp lý không được hỗ trợ bởi các ngữ cảnh được cung cấp.

            CÂU TRẢ LỜI:"""
    # Generate answer with Gemini
    response = model.generate_content(prompt)
    answer = response.text.strip()
    
    return answer

# Your OpenRouter API Key

# The OpenRouter API endpoint for chat completions

def rag_openrouter_response(query, retrieval_results, model_type="openai/gpt-4o-mini", max_tokens=2000):
    """
    Generate a legal response using OpenRouter GPT based on query and retrieval results.
    
    Args:
        query (str): The legal question to answer
        retrieval_results (dict): Dictionary containing retrieval results with 'final_chunks'
        model_type (str): The OpenRouter model to use (default: "openai/gpt-4o-mini")
        max_tokens (int): Maximum tokens for the response
    
    Returns:
        str: The generated answer or error message
    """
    
    # Build context from retrieval results
    context = ""
    for chunk in retrieval_results["final_chunks"]:
        context += f"""
        Nguồn: {chunk['doc_title']}, Điều {chunk['dieu_number']}, Nội dung: {chunk['content']}"""
    
    # Create the prompt
    prompt = f"""Là một trợ lý pháp lý, hãy cung cấp câu trả lời toàn diện cho câu hỏi pháp lý sau dựa CHỈ trên các ngữ cảnh pháp lý được cung cấp.

            CÂU HỎI PHÁP LÝ:
            {query}

            CÁC NGỮ CẢNH PHÁP LÝ LIÊN QUAN:
            {context}

            HƯỚNG DẪN:
            1. Trả lời CHỈ dựa trên các ngữ cảnh pháp lý được cung cấp.
            2. Nếu các ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi một cách đầy đủ, hãy thừa nhận sự hạn chế này trong câu trả lời của bạn.
            3. Trích dẫn cụ thể các luật, điều khoản và quy định từ các ngữ cảnh khi có thể.
            4. Cấu trúc câu trả lời rõ ràng với các điểm pháp lý liên quan.
            5. Không tự ý tạo ra hoặc suy diễn thông tin pháp lý không được hỗ trợ bởi các ngữ cảnh được cung cấp.

            CÂU TRẢ LỜI:"""
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Prepare payload
    payload = {
        "model": model_type,
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý pháp lý chuyên nghiệp, chỉ trả lời dựa trên các tài liệu pháp lý được cung cấp."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,  # Lower temperature for more consistent legal responses
    }
    
    try:
        # Make the API request
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload))
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse the response
        api_response = response.json()
        
        # Extract the assistant's message
        if "choices" in api_response and len(api_response["choices"]) > 0:
            answer = api_response["choices"][0]["message"]["content"].strip()
            return answer
        else:
            return "Lỗi: Không thể lấy phản hồi từ mô hình AI."
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if hasattr(response, 'content'):
            print(f"Response content: {response.content.decode()}")
        return f"Lỗi HTTP: {http_err}"
        
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return f"Lỗi kết nối: {req_err}"
        
    except KeyError as key_err:
        print(f"KeyError: {key_err}")
        print(f"API Response structure: {api_response if 'api_response' in locals() else 'No response received'}")
        return "Lỗi: Cấu trúc phản hồi từ API không như mong đợi."
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"Lỗi không xác định: {e}"



# Example usage
if __name__ == "__main__":
    # Initialize the retrieval flow with Elasticsearch connection
    embedding_model_list = [
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    ]
    choose_model = embedding_model_list[3]
    retrieval_flow = RAGLawRetrieval(
        es_host='localhost',
        es_port=9200,
        es_index=f'chunks_{choose_model.replace("/", "_").lower()}',
        embedding_model = choose_model,
        query_process_model="gemini-1.5-flash"
    )
    
    # Process a sample query
    query =  "Tai nạn giao thông"
    begin_time = time.time()
    results = retrieval_flow.process_query(query, top_k_categories=2, top_bm25=100, top_k_chunks=10)
    
    # Print results
    print("Your query:", query)
    
    print("\nStep 2: BM25 Candidates")
    print(f"  Found {len(results['step2_bm25_candidates'])} candidates")
    
    print("\nStep 3: Vector Reranked Chunks")
    for i, chunk in enumerate(results["step3_reranked_chunks"]):
        vector_score = chunk.get('vector_similarity_score', 'N/A')
        bm25_score = chunk.get('bm25_score', 'N/A')
        print(f"  {i+1}.chunk_id {chunk['chunk_id']}")
        print(f"     Vector Score: {vector_score}, BM25 Score: {bm25_score}")
        print(f"     Content: {chunk['content'][:100]}...")
    
    print("\nFinal Set of Chunks")
    for i, chunk in enumerate(results["final_chunks"]):
        print(f"  {i+1}.chunk_id {chunk['chunk_id']}")
        print(f"     Content: {chunk['content'][:100]}...")
    
    print("Total time:", time.time() - begin_time)

# Utility function to get all unique categories from Elasticsearch
def get_unique_categories(es_host='localhost', es_port=9200, es_index='legal_documents'):
    """
    Get all unique categories from the Elasticsearch index
    
    Parameters:
    -----------
    es_host : str
        Elasticsearch host
    es_port : int
        Elasticsearch port
    es_index : str
        Elasticsearch index name
        
    Returns:
    --------
    list
        List of unique categories
    """
    es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': 'http'}])
    
    # Aggregation query to get unique categories
    agg_query = {
        "size": 0,
        "aggs": {
            "unique_categories": {
                "terms": {
                    "field": "meta_data.category",
                    "size": 100
                }
            }
        }
    }
    
    # Execute query
    response = es.search(index=es_index, body=agg_query)
    
    # Extract category names
    categories = [bucket['key'] for bucket in response['aggregations']['unique_categories']['buckets']]
    
    return categories