
from sentence_transformers import SentenceTransformer
import json, os
import google.generativeai as genai
import logging
from typing import Dict
import time



law_category_contexts = [
    "[linh-vuc-khac] -> [Lĩnh vực khác] : Các quy định pháp luật không thuộc các lĩnh vực đã được phân loại cụ thể, hoặc các vấn đề pháp lý mới phát sinh chưa được xếp vào ngành luật nào.",
    "[chinh-sach] -> [Chính sách] : Các định hướng, giải pháp của Nhà nước để giải quyết vấn đề của thực tiễn nhằm đạt được mục tiêu nhất định.",
    "[thue] -> [Thuế] : Tổng hợp các quy phạm pháp luật điều chỉnh các quan hệ xã hội phát sinh trong quá trình thu, nộp thuế giữa cơ quan nhà nước có thẩm quyền và người nộp thuế.",
    "[tai-chinh] -> [Tài chính] : Tổng hợp các quy phạm pháp luật điều chỉnh các quan hệ xã hội phát sinh trong quá trình hình thành, phân phối và sử dụng các quỹ tiền tệ của nhà nước.",
    "[an-ninh-quoc-gia] -> [An ninh quốc gia] : Các quy định về bảo vệ chế độ xã hội chủ nghĩa, Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam, bảo vệ độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ của Tổ quốc.",
    "[hanh-chinh] -> [Hành chính] : Các quy định về hoạt động quản lý nhà nước trên các lĩnh vực của đời sống xã hội.",
    "[tu-phap] -> [Tư pháp] : Các quy định về tổ chức và hoạt động của các cơ quan tư pháp (tòa án, viện kiểm sát, cơ quan thi hành án) và các quan hệ pháp luật phát sinh trong hoạt động tư pháp.",
    "[doanh-nghiep] -> [Doanh nghiệp] : Các quy định về việc thành lập, tổ chức quản lý, hoạt động, giải thể và các hoạt động có liên quan của doanh nghiệp.",
    "[dat-dai] -> [Đất đai] : Các quy định về chế độ sở hữu, quản lý và sử dụng đất đai.",
    "[y-te] -> [Y tế] : Các quy định về chăm sóc sức khỏe nhân dân, phòng chống dịch bệnh, khám chữa bệnh, dược phẩm, và các dịch vụ y tế khác.",
    "[an-ninh-trat-tu] -> [An ninh trật tự] : Các quy định về bảo vệ trật tự công cộng, an toàn xã hội, phòng chống tội phạm và các vi phạm pháp luật khác.",
    "[dau-tu] -> [Đầu tư] : Các quy định về hoạt động đầu tư kinh doanh tại Việt Nam và từ Việt Nam ra nước ngoài.",
    "[co-cau-to-chuc] -> [Cơ cấu tổ chức] : Các quy định về mô hình, cấu trúc, chức năng, nhiệm vụ, quyền hạn của các cơ quan, tổ chức trong hệ thống chính trị.",
    "[tai-nguyen] -> [Tài nguyên] : Các quy định về quản lý, bảo vệ, khai thác và sử dụng tài nguyên thiên nhiên (đất, nước, khoáng sản, rừng, nguồn lợi thủy sản...).",
    "[giao-thong] -> [Giao thông] : Các quy định về trật tự, an toàn giao thông đường bộ, đường sắt, đường thủy, hàng không.",
    "[giao-duc] -> [Giáo dục] : Các quy định về hệ thống giáo dục quốc dân, nhà trường, giáo viên, người học và quản lý nhà nước về giáo dục.",
    "[lao-dong] -> [Lao động] : Các quy định về quan hệ lao động giữa người lao động và người sử dụng lao động, an toàn lao động, vệ sinh lao động, bảo hiểm xã hội.",
    "[thong-tin] -> [Thông tin] : Các quy định về hoạt động báo chí, xuất bản, bưu chính, viễn thông, công nghệ thông tin và truyền thông.",
    "[xay-dung] -> [Xây dựng] : Các quy định về hoạt động quy hoạch, thiết kế, thi công, giám sát, quản lý dự án đầu tư xây dựng.",
    "[van-hoa] -> [Văn hóa] : Các quy định về bảo tồn và phát huy di sản văn hóa, nghệ thuật biểu diễn, điện ảnh, mỹ thuật, nhiếp ảnh, quảng cáo.",
    "[cong-nghiep] -> [Công nghiệp] : Các quy định về phát triển các ngành công nghiệp, quản lý sản xuất công nghiệp, sản phẩm công nghiệp.",
    "[ngoai-giao] -> [Ngoại giao] : Các quy định về quan hệ ngoại giao, lãnh sự giữa Việt Nam và các nước, các tổ chức quốc tế.",
    "[nong-nghiep] -> [Nông nghiệp] : Các quy định về phát triển nông nghiệp, lâm nghiệp, ngư nghiệp, thủy lợi và phát triển nông thôn.",
    "[thuong-mai] -> [Thương mại] : Các quy định về hoạt động mua bán hàng hóa, cung ứng dịch vụ của thương nhân.",
    "[hinh-su] -> [Hình sự] : Các quy định về tội phạm và hình phạt.",
    "[khieu-nai] -> [Khiếu nại] : Các quy định về trình tự, thủ tục giải quyết khiếu nại của công dân, cơ quan, tổ chức.",
    "[khoa-hoc] -> [Khoa học] : Các quy định về hoạt động nghiên cứu khoa học, phát triển và ứng dụng công nghệ.",
    "[quoc-phong] -> [Quốc phòng] : Các quy định về xây dựng nền quốc phòng toàn dân, bảo vệ Tổ quốc.",
    "[xuat-nhap-canh] -> [Xuất nhập cảnh] : Các quy định về việc xuất cảnh, nhập cảnh, quá cảnh, cư trú của người nước ngoài tại Việt Nam và của công dân Việt Nam ở nước ngoài.",
    "[can-bo] -> [Cán bộ] : Các quy định về cán bộ, công chức, viên chức trong các cơ quan nhà nước, tổ chức chính trị, tổ chức chính trị - xã hội.",
    "[dan-su] -> [Dân sự] : Các quy định về địa vị pháp lý, chuẩn mực pháp lý cho cách ứng xử của cá nhân, pháp nhân, chủ thể khác; quyền, nghĩa vụ về nhân thân và tài sản của các chủ thể trong các quan hệ được hình thành trên cơ sở bình đẳng, tự do ý chí, độc lập về tài sản và tự chịu trách nhiệm.",
    "[dau-thau] -> [Đấu thầu] : Các quy định về hoạt động đấu thầu để lựa chọn nhà thầu cung cấp dịch vụ tư vấn, phi tư vấn, hàng hóa, xây lắp cho các dự án, gói thầu.",
    "[ke-toan] -> [Kế toán] : Các quy định về công tác kế toán, kiểm toán trong các cơ quan, đơn vị, doanh nghiệp.",
    "[so-huu-tri-tue] -> [Sở hữu trí tuệ] : Các quy định về quyền tác giả, quyền liên quan, quyền sở hữu công nghiệp và quyền đối với giống cây trồng.",
    "[bao-hiem] -> [Bảo hiểm] : Các quy định về hoạt động kinh doanh bảo hiểm, hợp đồng bảo hiểm.",
    "[hai-quan] -> [Hải quan] : Các quy định về thủ tục hải quan, kiểm tra, giám sát hải quan đối với hàng hóa xuất khẩu, nhập khẩu, quá cảnh, phương tiện vận tải xuất cảnh, nhập cảnh, quá cảnh.",
    "[hang-hai] -> [Hàng hải] : Các quy định về hoạt động của tàu thuyền trong lãnh hải, cảng biển Việt Nam.",
    "[hon-nhan-gia-dinh] -> [Hôn nhân gia đình] : Các quy định về kết hôn, ly hôn, quyền và nghĩa vụ giữa vợ và chồng, cha mẹ và con cái.",
    "[thi-dua] -> [Thi đua] : Các quy định về tổ chức phong trào thi đua, khen thưởng đối với tập thể, cá nhân có thành tích xuất sắc.",
    "[tiet-kiem] -> [Tiết kiệm] : Các quy định về thực hành tiết kiệm, chống lãng phí trong quản lý, sử dụng ngân sách, tài sản nhà nước.",
    "[chung-khoan] -> [Chứng khoán] : Các quy định về hoạt động chào bán, niêm yết, giao dịch, kinh doanh, đầu tư chứng khoán, dịch vụ về chứng khoán và thị trường chứng khoán.",
    "[cu-tru] -> [Cư trú] : Các quy định về việc đăng ký, quản lý cư trú của công dân tại một địa điểm thuộc đơn vị hành chính cấp xã hoặc cấp huyện.",
    "[toa-an] -> [Tòa án] : Các quy định về tổ chức và hoạt động của tòa án nhân dân, về các nguyên tắc, trình tự, thủ tục tố tụng tại tòa án.",
    "[xuat-nhap-khau] -> [Xuất nhập khẩu] : Các quy định về quản lý hoạt động mua bán hàng hóa quốc tế của thương nhân."
]

law_categoies = [
    "linh-vuc-khac",
    "chinh-sach",
    "thue",
    "tai-chinh",
    "an-ninh-quoc-gia",
    "hanh-chinh",
    "tu-phap",
    "doanh-nghiep",
    "dat-dai",
    "y-te",
    "an-ninh-trat-tu",
    "dau-tu",
    "co-cau-to-chuc",
    "tai-nguyen",
    "giao-thong",
    "giao-duc",
    "lao-dong",
    "thong-tin",
    "xay-dung",
    "van-hoa",
    "cong-nghiep",
    "ngoai-giao",
    "nong-nghiep",
    "thuong-mai",
    "hinh-su",
    "khieu-nai",
    "khoa-hoc",
    "quoc-phong",
    "xuat-nhap-canh",
    "can-bo",
    "dan-su",
    "dau-thau",
    "ke-toan",
    "so-huu-tri-tue",
    "bao-hiem",
    "hai-quan",
    "hang-hai",
    "hon-nhan-gia-dinh",
    "thi-dua",
    "tiet-kiem",
    "chung-khoan",
    "cu-tru",
    "toa-an",
    "xuat-nhap-khau"
]

class RAGLawRetrieval:
    def __init__(self, es_client, es_index='chunks', 
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

        self.index_name = es_index
        self.es = es_client
        # Get legal categories from Elasticsearch
        self.legal_categories_fulltext = self._get_legal_categories()
        self.legal_categories = law_categoies

        # Load models
        print("Using Classifier:", query_process_model)
        print("Using Embeder:", embedding_model)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel(query_process_model)
        self._expand_and_classify_query = self._expand_and_classify_query_with_gemini
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
        return law_category_contexts

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
                "doc_link": meta_data.get('doc_link', ''),
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


