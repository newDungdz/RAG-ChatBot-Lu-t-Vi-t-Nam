## Giới thiệu Tổng quan hệ thống

Dự án này xây dựng một hệ thống **Retrieval-Augmented Generation (RAG)** cho văn bản pháp luật tiếng Việt lấy từ https://luatvietnam.vn/.

Mục tiêu chính là tạo ra một **lớp truy hồi ngữ nghĩa chính xác và có thể kiểm soát**, phù hợp với các văn bản pháp luật có cấu trúc phân cấp phức tạp (Chương – Điều – Khoản), nơi yêu cầu cao về tính đúng đắn và khả năng truy vết nguồn.

Hệ thống RAG gồm ba thành phần chính:

### 1. Tiền xử lý & cấu trúc dữ liệu
Văn bản pháp luật được chuyển từ dạng phân cấp sang dạng **flattened**, trong đó mỗi chunk tương ứng với một điều hoặc khoản cụ thể.  
Metadata được giữ đầy đủ để bảo toàn **ngữ cảnh pháp lý** và **khả năng truy vết**.

### 2. Truy hồi dựa trên vector (Recall)
Các chunk được mã hóa bằng mô hình embedding tiếng Việt và lưu trữ trong **Elasticsearch**.  
Giai đoạn này ưu tiên **recall cao**, nhằm đảm bảo không bỏ sót các điều khoản liên quan.

### 3. Reranking (Precision)
Một mô hình transformer reranker được áp dụng để sắp xếp lại các kết quả truy hồi, giúp cải thiện độ chính xác cho các truy vấn pháp luật có ngữ nghĩa phức tạp.


## Hướng dẫn sử dụng:

Các quy định:
- Có môi trường chạy python trên máy ( như VSCode )
- Có Docker đã được tải về.


Tạo 1 file .env ở trong src/chatbot, ghi nội dung như sau:
GOOGLE_API_KEY= < Gemini API Key của bạn >
FLASK_ENV=development
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
LOCAL_MODE=True

Với Elasticsearch Local
- Vào src/chatbot/.env, chỉnh LOCAL_MODE = True
- Lên link Google Drive này: https://drive.google.com/drive/folders/176TyeZvesvSbiMOnK3cTo8hBISdtJfgL
- Chọn 1 file json tùy ý ( Khuyên chọn file chunks_embeddings_intfloat_multilingual-e5-small.json )
- Chỉnh trong docker dòng
 RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" 
 bằng mô hình embedding mà bạn dùng

- Chạy docker-compose up --build để mở elasticsearch
- Vào src/elasticsearch/upload_data, chỉnh JSON_FILE_PATH đúng với file vừa tải về, và chạy. ( Cần lib của elasticsearch, chưa có thì chạy pip install elasticsearch ) để đưa data lên elasticsearch
- Nếu muốn truy cập, phân tích dữ liệu hãy bỏ comment container kibana trước khi chạy docker-compose để có thể truy cập vào kibana.
- Khi xong, mở link được hiện trên log của docker ( mặc định http://localhost:5000 )


Link Google Drive Project: https://drive.google.com/drive/folders/10aCBigAKkBziuXMHkBcd70D96c3vAqDM
Trong Drive có chi tiết các tài liệu của project, gồm có:
- Dataset: raw dataset và dataset đã được xử lý, chunking.
- Video cài đặt và video demo.
- Báo cáo chi tiết về project.

Link Báo cáo:
https://docs.google.com/document/d/1RaDzGh5KaLjDyYfc6lCLI4380iur3DR4d6obFOoL7lo/edit?usp=sharing
