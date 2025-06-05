Link Google Drive:
https://drive.google.com/drive/folders/10aCBigAKkBziuXMHkBcd70D96c3vAqDM

Hướng dẫn sử dụng:

Các quy định:
- Có môi trường chạy python trên máy ( như VSCode )
- Có Docker đã được tải về.


Tạo 1 file .env ở trong src/chatbot, ghi nội dung như sau:
GOOGLE_API_KEY= < Gemini API Key của bạn >
FLASK_ENV=development
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
LOCAL_MODE=True ( False nếu dùng Elasticsearch Cloud )



Nếu sử dụng Elasticsearch Cloud:
- Vào src/chatbot/.env, chỉnh LOCAL_MODE = False
- Comment container elasticsearch ở trong docker-compose.yml
- Chạy docker-compose up --build
- Khi xong, mở link được hiện trên log của docker ( mặc định http://localhost:5000 )

Nếu dung Elasticsearch Local
- Vào src/chatbot/.env, chỉnh LOCAL_MODE = True
- Lên link Google Drive này: https://drive.google.com/drive/folders/176TyeZvesvSbiMOnK3cTo8hBISdtJfgL
- Chọn 1 file json tùy ý ( Khuyên chọn file chunks_embeddings_intfloat_multilingual-e5-small.json )
- Chỉnh trong docker dòng
 RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" 
 bằng mô hình embedding mà bạn dùng

- Chạy docker-compose up --build để mở elasticsearch
- Vào src/elasticsearch/upload_data, chỉnh JSON_FILE_PATH đúng với file vừa tải về, và chạy. ( Cần lib của elasticsearch, chưa có thì chạy pip install elasticsearch ) để đưa data lên elasticsearch
- Nếu muốn kiểm trả hãy bỏ comment container kibana trước khi chạy docker-compose, và kiểm tra đã có index nào chưa.
- Khi xong, mở link được hiện trên log của docker ( mặc định http://localhost:5000 )


