Hướng dẫn sử dụng:

Nếu sử dụng Elasticsearch Cloud:
- Vào chatbot/.env, chỉnh LOCAL_MODE = False
- Comment container elasticsearch ở trong docker-compose.yml
- Chạy docker-compose up --build

Nếu dung Elasticsearch Local
- Vào chatbot/.env, chỉnh LOCAL_MODE = True
- Lên link Google Drive này: https://drive.google.com/drive/folders/176TyeZvesvSbiMOnK3cTo8hBISdtJfgL
- Chọn 1 file json tùy ý ( Khuyên chọn file chunks_embeddings_intfloat_multilingual-e5-base.json )
- Chạy docker-compose up --build để mở elasticsearch
- Vào src/elasticsearch/upload_data, chỉnh JSON_FILE_PATH đúng với file vừa tải về, và chạy. ( Cần lib của elasticsearch, chưa có thì chạy pip install elasticsearch ) để đưa data lên elasticsearch
- Nếu muốn kiểm trả hãy bỏ comment container kibana ra, và kiểm tra đã có index nào chưa.
