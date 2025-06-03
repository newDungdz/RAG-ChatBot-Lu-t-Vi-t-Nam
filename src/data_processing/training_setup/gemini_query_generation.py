import os
import json
from tqdm import tqdm
from datetime import datetime
from google.generativeai import GenerativeModel, configure

# ==== Configurations ====
API_KEY = "AIzaSyBq4HTkU_PWUyHh7NmOuFPSjgzMQI86CCo"
TEMPERATURE = 0.7
N_SAMPLE = 100  # Total samples per category
SAVE_DIR = "generated_qna_data"
COMBINED_FILE = "combined_qna.json"

# ==== Category Mapping ====
CATEGORY_LABELS = {
    'chinh-sach': 'Chính sách',
    'thue': 'Thuế',
    'tai-chinh': 'Tài chính',
    'an-ninh-quoc-gia': 'An ninh quốc gia',
    'hanh-chinh': 'Hành chính',
    'tu-phap': 'Tư pháp',
    'doanh-nghiep': 'Doanh nghiệp',
    'dat-dai': 'Đất đai',
    'y-te': 'Y tế',
    'an-ninh-trat-tu': 'An ninh trật tự',
    'dau-tu': 'Đầu tư',
    'co-cau-to-chuc': 'Cơ cấu tổ chức',
    'tai-nguyen': 'Tài nguyên',
    'giao-thong': 'Giao thông',
    'giao-duc': 'Giáo dục',
    'lao-dong': 'Lao động',
    'thong-tin': 'Thông tin',
    'xay-dung': 'Xây dựng',
    'van-hoa': 'Văn hóa',
    'cong-nghiep': 'Công nghiệp',
    'ngoai-giao': 'Ngoại giao',
    'nong-nghiep': 'Nông nghiệp',
    'thuong-mai': 'Thương mại',
    'hinh-su': 'Hình sự',
    'khieu-nai': 'Khiếu nại',
    'khoa-hoc': 'Khoa học',
    'quoc-phong': 'Quốc phòng',
    'xuat-nhap-canh': 'Xuất nhập cảnh',
    'can-bo': 'Cán bộ',
    'dan-su': 'Dân sự',
    'dau-thau': 'Đấu thầu',
    'ke-toan': 'Kế toán',
    'so-huu-tri-tue': 'Sở hữu trí tuệ',
    'bao-hiem': 'Bảo hiểm',
    'hai-quan': 'Hải quan',
    'hang-hai': 'Hàng hải',
    'hon-nhan-gia-dinh': 'Hôn nhân gia đình',
    'thi-dua': 'Thi đua',
    'tiet-kiem': 'Tiết kiệm',
    'vi-pham-hanh-chinh': 'Vi phạm hành chính',
    'chung-khoan': 'Chứng khoán',
    'cu-tru': 'Cư trú',
    'toa-an': 'Tòa án',
    'xuat-nhap-khau': 'Xuất nhập khẩu'
}

# ==== Gemini Initialization ====
configure(api_key=API_KEY)
model = GenerativeModel("gemini-1.5-pro")

# ==== Helper Functions ====

def build_prompt(categories, n_samples):
    n_each_length = n_samples // 3
    prompt = "Hãy tạo danh sách các câu hỏi liên quan đến các lĩnh vực pháp luật sau. Mỗi lĩnh vực phải có các câu hỏi đặc trưng rõ ràng, không trùng lặp ý với các lĩnh vực khác. Câu hỏi phải thực tế, phù hợp với lĩnh vực đó.\n\n"

    prompt += f"Với mỗi lĩnh vực, hãy tạo:\n- {n_each_length} câu hỏi ngắn (dưới 15 từ)\n- {n_each_length} câu hỏi trung bình (15-30 từ)\n- {n_each_length} câu hỏi dài (trên 30 từ)\n\n"

    prompt += "Danh sách lĩnh vực:\n"
    for slug, name in categories.items():
        prompt += f"- {name}\n"

    prompt += "\nĐịnh dạng trả lời:\n[category_name]\n- câu hỏi 1\n- câu hỏi 2\n...\n\nBắt đầu nhé."
    
def generate_all_categories(categories, n_samples, temperature):
    response = model.generate_content(build_prompt(categories, n_samples), generation_config={"temperature": temperature})
    return response.text.strip() if response.text else ""

def parse_gemini_output(raw_text, categories):
    data = {slug: [] for slug in categories.keys()}
    current_category = None

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            slug = next((k for k, v in categories.items() if v == name), None)
            current_category = slug
            continue

        if current_category and line.startswith("- "):
            question = line[2:].strip()
            if question:
                data[current_category].append({"question": question, "category": current_category})

    return data

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ==== Main Process ====
os.makedirs(SAVE_DIR, exist_ok=True)
combined_data = load_json(os.path.join(SAVE_DIR, COMBINED_FILE))

# Exclude 'linh-vuc-khac'
target_categories = {k: v for k, v in CATEGORY_LABELS.items() if k != 'linh-vuc-khac'}

print(f"🔄 Generating QnA for {len(target_categories)} categories at once...")
raw_result = generate_all_categories(target_categories, N_SAMPLE, TEMPERATURE)

# Parse and Save
parsed_data = parse_gemini_output(raw_result, target_categories)

for slug, qna_list in parsed_data.items():
    if not qna_list:
        print(f"⚠️ Warning: No data for {slug}")
        continue

    checkpoint_file = os.path.join(SAVE_DIR, f"{slug}.json")
    save_json(qna_list, checkpoint_file)
    print(f"✅ Saved {len(qna_list)} questions to {checkpoint_file}")

    combined_data.extend(qna_list)
    save_json(combined_data, os.path.join(SAVE_DIR, COMBINED_FILE))

print(f"🎉 Done! Total combined samples: {len(combined_data)}")
