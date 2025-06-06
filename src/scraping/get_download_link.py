import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor
import time

# Cấu hình
JSON_INPUT_FILE = "bo_luat_links.json"
JSON_OUTPUT_FILE = "bo_luat_links_down.json"
HOME_URL = "https://luatvietnam.vn"
LOGIN_URL = "https://luatvietnam.vn/Account/DoLogin"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "application/x-www-form-urlencoded",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://luatvietnam.vn/"
}

# Thông tin đăng nhập
LOGIN_DATA = {
    "CustomerName": "giangson309@gmail.com",
    "CustomerPass": "khoai123",  
    "RememberMe": "true"
}

# Link lỗi cụ thể
ERROR_LINK = "https://cms.luatvietnam.vn/uploaded/Others/2023/08/29/Profile_LuatVietnam_8.8.2023_2908150600.pdf"
NO_LINK_FOUND = "Khong thay link"

def login(session):
    try:
        res = session.get("https://luatvietnam.vn/Account/Login", headers=HEADERS, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        token = soup.find("input", {"name": "__RequestVerificationToken"})["value"] if soup.find("input", {"name": "__RequestVerificationToken"}) else ""
        if not token:
            return False
        
        LOGIN_DATA["__RequestVerificationToken"] = token
        res = session.post(LOGIN_URL, headers=HEADERS, data=LOGIN_DATA, timeout=5)
        if res.status_code != 200:
            return False
        
        try:
            response_json = res.json()
            if response_json.get("Completed", False) and response_json.get("Data", "") == "LoginSuccess":
                test_res = session.get(HOME_URL, headers=HEADERS, timeout=5)
                return test_res.status_code == 200
            return False
        except ValueError:
            return False
    except RequestException:
        return False

def get_download_link(session, link, article_id, retries=3):
    for attempt in range(retries + 1):
        try:
            headers = HEADERS.copy()
            headers["Referer"] = link
            res = session.get(link, headers=headers, timeout=5)
            if res.status_code != 200:
                return "Khong thay link"
            soup = BeautifulSoup(res.text, "html.parser")

            word_link = None
            zip_link = None
            pdf_link = None
            
            div_download = soup.find("div", class_="divrow3")
            if div_download:
                for a in div_download.find_all("a", href=True):
                    href = urljoin(HOME_URL, a['href'])
                    if href.endswith((".doc", ".docx")):
                        word_link = href
                        break
                    elif href.endswith(".zip"):
                        zip_link = href
                    elif href.endswith(".pdf") and href != ERROR_LINK:
                        pdf_link = href
            
            if not word_link:
                for a in soup.find_all("a", href=True):
                    href = urljoin(HOME_URL, a['href'])
                    if href.endswith((".doc", ".docx")):
                        word_link = href
                        break
                    elif href.endswith(".zip"):
                        zip_link = href
                    elif href.endswith(".pdf") and href != ERROR_LINK:
                        pdf_link = href
            
            if not word_link:
                for a in soup.find_all("a", string=lambda text: text and ("tải về" in text.lower() or "download" in text.lower())):
                    href = urljoin(HOME_URL, a['href'])
                    if href.endswith((".doc", ".docx")):
                        word_link = href
                        break
                    elif href.endswith(".zip"):
                        zip_link = href
                    elif href.endswith(".pdf") and href != ERROR_LINK:
                        pdf_link = href
            
            if not word_link:
                for div in soup.find_all("div", class_="download"):
                    for a in div.find_all("a", href=True):
                        href = urljoin(HOME_URL, a['href'])
                        if href.endswith((".doc", ".docx")):
                            word_link = href
                            break
                        elif href.endswith(".zip"):
                            zip_link = href
                        elif href.endswith(".pdf") and href != ERROR_LINK:
                            pdf_link = href
            
            return word_link or zip_link or pdf_link or "Khong thay link"
        except RequestException:
            if attempt < retries:
                time.sleep(1)
                continue
            return "Khong thay link"

def process_article(article, session):
    result = article.copy()
    article_id = article.get("id")
    link = article.get("link")
    
    time.sleep(0.1)
    new_download_link = get_download_link(session, link, article_id) if link else "Khong thay link"
    result["download_link"] = new_download_link
    
    ordered_result = {}
    for key in ["title", "link", "issue_date", "id", "download_link", "summary"]:
        if key in result:
            ordered_result[key] = result[key]
    
    return ordered_result

def main():
    try:
        with open(JSON_INPUT_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except Exception:
        return
    
    session = requests.Session()
    if not login(session):
        return
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_article, article, session) for article in articles]
        for future in futures:
            results.append(future.result())
    
    with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(results)} link vào file {JSON_OUTPUT_FILE}")

if __name__ == "__main__":
    main()