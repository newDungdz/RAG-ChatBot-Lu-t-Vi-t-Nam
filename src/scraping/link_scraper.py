from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from bs4 import BeautifulSoup
import json, random, time
from tqdm import tqdm


options = Options()
options.add_argument("--headless=new")  # new headless mode
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service("chromedriver.exe"), options=options)

# Stealth mode
stealth(driver,
    languages=["en-US", "en"],
    vendor="Google Inc.",
    platform="Win32",
    webgl_vendor="Intel Inc.",
    renderer="Intel Iris OpenGL Engine",
    fix_hairline=True,
)

# Luật DocTypeID = 10, Bộ luật DocTypeID = 58, Nghị định DocTypeID = 11

base_url = "https://luatvietnam.vn/van-ban-luat-viet-nam.html"
doc_base_url = "https://luatvietnam.vn"
def scrape_page(page_number):
    url = f"{base_url}?OrderBy=0&keywords=&lFieldId=&EffectStatusId=0&DocTypeId=11&OrganId=0&page={page_number}&pSize=20&ShowSapo=0"
    driver.get(url)
    time.sleep(3)  # Wait for JavaScript to load content

    soup = BeautifulSoup(driver.page_source, "html.parser")

    doc_data = soup.find_all(["h2", "h3"], class_="doc-title")
    
    metadata_doc = soup.select("div.post-meta-doc")
    # print(doc_data)
    data = []
    for doc, meta in zip(doc_data, metadata_doc):
        a_tag = doc.find("a")
        issue_date = meta.select_one("div.doc-dmy span.w-doc-dmy2")
        data.append({
            "title": doc.get_text(strip=True),
            "link": doc_base_url + a_tag["href"],
            "issue_date": issue_date.get_text(strip=True) if issue_date else None
        })
    return data
    
    # for t, lk in zip(titles, download_links):
    #     print(f"Title: {t}\n Link: {lk}\n")

# Crawl multiple pages and assign unique IDs
all_documents = []
doc_id = 1
total_pages = 240  # Set this to the actual number of pages you want to scrape
total_docs = 0

# tqdm loop
with tqdm(range(1, total_pages + 1), desc="Crawling pages", unit="page") as pbar:
    for page in pbar:
        page_data = scrape_page(page)

        for doc in page_data:
            doc["id"] = doc_id
            doc_id += 1
            all_documents.append(doc)

        total_docs += len(page_data)
        pbar.set_postfix_str(f"Total docs: {total_docs}")

        # time.sleep(random.uniform(2, 5))

store_json  = "nghi_dinh_links.json"
# Save to JSON file
try:
    # Load existing data if the file exists
    with open(store_json, "w+", encoding="utf-8") as existing_file:
        # existing_data = json.load(existing_file)
        # if isinstance(existing_data, list):
        #     all_documents = existing_data + all_documents
        json.dump(all_documents, existing_file, ensure_ascii=False, indent=2)
        print(f"Scraping complete. Saved to {store_json}")
except FileNotFoundError:
    # If the file doesn't exist, proceed with the new data
    print("File not exist")

driver.quit()
