from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from bs4 import BeautifulSoup
import json, random, time
from datetime import datetime, timedelta
import re




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

# https://luatvietnam.vn/van-ban/tim-luat-su-tu-van.html?keywords=&RowAmount=20&PageIndex=2
base_url = "https://luatvietnam.vn/van-ban/tim-luat-su-tu-van.html"
doc_base_url = "https://luatvietnam.vn"

def scrape_page(page_number):
    def parse_relative_date(text):
        today = datetime.today()

        if "ngày trước" in text:
            days = int(re.search(r"(\d+)", text).group(1))
            return today - timedelta(days=days)

        # elif "tuần trước" in text:
        #     weeks = int(re.search(r"(\d+)", text).group(1))
        #     return today - timedelta(weeks=weeks)

        # elif "tháng trước" in text:
        #     months = int(re.search(r"(\d+)", text).group(1))
        #     return today - timedelta(days=30 * months)

        else:
            # Try parsing normal format dd/mm/yyyy
            try:
                return datetime.strptime(text, "%d/%m/%Y")
            except ValueError:
                return None

    url = f"{base_url}?keywords=&RowAmount=20&PageIndex={page_number}"
    driver.get(url)
    time.sleep(3)  # Wait for JavaScript to load content

    soup = BeautifulSoup(driver.page_source, "html.parser")

    articles_data = soup.find_all("h3", class_="article-title")
    dates = soup.find_all("p", class_= "article-meta")
        
    today = datetime.today()
    
    data = []
    for article, meta in zip(articles_data, dates):
        a_tag = article.find("a")
        create_date =  meta.get_text(strip=True)
        if "ngày trước" in create_date:
            days = int(re.search(r"(\d+)", create_date).group(1))
            create_date =  today - timedelta(days=days)
        if a_tag:
            title = a_tag.get_text(strip=True)
            link = a_tag["href"]
            data.append({
                "title": title,
                "link": link,
                "create_date": str(create_date).split(" ")[0]
            })

    return data

# Crawl multiple pages and assign unique IDs
all_qna = []
doc_id = 1
total_pages =  88 # Set this to the total number of pages you want to scrape

for page in range(1, total_pages + 1):
    print(f"Scraping page {page}...")
    page_data = scrape_page(page)
    print(f"Got {len(page_data)} QnA from page {page}")
    for doc in page_data:
        doc["id"] = doc_id
        doc_id += 1
        all_qna.append(doc)
    # Randomize delay between page requests
    time.sleep(random.uniform(2, 5))

# Save to JSON file
with open("QnA_links.json", "w", encoding="utf-8") as f:
    f.write("")
    json.dump(all_qna, f, ensure_ascii=False, indent=2)

driver.quit()
print("Scraping complete. Saved to QnA_links.json.")
