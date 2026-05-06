import json
import time
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "QnA_links.json"
OUTPUT_FILE = "QnA_data.json"
MAX_CRAWL   = 2
DELAY       = 0
# ────────────────────────────────────────────────────────────────────────────


def make_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {
        "userAgent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        )
    })
    return driver


def crawl_article(driver, url: str) -> dict:
    try:
        driver.get(url)
        time.sleep(DELAY)

        doc_type = driver.find_element(
            By.XPATH,
            "/html/body/main/div[4]/div/div[2]/div/article/div[1]/div[1]/a"
        ).text.replace(" ", "-").lower()

        full_question = driver.find_element(
            By.XPATH,
            "/html/body/main/div[4]/div/div[2]/div/article/div[2]/div/p"
        ).text.replace("Xin hỏi LuatVietnam:", "").replace("Xin cảm ơn!", "").strip()

        article_element = driver.find_element(
            By.XPATH,
            "/html/body/main/div[4]/div/div[2]/div/article/div[4]/div[1]"
        )

        article_content = driver.execute_script("""
            const container = arguments[0];
            const blockTags   = ['P','DIV','LI','H1','H2','H3','H4','H5','H6','BR'];
            const skipTags    = ['SCRIPT','STYLE','NOSCRIPT','IFRAME'];
            const skipClasses = ['advertisement','ads','ad-slot','gpt-ad'];

            function getCleanText(node) {
                let result = '';
                for (const child of node.childNodes) {
                    if (child.nodeType === Node.ELEMENT_NODE) {
                        if (skipTags.includes(child.tagName)) continue;
                        if (skipClasses.some(c => child.className?.includes?.(c))) continue;
                        if (child.id?.includes('div-gpt-ad')) continue;
                    }
                    if (child.nodeType === Node.TEXT_NODE) {
                        result += child.textContent;
                    } else if (child.tagName === 'BLOCKQUOTE') {
                        result += '\\n[QUOTE]\\n' + getCleanText(child).trim() + '\\n[/QUOTE]\\n';
                    } else {
                        const inner = getCleanText(child);
                        result += blockTags.includes(child.tagName) ? inner + '\\n' : inner;
                    }
                }
                return result;
            }
            return getCleanText(container);
        """, article_element)

        article_content = (
            article_content.split("Xem thêm")[0]
                           .replace("Trả lời:", "")
                           .strip()
        )

        return {
            "doc_type":        doc_type,
            "full_question":   full_question,
            "article_content": article_content,
        }

    except Exception as e:
        return {"crawl_status": "error", "crawl_error": str(e)}


def main():
    max_crawl = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_CRAWL

    with open(INPUT_FILE, encoding="utf-8") as f:
        items = json.load(f)

    to_crawl = items[:max_crawl]
    print(f"Crawling {len(to_crawl)} items sequentially (1 driver, 1 tab)\n")

    driver  = make_driver()
    results = []

    try:
        for item in tqdm(to_crawl, desc="Crawling", unit="page"):
            result = crawl_article(driver, item["link"])
            results.append({**item, **result})
    finally:
        driver.quit()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok     = sum(1 for r in results if "crawl_status" not in r)
    errors = sum(1 for r in results if r.get("crawl_status") == "error")
    print(f"\nDone — ok: {ok}  errors: {errors}")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()