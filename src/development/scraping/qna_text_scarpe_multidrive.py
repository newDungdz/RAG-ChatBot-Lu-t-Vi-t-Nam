import json
import asyncio
import sys
from playwright.async_api import async_playwright, Browser
from tqdm.asyncio import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE   = "QnA_links.json"
OUTPUT_FILE  = "QnA_data.json"
MAX_CRAWL    = 2000
NUM_WORKERS  = 10     # ← truly concurrent pages (no lock needed)
TIMEOUT      = 30000  # ms per page
# ────────────────────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

JS_EXTRACT = """
(container) => {
    const blockTags = ['P','DIV','LI','H1','H2','H3','H4','H5','H6','BR'];
    const skipTags = ['SCRIPT','STYLE','NOSCRIPT','IFRAME'];
    const skipClasses = ['advertisement','ads','ad-slot','gpt-ad'];

    function getCleanText(node) {
        let result = '';
        for (const child of node.childNodes) {
            if (child.nodeType === Node.ELEMENT_NODE) {
                if (skipTags.includes(child.tagName)) continue;
                if (skipClasses.some(c => child.className && child.className.includes(c))) continue;
                if (child.id && child.id.includes('div-gpt-ad')) continue;
            }

            if (child.nodeType === Node.TEXT_NODE) {
                result += child.textContent;
            }

            else if (child.tagName === 'STRONG' || child.tagName === 'B') {
                result += '<strong>' + getCleanText(child).trim() + '</strong>';
            }

            else if (child.tagName === 'EM' || child.tagName === 'I') {
                result += '<em>' + getCleanText(child).trim() + '</em>';
            }

            else if (child.tagName === 'U') {
                result += '<u>' + getCleanText(child).trim() + '</u>';
            }

            else if (child.tagName === 'BLOCKQUOTE') {
                result += '\\n[QUOTE]\\n' + getCleanText(child).trim() + '\\n[/QUOTE]\\n';
            }

            else {
                const inner = getCleanText(child);
                if (blockTags.includes(child.tagName)) {
                    result += inner + '\\n';
                } else {
                    result += inner;
                }
            }
        }
        return result;
    }

    return getCleanText(container);
}
"""

async def main():
    max_crawl   = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_CRAWL
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_WORKERS

    with open(INPUT_FILE, encoding="utf-8") as f:
        items = json.load(f)

    to_crawl = items[:max_crawl]
    print(f"Crawling {len(to_crawl)} items with {num_workers} concurrent pages (1 browser)\n")

    # Semaphore caps how many pages are open at once
    sem = asyncio.Semaphore(num_workers)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        # Single shared context so all pages share cookies/cache
        context = await browser.new_context(user_agent=USER_AGENT)

        # Override new_page to use the shared context
        async def crawl_with_context(item):
            async with sem:
                page = await context.new_page()
                try:
                    await page.goto(item["link"], wait_until="domcontentloaded", timeout=TIMEOUT)

                    doc_type = await page.locator(
                        "xpath=/html/body/main/div[4]/div/div[2]/div/article/div[1]/div[1]/a"
                    ).inner_text(timeout=TIMEOUT)
                    doc_type = doc_type.replace(" ", "-").lower()

                    paragraphs = await page.locator(
                        "xpath=/html/body/main/div[4]/div/div[2]/div/article/div[2]/div/p"
                    ).all_inner_texts()
                    full_question = "\n".join(paragraphs)

                    full_question = (
                        full_question
                        .replace("Xin hỏi LuatVietnam:", "")
                        .replace("Xin cảm ơn!", "")
                        .strip()
                    )

                    article_element = page.locator(
                        "xpath=/html/body/main/div[4]/div/div[2]/div/article/div[4]/div[1]"
                    )
                    article_content = await article_element.evaluate(JS_EXTRACT)
                    article_content = (
                        article_content.split("<strong><em>Xem thêm:</em></strong>")[0]
                                       .replace("<strong>Trả lời:</strong>", "")
                                       .strip()
                    )

                    return {
                        **item,
                        "doc_type":        doc_type,
                        "full_question":   full_question,
                        "article_content": article_content,
                    }

                except Exception as e:
                    return {**item, "crawl_status": "error", "crawl_error": str(e)}

                finally:
                    await page.close()

        tasks = [crawl_with_context(item) for item in to_crawl]

        # tqdm.gather gives a live progress bar over async tasks
        results = await tqdm.gather(*tasks, desc="Crawling", unit="page")

        await context.close()
        await browser.close()

    results = sorted(results, key=lambda r: r["id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok     = sum(1 for r in results if "crawl_status" not in r)
    errors = sum(1 for r in results if r.get("crawl_status") == "error")
    print(f"\nDone — ok: {ok}  errors: {errors}")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())