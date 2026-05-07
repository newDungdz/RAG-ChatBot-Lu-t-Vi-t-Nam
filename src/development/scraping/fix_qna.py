import json
import asyncio
import sys
import unicodedata
from pathlib import Path
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE   = "QnA_data.json"   # existing crawled data (may contain errors)
OUTPUT_FILE  = "QnA_data.json"   # overwrite in-place (change if you want a backup)
NUM_WORKERS  = 1
TIMEOUT      = 30_000            # ms
MAX_RETRIES  = 3                 # how many times to retry each error URL
# ────────────────────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

JS_EXTRACT_HTML = """
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
            } else if (child.tagName === 'STRONG' || child.tagName === 'B') {
                result += '<strong>' + getCleanText(child).trim() + '</strong>';
            } else if (child.tagName === 'EM' || child.tagName === 'I') {
                result += '<em>' + getCleanText(child).trim() + '</em>';
            } else if (child.tagName === 'U') {
                result += '<u>' + getCleanText(child).trim() + '</u>';
            } else if (child.tagName === 'BLOCKQUOTE') {
                result += '<blockquote>' + getCleanText(child).trim() + '</blockquote>';
            } else {
                const inner = getCleanText(child);
                if (blockTags.includes(child.tagName)) {
                    result += inner + '<br>';
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

JS_EXTRACT_MARKDOWN = """
(container) => {
    const blockTags = ['P','DIV','LI','H1','H2','H3','H4','H5','H6'];
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
            } else if (child.tagName === 'STRONG' || child.tagName === 'B') {
                result += '**' + getCleanText(child).trim() + '**';
            } else if (child.tagName === 'EM' || child.tagName === 'I') {
                result += '*' + getCleanText(child).trim() + '*';
            } else if (child.tagName === 'U') {
                result += getCleanText(child).trim();
            } else if (child.tagName === 'BLOCKQUOTE') {
                const inner = getCleanText(child).trim()
                    .split('\\n').map(line => '> ' + line).join('\\n');
                result += '\\n' + inner + '\\n';
            } else if (child.tagName && child.tagName.match(/^H[1-6]$/)) {
                const level = parseInt(child.tagName[1]);
                result += '\\n' + '#'.repeat(level) + ' ' + getCleanText(child).trim() + '\\n';
            } else if (child.tagName === 'LI') {
                result += '\\n- ' + getCleanText(child).trim();
            } else {
                const inner = getCleanText(child).trim();
                if (blockTags.includes(child.tagName)) {
                    result += '\\n' + inner + '\\n';
                } else {
                    result += inner;
                }
            }
        }
        return result;
    }
    return getCleanText(container)
        .replace(/\\n{3,}/g, '\\n\\n')
        .trim();
}
"""


def remove_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')


async def crawl_one(context, item: dict, sem: asyncio.Semaphore) -> dict:
    """Crawl a single item, returning a fixed record or an error record."""
    async with sem:
        page = await context.new_page()
        try:
            await page.goto(item["link"], wait_until="domcontentloaded", timeout=TIMEOUT)

            doc_type = await page.locator(
                "xpath=/html/body/main/div[4]/div/div[2]/div/article/div[1]/div[1]/a"
            ).inner_text(timeout=TIMEOUT)
            doc_type = remove_diacritics(doc_type).replace(" ", "-").lower()

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
            article_content_markdown = await article_element.evaluate(JS_EXTRACT_MARKDOWN)
            article_content_markdown = (
                article_content_markdown.split("***Xem thêm:***")[0]
                                        .replace("**Trả lời:**", "")
                                        .strip()
            )

            article_content_html = await article_element.evaluate(JS_EXTRACT_HTML)
            article_content_html = (
                article_content_html.split("<strong><em>Xem thêm:</em></strong>")[0]
                                    .replace("<strong>Trả lời:</strong>", "")
                                    .strip()
            )

            # Build a clean record: keep original fields, add scraped fields,
            # and explicitly remove any leftover error keys.
            result = {k: v for k, v in item.items() if k not in ("crawl_status", "crawl_error")}
            result.update({
                "doc_type":                   doc_type,
                "full_question":              full_question,
                "article_content_markdown":   article_content_markdown,
                "article_content_html":       article_content_html,
            })
            return result

        except Exception as e:
            return {**item, "crawl_status": "error", "crawl_error": str(e)}

        finally:
            await page.close()


async def main():
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_WORKERS

    # ── Load existing data ──────────────────────────────────────────────────
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: {INPUT_FILE} not found.")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        all_records: list[dict] = json.load(f)

    # ── Identify errors ─────────────────────────────────────────────────────
    error_records = [r for r in all_records if r.get("crawl_status") == "error"]
    ok_records    = [r for r in all_records if r.get("crawl_status") != "error"]

    if not error_records:
        print("No error entries found in the data. Nothing to re-crawl. ✓")
        return

    print(f"Total records : {len(all_records)}")
    print(f"OK records    : {len(ok_records)}")
    print(f"Error records : {len(error_records)}  ← will re-crawl these")
    print(f"Workers       : {num_workers}\n")

    # ── Re-crawl with retries ───────────────────────────────────────────────
    sem = asyncio.Semaphore(num_workers)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(user_agent=USER_AGENT)

        remaining = error_records[:]
        all_fixed: list[dict] = []

        for attempt in range(1, MAX_RETRIES + 1):
            if not remaining:
                break

            print(f"── Attempt {attempt}/{MAX_RETRIES}  ({len(remaining)} URLs) ──")
            tasks = [crawl_one(context, item, sem) for item in remaining]
            results = await tqdm.gather(*tasks, desc=f"  Crawling (attempt {attempt})", unit="page")

            still_failing = []
            for r in results:
                if r.get("crawl_status") == "error":
                    still_failing.append(r)
                else:
                    all_fixed.append(r)

            fixed_this_round = len(remaining) - len(still_failing)
            print(f"  Fixed: {fixed_this_round}  Still failing: {len(still_failing)}\n")
            remaining = still_failing

        # Any permanently failing items stay with error status
        all_fixed.extend(remaining)

        await context.close()
        await browser.close()

    # ── Merge & save ────────────────────────────────────────────────────────
    # Build a lookup of re-crawled results by id
    fixed_by_id = {r["id"]: r for r in all_fixed}

    # Replace error records in the original list with fixed versions
    merged = []
    for record in all_records:
        if record.get("crawl_status") == "error" and record["id"] in fixed_by_id:
            merged.append(fixed_by_id[record["id"]])
        else:
            merged.append(record)

    # Re-sort by id just in case
    merged.sort(key=lambda r: r["id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    final_errors = sum(1 for r in merged if r.get("crawl_status") == "error")
    final_ok     = len(merged) - final_errors

    print("══════════════════════════════════")
    print(f"Done! Output written to: {OUTPUT_FILE}")
    print(f"  OK      : {final_ok}")
    print(f"  Errors  : {final_errors}  (gave up after {MAX_RETRIES} attempts)")
    if final_errors:
        failed_ids = [r["id"] for r in merged if r.get("crawl_status") == "error"]
        print(f"  Failed IDs: {failed_ids}")
    print("══════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())