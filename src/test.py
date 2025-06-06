
import re

def remove_duplicate_urls(text):
    """
    Remove duplicate URLs in Markdown links where the URL appears as both text and link
    Example: 
    "[https://example.com](https://example.com)" 
    becomes 
    "https://example.com"
    """
    if not text:
        return text
    
    # Pattern to match Markdown links where URL is duplicated as text and link
    # [URL](URL) -> URL
    markdown_duplicate_pattern = r'\[(https?://[^\]]+)\]\(\1\)'
    
    # Replace duplicated Markdown links with just the URL
    cleaned_text = re.sub(markdown_duplicate_pattern, r'\1', text)
    
    return cleaned_text

texts = """

* Bộ luật Hình sự 2015: [https://luatvietnam.vn/hinh-su/bo-luat-hinh-su-2015-101324-d1.html](https://luatvietnam.vn/hinh-su/bo-luat-hinh-su-2015-101324-d1.html)
* Luật Thi hành án hình sự 2019: [https://luatvietnam.vn/hinh-su/luat-thi-hanh-an-hinh-su-2019-175008-d1.html](https://luatvietnam.vn/hinh-su/luat-thi-hanh-an-hinh-su-2019-175008-d1.html)

"""
print("Before")
print(texts)
print("After")
print(remove_duplicate_urls(texts))