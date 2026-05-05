import pandas as pd
import json

def generate_df_from_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        pre_id = 0
        json_data = json.loads(json_file.read())
        json_data = [data for data in json_data if data != None]
        return pd.DataFrame(json_data)

def get_link_name(link):
    unique_part = link.split("/")[-1]
    return unique_part.split("#")[0].strip()

document_df = generate_df_from_json_file("luat_links.json")
qna_df = generate_df_from_json_file("QnA_links.json")
qna_info_df = generate_df_from_json_file("QnA_info.json")


count = 0
document_df["link_name"] = document_df["link"].apply(get_link_name)
unavailable_doc = set()
total_doc = set()
doc_type = set()

document_data = []

for related_links in qna_info_df['related_links']:
    count += 1
    for link in related_links:
        if(not link['url'].startswith("https") and "luatvietnam.vn" not in link['url'] ): continue
        category = link['url'].split("/")[3].strip()
        doc_type.add(category)
        link_name = get_link_name(link['url'])
        if( 'd1' in link_name and link['url'] not in total_doc):
            document_data.append({
                "title": link['tag'],
                "link" : link['url']
            })
        total_doc.add(link['url'])

with open("luat_khac.json", "w+", encoding="utf-8") as existing_file:
    json.dump(document_data, existing_file, ensure_ascii=False, indent=2)
    print("Scraping complete. Saved to document_links.json.")
# print("Total:", len(total_doc))
# print("Document don't in database:", len(unavailable_doc))
# print(sort(list(doc_type)))
# for doc in unavailable_doc:
#     print(doc)
