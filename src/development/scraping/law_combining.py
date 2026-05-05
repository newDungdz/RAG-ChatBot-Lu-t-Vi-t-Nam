import json

def generate_df_from_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        pre_id = 0
        json_data = json.loads(json_file.read())
        json_data = [data for data in json_data if data != None]
        return json_data

def get_link_name(link):
    unique_part = link.split("/")[-1]
    return unique_part.split("#")[0].strip()

doc_set = set()

luat_data = generate_df_from_json_file("luat_links.json")
bo_luat_data = generate_df_from_json_file("bo_luat_links.json")
other_data = generate_df_from_json_file("luat_khac.json")

doc_set.update([get_link_name(data['link']) for data in luat_data])
doc_set.update([get_link_name(data['link']) for data in bo_luat_data])


other_data = [data for data in other_data if get_link_name(data['link']) not in doc_set]
all_law = luat_data + bo_luat_data + other_data
for i in range(len(all_law)):
     all_law[i]['id'] = i+1
# print(all_law)
with open("all_law.json", "w+", encoding="utf-8") as existing_file:
        json.dump(all_law, existing_file, ensure_ascii=False, indent=2)
        print("Combine law data to all_law.json")