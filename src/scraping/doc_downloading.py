import os
import requests
from pathlib import Path
import urllib.parse
import re, json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Document type
doc_type = "luat"

# Load document links
with open(f"data\\json_data\\download_link\\{doc_type}_download.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Output folder
output_dir = Path(f"data\\docs\\raw_doc_file\\{doc_type}")
output_dir.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Referer": "https://luatvietnam.vn/",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9,vi;q=0.8",
    "referer": "https://luatvietnam.vn/",
    "priority": "u=0, i",
    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-site",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}

cookies = {
    "__uidac": "7728fe5cbc7c615c0c07fcec40122091",
    "_fbp": "fb.1.1731986981388.553086552651665882",
    "_hjSessionUser_5192214": "eyJpZCI6ImNlOTE3NTA4LWYxMDctNTY0OC1hOGI3LTk1YmI0OGM3MmYyOCIsImNyZWF0ZWQiOjE3MzE5ODY5ODIxMjYsImV4aXN0aW5nIjp0cnVlfQ==",
    "_uidcms": "1977585241815907887",
    "_cc_id": "95717037030acb8a4e21f8f543bb5299",
    "_ga_DZNHRSYZR1": "GS1.1.1743416506.2.1.1743416549.0.0.0",
    "LAWSVN_AUTH": "EA0425FE7243F5C8779DF6DD4873F7B5916B89769D2914B3C9CAA8E2C1AA637EDE88F3E8901FA3C6C155066CBC2FDF5C92061DAB7EB76D3D84416281FDA166DB8340C33CB64A8FAF37592B790113B6CCF5C821BAB6E8B4BB00A0331B14DA06E71915F136C9C6EE99346E26DFC5525C5CDD740F8B6F5AF81374973B43EE377A34514DA45E9142E32631217047841E24BFD4392C76645EB52B9A00D92FFA845AD44EDC46694E5244A369DCC5C9DD2AF1C39FC40E5B31D36511F95A1C6BE4C3DAA52D04A1EAA103D135C3E3C9CD5384563BC3E3E51647850DB595A4D4731A778C134529488FB9F2D524E7788A788930DA630F098F2600F0392C3BCEF6F7F9B76BEE908EFBC19FFF73369E7618087FF7F4A2245B7A0B1BAEDC6FD8375DFCF97BF18AAAD3C69C740FE1CBD4F73D001D8EBEBB9B551C3BCE7F415E50B8DF1DCB2282ABDE4DFBEACB12CCA2526669468E7151AF3692A5D22C3DDF6AE8FCF5CC790FFA3C82DCA66429646308D7E634F9D1CDE28060771C459661DAC3C0C177711AAAEA10DB8D9FA6F5F8014B88A381EB1E55F2DD5A4ACC31FF24240F4B2E9F012E870465591FD15EE7216A41E1926D28819D9A7830E426FAD5394049E9821C955F6CEA7350DC903D24D58E5B958D805D49CF62F12877E8C4626D1A11DB65FAE2148A604D0B5747EDB3136A1E7ADC4D568C75BEF6D45247EC0E31C4BB24536A6DB1E3EBE7C97D81E5DBEF2963B4F869C8D3F93FF85AE0CB7218A816A75485E94D7B55632306300AA9931A03798842953E515F80BEE45C14785CA0411AFE9DEB8015EDEF00E2844292103CCFFB6B7191B4F41BA280F5D358B9078F7374E4031B341742196F805C6C1D27083771CF19A3E2C3D5307E5EA92395983FA1E9C1279FD1484CB50EB522A0A1E5AE87EFFCEEBE00960486586DF12831EE912E402017DC5D79604579348CC1CAB8BF4B62D99774D833549CD48134FEB6BF5143BB92F1C59128074D79BB950D6113777933B88D6C5716DD01EB8004D0BD37E5187F97C957E82A6DB74E14BECDBAF85876E5239000DCDF51819350CF9E61C17CC09C8A25C063F77D3AB72E8946437C8B8329513A6EEF482A4BDE1A06857A10EBE7A86BE57FE9F5E617A87FB49F084F87D1806B9FF522F35BDFB02B363DC62DE6263710C9EEBB1D16F1CE1C918C5C23A85CB1585858D59BE1607FA2F91074A9291B526A011FC12FB3ED9499B12C234C7720893687C2F1718E28DC4AE2C1448A9020A685CE66AD9D265F6AE82F11D43A44BE26BEA72C04407F25FBE13CBC8F4005302D07C0D82F068A753FE993AA8DBFF354C433A632BCC06F4D6B5820DF14F14E994227D3E2914484ADF86A14D4D43F6F97D4CA1EDFABF43B082B3783EB565BB487AFFA49F5901EABE4C0CD622A78233567EFC03D652B117772FFB483DD5A7504472B8D74EA695ED8D6C989F42BE9A7DD1D241FA756793423684A1BD00720B9D6F7A53545FF9FD3ECFD4570BA0E8C6B3D81CC0EFEA38104927580B1470285FEF5AB295EF2E68A598071425527BD9E83D75274054F6697E8B9B4989F03080C104D206C704CA6BD904C3479B026AFA560E3CD9D9B2B5F6709A6E5673C6EEE60AFA21F1F4DA631D0FA5CE4C725185196721FD696B1A8FE0FF3C640AA4DDC317562171BC17E91BB66DF635D0B9F22DCEF1642425991690492BAA4BAECB6E92E4949305AFFA2E3A300F488A74A353D019A8F1E6797166497D4495804C3E49A8AF73CD5C74C3D02133470EFA347425EB7AFA42E69FF319D1549E7D8E4BE57407CE472A1701ADEF405C12CC32518409C1EB8CA030D3FFB3470EF1DC2A3FA95D50C37CE5BE636D6588065A23FBB05C03ECABC59D20526004F68BCC615BB600EBA7CDAEB30310D6F6BBC86D0890E301537B1E12A70E5C83C339EA12DB01BEC379808E2CCE0D7D599B98A6D9A801356EC0F2C85",
    "_gcl_au": "1.1.581885547.1746596432",
    "_ga_J8TZJ65FPH": "GS2.1.s1746596432$o7$g0$t1746596432$j60$l0$h0",
    "dable_uid": "39443778.1746596434875",
    "_ga_RT540960JS": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
    "_ga_EYJSHRXPKN": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
    "_ga_DRR9FLY33Q": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
    "_ga_EHF2XGTXWS": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
    "_ga_J8FZDKLOH3": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
    "_ga_E2FE4ZMF23": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
    "_ga_CZGZYX6FQ1": "GS2.1.s1746596430$o6$g1$t1746597066$j0$l0$h0",
}

def extract_filename_from_header(header_value):
    if not header_value:
        return None
    match = re.findall(r'filename(?:\*)?[^;=\n]*=(\"?)([^";\n]+)\1', header_value, re.IGNORECASE)
    if match:
        raw_filename = match[0][1]
        fixed_filename = raw_filename.replace('uáº­t', 'uật').replace('á»', 'ộ').replace("Ä", "Đ")
        fixed_filename = fixed_filename.replace('Nghá» Äá»nh', 'Nghị Định')
        return fixed_filename
    return None

new_data = []

# Download function for multiprocessing
def download_document(doc):
    url = doc["download_link"]
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        header = response.headers.get("Content-Disposition")
        filename = extract_filename_from_header(header) or os.path.basename(url)
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        new_data.append({
            "title": doc["title"],
            "doc_name": filename.replace(".zip", ".docx"),
            "link": doc["link"],
            "issue_date": doc["issue_date"],
            "category": doc["link"].split("/")[3],
            # "summary": doc["summary"]
        })
    except requests.RequestException as e:
        print(f"❌ Download failed: {e}")

# Download files with multiprocessing
with ThreadPoolExecutor(max_workers=8) as executor:
    with tqdm(total=len(documents), desc="Downloading files", unit="file") as pbar:
        futures = [executor.submit(download_document, doc) for doc in documents]
        for future in futures:
            future.result()
            pbar.update(1)

# Save the new data to a JSON file
for i, data in enumerate(new_data):
    data["id"] = i
with open(f"data\\json_data\\doc_metadata\\normal_law\\{doc_type}_data.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
print("Data saved")
