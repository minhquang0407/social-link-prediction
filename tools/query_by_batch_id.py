import requests
import time
import json
import os
from tqdm import tqdm

def fetch_and_save_json_by_year(df_id, file_name, step):
    endpoint_url = "https://query.wikidata.org/sparql"
    # Quan trọng: Khi dùng POST, headers cần thêm Content-Type
    headers = {
        'User-Agent': 'WikidataBot/1.3 (Contact: your-email@example.com)',
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    output_dir = "data_output/raw/"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"raw_data_{file_name}_en.jsonl")

    # Step 300 là con số ổn định nhất cho SPARQL phức tạp
    step = step
    stop = len(df_id)
    with open(file_path, 'a', encoding='utf-8') as f:
        for start in tqdm(range(0, stop, step), desc="Fetching Wikidata"):
            end = min(start + step, stop)
            ids_batch = ' '.join([f"wd:{i}" if not i.startswith('wd:') else i for i in df_id[start:end]])

# 2. Query tối ưu
            query = f"""
            SELECT ?id ?type ?typeLabel
            WHERE {{
              VALUES ?id {{ {ids_batch} }}
              ?id wdt:P31 ?type.
              SERVICE wikibase:label {{bd:serviceParam wikibase:language "en".}}
            }}
            """

            retry = 0
            while retry < 5:
                try:
                    # Dùng POST thay vì GET để tránh lỗi URI Too Long (414)
                    response = requests.post(endpoint_url, data={'query': query}, headers=headers, timeout=300)

                    if response.status_code == 429: # Too Many Requests
                        time.sleep(20 * (retry + 1))
                        retry += 1
                        continue

                    response.raise_for_status()
                    bindings = response.json()['results']['bindings']

                    for row in bindings:
                        clean_record = {key: row[key]['value'] for key in row}
                        if 'id' in clean_record:
                            clean_record['id'] = clean_record['id'].rsplit('/', 1)[-1]
                        if 'type' in clean_record:
                            clean_record['type'] = clean_record['type'].rsplit('/', 1)[-1]
                        f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')

                    f.flush()
                    break
                except Exception as e:
                    retry += 1
                    print(f"\n⚠️ Lỗi tại batch {start}, đang thử lại lần {retry}...")
                    time.sleep(10 * retry)

            time.sleep(1) # Nghỉ 1 giây giữa các batch để tránh bị khóa IP

    print(f"\n✅ Hoàn tất! Dữ liệu tại: {file_path}")