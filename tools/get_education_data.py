import requests
import time
import json
import os
import sys
from tqdm import tqdm
def fetch_and_save_json_by_year(name, object_type, filter_key):
    endpoint_url = "https://query.wikidata.org/sparql"
    start_year = 1950
    end_year = 2026
    limit = 5000

    headers = {
        'User-Agent': 'WikidataBot/1.3 (Contact: your-email@example.com)',
        'Accept': 'application/sparql-results+json'
    }

    output_dir = "data_output/raw/json"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for year in range(start_year, end_year + 1):
        print(f"\n--- 📅 Processing Year: {year} ---")
        file_path = os.path.join(output_dir, f"raw_data_{name}{year}_{object_type}.jsonl")

        # 3 giai đoạn để tránh Timeout
        periods = [
            (f"{year}-01-01T00:00:00", f"{year}-03-01T00:00:00"),
            (f"{year}-03-01T00:00:00", f"{year}-06-01T00:00:00"),
            (f"{year}-06-01T00:00:00", f"{year}-09-01T00:00:00"),
            (f"{year}-09-01T00:00:00", f"{year+1}-01-01T00:00:00")
        ]

        total_year_records = 0

        # Mở file ở chế độ 'w' (ghi mới)
        with open(file_path, 'w', encoding='utf-8') as f:
            for start_date, end_date in periods:
                offset = 0
                retry_count = 0
                max_retries = 50
                pbar = tqdm(desc=f"Giai đoạn {start_date[:10]}", unit=" rec", leave=False)
                while True:
                    query = f"""
                    SELECT ?person ?birthYear ?object ?objectLabel
                    WHERE {{

                        ?person wdt:P31 wd:Q5;
                                wdt:P569 ?person_dob.
                        BIND(YEAR(?person_dob) AS ?birthYear)
                        FILTER (?person_dob >= "{start_date}"^^xsd:dateTime && ?person_dob < "{end_date}"^^xsd:dateTime)
                        {{   ##FILTER KEY##  }}
                        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                    }}
                    LIMIT {limit}
                    OFFSET {offset}
                    """
                    query = query.replace("##FILTER KEY##",filter_key)
                    try:
                        response = requests.get(endpoint_url, params={'query': query}, headers=headers, timeout=300)
                        response.raise_for_status()
                        data = response.json()
                        bindings = data['results']['bindings']

                        if not bindings:
                            break

                        for row in bindings:
                            # Trích xuất toàn bộ các biến có trong row
                            record = {var: row[var]['value'] for var in row}

                            # Ghi mỗi record thành 1 dòng JSON
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')

                        f.flush() # Ghi ngay xuống đĩa

                        total_year_records += len(bindings)
                        pbar.update(len(bindings))
                        print(f"   > {start_date[:10]}: Fetched {total_year_records} records...", end="\r")

                        if not bindings or len(bindings) < limit:
                            print(f"\n-------> Đã lấy hết!")
                            break

                        offset += limit
                        time.sleep(0.5)

                    except Exception as e:
                        retry_count += 1

                        if retry_count > max_retries:
                            tqdm.write(f"\n❌ Lỗi nghiêm trọng năm {year} tại offset {offset}: {e}")
                            break
                        else:
                            if retry_count % 5 == 0 and retry_count > 0:
                                sleep_time = 60 * retry_count
                            else:
                                sleep_time = 5 * retry_count

                            if limit > 1000:
                                limit -= 1000

                            tqdm.write(f"⚠️ Thử lại lần {retry_count} (Limit: {limit})...")
                            time.sleep(sleep_time)

        print(f"\n✅ Finished {year}. Total: {total_year_records} Saved to: {file_path}")

if __name__ == "__main__":
    sport = """?person wdt:P641 ?object."""
    occupation = """?person wdt:P106 ?object. """
    param = {
        "sport": (sport, "sport"),
        #"occupation": (occupation, "occupation"),
    }
    for name, (filter_key, object_type) in param.items():
        fetch_and_save_json_by_year(name, object_type, filter_key)