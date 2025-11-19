import sys
import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from datetime import datetime

def log_query_info(file_name, total_count, log_file="query_log.txt"):
    """
    Ghi log thông tin truy vấn vào file văn bản (.txt).
    Định dạng: "{query_name} đã truy vấn {total_count} kết quả, hoàn thành lúc {time}"
    """
    now = datetime.now()
    timestamp_str = now.strftime("%H:%M:%S %d/%m/%Y")

    log_message = f"{file_name} đã truy vấn {total_count} kết quả, hoàn thành lúc {timestamp_str}\n"

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message)
        print(f"--> [LOG] Đã ghi thông tin vào '{log_file}'")
    except Exception as e:
        print(f"Lỗi khi ghi log txt: {e}", file=sys.stderr)

# ---------------------------------------------------
class WikidataExtractor:

    def __init__(self, user_agent):
        if not user_agent:
            raise ValueError("User-Agent là bắt buộc để truy vấn Wikidata.")

        self.endpoint_url = "https://query.wikidata.org/sparql"
        # Khởi tạo SPARQLWrapper
        self.sparql = SPARQLWrapper(self.endpoint_url)
        self.sparql.agent = user_agent
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(300)
    def _run_paginated_query(self, base_query, page_size = 10000):
        """
        (Hàm "private") Thực thi một truy vấn SPARQL, tự động lật trang (dùng OFFSET)
        cho đến khi lấy hết kết quả và lưu vào file JSON.

        Tham số:
            base_query (str): Câu truy vấn SPARQL (chưa bao gồm LIMIT/OFFSET).
            output_filename (str): Tên file để lưu kết quả.
        """
        page = 1

        const_page_size = page_size

        all_bindings = []
        json_head = None
        retry_count = 0
        offset_num = 0

        max_retries = 20
        while True:
            paginated_query = base_query + f"\nLIMIT {page_size}\nOFFSET {offset_num}"
            self.sparql.setQuery(paginated_query)
            print(f"{datetime.now().strftime("%H:%M:%S")} Đang lấy trang {page}, {page_size}/page...", end="", flush=True)
            try:
                start_time = time.monotonic()
                response = self.sparql.query()
                raw_data_bytes = response.response.read()
                cleaned_data_string = raw_data_bytes.decode('utf-8', errors='ignore')
                results = json.loads(cleaned_data_string)
                #Đo thời gian
                end_time = time.monotonic()
                duration = end_time - start_time
                if json_head is None:
                    json_head = results["head"]
                bindings = results["results"]["bindings"]
                retry_count = 0
                page_size = const_page_size
                all_bindings.extend(bindings)
                if not bindings or len(bindings) < page_size:
                    print(f" "
                          f"\n-------> Đã lấy hết! Tổng {len(all_bindings)}")
                    break
                print("Truy vấn đã xong!",end = "", flush=True)


                print(f" Đã lấy {len(bindings)}, mất {int(duration)}s", end="\n", flush=True)
                page += 1
                offset_num += page_size
                time.sleep(1)
            except Exception as e:
                print(f"\n!!! LỖI KHI ĐANG TRUY VẤN (offset {offset_num}): {e}", file=sys.stderr)

                retry_count += 1
                if retry_count > max_retries:
                    print(f"    Đã thử lại {max_retries} lần thất bại. TỪ BỎ truy vấn này.", file=sys.stderr)
                    break
                else:
                    if retry_count % 5 == 0 and retry_count > 0:
                        sleep_time = 60 * retry_count
                    else:
                        sleep_time = 5 * retry_count
                    if page_size > 3000 and retry_count % 2 == 0:
                        page_size -= 2000
                    print(f"    Đang thử lại (lần {retry_count}/{max_retries}) sau {sleep_time}s với {page_size}/page...",file=sys.stderr)
                    time.sleep(sleep_time)
        return all_bindings

    def _create_intervals(self,start_val, end_val, step = 10):

        intervals = []
        current_start = start_val

        while current_start < end_val:
            current_end = current_start + step
            if current_end > end_val:
                current_end = end_val

            interval = (current_start, current_end)
            intervals.append(interval)

            current_start = current_end

        return intervals
    def _iterative_query(self, start, end , base_query,page_size,output_filename):
        all_bindings = []
        intervals = self._create_intervals(start, end+1)
        print(f"===========BẮT ĐẦU TRUY VẤN CHO {output_filename}===========")
        print("===========TRUY VẤN THEO TỪNG KHOẢNG==========")
        start_time = datetime.now()
        for start_year, end_year in intervals:
            print(f"\n--- {start_year}-{end_year} ---")
            year_filter_str = f"FILTER(YEAR(?person_dob) > {start_year} && YEAR(?person_dob) <= {end_year})"
            era_query = base_query.replace("##YEAR_FILTER_HOOK##", year_filter_str)
            binding = self._run_paginated_query(era_query,page_size)
            all_bindings.extend(binding)
            print(f"--- KẾT THÚC {start_year}-{end_year}. (Tổng hiện tại: {len(all_bindings)}) ---")
        end_time = datetime.now()
        print(f"==================TỔNG TOÀN TRUY VẤN = {len(all_bindings)}, THỜI GIAN = {end_time- start_time}=============== ")
        # GHI LOG
        log_file_path = os.path.join(OUTPUT_DIR, "tracking_log.txt")
        log_query_info(output_filename, len(all_bindings), log_file_path)
        final_json_output = {
            "head":  {"vars": []},
            "results": {"bindings": all_bindings}
        }
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(final_json_output, f, ensure_ascii=False, indent=2)
            print(f"\n==> ĐÃ LƯU: {len(all_bindings)} kết quả vào file '{output_filename}'")
        except IOError as e:
            print(f"\n!!! LỖI KHI LƯU FILE: {e}", file=sys.stderr)

        return all_bindings
    def fetch_all_relationships(self, relationship_queries, start, end, output_dir="data"):
        """
        Hàm chính để chạy nhiều truy vấn lật trang.

        Tham số:
            relationship_queries (dict): Một dict {"tên": "truy vấn SPARQL"}.
            output_dir (str): Thư mục để lưu các file JSON.
        """
        # Đảm bảo thư mục output tồn tại
        os.makedirs(output_dir, exist_ok=True)
        # Đường dẫn file log
        # log_file_path = os.path.join(output_dir, "tracking_log.txt")

        all_data = {}

        for name, (query,page_size) in relationship_queries.items():
            output_filename = os.path.join(output_dir, f"raw_data_{name}.json")
            self._iterative_query(start, end, query, page_size, output_filename)
            try:
                with open(output_filename, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    all_data[name] = loaded_data["results"]["bindings"]

            except Exception:
                all_data[name] = []
            time.sleep(5)


        print("\n*** HOÀN TẤT TẤT CẢ TRUY VẤN! ***")
        return all_data

if __name__ == "__main__":


    YOUR_USER_AGENT = "nqaq2005@gmail.com"

    OUTPUT_DIR = "/content/drive/MyDrive/data_output3"

    #truy vấn anh/chị/em của một người

    query_siblings = """
        SELECT  ?person       ?personLabel       ?person_dob       ?personDescription
                ?familyMember ?familyMemberLabel ?familyMember_dob ?familyMemberDescription
                ?relationshipLabel
        WHERE {
            ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P3373 ?familyMember.
            OPTIONAL { ?familyMember wdt:P569 ?familyMember_dob . }
            BIND("siblings" AS ?relationshipLabel).
            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """
    query_mother = """
        SELECT  ?person       ?personLabel       ?person_dob       ?personDescription
                ?familyMember ?familyMemberLabel ?familyMember_dob ?familyMemberDescription
                ?relationshipLabel
        WHERE {
            ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P25 ?familyMember.
            OPTIONAL { ?familyMember wdt:P569 ?familyMember_dob . }
            BIND("mother" AS ?relationshipLabel).
            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """

    query_father = """
        SELECT  ?person       ?personLabel       ?person_dob       ?personDescription
                ?familyMember ?familyMemberLabel ?familyMember_dob ?familyMemberDescription
                ?relationshipLabel
        WHERE {
            ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P22 ?familyMember. #father
            OPTIONAL { ?familyMember wdt:P569 ?familyMember_dob . }
            BIND("father" AS ?relationshipLabel).
            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """

    query_spouse = """
        SELECT  ?person       ?personLabel       ?person_dob       ?personDescription
                ?familyMember ?familyMemberLabel ?familyMember_dob ?familyMemberDescription
                ?relationshipLabel
        WHERE {
            ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P26 ?familyMember. #spouse
            OPTIONAL { ?familyMember wdt:P569 ?familyMember_dob . }
            BIND("spouse" AS ?relationshipLabel).
            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """

    #truy vấn các trường học của một người từng theo học

    query_educated = """
        SELECT ?person ?personLabel ?person_dob ?personDescription ?hub ?hubLabel ?hubDescription ?relationshipLabel
        WHERE {
            ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P69 ?hub.
            BIND("educated at" AS ?relationshipLabel)
            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """



    #truy vấn đảng phái của một người #160k

    query_party = """
        SELECT ?person ?personLabel ?person_dob ?personDescription ?hub ?hubLabel ?hubDescription ?relationshipLabel
        WHERE {
      ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P102 ?hub.

            BIND("party"@vi AS ?relationshipLabel).

            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """



    #truy vấn band nhạc của ai đó #27750

    query_band = """
        SELECT ?person ?personLabel ?person_dob ?personDescription ?hub ?hubLabel ?hubDescription ?relationshipLabel
        WHERE
        {
            ?person wdt:P31 wd:Q5.
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            ?person wdt:P463 ?hub.
            ?hub wdt:P31 wd:Q215380.

            BIND("band"@vi AS ?relationshipLabel).

            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """


    #truy vấn các phim đã đóng

    query_actor = """
        SELECT ?person ?personLabel ?person_dob ?personDescription ?hub ?hubLabel ?hubDescription ?relationshipLabel
        WHERE
        {
              ?person wdt:P31 wd:Q5.
              optional{?person wdt:P569 ?person_dob.}
              ##YEAR_FILTER_HOOK##
              ?hub wdt:P161 ?person.
              ?hub wdt:P31 wd:Q11424.

              BIND("actor"@vi AS ?relationshiplabel).

              SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """


    #Trình diễn trong tác phẩm (âm nhạc)

    query_artist = """
        SELECT ?person ?personLabel ?person_dob ?personDescription ?hub ?hubLabel ?hubDescription ?relationshipLabel
        WHERE
        {
            ?class wdt:P279* wd:Q2188189 .
            ?hub wdt:P31 ?class .
            ?hub wdt:P175 ?person.
            ?person wdt:P31 wd:Q5 .
            ?person wdt:P569 ?person_dob.
            ##YEAR_FILTER_HOOK##
            BIND("artist"@vi AS ?relationshipLabel)

            SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
        }
        """


    """Lưu query và số dòng cần lấy"""
    queries_to_run = {
        "siblings": (query_siblings,20000),
        "mother": (query_mother,20000),
        "father": (query_father,20000),
        "spouse": (query_spouse,20000),
        "educated": (query_educated, 10000),
        "party": (query_party, 20000),
        "band": (query_band, 20000),
        "actor": (query_actor, 20000),
        "artist": (query_artist, 20000),
    }

    # Khởi tạo Extractor
    extractor = WikidataExtractor(user_agent=YOUR_USER_AGENT)

    data = extractor.fetch_all_relationships(queries_to_run,1800,2025, OUTPUT_DIR)


    if data:
        for name, query in queries_to_run.items():
            if name in data:
                print(f"\nĐã lấy xong {len(data[name])} quan hệ '{name}'.")
    else:
        print("\nHàm fetch_all_relationships không trả về dữ liệu.", file=sys.stderr)
