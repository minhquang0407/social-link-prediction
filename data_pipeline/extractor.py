import sys
import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataExtractor:
    """
    Class này kết nối với Wikidata SPARQL endpoint để chạy các truy vấn
    và tự động xử lý việc "lật trang" (pagination) để lấy TẤT CẢ kết quả.
    """

    def __init__(self, user_agent):
        """
        Khởi tạo Wrapper, trỏ đến endpoint của Wikidata và đặt User-Agent.

        Tham số:
            user_agent (str): Một User-Agent duy nhất (BẮT BUỘC).
                             (ví dụ: "MyDataProject/1.0 (myemail@example.com)")
        """
        if not user_agent:
            raise ValueError("User-Agent là bắt buộc để truy vấn Wikidata.")

        self.endpoint_url = "https://query.wikidata.org/sparql"
        # Khởi tạo SPARQLWrapper
        self.sparql = SPARQLWrapper(self.endpoint_url)

        # Wikidata YÊU CẦU bạn đặt User-Agent
        self.sparql.agent = user_agent

        # Đặt định dạng trả về mặc định
        self.sparql.setReturnFormat(JSON)

        # Đặt Socket Timeout (Thời gian chờ kết nối)
        self.sparql.setTimeout(600)

    def _run_paginated_query(self, base_query, output_filename):
        """
        (Hàm "private") Thực thi một truy vấn SPARQL, tự động lật trang (dùng OFFSET)
        cho đến khi lấy hết kết quả và lưu vào file JSON.

        Tham số:
            base_query (str): Câu truy vấn SPARQL (chưa bao gồm LIMIT/OFFSET).
            output_filename (str): Tên file để lưu kết quả.
        """
        print(f"--- Bắt đầu truy vấn cho: {output_filename} ---")
        print("Đang lấy trang 1...", end="", flush=True)

        offset_num = 0
        page_size = 2000
        all_results = []

        # Thêm bộ đếm thử lại (retry counter)
        retry_count = 0
        max_retries = 10 # Thử lại tối đa 5 lần cho một trang

        while True:
            # 1. Thêm LIMIT và OFFSET vào truy vấn
            paginated_query = base_query + f"\nLIMIT {page_size}\nOFFSET {offset_num}"

            self.sparql.setQuery(paginated_query)

            try:
                # 2. Thực thi truy vấn
                results = self.sparql.query().convert()

                bindings = results["results"]["bindings"]

                # 3. Nếu thành công, reset bộ đếm thử lại
                retry_count = 0

                # 4. Kiểm tra xem trang này có rỗng không
                if not bindings:
                    print(" Đã lấy hết!")
                    break # Dừng vòng lặp (while True) nếu không còn kết quả

                # 5. Thêm kết quả của trang này vào danh sách
                all_results.extend(bindings)

                print(f" Đã lấy {len(bindings)} (Tổng: {len(all_results)}). Đang lấy trang tiếp theo...", end="\n", flush=True)

                # 6. Tăng OFFSET cho lần lặp tiếp theo
                offset_num += page_size

                # 7. Lịch sự: Chờ 1 giây giữa các yêu cầu
                time.sleep(1)

            except Exception as e:
                # 8. Xử lý thử lại (retry)
                print(f"\n!!! LỖI KHI ĐANG TRUY VẤN (offset {offset_num}): {e}", file=sys.stderr)

                retry_count += 1
                if retry_count > max_retries:
                    print(f"    Đã thử lại {max_retries} lần thất bại. TỪ BỎ truy vấn này.", file=sys.stderr)
                    break # Thoát khỏi vòng lặp while True
                else:
                    print(f"    Đang thử lại (lần {retry_count}/{max_retries}) sau 5 giây...", file=sys.stderr)
                    time.sleep(5) # Chờ 5 giây rồi thử lại vòng lặp này

        # 9. Sau khi vòng lặp kết thúc, lưu tất cả kết quả vào file
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n==> ĐÃ LƯU: {len(all_results)} kết quả vào file '{output_filename}'")
        except IOError as e:
            print(f"\n!!! LỖI KHI LƯU FILE: {e}", file=sys.stderr)

        return all_results

    def fetch_all_relationships(self, relationship_queries, output_dir="data"):
        """
        Hàm chính để chạy nhiều truy vấn lật trang.

        Tham số:
            relationship_queries (dict): Một dict {"tên": "truy vấn SPARQL"}.
            output_dir (str): Thư mục để lưu các file JSON.
        """
        # Đảm bảo thư mục output tồn tại
        os.makedirs(output_dir, exist_ok=True)

        all_data = {}

        for name, query in relationship_queries.items():
            output_filename = os.path.join(output_dir, f"raw_data_{name}.json")
            results = self._run_paginated_query(query, output_filename)
            all_data[name] = results

        print("\n*** HOÀN TẤT TẤT CẢ TRUY VẤN! ***")
        return all_data # <-- Dòng này sửa lỗi 'NoneType'

# --- VÍ DỤ SỬ DỤNG ---
if __name__ == "__main__":


    YOUR_USER_AGENT = "nqaq2005@gmail.com"


    # ---------------------------------------------
    #tạo đường dẫn lưu file json
    try:
        script_path = os.path.abspath(__file__)
    except NameError:
        script_path = os.path.abspath('.')
    current_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(current_dir)
    OUTPUT_DIR = os.path.join(project_root, "data_output")
    print(f"--- Đã đặt thư mục lưu trữ là: {OUTPUT_DIR} ---")

    # ---------------------------------------------

    # Định nghĩa các truy vấn SPARQL (không có LIMIT/OFFSET)

    query_spouse = """
    SELECT ?person ?personLabel ?personDescription ?person2 ?person2Label ?person2Description ?relationship
    WHERE {
      ?person wdt:P31 wd:Q5.
      ?person wdt:P26 ?person2.
      BIND("vợ/chồng"@vi AS ?relationship).
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,vi". }
    }
    """

    query_father = """
    SELECT ?person ?personLabel ?personDescription ?person2 ?person2Label ?person2Description ?relationship
    WHERE {
      ?person wdt:P31 wd:Q5.
      ?person wdt:P22 ?person2.
      BIND("cha"@vi AS ?relationship).
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,vi". }
    }
    """

    query_edu = """
    SELECT ?person ?personLabel ?personDescription ?school ?schoolLabel ?relationship
    WHERE {

      ?person wdt:P69 ?school.
      ?person wdt:P569 ?ngay_sinh.
      FILTER(?ngay_sinh > "1950-01-01T00:00:00Z"^^xsd:dateTime).
      ?person wdt:P31 wd:Q5.
      BIND("học tại"@vi AS ?relationship).

      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,vi". }
    }
    """

    # Tạo một dict để chứa các queries
    queries_to_run = {
        "spouse_P26": query_spouse,
        "father_P22": query_father,
        "edu_P69": query_edu
    }

    # Khởi tạo Extractor
    extractor = WikidataExtractor(user_agent=YOUR_USER_AGENT)

    # 3. Truy vấn
    data = extractor.fetch_all_relationships(queries_to_run, OUTPUT_DIR)
    print('Đã chạy xog.')